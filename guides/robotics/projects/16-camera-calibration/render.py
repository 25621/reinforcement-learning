"""A tiny ray-traced renderer for textured planes, plus the checkerboard.

Why render images at all instead of just simulating "detected corners"?
Because a calibration that is only ever fed perfect synthetic corner
coordinates never meets the two things that actually break real
calibrations: a detector that finds corners slightly in the wrong place,
and lens distortion that the renderer applied for real rather than as an
afterthought.  Rendering through the SAME distortion model we later try to
recover gives us ground truth we can check against -- which a real webcam
can never give you.

The renderer shoots one ray per (sub-)pixel, intersects it with every plane,
keeps the nearest hit, and samples that plane's texture.  That is enough for
checkerboards (project 16), fiducial tags (17), stereo scenes (18), depth
scans (19) and a corridor of speckle for visual odometry (20).
"""

import numpy as np

from camera import invert_pose


class Plane:
    """A finite textured rectangle in the world.

    origin : the rectangle's corner at texture coordinate (0, 0)
    e1, e2 : the two edge vectors (their lengths ARE the rectangle's size)
    texture: an (h, w, 3) uint8 image, sampled with u along e1, v along e2
    """

    def __init__(self, origin, e1, e2, texture, name="plane", repeat=(1, 1)):
        self.origin = np.asarray(origin, float)
        self.e1 = np.asarray(e1, float)
        self.e2 = np.asarray(e2, float)
        self.texture = np.asarray(texture)
        self.name = name
        # How many times the texture tiles across the rectangle.  A 100 m
        # corridor wall (project 20) needs fine texture along its whole
        # length; one stretched image would give the tracker nothing to hold.
        self.repeat = (float(repeat[0]), float(repeat[1]))
        n = np.cross(self.e1, self.e2)
        self.normal = n / np.linalg.norm(n)
        self.L1 = float(np.dot(self.e1, self.e1))
        self.L2 = float(np.dot(self.e2, self.e2))

    def sample(self, u, v):
        """Bilinear texture lookup at (u, v) in [0,1]^2.

        Note `u * w - 0.5`, not `u * (w - 1)`: a texel covers an interval, and
        its colour belongs at the interval's CENTRE.  Getting this wrong
        stretches the texture by w/(w-1) -- only 0.2%, but that was a
        systematic 0.7 px corner shift here, which is larger than the whole
        error budget of a calibration.
        """
        h, w = self.texture.shape[:2]
        if self.repeat != (1.0, 1.0):
            u = np.mod(u * self.repeat[0], 1.0)
            v = np.mod(v * self.repeat[1], 1.0)
        x = np.clip(u * w - 0.5, 0, w - 1)
        y = np.clip(v * h - 0.5, 0, h - 1)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        ax = (x - x0)[:, None]
        ay = (y - y0)[:, None]
        t = self.texture.astype(np.float32)
        top = t[y0, x0] * (1 - ax) + t[y0, x1] * ax
        bot = t[y1, x0] * (1 - ax) + t[y1, x1] * ax
        return top * (1 - ay) + bot * ay


def render(cam, planes, R_wc, t_wc, supersample=2, background=(30, 30, 34),
           light=None, seed=None, noise=0.0):
    """Render `planes` seen by `cam` at world pose (R_wc, t_wc).

    Returns (image uint8 HxWx3, depth float HxW, plane_id int HxW).
    Depth is the z-coordinate in the camera frame (metres); depth is inf and
    plane_id is -1 where nothing was hit.
    """
    R_wc = np.asarray(R_wc, float)
    t_wc = np.asarray(t_wc, float).reshape(3)
    s = int(supersample)
    dirs_cam = cam.rays(supersample=s)                      # (Hs, Ws, 3)
    Hs, Ws, _ = dirs_cam.shape
    d = (dirs_cam.reshape(-1, 3) @ R_wc.T)                  # world directions
    o = t_wc[None, :]

    best_t = np.full(d.shape[0], np.inf)
    color = np.tile(np.asarray(background, float), (d.shape[0], 1))
    pid = np.full(d.shape[0], -1, dtype=np.int32)

    for i, pl in enumerate(planes):
        denom = d @ pl.normal
        ok = np.abs(denom) > 1e-9
        tt = np.full(d.shape[0], np.inf)
        tt[ok] = ((pl.origin - o[0]) @ pl.normal) / denom[ok]
        hit = ok & (tt > 1e-6) & (tt < best_t)
        if not np.any(hit):
            continue
        P = o + d[hit] * tt[hit][:, None]
        rel = P - pl.origin
        u = (rel @ pl.e1) / pl.L1
        v = (rel @ pl.e2) / pl.L2
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        idx = np.flatnonzero(hit)[inside]
        if idx.size == 0:
            continue
        c = pl.sample(u[inside], v[inside])
        if light is not None:                                # simple Lambert shading
            lam = abs(float(np.dot(pl.normal, light)))
            c = c * (0.45 + 0.55 * lam)
        color[idx] = c
        best_t[idx] = tt[hit][inside]
        pid[idx] = i

    # camera-frame z (not ray length) -- that is what "depth" means everywhere
    z = best_t * (dirs_cam.reshape(-1, 3)[:, 2])
    z[np.isinf(best_t)] = np.inf

    img = color.reshape(Hs, Ws, 3)
    depth = z.reshape(Hs, Ws)
    ids = pid.reshape(Hs, Ws)
    if s > 1:
        img = img.reshape(cam.height, s, cam.width, s, 3).mean(axis=(1, 3))
        depth = depth.reshape(cam.height, s, cam.width, s)[:, 0, :, 0]
        ids = ids.reshape(cam.height, s, cam.width, s)[:, 0, :, 0]
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), depth, ids


# --------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------

def checker_texture(cols, rows, px=40, margin=1, dark=25, light=235):
    """A (rows x cols) checkerboard with a light border.

    `cols` and `rows` count SQUARES.  OpenCV's detector finds INNER corners,
    so a 9x6-square board has 8x5 = 40 inner corners.
    """
    t = np.full(((rows + 2 * margin) * px, (cols + 2 * margin) * px, 3),
                float(light))
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0, x0 = (r + margin) * px, (c + margin) * px
                t[y0:y0 + px, x0:x0 + px] = dark
    return t.astype(np.uint8)


def speckle_texture(h, w, seed=0, scale=3, lo=35, hi=245):
    """Random blotches: high-frequency texture that corner and flow trackers
    can actually latch onto.  A featureless wall would give them nothing."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(lo, hi, (max(2, h // scale), max(2, w // scale)))
    ys = np.linspace(0, small.shape[0] - 1, h)
    xs = np.linspace(0, small.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, small.shape[0] - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, small.shape[1] - 1)
    ay = (ys - y0)[:, None]; ax = (xs - x0)[None, :]
    top = small[np.ix_(y0, x0)] * (1 - ax) + small[np.ix_(y0, x1)] * ax
    bot = small[np.ix_(y1, x0)] * (1 - ax) + small[np.ix_(y1, x1)] * ax
    g = top * (1 - ay) + bot * ay
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)


# --------------------------------------------------------------------------
# the checkerboard as a calibration target
# --------------------------------------------------------------------------

class Checkerboard:
    """A physical checkerboard: geometry, texture, and its inner corners.

    The board lies in its own frame with z = 0, corner (0,0,0) at the first
    inner corner, x to the right and y down -- matching how the detector
    orders the corners it finds.
    """

    def __init__(self, cols=9, rows=6, square=0.025):
        self.cols, self.rows, self.square = cols, rows, square
        self.inner = (cols - 1, rows - 1)
        gx, gy = np.meshgrid(np.arange(cols - 1), np.arange(rows - 1))
        self.object_points = np.stack(
            [gx.reshape(-1) * square, gy.reshape(-1) * square,
             np.zeros((cols - 1) * (rows - 1))], axis=1)
        self.texture = checker_texture(cols, rows)
        self.margin = 1                                     # squares of border

    def plane(self, R_wb, t_wb):
        """The board as a renderable Plane at world pose (R_wb, t_wb)."""
        px = self.square
        # texture (0,0) is the top-left of the border, one square out from the
        # first inner corner in both directions
        origin = t_wb + R_wb @ np.array([-(1 + self.margin) * px,
                                         -(1 + self.margin) * px, 0.0])
        w = (self.cols + 2 * self.margin) * px
        h = (self.rows + 2 * self.margin) * px
        return Plane(origin, R_wb @ np.array([w, 0, 0]),
                     R_wb @ np.array([0, h, 0]), self.texture, "board")


def board_pose(dist, tilt_x, tilt_y, roll, shift=(0.0, 0.0), rng=None):
    """A board pose in front of a camera sitting at the origin looking along +z.

    Returns (R_cb, t_cb): board-from-camera... strictly, the pose of the board
    expressed in the camera frame, which is what calibration solves for.
    """
    from camera import rodrigues
    R = rodrigues([np.radians(tilt_x), 0, 0]) @ rodrigues([0, np.radians(tilt_y), 0])
    R = R @ rodrigues([0, 0, np.radians(roll)])
    t = np.array([shift[0], shift[1], dist], float)
    return R, t


def camera_pose_from_board(R_cb, t_cb):
    """Convert 'board in camera frame' to a world pose for the renderer, with
    the board fixed at the world origin."""
    return invert_pose(R_cb, t_cb)
