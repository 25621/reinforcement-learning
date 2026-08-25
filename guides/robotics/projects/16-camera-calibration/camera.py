"""The pinhole camera model with lens distortion, written out by hand.

This file is the shared camera of Phase 3.  Projects 17-23 import it rather
than re-deriving the same twenty lines, so every project in the phase agrees
on exactly what "the camera" means.

Two directions matter and they are NOT inverses of each other in closed form:

    project()    3D point in the camera frame  ->  pixel      (easy, direct)
    ray()        pixel  ->  3D direction in the camera frame  (needs an
                 iterative undistort, because the distortion polynomial has
                 no elementary inverse)

Everything is plain NumPy.  OpenCV is used in the projects only as an
independent second opinion, never as the implementation.
"""

import numpy as np


# --------------------------------------------------------------------------
# rotation helpers (Rodrigues, both directions)
# --------------------------------------------------------------------------

def rodrigues(rvec):
    """Axis-angle 3-vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rot_to_rvec(R):
    """3x3 rotation matrix -> axis-angle 3-vector."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > np.pi - 1e-6:                     # near 180 deg: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = int(np.argmax(k))
        k = k * np.sign(A[i]) * np.sign(k[i] if k[i] != 0 else 1.0)
        k = k / np.linalg.norm(k)
        return k * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(theta)) * theta


def orthonormalize(R):
    """Snap a nearly-rotation matrix onto the closest true rotation (SVD)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def rot_angle_deg(R_a, R_b):
    """Angle of the rotation that takes R_a to R_b, in degrees."""
    return np.degrees(np.linalg.norm(rot_to_rvec(R_a.T @ R_b)))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera pose that puts `target` in the middle of the image.

    Returns (R_wc, t_wc): columns of R_wc are the camera axes in world
    coordinates, t_wc is the camera centre.  Camera convention is the usual
    computer-vision one: +z forward (into the scene), +x right, +y down.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    z = target - eye
    z = z / np.linalg.norm(z)
    up = np.asarray(up, float)
    x = np.cross(z, -up)                          # -up because +y points DOWN
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)
    return orthonormalize(R_wc), eye


def invert_pose(R_wc, t_wc):
    """World-from-camera -> camera-from-world."""
    R_cw = R_wc.T
    return R_cw, -R_cw @ t_wc


# --------------------------------------------------------------------------
# the camera
# --------------------------------------------------------------------------

class Camera:
    """A pinhole camera plus the standard 5-term radial/tangential distortion.

    Parameters
    ----------
    fx, fy : focal length in PIXELS (the physical focal length in millimetres
             divided by the pixel pitch; you never measure it in millimetres
             from an image, so calibration reports pixels).
    cx, cy : principal point -- where the optical axis pierces the sensor.
             Close to the image centre, but never exactly there.
    dist   : (k1, k2, p1, p2, k3).  k* are radial, p* are tangential.
    """

    def __init__(self, fx, fy, cx, cy, dist=(0, 0, 0, 0, 0), width=640, height=480):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.dist = np.zeros(5)
        self.dist[:len(dist)] = np.asarray(dist, float)
        self.width, self.height = int(width), int(height)

    # -- parameter plumbing -------------------------------------------------
    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @staticmethod
    def from_K(K, dist=(0, 0, 0, 0, 0), width=640, height=480):
        return Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist, width, height)

    def copy(self):
        return Camera(self.fx, self.fy, self.cx, self.cy, self.dist.copy(),
                      self.width, self.height)

    def as_vector(self):
        return np.array([self.fx, self.fy, self.cx, self.cy, *self.dist])

    # -- the two directions -------------------------------------------------
    def distort(self, xn):
        """Ideal normalized coords -> distorted normalized coords.

        `xn` is (N, 2) = (X/Z, Y/Z).  This is where the lens bends straight
        lines: a point twice as far from the centre is pushed by more than
        twice as much, because the correction is a polynomial in the radius.
        """
        xn = np.asarray(xn, float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([xd, yd], axis=1)

    def undistort(self, xd, iters=30):
        """Distorted normalized coords -> ideal normalized coords.

        The polynomial above cannot be inverted in closed form, so we iterate.
        The naive fixed point `x -= distort(x) - xd` DIVERGES at the image
        corners once the radial terms get strong.  The form below -- divide
        out the radial factor, subtract the tangential part -- is what OpenCV
        uses and it converges for any lens you will meet.
        """
        xd = np.asarray(xd, float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = self.dist
        x = xd.copy()
        for _ in range(iters):
            r2 = x[:, 0] ** 2 + x[:, 1] ** 2
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            radial = np.where(np.abs(radial) < 1e-6, 1e-6, radial)
            dx = 2.0 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2.0 * x[:, 0] ** 2)
            dy = p1 * (r2 + 2.0 * x[:, 1] ** 2) + 2.0 * p2 * x[:, 0] * x[:, 1]
            xn = np.stack([(xd[:, 0] - dx) / radial, (xd[:, 1] - dy) / radial], axis=1)
            if np.max(np.abs(xn - x)) < 1e-13:
                x = xn
                break
            x = xn
        return x

    def project(self, X_cam):
        """(N,3) points in the CAMERA frame -> (N,2) pixels."""
        X_cam = np.asarray(X_cam, float).reshape(-1, 3)
        z = np.where(np.abs(X_cam[:, 2]) < 1e-9, 1e-9, X_cam[:, 2])
        xn = X_cam[:, :2] / z[:, None]
        xd = self.distort(xn)
        return np.stack([self.fx * xd[:, 0] + self.cx,
                         self.fy * xd[:, 1] + self.cy], axis=1)

    def project_world(self, X_world, R_cw, t_cw):
        """(N,3) world points, camera-from-world pose -> (N,2) pixels."""
        X_world = np.asarray(X_world, float).reshape(-1, 3)
        X_cam = X_world @ np.asarray(R_cw).T + np.asarray(t_cw).reshape(1, 3)
        return self.project(X_cam), X_cam

    def pixel_to_normalized(self, uv):
        """Pixels -> ideal normalized coords (undoes K, then the distortion)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        xd = np.stack([(uv[:, 0] - self.cx) / self.fx,
                       (uv[:, 1] - self.cy) / self.fy], axis=1)
        return self.undistort(xd)

    def rays(self, supersample=1):
        """Unit direction in the CAMERA frame for every pixel.

        Used by the renderer.  With supersample=s each pixel is split into
        s x s sub-rays whose colours are averaged -- cheap anti-aliasing, and
        without it every checkerboard corner would land on a jagged staircase
        and the corner detector would inherit that as noise.
        """
        s = int(supersample)
        cache = getattr(self, "_ray_cache", None)
        if cache is None:
            cache = self._ray_cache = {}
        if s in cache:
            return cache[s]                       # rays never change; renders repeat
        H, W = self.height * s, self.width * s
        # Pixel i's CENTRE is at coordinate i, not i+0.5 -- that is OpenCV's
        # convention and the one `project()` above assumes.  Rendering with
        # the other convention shifts every image by exactly half a pixel,
        # which shows up later as a rock-steady 0.5 px "detector bias".
        off = (np.arange(s) + 0.5) / s - 0.5
        u = (np.arange(self.width)[:, None] + off[None, :]).reshape(-1)
        v = (np.arange(self.height)[:, None] + off[None, :]).reshape(-1)
        uu, vv = np.meshgrid(u, v)
        uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
        xn = self.pixel_to_normalized(uv)
        d = np.concatenate([xn, np.ones((xn.shape[0], 1))], axis=1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cache[s] = d.reshape(H, W, 3)
        return cache[s]

    def __repr__(self):
        return (f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, "
                f"cy={self.cy:.2f}, dist={np.round(self.dist, 4).tolist()})")


# --------------------------------------------------------------------------
# the camera used everywhere in Phase 3
# --------------------------------------------------------------------------

def default_camera():
    """A plausible 640x480 webcam: ~60 deg horizontal field of view, barrel
    distortion strong enough to be visible (about 9 px at the corners)."""
    return Camera(fx=540.0, fy=538.0, cx=322.0, cy=241.0,
                  dist=(-0.28, 0.10, 0.0009, -0.0012, -0.02),
                  width=640, height=480)
