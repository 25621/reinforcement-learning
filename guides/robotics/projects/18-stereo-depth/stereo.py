"""A stereo rig, and block matching written out by hand.

The whole of stereo is one equation:

        Z = f * B / d

    Z = distance to the point (metres)
    f = focal length (pixels)          -- from project 16
    B = baseline, the gap between the two cameras (metres)
    d = disparity, how far the point moved sideways between the two
        images (pixels)

Everything else -- rectification, matching, filtering -- exists only to get a
trustworthy `d` for as many pixels as possible.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "16-camera-calibration"))

from camera import Camera, default_camera                        # noqa: E402
from render import Plane, speckle_texture, render                # noqa: E402


# --------------------------------------------------------------------------
# the rig
# --------------------------------------------------------------------------

class StereoRig:
    """Two identical cameras, side by side, pointing the same way.

    This is the *rectified* arrangement: rows line up, so a point at row v in
    the left image is on row v in the right image too, and matching is a 1-D
    search along the row instead of a 2-D search over the image.  Real rigs
    are never built this well; you get here by calibrating both cameras and
    warping both images (see `rectify` below).
    """

    def __init__(self, baseline=0.09, cam=None):
        self.cam = cam or default_camera()
        self.baseline = float(baseline)

    def render(self, planes, R_wc=None, t_wc=None, supersample=2, noise=1.5):
        R = np.eye(3) if R_wc is None else R_wc
        t = np.zeros(3) if t_wc is None else np.asarray(t_wc, float)
        left, dl, _ = render(self.cam, planes, R, t, supersample=supersample,
                             seed=1, noise=noise)
        right, dr, _ = render(self.cam, planes, R, t + R @ np.array([self.baseline, 0, 0]),
                              supersample=supersample, seed=2, noise=noise)
        return left, right, dl, dr

    def disparity_to_depth(self, disp):
        """d (px) -> Z (m).  Zero and negative disparities are invalid."""
        Z = np.full(disp.shape, np.nan)
        ok = disp > 0.05
        Z[ok] = self.cam.fx * self.baseline / disp[ok]
        return Z

    def depth_to_disparity(self, Z):
        return self.cam.fx * self.baseline / np.maximum(Z, 1e-6)


def rectify_maps(cam):
    """Undistortion maps: the step that turns a real camera pair into the
    idealized rig above.  Without it, a straight epipolar line is a curve and
    the matcher's 1-D row search looks in the wrong place."""
    import cv2
    return cv2.initUndistortRectifyMap(cam.K, cam.dist.reshape(1, 5), None,
                                       cam.K, (cam.width, cam.height), cv2.CV_32FC1)


# --------------------------------------------------------------------------
# block matching, by hand
# --------------------------------------------------------------------------

def _box_sum(x, win):
    """Sum over a win x win window at every pixel, via an integral image.

    The integral image (running sum from the top-left corner) turns the cost
    of a window sum from win^2 additions into exactly four lookups, whatever
    the window size.  With 96 disparities to test that is the difference
    between seconds and minutes.
    """
    h = win // 2
    p = np.pad(x, ((h + 1, h), (h + 1, h)), mode="edge")
    ii = p.cumsum(0).cumsum(1)
    H, W = x.shape
    return (ii[win:win + H, win:win + W] - ii[0:H, win:win + W]
            - ii[win:win + H, 0:W] + ii[0:H, 0:W])


def block_match(left, right, max_disp=96, win=9, subpixel=True, lr_check=True,
                uniqueness=0.0, return_cost=False):
    """Disparity of every LEFT pixel, by searching along its row in the right image.

    Returns (disparity float array, valid mask).  Invalid pixels are NaN.

    Cost is SAD -- the Sum of Absolute Differences between the window in the
    left image and the candidate window in the right.  It is the cheapest
    sensible similarity measure and, for a pair of images from the same
    camera model, a perfectly good one.
    """
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    if L.ndim == 3:
        L = L.mean(axis=2)
        R = R.mean(axis=2)
    H, W = L.shape

    costs = np.full((max_disp + 1, H, W), np.inf, dtype=np.float32)
    for d in range(max_disp + 1):
        # A point at column x in the LEFT image appears at x - d in the RIGHT
        # image (the right camera sits to the right, so everything shifts
        # left).  Columns with x < d have no partner and stay at infinity.
        diff = np.abs(L[:, d:] - R[:, :W - d])
        costs[d, :, d:] = _box_sum(diff, win)

    best = np.argmin(costs, axis=0)
    cmin = np.take_along_axis(costs, best[None], 0)[0]
    disp = best.astype(np.float32)
    valid = np.isfinite(cmin)

    if subpixel:
        # Fit a parabola through the cost at (d-1, d, d+1) and take its
        # minimum.  The true disparity is almost never a whole number of
        # pixels; without this the depth map is visibly banded into steps.
        dm = np.clip(best - 1, 0, max_disp)
        dp = np.clip(best + 1, 0, max_disp)
        c0 = np.take_along_axis(costs, dm[None], 0)[0]
        c2 = np.take_along_axis(costs, dp[None], 0)[0]
        den = (c0 - 2 * cmin + c2)
        shift = np.where(np.abs(den) > 1e-6, 0.5 * (c0 - c2) / np.where(den == 0, 1, den), 0.0)
        interior = (best > 0) & (best < max_disp) & np.isfinite(c0) & np.isfinite(c2)
        disp = disp + np.where(interior, np.clip(shift, -1, 1), 0.0)

    if uniqueness > 0:
        # Reject a match whose runner-up (at a genuinely different disparity)
        # is nearly as good.  That is the sign of a repeating or blank
        # texture, where the matcher has no way to tell candidates apart.
        c = costs.copy()
        for k in (-1, 0, 1):
            idx = np.clip(best + k, 0, max_disp)
            np.put_along_axis(c, idx[None], np.inf, 0)
        second = c.min(axis=0)
        valid &= (second > cmin * (1.0 + uniqueness)) | ~np.isfinite(second)

    if lr_check:
        # Match right-to-left as well and keep only pixels where the two
        # agree.  This is what removes matches in areas visible to only one
        # camera -- the strip behind every object's left edge, which the right
        # camera simply cannot see.
        dr = _match_right(L, R, max_disp, win)
        x = np.arange(W)[None, :].repeat(H, 0)
        xr = np.clip(np.rint(x - disp).astype(int), 0, W - 1)
        back = np.take_along_axis(dr, xr, 1)
        valid &= np.abs(back - disp) <= 1.5

    disp = np.where(valid, disp, np.nan)
    if return_cost:
        return disp, valid, costs
    return disp, valid


def _match_right(L, R, max_disp, win):
    H, W = L.shape
    costs = np.full((max_disp + 1, H, W), np.inf, dtype=np.float32)
    for d in range(max_disp + 1):
        diff = np.abs(R[:, :W - d] - L[:, d:])
        costs[d, :, :W - d] = _box_sum(diff, win)
    return np.argmin(costs, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# from a depth map to a point cloud
# --------------------------------------------------------------------------

def depth_to_cloud(cam, Z, color=None, step=2):
    """Lift a depth map into 3D points in the camera frame.

    Note this needs the camera's *undistorted* ray for every pixel -- the
    same `pixel_to_normalized` used by every other project in the phase.
    A depth map without intrinsics is just a grey picture.
    """
    H, W = Z.shape
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    z = Z[::step, ::step]
    ok = np.isfinite(z)
    uv = np.stack([uu[ok], vv[ok]], axis=1).astype(float)
    xn = cam.pixel_to_normalized(uv)
    pts = np.concatenate([xn * z[ok][:, None], z[ok][:, None]], axis=1)
    if color is None:
        return pts, None
    return pts, color[::step, ::step][ok]


# --------------------------------------------------------------------------
# the test scene
# --------------------------------------------------------------------------

def build_scene(blank_poster=True):
    """A back wall plus four posters at known distances, one of them blank.

    Fronto-parallel slabs make the accuracy question easy to score: every
    pixel on one poster has the same true depth, so "depth error on the
    poster at 1.6 m" is a well-defined number.
    """
    planes = []
    truth = {}

    def poster(cx, cy, z, w, h, seed, blank=False):
        tex = (np.full((64, 64, 3), 180.0).astype(np.uint8) if blank
               else speckle_texture(220, 220, seed=seed, scale=3))
        planes.append(Plane([cx - w / 2, cy - h / 2, z], [w, 0, 0], [0, h, 0], tex,
                            f"poster{z}"))
        truth[len(planes) - 1] = z

    # Placed by working backwards from where they should land in the image
    # (X = (u - cx) Z / fx), so that all four are visible side by side rather
    # than hiding behind each other.
    poster(-0.165, -0.093, 0.55, 0.183, 0.143, 11)
    poster(0.247, -0.152, 0.90, 0.300, 0.234, 12)
    poster(-0.480, 0.354, 1.60, 0.533, 0.416, 13)
    poster(0.712, 0.575, 2.60, 0.867, 0.677, 14, blank=blank_poster)
    planes.append(Plane([-3.0, -2.2, 4.0], [6, 0, 0], [0, 4.4, 0],
                        speckle_texture(400, 400, seed=20, scale=4), "wall"))
    truth[len(planes) - 1] = 4.0
    return planes, truth
