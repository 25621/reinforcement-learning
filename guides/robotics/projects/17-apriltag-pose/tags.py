"""Tags: making them, putting them in a scene, finding them again.

A tag is just a printed square with a pattern inside.  Everything a robot
gets out of it comes from four numbers per tag -- the pixel coordinates of
its four corners -- plus one number you measure with a ruler: how big the
square is in metres.
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "16-camera-calibration"))

from camera import rodrigues                                    # noqa: E402
from render import Plane                                        # noqa: E402

DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_SIZE = 0.06                       # 6 cm of black square, edge to edge


def tag_texture(tag_id, px=240, quiet=1.0):
    """One tag as an image, with the white quiet zone around it.

    `quiet` is the border width in units of the tag's own size.  The border
    is not decoration: the detector finds the tag by looking for a black
    quadrilateral against a lighter background, so a tag printed edge to
    edge on dark paper is often simply invisible.
    """
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT),
                                      tag_id, px)
    b = int(px * quiet / 2)
    img = np.pad(m, b, constant_values=235)
    return np.repeat(img[:, :, None], 3, axis=2)


def tag_object_points(size=TAG_SIZE):
    """The four corners in the TAG's own frame, z = 0, in the order the
    detector returns them: top-left, top-right, bottom-right, bottom-left,
    with x right, y down, z out of the tag face."""
    s = size / 2
    return np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])


def tag_plane(tag_id, R_wt, t_wt, size=TAG_SIZE, quiet=1.0, px=240):
    """The tag as a renderable Plane at world pose (R_wt, t_wt)."""
    tex = tag_texture(tag_id, px=px, quiet=quiet)
    full = size * (1 + quiet)
    origin = t_wt + R_wt @ np.array([-full / 2, -full / 2, 0.0])
    return Plane(origin, R_wt @ np.array([full, 0, 0]),
                 R_wt @ np.array([0, full, 0]), tex, f"tag{tag_id}")


def detect_tags(img, refine=True):
    """Detect every tag.  Returns {id: (4,2) corner array}.

    `refine` runs the same subpixel corner refinement used on checkerboards
    in project 16.  The detector's raw corners are quantized to the polygon
    fit; the refinement re-reads the intensity gradient around each corner
    and typically halves the pose error, for microseconds of work.
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT))
    corners, ids, _ = det.detectMarkers(g)
    if ids is None:
        return {}
    out = {}
    for c, i in zip(corners, ids.reshape(-1)):
        c = np.asarray(c, np.float32).reshape(-1, 1, 2)
        if refine:
            c = cv2.cornerSubPix(
                g, c, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
        out[int(i)] = c.reshape(-1, 2).astype(float)
    return out


def draw_axes(img, cam, R, t, length=0.04, thickness=2):
    """Project the tag's own x/y/z axes into the image -- the honest check.

    If the drawn axes sit squarely on the tag, the pose is right; if they
    lean off it, the pose is wrong in a way no number will make obvious.
    """
    pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], float)
    uv = cam.project(pts @ R.T + t).astype(int)
    out = img.copy()
    for k, col in zip((1, 2, 3), ((214, 39, 40), (44, 160, 44), (31, 119, 180))):
        cv2.line(out, tuple(uv[0]), tuple(uv[k]), col, thickness, cv2.LINE_AA)
    return out


def tag_board(n=4, spacing=0.10, size=TAG_SIZE, first_id=0):
    """A rigid plate carrying `n` tags in a row/grid, all in one frame.

    Returns (object_points_by_id, plane_maker).  Solving all the tags at once
    is the standard cure for the pose ambiguity of project 17's experiment 3:
    the four tags see the plate from four different offsets, so the two
    candidate tilts no longer explain the corners equally well.
    """
    k = int(np.ceil(np.sqrt(n)))
    offsets = []
    for i in range(n):
        r, c = divmod(i, k)
        offsets.append(np.array([(c - (k - 1) / 2) * spacing,
                                 (r - (k - 1) / 2) * spacing, 0.0]))
    objs = {first_id + i: tag_object_points(size) + offsets[i] for i in range(n)}

    def make_planes(R_wb, t_wb):
        return [tag_plane(first_id + i, R_wb, t_wb + R_wb @ offsets[i], size=size,
                          quiet=(spacing - size) / size)
                for i in range(n)]

    return objs, make_planes


def pose_from_angles(dist, tilt_deg, azim_deg=0.0, roll_deg=0.0):
    """A tag pose in the camera frame: `dist` metres away, tilted `tilt_deg`
    from facing the camera, with the tilt axis rotated by `azim_deg`."""
    R = (rodrigues([0, 0, np.radians(azim_deg)]) @
         rodrigues([np.radians(tilt_deg), 0, 0]) @
         rodrigues([0, 0, np.radians(roll_deg)]))
    return R, np.array([0.0, 0.0, dist])
