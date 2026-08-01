"""Segmentation without a segmentation model, and an antipodal grasp search.

Two pieces:

  segment_by_depth  -- the stand-in for SAM.  Everything sticking up off the
                       table is an object; separate blobs are separate
                       objects.  This is what a bin-picking system did before
                       SAM existed, and it is still what many of them do.
  best_grasp        -- given a mask, choose where to close the fingers.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

def segment_by_depth(depth, table_z, min_height=0.006, min_area=250):
    """Every connected blob that stands more than `min_height` off the table.

    Why this is a legitimate stand-in for [SAM]: SAM's job in a grasping
    pipeline is to turn "somewhere around here" into an exact set of pixels
    belonging to one object.  On a tabletop with a depth camera, geometry
    already answers that question -- and answers it with metric accuracy that
    an RGB model cannot match.  What SAM adds is working on *cluttered,
    touching, non-tabletop* scenes where the depth blobs merge.  Experiment 5
    measures exactly that limit by pushing the objects together.
    """
    above = (table_z - depth) > min_height
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(above, 8)
    masks = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            masks.append(lab == i)
    return masks


# --------------------------------------------------------------------------
# grasping
# --------------------------------------------------------------------------

def principal_axis_grasp(mask, points):
    """The naive grasp: close across the object's narrow direction at its centre.

    Take the mask's pixels, find the direction they spread out MOST (the
    principal axis, the largest eigenvector of their covariance), and close
    the fingers perpendicular to it, through the centroid.  For a convex
    object this is exactly right and takes three lines.  For anything with a
    handle, a hole, or a hook it can put the closing line through empty air --
    which experiment 4 measures.
    """
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    cov = np.cov((P - c).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, -1]
    angle = float(np.arctan2(major[1], major[0])) + np.pi / 2
    return c, angle


def grasp_line(center, angle, half_len, n=64):
    d = np.array([np.cos(angle), np.sin(angle)])
    s = np.linspace(-half_len, half_len, n)
    return center[None, :] + s[:, None] * d[None, :], d


def evaluate_grasp(center, angle, mask, points, others, max_width,
                   friction_deg=25.0):
    """Score one candidate (centre, angle).  Returns a dict, or None if illegal.

    A grasp is a pair of CONTACT points.  Walk outward from the centre along
    the closing direction until you leave the mask on each side: those are
    where the fingers touch.  Then three things must hold.

      1. the two contacts are actually on the object (not across a hole)
      2. the gap is no wider than the gripper opens
      3. the two contact surfaces face each other closely enough that
         friction can hold -- the ANTIPODAL condition.  "Antipodal" is from
         the Greek for "feet opposite", as in antipodes: two points on
         opposite sides.  Formally, the line joining the contacts must lie
         inside both friction cones, and a friction cone of half-angle
         atan(mu) is why a rubber pad grips where steel slips.
    """
    H, W = mask.shape
    d = np.array([np.cos(angle), np.sin(angle)])
    contacts = []
    for s in (1, -1):
        hit = None
        inside = False
        for r in np.arange(0, 90, 0.5):
            p = center + s * r * d
            xi, yi = int(round(p[0])), int(round(p[1]))
            if not (0 <= xi < W and 0 <= yi < H):
                break
            if mask[yi, xi]:
                inside = True
                hit = (xi, yi)
            elif inside:
                break
        if hit is None:
            return None
        contacts.append(hit)
    (x1, y1), (x2, y2) = contacts
    p1, p2 = points[y1, x1], points[y2, x2]
    width = float(np.linalg.norm(p1 - p2))

    # the closing line must stay ON the object between the contacts
    line, _ = grasp_line(center, angle, np.hypot(x2 - x1, y2 - y1) / 2 * 0.95)
    li = np.clip(np.round(line).astype(int), [0, 0], [W - 1, H - 1])
    on_object = float(mask[li[:, 1], li[:, 0]].mean())

    # Contact normals, from the SILHOUETTE, not from the depth surface.
    #
    # This is the one place where looking straight down bites.  A top-down
    # depth camera sees only the TOP of the object; the vertical side walls
    # the fingers will actually squeeze are invisible, and their 3-D normal
    # cannot be measured.  Taking the normal of the visible surface gives the
    # top face -- which points at the camera, perpendicular to every possible
    # closing direction, so every grasp fails the antipodal test at 90
    # degrees.  (That was the first version of this code, and every grasp
    # scored 50-90 degrees.)
    #
    # What IS measurable is the object's OUTLINE.  Where the silhouette runs
    # is where the side wall is, and the outline's normal in the image is the
    # side wall's normal in the world -- which is the number the antipodal
    # condition actually needs.
    n1 = _boundary_normal(mask, x1, y1)
    n2 = _boundary_normal(mask, x2, y2)
    axis = np.array([x2 - x1, y2 - y1], float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    a1 = np.degrees(np.arccos(np.clip(abs(float(n1 @ axis)), 0, 1)))
    a2 = np.degrees(np.arccos(np.clip(abs(float(n2 @ axis)), 0, 1)))

    # would the fingers hit a neighbouring object on the way in?
    collision = False
    for s in (1, -1):
        q = center + s * (np.hypot(x2 - x1, y2 - y1) / 2 + 6) * d
        xi, yi = int(round(q[0])), int(round(q[1]))
        if 0 <= xi < W and 0 <= yi < H and others[yi, xi]:
            collision = True

    ok = (width <= max_width and on_object > 0.97 and
          max(a1, a2) <= friction_deg and not collision)
    return dict(ok=bool(ok), width=width, on_object=on_object,
                normal_angle=float(max(a1, a2)), collision=collision,
                contacts=((x1, y1), (x2, y2)),
                quality=float(on_object * (1 - max(a1, a2) / 90.0) *
                              (1.0 if width <= max_width else 0.0) *
                              (0.0 if collision else 1.0)))


_BOUNDARY_CACHE = {}


def _boundary_normal(mask, x, y, r=5):
    """Outward normal of the object's OUTLINE at a pixel, in image coordinates.

    Blur the binary mask and take its gradient: the gradient of a smoothed
    step points straight across the edge, which is the outline's normal.  The
    blur radius sets how much of the outline is averaged -- too small and a
    single jagged pixel decides the answer, too large and a corner is rounded
    away.
    """
    key = id(mask)
    if key not in _BOUNDARY_CACHE:
        f = cv2.GaussianBlur(mask.astype(np.float32), (2 * r + 1, 2 * r + 1), r / 2.0)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        _BOUNDARY_CACHE[key] = (gx, gy)
        if len(_BOUNDARY_CACHE) > 64:
            _BOUNDARY_CACHE.pop(next(iter(_BOUNDARY_CACHE)))
    gx, gy = _BOUNDARY_CACHE[key]
    v = np.array([float(gx[y, x]), float(gy[y, x])])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def best_grasp(mask, points, others, max_width, n_angles=24, n_offsets=7):
    """Search over closing angles and offsets from the centroid.

    The naive grasp tries exactly one (centre, angle).  This tries a few
    hundred and keeps the best legal one, which is what lets it find a grip on
    the rim of a mug rather than through its handle.
    """
    ys, xs = np.nonzero(mask)
    c0 = np.array([xs.mean(), ys.mean()])
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    best = None
    for ai in range(n_angles):
        angle = np.pi * ai / n_angles
        perp = np.array([-np.sin(angle), np.cos(angle)])
        for oi in np.linspace(-0.35, 0.35, n_offsets):
            c = c0 + perp * oi * span
            xi, yi = int(round(c[0])), int(round(c[1]))
            if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
                continue
            if not mask[yi, xi]:
                continue
            g = evaluate_grasp(c, angle, mask, points, others, max_width)
            if g is None:
                continue
            g["center"], g["angle"] = c, angle
            if best is None or g["quality"] > best["quality"]:
                best = g
    return best


def draw_grasp(img, g, color=(30, 220, 90)):
    out = img.copy()
    if g is None:
        return out
    (x1, y1), (x2, y2) = g["contacts"]
    cv2.line(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for (x, y) in ((x1, y1), (x2, y2)):
        cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)
    return out
