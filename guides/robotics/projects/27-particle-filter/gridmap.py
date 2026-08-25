"""An occupancy grid, a laser scanner that reads it, and the two sensor models.

Shared with project 29, which builds its own occupancy grid from scratch and
needs the same ray caster to score it.

An OCCUPANCY GRID is the simplest useful map: chop the world into square cells
and store one bit per cell, "something is here" or "nothing is here".  It is
called that because each cell records whether the cell is occupied.  Crude, but
it supports the one query a laser needs -- "marching outward from here in this
direction, where do I first hit something?" -- in a handful of array lookups.
"""

import numpy as np


class GridMap:
    def __init__(self, occ, res, origin=(0.0, 0.0)):
        self.occ = np.asarray(occ, dtype=bool)      # [row=y, col=x]
        self.res = float(res)
        self.origin = np.asarray(origin, dtype=float)
        self.h, self.w = self.occ.shape
        self._df = None

    # ------------------------------------------------------------- geometry
    def extent(self):
        return (self.origin[0], self.origin[0] + self.w * self.res,
                self.origin[1], self.origin[1] + self.h * self.res)

    def world_to_cell(self, xy):
        c = (np.asarray(xy) - self.origin) / self.res
        return np.floor(c).astype(int)

    def is_free(self, xy):
        """True where the world point lies in a free, in-bounds cell."""
        c = self.world_to_cell(xy)
        cx, cy = c[..., 0], c[..., 1]
        ok = (cx >= 0) & (cx < self.w) & (cy >= 0) & (cy < self.h)
        out = np.zeros(ok.shape, dtype=bool)
        out[ok] = ~self.occ[cy[ok], cx[ok]]
        return out

    # ------------------------------------------------------------ ray casting
    def raycast(self, poses, angles, max_range=8.0, step=None, block=24):
        """Distance to the first obstacle, for every (pose, angle) pair.

        poses:  (N, 3) array of [x, y, theta]
        angles: (M,)   beam angles, relative to the robot's heading
        returns (N, M) ranges, clipped to max_range.

        Two speed decisions worth explaining, because this function is called
        several hundred thousand times and a naive version makes the project
        take twenty minutes instead of five:

        1. The march is vectorized over ALL rays at once, not looped per ray.
        2. It advances in blocks of `block` steps and drops rays that have
           already hit.  Indoors most beams stop within a couple of metres, so
           after the first block only a small minority are still travelling and
           every later block is cheap.  Marching all rays the full 8 m would do
           roughly three times the arithmetic for the same answer.
        """
        step = self.res if step is None else step
        n_steps = int(np.ceil(max_range / step)) + 1
        poses = np.atleast_2d(poses)
        th = (poses[:, 2:3] + np.asarray(angles)[None, :]).ravel()   # (N*M,)
        ox = np.repeat(poses[:, 0], len(angles))
        oy = np.repeat(poses[:, 1], len(angles))
        cs, sn = np.cos(th), np.sin(th)
        out = np.full(th.shape, max_range)
        live = np.arange(th.size)
        occ_flat = self.occ.ravel()
        s0 = 0
        while s0 < n_steps and live.size:
            ss = np.arange(s0, min(s0 + block, n_steps))
            d = (ss * step)[None, :]
            cx = ((ox[live, None] + cs[live, None] * d - self.origin[0])
                  / self.res).astype(np.int32)
            cy = ((oy[live, None] + sn[live, None] * d - self.origin[1])
                  / self.res).astype(np.int32)
            inb = (cx >= 0) & (cx < self.w) & (cy >= 0) & (cy < self.h)
            idx = np.where(inb, cy * self.w + cx, 0)
            hit = np.where(inb, occ_flat[idx], True)   # leaving the map = a hit
            any_hit = hit.any(axis=1)
            if any_hit.any():
                # argmax on a boolean row gives the FIRST True; it also gives 0
                # for an all-False row, which is why any_hit gates it.
                first = np.argmax(hit[any_hit], axis=1)
                out[live[any_hit]] = (s0 + first) * step
                live = live[~any_hit]
            s0 += block
        return np.minimum(out.reshape(len(poses), len(angles)), max_range)

    # ------------------------------------------------- the likelihood field
    def distance_field(self):
        """For every cell, the distance to the nearest occupied cell.

        Computed with a two-pass chamfer sweep: walk the grid forwards
        propagating "best distance so far" from up/left neighbours, then
        backwards from down/right.  Two passes are enough because any shortest
        path through the grid is monotone in one of the two sweep directions.
        Exact Euclidean distance would need a heavier algorithm; the chamfer
        approximation is within a few percent, which is far inside the laser's
        own noise.
        """
        if self._df is not None:
            return self._df
        BIG = 1e9
        d = np.where(self.occ, 0.0, BIG)
        r = self.res
        diag = r * np.sqrt(2.0)
        for i in range(self.h):
            for j in range(self.w):
                v = d[i, j]
                if i > 0:
                    v = min(v, d[i - 1, j] + r)
                    if j > 0:
                        v = min(v, d[i - 1, j - 1] + diag)
                    if j + 1 < self.w:
                        v = min(v, d[i - 1, j + 1] + diag)
                if j > 0:
                    v = min(v, d[i, j - 1] + r)
                d[i, j] = v
        for i in range(self.h - 1, -1, -1):
            for j in range(self.w - 1, -1, -1):
                v = d[i, j]
                if i + 1 < self.h:
                    v = min(v, d[i + 1, j] + r)
                    if j > 0:
                        v = min(v, d[i + 1, j - 1] + diag)
                    if j + 1 < self.w:
                        v = min(v, d[i + 1, j + 1] + diag)
                if j + 1 < self.w:
                    v = min(v, d[i, j + 1] + r)
                d[i, j] = v
        self._df = d
        return d

    def lookup_distance(self, xy):
        """Distance to the nearest obstacle at a world point (out of map = big)."""
        df = self.distance_field()
        c = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        cx = np.clip(c[..., 0], 0, self.w - 1)
        cy = np.clip(c[..., 1], 0, self.h - 1)
        out = df[cy, cx]
        inb = ((c[..., 0] >= 0) & (c[..., 0] < self.w) &
               (c[..., 1] >= 0) & (c[..., 1] < self.h))
        return np.where(inb, out, 10.0)


# ---------------------------------------------------------------- the worlds
def office_map(res=0.1, symmetric=False):
    """A room with internal walls.  `symmetric` mirrors it left-right, which
    makes two positions produce literally identical laser scans."""
    w_m, h_m = 16.0, 12.0
    w, h = int(w_m / res), int(h_m / res)
    occ = np.zeros((h, w), dtype=bool)
    t = max(1, int(0.2 / res))

    def box(x0, y0, x1, y1):
        i0, j0 = int(y0 / res), int(x0 / res)
        i1, j1 = int(y1 / res), int(x1 / res)
        occ[i0:i1, j0:j1] = True

    box(0, 0, w_m, t * res)                      # outer walls
    box(0, h_m - t * res, w_m, h_m)
    box(0, 0, t * res, h_m)
    box(w_m - t * res, 0, w_m, h_m)
    if symmetric:
        # Two identical alcoves, mirrored.  A robot in one cannot tell which.
        for x0 in (3.0, 11.0):
            box(x0, 4.0, x0 + 0.2, 8.0)
            box(x0, 4.0, x0 + 2.0, 4.2)
            box(x0, 7.8, x0 + 2.0, 8.0)
    else:
        # Deliberately irregular.  A plain rectangular room is close to
        # rotationally and reflectionally symmetric, and a laser scan taken in
        # one corner looks a lot like a scan taken in another -- which turns
        # "global localization" into a coin flip that has nothing to teach.
        # These spurs and blocks break every symmetry the outer walls have.
        box(4.0, 2.0, 4.2, 8.0)                  # a long spur
        box(4.0, 2.0, 9.0, 2.2)
        box(10.0, 5.0, 12.0, 5.2)                # a stub at a different angle
        box(11.8, 5.0, 12.0, 9.5)
        box(6.0, 9.0, 6.2, 11.0)
        box(1.5, 9.0, 2.6, 10.1)                 # scattered furniture
        box(7.5, 5.5, 8.3, 6.3)
        box(13.5, 1.5, 14.6, 2.3)
        box(9.0, 10.0, 10.4, 10.6)
        box(2.0, 5.0, 2.6, 6.4)
        box(13.0, 7.0, 13.6, 8.6)
    return GridMap(occ, res)


def free_poses(gmap, n, rng, margin=0.4):
    """Sample n poses that are not inside a wall (or hugging one)."""
    x0, x1, y0, y1 = gmap.extent()
    out = np.empty((n, 3))
    got = 0
    while got < n:
        cand = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n),
                         rng.uniform(-np.pi, np.pi, n)], axis=1)
        ok = gmap.lookup_distance(cand[:, :2]) > margin
        take = cand[ok][:n - got]
        out[got:got + len(take)] = take
        got += len(take)
    return out
