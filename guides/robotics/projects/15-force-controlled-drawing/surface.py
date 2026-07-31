"""The surface the robot draws on, and the contact that happens when it does.

The surface is a gently curved cylinder lying along y: a bulge in the middle of
the drawing area, like a book that will not lie flat.

    z_surface(x) = z_c + sqrt(R^2 - (x - x_c)^2)

Nothing about it is special except that the ROBOT DOES NOT KNOW IT.  The planner
is given a WRONG surface -- flat, or the right shape shifted a few millimetres
-- because that is the actual situation: you measured the fixture with a ruler,
the part is 2 mm thicker than the drawing said, and the table sags under load.

Contact is a one-sided spring-damper along the surface normal ("penalty"
contact), plus Coulomb friction along the surface.  Writing it out by hand
rather than using a physics engine's contact solver keeps the experiment about
the CONTROLLER: the paper stiffness is a number we chose and can quote, not a
solver setting we would have to reverse-engineer afterwards.
"""

import numpy as np


class Cylinder:
    """A convex surface: flat in y, circular in x."""

    def __init__(self, R=0.55, x_c=0.60, z_top=0.40, k_contact=25000.0,
                 d_contact=80.0, mu=0.25):
        self.R, self.x_c = R, x_c
        self.z_c = z_top - R  # so that z_surface(x_c) = z_top
        self.k, self.d, self.mu = k_contact, d_contact, mu

    def height(self, x):
        dx = np.clip(np.asarray(x, dtype=float) - self.x_c, -0.999 * self.R, 0.999 * self.R)
        return self.z_c + np.sqrt(self.R ** 2 - dx ** 2)

    def normal(self, x):
        """Outward unit normal at a point above the surface, pointing up."""
        dx = float(np.clip(x - self.x_c, -0.999 * self.R, 0.999 * self.R))
        z = np.sqrt(self.R ** 2 - dx ** 2)
        n = np.array([dx, 0.0, z])
        return n / np.linalg.norm(n)

    def slope_deg(self, x):
        n = self.normal(x)
        return float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))

    def contact_wrench(self, p, v):
        """Force the surface applies to the pen tip at ``p`` moving at ``v``.

        Returns a 6-vector (force, torque) in world axes.  Zero unless the tip
        is below the surface.
        """
        n = self.normal(p[0])
        depth = float((self.height(p[0]) - p[2]) * n[2])  # along the normal
        if depth <= 0.0:
            return np.zeros(6), 0.0
        v_n = float(v[:3] @ n)
        f_n = self.k * depth + self.d * max(-v_n, 0.0)
        f = f_n * n
        # Coulomb friction opposing the sliding direction, capped at mu * f_n.
        v_t = v[:3] - v_n * n
        speed = float(np.linalg.norm(v_t))
        if speed > 1e-6:
            f = f - self.mu * f_n * (v_t / speed)
        return np.concatenate([f, np.zeros(3)]), f_n


class BelievedSurface:
    """What the planner THINKS the surface is.  Deliberately not the truth."""

    def __init__(self, kind="flat", truth=None, offset=0.0):
        self.kind, self.truth, self.offset = kind, truth, offset

    def height(self, x):
        if self.kind == "flat":
            # A flat table at the height of the bulge's summit.
            return self.truth.height(self.truth.x_c) + self.offset
        return self.truth.height(x) + self.offset

    def normal(self, x):
        if self.kind == "flat":
            return np.array([0.0, 0.0, 1.0])
        return self.truth.normal(x)
