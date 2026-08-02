"""A planar peg, a chamfered hole, and a controller that is allowed to yield.

Why planar.  Insertion is a *contact* problem, and the classic analysis of it
(Whitney, 1982) is planar: a rectangular peg, a slot, two contact points, and
friction.  Everything that goes wrong in a real round-peg-in-round-hole -- one
point contact, two point contact, jamming, wedging -- already goes wrong here,
with three state variables instead of six.  The one thing the plane cannot show
is the shape of the search pattern: in 3D you sweep an Archimedean spiral
across the surface, here you sweep a line back and forth.  The spiral is the
same idea with one more dimension, and nothing else in this file changes.

Why a hand-written contact model instead of a physics engine.  The whole
experiment is about the controller's stiffness competing with the contact's
stiffness.  If the contact stiffness were a solver setting we did not choose,
the sweeps would be measuring the solver.  Here `K_CONTACT` is a number in this
file, so "how stiff is the world compared to the arm" is a question with an
answer.
"""

import numpy as np

G = 9.81

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class Hole:
    """A plate with a slot in it, as two polygons.  Plate top surface is z = 0."""

    def __init__(self, width=0.0204, thickness=0.018, chamfer=0.0015, span=0.10):
        self.w = width
        self.t = thickness
        self.ch = chamfer
        h = width / 2
        # With no chamfer the two middle vertices coincide, which leaves a
        # zero-length edge whose normal is 0/0.  Drop it rather than divide.
        if chamfer > 1e-9:
            self.left = np.array([(-span, -thickness), (-h, -thickness),
                                  (-h, -chamfer), (-h - chamfer, 0.0),
                                  (-span, 0.0)])
        else:
            self.left = np.array([(-span, -thickness), (-h, -thickness),
                                  (-h, 0.0), (-span, 0.0)])
        self.right = self.left * np.array([-1.0, 1.0])
        self.right = self.right[::-1]
        self.plates = [self.left, self.right]
        # the inner corners are the only plate vertices a peg can ever touch
        self.corners = np.array([(-h, -thickness), (-h, -chamfer),
                                 (-h - chamfer, 0.0), (h, -thickness),
                                 (h, -chamfer), (h + chamfer, 0.0)])
        # pre-baked edge tables, so the inner loop is two array operations
        # instead of a hundred small ones
        self.edgeA, self.edgeN = [], []
        for P in self.plates:
            e = np.roll(P, -1, axis=0) - P
            L = np.linalg.norm(e, axis=1)
            self.edgeA.append(P)
            self.edgeN.append(np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None])


class Peg:
    """A rectangular peg.  Body frame origin at the centre of mass."""

    def __init__(self, width=0.0200, length=0.060, mass=0.20):
        self.w = width
        self.L = length
        self.m = mass
        self.I = mass * (width ** 2 + length ** 2) / 12.0
        hw, hl = width / 2, length / 2
        self.corners = np.array([(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)])

    def poly(self, q):
        c, s = np.cos(q[2]), np.sin(q[2])
        R = np.array([[c, -s], [s, c]])
        return self.corners @ R.T + q[:2]

    def tip_z(self, q):
        return self.poly(q)[:, 1].min()


def clearance(peg, hole):
    return 0.5 * (hole.w - peg.w)


# ---------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------

K_CONTACT = 2.0e5      # N/m -- steel-on-steel is 1e8; this is a stiff-ish plastic
D_CONTACT = 120.0      # N.s/m
MU = 0.25              # sliding friction between peg and plate
V_STICK = 1e-4         # below this tangential speed friction is treated as sticking


def _deepest(poly, p):
    """If p is inside the convex-ish polygon, return (depth, outward normal).

    The polygon's least-penetrated face is the one we push out of.  With the
    penetrations here (tens of microns) any face would do; picking the nearest
    keeps the force pointing the way a real surface would push.
    """
    A = poly
    B = np.roll(poly, -1, axis=0)
    e = B - A
    L = np.linalg.norm(e, axis=1)
    n = np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None]     # outward for CCW
    d = np.einsum("ij,ij->i", p[None, :] - A, n)
    if np.any(d > 0):
        return 0.0, None
    k = int(np.argmax(d))
    return float(-d[k]), n[k]


def contacts(peg, hole, q, qd):
    """All contact wrenches on the peg, as (force_world, application_point).

    Two families, and you need both:

      * a PEG corner poking into a plate  -- the usual case going in
      * a HOLE corner poking into the peg -- what happens when the peg tilts
        and its flat side lands on the sharp lip of the hole

    Skip the second family and a tilted peg slides through solid steel.
    """
    out = []
    pts = peg.poly(q)
    w = qd[2]
    for A, N in zip(hole.edgeA, hole.edgeN):
        # D[p, i] = how far peg corner p is OUTSIDE plate edge i.  All negative
        # means the corner is inside the plate, and the least negative entry is
        # the face it is closest to escaping through.
        D = ((pts[:, None, :] - A[None, :, :]) * N[None, :, :]).sum(-1)
        inside = np.all(D < 0.0, axis=1)
        for p_i in np.flatnonzero(inside):
            k = int(np.argmax(D[p_i]))
            out.append(_penalty(peg, q, qd, pts[p_i], N[k], -D[p_i, k], w))
    eP = np.roll(pts, -1, axis=0) - pts
    NP = np.stack([eP[:, 1], -eP[:, 0]], 1) / np.linalg.norm(eP, axis=1)[:, None]
    Dc = ((hole.corners[:, None, :] - pts[None, :, :]) * NP[None, :, :]).sum(-1)
    inside = np.all(Dc < 0.0, axis=1)
    for c_i in np.flatnonzero(inside):
        k = int(np.argmax(Dc[c_i]))
        out.append(_penalty(peg, q, qd, hole.corners[c_i], -NP[k],
                            -Dc[c_i, k], w))
    return [o for o in out if o is not None]


def _penalty(peg, q, qd, p, n, depth, w):
    """One-sided spring-damper along n, plus regularized Coulomb friction.

    Coulomb's law says the tangential force cannot exceed mu times the normal
    force -- the friction cone again, the same object project 39 used, now
    doing work.  The regularization replaces the ideal "sticks until it
    slips" discontinuity with a very steep ramp, because a discontinuous force
    makes an explicit integrator explode.
    """
    r = p - q[:2]
    v = qd[:2] + w * np.array([-r[1], r[0]])
    vn = v @ n
    fn = K_CONTACT * depth - D_CONTACT * vn
    if fn <= 0:
        return None
    t = np.array([-n[1], n[0]])
    vt = v @ t
    ft = -np.clip(vt / V_STICK, -1.0, 1.0) * MU * fn
    return (fn * n + ft * t, p)


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

class Impedance:
    """A virtual spring-damper between a moving reference and a point on the peg.

    `cc` is the COMPLIANCE CENTRE: the offset, in the peg's own frame, of the
    point the spring is attached to.  A beginner reasonably asks why this
    matters, since the spring is the same spring wherever you hook it on.  It
    matters because a force applied away from a point also creates a TORQUE
    about that point.  Hook the spring at the wrist and a sideways push on the
    peg tip rotates the peg (tilting it into a jam); hook it at the tip and the
    same push slides the peg sideways (which is the correction you wanted).
    The mechanical version of this trick, a rubber-and-steel wrist that puts
    the compliance centre at the tip, is the Remote Centre of Compliance -- the
    single part that made robotic assembly work in the 1980s.  Experiment 6
    measures it.
    """

    def __init__(self, kx=800.0, kz=1600.0, kth=8.0, zeta=1.0, cc=0.0,
                 mass=0.20, inertia=1e-4, grav=True, f_push=12.0):
        self.K = np.array([kx, kz])
        self.kth = kth
        self.D = 2 * zeta * np.sqrt(self.K * mass)
        self.dth = 2 * zeta * np.sqrt(kth * inertia)
        self.cc = cc              # metres BELOW the peg centre of mass
        self.m = mass
        self.grav = grav
        # A cap on how hard the controller ever pushes DOWN.  Without it the
        # descending reference keeps sinking into the plate and the spring
        # force grows without limit -- which is exactly what a position
        # controller does, and exactly what breaks parts.  With it, "press
        # gently and feel your way in" is one number.
        self.f_push = f_push

    def wrench(self, q, qd, ref):
        c, s = np.cos(q[2]), np.sin(q[2])
        R = np.array([[c, -s], [s, c]])
        r = R @ np.array([0.0, -self.cc])
        p_c = q[:2] + r
        v_c = qd[:2] + qd[2] * np.array([-r[1], r[0]])
        F = self.K * (ref[:2] - p_c) - self.D * v_c
        F[1] = max(F[1], -self.f_push)
        if self.grav:
            F = F + np.array([0.0, self.m * G])
        tau = (r[0] * F[1] - r[1] * F[0]) + self.kth * (ref[2] - q[2]) - self.dth * qd[2]
        return F, tau

    def search_margin(self, amp, mu=MU):
        """Sideways pull the search can exert, minus the friction holding it.

        Positive means the sweep can actually drag the peg across the plate;
        negative means the controller is pressing down harder than it can push
        sideways, and the peg will sit still while the reference sweeps past
        it.  This one inequality explains most failed searches.
        """
        return self.K[0] * amp - mu * self.f_push


# ---------------------------------------------------------------------------
# search strategies
# ---------------------------------------------------------------------------

def straight(t, x0, params):
    """No search at all: drive the reference straight down."""
    return np.array([x0, params["z0"] - params["vz"] * t, 0.0])


def sweep(t, x0, params):
    """The planar shadow of a spiral search: sweep sideways, amplitude growing.

    The reason a search is needed at all: the arm does not know where the hole
    is to better than its own calibration error, which is millimetres, while
    the clearance is tens of microns.  Pushing straight down on a surface tells
    you only that you are not in the hole.  Sweeping turns "I am blocked" into
    "I am blocked HERE", and the tip eventually crosses the opening.
    """
    A = min(params["amp"], params["amp_rate"] * t)
    return np.array([x0 + A * np.sin(2 * np.pi * params["freq"] * t),
                     params["z0"] - params["vz"] * t, 0.0])


def triggered_sweep(t, x0, params, state):
    """Sweep only while the peg is blocked; the moment it drops, stop and push.

    A blind sweep keeps wiggling after the peg is already in the hole, which
    scrapes the sides and can pull it back out.  Watching the tip's height is
    a one-line contact-state estimator and it is the difference between an
    insertion that finishes and one that oscillates forever.
    """
    if state["entered"]:
        return np.array([state["x_at_entry"],
                         params["z0"] - params["vz"] * t, 0.0])
    return sweep(t, x0, params)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def simulate(peg, hole, ctrl, x0=0.0, tilt0=0.0, strategy="triggered",
             T=4.0, dt=4e-4, params=None, log_every=5):
    """Run one insertion attempt.  Returns a dict of traces and a verdict."""
    params = dict(z0=0.010, vz=0.012, amp=0.012, amp_rate=0.006, freq=3.0,
                  **(params or {}))
    q = np.array([x0, params["z0"] + peg.L / 2, tilt0])
    qd = np.zeros(3)
    state = dict(entered=False, x_at_entry=x0)
    n = int(T / dt)
    tr = dict(t=[], q=[], f=[], fx=[], fz=[], m=[], depth=[], ref=[])
    success = False
    t_ins = np.nan
    for k in range(n):
        t = k * dt
        if strategy == "straight":
            ref = straight(t, x0, params)
        elif strategy == "sweep":
            ref = sweep(t, x0, params)
        else:
            ref = triggered_sweep(t, x0, params, state)
        # the strategies command where the TIP should be; the spring pulls on
        # the compliance centre, which sits (L/2 - cc) above the tip
        ref = ref + np.array([0.0, peg.L / 2 - ctrl.cc, 0.0])
        # The commanded ANGLE is the peg's starting angle, not zero.  `tilt0`
        # models a crooked GRASP: the arm believes the peg is straight and
        # holds it there, so nothing but contact can ever straighten it.  Reset
        # the command to zero instead and the angular spring quietly fixes the
        # misalignment in mid-air, and the experiment measures nothing.
        ref[2] = tilt0
        F, tau = ctrl.wrench(q, qd, ref)
        cs = contacts(peg, hole, q, qd)
        Fc = np.zeros(2)
        Mc = 0.0
        for f, p in cs:
            Fc = Fc + f
            r = p - q[:2]
            Mc += r[0] * f[1] - r[1] * f[0]
        acc = np.array([(F[0] + Fc[0]) / peg.m,
                        (F[1] + Fc[1]) / peg.m - G,
                        (tau + Mc) / peg.I])
        qd = qd + dt * acc
        qd = np.clip(qd, -5.0, 5.0)
        q = q + dt * qd
        tip = peg.tip_z(q)
        if not state["entered"] and tip < -hole.ch - 0.0015:
            state["entered"] = True
            state["x_at_entry"] = q[0]
        if k % log_every == 0:
            tr["t"].append(t)
            tr["q"].append(q.copy())
            tr["f"].append(np.linalg.norm(Fc))
            tr["fx"].append(Fc[0])
            tr["fz"].append(Fc[1])
            tr["m"].append(Mc)
            tr["depth"].append(-tip)
            tr["ref"].append(ref.copy())
        if tip < -(hole.t - 0.002):
            success = True
            t_ins = t
            break
    for k in tr:
        tr[k] = np.array(tr[k])
    tr["success"] = success
    tr["t_insert"] = t_ins
    tr["peak_force"] = float(tr["f"].max()) if len(tr["f"]) else 0.0
    tr["final_depth"] = float(tr["depth"][-1]) if len(tr["depth"]) else 0.0
    tr["entered"] = state["entered"]
    return tr
