"""Convex MPC over the single-rigid-body model, and the leg controller under it.

The full dynamics of a quadruped are 18-dimensional, non-linear, and change
discontinuously every time a foot touches down.  Optimising over that at
50 Hz is not realistic.  The trick that made modern quadrupeds work (Di Carlo,
Wensing, Katz, Bledt & Kim, 2018) is to throw away almost all of it:

  **Pretend the robot is a single rigid brick, and that the legs are
  massless force generators attached to it at known points.**

That is the *centroidal* or *single-rigid-body* model.  The legs weigh about
14% of this robot, so the approximation is not free -- but what it buys is
enormous: the dynamics become LINEAR in the ground reaction forces, and the
whole problem becomes a quadratic program, which solves in milliseconds and
always finds the global optimum.

The state is 13 numbers: roll-pitch-yaw, position, angular rate, linear
velocity, and gravity.  Gravity is carried as a 13th state that never changes,
purely so the dynamics can be written as x' = A x + B u with no constant
term -- a standard trick, and the reason you will see a stray "1" in every
published version of this matrix.
"""

import math

import casadi as ca
import numpy as np


def hat(r):
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


class ConvexMPC:
    def __init__(self, N=10, dt=0.03, mass=14.0, inertia=(0.09, 0.30, 0.33),
                 mu=0.6, f_max=180.0, f_min=1.0,
                 q_diag=(60.0, 60.0, 6.0, 4.0, 4.0, 220.0,
                         1.0, 1.0, 1.0, 12.0, 12.0, 6.0, 0.0),
                 r_weight=1e-4, friction_cone=True):
        self.N, self.dt, self.m, self.mu = N, dt, mass, mu
        self.I = np.diag(inertia)
        self.f_max, self.f_min = f_max, f_min
        self.Q = np.diag(q_diag)
        self.R = r_weight
        self.cone = friction_cone
        nu = 12 * N
        nc = 16 * N if friction_cone else 0
        self.nu, self.nc = nu, nc
        # OSQP, not an active-set solver.  Half of the 12N force variables
        # are pinned to zero at any moment (the swinging feet), which makes
        # the problem badly degenerate; an active-set method spends hundreds
        # of iterations discovering that, while OSQP's splitting method does
        # not care.  Measured on this problem: 360 ms vs a few ms.
        opts = {"print_time": False, "error_on_fail": False, "warm_start_primal": True,
                "osqp": {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4,
                         "max_iter": 900,
                         "polish": False}}
        self.solver = ca.conic("qp", "osqp",
                               {"h": ca.Sparsity.dense(nu, nu),
                                "a": ca.Sparsity.dense(max(nc, 1), nu)}, opts)
        self.last = np.zeros(nu)

    # ------------------------------------------------------------- dynamics
    def _AB(self, yaw, r_feet):
        """Linearised single-rigid-body dynamics about the current yaw.

        The only non-linearity kept is the yaw rotation, because a robot
        walking in a circle really does need to know which way it is facing.
        Roll and pitch are assumed small, which for a trotting robot is true
        by construction -- if they are not, the controller has already failed.
        """
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Iw = Rz @ self.I @ Rz.T
        Iw_inv = np.linalg.inv(Iw)

        A = np.eye(13)
        A[0:3, 6:9] = Rz.T * self.dt     # rpy_dot = Rz^T * omega
        A[3:6, 9:12] = np.eye(3) * self.dt
        A[11, 12] = self.dt              # v_z picks up gravity

        B = np.zeros((13, 12))
        for i in range(4):
            B[6:9, 3 * i:3 * i + 3] = Iw_inv @ hat(r_feet[i]) * self.dt
            B[9:12, 3 * i:3 * i + 3] = np.eye(3) * self.dt / self.m
        return A, B

    def solve(self, x0, x_ref, r_feet, contact):
        """One MPC step.  `contact` is (N, 4) booleans from the gait schedule."""
        N = self.N
        A, B = self._AB(float(x0[2]), r_feet)

        # Condense: express every future state in terms of the force sequence,
        # so the QP has only the forces as unknowns.  For a 10-step horizon
        # that is 120 variables instead of 250.
        Aqp = np.zeros((13 * N, 13))
        Bqp = np.zeros((13 * N, 12 * N))
        Apow = np.eye(13)
        pows = [np.eye(13)]
        for k in range(N):
            Apow = A @ Apow
            pows.append(Apow.copy())
            Aqp[13 * k:13 * k + 13] = Apow
        for k in range(N):
            for j in range(k + 1):
                Bqp[13 * k:13 * k + 13, 12 * j:12 * j + 12] = pows[k - j] @ B

        Qb = np.kron(np.eye(N), self.Q)
        H = 2.0 * (Bqp.T @ Qb @ Bqp + self.R * np.eye(12 * N))
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(12 * N)
        e = Aqp @ x0 - x_ref.reshape(-1)
        g = 2.0 * Bqp.T @ Qb @ e

        lbx = np.zeros(12 * N)
        ubx = np.zeros(12 * N)
        for k in range(N):
            for i in range(4):
                s = 12 * k + 3 * i
                if contact[k, i]:
                    lbx[s:s + 2] = -self.mu * self.f_max
                    ubx[s:s + 2] = self.mu * self.f_max
                    lbx[s + 2] = self.f_min
                    ubx[s + 2] = self.f_max
                # a swinging foot pushes on nothing: bounds stay 0

        if self.cone:
            # The friction pyramid: |fx| <= mu*fz and |fy| <= mu*fz.  The true
            # constraint is a CONE (sqrt(fx^2+fy^2) <= mu*fz); the pyramid is
            # its inscribed square, which keeps the problem a plain QP instead
            # of a second-order cone program.  It is slightly conservative --
            # it forbids some forces that friction would actually allow --
            # and every published convex-MPC quadruped uses it anyway.
            Ac = np.zeros((16 * N, 12 * N))
            uba = np.zeros(16 * N)
            lba = np.full(16 * N, -1e20)
            for k in range(N):
                for i in range(4):
                    s = 12 * k + 3 * i
                    row = 16 * k + 4 * i
                    for d, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
                        Ac[row, s + d] = sgn
                        Ac[row, s + 2] = -self.mu
                        row += 1
            sol = self.solver(h=H, g=g, a=Ac, lba=lba, uba=uba,
                              lbx=lbx, ubx=ubx, x0=self.last)
        else:
            sol = self.solver(h=H, g=g, a=np.zeros((1, 12 * N)),
                              lba=[-1e20], uba=[1e20], lbx=lbx, ubx=ubx,
                              x0=self.last)
        u = np.asarray(sol["x"]).ravel()
        if not np.all(np.isfinite(u)):
            u = np.zeros_like(u)
        self.last = u
        return u[:12].reshape(4, 3)


def stance_torque(J, f_world):
    """Joint torques that produce a given ground reaction force.

    tau = -J^T f.  The minus sign is the whole of Newton's third law: the QP
    solved for the force the GROUND pushes on the foot with, and the leg has
    to push down on the ground by the same amount.  J^T maps a force at the
    foot into the torques at the joints -- the same transpose that appears in
    every manipulator's static force analysis.
    """
    return -J.T @ f_world


def swing_torque(q, dq, q_des, kp=42.0, kd=1.2):
    """A joint-space PD for a leg in the air.

    A swinging leg carries no load, so there is nothing to optimise: it just
    has to arrive at the next foothold on time.  Splitting the controller
    this way -- QP for the loaded legs, PD for the free ones -- is what makes
    the whole stack fast enough to run at 500 Hz.
    """
    return kp * (q_des - q) - kd * dq
