"""The cart-pole from project 09, run many copies at a time.

Project 09 solved this plant with LQR: given the linearised model it computes
the provably optimal feedback gain, with no data at all.  That makes it the
perfect opponent for reinforcement learning -- we know exactly what the best
possible answer is, so "did PPO learn well?" has a number rather than a vibe.

Two changes from 09, both about making the comparison honest:

* The reward is the NEGATIVE of the LQR cost, so both controllers are graded on
  the same objective.  A "+1 for staying alive" reward would be easier for PPO
  and would make the comparison meaningless.
* There is no early termination.  A pole that falls simply keeps accruing cost.
  Termination would hand PPO a second, hidden objective (survive) on top of the
  stated one, and the classic result "my agent learned to end the episode" is
  exactly the reward-design trap this phase warns about.

The physics is 09's, vectorised over ``n_env`` copies so that PPO can collect
thousands of transitions per second; ``verify()`` checks the vectorised version
against 09's scalar class step for step.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "09-cart-pole-lqr"))

import cartpole as CP9      # noqa: E402
import lqr as LQR9          # noqa: E402

G = 9.81
DT = 0.02
EP_LEN = 200                          # 4 seconds
U_MAX = 15.0
Q = np.diag([1.0, 0.1, 10.0, 0.1])    # position, velocity, ANGLE, angular rate
R = 0.01
COST_SCALE = 10.0                     # keeps rewards near unit size
COST_CAP = 2.0                        # see step(): the training reward is capped


class BatchCartPole:
    """``n_env`` independent cart-poles stepped together."""

    obs_dim = 5
    act_dim = 1

    def __init__(self, n_env=32, seed=0, M=1.0, m=0.1, l=0.5,
                 init_angle=0.20, init_rate=0.5, cap=True):
        self.cap = cap
        self.n = n_env
        self.M, self.m, self.l = M, m, l
        self.rng = np.random.default_rng(seed)
        self.init_angle, self.init_rate = init_angle, init_rate
        self.reset()

    # -- physics (identical equations to project 09, applied to arrays) ------
    def deriv(self, s, u):
        xd, th, thd = s[:, 1], s[:, 2], s[:, 3]
        st, ct = np.sin(th), np.cos(th)
        xdd = (u + self.m * self.l * thd * thd * st - self.m * G * st * ct) \
            / (self.M + self.m * st * st)
        thdd = (G * st - xdd * ct) / self.l
        return np.stack([xd, xdd, thd, thdd], axis=1)

    def rk4(self, s, u, dt=DT):
        k1 = self.deriv(s, u)
        k2 = self.deriv(s + 0.5 * dt * k1, u)
        k3 = self.deriv(s + 0.5 * dt * k2, u)
        k4 = self.deriv(s + dt * k3, u)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- gym-ish interface --------------------------------------------------
    def sample_states(self, n, rng=None):
        rng = rng or self.rng
        s = np.zeros((n, 4))
        s[:, 0] = rng.uniform(-0.05, 0.05, n)
        s[:, 1] = rng.uniform(-self.init_rate, self.init_rate, n)
        s[:, 2] = rng.uniform(-self.init_angle, self.init_angle, n)
        s[:, 3] = rng.uniform(-self.init_rate, self.init_rate, n)
        return s

    def reset(self):
        self.s = self.sample_states(self.n)
        self.t = np.zeros(self.n, dtype=int)
        return self.obs()

    def obs(self):
        """Angle enters as sine and cosine.

        A raw angle has a seam: +3.14 and -3.14 are the same pole but the
        opposite ends of the number line, and a network has to spend capacity
        learning that wrap-around.  Sine and cosine remove the seam.
        """
        s = self.s
        return np.stack([s[:, 0], s[:, 1], np.sin(s[:, 2]), np.cos(s[:, 2]),
                         s[:, 3]], axis=1)

    def cost(self, s, u):
        return (np.einsum("ij,jk,ik->i", s, Q, s) + R * u ** 2) * DT

    def step(self, u):
        """One control step.  The reward is a CAPPED version of the cost.

        The true cost of a fallen pole is unbounded -- the cart accelerates
        away and the position term grows without limit -- so returns span four
        orders of magnitude and the critic cannot fit them.  Capping says "all
        disasters are equally bad", which is true enough for a controller and
        makes the learning problem well-conditioned.  Evaluation still uses the
        uncapped cost, so the score is not softened, only the training signal.
        """
        u = np.clip(u, -U_MAX, U_MAX)
        c = np.minimum(self.cost(self.s, u), COST_CAP) if self.cap else self.cost(self.s, u)
        self.s = self.rk4(self.s, u)
        self.t += 1
        done = self.t >= EP_LEN
        if done.any():
            idx = np.where(done)[0]
            self.s[idx] = self.sample_states(len(idx))
            self.t[idx] = 0
        return self.obs(), -c / COST_SCALE, done, c


def episode_cost(controller, n_ep=200, seed=12345, init_angle=0.20,
                 init_rate=0.5, M=1.0, m=0.1, l=0.5):
    """Total LQR cost over an episode, averaged over ``n_ep`` starts.

    ``controller(obs, state) -> force``.  Both PPO and LQR are graded with this
    one function, from the same initial states, so nothing but the controller
    differs.
    """
    env = BatchCartPole(n_ep, seed=seed, init_angle=init_angle,
                        init_rate=init_rate, M=M, m=m, l=l)
    total = np.zeros(n_ep)
    fell = np.zeros(n_ep, dtype=bool)
    for _ in range(EP_LEN):
        u = controller(env.obs(), env.s)
        u = np.clip(u, -U_MAX, U_MAX)
        total += env.cost(env.s, u)
        env.s = env.rk4(env.s, u)
        fell |= np.abs(env.s[:, 2]) > 1.0
        env.t += 1
    return dict(cost=float(total.mean()), cost_med=float(np.median(total)),
                fell=float(fell.mean()))


def lqr_controller(M=1.0, m=0.1, l=0.5):
    """Project 09's design, on whatever model you tell it to believe in."""
    plant = CP9.CartPole(M=M, m=m, l=l)
    A, B = plant.linearize()
    K, _ = LQR9.dlqr(A, B, Q, R, DT)
    K = np.asarray(K).ravel()

    def ctrl(obs, state):
        return -(state @ K)
    return ctrl, K


def verify():
    """Batched physics must equal project 09's scalar physics exactly."""
    env = BatchCartPole(8, seed=0)
    plant = CP9.CartPole()
    s = env.sample_states(8, np.random.default_rng(1))
    u = np.random.default_rng(2).uniform(-10, 10, 8)
    mine = env.rk4(s.copy(), u)
    theirs = np.array([plant.rk4(s[i], u[i], DT) for i in range(8)])
    return float(np.abs(mine - theirs).max())


if __name__ == "__main__":
    print("max |batched - project 09 scalar| :", verify())
    ctrl, K = lqr_controller()
    print("LQR gain:", np.round(K, 3))
    print("LQR cost:", episode_cost(ctrl))
    print("zero-force cost:", episode_cost(lambda o, s: np.zeros(len(s))))
