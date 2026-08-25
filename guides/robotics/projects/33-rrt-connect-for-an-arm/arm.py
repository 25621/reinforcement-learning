"""A 7-joint arm in MuJoCo, used purely as a collision checker.

The whole of sampling-based planning needs exactly three things from a robot:

  1. the joint limits, so it knows what to sample;
  2. a yes/no answer to "is this configuration in collision?";
  3. forward kinematics, so it can tell you where the hand ended up.

MuJoCo gives us all three, and we use it for nothing else -- no physics, no
integration, no contact forces.  `mj_kinematics` places the links and
`mj_collision` runs the broad and narrow phase; if `data.ncon > 0`, some pair
of geoms overlaps.  That is the entire interface.

A beginner might ask why we drag a whole physics engine in just to answer a
yes/no question.  Because the yes/no question is the expensive, fiddly part:
it needs every link's pose (forward kinematics), a bounding-volume pass to
skip the 90% of geom pairs that are obviously far apart, and an exact
mesh-to-mesh test for the rest.  MuJoCo has a tuned C implementation of all of
it.  Writing that ourselves would be the project, and it would not teach us
anything about planning.
"""

import os

import numpy as np

import mujoco

# 7 revolute joints, axes alternating z / y -- the standard "anthropomorphic"
# layout that gives an arm a shoulder, an elbow, and a wrist.
XML = """
<mujoco model="arm7">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <default>
    <geom rgba="0.55 0.62 0.70 1" friction="1 0.005 0.0001"/>
    <joint damping="1" limited="true"/>
  </default>

  <worldbody>
    <light pos="1.2 -1.2 2.2" dir="-0.5 0.5 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.92 0.92 0.92 1"/>

    <!-- The obstacle is a SHELF, not a bare table.  A bare table is a poor
         planning problem: the arm simply lifts over it, and RRT-Connect solves
         that on its first sample.  A shelf with a top, a bottom, a back and
         two sides leaves only one way in and out -- the front opening -- so
         the plan has to reverse the hand out of a pocket before it can go
         anywhere.  That is a narrow passage in joint space, which is what
         makes the problem worth planning.  The post adds a second obstacle
         between the shelf and the drop-off point.
         The shelf panels are drawn semi-transparent so the figures can show
         the arm inside; transparency is a rendering property only and has no
         effect whatsoever on collision checking. -->
    <geom name="shelf_bottom" type="box" pos="0.46 0.0 0.30" size="0.16 0.30 0.015"
          rgba="0.75 0.55 0.35 0.45"/>
    <geom name="shelf_top" type="box" pos="0.46 0.0 0.58" size="0.16 0.30 0.015"
          rgba="0.75 0.55 0.35 0.45"/>
    <geom name="shelf_back" type="box" pos="0.61 0.0 0.44" size="0.015 0.30 0.16"
          rgba="0.62 0.45 0.28 0.45"/>
    <geom name="shelf_left" type="box" pos="0.46 0.30 0.44" size="0.16 0.015 0.16"
          rgba="0.62 0.45 0.28 0.45"/>
    <geom name="shelf_right" type="box" pos="0.46 -0.30 0.44" size="0.16 0.015 0.16"
          rgba="0.62 0.45 0.28 0.45"/>
    <geom name="post" type="cylinder" pos="0.02 -0.42 0.40" size="0.05 0.40"
          rgba="0.80 0.35 0.30 0.85"/>

    <body name="link0" pos="0 0 0.06">
      <geom name="g0" type="cylinder" size="0.075 0.05"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="g1" type="capsule" fromto="0 0 0.055 0 0 0.105" size="0.055"/>
        <body name="link2" pos="0 0 0.16">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.9 1.9"/>
          <geom name="g2" type="capsule" fromto="0 0 0.052 0 0 0.208" size="0.052"/>
          <body name="link3" pos="0 0 0.26">
            <joint name="j3" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
            <geom name="g3" type="sphere" pos="0 0 0.04" size="0.04"/>
            <body name="link4" pos="0 0 0.08">
              <joint name="j4" type="hinge" axis="0 1 0" range="-2.6 0.1"/>
              <geom name="g4" type="capsule" fromto="0 0 0.045 0 0 0.195" size="0.045"/>
              <body name="link5" pos="0 0 0.24">
                <joint name="j5" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
                <geom name="g5" type="sphere" pos="0 0 0.035" size="0.035"/>
                <body name="link6" pos="0 0 0.07">
                  <joint name="j6" type="hinge" axis="0 1 0" range="-1.7 2.1"/>
                  <geom name="g6" type="capsule" fromto="0 0 0.038 0 0 0.072" size="0.038"/>
                  <body name="link7" pos="0 0 0.11">
                    <joint name="j7" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
                    <geom name="g7" type="box" size="0.035 0.06 0.02" pos="0 0 0.03"
                          rgba="0.30 0.45 0.60 1"/>
                    <body name="tool" pos="0 0 0.09">
                      <geom name="gtool" type="sphere" size="0.022"
                            rgba="0.85 0.45 0.10 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- Neighbouring links share a joint, so their geometry touches by design.
       Every real robot model excludes those pairs; otherwise the arm would be
       "in collision with itself" while standing perfectly still.  Note what
       is NOT excluded: link1-vs-link6, link2-vs-tool and so on.  Those are the
       self-collisions a planner genuinely has to avoid, and they stay on. -->
  <contact>
    <exclude body1="link0" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link6" body2="link7"/>
    <exclude body1="link7" body2="tool"/>
    <exclude body1="link0" body2="link2"/>
    <exclude body1="link2" body2="link4"/>
    <exclude body1="link4" body2="link6"/>
    <exclude body1="link6" body2="tool"/>
  </contact>
</mujoco>
"""


class Arm:
    """Joint limits, a collision oracle, and forward kinematics."""

    def __init__(self, xml=XML):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.lo = self.model.jnt_range[:, 0].copy()
        self.hi = self.model.jnt_range[:, 1].copy()
        self.tool_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "tool")
        self.n_checks = 0

    # -------------------------------------------------- collision
    def collides(self, q):
        """True if configuration q puts any geom pair in contact.

        `mj_kinematics` alone would place the bodies but not look for
        contacts; `mj_collision` alone would use stale body poses.  Both are
        needed, in that order, and neither of them integrates physics -- that
        is why this is fast enough to call a hundred thousand times.
        """
        self.n_checks += 1
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return self.data.ncon > 0

    def free(self, q):
        if np.any(q < self.lo) or np.any(q > self.hi):
            return False
        return not self.collides(q)

    def segment_free(self, a, b, res=0.05):
        """Check the straight joint-space segment a->b at spacing `res` rad.

        This is a DISCRETE check of a CONTINUOUS motion.  Everything between
        two sample points is assumed safe, which it is not.  Experiment 3
        measures how often that assumption is wrong.
        """
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = int(max(2, np.ceil(np.max(np.abs(b - a)) / res) + 1))
        for t in np.linspace(0.0, 1.0, n):
            if not self.free(a + t * (b - a)):
                return False
        return True

    def sample(self, rng):
        return self.lo + rng.random(self.nq) * (self.hi - self.lo)

    def sample_free(self, rng, tries=2000):
        for _ in range(tries):
            q = self.sample(rng)
            if self.free(q):
                return q
        raise RuntimeError("no free configuration found")

    # -------------------------------------------------- kinematics
    def tool_pos(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.tool_id].copy()

    def tool_jac(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.tool_id)
        return jacp

    def ik(self, target, q0, rng=None, iters=200, lam=0.08, tol=2e-3):
        """Damped least-squares IK for the tool position only (3 equations,
        7 unknowns -- so there are infinitely many answers, and which one you
        get depends entirely on where you start).  Same method as project 05.
        """
        q = np.asarray(q0, float).copy()
        for _ in range(iters):
            e = np.asarray(target, float) - self.tool_pos(q)
            if np.linalg.norm(e) < tol:
                break
            J = self.tool_jac(q)
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(np.asarray(target) - self.tool_pos(q)))


# ------------------------------------------------------------------ planners
def _steer(a, b, step):
    d = b - a
    n = np.linalg.norm(d)
    return b.copy() if n <= step else a + (step / n) * d


class _Tree:
    def __init__(self, root, cap=60000):
        self.pts = np.empty((cap, len(root)))
        self.pts[0] = root
        self.parent = [-1]
        self.n = 1

    def add(self, q, p):
        self.pts[self.n] = q
        self.parent.append(p)
        self.n += 1
        return self.n - 1

    def nearest(self, q):
        return int(np.argmin(np.linalg.norm(self.pts[:self.n] - q, axis=1)))

    def path_to(self, i):
        out = []
        while i != -1:
            out.append(self.pts[i].copy())
            i = self.parent[i]
        return out[::-1]


def rrt(arm, start, goal, rng, step=0.4, goal_bias=0.05, max_iters=20000,
        res=0.05):
    """Single-tree RRT in the arm's 7-dimensional joint space."""
    import time
    t0 = time.perf_counter()
    arm.n_checks = 0
    tree = _Tree(np.asarray(start, float))
    goal = np.asarray(goal, float)
    for it in range(max_iters):
        target = goal if rng.random() < goal_bias else arm.sample(rng)
        ni = tree.nearest(target)
        new = _steer(tree.pts[ni], target, step)
        if not arm.segment_free(tree.pts[ni], new, res):
            continue
        idx = tree.add(new, ni)
        if np.linalg.norm(new - goal) < step and arm.segment_free(new, goal, res):
            gi = tree.add(goal, idx)
            return tree, tree.path_to(gi), dict(
                found=True, iters=it + 1, nodes=tree.n, checks=arm.n_checks,
                time=time.perf_counter() - t0)
    return tree, None, dict(found=False, iters=max_iters, nodes=tree.n,
                            checks=arm.n_checks, time=time.perf_counter() - t0)


def _connect(arm, tree, q, step, res, max_steps=1000):
    """Grow `tree` toward q with repeated steps until it arrives or is blocked.

    This greedy inner loop is what makes RRT-*Connect* different from plain
    bidirectional RRT.  Plain RRT takes ONE step per sample; Connect keeps
    stepping in the same direction as long as it is legal, so a tree can cross
    an empty corridor in a single iteration instead of one node at a time.
    Returns ("reached" | "advanced" | "trapped", index of the last node).
    """
    i = tree.nearest(q)
    status = "trapped"
    for _ in range(max_steps):
        new = _steer(tree.pts[i], q, step)
        if not arm.segment_free(tree.pts[i], new, res):
            return status, i
        i = tree.add(new, i)
        if np.linalg.norm(new - q) < 1e-9:
            return "reached", i
        status = "advanced"
    return status, i


def rrt_connect(arm, start, goal, rng, step=0.4, max_iters=20000, res=0.05):
    """Bidirectional RRT-Connect.

    Two trees, one rooted at the start and one at the goal.  Each round: grow
    tree A one step toward a random sample, then let tree B *connect* greedily
    to whatever node A just made.  Then swap the roles.

    Why two trees rather than one that is twice as big?  Because the chance of
    a single tree hitting a specific goal configuration is tiny -- a goal is a
    point, and points have no volume.  The chance of two trees meeting is the
    chance of hitting each other's whole frontier, which is enormous by
    comparison.  Bidirectional search converts "hit this point" into "meet
    somewhere", and that is nearly all of the speed-up.
    """
    import time
    t0 = time.perf_counter()
    arm.n_checks = 0
    ta = _Tree(np.asarray(start, float))
    tb = _Tree(np.asarray(goal, float))
    a_is_start = True
    for it in range(max_iters):
        target = arm.sample(rng)
        ni = ta.nearest(target)
        new = _steer(ta.pts[ni], target, step)
        if arm.segment_free(ta.pts[ni], new, res):
            ia = ta.add(new, ni)
            status, ib = _connect(arm, tb, new, step, res)
            if status == "reached":
                pa = ta.path_to(ia)
                pb = tb.path_to(ib)
                path = pa + pb[::-1] if a_is_start else pb + pa[::-1]
                return (ta, tb), path, dict(
                    found=True, iters=it + 1, nodes=ta.n + tb.n,
                    checks=arm.n_checks, time=time.perf_counter() - t0)
        ta, tb = tb, ta
        a_is_start = not a_is_start
    return (ta, tb), None, dict(found=False, iters=max_iters,
                                nodes=ta.n + tb.n, checks=arm.n_checks,
                                time=time.perf_counter() - t0)


def path_cost(path):
    if path is None or len(path) < 2:
        return float("inf")
    p = np.asarray(path)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def path_free(arm, path, res=0.01):
    """Re-verify a whole path at a (much) finer resolution."""
    for a, b in zip(path[:-1], path[1:]):
        if not arm.segment_free(a, b, res):
            return False
    return True
