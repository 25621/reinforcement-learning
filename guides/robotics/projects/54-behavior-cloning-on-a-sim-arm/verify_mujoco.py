"""Is ``arm.py`` really a robot simulator, or just plausible-looking algebra?

Phase 8 replaces MuJoCo with hand-written equations because the learning
projects need to re-simulate from arbitrary states, thousands of times a
second, with the physics parameters changing underneath them.  That is a
speed win only if the equations are right, so this script builds the *same*
two-link arm in MuJoCo and compares joint accelerations on random states.

MuJoCo is used here as the referee, not as the engine -- the same role
project 02 gave it for URDF parsing.

Run:  python3 verify_mujoco.py
"""

import numpy as np
import mujoco

import arm as A

# Gravity is zeroed: the arm in this phase lies flat on a table.  Link inertia
# is given explicitly (uniform rod: centre at l/2, izz = m l^2 / 12) so that
# MuJoCo models exactly the body arm.py assumes, rather than something derived
# from a mesh.  (The two off-axis entries are set to half of izz purely to
# satisfy MuJoCo's A + B >= C check on a physically-consistent inertia; a
# planar arm only ever rotates about z, so they never enter the answer.)
XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0" integrator="Euler"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="{b1}"/>
      <inertial pos="{r1} 0 0" mass="{m1}" diaginertia="{h1} {h1} {i1}"/>
      <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="0.01" mass="0"/>
      <body name="link2" pos="{l1} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="{b2}"/>
        <inertial pos="{r2} 0 0" mass="{m2}" diaginertia="{h2} {h2} {i2}"/>
        <geom type="capsule" fromto="0 0 0 {l2} 0 0" size="0.01" mass="0"/>
        <site name="tip" pos="{l2} 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def build(a):
    xml = XML.format(l1=a.l[0], l2=a.l[1], r1=a.r[0], r2=a.r[1],
                     m1=a.m[0], m2=a.m[1], i1=a.I[0], i2=a.I[1],
                     b1=a.b[0], b2=a.b[1],
                     h1=a.I[0] / 2, h2=a.I[1] / 2)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def main():
    a = A.PlanarArm()
    model, data = build(a)
    rng = np.random.default_rng(0)

    worst_acc, worst_tip, worst_M = 0.0, 0.0, 0.0
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-6, 6, 2)
        tau = rng.uniform(-3, 3, 2)

        data.qpos[:] = q
        data.qvel[:] = qd
        data.ctrl[:] = 0
        data.qfrc_applied[:] = tau
        mujoco.mj_forward(model, data)

        worst_acc = max(worst_acc, np.abs(data.qacc - a.forward_dynamics(q, qd, tau)).max())
        worst_tip = max(worst_tip, np.abs(data.site_xpos[0][:2] - a.tip(q)).max())

        M = np.zeros((2, 2))
        mujoco.mj_fullM(model, data, M)  # dense mass matrix from the sparse one
        worst_M = max(worst_M, np.abs(M - a.mass_matrix(q)).max())

    # the readable loop version and the fast closed form must also agree
    worst_fast = 0.0
    for _ in range(500):
        q, qd = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-6, 6, 2)
        tau = rng.uniform(-3, 3, 2)
        generic = np.linalg.solve(a.mass_matrix(q),
                                  tau - a.rnea(q, qd, np.zeros(2)) - a.b * qd)
        worst_fast = max(worst_fast, np.abs(generic - a._fd2(q, qd, tau)).max())

    print(f"max |qacc  - MuJoCo| over 500 random states : {worst_acc:.3e} rad/s^2")
    print(f"max |M(q)  - MuJoCo| over 500 random states : {worst_M:.3e} kg m^2")
    print(f"max |tip   - MuJoCo| over 500 random states : {worst_tip:.3e} m")
    print(f"max |fast path - reference RNEA|            : {worst_fast:.3e} rad/s^2")
    ok = max(worst_acc, worst_M, worst_tip, worst_fast) < 1e-9
    print("VERDICT:", "identical to numerical precision" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
