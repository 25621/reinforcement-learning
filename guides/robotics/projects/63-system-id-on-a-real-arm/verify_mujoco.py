"""Check the hand-written dynamics and regressor against MuJoCo.

Everything in this project rests on one claim: torque is EXACTLY a linear
function of ten numbers, and ``dyn.regressor`` builds the matrix.  If that is
wrong, every fit below is fitting the wrong thing.  So we build the same arm in
MuJoCo, ask it for inverse dynamics at random states, and compare.

MuJoCo has no Coulomb-friction term we can match exactly (its ``frictionloss``
is a constraint, not a passive force), so the comparison runs with the friction
parameters set to zero.  The friction columns are two `tanh` terms we add
ourselves and can check by hand.
"""

import mujoco
import numpy as np

import dyn

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 -1 0" damping="{d1}" armature="{Ia1}"/>
      <inertial pos="{lc1} 0 0" mass="{m1}" diaginertia="1e-8 {I1} {I1}"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 {L1} 0 0" density="0"/>
      <body name="l2" pos="{L1} 0 0">
        <joint name="j2" type="hinge" axis="0 -1 0" damping="{d2}" armature="{Ia2}"/>
        <inertial pos="{lc2} 0 0" mass="{m2}" diaginertia="1e-8 {I2} {I2}"/>
        <geom type="capsule" size="0.018" fromto="0 0 0 {L2} 0 0" density="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def build():
    p = dict(dyn.TRUE_PHYS)
    p["f1"] = p["f2"] = 0.0
    m = mujoco.MjModel.from_xml_string(
        XML.format(L1=dyn.L1, L2=dyn.L2, **p))
    # links must not collide with each other: this is a pure-dynamics check
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    return m, p


def main():
    m, p = build()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst_tau, worst_acc = 0.0, 0.0
    th = dyn.to_theta(p)

    for _ in range(300):
        q = rng.uniform(-2.5, 2.5, 2)
        qd = rng.uniform(-4, 4, 2)
        qdd = rng.uniform(-20, 20, 2)

        d.qpos[:], d.qvel[:], d.qacc[:] = q, qd, qdd
        mujoco.mj_inverse(m, d)
        tau_mj = d.qfrc_inverse.copy()
        tau_ours = dyn.regressor(q, qd, qdd)[0] @ th
        worst_tau = max(worst_tau, np.abs(tau_mj - tau_ours).max())

        # and the other direction: same torque -> same acceleration
        d.qfrc_applied[:] = tau_mj
        d.qacc[:] = 0
        mujoco.mj_forward(m, d)
        worst_acc = max(worst_acc, np.abs(d.qacc - qdd).max())
        d.qfrc_applied[:] = 0

    print("regressor vs MuJoCo inverse dynamics : max |dtau| = %.3e N m" % worst_tau)
    print("forward dynamics round-trip          : max |dqdd| = %.3e rad/s^2" % worst_acc)

    # the friction columns, checked by hand at one state
    qd = np.array([1.0, -2.0])
    Y = dyn.regressor(np.zeros(2), qd, np.zeros(2))[0]
    print("friction column f1 at qd=+1.0 rad/s  : %.6f (want ~ +1)" % Y[0, 8])
    print("friction column f2 at qd=-2.0 rad/s  : %.6f (want ~ -1)" % Y[1, 9])
    assert worst_tau < 1e-9 and worst_acc < 1e-9, "the model does not match MuJoCo"
    print("\nOK")


if __name__ == "__main__":
    main()
