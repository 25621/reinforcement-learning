"""Project 01 -- Transform calculator.

Seven experiments, all on the same library (``transforms.py``):

  1. round-trip 10,000 uniformly random rotations through every conversion path
  2. the textbook axis-angle formula vs a robust one, as a function of angle
  3. gimbal lock: what roll-pitch-yaw loses near pitch = +-90 degrees
  4. the quaternion double cover, and the one-line fix
  5. interpolation: slerp vs Euler-lerp vs straight-line matrix lerp
  6. numerical drift when you compose a million rotations
  7. timing: are quaternions actually faster than matrices?

Runs in well under a minute on a CPU.
"""

import os
import time

import numpy as np

import transforms as tf
from plot_style import COLORS, save, use_style

import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<46s} {value:>12.3e} {unit}")


# ---------------------------------------------------------------------------
# 1. Round-trip accuracy
# ---------------------------------------------------------------------------
def exp1_roundtrip(n=10000, seed=0):
    print("[1] round-tripping", n, "uniformly random rotations")
    rng = np.random.default_rng(seed)
    Rs = tf.random_Rs(n, rng)

    paths = {
        "R -> quat -> R": lambda R: tf.quat_to_R(tf.R_to_quat(R)),
        "R -> axis-angle -> R": lambda R: tf.axis_angle_to_R(tf.R_to_axis_angle(R)),
        "R -> rpy -> R": lambda R: tf.rpy_to_R(tf.R_to_rpy(R)),
        "R -> T -> T^-1 -> R": lambda R: tf.T_inv(tf.T_inv(tf.T_from_Rp(R, np.zeros(3))))[:3, :3],
        "R -> se3 log -> exp -> R": lambda R: tf.se3_exp(tf.se3_log(tf.T_from_Rp(R, np.zeros(3))))[:3, :3],
        "R -> quat -> aa -> quat -> R": lambda R: tf.quat_to_R(
            tf.axis_angle_to_quat(tf.quat_to_axis_angle(tf.R_to_quat(R)))
        ),
    }

    errs = {}
    for name, fn in paths.items():
        # Two ways to score the same thing.  The GEODESIC error is the honest
        # physical one ("by what angle did the rotation move?").  The
        # ELEMENTWISE one ("how far did any single matrix entry move?") is what
        # a unit test usually checks.  They agree here to within a factor of 2.
        out = [fn(R) for R in Rs]
        e = np.array([tf.rot_geodesic(R, R2) for R, R2 in zip(Rs, out)])
        e_el = np.array([np.abs(R - R2).max() for R, R2 in zip(Rs, out)])
        errs[name] = np.maximum(e, e_el)
        record("roundtrip", f"max geodesic error [{name}]", e.max(), "rad")
        record("roundtrip", f"max elementwise error [{name}]", e_el.max(), "")

    # A full SE(3) round trip with translation as well.
    Ts = [tf.random_T(rng, 1.5) for _ in range(1000)]
    e_se3 = np.array([np.abs(tf.se3_exp(tf.se3_log(T)) - T).max() for T in Ts])
    errs["T -> se3 log -> exp -> T"] = e_se3
    record("roundtrip", "max error [SE(3) log/exp, incl. translation]", e_se3.max(), "m")

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    names = list(errs)
    maxes = [max(errs[k].max(), 1e-18) for k in names]
    ax.barh(names, maxes, color=COLORS[0], height=0.6)
    ax.axvline(1e-10, color=COLORS[1], ls="--", lw=1.5)
    ax.text(1.3e-10, -0.45, "project target 1e-10", color=COLORS[1], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("worst error over 10,000 random rotations (rad, or m)")
    ax.set_title("Every conversion path round-trips at machine precision")
    ax.invert_yaxis()
    save(fig, f"{OUT}/roundtrip.png")
    return errs


# ---------------------------------------------------------------------------
# 2. Robust vs naive axis-angle extraction
# ---------------------------------------------------------------------------
def exp2_logmap(seed=1):
    print("[2] naive vs robust axis-angle extraction, swept over the angle")
    rng = np.random.default_rng(seed)
    thetas = np.concatenate(
        [
            np.logspace(-9, -1, 60),
            np.linspace(0.1, np.pi - 0.1, 60),
            np.pi - np.logspace(-9, -1, 60)[::-1],
        ]
    )
    del rng
    axis = np.array([0.3, -0.7, 0.65])
    axis /= np.linalg.norm(axis)

    e_naive, e_robust = [], []
    for th in thetas:
        R = tf.axis_angle_to_R(th * axis)
        for fn, store in ((tf.R_to_axis_angle_naive, e_naive), (tf.R_to_axis_angle, e_robust)):
            r = fn(R)
            # Error measured as the rotation angle between truth and recovery.
            store.append(tf.rot_geodesic(R, tf.axis_angle_to_R(r)))
    e_naive = np.array(e_naive)
    e_robust = np.array(e_robust)

    record("logmap", "naive worst error", e_naive.max(), "rad")
    record("logmap", "robust worst error", e_robust.max(), "rad")
    record("logmap", "naive error at theta = pi - 1e-9", e_naive[-1], "rad")
    record("logmap", "robust error at theta = pi - 1e-9", e_robust[-1], "rad")

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.minimum(thetas, np.pi - thetas)  # distance to the nearest bad point
    ax.loglog(x, np.maximum(e_naive, 1e-18), ".", ms=4, color=COLORS[1], label="naive: arccos + 1/sin(theta)")
    ax.loglog(x, np.maximum(e_robust, 1e-18), ".", ms=4, color=COLORS[0], label="robust: via quaternion + atan2")
    ax.set_xlabel("distance of the rotation angle from 0 or from pi  (rad)")
    ax.set_ylabel("recovered-vs-true angle error (rad)")
    ax.set_title("The textbook axis-angle formula loses 8 digits near 0 and near pi")
    ax.legend(loc="upper right")
    save(fig, f"{OUT}/logmap_error.png")


# ---------------------------------------------------------------------------
# 3. Gimbal lock
# ---------------------------------------------------------------------------
def exp3_gimbal():
    print("[3] gimbal lock near pitch = 90 degrees")
    dp = np.logspace(-9, -1, 90)
    pitches = np.pi / 2 - dp
    roll, yaw = 0.4, -1.1

    rpy_err, R_err = [], []
    for p in pitches:
        rpy0 = np.array([roll, p, yaw])
        R = tf.rpy_to_R(rpy0)
        rpy1 = tf.R_to_rpy(R)
        rpy_err.append(np.abs(rpy1 - rpy0).max())  # do the ANGLES come back?
        R_err.append(tf.rot_geodesic(R, tf.rpy_to_R(rpy1)))  # does the ROTATION?

    rpy_err, R_err = np.array(rpy_err), np.array(R_err)
    record("gimbal", "worst roll/pitch/yaw angle error near lock", rpy_err.max(), "rad")
    record("gimbal", "worst rotation error near lock", R_err.max(), "rad")

    # The degeneracy itself: at pitch = 90 deg, changing roll and yaw together
    # by equal and opposite amounts leaves the rotation unchanged.
    Ra = tf.rpy_to_R([0.0, np.pi / 2, 0.0])
    Rb = tf.rpy_to_R([0.8, np.pi / 2, 0.8])
    record("gimbal", "rotation distance between (0,90,0) and (0.8,90,0.8)", tf.rot_geodesic(Ra, Rb), "rad")

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.loglog(dp, np.maximum(rpy_err, 1e-18), color=COLORS[1], label="error in the recovered ANGLES")
    ax.loglog(dp, np.maximum(R_err, 1e-18), color=COLORS[0], label="error in the recovered ROTATION")
    ax.set_xlabel("distance of pitch from 90 degrees (rad)")
    ax.set_ylabel("error")
    ax.set_title("Gimbal lock: the rotation survives, the three numbers do not")
    ax.legend()
    save(fig, f"{OUT}/gimbal_lock.png")


# ---------------------------------------------------------------------------
# 4. The quaternion double cover
# ---------------------------------------------------------------------------
def exp4_double_cover(n=4000, seed=2):
    print("[4] the double cover: q and -q are the same rotation")
    rng = np.random.default_rng(seed)
    qa = tf.random_quats(n, rng)
    qb = tf.random_quats(n, rng)
    # Randomly flip the sign of half of them -- a real system does this all the
    # time, because a solver, a filter or a network has no reason to prefer one.
    flip = rng.random(n) < 0.5
    qb_flipped = np.where(flip[:, None], -qb, qb)

    true_ang = np.array([tf.rot_geodesic(tf.quat_to_R(a), tf.quat_to_R(b)) for a, b in zip(qa, qb)])
    naive = np.linalg.norm(qa - qb_flipped, axis=1)  # what a careless loss does
    fixed_q = np.where((np.sum(qa * qb_flipped, axis=1) < 0)[:, None], -qb_flipped, qb_flipped)
    fixed = np.linalg.norm(qa - fixed_q, axis=1)

    # Correlation with the true geodesic distance tells the story numerically.
    record("double_cover", "corr(naive ||qa-qb||, true angle)", float(np.corrcoef(naive, true_ang)[0, 1]))
    record("double_cover", "corr(sign-fixed ||qa-qb||, true angle)", float(np.corrcoef(fixed, true_ang)[0, 1]))
    worst = true_ang[np.argmin(naive)]
    record("double_cover", "true angle of the 'closest' pair under naive distance", worst, "rad")

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    axs[0].plot(true_ang, naive, ".", ms=2, color=COLORS[1])
    axs[0].set_title("naive  ||q_a - q_b||")
    axs[0].set_xlabel("true rotation angle between them (rad)")
    axs[0].set_ylabel("reported distance")
    axs[1].plot(true_ang, fixed, ".", ms=2, color=COLORS[0])
    axs[1].set_title("after aligning the sign  (dot < 0  ->  negate)")
    axs[1].set_xlabel("true rotation angle between them (rad)")
    fig.suptitle("Ignoring the double cover splits one curve into two branches", y=1.02)
    save(fig, f"{OUT}/double_cover.png")


# ---------------------------------------------------------------------------
# 5. Interpolation
# ---------------------------------------------------------------------------
def exp5_interpolation():
    print("[5] slerp vs Euler-lerp vs straight-line matrix lerp")
    rpy_a = np.array([0.0, 0.0, 0.0])
    rpy_b = np.array([2.6, 1.2, -2.8])
    Ra, Rb = tf.rpy_to_R(rpy_a), tf.rpy_to_R(rpy_b)
    qa, qb = tf.R_to_quat(Ra), tf.R_to_quat(Rb)

    ts = np.linspace(0.0, 1.0, 201)
    R_slerp = [tf.quat_to_R(tf.slerp(qa, qb, t)) for t in ts]
    R_euler = [tf.rpy_to_R((1 - t) * rpy_a + t * rpy_b) for t in ts]
    R_lerp_raw = [(1 - t) * Ra + t * Rb for t in ts]
    R_lerp = [tf.orthonormalize(M) for M in R_lerp_raw]

    def angular_speed(seq):
        dt = ts[1] - ts[0]
        return np.array([tf.rot_geodesic(seq[i], seq[i + 1]) / dt for i in range(len(seq) - 1)])

    s_slerp, s_euler, s_lerp = angular_speed(R_slerp), angular_speed(R_euler), angular_speed(R_lerp)
    for nm, s in (("slerp", s_slerp), ("euler-lerp", s_euler), ("matrix-lerp", s_lerp)):
        record("interp", f"{nm}: angular-speed max/min ratio", float(s.max() / s.min()))
    defect = np.array([tf.so3_defect(M) for M in R_lerp_raw])
    record("interp", "worst ||R^T R - I|| of the raw matrix lerp", defect.max())

    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axs[0].plot(ts[:-1], s_slerp, color=COLORS[0], label="slerp")
    axs[0].plot(ts[:-1], s_euler, color=COLORS[1], label="lerp on roll/pitch/yaw")
    axs[0].plot(ts[:-1], s_lerp, color=COLORS[2], label="lerp on the matrix, then re-orthonormalise")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("angular speed (rad per unit t)")
    axs[0].set_title("Only slerp turns at a constant rate")
    axs[0].legend(fontsize=7)
    axs[1].plot(ts, defect, color=COLORS[2])
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$\|R^\top R - I\|_F$")
    axs[1].set_title("A straight line between two rotations\nleaves the set of rotations")
    save(fig, f"{OUT}/interpolation.png")


# ---------------------------------------------------------------------------
# 6. Drift under repeated composition
# ---------------------------------------------------------------------------
def exp6_drift(n=200000, seed=3):
    print("[6] composing", n, "small rotations: how fast does each form drift?")
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=1e-3, size=(n, 3))
    R = np.eye(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_renorm = q.copy()
    ks, dR, dq, dqn = [], [], [], []
    for i, s in enumerate(steps):
        dR_i = tf.axis_angle_to_R(s)
        dq_i = tf.axis_angle_to_quat(s)
        R = R @ dR_i
        q = tf.quat_mul(q, dq_i)
        q_renorm = tf.quat_normalize(tf.quat_mul(q_renorm, dq_i))
        if (i + 1) % 2000 == 0:
            ks.append(i + 1)
            dR.append(tf.so3_defect(R))
            dq.append(abs(np.linalg.norm(q) - 1.0))
            dqn.append(abs(np.linalg.norm(q_renorm) - 1.0))

    record("drift", "matrix ||R^T R - I|| after 200k compositions", dR[-1])
    record("drift", "quaternion |‖q‖-1| after 200k compositions (no renorm)", dq[-1])
    record("drift", "quaternion |‖q‖-1| after 200k compositions (renormalised)", dqn[-1])

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.loglog(ks, np.maximum(dR, 1e-20), color=COLORS[1], label=r"matrix: $\|R^\top R - I\|_F$")
    ax.loglog(ks, np.maximum(dq, 1e-20), color=COLORS[0], label=r"quaternion: $|\,\|q\|-1|$")
    ax.loglog(ks, np.maximum(dqn, 1e-20), color=COLORS[2], label="quaternion, renormalised each step")
    ax.set_xlabel("number of composed rotations")
    ax.set_ylabel("distance from a valid rotation")
    ax.set_title("Both forms drift; one divide per step removes it")
    ax.legend(fontsize=8)
    save(fig, f"{OUT}/drift.png")


# ---------------------------------------------------------------------------
# 7. Timing
# ---------------------------------------------------------------------------
def exp7_timing(n=200000, seed=4):
    print("[7] timing: matrix composition vs quaternion composition")
    rng = np.random.default_rng(seed)
    Rs = tf.random_Rs(300, rng)
    qs = np.array([tf.R_to_quat(R) for R in Rs])

    t0 = time.perf_counter()
    A = np.eye(3)
    for i in range(n):
        A = A @ Rs[i % 300]
    t_mat = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    a = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(n):
        a = tf.quat_mul(a, qs[i % 300])
    t_quat = (time.perf_counter() - t0) / n

    record("timing", "one 3x3 @ 3x3 matrix composition", t_mat * 1e9, "ns")
    record("timing", "one quaternion product (numpy arrays)", t_quat * 1e9, "ns")
    record("timing", "quaternion speed-up (>1 means quaternions win)", t_mat / t_quat)
    record("timing", "floating-point multiplies, matrix product", 27, "mults")
    record("timing", "floating-point multiplies, quaternion product", 16, "mults")
    record("timing", "bytes to store one rotation, matrix", 9 * 8, "B")
    record("timing", "bytes to store one rotation, quaternion", 4 * 8, "B")


def main():
    exp1_roundtrip()
    exp2_logmap()
    exp3_gimbal()
    exp4_double_cover()
    exp5_interpolation()
    exp6_drift()
    exp7_timing()

    import csv

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
