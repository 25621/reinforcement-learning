"""Price every defect in the hobby URDF.

Five experiments, each one asking "what does this particular sloppy field
actually cost me?":

  1. what the numbers are:  URDF-as-imported vs. what the geometry implies
  2. gravity-compensation torque: the arm sags before it has done anything
  3. free swing: how differently the two robots move
  4. armature and the largest usable timestep
  5. mesh vs. capsule collision: simulation throughput

Runs in well under a minute.
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import migrate
from migrate import DEFECTS, LINKS, build_mjcf, merged_link3, principal_inertia

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
JOINTS = ["j_yaw", "j_shoulder", "j_elbow"]
ROWS = []


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


def load(defects=(), **kw):
    return mujoco.MjModel.from_xml_string(build_mjcf(defects, **kw))


def urdf_model():
    return mujoco.MjModel.from_xml_path(os.path.join(HERE, "hobby_arm.urdf"))


def no_contact(model):
    """Turn every geom into a decoration.

    Experiments 2 and 3 are about the *dynamics* numbers -- masses, inertias,
    damping.  If the links are allowed to touch each other the contact solver
    adds forces of its own and we would be measuring the collision model
    instead.  MuJoCo only excludes parent-child geom pairs automatically, so
    link1 and link3 collide happily; that was worth a debugging cycle.
    """
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model


# ---------------------------------------------------------------------------
# 1. what the importer actually gave us
# ---------------------------------------------------------------------------
def exp1_numbers():
    print("\n=== 1. the numbers, side by side " + "=" * 40)
    mu, mc = urdf_model(), load()
    print("bodies:  URDF import %d   clean MJCF %d" % (mu.nbody, mc.nbody))
    print("  URDF: %s" % [mujoco.mj_id2name(mu, mujoco.mjtObj.mjOBJ_BODY, i)
                          for i in range(mu.nbody)])
    print("  MJCF: %s" % [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_BODY, i)
                          for i in range(mc.nbody)])
    gone = {mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mc.nbody)} \
        - {mujoco.mj_id2name(mu, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mu.nbody)}
    print("  frames the weld destroyed: %s" % sorted(gone))
    record("weld", urdf_bodies=mu.nbody, mjcf_bodies=mc.nbody,
           lost_frames=";".join(sorted(gone)))

    print("\n  link      mass    URDF Izz     true Izz      ratio   COM offset")
    for name in ("link1", "link2", "link3"):
        iu = mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_BODY, name)
        ic = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, name)
        # compare the largest principal inertia, which is what a swing feels
        iu_v, ic_v = mu.body_inertia[iu].max(), mc.body_inertia[ic].max()
        com_err = np.linalg.norm(mu.body_ipos[iu] - mc.body_ipos[ic])
        print("  %-8s %5.2f  %10.3g  %11.3g  %8.0fx   %5.1f mm"
              % (name, mc.body_mass[ic], iu_v, ic_v, iu_v / ic_v, 1e3 * com_err))
        record("inertia", link=name, mass=float(mc.body_mass[ic]),
               urdf_inertia=float(iu_v), true_inertia=float(ic_v),
               ratio=float(iu_v / ic_v), com_error_mm=float(1e3 * com_err))


# ---------------------------------------------------------------------------
# 2. gravity compensation: the torque needed to hold still
# ---------------------------------------------------------------------------
GPOSE = np.array([0.0, 0.6, -0.8])      # arm out and a bit down


def gravity_torque(model, q):
    """Torque each motor must produce to hold the arm frozen at q.

    mj_inverse answers "what forces produce this acceleration?".  Set velocity
    and acceleration to zero and the answer is exactly the gravity load.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = q
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_inverse(model, d)
    return d.qfrc_inverse.copy()


def exp2_gravity():
    print("\n=== 2. gravity-compensation torque at a fixed pose " + "=" * 22)
    models = {"clean": no_contact(load()),
              "URDF as imported": no_contact(urdf_model()),
              "clean + com": no_contact(load({"com"})),
              "clean + inertia": no_contact(load({"inertia"}))}
    tc = gravity_torque(models["clean"], GPOSE)
    print("  model              shoulder (N m)   elbow (N m)   shoulder error")
    for name, m in models.items():
        t = gravity_torque(m, GPOSE)
        err = 100 * abs(t[1] - tc[1]) / abs(tc[1])
        print("  %-18s %12.4f  %12.4f   %10.1f %%" % (name, t[1], t[2], err))
        record("gravity", model=name, shoulder_Nm=float(t[1]),
               elbow_Nm=float(t[2]), shoulder_err_pct=float(err))
    print("  -> statics depend on mass and COM, not on inertia: the inertia")
    print("     defect scores 0 % here and is invisible to a static check.")


# ---------------------------------------------------------------------------
# 3. free swing: let go and watch
# ---------------------------------------------------------------------------
Q0 = np.array([0.0, 0.0, 0.0])          # arm straight out, horizontal
SWING_T = 2.0


def swing(model, T=SWING_T):
    d = mujoco.MjData(model)
    d.qpos[:] = Q0
    n = int(T / model.opt.timestep)
    ts, qs = [], []
    for k in range(n):
        mujoco.mj_step(model, d)
        if k % 5 == 0:
            ts.append(d.time)
            qs.append(d.qpos.copy())
    return np.array(ts), np.array(qs)


def fall_time(t, q, level=0.5):
    """When does the shoulder first get half a radian away from horizontal?"""
    hit = np.nonzero(np.abs(q[:, 1]) > level)[0]
    return float(t[hit[0]]) if len(hit) else float("nan")


def exp3_swing():
    print("\n=== 3. free swing under gravity " + "=" * 40)
    models = {"clean": no_contact(load()),
              "URDF as imported": no_contact(urdf_model())}
    for defect in DEFECTS:
        if defect == "mesh":           # no contacts here, so mesh is a no-op
            continue
        models["clean + " + defect] = no_contact(load({defect}))

    t_ref, q_ref = swing(models["clean"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    print("  model                      falls 0.5 rad in   settles at   RMS gap")
    for name, m in models.items():
        t, q = swing(m)
        n = min(len(q), len(q_ref))
        rms = float(np.sqrt(np.mean((q[:n, 1] - q_ref[:n, 1]) ** 2)))
        tf = fall_time(t, q)
        print("  %-26s %8.3f s %+12.3f rad %8.3f" % (name, tf, q[-1, 1], rms))
        record("swing", model=name, fall_time_s=tf,
               shoulder_2s=float(q[-1, 1]), rms_vs_clean=rms)
        style = dict(lw=2.4 if name == "clean" else 1.4,
                     ls="-" if "+" not in name else "--")
        ax[0].plot(t, q[:, 1], label=name, **style)
        ax[1].plot(t, q[:, 2], label=name, **style)
    for a, ttl in zip(ax, ("shoulder j_shoulder", "elbow j_elbow")):
        a.set_xlabel("time (s)"); a.set_ylabel("angle (rad)"); a.set_title(ttl)
        a.grid(alpha=.3)
    ax[1].legend(fontsize=7, loc="best")
    fig.suptitle("Let go of a horizontal arm: one URDF, several sets of numbers")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "free_swing.png"), dpi=120)
    plt.close(fig)

    # a headline bar chart: which single defect moves the robot most?
    rows = [r for r in ROWS if r["experiment"] == "swing" and r["model"] != "clean"]
    rows.sort(key=lambda r: r["rms_vs_clean"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = [r["model"].replace("clean + ", "") for r in rows]
    vals = [r["rms_vs_clean"] for r in rows]
    cols = ["#455a64" if r["model"] == "URDF as imported" else "#1976d2" for r in rows]
    ax.barh(names, vals, color=cols)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, "%.2f" % v, va="center", fontsize=8)
    ax.set_xlabel("RMS shoulder-angle gap vs. the clean model over 2 s (rad)")
    ax.set_title("What each single sloppy field costs")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "defect_cost.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. armature and the largest timestep that still works
# ---------------------------------------------------------------------------
def stable(model, kp=(300.0, 300.0, 150.0), T=1.0):
    """Run a stiff position servo and report whether the sim survives."""
    d = mujoco.MjData(model)
    d.qpos[:] = np.array([0.3, 0.4, -0.5])
    target = np.zeros(3)
    kp = np.array(kp)
    kd = 2 * np.sqrt(kp) * 0.1
    for _ in range(int(T / model.opt.timestep)):
        d.ctrl[:] = np.clip(kp * (target - d.qpos) - kd * d.qvel, -20, 20)
        mujoco.mj_step(model, d)
        if not np.all(np.isfinite(d.qpos)) or np.abs(d.qvel).max() > 1e3:
            return False
    return np.abs(d.qpos - target).max() < 0.5


def exp4_timestep():
    print("\n=== 4. largest usable timestep " + "=" * 41)
    steps = [0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.012, 0.020, 0.030, 0.050]
    grid = {}
    configs = (("clean (armature 0.01)", ()),
               ("no armature", {"armature"}),
               ("no armature, unit inertia", {"armature", "inertia"}))
    for label, defects in configs:
        row = []
        for dt in steps:
            m = no_contact(load(defects, timestep=dt, integrator="Euler"))
            row.append(stable(m))
        grid[label] = row
        biggest = max([dt for dt, ok in zip(steps, row) if ok], default=0.0)
        print("  %-20s largest stable dt = %6.4f s  (%d Hz)"
              % (label, biggest, round(1 / biggest) if biggest else 0))
        record("timestep", config=label, max_dt=biggest,
               rate_hz=round(1 / biggest) if biggest else 0)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    for i, (label, row) in enumerate(grid.items()):
        ax.scatter(steps, [i] * len(steps), s=170,
                   c=["#2e7d32" if ok else "#c62828" for ok in row],
                   marker="s")
    ax.set_yticks(range(len(grid))); ax.set_yticklabels(list(grid), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("timestep (s)")
    ax.set_title("green = the stiff servo survives, red = the sim blows up")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timestep_stability.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. mesh vs. capsule collision throughput
# ---------------------------------------------------------------------------
def throughput(model, n=6000):
    d = mujoco.MjData(model)
    d.qpos[:] = np.array([0.0, 1.9, 0.9])
    for _ in range(200):
        mujoco.mj_step(model, d)
    ncon = 0
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, d)
        ncon += d.ncon
    el = time.perf_counter() - t0
    return n / el, ncon / n


def exp5_collision():
    print("\n=== 5. collision geometry and throughput " + "=" * 31)
    counts = {L["name"]: sum(1 for _ in open(
        os.path.join(migrate.MESH_DIR, L["name"] + ".obj")) if _.startswith("v "))
        for L in LINKS}
    print("  mesh vertices: %s" % counts)
    for label, defects in (("capsules + box", ()), ("collision = visual mesh", {"mesh"})):
        # three repeats, median: a single timing run on a busy laptop is noise
        runs = [throughput(load(defects, floor=True)) for _ in range(3)]
        sps = float(np.median([r[0] for r in runs]))
        ncon = float(np.median([r[1] for r in runs]))
        print("  %-24s %8.0f steps/s   %.2f contacts/step" % (label, sps, ncon))
        record("collision", config=label, steps_per_s=float(sps),
               contacts_per_step=float(ncon))
    a = [r for r in ROWS if r["experiment"] == "collision"]
    print("  -> the mesh costs %.2fx of the throughput"
          % (a[0]["steps_per_s"] / a[1]["steps_per_s"]))


if __name__ == "__main__":
    migrate.make_meshes()
    open(os.path.join(HERE, "arm_clean.mjcf"), "w").write(build_mjcf())
    exp1_numbers()
    exp2_gravity()
    exp3_swing()
    exp4_timestep()
    exp5_collision()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader()
        w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
