"""Find the planted defect, using only evidence a real team could collect.

  1. the symptom          -- the policy works in sim and not on the robot
  2. add-one-in           -- break the simulator one way at a time
  3. leave-one-out        -- fix the robot one way at a time
  4. the full factorial   -- main effects and interactions, all 8 robots
  5. probes that need no policy -- open-loop replay, and an oracle sensor
  6. acting on the diagnosis    -- three fixes, one of which is wrong

About 6 minutes on 12 cores.
"""

import csv
import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

# "spawn", not the Linux default "fork".  torch keeps a pool of worker threads
# alive, and forking a process that holds one of their locks deadlocks the
# child -- silently, with no error and no traceback, forever.  Spawn starts a
# clean interpreter instead.  This cost an hour to find.
MP = multiprocessing.get_context("spawn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gap
from gap import DEFECTS, evaluate, open_loop, train_policy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
NAMES = list(DEFECTS)
ALL = tuple(NAMES)
N_EVAL = 80
N_DEMOS = 400


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


def label(subset):
    return "+".join(s[0] for s in subset) if subset else "none"


# ---------------------------------------------------------------------------
# the policy under investigation
# ---------------------------------------------------------------------------
def build_policy():
    print("training the sim policy (%d demonstrations)..." % N_DEMOS)
    policy, meta = train_policy(N_DEMOS, seed=0)
    return policy


def _eval_subset(args):
    policy, subset = args
    return subset, evaluate(policy, subset, n=N_EVAL)


def eval_all(policy, subsets):
    with ProcessPoolExecutor(max_workers=8, mp_context=MP) as ex:
        return dict(ex.map(_eval_subset, [(policy, s) for s in subsets]))


# ---------------------------------------------------------------------------
# 1, 2, 3, 4 -- all of them read off one factorial sweep
# ---------------------------------------------------------------------------
def exp_factorial(policy):
    subsets = [tuple(c) for k in range(4)
               for c in itertools.combinations(NAMES, k)]
    res = eval_all(policy, subsets)
    sim, real = res[()], res[ALL]

    print("\n=== 1. the symptom " + "=" * 53)
    print("  in simulation : %.3f success   (mean final error %.1f mm)"
          % (sim["success"], 1e3 * sim["err"]))
    print("  on the robot  : %.3f success   (mean final error %.1f mm)"
          % (real["success"], 1e3 * real["err"]))
    print("  the gap       : %.3f" % (sim["success"] - real["success"]))
    record("symptom", sim=sim["success"], real=real["success"],
           gap=sim["success"] - real["success"])

    print("\n=== 2. add one defect to the simulator " + "=" * 33)
    print("  robot                    success    drop from sim")
    aoi = {}
    for name in NAMES:
        s = res[(name,)]["success"]
        aoi[name] = sim["success"] - s
        print("  sim + %-18s %7.3f %14.3f" % (name, s, aoi[name]))
        record("add_one_in", defect=name, success=s, drop=aoi[name])

    print("\n=== 3. remove one defect from the robot " + "=" * 32)
    print("  robot                    success    recovery from real")
    loo = {}
    for name in NAMES:
        rest = tuple(n for n in NAMES if n != name)
        s = res[rest]["success"]
        loo[name] = s - real["success"]
        print("  real - %-17s %7.3f %14.3f" % (name, s, loo[name]))
        record("leave_one_out", defect=name, success=s, recovery=loo[name])

    print("\n  the two rankings, side by side:")
    print("  defect              add-one-in drop   leave-one-out recovery")
    for name in NAMES:
        print("  %-20s %14.3f %20.3f" % (name, aoi[name], loo[name]))
    a_rank = sorted(NAMES, key=lambda n: -aoi[n])
    l_rank = sorted(NAMES, key=lambda n: -loo[n])
    print("  ranked by add-one-in    : %s" % " > ".join(a_rank))
    print("  ranked by leave-one-out : %s" % " > ".join(l_rank))
    print("  the two agree" if a_rank == l_rank else
          "  THE TWO DISAGREE -- which means the defects interact")
    record("ranking", add_one_in=" > ".join(a_rank),
           leave_one_out=" > ".join(l_rank), agree=a_rank == l_rank)

    print("\n=== 4. the full factorial: all 8 robots " + "=" * 32)
    print("  defects present        success")
    for s in subsets:
        print("  %-22s %7.3f" % (label(s), res[s]["success"]))
        record("factorial", subset=label(s), success=res[s]["success"],
               n_defects=len(s))
    print("\n  main effect of each defect (mean success with it minus without,")
    print("  averaged over every combination of the other two):")
    for name in NAMES:
        with_ = np.mean([res[s]["success"] for s in subsets if name in s])
        without = np.mean([res[s]["success"] for s in subsets if name not in s])
        print("  %-20s %+7.3f   (with %.3f, without %.3f)"
              % (name, with_ - without, with_, without))
        record("main_effect", defect=name, effect=with_ - without,
               with_it=with_, without_it=without)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))
    order = sorted(subsets, key=lambda s: -res[s]["success"])
    ax[0].barh([label(s) for s in order], [res[s]["success"] for s in order],
               color=["#2e7d32" if not s else
                      "#c62828" if len(s) == 3 else "#1976d2" for s in order])
    ax[0].set_xlabel("success rate"); ax[0].invert_yaxis()
    ax[0].set_title("all 8 robots (sim = none, robot = D+A+P)")
    ax[0].grid(alpha=.3, axis="x")
    x = np.arange(len(NAMES))
    ax[1].bar(x - 0.2, [aoi[n] for n in NAMES], 0.4, color="#c62828",
              label="add-one-in: drop from sim")
    ax[1].bar(x + 0.2, [loo[n] for n in NAMES], 0.4, color="#2e7d32",
              label="leave-one-out: recovery from real")
    ax[1].set_xticks(x); ax[1].set_xticklabels(NAMES, fontsize=8)
    ax[1].set_ylabel("change in success rate"); ax[1].legend(fontsize=8)
    ax[1].set_title("the two ablations disagree"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "factorial.png"), dpi=120)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# 5. probes that need no policy
# ---------------------------------------------------------------------------
def exp5_probes(policy):
    print("\n=== 5. two probes that do not involve the policy's decisions " + "=" * 11)
    r = evaluate(policy, (), n=12, seed=555, record_actions=True)
    acts = r["actions"]

    print("  A. open-loop replay: send the SAME actions to each robot")
    print("     robot                 tip divergence (mean / worst)")
    base = open_loop(acts, (), seed=555)
    for subset in [(), ("D dynamics",), ("A actuation",), ("P perception",), ALL]:
        tips = open_loop(acts, subset, seed=555)
        d = [np.linalg.norm(a - b, axis=1) for a, b in zip(base, tips)]
        mean = float(np.mean([x.mean() for x in d]))
        worst = float(np.max([x.max() for x in d]))
        print("     %-20s %8.2f mm %10.2f mm"
              % (label(subset), 1e3 * mean, 1e3 * worst))
        record("open_loop", subset=label(subset), mean_mm=1e3 * mean,
               worst_mm=1e3 * worst)
    print("     -> perception moves the tip by exactly 0: an open-loop replay")
    print("        is blind to it, which is what makes the probe useful.")


# ---------------------------------------------------------------------------
# 6. acting on the diagnosis
# ---------------------------------------------------------------------------
FIXES = [
    ("no fix (the sim policy)", "physics", {}),
    ("randomise the physics, widely", "physics",
     dict(randomize={"mass_scale": (0.8, 2.0), "damp_scale": (0.7, 2.6),
                     "gear": (0.7, 1.15), "latency": (0, 2)})),
    ("system ID: train on the real physics", "physics",
     dict(params_all=True)),
    ("randomise the camera offset", "perception", dict(perception_dr=0.030)),
    ("calibrate the camera", "perception", dict(calibrated=True)),
]


def _fix_job(args):
    kind, layer, kw = args
    kw = dict(kw)
    calibrated = kw.pop("calibrated", False)
    if kw.pop("params_all", False):
        kw["params"] = gap.params_for(ALL)
    pol, _ = train_policy(N_DEMOS, seed=0, **kw)
    return (kind, layer,
            evaluate(pol, ALL, n=N_EVAL, calibrated=calibrated)["success"],
            evaluate(pol, (), n=N_EVAL)["success"])


def exp6_fixes():
    print("\n=== 6. acting on the diagnosis " + "=" * 41)
    print("  The diagnosis said PERCEPTION.  Three of these five fixes are")
    print("  aimed at the physics, which is where a team without the")
    print("  forensics would have started.")
    with ProcessPoolExecutor(max_workers=5, mp_context=MP) as ex:
        out = list(ex.map(_fix_job, FIXES))
    print("  fix                                    layer      robot     sim")
    for kind, layer, real, sim in out:
        print("  %-38s %-10s %6.3f %7.3f" % (kind, layer, real, sim))
        record("fix", fix=kind, layer=layer, real_success=real,
               sim_success=sim)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh([o[0] for o in out], [o[2] for o in out],
            color=["#455a64" if o[1] == "physics" else "#2e7d32" for o in out])
    for y, o in enumerate(out):
        ax.text(o[2], y, " %.3f" % o[2], va="center", fontsize=8)
    ax.axvline(out[0][2], ls="--", color="#c62828", lw=1.2, label="no fix")
    ax.set_xlabel("success on the real robot"); ax.invert_yaxis()
    ax.set_title("grey = a fix aimed at the physics, green = at perception",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fixes.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    pol = build_policy()
    exp_factorial(pol)
    exp5_probes(pol)
    exp6_fixes()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
