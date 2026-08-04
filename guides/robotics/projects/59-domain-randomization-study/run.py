"""Domain randomisation: train on many robots so one unseen robot works.

The setup is project 54's push task and project 54's cloned policy, with one
change: during data collection the simulator's physics is redrawn every
episode -- link masses, joint damping, motor strength, control latency.  The
policy is never told which robot it is on, so the only way to score well across
all of them is to behave in a way that does not depend on knowing.

  1. how badly does a policy trained on one robot degrade on another?
  2. randomised vs nominal, on a held-out fleet -- and what it costs at home
  3. which knob is worth randomising?
  4. how wide should the ranges be?
  5. measure the robot first, then randomise narrowly around it
  6. outside the training range, where the promise runs out

Every robot in the study is a robot the simulator can actually integrate.  The
servo gains are fixed while the physics changes, so a very light link with very
high damping produces a decay rate faster than the 200 Hz time step can follow
and the arm diverges -- not a hard robot, a broken one.  The ranges below keep
the worst-case rate (an eigenvalue of M^-1 B) under about 2 / dt.

Note on the demonstrator: it is a feedback controller, so it works on every
robot in the fleet without being retuned -- that is what makes randomised
demonstrations possible at all.  Its success rate does drop on the extreme
robots, and failed demonstrations are discarded, so a very wide range quietly
trains on an easier subset than it claims.  Experiment 4 is where that shows up.
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

N_DEMOS = 100
EPOCHS = 350
EVAL_N = 40

RANGES = {
    "nominal": {},
    "narrow": dict(mass_scale=(0.9, 1.1), gear=(0.95, 1.05),
                   damp_scale=(0.9, 1.1)),
    "medium": dict(mass_scale=(0.7, 1.8), gear=(0.75, 1.3),
                   damp_scale=(0.6, 1.8), latency=(0, 1)),
    "wide": dict(mass_scale=(0.55, 3.0), gear=(0.55, 1.6),
                 damp_scale=(0.4, 2.5), latency=(0, 2)),
    "extreme": dict(mass_scale=(0.5, 6.0), gear=(0.3, 2.5),
                    damp_scale=(0.35, 2.5), latency=(0, 4)),
    "mass-only": dict(mass_scale=(0.55, 3.0)),
    "gear-only": dict(gear=(0.55, 1.6)),
    "damp-only": dict(damp_scale=(0.4, 2.5)),
    "latency-only": dict(latency=(0, 2)),
    # experiment 5: the robot was measured first, so randomise a little
    # around the measured value instead of over everything imaginable
    "system-id": dict(mass_scale=(1.7, 2.1), gear=(0.62, 0.78),
                      damp_scale=(2.0, 2.4)),
}

# The fleet the policies are graded on.  The first block sits inside the "wide"
# training range; the second block is deliberately outside every range.
FLEET_IN = [
    dict(mass_scale=1.0),
    dict(mass_scale=0.6), dict(mass_scale=2.5),
    dict(gear=0.6), dict(gear=1.5),
    dict(damp_scale=0.45), dict(damp_scale=2.2),
    dict(latency=1), dict(latency=2),
    dict(mass_scale=1.9, gear=0.7, damp_scale=2.2),     # the "real" robot
]
FLEET_OUT = [
    dict(mass_scale=4.0), dict(gear=0.4), dict(gear=2.0), dict(latency=3),
]
NOMINAL = FLEET_IN[0]        # mass_scale=1.0 is the default robot
REAL_ROBOT = FLEET_IN[-1]

ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:44s} {value}  {unit}", flush=True)


def label(p):
    if not p:
        return "nominal"
    return ", ".join(f"{k.replace('_scale', '')}={v}" for k, v in p.items())


def job(cfg):
    """Train one policy under one randomisation range, grade it on the fleet."""
    torch.set_num_threads(2)
    name, seed = cfg["name"], cfg["seed"]
    rand = RANGES[name]
    O, Ac, meta = collect(N_DEMOS, seed=seed, randomize=rand)
    net, norm, hist = nets.train_bc(O, Ac, epochs=EPOCHS, seed=seed)
    pol = nets.make_policy(net, norm)
    scores = {}
    for p in FLEET_IN + FLEET_OUT:
        scores[label(p)] = A.evaluate(pol, n=EVAL_N, seed=999, params=p)["success"]
    return dict(name=name, seed=seed, scores=scores, demo_yield=meta["n"] / meta["tries"])


def collect(n_demos, seed, randomize):
    """Demonstrations, optionally on a fresh robot every episode."""
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, randomize=randomize)
    O, Ac, ok, tries = [], [], 0, 0
    while ok < n_demos and tries < n_demos * 6:
        tries += 1
        r = A.rollout(env, None, side=1, record=True)
        if not r["success"]:
            continue
        O.append(r["obs"])
        Ac.append(r["act"])
        ok += 1
    return (np.concatenate(O).astype(np.float32),
            np.concatenate(Ac).astype(np.float32),
            dict(n=ok, tries=tries))


def main():
    torch.set_num_threads(2)

    # -- 0. the demonstrator on a randomised fleet --------------------------
    # The expert's own score on each robot is the CEILING for any policy cloned
    # from it.  Two of the fleet robots are genuinely hard for a feedback
    # controller (a very light arm oscillates under fixed gains; a delayed one
    # chases its own past), so a low policy score there is not evidence about
    # randomisation at all.
    for p in FLEET_IN + FLEET_OUT:
        rng = np.random.default_rng(3)
        env = A.PushEnv(rng, params=p)
        sc = np.mean([A.rollout(env, None, side=1)["success"] for _ in range(30)])
        record("0-demonstrator", f"expert ceiling on [{label(p)}]", round(float(sc), 3))
    for name in ("nominal", "wide", "extreme"):
        rng = np.random.default_rng(5)
        env = A.PushEnv(rng, randomize=RANGES[name])
        sc = np.mean([A.rollout(env, None, side=1)["success"] for _ in range(60)])
        record("0-demonstrator", f"expert success under '{name}' randomisation",
               round(float(sc), 3))

    jobs = [dict(name=n, seed=s) for n in RANGES for s in (0, 1)]
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        res = list(ex.map(job, jobs))
    R = {}
    for r in res:
        R.setdefault(r["name"], []).append(r)
    print(f"[{time.time() - T0:6.1f}s] {len(res)} policies trained", flush=True)

    def score(name, p):
        return float(np.mean([r["scores"][label(p)] for r in R[name]]))

    def mean_over(name, fleet):
        return float(np.mean([score(name, p) for p in fleet]))

    # -- 1. the transfer gap ------------------------------------------------
    for p in FLEET_IN:
        record("1-transfer-gap", f"nominal-trained on [{label(p)}]",
               round(score("nominal", p), 3))
    record("1-transfer-gap", "nominal-trained: mean over the in-range fleet",
           round(mean_over("nominal", FLEET_IN), 3))
    record("1-transfer-gap", "nominal-trained: on its own robot",
           round(score("nominal", NOMINAL), 3))

    # -- 2. randomised vs nominal ------------------------------------------
    for name in ("nominal", "medium", "wide"):
        record("2-randomised-vs-nominal", f"{name}: mean over in-range fleet",
               round(mean_over(name, FLEET_IN), 3))
        record("2-randomised-vs-nominal", f"{name}: on the nominal robot",
               round(score(name, NOMINAL), 3))
    prem = score("nominal", NOMINAL) - score("wide", NOMINAL)
    record("2-randomised-vs-nominal", "premium paid on the nominal robot",
           round(prem, 3))
    record("2-randomised-vs-nominal", "gain on the fleet",
           round(mean_over("wide", FLEET_IN) - mean_over("nominal", FLEET_IN), 3))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = [label(p) for p in FLEET_IN + FLEET_OUT]
    x = np.arange(len(labels))
    for i, (name, c) in enumerate([("nominal", "tab:red"), ("medium", "tab:orange"),
                                   ("wide", "tab:green")]):
        ax.bar(x + (i - 1) * 0.27, [score(name, p) for p in FLEET_IN + FLEET_OUT],
               0.27, label=f"trained with {name} randomisation", color=c)
    ax.axvline(len(FLEET_IN) - 0.5, c="k", ls="--", lw=1)
    ax.text(len(FLEET_IN) - 0.4, 1.02, "outside every training range", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("success rate")
    ax.set_title("The same task on fourteen different robots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fleet.png"), dpi=110)
    plt.close(fig)

    # -- 3. which axis matters ---------------------------------------------
    axis_scores = {}
    for name in ("nominal", "mass-only", "gear-only", "damp-only", "latency-only",
                 "wide"):
        axis_scores[name] = mean_over(name, FLEET_IN)
        record("3-which-axis", f"{name}: mean over in-range fleet",
               round(axis_scores[name], 3))
    # what each axis is worth on ITS OWN test robots
    per_axis_tests = {"mass-only": [dict(mass_scale=0.6), dict(mass_scale=2.5)],
                      "gear-only": [dict(gear=0.6), dict(gear=1.5)],
                      "damp-only": [dict(damp_scale=0.45), dict(damp_scale=2.2)],
                      "latency-only": [dict(latency=1), dict(latency=2)]}
    for name, tests in per_axis_tests.items():
        record("3-which-axis", f"{name}: on the robots it was aimed at",
               f"{mean_over(name, tests):.3f} (nominal-trained: "
               f"{mean_over('nominal', tests):.3f})")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(list(axis_scores.keys()), list(axis_scores.values()),
           color=["tab:red"] + ["tab:blue"] * 4 + ["tab:green"])
    ax.set_ylabel("mean success over the fleet")
    ax.set_title("Randomising one thing at a time")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "axes.png"), dpi=110)
    plt.close(fig)

    # -- 4. how wide? -------------------------------------------------------
    widths = ["nominal", "narrow", "medium", "wide", "extreme"]
    inr = [mean_over(w, FLEET_IN) for w in widths]
    outr = [mean_over(w, FLEET_OUT) for w in widths]
    for w, a, b in zip(widths, inr, outr):
        yld = float(np.mean([r["demo_yield"] for r in R[w]]))
        record("4-range-width", f"{w}: fleet / outside-range / usable demos",
               f"{a:.3f} / {b:.3f} / {yld:.2f}")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(widths, inr, "o-", label="held-out fleet (inside the wide range)")
    ax.plot(widths, outr, "s--", label="robots outside every range")
    ax.plot(widths, [score(w, NOMINAL) for w in widths], "^:", label="the nominal robot")
    ax.set_ylabel("success rate")
    ax.set_title("Wider is better, until it is not")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "width.png"), dpi=110)
    plt.close(fig)

    # -- 5. system identification ------------------------------------------
    for name in ("nominal", "wide", "system-id"):
        record("5-system-id", f"{name}: on the real robot {label(REAL_ROBOT)}",
               round(score(name, REAL_ROBOT), 3))
    record("5-system-id", "system-id vs wide, on the real robot",
           round(score("system-id", REAL_ROBOT) - score("wide", REAL_ROBOT), 3))
    record("5-system-id", "system-id: mean over the whole fleet",
           round(mean_over("system-id", FLEET_IN), 3))

    # -- 6. outside the range ----------------------------------------------
    for p in FLEET_OUT:
        record("6-outside-the-range", f"[{label(p)}] nominal / wide / extreme",
               f"{score('nominal', p):.3f} / {score('wide', p):.3f} / "
               f"{score('extreme', p):.3f}")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
