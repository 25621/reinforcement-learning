"""DAgger vs behaviour cloning: six experiments.

  1. the DAgger curve, round by round
  2. the control that matters: BC given the SAME number of expert labels
  3. the beta schedule -- who drives during data collection
  4. the covariate-shift gap, before and after
  5. does aggregation matter, or only the newest round?
  6. the cheap offline alternative (noisy demonstrations) at equal cost

Independent configurations are run in parallel processes, because each one is
a full train-rollout-retrain loop and there are twelve cores sitting idle.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402
import dagger as D         # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

ROUNDS = 4
EPS_PER_ROUND = 20
N_INIT = 25
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:20s} {key:38s} {value}  {unit}", flush=True)


# ---------------------------------------------------------------------------
# jobs (top-level so they can be shipped to worker processes)
# ---------------------------------------------------------------------------
def job_dagger(args):
    import torch
    torch.set_num_threads(2)
    beta, seed, aggregate = args
    rows, net, norm = D.run_dagger(n_init=N_INIT, rounds=ROUNDS,
                                   eps_per_round=EPS_PER_ROUND,
                                   beta=beta, seed=seed, aggregate=aggregate)
    gap = D.shift_gap(net, norm)
    return dict(kind="dagger", beta=beta, seed=seed, aggregate=aggregate,
                rows=rows, gap=gap)


def job_bc(args):
    import torch
    torch.set_num_threads(2)
    n_demos, seed, noise = args
    if noise == 0.0:
        res, net, norm = D.bc_baseline(n_demos, seed=seed)
    else:
        O, Y, _ = A.collect_demos(n_demos, seed=seed, side_mode=1, noise=noise)
        net, norm, hist = nets.train_bc(O, Y, epochs=350, seed=seed)
        ev = A.evaluate(nets.make_policy(net, norm), n=60, seed=999)
        res = dict(labels=len(O), success=ev["success"], err=ev["err"],
                   val=hist[-1][1])
    gap = D.shift_gap(net, norm)
    return dict(kind="bc", n_demos=n_demos, seed=seed, noise=noise,
                res=res, gap=gap)


def main():
    import torch
    torch.set_num_threads(2)

    dagger_jobs = [("zero", 0, True), ("zero", 1, True),
                   ("decay", 0, True), ("one", 0, True),
                   ("zero", 0, False)]
    bc_jobs = [(25, 0, 0.0), (50, 0, 0.0), (100, 0, 0.0), (150, 0, 0.0),
               (200, 0, 0.0), (100, 0, 0.15), (200, 0, 0.15)]

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        fut_d = [ex.submit(job_dagger, j) for j in dagger_jobs]
        fut_b = [ex.submit(job_bc, j) for j in bc_jobs]
        dag = [f.result() for f in fut_d]
        bcs = [f.result() for f in fut_b]
    print(f"[{time.time() - T0:6.1f}s] all jobs done", flush=True)

    by = {(d["beta"], d["seed"], d["aggregate"]): d for d in dag}

    # -- 1. the DAgger curve ------------------------------------------------
    main_runs = [by[("zero", 0, True)], by[("zero", 1, True)]]
    curve = np.array([[r["success"] for r in run["rows"]] for run in main_runs])
    labels = np.array([r["labels"] for r in main_runs[0]["rows"]])
    for i in range(ROUNDS + 1):
        record("1-dagger-curve", f"round {i}: success",
               f"{curve[:, i].mean():.3f} +- {curve[:, i].std():.3f}")
        record("1-dagger-curve", f"round {i}: expert labels", int(labels[i]))
    record("1-dagger-curve", "round 0 -> final gain",
           f"{curve[:, 0].mean():.3f} -> {curve[:, -1].mean():.3f}")
    coll = [r["collect_success"] for r in main_runs[0]["rows"][1:]]
    record("1-dagger-curve", "success DURING collection, round 1",
           round(float(coll[0]), 3))
    record("1-dagger-curve", "success DURING collection, last round",
           round(float(coll[-1]), 3))

    # -- 2. BC at the same label budget -------------------------------------
    bc_plain = sorted([b for b in bcs if b["noise"] == 0.0],
                      key=lambda b: b["res"]["labels"])
    bc_lab = np.array([b["res"]["labels"] for b in bc_plain])
    bc_suc = np.array([b["res"]["success"] for b in bc_plain])
    for b in bc_plain:
        record("2-bc-control", f"BC {b['n_demos']:3d} demos "
               f"({b['res']['labels']} labels): success", round(b["res"]["success"], 3))
    # interpolate BC at DAgger's final label count
    bc_at = float(np.interp(labels[-1], bc_lab, bc_suc))
    record("2-bc-control", f"BC interpolated at {labels[-1]} labels", round(bc_at, 3))
    record("2-bc-control", "DAgger at the same labels", round(float(curve[:, -1].mean()), 3))
    record("2-bc-control", "DAgger advantage at equal labels",
           round(float(curve[:, -1].mean()) - bc_at, 3))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].errorbar(range(ROUNDS + 1), curve.mean(0), yerr=curve.std(0),
                     marker="o", capsize=4, label="DAgger (beta=0)")
    axes[0].axhline(curve[:, 0].mean(), ls=":", c="gray", label="starting BC policy")
    axes[0].axhline(1.0, ls="--", c="g", label="expert")
    axes[0].set_xlabel("DAgger round")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Every round, the expert re-labels the mess")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].errorbar(labels, curve.mean(0), yerr=curve.std(0), marker="o",
                     capsize=4, label="DAgger")
    axes[1].plot(bc_lab, bc_suc, "s--", label="BC on fresh demos")
    nz = sorted([b for b in bcs if b["noise"] > 0], key=lambda b: b["res"]["labels"])
    axes[1].plot([b["res"]["labels"] for b in nz], [b["res"]["success"] for b in nz],
                 "^:", label="BC on noisy demos")
    axes[1].set_xlabel("expert labels used (the thing you pay for)")
    axes[1].set_ylabel("success rate")
    axes[1].set_title("Same budget, three ways to spend it")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dagger_curve.png"), dpi=110)
    plt.close(fig)

    # -- 3. beta schedule ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for beta, lab in [("zero", "beta=0 (policy drives)"),
                      ("decay", "beta=0.5^i (hand-over)"),
                      ("one", "beta=1 (expert drives = plain BC)")]:
        run = by[(beta, 0, True)]
        sc = [r["success"] for r in run["rows"]]
        record("3-beta", f"beta={beta}: final success", round(sc[-1], 3))
        record("3-beta", f"beta={beta}: final labels", int(run["rows"][-1]["labels"]))
        ax.plot(range(ROUNDS + 1), sc, "o-", label=lab)
    ax.set_xlabel("round")
    ax.set_ylabel("success rate")
    ax.set_title("Who should be holding the joystick?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "beta.png"), dpi=110)
    plt.close(fig)

    # -- 4. the shift gap ---------------------------------------------------
    g_bc = [b["gap"] for b in bc_plain if b["n_demos"] == 25][0]
    g_dag = by[("zero", 0, True)]["gap"]
    record("4-shift-gap", "BC 25 demos: MSE expert / policy states",
           f"{g_bc[0]:.4f} / {g_bc[1]:.4f}  ({g_bc[1] / g_bc[0]:.2f}x)")
    record("4-shift-gap", "after DAgger: MSE expert / policy states",
           f"{g_dag[0]:.4f} / {g_dag[1]:.4f}  ({g_dag[1] / g_dag[0]:.2f}x)")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.18, [g_bc[0], g_bc[1]], 0.36, label="BC (25 demos)")
    ax.bar(x + 0.18, [g_dag[0], g_dag[1]], 0.36, label="after DAgger")
    ax.set_xticks(x)
    ax.set_xticklabels(["on expert states", "on its own states"])
    ax.set_ylabel("action MSE")
    ax.set_title("DAgger does not fit the expert better -- it fits it where it matters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shift_gap.png"), dpi=110)
    plt.close(fig)

    # -- 5. aggregation ablation -------------------------------------------
    agg = by[("zero", 0, True)]
    noagg = by[("zero", 0, False)]
    record("5-aggregation", "keep all rounds: final success",
           round(agg["rows"][-1]["success"], 3))
    record("5-aggregation", "newest round only: final success",
           round(noagg["rows"][-1]["success"], 3))
    record("5-aggregation", "newest round only: final labels",
           int(noagg["rows"][-1]["labels"]))

    # -- 6. cost accounting -------------------------------------------------
    noisy = sorted([b for b in bcs if b["noise"] > 0],
                   key=lambda b: b["res"]["labels"])
    for b in noisy:
        record("6-cheap-alternative",
               f"noisy BC {b['n_demos']} demos ({b['res']['labels']} labels): success",
               round(b["res"]["success"], 3))
    record("6-cheap-alternative", "labels per success point, DAgger",
           round(float(labels[-1] / max(1e-9, curve[:, -1].mean()) / 100), 1))
    record("6-cheap-alternative", "labels per success point, BC",
           round(float(bc_lab[-1] / max(1e-9, bc_suc[-1]) / 100), 1))

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
