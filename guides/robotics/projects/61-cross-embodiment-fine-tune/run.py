"""Cross-embodiment pretraining and fine-tuning: six experiments.

  1. the zoo, and how badly a policy transfers between robots on its own
  2. sample efficiency: fine-tuning vs starting from scratch, demo by demo
  3. is it the DIVERSITY of robots or just the amount of data?
  4. does the policy need to be told which robot it is driving?
  5. fine-tune everything, or only the last layer?
  6. what happens once the target robot has plenty of data
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import embodiment as EMB   # noqa: E402
import ft                  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

N_PRE = 90            # demos per source robot
EVAL_N = 40
KS = [2, 5, 10, 25, 50, 200]
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:46s} {value}  {unit}", flush=True)


# ---------------------------------------------------------------------------
def job_single(name):
    """Train on one robot only; evaluate on every robot (experiment 1)."""
    torch.set_num_threads(2)
    O, Y, M, _ = EMB.collect(name, N_PRE, seed=0)
    net, norm = ft.train(O, Y, M, epochs=250, seed=0)
    pol = ft.policy_of(net, norm)
    return dict(kind="single", name=name,
                scores={n: EMB.evaluate(pol, n, n=EVAL_N)["success"] for n in EMB.ZOO})


def job_pretrain(cfg):
    torch.set_num_threads(2)
    net, norm, n_rows = ft.pretrain(cfg["sources"], n_demos=cfg["n_demos"],
                                    seed=0, drop_emb=cfg.get("drop_emb", False))
    pol = ft.policy_of(net, norm, drop_emb=cfg.get("drop_emb", False))
    zero = EMB.evaluate(pol, EMB.TARGET, n=EVAL_N)
    return dict(kind="pretrain", tag=cfg["tag"], rows=n_rows,
                state={k: v.clone() for k, v in net.state_dict().items()},
                norm=(norm.mu, norm.sd), zero_shot=zero["success"],
                on_sources={n: EMB.evaluate(pol, n, n=20)["success"]
                            for n in cfg["sources"]})


def job_finetune(cfg):
    torch.set_num_threads(2)
    O, Y, M, _ = EMB.collect(EMB.TARGET, cfg["k"], seed=500 + cfg["seed"])
    # A policy pretrained WITHOUT the embodiment vector must also be
    # fine-tuned and evaluated without it.  Handing it real link lengths in
    # those four slots at fine-tuning time feeds it inputs it has never seen,
    # and the resulting collapse says nothing about embodiment conditioning.
    drop = cfg.get("drop_emb", False)
    if drop:
        O = O.copy()
        O[:, -4:] = 0.0
    norm = None
    if cfg["init"] is not None:
        norm = ft.nets.Norm(np.zeros((1, O.shape[1]), np.float32))
        norm.mu, norm.sd = cfg["norm"]
    net, norm = ft.train(O, Y, M, epochs=cfg.get("epochs", 250), seed=cfg["seed"],
                         init=cfg["init"], norm=norm,
                         freeze_trunk=cfg.get("freeze_trunk", False),
                         lr=cfg.get("lr", 1e-3))
    ev = EMB.evaluate(ft.policy_of(net, norm, drop_emb=drop), EMB.TARGET, n=EVAL_N)
    return dict(kind="ft", tag=cfg["tag"], k=cfg["k"], seed=cfg["seed"],
                success=ev["success"], err=ev["err"], rows=len(O))


def main():
    torch.set_num_threads(2)
    ctx = __import__("multiprocessing").get_context("spawn")

    # -- 0. the zoo ---------------------------------------------------------
    for name in EMB.ZOO:
        arm = EMB.get_arm(name)
        record("0-zoo", f"{name}: links / reach",
               f"{np.round(arm.l, 2).tolist()} / {arm.reach:.2f} m")
    record("0-zoo", "expert success on the target robot",
           EMB.expert_score(EMB.TARGET, n=40))

    # -- 1. single-robot policies, evaluated everywhere ---------------------
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as ex:
        singles = list(ex.map(job_single, list(EMB.ZOO)))
    names = list(EMB.ZOO)
    mat = np.array([[s["scores"][n] for n in names] for s in singles])
    for i, s in enumerate(singles):
        record("1-transfer-matrix", f"trained on {s['name']}: own robot",
               round(s["scores"][s["name"]], 3))
        others = [v for n, v in s["scores"].items() if n != s["name"]]
        record("1-transfer-matrix", f"trained on {s['name']}: other robots (mean)",
               round(float(np.mean(others)), 3))
    record("1-transfer-matrix", "diagonal mean", round(float(np.mean(np.diag(mat))), 3))
    off = mat[~np.eye(len(names), dtype=bool)]
    record("1-transfer-matrix", "off-diagonal mean", round(float(off.mean()), 3))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("source-", "").replace("target-", "*")
                        for n in names], rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("source-", "").replace("target-", "*")
                        for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="w" if mat[i, j] < 0.6 else "k", fontsize=9)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(f"One robot's policy on another robot ({N_PRE} demos each)")
    fig.colorbar(im, label="success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_matrix.png"), dpi=110)
    plt.close(fig)

    # -- pretraining variants ----------------------------------------------
    pre_cfgs = [
        dict(tag="all4", sources=EMB.SOURCES, n_demos=N_PRE),
        dict(tag="one", sources=["source-A-even"], n_demos=N_PRE * 4),
        dict(tag="two", sources=EMB.SOURCES[:2], n_demos=N_PRE * 2),
        dict(tag="all4-noemb", sources=EMB.SOURCES, n_demos=N_PRE, drop_emb=True),
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as ex:
        pres = {p["tag"]: p for p in ex.map(job_pretrain, pre_cfgs)}
    for tag, p in pres.items():
        record("2-pretraining", f"{tag}: transitions", p["rows"])
        record("2-pretraining", f"{tag}: zero-shot on the target robot",
               round(p["zero_shot"], 3))
        record("2-pretraining", f"{tag}: mean success on its own source robots",
               round(float(np.mean(list(p["on_sources"].values()))), 3))

    # -- 3-6. fine-tuning jobs ---------------------------------------------
    jobs = []
    for k in KS:
        for s in (0, 1):
            jobs.append(dict(tag=f"scratch-k{k}", k=k, seed=s, init=None, norm=None))
            jobs.append(dict(tag=f"ft-k{k}", k=k, seed=s,
                             init=pres["all4"]["state"], norm=pres["all4"]["norm"]))
    for tag in ("one", "two", "all4-noemb"):
        for s in (0, 1):
            jobs.append(dict(tag=f"ft{tag}-k10", k=10, seed=s,
                             init=pres[tag]["state"], norm=pres[tag]["norm"],
                             drop_emb=(tag == "all4-noemb")))
    for s in (0, 1):
        jobs.append(dict(tag="ft-frozen-k10", k=10, seed=s, freeze_trunk=True,
                         init=pres["all4"]["state"], norm=pres["all4"]["norm"]))
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        fts = list(ex.map(job_finetune, jobs))
    R = {}
    for r in fts:
        R.setdefault(r["tag"], []).append(r["success"])
    print(f"[{time.time() - T0:6.1f}s] {len(fts)} fine-tuning runs finished", flush=True)

    def mean(tag):
        return float(np.mean(R[tag])), float(np.std(R[tag]))

    # -- 3. sample efficiency ----------------------------------------------
    sc_ft, sc_sc = [], []
    for k in KS:
        m1, s1 = mean(f"ft-k{k}")
        m0, s0 = mean(f"scratch-k{k}")
        sc_ft.append(m1)
        sc_sc.append(m0)
        record("3-sample-efficiency", f"{k:3d} target demos: fine-tuned",
               f"{m1:.3f} +- {s1:.3f}")
        record("3-sample-efficiency", f"{k:3d} target demos: from scratch",
               f"{m0:.3f} +- {s0:.3f}")
    # how many demos does scratch need to match fine-tuning at 10?
    target_level = sc_ft[KS.index(10)]
    if sc_sc[-1] >= target_level:
        need = float(np.interp(target_level, sc_sc, KS))
        record("3-sample-efficiency", "demos from scratch to match fine-tuning at 10",
               round(need, 1))
        record("3-sample-efficiency", "sample-efficiency multiplier",
               round(need / 10, 1), "x")
    else:
        record("3-sample-efficiency", "demos from scratch to match fine-tuning at 10",
               f"more than {KS[-1]} (scratch tops out at {sc_sc[-1]:.3f} "
               f"vs {target_level:.3f})")
        record("3-sample-efficiency", "sample-efficiency multiplier",
               f">{KS[-1] / 10:.0f}", "x")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(KS, sc_ft, "o-", label="fine-tuned from the 4-robot policy")
    ax.plot(KS, sc_sc, "s--", label="trained from scratch")
    ax.axhline(pres["all4"]["zero_shot"], c="gray", ls=":",
               label="pretrained policy, zero target demos")
    ax.set_xscale("log")
    ax.set_xlabel("demonstrations on the target robot")
    ax.set_ylabel("success rate on the target robot")
    ax.set_title("What pretraining on other robots is worth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sample_efficiency.png"), dpi=110)
    plt.close(fig)

    # -- 4. diversity vs volume; 5. embodiment vector; 6. freezing ---------
    bars = {}
    for label, tag in [("4 robots", "ft-k10"), ("2 robots", "fttwo-k10"),
                       ("1 robot,\nsame total data", "ftone-k10"),
                       ("4 robots,\nno embodiment vector", "ftall4-noemb-k10"),
                       ("4 robots,\nlast layer only", "ft-frozen-k10"),
                       ("no pretraining", "scratch-k10")]:
        m, s = mean(tag)
        bars[label] = m
        record("4-what-matters", f"{label.replace(chr(10), ' ')} (10 target demos)",
               f"{m:.3f} +- {s:.3f}")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(list(bars.keys()), list(bars.values()),
           color=["tab:green", "tab:blue", "tab:cyan", "tab:orange", "tab:purple",
                  "tab:red"])
    ax.set_ylabel("success on the target robot")
    ax.set_title("Ten demonstrations on the target robot, six starting points")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "what_matters.png"), dpi=110)
    plt.close(fig)

    # -- 6. plenty of data --------------------------------------------------
    m1, _ = mean("ft-k200")
    m0, _ = mean("scratch-k200")
    record("5-plenty-of-data", "200 demos: fine-tuned vs scratch",
           f"{m1:.3f} vs {m0:.3f}  (advantage {m1 - m0:+.3f})")
    m1s, _ = mean("ft-k5")
    m0s, _ = mean("scratch-k5")
    record("5-plenty-of-data", "5 demos: fine-tuned vs scratch",
           f"{m1s:.3f} vs {m0s:.3f}  (advantage {m1s - m0s:+.3f})")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
