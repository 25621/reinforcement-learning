"""Fifty steps in a row: what p**N gets right, and what it gets wrong.

Six experiments:

1. the forecast -- measure per-stage success the way a unit test does, predict
   the fifty-stage success rate, then run fifty stages and compare
2. where the stages stop being identical -- success against stage index
3. where they stop being independent -- failure given the previous stage
4. recovery -- one retry, and what it is worth compared with a better policy
5. the horizon you can honestly claim, as a function of per-step reliability
6. long chains as a regression test: a 1 % per-step change, amplified

Run:  python3 run.py     (about 6 minutes; needs numpy, torch, matplotlib)
"""

import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402
import chain as C          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
N_CHAINS = 120
N_FRESH = 600
POOL = None


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<50s} {value:>10}   {note}")


def chains(make_policy, n=N_CHAINS, seed0=20000, **kw):
    return C.run_many(make_policy, n, seed0=seed0, pool=POOL, **kw)


def fresh(make_policy, n=N_FRESH, seed=5000, budget=C.STAGE_LEN):
    """Per-stage rate, split across the pool."""
    k = 12
    per = max(1, n // k)
    jobs = [(make_policy, seed + 991 * i, per, budget) for i in range(k)]
    if POOL is None:
        vals = [C._fresh_worker(j) for j in jobs]
    else:
        vals = list(POOL.map(C._fresh_worker, jobs, chunksize=1))
    return float(np.mean(vals))


def survival(rs, n_stages=C.N_STAGES):
    """Fraction of chains still alive after k stages, k = 0..N."""
    reached = np.array([r["reached"] for r in rs])
    return np.array([float((reached >= k).mean()) for k in range(n_stages + 1)])


def main():
    t0 = time.time()
    nets.seed_all(0)

    systems = {}
    for sig in (0.20, 0.30, 0.40):
        systems[f"expert s={sig:.2f}"] = C.NoisyExpert(sig, seed=3)
    print("\n[0] a cloned policy, as the realistic system")
    obs, act, _ = A.collect_demos(400, seed=0, noise=0.25)
    net, norm, _ = nets.train_bc(obs, act, epochs=150, seed=0)
    systems["cloned policy"] = C.ClonedPolicy(net, norm)

    # -- 1. the forecast -----------------------------------------------------
    print("\n[1] measured per-stage rate, forecast, and what actually happened")
    table = {}
    for name, mk in systems.items():
        p_fresh = fresh(mk)
        rs = chains(mk)
        obs_full = float(np.mean([r["complete"] for r in rs]))
        reached = float(np.mean([r["reached"] for r in rs]))
        # per-stage rate as seen INSIDE the chains
        att = sum(len(r["per_stage"]) for r in rs)
        won = sum(sum(r["per_stage"]) for r in rs)
        p_chain = won / att
        table[name] = dict(p_fresh=p_fresh, p_chain=p_chain, rs=rs,
                           obs=obs_full, reached=reached)
        record("forecast", f"{name}: fresh per-stage p", round(p_fresh, 4))
        record("forecast", f"{name}: p**50 forecast",
               round(p_fresh ** C.N_STAGES, 4))
        record("forecast", f"{name}: observed 50-stage success",
               round(obs_full, 4),
               f"mean stages reached {reached:.1f}/{C.N_STAGES}")
        record("forecast", f"{name}: in-chain per-stage p", round(p_chain, 4),
               f"forecast from it {p_chain ** C.N_STAGES:.4f}")

    # -- 2. are the stages identical? ---------------------------------------
    print("\n[2] success against stage index (are the stages the same task?)")
    key = "expert s=0.30"
    rs = table[key]["rs"]
    bins = [(0, 5), (5, 15), (15, 30), (30, 50)]
    for lo, hi in bins:
        num = den = 0
        for r in rs:
            for k in range(lo, min(hi, len(r["per_stage"]))):
                den += 1
                num += r["per_stage"][k]
        if den:
            record("drift", f"{key}: p over stages {lo}-{hi}",
                   round(num / den, 4), f"{den} stage attempts")

    # -- 3. are the stages independent? -------------------------------------
    print("\n[3] correlation between neighbouring stages")
    # If the stages really were independent coin flips, the number of stages a
    # chain survives would follow a GEOMETRIC distribution -- "how many heads
    # before the first tail" -- whose spread is fixed once you know p.  Any
    # extra spread beyond that is evidence that the chains are not exchangeable
    # with one another: some episodes were dealt an easy hand and some a hard
    # one, and the coin-flip model has no way to say so.
    lens = np.array([r["reached"] for r in table[key]["rs"]])
    record("independence", "stages reached: mean", round(float(lens.mean()), 2))
    record("independence", "stages reached: std", round(float(lens.std()), 2))
    p = table[key]["p_chain"]
    geo_std = float(np.sqrt(1 - p) / p)
    record("independence", "std a coin-flip model predicts",
           round(min(geo_std, C.N_STAGES), 2),
           "geometric distribution with the same p")

    # per-episode heterogeneity: split chains by their own difficulty
    het = []
    for r in table[key]["rs"]:
        het.append(r["reached"])
    het = np.array(het)
    record("independence", "chains that died in the first 5 stages",
           round(float((het < 5).mean()), 3))
    record("independence", "chains that finished all 50",
           round(float((het == C.N_STAGES).mean()), 3))

    # -- 4. recovery ---------------------------------------------------------
    print("\n[4] one retry versus a better policy")
    rec = {}
    for name, mk, rt in [("s=0.40, 0 retries", C.NoisyExpert(0.40, 3), 0),
                         ("s=0.40, 1 retry", C.NoisyExpert(0.40, 3), 1),
                         ("s=0.40, 2 retries", C.NoisyExpert(0.40, 3), 2),
                         ("s=0.30, 0 retries", C.NoisyExpert(0.30, 3), 0)]:
        rr = chains(mk, retries=rt)
        rec[name] = rr
        cost = float(np.mean([r["attempts"] for r in rr]))
        record("recovery", name, round(float(np.mean([r["complete"] for r in rr])), 3),
               f"stages reached {np.mean([r['reached'] for r in rr]):.1f}  "
               f"attempts {cost:.1f}")

    # -- 5. the horizon you may claim ---------------------------------------
    print("\n[5] the horizon arithmetic")
    for pp in (0.999, 0.99, 0.98, 0.95, 0.90, 0.80):
        n50 = np.log(0.5) / np.log(pp)
        record("arithmetic", f"p={pp}: stages at 50% chain success",
               int(n50), f"p**50 = {pp ** 50:.4f}")
    need = 0.5 ** (1.0 / C.N_STAGES)
    record("arithmetic", "per-stage p needed for 50% over 50 stages",
           round(need, 5))
    need99 = 0.99 ** (1.0 / C.N_STAGES)
    record("arithmetic", "per-stage p needed for 99% over 50 stages",
           round(need99, 6))

    # -- 6. the amplifier ----------------------------------------------------
    print("\n[6] a long chain as a regression test")
    base = C.NoisyExpert(0.20, seed=3)
    regressed = C.NoisyExpert(0.26, seed=3)      # a slightly shakier hand
    p_b = fresh(base)
    p_r = fresh(regressed)
    rb = chains(base, n=N_CHAINS, seed0=60000)
    rr = chains(regressed, n=N_CHAINS, seed0=60000)
    sb = float(np.mean([r["reached"] for r in rb]))
    sr = float(np.mean([r["reached"] for r in rr]))
    record("amplifier", "per-stage p: base -> regressed",
           f"{p_b:.4f} -> {p_r:.4f}", f"drop {p_b - p_r:.4f}")
    record("amplifier", "stages reached: base -> regressed",
           f"{sb:.1f} -> {sr:.1f}",
           f"relative drop {(sb - sr) / max(sb, 1e-9):.3f}")
    amp = ((sb - sr) / max(sb, 1e-9)) / max(p_b - p_r, 1e-9)
    record("amplifier", "amplification (relative chain drop / per-stage drop)",
           round(amp, 1))
    # how many episodes each test needs to see the same effect
    def n_needed(p1, p2, sd1=None):
        d = abs(p1 - p2)
        v = p1 * (1 - p1) + p2 * (1 - p2)
        return int(np.ceil(2 * (1.96 ** 2) * v / max(d * d, 1e-12) / 2))
    n_stage = n_needed(p_b, p_r)
    v = np.var([r["reached"] for r in rb]) + np.var([r["reached"] for r in rr])
    n_chain = int(np.ceil(2 * (1.96 ** 2) * v / max((sb - sr) ** 2, 1e-9) / 2))
    record("amplifier", "stage trials needed to see it (95%)", n_stage)
    record("amplifier", "chains needed to see it (95%)", n_chain,
           f"= {n_chain * sb:.0f} stage attempts")

    record("cost", "total runtime (s)", round(time.time() - t0, 1))

    # ---------------- figures ----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ks = np.arange(C.N_STAGES + 1)
    cols = {"expert s=0.20": "#264653", "expert s=0.30": "#2a9d8f",
            "expert s=0.40": "#e76f51", "cloned policy": "#d1495b"}
    for name, d in table.items():
        s = survival(d["rs"])
        ax[0].plot(ks, s, "-", lw=1.8, c=cols.get(name, "k"), label=name)
        ax[0].plot(ks, d["p_fresh"] ** ks, ":", lw=1.4, c=cols.get(name, "k"))
    ax[0].set_xlabel("stages completed")
    ax[0].set_ylabel("fraction of chains still alive")
    ax[0].set_title("solid = measured,  dotted = p**N forecast")
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(-0.02, 1.02)
    xs, ys = [], []
    for lo, hi in bins:
        num = den = 0
        for r in table[key]["rs"]:
            for k in range(lo, min(hi, len(r["per_stage"]))):
                den += 1
                num += r["per_stage"][k]
        if den:
            xs.append((lo + hi) / 2)
            ys.append(num / den)
    ax[1].plot(xs, ys, "o-", c="#2a9d8f")
    ax[1].axhline(table[key]["p_fresh"], ls="--", c="#457b9d",
                  label="fresh-reset p (the unit test)")
    ax[1].set_xlabel("stage index inside the chain")
    ax[1].set_ylabel("per-stage success")
    ax[1].set_title(f"{key}: the stages are not the same task")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "survival.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    names = list(rec.keys())
    vals = [float(np.mean([r["reached"] for r in rec[n]])) for n in names]
    costs = [float(np.mean([r["attempts"] for r in rec[n]])) for n in names]
    ax[0].bar(range(len(names)), vals, color="#2a9d8f")
    for i, (v_, c_) in enumerate(zip(vals, costs)):
        ax[0].text(i, v_ + 0.6, f"{v_:.1f}\n({c_:.0f} tries)", ha="center",
                   fontsize=8)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=7)
    ax[0].set_ylabel("stages reached out of 50")
    ax[0].set_title("recovery vs a steadier hand")
    ax[0].set_ylim(0, C.N_STAGES * 1.15)
    pp = np.linspace(0.80, 1.0, 400)
    for N in (10, 50, 200):
        ax[1].plot(pp, pp ** N, label=f"N = {N}")
    ax[1].axhline(0.5, ls=":", c="k", lw=0.9)
    ax[1].set_xlabel("per-step success p")
    ax[1].set_ylabel("chain success p**N")
    ax[1].set_title("why the last percent is the expensive one")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recovery.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    # spawn, not fork: torch keeps worker threads alive, and forking a process
    # that holds one of their locks hangs the child forever with no error.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count()),
                             mp_context=ctx) as ex:
        POOL = ex
        main()
