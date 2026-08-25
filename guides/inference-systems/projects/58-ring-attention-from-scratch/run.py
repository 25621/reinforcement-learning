"""Project 58 -- Ring attention from scratch.

Four ranks, one long sequence, no rank ever holding more than its slice.

  A. Is it exact? Gather the ring's output and compare it against PyTorch's
     own fused attention over the whole sequence.
  B. What does it save? Peak score-matrix bytes and resident K/V per rank,
     against the single-device number.
  C. Does it scale? The same 8,192-token sequence at world size 1, 2 and 4.
  D. The imbalance nobody warns you about: causal masking makes contiguous
     slices unequal, and the zigzag layout fixes it. Measured as chunk-pairs
     of real work per rank, and as seconds.

The guide's version of this project says "4 GPUs". This machine's GPU
(compute capability 6.1) is not usable from this PyTorch build, so the four
ranks are four CPU processes talking over gloo on loopback. The algorithm,
the message pattern and the load imbalance are the same; only the wire is
slower, and every number below says which is which.

Usage:
    python3 run.py            # ~4 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SEQ = 8192
HEADS = 8
DIM = 64
TOTAL_THREADS = 12


def launch(world, layout, seq=SEQ, check=False, tag=""):
    """One torchrun of ring.py, returning every rank's JSON."""
    stem = os.path.join(OUT, f"rank_{layout}_{world}_{seq}{tag}")
    for f in glob.glob(stem + ".*.json"):
        os.remove(f)
    # Split the box's threads evenly. Giving every rank 12 threads would
    # oversubscribe the 12 cores 4x and turn a scaling study into a study of
    # the Linux scheduler.
    threads = max(1, TOTAL_THREADS // world)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(world),
           "--master_port", str(random.randrange(29500, 29900)),
           os.path.join(HERE, "ring.py"),
           "--seq", str(seq), "--heads", str(HEADS), "--dim", str(DIM),
           "--layout", layout, "--threads", str(threads), "--out", stem]
    if check:
        cmd.append("--check")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"torchrun failed for world={world} {layout}")
    ranks = []
    for f in sorted(glob.glob(stem + ".*.json")):
        with open(f) as fh:
            ranks.append(json.load(fh))
    for r in ranks:
        r["launch_s"] = time.perf_counter() - t0
    return ranks


def summarise(ranks):
    comp = [r["comp_s"] for r in ranks]
    done = [r["pairs_done"] for r in ranks]
    return {
        "world": ranks[0]["world"], "layout": ranks[0]["layout"],
        "seq": ranks[0]["seq"], "threads": ranks[0]["threads"],
        "wall_s": max(r["wall_s"] for r in ranks),
        "comp_s": comp, "comm_s": [r["comm_s"] for r in ranks],
        "pairs_done": done, "pairs_skipped": [r["pairs_skipped"] for r in ranks],
        "comm_mb": [r["comm_bytes"] / 1e6 for r in ranks],
        "peak_score_mb": ranks[0]["peak_score_mb"],
        "kv_resident_mb": ranks[0]["kv_resident_mb"],
        # Work efficiency: if every rank did the same amount, the fleet would
        # finish when the average rank finishes. It actually finishes when the
        # SLOWEST rank does, so mean/max is the fraction of the fleet's
        # capacity the layout manages to use.
        "balance": (sum(done) / len(done)) / max(done) if max(done) else 1.0,
        "time_balance": (sum(comp) / len(comp)) / max(comp),
        "max_abs_err": ranks[0].get("max_abs_err"),
        "ref_abs_mean": ranks[0].get("ref_abs_mean"),
        "full_score_mb": ranks[0].get("full_score_mb"),
    }


def measure():
    os.makedirs(OUT, exist_ok=True)
    res = {"seq": SEQ, "heads": HEADS, "dim": DIM,
           "total_threads": TOTAL_THREADS, "runs": []}

    print("== A/B. correctness and memory, world=4 ==")
    for layout in ("contiguous", "zigzag"):
        s = summarise(launch(4, layout, check=True))
        res["runs"].append(s)
        print(f"  {layout:11} max|err| {s['max_abs_err']:.2e} "
              f"(mean |out| {s['ref_abs_mean']:.3f})  "
              f"peak score/rank {s['peak_score_mb']:.1f} MB vs "
              f"{s['full_score_mb']:.0f} MB whole-sequence")
        print(f"              pairs/rank {s['pairs_done']}  "
              f"balance {s['balance']*100:.1f}%  "
              f"comp {['%.2f' % c for c in s['comp_s']]}  "
              f"wall {s['wall_s']:.2f}s", flush=True)

    print("\n== C. scaling, same 8,192-token sequence ==")
    for world in (1, 2, 4):
        s = summarise(launch(world, "zigzag", tag="_scale"))
        s["scaling"] = True
        res["runs"].append(s)
        print(f"  world={world} threads/rank={s['threads']}  "
              f"wall {s['wall_s']:.2f}s  "
              f"comm {sum(s['comm_mb']):.1f} MB  "
              f"peak score/rank {s['peak_score_mb']:.1f} MB", flush=True)

    print("\n== D. layout at every world size ==")
    # World size 4 is already measured above -- reuse it rather than paying
    # for the same two runs again.
    for r in res["runs"][:2]:
        r["layout_study"] = True
    for world in (2,):
        for layout in ("contiguous", "zigzag"):
            s = summarise(launch(world, layout, tag="_layout"))
            s["layout_study"] = True
            res["runs"].append(s)
    for s in [r for r in res["runs"] if r.get("layout_study")]:
        print(f"  world={s['world']} {s['layout']:11} pairs {s['pairs_done']} "
              f"balance {s['balance']*100:5.1f}%  "
              f"time-balance {s['time_balance']*100:5.1f}%  "
              f"wall {s['wall_s']:.2f}s", flush=True)

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = res["runs"]
    check = [r for r in runs if r.get("max_abs_err") is not None]
    scale = [r for r in runs if r.get("scaling")]
    lay = [r for r in runs if r.get("layout_study")]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    a = ax[0][0]
    r = check[0]
    vals = [r["full_score_mb"], r["peak_score_mb"]]
    a.bar(["one device\n(whole T x T)", "per rank\n(chunk x chunk)"], vals,
          color=["#c0392b", "#27ae60"], width=.5)
    a.set_yscale("log")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.1f} MB", ha="center", va="bottom")
    a.set_ylabel("peak attention-score bytes")
    a.set_title(f"B. Memory — {vals[0]/vals[1]:.0f}x smaller score block\n"
                f"exactness: max|err| {r['max_abs_err']:.1e} against "
                f"mean |out| {r['ref_abs_mean']:.2f}")

    a = ax[0][1]
    w = [r["world"] for r in scale]
    a.plot(w, [r["wall_s"] for r in scale], "o-", color="#c0392b",
           label="wall seconds")
    a.set_xlabel("ranks (12 cores split between them)")
    a.set_ylabel("seconds")
    a.set_xticks(w)
    a2 = a.twinx()
    a2.plot(w, [r["peak_score_mb"] for r in scale], "s--", color="#2980b9")
    a2.set_ylabel("peak score MB per rank", color="#2980b9")
    a.grid(alpha=.3)
    a.legend(fontsize=8, loc="upper center")
    a.set_title("C. Same sequence, more ranks — memory falls,\n"
                "time does not (one box, one memory bus)")

    a = ax[1][0]
    r4 = sorted([r for r in lay if r["world"] == 4],
                key=lambda r: r["layout"])
    x = np.arange(4)
    for r, off, col in ((r4[0], -.2, "#c0392b"), (r4[1], .2, "#27ae60")):
        a.bar(x + off, r["pairs_done"], .4, label=r["layout"], color=col)
    a.set_xticks(x, [f"rank {i}" for i in x])
    a.set_ylabel("chunk-pairs of real work")
    a.legend(fontsize=8)
    a.set_title(f"D. Causal imbalance — balance "
                f"{r4[0]['balance']*100:.0f}% vs {r4[1]['balance']*100:.0f}%")

    a = ax[1][1]
    labels, vals, cols = [], [], []
    for r in sorted(lay, key=lambda r: (r["world"], r["layout"])):
        labels.append(f"P={r['world']}\n{r['layout']}")
        vals.append(r["time_balance"] * 100)
        cols.append("#c0392b" if r["layout"] == "contiguous" else "#27ae60")
    a.bar(labels, vals, color=cols, width=.55)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.0f}%", ha="center", va="bottom")
    a.set_ylabel("mean / max compute time across ranks (%)")
    a.set_ylim(0, 110)
    a.set_title("D2. The same imbalance in seconds")

    fig.suptitle("Ring attention: exact, memory-bounded, and only balanced "
                 "if you assign the sequence carefully", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "ring.png"), dpi=120)
    print("wrote", os.path.join(OUT, "ring.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure())
        print(f"total {time.time()-t0:.0f}s")
