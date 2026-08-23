"""Project 44 -- tensor parallelism (TP=2) from scratch.

  A. Correctness: sharded model vs whole model, logit by logit.
  B. The cost: decode step time and the all-reduce's share, batch 1..32,
     plus one 512-token prefill.
  C. The wire: gloo all-reduce latency vs payload size (why decode's
     collectives are pure latency, not bandwidth).
  D. Memory: what sharding actually saves per rank.

    python3 run.py           # ~4 minutes (launches torchrun with 2 ranks)
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def run_dist():
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "3")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=2", "--master_port=29617", "tp_run.py"],
        cwd=HERE, env=env, check=True)
    f = json.load(open(os.path.join(OUT, "findings_dist.json")))
    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(f, fh, indent=1)
    os.remove(os.path.join(OUT, "findings_dist.json"))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    # A. per-step logit difference
    d = f["correct"]["per_step_diff"]
    ax[0].plot(range(len(d)), d, color="#2471a3", lw=1.5)
    ax[0].axhline(f["correct"]["ref_logit_scale"], color="#c0392b", ls="--", lw=1,
                  label=f"mean |logit| = {f['correct']['ref_logit_scale']:.1f}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step (0 = prefill)")
    ax[0].set_ylabel("max |TP − single| (log)")
    ax[0].set_title(f"A. exact to fp32 noise\nargmax agree "
                    f"{f['correct']['argmax_agree']}/{f['correct']['steps']}")
    ax[0].legend(fontsize=8)

    # B. step time, single vs TP (comm stacked)
    rows = f["steps"]
    x = range(len(rows))
    ax[1].bar([i - .2 for i in x], [r["single_ms"] for r in rows], .4,
              color="#c0392b", label="single (6 threads)")
    ax[1].bar([i + .2 for i in x], [r["tp_ms"] - r["comm_ms"] for r in rows], .4,
              color="#2471a3", label="TP=2 compute")
    ax[1].bar([i + .2 for i in x], [r["comm_ms"] for r in rows], .4,
              bottom=[r["tp_ms"] - r["comm_ms"] for r in rows],
              color="#f39c12", label="TP=2 all-reduce")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels([r["batch"] for r in rows])
    ax[1].set_xlabel("batch size (ctx 256)")
    ax[1].set_ylabel("decode step, ms")
    ax[1].set_title("B. one box: TP=2 loses everywhere")
    ax[1].legend(fontsize=8)

    # C. all-reduce latency vs size
    m = f["allreduce_micro"]
    ax[2].plot([r["bytes"] for r in m], [r["lat_us"] for r in m],
               marker="o", ms=3, color="#2471a3")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    b1 = f["steps"][0]["payload_bytes"]
    ax[2].axvline(b1, color="#c0392b", ls="--", lw=1,
                  label=f"decode payload B=1 ({b1/1024:.1f} KiB)")
    ax[2].axhline(m[0]["lat_us"], color="k", ls=":", lw=1,
                  label=f"floor {m[0]['lat_us']:.0f} us")
    ax[2].set_xlabel("payload bytes")
    ax[2].set_ylabel("all-reduce, us (log)")
    ax[2].set_title("C. small collectives are all latency")
    ax[2].legend(fontsize=7)

    # D. comm share vs batch
    ax[3].plot([r["batch"] for r in rows], [100 * r["comm_share"] for r in rows],
               marker="o", color="#2471a3", label="decode (measured)")
    p = f["prefill_512"]
    ax[3].axhline(100 * p["comm_share"], color="#27ae60", ls="--",
                  label=f"prefill 512 ({100*p['comm_share']:.1f}%)")
    ax[3].set_xlabel("batch size")
    ax[3].set_ylabel("all-reduce share of step, %")
    ax[3].set_ylim(0, 16)
    ax[3].set_title("D. the comm share barely moves with batch")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tp2.png"), dpi=110)
    print("wrote outputs/tp2.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        run_dist()
    plot()
