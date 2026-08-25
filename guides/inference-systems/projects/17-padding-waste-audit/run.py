"""Project 17 -- padding waste audit.

Static batching pads. This project puts a number on it.

  A. One concrete batch of 8, taken apart: how many token-positions are
     filler, and how much arithmetic do they cost?
  B. Batch size is the main dial. Sweep it.
  C. Is it the batch size or the *spread* of lengths? Sweep the spread with
     the batch size fixed.
  D. Length bucketing -- sort the queue by length before batching. How much of
     the waste does that recover, for free?
  E. Reality check: does saving X% of FLOPs save X% of time?

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))

import batchlib  # noqa: E402
from batchlib import make_workload  # noqa: E402
from padding import audit_group, audit_trace  # noqa: E402
from schedulers import run_static  # noqa: E402

F = {}


def synth_lens(n, seed, p_med=64, p_sigma=0.75, o_med=32, o_sigma=0.7,
               p_max=192, o_max=72):
    """(prompt_len, out_len) pairs from the same lognormal as project 16."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = int(min(p_max, max(8, rng.lognormvariate(math.log(p_med), p_sigma))))
        o = int(min(o_max, max(4, rng.lognormvariate(math.log(o_med), o_sigma))))
        out.append((p, o))
    return out


# ---------------------------------------------------------------------------


def section_a(runner):
    print("\n=== A. one batch of 8, taken apart ===")
    lens = synth_lens(8, seed=3)
    pre, dec = audit_group(lens, runner.flops_per_token)
    print(f"  lengths (prompt, output): {lens}")
    print(f"  prefill : {pre.pad_slots:5d} pad / {pre.pad_slots + pre.real_slots:5d} slots"
          f"  = {pre.slot_frac:.1%} slots, {pre.flop_frac:.1%} FLOPs")
    print(f"  decode  : {dec.pad_slots:5d} pad / {dec.pad_slots + dec.real_slots:5d} slots"
          f"  = {dec.slot_frac:.1%} slots, {dec.flop_frac:.1%} FLOPs")
    tot = pre + dec
    print(f"  TOTAL   : {tot.slot_frac:.1%} of slots, {tot.flop_frac:.1%} of FLOPs are padding")
    F["A"] = {
        "lengths": lens,
        "prefill": {"pad_slots": pre.pad_slots, "slots": pre.pad_slots + pre.real_slots,
                    "slot_frac": round(pre.slot_frac, 4), "flop_frac": round(pre.flop_frac, 4),
                    "pad_tflops": round(pre.pad_flops / 1e12, 3)},
        "decode": {"pad_slots": dec.pad_slots, "slots": dec.pad_slots + dec.real_slots,
                   "slot_frac": round(dec.slot_frac, 4), "flop_frac": round(dec.flop_frac, 4),
                   "pad_tflops": round(dec.pad_flops / 1e12, 3)},
        "total_slot_frac": round(tot.slot_frac, 4),
        "total_flop_frac": round(tot.flop_frac, 4),
        "decode_share_of_all_pad_flops": round(
            dec.pad_flops / (dec.pad_flops + pre.pad_flops), 4),
    }


def section_b(runner):
    print("\n=== B. batch-size sweep (256 requests) ===")
    lens = synth_lens(256, seed=11)
    rows = []
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        pre, dec = audit_trace(lens, bs, runner.flops_per_token)
        tot = pre + dec
        rows.append({"batch": bs,
                     "prefill_pad_flop_frac": round(pre.flop_frac, 4),
                     "decode_pad_flop_frac": round(dec.flop_frac, 4),
                     "total_pad_flop_frac": round(tot.flop_frac, 4),
                     "total_pad_slot_frac": round(tot.slot_frac, 4)})
        print(f"  batch {bs:3d}: prefill pad {pre.flop_frac:6.1%}   decode pad "
              f"{dec.flop_frac:6.1%}   overall {tot.flop_frac:6.1%}")
    F["B"] = rows


def section_c(runner):
    print("\n=== C. spread sweep (batch 8 fixed) ===")
    rows = []
    for sig in [0.0, 0.2, 0.4, 0.7, 1.0, 1.3]:
        lens = synth_lens(256, seed=11, o_sigma=sig, p_sigma=sig, o_max=10_000,
                          p_max=10_000)
        pre, dec = audit_trace(lens, 8, runner.flops_per_token)
        tot = pre + dec
        omax = max(o for _, o in lens)
        omed = sorted(o for _, o in lens)[128]
        rows.append({"sigma": sig, "out_median": omed, "out_max": omax,
                     "total_pad_flop_frac": round(tot.flop_frac, 4)})
        print(f"  sigma {sig:.1f}: output median {omed:4d} max {omax:5d}"
              f"  ->  pad {tot.flop_frac:6.1%}")
    F["C"] = rows


def section_d(runner):
    print("\n=== D. length bucketing: which length do you sort by? ===")
    lens = synth_lens(256, seed=11)
    fpt = runner.flops_per_token
    base = audit_trace(lens, 8, fpt)
    base_tot = base[0] + base[1]
    base_f = base_tot.real_flops + base_tot.pad_flops
    rows = []
    for name, sort, window, honest in [
            ("arrival order", "arrival", None, True),
            ("sort by PROMPT len, window 32", "prompt", 32, True),
            ("sort by PROMPT len, whole trace", "prompt", None, False),
            ("sort by OUTPUT len (oracle)", "output", None, False),
            ("sort by prompt+output (oracle)", "both", None, False)]:
        pre, dec = audit_trace(lens, 8, fpt, sort=sort, window=window)
        tot = pre + dec
        f = tot.real_flops + tot.pad_flops
        rows.append({"policy": name, "implementable": honest,
                     "prefill_pad": round(pre.flop_frac, 4),
                     "decode_pad": round(dec.flop_frac, 4),
                     "total_pad": round(tot.flop_frac, 4),
                     "flops_saved": round(1 - f / base_f, 4)})
        print(f"  {name:32s}: prefill pad {pre.flop_frac:6.1%}  decode pad "
              f"{dec.flop_frac:6.1%}  total {tot.flop_frac:6.1%}"
              f"  FLOPs {-(1 - f / base_f):+.1%}")
    F["D"] = rows


def section_e(runner, tok):
    print("\n=== E. do the saved FLOPs turn into saved time? ===")
    reqs = make_workload(tok, n=32, rate=100.0, seed=5)   # all present at t=0
    out = {}
    arms = [("arrival order", None),
            ("sorted by prompt len", lambda r: r.prompt_len),
            ("sorted by output len (oracle)", lambda r: r.max_new)]
    for name, key in arms:
        rs = copy.deepcopy(reqs)
        if key:
            rs.sort(key=key)
            for i, r in enumerate(rs):
                r.arrive = i * 1e-6           # keep the sorted order stable
        run_static(runner, rs, batch_size=8)
        c = runner.counters
        out[name] = {
            "model_s": round(c.model_s, 2),
            "pad_flop_frac": round(c.pad_flop_frac, 4),
            "total_tflops": round((c.useful_flops + c.pad_flops) / 1e12, 2),
            "forward_passes": c.forward_passes,
        }
        print(f"  {name:24s}: model time {out[name]['model_s']:6.2f}s   "
              f"TFLOPs {out[name]['total_tflops']:6.2f}   pad {c.pad_flop_frac:.1%}")
    a = out["arrival order"]
    F["E"] = {"runs": out, "deltas": {}}
    for name, _ in arms[1:]:
        b = out[name]
        d = {
            "flops_saved": round(1 - b["total_tflops"] / a["total_tflops"], 4),
            "time_saved": round(1 - b["model_s"] / a["model_s"], 4),
            "passes_saved": round(1 - b["forward_passes"] / a["forward_passes"], 4),
        }
        F["E"]["deltas"][name] = d
        print(f"  vs arrival order, {name:30s}: FLOPs {-d['flops_saved']:+.1%}"
              f"   time {-d['time_saved']:+.1%}   passes {-d['passes_saved']:+.1%}")


# ---------------------------------------------------------------------------


def write_csv():
    with open(os.path.join(OUT, "batch_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["B"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(F["B"])
    with open(os.path.join(OUT, "spread_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["C"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(F["C"])


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    b = f["B"]
    ax[0].plot([r["batch"] for r in b], [r["total_pad_flop_frac"] * 100 for r in b],
               "o-", color="#c0392b", label="all FLOPs")
    ax[0].plot([r["batch"] for r in b], [r["decode_pad_flop_frac"] * 100 for r in b],
               "s--", color="#e67e22", label="decode only")
    ax[0].plot([r["batch"] for r in b], [r["prefill_pad_flop_frac"] * 100 for r in b],
               "^--", color="#2471a3", label="prefill only")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("static batch size")
    ax[0].set_ylabel("% of FLOPs on padding")
    ax[0].set_title("B. padding vs batch size")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    c = f["C"]
    ax[1].plot([r["sigma"] for r in c], [r["total_pad_flop_frac"] * 100 for r in c],
               "o-", color="#8e44ad")
    ax[1].set_xlabel("lognormal sigma (spread of lengths)")
    ax[1].set_ylabel("% of FLOPs on padding")
    ax[1].set_title("C. padding vs length spread (batch 8)")
    ax[1].grid(alpha=.3)

    d = f["D"]
    x = range(len(d))
    ax[2].barh(list(x), [r["total_pad"] * 100 for r in d],
               color=["#27ae60" if r["implementable"] else "#95a5a6" for r in d])
    ax[2].set_yticks(list(x))
    ax[2].set_yticklabels([r["policy"].replace(", ", ",\n") for r in d], fontsize=7)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("% of FLOPs on padding")
    ax[2].set_title("D. bucketing (green = implementable)")
    for i, r in enumerate(d):
        ax[2].text(r["total_pad"] * 100, i, f" {r['total_pad']:.1%}",
                   va="center", fontsize=7)

    e = f["E"]["deltas"]
    keys = list(e)
    w = 0.35
    for j, k in enumerate(keys):
        ax[3].bar([i + (j - .5) * w for i in range(3)],
                  [e[k]["flops_saved"] * 100, e[k]["time_saved"] * 100,
                   e[k]["passes_saved"] * 100], w,
                  color=["#27ae60", "#95a5a6"][j], label=k)
    ax[3].set_xticks([0, 1, 2])
    ax[3].set_xticklabels(["FLOPs", "model time", "forward passes"])
    ax[3].set_ylabel("% saved by bucketing")
    ax[3].set_title("E. FLOPs saved is not time saved")
    ax[3].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "padding_waste.png"), dpi=110)
    print("wrote outputs/padding_waste.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    runner, tok = batchlib.load_runner()
    F["model"] = {"id": batchlib.MODEL_ID, "threads": batchlib.N_THREADS}
    section_a(runner)
    section_b(runner)
    section_c(runner)
    section_d(runner)
    section_e(runner, tok)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv()
    plot()


if __name__ == "__main__":
    main()
