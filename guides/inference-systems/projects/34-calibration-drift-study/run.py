"""Project 34 -- calibration drift: the quality regression nobody deploys.

You calibrated the quantizer on a sample of production traffic in week 0. It
passed the gate, it shipped, nobody touched it. Twelve weeks later the traffic
mix has moved -- more code, more structured output, fewer plain-prose chats --
and the calibration is describing a distribution that no longer exists. Nothing
alerts, because nothing failed.

  A. Model the drift. Two traffic mixtures, and how far apart the model's own
     activation statistics say they are.
  B. The gap over time: one week-0 calibration, evaluated against weeks 0-12.
  C. The full cross-domain matrix -- calibrate on each of four domains,
     evaluate on all four. The diagonal is the best case; the rest is drift.
  D. Is a *wrong* calibration worse than *no* calibration? (If not, the whole
     worry is misplaced.)
  E. Two fixes: recalibrate, or calibrate on a deliberately mixed set.
  F. A cheap drift detector that needs no eval labels, and whether its signal
     actually tracks the quality gap.

    python3 run.py           # ~9 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "30-quantize-a-7b-model-end-to-end"))

import torch  # noqa: E402

import quantlib as Q  # noqa: E402

WIN = 512
BITS, GROUP = 4, 128
DOMAINS = ("wiki", "chat", "code", "json")

# A plausible twelve weeks: an assistant that starts out answering prose
# questions and gradually becomes a coding and tool-calling endpoint.
WEEKS = {
    0:  {"wiki": 0.60, "chat": 0.35, "code": 0.05, "json": 0.00},
    3:  {"wiki": 0.50, "chat": 0.32, "code": 0.13, "json": 0.05},
    6:  {"wiki": 0.38, "chat": 0.27, "code": 0.25, "json": 0.10},
    9:  {"wiki": 0.25, "chat": 0.22, "code": 0.37, "json": 0.16},
    12: {"wiki": 0.15, "chat": 0.15, "code": 0.45, "json": 0.25},
}
F = {}


def mixture(pools, weights, n):
    """Build an n-window sample from per-domain pools of token windows."""
    out = []
    for dom, w in weights.items():
        k = int(round(w * n))
        pool = pools[dom]
        for i in range(k):
            out.append(pool[i % len(pool)])
    while len(out) < n:
        out.append(pools["wiki"][len(out) % len(pools["wiki"])])
    return torch.stack(out[:n])


def penalty(model, ev, calib_scales):
    """The only honest metric here: quantized perplexity divided by the *same
    data's* fp32 perplexity.

    Raw perplexity is useless for this study because the evaluation data itself
    changes from week to week -- code is easier to predict than prose, so a
    drifted week can show a *lower* raw perplexity while the quantization is
    doing more damage than ever. The ratio removes the data and leaves the
    quantizer."""
    base = Q.eval_chunks(model, ev, want_argmax=True)
    with Q.Quantized(model, bits=BITS, group=GROUP, awq_scales=calib_scales):
        q = Q.eval_chunks(model, ev, want_argmax=True)
    return {"base_ppl": base["ppl"], "q_ppl": q["ppl"],
            "penalty": q["ppl"] / base["ppl"],
            "agree": Q.agreement(base["argmax"], q["argmax"])}


def calibrate(model, chunks):
    st = Q.act_stats(model, chunks, sample_rows=96)
    sc, _ = Q.awq_scales(model, st, BITS, GROUP)
    return sc, st


# ---------------------------------------------------------------------------
# A. how far apart are the weeks?
# ---------------------------------------------------------------------------


def drift_signal(st_a, st_b):
    """Mean absolute log-ratio of per-channel |activation| between two traffic
    samples, averaged over every linear.

    Why a log-ratio and not a plain difference: the quantities span several
    orders of magnitude across channels, so a raw difference is dominated by
    the few biggest channels. `|log(a/b)|` asks "by what factor did this channel
    move?", which treats a 2x rise in a small channel the same as a 2x rise in
    a big one -- and a factor is exactly what a scale is."""
    vals = []
    for name, a in st_a.items():
        b = st_b.get(name)
        if b is None:
            continue
        ra = a["absmean"].clamp_min(1e-6)
        rb = b["absmean"].clamp_min(1e-6)
        vals.append(float((ra / rb).log().abs().mean()))
    return sum(vals) / len(vals)


def section_a(model, pools):
    print("\n=== A. the two ends of the drift ===")
    st = {}
    for wk in (0, 12):
        st[wk] = Q.act_stats(model, mixture(pools, WEEKS[wk], 8))
    d = drift_signal(st[0], st[12])
    per_dom = {}
    st_dom = {dom: Q.act_stats(model, pools[dom][:4]) for dom in DOMAINS}
    for dom in DOMAINS:
        per_dom[dom] = drift_signal(st_dom["wiki"], st_dom[dom])
    F["A"] = {"weeks": WEEKS, "drift_week0_to_12": d,
              "drift_vs_wiki": per_dom}
    print(f"  activation drift week 0 -> week 12: {d:.4f}")
    for k, v in per_dom.items():
        print(f"    wiki vs {k:5s}: {v:.4f}")
    return st_dom


# ---------------------------------------------------------------------------
# B. one calibration, twelve weeks
# ---------------------------------------------------------------------------


def section_b(model, pools):
    print("\n=== B. week-0 calibration, weeks 0-12 of traffic ===")
    calib0 = mixture(pools, WEEKS[0], 12)
    sc0, st0 = calibrate(model, calib0)
    rows = []
    for wk, mix in WEEKS.items():
        ev = mixture(pools, mix, 6)
        p = penalty(model, ev, sc0)
        rows.append({"week": wk, **p, **{"mix_" + k: v for k, v in mix.items()}})
        print(f"  week {wk:2d}: fp32 ppl {p['base_ppl']:8.3f}  quantized "
              f"{p['q_ppl']:8.3f}  penalty x{p['penalty']:.4f}  "
              f"agree {p['agree']*100:5.2f}%")
    # recalibrate at week 12 and re-measure
    sc12, _ = calibrate(model, mixture(pools, WEEKS[12], 12))
    ev12 = mixture(pools, WEEKS[12], 6)
    p12 = penalty(model, ev12, sc12)
    rtn = penalty(model, ev12, None)
    F["B"] = {"rows": rows, "week12_recalibrated": p12, "week12_rtn": rtn,
              "stale_penalty": rows[-1]["penalty"],
              "fresh_penalty": p12["penalty"], "rtn_penalty": rtn["penalty"]}
    print(f"  week 12 recalibrated: penalty x{p12['penalty']:.4f} "
          f"(stale x{rows[-1]['penalty']:.4f}, no calibration at all "
          f"x{rtn['penalty']:.4f})")
    return sc0, st0


# ---------------------------------------------------------------------------
# C + D. the cross-domain matrix
# ---------------------------------------------------------------------------


def section_cd(model, pools):
    print("\n=== C/D. calibrate on X, serve Y ===")
    scales, stats = {}, {}
    for dom in DOMAINS:
        scales[dom], stats[dom] = calibrate(model, pools[dom][8:14])
    scales["mixed"], stats["mixed"] = calibrate(
        model, torch.stack([pools[d][8 + i] for i in range(2) for d in DOMAINS]))

    matrix, base_ppl, rtn_row = {}, {}, {}
    for serve in DOMAINS:
        ev = pools[serve][:5]
        b = Q.eval_chunks(model, ev, want_argmax=True)
        base_ppl[serve] = b["ppl"]
        with Q.Quantized(model, bits=BITS, group=GROUP):
            r = Q.eval_chunks(model, ev, want_argmax=True)
        rtn_row[serve] = {"penalty": r["ppl"] / b["ppl"],
                          "agree": Q.agreement(b["argmax"], r["argmax"])}
        matrix[serve] = {}
        for calib in list(DOMAINS) + ["mixed"]:
            with Q.Quantized(model, bits=BITS, group=GROUP,
                             awq_scales=scales[calib]):
                r = Q.eval_chunks(model, ev, want_argmax=True)
            matrix[serve][calib] = {"penalty": r["ppl"] / b["ppl"],
                                    "agree": Q.agreement(b["argmax"], r["argmax"])}
        row = "  ".join(f"{c}:{matrix[serve][c]['penalty']:.3f}"
                        for c in list(DOMAINS) + ["mixed"])
        print(f"  serve {serve:5s} (fp32 ppl {b['ppl']:7.2f})  no-calib:"
              f"{rtn_row[serve]['penalty']:.3f}  {row}")

    # D: is a wrong calibration ever worse than none?
    worse = []
    for serve in DOMAINS:
        for calib in DOMAINS:
            if calib == serve:
                continue
            if matrix[serve][calib]["penalty"] > rtn_row[serve]["penalty"]:
                worse.append({"serve": serve, "calib": calib,
                              "wrong": matrix[serve][calib]["penalty"],
                              "none": rtn_row[serve]["penalty"]})
    F["C"] = {"matrix": matrix, "rtn": rtn_row, "base_ppl": base_ppl,
              "worse_than_no_calibration": worse}
    print(f"  wrong-calibration-worse-than-none cases: {len(worse)} / 12")
    return stats, matrix, rtn_row


# ---------------------------------------------------------------------------
# E. how good is the mixed calibration?
# ---------------------------------------------------------------------------


def section_e(matrix, rtn_row):
    print("\n=== E. matched vs mixed vs worst-wrong calibration ===")
    rows = []
    for serve in DOMAINS:
        matched = matrix[serve][serve]["penalty"]
        mixed = matrix[serve]["mixed"]["penalty"]
        wrong = max(matrix[serve][c]["penalty"] for c in DOMAINS if c != serve)
        none = rtn_row[serve]["penalty"]
        rows.append({"serve": serve, "matched": matched, "mixed": mixed,
                     "worst_wrong": wrong, "none": none,
                     "mixed_vs_matched": mixed / matched})
        print(f"  {serve:5s}: matched x{matched:.4f}  mixed x{mixed:.4f}  "
              f"worst-wrong x{wrong:.4f}  none x{none:.4f}")
    F["E"] = {"rows": rows,
              "mean_mixed_vs_matched": sum(r["mixed_vs_matched"] for r in rows)
              / len(rows)}


# ---------------------------------------------------------------------------
# F. a detector you can run without labels
# ---------------------------------------------------------------------------


def section_f(stats, matrix):
    print("\n=== F. does the drift signal predict the damage? ===")
    xs, ys, pts = [], [], []
    for serve in DOMAINS:
        for calib in DOMAINS:
            d = drift_signal(stats[calib], stats[serve])
            p = matrix[serve][calib]["penalty"]
            xs.append(d)
            ys.append(p)
            pts.append({"calib": calib, "serve": serve, "drift": d, "penalty": p})
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    r = cov / max(sx * sy, 1e-12)
    F["F"] = {"points": pts, "pearson_r": r}
    print(f"  Pearson r between activation drift and quantization penalty: "
          f"{r:+.3f}  (n={n})")


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    b = f["B"]["rows"]
    ax[0].plot([r["week"] for r in b], [r["penalty"] for r in b], "o-",
               color="#c0392b", label="week-0 calibration")
    ax[0].axhline(f["B"]["fresh_penalty"], color="#27ae60", ls="--", lw=1.2,
                  label="recalibrated at week 12")
    ax[0].axhline(f["B"]["rtn_penalty"], color="#34495e", ls=":", lw=1.2,
                  label="no calibration at all")
    ax[0].set_xlabel("week")
    ax[0].set_ylabel("perplexity / fp32 on the same data")
    ax[0].set_title("B. one calibration, twelve weeks")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax0b = ax[1]
    doms = list(f["A"]["weeks"]["0"].keys()) if "0" in f["A"]["weeks"] else \
        list(next(iter(f["A"]["weeks"].values())).keys())
    weeks = sorted(int(k) for k in f["A"]["weeks"])
    bottom = [0.0] * len(weeks)
    cols = {"wiki": "#2471a3", "chat": "#27ae60", "code": "#e67e22",
            "json": "#8e44ad"}
    for d in doms:
        ys = [f["A"]["weeks"][str(w)][d] for w in weeks]
        ax0b.bar([str(w) for w in weeks], ys, bottom=bottom, label=d,
                 color=cols.get(d))
        bottom = [a + c for a, c in zip(bottom, ys)]
    ax0b.set_xlabel("week")
    ax0b.set_ylabel("share of traffic")
    ax0b.set_title("A. the drift being modelled")
    ax0b.legend(fontsize=7)

    mat = f["C"]["matrix"]
    serves = list(mat)
    calibs = list(mat[serves[0]])
    import numpy as np
    M = np.array([[mat[s][c]["penalty"] for c in calibs] for s in serves])
    im = ax[2].imshow(M, cmap="RdYlGn_r")
    ax[2].set_xticks(range(len(calibs)))
    ax[2].set_xticklabels(calibs, rotation=40, ha="right", fontsize=7)
    ax[2].set_yticks(range(len(serves)))
    ax[2].set_yticklabels(serves, fontsize=7)
    for i in range(len(serves)):
        for j in range(len(calibs)):
            ax[2].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                       fontsize=6)
    ax[2].set_xlabel("calibrated on")
    ax[2].set_ylabel("serving")
    ax[2].set_title("C. the drift matrix (penalty)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    p = f["F"]["points"]
    ax[3].scatter([x["drift"] for x in p], [x["penalty"] for x in p], s=50,
                  c=["#27ae60" if x["calib"] == x["serve"] else "#c0392b"
                     for x in p])
    ax[3].set_xlabel("activation drift  (mean |log ratio|)")
    ax[3].set_ylabel("quantization penalty")
    ax[3].set_title(f"F. detector, r = {f['F']['pearson_r']:+.2f}")
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "calibration_drift.png"), dpi=110)
    print("wrote outputs/calibration_drift.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    tok, model = Q.load()
    texts = Q.corpora(tok, DOMAINS if "wiki" in DOMAINS else DOMAINS)
    pools = {d: Q.token_chunks(tok, texts[d], WIN, 20) for d in DOMAINS}
    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS, "window": WIN,
                  "bits": BITS, "group": GROUP, "domains": list(DOMAINS)}

    section_a(model, pools)
    section_b(model, pools)
    stats, matrix, rtn_row = section_cd(model, pools)
    section_e(matrix, rtn_row)
    section_f(stats, matrix)

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
