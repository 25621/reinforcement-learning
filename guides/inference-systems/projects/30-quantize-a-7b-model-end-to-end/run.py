"""Project 30 -- quantize a model end-to-end, and gate the result.

The whole serving pipeline, start to finish:

  A. The arithmetic. What each bit plan costs in bytes, for 0.5B / 1.5B / 7B /
     70B, and how many H100s each one needs. This is the reason anyone does it.
  B. Round-to-nearest: the free baseline. Sweep bit width and scale
     granularity, measure perplexity and greedy agreement against fp32.
  C. AWQ: calibrate on 128 prompts, search the per-layer scale, and measure how
     much of the round-to-nearest damage it takes back.
  D. Does more calibration data help? Sweep 2 -> 128 windows.
  E. The control: calibrate on random tokens. If that works as well, the word
     "activation-aware" was doing nothing.
  F. The quality gate. Perplexity, MMLU, shadow agreement and free-running
     generation, with pass/fail thresholds -- run on a model that should pass,
     one that should just pass, and one that must be blocked.

    python3 run.py           # ~9 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import quantlib as Q  # noqa: E402

EVAL_CHUNKS = 8           # 8 x 512 = 4,096 tokens of held-out text
CALIB_CHUNKS = 16         # 16 x 512 = 8,192 tokens ~ "128 production prompts"
MMLU_N = 100
F = {}


def _label(bits, group, awq):
    g = "per-channel" if group in (0, None) else f"group-{group}"
    return f"{'AWQ' if awq else 'RTN'} int{bits} {g}"


# ---------------------------------------------------------------------------
# A. what quantization is actually for
# ---------------------------------------------------------------------------


def section_a():
    print("\n=== A. bytes on the wire ===")
    plans = [("BF16", 16, 16), ("FP8 / INT8", 8, 8), ("INT4 body, INT8 head", 4, 8),
             ("INT4 everything", 4, 4)]
    rows = []
    for name in ("Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B", "Llama-3-70B"):
        for pname, wb, hb in plans:
            r = Q.size_report(name, wb, hb, group=128)
            r["plan"] = pname
            # An H100-80GB cannot be filled to the brim: KV cache, activations
            # and CUDA context need room. 90% is the usual planning number.
            r["h100s"] = max(1, -(-int(r["bytes"]) // int(0.90 * 80 * 2**30)))
            rows.append(r)
            print(f"  {name:14s} {pname:22s} {r['gib']:8.2f} GiB  "
                  f"{r['eff_bits_per_weight']:5.2f} bits/w  {r['h100s']} x H100")
    F["A"] = {"rows": rows}


# ---------------------------------------------------------------------------
# B + C + E. quality of every recipe
# ---------------------------------------------------------------------------


def section_bce(tok, model, ev, calib, calib_rand):
    print("\n=== B/C. quality of each recipe ===")
    base = Q.eval_chunks(model, ev, want_argmax=True)
    ref = base["argmax"]
    print(f"  fp32 baseline ppl {base['ppl']:.3f}")

    t0 = time.time()
    stats = Q.act_stats(model, calib, sample_rows=128)
    calib_s = time.time() - t0
    t0 = time.time()
    sc128, rep128 = Q.awq_scales(model, stats, 4, 128)
    sc64, _ = Q.awq_scales(model, stats, 4, 64)
    sc3, _ = Q.awq_scales(model, stats, 3, 128)
    search_s = time.time() - t0
    print(f"  calibration pass {calib_s:.1f}s, alpha search {search_s:.1f}s")

    scales = {(4, 128): sc128, (4, 64): sc64, (3, 128): sc3}

    configs = [
        (8, 0, False),
        (4, 0, False), (4, 128, False), (4, 64, False),
        (3, 128, False),
        (4, 128, True), (4, 64, True), (3, 128, True),
    ]
    rows = [{"label": "fp32 baseline", "bits": 16, "group": 0, "awq": False,
             "ppl": base["ppl"], "agree": 1.0, "eff_bits": 16.0}]
    for bits, group, awq in configs:
        kw = {"awq_scales": scales[(bits, group)]} if awq else {}
        with Q.Quantized(model, bits=bits, group=group, **kw):
            r = Q.eval_chunks(model, ev, want_argmax=True)
        eff = bits + (0.0 if group in (0, None) else 32.0 / group)
        row = {"label": _label(bits, group, awq), "bits": bits, "group": group,
               "awq": awq, "ppl": r["ppl"], "agree": Q.agreement(ref, r["argmax"]),
               "eff_bits": eff}
        rows.append(row)
        print(f"  {row['label']:24s} ppl {row['ppl']:8.3f}  "
              f"agree {row['agree']*100:5.1f}%  {eff:.2f} bits/w")

    F["B"] = {"baseline_ppl": base["ppl"], "rows": rows,
              "eval_tokens": int(base["ntok"]),
              "alpha_hist_g128": dict(Counter(str(r["alpha"]) for r in rep128)),
              "alpha_by_proj": _alpha_by_proj(rep128),
              "calib_s": calib_s, "search_s": search_s}

    # -- E. the control: calibrate on noise -----------------------------------
    print("\n=== E. control: calibrate on random tokens ===")
    st_rand = Q.act_stats(model, calib_rand, sample_rows=128)
    sc_rand, _ = Q.awq_scales(model, st_rand, 4, 128)
    with Q.Quantized(model, bits=4, group=128, awq_scales=sc_rand):
        r = Q.eval_chunks(model, ev, want_argmax=True)
    rtn = [x for x in rows if x["label"] == "RTN int4 group-128"][0]
    awq = [x for x in rows if x["label"] == "AWQ int4 group-128"][0]
    F["E"] = {"rtn_ppl": rtn["ppl"], "awq_real_ppl": awq["ppl"],
              "awq_random_ppl": r["ppl"],
              "awq_random_agree": Q.agreement(ref, r["argmax"]),
              "recovered_real": (rtn["ppl"] - awq["ppl"]) / (rtn["ppl"] - base["ppl"]),
              "recovered_random": (rtn["ppl"] - r["ppl"]) / (rtn["ppl"] - base["ppl"])}
    print(f"  RTN {rtn['ppl']:.3f} | AWQ(real text) {awq['ppl']:.3f} | "
          f"AWQ(random tokens) {r['ppl']:.3f}")
    print(f"  gap recovered: real {F['E']['recovered_real']*100:.0f}%, "
          f"random {F['E']['recovered_random']*100:.0f}%")
    return base, stats, sc128


def _alpha_by_proj(rep):
    by = {}
    for r in rep:
        by.setdefault(Q.group_of(r["linear"]), []).append(r["alpha"])
    return {k: sum(v) / len(v) for k, v in by.items()}


# ---------------------------------------------------------------------------
# D. how much calibration data is enough
# ---------------------------------------------------------------------------


def section_d(tok, model, ev_small, calib_all):
    print("\n=== D. calibration-set size sweep ===")
    base = Q.eval_chunks(model, ev_small)
    with Q.Quantized(model, bits=4, group=128):
        rtn = Q.eval_chunks(model, ev_small)
    print(f"  (this subset: fp32 {base['ppl']:.3f}, no calibration "
          f"{rtn['ppl']:.3f})")
    rows = []
    for n in (2, 8, 16):
        st = Q.act_stats(model, calib_all[:n], sample_rows=128)
        sc, rep = Q.awq_scales(model, st, 4, 128)
        with Q.Quantized(model, bits=4, group=128, awq_scales=sc):
            r = Q.eval_chunks(model, ev_small)
        rows.append({"windows": n, "tokens": n * 512, "ppl": r["ppl"],
                     "recovered": (rtn["ppl"] - r["ppl"])
                     / max(rtn["ppl"] - base["ppl"], 1e-9),
                     "mean_alpha": sum(x["alpha"] for x in rep) / len(rep)})
        print(f"  {n:3d} windows ({n*512:5d} tokens): ppl {r['ppl']:.3f} "
              f"mean alpha {rows[-1]['mean_alpha']:.3f}")
    F["D"] = {"baseline_ppl": base["ppl"], "rtn_ppl": rtn["ppl"], "rows": rows}


# ---------------------------------------------------------------------------
# F. the deploy gate
# ---------------------------------------------------------------------------

GATE = {"max_ppl_ratio": 1.15,      # perplexity may rise by at most 15%
        "max_mmlu_drop": 3.0,       # points
        "min_shadow_agree": 0.95}   # greedy top-1 agreement with production


def section_f(tok, model, ev, sc128):
    print("\n=== F. the deploy gate ===")
    prompts = Q.chat_prompts(tok, n=4, seed=7)
    items = Q.mmlu_items(MMLU_N, seed=3)

    def measure(name, ctx):
        with ctx() if ctx else _null():
            e = Q.eval_chunks(model, ev, want_argmax=True)
            m = Q.mmlu_eval(model, tok, items)
            g = Q.greedy_generate(model, tok, prompts, max_new=20)
        return {"name": name, "ppl": e["ppl"], "argmax": e["argmax"],
                "mmlu": m["acc"], "mmlu_se": m["stderr"], "gen": g}

    base = measure("fp32 baseline", None)
    example = {"prompt": prompts[0], "fp32 baseline": tok.decode(base["gen"][0])}
    cands = [
        ("INT8 per-channel", lambda: Q.Quantized(model, bits=8, group=0)),
        ("AWQ INT4 group-128", lambda: Q.Quantized(model, bits=4, group=128,
                                                   awq_scales=sc128)),
        ("RTN INT4 per-channel", lambda: Q.Quantized(model, bits=4, group=0)),
        ("RTN INT3 group-128", lambda: Q.Quantized(model, bits=3, group=128)),
    ]
    rows = []
    for name, ctx in cands:
        m = measure(name, ctx)
        agree = Q.agreement(base["argmax"], m["argmax"])
        div = [Q.first_divergence(a, b) for a, b in zip(base["gen"], m["gen"])]
        checks = {
            "ppl_ratio": m["ppl"] / base["ppl"],
            "mmlu_drop": (base["mmlu"] - m["mmlu"]) * 100,
            "shadow_agree": agree,
        }
        fails = []
        if checks["ppl_ratio"] > GATE["max_ppl_ratio"]:
            fails.append("perplexity")
        if checks["mmlu_drop"] > GATE["max_mmlu_drop"]:
            fails.append("mmlu")
        if checks["shadow_agree"] < GATE["min_shadow_agree"]:
            fails.append("shadow")
        example[name] = tok.decode(m["gen"][0])
        rows.append({"name": name, "ppl": m["ppl"], "mmlu": m["mmlu"],
                     "mmlu_se": m["mmlu_se"], **checks,
                     "mean_first_divergence": sum(div) / len(div),
                     "identical_gens": sum(1 for d, g in zip(div, m["gen"])
                                           if d >= len(g)),
                     "n_gens": len(div), "fails": fails,
                     "verdict": "BLOCK" if fails else "PASS"})
        print(f"  {name:22s} ppl x{checks['ppl_ratio']:.3f}  "
              f"MMLU {m['mmlu']*100:5.1f}% ({-checks['mmlu_drop']:+.1f})  "
              f"agree {agree*100:5.1f}%  -> {rows[-1]['verdict']}"
              + (f" [{','.join(fails)}]" if fails else ""))


    F["F"] = {"gate": GATE, "baseline": {"ppl": base["ppl"], "mmlu": base["mmlu"],
                                         "mmlu_se": base["mmlu_se"]},
              "mmlu_n": MMLU_N, "rows": rows, "example": example}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))

    # A. memory
    names = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B", "Llama-3-70B"]
    plans = ["BF16", "FP8 / INT8", "INT4 everything"]
    cols = {"BF16": "#34495e", "FP8 / INT8": "#2471a3", "INT4 everything": "#27ae60"}
    w = 0.26
    for j, p in enumerate(plans):
        ys = [r["gib"] for n in names for r in f["A"]["rows"]
              if r["model"] == n and r["plan"] == p]
        ax[0].bar([i + (j - 1) * w for i in range(len(names))], ys, w,
                  label=p, color=cols[p])
    ax[0].axhline(0.90 * 80, color="#c0392b", ls="--", lw=1,
                  label="one H100-80GB (90% usable)")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.replace("Qwen2.5-", "Q").replace("Llama-3-", "L3-")
                           for n in names], fontsize=8)
    ax[0].set_ylabel("weights (GiB)")
    ax[0].set_title("A. what the bits buy")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")

    # B. quality vs bits
    rows = f["B"]["rows"]
    for awq, c, m in ((False, "#c0392b", "o"), (True, "#27ae60", "s")):
        pts = [r for r in rows if r.get("awq") == awq and r["bits"] < 16]
        ax[1].scatter([r["eff_bits"] for r in pts], [r["ppl"] for r in pts],
                      color=c, marker=m, s=60, label="AWQ" if awq else "RTN")
    ax[1].axhline(f["B"]["baseline_ppl"], color="#34495e", ls="--", lw=1,
                  label="fp32 baseline")
    ax[1].set_xlabel("effective bits per weight (scales included)")
    ax[1].set_ylabel("perplexity")
    ax[1].set_yscale("log")
    ax[1].set_title("B/C. granularity and AWQ")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # D/E. calibration size and the noise control, on one comparable axis:
    # the fraction of the round-to-nearest damage each variant takes back.
    d = f["D"]["rows"]
    ax[2].plot([r["tokens"] for r in d], [100 * r["recovered"] for r in d], "o-",
               color="#2471a3", label="AWQ int4 g128, real text")
    ax[2].axhline(0.0, color="#c0392b", ls="--", lw=1,
                  label="no calibration (RTN)")
    ax[2].axhline(100 * f["E"]["recovered_random"], color="#e67e22", ls=":",
                  lw=1.5, label="calibrated on random tokens")
    ax[2].axhline(100 * f["E"]["recovered_real"], color="#27ae60", ls="-.",
                  lw=1.2, label="8k real tokens (section C eval)")
    ax[2].set_xscale("log")
    ax[2].set_ylim(-5, 60)
    ax[2].set_xlabel("calibration tokens")
    ax[2].set_ylabel("% of the int4 damage recovered")
    ax[2].set_title("D/E. how much calibration is enough")
    ax[2].legend(fontsize=6.5)
    ax[2].grid(alpha=0.3)

    # F. the gate
    g = f["F"]["rows"]
    labs = [r["name"].replace(" group-128", "\ng128").replace(" per-channel", "\npc")
            for r in g]
    y = [r["shadow_agree"] * 100 for r in g]
    bars = ax[3].bar(range(len(g)), y,
                     color=["#27ae60" if r["verdict"] == "PASS" else "#c0392b"
                            for r in g])
    ax[3].axhline(f["F"]["gate"]["min_shadow_agree"] * 100, color="k", ls="--",
                  lw=1, label="gate threshold")
    for i, r in enumerate(g):
        ax[3].text(i, y[i] + 1.5, r["verdict"], ha="center", fontsize=7)
    ax[3].set_xticks(range(len(g)))
    ax[3].set_xticklabels(labs, fontsize=7)
    ax[3].set_ylabel("greedy agreement with fp32 (%)")
    ax[3].set_ylim(0, 108)
    ax[3].set_title("F. the deploy gate")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "quantize_end_to_end.png"), dpi=110)
    print("wrote outputs/quantize_end_to_end.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    tok, model = Q.load()
    text = Q._wikitext(600_000)
    ev = Q.token_chunks(tok, text, 512, EVAL_CHUNKS)
    calib = Q.token_chunks(tok, text, 512, CALIB_CHUNKS, skip=EVAL_CHUNKS + 4)
    torch.manual_seed(0)
    calib_rand = torch.randint(0, 150_000, (CALIB_CHUNKS, 512))

    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS,
                  "eval_windows": EVAL_CHUNKS, "calib_windows": CALIB_CHUNKS,
                  "window_tokens": 512, "mmlu_n": MMLU_N}
    section_a()
    _, _, sc128 = section_bce(tok, model, ev, calib, calib_rand)
    section_d(tok, model, ev[:5], calib)
    section_f(tok, model, ev, sc128)

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
