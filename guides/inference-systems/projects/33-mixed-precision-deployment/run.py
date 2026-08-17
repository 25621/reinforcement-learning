"""Project 33 -- mixed precision: spend your extra bits where they pay.

"Quantize the model to int4" is never literally what ships. Real recipes hold
one or two weight families back at higher precision. The usual advice is
"attention output projections and `lm_head`". This project checks that advice
by measuring every family's sensitivity and then pricing the bytes.

  A. Where the parameters actually are. The same recipe costs a very different
     fraction of the model at 0.5B and at 7B.
  B. Sensitivity, measured two ways that answer different questions:
       leave-one-out  -- int4 everything except family X. "What does protecting
                         X buy me?"
       only-one       -- fp32 everything except family X. "What does quantizing
                         X alone cost me?"
  C. The guide's recipe (o_proj + lm_head in fp32) against a greedy allocator
     that spends the same bytes on whatever section B says is most sensitive.
  D. Is sensitivity positional? Protect the first / last few layers instead.
  E. The output head on its own: fp32 / int8 / int4, and the tied-embedding
     trap that makes this decision bigger than it looks.
  F. The Pareto frontier: quality against bytes, for every plan measured.

    python3 run.py           # ~8 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import contextlib
import json
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
EVAL_N = 10          # for the headline plans
SENS_N = 6           # for the 16-config sensitivity sweep
BASE_BITS, BASE_GROUP = 4, 128
FAMILIES = list(Q.PROJ_NAMES)
F = {}


def run(model, ev, ref, *ctxs):
    with contextlib.ExitStack() as st:
        for c in ctxs:
            if c is not None:
                st.enter_context(c)
        r = Q.eval_chunks(model, ev, want_argmax=True)
    return {"ppl": r["ppl"], "agree": Q.agreement(ref, r["argmax"])}


def plan_ctx(model, keep_fp32=(), head_bits=4, awq=None, per_layer=None):
    """A deployment plan: int4 g128 body, `keep_fp32` families left alone,
    `head_bits` for the output head."""
    ctxs = [Q.Quantized(model, bits=BASE_BITS, group=BASE_GROUP,
                        skip=tuple(keep_fp32), awq_scales=awq,
                        per_layer=per_layer)]
    if head_bits < 16:
        ctxs.append(Q.quantize_head(model, bits=head_bits, group=BASE_GROUP))
    return ctxs


# ---------------------------------------------------------------------------
# A. where the parameters are
# ---------------------------------------------------------------------------


def section_a(model):
    print("\n=== A. parameter budget by family ===")
    rows = []
    for shape in ("Qwen2.5-0.5B", "Qwen2.5-7B", "Llama-3-70B"):
        gp = Q.group_params(Q.MODEL_SHAPES[shape])
        tot = sum(gp.values())
        rows.append({"model": shape, "total": tot,
                     **{k: v / tot for k, v in gp.items()}})
        print(f"  {shape:14s} " + "  ".join(
            f"{k.replace('_proj','')}={100*v/tot:4.1f}%" for k, v in gp.items()))
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    F["A"] = {"rows": rows, "lm_head_tied_to_embedding": bool(tied)}
    print(f"  lm_head tied to the embedding table: {tied}")


# ---------------------------------------------------------------------------
# B. sensitivity, both directions
# ---------------------------------------------------------------------------


def section_b(model, ev, ref, base_ppl):
    print("\n=== B. per-family sensitivity ===")
    all4 = run(model, ev, ref, *plan_ctx(model, (), BASE_BITS))
    print(f"  int4 everything (incl. head): ppl {all4['ppl']:.3f}")

    loo, only = [], []
    for fam in FAMILIES + ["embed_head"]:
        keep = (fam,) if fam != "embed_head" else ()
        head = 16 if fam == "embed_head" else BASE_BITS
        r = run(model, ev, ref, *plan_ctx(model, keep, head))
        saved = Q.group_params(Q.MODEL_SHAPES["Qwen2.5-0.5B"])[fam]
        extra_bytes_7b = (Q.group_params(Q.MODEL_SHAPES["Qwen2.5-7B"])[fam]
                          * (16 - (BASE_BITS + 32 / BASE_GROUP)) / 8)
        loo.append({"family": fam, "ppl": r["ppl"], "agree": r["agree"],
                    "recovered": (all4["ppl"] - r["ppl"]) / max(all4["ppl"] - base_ppl, 1e-9),
                    "params_05b": saved, "extra_gib_7b": extra_bytes_7b / 2**30})
        print(f"  leave-out {fam:11s} ppl {r['ppl']:8.3f}  "
              f"recovers {loo[-1]['recovered']*100:5.1f}% for "
              f"{loo[-1]['extra_gib_7b']:.2f} GiB on a 7B")

    for fam in FAMILIES + ["embed_head"]:
        if fam == "embed_head":
            r = run(model, ev, ref, Q.quantize_head(model, bits=BASE_BITS,
                                                    group=BASE_GROUP))
        else:
            r = run(model, ev, ref,
                    Q.Quantized(model, bits=BASE_BITS, group=BASE_GROUP,
                                names=[n for n in Q.block_linears(model)
                                       if Q.group_of(n) == fam]))
        only.append({"family": fam, "ppl": r["ppl"], "agree": r["agree"],
                     "delta_pct": 100 * (r["ppl"] / base_ppl - 1)})
        print(f"  only      {fam:11s} ppl {r['ppl']:8.3f} "
              f"({only[-1]['delta_pct']:+6.2f}%)")

    F["B"] = {"all_int4_ppl": all4["ppl"], "base_ppl": base_ppl,
              "leave_one_out": loo, "only_one": only}
    return all4, loo


# ---------------------------------------------------------------------------
# C. the standard recipe against a measured allocator
# ---------------------------------------------------------------------------


def section_c(model, ev, ref, base_ppl, all4, loo):
    print("\n=== C. the standard recipe vs a measured allocator ===")
    byfam = {r["family"]: r for r in loo}

    # "recovery per extra GiB on a 7B" -- the only ranking that respects cost
    ranked = sorted(byfam.values(),
                    key=lambda r: r["recovered"] / max(r["extra_gib_7b"], 1e-9),
                    reverse=True)
    print("  efficiency ranking (perplexity recovered per GiB spent on a 7B):")
    for r in ranked:
        print(f"    {r['family']:11s} {r['recovered']/max(r['extra_gib_7b'],1e-9):8.2f} "
              f"per GiB  ({r['recovered']*100:5.1f}% for {r['extra_gib_7b']:.2f} GiB)")

    guide = ("o_proj",)
    guide_bytes = (byfam["o_proj"]["extra_gib_7b"]
                   + byfam["embed_head"]["extra_gib_7b"])
    # Greedy: take families in efficiency order until the same budget is used.
    picked, spent = [], 0.0
    for r in ranked:
        if spent + r["extra_gib_7b"] <= guide_bytes + 1e-9:
            picked.append(r["family"])
            spent += r["extra_gib_7b"]

    plans = [
        ("int4 everything", (), BASE_BITS),
        ("guide: o_proj + head fp32", guide, 16),
        (f"greedy: {'+'.join(picked)}",
         tuple(f for f in picked if f != "embed_head"),
         16 if "embed_head" in picked else BASE_BITS),
    ]
    rows = []
    for label, keep, head in plans:
        r = run(model, ev, ref, *plan_ctx(model, keep, head))
        bits = {f: 16 for f in keep}
        if head >= 16:
            bits["embed_head"] = 16
        sz = Q.plan_bytes("Qwen2.5-7B", bits, BASE_BITS, BASE_GROUP)
        rows.append({"plan": label, "ppl": r["ppl"], "agree": r["agree"],
                     "gib_7b": sz["gib"], "eff_bits": sz["eff_bits"],
                     "recovered": (all4["ppl"] - r["ppl"])
                     / max(all4["ppl"] - base_ppl, 1e-9)})
        print(f"  {label:34s} ppl {r['ppl']:8.3f}  "
              f"recovers {rows[-1]['recovered']*100:5.1f}%  "
              f"{sz['gib']:.2f} GiB on a 7B")
    F["C"] = {"ranked": ranked, "guide_budget_gib_7b": guide_bytes,
              "greedy_pick": picked, "rows": rows}


# ---------------------------------------------------------------------------
# D. is sensitivity positional?
# ---------------------------------------------------------------------------


def section_d(model, ev, ref, base_ppl, all4):
    print("\n=== D. protect layers instead of families ===")
    lins = list(Q.block_linears(model))
    nl = max(int(n.split(".")[1]) for n in lins) + 1
    rows = []
    for label, keep_layers in (("first 2", range(2)), ("last 2", range(nl - 2, nl)),
                               ("first 4", range(4)), ("last 4", range(nl - 4, nl))):
        keep = {n for n in lins if int(n.split(".")[1]) in set(keep_layers)}
        r = run(model, ev, ref,
                Q.Quantized(model, bits=BASE_BITS, group=BASE_GROUP,
                            names=[n for n in lins if n not in keep]),
                Q.quantize_head(model, bits=BASE_BITS, group=BASE_GROUP))
        rows.append({"plan": label, "n_layers_kept": len(set(keep_layers)),
                     "ppl": r["ppl"], "agree": r["agree"],
                     "recovered": (all4["ppl"] - r["ppl"])
                     / max(all4["ppl"] - base_ppl, 1e-9),
                     "frac_params_fp32": len(set(keep_layers)) / nl})
        print(f"  keep {label:8s} fp32: ppl {r['ppl']:8.3f}  "
              f"recovers {rows[-1]['recovered']*100:5.1f}%  "
              f"({100*rows[-1]['frac_params_fp32']:.1f}% of the body)")
    F["D"] = {"rows": rows, "n_layers": nl}


# ---------------------------------------------------------------------------
# E. the output head alone
# ---------------------------------------------------------------------------


def section_e(model, ev, ref, base_ppl):
    print("\n=== E. the output head on its own ===")
    rows = []
    for bits in (16, 8, 4, 3):
        ctx = None if bits >= 16 else Q.quantize_head(model, bits=bits,
                                                      group=BASE_GROUP)
        r = run(model, ev, ref, ctx)
        rows.append({"head_bits": bits, "ppl": r["ppl"], "agree": r["agree"],
                     "delta_pct": 100 * (r["ppl"] / base_ppl - 1)})
        print(f"  head int{bits:<2d} (body fp32): ppl {r['ppl']:8.3f} "
              f"({rows[-1]['delta_pct']:+6.2f}%)  agree {r['agree']*100:5.1f}%")
    F["E"] = {"rows": rows, "base_ppl": base_ppl,
              "head_frac_05b": Q.group_params(Q.MODEL_SHAPES["Qwen2.5-0.5B"])["embed_head"]
              / sum(Q.group_params(Q.MODEL_SHAPES["Qwen2.5-0.5B"]).values()),
              "head_frac_7b": Q.group_params(Q.MODEL_SHAPES["Qwen2.5-7B"])["embed_head"]
              / sum(Q.group_params(Q.MODEL_SHAPES["Qwen2.5-7B"]).values())}


# ---------------------------------------------------------------------------
# F. the frontier
# ---------------------------------------------------------------------------


def section_f(model, ev, ref, base_ppl, stats):
    print("\n=== F. quality vs bytes ===")
    awq, _ = Q.awq_scales(model, stats, BASE_BITS, BASE_GROUP)
    plans = [
        ("int4 all", (), 4, None),
        ("int4 all + AWQ", (), 4, awq),
        ("int4 body, int8 head", (), 8, awq),
        ("int4 body, fp32 head", (), 16, awq),
        ("int4, o_proj fp32, fp32 head", ("o_proj",), 16, awq),
        ("int4, down_proj fp32, fp32 head", ("down_proj",), 16, awq),
        ("int4, k+v fp32, int8 head", ("k_proj", "v_proj"), 8, awq),
        ("int8 all", (), 8, None),
    ]
    rows = []
    for label, keep, head, sc in plans:
        bits = 8 if label == "int8 all" else BASE_BITS
        ctxs = [Q.Quantized(model, bits=bits, group=BASE_GROUP,
                            skip=tuple(keep), awq_scales=sc)]
        if head < 16:
            ctxs.append(Q.quantize_head(model, bits=head, group=BASE_GROUP))
        r = run(model, ev, ref, *ctxs)
        bg = {f: 16 for f in keep}
        bg["embed_head"] = head
        sz = Q.plan_bytes("Qwen2.5-7B", bg, bits, BASE_GROUP)
        rows.append({"plan": label, "ppl": r["ppl"], "agree": r["agree"],
                     "gib_7b": sz["gib"], "eff_bits": sz["eff_bits"]})
        print(f"  {label:32s} ppl {r['ppl']:8.3f}  agree {r['agree']*100:5.1f}%"
              f"  {sz['gib']:5.2f} GiB  {sz['eff_bits']:.2f} bits/w")
    F["F"] = {"rows": rows, "base_ppl": base_ppl,
              "bf16_gib_7b": Q.plan_bytes("Qwen2.5-7B", {}, 16)["gib"]}


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    loo = f["B"]["leave_one_out"]
    fams = [r["family"] for r in loo]
    ax[0].barh(range(len(loo)), [r["recovered"] * 100 for r in loo],
               color="#2471a3")
    ax[0].set_yticks(range(len(loo)))
    ax[0].set_yticklabels([x.replace("_proj", "") for x in fams], fontsize=8)
    ax[0].set_xlabel("% of the int4 damage recovered")
    ax[0].set_title("B. leave one family in fp32")
    ax[0].grid(alpha=0.3, axis="x")
    ax[0].invert_yaxis()

    ax[1].scatter([r["extra_gib_7b"] for r in loo],
                  [r["recovered"] * 100 for r in loo], s=60, color="#c0392b")
    for r in loo:
        ax[1].annotate(r["family"].replace("_proj", ""),
                       (r["extra_gib_7b"], r["recovered"] * 100), fontsize=7,
                       xytext=(4, 2), textcoords="offset points")
    ax[1].set_xlabel("extra GiB on a 7B")
    ax[1].set_ylabel("% of damage recovered")
    ax[1].set_title("C. recovery per byte spent")
    ax[1].grid(alpha=0.3)

    e = f["E"]["rows"]
    ax[2].plot([r["head_bits"] for r in e], [r["delta_pct"] for r in e], "o-",
               color="#8e44ad")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set_xlabel("bits in the output head (body left fp32)")
    ax[2].set_ylabel("perplexity change (%)")
    ax[2].set_title("E. the head alone")
    ax[2].grid(alpha=0.3)

    fr = f["F"]["rows"]
    ax[3].scatter([r["gib_7b"] for r in fr], [r["ppl"] for r in fr], s=60,
                  color="#27ae60")
    for r in fr:
        ax[3].annotate(r["plan"], (r["gib_7b"], r["ppl"]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[3].axhline(f["F"]["base_ppl"], color="#34495e", ls="--", lw=1,
                  label="fp32 baseline")
    ax[3].set_xlabel("7B weights (GiB)")
    ax[3].set_ylabel("perplexity")
    ax[3].set_title("F. the frontier")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mixed_precision.png"), dpi=110)
    print("wrote outputs/mixed_precision.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    tok, model = Q.load()
    text = Q._wikitext(500_000)
    ev = Q.token_chunks(tok, text, WIN, EVAL_N)
    sens_ev = ev[:SENS_N]
    calib = Q.token_chunks(tok, text, WIN, 8, skip=EVAL_N + 4)
    base = Q.eval_chunks(model, ev, want_argmax=True)
    base_s = Q.eval_chunks(model, sens_ev, want_argmax=True)
    print(f"fp32 baseline ppl {base['ppl']:.4f} "
          f"(sensitivity subset {base_s['ppl']:.4f})")
    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS, "window": WIN,
                  "eval_windows": EVAL_N, "sens_windows": SENS_N,
                  "base_bits": BASE_BITS, "base_group": BASE_GROUP,
                  "base_ppl": base["ppl"], "sens_base_ppl": base_s["ppl"]}

    section_a(model)
    all4, loo = section_b(model, sens_ev, base_s["argmax"], base_s["ppl"])
    section_c(model, sens_ev, base_s["argmax"], base_s["ppl"], all4, loo)
    section_d(model, sens_ev, base_s["argmax"], base_s["ppl"], all4)
    section_e(model, ev, base["argmax"], base["ppl"])
    stats = Q.act_stats(model, calib, sample_rows=128)
    section_f(model, ev, base["argmax"], base["ppl"], stats)

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
