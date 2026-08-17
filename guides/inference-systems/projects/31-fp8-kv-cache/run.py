"""Project 31 -- switch a deployment's KV cache from 16-bit to FP8.

Project 13 ranked eleven storage formats by quality. This is the operational
follow-up: you are flipping the flag on a live service.

  A. Headroom. FP8 e4m3 stops at 448 and turns into NaN above 464. Measure the
     real |k| and |v| this model produces and decide whether an unscaled cast
     is even legal.
  B. Quality, for the three scaling shapes a serving engine offers (none /
     static / per-token) and both fp8 layouts, plus the int8 comparison.
  C. Memory: measured cache bytes, the per-token-scale overhead nobody budgets
     for, and how many concurrent users each plan buys on an H100.
  D. Speed. Measured on this CPU (which has no fp8 hardware, and the result is
     the honest one), and computed for an H100 from bytes moved per step.
  E. Does the damage grow with context? Sweep 256 -> 4096 tokens.
  F. The deploy gate, plus the failure this project exists to warn about: a
     static scale calibrated on one workload, serving another.

    python3 run.py           # ~5 minutes on 6 CPU threads
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "13-kv-quantization-study"))
sys.path.insert(0, os.path.join(HERE, "..", "30-quantize-a-7b-model-end-to-end"))

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import kvlib  # noqa: E402
import quantlib as Q  # noqa: E402
from fp8kv import FP8Cache, calibrate_static, kv_absmax_profile  # noqa: E402
from quantcache import QuantCache  # noqa: E402

EVAL_N = 6          # windows of 512 tokens for the quality table
WIN = 512
F = {}


@torch.inference_mode()
def score(runner, make_cache, chunks, ref=None):
    """Perplexity and greedy top-1 predictions with one storage plan."""
    nll, ntok, preds = 0.0, 0, []
    for ch in chunks:
        cache = make_cache()
        logits = runner.forward(ch.unsqueeze(0), cache, start_pos=0)[0].float()
        lp, tgt = logits[:-1], ch[1:]
        nll += TF.cross_entropy(lp, tgt, reduction="sum").item()
        ntok += tgt.numel()
        preds.append(lp.argmax(-1))
    preds = torch.cat(preds)
    out = {"ppl": math.exp(nll / ntok), "ntok": ntok, "preds": preds}
    if ref is not None:
        out["agree"] = (preds == ref).float().mean().item()
    return out


# ---------------------------------------------------------------------------
# A. is an unscaled cast even legal?
# ---------------------------------------------------------------------------


def section_a(runner, chunks):
    print("\n=== A. fp8 headroom ===")
    prof = kv_absmax_profile(runner, chunks[:3])
    kmax, vmax = max(prof["k"]), max(prof["v"])
    # The format's own cliff, measured rather than quoted.
    probe = {}
    for x in (447.0, 449.0, 463.0, 465.0, 1000.0):
        y = float(torch.tensor([x]).to(torch.float8_e4m3fn).float())
        probe[str(x)] = "nan" if y != y else y
    # What happens if the conversion does NOT saturate: one out-of-range key
    # becomes NaN, and a NaN inside a softmax row destroys the whole row.
    nl = runner.n_layers
    tight = {(i, w): (prof[w][i] / 448.0) * 0.5      # a scale 2x too small
             for i in range(nl) for w in ("k", "v")}
    nan_out = {}
    for sat in (False, True):
        c = FP8Cache(nl, "e4m3", "static", static=dict(tight), saturate=sat)
        lg = runner.forward(chunks[0].unsqueeze(0), c, start_pos=0)
        nan_out["saturate" if sat else "raw_cast"] = {
            "nan_logits": bool(torch.isnan(lg).any()),
            "clip_rate_k": c.clip_rate()["k"]}
    print(f"  with a scale 2x too small: {nan_out}")

    F["A"] = {"k_absmax_per_layer": prof["k"], "v_absmax_per_layer": prof["v"],
              "half_scale_probe": nan_out,
              "k_absmax": kmax, "v_absmax": vmax,
              "e4m3_max": 448.0, "e4m3_nan_above": 464.0, "probe": probe,
              "k_headroom": 448.0 / kmax, "v_headroom": 448.0 / vmax,
              "worst_layer_k": int(max(range(len(prof["k"])),
                                       key=lambda i: prof["k"][i]))}
    print(f"  max |k| = {kmax:.1f} (layer {F['A']['worst_layer_k']}), "
          f"max |v| = {vmax:.1f}")
    print(f"  e4m3 ceiling 448, NaN above 464 -> headroom "
          f"k {F['A']['k_headroom']:.2f}x, v {F['A']['v_headroom']:.2f}x")
    print(f"  cast probe: {probe}")


# ---------------------------------------------------------------------------
# B. quality of every storage plan
# ---------------------------------------------------------------------------


def section_b(runner, chunks, static):
    print("\n=== B. quality per storage plan ===")
    nl = runner.n_layers
    base = score(runner, lambda: kvlib.ContiguousCache(nl), chunks)
    ref = base["preds"]
    print(f"  fp32 cache ppl {base['ppl']:.4f}")

    plans = [
        ("fp8 e4m3, unscaled", lambda: FP8Cache(nl, "e4m3", "none")),
        ("fp8 e4m3, static scale", lambda: FP8Cache(nl, "e4m3", "static",
                                                    static=dict(static))),
        ("fp8 e4m3, per-token scale", lambda: FP8Cache(nl, "e4m3", "per-token")),
        ("fp8 e5m2, unscaled", lambda: FP8Cache(nl, "e5m2", "none")),
        ("fp8 e5m2, per-token scale", lambda: FP8Cache(nl, "e5m2", "per-token")),
        ("int8 per-token", lambda: QuantCache(nl, bits=8, granularity="token")),
        ("fp8 e4m3 keys only", lambda: FP8Cache(nl, "e4m3", "per-token",
                                                quant_v=False)),
        ("fp8 e4m3 values only", lambda: FP8Cache(nl, "e4m3", "per-token",
                                                  quant_k=False)),
    ]
    rows = [{"plan": "fp32 cache (baseline)", "ppl": base["ppl"], "agree": 1.0,
             "delta_pct": 0.0}]
    for name, mk in plans:
        r = score(runner, mk, chunks, ref)
        rows.append({"plan": name, "ppl": r["ppl"], "agree": r["agree"],
                     "delta_pct": 100 * (r["ppl"] / base["ppl"] - 1)})
        print(f"  {name:28s} ppl {r['ppl']:8.4f} ({rows[-1]['delta_pct']:+6.2f}%)"
              f"  agree {r['agree']*100:5.2f}%")
    F["B"] = {"baseline_ppl": base["ppl"], "rows": rows,
              "eval_tokens": base["ntok"]}
    return ref


# ---------------------------------------------------------------------------
# C. memory and concurrency
# ---------------------------------------------------------------------------


def section_c(runner, chunks):
    print("\n=== C. bytes and seats ===")
    nl = runner.n_layers
    ch = chunks[0]
    measured = {}
    for name, mk in (("fp32", lambda: None),
                     ("fp8 unscaled", lambda: FP8Cache(nl, "e4m3", "none")),
                     ("fp8 static", lambda: FP8Cache(nl, "e4m3", "static")),
                     ("fp8 per-token", lambda: FP8Cache(nl, "e4m3", "per-token"))):
        cache = mk() or kvlib.ContiguousCache(nl)
        runner.forward(ch.unsqueeze(0), cache, start_pos=0)
        b = cache.nbytes() if name == "fp32" else cache.stored_bytes()
        measured[name] = {"bytes": b, "bytes_per_token": b / WIN}
        print(f"  {name:14s} {b/2**20:7.2f} MiB for {WIN} tokens "
              f"({b/WIN:7.1f} B/token)")

    # The scale overhead, spelled out. One fp32 scale per (token, kv-head)
    # sits on top of d_head bytes of payload.
    d_head = runner.d_head
    overhead = 4.0 / d_head
    F["C"] = {"measured": measured, "d_head": d_head,
              "per_token_scale_overhead_pct": 100 * overhead, "fleet": []}
    print(f"  per-token scale overhead at d_head={d_head}: "
          f"{100*overhead:.1f}% of the fp8 payload")

    # Fleet arithmetic on models whose weights do not fit on this box.
    # Reported per *card* as well as in total, because a plan that halves the
    # weights can also halve the card count -- and then "more seats" and "more
    # seats per card" are different questions with different answers.
    card = 0.90 * 80 * 2**30                      # usable bytes on an H100-80GB
    for shape in ("Qwen2.5-7B", "Llama-3-70B"):
        for ctx in (8192, 32768):
            for wbits, kvbits in ((16, 16), (16, 8), (8, 16), (8, 8)):
                wb = Q.size_report(shape, wbits, wbits)["bytes"]
                per_user = Q.kv_bytes_per_token(shape, kvbits) * ctx
                cards = max(1, math.ceil(wb / card))
                seats = max(0, int((cards * card - wb) // per_user))
                row = {"model": shape, "ctx": ctx, "w_bits": wbits,
                       "kv_bits": kvbits, "cards": cards, "seats": seats,
                       "seats_per_card": seats / cards,
                       "kv_B_per_token": Q.kv_bytes_per_token(shape, kvbits),
                       "weights_gib": wb / 2**30}
                F["C"]["fleet"].append(row)
                print(f"  {shape:13s} ctx {ctx:5d}  W{wbits}/KV{kvbits}: "
                      f"{cards} card(s), {seats:4d} seats "
                      f"({seats/cards:6.1f} per card)")


# ---------------------------------------------------------------------------
# D. speed: measured here, computed for the hardware that has the instruction
# ---------------------------------------------------------------------------


def section_d(runner, chunks):
    print("\n=== D. decode step time ===")
    nl = runner.n_layers
    rows = []
    for ctx in (512, 2048):
        prompt = torch.cat([chunks[i] for i in range(ctx // WIN)]).unsqueeze(0)
        one = torch.tensor([[1000]])

        def mk(plan):
            return {"fp32": lambda: kvlib.ContiguousCache(nl),
                    "fp8 per-token": lambda: FP8Cache(nl, "e4m3", "per-token"),
                    "fp8 static": lambda: FP8Cache(nl, "e4m3", "static")}[plan]()

        caches = {}
        for plan in ("fp32", "fp8 per-token", "fp8 static"):
            c = mk(plan)
            runner.forward(prompt, c, start_pos=0)
            caches[plan] = c

        # Each timed call appends one more token, so `start_pos` has to advance
        # with it -- the causal mask is built from absolute positions and would
        # be the wrong width otherwise. Over four rounds the context grows by
        # four tokens out of 2048, which is below the timing noise; rebuilding
        # the cache each round instead would time the prefill.
        pos = {plan: ctx for plan in caches}

        def step(plan):
            c = caches[plan]
            runner.forward(one, c, start_pos=pos[plan])
            pos[plan] += 1

        fns = {plan: (lambda p=plan: step(p)) for plan in caches}
        t = kvlib.interleaved(fns, rounds=3, warmup=1)
        row = {"ctx": ctx, **{k: v * 1000 for k, v in t.items()}}
        row["fp8_vs_fp32"] = t["fp8 per-token"] / t["fp32"]
        rows.append(row)
        print(f"  ctx {ctx}: " + "  ".join(f"{k} {v*1000:.1f} ms"
                                           for k, v in t.items()))

    # What the same change does on hardware that loads fp8 natively.
    #
    # The batch axis is the whole story and it is easy to miss. A decode step
    # reads the weights ONCE no matter how many requests are in the batch, but
    # it reads each request's KV cache separately. So at batch 1 the cache is a
    # rounding error next to the weights, and at batch 128 it is most of the
    # traffic -- which is exactly the regime a busy server runs in.
    h100_bw = 3.35e12          # HBM3 bytes/s on an H100-SXM
    arith = []
    for shape in ("Qwen2.5-7B", "Llama-3-70B"):
        wb = Q.size_report(shape, 16, 16)["bytes"]
        for ctx in (2048, 8192, 32768):
            for batch in (1, 32, 128):
                kv16 = Q.kv_bytes_per_token(shape, 16) * ctx * batch
                kv8 = Q.kv_bytes_per_token(shape, 8) * ctx * batch
                t16 = (wb + kv16) / h100_bw
                t8 = (wb + kv8) / h100_bw
                arith.append({"model": shape, "ctx": ctx, "batch": batch,
                              "kv_share_bf16": kv16 / (wb + kv16),
                              "step_ms_bf16_kv": t16 * 1000,
                              "step_ms_fp8_kv": t8 * 1000,
                              "speedup": t16 / t8})
    F["D"] = {"measured": rows, "h100_bandwidth_B_s": h100_bw, "arithmetic": arith}
    for a in arith:
        if a["ctx"] in (2048, 32768):
            print(f"  [arithmetic] {a['model']:13s} ctx {a['ctx']:5d} batch "
                  f"{a['batch']:3d}: kv is {a['kv_share_bf16']*100:4.1f}% of "
                  f"the read -> fp8 kv {a['speedup']:.2f}x")


# ---------------------------------------------------------------------------
# E. does it get worse with context?
# ---------------------------------------------------------------------------


def section_e(runner, text, tok):
    print("\n=== E. quality vs context length ===")
    nl = runner.n_layers
    rows = []
    for ctx in (256, 512, 1024, 2048):
        n = max(1, 1024 // ctx)
        chunks = Q.token_chunks(tok, text, ctx, n)
        base = score(runner, lambda: kvlib.ContiguousCache(nl), chunks)
        r_pt = score(runner, lambda: FP8Cache(nl, "e4m3", "per-token"), chunks,
                     base["preds"])
        r_st = score(runner, lambda: FP8Cache(nl, "e4m3", "static"), chunks,
                     base["preds"])
        rows.append({"ctx": ctx, "windows": n, "base_ppl": base["ppl"],
                     "pertoken_ppl": r_pt["ppl"], "pertoken_agree": r_pt["agree"],
                     "static_ppl": r_st["ppl"], "static_agree": r_st["agree"],
                     "pertoken_delta_pct": 100 * (r_pt["ppl"] / base["ppl"] - 1),
                     "static_delta_pct": 100 * (r_st["ppl"] / base["ppl"] - 1)})
        print(f"  ctx {ctx:5d}: base {base['ppl']:7.3f}  per-token "
              f"{rows[-1]['pertoken_delta_pct']:+.2f}% agree "
              f"{r_pt['agree']*100:.2f}%  static "
              f"{rows[-1]['static_delta_pct']:+.2f}% agree {r_st['agree']*100:.2f}%")
    F["E"] = {"rows": rows}


# ---------------------------------------------------------------------------
# F. the gate, and the stale-scale failure
# ---------------------------------------------------------------------------


def section_f(runner, tok, texts):
    print("\n=== F. a static scale is a calibration you can get wrong ===")
    nl = runner.n_layers
    doms = ("wiki", "code", "chat")
    scales, absmax = {}, {}
    for d in doms:
        scales[d], absmax[d] = calibrate_static(
            runner, None, Q.token_chunks(tok, texts[d], WIN, 4, skip=30))

    ratios = {d: [absmax[d][(i, "k")] / absmax["wiki"][(i, "k")]
                  for i in range(nl)] for d in doms}
    rows = []
    for serve in doms:
        ev = Q.token_chunks(tok, texts[serve], WIN, 3, skip=2)
        base = score(runner, lambda: kvlib.ContiguousCache(nl), ev)
        row = {"domain": serve, "base_ppl": base["ppl"], "static": {}}
        for calib in doms:
            r = score(runner, lambda c=calib: FP8Cache(
                nl, "e4m3", "static", static=dict(scales[c])), ev, base["preds"])
            row["static"][calib] = {"ppl": r["ppl"], "agree": r["agree"],
                                    "ratio": r["ppl"] / base["ppl"]}
        r = score(runner, lambda: FP8Cache(nl, "e4m3", "per-token"), ev,
                  base["preds"])
        row["pertoken"] = {"ppl": r["ppl"], "agree": r["agree"],
                           "ratio": r["ppl"] / base["ppl"]}
        rows.append(row)
        print(f"  serve {serve:5s} (fp32 {base['ppl']:7.3f}): "
              + "  ".join(f"{c}-scale x{row['static'][c]['ratio']:.4f}"
                          for c in doms)
              + f"   per-token x{row['pertoken']['ratio']:.4f}")

    gate = {"max_ppl_ratio": 1.02, "min_agree": 0.90}
    for r in rows:
        for c in doms:
            v = r["static"][c]
            v["verdict"] = ("PASS" if v["ratio"] <= gate["max_ppl_ratio"]
                            and v["agree"] >= gate["min_agree"] else "BLOCK")
        v = r["pertoken"]
        v["verdict"] = ("PASS" if v["ratio"] <= gate["max_ppl_ratio"]
                        and v["agree"] >= gate["min_agree"] else "BLOCK")
    F["F"] = {"rows": rows, "gate": gate, "domains": list(doms),
              "k_absmax_ratio_vs_wiki": {
                  d: {"min": min(v), "max": max(v),
                      "mean": sum(v) / len(v)} for d, v in ratios.items()}}


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))

    a = f["A"]
    ax[0].plot(a["k_absmax_per_layer"], "o-", color="#c0392b", label="max |k|")
    ax[0].plot(a["v_absmax_per_layer"], "s-", color="#2471a3", label="max |v|")
    ax[0].axhline(448, color="k", ls="--", lw=1, label="e4m3 max (448)")
    ax[0].axhline(464, color="#e67e22", ls=":", lw=1.5, label="NaN above 464")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("largest absolute value")
    ax[0].set_title("A. how much fp8 headroom is left")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    rows = [r for r in f["B"]["rows"] if r["plan"] != "fp32 cache (baseline)"]
    y = [r["delta_pct"] for r in rows]
    ax[1].barh(range(len(rows)), y,
               color=["#27ae60" if v < 2 else "#e67e22" if v < 10 else "#c0392b"
                      for v in y])
    ax[1].set_yticks(range(len(rows)))
    ax[1].set_yticklabels([r["plan"] for r in rows], fontsize=7)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("perplexity change vs fp32 cache (%)")
    ax[1].set_title("B. what each plan costs")
    ax[1].grid(alpha=0.3, axis="x")
    ax[1].invert_yaxis()

    e = f["E"]["rows"]
    ax[2].plot([r["ctx"] for r in e], [r["pertoken_agree"] * 100 for r in e],
               "o-", color="#27ae60", label="per-token scale")
    ax[2].plot([r["ctx"] for r in e], [r["static_agree"] * 100 for r in e],
               "s-", color="#e67e22", label="static scale")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("context length (tokens)")
    ax[2].set_ylabel("greedy agreement with fp32 (%)")
    ax[2].set_title("E. does damage grow with context?")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fr = f["F"]["rows"]
    doms = f["F"]["domains"]
    xs = range(len(fr))
    w = 0.22
    cols = {"wiki": "#2471a3", "code": "#e67e22", "chat": "#8e44ad"}
    for j, c in enumerate(doms):
        ax[3].bar([x + (j - 1.5) * w for x in xs],
                  [100 * (r["static"][c]["ratio"] - 1) for r in fr], w,
                  label=f"static scale from {c}", color=cols[c])
    ax[3].bar([x + 1.5 * w for x in xs],
              [100 * (r["pertoken"]["ratio"] - 1) for r in fr], w,
              label="per-token (no calibration)", color="#27ae60")
    ax[3].set_xticks(list(xs))
    ax[3].set_xticklabels([r["domain"] for r in fr])
    ax[3].axhline(2.0, color="k", ls="--", lw=1, label="gate: +2%")
    ax[3].set_ylabel("perplexity change vs fp32 (%)")
    ax[3].set_title("F. a static scale is a calibration you can get wrong")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fp8_kv_cache.png"), dpi=110)
    print("wrote outputs/fp8_kv_cache.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    runner, tok, _model = kvlib.load_runner()
    texts = {"wiki": Q._wikitext(400_000), "code": Q._code(300_000),
             "chat": Q._chat(tok, 200_000)}
    chunks = Q.token_chunks(tok, texts["wiki"], WIN, EVAL_N)
    F["setup"] = {"model": kvlib.MODEL_ID, "threads": kvlib.N_THREADS,
                  "window": WIN, "eval_windows": EVAL_N,
                  "layers": runner.n_layers, "kv_heads": runner.n_kv_heads,
                  "d_head": runner.d_head}

    section_a(runner, chunks)
    calib = Q.token_chunks(tok, texts["wiki"], WIN, 4, skip=EVAL_N + 2)
    static, _ = calibrate_static(runner, None, calib)
    section_b(runner, chunks, static)
    section_c(runner, chunks)
    section_d(runner, chunks)
    section_e(runner, texts["wiki"], tok)
    section_f(runner, tok, texts)

    F["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(F, fh, indent=2, default=lambda o: str(o))
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
