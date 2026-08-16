"""Project 01 — Manual inference loop.

Writes the prefill/decode loop by hand, proves it matches `model.generate()`,
and then measures the two things the loop exists to teach:

  A. correctness  — manual loop == HF generate, token for token
  B. the cache    — same output with and without a KV cache, very different cost
  C. prefill      — cost grows with prompt length (compute-bound shape)
  D. decode       — cost barely grows with batch size (memory-bound shape)
  E. roofline     — the arithmetic that explains C and D

Run:  python3 run.py          (~3 min on 6 CPU threads)
      python3 run.py --plot   (redraw the figure from outputs/findings.json)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, HERE)

import loop_lib as L  # noqa: E402
import torch  # noqa: E402

PROMPT = "Once upon a time in a distant kingdom, a young engineer discovered"


# ---------------------------------------------------------------------------


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def section_a(tok, model, ids, findings, n_new=24):
    """Manual loop vs transformers' own greedy generate.

    Twice: once with the model's shipped generation_config (which quietly
    carries repetition_penalty=1.1), once with that penalty switched off.
    """
    mine = L.generate_with_cache(model, ids, max_new_tokens=n_new)
    with torch.inference_mode():
        ref_default = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        ref_raw = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                                 repetition_penalty=1.0,
                                 pad_token_id=tok.eos_token_id)
    d_ids = ref_default[0, ids.shape[1]:].tolist()
    r_ids = ref_raw[0, ids.shape[1]:].tolist()

    findings["A_correctness"] = {
        "manual_token_ids": mine.token_ids,
        "hf_default_token_ids": d_ids,
        "hf_raw_argmax_token_ids": r_ids,
        "matches_hf_default": mine.token_ids == d_ids,
        "matches_hf_raw_argmax": mine.token_ids == r_ids,
        "first_divergence_vs_default": first_divergence(mine.token_ids, d_ids),
        "shipped_repetition_penalty": 1.1,
        "manual_text": tok.decode(mine.token_ids),
        "hf_default_text": tok.decode(d_ids),
        "prefill_s": round(mine.prefill_s, 4),
        "median_decode_step_s": round(mine.median_itl_s, 4),
    }
    print(f"  A: manual == generate(default):     {mine.token_ids == d_ids} "
          f"(first divergence at token "
          f"{findings['A_correctness']['first_divergence_vs_default']})")
    print(f"  A: manual == generate(rep_pen=1.0): {mine.token_ids == r_ids}")
    print(f"     {tok.decode(mine.token_ids)[:70]!r}")
    return mine


def section_b(tok, model, findings, n_new=32, prompt_tokens=512):
    """The KV cache: identical tokens, very different wall clock.

    Uses a long (512-token) prompt on purpose: the cost the cache removes is
    "re-read the prefix", so it only becomes visible when the prefix is long.
    """
    torch.manual_seed(0)
    ids = torch.randint(1000, 20000, (1, prompt_tokens))
    cached = L.generate_with_cache(model, ids, max_new_tokens=n_new)
    plain = L.generate_no_cache(model, ids, max_new_tokens=n_new)
    cached_total = cached.prefill_s + sum(cached.decode_step_s)
    plain_total = plain.prefill_s + sum(plain.decode_step_s)

    # Per-step curves: with a cache the step cost is flat, without it, linear.
    findings["B_cache_vs_no_cache"] = {
        "n_new_tokens": n_new,
        "prompt_tokens": int(ids.shape[1]),
        "identical_output": cached.token_ids == plain.token_ids,
        "cached_total_s": round(cached_total, 3),
        "no_cache_total_s": round(plain_total, 3),
        "speedup": round(plain_total / cached_total, 2),
        "cached_step_s": [round(x, 4) for x in cached.decode_step_s],
        "no_cache_step_s": [round(x, 4) for x in plain.decode_step_s],
        "cached_first_vs_last_step": round(
            cached.decode_step_s[-1] / cached.decode_step_s[0], 2),
        "no_cache_first_vs_last_step": round(
            plain.decode_step_s[-1] / plain.decode_step_s[0], 2),
    }
    print(f"  B: cache {cached_total:.2f}s vs no-cache {plain_total:.2f}s "
          f"({plain_total / cached_total:.2f}x), same tokens: "
          f"{cached.token_ids == plain.token_ids}")


def section_c(model, findings, lengths=(16, 64, 256, 1024)):
    """Prefill: one pass over L tokens. Compute-bound => time grows with L."""
    rows = []
    with torch.inference_mode():
        fns = {}
        for n in lengths:
            batch = torch.randint(0, 5000, (1, n))
            fns[n] = (lambda b=batch: model(b, use_cache=True))
        best = L.interleaved(fns, rounds=3, warmup=1)
    for n in lengths:
        rows.append({"prompt_tokens": n,
                     "prefill_s": round(best[n], 4),
                     "tokens_per_s": round(n / best[n], 1),
                     "ms_per_token": round(1000 * best[n] / n, 3)})
        print(f"  C: prefill {n:5d} tok  {best[n]:.3f}s  "
              f"{n / best[n]:7.1f} tok/s")
    findings["C_prefill_scaling"] = rows
    return rows


def section_d(model, findings, batches=(1, 2, 4, 8, 16, 32), ctx=64):
    """Decode: one token per sequence. Memory-bound => step time barely moves."""
    rows = []
    with torch.inference_mode():
        prepared = {}
        for b in batches:
            warm = model(torch.randint(0, 5000, (b, ctx)), use_cache=True)
            prepared[b] = (warm.past_key_values, torch.randint(0, 5000, (b, 1)))
        fns = {}
        for b in batches:
            past, nid = prepared[b]

            def one_step(past=past, nid=nid):
                model(nid, past_key_values=copy.deepcopy(past), use_cache=True)

            fns[b] = one_step
        best = L.interleaved(fns, rounds=3, warmup=1)
    base = best[batches[0]]
    for b in batches:
        rows.append({"batch": b,
                     "step_s": round(best[b], 4),
                     "tokens_per_s": round(b / best[b], 1),
                     "step_cost_vs_b1": round(best[b] / base, 2),
                     "throughput_vs_b1": round((b / best[b]) / (1 / base), 2)})
        print(f"  D: decode B={b:3d}  step {best[b] * 1000:6.1f} ms  "
              f"{b / best[b]:7.1f} tok/s  (step cost {best[b] / base:.2f}x b=1)")
    findings["D_decode_batch_scaling"] = rows
    return rows


def section_e(shape, findings, prefill_rows, decode_rows, ctx=64):
    """The arithmetic that predicts C and D: bytes moved vs FLOPs done."""
    w_bytes = shape["weight_bytes_fp32"]
    kv_per_tok = L.kv_bytes_per_token(shape, dtype_bytes=4)
    # Two FLOPs (one multiply, one add) per parameter per token.
    flops_per_token = 2 * shape["n_params"]

    rows = []
    for b in (1, 8, 32):
        bytes_read = w_bytes + b * ctx * kv_per_tok
        flops = b * flops_per_token
        rows.append({"batch": b,
                     "bytes_read_per_step": bytes_read,
                     "flops_per_step": flops,
                     "arithmetic_intensity_flop_per_byte": round(flops / bytes_read, 2)})
    # Measured effective bandwidth at batch 1: we must read the weights once.
    b1 = [r for r in decode_rows if r["batch"] == 1][0]
    eff_bw = w_bytes / b1["step_s"] / 1e9
    p_long = prefill_rows[-1]
    d_1 = b1["tokens_per_s"]
    findings["E_roofline"] = {
        "weight_bytes_fp32": w_bytes,
        "kv_bytes_per_token_fp32": kv_per_tok,
        "flops_per_token": flops_per_token,
        "intensity": rows,
        "decode_effective_bandwidth_GB_s": round(eff_bw, 1),
        "prefill_tok_per_s_at_1024": p_long["tokens_per_s"],
        "decode_tok_per_s_at_b1": d_1,
        "prefill_over_decode": round(p_long["tokens_per_s"] / d_1, 1),
    }
    print(f"  E: decode moves {w_bytes / 1e9:.2f} GB/step -> "
          f"{eff_bw:.1f} GB/s effective; prefill is "
          f"{p_long['tokens_per_s'] / d_1:.1f}x faster per token")


# ---------------------------------------------------------------------------


def plot(findings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    b = findings["B_cache_vs_no_cache"]
    ax[0].plot([1000 * x for x in b["cached_step_s"]], "o-", label="with KV cache")
    ax[0].plot([1000 * x for x in b["no_cache_step_s"]], "s-", label="no cache")
    ax[0].set_xlabel("decode step")
    ax[0].set_ylabel("time for that step (ms)")
    ax[0].set_title(f"B. one step's cost\n(no-cache total is "
                    f"{b['speedup']}x the cached total)")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    c = findings["C_prefill_scaling"]
    ax[1].plot([r["prompt_tokens"] for r in c],
               [r["tokens_per_s"] for r in c], "o-", label="prefill")
    d1 = [r for r in findings["D_decode_batch_scaling"] if r["batch"] == 1][0]
    ax[1].axhline(d1["tokens_per_s"], color="crimson", ls="--",
                  label=f"decode, batch 1 ({d1['tokens_per_s']} tok/s)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("prompt tokens")
    ax[1].set_ylabel("tokens / second")
    ax[1].set_title("C. prefill is fast per token,\ndecode is not")
    ax[1].legend()
    ax[1].grid(alpha=.3)

    d = findings["D_decode_batch_scaling"]
    bs = [r["batch"] for r in d]
    ax2 = ax[2]
    ax2.plot(bs, [r["tokens_per_s"] for r in d], "o-", color="tab:green",
             label="throughput (tok/s)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("decode batch size")
    ax2.set_ylabel("tokens / second", color="tab:green")
    ax2.grid(alpha=.3)
    ax3 = ax2.twinx()
    ax3.plot(bs, [1000 * r["step_s"] for r in d], "s--", color="tab:red",
             label="step time (ms)")
    ax3.set_ylabel("time for one decode step (ms)", color="tab:red")
    ax3.set_ylim(0, max(1000 * r["step_s"] for r in d) * 1.6)
    ax2.set_title("D. 32x the work for "
                  f"{d[-1]['step_cost_vs_b1']}x the time\n(this is what "
                  "'memory-bound' looks like)")
    fig.tight_layout()
    path = os.path.join(OUT, "manual_loop.png")
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")

    if "--plot" in sys.argv:
        with open(fpath) as f:
            plot(json.load(f))
        return

    t_start = time.time()
    print("loading model ...")
    tok, model = L.load()
    shape = L.model_shape(model)
    print(f"  {shape['n_params'] / 1e6:.0f}M params, {shape['n_layers']} layers, "
          f"{shape['n_kv_heads']} kv-heads, vocab {shape['vocab_size']}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    findings = {"model": shape, "prompt": PROMPT,
                "prompt_tokens": int(ids.shape[1]), "threads": L.N_THREADS}

    section_a(tok, model, ids, findings)
    section_b(tok, model, findings)
    prefill_rows = section_c(model, findings)
    decode_rows = section_d(model, findings)
    section_e(shape, findings, prefill_rows, decode_rows)

    findings["wall_clock_s"] = round(time.time() - t_start, 1)
    with open(fpath, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"  wrote {fpath}")

    with open(os.path.join(OUT, "findings.csv"), "w") as f:
        f.write("section,key,value\n")
        for r in findings["C_prefill_scaling"]:
            f.write(f"C_prefill,tok_per_s@{r['prompt_tokens']},{r['tokens_per_s']}\n")
        for r in findings["D_decode_batch_scaling"]:
            f.write(f"D_decode,tok_per_s@b{r['batch']},{r['tokens_per_s']}\n")
            f.write(f"D_decode,step_ms@b{r['batch']},{round(1000 * r['step_s'], 2)}\n")
        e = findings["E_roofline"]
        for k in ("decode_effective_bandwidth_GB_s", "prefill_over_decode"):
            f.write(f"E_roofline,{k},{e[k]}\n")
        b = findings["B_cache_vs_no_cache"]
        f.write(f"B_cache,speedup,{b['speedup']}\n")

    plot(findings)
    print(f"done in {findings['wall_clock_s']}s")


if __name__ == "__main__":
    main()
