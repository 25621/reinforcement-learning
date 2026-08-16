"""Project 09 -- KV cache from scratch.

Builds the phase's engine (`kvlib.Qwen2Runner`: a hand-written Qwen2 forward
pass with a pluggable cache) and then answers four questions with numbers:

  A. Is our hand-written forward pass actually the same model? (vs HuggingFace)
  B. Does the cache change the output, or only the time?
  C. How big does the cache get, and does the textbook formula predict it?
  D. What does a decode step cost as the cache grows?

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw the figure from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import kvlib  # noqa: E402
import torch  # noqa: E402


def build_prompt(tok, n_tokens):
    """A prompt of roughly `n_tokens` tokens, made of real text."""
    body = ("The cache stores keys and values so that attention never "
            "recomputes them. ") * 400
    ids = tok(body, return_tensors="pt").input_ids[:, :n_tokens]
    return ids


def main():
    findings = {}
    runner, tok, model = kvlib.load_runner()
    findings["model"] = {
        "id": kvlib.MODEL_ID,
        "n_layers": runner.n_layers,
        "n_heads": runner.n_heads,
        "n_kv_heads": runner.n_kv_heads,
        "d_head": runner.d_head,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    # ---------------------------------------------------------------- A
    # Our arithmetic vs HuggingFace's, greedy, same weights.
    print("A. correctness vs HuggingFace ...", flush=True)
    msgs = [{"role": "user", "content": "Explain what a KV cache is, briefly."}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat, return_tensors="pt").input_ids

    cache = kvlib.ContiguousCache(runner.n_layers)
    ours, _, _ = runner.generate(chat_ids, cache, max_new_tokens=24)
    with torch.inference_mode():
        hf = model.generate(chat_ids, max_new_tokens=24, do_sample=False,
                            repetition_penalty=1.0)
    hf_ids = hf[0][chat_ids.shape[1]:].tolist()[:len(ours)]

    # And a numeric check on the logits, not just the argmax.
    cache.reset()
    with torch.inference_mode():
        our_logits = runner.forward(chat_ids, cache)[0, -1]
        hf_logits = model(chat_ids).logits[0, -1]
    findings["A_correctness"] = {
        "our_text": tok.decode(ours),
        "hf_text": tok.decode(hf_ids),
        "tokens_identical": ours == hf_ids,
        "max_abs_logit_diff": float((our_logits - hf_logits).abs().max()),
        "logit_scale": float(hf_logits.abs().max()),
    }
    print("   identical tokens:", findings["A_correctness"]["tokens_identical"])

    # ---------------------------------------------------------------- B
    # Cache vs no cache: same text, very different clock.
    print("B. cache vs no cache ...", flush=True)
    P, N = 256, 24
    prompt = build_prompt(tok, P)

    cache = kvlib.ContiguousCache(runner.n_layers)
    t0 = time.perf_counter()
    ids_c, pf_c, steps_c = runner.generate(prompt, cache, max_new_tokens=N,
                                           stop_on_eos=False)
    total_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    ids_n, pf_n, steps_n = runner.generate_no_cache(prompt, max_new_tokens=N,
                                                    stop_on_eos=False)
    total_n = time.perf_counter() - t0

    findings["B_cache_vs_none"] = {
        "prompt_tokens": P, "new_tokens": N,
        "cached_total_s": total_c, "nocache_total_s": total_n,
        "speedup": total_n / total_c,
        "cached_prefill_s": pf_c, "nocache_first_pass_s": pf_n,
        "cached_step_s": steps_c, "nocache_step_s": steps_n,
        "cached_median_step_s": statistics.median(steps_c),
        "nocache_median_step_s": statistics.median(steps_n),
        "tokens_identical": ids_c == ids_n,
        "text": tok.decode(ids_c),
    }
    print(f"   {total_n / total_c:.2f}x faster, identical={ids_c == ids_n}")

    # ---------------------------------------------------------------- C
    # Cache size: measured bytes vs the formula.
    print("C. cache size ...", flush=True)
    per_tok_formula = kvlib.kv_bytes_per_token(
        runner.n_layers, runner.n_kv_heads, runner.d_head, dtype_bytes=4)
    growth = []
    cache = kvlib.ContiguousCache(runner.n_layers)
    runner.forward(prompt, cache)
    growth.append((cache.n_tokens(), cache.nbytes()))
    pos = P
    nxt = torch.tensor([[100]])
    for _ in range(8):
        runner.forward(nxt, cache, start_pos=pos)
        pos += 1
        growth.append((cache.n_tokens(), cache.nbytes()))
    measured_per_tok = (growth[-1][1] - growth[0][1]) / (growth[-1][0] - growth[0][0])
    findings["C_size"] = {
        "formula_bytes_per_token_fp32": per_tok_formula,
        "measured_bytes_per_token_fp32": measured_per_tok,
        "growth": growth,
        "weight_bytes": findings["model"]["n_params"] * 4,
    }
    print(f"   formula {per_tok_formula} B/token, measured {measured_per_tok:.0f}")

    # ---------------------------------------------------------------- D
    # Decode step cost as the cache grows. This is the read-traffic curve.
    print("D. decode cost vs context ...", flush=True)
    ctx_points = [128, 512, 1024, 2048, 4096]
    d_rows = []
    for ctx in ctx_points:
        p = build_prompt(tok, ctx)
        cache = kvlib.ContiguousCache(runner.n_layers)
        runner.forward(p, cache)
        nxt = torch.tensor([[100]])
        # 5 steps, keep the median; the cache grows by 5 tokens which is
        # negligible next to ctx.
        ts = []
        pos = ctx
        for _ in range(5):
            t0 = time.perf_counter()
            runner.forward(nxt, cache, start_pos=pos)
            ts.append(time.perf_counter() - t0)
            pos += 1
        kv_mb = cache.nbytes() / 1e6
        d_rows.append({"ctx": ctx, "step_s": statistics.median(ts),
                       "kv_mb": kv_mb})
        print(f"   ctx {ctx:5d}: {statistics.median(ts) * 1e3:7.1f} ms, "
              f"cache {kv_mb:6.1f} MB")
    findings["D_step_vs_ctx"] = d_rows

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    write_csv(findings)
    plot(findings)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["section", "key", "value"])
        b = f["B_cache_vs_none"]
        for k in ("cached_total_s", "nocache_total_s", "speedup",
                  "cached_median_step_s", "nocache_median_step_s",
                  "tokens_identical"):
            w.writerow(["B", k, b[k]])
        w.writerow(["C", "formula_bytes_per_token_fp32",
                    f["C_size"]["formula_bytes_per_token_fp32"]])
        w.writerow(["C", "measured_bytes_per_token_fp32",
                    round(f["C_size"]["measured_bytes_per_token_fp32"], 1)])
        for r in f["D_step_vs_ctx"]:
            w.writerow(["D", f"ctx_{r['ctx']}_step_ms", round(r["step_s"] * 1e3, 2)])
            w.writerow(["D", f"ctx_{r['ctx']}_kv_mb", round(r["kv_mb"], 2)])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    b = f["B_cache_vs_none"]
    ax[0].plot(range(1, len(b["cached_step_s"]) + 1),
               [s * 1e3 for s in b["cached_step_s"]], "o-", label="with KV cache")
    ax[0].plot(range(1, len(b["nocache_step_s"]) + 1),
               [s * 1e3 for s in b["nocache_step_s"]], "s-", label="no cache")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("decode step")
    ax[0].set_ylabel("ms per step (log)")
    ax[0].set_title(f"A. same tokens, {b['speedup']:.1f}x the time\n"
                    f"(prompt {b['prompt_tokens']}, {b['new_tokens']} new)")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    c = f["C_size"]
    n = [g[0] for g in c["growth"]]
    mb = [g[1] / 1e6 for g in c["growth"]]
    ax[1].plot(n, mb, "o-")
    ax[1].set_xlabel("tokens in cache")
    ax[1].set_ylabel("cache size (MB, fp32)")
    ax[1].set_title(f"B. {c['measured_bytes_per_token_fp32']:.0f} bytes/token\n"
                    f"(formula says {c['formula_bytes_per_token_fp32']})")
    ax[1].grid(alpha=.3)

    d = f["D_step_vs_ctx"]
    ax2 = ax[2]
    ax2.plot([r["ctx"] for r in d], [r["step_s"] * 1e3 for r in d], "o-",
             color="tab:red", label="decode step (ms)")
    ax2.set_xlabel("context length (tokens)")
    ax2.set_ylabel("ms per decode step", color="tab:red")
    ax2.set_xscale("log", base=2)
    ax3 = ax2.twinx()
    ax3.plot([r["ctx"] for r in d], [r["kv_mb"] for r in d], "s--",
             color="tab:blue")
    ax3.set_ylabel("cache size (MB)", color="tab:blue")
    ax2.set_title("C. the cache you keep is the cache you re-read")
    ax2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv_cache.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        with open(os.path.join(OUT, "findings.json")) as fh:
            plot(json.load(fh))
    else:
        main()
