"""Project 13 -- KV-quantization study.

Stores the KV cache in fewer bits and asks, for each format, two questions the
guide insists on pairing: *how much memory did it save* and *what did it cost
in quality*. Quality is measured the way serving actually uses the cache -- a
prompt is prefilled, then real text is stepped through one token at a time, so
every measured token is predicted from a cache that has been through the
quantizer.

    python3 run.py           # ~5 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

import torch  # noqa: E402
import kvlib  # noqa: E402
from quantcache import QuantCache, SinkProtectedCache  # noqa: E402

PREFILL = 256
EVAL_STEPS = 160


def variants(n_layers):
    """(label, factory, bits-per-value-for-the-plot)."""
    Q = lambda **kw: (lambda: QuantCache(n_layers, **kw))   # noqa: E731
    return [
        ("fp32 (baseline)", lambda: kvlib.ContiguousCache(n_layers), 32),
        ("bf16", Q(narrow_dtype=torch.bfloat16), 16),
        ("fp8 e4m3", Q(narrow_dtype=torch.float8_e4m3fn), 8),
        ("fp8 e5m2", Q(narrow_dtype=torch.float8_e5m2), 8),
        ("int8 per-tensor", Q(bits=8, granularity="tensor"), 8),
        ("int8 per-token", Q(bits=8, granularity="token"), 8),
        ("int8 per-channel", Q(bits=8, granularity="channel"), 8),
        ("int4 per-token", Q(bits=4, granularity="token"), 4),
        ("int4, K only", Q(bits=4, granularity="token", quant_v=False), 4),
        ("int4, V only", Q(bits=4, granularity="token", quant_k=False), 4),
        ("int4 + fp32 sink",
         lambda: SinkProtectedCache(n_layers, keep_first=4, bits=4,
                                    granularity="token"), 4),
    ]


@torch.inference_mode()
def evaluate(runner, cache, ids, baseline_argmax=None):
    """Prefill, then teacher-force real tokens through the cache.

    Teacher forcing means: at every step we feed the token the corpus actually
    has, not the one the model predicted. All variants therefore see exactly
    the same context, so any difference in the numbers comes from the cache
    and nothing else.
    """
    nll, argmaxes = [], []
    t0 = time.perf_counter()
    logits = runner.forward(ids[:, :PREFILL], cache, start_pos=0)
    prefill_s = time.perf_counter() - t0
    step_s = []
    for i in range(EVAL_STEPS):
        target = int(ids[0, PREFILL + i])
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        nll.append(-float(logp[target]))
        argmaxes.append(int(logits[0, -1].argmax()))
        t0 = time.perf_counter()
        logits = runner.forward(ids[:, PREFILL + i:PREFILL + i + 1], cache,
                                start_pos=PREFILL + i)
        step_s.append(time.perf_counter() - t0)
    res = {"ppl": math.exp(statistics.mean(nll)),
           "mean_nll": statistics.mean(nll),
           "prefill_s": prefill_s,
           "median_step_s": statistics.median(step_s),
           "argmax": argmaxes}
    if baseline_argmax is not None:
        agree = sum(a == b for a, b in zip(argmaxes, baseline_argmax))
        res["top1_agreement"] = agree / len(argmaxes)
        first = next((i for i, (a, b) in enumerate(zip(argmaxes, baseline_argmax))
                      if a != b), None)
        res["first_disagreement_step"] = first
    if hasattr(cache, "stored_bytes"):
        res["stored_bytes"] = cache.stored_bytes()
        res["err"] = cache.mean_err()
    else:
        res["stored_bytes"] = cache.nbytes()
        res["err"] = {"k": 0.0, "v": 0.0}
    return res


def main():
    f = {"config": {"prefill": PREFILL, "eval_steps": EVAL_STEPS}}
    runner, tok, _ = kvlib.load_runner()
    text = kvlib.wikitext_lines(40_000)
    ids = tok(text, return_tensors="pt").input_ids[:, :PREFILL + EVAL_STEPS + 1]
    assert ids.shape[1] >= PREFILL + EVAL_STEPS + 1
    print(f"corpus: wikitext-2 test, {ids.shape[1]} tokens "
          f"({PREFILL} prefill + {EVAL_STEPS} scored)")

    rows = []
    base_argmax = None
    for label, factory, bits in variants(runner.n_layers):
        cache = factory()
        r = evaluate(runner, cache, ids, base_argmax)
        if base_argmax is None:
            base_argmax = r["argmax"]
            r["top1_agreement"] = 1.0
            r["first_disagreement_step"] = None
        r.pop("argmax")
        r.update({"label": label, "bits": bits})
        rows.append(r)
        print(f"   {label:>18}: ppl {r['ppl']:7.3f}  "
              f"cache {r['stored_bytes']/1e6:6.2f} MB  "
              f"top-1 agree {r['top1_agreement']*100:5.1f}%  "
              f"K-err {r['err']['k']:.2e}  V-err {r['err']['v']:.2e}", flush=True)
    f["variants"] = rows

    base = rows[0]
    for r in rows:
        r["ppl_delta_pct"] = 100 * (r["ppl"] - base["ppl"]) / base["ppl"]
        r["bytes_vs_fp32"] = r["stored_bytes"] / base["stored_bytes"]

    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    keys = ["label", "bits", "ppl", "ppl_delta_pct", "stored_bytes",
            "bytes_vs_fp32", "top1_agreement", "first_disagreement_step",
            "median_step_s"]
    with open(os.path.join(OUT, "variants.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(keys + ["k_err", "v_err"])
        for r in f["variants"]:
            w.writerow([r.get(k) for k in keys] + [r["err"]["k"], r["err"]["v"]])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = f["variants"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1: the memory/quality plane -- the only plot that matters here.
    for r in rows:
        ax[0].scatter(r["stored_bytes"] / 1e6, r["ppl"], s=70)
        ax[0].annotate(r["label"], (r["stored_bytes"] / 1e6, r["ppl"]),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[0].axhline(rows[0]["ppl"], color="k", ls="--", lw=1)
    ax[0].set_xlabel("cache size (MB) -- smaller is cheaper")
    ax[0].set_ylabel("perplexity -- lower is better")
    ax[0].set_title("A. every format, both axes at once")
    ax[0].grid(alpha=.3)

    # 2: perplexity cost as a bar chart, sorted.
    s = sorted(rows, key=lambda r: r["ppl_delta_pct"])
    cols = ["tab:green" if r["ppl_delta_pct"] < 1 else
            "tab:orange" if r["ppl_delta_pct"] < 10 else "tab:red" for r in s]
    ax[1].barh([r["label"] for r in s], [r["ppl_delta_pct"] for r in s], color=cols)
    ax[1].set_xlabel("perplexity vs fp32 (%)")
    ax[1].set_title("B. what each format costs in quality")
    ax[1].grid(alpha=.3, axis="x")
    ax[1].tick_params(labelsize=8)

    # 3: which half of the cache is the fragile one.
    kv = [r for r in rows if r["err"]["k"] > 0 or r["err"]["v"] > 0]
    x = range(len(kv))
    ax[2].bar([i - 0.2 for i in x], [r["err"]["k"] for r in kv], width=0.4,
              label="K error")
    ax[2].bar([i + 0.2 for i in x], [r["err"]["v"] for r in kv], width=0.4,
              label="V error")
    ax[2].set_xticks(list(x))
    ax[2].set_xticklabels([r["label"] for r in kv], rotation=40, ha="right",
                          fontsize=7)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("mean squared reconstruction error")
    ax[2].set_title("C. keys are harder to store than values")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv_quant.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
