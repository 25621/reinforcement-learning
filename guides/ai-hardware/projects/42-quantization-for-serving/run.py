"""Project 42 -- Quantization for serving.

Project 34 asked what quantization costs in quality.  This one asks what it buys
a *server*, which is a different question with a different answer: on hardware
with no low-precision matmul instruction the arithmetic gets no faster at all,
and the win arrives through the back door -- smaller weights leave more room for
the KV cache, and KV cache is what limits how many users you can serve at once.

Sections:
  A. quality and size of five weight formats, measured on the phase-8 engine
  B. decode speed when weights must be de-quantized on the fly (no INT4 kernel)
  C. the flagship: a fixed 3 GB memory budget, split between weights and KV
  D. the same arithmetic on an 80 GB H100 serving Llama-3 8B

Runs in about 70 seconds on 12 CPU threads.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "39-deploy-with-vllm"))
sys.path.insert(0, os.path.join(HERE, "..", "34-quantize-a-small-llm"))
import servelib as S  # noqa: E402
import quantlib as ql  # noqa: E402

OUT = S.outdir(__file__)
PPL_SEQ, SEQLEN = 4, 512
BUDGET_GB = 3.0
CTX = 1024
LINEARS = ["wq", "wk", "wv", "wo", "wgate", "wup", "wdown"]

results = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------ weight formats
def fp8_quant(W, per_channel=True):
    """Cast to FP8 E4M3 with a scale, then back -- what an FP8 kernel would see.

    E4M3's largest value is 448, so a raw cast would flush most weights to zero
    or to infinity depending on scale.  We first rescale each output channel to
    fill the range, exactly as an FP8 serving kernel does.
    """
    dim = 1 if per_channel else None
    amax = W.abs().amax(dim=dim, keepdim=True) if per_channel else W.abs().amax()
    scale = amax.clamp(min=1e-8) / 448.0
    q = (W / scale).to(torch.float8_e4m3fn)
    return q.float() * scale


FORMATS = [
    ("fp32 (baseline)", None, 32.0),
    ("fp8 e4m3, per-channel", lambda W: fp8_quant(W), 8.0),
    ("int8, per-channel", lambda W: ql.fake_quant(W, 8), 8.0 + 16 / 896),
    ("int4, group 128", lambda W: ql.fake_quant(W, 4, 128), 4.125),
    ("int3, group 128", lambda W: ql.fake_quant(W, 3, 128), 3.125),
]


def apply_format(w, fn):
    """Fake-quantize every transformer-block matrix; keep embeddings in fp32."""
    saved = []
    for L in w.layers:
        for k in LINEARS:
            saved.append((L, k, L[k]))
            L[k] = fn(L[k])
    return saved


def restore(saved):
    for L, k, t in saved:
        L[k] = t


# ------------------------------------------------------------------- quality
@torch.no_grad()
def engine_perplexity(w, batches):
    """Perplexity measured through the serving engine itself, not a side model."""
    total_nll, total_tok = 0.0, 0
    for i in range(batches.shape[0]):
        ids = batches[i].tolist()
        eng = S.Engine(w, num_blocks=len(ids) // 16 + 2)
        seq = S.Sequence(0, ids)
        logits = eng.forward([seq], [ids], last_only=False)
        nll = torch.nn.functional.cross_entropy(
            logits[:-1].float(), torch.tensor(ids[1:]), reduction="sum")
        total_nll += float(nll)
        total_tok += len(ids) - 1
    return float(torch.exp(torch.tensor(total_nll / total_tok)))


def section_a(w):
    log("A. quality and size of five weight formats")
    text = ql.wikitext_text()
    batches = ql.token_batches(w.tok, text, PPL_SEQ, SEQLEN)
    n_quant = sum(L[k].numel() for L in w.layers for k in LINEARS)
    n_other = w.embed.numel() + w.final_norm.numel() + \
        sum(L[k].numel() for L in w.layers for k in ("ln1", "ln2", "bq", "bk", "bv"))
    rows = []
    base_ppl = None
    base_gb = (n_quant * 4 + n_other * 4) / 1e9
    for name, fn, bpw in FORMATS:
        saved = apply_format(w, fn) if fn else []
        t0 = time.perf_counter()
        ppl = engine_perplexity(w, batches)
        restore(saved)
        if base_ppl is None:
            base_ppl = ppl
        gb = (n_quant * bpw / 8 + n_other * 4) / 1e9
        rows.append(dict(name=name, bits_per_weight=bpw, ppl=ppl,
                         ppl_ratio=ppl / base_ppl, weights_GB=gb,
                         shrink=base_gb / gb, seconds=time.perf_counter() - t0))
        log(f"   {name:24s} ppl {ppl:8.3f} ({ppl / base_ppl:5.3f}x)  "
            f"weights {gb:.3f} GB ({base_gb / gb:.2f}x smaller)")
    results["formats"] = rows
    results["quantized_params"] = n_quant
    results["other_params"] = n_other
    return rows


# --------------------------------------------------------------------- speed
class DequantOnUse:
    """An int8 weight matrix that reconstructs itself every time it is used.

    Real INT8/INT4 serving kernels *fuse* the reconstruction into the matmul, so
    the wide values never touch memory.  We have no such kernel on this CPU, so
    this class does the honest, unfused thing -- and section B measures what that
    costs.  `.T` is what the engine's forward pass asks for.
    """

    def __init__(self, W, bits=8, group=None):
        q, scale, zero, shape = ql.quantize(W, bits, group)
        self.q = q.to(torch.int8)          # store small, as a real engine would
        self.scale, self.zero, self.shape = scale, zero, shape
        self.bytes = q.numel() * bits / 8 + scale.numel() * 2

    @property
    def T(self):
        return ql.dequantize(self.q, self.scale, self.zero, self.shape).T


def section_b(w):
    log("\nB. decode speed with no low-precision kernel (batch 1 and 16, ctx 512)")
    rows = []
    for name, bits, group in [("fp32", None, None), ("int8 per-channel", 8, None),
                              ("int4 group 128", 4, 128)]:
        saved = []
        if bits:
            for L in w.layers:
                for k in LINEARS:
                    saved.append((L, k, L[k]))
                    L[k] = DequantOnUse(L[k], bits, group)
        row = dict(name=name)
        for B in (1, 16):
            eng = S.Engine(w, num_blocks=B * (512 // 16 + 2))
            seqs = S.synthetic_seqs(eng, B, 512)
            row[f"step_ms_b{B}"] = S.time_decode(eng, seqs, rounds=3) * 1e3
            row[f"tok_s_b{B}"] = B / (row[f"step_ms_b{B}"] / 1e3)
            del eng, seqs
        restore(saved)
        rows.append(row)
        log(f"   {name:18s} batch 1: {row['step_ms_b1']:7.1f} ms   "
            f"batch 16: {row['step_ms_b16']:7.1f} ms "
            f"({row['tok_s_b16']:.1f} tok/s)")
    base = rows[0]
    for r in rows[1:]:
        r["slowdown_b1"] = r["step_ms_b1"] / base["step_ms_b1"]
        r["slowdown_b16"] = r["step_ms_b16"] / base["step_ms_b16"]
        log(f"   -> {r['name']} is {r['slowdown_b1']:.2f}x the fp32 step time at "
            f"batch 1, {r['slowdown_b16']:.2f}x at batch 16")
    results["speed"] = rows
    return rows


# ------------------------------------------------- the fixed-budget experiment
def section_c(w):
    log(f"\nC. one {BUDGET_GB:.0f} GB budget, split between weights and KV "
        f"(context {CTX})")
    bpt = S.KVPool(w, 1, 16).bytes_per_token()
    n_quant = results["quantized_params"]
    n_other = results["other_params"]
    rows = []
    for name, bpw in [("fp32 weights", 32.0), ("int8 weights", 8.0),
                      ("int4 weights", 4.125)]:
        wgb = (n_quant * bpw / 8 + n_other * 4) / 1e9
        kv_bytes = BUDGET_GB * 1e9 - wgb * 1e9
        seats = int(kv_bytes / (bpt * CTX))
        blocks = seats * (CTX // 16 + 1)
        eng = S.Engine(w, num_blocks=blocks)
        seqs = S.synthetic_seqs(eng, seats, CTX)
        step = S.time_decode(eng, seqs, rounds=3)
        rows.append(dict(name=name, weights_GB=wgb, kv_GB=kv_bytes / 1e9,
                         seats=seats, step_ms=step * 1e3, tok_s=seats / step,
                         user_tok_s=1 / step))
        log(f"   {name:13s} weights {wgb:.2f} GB -> {kv_bytes / 1e9:.2f} GB of KV "
            f"= {seats:3d} concurrent requests, measured {seats / step:6.1f} tok/s "
            f"({step * 1e3:.0f} ms/step)")
        del eng, seqs
    gain = rows[1]["tok_s"] / rows[0]["tok_s"]
    results["budget"] = dict(budget_GB=BUDGET_GB, context=CTX,
                             bytes_per_token=bpt, rows=rows,
                             int8_throughput_gain=gain,
                             int8_seat_gain=rows[1]["seats"] / rows[0]["seats"])
    log(f"   -> int8 weights fit {rows[1]['seats'] / rows[0]['seats']:.1f}x more "
        f"requests and deliver {gain:.2f}x the throughput, on a CPU whose matmul "
        f"never once ran in int8")
    return rows


# ----------------------------------------------------- D. the same on an H100
def section_d():
    log("\nD. the same arithmetic on an 80 GB H100 serving Llama-3 8B, context 8192")
    params, layers, kv_heads, head_dim = 8.03e9, 32, 8, 128
    hbm_GB, hbm_BW = 80.0, 3350e9        # H100 SXM: 80 GB at 3.35 TB/s
    kv_bpt = 2 * layers * kv_heads * head_dim * 2
    rows = []
    for name, bpw in [("fp16", 16), ("fp8", 8), ("int4 (AWQ/GPTQ)", 4.25)]:
        wgb = params * bpw / 8 / 1e9
        free = hbm_GB * 1e9 - wgb * 1e9 - 4e9      # 4 GB for activations etc.
        seats = free / (kv_bpt * 8192)
        # Decode is bandwidth-bound: one step re-reads every weight once.
        step_s = params * bpw / 8 / hbm_BW
        rows.append(dict(name=name, weights_GB=wgb, seats=seats,
                         step_ms=step_s * 1e3, single_user_tok_s=1 / step_s,
                         server_tok_s=seats / step_s))
        log(f"   {name:16s} weights {wgb:5.2f} GB | {seats:5.1f} concurrent 8k "
            f"requests | {1 / step_s:6.0f} tok/s for one user | "
            f"{seats / step_s:8.0f} tok/s in total")
    results["h100"] = dict(kv_bytes_per_token=kv_bpt, rows=rows,
                           int4_vs_fp16_total=rows[2]["server_tok_s"] / rows[0]["server_tok_s"])
    log(f"   -> int4 weights: {rows[2]['single_user_tok_s'] / rows[0]['single_user_tok_s']:.1f}x "
        f"faster for one user, "
        f"{rows[2]['server_tok_s'] / rows[0]['server_tok_s']:.1f}x more total "
        f"throughput once the freed memory is spent on batch")


# -------------------------------------------------------------------- figures
def make_plots(res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    f = res["formats"]
    names = [r["name"] for r in f]
    ppl = [r["ppl"] for r in f]
    colors = ["#1f77b4" if r["ppl_ratio"] < 1.5 else "#d62728" for r in f]
    ax[0].barh(names, ppl, color=colors)
    for i, r in enumerate(f):
        ax[0].text(r["ppl"], i, f"  {r['ppl']:.2f} ({r['weights_GB']:.2f} GB)",
                   va="center", fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("WikiText-2 perplexity (log scale)")
    ax[0].set_title("Quality vs format")
    ax[0].invert_yaxis()
    ax[0].set_xlim(right=max(ppl) * 3)

    b = res["budget"]["rows"]
    x = range(len(b))
    ax[1].bar([i - 0.2 for i in x], [r["weights_GB"] for r in b], 0.4,
              label="weights", color="#ff7f0e")
    ax[1].bar([i + 0.2 for i in x], [r["kv_GB"] for r in b], 0.4,
              label="KV cache", color="#1f77b4")
    for i, r in enumerate(b):
        ax[1].text(i + 0.2, r["kv_GB"], f"{r['seats']} seats", ha="center",
                   va="bottom", fontsize=8)
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels([r["name"] for r in b], fontsize=8)
    ax[1].set_ylabel("GB")
    ax[1].set_title(f"A fixed {res['budget']['budget_GB']:.0f} GB, split two ways")
    ax[1].legend(fontsize=8)

    ax2 = ax[2]
    tok = [r["tok_s"] for r in b]
    ax2.bar([r["name"] for r in b], tok, color="#2ca02c")
    for i, t in enumerate(tok):
        ax2.text(i, t, f"{t:.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("measured decode tokens/s")
    ax2.set_title("Throughput that quantization actually bought")
    ax2.tick_params(axis="x", labelsize=8)
    fig.savefig(f"{OUT}/quant_serving.png", dpi=130)
    log(f"   wrote {OUT}/quant_serving.png")


def main():
    S.setup()
    t0 = time.time()
    w = S.Weights(S.SMALL)
    section_a(w)
    section_b(w)
    section_c(w)
    section_d()
    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("format,bits_per_weight,ppl,ppl_ratio,weights_GB\n")
        for r in results["formats"]:
            f.write(f"\"{r['name']}\",{r['bits_per_weight']:.3f},{r['ppl']:.4f},"
                    f"{r['ppl_ratio']:.4f},{r['weights_GB']:.4f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
