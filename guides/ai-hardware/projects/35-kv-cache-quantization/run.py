"""Project 35 -- KV-cache quantization.

Stores the attention keys and values in INT8 / INT4 instead of FP16 and measures
what that costs in quality and buys in memory. Uses the same model as project 34
(Qwen2.5-0.5B-Instruct) so the numbers are comparable.

Runs in about 2 minutes on 12 CPU threads.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers.cache_utils import DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "34-quantize-a-small-llm"))
import quantlib as ql  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

SEQLEN, N_SEQ, TOKEN_GROUP = 2048, 2, 128
results = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------- the quantizers
def q_per_token(x, bits):
    """One scale per (token, head): the scale covers the head_dim axis.

    x is (batch, heads, tokens, head_dim). Every token's slice is finished the
    moment that token is generated, so a per-token scale can be computed online
    with no lookahead -- which is why every serving stack uses it for values.
    """
    return ql.fake_quant(x.reshape(-1, x.shape[-1]), bits, sym=False).reshape(x.shape)


def q_per_channel(x, bits, group=TOKEN_GROUP):
    """One scale per (channel, block of `group` tokens).

    Keys have a few channels whose magnitude is far above the rest. A per-token
    scale is set by those outliers and then wastes all its levels on them, so
    every other channel in that token collapses. Grouping along the token axis
    instead gives each channel its own scale. It needs `group` tokens in hand,
    which is why real systems (KIVI) keep the newest partial group in FP16.
    """
    B, H, T, D = x.shape
    pad = (-T) % group
    if pad:
        x = torch.cat([x, x[:, :, -1:].expand(B, H, pad, D)], dim=2)
    y = x.reshape(B, H, -1, group, D).transpose(-1, -2).reshape(-1, group)
    y = ql.fake_quant(y, bits, sym=False)
    y = y.reshape(B, H, -1, D, group).transpose(-1, -2).reshape(B, H, -1, D)
    return y[:, :, :T]


class QuantCache(DynamicCache):
    """A KV cache that stores keys and values at reduced precision.

    Config lives on the class so we can swap it between runs without fighting the
    library's constructor signature.
    """
    kbits = vbits = 16
    kmode = "per_channel"
    vmode = "per_token"
    sink = 0            # how many leading tokens stay in FP16

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        key_states = self._q(key_states, QuantCache.kbits, QuantCache.kmode)
        value_states = self._q(value_states, QuantCache.vbits, QuantCache.vmode)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    @staticmethod
    def _q(x, bits, mode):
        if bits >= 16:
            return x
        s = QuantCache.sink
        head, tail = (x[:, :, :s], x[:, :, s:]) if s else (None, x)
        tail = (q_per_token(tail, bits) if mode == "per_token"
                else q_per_channel(tail, bits))
        return torch.cat([head, tail], dim=2) if s else tail


@torch.no_grad()
def evaluate(model, batches, per_position=False):
    """Perplexity with the cache active, optionally bucketed by position."""
    total_nll, total_tok, preds = 0.0, 0, []
    bucket_nll = torch.zeros(8)
    bucket_n = torch.zeros(8)
    for i in range(batches.shape[0]):
        x = batches[i: i + 1]
        out = model(x, use_cache=True, past_key_values=QuantCache())
        logits = out.logits[:, :-1].float()
        target = x[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            reduction="none")
        total_nll += float(nll.sum())
        total_tok += nll.numel()
        preds.append(logits[0].argmax(-1))
        if per_position:
            keep = (nll.numel() // 8) * 8
            chunks = nll[:keep].reshape(8, -1)
            bucket_nll += chunks.sum(1)
            bucket_n += chunks.shape[1]
    ppl = float(torch.exp(torch.tensor(total_nll / total_tok)))
    per_pos = torch.exp(bucket_nll / bucket_n).tolist() if per_position else None
    return ppl, torch.cat(preds), per_pos


def make_plots(res):
    """Rebuild the figures from findings.json, as ratios to the FP16 baseline.

    Plotting raw perplexity hides everything interesting: the six good
    configurations all sit within 4% of each other and the two broken ones are
    7x away, so on one axis the good ones are a single indistinguishable smear.
    Dividing by the baseline puts 1.00 in the middle of the picture.
    """
    rows = res["quality"]
    base = rows[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    ratios = [r["ppl"] / base["ppl"] for r in rows]
    bad = ["#444444"] + ["#d62728" if x > 2 else "#1f77b4" for x in ratios[1:]]
    axes[0].barh([r["name"] for r in rows], ratios, color=bad)
    axes[0].axvline(1.0, ls="--", color="k", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("perplexity relative to the FP16 cache "
                       "(1.00 = free, log scale)")
    axes[0].invert_yaxis()
    axes[0].set_title("Cache precision vs quality")
    for r, ratio in zip(rows, ratios):
        axes[0].annotate(f"{ratio:.2f}x", (ratio, r["name"]), fontsize=8,
                         xytext=(4, -3), textcoords="offset points")

    seqlen = res["seqlen"]
    n = len(base["per_position_ppl"])
    xs = [(i + 0.5) * seqlen / n for i in range(n)]
    ok_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    bad_colors = ["#d62728", "#ff7f0e"]
    for r, ratio in zip(rows[1:], ratios[1:]):
        rel = [a / b for a, b in zip(r["per_position_ppl"],
                                     base["per_position_ppl"])]
        color = bad_colors.pop(0) if ratio > 2 else ok_colors.pop(0)
        axes[1].plot(xs, rel, marker="o", markersize=3, label=r["name"],
                     color=color)
    axes[1].axhline(1.0, ls="--", color="k", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("position in the sequence (token index)")
    axes[1].set_ylabel("perplexity relative to the FP16 cache")
    axes[1].set_title("Does the damage grow with context length?")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Quantizing the KV cache of " + ql.MODEL)
    fig.savefig(f"{OUT}/kv_quant.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/kv_quant.png")


def main():
    ql.setup()
    t_start = time.time()
    tok, model = ql.load(ql.MODEL)
    cfg = model.config

    # ---------------------------------------------------------- A. the math
    log("=== A. How big is a KV cache, exactly? ===")
    n_layer = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    elems_per_token = 2 * n_layer * n_kv * head_dim      # 2 = one K and one V
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  layers {n_layer}, key/value heads {n_kv}, head_dim {head_dim}")
    log(f"  elements cached per token = 2 x {n_layer} x {n_kv} x {head_dim} "
        f"= {elems_per_token}")
    log(f"  FP16 -> {elems_per_token * 2 / 1024:.1f} KiB per token, "
        f"INT8 -> {elems_per_token / 1024:.1f} KiB, "
        f"INT4 -> {elems_per_token * 0.5 / 1024:.1f} KiB")
    weights_mb = n_params * 2 / 1e6
    log(f"  model weights in FP16: {weights_mb:.0f} MB")
    crossover = int(weights_mb * 1e6 / (elems_per_token * 2))
    log(f"  a single FP16 cache equals the whole model at {crossover:,} tokens "
        f"of context")

    table = []
    for batch in [1, 8, 32]:
        for ctx in [4096, 32768, 131072]:
            row = {"batch": batch, "ctx": ctx}
            for bits, name in [(16, "fp16"), (8, "int8"), (4, "int4")]:
                row[name] = batch * ctx * elems_per_token * bits / 8 / 1e9
            table.append(row)
            log(f"  batch {batch:3d} x {ctx:7,d} tokens: "
                f"fp16 {row['fp16']:7.2f} GB   int8 {row['int8']:7.2f} GB   "
                f"int4 {row['int4']:7.2f} GB")
    results["memory_math"] = {
        "elems_per_token": elems_per_token, "n_layer": n_layer, "n_kv": n_kv,
        "head_dim": head_dim, "weights_MB": weights_mb,
        "crossover_tokens": crossover, "table": table}

    text = ql.wikitext_text()
    ev = ql.token_batches(tok, text, N_SEQ, SEQLEN)
    log(f"\n  evaluating on {N_SEQ} x {SEQLEN} = {N_SEQ * SEQLEN} tokens "
        f"of WikiText-2")

    # -------------------------------------- B. why keys and values differ
    log("\n=== B. Keys and values do not look alike ===")
    # Collect the tensors the cache actually stores. They are NOT the raw output
    # of k_proj: rotary position embeddings are applied first, and a cache holds
    # the post-rotary keys. Reading them from inside the cache is the only way to
    # measure what quantization will really see.
    grabbed = {}
    watch = n_layer // 2

    class StatCache(DynamicCache):
        def update(self, k, v, layer_idx, *args, **kwargs):
            if layer_idx == watch and "k" not in grabbed:
                grabbed["k"] = k.detach()[0].transpose(0, 1).reshape(
                    k.shape[2], -1).float()
                grabbed["v"] = v.detach()[0].transpose(0, 1).reshape(
                    v.shape[2], -1).float()
            return super().update(k, v, layer_idx, *args, **kwargs)

    with torch.no_grad():
        model(ev[:1, :512], use_cache=True, past_key_values=StatCache())

    stats = {}
    for kind in ["k", "v"]:
        x = grabbed[kind]                       # (tokens, channels)
        per_ch = x.abs().amax(0)
        per_tok = x.abs().amax(1)
        stats[kind] = {
            "channel_amax_max": float(per_ch.max()),
            "channel_amax_median": float(per_ch.median()),
            "channel_spread": float(per_ch.max() / per_ch.median()),
            "token_spread": float(per_tok.max() / per_tok.median()),
            "per_channel_amax": per_ch.tolist(),
        }
        log(f"  {kind}: biggest channel is "
            f"{stats[kind]['channel_spread']:6.1f}x the median channel; "
            f"biggest token is {stats[kind]['token_spread']:5.1f}x the median token")
    results["kv_stats"] = {k: {kk: vv for kk, vv in v.items()
                               if kk != "per_channel_amax"}
                           for k, v in stats.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, kind, title in zip(axes, ["k", "v"], ["Keys", "Values"]):
        ax.plot(stats[kind]["per_channel_amax"], lw=0.9,
                color="#d62728" if kind == "k" else "#2ca02c")
        ax.set_xlabel("channel")
        ax.set_ylabel("max |value| over 512 tokens")
        ax.set_title(f"{title}: spread "
                     f"{stats[kind]['channel_spread']:.0f}x across channels")
        ax.grid(alpha=0.3)
    fig.suptitle("A per-token scale has to cover every channel at once -- "
                 "that is why keys suffer")
    fig.savefig(f"{OUT}/kv_channels.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------ C. quality sweep
    log("\n=== C. What quantizing the cache costs ===")
    configs = [
        ("FP16 cache (baseline)", 16, 16, "per_channel", "per_token", 0),
        ("INT8 K+V, per-token", 8, 8, "per_token", "per_token", 0),
        ("INT8 K per-channel, V per-token", 8, 8, "per_channel", "per_token", 0),
        ("INT4 K+V, per-token", 4, 4, "per_token", "per_token", 0),
        ("INT4 K per-channel, V per-token", 4, 4, "per_channel", "per_token", 0),
        ("INT4 + 32 FP16 sink tokens", 4, 4, "per_channel", "per_token", 32),
        ("INT2 K per-channel, V per-token", 2, 2, "per_channel", "per_token", 0),
    ]
    rows, base_pred = [], None
    for name, kb, vb, km, vm, sink in configs:
        QuantCache.kbits, QuantCache.vbits = kb, vb
        QuantCache.kmode, QuantCache.vmode, QuantCache.sink = km, vm, sink
        with ql.Timer() as t:
            ppl, pred, per_pos = evaluate(model, ev, per_position=True)
        if base_pred is None:
            base_pred = pred
        bits_avg = (kb + vb) / 2
        rows.append({"name": name, "kbits": kb, "vbits": vb, "kmode": km,
                     "sink": sink, "ppl": ppl,
                     "agree": ql.agreement(pred, base_pred),
                     "cache_GB_at_32k": 32768 * elems_per_token * bits_avg / 8 / 1e9,
                     "per_position_ppl": per_pos, "seconds": t.dt})
        log(f"  {name:34s} ppl {ppl:8.3f}   agree {rows[-1]['agree'] * 100:5.1f}%"
            f"   ({t.dt:.0f}s)")
    QuantCache.kbits = QuantCache.vbits = 16
    results["quality"] = rows
    results["seqlen"] = SEQLEN

    # --------------------------------------------------- D. what it buys you
    log("\n=== D. What the saving is for ===")
    budget_gb = 8.0
    free = budget_gb - weights_mb / 1000
    log(f"  Suppose {budget_gb:.0f} GB of memory and "
        f"{weights_mb / 1000:.2f} GB of weights -> "
        f"{free:.2f} GB left for the cache.")
    seats = {}
    for bits, name in [(16, "fp16"), (8, "int8"), (4, "int4")]:
        per_seq = 32768 * elems_per_token * bits / 8 / 1e9
        seats[name] = int(free / per_seq)
        log(f"  {name}: {per_seq:.2f} GB per 32k-token sequence -> "
            f"{seats[name]} concurrent sequences")
    results["batch_capacity"] = {"budget_GB": budget_gb, "seats": seats}

    make_plots(results)
    results["total_seconds"] = time.time() - t_start
    ql.save_json(f"{OUT}/findings.json", results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("name,kbits,vbits,kmode,sink,ppl,agree,cache_GB_at_32k\n")
        for r in rows:
            f.write(f"{r['name']},{r['kbits']},{r['vbits']},{r['kmode']},"
                    f"{r['sink']},{r['ppl']:.4f},{r['agree']:.4f},"
                    f"{r['cache_GB_at_32k']:.4f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:      # redraw from the committed findings.json
        make_plots(json.load(open(f"{OUT}/findings.json")))
    else:
        main()
