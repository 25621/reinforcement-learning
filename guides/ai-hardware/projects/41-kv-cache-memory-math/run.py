"""Project 41 -- KV-cache memory math.

One formula, checked three ways:

    bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_value

Section A verifies it against tensors that really exist (Hugging Face's own cache
and this phase's engine) and against the resident memory the OS reports.
Sections B-E apply it to real 2024-2026 architectures downloaded from their
config files -- Llama-2/3, Mistral, Qwen, Gemma-2, Phi-3 and DeepSeek's MLA --
and turn the bytes into the only number a serving engineer cares about: how many
requests fit at once.

Runs in about 10 seconds on 12 CPU threads.
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
import servelib as S  # noqa: E402

OUT = S.outdir(__file__)
H100_GB = 80.0
MODELS = [
    ("Llama-2 7B (MHA)", "NousResearch/Llama-2-7b-hf", 6.74e9),
    ("Llama-3 8B (GQA 8)", "NousResearch/Meta-Llama-3-8B", 8.03e9),
    ("Mistral 7B (GQA 8 + SWA)", "mistralai/Mistral-7B-v0.1", 7.24e9),
    ("Qwen2.5 7B (GQA 4)", "Qwen/Qwen2.5-7B-Instruct", 7.62e9),
    ("Gemma-2 9B (GQA 8)", "unsloth/gemma-2-9b", 9.24e9),
    ("Phi-3 mini (MHA)", "microsoft/Phi-3-mini-4k-instruct", 3.82e9),
    ("DeepSeek-V2-Lite (MLA)", "deepseek-ai/DeepSeek-V2-Lite", 15.7e9),
]
results = {}


def log(*a):
    print(*a, flush=True)


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


# ----------------------------------------------------- A. verify the formula
def section_a():
    log("A. verifying the formula against tensors that exist")
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    name = S.SMALL
    w = S.Weights(name)
    T = 128
    predicted_per_token = 2 * w.n_layer * w.n_kv * w.head_dim * 4     # fp32
    ids = torch.tensor([S.prompt_ids(w.tok, "memory arithmetic. ", T)])

    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    cache = DynamicCache()
    with torch.no_grad():
        hf(ids, past_key_values=cache, use_cache=True)
    seen, measured = set(), 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                measured += t.numel() * t.element_size()
    del hf, cache

    eng = S.Engine(w, num_blocks=16, block_size=16)
    engine_per_token = eng.pool.bytes_per_token()

    a = dict(model=name, tokens=T,
             predicted_bytes_per_token=predicted_per_token,
             hf_cache_bytes=measured,
             hf_bytes_per_token=measured / T,
             engine_bytes_per_token=engine_per_token,
             formula_error_pct=100 * abs(measured / T - predicted_per_token) / predicted_per_token)
    log(f"   formula:  2 * {w.n_layer} layers * {w.n_kv} kv-heads * {w.head_dim} dim "
        f"* 4 B = {predicted_per_token} B/token")
    log(f"   Hugging Face DynamicCache after {T} tokens: {measured} B "
        f"= {measured / T:.0f} B/token")
    log(f"   this phase's paged pool: {engine_per_token:.0f} B/token   "
        f"(error {a['formula_error_pct']:.3f}%)")

    # Asking for memory and *having* it are different events.  A KV pool that
    # has only been asked for costs nothing until something writes to it.
    gb, res = 2.0, {}
    for label, alloc in [("torch.empty", torch.empty), ("torch.zeros", torch.zeros)]:
        base = rss_mb()
        buf = alloc(int(gb * 1e9 / 4), dtype=torch.float32)
        after_alloc = rss_mb() - base
        buf.fill_(1.0)
        after_touch = rss_mb() - base
        del buf
        res[label] = dict(rss_after_alloc_MB=after_alloc, rss_after_touch_MB=after_touch)
        log(f"   {label}({gb:.0f} GB): RSS +{after_alloc:.0f} MB on allocation, "
            f"+{after_touch:.0f} MB once written")
    a["reservation"] = dict(requested_MB=gb * 1e3, **res)
    results["verification"] = a
    return w


# ------------------------------------------------- the model-independent maths
def kv_per_token(cfg, bits=16):
    """Bytes of KV cache one token costs, from a config dict."""
    L = cfg["num_hidden_layers"]
    if "kv_lora_rank" in cfg:            # DeepSeek MLA: one compressed vector
        per_layer = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        kind = "MLA"
    else:
        h = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        per_layer = 2 * cfg["num_key_value_heads"] * h
        kind = ("MHA" if cfg["num_key_value_heads"] == cfg["num_attention_heads"]
                else "MQA" if cfg["num_key_value_heads"] == 1 else "GQA")
    return per_layer * L * bits / 8, kind


def kv_at_context(cfg, ctx, bits=16):
    """Bytes for `ctx` tokens, honouring per-layer sliding windows.

    A "sliding window" layer only keeps the last W tokens, so its cache stops
    growing at W.  Gemma-2 alternates sliding and full layers, so only half of
    its layers keep growing -- the config's `layer_types` list says which.
    """
    per_token, kind = kv_per_token(cfg, bits)
    L = cfg["num_hidden_layers"]
    per_layer = per_token / L
    win = cfg.get("sliding_window")
    types = cfg.get("layer_types") or ["sliding_attention" if win else "full_attention"] * L
    total = 0.0
    for t in types:
        keep = min(ctx, win) if (t == "sliding_attention" and win) else ctx
        total += per_layer * keep
    return total, kind


def section_b():
    log("\nB. the table the guide asks for: Llama-3 8B, context 32768")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3-8B").to_dict()
    bpt, kind = kv_per_token(cfg, 16)
    rows = []
    for batch in (1, 8, 32):
        gb = bpt * 32768 * batch / 1e9
        rows.append(dict(batch=batch, GB=gb, pct_of_H100=100 * gb / H100_GB,
                         fits=gb + 16.06 <= H100_GB))
        log(f"   batch {batch:2d}: {gb:7.2f} GB of KV "
            f"({rows[-1]['pct_of_H100']:5.1f}% of an 80 GB H100) "
            f"{'fits with the weights' if rows[-1]['fits'] else 'DOES NOT FIT'}")
    results["llama3_32k"] = dict(bytes_per_token=bpt, kind=kind, rows=rows,
                                 weights_fp16_GB=16.06)
    log(f"   ({bpt / 1024:.0f} KiB per token; the fp16 weights are 16.06 GB, so a "
        f"single 32k request already costs {bpt * 32768 / 1e9 / 16.06:.2f}x "
        f"the whole model)")
    return cfg


def section_c():
    log("\nC. every architecture, per token of context")
    from transformers import AutoConfig
    rows = []
    for label, repo, params in MODELS:
        cfg = AutoConfig.from_pretrained(repo).to_dict()
        bpt, kind = kv_per_token(cfg, 16)
        win = cfg.get("sliding_window")
        eff32k, _ = kv_at_context(cfg, 32768, 16)
        weights_GB = params * 2 / 1e9
        rows.append(dict(
            label=label, repo=repo, kind=kind, params=params,
            bytes_per_token=bpt, KiB_per_token=bpt / 1024,
            layers=cfg["num_hidden_layers"],
            kv_heads=cfg.get("num_key_value_heads"),
            q_heads=cfg["num_attention_heads"],
            sliding_window=win,
            sliding_layers=sum(1 for t in (cfg.get("layer_types") or [])
                               if t == "sliding_attention"),
            GB_at_32k=bpt * 32768 / 1e9,
            GB_at_32k_effective=eff32k / 1e9,
            weights_GB=weights_GB,
            crossover_tokens=weights_GB * 1e9 / bpt,
            seats_at_32k=int((H100_GB * 1e9 - weights_GB * 1e9) / (bpt * 32768)),
        ))
        r = rows[-1]
        swa = ("" if r["GB_at_32k_effective"] >= r["GB_at_32k"] - 1e-9
               else f" -> {r['GB_at_32k_effective']:5.2f} GB with its sliding window "
                    f"({r['sliding_layers'] or r['layers']}/{r['layers']} layers capped)")
        log(f"   {label:26s} {r['KiB_per_token']:7.1f} KiB/tok  "
            f"{r['GB_at_32k']:6.2f} GB @32k  weights={r['weights_GB']:5.1f} GB  "
            f"KV=weights at {r['crossover_tokens'] / 1000:6.1f}k tokens  "
            f"{r['seats_at_32k']:3d} seats{swa}")
    results["architectures"] = rows
    mha = next(r for r in rows if r["label"].startswith("Llama-2"))
    gqa = next(r for r in rows if r["label"].startswith("Llama-3"))
    mla = next(r for r in rows if "MLA" in r["kind"])
    log(f"   GQA vs MHA at the same 7-8B scale: "
        f"{mha['KiB_per_token'] / gqa['KiB_per_token']:.1f}x less KV")
    log(f"   MLA (DeepSeek) vs MHA per layer-token: "
        f"{mha['bytes_per_token'] / mha['layers']:.0f} B vs "
        f"{mla['bytes_per_token'] / mla['layers']:.0f} B")
    return rows


def section_d(rows):
    log("\nD. what each trick buys on Llama-3 8B (80 GB card, 16.06 GB of weights)")
    base = next(r for r in rows if r["label"].startswith("Llama-3"))
    free = H100_GB * 1e9 - base["weights_GB"] * 1e9
    tricks = []
    for name, bpt in [
        ("fp16 KV (baseline)", base["bytes_per_token"]),
        ("fp8 KV", base["bytes_per_token"] / 2),
        ("int4 KV", base["bytes_per_token"] / 4),
        ("MHA instead of GQA", base["bytes_per_token"] * 4),
        ("MQA (1 kv head)", base["bytes_per_token"] / 8),
        ("sliding window 4096", base["bytes_per_token"] * 4096 / 32768),
    ]:
        seats = free / (bpt * 32768)
        tricks.append(dict(name=name, bytes_per_token=bpt, seats_32k=seats,
                           GB_per_request=bpt * 32768 / 1e9))
        log(f"   {name:22s} {bpt / 1024:7.1f} KiB/tok  "
            f"{tricks[-1]['GB_per_request']:6.2f} GB/request  "
            f"{seats:6.1f} concurrent 32k requests")
    results["tricks"] = tricks


def section_e(rows):
    log("\nE. the crossover: at what context does KV outweigh the model?")
    out = {}
    for r in rows:
        out[r["label"]] = r["crossover_tokens"]
    results["crossover"] = out
    l3 = next(r for r in rows if r["label"].startswith("Llama-3"))
    log(f"   Llama-3 8B: KV overtakes the fp16 weights at "
        f"{l3['crossover_tokens'] / 1000:.1f}k tokens of *total* context -- "
        f"that is {l3['crossover_tokens'] / 8192:.1f} full 8k requests, or one "
        f"128k request")


# -------------------------------------------------------------------- figures
def make_plots(res):
    rows = res["architectures"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    labels = [r["label"].split(" (")[0] for r in rows]
    kinds = [r["kind"] for r in rows]
    cmap = {"MHA": "#d62728", "GQA": "#1f77b4", "MQA": "#ff7f0e", "MLA": "#2ca02c"}
    ax[0].barh(labels, [r["KiB_per_token"] for r in rows],
               color=[cmap[k] for k in kinds])
    for i, r in enumerate(rows):
        ax[0].text(r["KiB_per_token"], i, f" {r['KiB_per_token']:.0f} ({r['kind']})",
                   va="center", fontsize=8)
    ax[0].set_xlabel("KiB of KV cache per token (fp16)")
    ax[0].set_title("The same 7-9B class, 17x apart")
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, max(r["KiB_per_token"] for r in rows) * 1.35)

    l3 = next(r for r in rows if r["label"].startswith("Llama-3"))
    ctx = [2 ** k for k in range(10, 21)]
    ax[1].plot(ctx, [l3["bytes_per_token"] * c / 1e9 for c in ctx], "o-",
               color="#1f77b4", label="KV cache, 1 request")
    ax[1].axhline(l3["weights_GB"], ls="--", color="#d62728",
                  label=f"fp16 weights ({l3['weights_GB']:.1f} GB)")
    ax[1].axhline(H100_GB, ls=":", color="k", label="80 GB card")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("context length (tokens)")
    ax[1].set_ylabel("GB")
    ax[1].set_title("Llama-3 8B: KV overtakes the model")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    tricks = res["tricks"]
    names = [t["name"] for t in tricks]
    seats = [t["seats_32k"] for t in tricks]
    colors = ["#1f77b4" if s >= tricks[0]["seats_32k"] else "#d62728" for s in seats]
    ax[2].barh(names, seats, color=colors)
    for i, s in enumerate(seats):
        ax[2].text(s, i, f" {s:.1f}", va="center", fontsize=8)
    ax[2].set_xlabel("concurrent 32k-token requests on one 80 GB card")
    ax[2].set_title("Same card, same model, six choices")
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, max(seats) * 1.3)
    fig.savefig(f"{OUT}/kv_memory.png", dpi=130)
    log(f"   wrote {OUT}/kv_memory.png")


def main():
    S.setup()
    t0 = time.time()
    section_a()
    section_b()
    rows = section_c()
    section_d(rows)
    section_e(rows)
    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("model,kind,layers,q_heads,kv_heads,KiB_per_token,GB_at_32k,"
                "weights_GB,crossover_tokens,seats_at_32k\n")
        for r in rows:
            f.write(f"\"{r['label']}\",{r['kind']},{r['layers']},{r['q_heads']},"
                    f"{r['kv_heads']},{r['KiB_per_token']:.2f},{r['GB_at_32k']:.3f},"
                    f"{r['weights_GB']:.2f},{r['crossover_tokens']:.0f},"
                    f"{r['seats_at_32k']}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
