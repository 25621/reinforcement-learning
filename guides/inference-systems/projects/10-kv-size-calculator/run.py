"""Project 10 -- KV size calculator.

Turns the KV-cache size formula into code, checks it against a cache we
actually measured in project 09, and then sweeps the three knobs that matter
(kv-heads, sequence length, dtype) across real published model shapes.

    python3 run.py           # ~5 seconds, no model download
    python3 run.py --plot    # redraw the figure from outputs/findings.json

Everything here is arithmetic, so it runs instantly -- which is the point.
Capacity planning should cost you a coffee's worth of thought, not a cluster
reservation.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
NINE = os.path.join(HERE, "..", "09-kv-cache-from-scratch")

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp8": 1, "int4": 0.5}

# Published shapes. `attn` is the flavour; `params` is in billions.
# MLA (DeepSeek) caches one shared low-rank latent per token instead of
# per-head K and V, so it gets its own bytes-per-token rule below.
MODELS = [
    dict(name="Qwen2.5-0.5B",   attn="GQA", layers=24, heads=14, kv_heads=2,  d_head=64,  params=0.49),
    dict(name="Llama-2-13B",    attn="MHA", layers=40, heads=40, kv_heads=40, d_head=128, params=13.0),
    dict(name="Falcon-7B",      attn="MQA", layers=32, heads=71, kv_heads=1,  d_head=64,  params=7.2),
    dict(name="Llama-3.1-8B",   attn="GQA", layers=32, heads=32, kv_heads=8,  d_head=128, params=8.03),
    dict(name="Llama-3.1-70B",  attn="GQA", layers=80, heads=64, kv_heads=8,  d_head=128, params=70.6),
    dict(name="DeepSeek-V2-Lite", attn="MLA", layers=27, heads=16, kv_heads=16, d_head=128,
         params=15.7, mla_rank=512, mla_rope=64),
]

GPU_HBM_GB = {"A100-40GB": 40, "A100-80GB": 80, "H100-80GB": 80, "H200-141GB": 141}


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def kv_bytes_per_token(layers, kv_heads, d_head, dtype_bytes=2, mla=None):
    """Bytes of KV cache one token costs, for the whole model.

        2 (K and V) x layers x kv_heads x d_head x bytes_per_number

    The 2 is "one key vector and one value vector". kv_heads is the number of
    key/value heads *after* GQA sharing, not the number of query heads -- the
    single most common way to get this formula wrong by 4-8x.

    MLA (multi-head latent attention) breaks the shape: it stores one
    compressed latent of width `mla_rank` plus a small rotary part of width
    `mla_rope`, once per layer, shared by every head. No factor of 2, no
    factor of kv_heads.
    """
    if mla is not None:
        return layers * (mla["rank"] + mla["rope"]) * dtype_bytes
    return 2 * layers * kv_heads * d_head * dtype_bytes


def model_kv_per_token(m, dtype="bf16"):
    b = DTYPE_BYTES[dtype]
    if m["attn"] == "MLA":
        return kv_bytes_per_token(m["layers"], 0, 0, b,
                                  mla={"rank": m["mla_rank"], "rope": m["mla_rope"]})
    return kv_bytes_per_token(m["layers"], m["kv_heads"], m["d_head"], b)


def weight_bytes(m, dtype="bf16"):
    return m["params"] * 1e9 * DTYPE_BYTES[dtype]


def main():
    f = {"dtype_bytes": DTYPE_BYTES}

    # ------------------------------------------------------------------ A
    # Check the formula against a cache project 09 actually allocated.
    print("A. validating the formula against project 09's measured cache")
    p = os.path.join(NINE, "outputs", "findings.json")
    if os.path.exists(p):
        n9 = json.load(open(p))
        mm = n9["model"]
        pred = kv_bytes_per_token(mm["n_layers"], mm["n_kv_heads"], mm["d_head"], 4)
        meas = n9["C_size"]["measured_bytes_per_token_fp32"]
        f["A_validation"] = {"predicted_bytes_per_token": pred,
                             "measured_bytes_per_token": meas,
                             "error_pct": 100 * (meas - pred) / pred,
                             "source": "09-kv-cache-from-scratch/outputs/findings.json"}
        print(f"   predicted {pred} B/token, measured {meas:.0f} "
              f"({f['A_validation']['error_pct']:+.3f}%)")
    else:
        print("   (run project 09 first to fill this in)")

    # ------------------------------------------------------------------ B
    # Per-token cost of every model shape, and the GQA/MQA/MLA ratios.
    print("B. bytes per token, by attention flavour")
    rows = []
    mha_ref = None
    for m in MODELS:
        per_tok = model_kv_per_token(m, "bf16")
        # What the same model would cost with no KV sharing at all (MHA).
        mha = kv_bytes_per_token(m["layers"], m["heads"], m["d_head"], 2)
        rows.append({
            "name": m["name"], "attn": m["attn"], "layers": m["layers"],
            "heads": m["heads"], "kv_heads": m["kv_heads"], "d_head": m["d_head"],
            "params_b": m["params"],
            "kv_bytes_per_token_bf16": per_tok,
            "mha_equivalent_bytes_per_token": mha,
            "sharing_factor": mha / per_tok,
            "weight_gb_bf16": weight_bytes(m, "bf16") / 1e9,
            "kv_gb_at_8k_bf16": per_tok * 8192 / 1e9,
        })
        print(f"   {m['name']:>18} {m['attn']:>3}  {per_tok/1024:7.1f} KB/token  "
              f"sharing {mha/per_tok:5.1f}x")
    f["B_per_model"] = rows

    # ------------------------------------------------------------------ C
    # Concurrency: how many 8k-context users fit on one GPU after weights.
    print("C. concurrency on one GPU (8k context, bf16 weights)")
    conc = []
    for m in MODELS:
        w = weight_bytes(m, "bf16")
        for gpu, gb in GPU_HBM_GB.items():
            hbm = gb * 1e9 * 0.90         # 90%: framework + activations overhead
            free = hbm - w
            row = {"model": m["name"], "gpu": gpu, "free_gb": free / 1e9}
            for dt in ("bf16", "fp8", "int4"):
                per_req = model_kv_per_token(m, dt) * 8192
                row[dt] = max(0, int(free // per_req)) if free > 0 else 0
            conc.append(row)
    f["C_concurrency_8k"] = conc
    for r in conc:
        if r["gpu"] == "H100-80GB":
            print(f"   {r['model']:>18} on H100-80GB: bf16 {r['bf16']:>5} users, "
                  f"fp8 {r['fp8']:>5}, int4 {r['int4']:>5}")

    # C2. Some models do not fit on one GPU at all. Work out the smallest
    # number of H100s that holds the weights, then re-run the same sum.
    print("C2. same question on the smallest H100 box that holds the weights")
    conc_n = []
    for m in MODELS:
        w = weight_bytes(m, "bf16")
        n_gpu = 1
        while w > 0.90 * n_gpu * 80e9 * 0.80:      # leave 20% of HBM for cache
            n_gpu *= 2
        free = 0.90 * n_gpu * 80e9 - w
        row = {"model": m["name"], "n_h100": n_gpu, "free_gb": free / 1e9}
        for dt in ("bf16", "fp8", "int4"):
            row[dt] = int(free // (model_kv_per_token(m, dt) * 8192))
        conc_n.append(row)
        print(f"   {m['name']:>18}: {n_gpu} x H100  bf16 {row['bf16']:>5} users, "
              f"fp8 {row['fp8']:>5}, int4 {row['int4']:>5}")
    f["C2_concurrency_multigpu"] = conc_n

    # ------------------------------------------------------------------ D
    # The crossover: at what context does the cache outweigh the weights?
    print("D. context length where cache == weights (batch 1 and batch 32)")
    cross = []
    for m in MODELS:
        w = weight_bytes(m, "bf16")
        per_tok = model_kv_per_token(m, "bf16")
        cross.append({
            "model": m["name"],
            "weight_gb": w / 1e9,
            "ctx_batch1": w / per_tok,
            "ctx_batch32": w / (per_tok * 32),
        })
        print(f"   {m['name']:>18}: {w/per_tok:>12,.0f} tokens at batch 1, "
              f"{w/(per_tok*32):>10,.0f} at batch 32")
    f["D_crossover"] = cross

    # ------------------------------------------------------------------ E
    # The sweep the plot draws: memory vs concurrency vs context.
    seqs = [1024, 2048, 4096, 8192, 16384, 32768, 131072]
    batches = [1, 4, 8, 16, 32, 64, 128, 256]
    f["E_sweep"] = {
        "seq_lens": seqs, "batches": batches,
        "llama70b_gb": {dt: [[model_kv_per_token(MODELS[4], dt) * s * b / 1e9
                              for b in batches] for s in seqs]
                        for dt in ("bf16", "fp8", "int4")},
    }

    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "per_model.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["B_per_model"][0].keys()))
        w.writeheader()
        for r in f["B_per_model"]:
            w.writerow(r)
    with open(os.path.join(OUT, "concurrency.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["C_concurrency_8k"][0].keys()))
        w.writeheader()
        for r in f["C_concurrency_8k"]:
            w.writerow(r)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # 1: KB per token by model, log scale, coloured by attention flavour.
    rows = sorted(f["B_per_model"], key=lambda r: r["kv_bytes_per_token_bf16"])
    colors = {"MHA": "tab:red", "GQA": "tab:blue", "MQA": "tab:green",
              "MLA": "tab:purple"}
    ax[0].barh([r["name"] for r in rows],
               [r["kv_bytes_per_token_bf16"] / 1024 for r in rows],
               color=[colors[r["attn"]] for r in rows])
    for i, r in enumerate(rows):
        ax[0].text(r["kv_bytes_per_token_bf16"] / 1024 * 1.05, i,
                   f"{r['attn']}", va="center", fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("KB of KV cache per token (bf16, log)")
    ax[0].set_title("A. attention flavour sets the bill")
    ax[0].grid(alpha=.3, axis="x")

    # 2: Llama-70B cache GB vs batch, one line per context length.
    sw = f["E_sweep"]
    for si, s in enumerate(sw["seq_lens"]):
        ax[1].plot(sw["batches"], sw["llama70b_gb"]["bf16"][si], "o-",
                   label=f"{s // 1024}k ctx")
    ax[1].axhline(80, color="k", ls="--", lw=1)
    ax[1].text(1.2, 84, "H100 80 GB", fontsize=8)
    ax[1].axhline(141 - 141, color="none")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("concurrent requests")
    ax[1].set_ylabel("KV cache (GB, bf16)")
    ax[1].set_title("B. Llama-3.1-70B: the cache runs out first")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=.3)

    # 3: concurrency on an H100 by dtype.
    h = [r for r in f["C_concurrency_8k"] if r["gpu"] == "H100-80GB"]
    names = [r["model"] for r in h]
    x = range(len(names))
    for i, dt in enumerate(("bf16", "fp8", "int4")):
        ax[2].bar([xx + i * 0.27 for xx in x], [r[dt] for r in h], width=0.27,
                  label=dt)
    ax[2].set_xticks([xx + 0.27 for xx in x])
    ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[2].set_yscale("symlog")
    ax[2].set_ylabel("concurrent 8k-context requests")
    ax[2].set_title("C. one H100-80GB, after weights")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv_size.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
