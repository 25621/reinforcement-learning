"""Project 48 - deploying to the edge, without an edge board.

There is no Jetson on this machine. What there is: a memory system, a small
real LLM, and the arithmetic that decides an edge deployment. So the project
builds a predictor for decode speed on *any* device from one number - its
memory bandwidth - validates the predictor on the two devices we do have, and
only then applies it to Jetson modules.

Sections
  A. the devices           - bandwidth / TOPS / watts (datasheet, arithmetic)
  B. the local ground truth- measured CPU memory bandwidth, and measured
                             decode speed of Qwen2.5-0.5B at fp32 and int8
  C. does the predictor work? - measured vs predicted on this CPU
  D. the Jetson prediction - the same predictor on edge modules
  E. why TOPS is the wrong number - the arithmetic intensity a device needs
                             before its TOPS rating means anything
  F. what actually helps   - batching, measured

Labels in findings.json: `measured` on this machine, `arithmetic` for anything
about a device we do not own.
"""

import json
import os
import sys
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain what a Jetson module is, in two sentences."
GEN_TOKENS = 24

# ---------------------------------------------------------------- A
# Datasheet numbers. TOPS are int8 dense (NVIDIA quotes sparse; halved here).
# "shared" bandwidth means CPU and GPU draw from the same pool.
DEVICES = [
    # name,                    GB/s, int8 TOPS, watts, memory, price
    ("Jetson Orin Nano 8GB",     68,   20,  (7, 15),  "shared LPDDR5",  249),
    ("Jetson Orin NX 16GB",     102,   50,  (10, 25), "shared LPDDR5",  699),
    ("Jetson AGX Orin 64GB",    204,  137,  (15, 60), "shared LPDDR5", 1999),
    ("Raspberry Pi 5 (CPU)",     17,  0.1,  (5, 12),  "shared LPDDR4X",   80),
    # 30.5 TOPS int8 is measured on this card in project 49 (dp4a), not a spec
    ("GTX 1070 Ti (this box)",  197,   30.5, (30, 180), "discrete GDDR5", 150),
    ("RTX 5090",               1792,  838,  (60, 575), "discrete GDDR7", 2200),
]
DEV_KEYS = ["name", "bw_gbs", "tops_int8", "watts", "memory", "price_usd"]


def devices():
    return [dict(zip(DEV_KEYS, d)) for d in DEVICES]


# ---------------------------------------------------------------- B
def cpu_bandwidth(mb=256, reps=6):
    """A STREAM-style copy: read one array, write another. The number that
    matters for decoding is *read* bandwidth of a big, cold array."""
    import torch
    n = mb * 1024 * 1024 // 4
    a = torch.randn(n, dtype=torch.float32)
    b = torch.empty_like(a)
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        b.copy_(a)
        dt = time.perf_counter() - t0
        best = max(best, 2 * n * 4 / dt / 1e9)     # read + write
    # a pure read (sum) is closer to what weight loading does
    best_read = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        a.sum()
        dt = time.perf_counter() - t0
        best_read = max(best_read, n * 4 / dt / 1e9)
    return dict(copy_gbs=best, read_gbs=best_read, mb=mb)


def decode_bench(quantized=False, batch=1):
    """Measured tokens/second for greedy decoding, plus the bytes of weights
    that have to be read to produce one token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    # What matters for decode speed is the weight bytes *traversed per token*,
    # not the checkpoint size. Every Linear is read in full for every token;
    # the embedding table is not - a lookup touches one row (896 floats here),
    # which is nothing. In this model the embedding is tied to the output
    # Linear, so counting Linear weights already covers it exactly once.
    n_params = sum(p.numel() for p in model.parameters())
    linear_params = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m, torch.nn.Linear))
    label = "fp32"
    weight_bytes = n_params * 4
    if quantized:
        # Dynamic int8: Linear weights are stored as int8, activations are
        # quantized on the fly. This is what an edge runtime applies when you
        # have no calibration data (compare project 36). Everything that is not
        # a Linear - embeddings, norms - stays fp32, so the bytes read per token
        # are counted accordingly rather than assumed to be 1 per parameter.
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        label = "int8-dynamic"
        weight_bytes = linear_params * 1 + (n_params - linear_params) * 4

    ids = tok([PROMPT] * batch, return_tensors="pt")

    with torch.no_grad():
        model.generate(**ids, max_new_tokens=2, do_sample=False)   # warm up
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False)
        dt = time.perf_counter() - t0
    new_tokens = (out.shape[1] - ids["input_ids"].shape[1]) * batch
    return dict(label=label, batch=batch, seconds=dt, tokens=new_tokens,
                tok_per_s=new_tokens / dt, params=n_params,
                linear_params=linear_params, weight_bytes=weight_bytes,
                bytes_per_token=weight_bytes,
                text=tok.decode(out[0], skip_special_tokens=True)[:200])


# ---------------------------------------------------------------- C/D
def predict(bw_gbs, weight_bytes, efficiency=1.0):
    """Decode is memory-bound: one token means reading every weight once.
        tokens/s = bandwidth / bytes_per_token
    `efficiency` is the fraction of peak bandwidth a real runtime achieves."""
    return bw_gbs * 1e9 * efficiency / weight_bytes


MODEL_SIZES = [
    ("Qwen2.5-0.5B", 0.5e9), ("Llama-3.2-3B", 3.2e9), ("Llama-3-8B", 8.0e9),
]
QUANTS = [("fp16", 2), ("int8", 1), ("int4", 0.5)]


def jetson_table(efficiency):
    rows = []
    for d in devices():
        for mname, params in MODEL_SIZES:
            for qname, bpw in QUANTS:
                wb = params * bpw
                fits = wb / 2**30 < 6.5 if "Orin Nano" in d["name"] else True
                rows.append(dict(device=d["name"], model=mname, quant=qname,
                                 weight_gib=wb / 2**30,
                                 tok_per_s=predict(d["bw_gbs"], wb, efficiency),
                                 fits_8gb=fits))
    return rows


# ---------------------------------------------------------------- E
def tops_reality():
    """How much arithmetic per byte a device needs before its TOPS rating is
    the binding limit - and how much decoding actually has."""
    rows = []
    for d in devices():
        need = d["tops_int8"] * 1e12 / (d["bw_gbs"] * 1e9)
        # int8 decode at batch 1: each weight byte is used for one multiply
        # and one add, so 2 operations per byte and no more.
        have = 2.0
        rows.append(dict(device=d["name"], tops=d["tops_int8"],
                         bw_gbs=d["bw_gbs"], ridge_ops_per_byte=need,
                         decode_ops_per_byte=have,
                         tops_used_pct=100 * have / need,
                         batch_needed_for_ridge=need / have))
    return rows


def main():
    findings = {"model": MODEL_ID, "gen_tokens": GEN_TOKENS}

    print("== B. this machine's memory system ==")
    bw = cpu_bandwidth()
    findings["cpu_bandwidth"] = bw
    print(f"   copy {bw['copy_gbs']:.1f} GB/s, pure read {bw['read_gbs']:.1f} GB/s")

    print("== B. decoding a real 0.5B model on the CPU ==")
    runs = []
    for q in (False, True):
        r = decode_bench(quantized=q)
        runs.append(r)
        print(f"   {r['label']:>13}: {r['tok_per_s']:6.2f} tok/s "
              f"({r['weight_bytes']/2**20:.0f} MiB of weights per token)")
    findings["decode"] = runs

    print("== C. does the memory-bound predictor work here? ==")
    checks = []
    for r in runs:
        pred = predict(bw["read_gbs"], r["weight_bytes"])
        checks.append(dict(label=r["label"], measured=r["tok_per_s"],
                           predicted_at_peak=pred,
                           efficiency=r["tok_per_s"] / pred))
        print(f"   {r['label']:>13}: measured {r['tok_per_s']:6.2f} tok/s, "
              f"ceiling {pred:6.2f} tok/s -> runtime reaches "
              f"{100*r['tok_per_s']/pred:.0f}% of memory bandwidth")
    findings["predictor_check"] = checks
    eff = min(c["efficiency"] for c in checks)
    findings["efficiency_used"] = eff

    print(f"== D. the same predictor on edge modules (at {100*eff:.0f}% "
          f"of peak bandwidth) ==")
    findings["jetson"] = jetson_table(eff)
    for r in findings["jetson"]:
        if r["device"].startswith("Jetson Orin Nano") and r["quant"] != "fp16":
            print(f"   {r['device']} {r['model']:>14} {r['quant']:>5}: "
                  f"{r['tok_per_s']:6.1f} tok/s "
                  f"({r['weight_gib']:.1f} GiB, fits 8 GB: {r['fits_8gb']})")

    print("== E. why the TOPS number on the box is not your number ==")
    findings["tops"] = tops_reality()
    for r in findings["tops"]:
        print(f"   {r['device']:>26}: needs {r['ridge_ops_per_byte']:7.0f} "
              f"ops/byte to use its {r['tops']:g} TOPS; batch-1 decode has 2 "
              f"-> {r['tops_used_pct']:.2f}% of the rating "
              f"(batch {r['batch_needed_for_ridge']:.0f} would be needed)")

    print("== F. what actually helps: batching ==")
    batched = [decode_bench(quantized=False, batch=b) for b in (1, 4, 8)]
    findings["batching"] = batched
    base = batched[0]["tok_per_s"]
    for r in batched:
        r["speedup"] = r["tok_per_s"] / base
        print(f"   batch {r['batch']}: {r['tok_per_s']:6.2f} tok/s total "
              f"({r['speedup']:.2f}x), {r['seconds']/GEN_TOKENS*1000:6.1f} ms "
              f"per step")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, edge.png")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        w.writerow(["B", "cpu read bandwidth", round(f["cpu_bandwidth"]["read_gbs"], 2),
                    "GB/s", "measured"])
        for r in f["decode"]:
            w.writerow(["B", f"decode {r['label']}", round(r["tok_per_s"], 3),
                        "tok/s", "measured"])
        for r in f["predictor_check"]:
            w.writerow(["C", f"bandwidth efficiency {r['label']}",
                        round(100 * r["efficiency"], 1), "%", "measured"])
        for r in f["jetson"]:
            w.writerow(["D", f"{r['device']} | {r['model']} {r['quant']}",
                        round(r["tok_per_s"], 2), "tok/s", "arithmetic"])
        for r in f["tops"]:
            w.writerow(["E", f"{r['device']} ridge", round(r["ridge_ops_per_byte"], 1),
                        "ops/byte", "arithmetic"])
        for r in f["batching"]:
            w.writerow(["F", f"batch {r['batch']}", round(r["tok_per_s"], 3),
                        "tok/s", "measured"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1. measured vs predicted on this CPU
    a = ax[0]
    labels = [c["label"] for c in f["predictor_check"]]
    x = range(len(labels))
    a.bar([i - 0.2 for i in x], [c["predicted_at_peak"] for c in f["predictor_check"]],
          width=0.4, label="ceiling = bandwidth / bytes-per-token", color="#95a5a6")
    a.bar([i + 0.2 for i in x], [c["measured"] for c in f["predictor_check"]],
          width=0.4, label="measured", color="#27ae60")
    for i, c in enumerate(f["predictor_check"]):
        a.text(i + 0.2, c["measured"], f"{100*c['efficiency']:.0f}% of ceiling",
               ha="center", va="bottom", fontsize=7)
    a.set_xticks(list(x)); a.set_xticklabels(labels)
    a.set_ylabel("tokens/s"); a.set_title("C. the predictor, checked locally",
                                          fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3, axis="y")

    # 2. predicted tok/s per device for a 3B int4
    a = ax[1]
    rows = [r for r in f["jetson"] if r["model"] == "Llama-3.2-3B"
            and r["quant"] == "int4"]
    a.barh([r["device"] for r in rows], [r["tok_per_s"] for r in rows],
           color="#2980b9")
    for i, r in enumerate(rows):
        a.text(r["tok_per_s"], i, f" {r['tok_per_s']:.0f}", va="center", fontsize=8)
    a.set_xscale("log")
    a.set_xlabel("predicted tokens/s (log)")
    a.set_title("D. Llama-3.2-3B int4, decode ceiling", fontsize=10)
    a.grid(alpha=.3, axis="x")

    # 3. the TOPS gap
    a = ax[2]
    rows = f["tops"]
    a.barh([r["device"] for r in rows], [r["ridge_ops_per_byte"] for r in rows],
           color="#8e44ad", label="ops/byte needed to use the TOPS rating")
    a.axvline(2, color="#c0392b", lw=2, label="ops/byte a batch-1 decode has")
    a.set_xscale("log"); a.set_xlabel("arithmetic intensity (ops/byte, log)")
    a.set_title("E. the TOPS on the box needs 100x more work per byte",
                fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3, axis="x")

    fig.suptitle("Project 48 - edge deployment, predicted from bandwidth",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "edge.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
