"""Project 37 -- Per-channel vs per-tensor quantization.

One scale for the whole matrix, one per output channel, or one per group of 32
weights: the arithmetic is identical, only the number of scales changes. This
project measures what each choice costs in quality, in storage, and in the speed
of the matmul that has to apply those scales.

Runs in about 3 minutes on 12 CPU threads.
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
sys.path.insert(0, os.path.join(HERE, "..", "34-quantize-a-small-llm"))
import quantlib as ql  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

EVAL_SEQ, SEQLEN = 6, 512
results = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------- speed harness
def timeit(fn, seconds=0.4, rounds=5):
    """Best-of-N median timing: N short rounds, report the fastest.

    On a machine shared with other work the *minimum* is the honest estimate --
    a run can only be slowed down by interference, never sped up.
    """
    fn()
    best = float("inf")
    for _ in range(rounds):
        t = time.perf_counter()
        n = 0
        while time.perf_counter() - t < seconds / rounds:
            fn()
            n += 1
        best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3


def dequant_matmul_time(out_f, in_f, group_size, per_tensor):
    """Time 'unpack INT4 -> fp32, then matmul', the shape a weight-only kernel takes.

    A weight-only INT4 kernel does not do integer arithmetic. It reads packed
    4-bit weights, multiplies them by their scale to get floats, and feeds an
    ordinary float matmul. So the cost of a finer granularity is the cost of
    reading and broadcasting more scales -- which is what this measures.
    """
    x = torch.randn(256, in_f)
    q = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8).float()
    if per_tensor:
        scale = torch.randn(1, 1).abs() + 0.1
    elif group_size is None:
        scale = torch.randn(out_f, 1).abs() + 0.1
    else:
        scale = torch.randn(out_f, in_f // group_size, 1).abs() + 0.1

    def once():
        if per_tensor or group_size is None:
            w = q * scale
        else:
            w = (q.reshape(out_f, in_f // group_size, group_size) * scale
                 ).reshape(out_f, in_f)
        return x @ w.T

    return timeit(once)


def make_plots(res):
    """Rebuild the figure from findings.json (so `run.py --plot` is instant)."""
    rows = res["quality"]
    ppl_fp32 = res["fp32_ppl"]
    speed = res["speed"]["rows"]
    fp32_ms = res["speed"]["fp32_ms"]
    out_f, in_f = res["speed"]["shape"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, bits, ylim in [(axes[0], 4, (17, 60)), (axes[1], 3, (50, 4000))]:
        for sym, style, color, label in [(True, "-o", "#1f77b4", "symmetric"),
                                         (False, "--s", "#d62728", "asymmetric")]:
            sub = [r for r in rows if r["bits"] == bits and r["sym"] == sym
                   and not r["per_tensor"]]
            sub.sort(key=lambda r: r["bits_per_weight"])
            ax.plot([r["bits_per_weight"] for r in sub],
                    [min(r["ppl"], ylim[1] * 0.98) for r in sub],
                    style, color=color, markersize=5, label=label)
            for r in sub:
                if r["ppl"] <= ylim[1]:
                    ax.annotate(r["label"].split(" ", 1)[1].replace(" asym", ""),
                                (r["bits_per_weight"], r["ppl"]), fontsize=7,
                                xytext=(4, 4), textcoords="offset points")
        ax.axhline(ppl_fp32, ls=":", color="k", label="fp32")
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlabel("effective bits per weight (scales included)")
        ax.set_ylabel("WikiText-2 perplexity")
        ax.set_title(f"INT{bits}: granularity buys quality, costs storage")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        pt = [r for r in rows if r["bits"] == bits and r["per_tensor"]][0]
        ax.text(0.03, 0.94, f"per-tensor is off the chart at {pt['ppl']:.3g}",
                transform=ax.transAxes, fontsize=8, color="#d62728")

    axes[2].bar([r["label"] for r in speed], [r["ms"] for r in speed],
                color="#1f77b4")
    axes[2].axhline(fp32_ms, ls="--", color="k", label="plain fp32 matmul")
    axes[2].set_ylabel("ms per dequantize+matmul")
    axes[2].set_title(f"Applying the scales: {out_f}x{in_f} weight, 256 tokens")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle("Quantization granularity for " + ql.MODEL)
    fig.savefig(f"{OUT}/granularity.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/granularity.png")


def main():
    ql.setup()
    t_start = time.time()
    tok, model = ql.load(ql.MODEL)
    linears = ql.quantizable_linears(model)
    text = ql.wikitext_text()
    ev = ql.token_batches(tok, text, EVAL_SEQ, SEQLEN)

    # -------------------------------------------- A. why granularity exists
    log("=== A. The dynamic range inside one weight matrix ===")
    name = "layers.12.mlp.down_proj"
    W = linears[name].weight.data
    row_amax = W.abs().amax(1)
    log(f"  {name}  shape {tuple(W.shape)}")
    log(f"  |w| max over the whole tensor : {float(W.abs().max()):.4f}")
    log(f"  biggest output channel        : {float(row_amax.max()):.4f}")
    log(f"  median  output channel        : {float(row_amax.median()):.4f}")
    log(f"  smallest output channel       : {float(row_amax.min()):.4f}")
    spread = float(row_amax.max() / row_amax.min())
    log(f"  -> the loudest channel is {spread:.1f}x the quietest one.")
    log(f"     One scale for the whole tensor is set by the loudest channel, so "
        f"the quietest\n     channel only gets "
        f"{7 / spread:.2f} of INT4's 7 positive levels to work with.")
    results["range"] = {"layer": name, "shape": list(W.shape),
                        "row_amax_max": float(row_amax.max()),
                        "row_amax_median": float(row_amax.median()),
                        "row_amax_min": float(row_amax.min()),
                        "spread": spread}

    groups = W.reshape(W.shape[0], -1, 128).abs().amax(-1)
    log(f"  Within a single output channel, the loudest group of 128 is on "
        f"average\n     "
        f"{float((groups.amax(1) / groups.amin(1)).mean()):.1f}x the quietest -- "
        f"which is what per-group scales pick up.")
    results["range"]["within_channel_spread"] = float(
        (groups.amax(1) / groups.amin(1)).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), constrained_layout=True)
    axes[0].hist(row_amax.numpy(), bins=80, color="#1f77b4")
    axes[0].axvline(float(W.abs().max()), color="r", ls="--",
                    label="the single per-tensor scale")
    axes[0].set_xlabel("max |weight| in an output channel")
    axes[0].set_ylabel("channels")
    axes[0].set_title(f"{name}: channels are not the same size")
    axes[0].legend(fontsize=8)
    axes[1].plot(groups[0].numpy(), lw=0.9)
    axes[1].set_xlabel("group of 128 input weights, within output channel 0")
    axes[1].set_ylabel("max |weight| in the group")
    axes[1].set_title("...and neither are groups inside one channel")
    axes[1].grid(alpha=0.3)
    fig.savefig(f"{OUT}/weight_ranges.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------ B. quality vs granularity
    log("\n=== B. Quality at every granularity (round-to-nearest INT4) ===")
    ppl_fp32 = ql.perplexity(model, ev)
    log(f"  fp32 baseline: ppl {ppl_fp32:.3f}")

    specs = []
    for bits in [4, 3]:
        specs.append((f"INT{bits} per-tensor", bits, None, True, True))
        specs.append((f"INT{bits} per-channel", bits, None, False, True))
        for g in [256, 128, 64, 32]:
            specs.append((f"INT{bits} group {g}", bits, g, False, True))
        specs.append((f"INT{bits} group 128 asym", bits, 128, False, False))
        specs.append((f"INT{bits} per-channel asym", bits, None, False, False))

    rows = []
    for label, bits, group, per_tensor, sym in specs:
        with ql.QuantizedWeights(
                model, lambda n, w: ql.fake_quant(w, bits, group, sym, per_tensor)):
            ppl = ql.perplexity(model, ev)
        bpw = sum(
            ql.bits_per_weight(m.weight.shape[1], bits, group, per_tensor, sym=sym)
            * m.weight.numel() for m in linears.values())
        bpw /= sum(m.weight.numel() for m in linears.values())
        rows.append({"label": label, "bits": bits, "group": group,
                     "per_tensor": per_tensor, "sym": sym, "ppl": ppl,
                     "bits_per_weight": bpw})
        log(f"  {label:24s} ppl {ppl:9.3f}   {bpw:5.2f} bits/weight")
    results["quality"] = rows
    results["fp32_ppl"] = ppl_fp32

    # ------------------------------------------------------- C. speed cost
    log("\n=== C. What the extra scales cost at inference time ===")
    out_f, in_f = 4864, 896
    speed = []
    for label, group, per_tensor in [("per-tensor", None, True),
                                     ("per-channel", None, False),
                                     ("group 128", 128, False),
                                     ("group 32", 32, False)]:
        ms = dequant_matmul_time(out_f, in_f, group, per_tensor)
        speed.append({"label": label, "ms": ms})
        log(f"  {label:14s} dequantize+matmul {ms:7.2f} ms")
    x = torch.randn(256, in_f)
    w = torch.randn(out_f, in_f)
    fp32_ms = timeit(lambda: x @ w.T)
    log(f"  {'plain fp32':14s} matmul            {fp32_ms:7.2f} ms")
    slowest = max(speed, key=lambda r: r["ms"])
    log(f"  finest granularity costs "
        f"{slowest['ms'] / speed[0]['ms']:.2f}x the coarsest, and "
        f"{slowest['ms'] / fp32_ms:.2f}x a plain fp32 matmul")
    results["speed"] = {"rows": speed, "fp32_ms": fp32_ms,
                        "shape": [out_f, in_f]}

    # ------------------------------------------- D. activations are different
    log("\n=== D. The same question for activations ===")
    grabbed = {}

    def hook(mod, args):
        if "x" not in grabbed:
            grabbed["x"] = args[0].detach()[0].float()

    h = linears["layers.12.mlp.down_proj"].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(ev[:1, :512])
    h.remove()
    act = grabbed["x"]
    ch_amax = act.abs().amax(0)
    tok_amax = act.abs().amax(1)
    log(f"  activation tensor {tuple(act.shape)} (tokens x channels)")
    log(f"  channel spread (max/median): "
        f"{float(ch_amax.max() / ch_amax.median()):8.1f}x")
    log(f"  token   spread (max/median): "
        f"{float(tok_amax.max() / tok_amax.median()):8.1f}x")
    a_rows = []
    for label, dim in [("per-tensor", None), ("per-token", 1), ("per-channel", 0)]:
        if dim is None:
            aq = ql.fake_quant(act.reshape(1, -1), 8, sym=False).reshape(act.shape)
        elif dim == 1:
            aq = ql.fake_quant(act, 8, sym=False)
        else:
            aq = ql.fake_quant(act.T, 8, sym=False).T
        err = float((act - aq).norm() / act.norm())
        a_rows.append({"label": label, "rel_error": err})
        log(f"  INT8 activations, {label:12s} relative error {err:.5f}")
    log("  Activations are quantized per-token in practice: a per-channel scale "
        "would\n  have to be known before the token arrives, and it is not.")
    results["activations"] = {
        "channel_spread": float(ch_amax.max() / ch_amax.median()),
        "token_spread": float(tok_amax.max() / tok_amax.median()),
        "rows": a_rows,
        "channel_amax": ch_amax.tolist()}

    make_plots(results)
    results["total_seconds"] = time.time() - t_start
    ql.save_json(f"{OUT}/findings.json", results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("label,bits,group,per_tensor,sym,ppl,bits_per_weight\n")
        for r in rows:
            f.write(f"{r['label']},{r['bits']},{r['group']},{r['per_tensor']},"
                    f"{r['sym']},{r['ppl']:.4f},{r['bits_per_weight']:.4f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:      # redraw from the committed findings.json
        make_plots(json.load(open(f"{OUT}/findings.json")))
    else:
        main()
