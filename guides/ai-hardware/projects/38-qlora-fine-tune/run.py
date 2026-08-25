"""Project 38 -- QLoRA fine-tune.

Fine-tunes SmolLM2-135M on Python source code three ways -- full fine-tune, LoRA
on an FP32 base, and QLoRA on an NF4 base -- and accounts for every byte of
memory each one needs.

Runs in about 6 minutes on an idle 12-thread CPU.
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
import nf4 as N        # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

MODEL = ql.TINY
STEPS, BS, SEQ, RANK = 50, 2, 256, 16
results = {}


def log(*a):
    print(*a, flush=True)


# =========================================================== A. the NF4 grid
def section_a():
    log("=== A. Deriving the NF4 grid ===")
    levels = N.NF4
    log("  16 levels: " + ", ".join(f"{v:+.4f}" for v in levels.tolist()))
    published = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0])
    diff = float((levels - published).abs().max())
    log(f"  max difference from the published QLoRA table: {diff:.2e}")

    torch.manual_seed(0)
    gauss = torch.randn(1 << 20)
    int4 = ql.fake_quant(gauss.reshape(-1, 64), 4, sym=True).reshape(-1)
    nf4v = N.nf4_fake_quant(gauss, block=64)
    e_int = float((gauss - int4).pow(2).mean() / gauss.pow(2).mean())
    e_nf4 = float((gauss - nf4v).pow(2).mean() / gauss.pow(2).mean())
    log(f"  on Gaussian data, block 64: INT4 rel MSE {e_int:.5f}, "
        f"NF4 rel MSE {e_nf4:.5f}  ({e_int / e_nf4:.2f}x better)")
    results["nf4_grid"] = {"levels": levels.tolist(), "max_diff": diff,
                           "int4_relmse": e_int, "nf4_relmse": e_nf4}
    return levels


# ============================================== B. what NF4 costs to store
def section_b(model):
    log("\n=== B. Bits per weight, once the scales are counted ===")
    linears = ql.quantizable_linears(model)
    n = sum(m.weight.numel() for m in linears.values())
    block, block2 = 64, 256
    plain = 4 + 32 / block
    doubled = 4 + 8 / block + 32 / (block * block2)
    log(f"  NF4, one FP32 scale per {block} weights : {plain:.3f} bits/weight")
    log(f"  + double quantization (INT8 scales)     : {doubled:.3f} bits/weight")
    log(f"  saving on {n / 1e6:.0f} M quantized weights: "
        f"{(plain - doubled) * n / 8 / 1e6:.1f} MB")

    W = linears["layers.10.mlp.down_proj"].weight.data
    codes, scale, meta = N.nf4_quantize(W, block)
    err_single = float((W - N.nf4_dequantize(codes, scale, meta)).pow(2).mean())
    scale2 = N.double_quantize_scales(scale, block2)
    err_double = float((W - N.nf4_dequantize(codes, scale2, meta)).pow(2).mean())
    log(f"  measured error on one real weight matrix: "
        f"{err_single:.3e} -> {err_double:.3e} "
        f"({100 * (err_double / err_single - 1):+.2f}% from double quantization)")
    results["bits"] = {"plain": plain, "doubled": doubled,
                       "mse_single": err_single, "mse_double": err_double,
                       "quantized_weights": n}


# ============================================ C/D. memory model + training
def build(mode, tok):
    _, model = ql.load(MODEL)
    n_lora = 0
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        if mode == "qlora":
            # The base is frozen, so its NF4 values never change: quantize once
            # up front. A real QLoRA keeps the 4-bit codes in memory and expands
            # them inside the kernel; the *numbers* the forward pass sees are the
            # same ones, which is what quality depends on.
            for name, mod in ql.quantizable_linears(model).items():
                mod.weight.data = N.nf4_fake_quant(
                    mod.weight.data, block=64).contiguous()
        for p in model.parameters():
            p.requires_grad_(False)
        n_lora = N.inject_lora(model, r=RANK)
    return model, n_lora


def memory_model(model, mode, act_bytes):
    """Every byte a training step needs, by category."""
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # A real deployment stores the frozen base at its native width: 4.5 bits for
    # QLoRA's NF4, 16 bits for an FP16 LoRA base, and trains in FP32.
    base_bits = 4.5 if mode == "qlora" else 16
    weights = frozen * base_bits / 8 + train * 4
    grads = train * 4
    adam = train * 8                      # AdamW keeps two fp32 moments
    return {"mode": mode, "frozen_params": frozen, "trainable_params": train,
            "weights_MB": weights / 1e6, "grads_MB": grads / 1e6,
            "optimizer_MB": adam / 1e6, "activations_MB": act_bytes / 1e6,
            "total_MB": (weights + grads + adam + act_bytes) / 1e6}


def interleaved_step_times(tok, x, rounds=4):
    """Time one training step of each mode, round-robin.

    The three modes cannot be timed one after another on this machine: it is
    shared, its load moves on a scale of minutes, and running them in sequence
    would charge whichever mode went last for whatever else started meanwhile.
    Interleaving spreads that interference evenly, and reporting the *minimum*
    of several rounds reports the run that was interfered with least -- a step
    can only be slowed down, never sped up.
    """
    built = {}
    for mode in ["full", "lora", "qlora"]:
        model, _ = build(mode, tok)
        params = [p for p in model.parameters() if p.requires_grad]
        built[mode] = (model, params, torch.optim.AdamW(params, lr=1e-9))
        model.train()

    best = {m: float("inf") for m in built}
    for _ in range(rounds):
        for mode, (model, params, opt) in built.items():
            t = time.perf_counter()
            loss = model(x, labels=x).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            best[mode] = min(best[mode], time.perf_counter() - t)
    del built
    return best


def batches(tok, text, n, seq, seed=0):
    ids = tok(text[: 8 * n * seq + 20000], return_tensors="pt").input_ids[0]
    ids = ids[: (ids.numel() // seq) * seq].view(-1, seq)
    g = torch.Generator().manual_seed(seed)
    return ids[torch.randperm(ids.shape[0], generator=g)][:n]


def train(mode, tok, train_b, eval_b, lr):
    model, n_lora = build(mode, tok)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()

    act_bytes = None
    curve = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        start = (step * BS) % (train_b.shape[0] - BS + 1)
        x = train_b[start:start + BS]
        if step == 0:
            with N.ActivationBytes() as ab:
                loss = model(x, labels=x).loss
            act_bytes = ab.total
        else:
            loss = model(x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 5 == 0 or step == STEPS - 1:
            curve.append((step, float(loss)))
    dt = time.perf_counter() - t0

    model.eval()
    ppl = ql.perplexity(model, eval_b)
    mem = memory_model(model, mode, act_bytes)
    mem.update({"lora_layers": n_lora, "ppl_after": ppl, "seconds": dt,
                "curve": curve, "s_per_step": dt / STEPS})
    log(f"  {mode:6s} trainable {mem['trainable_params'] / 1e6:7.3f} M "
        f"({100 * mem['trainable_params'] / (mem['frozen_params'] + mem['trainable_params']):5.2f}%)"
        f"  code ppl {ppl:7.3f}  total {mem['total_MB']:7.1f} MB  "
        f"{dt / STEPS:.2f} s/step")
    del model, opt
    return mem


def make_plots(res):
    """Rebuild the figure from findings.json (so `run.py --plot` is instant)."""
    rows = res["runs"]
    base_ppl = res["base_ppl"]
    nf4_ppl = res["nf4_base_ppl"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    for r in rows:
        axes[0].plot([c[0] for c in r["curve"]], [c[1] for c in r["curve"]],
                     marker="o", markersize=3, label=r["mode"])
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Fine-tuning on Python source")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].bar([r["mode"] for r in rows], [r["ppl_after"] for r in rows],
                color=["#444444", "#1f77b4", "#2ca02c"])
    axes[1].axhline(base_ppl, ls="--", color="k",
                    label=f"fp32 base, untrained ({base_ppl:.2f})")
    axes[1].axhline(nf4_ppl, ls=":", color="#2ca02c",
                    label=f"NF4 base, untrained ({nf4_ppl:.2f})")
    axes[1].set_ylabel("held-out Python perplexity")
    axes[1].set_ylim(0, max(nf4_ppl, base_ppl) * 1.25)
    axes[1].set_title(f"Quality after {res.get('steps', '?')} steps")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    keys = ["weights_MB", "grads_MB", "optimizer_MB", "activations_MB"]
    bottom = [0.0] * len(rows)
    for key in keys:
        vals = [r[key] for r in rows]
        axes[2].bar([r["mode"] for r in rows], vals, bottom=bottom,
                    label=key.replace("_MB", ""))
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[2].set_ylabel("MB")
    axes[2].set_title("Memory for one training step")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"QLoRA on {MODEL}")
    fig.savefig(f"{OUT}/qlora.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/qlora.png")


def main():
    ql.setup()
    t_start = time.time()
    section_a()

    tok, probe = ql.load(MODEL)
    section_b(probe)
    code = ql.code_text()
    train_b = batches(tok, code, 64, SEQ, seed=1)
    eval_b = batches(tok, code, 6, SEQ, seed=2)
    base_ppl = ql.perplexity(probe, eval_b)
    log(f"\n  before any fine-tuning, the model scores "
        f"{base_ppl:.3f} perplexity on held-out Python")
    # QLoRA starts from a *damaged* model, so this is the number its adapter has
    # to climb back from -- comparing it against the fp32 base would confuse
    # "what NF4 cost" with "what the fine-tune gained".
    with ql.QuantizedWeights(probe, lambda n, w: N.nf4_fake_quant(w, block=64)):
        nf4_ppl = ql.perplexity(probe, eval_b)
    log(f"  after NF4 quantization but before any training: {nf4_ppl:.3f} "
        f"({100 * (nf4_ppl / base_ppl - 1):+.1f}%)")
    results["nf4_base_ppl"] = nf4_ppl
    del probe

    log("\n=== C. Three ways to fine-tune the same model ===")
    rows = []
    for mode, lr in [("full", 1e-4), ("lora", 4e-4), ("qlora", 4e-4)]:
        rows.append(train(mode, tok, train_b, eval_b, lr))
    results["runs"] = rows
    results["steps"] = STEPS
    results["base_ppl"] = base_ppl

    log("\n=== D. Where the memory goes ===")
    log(f"  {'mode':7s} {'weights':>9s} {'grads':>8s} {'optimizer':>10s} "
        f"{'activations':>12s} {'total':>9s}")
    for r in rows:
        log(f"  {r['mode']:7s} {r['weights_MB']:8.1f}M {r['grads_MB']:7.1f}M "
            f"{r['optimizer_MB']:9.1f}M {r['activations_MB']:11.1f}M "
            f"{r['total_MB']:8.1f}M")
    full, qlora = rows[0], rows[2]
    log(f"  QLoRA needs {full['total_MB'] / qlora['total_MB']:.2f}x less than a "
        f"full fine-tune of the same model.")
    results["memory_ratio"] = full["total_MB"] / qlora["total_MB"]

    log("\n=== D2. Step time, measured round-robin ===")
    step_ms = interleaved_step_times(tok, train_b[:BS])
    for mode, sec in step_ms.items():
        log(f"  {mode:7s} {sec:6.2f} s/step   "
            f"({sec / step_ms['full']:.2f}x full fine-tune)")
    log("  Training 0.68% of the parameters does NOT make the step 100x cheaper:"
        "\n  the backward pass still traverses every frozen layer, because the "
        "*inputs* of\n  each layer need gradients even when its weights do not. "
        "Only the weight-gradient\n  and optimizer work is saved.")
    log("  Compare these with the s/step printed in section C, which were "
        "measured\n  sequentially on a shared machine and disagree -- that is "
        "the whole reason\n  this section exists.")
    results["step_seconds"] = step_ms

    # ------------------------------------------------------------ scaling up
    log("\n=== E. The same accounting for a 7B model ===")
    seven = []
    for mode in ["full", "lora", "qlora"]:
        n = 7.0e9
        if mode == "full":
            w, tr = n * 4, n
        else:
            trainable = n * (rows[1]["trainable_params"] /
                             (rows[1]["frozen_params"] + rows[1]["trainable_params"]))
            w = n * (4.5 if mode == "qlora" else 16) / 8 + trainable * 4
            tr = trainable
        gb = (w + tr * 4 + tr * 8) / 1e9
        seven.append({"mode": mode, "GB": gb})
        log(f"  {mode:6s} weights+grads+optimizer = {gb:7.2f} GB"
            f"   {'(fits a 24 GB card)' if gb < 22 else '(does not fit 24 GB)'}")
    results["seven_b"] = seven

    make_plots(results)
    results["total_seconds"] = time.time() - t_start
    ql.save_json(f"{OUT}/findings.json", results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("mode,trainable_params,frozen_params,ppl_after,weights_MB,"
                "grads_MB,optimizer_MB,activations_MB,total_MB,s_per_step\n")
        for r in rows:
            f.write(f"{r['mode']},{r['trainable_params']},{r['frozen_params']},"
                    f"{r['ppl_after']:.4f},{r['weights_MB']:.2f},"
                    f"{r['grads_MB']:.2f},{r['optimizer_MB']:.2f},"
                    f"{r['activations_MB']:.2f},{r['total_MB']:.2f},"
                    f"{r['s_per_step']:.3f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:      # redraw from the committed findings.json
        make_plots(json.load(open(f"{OUT}/findings.json")))
    else:
        main()
