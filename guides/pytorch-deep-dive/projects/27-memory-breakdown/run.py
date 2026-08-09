"""Project 27 — Memory breakdown.

Predict the four buckets of training memory for a transformer, then measure each
one:

  1. parameters, gradients, optimizer state — exact byte counts vs the formula
  2. activations — measured with a saved-tensor hook, vs a hand-derived estimate
  3. which optimizer costs what (SGD / SGD+momentum / Adam / AdamW)
  4. the batch-size sweep: only one bucket grows, and where it takes over
  5. gradient checkpointing, and what it does to the activation bucket
  6. set_to_none=True: the bucket you can free between steps
  7. the same arithmetic at 1B and 7B parameters

Runtime ~2 min. Needs torch, numpy, matplotlib.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24-profile-a-training-step"))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

rows = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<42} {value:>14}  {note}")


def measure(model, opt, batch=P.BATCH, seq=P.SEQ, use_checkpoint=False):
    """One step; return the four buckets in bytes."""
    x, y = P.make_batch(batch=batch, seq=seq)
    with P.ActivationBytes(model) as tracker:
        if use_checkpoint:
            h = model.tok(x) + model.pos(torch.arange(seq))
            for blk in model.blocks:
                h = checkpoint(blk, h, use_reentrant=False)
            logits = model.head(model.nf(h))
        else:
            logits = model(x)
        loss = P.loss_fn(logits, y)
        # backward inside the tracker: with checkpointing the recomputed
        # activations only exist during the backward pass, and they count
        loss.backward()
        act_peak = tracker.peak
    params = P.param_bytes(model)
    grads = P.grad_bytes(model)
    opt.step()
    state = P.optimizer_bytes(opt)
    opt.zero_grad(set_to_none=True)
    return dict(params=params, grads=grads, optim=state, acts=act_peak)


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | CPU only\n")

    # -----------------------------------------------------------------
    # 1. the three easy buckets
    # -----------------------------------------------------------------
    print("[1] parameters, gradients, optimizer state")
    model = P.new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n = P.count_params(model)
    record("params", "parameter count", f"{n:,}")
    record("params", "parameters (fp32)", P.human(P.param_bytes(model)),
           f"{n} x 4 bytes")

    d, L, V, T = P.D_MODEL, P.N_LAYER, P.VOCAB, P.SEQ
    predicted = L * (12 * d * d + 13 * d) + (V + T) * d + 2 * d + V * d
    record("params", "predicted by formula", f"{predicted:,}",
           "12*d^2 per block + embeddings + head")
    record("params", "formula error", f"{100*abs(predicted-n)/n:.2f} %")

    buckets = measure(model, opt)
    for k in ("params", "grads", "optim", "acts"):
        record("buckets", k, P.human(buckets[k]),
               f"{100*buckets[k]/sum(buckets.values()):.1f} % of total")
    record("buckets", "total", P.human(sum(buckets.values())))
    record("buckets", "grads / params", f"{buckets['grads']/buckets['params']:.2f}x",
           "one gradient per parameter")
    record("buckets", "AdamW state / params", f"{buckets['optim']/buckets['params']:.2f}x",
           "exp_avg + exp_avg_sq")

    # -----------------------------------------------------------------
    # 2. the activation estimate
    # -----------------------------------------------------------------
    print("\n[2] activations, predicted vs measured")
    B = P.BATCH
    # per block, in elements: layernorm in/out, qkv, attention out, proj in,
    # layernorm, fc1 in, fc1 out (4d), gelu out (4d)
    per_block = B * T * d * (2 + 3 + 1 + 1 + 2 + 4 + 4)
    est = 4 * (L * per_block + B * T * d * 2 + B * T * V)
    record("activations", "measured peak", P.human(buckets["acts"]))
    record("activations", "hand estimate", P.human(est),
           "~17 tensors of B*T*d per block")
    record("activations", "ratio", f"{buckets['acts']/est:.2f}x")
    record("activations", "per sample", P.human(buckets["acts"] / B))

    # -----------------------------------------------------------------
    # 3. optimizers
    # -----------------------------------------------------------------
    print("\n[3] what each optimizer stores")
    for name, make in (
        ("SGD", lambda p: torch.optim.SGD(p, lr=1e-2)),
        ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9)),
        ("Adam", lambda p: torch.optim.Adam(p, lr=1e-3)),
        ("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3)),
    ):
        m = P.new_model()
        o = make(m.parameters())
        b = measure(m, o)
        record("optimizers", name, P.human(b["optim"]),
               f"{b['optim']/b['params']:.0f} x parameters")

    # -----------------------------------------------------------------
    # 4. only one bucket grows with the batch
    # -----------------------------------------------------------------
    print("\n[4] batch-size sweep")
    sizes = [1, 2, 4, 8, 16, 32]
    sweep = []
    for b in sizes:
        m = P.new_model()
        o = torch.optim.AdamW(m.parameters(), lr=3e-4)
        bk = measure(m, o, batch=b)
        sweep.append(bk)
        record("sweep", f"batch {b}", P.human(bk["acts"]),
               f"activations = {100*bk['acts']/sum(bk.values()):.0f} % of total")
    fixed = sweep[0]["params"] + sweep[0]["grads"] + sweep[0]["optim"]
    per_sample = sweep[-1]["acts"] / sizes[-1]
    record("sweep", "fixed cost (params+grads+optim)", P.human(fixed))
    record("sweep", "activations per sample", P.human(per_sample))
    record("sweep", "crossover batch size", f"{fixed/per_sample:.1f}",
           "above this, activations dominate")

    # -----------------------------------------------------------------
    # 5. checkpointing
    # -----------------------------------------------------------------
    print("\n[5] gradient checkpointing")
    m = P.new_model()
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    ck = measure(m, o, use_checkpoint=True)
    record("checkpoint", "activations held after the forward", P.human(ck["acts"]))
    record("checkpoint", "vs plain", f"{buckets['acts']/ck['acts']:.2f}x smaller",
           "an upper bound — see the note below")
    record("checkpoint", "total, checkpointed", P.human(sum(ck.values())),
           f"vs {P.human(sum(buckets.values()))} plain")
    record("checkpoint", "what this hook cannot see", "the recompute",
           "torch.utils.checkpoint installs its own saved_tensors_hooks inside")

    # -----------------------------------------------------------------
    # 6. set_to_none
    # -----------------------------------------------------------------
    print("\n[6] freeing the gradient bucket")
    m = P.new_model()
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x, y = P.make_batch()
    P.loss_fn(m(x), y).backward()
    o.step()
    before = P.grad_bytes(m)
    o.zero_grad(set_to_none=False)
    record("zero_grad", "grad bytes after zero_grad(False)", P.human(P.grad_bytes(m)),
           f"was {P.human(before)}")
    o.zero_grad(set_to_none=True)
    record("zero_grad", "grad bytes after zero_grad(True)", P.human(P.grad_bytes(m)),
           "the tensors are gone, not zeroed")

    # -----------------------------------------------------------------
    # 7. the same arithmetic at scale
    # -----------------------------------------------------------------
    print("\n[7] extrapolation")
    GB = 1024 ** 3
    for label, N in (("1B", 1e9), ("7B", 7e9)):
        fp32 = N * (4 + 4 + 8) / GB
        mixed = N * (2 + 4 + 4 + 8) / GB
        lora = N * (2) / GB
        record("scale", f"{label} fp32 AdamW (GB)", f"{fp32:.1f}",
               "4 params + 4 grads + 8 state")
        record("scale", f"{label} mixed-precision AdamW (GB)", f"{mixed:.1f}",
               "bf16 copy + fp32 master + grads + state")
        record("scale", f"{label} frozen bf16 weights only (GB)", f"{lora:.1f}",
               "what LoRA / inference needs")

    # -----------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------
    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    keys = ["params", "grads", "optim", "acts"]
    vals = [P.mb(buckets[k]) for k in keys]
    ax.bar(np.arange(4), vals, color=ps.SERIES[:4])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.1f}", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["parameters", "gradients", "optimizer", "activations"], fontsize=9)
    ax.set_title(f"Four buckets, batch {P.BATCH} x {P.SEQ}", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_ylabel("MB", color=ps.INK_SECONDARY)

    ax = axes[1]
    bottom = np.zeros(len(sizes))
    for i, k in enumerate(keys):
        v = np.array([P.mb(s[k]) for s in sweep])
        ax.bar(np.arange(len(sizes)), v, bottom=bottom, color=ps.SERIES[i], label=k)
        bottom += v
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Only activations grow with the batch", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("batch size", color=ps.INK_SECONDARY)
    ax.set_ylabel("MB", color=ps.INK_SECONDARY)

    ax = axes[2]
    labels = ["plain", "checkpointed"]
    plain_v = [P.mb(buckets[k]) for k in keys]
    ck_v = [P.mb(ck[k]) for k in keys]
    bottom = np.zeros(2)
    for i, k in enumerate(keys):
        v = np.array([plain_v[i], ck_v[i]])
        ax.bar(np.arange(2), v, bottom=bottom, color=ps.SERIES[i], label=k, width=0.5)
        bottom += v
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Checkpointing trims one bucket only", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_ylabel("MB", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "memory_breakdown.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
