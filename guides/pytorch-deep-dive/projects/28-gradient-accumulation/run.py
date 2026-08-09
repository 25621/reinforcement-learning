"""Project 28 — Gradient accumulation.

Show that k micro-batches really do reproduce one k-times-bigger batch, then
break it in the four ways people break it in practice:

  1. exactness: 4 x 8 vs 1 x 32, gradient by gradient
  2. the forgotten 1/k
  3. uneven micro-batches: the mean of means is not the mean
  4. BatchNorm breaks the equivalence; LayerNorm does not
  5. dropout: same weights, different masks
  6. what it buys and what it costs: activation memory and step time vs k
  7. 100 real training steps, accumulated vs not

Runtime ~3 min. Needs torch, numpy, matplotlib.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24-profile-a-training-step"))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

BIG = 32
rows = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<44} {value:>14}  {note}")


def flat_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                      if p.grad is not None])


def big_batch_grads(model, x, y):
    model.zero_grad(set_to_none=True)
    P.loss_fn(model(x), y).backward()
    return flat_grads(model).clone()


def accumulated_grads(model, x, y, k, divide=True, weights=None):
    model.zero_grad(set_to_none=True)
    chunks = torch.chunk(x, k) if weights is None else torch.split(x, weights)
    ychunks = torch.chunk(y, k) if weights is None else torch.split(y, weights)
    for xc, yc in zip(chunks, ychunks):
        loss = P.loss_fn(model(xc), yc)
        (loss / len(chunks) if divide else loss).backward()
    return flat_grads(model).clone()


def rel_diff(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | CPU only\n")
    model = P.new_model(d=128, n_layer=2, seq=64)
    x, y = P.make_batch(batch=BIG, seq=64)

    # -----------------------------------------------------------------
    # 1. exactness
    # -----------------------------------------------------------------
    print("[1] one big batch vs k micro-batches")
    ref = big_batch_grads(model, x, y)
    for k in (2, 4, 8):
        acc = accumulated_grads(model, x, y, k)
        record("exact", f"k={k} max abs diff", f"{(acc-ref).abs().max():.3e}",
               f"largest gradient {ref.abs().max():.3f}")
        record("exact", f"k={k} relative", f"{rel_diff(acc, ref):.3e}")
        cos = torch.nn.functional.cosine_similarity(acc.double(), ref.double(), dim=0)
        record("exact", f"k={k} cosine similarity", f"{cos.item():.10f}")
    record("exact", "bit-identical?", "no",
           "float addition is not associative — order changed")

    # -----------------------------------------------------------------
    # 2. the forgotten 1/k
    # -----------------------------------------------------------------
    print("\n[2] forgetting to divide by k")
    for k in (2, 4, 8):
        acc = accumulated_grads(model, x, y, k, divide=False)
        record("nodivide", f"k={k} norm ratio vs big batch",
               f"{(acc.norm()/ref.norm()):.4f}", f"exactly k = {k}")

    # -----------------------------------------------------------------
    # 3. uneven micro-batches
    # -----------------------------------------------------------------
    print("\n[3] uneven micro-batches")
    splits = [20, 8, 4]
    acc = accumulated_grads(model, x, y, k=None, weights=splits)
    record("uneven", f"splits {splits}, mean of means",
           f"{rel_diff(acc, ref):.3e}", "relative error vs the true mean")
    # the fix: weight each micro-batch by its share of the samples
    model.zero_grad(set_to_none=True)
    for xc, yc in zip(torch.split(x, splits), torch.split(y, splits)):
        (P.loss_fn(model(xc), yc) * len(xc) / BIG).backward()
    fixed = flat_grads(model).clone()
    record("uneven", "weighted by sample count", f"{rel_diff(fixed, ref):.3e}",
           "back to float noise")

    # -----------------------------------------------------------------
    # 4. normalization layers
    # -----------------------------------------------------------------
    print("\n[4] BatchNorm vs LayerNorm")
    torch.manual_seed(0)
    xs = torch.randn(BIG, 64)
    ys = torch.randn(BIG, 8)
    for name, norm in (("BatchNorm1d", nn.BatchNorm1d(128)), ("LayerNorm", nn.LayerNorm(128))):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(64, 128), norm, nn.GELU(), nn.Linear(128, 8))
        net.zero_grad(set_to_none=True)
        ((net(xs) - ys) ** 2).mean().backward()
        r = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
        net.zero_grad(set_to_none=True)
        for xc, yc in zip(torch.chunk(xs, 4), torch.chunk(ys, 4)):
            (((net(xc) - yc) ** 2).mean() / 4).backward()
        a = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
        record("norm", f"{name} relative diff", f"{rel_diff(a, r):.3e}",
               "batch statistics change with the micro-batch"
               if "Batch" in name else "per-sample statistics — unaffected")

    # -----------------------------------------------------------------
    # 5. dropout
    # -----------------------------------------------------------------
    print("\n[5] dropout")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 8))
    torch.manual_seed(123)
    net.zero_grad(set_to_none=True)
    ((net(xs) - ys) ** 2).mean().backward()
    r = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
    torch.manual_seed(123)
    net.zero_grad(set_to_none=True)
    for xc, yc in zip(torch.chunk(xs, 4), torch.chunk(ys, 4)):
        (((net(xc) - yc) ** 2).mean() / 4).backward()
    a = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
    record("dropout", "same seed, relative diff", f"{rel_diff(a, r):.3e}",
           "the RNG is consumed in a different pattern")
    net.eval()
    net.zero_grad(set_to_none=True)
    ((net(xs) - ys) ** 2).mean().backward()
    r2 = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
    net.zero_grad(set_to_none=True)
    for xc, yc in zip(torch.chunk(xs, 4), torch.chunk(ys, 4)):
        (((net(xc) - yc) ** 2).mean() / 4).backward()
    a2 = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
    record("dropout", "eval mode, relative diff", f"{rel_diff(a2, r2):.3e}",
           "no randomness left")
    # why it matched: on CPU one big draw and k small draws walk the same stream
    torch.manual_seed(5)
    one = torch.bernoulli(torch.full((32, 128), 0.8))
    torch.manual_seed(5)
    many = torch.cat([torch.bernoulli(torch.full((8, 128), 0.8)) for _ in range(4)])
    record("dropout", "one 32x128 mask == four 8x128 masks",
           str(bool((one == many).all())),
           "CPU draws sequentially — CUDA gives each launch its own offset")

    # -----------------------------------------------------------------
    # 6. memory and time
    # -----------------------------------------------------------------
    print("\n[6] what accumulation costs")
    ks = (1, 2, 4, 8)
    mem, times = {}, {k: [] for k in ks}
    steps = {}
    for k in ks:
        m = P.new_model(d=128, n_layer=2, seq=64)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        peaks = []

        def one_step(k=k, m=m, opt=opt, peaks=peaks):
            opt.zero_grad(set_to_none=True)
            for xc, yc in zip(torch.chunk(x, k), torch.chunk(y, k)):
                with P.ActivationBytes(m) as tr:
                    loss = P.loss_fn(m(xc), yc) / k
                    peaks.append(tr.peak)
                loss.backward()
            opt.step()

        steps[k] = one_step
        one_step()
        mem[k] = max(peaks)
    # interleave the four configurations across rounds: measuring all of k=1 and
    # then all of k=8 lets a busy minute on a shared machine land on one of them
    for _ in range(5):
        for k in ks:
            times[k].append(P.best_of(steps[k], repeats=3, warmup=1)[0])
    for k in ks:
        record("cost", f"k={k} peak activations", P.human(mem[k]),
               f"micro-batch {BIG//k}")
        record("cost", f"k={k} step time (ms)", f"{min(times[k]):.1f}",
               f"spread over rounds {max(times[k])-min(times[k]):.1f}")
    times = {k: min(v) for k, v in times.items()}
    record("cost", "memory saving k=8 vs k=1", f"{mem[1]/mem[8]:.2f}x")
    record("cost", "time cost k=8 vs k=1", f"{times[8]/times[1]:.2f}x")

    # -----------------------------------------------------------------
    # 7. 100 real steps
    # -----------------------------------------------------------------
    print("\n[7] 100 training steps, accumulated vs not")
    curves = {}
    finals = {}
    for k in (1, 4):
        m = P.new_model(seed=0, d=128, n_layer=2, seq=64)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(0)
        losses = []
        for _ in range(100):
            xb, yb = P.make_batch(batch=BIG, seq=64, gen=gen)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for xc, yc in zip(torch.chunk(xb, k), torch.chunk(yb, k)):
                loss = P.loss_fn(m(xc), yc) / k
                loss.backward()
                tot += loss.item()
            opt.step()
            losses.append(tot)
        curves[k] = losses
        finals[k] = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
        record("train", f"k={k} final loss", f"{losses[-1]:.5f}")
    drift = (finals[1] - finals[4]).abs().max().item()
    record("train", "max weight difference after 100 steps", f"{drift:.3e}",
           f"weights are ~{finals[1].abs().mean():.3f} on average")
    record("train", "final loss difference",
           f"{abs(curves[1][-1]-curves[4][-1]):.3e}")

    # -----------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------
    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    ks = [1, 2, 4, 8]
    ax.plot(ks, [P.mb(mem[k]) for k in ks], "o-", color=ps.SERIES[0], lw=1.8,
            label="peak activations (MB)")
    ax.plot(ks, [P.mb(mem[1]) / k for k in ks], "--", color=ps.BASELINE, lw=1.2,
            label="ideal 1/k")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Activation memory falls as 1/k", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("accumulation steps k", color=ps.INK_SECONDARY)
    ax.set_ylabel("MB", color=ps.INK_SECONDARY)

    ax = axes[1]
    ax.plot(ks, [times[k] / times[1] for k in ks], "o-", color=ps.SERIES[2], lw=1.8)
    ax.axhline(1.0, color=ps.BASELINE, ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_title("...and what that costs in time", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("accumulation steps k", color=ps.INK_SECONDARY)
    ax.set_ylabel("step time relative to k=1", color=ps.INK_SECONDARY)

    ax = axes[2]
    for i, k in enumerate((1, 4)):
        ax.plot(curves[k], color=ps.SERIES[i], lw=1.6,
                label=f"batch 32, k={k}" if k > 1 else "batch 32, one shot")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The two curves lie on top of each other", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY)
    ax.set_ylabel("loss", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "gradient_accumulation.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
