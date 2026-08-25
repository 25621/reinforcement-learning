"""Project 39 — Cut one attention layer in half and keep the answer identical.

Run:  python3 run.py           (~3 minutes)

Sections
  1. the split: which weight is cut which way, and why
  2. forward equality with the single-process reference
  3. f and g: the two collectives, and what breaks without each
  4. the head-boundary trap (a split that runs fine and is wrong)
  5. PyTorch's own ColwiseParallel / RowwiseParallel, same numbers
  6. what it costs: memory saved, bytes moved, wall time
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "36-two-gpu-ddp"))
sys.path.insert(0, os.path.join(HERE, "..", "38-fsdp-a-transformer"))
import dist_lib as D  # noqa: E402
from model import Block  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []

D_MODEL, N_HEADS, BATCH, SEQ = 512, 8, 8, 64


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<52} {value}")


def reference_block(seed=0):
    torch.manual_seed(seed)
    return Block(D_MODEL, N_HEADS)


def reference_input(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ, D_MODEL, generator=g)


# ---------------------------------------------------------------------------
# the two collectives, as autograd functions
# ---------------------------------------------------------------------------

class CopyToRanks(torch.autograd.Function):
    """Megatron calls this `f`.

    Forward: nothing — every rank already has the same input activations.
    Backward: all-reduce. Each rank computed a *partial* derivative with respect
    to that shared input (its own heads' share), and the true gradient is the
    sum of all of them. Skip this and every rank keeps only its own fragment,
    so anything upstream of the attention layer trains on a fraction of the
    signal.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad


class ReduceFromRanks(torch.autograd.Function):
    """Megatron calls this `g` — the mirror image of `f`.

    Forward: all-reduce, because after a row-parallel matmul each rank holds a
    partial sum of the same output and they must be added.
    Backward: nothing, because the gradient arriving at that output is already
    identical on every rank.
    """

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


# ---------------------------------------------------------------------------
# the tensor-parallel block
# ---------------------------------------------------------------------------

def slice_qkv(qkv_weight, rank, world, n_heads, d_model, by_head=True):
    """Take this rank's slice of the fused qkv weight.

    `qkv.weight` has shape [3*d, d]: the q rows, then the k rows, then the v
    rows. Column parallelism means "give each rank some of the OUTPUT
    features" — but the output features are grouped into heads, and a head only
    makes sense whole. So we slice q, k and v *separately*, each by head, and
    stack the three slices back together.

    `by_head=False` reproduces the tempting one-liner `qkv_weight.chunk(world,
    dim=0)`, which hands rank 0 all of q and half of k. It runs. It is wrong.
    """
    if not by_head:
        return qkv_weight.chunk(world, dim=0)[rank].contiguous()
    d_local = d_model // world
    parts = []
    for i in range(3):                       # q, k, v
        block = qkv_weight[i * d_model:(i + 1) * d_model]
        parts.append(block[rank * d_local:(rank + 1) * d_local])
    return torch.cat(parts, dim=0).contiguous()


class TPAttention(nn.Module):
    """Multi-head attention split over `world` ranks, Megatron style."""

    def __init__(self, ref: Block, rank, world, by_head=True, use_f=True, use_g=True):
        super().__init__()
        self.rank, self.world = rank, world
        self.n_heads_local = N_HEADS // world
        self.d_head = D_MODEL // N_HEADS
        self.use_f, self.use_g = use_f, use_g

        d_local = D_MODEL // world
        self.qkv = nn.Linear(D_MODEL, 3 * d_local, bias=False)
        self.proj = nn.Linear(d_local, D_MODEL, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(slice_qkv(ref.qkv.weight, rank, world,
                                            N_HEADS, D_MODEL, by_head))
            # row parallel: split the proj weight along its INPUT dimension,
            # because its input is the concatenation of the heads
            self.proj.weight.copy_(
                ref.proj.weight[:, rank * d_local:(rank + 1) * d_local])

    def forward(self, x):
        if self.use_f:
            x = CopyToRanks.apply(x)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.n_heads_local, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = self.proj(o)                      # partial sum on every rank
        if self.use_g:
            o = ReduceFromRanks.apply(o)
        return o


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def w_tp(rank, world, by_head=True, use_f=True, use_g=True):
    ref = reference_block()
    x = reference_input()

    # single-process reference, computed on every rank so we can compare locally
    xr = x.clone().requires_grad_(True)
    out_ref = ref.attn(xr)
    out_ref.sum().backward()
    gx_ref = xr.grad.clone()
    gq_ref = ref.qkv.weight.grad.clone()

    tp = TPAttention(ref, rank, world, by_head, use_f, use_g)
    xt = x.clone().requires_grad_(True)
    out_tp = tp(xt)
    out_tp.sum().backward()

    # gather the qkv weight gradient back into the reference layout
    d_local = D_MODEL // world
    g_local = tp.qkv.weight.grad
    gathered = [torch.empty_like(g_local) for _ in range(world)]
    dist.all_gather(gathered, g_local.contiguous())
    if by_head:
        rebuilt = torch.cat([torch.cat([g[i * d_local:(i + 1) * d_local]
                                        for g in gathered], dim=0)
                             for i in range(3)], dim=0)
    else:
        rebuilt = torch.cat(gathered, dim=0)

    out_tp, out_ref = out_tp.detach(), out_ref.detach()
    return {
        "fwd_err": float((out_tp - out_ref).abs().max()),
        "fwd_rel": float((out_tp - out_ref).abs().max() / out_ref.abs().max()),
        "gx_err": float((xt.grad - gx_ref).abs().max()),
        "gx_rel": float((xt.grad - gx_ref).abs().max() / gx_ref.abs().max()),
        "gw_err": float((rebuilt - gq_ref).abs().max()),
        "gw_rel": float((rebuilt - gq_ref).abs().max() / gq_ref.abs().max()),
        "local_qkv_shape": tuple(tp.qkv.weight.shape),
        "local_proj_shape": tuple(tp.proj.weight.shape),
        "local_params": sum(p.numel() for p in tp.parameters()),
        "ref_params": ref.qkv.weight.numel() + ref.proj.weight.numel(),
    }


def section_1_2():
    print("\n[1] the split")
    ref = reference_block()
    record("split", "d_model / heads / head dim",
           f"{D_MODEL} / {N_HEADS} / {D_MODEL // N_HEADS}")
    record("split", "qkv.weight (fused q,k,v)", tuple(ref.qkv.weight.shape))
    record("split", "proj.weight", tuple(ref.proj.weight.shape))

    r2 = D.launch(w_tp, 2, threads=3)
    record("split", "rank 0 qkv slice, 2 ranks", r2[0]["local_qkv_shape"])
    record("split", "rank 0 proj slice, 2 ranks", r2[0]["local_proj_shape"])
    record("split", "attention params per rank / single process",
           f"{r2[0]['local_params']:,} / {r2[0]['ref_params']:,}"
           f"  ({r2[0]['ref_params'] / r2[0]['local_params']:.1f}x smaller)")

    print("\n[2] does it give the same answer?")
    record("equal", "2 ranks: max |out(TP) - out(1 process)|", f"{r2[0]['fwd_err']:.3e}")
    record("equal", "2 ranks: relative to the largest output value",
           f"{r2[0]['fwd_rel']:.3e}")
    record("equal", "2 ranks: max error in d(loss)/d(input)", f"{r2[0]['gx_err']:.3e}")
    record("equal", "2 ranks: max error in d(loss)/d(qkv weight)", f"{r2[0]['gw_err']:.3e}")
    record("equal", "2 ranks: the same three, relative to the largest true value",
           f"{r2[0]['fwd_rel']:.1e} / {r2[0]['gx_rel']:.1e} / {r2[0]['gw_rel']:.1e}",
           "float32 has ~1e-7 of relative precision, so this is as equal as it gets")
    r4 = D.launch(w_tp, 4, threads=2)
    record("equal", "4 ranks: forward / input-grad / weight-grad, relative",
           f"{r4[0]['fwd_rel']:.1e} / {r4[0]['gx_rel']:.1e} / {r4[0]['gw_rel']:.1e}")
    return r2, r4


def section_3():
    print("\n[3] remove f, then remove g")
    no_f = D.launch(w_tp, 2, threads=3, args=(True, False, True))
    no_g = D.launch(w_tp, 2, threads=3, args=(True, True, False))
    record("fg", "no f (no all-reduce in backward): forward error",
           f"{no_f[0]['fwd_err']:.3e}", "forward is untouched - nothing looks wrong")
    record("fg", "no f: input-gradient error (relative "
           f"{no_f[0]['gx_rel']:.2f})", f"{no_f[0]['gx_err']:.3e}",
           "each rank keeps only its own heads' share of the gradient")
    record("fg", "no f: weight-gradient error", f"{no_f[0]['gw_err']:.3e}",
           "this layer's own weights are still correct - the damage is upstream")
    record("fg", "no g (no all-reduce in forward): forward error",
           f"{no_g[0]['fwd_err']:.3e}", "each rank returns only its own partial sum")
    record("fg", "no g: input-gradient error", f"{no_g[0]['gx_err']:.3e}")
    return no_f, no_g


def section_4():
    print("\n[4] the head-boundary trap: qkv.weight.chunk(2, dim=0)")
    bad = D.launch(w_tp, 2, threads=3, args=(False, True, True))
    record("trap", "chunked across the q/k boundary: forward error",
           f"{bad[0]['fwd_err']:.3e}")
    record("trap", "  relative to the largest output value", f"{bad[0]['fwd_rel']:.3e}",
           "no exception, no NaN - just a different function")
    record("trap", "  rank 0 slice shape (looks perfectly reasonable)",
           bad[0]["local_qkv_shape"])
    return bad


# ---------------------------------------------------------------------------
# 5. the built-in API
# ---------------------------------------------------------------------------

def w_dtensor_tp(rank, world):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel)

    mesh = init_device_mesh("cpu", (world,))
    ref = reference_block()
    x = reference_input()
    out_ref = ref.attn(x)

    # A "wide then narrow" pair: column-parallel first layer, row-parallel
    # second. This is the same shape of computation as qkv -> proj, and it is
    # exactly the pattern PyTorch's helpers are built for.
    torch.manual_seed(3)
    mlp = nn.Sequential()
    mlp.add_module("up", nn.Linear(D_MODEL, 4 * D_MODEL, bias=False))
    mlp.add_module("act", nn.GELU())
    mlp.add_module("down", nn.Linear(4 * D_MODEL, D_MODEL, bias=False))
    ref_out = mlp(x).clone()

    plan = {"up": ColwiseParallel(), "down": RowwiseParallel()}
    parallelize_module(mlp, mesh, plan)
    tp_out = mlp(x)
    tp_out = tp_out.to_local() if hasattr(tp_out, "to_local") else tp_out

    up_w = mlp.up.weight
    tp_out, ref_out = tp_out.detach(), ref_out.detach()
    return {"err": float((tp_out - ref_out).abs().max()),
            "up_global": tuple(up_w.shape),
            "up_local": tuple(up_w.to_local().shape),
            "up_placement": str(up_w.placements),
            "down_local": tuple(mlp.down.weight.to_local().shape),
            "down_placement": str(mlp.down.weight.placements),
            "ref_max": float(out_ref.abs().max())}


def section_5():
    print("\n[5] the same thing with PyTorch's own helpers")
    r = D.launch(w_dtensor_tp, 2, threads=3)[0]
    record("api", "ColwiseParallel: global shape -> local shape",
           f"{r['up_global']} -> {r['up_local']}  {r['up_placement']}")
    record("api", "RowwiseParallel: local shape", f"{r['down_local']}  {r['down_placement']}")
    record("api", "max |out(parallelize_module) - out(1 process)|", f"{r['err']:.3e}")
    return r


# ---------------------------------------------------------------------------
# 6. cost
# ---------------------------------------------------------------------------

def w_tp_speed(rank, world, mode):
    ref = reference_block()
    x = reference_input()
    if mode == "tp":
        layer = TPAttention(ref, rank, world)
    else:
        layer = ref.attn

    def step():
        xt = x.clone().requires_grad_(True)
        out = layer(xt)
        out.sum().backward()

    step()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(10):
        step()
    dist.barrier()
    return {"secs": (time.perf_counter() - t0) / 10}


def section_6(r2):
    print("\n[6] what tensor parallelism costs")
    act_bytes = BATCH * SEQ * D_MODEL * 4
    ref = reference_block()
    attn_params = ref.qkv.weight.numel() + ref.proj.weight.numel()
    record("cost", "activation all-reduced per layer, per step (forward)",
           D.fmt_bytes(act_bytes))
    record("cost", "  and once more in the backward pass", D.fmt_bytes(act_bytes))
    record("cost", "gradient all-reduced per step by DDP for this layer",
           D.fmt_bytes(attn_params * 4))
    record("cost", "TP traffic / DDP traffic for one layer, one step",
           f"{2 * act_bytes / (attn_params * 4):.2f}x",
           "TP traffic grows with batch and sequence length; DDP traffic does not")
    bs = BATCH * SEQ
    crossover = attn_params * 4 / (2 * D_MODEL * 4)
    record("cost", "tokens per step at which TP traffic overtakes DDP traffic",
           f"{crossover:.0f}  (we use {bs})")

    t = {"single": [], "tp2": []}
    for _ in range(4):
        t["single"].append(D.launch(w_tp_speed, 1, threads=4, args=("single",))[0]["secs"])
        t["tp2"].append(max(r["secs"] for r in
                            D.launch(w_tp_speed, 2, threads=2, args=("tp",))))
    med = {k: statistics.median(v) for k, v in t.items()}
    record("cost", "1 process x 4 threads, fwd+bwd", f"{med['single'] * 1e3:.1f} ms")
    record("cost", "2 ranks x 2 threads, fwd+bwd (TP)",
           f"{med['tp2'] * 1e3:.1f} ms   ({med['single'] / med['tp2']:.2f}x)")
    return med, act_bytes, attn_params


def figure(r2, r4, no_f, no_g, bad, med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    labels = ["correct TP\n(2 ranks)", "correct TP\n(4 ranks)", "no f\n(bwd all-reduce)",
              "no g\n(fwd all-reduce)", "chunked across\nthe q/k boundary"]
    fwd = [r2[0]["fwd_err"], r4[0]["fwd_err"], no_f[0]["fwd_err"],
           no_g[0]["fwd_err"], bad[0]["fwd_err"]]
    gx = [r2[0]["gx_err"], r4[0]["gx_err"], no_f[0]["gx_err"],
          no_g[0]["gx_err"], bad[0]["gx_err"]]
    xpos = range(len(labels))
    ax[0].bar([p - 0.2 for p in xpos], [max(v, 1e-12) for v in fwd], 0.4, label="forward")
    ax[0].bar([p + 0.2 for p in xpos], [max(v, 1e-12) for v in gx], 0.4, label="input grad")
    ax[0].set_yscale("log")
    ax[0].set_xticks(list(xpos)); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("max abs error vs 1 process")
    ax[0].set_title("Which mistake shows up where")
    ax[0].legend()

    toks = [64, 256, 1024, 4096, 16384]
    attn_params = D_MODEL * D_MODEL * 4
    ax[1].plot(toks, [2 * t * D_MODEL * 4 / 1e6 for t in toks], "o-", label="TP (activations)")
    ax[1].axhline(attn_params * 4 / 1e6, color="#c0392b", ls="--",
                  label="DDP (gradients)")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("tokens per step (batch x seq)")
    ax[1].set_ylabel("MB moved per layer per step")
    ax[1].set_title("Why TP stays inside one node")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    n = ["1 process\n4 threads", "2 ranks\n2 threads (TP)"]
    v = [med["single"] * 1e3, med["tp2"] * 1e3]
    ax[2].bar(n, v, color=["#7f8c8d", "#2980b9"])
    ax[2].set_ylabel("ms per fwd+bwd")
    ax[2].set_title("TP buys memory, not speed (at this size)")
    for i, val in enumerate(v):
        ax[2].text(i, val, f"{val:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    p = os.path.join(OUT, "tensor_parallel.png")
    fig.savefig(p, dpi=120)
    print(f"\n  figure -> {p}")


def main():
    t0 = time.time()
    r2, r4 = section_1_2()
    no_f, no_g = section_3()
    bad = section_4()
    section_5()
    med, _, _ = section_6(r2)
    figure(r2, r4, no_f, no_g, bad, med)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader(); w.writerows(FINDINGS)
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
