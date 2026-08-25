"""Project 38 — FSDP: shard the parameters, borrow them back one layer at a time.

Run:  python3 run.py           (~5 minutes)

FSDP1 (`FullyShardedDataParallel`) refuses to run without a GPU on this machine:
    RuntimeError: FSDP needs a non-CPU accelerator device, but none is detected.
FSDP2 (`fully_shard`, the DTensor-based rewrite) runs on a CPU device mesh, so
that is what this project uses. The concepts are identical.

Sections
  1. the memory arithmetic (16 bytes per parameter, and where they go)
  2. what sharding actually does to a parameter
  3. measured per-rank memory: DDP vs FSDP vs the ZeRO stages
  4. the model that does not fit under DDP but fits under FSDP
  5. is it the same training run? (loss and weights)
  6. what FSDP costs: communication volume and wall time
  7. two traps: the optimizer built too early, and the sharded state_dict
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "36-two-gpu-ddp"))
sys.path.insert(0, HERE)
import dist_lib as D  # noqa: E402
from model import build_gpt, token_batches  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []

VOCAB, D_MODEL, N_LAYERS, N_HEADS, SEQ = 2048, 256, 8, 8, 32
BATCH, STEPS = 4, 6


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<50} {value}")


def make_model():
    return build_gpt(vocab=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                     n_heads=N_HEADS, seq=SEQ)


def shard_it(model, mesh, per_block=True, reshard_after_forward=True):
    """Wrap the model for FSDP2.

    Wrapping *every block* is the point. If you only wrap the root module, FSDP
    has exactly one unit to gather, so the first forward pass materialises the
    entire model at once and you save nothing during the step. Wrapping each
    block means only one block is un-sharded at any moment.
    """
    from torch.distributed.fsdp import fully_shard
    kw = {"mesh": mesh, "reshard_after_forward": reshard_after_forward}
    if per_block:
        for blk in model.blocks:
            fully_shard(blk, **kw)
    fully_shard(model, **kw)
    return model


def full_state_bytes(model):
    """Bytes the *whole* model would take if every rank held all of it."""
    p = sum(x.numel() for x in model.parameters())
    return {"params": 4 * p, "grads": 4 * p, "adam": 8 * p, "total": 16 * p, "n": p}


# ---------------------------------------------------------------------------
# 1-3. memory
# ---------------------------------------------------------------------------

def w_memory(rank, world, mode, per_block=True, reshard=True, optim_early=False):
    from torch.distributed.device_mesh import init_device_mesh

    rss0 = D.rss_peak_mb()
    model = make_model()
    mesh = init_device_mesh("cpu", (world,))

    opt = None
    if mode == "fsdp":
        if optim_early:                     # the trap: parameters are replaced
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model = shard_it(model, mesh, per_block, reshard)
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        net = model
    elif mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    else:                                    # single rank, no wrapper
        net = model
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    x, y = token_batches(STEPS, BATCH, SEQ, VOCAB, seed=4)
    losses = []

    def wsum():
        tot = 0.0
        for p in model.parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            tot += float(local.detach().sum())
        return tot

    w_before = wsum()
    for s in range(STEPS):
        xb = x[s * BATCH:(s + 1) * BATCH]
        yb = y[s * BATCH:(s + 1) * BATCH]
        opt.zero_grad(set_to_none=True)
        logits = net(xb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    acct = D.model_state_bytes(model, opt)
    p0 = next(model.parameters())
    shapes = {"global": tuple(p0.shape),
              "local": tuple(p0.to_local().shape) if hasattr(p0, "to_local") else tuple(p0.shape),
              "type": type(p0.data).__name__}
    return {"acct": acct, "shapes": shapes, "losses": losses,
            "w_before": w_before, "w_after": wsum(),
            "n_optim_states": len(opt.state),
            "rss_start_mb": rss0, "rss_peak_mb": D.rss_peak_mb(),
            "n_params": sum(x.numel() for x in
                            (p.to_local() if hasattr(p, "to_local") else p
                             for p in model.parameters()))}


def section_1_3():
    print("\n[1] the arithmetic: what one parameter costs to train with AdamW")
    m = make_model()
    fb = full_state_bytes(m)
    record("arith", "parameters", f"{fb['n']:,}")
    for k in ("params", "grads", "adam"):
        record("arith", f"{k} (fp32)", D.fmt_bytes(fb[k]))
    record("arith", "total per rank if nothing is sharded", D.fmt_bytes(fb["total"]),
           "16 bytes per parameter: 4 weight + 4 grad + 8 Adam moments")

    print("\n[2] what fully_shard does to one parameter")
    f2 = D.launch(w_memory, 2, threads=3, args=("fsdp",))
    record("shard", "parameter class after fully_shard", f2[0]["shapes"]["type"])
    record("shard", "p.shape (what your code sees)", f2[0]["shapes"]["global"])
    record("shard", "p.to_local().shape (what this rank stores)", f2[0]["shapes"]["local"])
    record("shard", "local parameter count, 2 ranks", f"{f2[0]['n_params']:,}")

    print("\n[3] measured bytes per rank")
    ddp2 = D.launch(w_memory, 2, threads=3, args=("ddp",))
    f4 = D.launch(w_memory, 4, threads=2, args=("fsdp",))
    ddp4 = D.launch(w_memory, 4, threads=2, args=("ddp",))

    rows = []
    for name, res, world in (("DDP, 2 ranks", ddp2, 2), ("FSDP, 2 ranks", f2, 2),
                             ("DDP, 4 ranks", ddp4, 4), ("FSDP, 4 ranks", f4, 4)):
        a = res[0]["acct"]
        rows.append((name, a["total"], res[0]["rss_peak_mb"] - res[0]["rss_start_mb"]))
        record("mem", f"{name}: params+grads+Adam on rank 0",
               f"{D.fmt_bytes(a['total']):>10}   "
               f"(p {D.fmt_bytes(a['params'])}, g {D.fmt_bytes(a['grads'])}, "
               f"o {D.fmt_bytes(a['optim'])})")
    for name, res in (("DDP, 4 ranks", ddp4), ("FSDP, 4 ranks", f4)):
        record("mem", f"{name}: process RSS growth (VmHWM - start)",
               f"{res[0]['rss_peak_mb'] - res[0]['rss_start_mb']:.0f} MB")
    record("mem", "FSDP saving at 4 ranks",
           f"{ddp4[0]['acct']['total'] / f4[0]['acct']['total']:.2f}x smaller")

    # the ZeRO stage table, computed for this model
    print("\n    ZeRO stages, this model, N ranks (bytes per rank):")
    for world in (1, 2, 4, 8):
        p, g, o = fb["params"], fb["grads"], fb["adam"]
        s1 = p + g + o / world
        s2 = p + g / world + o / world
        s3 = (p + g + o) / world
        record("zero", f"world={world}: stage 1 / stage 2 / stage 3 (=FSDP)",
               f"{D.fmt_bytes(s1)} / {D.fmt_bytes(s2)} / {D.fmt_bytes(s3)}")
    return f2, ddp2, f4, ddp4, fb, rows


# ---------------------------------------------------------------------------
# 4. the model that does not fit
# ---------------------------------------------------------------------------

BUDGET_MB = 100


def w_budget(rank, world, mode):
    """Refuse to allocate more than BUDGET_MB of model state, as a stand-in for
    a GPU that is out of memory. A real GPU raises `CUDA out of memory`; we have
    no GPU, so we check the same number ourselves and raise the same way."""
    from torch.distributed.device_mesh import init_device_mesh

    big = dict(vocab=VOCAB, d_model=384, n_layers=10, n_heads=8, seq=SEQ)
    model = build_gpt(**big)
    n = sum(p.numel() for p in model.parameters())
    mesh = init_device_mesh("cpu", (world,))
    if mode == "fsdp":
        model = shard_it(model, mesh)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = token_batches(2, BATCH, SEQ, VOCAB, seed=4)
    ok, msg, used = True, "", 0
    for s in range(2):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[s * BATCH:(s + 1) * BATCH]).reshape(-1, VOCAB),
                               y[s * BATCH:(s + 1) * BATCH].reshape(-1))
        loss.backward()
        opt.step()
        used = D.model_state_bytes(model, opt)["total"]
        if used > BUDGET_MB * 1024 * 1024:
            ok, msg = False, (f"OutOfMemoryError (simulated): tried to keep "
                              f"{used / 1e6:.0f} MB of model state on a device with "
                              f"a {BUDGET_MB} MB budget")
            break
    return {"ok": ok, "msg": msg, "bytes": used, "n_params": n}


def section_4():
    print(f"\n[4] a bigger model against a {BUDGET_MB} MB per-rank budget")
    ddp = D.launch(w_budget, 4, threads=2, args=("ddp",))
    fsdp = D.launch(w_budget, 4, threads=2, args=("fsdp",))
    record("budget", "model", f"{ddp[0]['n_params']:,} parameters")
    record("budget", "DDP, 4 ranks", "FITS" if ddp[0]["ok"] else "FAILS")
    record("budget", "  -> ", ddp[0]["msg"] or D.fmt_bytes(ddp[0]["bytes"]))
    record("budget", "FSDP, 4 ranks", "FITS" if fsdp[0]["ok"] else "FAILS")
    record("budget", "  -> ", fsdp[0]["msg"] or
           f"{D.fmt_bytes(fsdp[0]['bytes'])} of model state per rank")
    return ddp, fsdp


# ---------------------------------------------------------------------------
# 5. is it the same training run?
# ---------------------------------------------------------------------------

def section_5(f2, ddp2, f4):
    print("\n[5] same maths, different storage?")
    d = max(abs(a - b) for a, b in zip(f2[0]["losses"], ddp2[0]["losses"]))
    record("equal", "max |loss(FSDP,2) - loss(DDP,2)| over 6 steps", f"{d:.3e}")
    record("equal", "loss curve, DDP  ", " ".join(f"{v:.3f}" for v in ddp2[0]["losses"]))
    record("equal", "loss curve, FSDP ", " ".join(f"{v:.3f}" for v in f2[0]["losses"]))
    record("equal", "max |loss(FSDP,4) - loss(DDP,2)|",
           f"{max(abs(a - b) for a, b in zip(f4[0]['losses'], ddp2[0]['losses'])):.3e}",
           "4 ranks means 4x the global batch, so this one SHOULD differ")
    return d


# ---------------------------------------------------------------------------
# 6. what it costs
# ---------------------------------------------------------------------------

def w_speed(rank, world, mode, reshard=True, per_block=True):
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cpu", (world,))
    model = make_model()
    if mode == "fsdp":
        model = shard_it(model, mesh, per_block, reshard)
        net = model
    elif mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model)
    else:
        net = model
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x, y = token_batches(STEPS, BATCH, SEQ, VOCAB, seed=4)

    def one():
        for s in range(STEPS):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(x[s * BATCH:(s + 1) * BATCH]).reshape(-1, VOCAB),
                                   y[s * BATCH:(s + 1) * BATCH].reshape(-1))
            loss.backward()
            opt.step()

    one()
    dist.barrier()
    t0 = time.perf_counter()
    one()
    dist.barrier()
    dt = time.perf_counter() - t0
    return {"secs": dt, "rss": D.rss_peak_mb(),
            "state": D.model_state_bytes(model, opt)["total"]}


def section_6(fb):
    print("\n[6] the price of sharding")
    P = fb["n"]
    world = 4
    ddp_vol = 2 * (world - 1) / world * 4 * P
    fsdp_vol = (2 * (world - 1) / world * 4 * P     # reduce-scatter grads (half a ring)
                + 2 * (world - 1) / world * 4 * P)  # all-gather params, fwd + bwd
    record("cost", "DDP bytes moved per rank per step (ring all-reduce)",
           D.fmt_bytes(ddp_vol))
    record("cost", "FSDP bytes moved per rank per step (gather x2 + scatter)",
           D.fmt_bytes(fsdp_vol) + f"  ({fsdp_vol / ddp_vol:.1f}x DDP)")

    times = {"ddp": [], "fsdp": [], "fsdp_noreshard": []}
    for _ in range(3):                              # interleaved rounds
        times["ddp"].append(max(r["secs"] for r in
                                D.launch(w_speed, 4, threads=2, args=("ddp",))))
        times["fsdp"].append(max(r["secs"] for r in
                                 D.launch(w_speed, 4, threads=2, args=("fsdp", True))))
        times["fsdp_noreshard"].append(max(r["secs"] for r in
                                           D.launch(w_speed, 4, threads=2,
                                                    args=("fsdp", False))))
    med = {k: statistics.median(v) for k, v in times.items()}
    record("cost", "4 ranks, 6 steps: DDP", f"{med['ddp'] * 1e3:.0f} ms")
    record("cost", "4 ranks, 6 steps: FSDP (reshard after forward)",
           f"{med['fsdp'] * 1e3:.0f} ms   ({med['fsdp'] / med['ddp']:.2f}x DDP)")
    record("cost", "4 ranks, 6 steps: FSDP (keep params after forward)",
           f"{med['fsdp_noreshard'] * 1e3:.0f} ms   "
           f"({med['fsdp_noreshard'] / med['ddp']:.2f}x DDP)")

    root = D.launch(w_speed, 4, threads=2, args=("fsdp", True, False))
    perb = D.launch(w_speed, 4, threads=2, args=("fsdp", True, True))
    record("cost", "root-only wrapping: RSS peak / model-state bytes",
           f"{root[0]['rss']:.0f} MB / {D.fmt_bytes(root[0]['state'])}")
    record("cost", "per-block wrapping: RSS peak / model-state bytes",
           f"{perb[0]['rss']:.0f} MB / {D.fmt_bytes(perb[0]['state'])}")
    return med


# ---------------------------------------------------------------------------
# 7. two traps
# ---------------------------------------------------------------------------

def w_statedict(rank, world):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, StateDictOptions)

    mesh = init_device_mesh("cpu", (world,))
    model = shard_it(make_model(), mesh)
    naive = model.state_dict()
    k0 = "blocks.0.qkv.weight"
    naive_type = type(naive[k0]).__name__
    naive_shape = tuple(naive[k0].shape)
    naive_local = (tuple(naive[k0].to_local().shape)
                   if hasattr(naive[k0], "to_local") else naive_shape)

    # cpu_offload=True gathers the whole model onto RANK 0 ONLY; the other ranks
    # get an empty dict back. Saving from every rank would therefore write
    # three empty files and one real one.
    full = get_model_state_dict(model, options=StateDictOptions(
        full_state_dict=True, cpu_offload=True))
    has = k0 in full
    full_type = type(full[k0]).__name__ if has else "(this rank got nothing)"
    full_shape = tuple(full[k0].shape) if has else ()
    full_bytes = sum(v.numel() * v.element_size() for v in full.values())
    naive_bytes = sum(D.tensor_bytes(v) for v in naive.values())
    return {"naive_type": naive_type, "naive_shape": naive_shape,
            "naive_local": naive_local, "full_type": full_type,
            "full_shape": full_shape, "full_bytes": full_bytes,
            "naive_bytes": naive_bytes, "n_full_keys": len(full),
            "n_naive_keys": len(naive)}


def section_7():
    print("\n[7] two traps")
    early = D.launch(w_memory, 4, threads=2, args=("fsdp", True, True, True))
    late = D.launch(w_memory, 4, threads=2, args=("fsdp", True, True, False))
    record("traps", "optimizer built BEFORE fully_shard: Adam state per rank",
           D.fmt_bytes(early[0]["acct"]["optim"]) +
           f"  ({early[0]['n_optim_states']} tensors have state)")
    record("traps", "optimizer built AFTER fully_shard: Adam state per rank",
           D.fmt_bytes(late[0]["acct"]["optim"]) +
           f"  ({late[0]['n_optim_states']} tensors have state)")
    record("traps", "optimizer early: did the weights move? sum before -> after",
           f"{early[0]['w_before']:.6f} -> {early[0]['w_after']:.6f}",
           "no exception, no warning, and no training")
    record("traps", "optimizer late: sum before -> after",
           f"{late[0]['w_before']:.6f} -> {late[0]['w_after']:.6f}")
    record("traps", "final loss, optimizer early / late",
           f"{early[0]['losses'][-1]:.4f} / {late[0]['losses'][-1]:.4f}")

    sds = D.launch(w_statedict, 4, threads=2)
    sd = sds[0]
    record("traps", "model.state_dict() entry type", sd["naive_type"])
    record("traps", "  its .shape / its local shape",
           f"{sd['naive_shape']} / {sd['naive_local']}")
    record("traps", "  bytes actually on this rank", D.fmt_bytes(sd["naive_bytes"]),
           "torch.save() on this gives you a quarter of a model in a DTensor wrapper")
    record("traps", "get_model_state_dict(full_state_dict=True, cpu_offload=True)",
           f"rank 0: {sd['full_type']}, shape {sd['full_shape']}, "
           f"{D.fmt_bytes(sd['full_bytes'])}")
    record("traps", "  same call on rank 1: keys returned",
           f"{sds[1]['n_full_keys']} (rank 0 got {sd['n_full_keys']})",
           "with cpu_offload the full model is gathered onto rank 0 only")
    return early, late, sd


# ---------------------------------------------------------------------------

def figure(fb, rows, med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    names = [r[0] for r in rows]
    vals = [r[1] / 1e6 for r in rows]
    ax[0].barh(names, vals, color=["#7f8c8d", "#27ae60", "#7f8c8d", "#27ae60"])
    ax[0].set_xlabel("MB of params + grads + Adam, per rank")
    ax[0].set_title("Measured model state")
    for i, v in enumerate(vals):
        ax[0].text(v, i, f" {v:.0f}", va="center", fontsize=9)

    worlds = [1, 2, 4, 8, 16, 32]
    p, g, o = fb["params"], fb["grads"], fb["adam"]
    ax[1].plot(worlds, [(p + g + o / w) / 1e6 for w in worlds], "o-", label="ZeRO-1 (optim)")
    ax[1].plot(worlds, [(p + (g + o) / w) / 1e6 for w in worlds], "s-", label="ZeRO-2 (+grads)")
    ax[1].plot(worlds, [(p + g + o) / w / 1e6 for w in worlds], "^-",
               label="ZeRO-3 / FSDP (+params)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("ranks"); ax[1].set_ylabel("MB per rank")
    ax[1].set_title("Which stage keeps shrinking")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    n = ["DDP", "FSDP\nreshard", "FSDP\nkeep params"]
    v = [med["ddp"] * 1e3, med["fsdp"] * 1e3, med["fsdp_noreshard"] * 1e3]
    ax[2].bar(n, v, color=["#7f8c8d", "#2980b9", "#8e44ad"])
    ax[2].set_ylabel("ms for 6 steps, 4 ranks")
    ax[2].set_title("Memory is not free")
    for i, val in enumerate(v):
        ax[2].text(i, val, f"{val:.0f}", ha="center", va="bottom")

    fig.tight_layout()
    p_ = os.path.join(OUT, "fsdp.png")
    fig.savefig(p_, dpi=120)
    print(f"\n  figure -> {p_}")


def main():
    t0 = time.time()
    f2, ddp2, f4, ddp4, fb, rows = section_1_3()
    section_4()
    section_5(f2, ddp2, f4)
    med = section_6(fb)
    section_7()
    figure(fb, rows, med)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader(); w.writerows(FINDINGS)
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
