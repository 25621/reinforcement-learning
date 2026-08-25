"""Project 37 — Write the all-reduce yourself, then check DDP against it.

Run:  python3 run.py           (~4 minutes)

Sections
  1. hand-rolled DDP in 3 lines, checked against real DDP
  2. the two bugs you will actually write (forgot /world, forgot broadcast)
  3. ring all-reduce from send/recv, and the bytes it moves
  4. bucketing: 1 big collective beats 200 small ones
  5. overlap: starting the all-reduce during backward, and DDP's own numbers
  6. no_sync(): skipping the all-reduce on purpose
"""

from __future__ import annotations

import csv
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "36-two-gpu-ddp"))
import dist_lib as D  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<46} {value}")


# ---------------------------------------------------------------------------
# shared model + task
# ---------------------------------------------------------------------------

IN, HID, OUTD = 64, 128, 8
STEPS = 25


def build(seed=1234):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN, HID), nn.Tanh(),
                         nn.Linear(HID, HID), nn.Tanh(),
                         nn.Linear(HID, OUTD))


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def batches(rank, world, n_steps, per_rank=32):
    x, y = D.make_teacher_data(per_rank * world * n_steps, IN, OUTD, seed=5)
    for s in range(n_steps):
        lo = s * per_rank * world + rank * per_rank
        yield x[lo:lo + per_rank], y[lo:lo + per_rank]


# ---------------------------------------------------------------------------
# 1 + 2. hand-rolled DDP
# ---------------------------------------------------------------------------

def manual_sync_(model, world):
    """This is DDP, in three lines.

    all_reduce(SUM) leaves *every* rank holding the sum of all ranks' copies of
    the tensor, so after dividing by the world size every rank holds the same
    average gradient. Same gradient + same starting weights + same optimiser
    settings => the replicas can never drift apart.
    """
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


def w_train(rank, world, mode):
    """mode: 'ddp' | 'manual' | 'no_divide' | 'no_broadcast' | 'solo'"""
    seed = 1234 if mode != "no_broadcast" else 1234 + rank
    model = build(seed)
    if mode == "no_broadcast":
        # what DDP does for you at construction time, and we are skipping:
        pass
    net = DDP(model) if mode == "ddp" else model
    opt = torch.optim.SGD(net.parameters(), lr=0.2)

    losses = []
    for xb, yb in batches(rank, world, STEPS):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(xb), yb)
        loss.backward()
        if mode in ("manual", "no_broadcast"):
            manual_sync_(model, world)
        elif mode == "no_divide":
            for p in model.parameters():          # sum, never averaged
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        opt.step()
        losses.append(loss.detach().item())
    return {"w": flat(model), "losses": losses,
            "grad": torch.cat([p.grad.reshape(-1) for p in model.parameters()])}


def section_1_2():
    print("\n[1] three lines of all_reduce vs the real DDP")
    ddp = D.launch(w_train, 2, threads=3, args=("ddp",))
    man = D.launch(w_train, 2, threads=3, args=("manual",))
    solo = D.launch(w_train, 1, threads=3, args=("solo",))

    record("manual", "max |w(manual all-reduce) - w(DDP)| after 25 steps",
           f"{float((man[0]['w'] - ddp[0]['w']).abs().max()):.3e}")
    record("manual", "max |grad(manual) - grad(DDP)| at the last step",
           f"{float((man[0]['grad'] - ddp[0]['grad']).abs().max()):.3e}")
    record("manual", "final loss  DDP / manual",
           f"{ddp[0]['losses'][-1]:.4f} / {man[0]['losses'][-1]:.4f}")
    record("manual", "max |w(rank0) - w(rank1)| , manual all-reduce",
           f"{float((man[0]['w'] - man[1]['w']).abs().max()):.3e}")

    print("\n[2] the two bugs you will write")
    nod = D.launch(w_train, 2, threads=3, args=("no_divide",))
    nob = D.launch(w_train, 2, threads=3, args=("no_broadcast",))
    record("bugs", "forgot / world: max |w - w(DDP)|",
           f"{float((nod[0]['w'] - ddp[0]['w']).abs().max()):.3e}",
           "gradients are 2x too large = a silently doubled learning rate")
    record("bugs", "forgot / world: final loss vs DDP",
           f"{nod[0]['losses'][-1]:.4f} vs {ddp[0]['losses'][-1]:.4f}")
    record("bugs", "forgot the initial broadcast: max |w(rank0) - w(rank1)|",
           f"{float((nob[0]['w'] - nob[1]['w']).abs().max()):.3e}",
           "identical gradients, different starting points -> ranks never agree")
    record("bugs", "forgot the initial broadcast: final loss rank0 / rank1",
           f"{nob[0]['losses'][-1]:.4f} / {nob[1]['losses'][-1]:.4f}")
    return ddp[0]["losses"], solo[0]["losses"], man[0]["losses"], nod[0]["losses"]


# ---------------------------------------------------------------------------
# 3. ring all-reduce written with send/recv
# ---------------------------------------------------------------------------

def ring_all_reduce_(t, rank, world):
    """All-reduce a flat tensor using only point-to-point messages.

    Two phases, each of world-1 steps:

      reduce-scatter : chunk k travels around the ring accumulating, so that at
                       the end rank r owns the fully summed chunk r
      all-gather     : those finished chunks travel around the ring again so
                       every rank ends up with all of them

    Each rank sends (world-1)/world of the tensor twice, i.e. 2*(N-1)/N * D
    bytes -- almost 2D no matter how many ranks there are. A gather-to-rank-0
    then broadcast moves (N-1)*D bytes *through rank 0*, which is why the naive
    version gets worse and worse as the cluster grows.
    """
    chunks = list(t.chunk(world))
    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world
    moved = 0

    for step in range(world - 1):                       # reduce-scatter
        s_idx = (rank - step) % world
        r_idx = (rank - step - 1) % world
        recv_buf = torch.empty_like(chunks[r_idx])
        reqs = [dist.isend(chunks[s_idx].contiguous(), send_to),
                dist.irecv(recv_buf, recv_from)]
        for r in reqs:
            r.wait()
        chunks[r_idx] += recv_buf
        moved += chunks[s_idx].numel() * chunks[s_idx].element_size()

    for step in range(world - 1):                       # all-gather
        s_idx = (rank + 1 - step) % world
        r_idx = (rank - step) % world
        recv_buf = torch.empty_like(chunks[r_idx])
        reqs = [dist.isend(chunks[s_idx].contiguous(), send_to),
                dist.irecv(recv_buf, recv_from)]
        for r in reqs:
            r.wait()
        chunks[r_idx].copy_(recv_buf)
        moved += chunks[s_idx].numel() * chunks[s_idx].element_size()

    out = torch.cat(chunks)
    t.copy_(out)
    return moved


def w_ring(rank, world, numel):
    torch.manual_seed(rank)
    ref = torch.randn(numel)

    mine = ref.clone()
    moved = ring_all_reduce_(mine, rank, world)

    theirs = ref.clone()
    dist.all_reduce(theirs)

    def time_it(fn, n=5):
        fn()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dist.barrier()
        return (time.perf_counter() - t0) / n

    buf = ref.clone()
    t_ring = time_it(lambda: ring_all_reduce_(buf, rank, world))
    buf2 = ref.clone()
    t_builtin = time_it(lambda: dist.all_reduce(buf2))

    naive_bytes = (world - 1) * numel * 4 * 2      # gather to root + broadcast back
    return {"err": float((mine - theirs).abs().max()),
            "ring_bytes": moved, "naive_bytes": naive_bytes,
            "t_ring": t_ring, "t_builtin": t_builtin}


def section_3():
    print("\n[3] ring all-reduce from send/recv")
    numel = 1 << 20                                  # 4 MB of float32
    for world in (2, 4):
        res = D.launch(w_ring, world, threads=2, args=(numel,))
        r0 = res[0]
        record("ring", f"world={world}: max error vs dist.all_reduce", f"{r0['err']:.3e}")
        record("ring", f"world={world}: bytes sent per rank, ring",
               D.fmt_bytes(r0["ring_bytes"]) +
               f"  = 2*(N-1)/N * D = {2 * (world - 1) / world:.2f} D")
        record("ring", f"world={world}: bytes through the root, gather+broadcast",
               D.fmt_bytes(r0["naive_bytes"]))
        record("ring", f"world={world}: our python ring / built-in all_reduce",
               f"{r0['t_ring'] * 1e3:.1f} ms / {r0['t_builtin'] * 1e3:.1f} ms"
               f"   ({r0['t_ring'] / r0['t_builtin']:.1f}x slower)")
    return numel


# ---------------------------------------------------------------------------
# 4. bucketing
# ---------------------------------------------------------------------------

def w_buckets(rank, world, n_tensors, total_numel):
    each = total_numel // n_tensors
    tensors = [torch.randn(each) for _ in range(n_tensors)]

    def one_by_one():
        for t in tensors:
            dist.all_reduce(t)

    flat_buf = torch.zeros(total_numel)

    def bucketed():
        torch._foreach_copy_(list(flat_buf.split(each)), tensors)
        dist.all_reduce(flat_buf)
        torch._foreach_copy_(tensors, list(flat_buf.split(each)))

    res = D.interleaved({"one_by_one": one_by_one, "bucketed": bucketed},
                        rounds=5, warmup=1)
    return {"one": res["one_by_one"]["min"], "bucket": res["bucketed"]["min"],
            "n_calls_one": n_tensors, "n_calls_bucket": 1}


def section_4():
    print("\n[4] many small collectives vs one big one (4 MB total either way)")
    total = 1 << 20
    rows = []
    for n_tensors in (1, 8, 64, 512):
        res = D.launch(w_buckets, 2, threads=3, args=(n_tensors, total))[0]
        speed = res["one"] / res["bucket"]
        rows.append((n_tensors, res["one"] * 1e3, res["bucket"] * 1e3, speed))
        record("bucket", f"{n_tensors:>4} tensors: one-by-one / bucketed",
               f"{res['one'] * 1e3:7.2f} ms / {res['bucket'] * 1e3:6.2f} ms"
               f"   ({speed:.2f}x)")
    return rows


# ---------------------------------------------------------------------------
# 5. overlap, and DDP's own bucket report
# ---------------------------------------------------------------------------

class OverlappedDDP:
    """Start each gradient's all-reduce the moment autograd produces it.

    `register_post_accumulate_grad_hook` fires per parameter during the backward
    pass, not after it. The async all_reduce (`async_op=True`) returns a handle
    immediately, so the communication for layer L overlaps with the backward
    computation of layer L-1. We only block at `finish()`.
    """

    def __init__(self, module, world):
        self.module = module
        self.world = world
        self.handles = []
        for p in module.parameters():
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        h = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((h, p))

    def finish(self):
        for h, p in self.handles:
            h.wait()
            p.grad /= self.world
        self.handles.clear()


def big_model(seed=7):
    torch.manual_seed(seed)
    layers = []
    for _ in range(24):
        layers += [nn.Linear(256, 256), nn.Tanh()]
    layers.append(nn.Linear(256, OUTD))
    return nn.Sequential(nn.Linear(IN, 256), nn.Tanh(), *layers)


def w_overlap(rank, world):
    x, y = D.make_teacher_data(64 * 12, IN, OUTD, seed=3)

    def run(sync):
        model = big_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        over = OverlappedDDP(model, world) if sync == "overlap" else None
        ddp = DDP(model) if sync == "ddp" else None
        net = ddp if ddp is not None else model

        def step():
            for s in range(8):
                xb, yb = x[s * 64:(s + 1) * 64], y[s * 64:(s + 1) * 64]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(net(xb), yb).backward()
                if sync == "after":
                    manual_sync_(model, world)
                elif sync == "overlap":
                    over.finish()
                opt.step()
        return step, ddp

    step_after, _ = run("after")
    step_over, _ = run("overlap")
    step_ddp, ddp = run("ddp")

    res = D.interleaved({"after": step_after, "overlap": step_over, "ddp": step_ddp},
                        rounds=4, warmup=1)
    log = {}
    if ddp is not None:
        d = ddp._get_ddp_logging_data()
        log = {k: d[k] for k in
               ("bucket_cap_bytes", "num_buckets_created" if
                "num_buckets_created" in d else "bucket_cap_bytes")
               if k in d}
        log["avg_backward_compute_time"] = d.get("avg_backward_compute_time", 0)
        log["avg_backward_comm_time"] = d.get("avg_backward_comm_time", 0)
        log["avg_backward_compute_comm_overlap_time"] = d.get(
            "avg_backward_compute_comm_overlap_time", 0)
    n_params = sum(p.numel() for p in big_model().parameters())
    return {"t": {k: v["min"] for k, v in res.items()}, "log": log,
            "n_params": n_params,
            "n_tensors": len(list(big_model().parameters()))}


def section_5():
    print("\n[5] overlapping communication with the backward pass")
    r = D.launch(w_overlap, 2, threads=3)[0]
    t = r["t"]
    record("overlap", "model", f"{r['n_params']:,} params in {r['n_tensors']} tensors")
    record("overlap", "all-reduce AFTER backward (8 steps)", f"{t['after'] * 1e3:.0f} ms")
    record("overlap", "all-reduce DURING backward (hooks)", f"{t['overlap'] * 1e3:.0f} ms"
           f"   ({t['after'] / t['overlap']:.2f}x)")
    record("overlap", "real DDP (buckets + overlap, C++)", f"{t['ddp'] * 1e3:.0f} ms"
           f"   ({t['after'] / t['ddp']:.2f}x)")
    for k, v in r["log"].items():
        if "time" in k and v:
            record("overlap", f"DDP reports {k}", f"{v / 1e6:.1f} ms")
        else:
            record("overlap", f"DDP reports {k}", v)
    return t


# ---------------------------------------------------------------------------
# 6. no_sync
# ---------------------------------------------------------------------------

def w_nosync(rank, world):
    x, y = D.make_teacher_data(32 * 16, IN, OUTD, seed=13)
    calls = {"n": 0}
    real = dist.all_reduce

    def counting(tensor, *a, **kw):
        calls["n"] += 1
        return real(tensor, *a, **kw)

    dist.all_reduce = counting
    try:
        out = {}
        for use_nosync in (False, True):
            calls["n"] = 0
            model = build()
            net = model
            opt = torch.optim.SGD(net.parameters(), lr=0.05)
            accum = 4
            for s in range(16):
                xb, yb = x[s * 32:(s + 1) * 32], y[s * 32:(s + 1) * 32]
                loss = F.cross_entropy(net(xb), yb) / accum
                loss.backward()
                last_micro = (s % accum) == accum - 1
                if last_micro or not use_nosync:
                    manual_sync_(model, world)
                if last_micro:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            out["nosync" if use_nosync else "every_step"] = calls["n"]
            out[("w_nosync" if use_nosync else "w_every")] = flat(model)
    finally:
        dist.all_reduce = real
    return out


def section_6():
    print("\n[6] gradient accumulation: no_sync() skips the collectives you do not need")
    r = D.launch(w_nosync, 2, threads=3)[0]
    record("nosync", "all_reduce calls, syncing every micro-batch", r["every_step"])
    record("nosync", "all_reduce calls, syncing once per optimiser step", r["nosync"])
    record("nosync", "max |w(sync every) - w(sync once)|",
           f"{float((r['w_every'] - r['w_nosync']).abs().max()):.3e}",
           "averaging then summing == summing then averaging")
    return r


# ---------------------------------------------------------------------------

def figure(losses, bucket_rows, overlap_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ddp_l, solo_l, man_l, nod_l = losses
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(ddp_l, lw=3, alpha=0.6, label="real DDP")
    ax[0].plot(man_l, "--", label="3-line manual all-reduce")
    ax[0].plot(nod_l, ":", color="#c0392b", label="forgot / world_size")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].set_title("Manual all-reduce IS DDP")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    n = [r[0] for r in bucket_rows]
    ax[1].plot(n, [r[1] for r in bucket_rows], "o-", label="one all_reduce per tensor")
    ax[1].plot(n, [r[2] for r in bucket_rows], "s-", label="one all_reduce, flat bucket")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("number of tensors (4 MB total)")
    ax[1].set_ylabel("ms per all-reduce round")
    ax[1].set_title("Why DDP buckets")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    names = ["all-reduce\nafter backward", "all-reduce\nduring backward", "real DDP\n(C++)"]
    vals = [overlap_t["after"] * 1e3, overlap_t["overlap"] * 1e3, overlap_t["ddp"] * 1e3]
    ax[2].bar(names, vals, color=["#7f8c8d", "#2980b9", "#27ae60"])
    ax[2].set_ylabel("ms for 8 steps")
    ax[2].set_title("Overlap")
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.0f}", ha="center", va="bottom")

    fig.tight_layout()
    p = os.path.join(OUT, "allreduce.png")
    fig.savefig(p, dpi=120)
    print(f"\n  figure -> {p}")


def main():
    t0 = time.time()
    ddp_l, solo_l, man_l, nod_l = section_1_2()
    section_3()
    rows = section_4()
    ot = section_5()
    section_6()
    figure((ddp_l, solo_l, man_l, nod_l), rows, ot)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader(); w.writerows(FINDINGS)
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
