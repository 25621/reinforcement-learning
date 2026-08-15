"""Project 29 - DDP, and where a data-parallel step actually spends its time.

No usable multi-GPU here, so the replicas are processes on 12 CPU cores talking
gloo over loopback. Every mechanism being measured (the bucketed all-reduce, the
overlap with backward, the averaging that makes DDP correct) is the *same* code
path PyTorch runs on 8 H100s; only the transport underneath differs.

Sections
  A  the correctness invariant: 4 ranks x batch 32 == 1 process x batch 128
  B  step-time breakdown: compute, all-reduce, exposed communication
  C  gradient bucketing: bucket_cap_mb -> message count -> time
  D  overlap: DDP's bucketed all-reduce vs one flat all-reduce after backward
  E  the only knob that changes the comm:compute ratio of an MLP

Runtime: ~36 s.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "28-nccl-tests"))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from commlib import run_ranks  # noqa: E402
from model import fake_batch, make_model, param_bytes  # noqa: E402

THREADS = 2          # per rank, so 4 ranks (8 threads) still fit in 12 cores
STEPS = 12
findings: dict = {}


# ---------------------------------------------------------------- utilities

def hook_state(world):
    return {"time": 0.0, "bytes": 0, "count": 0, "world": world}


def timing_hook(state, bucket):
    """Same job as DDP's default hook (average the gradients) plus a stopwatch
    and a byte counter, which is the only way to see the messages from inside."""
    buf = bucket.buffer()
    t0 = time.perf_counter()
    fut = dist.all_reduce(buf, async_op=True).get_future()

    def done(f):
        state["time"] += time.perf_counter() - t0
        state["bytes"] += buf.numel() * buf.element_size()
        state["count"] += 1
        val = f.value()
        out = val[0] if isinstance(val, (list, tuple)) else val
        return out / state["world"]

    return fut.then(done)


def reset(state):
    """Zero the counters after warm-up. DDP rebuilds its buckets after the first
    iteration, so counting from step 0 mixes two different bucket layouts."""
    dist.barrier()
    state["time"], state["bytes"], state["count"] = 0.0, 0, 0


def train_steps(mod, opt, batch, steps, seed_off=0):
    """A plain training loop. Returns seconds per step (median of the steps
    after the first two, which pay for lazy allocation)."""
    times = []
    for s in range(steps):
        x, y = fake_batch(batch, seed=100 + s + seed_off)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(mod(x), y)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    times = sorted(times[2:])
    return times[len(times) // 2]


# ------------------------------------------------------------------ A

def _correctness(rank, world, batch_per_rank):
    torch.set_num_threads(THREADS)
    model = make_model()
    ddp = DDP(model)
    state = hook_state(world)
    ddp.register_comm_hook(state, timing_hook)

    x, y = fake_batch(batch_per_rank * world, seed=7)
    xs = x.chunk(world)[rank]
    ys = y.chunk(world)[rank]
    F.cross_entropy(ddp(xs), ys).backward()
    ddp_grad = torch.cat([p.grad.flatten() for p in ddp.module.parameters()])

    # the reference: one process, the whole batch, no communication at all
    ref = make_model()
    F.cross_entropy(ref(x), y).backward()
    ref_grad = torch.cat([p.grad.flatten() for p in ref.parameters()])

    return dict(
        max_abs_diff=float((ddp_grad - ref_grad).abs().max()),
        grad_norm=float(ref_grad.norm()),
        bytes_per_step=state["bytes"],
        messages_per_step=state["count"],
        param_bytes=param_bytes(ref),
    )


def section_a():
    res = {}
    for world in [2, 4]:
        res[world] = run_ranks(_correctness, world, 32, threads=THREADS)
        print(f"A: world={world} max|DDP-single| = {res[world]['max_abs_diff']:.3e} "
              f"(grad norm {res[world]['grad_norm']:.3f}), "
              f"{res[world]['messages_per_step']} messages, "
              f"{res[world]['bytes_per_step']/1e6:.2f} MB/step")
    findings["A_correctness"] = res


# ------------------------------------------------------------------ B

def _breakdown(rank, world, batch):
    torch.set_num_threads(THREADS)
    local = make_model()
    lopt = torch.optim.SGD(local.parameters(), lr=0.01)
    t_local = train_steps(local, lopt, batch, STEPS)

    model = make_model()
    ddp = DDP(model, bucket_cap_mb=25)
    state = hook_state(world)
    ddp.register_comm_hook(state, timing_hook)
    dopt = torch.optim.SGD(ddp.parameters(), lr=0.01)
    train_steps(ddp, dopt, batch, 3)
    reset(state)
    t_ddp = train_steps(ddp, dopt, batch, STEPS)

    return dict(local_s=t_local, ddp_s=t_ddp,
                comm_s=state["time"] / STEPS,
                bytes=state["bytes"] / STEPS,
                messages=state["count"] / STEPS)


def section_b():
    res = {}
    for world in [1, 2, 4]:
        r = run_ranks(_breakdown, world, 64, threads=THREADS)
        r["exposed_s"] = r["ddp_s"] - r["local_s"]
        r["exposed_frac"] = r["exposed_s"] / r["ddp_s"]
        r["hidden_s"] = max(r["comm_s"] - r["exposed_s"], 0.0)
        res[world] = r
        print(f"B: world={world} local={r['local_s']*1e3:7.2f} ms  ddp={r['ddp_s']*1e3:7.2f} ms  "
              f"in-allreduce={r['comm_s']*1e3:7.2f} ms  exposed={r['exposed_s']*1e3:6.2f} ms "
              f"({r['exposed_frac']*100:.1f}%)")
    findings["B_breakdown"] = res


# ------------------------------------------------------------------ C

def _buckets(rank, world, batch):
    torch.set_num_threads(THREADS)
    out = {}
    for cap in [0.1, 0.5, 2.0, 25.0]:
        model = make_model()
        ddp = DDP(model, bucket_cap_mb=cap)
        state = hook_state(world)
        ddp.register_comm_hook(state, timing_hook)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        train_steps(ddp, opt, batch, 3)
        reset(state)
        t = train_steps(ddp, opt, batch, STEPS)
        out[str(cap)] = dict(step_s=t, messages=state["count"] / STEPS,
                             comm_s=state["time"] / STEPS,
                             bytes=state["bytes"] / STEPS)
    return out


def section_c():
    res = run_ranks(_buckets, 4, 64, threads=THREADS)
    findings["C_bucketing"] = res
    for cap, r in res.items():
        print(f"C: cap={cap:>5s} MB  {r['messages']:5.1f} messages/step  "
              f"step={r['step_s']*1e3:7.2f} ms  in-allreduce={r['comm_s']*1e3:6.2f} ms")


# ------------------------------------------------------------------ D

def _overlap(rank, world, batch):
    torch.set_num_threads(THREADS)
    model = make_model()
    ddp = DDP(model, bucket_cap_mb=25)
    state = hook_state(world)
    ddp.register_comm_hook(state, timing_hook)
    dopt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    manual = make_model()
    mopt = torch.optim.SGD(manual.parameters(), lr=0.01)
    flat = torch.cat([p.detach().flatten() for p in manual.parameters()])

    def manual_step(s):
        x, y = fake_batch(batch, seed=100 + s)
        mopt.zero_grad(set_to_none=True)
        F.cross_entropy(manual(x), y).backward()
        # one flat message, issued only after the whole backward is finished
        off = 0
        for p in manual.parameters():
            n = p.numel()
            flat[off:off + n] = p.grad.flatten()
            off += n
        dist.all_reduce(flat)
        flat.div_(world)
        off = 0
        for p in manual.parameters():
            n = p.numel()
            p.grad.copy_(flat[off:off + n].view_as(p))
            off += n
        mopt.step()

    def ddp_step(s):
        x, y = fake_batch(batch, seed=100 + s)
        dopt.zero_grad(set_to_none=True)
        F.cross_entropy(ddp(x), y).backward()
        dopt.step()

    res = {}
    for name, fn in [("ddp_overlapped", ddp_step), ("manual_flat", manual_step)]:
        for s in range(2):
            fn(s)
        dist.barrier()
        ts = []
        for s in range(STEPS):
            t0 = time.perf_counter()
            fn(s)
            ts.append(time.perf_counter() - t0)
        ts = sorted(ts[2:])
        t = torch.tensor([ts[len(ts) // 2]])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        res[name] = float(t.item())
    res["ddp_messages"] = state["count"] / (STEPS + 2)
    return res


def section_d():
    res = {}
    for world in [2, 4]:
        r = run_ranks(_overlap, world, 64, threads=THREADS)
        r["speedup_of_overlap"] = r["manual_flat"] / r["ddp_overlapped"]
        res[world] = r
        print(f"D: world={world} ddp(overlapped, {r['ddp_messages']:.1f} msgs)="
              f"{r['ddp_overlapped']*1e3:7.2f} ms   manual(1 msg, no overlap)="
              f"{r['manual_flat']*1e3:7.2f} ms   ratio={r['speedup_of_overlap']:.2f}x")
    findings["D_overlap"] = res


# ------------------------------------------------------------------ E

def _batch_sweep(rank, world):
    torch.set_num_threads(THREADS)
    out = {}
    for batch in [8, 32, 128, 512]:
        model = make_model()
        ddp = DDP(model, bucket_cap_mb=25)
        state = hook_state(world)
        ddp.register_comm_hook(state, timing_hook)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        train_steps(ddp, opt, batch, 3)
        reset(state)
        t = train_steps(ddp, opt, batch, STEPS)

        local = make_model()
        lopt = torch.optim.SGD(local.parameters(), lr=0.01)
        tl = train_steps(local, lopt, batch, STEPS)
        out[str(batch)] = dict(step_s=t, local_s=tl, comm_s=state["time"] / STEPS,
                               bytes=state["bytes"] / STEPS,
                               samples_per_s=batch * world / t)
    return out


def section_e():
    res = run_ranks(_batch_sweep, 4, threads=THREADS)
    for b, r in res.items():
        r["comm_frac"] = (r["step_s"] - r["local_s"]) / r["step_s"]
        print(f"E: batch/rank={b:>4s}  compute={r['local_s']*1e3:7.2f} ms  "
              f"step={r['step_s']*1e3:7.2f} ms  exposed comm={r['comm_frac']*100:5.1f}%  "
              f"{r['samples_per_s']:8.0f} samples/s")
    findings["E_batch_sweep"] = res


# ------------------------------------------------------------------ plot

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    b = findings["B_breakdown"]
    ws = sorted(int(k) for k in b)
    comp = [b[w]["local_s"] * 1e3 for w in ws]
    exp = [max(b[w]["exposed_s"], 0) * 1e3 for w in ws]
    ax[0][0].bar([str(w) for w in ws], comp, label="compute")
    ax[0][0].bar([str(w) for w in ws], exp, bottom=comp, label="exposed comm")
    ax[0][0].set_xlabel("world size")
    ax[0][0].set_ylabel("ms / step")
    ax[0][0].set_title("A. a DDP step, split")
    ax[0][0].legend(fontsize=7)
    ax[0][0].grid(alpha=.3)

    c = findings["C_bucketing"]
    caps = sorted(c, key=float)
    ax[0][1].plot([c[k]["messages"] for k in caps], [c[k]["step_s"] * 1e3 for k in caps], "o-")
    for k in caps:
        ax[0][1].annotate(f"{k} MB", (c[k]["messages"], c[k]["step_s"] * 1e3), fontsize=7)
    ax[0][1].set_xlabel("all-reduce messages per step")
    ax[0][1].set_ylabel("ms / step")
    ax[0][1].set_title("B. bucket size -> message count -> time (world=4)")
    ax[0][1].grid(alpha=.3)

    d = findings["D_overlap"]
    ws = sorted(int(k) for k in d)
    w0 = [d[w]["ddp_overlapped"] * 1e3 for w in ws]
    w1 = [d[w]["manual_flat"] * 1e3 for w in ws]
    xs = range(len(ws))
    ax[1][0].bar([x - .2 for x in xs], w0, width=.4, label="DDP (overlapped)")
    ax[1][0].bar([x + .2 for x in xs], w1, width=.4, label="manual flat (not overlapped)")
    ax[1][0].set_xticks(list(xs))
    ax[1][0].set_xticklabels([str(w) for w in ws])
    ax[1][0].set_xlabel("world size")
    ax[1][0].set_ylabel("ms / step")
    ax[1][0].set_title("C. does overlapping pay?")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=.3)

    e = findings["E_batch_sweep"]
    bs = sorted(e, key=int)
    ax[1][1].plot([int(k) for k in bs], [e[k]["comm_frac"] * 100 for k in bs], "o-")
    ax[1][1].set_xscale("log", base=2)
    ax[1][1].set_xlabel("batch per rank")
    ax[1][1].set_ylabel("exposed comm (% of step)")
    ax[1][1].set_title("D. same model, same bytes: only the batch moves the ratio")
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "multi_gpu_ddp.png", dpi=120)


def main():
    t0 = time.perf_counter()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    plot()
    findings["runtime_s"] = time.perf_counter() - t0
    print(f"total runtime {findings['runtime_s']:.1f} s")
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    with open(OUT / "findings.csv", "w") as f:
        f.write("section,key,world,step_ms,comm_ms,messages,bytes\n")
        for w, r in findings["B_breakdown"].items():
            f.write(f"B,breakdown,{w},{r['ddp_s']*1e3:.3f},{r['comm_s']*1e3:.3f},"
                    f"{r['messages']:.2f},{r['bytes']:.0f}\n")
        for cap, r in findings["C_bucketing"].items():
            f.write(f"C,cap_{cap}MB,4,{r['step_s']*1e3:.3f},{r['comm_s']*1e3:.3f},"
                    f"{r['messages']:.2f},{r['bytes']:.0f}\n")
        for b, r in findings["E_batch_sweep"].items():
            f.write(f"E,batch_{b},4,{r['step_s']*1e3:.3f},{r['comm_s']*1e3:.3f},"
                    f"{r['messages'] if 'messages' in r else ''},{r['bytes']:.0f}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
