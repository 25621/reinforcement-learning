"""Project 30 - FSDP: what sharding buys, and what it charges for it.

FSDP2 (`fully_shard`) refuses to be interesting on one device, so this runs
2/4/6 CPU processes over gloo. The sharding, the all-gathers and the
reduce-scatter are the real FSDP code path; only the wire is different.

Sections
  A  sharding, verified: every rank physically holds 1/n of every parameter
  B  the bill: FSDP moves 1.5x DDP's bytes to save (n-1)/n of the memory
  C  measured step time, world 2/4/6 -- memory is bought with time
  D  reshard_after_forward: one all-gather fewer, or 1/n of the memory
  E  the arithmetic that decides this for a real 7B model

Runtime: ~46 s.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # FSDP2 would pick the unusable GPU

import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "28-nccl-tests"))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from commlib import run_ranks  # noqa: E402

D, LAYERS, BATCH, STEPS = 768, 8, 64, 8
THREADS = 2
findings: dict = {}


# ---------------------------------------------------------------- model

def build(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    layers = []
    for _ in range(LAYERS):
        layers += [nn.Linear(D, D), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(D, 10))


def batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, D, generator=g), torch.randint(0, 10, (BATCH,), generator=g)


# ------------------------------------------------- collective byte counter

class Counter:
    """FSDP's collectives go through these two functions. Wrapping them is the
    cheapest honest way to see how many bytes a step really moves -- the
    alternative is reading the source and trusting the arithmetic."""

    def __init__(self):
        self.reset()
        self._ag = dist.all_gather_into_tensor
        self._rs = dist.reduce_scatter_tensor
        self._ar = dist.all_reduce

    def reset(self):
        self.n = {"all_gather": 0, "reduce_scatter": 0, "all_reduce": 0}
        self.b = {"all_gather": 0, "reduce_scatter": 0, "all_reduce": 0}

    def _note(self, kind, t):
        self.n[kind] += 1
        self.b[kind] += t.numel() * t.element_size()

    def __enter__(self):
        c = self

        def ag(*a, **k):
            out = k.get("output", a[0] if a else None)
            c._note("all_gather", out)
            return c._ag(*a, **k)

        def rs(*a, **k):
            inp = k.get("input", a[1] if len(a) > 1 else None)
            c._note("reduce_scatter", inp)
            return c._rs(*a, **k)

        def ar(*a, **k):
            t = k.get("tensor", a[0] if a else None)
            c._note("all_reduce", t)
            return c._ar(*a, **k)

        dist.all_gather_into_tensor = ag
        dist.reduce_scatter_tensor = rs
        dist.all_reduce = ar
        return self

    def __exit__(self, *exc):
        dist.all_gather_into_tensor = self._ag
        dist.reduce_scatter_tensor = self._rs
        dist.all_reduce = self._ar


def ddp_counter(world):
    """DDP's gradient all-reduce is issued from C++, so wrapping the Python
    `dist.all_reduce` does not see it. A comm hook does -- it is the documented
    place to intercept exactly that message."""
    state = {"n": 0, "b": 0, "world": world}

    def hook(st, bucket):
        buf = bucket.buffer()
        st["n"] += 1
        st["b"] += buf.numel() * buf.element_size()
        fut = dist.all_reduce(buf, async_op=True).get_future()

        def done(f):
            val = f.value()
            out = val[0] if isinstance(val, (list, tuple)) else val
            return out / st["world"]

        return fut.then(done)

    return state, hook


# ---------------------------------------------------------------- workers

def _shard(model, mesh, reshard_after_forward=True):
    from torch.distributed.fsdp import fully_shard
    for layer in model:
        if list(layer.parameters()):
            fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


def _mesh(world):
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh("cpu", (world,))


def local_bytes(model) -> int:
    tot = 0
    for p in model.parameters():
        t = p.to_local() if hasattr(p, "to_local") else p
        tot += t.numel() * t.element_size()
    return tot


def run_steps(model, opt, steps, sync=True):
    """`sync=False` skips the max-over-ranks all-reduce, which would otherwise
    show up inside the byte counter as traffic this project did not intend."""
    times = []
    for s in range(steps):
        x, y = batch(s)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    times = sorted(times[max(len(times) - 3, 1):]) if len(times) < 5 else sorted(times[2:])
    med = times[len(times) // 2]
    if not sync:
        return med
    t = torch.tensor([med])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


# ------------------------------------------------------------------ A + B

def _shard_and_count(rank, world):
    torch.set_num_threads(THREADS)
    mesh = _mesh(world)

    full = build()
    global_bytes = sum(p.numel() * p.element_size() for p in full.parameters())

    fsdp = _shard(build(), mesh)
    fsdp_local = local_bytes(fsdp)
    fopt = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)

    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp = DDP(build())
    dstate, dhook = ddp_counter(world)
    ddp.register_comm_hook(dstate, dhook)
    ddp_local = local_bytes(ddp)
    dopt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    # warm up (first step allocates and, for DDP, rebuilds buckets)
    run_steps(fsdp, fopt, 3)
    run_steps(ddp, dopt, 3)

    with Counter() as cf:
        run_steps(fsdp, fopt, 3, sync=False)
        fsdp_traffic = dict(n=dict(cf.n), b=dict(cf.b))
    dist.barrier()
    dstate["n"], dstate["b"] = 0, 0
    run_steps(ddp, dopt, 3, sync=False)
    dist.barrier()
    ddp_traffic = dict(n={"all_reduce": dstate["n"], "all_gather": 0, "reduce_scatter": 0},
                       b={"all_reduce": dstate["b"], "all_gather": 0, "reduce_scatter": 0})

    # Optimiser state is sharded too: Adam keeps 2 moments per element, and
    # under FSDP those moments only exist for the elements this rank owns.
    # NOTE: a DTensor reports the *global* numel, so ask for the local shard.
    def state_bytes(opt):
        tot = 0
        for st in opt.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.dim() > 0:
                    t = v.to_local() if hasattr(v, "to_local") else v
                    tot += t.numel() * t.element_size()
        return tot

    fopt_state = state_bytes(fopt)
    dopt_state = state_bytes(dopt)

    return dict(global_param_bytes=global_bytes,
                fsdp_local_param_bytes=fsdp_local, ddp_local_param_bytes=ddp_local,
                fsdp_opt_state_bytes=fopt_state, ddp_opt_state_bytes=dopt_state,
                fsdp_traffic={k: {kk: vv / 3 for kk, vv in v.items()} for k, v in fsdp_traffic.items()},
                ddp_traffic={k: {kk: vv / 3 for kk, vv in v.items()} for k, v in ddp_traffic.items()})


def section_ab():
    res = {}
    for world in [2, 4]:
        r = run_ranks(_shard_and_count, world, threads=THREADS)
        r["shard_ratio"] = r["fsdp_local_param_bytes"] / r["global_param_bytes"]
        fb = r["fsdp_traffic"]["b"]
        db = r["ddp_traffic"]["b"]
        r["fsdp_user_bytes"] = fb["all_gather"] + fb["reduce_scatter"]
        r["ddp_user_bytes"] = db["all_reduce"]
        # wire bytes: ring collectives move (n-1)/n per participant, twice for all-reduce
        r["fsdp_wire_bytes"] = ((fb["all_gather"] + fb["reduce_scatter"]) * (world - 1) / world)
        r["ddp_wire_bytes"] = db["all_reduce"] * 2 * (world - 1) / world
        r["wire_ratio"] = r["fsdp_wire_bytes"] / r["ddp_wire_bytes"]
        res[world] = r
        print(f"A: world={world} params/rank FSDP={r['fsdp_local_param_bytes']/1e6:.2f} MB "
              f"DDP={r['ddp_local_param_bytes']/1e6:.2f} MB "
              f"(1/{1/r['shard_ratio']:.1f} of the model), "
              f"Adam state {r['fsdp_opt_state_bytes']/1e6:.2f} vs {r['ddp_opt_state_bytes']/1e6:.2f} MB")
        print(f"B: world={world} FSDP {r['fsdp_traffic']['n']} bytes={r['fsdp_user_bytes']/1e6:.2f} MB "
              f"| DDP {r['ddp_traffic']['n']} bytes={r['ddp_user_bytes']/1e6:.2f} MB "
              f"| wire ratio {r['wire_ratio']:.2f}x")
    findings["AB_shard_and_traffic"] = res


# ------------------------------------------------------------------ C

def _timing(rank, world):
    torch.set_num_threads(THREADS)
    mesh = _mesh(world)

    plain = build()
    popt = torch.optim.AdamW(plain.parameters(), lr=1e-3)
    fsdp = _shard(build(), mesh)
    fopt = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp = DDP(build())
    dopt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    for m, o in [(plain, popt), (fsdp, fopt), (ddp, dopt)]:
        run_steps(m, o, 3)

    out = {}
    for name, m, o in [("local", plain, popt), ("ddp", ddp, dopt), ("fsdp", fsdp, fopt)]:
        out[name] = run_steps(m, o, STEPS)
    out["params_per_rank_MB"] = local_bytes(fsdp) / 1e6
    return out


def section_c():
    res = {}
    for world in [2, 4, 6]:
        r = run_ranks(_timing, world, threads=THREADS)
        r["fsdp_over_ddp"] = r["fsdp"] / r["ddp"]
        res[world] = r
        print(f"C: world={world} local={r['local']*1e3:7.2f} ms  ddp={r['ddp']*1e3:7.2f} ms  "
              f"fsdp={r['fsdp']*1e3:7.2f} ms  ({r['fsdp_over_ddp']:.2f}x DDP, "
              f"{r['params_per_rank_MB']:.2f} MB params/rank)")
    findings["C_timing"] = res


# ------------------------------------------------------------------ D

def _reshard(rank, world):
    torch.set_num_threads(THREADS)
    mesh = _mesh(world)
    out = {}
    for flag in [True, False]:
        m = _shard(build(), mesh, reshard_after_forward=flag)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        run_steps(m, o, 3)
        with Counter() as c:
            run_steps(m, o, 3, sync=False)
            traffic = dict(n=dict(c.n), b=dict(c.b))
        t = run_steps(m, o, STEPS)
        out[str(flag)] = dict(step_s=t,
                              all_gathers=traffic["n"]["all_gather"] / 3,
                              ag_bytes=traffic["b"]["all_gather"] / 3,
                              rs_bytes=traffic["b"]["reduce_scatter"] / 3)
    return out


def section_d():
    res = run_ranks(_reshard, 4, threads=THREADS)
    a, b = res["True"], res["False"]
    res["ag_saved"] = a["ag_bytes"] - b["ag_bytes"]
    res["speedup"] = a["step_s"] / b["step_s"]
    findings["D_reshard"] = res
    for k in ["True", "False"]:
        r = res[k]
        print(f"D: reshard_after_forward={k:5s} {r['all_gathers']:4.1f} all-gathers/step "
              f"({r['ag_bytes']/1e6:6.2f} MB)  step={r['step_s']*1e3:7.2f} ms")
    print(f"D: keeping the parameters gathered is {res['speedup']:.2f}x, "
          f"and saves {res['ag_saved']/1e6:.2f} MB/step of all-gather")


# ------------------------------------------------------------------ E

def section_e():
    """Pure arithmetic for a model nobody here can hold: 7B params, AdamW,
    bf16 params + fp32 optimizer states (the standard mixed-precision recipe)."""
    P, BLOCKS = 7e9, 32
    per_param = dict(param_bf16=2, grad_bf16=2, adam_m_fp32=4, adam_v_fp32=4, master_fp32=4)
    total = sum(per_param.values()) * P
    # FSDP still has to materialise one whole block at a time, unsharded, in bf16
    transient = 2 * P / BLOCKS / 1e9
    rows = []
    for n in [1, 2, 4, 8, 16, 64]:
        ddp_gb = total / 1e9
        fsdp_gb = total / n / 1e9 + transient
        rows.append(dict(world=n, ddp_GB=ddp_gb, fsdp_GB=fsdp_gb,
                         fits_80GB_ddp=ddp_gb <= 80, fits_80GB_fsdp=fsdp_gb <= 80,
                         wire_ratio=1.5))
    findings["E_7b_arithmetic"] = dict(bytes_per_param=per_param, params=P,
                                       blocks=BLOCKS, transient_GB=transient, rows=rows)
    for r in rows:
        print(f"E: n={r['world']:3d}  DDP {r['ddp_GB']:7.1f} GB/GPU ({'fits' if r['fits_80GB_ddp'] else 'NO'})   "
              f"FSDP {r['fsdp_GB']:6.1f} GB/GPU ({'fits' if r['fits_80GB_fsdp'] else 'NO'})")


# ------------------------------------------------------------------ plot

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    c = findings["C_timing"]
    ws = sorted(int(k) for k in c)
    for name in ["local", "ddp", "fsdp"]:
        ax[0][0].plot(ws, [c[w][name] * 1e3 for w in ws], "o-", label=name)
    ax[0][0].set_xlabel("world size")
    ax[0][0].set_ylabel("ms / step")
    ax[0][0].set_title("A. step time: FSDP pays for the memory it saves")
    ax[0][0].legend(fontsize=7)
    ax[0][0].grid(alpha=.3)

    ax[0][1].plot(ws, [c[w]["params_per_rank_MB"] for w in ws], "o-")
    ax[0][1].set_xlabel("world size")
    ax[0][1].set_ylabel("MB of parameters per rank")
    ax[0][1].set_title("B. measured shard size = P/n exactly")
    ax[0][1].grid(alpha=.3)

    ab = findings["AB_shard_and_traffic"]
    ws2 = sorted(int(k) for k in ab)
    xs = range(len(ws2))
    ax[1][0].bar([x - .2 for x in xs], [ab[w]["ddp_wire_bytes"] / 1e6 for w in ws2],
                 width=.4, label="DDP")
    ax[1][0].bar([x + .2 for x in xs], [ab[w]["fsdp_wire_bytes"] / 1e6 for w in ws2],
                 width=.4, label="FSDP")
    ax[1][0].set_xticks(list(xs))
    ax[1][0].set_xticklabels([str(w) for w in ws2])
    ax[1][0].set_xlabel("world size")
    ax[1][0].set_ylabel("MB on the wire per rank per step")
    ax[1][0].set_title("C. FSDP's 1.5x traffic")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=.3)

    rows = findings["E_7b_arithmetic"]["rows"]
    ax[1][1].plot([r["world"] for r in rows], [r["ddp_GB"] for r in rows], "o-", label="DDP")
    ax[1][1].plot([r["world"] for r in rows], [r["fsdp_GB"] for r in rows], "s-", label="FSDP")
    ax[1][1].axhline(80, ls="--", color="red", lw=1, label="80 GB HBM")
    ax[1][1].set_xscale("log", base=2)
    ax[1][1].set_yscale("log")
    ax[1][1].set_xlabel("GPUs")
    ax[1][1].set_ylabel("GB per GPU")
    ax[1][1].set_title("D. 7B + AdamW: where each strategy fits")
    ax[1][1].legend(fontsize=7)
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "fsdp_scaling.png", dpi=120)


def main():
    t0 = time.perf_counter()
    section_ab()
    section_c()
    section_d()
    section_e()
    plot()
    findings["runtime_s"] = time.perf_counter() - t0
    print(f"total runtime {findings['runtime_s']:.1f} s")
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    with open(OUT / "findings.csv", "w") as f:
        f.write("section,world,metric,value\n")
        for w, r in findings["C_timing"].items():
            for k in ["local", "ddp", "fsdp", "params_per_rank_MB"]:
                f.write(f"C,{w},{k},{r[k]}\n")
        for w, r in findings["AB_shard_and_traffic"].items():
            for k in ["fsdp_wire_bytes", "ddp_wire_bytes", "wire_ratio", "shard_ratio"]:
                f.write(f"AB,{w},{k},{r[k]}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
