"""Project 28 - a faithful nccl-tests clone on the only transport this machine has.

Sections
  A  can NCCL run here at all? (measured refusal)
  B  the nccl-tests sweep: 4 collectives x 5 message sizes x 3 world sizes
  C  latency/bandwidth (alpha-beta) fit and the crossover message size
  D  three hand-written all-reduce algorithms vs gloo's built-in
  E  why nccl-tests prints busbw and not algbw

Runtime: ~35 s on 12 CPU cores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from collectives import ALGORITHMS, steps_and_bytes  # noqa: E402
from commlib import (alpha_beta, bandwidths, bus_factor, reps_for,  # noqa: E402
                     run_ranks, timed, timed_many)

SIZES = [1 << 12, 1 << 16, 1 << 20, 1 << 23, 1 << 24]  # 4 KiB .. 16 MiB
LAT_SIZES = [1 << 2, 1 << 8, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 23]
WORLDS = [2, 4, 6]

findings: dict = {}


# ------------------------------------------------------------------ A

def section_a():
    probe = (
        "import os,torch,torch.distributed as dist;"
        "os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']='29876';"
        "dist.init_process_group('nccl',rank=0,world_size=1);"
        "x=torch.ones(16,device='cuda');dist.all_reduce(x);torch.cuda.synchronize();"
        "print('NCCL_OK')"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=180)
    ok = "NCCL_OK" in r.stdout
    err = ""
    if not ok:
        for line in (r.stderr or "").splitlines():
            if "Error" in line or "error" in line:
                err = line.strip()
                break
    a = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
        "torch_supported_arch": torch.cuda.get_arch_list(),
        "nccl_built_in": dist.is_nccl_available(),
        "nccl_runs": ok,
        "nccl_error": err[:200],
        "fallback": "gloo over TCP loopback",
    }
    findings["A_nccl_probe"] = a
    print("A:", json.dumps(a)[:300])


# ------------------------------------------------------------------ B

def _sweep(rank, world):
    out = []
    for nbytes in SIZES:
        n = nbytes // 4
        reps = reps_for(nbytes)
        x = torch.randn(n)
        t = timed(lambda: dist.all_reduce(x), reps, rounds=3)
        out.append(("all_reduce", nbytes, t))

        src = torch.randn(n // world)
        dst = [torch.empty(n // world) for _ in range(world)]
        t = timed(lambda: dist.all_gather(dst, src), reps, rounds=3)
        out.append(("all_gather", nbytes, t))

        parts = [torch.randn(n // world) for _ in range(world)]
        o = torch.empty(n // world)
        t = timed(lambda: dist.reduce_scatter(o, parts), reps, rounds=3)
        out.append(("reduce_scatter", nbytes, t))

        b = torch.randn(n)
        t = timed(lambda: dist.broadcast(b, 0), reps, rounds=3)
        out.append(("broadcast", nbytes, t))
    return out


def section_b():
    rows = []
    for world in WORLDS:
        for coll, nbytes, t in _sweep_all(world):
            alg, bus = bandwidths(coll, nbytes, t, world)
            rows.append(
                dict(world=world, collective=coll, bytes=nbytes, us=t * 1e6,
                     algbw=alg, busbw=bus, factor=bus_factor(coll, world))
            )
    findings["B_sweep"] = rows
    for r in rows:
        if r["bytes"] == 1 << 24:
            print(f"B: world={r['world']} {r['collective']:15s} {r['us']/1000:8.2f} ms "
                  f"algbw={r['algbw']:.3f} busbw={r['busbw']:.3f} GB/s")


def _sweep_all(world):
    return run_ranks(_sweep, world)


# ------------------------------------------------------------------ C

def _latency(rank, world):
    out = []
    for nbytes in LAT_SIZES:
        x = torch.randn(max(nbytes // 4, world))
        t = timed(lambda: dist.all_reduce(x), reps_for(nbytes), rounds=5)
        out.append((nbytes, t))
    return out


def section_c():
    res = {}
    for world in WORLDS:
        pts = _latency_all(world)
        alpha, bw = alpha_beta([p[0] for p in pts], [p[1] for p in pts])
        # crossover: message size where the transfer term equals the fixed cost
        res[world] = dict(points=pts, alpha_us=alpha * 1e6, bw_GBs=bw / 1e9,
                          crossover_bytes=alpha * bw)
        print(f"C: world={world} alpha={alpha*1e6:8.1f} us  bw={bw/1e9:.3f} GB/s  "
              f"crossover={alpha*bw/1024:.1f} KiB")
    findings["C_alpha_beta"] = res


def _latency_all(world):
    return run_ranks(_latency, world)


# ------------------------------------------------------------------ D

def _algs(rank, world):
    out = []
    for nbytes in [1 << 12, 1 << 16, 1 << 20, 1 << 23]:
        n = nbytes // 4
        ref = torch.randn(n)
        gold = ref.clone()
        dist.all_reduce(gold)

        errs, ops = {}, {}
        for name, fn in ALGORITHMS.items():
            if name == "recursive_doubling" and (world & (world - 1)):
                continue
            x = ref.clone()
            fn(x, rank, world)
            errs[name] = float((x - gold).abs().max())
            ops[name] = (lambda f=fn: f(ref.clone(), rank, world))
        errs["gloo_builtin"] = 0.0
        ops["gloo_builtin"] = lambda: dist.all_reduce(ref.clone())

        reps = max(reps_for(nbytes) // 3, 2)
        for name, t in timed_many(ops, reps, rounds=3).items():
            out.append((name, nbytes, t, errs[name]))
    return out


def section_d():
    rows = []
    for world in [2, 4]:
        for name, nbytes, t, err in run_ranks(_algs, world):
            steps, sent = steps_and_bytes(name, world, nbytes) if name in ALGORITHMS else (None, None)
            rows.append(dict(world=world, alg=name, bytes=nbytes, us=t * 1e6,
                             max_err=err, steps=steps, bytes_sent=sent))
    findings["D_algorithms"] = rows
    for r in rows:
        if r["world"] == 4 and r["bytes"] in (1 << 12, 1 << 23):
            print(f"D: world=4 {r['alg']:20s} {r['bytes']:>9d} B  {r['us']:9.1f} us  err={r['max_err']:.1e}")


# ------------------------------------------------------------------ E

def section_e():
    rows = [r for r in findings["B_sweep"] if r["collective"] == "all_reduce" and r["bytes"] == 1 << 24]
    alg = {r["world"]: r["algbw"] for r in rows}
    bus = {r["world"]: r["busbw"] for r in rows}
    spread = lambda d: max(d.values()) / min(d.values())
    findings["E_busbw_invariance"] = dict(
        algbw=alg, busbw=bus, algbw_spread=spread(alg), busbw_spread=spread(bus)
    )
    print(f"E: 16 MiB all-reduce  algbw spread over worlds {spread(alg):.2f}x, "
          f"busbw spread {spread(bus):.2f}x")


# ------------------------------------------------------------------ plot

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    rows = findings["B_sweep"]
    for coll in ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]:
        xs = [r["bytes"] for r in rows if r["collective"] == coll and r["world"] == 4]
        ys = [r["busbw"] for r in rows if r["collective"] == coll and r["world"] == 4]
        ax[0][0].plot(xs, ys, "o-", label=coll)
    ax[0][0].set_xscale("log", base=2)
    ax[0][0].set_xlabel("message bytes")
    ax[0][0].set_ylabel("busbw (GB/s)")
    ax[0][0].set_title("A. bus bandwidth vs message size (world=4)")
    ax[0][0].legend(fontsize=7)
    ax[0][0].grid(alpha=.3)

    for world, d in findings["C_alpha_beta"].items():
        xs = [p[0] for p in d["points"]]
        ys = [p[1] * 1e6 for p in d["points"]]
        ax[0][1].plot(xs, ys, "o-", label=f"world={world}")
        ax[0][1].axhline(d["alpha_us"], ls=":", lw=.8, color="grey")
    ax[0][1].set_xscale("log", base=2)
    ax[0][1].set_yscale("log")
    ax[0][1].set_xlabel("message bytes")
    ax[0][1].set_ylabel("all-reduce time (us)")
    ax[0][1].set_title("B. latency floor, then bandwidth slope")
    ax[0][1].legend(fontsize=7)
    ax[0][1].grid(alpha=.3)

    drows = [r for r in findings["D_algorithms"] if r["world"] == 4]
    for alg in sorted({r["alg"] for r in drows}):
        xs = [r["bytes"] for r in drows if r["alg"] == alg]
        ys = [r["us"] for r in drows if r["alg"] == alg]
        ax[1][0].plot(xs, ys, "o-", label=alg)
    ax[1][0].set_xscale("log", base=2)
    ax[1][0].set_yscale("log")
    ax[1][0].set_xlabel("message bytes")
    ax[1][0].set_ylabel("time (us)")
    ax[1][0].set_title("C. all-reduce algorithms, world=4")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=.3)

    e = findings["E_busbw_invariance"]
    ws = sorted(int(w) for w in e["algbw"])
    ax[1][1].plot(ws, [e["algbw"][w] if w in e["algbw"] else e["algbw"][str(w)] for w in ws],
                  "o-", label="algbw")
    ax[1][1].plot(ws, [e["busbw"][w] if w in e["busbw"] else e["busbw"][str(w)] for w in ws],
                  "s-", label="busbw")
    ax[1][1].set_xlabel("world size")
    ax[1][1].set_ylabel("GB/s")
    ax[1][1].set_title("D. 16 MiB all-reduce: algbw falls, busbw is the invariant")
    ax[1][1].legend(fontsize=7)
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "nccl_tests.png", dpi=120)


def main():
    import time as _t
    t0 = _t.perf_counter()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    plot()
    findings["runtime_s"] = _t.perf_counter() - t0
    print(f"total runtime {findings['runtime_s']:.1f} s")
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    with open(OUT / "findings.csv", "w") as f:
        f.write("section,world,key,bytes,value_us,algbw_GBs,busbw_GBs\n")
        for r in findings["B_sweep"]:
            f.write(f"B,{r['world']},{r['collective']},{r['bytes']},{r['us']:.2f},"
                    f"{r['algbw']:.4f},{r['busbw']:.4f}\n")
        for r in findings["D_algorithms"]:
            f.write(f"D,{r['world']},{r['alg']},{r['bytes']},{r['us']:.2f},,\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
