"""Project 38 -- Profile a single decode step.

Nsight Systems cannot run on this machine (see the README), so the engine
profiles itself: every kernel launch is bracketed by a CUDA event pair, and the
launch overhead is measured against an empty kernel.

  python3 run.py          # full run, ~3 minutes
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "37-roofline-plot-for-your-engine"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import enginelib as E  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CTX = 1024


@triton.jit
def k_null(p):
    """A kernel that does nothing.  Timing a chain of these is the only way to
    see the launch cost on its own, with zero work hiding inside it."""
    tl.store(p, tl.load(p))


# ------------------------------------------------------------------ tracing
def trace_step(eng, B: int, reps: int = 10) -> dict:
    """Method 1 -- in situ.  Bracket every launch with a CUDA event pair and
    aggregate by kernel name.  Sees the real order and the real cache state,
    and pays for it (section C)."""
    eng.set_len(CTX - 1)
    for _ in range(3):
        eng.decode_step(B, advance=False)
    torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: {"count": 0, "ms": 0.0, "bytes": 0})
    for _ in range(reps):
        E.warm(5)
        eng.set_len(CTX - 1)
        E.trace_start()
        eng.decode_step(B, advance=False)
        for name, nbytes, ms in E.trace_stop():
            a = agg[name]
            a["count"] += 1
            a["ms"] += ms
            a["bytes"] = nbytes
    rows = []
    for name, a in agg.items():
        cnt = a["count"] // reps
        rows.append({"name": name, "count": cnt, "ms": a["ms"] / reps})
    rows.sort(key=lambda r: -r["ms"])
    return {"rows": rows, "total_ms": sum(r["ms"] for r in rows),
            "n_launches": sum(r["count"] for r in rows)}


def replay_step(eng, B: int) -> dict:
    """Method 2 -- isolated.  Record every launch's arguments, then time each
    one on its own in a tight loop.  No observer overhead, but every kernel now
    runs against a warm cache it would not normally have."""
    eng.set_len(CTX - 1)
    eng.decode_step(B, advance=False)
    torch.cuda.synchronize()
    eng.set_len(CTX - 1)
    E.record_start()
    eng.decode_step(B, advance=False)
    entries = E.record_stop()
    torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: {"count": 0, "ms": 0.0, "bytes": 0})
    singles = []
    for i, e in enumerate(entries):
        eng.set_len(CTX - 1)
        ms = E.replay_launch(e)
        a = agg[e[0]]
        a["count"] += 1
        a["ms"] += ms
        a["bytes"] = e[1]
        singles.append({"i": i, "name": e[0], "ms": ms, "bytes": e[1]})
    rows = []
    for name, a in agg.items():
        ms = a["ms"]
        rows.append({"name": name, "count": a["count"], "ms": ms,
                     "bytes_per_launch": a["bytes"],
                     "bytes": a["bytes"] * a["count"],
                     "gbs": a["bytes"] * a["count"] / ms / 1e6 if ms else 0.0,
                     "us_per_launch": ms * 1000 / a["count"]})
    rows.sort(key=lambda r: -r["ms"])
    singles.sort(key=lambda r: -r["ms"])
    return {"rows": rows, "total_ms": sum(r["ms"] for r in rows),
            "n_launches": len(entries), "top_launches": singles[:6]}


# ------------------------------------------------------------ launch costs
def launch_costs(eng, B: int) -> dict:
    p = torch.empty(1, device=E.DEV)
    n = 2000
    k_null[(1,)](p)
    torch.cuda.synchronize()
    # CPU side: how long Python + Triton take to ISSUE one launch.
    t0 = time.perf_counter()
    for _ in range(n):
        k_null[(1,)](p)
    cpu_us = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    # Eager back-to-back: the GPU cannot go faster than the CPU feeds it.
    eager_us = E.gpu_time(lambda: [k_null[(1,)](p) for _ in range(n)], reps=3) * 1000 / n
    # Hardware floor: the same n launches replayed from a CUDA graph, where no
    # CPU is involved at all.
    gnull = E.Graph(lambda: [k_null[(1,)](p) for _ in range(n)], warmup=1)
    gpu_us = E.gpu_time(gnull.replay, reps=5) * 1000 / n

    eng.set_len(CTX - 1)
    g = E.Graph(lambda: eng.decode_step(B, advance=False))
    # Interleaved so a background job on this shared machine cannot land on
    # only one of the two variants.
    res = E.interleaved({"eager": lambda: eng.decode_step(B, advance=False),
                         "graph": g.replay}, reps=7, inner=15)
    eager, graphed = res["eager"], res["graph"]
    wall = E.wall_time(lambda: eng.decode_step(B, advance=False), reps=20)
    graph_wall = E.wall_time(g.replay, reps=30)
    nk = eng.n_kernels_per_decode(advance=False)
    return {"null_gpu_us": gpu_us, "null_cpu_us": cpu_us,
            "null_eager_us": eager_us, "n_kernels": nk,
            "graph_nodes": g.n_nodes,
            "eager_ms": eager, "eager_wall_ms": wall,
            "graph_ms": graphed, "graph_wall_ms": graph_wall,
            "floor_ms": nk * gpu_us / 1000,
            "cpu_issue_ms": nk * cpu_us / 1000,
            "graph_saving_pct": 100 * (eager - graphed) / eager}


# ------------------------------------------------------------- byte budget
def byte_budget(eng, B: int, rows: list) -> dict:
    """Where does a decode step's HBM traffic actually come from?"""
    cfg = eng.cfg
    w = cfg.weight_bytes()
    kv = B * CTX * cfg.kv_bytes_per_token()
    act = sum(r["bytes"] for r in rows
              if not r["name"].startswith("gemm") and not r["name"].startswith("attn"))
    tot = w + kv + act
    return {"weights": w, "kv": kv, "activations": act, "total": tot,
            "weight_share": w / tot, "kv_share": kv / tot, "act_share": act / tot}


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = f["replay_b1"]["rows"]
    lc = f["launch"]
    ceil = f["copy_gbs"]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    a = ax[0]
    names = [r["name"] for r in tr][::-1]
    ms = [r["ms"] for r in tr][::-1]
    tot = sum(ms)
    colors = ["#c0392b" if n.startswith("gemm") else
              "#1f6f8b" if n.startswith("attn") else "#e8b04b" for n in names]
    a.barh(names, ms, color=colors)
    for i, (m, n) in enumerate(zip(ms, names)):
        a.text(m + tot * 0.008, i, f"{100*m/tot:.1f}%", va="center", fontsize=8,
               color="0.3")
    a.set_xlabel("milliseconds per decode step (batch 1)")
    a.set_title(f"Where the {f['replay_b1']['n_launches']} kernels spend the step")
    a.set_xlim(0, max(ms) * 1.22)
    a.grid(alpha=0.2, axis="x", lw=0.4)

    a = ax[1]
    for r in tr:
        if r["bytes"] <= 0:
            continue
        c = ("#c0392b" if r["name"].startswith("gemm") else
             "#1f6f8b" if r["name"].startswith("attn") else "#e8b04b")
        a.scatter(r["bytes_per_launch"], r["us_per_launch"], s=48, color=c, zorder=5)
        a.annotate(r["name"], (r["bytes_per_launch"], r["us_per_launch"]),
                   textcoords="offset points", xytext=(6, 3), fontsize=7, color="0.3")
    xs = [10 ** (i / 4) for i in range(12, 32)]
    a.plot(xs, [x / ceil / 1e3 for x in xs], "--", color="0.35",
           label=f"{ceil:.0f} GB/s (copy ceiling)")
    a.axhline(lc["null_gpu_us"], color="0.6", ls=":",
              label=f"launch floor {lc['null_gpu_us']:.1f} us")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("bytes moved per launch")
    a.set_ylabel("microseconds per launch")
    a.set_title("Small kernels are priced by latency, not bytes")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.2, which="both", lw=0.4)

    a = ax[2]
    kern = f["replay_b1"]["total_ms"]
    labels = ["kernels alone\n(isolated replay)", "real step\n(eager)",
              "with an event pair\nper launch"]
    work = [kern, kern, kern]
    extra = [0.0, max(0.0, lc["eager_ms"] - kern),
             max(0.0, f["trace_b1"]["total_ms"] - kern)]
    a.bar(labels, work, color="#1f6f8b", label="kernel time (isolated)")
    a.bar(labels, extra, bottom=work, color="#e8b04b", label="everything else")
    a.set_ylabel("ms per decode step")
    a.set_title("Measuring it changes it")
    for i, (w, e) in enumerate(zip(work, extra)):
        a.text(i, w + e + 0.03, f"{w+e:.2f} ms", ha="center", fontsize=9)
    a.legend(fontsize=8)
    a.grid(alpha=0.2, axis="y", lw=0.4)

    fig.suptitle("Project 38 - anatomy of one decode step - 159 kernels, batch 1",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "decode_step_profile.png"), dpi=125)
    print("wrote", os.path.join(OUT, "decode_step_profile.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    cfg = E.Config()
    eng = E.Engine(cfg, max_batch=32, max_seq=CTX, max_tokens=32)
    eng.ctx_hint = CTX

    print("A. copy ceiling")
    n = 1 << 24
    a = torch.empty(n, device=E.DEV)
    b = torch.empty(n, device=E.DEV)
    t = E.gpu_time(lambda: E.k_copy[(triton.cdiv(n, 1024),)](a, b, n, BLOCK=1024), reps=50)
    copy_gbs = 2 * n * 4 / t / 1e6
    print(f"   {copy_gbs:.1f} GB/s")

    print("B. isolated replay, batch 1")
    r1 = replay_step(eng, 1)
    for r in r1["rows"]:
        print(f"   {r['name']:20s} x{r['count']:3d}  {r['ms']:7.3f} ms  "
              f"{r['us_per_launch']:7.1f} us/launch  {r['gbs']:6.1f} GB/s")
    print(f"   kernels alone: {r1['total_ms']:.3f} ms over {r1['n_launches']} launches")

    print("C. in-situ trace, batch 1 (the observer effect)")
    t1 = trace_step(eng, 1)
    print(f"   traced total {t1['total_ms']:.3f} ms")

    print("C2. batch 32")
    r32 = replay_step(eng, 32)
    for r in r32["rows"][:6]:
        print(f"   {r['name']:20s} x{r['count']:3d}  {r['ms']:7.3f} ms")

    print("D. launch costs")
    lc = launch_costs(eng, 1)
    print(f"   null kernel: {lc['null_gpu_us']:.2f} us GPU, {lc['null_cpu_us']:.2f} us CPU issue")
    print(f"   {lc['n_kernels']} kernels -> floor {lc['floor_ms']:.3f} ms, "
          f"cpu issue {lc['cpu_issue_ms']:.3f} ms")
    print(f"   eager {lc['eager_ms']:.3f} ms  graph {lc['graph_ms']:.3f} ms "
          f"({lc['graph_saving_pct']:.1f}% saved)")

    f = {
        "copy_gbs": copy_gbs,
        "model": {"params": cfg.n_params(), "weight_bytes": cfg.weight_bytes(),
                  "kv_bytes_per_token": cfg.kv_bytes_per_token()},
        "ctx": CTX,
        "replay_b1": r1,
        "replay_b32": r32,
        "trace_b1": t1,
        "launch": lc,
        "budget_b1": byte_budget(eng, 1, r1["rows"]),
        "budget_b32": byte_budget(eng, 32, r32["rows"]),
    }
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
