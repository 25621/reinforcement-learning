"""Project 41 -- CUDA Graphs for decode.

Capture the per-token kernel sequence once, replay it with a single call, and
measure what that is worth.  Project 38 found 0.2% on the full model; this
project sweeps model size until the answer changes, and checks that a replayed
graph still produces the right tokens.

  python3 run.py          # full run, ~5 minutes
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
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

CTX = 512

# Four models spanning 30x in per-step GPU work, same architecture throughout.
MODELS = {
    "tiny (0.6M)": E.Config(d_model=128, n_heads=4, n_kv_heads=1, n_layers=4,
                            d_ff=352, vocab=2048),
    "small (4M)": E.Config(d_model=256, n_heads=8, n_kv_heads=2, n_layers=6,
                           d_ff=704, vocab=4096),
    "medium (26M)": E.Config(d_model=512, n_heads=8, n_kv_heads=2, n_layers=8,
                             d_ff=1408, vocab=8192),
    "full (152M)": E.Config(),
}
LAYER_SWEEP = [2, 4, 8, 16, 24]


@triton.jit
def k_null(p):
    tl.store(p, tl.load(p))


def cpu_issue_ms(fn, reps: int = 3) -> float:
    """How long the CPU takes to *issue* the step, with the GPU running behind.

    The CUDA queue is over a thousand launches deep, so a couple of steps can
    be pushed into it before the driver makes the caller wait -- which is
    exactly the quantity we want.
    """
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t0) / reps
    torch.cuda.synchronize()
    return dt * 1e3


def measure(cfg: E.Config, batch: int = 1) -> dict:
    eng = E.Engine(cfg, max_batch=batch, max_seq=CTX, max_tokens=batch)
    eng.ctx_hint = CTX
    eng.set_len(CTX - 1)
    nk = eng.n_kernels_per_decode()

    t0 = time.perf_counter()
    g = E.Graph(lambda: eng.decode_step(batch, advance=False))
    build_ms = (time.perf_counter() - t0) * 1e3

    eng.set_len(CTX - 1)
    res = E.interleaved({"eager": lambda: eng.decode_step(batch, advance=False),
                         "graph": g.replay}, reps=7, inner=15)
    eng.set_len(CTX - 1)
    issue = cpu_issue_ms(lambda: eng.decode_step(batch, advance=False))
    eager_wall = E.wall_time(lambda: eng.decode_step(batch, advance=False), reps=20)
    graph_wall = E.wall_time(g.replay, reps=30)
    nodes = g.n_nodes
    g.close()
    del eng
    torch.cuda.empty_cache()
    return {"params": cfg.n_params(), "d_model": cfg.d_model,
            "n_layers": cfg.n_layers, "kernels": nk, "graph_nodes": nodes,
            "eager_ms": res["eager"], "graph_ms": res["graph"],
            "eager_wall_ms": eager_wall, "graph_wall_ms": graph_wall,
            "cpu_issue_ms": issue, "build_ms": build_ms,
            "speedup": res["eager"] / res["graph"],
            "ratio": issue / res["graph"],
            "predicted": max(1.0, issue / res["graph"])}


def correctness() -> dict:
    """A graph replays fixed pointers.  Does the sequence position still move?

    Two graphs of the same step: one that includes the kernel incrementing the
    device-side position counter, one that does not.  Both are replayed eight
    times and compared with an eager eight-token generation.
    """
    cfg = MODELS["small (4M)"]
    eng = E.Engine(cfg, max_batch=1, max_seq=CTX, max_tokens=1)
    eng.ctx_hint = CTX
    x0 = eng.x[:cfg.d_model].cpu().clone()

    def gen_eager(n=8):
        eng.set_len(64)
        eng.x[:cfg.d_model].copy_(x0)
        out = []
        for _ in range(n):
            eng.decode_step(1, advance=True)
            torch.cuda.synchronize()
            out.append(int(eng.tok[0].cpu()))
        return out

    ref = gen_eager()

    def gen_graph(advance, n=8):
        g = E.Graph(lambda: eng.decode_step(1, advance=advance))
        eng.set_len(64)
        eng.x[:cfg.d_model].copy_(x0)
        out = []
        for _ in range(n):
            g.replay()
            torch.cuda.synchronize()
            out.append(int(eng.tok[0].cpu()))
        end = int(eng.seqlen[0].cpu())
        g.close()
        return out, end

    with_ctr, end_with = gen_graph(True)
    without_ctr, end_without = gen_graph(False)
    del eng
    torch.cuda.empty_cache()
    return {"eager": ref, "graph_with_counter": with_ctr,
            "graph_without_counter": without_ctr,
            "with_counter_matches": with_ctr == ref,
            "without_counter_matches": without_ctr == ref,
            "final_len_with": end_with, "final_len_without": end_without}


def graph_memory(cfg: E.Config, n: int = 16) -> dict:
    """Each captured graph costs device memory.  How much, per bucket?"""
    eng = E.Engine(cfg, max_batch=1, max_seq=CTX, max_tokens=1)
    eng.ctx_hint = CTX
    eng.set_len(CTX - 1)
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    graphs = [E.Graph(lambda: eng.decode_step(1, advance=False)) for _ in range(n)]
    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info()[0]
    for g in graphs:
        g.close()
    torch.cuda.synchronize()
    free2 = torch.cuda.mem_get_info()[0]
    del eng
    torch.cuda.empty_cache()
    return {"n": n, "bytes_per_graph": (free0 - free1) / n,
            "reclaimed_frac": (free2 - free1) / max(1, free0 - free1)}


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    rows = f["models"]
    xs = [r["graph_ms"] for r in rows]
    a.plot(xs, [r["speedup"] for r in rows], "o-", color="#c0392b",
           label="measured speedup")
    a.plot(xs, [r["predicted"] for r in rows], "s--", color="0.45",
           label="max(1, CPU issue / GPU work)")
    for r in rows:
        a.annotate(r["name"].split(" ")[0], (r["graph_ms"], r["speedup"]),
                   textcoords="offset points", xytext=(5, 6), fontsize=8)
    a.axhline(1.0, color="0.7", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("GPU work per decode step (ms)")
    a.set_ylabel("speedup from replaying a CUDA graph")
    a.set_title("Graphs pay only when the CPU cannot keep up")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[1]
    rows = f["layers"]
    n = [r["n_layers"] for r in rows]
    a.plot(n, [r["cpu_issue_ms"] for r in rows], "o-", color="#e8b04b",
           label="CPU: issue the launches")
    a.plot(n, [r["graph_ms"] for r in rows], "s-", color="#1f6f8b",
           label="GPU: run the kernels")
    a.plot(n, [r["eager_ms"] for r in rows], "^-", color="#c0392b",
           label="eager step (the max of the two)")
    a.set_xlabel("layers (at d_model = 256)")
    a.set_ylabel("ms per decode step")
    a.set_title("Two clocks racing; the slower one wins")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, lw=0.4)

    a = ax[2]
    rows = f["models"]
    names = [r["name"] for r in rows]
    idx = range(len(names))
    w = 0.38
    a.bar([i - w / 2 for i in idx], [r["cpu_issue_ms"] for r in rows], w,
          color="#e8b04b", label="CPU issue time")
    a.bar([i + w / 2 for i in idx], [r["graph_ms"] for r in rows], w,
          color="#1f6f8b", label="GPU work")
    for i, r in enumerate(rows):
        a.text(i, max(r["cpu_issue_ms"], r["graph_ms"]) * 1.06,
               f"{r['speedup']:.2f}x", ha="center", fontsize=9)
    a.set_xticks(list(idx))
    a.set_xticklabels([n.split(" ")[0] for n in names])
    a.set_yscale("log")
    a.set_ylabel("ms per decode step")
    a.set_title("Where the crossover is")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y", which="both", lw=0.4)

    fig.suptitle("Project 41 - CUDA Graphs: replacing 160 launches with one",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "cuda_graphs.png"), dpi=125)
    print("wrote", os.path.join(OUT, "cuda_graphs.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    print("A. does a replayed graph still generate the right tokens?")
    corr = correctness()
    print("   eager           ", corr["eager"])
    print("   graph +counter  ", corr["graph_with_counter"],
          "match" if corr["with_counter_matches"] else "MISMATCH")
    print("   graph -counter  ", corr["graph_without_counter"],
          "match" if corr["without_counter_matches"] else "MISMATCH")

    p = torch.empty(1, device=E.DEV)
    k_null[(1,)](p)
    torch.cuda.synchronize()
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        k_null[(1,)](p)
    per_launch_us = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    print(f"B. one launch costs the CPU {per_launch_us:.2f} us to issue")

    print("C. model-size sweep")
    models = []
    for name, cfg in MODELS.items():
        r = measure(cfg)
        r["name"] = name
        models.append(r)
        print(f"   {name:14s} {r['kernels']:4d} kernels  gpu {r['graph_ms']:7.3f} ms  "
              f"cpu {r['cpu_issue_ms']:7.3f} ms  eager {r['eager_ms']:7.3f}  "
              f"-> {r['speedup']:.2f}x (predicted {r['predicted']:.2f}x)")

    print("D. layer sweep at d_model = 256")
    layers = []
    for L in LAYER_SWEEP:
        cfg = E.Config(d_model=256, n_heads=8, n_kv_heads=2, n_layers=L,
                       d_ff=704, vocab=4096)
        r = measure(cfg)
        r["name"] = f"{L} layers"
        layers.append(r)
        print(f"   {L:3d} layers  {r['kernels']:4d} kernels  gpu {r['graph_ms']:7.3f} ms  "
              f"cpu {r['cpu_issue_ms']:7.3f} ms -> {r['speedup']:.2f}x")

    print("E. what a graph costs")
    mem = graph_memory(MODELS["small (4M)"])
    print(f"   {mem['bytes_per_graph']/1024:.0f} KiB per captured graph, "
          f"{100*mem['reclaimed_frac']:.0f}% reclaimed on destroy")

    f = {"device": E.device_info(), "ctx": CTX,
         "per_launch_us": per_launch_us, "correctness": corr,
         "models": models, "layers": layers, "memory": mem}
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
