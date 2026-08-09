"""Project 29 — Bottleneck fix.

Take the workload `torch.compile` is best at — project 26's element-wise chain,
2.7x faster compiled — and break it with four ordinary lines of Python. Then
find the breaks and fix them:

  1. the patient, and the control (eager, compiled clean)
  2. what Dynamo says about each planted line, in its own words
  3. what a break costs: captured operations, and time
  4. fixing them one at a time
  5. the rewrite that keeps the branch and the speedup
  6. the other recompilation trap: a Python number baked into a graph

Runtime ~6 min (most of it compiling). Needs torch, numpy, matplotlib and a
C++ compiler.
"""

import contextlib
import csv
import io
import os
import sys
import tempfile
import time
from pathlib import Path

# a fresh Inductor cache, and no cross-run reuse: the compile times and the
# capture counts below are cold ones
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor-cache-")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._inductor.config  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Every variant below is a different closure over the SAME code object, and
# Dynamo caches (and gives up on) compiled code per code object. Compiling them
# one after another in one process makes them contaminate each other: after one
# variant breaks, the next is skipped without being traced at all. The fix is a
# `torch._dynamo.reset()` before every single compile, which `compile_once`
# below does. Inductor's on-disk cache is left ON so that re-compiling the same
# graph in a later round is cheap; the *cache directory* is fresh per run, so
# the first compile of each graph is still a cold one.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24-profile-a-training-step"))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ROUNDS = 8
REPEATS, WARMUP = 7, 3
rows = []
LOG = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<40} {value:>16}  {note}")


# ---------------------------------------------------------------------------
# the patient: eight rounds of element-wise work, with four things people write
# ---------------------------------------------------------------------------
def make_chain(problems=()):
    problems = set(problems)
    log = []

    def chain(x):
        for i in range(ROUNDS):
            x = F.gelu(x) * 1.01 + torch.tanh(x) * 0.5
            if "print" in problems and i == 3:            # (1) a debug print
                print(f"    round {i}: ok")
            if "branch" in problems and i == 3:           # (2) data-dependent branch
                if x.abs().max().item() > 1e9:
                    x = x * 0.5
            if "numpy" in problems and i == 3:            # (3) a numpy round trip
                x = x - float(np.mean(x.detach().numpy()))
            if "log" in problems and i == 3:              # (4) a running statistic
                log.append(x.std().item())
        return x
    return chain


def chain_fixed(x):
    """The same maths, with the branch expressed as a tensor operation."""
    for i in range(ROUNDS):
        x = F.gelu(x) * 1.01 + torch.tanh(x) * 0.5
        if i == 3:
            x = torch.where(x.abs().max() > 1e9, x * 0.5, x)
    return x


def compile_once(fn, x, quiet=True):
    """Compile fn from a clean Dynamo state and report what it captured."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(fn)
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        compiled(x)
    compile_s = time.perf_counter() - t0
    unimpl = torch._dynamo.utils.counters["unimplemented"]
    stats = torch._dynamo.utils.counters["stats"]
    return compiled, {
        "breaks": sum(unimpl.values()),
        "reasons": list(unimpl.keys()),
        "captured_ops": stats.get("calls_captured", 0),
        "graphs": stats.get("unique_graphs", 0),
        "compile_s": compile_s,
    }


def timed(fn, x, quiet=True):
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        return P.best_of(lambda: fn(x), repeats=REPEATS, warmup=WARMUP)


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"load average {os.getloadavg()[0]:.1f}\n")
    x = torch.randn(16, 128, 256)

    # -----------------------------------------------------------------
    # compile every variant first, then time them all interleaved: measuring
    # one variant to exhaustion before starting the next lets a busy minute on
    # a shared machine land entirely on one of them and look like a finding
    # -----------------------------------------------------------------
    print("[1] compiling and timing every variant, interleaved")
    stages = [
        ("all four", ("print", "branch", "numpy", "log")),
        ("- print", ("branch", "numpy", "log")),
        ("- numpy", ("branch", "log")),
        ("- log", ("branch",)),
    ]
    factories = {"eager": lambda: make_chain()}
    for name in ("print", "branch", "numpy", "log"):
        factories[f"only {name}"] = (lambda n=name: make_chain((n,)))
    for label, probs in stages:
        factories[label] = (lambda pr=probs: make_chain(pr))
    factories["clean"] = lambda: make_chain()
    factories["torch.where"] = lambda: chain_fixed

    times = {k: [] for k in factories}
    infos = {}
    for r in range(3):
        for k, factory in factories.items():
            fn = factory()
            if k == "eager":
                times[k].append(timed(fn, x)[0])
                continue
            compiled, inf = compile_once(fn, x)
            infos.setdefault(k, inf)
            times[k].append(timed(compiled, x)[0])
        print(f"    round {r + 1} done")
    ms = {k: min(v) for k, v in times.items()}
    spread = {k: max(v) - min(v) for k, v in times.items()}
    e_ms = ms["eager"]
    variants = {k: (None, infos.get(k)) for k in factories}

    record("control", "eager (ms)", f"{e_ms:.2f}",
           f"spread over 3 rounds {spread['eager']:.2f}")
    record("control", "noise floor",
           f"{100*spread['eager']/e_ms:.1f} %", "of the eager time")
    info = variants["clean"][1]
    record("clean", "compiled, no problems (ms)", f"{ms['clean']:.2f}",
           f"spread {spread['clean']:.2f}")
    record("clean", "speedup", f"{e_ms/ms['clean']:.2f}x", "this is what is at stake")
    record("clean", "operations captured", f"{info['captured_ops']}",
           f"{info['graphs']} graph, compiled in {info['compile_s']:.0f} s")

    # -----------------------------------------------------------------
    # 2-3. each problem on its own
    # -----------------------------------------------------------------
    print("\n[2-3] one problem at a time")
    per_problem = {}
    for name in ("print", "branch", "numpy", "log"):
        key = f"only {name}"
        inf = variants[key][1]
        per_problem[name] = (ms[key], e_ms / ms[key], inf)
        record("problem", f"'{name}' compiled (ms)", f"{ms[key]:.2f}",
               f"{e_ms/ms[key]:.2f}x vs eager | {inf['captured_ops']} ops captured "
               f"| {inf['breaks']} break")
        if inf["reasons"]:
            first = inf["reasons"][0]
            headline = first.split("\n")[0][:70]
            record("reason", name, headline[:16] + "...", headline)
            LOG.append(f"=== {name} ===\n{first}\n")
    (OUT / "graph_breaks.txt").write_text("\n".join(LOG))

    # -----------------------------------------------------------------
    # 4. fixing them one at a time
    # -----------------------------------------------------------------
    print("\n[4] removing the problems one at a time")
    results = []
    for label, probs in stages + [("- branch (clean)", ())]:
        key = label if label != "- branch (clean)" else "clean"
        inf = variants[key][1]
        results.append((label, ms[key], e_ms / ms[key], inf["captured_ops"], len(probs)))
        record("fix", label, f"{ms[key]:.2f} ms",
               f"{e_ms/ms[key]:.2f}x vs eager | {inf['captured_ops']} ops captured "
               f"| {len(probs)} problems left")

    # -----------------------------------------------------------------
    # 5. the rewrite
    # -----------------------------------------------------------------
    print("\n[5] keeping the branch, without the break")
    finfo = variants["torch.where"][1]
    f_ms = ms["torch.where"]
    record("rewrite", "torch.where instead of if/.item() (ms)", f"{f_ms:.2f}",
           f"{e_ms/f_ms:.2f}x vs eager")
    record("rewrite", "operations captured", f"{finfo['captured_ops']}",
           f"{finfo['breaks']} breaks")
    same = torch.allclose(chain_fixed(x), make_chain(("branch",))(x), atol=1e-5)
    record("rewrite", "same result as the branch version", str(same))
    c_ms = ms["clean"]

    # -----------------------------------------------------------------
    # 6. a Python number baked into the graph
    # -----------------------------------------------------------------
    print("\n[6] the other trap: a Python number that changes")
    torch._dynamo.config.recompile_limit = 8          # back to the default
    record("recompile", "recompile_limit", f"{torch._dynamo.config.recompile_limit}",
           "compiles allowed per function before Dynamo gives up")

    def variable_rounds(x, n):
        for _ in range(n):
            x = F.gelu(x) * 1.01 + torch.tanh(x) * 0.5
        return x

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    vr = torch.compile(variable_rounds)
    per_call = []
    for n in range(1, 13):
        t0 = time.perf_counter()
        vr(x, n)
        per_call.append(time.perf_counter() - t0)
    compiles = sum(1 for t in per_call if t > 0.5)
    record("recompile", "calls that compiled", f"{compiles} of 12",
           f"{sum(per_call):.1f} s in total")
    record("recompile", "call 1 / call 12 (s)",
           f"{per_call[0]:.2f} / {per_call[-1]:.3f}",
           "after the limit, Dynamo stops trying and runs eager")
    record("recompile", "unique graphs",
           f"{torch._dynamo.utils.counters['stats'].get('unique_graphs', 0)}",
           "one per trip count — the loop is unrolled into the graph")

    # -----------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------
    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    names = ["eager"] + list(per_problem.keys()) + ["clean"]
    vals = [e_ms] + [per_problem[k][0] for k in per_problem] + [c_ms]
    colors = [ps.INK_MUTED] + [ps.SERIES[2]] * len(per_problem) + [ps.SERIES[1]]
    ax.barh(np.arange(len(names)), vals, color=colors)
    ax.axvline(e_ms, color=ps.INK_MUTED, ls="--", lw=1.2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("One bad line each, compiled", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("ms per call (dashed = eager)", color=ps.INK_SECONDARY)

    ax = axes[1]
    labels = [r[0] for r in results]
    caps = [r[3] for r in results]
    ax.bar(np.arange(len(labels)), caps, color=ps.SERIES[0])
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(labels)), [r[2] for r in results], "o-",
             color=ps.SERIES[2], lw=1.8)
    ax2.set_ylabel("speedup vs eager", color=ps.SERIES[2])
    ax2.tick_params(colors=ps.INK_MUTED, labelsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_title("Nothing improves until the last break is gone", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_ylabel("operations captured", color=ps.INK_SECONDARY)

    ax = axes[2]
    xs = np.arange(1, 13)
    ax.plot(xs, per_call, "o-", color=ps.SERIES[2], lw=1.8)
    ax.axvline(torch._dynamo.config.recompile_limit, color=ps.BASELINE, ls="--", lw=1.2)
    ax.text(torch._dynamo.config.recompile_limit + 0.15, max(per_call) * 0.3,
            "recompile\nlimit", color=ps.INK_MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_title("A new Python value means a new compile", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("call number (loop count n = 1..12)", color=ps.INK_SECONDARY)
    ax.set_ylabel("call time (s, log scale)", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "bottleneck_fix.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
