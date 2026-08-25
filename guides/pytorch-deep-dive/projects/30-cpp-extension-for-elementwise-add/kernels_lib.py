"""Shared pieces for the Phase-6 (custom kernels) projects 30-35.

One compiler wrapper, one interleaved timer, one FLOP/byte bookkeeper.

Why a shared module at all: every project in this phase compiles C++ with the
same flags, on the same six threads, and reports numbers that are only
comparable if they were taken the same way. Project 32's "my matmul reaches 8 %
of oneDNN" and project 33's "fusion saves 1.4x" are the same machine or they are
nothing.

Hardware note (the reason this phase is not a GPU phase)
--------------------------------------------------------
`torch.cuda.is_available()` returns True on this machine, and then the first
kernel launch fails with `no kernel image is available for execution on the
device`. The GPU is a GTX 1070 Ti, compute capability sm_61 (Pascal, 2017);
this PyTorch build (2.10+cu128) ships no Pascal kernels, and Triton's own
minimum is sm_70. So CUDA and Triton cannot run here at all -- not slowly, not
at all.

Everything Phase 6 teaches still applies, because the *ideas* are not
GPU-specific:

    GPU concept              CPU equivalent used in this phase
    ---------------------    --------------------------------------------
    thread block / program   one `at::parallel_for` chunk
    shared memory (on-chip)  the L1/L2 cache, reached by blocking loops
    warp / SIMT lane         an AVX2 SIMD lane (8 floats wide here)
    HBM bandwidth            DRAM bandwidth
    kernel launch            a C++ function call through the dispatcher
    `triton.jit` compile     `load_inline` compile (~20 s, cached on disk)

Each project prints the Triton or CUDA source it *would* run next to the C++ it
actually ran, so the mapping stays visible.
"""

import os

# Must happen before torch is imported: otherwise `torch.cuda.is_available()`
# says True and every later `.cuda()` raises at the first launch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

THREADS = 6
torch.set_num_threads(THREADS)
torch.manual_seed(0)

HERE = Path(__file__).resolve().parent

# Reuse the Phase-5 timing helpers so that a millisecond in project 31 means the
# same thing as a millisecond in project 25.
sys.path.insert(0, str(HERE.parent / "24-profile-a-training-step"))
from perf_lib import best_of, cpu_time, human, mb  # noqa: E402,F401


# ---------------------------------------------------------------------------
# compiling C++ at runtime
# ---------------------------------------------------------------------------
# -O3             let gcc unroll and vectorize; without it a hand loop is much slower
# -march=native   allow AVX2 on this CPU (8 floats per instruction)
# -funroll-loops  fewer loop-counter checks per useful add
# -fopenmp        at::parallel_for needs the OpenMP runtime to have threads
# (`-ffast-math` is deliberately NOT here: it lets gcc reassociate float adds,
#  which changes results. Project 31 adds it to one kernel on purpose and
#  measures what it buys and what it costs.)
CFLAGS = ["-O3", "-march=native", "-funroll-loops", "-fopenmp"]
LDFLAGS = ["-fopenmp"]


# Compiled objects live outside the repo: a .so is a build artifact, not a
# result worth committing.
BUILD_ROOT = Path(os.environ.get("PHASE6_BUILD_DIR",
                                 Path(tempfile.gettempdir()) / "pytorch-phase6-kernels"))


def build(name, source, functions=None, extra_cflags=(), verbose=False, force=False):
    """Compile a C++ string into an importable module, and time it.

    Returns (module, seconds).

    Why this wrapper caches by hand: `load_inline` does NOT skip work when the
    source is unchanged. It bakes the module name into the compile command as
    `-DTORCH_EXTENSION_NAME=<name>_v<version>`, and it bumps that version on
    every call whose inputs it has not seen *in this process*. A different
    define means a different compile command, so ninja recompiles -- ~22 s,
    every run, forever. Project 30 measures this.

    So we key a directory on sha256(source + flags) ourselves and, if the .so
    is already sitting there, import it directly. Second run: ~2 ms.
    """
    key = hashlib.sha256(
        (source + repr(sorted(functions or [])) + repr(list(extra_cflags))).encode()
    ).hexdigest()[:12]
    build_dir = BUILD_ROOT / f"{name}-{key}"
    build_dir.mkdir(parents=True, exist_ok=True)
    so = build_dir / f"{name}.so"

    if so.exists() and not force:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location(name, so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, time.perf_counter() - t0

    # If an earlier run was killed mid-compile it left load_inline's lock file
    # behind, and the next run waits on it forever with no message. Clear one
    # that nobody can still be holding.
    lock = build_dir / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > 300:
        lock.unlink()

    t0 = time.perf_counter()
    mod = load_inline(
        name=name,
        cpp_sources=source,
        functions=list(functions) if functions else None,
        extra_cflags=CFLAGS + list(extra_cflags),
        extra_ldflags=LDFLAGS,
        build_directory=str(build_dir),
        verbose=verbose,
    )
    return mod, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def interleaved(fns, rounds=7, warmup=2):
    """Time several callables by rotating between them, best-of-rounds.

    This machine is shared. If you time all 20 repetitions of version A and
    then all 20 of version B, a neighbour that wakes up in between is charged
    entirely to B. Rotating A, B, C, A, B, C ... spreads any interference over
    every candidate, so a difference that survives is a difference in the code.

    `fns` is a dict {label: callable}. Returns {label: (best_ms, spread_ms)}.
    """
    labels = list(fns)
    for lb in labels:
        for _ in range(warmup):
            fns[lb]()
    times = {lb: [] for lb in labels}
    for _ in range(rounds):
        for lb in labels:
            t0 = time.perf_counter()
            fns[lb]()
            times[lb].append(time.perf_counter() - t0)
    return {
        lb: (min(ts) * 1e3, (max(ts) - min(ts)) * 1e3) for lb, ts in times.items()
    }


def noise_floor(fn, rounds=15):
    """How much the *same* callable varies round to round, in percent.

    Any speedup smaller than this number is not a speedup, it is the machine.
    """
    for _ in range(3):
        fn()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return (max(ts) - min(ts)) / min(ts) * 100.0, statistics.median(ts) * 1e3


# ---------------------------------------------------------------------------
# bookkeeping: bytes moved and FLOPs done -- the two numbers that decide
# whether a kernel is memory-bound or compute-bound
# ---------------------------------------------------------------------------
def gbps(total_bytes, ms):
    """Effective DRAM bandwidth in GB/s."""
    return total_bytes / (ms * 1e-3) / 1e9


def gflops(total_flops, ms):
    return total_flops / (ms * 1e-3) / 1e9


def arithmetic_intensity(total_flops, total_bytes):
    """FLOPs per byte. Low = memory-bound, high = compute-bound."""
    return total_flops / total_bytes


def max_abs_diff(a, b):
    return (a.double() - b.double()).abs().max().item()


def rel_err(a, b):
    d = (a.double() - b.double()).abs().max()
    s = b.double().abs().max().clamp_min(1e-12)
    return (d / s).item()


# ---------------------------------------------------------------------------
# output files
# ---------------------------------------------------------------------------
def outputs_dir(project_file):
    d = Path(project_file).resolve().parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)
