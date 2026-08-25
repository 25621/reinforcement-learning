"""Project 55 - what building PyTorch from source actually costs, measured.

A full build takes hours, so this project does not pretend to run one. It does
something more useful in ten minutes: it runs the *first real stage* of the
build for real (code generation), verifies the output against the wheel you
already have, and then measures the second stage (compilation) on single files
until the hours-long total stops being a mystery and becomes an arithmetic
result you can predict.

Sections:
  1. what your wheel was built with, and what came out
  2. stage 1 for real: run torchgen
  3. the file the dispatcher pointed at in project 53
  4. stage 2: the header tax
  5. extrapolating the build
  6. -O0 vs -O2 vs -O3: what optimisation costs and buys
  7. why an incremental rebuild is still slow
  8. the real recipe

Run:  python3 run.py        (~5 minutes)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "53-trace-one-op-end-to-end"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import source_lib as S  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F = S.Findings()

TORCH = S.torch_root()
INCLUDES = [
    f"-I{os.path.join(TORCH, 'include')}",
    f"-I{os.path.join(TORCH, 'include', 'torch', 'csrc', 'api', 'include')}",
    f"-I{sysconfig.get_paths()['include']}",
]
NPROC = os.cpu_count() or 1


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ===========================================================================
# 1. What your wheel was built with
# ===========================================================================
F.head("1. the build you already have")

config = torch.__config__.show()
with open(os.path.join(OUT, "torch_config.txt"), "w") as fh:
    fh.write(config + "\n\n" + torch.__config__.parallel_info())

for line in config.splitlines():
    line = line.strip()
    if line.startswith(("- GCC", "- C++ Version", "- CPU capability", "- CUDA Runtime", "- OpenMP")):
        F.note("built with", line.lstrip("- "))
F.note("torch version", torch.__version__)
F.note("commit the wheel was built from", torch.version.git_version[:12])

libs = {}
libdir = os.path.join(TORCH, "lib")
for fn in sorted(os.listdir(libdir)):
    if fn.endswith(".so") or ".so." in fn:
        libs[fn] = os.path.getsize(os.path.join(libdir, fn)) / 1e6
for fn, mb in sorted(libs.items(), key=lambda kv: -kv[1])[:6]:
    F.note(f"library {fn} (MB)", mb)
F.note("total shared libraries (MB)", sum(libs.values()))

for lib in ["libtorch_cpu.so", "libc10.so"]:
    out = run(["nm", "-D", "--defined-only", os.path.join(libdir, lib)])
    n = len(out.stdout.splitlines()) if out.returncode == 0 else -1
    F.note(f"{lib}: exported symbols", n)

py_files = sum(len([f for f in files if f.endswith(".py")]) for _, _, files in os.walk(TORCH))
F.note("python files in the wheel", py_files)
hdr = 0
for dirpath, _, files in os.walk(os.path.join(TORCH, "include")):
    hdr += sum(1 for f in files if f.endswith((".h", ".hpp", ".cuh")))
F.note("C++ headers in the wheel", hdr)


# ===========================================================================
# 2. Stage 1 for real: run torchgen
# ===========================================================================
F.head("2. running the code generator")

gen_dir = tempfile.mkdtemp(prefix="p55-torchgen-")
aten_src = os.path.join(S.torchgen_root(), "packaged", "ATen")
t0 = time.perf_counter()
# Run it from inside the output directory: torchgen also emits a few files at
# paths relative to the *current* directory (`torch/csrc/inductor/...`), and
# without this they land in the project folder.
proc = run([sys.executable, "-m", "torchgen.gen", "-s", aten_src, "-d", gen_dir,
            "--per-operator-headers"], cwd=gen_dir)
gen_seconds = time.perf_counter() - t0
F.note("torchgen exit code", proc.returncode)
F.note("torchgen wall time (s)", gen_seconds)

gen_files = []
for dirpath, _, files in os.walk(gen_dir):
    for fn in files:
        gen_files.append(os.path.join(dirpath, fn))
gen_bytes = sum(os.path.getsize(p) for p in gen_files)
F.note("files generated", len(gen_files))
F.note("bytes generated (MB)", gen_bytes / 1e6)
F.note("of those, .h", sum(1 for p in gen_files if p.endswith(".h")))
F.note("of those, .cpp", sum(1 for p in gen_files if p.endswith(".cpp")))
F.note("generated per second", len(gen_files) / gen_seconds)

# "Everything" files are a second copy of the same registrations for builds that
# do not split them; counting them as separate work would double the estimate.
tus = [p for p in gen_files if p.endswith(".cpp") and "Everything" not in os.path.basename(p)]
F.note("real translation units generated", len(tus))
tu_lines = sum(sum(1 for _ in open(p, errors="ignore")) for p in tus)
F.note("lines of C++ in them", tu_lines)

# Verify against the wheel: the same generator, the same input, so a generated
# header should match the one that shipped.
mine = os.path.join(gen_dir, "ops", "add_native.h")
theirs = os.path.join(TORCH, "include", "ATen", "ops", "add_native.h")
if os.path.exists(mine) and os.path.exists(theirs):
    a = open(mine).read().splitlines()
    b = open(theirs).read().splitlines()
    same = sum(1 for line in a if line in b)
    F.note("add_native.h: lines we generated", len(a))
    F.note("add_native.h: lines in the wheel", len(b))
    F.note("add_native.h: lines that match", same)
    shutil.copy(mine, os.path.join(OUT, "generated_add_native.h"))


# ===========================================================================
# 3. The file the dispatcher pointed at
# ===========================================================================
F.head("3. finding project 53's file")

reg = os.path.join(gen_dir, "RegisterCPU_0.cpp")
runtime_where = dict(S.dispatch_rows("aten::add.Tensor")).get("CPU", "")
F.note("dispatcher says the CPU kernel is at", runtime_where)
if os.path.exists(reg):
    lines = open(reg).read().splitlines()
    F.note("RegisterCPU_0.cpp: lines we generated", len(lines))
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if 'm.impl("add.Tensor"' in ln]
    for lineno, text in hits:
        F.note(f"  our line {lineno}", text)
    if hits and ":" in runtime_where:
        theirs_line = int(runtime_where.rsplit(":", 1)[1])
        F.note("line number reported by the running library", theirs_line)
        F.note("difference (lines)", abs(hits[0][0] - theirs_line))
    # The whole file is 500 KB, so save the neighbourhood of the registration
    # rather than all of it - that is the part worth reading.
    if hits:
        lo = max(0, hits[0][0] - 110)
        hi = min(len(lines), hits[0][0] + 10)
        with open(os.path.join(OUT, "generated_RegisterCPU_0_excerpt.cpp"), "w") as fh:
            fh.write(f"// Excerpt of the generated RegisterCPU_0.cpp ({len(lines)} lines total),\n"
                     f"// produced locally by `python -m torchgen.gen`. Lines {lo + 1}-{hi}.\n"
                     f"// The dispatcher reported add.Tensor's CPU kernel at {runtime_where}\n\n")
            for i in range(lo, hi):
                fh.write(f"{i + 1:6d}  {lines[i]}\n")

ufunc = os.path.join(gen_dir, "UfuncCPUKernel_add.cpp")
if os.path.exists(ufunc):
    text = open(ufunc).read()
    F.note("UfuncCPUKernel_add.cpp lines", text.count("\n") + 1)
    F.note("mentions cpu_kernel_vec (vectorised path)", "cpu_kernel_vec" in text)
    F.note("dtype cases generated (AT_DISPATCH_CASE)", text.count("AT_DISPATCH_CASE"))
    shutil.copy(ufunc, os.path.join(OUT, "generated_UfuncCPUKernel_add.cpp"))


# ===========================================================================
# 4. Stage 2: the header tax
# ===========================================================================
F.head("4. the header tax")

work = tempfile.mkdtemp(prefix="p55-compile-")
HEADERS = [
    ("c10/core/TensorImpl.h", "the core tensor struct"),
    ("ATen/ops/add.h", "one operator"),
    ("ATen/ATen.h", "every operator"),
    ("torch/extension.h", "every operator + the Python glue"),
]
header_rows = []
for header, what in HEADERS:
    src = os.path.join(work, "probe.cpp")
    with open(src, "w") as fh:
        fh.write(f"#include <{header}>\nint probe_symbol = 0;\n")
    pre = run(["g++", "-E", "-std=c++17", *INCLUDES, src])
    n_lines = pre.stdout.count("\n")
    dep = run(["g++", "-MM", "-std=c++17", *INCLUDES, src])
    n_deps = dep.stdout.count("\\\n") if dep.returncode == 0 else -1
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        c = run(["g++", "-c", "-O2", "-std=c++17", *INCLUDES, src,
                 "-o", os.path.join(work, "probe.o")])
        best = min(best, time.perf_counter() - t0)
    obj = os.path.join(work, "probe.o")
    size = os.path.getsize(obj) / 1e3 if os.path.exists(obj) else -1
    header_rows.append((header, n_lines, n_deps, best, size))
    F.note(f"{header:26s} preprocessed lines", n_lines)
    F.note(f"{header:26s} header files pulled in", n_deps)
    F.note(f"{header:26s} compile seconds (empty file)", best)
    F.note(f"{header:26s} object file (KB)", size)

per_op = next(r for r in header_rows if r[0] == "ATen/ops/add.h")
everything = next(r for r in header_rows if r[0] == "ATen/ATen.h")
F.note("one operator vs all operators, compile (x)", everything[3] / per_op[3])
F.note("adding pybind11 costs (s)", header_rows[3][3] - everything[3])
F.note("adding pybind11 costs (x)", header_rows[3][3] / everything[3])


# ===========================================================================
# 5. Extrapolating the build
# ===========================================================================
F.head("5. how long would the whole thing take")

# The empty-file probes measure the FIXED cost (headers). Real files also have
# code, and code costs more per line than headers do. Measure that slope
# directly: compile the same file with 0, 250 and 1000 lines of real ATen calls.
slope_rows = []
for n in [0, 250, 1000]:
    body = "\n".join(f"  s = at::add(s, t, {i % 7} + 1);" for i in range(n))
    src = os.path.join(work, "body.cpp")
    with open(src, "w") as fh:
        fh.write("#include <ATen/ATen.h>\nat::Tensor f(at::Tensor s, at::Tensor t) {\n"
                 + body + "\n  return s;\n}\n")
    t0 = time.perf_counter()
    run(["g++", "-c", "-O2", "-std=c++17", *INCLUDES, src, "-o", os.path.join(work, "body.o")])
    secs = time.perf_counter() - t0
    obj_kb = os.path.getsize(os.path.join(work, "body.o")) / 1e3
    slope_rows.append((n, secs, obj_kb))
    F.note(f"{n:>5d} lines of at::add: compile seconds", secs)
    F.note(f"{n:>5d} lines of at::add: object KB", obj_kb)

per_line = (slope_rows[-1][1] - slope_rows[0][1]) / (slope_rows[-1][0] - slope_rows[0][0])
fixed = slope_rows[0][1]
F.note("fixed cost per file, headers only (s)", fixed)
F.note("marginal cost per line of real code (ms)", per_line * 1000)
F.note("lines of code equal to the header cost", int(fixed / per_line))

avg_lines = tu_lines / max(len(tus), 1)
F.note("average lines per generated TU", avg_lines)
est_tu = fixed + per_line * avg_lines
F.note("estimated seconds per generated TU", est_tu)
serial_hours = est_tu * len(tus) / 3600
F.note("generated TUs alone, one core (hours)", serial_hours)
F.note(f"generated TUs alone, {NPROC} cores (hours)", serial_hours / NPROC)
F.note("...and this is a LOWER bound", "the repo adds thousands of hand-written TUs, plus CUDA and third-party libs")


# ===========================================================================
# 6. -O0 vs -O2 vs -O3
# ===========================================================================
F.head("6. what optimisation costs and buys")

KERNEL = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

// A deliberately plain loop: no at::vec, no OpenMP. The compiler is the only
// thing that can make this fast, so the -O level shows up directly.
at::Tensor saxpy(at::Tensor x, at::Tensor y, double a) {
  auto out = at::empty_like(x);
  const float* xp = x.data_ptr<float>();
  const float* yp = y.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const float af = static_cast<float>(a);
  const int64_t n = x.numel();
  for (int64_t i = 0; i < n; ++i) op[i] = af * xp[i] + yp[i];
  return out;
}
"""

from torch.utils.cpp_extension import load_inline  # noqa: E402

x = torch.randn(1 << 20)
y = torch.randn(1 << 20)
opt_rows = []
for level in ["-O0", "-O2", "-O3"]:
    name = f"p55_saxpy_{level.replace('-', '')}"
    build_dir = tempfile.mkdtemp(prefix=f"p55-{level}-")
    t0 = time.perf_counter()
    mod = load_inline(name=name, cpp_sources=[KERNEL], functions=["saxpy"],
                      extra_cflags=[level], build_directory=build_dir, verbose=False)
    build_s = time.perf_counter() - t0
    timing = S.interleaved({"saxpy": lambda mod=mod: mod.saxpy(x, y, 2.0)},
                           rounds=5, calls=20)["saxpy"]["best"]
    so = os.path.join(build_dir, f"{name}.so")
    so_kb = os.path.getsize(so) / 1e3 if os.path.exists(so) else -1
    opt_rows.append((level, build_s, timing * 1e6, so_kb))
    F.note(f"{level} build seconds", build_s)
    F.note(f"{level} kernel microseconds", timing * 1e6)
    F.note(f"{level} .so size (KB)", so_kb)
    shutil.rmtree(build_dir, ignore_errors=True)

o0 = opt_rows[0]
o2 = opt_rows[1]
F.note("-O0 vs -O2: build time (x faster at -O0)", o2[1] / o0[1])
F.note("-O0 vs -O2: runtime (x slower at -O0)", o0[2] / o2[2])
F.note("torch's own build uses", "-O2 (see section 1)")


# ===========================================================================
# 7. Why an incremental rebuild is still slow
# ===========================================================================
F.head("7. incremental rebuilds")

# If you edit a header, every translation unit that includes it (directly or
# not) must be recompiled. The dependency count from section 4 IS that fan-out,
# measured from the other side.
F.note("headers read by a TU including ATen/ATen.h", everything[2])
F.note("headers read by a TU including ATen/ops/add.h", per_op[2])
F.note("reduction from per-operator headers (x)", everything[2] / max(per_op[2], 1))

# Which headers does everyone include? Count how many of the generated
# per-operator headers mention the central tensor header.
ops_dir = os.path.join(gen_dir, "ops")
central = 0
sampled = 0
if os.path.isdir(ops_dir):
    names = sorted(os.listdir(ops_dir))[:800]
    for fn in names:
        p = os.path.join(ops_dir, fn)
        if not fn.endswith(".h"):
            continue
        sampled += 1
        text = open(p, errors="ignore").read()
        if "TensorBody.h" in text or "ATen/core/Tensor.h" in text:
            central += 1
    F.note("per-operator headers sampled", sampled)
    F.note("  ... that pull in the central tensor header", central)
    F.note("  ... percentage", 100 * central / max(sampled, 1))

F.note("so editing TensorBody.h rebuilds", "essentially everything")
F.note("and editing ATen/ops/add.h rebuilds", "only what uses add")


# ===========================================================================
# 8. The real recipe
# ===========================================================================
F.head("8. the recipe, with predicted times")

recipe = [
    ("git clone --recursive https://github.com/pytorch/pytorch", "~10 min, ~3 GB"),
    ("pip install -r requirements.txt", "~2 min"),
    ("export USE_CUDA=0", "removes the single largest cost"),
    ("export MAX_JOBS=%d" % NPROC, "one compiler per core; each needs ~2 GB RAM"),
    ("export CMAKE_C_COMPILER_LAUNCHER=ccache", "second build ~10x faster"),
    ("export DEBUG=1  (optional)", "-O0: see section 6 for the trade"),
    ("python setup.py develop", "the build itself"),
]
for cmd, note in recipe:
    F.note(cmd, note)

F.note("measured stage 1 (codegen) on this machine (s)", gen_seconds)
F.note("estimated stage 2 (generated TUs) on this machine (h)", serial_hours / NPROC)
F.note("RAM on this machine (GB)", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
F.note("RAM needed for MAX_JOBS=%d at ~2GB each (GB)" % NPROC, 2 * NPROC)

shutil.rmtree(gen_dir, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)


# ===========================================================================
# figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=110)
fig.patch.set_facecolor("#fcfcfb")
axes = axes.ravel()
for ax in axes:
    style_axes(ax)

ax = axes[0]
labs = [r[0] for r in header_rows]
vals = [r[3] for r in header_rows]
ax.bar(range(len(labs)), vals, color=SERIES[0], width=0.6)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([l.replace("/", "/\n") for l in labs], fontsize=7)
ax.set_ylabel("seconds to compile an EMPTY file")
ax.set_title("1. the header tax", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

ax = axes[1]
ax.bar(range(len(labs)), [r[1] / 1000 for r in header_rows], color=SERIES[1], width=0.6)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([l.replace("/", "/\n") for l in labs], fontsize=7)
ax.set_ylabel("thousands of preprocessed lines")
ax.set_title("2. what the compiler actually reads", loc="left", fontsize=11)
for i, r in enumerate(header_rows):
    ax.text(i, r[1] / 1000, f"{r[1]/1000:.0f}k", ha="center", va="bottom", fontsize=8)

ax = axes[2]
lv = [r[0] for r in opt_rows]
xs = np.arange(len(lv))
ax.bar(xs - 0.2, [r[1] for r in opt_rows], width=0.38, color=SERIES[0], label="build seconds")
ax.bar(xs + 0.2, [r[2] / 100 for r in opt_rows], width=0.38, color=SERIES[2],
       label="kernel microseconds / 100")
ax.set_xticks(xs)
ax.set_xticklabels(lv)
ax.set_title("3. optimisation: pay now or pay later", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

ax = axes[3]
parts = {
    "codegen (measured)": gen_seconds / 60,
    f"generated TUs, {NPROC} cores (est.)": serial_hours * 60 / NPROC,
}
ax.barh(range(len(parts)), list(parts.values()), color=[SERIES[1], SERIES[3]], height=0.5)
ax.set_yticks(range(len(parts)))
ax.set_yticklabels(list(parts), fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("minutes")
ax.set_xlim(0, max(parts.values()) * 1.25)
ax.set_title("4. where the build time goes", loc="left", fontsize=11)
for i, v in enumerate(parts.values()):
    ax.text(v, i, f" {v:.1f} min", va="center", fontsize=8)

fig.tight_layout()
save(fig, os.path.join(OUT, "build_cost.png"))
F.write(os.path.join(OUT, "findings.csv"))
