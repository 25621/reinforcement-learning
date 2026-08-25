"""Project 53 - following `torch.add` from a Python `+` down to the C++ kernel.

Everything here is measured against the *installed* PyTorch, using hooks the
library itself exposes: the dispatcher's debug dump, the tensor's dispatch key
set, and the code generator's own parse of `native_functions.yaml`. Nothing is
read off a blog post.

Sections:
  1. four ways to write it, one operator underneath
  2. overloads: which `add` did you call?
  3. the dispatch table, printed from the running library
  4. how the dispatcher chooses: the key set on the tensor
  5. what the source says: add.Tensor -> add.out -> ufunc_add_CPU
  6. where the maths happens: TensorIterator, seen through its side effects
  7. the tax: the same op called from C++, with no Python in the loop
  8. the map

Run:  python3 run.py        (~4 minutes; ~20 s of it is a C++ compile)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import source_lib as S  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F = S.Findings()


# ===========================================================================
# 1. Four ways to write it, one operator underneath
# ===========================================================================
F.head("1. four spellings, one operator")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class RecordAten(TorchDispatchMode):
    """Record every ATen operator that runs inside the `with` block.

    `TorchDispatchMode` is a hook the dispatcher offers to Python: before a
    kernel runs, the dispatcher calls this object with the operator it was
    about to run. It is the cheapest way to see the *real* operator name behind
    a piece of Python — no C++ debugger, no rebuild.
    """

    def __init__(self):
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


a = torch.randn(1000)
b = torch.randn(1000)

spellings = {
    "a + b": lambda: a + b,
    "a.add(b)": lambda: a.add(b),
    "torch.add(a, b)": lambda: torch.add(a, b),
    "torch.ops.aten.add.Tensor(a, b)": lambda: torch.ops.aten.add.Tensor(a, b),
}

for label, fn in spellings.items():
    with RecordAten() as rec:
        fn()
    F.note(f"{label:32s} -> aten op", ",".join(rec.ops))

timing = S.interleaved(spellings, rounds=7, calls=400)
for label in spellings:
    F.note(f"{label:32s} per call (us)", timing[label]["best"] * 1e6)

fastest = min(timing, key=lambda k: timing[k]["best"])
slowest = max(timing, key=lambda k: timing[k]["best"])
F.note("spread across spellings (x)", timing[slowest]["best"] / timing[fastest]["best"])


# ===========================================================================
# 2. Overloads: which `add` did you call?
# ===========================================================================
F.head("2. overload resolution")

schemas = torch._C._jit_get_schemas_for_operator("aten::add")
for sch in schemas:
    F.note("schema", str(sch))

cases = {
    "tensor + tensor": (torch.randn(4), torch.randn(4)),
    "tensor + python float": (torch.randn(4), 2.5),
    "tensor + 0-dim tensor": (torch.randn(4), torch.tensor(2.5)),
}
for label, (x, y) in cases.items():
    with RecordAten() as rec:
        x + y
    F.note(f"{label:24s} dispatches to", ",".join(rec.ops))

# add.Scalar exists, but Python's `+` never picks it: the binding wraps the
# number into a 0-dim tensor first. You can still reach it by name.
with RecordAten() as rec:
    torch.ops.aten.add.Scalar(torch.randn(4), 2.5)
F.note("aten.add.Scalar called by name", ",".join(rec.ops))


# ===========================================================================
# 3. The dispatch table, printed from the running library
# ===========================================================================
F.head("3. the dispatch table for aten::add.Tensor")

dump = S.dispatch_table("aten::add.Tensor")
with open(os.path.join(OUT, "dispatch_add_tensor.txt"), "w") as fh:
    fh.write(dump)
rows = S.dispatch_rows("aten::add.Tensor")
F.note("registered kernels (active)", len(rows))
for key, where in rows:
    F.note(f"  key {key}", where)

generated = sum(1 for _, w in rows if "/build/" in w or "generated" in w)
F.note("kernels living in generated files", f"{generated}/{len(rows)}")
python_kernels = [(k, w) for k, w in rows if w.endswith(".py") or ".py:" in w]
F.note("kernels registered from PYTHON, not C++", len(python_kernels))
for k, w in python_kernels:
    F.note(f"  python kernel for key {k}", w)


# ===========================================================================
# 4. How the dispatcher chooses: the key set on the tensor
# ===========================================================================
F.head("4. the key set decides")

plain = torch.randn(4)
grad = torch.randn(4, requires_grad=True)
meta = torch.randn(4, device="meta")

F.note("plain tensor keys", " ".join(S.key_set(plain)))
F.note("requires_grad=True keys", " ".join(S.key_set(grad)))
F.note("meta tensor keys", " ".join(S.key_set(meta)))

for key in ["CPU", "Autograd", "AutogradCPU", "Meta", "CUDA", "NestedTensorCPU"]:
    F.note(
        f"aten::add.Tensor has a kernel for {key}",
        torch._C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", key),
    )

x1 = torch.randn(1000)
x2 = torch.randn(1000, requires_grad=True)
F.note("grad_fn built by the autograd kernel", type(x2.add(x2).grad_fn).__name__)


def no_grad_add():
    with torch.no_grad():
        return x2 + x2


grad_cost = S.interleaved(
    {
        "requires_grad=False": lambda: x1 + x1,
        "requires_grad=True": lambda: x2 + x2,
        "requires_grad=True, no_grad()": no_grad_add,
    },
    rounds=7,
    calls=400,
)
no_ag = grad_cost["requires_grad=False"]["best"] * 1e6
ag = grad_cost["requires_grad=True"]["best"] * 1e6
ng = grad_cost["requires_grad=True, no_grad()"]["best"] * 1e6
F.note("add without autograd (us)", no_ag)
F.note("add with autograd (us)", ag)
F.note("add under no_grad() (us)", ng)
F.note("cost of the autograd hop (us)", ag - no_ag)
F.note("autograd overhead (x)", ag / no_ag)
F.note("no_grad() around each call costs (us)", ng - ag)

# Same comparison with `no_grad()` entered *once*, outside the timing loop.
# Both numbers below are measured inside one `with` block, so they are
# comparable with each other (and only roughly with the three above).
with torch.no_grad():
    hoisted = S.interleaved(
        {"plain": lambda: x1 + x1, "requires_grad=True": lambda: x2 + x2},
        rounds=7,
        calls=400,
    )
F.note("inside no_grad(): plain (us)", hoisted["plain"]["best"] * 1e6)
F.note("inside no_grad(): requires_grad=True (us)",
       hoisted["requires_grad=True"]["best"] * 1e6)

with RecordAten() as rec_meta:
    meta + meta
F.note("meta tensor add reaches", ",".join(rec_meta.ops))
F.note("meta result has storage bytes", (meta + meta).untyped_storage().nbytes())


# ===========================================================================
# 5. What the source says
# ===========================================================================
F.head("5. native_functions.yaml on add")

from torchgen.model import DispatchKey  # noqa: E402

by_name = S.native_functions_by_name()
_, backend_indices = S.native_functions()

add_tensor = by_name["add.Tensor"]
add_out = by_name["add.out"]

F.note("add.Tensor signature", str(add_tensor.func))
F.note("add.Tensor structured_delegate", str(add_tensor.structured_delegate))
F.note("add.Tensor own CPU kernel", str(backend_indices[DispatchKey.CPU].get_kernel(add_tensor)))
F.note("add.Tensor tags", " ".join(sorted(str(t) for t in add_tensor.tags)))
F.note("add.out signature", str(add_out.func))
F.note("add.out structured", add_out.structured)
cpu_meta = backend_indices[DispatchKey.CPU].get_kernel(add_out)
cuda_meta = backend_indices[DispatchKey.CUDA].get_kernel(add_out)
F.note("add.out CPU kernel name", cpu_meta.kernel)
F.note("add.out CUDA kernel name", cuda_meta.kernel)

# The generated declaration header ships inside the wheel. It is the C++
# signature the generator produced from the YAML entry above.
native_h = os.path.join(S.torch_root(), "include", "ATen", "ops", "add_native.h")
if os.path.exists(native_h):
    text = open(native_h).read()
    keep = [ln for ln in text.splitlines() if "ufunc_add" in ln or "structured_add" in ln]
    with open(os.path.join(OUT, "add_native_h_excerpt.txt"), "w") as fh:
        fh.write("\n".join(keep) + "\n")
    F.note("add_native.h lines mentioning the ufunc", len(keep))
    for ln in keep[:4]:
        F.note("  decl", ln.strip())


# ===========================================================================
# 6. Where the maths happens: TensorIterator, seen through side effects
# ===========================================================================
F.head("6. TensorIterator fingerprints")

# (a) broadcasting: no copy is made, so a (1000,1) + (1,1000) costs about as
#     much as writing 1000x1000 outputs, not as much as materialising inputs.
r = torch.randn(1000, 1)
c = torch.randn(1, 1000)
F.note("broadcast (1000,1)+(1,1000) result shape", tuple((r + c).shape))

# (b) type promotion: the iterator computes a common dtype before running.
promo = {
    "float32 + float64": (torch.randn(4), torch.randn(4, dtype=torch.float64)),
    "int64 + float32": (torch.ones(4, dtype=torch.int64), torch.randn(4)),
    "bool + int64": (torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.int64)),
}
for label, (u, v) in promo.items():
    F.note(f"{label:20s} -> result dtype", str((u + v).dtype))

# (c) memory layout: the same op on the same number of elements, laid out three
#     ways. TensorIterator reorders and coalesces dimensions when it can.
n = 512
base = torch.randn(n, n)
other = torch.randn(n, n)
layout = S.interleaved(
    {
        "contiguous": lambda: base + other,
        "transposed (both)": lambda: base.t() + other.t(),
        "transposed (one)": lambda: base + other.t(),
        "strided (every 2nd)": lambda: base[:, ::2] + other[:, ::2],
    },
    rounds=5,
    calls=20,
)
for label in layout:
    F.note(f"{label:22s} (us)", layout[label]["best"] * 1e6)
F.note(
    "one-transposed / both-transposed (x)",
    layout["transposed (one)"]["best"] / layout["transposed (both)"]["best"],
)

# (d) the size sweep: a flat region (fixed per-call cost) then a linear region
#     (memory bandwidth). The knee is where dispatch stops dominating.
sizes = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
sweep = []
for numel in sizes:
    u = torch.randn(numel)
    v = torch.randn(numel)
    calls = max(4, min(400, int(2_000_000 / max(numel, 1))))
    t = S.interleaved({"add": lambda u=u, v=v: u + v}, rounds=5, calls=calls)
    sweep.append(t["add"]["best"])
for numel, t in zip(sizes, sweep):
    F.note(f"add on {numel:>8d} elements (us)", t * 1e6)
# The fixed cost is the *plateau*, not the single smallest measurement: on a
# shared machine one sample can be 60% high, and reading the plateau off one
# point would inherit that noise. Median of everything below 1024 elements.
flat_region = [t * 1e6 for numel, t in zip(sizes, sweep) if numel <= 512]
fixed = float(np.median(flat_region))
biggest = sizes[-1]
gbps = (3 * biggest * 4) / sweep[-1] / 1e9  # read a, read b, write out
F.note("fixed cost per call (us)", fixed)
F.note("elements bought by that fixed cost", int(biggest * fixed / (sweep[-1] * 1e6)))
F.note("effective bandwidth at 4M elements (GB/s)", gbps)


# ===========================================================================
# 7. The tax: the same op with no Python in the loop
# ===========================================================================
F.head("7. calling at::add from C++")

CPP = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <chrono>

// Call at::add in a tight C++ loop. `at::add` is the same entry point the
// Python binding calls, so this measures the dispatcher without Python's
// argument parsing, and without the interpreter.
double add_loop(at::Tensor a, at::Tensor b, int64_t iters) {
  at::Tensor out = a + b;   // warm up allocation paths
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < iters; ++i) {
    out = at::add(a, b);
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / iters;
}

// Identical to add_loop except the previous result is released *before* the
// next one is allocated. That one line decides whether the caching allocator
// can hand the same block back, or has to keep two large buffers alive.
double add_loop_free(at::Tensor a, at::Tensor b, int64_t iters) {
  at::Tensor out = a + b;
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < iters; ++i) {
    out.reset();
    out = at::add(a, b);
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / iters;
}

// Same loop, but reusing the output buffer, so no allocation happens per call.
double add_out_loop(at::Tensor a, at::Tensor b, int64_t iters) {
  at::Tensor out = at::empty_like(a);
  at::add_out(out, a, b);
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < iters; ++i) {
    at::add_out(out, a, b);
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / iters;
}
"""

from torch.utils.cpp_extension import load_inline  # noqa: E402

cache = os.path.join(
    os.path.expanduser("~"), ".cache", "torch_extensions",
    f"py{sys.version_info.major}{sys.version_info.minor}_cu{torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'}",
    "p53_addloop",
)
cold = not os.path.exists(os.path.join(cache, "p53_addloop.so"))
t0 = time.perf_counter()
ext = load_inline(
    name="p53_addloop",
    cpp_sources=[CPP],
    functions=["add_loop", "add_loop_free", "add_out_loop"],
    extra_cflags=["-O2"],
    verbose=False,
)
F.note("C++ extension build was cold", cold)
F.note("C++ extension build time (s)", time.perf_counter() - t0)

small_a, small_b = torch.randn(1000), torch.randn(1000)
py_small = S.interleaved({"py": lambda: small_a + small_b}, rounds=7, calls=400)["py"]["best"]
cpp_small = min(ext.add_loop(small_a, small_b, 20000) for _ in range(5))
cpp_out_small = min(ext.add_out_loop(small_a, small_b, 20000) for _ in range(5))
F.note("python  a+b, 1000 elements (us)", py_small * 1e6)
F.note("C++ at::add, 1000 elements (us)", cpp_small * 1e6)
F.note("C++ at::add_out, 1000 elements (us)", cpp_out_small * 1e6)
F.note("python-side tax (us)", (py_small - cpp_small) * 1e6)
F.note("python-side tax (% of call)", 100 * (py_small - cpp_small) / py_small)
F.note("allocation cost per call (us)", (cpp_small - cpp_out_small) * 1e6)

big_a, big_b = torch.randn(4_194_304), torch.randn(4_194_304)
py_big = S.interleaved({"py": lambda: big_a + big_b}, rounds=5, calls=8)["py"]["best"]
cpp_big = min(ext.add_loop(big_a, big_b, 40) for _ in range(5))
cpp_big_free = min(ext.add_loop_free(big_a, big_b, 40) for _ in range(5))
F.note("python  a+b, 4M elements (us)", py_big * 1e6)
F.note("C++ at::add, 4M elements (us)", cpp_big * 1e6)
F.note("C++ at::add, previous result freed first (us)", cpp_big_free * 1e6)
F.note("cost of holding the old buffer (x)", cpp_big / cpp_big_free)
F.note("python-side tax at 4M (% of call)", 100 * (py_big - cpp_big_free) / py_big)


# ===========================================================================
# 8. The map
# ===========================================================================
F.head("8. the whole path")

path = [
    ("Python", "a + b", "torch/_tensor.py -> C binding"),
    ("C binding", "THPVariable_add", "torch/csrc/autograd/generated/python_variable_methods.cpp"),
    ("Op lookup", "aten::add.Tensor", "the schema string in native_functions.yaml"),
    ("Dispatcher", "highest key in the tensor's key set", "c10/core/DispatchKeySet.h"),
    ("Autograd", "AutogradCPU: record AddBackward0, redispatch", "torch/csrc/autograd/generated/VariableType_2.cpp"),
    ("Backend", "CPU: the generated wrapper", dict(rows).get("CPU", "RegisterCPU_*.cpp")),
    ("Structured", "add.out via structured_delegate", "aten/src/ATen/native/BinaryOps.cpp"),
    ("Kernel", cpu_meta.kernel, "generated from aten/src/ATen/native/ufunc/add.h"),
    ("Loop", "TensorIterator drives the elementwise loop", "aten/src/ATen/TensorIterator.cpp"),
]
for stage, what, where in path:
    F.note(f"{stage:11s} {what}", where)
with open(os.path.join(OUT, "call_path.txt"), "w") as fh:
    for stage, what, where in path:
        fh.write(f"{stage:11s} | {what}\n{'':11s} | {where}\n")


# ===========================================================================
# figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=110)
fig.patch.set_facecolor("#fcfcfb")
axes = axes.ravel()
for ax in axes:
    style_axes(ax)

ax = axes[0]
labels = list(spellings)
vals = [timing[k]["best"] * 1e6 for k in labels]
ax.barh(range(len(labels)), vals, color=SERIES[0], height=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([l.replace("(a, b)", "") for l in labels], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("microseconds per call")
ax.set_title("1. four spellings, one kernel", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)

ax = axes[1]
ax.loglog(sizes, [t * 1e6 for t in sweep], "o-", color=SERIES[0], lw=1.8, ms=5,
          label="measured")
flat = fixed
ax.axhline(flat, color=SERIES[2], ls="--", lw=1.2, label=f"fixed cost {flat:.2f} us")
ax.set_xlabel("elements")
ax.set_ylabel("microseconds per call")
ax.set_title("2. flat = dispatch, sloped = memory", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
bars = {
    "C++ at::add_out": cpp_out_small * 1e6,
    "C++ at::add": cpp_small * 1e6,
    "python a+b": py_small * 1e6,
    "python (requires_grad)": ag,
}
ax.barh(range(len(bars)), list(bars.values()),
        color=[SERIES[1], SERIES[1], SERIES[0], SERIES[3]], height=0.6)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels(list(bars), fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("microseconds per call (1000 elements)")
ax.set_title("3. what each layer costs", loc="left", fontsize=11)
for i, v in enumerate(bars.values()):
    ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)

ax = axes[3]
labs = list(layout)
vals = [layout[k]["best"] * 1e6 for k in labs]
ax.bar(range(len(labs)), vals, color=SERIES[4], width=0.6)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([l.replace(" (", "\n(") for l in labs], fontsize=8)
ax.set_ylabel("microseconds per call")
ax.set_title("4. TensorIterator and memory layout (512x512)", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
save(fig, os.path.join(OUT, "trace_one_op.png"))
F.write(os.path.join(OUT, "findings.csv"))
