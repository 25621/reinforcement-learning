"""Project 56 - patching a kernel and watching your own printf fire.

The classic version of this exercise edits a CUDA kernel and rebuilds PyTorch.
Neither half is available here: this machine's GPU is too old for the installed
wheel, and a full rebuild takes hours (project 55 measured why). So this project
does the same thing through the front door the dispatcher already provides -
`TORCH_LIBRARY_IMPL`, the same macro PyTorch's own generated code uses to
register every kernel you saw in project 53.

The result is stronger than the original exercise, not weaker: the edit-compile-
run loop takes 20 seconds instead of 2 hours, and the mechanism is identical.

Sections:
  1. baselines, measured before anything is patched
  2. route 1: patching from Python, with no compiler at all
  3. route 2: patching in C++, and watching printf fire
  4. proof: the dispatch table now points at your file
  5. the payoff: a kernel-level call counter that Python cannot build
  6. the price: your kernel is slower than theirs, and by how much
  7. the danger: the same patch at a different key silently breaks autograd
  8. the loop, and doing it for real

Run:  python3 run.py        (~3 minutes; ~40 s of it is two C++ compiles)
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "53-trace-one-op-end-to-end"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import source_lib as S  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from torch.utils.cpp_extension import load_inline  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F = S.Findings()


def cache_path(name: str) -> str:
    cu = torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    return os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions",
                        f"py{sys.version_info.major}{sys.version_info.minor}_cu{cu}",
                        name, f"{name}.so")


# ===========================================================================
# 1. Baselines, measured BEFORE anything is patched
# ===========================================================================
F.head("1. baselines (unpatched)")

a = torch.randn(1000)
b = torch.randn(1000)
reference = (a + b).clone()

F.note("stock CPU kernel registered at", dict(S.dispatch_rows("aten::add.Tensor"))["CPU"])
base_small = S.interleaved({"add": lambda: a + b}, rounds=7, calls=400)["add"]["best"]
F.note("stock add, 1000 elements (us)", base_small * 1e6)

big = torch.randn(1 << 20)
big2 = torch.randn(1 << 20)
base_big = S.interleaved({"add": lambda: big + big2}, rounds=5, calls=20)["add"]["best"]
F.note("stock add, 1M elements (us)", base_big * 1e6)


class TinyNet(nn.Module):
    """Small MLP with an explicit residual add, so `add` is used more than once."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = h + torch.relu(self.fc2(h))  # residual: one visible `add`
        return self.fc3(h)


torch.manual_seed(0)
net = TinyNet()
xb = torch.randn(32, 64)
yb = torch.randint(0, 10, (32,))
opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)


def train_steps(n=20):
    for _ in range(n):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xb), yb)
        loss.backward()
        opt.step()
    return float(loss)


base_loss = train_steps(5)
F.note("training loss after 5 warmup steps", base_loss)

# What a Python-level instrument can see, for comparison in section 5.
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class CountAten(TorchDispatchMode):
    def __init__(self):
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "add.Tensor" in str(func):
            self.n += 1
        return func(*args, **(kwargs or {}))


with CountAten() as counter:
    train_steps(20)
python_seen = counter.n
F.note("add.Tensor calls seen by TorchDispatchMode (20 steps)", python_seen)


# ===========================================================================
# 2. Route 1: patching from Python
# ===========================================================================
F.head("2. patching from Python")

py_calls = {"n": 0}
lib = torch.library.Library("aten", "IMPL")


def python_add(self, other, alpha=1):
    """Our replacement for the CPU kernel of aten::add.Tensor.

    It must not call `torch.add`: that would re-enter the dispatcher, arrive at
    this same kernel, and recurse forever. `add.out` is a *different* operator
    with its own CPU kernel, so calling it is safe - and it is exactly what the
    stock `add.Tensor` does anyway (project 53, section 5).
    """
    py_calls["n"] += 1
    out = torch.empty(0, dtype=self.dtype, device=self.device)
    return torch.ops.aten.add.out(self, other, alpha=alpha, out=out)


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    lib.impl("add.Tensor", python_add, "CPU")
F.note("registering produced a warning", len(w) > 0)
if w:
    text = str(w[0].message)
    F.note("warning mentions 'Overriding'", "Overriding" in text)
    F.note("warning names our file", os.path.basename(__file__) in text)

patched_result = a + b
F.note("our python kernel ran", py_calls["n"])
F.note("result still correct", bool(torch.equal(patched_result, reference)))
py_small = S.interleaved({"add": lambda: a + b}, rounds=5, calls=200)["add"]["best"]
F.note("python-patched add, 1000 elements (us)", py_small * 1e6)
F.note("slowdown vs stock (x)", py_small / base_small)

# Put the stock kernel back. Destroying the Library object removes every
# registration it made - which is why a Python patch is safe to experiment with.
lib._destroy()
after = a + b
F.note("after _destroy, result correct", bool(torch.equal(after, reference)))
restored = S.interleaved({"add": lambda: a + b}, rounds=5, calls=200)["add"]["best"]
F.note("restored add, 1000 elements (us)", restored * 1e6)


# ===========================================================================
# 3. Route 2: patching in C++
# ===========================================================================
F.head("3. patching in C++")

CPP = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdio>

static int64_t g_calls = 0;
static int64_t g_elements = 0;
static bool g_verbose = true;

// Our replacement for the CPU kernel of aten::add.Tensor.
//
// For the common case (same shape, float32, contiguous) it runs a plain loop so
// that section 6 can measure what the stock kernel's vectorisation and
// threading are worth. Everything else falls back to at::add_out, which is a
// DIFFERENT operator, so there is no recursion.
at::Tensor patched_add(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha) {
  g_calls += 1;
  g_elements += self.numel();
  if (g_verbose && g_calls <= 3) {
    printf("[patched C++ kernel] aten::add.Tensor call #%ld, numel=%ld\n",
           (long)g_calls, (long)self.numel());
    // Without this line the message appears at the END of the program: C's
    // stdout buffer is flushed at exit, while Python's print() has already
    // gone out. Two buffers, one terminal.
    fflush(stdout);
  }
  const bool simple = self.scalar_type() == at::kFloat &&
                      other.scalar_type() == at::kFloat &&
                      self.sizes() == other.sizes() &&
                      self.is_contiguous() && other.is_contiguous();
  if (!simple) {
    at::Tensor out = at::empty({0}, self.options());
    return at::add_out(out, self, other, alpha);
  }
  at::Tensor out = at::empty_like(self);
  const float* sp = self.data_ptr<float>();
  const float* op = other.data_ptr<float>();
  float* rp = out.data_ptr<float>();
  const float al = alpha.to<float>();
  const int64_t n = self.numel();
  for (int64_t i = 0; i < n; ++i) rp[i] = sp[i] + al * op[i];
  return out;
}

int64_t calls()    { return g_calls; }
int64_t elements() { return g_elements; }
void reset()       { g_calls = 0; g_elements = 0; }
void quiet()       { g_verbose = false; }

// This macro is the whole mechanism. It is the same one the generated file
// RegisterCPU_0.cpp uses - see project 55, which regenerated that file.
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(patched_add));
}
"""

cold = not os.path.exists(cache_path("p56_patch"))
t0 = time.perf_counter()
ext = load_inline(
    name="p56_patch",
    cpp_sources=[CPP],
    functions=["calls", "elements", "reset", "quiet"],
    extra_cflags=["-O2"],
    verbose=False,
)
build_s = time.perf_counter() - t0
F.note("C++ build was cold", cold)
F.note("C++ build seconds", build_s)

sys.stdout.flush()
ext.reset()
patched = a + b
F.note("printf fired for the first calls", "see stdout above")
F.note("our C++ kernel ran", ext.calls())
F.note("result still correct", bool(torch.equal(patched, reference)))
F.note("max |difference| from stock", float((patched - reference).abs().max()))

# Autograd still works, because we patched BELOW the autograd key.
xg = torch.randn(8, requires_grad=True)
yg = torch.randn(8)
(xg + yg).sum().backward()
F.note("gradient still flows through our kernel", bool(torch.equal(xg.grad, torch.ones(8))))
F.note("grad_fn still built", type((xg + yg).grad_fn).__name__)
ext.quiet()


# ===========================================================================
# 4. Proof: the dispatch table points at your file
# ===========================================================================
F.head("4. the dispatch table now names your file")

rows = dict(S.dispatch_rows("aten::add.Tensor"))
F.note("CPU kernel is now registered at", rows["CPU"])
F.note("it is our build directory", "torch_extensions" in rows["CPU"])
F.note("CUDA kernel is untouched", rows.get("CUDA", "-"))
with open(os.path.join(OUT, "dispatch_after_patch.txt"), "w") as fh:
    fh.write(S.dispatch_table("aten::add.Tensor"))


# ===========================================================================
# 5. The payoff: counting what Python cannot see
# ===========================================================================
F.head("5. two instruments, two blind spots")

# (a) The kernel-level counter is PASSIVE: the program runs exactly as it would
#     without it. Run the same 20 steps three ways and compare.
ext.reset()
train_steps(20)
cpp_alone = ext.calls()
F.note("kernel counter alone, 20 steps", cpp_alone)

ext.reset()
with CountAten() as both:
    train_steps(20)
cpp_with_mode = ext.calls()
F.note("kernel counter while a TorchDispatchMode is installed", cpp_with_mode)
F.note("TorchDispatchMode's own count, same run", both.n)
F.note("extra ADDS CAUSED BY the python instrument", cpp_with_mode - cpp_alone)
F.note("inflation factor", cpp_with_mode / max(cpp_alone, 1))
F.note("elements added in those 20 steps", ext.elements())

# Attribute them: one step, counted in three phases, with no mode installed.
ext.reset()
opt.zero_grad()
loss = nn.functional.cross_entropy(net(xb), yb)
forward_calls = ext.calls()
loss.backward()
backward_calls = ext.calls() - forward_calls
opt.step()
step_calls = ext.calls() - forward_calls - backward_calls
F.note("adds during forward (no mode)", forward_calls)
F.note("adds during backward (no mode)", backward_calls)
F.note("adds during optimizer.step() (no mode)", step_calls)

ext.reset()
with CountAten():
    opt.zero_grad()
    loss = nn.functional.cross_entropy(net(xb), yb)
    fwd_mode = ext.calls()
    loss.backward()
    bwd_mode = ext.calls() - fwd_mode
    opt.step()
F.note("adds during forward (mode installed)", fwd_mode)
F.note("adds during backward (mode installed)", bwd_mode)

# (b) The other direction: a call made from inside a C++ kernel never returns to
#     Python, so a TorchDispatchMode cannot see it. Only the kernel patch can.
u = torch.randn(64, 8)
v = torch.randn(64, 8)
probes = {
    "u + v": lambda: u + v,
    "torch.cdist(u, v)": lambda: torch.cdist(u, v),
    "F.pairwise_distance(u, v)": lambda: nn.functional.pairwise_distance(u, v),
    "torch.logaddexp(u, v)": lambda: torch.logaddexp(u, v),
    "torch.trapezoid(u)": lambda: torch.trapezoid(u),
}
blind_rows = []
for label, fn in probes.items():
    ext.reset()
    with CountAten() as pc:
        fn()
    blind_rows.append((label, pc.n, ext.calls()))
    F.note(f"{label:26s} python sees / kernel sees", f"{pc.n} / {ext.calls()}")
hidden = [r for r in blind_rows if r[2] > r[1]]
F.note("calls visible only to the kernel patch", len(hidden))
for r in hidden:
    F.note("  invisible to python", r[0])


# ===========================================================================
# 6. The price: your kernel vs theirs
# ===========================================================================
F.head("6. what the stock kernel was doing for you")

ext.reset()
patched_small = S.interleaved({"add": lambda: a + b}, rounds=7, calls=400)["add"]["best"]
patched_big = S.interleaved({"add": lambda: big + big2}, rounds=5, calls=20)["add"]["best"]
F.note("patched add, 1000 elements (us)", patched_small * 1e6)
F.note("patched add, 1M elements (us)", patched_big * 1e6)
F.note("slowdown at 1000 elements (x)", patched_small / base_small)
F.note("slowdown at 1M elements (x)", patched_big / base_big)
F.note("stock GB/s at 1M", (3 * (1 << 20) * 4) / base_big / 1e9)
F.note("patched GB/s at 1M", (3 * (1 << 20) * 4) / patched_big / 1e9)


# ===========================================================================
# 7. The danger: the same patch at a different key
# ===========================================================================
F.head("7. one word changed, autograd gone")

CPP_AUTOGRAD = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

// Byte for byte the same kernel as before, except for the last three lines:
// it is registered at the Autograd key instead of the CPU key.
//
// The autograd kernel's real job is to record a node in the graph and then
// redispatch. This one only redispatches. Nothing errors; the arithmetic is
// still right; the graph simply never gets built.
at::Tensor forgetful_add(const at::Tensor& self, const at::Tensor& other,
                         const at::Scalar& alpha) {
  at::AutoDispatchBelowADInplaceOrView guard;
  c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
  return at::add(self, other, alpha);
}

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("add.Tensor", TORCH_FN(forgetful_add));
}
"""

before_grad_fn = type((xg + yg).grad_fn).__name__
F.note("before: grad_fn", before_grad_fn)

cold2 = not os.path.exists(cache_path("p56_autograd"))
t0 = time.perf_counter()
ext2 = load_inline(name="p56_autograd", cpp_sources=[CPP_AUTOGRAD], functions=[],
                   extra_cflags=["-O2"], verbose=False)
F.note("second C++ build was cold", cold2)
F.note("second C++ build seconds", time.perf_counter() - t0)

z = xg + yg
F.note("after: grad_fn", str(type(z.grad_fn).__name__) if z.grad_fn else "None")
F.note("after: requires_grad on the result", bool(z.requires_grad))
F.note("values still correct", bool(torch.allclose(z, xg.detach() + yg)))
try:
    z.sum().backward()
    broke = False
except RuntimeError as exc:
    broke = True
    F.note("backward now raises", type(exc).__name__)
    F.note("  message", str(exc).split("\n")[0][:90])
F.note("backward broke", broke)

# And the failure is silent for anything that does not need THAT gradient.
w_ = torch.randn(4, 4, requires_grad=True)
out = (w_ @ torch.randn(4, 4)).sum()
out.backward()
F.note("gradients unaffected for ops we did not patch", w_.grad is not None)


# ===========================================================================
# 8. The loop, and doing it for real
# ===========================================================================
F.head("8. the edit-compile-run loop")

F.note("this project's C++ build (s)", build_s)
F.note("  was that a cold build", cold)
F.note("  (a warm build reuses ~/.cache/torch_extensions and takes <1 s)", not cold)
F.note("a full PyTorch rebuild (project 55 estimate, h)", "1+")
if cold:
    F.note("speedup of patching over rebuilding (x)", int(3600 / max(build_s, 1e-9)))
F.note("CUDA printf possible on this machine", torch.cuda.is_available())
F.note("  reason", "the installed wheel supports sm_70+; this GPU is sm_61")

for step, what in [
    ("1. find the kernel", "torch._C._dispatch_dump(op) -> file:line (project 53)"),
    ("2. copy its signature", "torch/include/ATen/ops/<op>_native.h"),
    ("3. write the replacement", "same signature, your body"),
    ("4. register it", "TORCH_LIBRARY_IMPL(aten, <Key>, m) { m.impl(...); }"),
    ("5. build", "load_inline(...) - seconds, not hours"),
    ("6. verify", "dispatch_dump again: the path is now yours"),
]:
    F.note(step, what)


# ===========================================================================
# figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=110)
fig.patch.set_facecolor("#fcfcfb")
axes = axes.ravel()
for ax in axes:
    style_axes(ax)

ax = axes[0]
labs = ["stock", "python patch", "C++ patch"]
vals = [base_small * 1e6, py_small * 1e6, patched_small * 1e6]
ax.bar(range(3), vals, color=[SERIES[1], SERIES[3], SERIES[0]], width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("microseconds per call")
ax.set_title("1. cost of a patched add (1000 elements)", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax = axes[1]
xs = np.arange(2)
ax.bar(xs - 0.2, [base_small * 1e6, base_big * 1e6], width=0.38, color=SERIES[1], label="stock")
ax.bar(xs + 0.2, [patched_small * 1e6, patched_big * 1e6], width=0.38, color=SERIES[0],
       label="our plain C loop")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["1,000 elements", "1,048,576 elements"], fontsize=9)
ax.set_ylabel("microseconds per call")
ax.set_title("2. what vectorisation and threads were worth", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
vals = [cpp_alone, cpp_with_mode]
ax.bar(range(2), vals, color=[SERIES[3], SERIES[0]], width=0.6)
ax.set_xticks(range(2))
ax.set_xticklabels(["program alone", "with a Python\ninstrument installed"], fontsize=8)
ax.set_ylabel("add.Tensor calls the KERNEL saw, 20 steps")
ax.set_title("3. measuring changed the program", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

ax = axes[3]
parts = {"forward": forward_calls, "backward": backward_calls, "optimizer.step()": step_calls}
ax.barh(range(len(parts)), list(parts.values()),
        color=[SERIES[0], SERIES[1], SERIES[4]], height=0.55)
ax.set_yticks(range(len(parts)))
ax.set_yticklabels(list(parts), fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("add.Tensor calls in ONE training step (no mode)")
ax.set_title("4. where the adds actually happen", loc="left", fontsize=11)
for i, v in enumerate(parts.values()):
    ax.text(v, i, f" {v}", va="center", fontsize=8)

fig.tight_layout()
save(fig, os.path.join(OUT, "patch_and_rebuild.png"))
F.write(os.path.join(OUT, "findings.csv"))
