"""Project 57 - hunting a real bug in the installed PyTorch, then fixing it.

Nothing here is a staged bug. The sweep runs against the wheel you installed,
using PyTorch's own operator test database, and every survivor is a real
behaviour of the library as shipped.

The finale of the phase, so it uses all four earlier projects: project 53's
dispatch table to find kernels, project 54's `native_functions.yaml` to explain
*why* the bug exists, project 55's build knowledge to know where the fix would
go, and project 56's `TORCH_LIBRARY_IMPL` to apply and validate the fix without
a rebuild.

Sections:
  1. the hunt, part 1: two properties, 697 operators
  2. triage: turning 55 failures into 0 bugs
  3. the hunt, part 2: the out= contract
  4. the root cause, read off native_functions.yaml
  5. testing the explanation at scale (and watching it half-fail)
  6. search before you file - in the right place
  7. the fix, written and validated
  8. the test, in PyTorch's own style

Run:  python3 run.py        (~4 minutes; ~20 s of it is a C++ compile)
"""

from __future__ import annotations

import io
import os
import sys
import time
import unittest
import warnings
from collections import Counter

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)
warnings.filterwarnings("ignore", message=".*beta.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "53-trace-one-op-end-to-end"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import source_lib as S  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F = S.Findings()

t_import = time.perf_counter()
from torch.testing._internal.common_methods_invocations import op_db  # noqa: E402

F.head("0. the test database that ships with your wheel")
F.note("op_db import seconds", time.perf_counter() - t_import)
F.note("OpInfo entries", len(op_db))
F.note("distinct operators covered", len({o.name for o in op_db}))
F.note("entries claiming out= support", sum(1 for o in op_db if o.supports_out))


# ===========================================================================
# 1. The hunt, part 1
# ===========================================================================
F.head("1. two properties, every operator")

# A "property" is something that must be true of a correct implementation, for
# every input, without knowing the right answer. That is what makes this kind of
# testing cheap: no expected outputs to write down.
#
# Property A: running an op on the `meta` device must predict the same shape and
#   dtype as running it for real. Meta tensors carry shape and dtype but no
#   data, so this is a pure "does the shape rule agree with the arithmetic" test.
# Property B: an op must give the same answer on a non-contiguous input as on a
#   contiguous copy of the same values. Memory layout is not part of the maths.


def to_meta(x):
    return x.to("meta") if isinstance(x, torch.Tensor) else x


def make_noncontiguous(t: torch.Tensor):
    """A tensor with the same values but a stride that skips every other element."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0 or t.dim() == 0:
        return None
    if t.layout is not torch.strided:
        return None
    big = torch.empty([2 * s for s in t.shape], dtype=t.dtype, device=t.device)
    view = big[tuple(slice(0, 2 * s, 2) for s in t.shape)]
    view.copy_(t)
    return view if not view.is_contiguous() else None


meta_fails, contig_fails = [], []
meta_checked = contig_checked = harness_skips = 0
t0 = time.perf_counter()
for oi in op_db:
    try:
        samples = list(oi.sample_inputs("cpu", torch.float32))
    except Exception:
        harness_skips += 1
        continue
    for s in samples[:1]:
        if not isinstance(s.input, torch.Tensor):
            break
        try:
            ref = oi.op(s.input, *s.args, **s.kwargs)
        except Exception:
            harness_skips += 1
            break
        if not isinstance(ref, torch.Tensor):
            break

        # Property A
        meta_checked += 1
        try:
            got = oi.op(to_meta(s.input), *[to_meta(a) for a in s.args],
                        **{k: to_meta(v) for k, v in s.kwargs.items()})
            if isinstance(got, torch.Tensor):
                if got.shape != ref.shape:
                    meta_fails.append((oi.name, "shape", str(tuple(ref.shape)),
                                       str(tuple(got.shape))))
                elif got.dtype != ref.dtype:
                    meta_fails.append((oi.name, "dtype", str(ref.dtype), str(got.dtype)))
        except Exception as exc:
            meta_fails.append((oi.name, "raises", type(exc).__name__, str(exc)[:70]))

        # Property B
        nc = make_noncontiguous(s.input)
        if nc is not None:
            try:
                other = oi.op(nc, *s.args, **s.kwargs)
            except Exception:
                other = None
            if isinstance(other, torch.Tensor):
                contig_checked += 1
                try:
                    torch.testing.assert_close(ref, other, rtol=0, atol=0)
                except Exception:
                    if ref.shape == other.shape:
                        d = float((ref.double() - other.double()).abs().max())
                        scale = max(float(ref.double().abs().max()), 1e-30)
                        contig_fails.append((oi.name, d, d / scale))
                    else:
                        contig_fails.append((oi.name, -1.0, -1.0))
        break

F.note("sweep seconds", time.perf_counter() - t0)
F.note("property A (meta agrees) checks", meta_checked)
F.note("property A failures", len(meta_fails))
F.note("property B (layout invariance) checks", contig_checked)
F.note("property B failures", len(contig_fails))
F.note("samples the harness could not build", harness_skips)
raw_failures = len(meta_fails) + len(contig_fails)
F.note("RAW failures to triage", raw_failures)


# ===========================================================================
# 2. Triage
# ===========================================================================
F.head("2. triage: which failures are real")

# Bucket A: ops whose output shape genuinely depends on the DATA. A meta tensor
# has no data, so no shape rule can exist. This is a documented limitation, not
# a bug - `nonzero` cannot know how many non-zeros there are.
data_dependent = {str(f.func.name.name.base) for f in S.native_functions()[0]
                  if any(str(t) in ("data_dependent_output", "dynamic_output_shape")
                         for t in f.tags)}
F.note("ops tagged data-dependent in native_functions.yaml", len(data_dependent))

buckets = Counter()
survivors_meta = []
for name, kind, a, b in meta_fails:
    base = name.split(".")[-1]
    if name in data_dependent or base in data_dependent or "item()" in str(b):
        buckets["data-dependent output (by design)"] += 1
    elif "sparse" in name.lower():
        buckets["sparse layouts (out of scope)"] += 1
    elif kind == "raises" and a == "NotImplementedError":
        buckets["no meta kernel written yet"] += 1
    elif kind == "raises":
        buckets["our harness passed a bad argument"] += 1
    else:
        buckets["SURVIVOR"] += 1
        survivors_meta.append((name, kind, a, b))

nondet = {str(f.func.name.name.base) for f in S.native_functions()[0]
          if any(str(t) == "nondeterministic_seeded" for t in f.tags)}
survivors_contig = []
for name, absolute, relative in contig_fails:
    base = name.split(".")[-1]
    if base in nondet or name in nondet:
        buckets["documented randomness"] += 1
    elif name in ("as_strided", "as_strided_copy", "as_strided_scatter"):
        buckets["stride IS the argument"] += 1
    elif absolute != absolute:  # NaN
        buckets["NaN vs NaN: our comparator's default"] += 1
    elif 0 <= relative < 1e-5:
        buckets["float rounding (different kernel path)"] += 1
    else:
        buckets["SURVIVOR"] += 1
        survivors_contig.append((name, absolute, relative))

for k, v in buckets.most_common():
    F.note(f"bucket: {k}", v)
F.note("survivors after triage", buckets["SURVIVOR"])
for s in (survivors_meta + survivors_contig)[:10]:
    F.note("  survivor", str(s))

worst = sorted([c for c in contig_fails if c[2] == c[2] and c[2] >= 0],
               key=lambda c: -c[2])[:5]
for name, absolute, relative in worst:
    F.note(f"largest layout differences: {name}", f"abs={absolute:.3g} rel={relative:.3g}")


# ===========================================================================
# 3. The hunt, part 2: the out= contract
# ===========================================================================
F.head("3. the out= contract")

# PyTorch documents this rule: if you pass an `out=` tensor of the wrong shape
# and it is not empty, the op resizes it AND warns you, because silently
# throwing away a buffer you supplied is how data is lost.
RESIZE_MSG = "An output with one or more elements"

out_rows = []
for oi in op_db:
    if not oi.supports_out:
        continue
    try:
        samples = list(oi.sample_inputs("cpu", torch.float32))
    except Exception:
        continue
    for s in samples[:1]:
        try:
            ref = oi.op(s.input, *s.args, **s.kwargs)
        except Exception:
            break
        if not isinstance(ref, torch.Tensor) or ref.numel() == 0:
            break
        # PyTorch's own test perturbs only the LAST dimension; copy that exactly,
        # so a difference cannot be blamed on a different test.
        wrong = list(ref.shape)
        wrong = [2] if not wrong else wrong[:-1] + [wrong[-1] + 1]
        bad = torch.empty(wrong, dtype=ref.dtype)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                oi.op(s.input, *s.args, out=bad, **s.kwargs)
            warned = any(RESIZE_MSG in str(m.message) for m in w)
        except Exception:
            break
        out_rows.append((oi.aten_name or oi.name, oi.name, warned))
        break

silent = sorted({r[0] for r in out_rows if not r[2]})
F.note("operators tested for the out= rule", len(out_rows))
F.note("that warn as documented", sum(1 for r in out_rows if r[2]))
F.note("that resize SILENTLY", len(silent))
for name in silent:
    F.note("  silent", name)

# The minimal reproduction - the thing an issue report is actually made of.
repro = """import torch, warnings
x = torch.randn(20)
out = torch.empty(21)                      # wrong shape, NOT empty
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.nn.functional.logsigmoid(x, out=out)
print(out.shape, [str(m.message)[:40] for m in w])
# torch.Size([20]) []      <- resized, no warning
torch.sigmoid(x, out=torch.empty(21))      # the sibling op DOES warn
"""
with open(os.path.join(OUT, "repro.py"), "w") as fh:
    fh.write(repro)

x = torch.randn(20)
pairs = [
    ("F.logsigmoid", lambda o: torch.nn.functional.logsigmoid(x, out=o)),
    ("torch.sigmoid", lambda o: torch.sigmoid(x, out=o)),
    ("torch.tanh", lambda o: torch.tanh(x, out=o)),
    ("torch.threshold", lambda o: torch.threshold(x, 0, 0, out=o)),
]
for label, fn in pairs:
    o = torch.empty(21)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn(o)
    F.note(f"{label:16s} warns on resize", any(RESIZE_MSG in str(m.message) for m in w))


# ===========================================================================
# 4. The root cause, from native_functions.yaml
# ===========================================================================
F.head("4. why: structured vs unstructured")

for name in ["log_sigmoid.out", "sigmoid.out", "tanh.out", "threshold.out",
             "addbmm.out", "baddbmm.out", "narrow_copy.out", "add.out"]:
    F.note(f"{name:18s} structured", str(S.is_structured(name)))

fn_by_name = S.native_functions_by_name()
lsf = fn_by_name.get("log_sigmoid_forward.output")
if lsf is not None:
    from torchgen.model import DispatchKey

    _, bi = S.native_functions()
    k = bi[DispatchKey.CPU].get_kernel(lsf)
    F.note("the kernel that actually resizes", k.kernel if k else "-")
    F.note("its declaration lives in", "torch/include/ATen/ops/log_sigmoid_forward_native.h")


# ===========================================================================
# 5. Testing the explanation at scale
# ===========================================================================
F.head("5. does 'unstructured' predict 'silent'?")

pred_rows = []
for aten_name, op_name, warned in out_rows:
    structured = S.is_structured(f"{aten_name}.out")
    if structured is None:
        continue
    pred_rows.append((aten_name, structured, warned))

matrix = Counter((p, w) for _, p, w in pred_rows)
F.note("ops with an entry we can read", len(pred_rows))
F.note("structured AND warns", matrix[(True, True)])
F.note("structured AND silent", matrix[(True, False)])
F.note("unstructured AND warns", matrix[(False, True)])
F.note("unstructured AND silent", matrix[(False, False)])
acc = sum(1 for _, p, w in pred_rows if p == w) / max(len(pred_rows), 1)
F.note("accuracy of 'structured <=> warns'", acc)
if matrix[(True, True)] + matrix[(True, False)]:
    F.note("P(warns | structured)",
           matrix[(True, True)] / (matrix[(True, True)] + matrix[(True, False)]))
if matrix[(False, True)] + matrix[(False, False)]:
    F.note("P(silent | unstructured)",
           matrix[(False, False)] / (matrix[(False, True)] + matrix[(False, False)]))


# ===========================================================================
# 6. Search before you file
# ===========================================================================
F.head("6. the codebase already tracks this")

known = {}
for oi in op_db:
    for skip in oi.skips:
        name = str(getattr(skip, "test_name", None))
        if name == "test_out_warning":
            known.setdefault(oi.aten_name or oi.name, oi.name)
F.note("ops marked as expected failures on test_out_warning", len(known))
F.note("  the list", ", ".join(sorted(known)))

found = set(silent)
tracked = set(known)
F.note("ops our sweep found silent", len(found))
F.note("of those, already tracked", len(found & tracked))
F.note("precision against the maintainers' own list", len(found & tracked) / max(len(found), 1))
F.note("tracked but not found by our sweep", len(tracked - found))
F.note("  (our sweep tests one sample per op, and skips some shapes)",
       ", ".join(sorted(tracked - found)[:8]))

all_skips = Counter()
for oi in op_db:
    for skip in oi.skips:
        all_skips[str(getattr(skip, "test_name", None))] += 1
F.note("total skip/xfail entries in op_db", sum(all_skips.values()))
F.note("distinct tests they cover", len(all_skips))
for name, n in all_skips.most_common(6):
    F.note(f"  {name}", n)
with open(os.path.join(OUT, "good_first_issue_menu.txt"), "w") as fh:
    for name, n in all_skips.most_common():
        fh.write(f"{n:5d}  {name}\n")


# ===========================================================================
# 7. The fix
# ===========================================================================
F.head("7. writing and validating the fix")

FIX_CPP = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/Resize.h>

// The one-line fix, in the form the real patch would take.
//
// The shipped kernel (aten/src/ATen/native/Activation.cpp) resizes its output
// with `result.resize_as_(input)`, which is silent. `at::native::resize_output`
// does the same resize AND emits the documented warning when the output was
// non-empty. Everything else is unchanged: the arithmetic still comes from the
// stock log_sigmoid_forward kernel.
at::Tensor& fixed_log_sigmoid_out(const at::Tensor& self, at::Tensor& out) {
  at::native::resize_output(out, self.sizes());
  at::Tensor buffer = at::empty({0}, self.options());
  at::log_sigmoid_forward_out(out, buffer, self);
  return out;
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("log_sigmoid.out", TORCH_FN(fixed_log_sigmoid_out));
}
"""


def out_warning_case(op, sample_shape=(20,), wrong_extra=1):
    """PyTorch's own test_out_warning "Case Zero", reproduced exactly."""
    x = torch.randn(*sample_shape)
    expected = op(x)
    wrong = list(expected.shape)
    wrong[-1] += wrong_extra
    out = torch.empty(wrong, dtype=expected.dtype)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = op(x, out=out)
    warned = any(RESIZE_MSG in str(m.message) for m in w)
    correct = torch.equal(got, expected)
    return warned, correct


before_warn, before_correct = out_warning_case(torch.nn.functional.logsigmoid)
F.note("before fix: warns", before_warn)
F.note("before fix: value correct", before_correct)

from torch.utils.cpp_extension import load_inline  # noqa: E402

t0 = time.perf_counter()
fix = load_inline(name="p57_fix", cpp_sources=[FIX_CPP], functions=[],
                  extra_cflags=["-O2"], verbose=False)
F.note("fix build seconds", time.perf_counter() - t0)
F.note("the CPU kernel is now registered at",
       dict(S.dispatch_rows("aten::log_sigmoid.out")).get("CPU", "-"))

after_warn, after_correct = out_warning_case(torch.nn.functional.logsigmoid)
F.note("after fix: warns", after_warn)
F.note("after fix: value correct", after_correct)
F.note("fix works", after_warn and after_correct and not before_warn)

# Regression check: the ops that already behaved must keep behaving.
regressions = []
for label, fn in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh)]:
    w_, c_ = out_warning_case(fn)
    if not (w_ and c_):
        regressions.append(label)
F.note("sibling ops still correct", not regressions)

# And the empty-output case must stay quiet - resizing an EMPTY out is normal.
o = torch.empty(0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.nn.functional.logsigmoid(torch.randn(20), out=o)
F.note("empty out= still resizes without warning", not any(RESIZE_MSG in str(m.message) for m in w))
F.note("  and produces the right shape", tuple(o.shape))


# ===========================================================================
# 8. The test, in PyTorch's own style
# ===========================================================================
F.head("8. the test that would go in the pull request")


class TestLogSigmoidOutWarning(unittest.TestCase):
    """The test a reviewer would ask for, using PyTorch's own assertion."""

    def test_out_warning_on_resize(self):
        x = torch.randn(20)
        out = torch.empty(21)
        with self.assertWarnsRegex(UserWarning, RESIZE_MSG):
            torch.nn.functional.logsigmoid(x, out=out)
        self.assertEqual(out.shape, torch.Size([20]))
        torch.testing.assert_close(out, torch.nn.functional.logsigmoid(x))

    def test_no_warning_when_out_is_empty(self):
        x = torch.randn(20)
        out = torch.empty(0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.nn.functional.logsigmoid(x, out=out)
        self.assertFalse(any(RESIZE_MSG in str(m.message) for m in w))


stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
    unittest.TestLoader().loadTestsFromTestCase(TestLogSigmoidOutWarning)
)
F.note("tests run", result.testsRun)
F.note("failures", len(result.failures))
F.note("errors", len(result.errors))
F.note("all pass with the fix loaded", result.wasSuccessful())
with open(os.path.join(OUT, "test_output.txt"), "w") as fh:
    fh.write(stream.getvalue())

issue = f"""# log_sigmoid's out= variant resizes silently

## Summary

`torch.nn.functional.logsigmoid(x, out=y)` resizes a non-empty `y` without
emitting the documented resize warning. Sibling unary ops (`sigmoid`, `tanh`,
`threshold`) all warn.

## Repro

```python
{repro}```

## Expected

A `UserWarning` starting "An output with one or more elements was resized",
as produced by `torch.sigmoid(x, out=torch.empty(21))`.

## Cause

`log_sigmoid.out` is not a structured kernel, so it resizes its output by hand.
`log_sigmoid_forward_out_cpu` in `aten/src/ATen/native/Activation.cpp` calls
`result.resize_as_(input)`, which is silent, instead of
`at::native::resize_output(result, input.sizes())`, which warns.

## Already tracked

`nn.functional.logsigmoid`'s OpInfo carries an `expectedFailure` for
`TestCommon.test_out_warning`. {len(known)} operators carry that marker.

## Environment

torch {torch.__version__}, built from {torch.version.git_version[:12]}, CPU.

## Same sweep, other operators affected

{", ".join(silent)}
"""
with open(os.path.join(OUT, "issue.md"), "w") as fh:
    fh.write(issue)
F.note("issue report written", "outputs/issue.md")


# ===========================================================================
# figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), dpi=110)
fig.patch.set_facecolor("#fcfcfb")
axes = axes.ravel()
for ax in axes:
    style_axes(ax)

ax = axes[0]
items = [(k, v) for k, v in buckets.most_common()]
ax.barh(range(len(items)), [v for _, v in items],
        color=[SERIES[2] if k == "SURVIVOR" else SERIES[0] for k, _ in items], height=0.6)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([k.replace(" (", "\n(") for k, _ in items], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("failures")
ax.set_title(f"1. {raw_failures} raw failures, triaged", loc="left", fontsize=11)
for i, (_, v) in enumerate(items):
    ax.text(v, i, f" {v}", va="center", fontsize=8)

ax = axes[1]
vals = [sum(1 for r in out_rows if r[2]), len(silent)]
ax.bar(range(2), vals, color=[SERIES[1], SERIES[2]], width=0.6)
ax.set_xticks(range(2))
ax.set_xticklabels(["warns\n(as documented)", "resizes silently"], fontsize=9)
ax.set_ylabel("operators")
ax.set_title("2. the out= contract, tested", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

ax = axes[2]
cells = np.array([[matrix[(True, True)], matrix[(True, False)]],
                  [matrix[(False, True)], matrix[(False, False)]]])
ax.imshow(cells, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cells[i, j]), ha="center", va="center", fontsize=13,
                color="white" if cells[i, j] > cells.max() / 2 else "#0b0b0b")
ax.set_xticks([0, 1]); ax.set_xticklabels(["warns", "silent"], fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["structured", "unstructured"], fontsize=9)
ax.set_title("3. structured predicts one direction only", loc="left", fontsize=11)
ax.grid(False)

ax = axes[3]
top = all_skips.most_common(8)
ax.barh([t[0] for t in top], [t[1] for t in top], color=SERIES[4], height=0.6)
ax.invert_yaxis()
ax.tick_params(labelsize=7)
ax.set_xlabel("operators marked as known-failing")
ax.set_title(f"4. the menu: {sum(all_skips.values())} entries across {len(all_skips)} tests",
             loc="left", fontsize=11)

fig.tight_layout()
save(fig, os.path.join(OUT, "good_first_issue.png"))
F.write(os.path.join(OUT, "findings.csv"))
