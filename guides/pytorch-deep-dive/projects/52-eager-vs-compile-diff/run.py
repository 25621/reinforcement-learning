"""Project 52 - finding, and then judging, a difference between eager and compiled.

`torch.compile` is allowed to give you a different answer. Most of the time the
difference is one unit in the last place of a float32 and means nothing. Some of
the time it means everything. This project measures both kinds and, crucially,
builds the test that tells them apart.

Sections:
  1. the survey: which ops disagree, and by how much
  2. is that a lot? the float64 referee
  3. the control: PyTorch disagreeing with itself
  4. when 1e-7 is not small: decisions downstream
  5. the bisector: narrowing a whole model to one layer
  6. a difference that is not rounding: randomness
  7. what actually changed: reading the generated kernel
  8. the checklist

Run:  python3 run.py        (~6 minutes; most of it is compilation)
"""

from __future__ import annotations

import contextlib
import io
import re
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "48-nan-forensics"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import debug_lib as D  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F_ = D.Findings()


def compiled(fn):
    """`dynamic=False` so each shape gets its own specialised kernel.

    Left dynamic, Dynamo may compile a *shape-generic* kernel on the second call
    with a new shape, which is a different program with different arithmetic —
    and then "eager vs compiled" is quietly comparing three things, not two.
    """
    return torch.compile(fn, dynamic=False)


def rel_diff(a, b) -> float:
    scale = max(float(a.abs().max()), 1e-30)
    return float((a - b).abs().max()) / scale


# ===========================================================================
# 1. The survey
# ===========================================================================

F_.head("1. The survey: which ops disagree")

g = torch.Generator().manual_seed(0)
X = torch.randn(256, 1024, generator=g)
W = torch.randn(1024, 256, generator=g)

CASES = {
    "matmul":     (lambda t: t @ W, X),
    "add/mul":    (lambda t: t * 2.0 + 1.0, X),
    "gelu":       (lambda t: F.gelu(t), X),
    "sum(-1)":    (lambda t: t.sum(-1), X),
    "mean(-1)":   (lambda t: t.mean(-1), X),
    "var(-1)":    (lambda t: t.var(-1), X),
    "softmax":    (lambda t: torch.softmax(t, -1), X),
    "logsumexp":  (lambda t: t.logsumexp(-1), X),
    "layer_norm": (lambda t: F.layer_norm(t, (1024,)), X),
    "fused chain": (lambda t: (t.sigmoid() * t).sum(-1), X),
}

survey = {}
for name, (fn, arg) in CASES.items():
    e = fn(arg)
    c = compiled(fn)(arg)
    survey[name] = {"abs": float((e - c).abs().max()), "rel": rel_diff(e, c)}
    F_.note(name, f"max |eager - compiled| = {survey[name]['abs']:.3e}   "
                  f"relative {survey[name]['rel']:.2e}")

exact = [n for n, v in survey.items() if v["abs"] == 0.0]
F_.note("bit-identical ops", f"{len(exact)} of {len(survey)}: {', '.join(exact)}")
F_.note("largest relative disagreement",
        f"{max(v['rel'] for v in survey.values()):.2e}")
F_.note("float32 machine epsilon (2^-23)", float(torch.finfo(torch.float32).eps))
F_.note("so the disagreements are", "about one unit in the last place - a rounding "
                                    "difference, not a logic difference")


# ===========================================================================
# 2. The float64 referee
# ===========================================================================

F_.head("2. Is that a lot? Ask a more accurate calculator")

REFEREE = {
    "sum(-1)":    lambda t: t.sum(-1),
    "mean(-1)":   lambda t: t.mean(-1),
    "logsumexp":  lambda t: t.logsumexp(-1),
    "layer_norm": lambda t: F.layer_norm(t, (1024,)),
    "fused chain": lambda t: (t.sigmoid() * t).sum(-1),
}
referee_rows = []
for name, fn in REFEREE.items():
    truth = fn(X.double())                      # float64: ~15 digits, not 7
    e = fn(X).double()
    c = compiled(fn)(X).double()
    err_e = float((e - truth).abs().max())
    err_c = float((c - truth).abs().max())
    winner = "compiled" if err_c < err_e else ("eager" if err_e < err_c else "tie")
    referee_rows.append((name, err_e, err_c, winner))
    F_.note(name, f"eager err {err_e:.3e} | compiled err {err_c:.3e} | closer: {winner}")

wins_c = sum(1 for _, _, _, w in referee_rows if w == "compiled")
F_.note("cases where the COMPILED answer is closer to the truth",
        f"{wins_c} of {len(referee_rows)}")
F_.note("the test that settles 'is this a bug?'",
        "compute the same thing in float64 and see which one is wrong")


# ===========================================================================
# 3. The control: PyTorch disagreeing with itself
# ===========================================================================

F_.head("3. The control: eager vs eager")


def sum_last(t):
    return t.sum(-1)


base = sum_last(X)
ctrl = {}

big = torch.cat([X, X], 0)                      # same rows, bigger batch
ctrl["row sum, batch 256 vs 512"] = float((base - sum_last(big)[:256]).abs().max())

# A row-wise sum gives each row to one thread, so the thread count cannot change
# it. A sum over the WHOLE tensor is split across threads, and then it can.
flat4 = X.sum()
torch.set_num_threads(1)
flat1 = X.sum()
torch.set_num_threads(4)
ctrl["full sum, 1 thread vs 4"] = float((flat4 - flat1).abs())

nc = X[:, ::2].contiguous()
ctrl["contiguous vs strided view"] = float((sum_last(nc) - sum_last(X[:, ::2])).abs().max())

ctrl["eager vs compiled"] = survey["sum(-1)"]["abs"]

for k, v in ctrl.items():
    F_.note(f"max |difference|, {k}", f"{v:.3e}")
worst_self = max(v for k, v in ctrl.items() if k != "eager vs compiled")
F_.note("PyTorch's largest disagreement with ITSELF", f"{worst_self:.3e}")
F_.note("torch.compile's disagreement", f"{ctrl['eager vs compiled']:.3e}")
F_.note("ratio", f"{ctrl['eager vs compiled'] / max(worst_self, 1e-30):.2f}x")


# ===========================================================================
# 4. When 1e-7 is not small
# ===========================================================================

F_.head("4. When a rounding difference changes the answer")


gz = torch.Generator().manual_seed(1)
XB = torch.randn(4096, 1024, generator=gz)


def make_head(spread):
    """16 output classes whose weight vectors are `spread` apart.

    spread=1.0 is an ordinary classifier: the classes want different things and
    the winner wins by a mile. As spread shrinks the classes become near-copies
    of each other, the top two scores come within a rounding error, and the
    argmax is decided by the last bit. Real models look like this whenever the
    input is genuinely ambiguous.
    """
    gg = torch.Generator().manual_seed(2)
    base_w = torch.randn(1024, 1, generator=gg)
    return base_w + spread * torch.randn(1024, 16, generator=gg)


flip_rows = []
for spread in (1.0, 1e-2, 1e-4, 1e-6):
    Wh = make_head(spread)

    def head(t, Wh=Wh):
        return F.gelu(t) @ Wh

    e = head(XB)
    torch._dynamo.reset()
    c = compiled(head)(XB)
    top2 = e.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    flipped = (e.argmax(-1) != c.argmax(-1))
    flips = int(flipped.sum())
    flip_rows.append((spread, float((e - c).abs().max()), float(margin.median()),
                      flips, len(XB)))
    F_.note(f"class spread {spread:g}",
            f"max diff {float((e - c).abs().max()):.2e} | median top-2 margin "
            f"{float(margin.median()):.2e} | argmax flips {flips}/{len(XB)} "
            f"({100 * flips / len(XB):.2f}%)")

F_.note("flips when the classes are well separated", f"{flip_rows[0][3]} of 4096")
F_.note("flips when the top-2 margin is the size of the rounding difference",
        f"{flip_rows[-1][3]} of 4096")
F_.note("the rule", "a rounding difference matters exactly where something "
                    "downstream compares, rounds, sorts or branches - and only "
                    "when the thing being compared is closer together than the "
                    "difference")


# ===========================================================================
# 5. The bisector
# ===========================================================================

F_.head("5. Narrowing a model to one layer")


class Stack(nn.Module):
    """Six named stages. Real models are shaped like this, which is why
    bisecting by stage is the first thing to try."""

    def __init__(self, d=512):
        super().__init__()
        gg = torch.Generator().manual_seed(3)

        def lin(a, b):
            m = nn.Linear(a, b)
            with torch.no_grad():
                m.weight.copy_(torch.randn(b, a, generator=gg) * (1 / a) ** 0.5)
                m.bias.zero_()
            return m

        self.stage0 = lin(d, d)
        self.stage1 = nn.GELU()
        self.stage2 = nn.LayerNorm(d)
        self.stage3 = lin(d, d)
        self.stage4 = nn.Softmax(dim=-1)
        self.stage5 = lin(d, 16)

    def stages(self):
        return [self.stage0, self.stage1, self.stage2,
                self.stage3, self.stage4, self.stage5]

    def forward(self, x):
        for s in self.stages():
            x = s(x)
        return x


model = Stack().eval()
xin = torch.randn(128, 512, generator=torch.Generator().manual_seed(4))


def run_prefix(m, x, k, compile_prefix):
    """Run the first k stages (compiled or not), the rest always eager."""
    stages = m.stages()
    if compile_prefix and k:
        pre = nn.Sequential(*stages[:k])
        h = compiled(lambda t: pre(t))(x)
    else:
        h = x
        for s in stages[:k]:
            h = s(h)
    for s in stages[k:]:
        h = s(h)
    return h


with torch.no_grad():
    ref = model(xin)
    prefix_diffs = []
    for k in range(len(model.stages()) + 1):
        out = run_prefix(model, xin, k, compile_prefix=True)
        prefix_diffs.append(float((ref - out).abs().max()))
        F_.note(f"compile the first {k} stage(s)", f"max diff {prefix_diffs[-1]:.3e}")

jumps = [(k, prefix_diffs[k] - prefix_diffs[k - 1]) for k in range(1, len(prefix_diffs))]
culprit = max(jumps, key=lambda kv: kv[1])
STAGE_NAMES = ["Linear", "GELU", "LayerNorm", "Linear", "Softmax", "Linear"]
F_.note("largest jump when stage k is added to the compiled prefix",
        f"stage {culprit[0] - 1} ({STAGE_NAMES[culprit[0] - 1]}), "
        f"+{culprit[1]:.3e}")
F_.note("compiling the whole model", f"{prefix_diffs[-1]:.3e}")
F_.note("why prefix-bisection and not per-op",
        "the compiler FUSES ops, so an individual op's output no longer exists "
        "inside the compiled kernel - the smallest thing you can compare is a "
        "boundary you kept")


# ===========================================================================
# 6. A difference that is not rounding
# ===========================================================================

F_.head("6. Randomness: a real, large, correct-looking difference")


def with_dropout(t):
    return F.dropout(t, p=0.5, training=True)


torch.manual_seed(7)
e = with_dropout(X)
torch._dynamo.reset()
torch.manual_seed(7)
c = compiled(with_dropout)(X)
F_.note("max |eager - compiled| with dropout", f"{float((e - c).abs().max()):.4f}")
F_.note("fraction of elements that differ", f"{float(((e - c).abs() > 0).float().mean()):.4f}")
F_.note("for comparison, the largest rounding difference in section 1",
        f"{max(v['abs'] for v in survey.values()):.3e}")
F_.note("zeros in each", f"eager {float((e == 0).float().mean()):.3f}, "
                         f"compiled {float((c == 0).float().mean()):.3f}")
F_.note("so both are correct dropout", "they just drew different masks")

import torch._inductor.config as icfg  # noqa: E402

icfg.fallback_random = True
torch._dynamo.reset()
torch.manual_seed(7)
e2 = with_dropout(X)
torch.manual_seed(7)
c2 = compiled(with_dropout)(X)
F_.note("with torch._inductor.config.fallback_random = True",
        f"max |eager - compiled| = {float((e2 - c2).abs().max()):.4f}")
icfg.fallback_random = False
torch._dynamo.reset()
F_.note("what that flag does", "make the compiled code call the same RNG kernels "
                               "eager does, instead of its own faster ones")
F_.note("when you need it", "only when comparing against an eager baseline; it "
                            "costs speed and buys no correctness")


# ===========================================================================
# 7. What actually changed
# ===========================================================================

F_.head("7. Reading the generated kernel")

code_text = ""
try:
    import logging

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    # The generated source is logged by the codecache module, not by the graph
    # module. Attach to the shared `torch._inductor` parent so we get it wherever
    # inside inductor it is emitted.
    logger = logging.getLogger("torch._inductor")
    logger.addHandler(handler)
    torch._logging.set_logs(output_code=True)
    torch._dynamo.reset()
    compiled(lambda t: t.sum(-1))(X)
    torch._logging.set_logs()
    logger.removeHandler(handler)
    code_text = buf.getvalue()
except Exception as exc:  # noqa: BLE001
    code_text = f"(could not capture generated code: {exc})"

# The logger prefixes every line with a timestamp and a tag; strip it so the
# saved file is readable C++.
clean = "\n".join(re.sub(r"^.*\[__output_code\] ?", "", ln)
                  for ln in code_text.splitlines())
with open(os.path.join(OUT, "generated_kernel.txt"), "w") as fh:
    fh.write(clean or "(empty)")
F_.note("generated code captured", f"{len(clean)} characters")
vec = [ln.strip() for ln in clean.splitlines()
       if "at::vec" in ln or "#pragma omp" in ln]
F_.note("lines of vectorised / OpenMP C++ inductor wrote for a single sum", len(vec))
for label, needle in (("the parallel directive", "#pragma omp parallel"),
                      ("the vector accumulator", "Vectorized<float> tmp_acc0_vec"),
                      ("the horizontal reduce at the end", "vec_reduce_all")):
    hit = next((ln.strip() for ln in clean.splitlines() if needle in ln), "-")
    F_.note(label, hit[:110])
step = next((ln.strip() for ln in clean.splitlines()
             if "x1<" in ln and "+=" in ln), "-")
F_.note("the inner loop it generated", step[:110])
F_.note("what this shows", "inductor writes its own C++ loop, so the ORDER the "
                           "1024 numbers are added in is its choice, not eager's")

# Build the compiled callable ONCE. Calling `torch.compile(...)` inside the
# timed lambda would time the compiler's cache lookup, not the kernel - which is
# how the first version of this section reported the compiled sum as 150x
# slower than eager.
_csum = compiled(lambda t: t.sum(-1))
_csum(X)                                        # warm: the first call compiles
speed = D.interleaved({
    "eager sum": lambda: X.sum(-1),
    "compiled sum": lambda: _csum(X),
}, rounds=5, calls=50)
for k, v in speed.items():
    F_.note(f"{k}", f"{v['best'] * 1e6:.1f} us")


# ===========================================================================
# 8. The checklist
# ===========================================================================

F_.head("8. Deciding what you are looking at")

F_.note("step 1", "measure the RELATIVE difference, not the absolute one")
F_.note("step 2", "compare it to eager-vs-eager at another batch size or thread count")
F_.note("step 3", "compute the same thing in float64; whoever is closer is right")
F_.note("step 4", "if the difference is far above one ulp, suspect RNG, dtype "
                  "promotion, or a real bug - not fusion")
F_.note("step 5", "bisect by compiling prefixes of your model, not individual ops")
F_.note("verdict for every op measured here",
        "rounding, except dropout, which is a different random draw")


# ===========================================================================
# figures
# ===========================================================================

fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.0), dpi=110)
for ax in axes:
    style_axes(ax)
fig.patch.set_facecolor("#fcfcfb")

ax = axes[0]
names = list(survey)
vals = [max(survey[n]["rel"], 1e-12) for n in names]
ax.barh(range(len(names)), vals, color=SERIES[0], height=0.62)
ax.set_xscale("log")
ax.axvline(float(torch.finfo(torch.float32).eps), color=SERIES[2], ls="--", lw=1.2,
           label="float32 eps")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("relative |eager - compiled|")
ax.set_title("1. how far apart, per op", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False, loc="lower right")

ax = axes[1]
labs = [r[0] for r in referee_rows]
xs = np.arange(len(labs))
ax.bar(xs - 0.2, [max(r[1], 1e-12) for r in referee_rows], width=0.38,
       color=SERIES[0], label="eager error")
ax.bar(xs + 0.2, [max(r[2], 1e-12) for r in referee_rows], width=0.38,
       color=SERIES[1], label="compiled error")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([l.replace("(", "\n(") for l in labs], fontsize=7)
ax.set_ylabel("distance from the float64 answer")
ax.set_title("2. which one is actually wrong", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
ks = list(ctrl)
vals = [max(ctrl[k], 1e-12) for k in ks]
ax.barh(range(len(ks)), vals,
        color=[SERIES[1]] * (len(ks) - 1) + [SERIES[2]], height=0.6)
ax.set_xscale("log")
ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("max |difference| on the same sum")
ax.set_title("3. eager also disagrees with eager", loc="left", fontsize=11)

ax = axes[3]
ax.plot(range(len(prefix_diffs)), np.maximum(prefix_diffs, 1e-12), "o-",
        color=SERIES[0], lw=1.8, ms=6)
ax.set_yscale("log")
ax.set_xticks(range(len(prefix_diffs)))
ax.set_xlabel("stages compiled (0 = all eager)")
ax.set_ylabel("max |difference| from all-eager")
ax.set_title("4. the bisector finds the boundary", loc="left", fontsize=11)
for k, name in enumerate(STAGE_NAMES):
    ax.annotate(name, (k + 1, max(prefix_diffs[k + 1], 1e-12)), fontsize=7,
                xytext=(0, 6), textcoords="offset points", ha="center")

save(fig, os.path.join(OUT, "eager_vs_compile.png"))
F_.write(os.path.join(OUT, "findings.csv"))
