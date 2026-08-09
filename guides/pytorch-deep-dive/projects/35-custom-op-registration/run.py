"""Project 35 -- registering a kernel so torch.compile can use it.

Project 30 ended with a working C++ kernel that autograd ignored and project 33
ended with one that made a model faster. Both are still second-class citizens:
`torch.compile` cannot see inside them, and `.backward()` does not know they
exist. This project fixes that with `torch.library.custom_op`, and measures
what each piece of the registration is worth.

Sections
  1. the kernel, as a plain pybind function
  2. what torch.compile does with it (a graph break)
  3. registering it as a custom op, and what changes
  4. the fake ("meta") implementation: what needs it and what it costs
  5. autograd: register_autograd, and checking the gradient
  6. opcheck, and the mutation bug it catches
  7. end to end: a small MLP, compiled, with and without registration
"""

import contextlib
import io
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "30-cpp-extension-for-elementwise-add"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import kernels_lib as K  # noqa: E402

OUT = K.outputs_dir(__file__)
ROWS = []
B, H = 2048, 2048


AVX2_FLAGS = ["-DCPU_CAPABILITY_AVX2", "-mavx2", "-mfma"]

CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>

using Vec = at::vec::Vectorized<float>;

static inline float gelu_scalar(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// out = gelu(h + bias), in one pass, 8 floats at a time. This is project 33's
// winning kernel: 3.4x faster than the two torch ops it replaces. Using the
// fast version here matters, because section 7 asks whether registration is
// worth it -- and that question is only meaningful if the kernel is worth it.
torch::Tensor bias_gelu(torch::Tensor h, torch::Tensor bias) {
  TORCH_CHECK(h.is_contiguous() && h.scalar_type() == torch::kFloat);
  TORCH_CHECK(bias.numel() == h.size(1));
  auto out = torch::empty_like(h);
  const int64_t rows = h.size(0), cols = h.size(1);
  const float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  float* po = out.data_ptr<float>();
  const int64_t W = Vec::size();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* hr = ph + r * cols;
      float* orow = po + r * cols;
      int64_t c = 0;
      for (; c + W <= cols; c += W) {
        Vec z = Vec::loadu(hr + c) + Vec::loadu(pb + c);
        Vec y = z * Vec(0.5f) * (Vec(1.f) + (z * Vec(0.70710678118654752440f)).erf());
        y.store(orow + c);
      }
      for (; c < cols; ++c) orow[c] = gelu_scalar(hr[c] + pb[c]);
    }
  });
  return out;
}

// The same thing, writing over h. Needed for section 6: an op that MUTATES an
// input has to say so, and this one will "forget" to.
torch::Tensor bias_gelu_inplace(torch::Tensor h, torch::Tensor bias) {
  const int64_t rows = h.size(0), cols = h.size(1);
  float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r)
      for (int64_t c = 0; c < cols; ++c)
        ph[r * cols + c] = gelu_scalar(ph[r * cols + c] + pb[c]);
  });
  return h;
}

// The backward pass, also in C++.
//   y   = gelu(z),  z = h + bias
//   dy/dz = 0.5*(1+erf(z/sqrt2)) + z * exp(-z^2/2) / sqrt(2*pi)
// grad_h is that, times the incoming gradient; grad_bias sums grad_h down
// the rows, because one bias value is added to every row.
std::vector<torch::Tensor> bias_gelu_backward(torch::Tensor grad, torch::Tensor h,
                                              torch::Tensor bias) {
  auto gh = torch::empty_like(h);
  auto gb = torch::zeros({h.size(1)}, h.options());
  const int64_t rows = h.size(0), cols = h.size(1);
  const float* pg = grad.contiguous().data_ptr<float>();
  const float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  float* pgh = gh.data_ptr<float>();
  float* pgb = gb.data_ptr<float>();
  const float inv_sqrt2 = 0.70710678118654752440f;
  const float inv_sqrt2pi = 0.39894228040143267794f;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const float z = ph[r * cols + c] + pb[c];
      const float d = 0.5f * (1.f + std::erf(z * inv_sqrt2))
                    + z * std::exp(-0.5f * z * z) * inv_sqrt2pi;
      const float g = pg[r * cols + c] * d;
      pgh[r * cols + c] = g;
      pgb[c] += g;
    }
  return {gh, gb};
}
"""


def compile_once(fn, *args):
    """Compile from a clean Dynamo state and report what it captured.

    torch._dynamo.reset() matters here for the same reason it mattered in
    project 29: Dynamo caches compiled code per Python code object, so a
    previous variant's result leaks into the next one's measurement.
    """
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(fn)
    with contextlib.redirect_stdout(io.StringIO()):
        out = compiled(*args)
    unimpl = torch._dynamo.utils.counters["unimplemented"]
    stats = torch._dynamo.utils.counters["stats"]
    return compiled, out, {
        "breaks": sum(unimpl.values()),
        "reasons": list(unimpl.keys()),
        "ops": stats.get("calls_captured", 0),
        "graphs": stats.get("unique_graphs", 0),
    }


def build_all():
    K.banner("[1] the kernel, as a plain pybind function")
    mod, s = K.build("p35_ops", CPP, extra_cflags=AVX2_FLAGS,
                     functions=["bias_gelu", "bias_gelu_inplace", "bias_gelu_backward"])
    h = torch.randn(4, 8)
    b = torch.randn(8)
    err = K.max_abs_diff(mod.bias_gelu(h, b), F.gelu(h + b))
    print(f"    compiled in {s:.1f} s; matches F.gelu(h + b) to {err:.2e}")
    print(f"    type: {type(mod.bias_gelu)} -- a plain Python callable, nothing more")
    ROWS.append(["compile", "p35_ops", f"{s:.1f}", "s"])
    ROWS.append(["correctness", "pybind kernel vs F.gelu", f"{err:.2e}", ""])
    return mod


def without_registration(mod):
    K.banner("[2] what torch.compile does with an unregistered kernel")
    h = torch.randn(256, 128)
    b = torch.randn(128)

    def with_torch_ops(h, b):
        return F.gelu(h + b) * 2.0

    def with_our_kernel(h, b):
        return mod.bias_gelu(h, b) * 2.0

    _, _, a = compile_once(with_torch_ops, h, b)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, c = compile_once(with_our_kernel, h, b)
    text = "\n\n".join(str(w.message) for w in caught)
    (OUT / "dynamo_warning.txt").write_text(text + "\n")
    print(f"    {'function':<34}{'ops captured':>14}{'graphs':>8}{'breaks':>8}")
    print(f"    {'pure torch ops':<34}{a['ops']:>14}{a['graphs']:>8}{a['breaks']:>8}")
    print(f"    {'our pybind kernel':<34}{c['ops']:>14}{c['graphs']:>8}{c['breaks']:>8}")
    print("\n    The kernel call is simply not in the graph: 3 operations become 1,")
    print("    and the missing one is ours. Dynamo says so itself (the full text is")
    print("    in outputs/dynamo_warning.txt):\n")
    for line in text.splitlines()[:3]:
        print(f"      {line[:88]}")
    ROWS.append(["unregistered", "dynamo warning",
                 text.splitlines()[0][:70] if text else "(none)", ""])
    ROWS.append(["unregistered", "pure torch ops captured", str(a["ops"]), f"{a['breaks']} breaks"])
    ROWS.append(["unregistered", "pybind kernel ops captured", str(c["ops"]),
                 f"{c['breaks']} breaks"])

    print("\n    -- autograd --")
    hh = torch.randn(8, 4, requires_grad=True)
    bb = torch.randn(4, requires_grad=True)
    y = mod.bias_gelu(hh, bb)
    print(f"    output requires_grad: {y.requires_grad}, grad_fn: {y.grad_fn}")
    try:
        y.sum().backward()
        msg = "backward ran"
    except RuntimeError as e:
        msg = "RuntimeError: " + str(e).split("\n")[0][:70]
    print(f"    {msg}")
    ROWS.append(["unregistered", "autograd", str(y.grad_fn), msg[:60]])
    return a, c


def register(mod):
    K.banner("[3] registering it properly")

    @torch.library.custom_op("p35::bias_gelu", mutates_args=())
    def bias_gelu(h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return mod.bias_gelu(h, bias)

    # The fake ("meta") implementation. It never touches data -- it only says
    # what SHAPE and DTYPE come out. torch.compile traces with fake tensors that
    # carry metadata but no values, so without this it cannot know the output
    # shape and refuses to trace the op.
    @bias_gelu.register_fake
    def _(h, bias):
        return torch.empty_like(h)

    def backward(ctx, grad):
        h, bias = ctx.saved
        gh, gb = torch.ops.p35.bias_gelu_backward(grad, h, bias)
        return gh, gb

    def setup_context(ctx, inputs, output):
        ctx.saved = inputs

    @torch.library.custom_op("p35::bias_gelu_backward", mutates_args=())
    def bias_gelu_backward(grad: torch.Tensor, h: torch.Tensor,
                           bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gh, gb = mod.bias_gelu_backward(grad, h, bias)
        return gh, gb

    @bias_gelu_backward.register_fake
    def _(grad, h, bias):
        return torch.empty_like(h), torch.empty_like(bias)

    bias_gelu.register_autograd(backward, setup_context=setup_context)

    print(f"    torch.ops.p35.bias_gelu -> {torch.ops.p35.bias_gelu}")
    print(f"    schema: {torch.ops.p35.bias_gelu.default._schema}")
    h = torch.randn(64, 32)
    b = torch.randn(32)
    err = K.max_abs_diff(torch.ops.p35.bias_gelu(h, b), F.gelu(h + b))
    print(f"    still correct: {err:.2e}")
    ROWS.append(["registered", "schema",
                 str(torch.ops.p35.bias_gelu.default._schema), ""])
    return bias_gelu


def with_registration(mod, op, before):
    K.banner("[4] the same compile, now that the op is registered")
    h = torch.randn(256, 128)
    b = torch.randn(128)

    def with_custom_op(h, b):
        return op(h, b) * 2.0

    _, _, r = compile_once(with_custom_op, h, b)
    print(f"    {'function':<34}{'ops captured':>14}{'graphs':>8}{'breaks':>8}")
    print(f"    {'pybind kernel (section 2)':<34}{before['ops']:>14}"
          f"{before['graphs']:>8}{before['breaks']:>8}")
    print(f"    {'the same kernel, registered':<34}{r['ops']:>14}"
          f"{r['graphs']:>8}{r['breaks']:>8}")
    ROWS.append(["registered", "ops captured", str(r["ops"]), f"{r['breaks']} breaks"])

    print("\n    -- the fake implementation, tested directly --")
    with torch.device("meta"):
        mh = torch.empty(1024, 512)
        mb = torch.empty(512)
    try:
        out = torch.ops.p35.bias_gelu(mh, mb)
        print(f"    on meta tensors: got shape {tuple(out.shape)}, dtype {out.dtype},"
              f" device {out.device}")
        print("    -- no data was touched; only the shape rule ran")
        ROWS.append(["fake impl", "meta call", str(tuple(out.shape)), "no data touched"])
    except Exception as e:
        print(f"    meta call failed: {type(e).__name__}: {str(e)[:80]}")

    print("\n    -- dynamic shapes: does one compile cover many batch sizes? --")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(with_custom_op)
    with contextlib.redirect_stdout(io.StringIO()):
        for rows in (64, 128, 256, 512):
            compiled(torch.randn(rows, 128), b)
    graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
    print(f"    four different batch sizes -> {graphs} compiled graphs")
    print("    (the fake implementation is what lets the second one be symbolic:")
    print("     `empty_like(h)` is a shape RULE, so it works for any number of rows)")
    ROWS.append(["fake impl", "graphs for 4 batch sizes", str(graphs), ""])
    return r


def autograd_check(op):
    K.banner("[5] autograd")
    torch.manual_seed(0)
    h = torch.randn(64, 32, requires_grad=True)
    b = torch.randn(32, requires_grad=True)
    h2 = h.detach().clone().requires_grad_()
    b2 = b.detach().clone().requires_grad_()

    y = op(h, b)
    print(f"    grad_fn is now: {y.grad_fn}")
    y.pow(2).sum().backward()
    F.gelu(h2 + b2).pow(2).sum().backward()

    gh = K.rel_err(h.grad, h2.grad)
    gb = K.rel_err(b.grad, b2.grad)
    print(f"    grad wrt h    : relative error {gh:.2e}")
    print(f"    grad wrt bias : relative error {gb:.2e}")
    ROWS.append(["autograd", "grad h vs torch", f"{gh:.2e}", ""])
    ROWS.append(["autograd", "grad bias vs torch", f"{gb:.2e}", ""])

    # torch.autograd.gradcheck is the usual tool, and it cannot be used here:
    # it feeds float64 inputs, and this kernel is float32-only, so it raises
    # before testing anything. That is worth knowing before you reach for it.
    try:
        torch.autograd.gradcheck(op, (h.detach().double().requires_grad_(),
                                      b.detach().double().requires_grad_()))
        gc_msg = "passed"
    except Exception as e:
        gc_msg = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
    print(f"\n    torch.autograd.gradcheck -> {gc_msg}")
    print("    (gradcheck compares against float64 finite differences; a float32")
    print("     kernel cannot take part, so we do the comparison ourselves)")
    ROWS.append(["autograd", "gradcheck on fp32 kernel", gc_msg[:50], "cannot be used"])

    # A hand-rolled central difference instead:  df/dx ~= (f(x+e) - f(x-e)) / 2e
    #
    # The objective is ONE output element, not a sum over all of them. With a
    # sum, f is a number around 1000 and the perturbation moves it by ~0.01 --
    # float32 keeps about 7 digits, so most of the difference is rounding noise.
    # Element by element, f is around 1 and the signal survives.
    h3 = h.detach().clone()
    b3 = b.detach().clone()
    eps = 1e-2
    worst = 0.0
    torch.manual_seed(1)
    for _ in range(8):
        i = int(torch.randint(0, h3.size(0), (1,)))
        j = int(torch.randint(0, h3.size(1), (1,)))
        probe = h3.clone().requires_grad_()
        op(probe, b3)[i, j].backward()
        analytic = probe.grad[i, j].item()
        plus, minus = h3.clone(), h3.clone()
        plus[i, j] += eps
        minus[i, j] -= eps
        num = (op(plus, b3)[i, j] - op(minus, b3)[i, j]).item() / (2 * eps)
        worst = max(worst, abs(num - analytic) / (abs(analytic) + 1e-6))
    print(f"    central differences at 8 random coordinates: worst relative"
          f" mismatch {worst:.2e}")
    ROWS.append(["autograd", "central difference check", f"{worst:.2e}", "8 coordinates"])


def opcheck_section(mod, op):
    K.banner("[6] opcheck, and the bug it catches")
    h = torch.randn(32, 16)
    b = torch.randn(16)

    print("    -- the honest op --")
    try:
        torch.library.opcheck(op, (h, b))
        print("    torch.library.opcheck: all tests passed")
        ROWS.append(["opcheck", "correct op", "passed", ""])
    except Exception as e:
        print(f"    opcheck FAILED: {type(e).__name__}: {str(e)[:120]}")
        ROWS.append(["opcheck", "correct op", "failed", str(e)[:60]])

    print("\n    -- lie #1: return the input tensor itself --")
    # The in-place kernel returns h. Declared with mutates_args=(), the op is
    # promising "I do not touch my inputs" while handing one of them back.

    @torch.library.custom_op("p35::lying_alias", mutates_args=())
    def lying_alias(h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return mod.bias_gelu_inplace(h, bias)     # returns h itself

    @lying_alias.register_fake
    def _(h, bias):
        return torch.empty_like(h)

    try:
        lying_alias(h.clone(), b)
        print("    the call succeeded -- nothing noticed")
        ROWS.append(["opcheck", "aliasing lie", "not caught", ""])
    except RuntimeError as e:
        print(f"    RuntimeError on the very first call: {str(e).split('.')[0][:110]}")
        print("    PyTorch checks this one for you: the output of a custom op may")
        print("    not BE one of its inputs, because the schema you signed says")
        print("    the op is functional and the compiler will rely on that.")
        ROWS.append(["opcheck", "aliasing lie", "caught at call time",
                     str(e).split(".")[0][:60]])

    print("\n    -- lie #2: mutate the input but return a fresh tensor --")
    # Now nothing aliases, so the automatic check has nothing to see. The
    # mutation is still there, and still undeclared.

    @torch.library.custom_op("p35::lying_mutate", mutates_args=())
    def lying_mutate(h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        mod.bias_gelu_inplace(h, bias)            # writes into h ...
        return h.clone()                          # ... and hides the evidence

    @lying_mutate.register_fake
    def _(h, bias):
        return torch.empty_like(h)

    hh = h.clone()
    before = hh.clone()
    lying_mutate(hh, b)
    changed = K.max_abs_diff(hh, before)
    print(f"    the call ran with no error, and changed its input by up to {changed:.3f}")
    ROWS.append(["opcheck", "silent input mutation", f"{changed:.3f}", "max change"])
    try:
        torch.library.opcheck(lying_mutate, (h.clone(), b))
        print("    opcheck: passed (!) -- this one slips through")
        ROWS.append(["opcheck", "mutation lie", "not caught", ""])
    except Exception as e:
        first = str(e).split("\n")[0][:110]
        print(f"    opcheck CAUGHT it: {type(e).__name__}: {first}")
        ROWS.append(["opcheck", "mutation lie", "caught by opcheck", first[:60]])

    print("\n    -- the honest in-place version --")

    @torch.library.custom_op("p35::bias_gelu_", mutates_args={"h"})
    def bias_gelu_(h: torch.Tensor, bias: torch.Tensor) -> None:
        mod.bias_gelu_inplace(h, bias)

    hh = h.clone()
    bias_gelu_(hh, b)
    err = K.max_abs_diff(hh, F.gelu(h + b))
    print(f"    mutates_args={{'h'}} and returns None: result correct to {err:.1e}")
    try:
        torch.library.opcheck(bias_gelu_, (h.clone(), b))
        print("    opcheck: all tests passed")
        ROWS.append(["opcheck", "honest in-place op", "passed", f"err {err:.1e}"])
    except Exception as e:
        print(f"    opcheck FAILED: {str(e).splitlines()[0][:90]}")
        ROWS.append(["opcheck", "honest in-place op", "failed", ""])


def end_to_end(mod, op):
    K.banner("[7] end to end: a two-layer MLP, compiled")
    torch.manual_seed(0)
    x = torch.randn(B, 512)
    W1 = torch.randn(512, H) * 0.05
    b1 = torch.randn(H) * 0.1
    W2 = torch.randn(H, 512) * 0.02

    def torch_mlp(x):
        return F.gelu(x @ W1 + b1) @ W2

    def pybind_mlp(x):
        return mod.bias_gelu(x @ W1, b1) @ W2

    def custom_op_mlp(x):
        return op(x @ W1, b1) @ W2

    infos = {}
    for name, fn in (("all torch ops", torch_mlp),
                     ("pybind kernel", pybind_mlp),
                     ("registered custom op", custom_op_mlp)):
        compiled, _, info = compile_once(fn, x)
        infos[name] = (compiled, info)

    ref = torch_mlp(x)
    for name, (compiled, info) in infos.items():
        print(f"    {name:<24} ops captured {info['ops']:>4}   graphs {info['graphs']}"
              f"   breaks {info['breaks']}   err {K.rel_err(compiled(x), ref):.1e}")
        ROWS.append(["end to end", name, f"{info['ops']} ops captured",
                     f"{info['breaks']} breaks"])

    fns = {"eager, all torch": lambda: torch_mlp(x),
           "eager, pybind kernel": lambda: pybind_mlp(x)}
    for name, (compiled, _) in infos.items():
        fns["compiled: " + name] = (lambda c=compiled: c(x))
    res = K.interleaved(fns, rounds=5, warmup=2)
    base = res["eager, all torch"][0]
    print()
    rows = []
    for lb, (ms, sp) in res.items():
        print(f"    {lb:<30}{ms:>9.1f} ms{base / ms:>8.2f}x")
        rows.append([lb, f"{ms:.1f}", f"{base / ms:.2f}"])
        ROWS.append(["end to end speed", lb, f"{ms:.2f} ms", f"{base / ms:.2f}x"])
    return infos, rows


def figure(infos, speed_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    names = list(infos)
    ops = [infos[n][1]["ops"] for n in names]
    ax.bar(range(len(ops)), ops, color=["#468", "#c44", "#4a7"])
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylabel("operations captured by torch.compile")
    ax.set_title("What the compiler can see")
    for i, v in enumerate(ops):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    labels = [r[0] for r in speed_rows]
    sp = [float(r[2]) for r in speed_rows]
    ax.barh(range(len(sp)), sp, color=["#999", "#c44", "#468", "#e8a33d", "#4a7"])
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#666", ls="--", lw=1)
    ax.set_xlabel("speedup vs eager torch")
    ax.set_title("And what that is worth")
    for i, v in enumerate(sp):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "custom_op.png", dpi=110)
    print(f"\n    wrote {OUT / 'custom_op.png'}")


def main():
    t0 = time.perf_counter()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()}")
    mod = build_all()
    _, unreg = without_registration(mod)
    op = register(mod)
    with_registration(mod, op, unreg)
    autograd_check(op)
    opcheck_section(mod, op)
    infos, speed_rows = end_to_end(mod, op)
    figure(infos, speed_rows)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"    wrote {OUT / 'findings.csv'}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
