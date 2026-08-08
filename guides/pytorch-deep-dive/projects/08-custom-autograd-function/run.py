"""Project 08 — Custom autograd.Function.

Write ReLU and Sigmoid as `torch.autograd.Function` subclasses, then answer the
question a beginner should be asking: torch already HAS relu and sigmoid, so why
write them again?

  1. the two functions, verified against torch
  2. gradcheck: what it actually does, and why it needs float64
  3. two broken backwards -- one gradcheck catches, one it cannot
  4. the real payoff: a fused op that saves fewer tensors than the eager version
  5. save_for_backward vs stashing on ctx -- the safety net you lose

Runs in about 15 seconds on CPU. No downloads.
"""

import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(1)
torch.manual_seed(0)

FINDINGS = {}


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# the functions
# =========================================================================
class MyReLU(Function):
    """max(x, 0), with its own backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # backward needs to know the SIGN of x
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # d/dx max(x,0) = 1 where x > 0, else 0. Multiply by the incoming
        # gradient -- that multiplication IS the chain rule, and forgetting it
        # is the single most common bug in a custom Function.
        return grad_output * (x > 0)


class MySigmoid(Function):
    """1 / (1 + exp(-x)), saving the OUTPUT instead of the input."""

    @staticmethod
    def forward(ctx, x):
        y = torch.sigmoid(x)
        ctx.save_for_backward(y)          # note: y, not x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        # dy/dx = y(1-y). Written in terms of the output, so backward needs no
        # exp() at all. torch's own SigmoidBackward does exactly this.
        return grad_output * y * (1 - y)


class Scale(Function):
    """y = alpha * x, where alpha is a plain Python float, not a tensor."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha                 # a float is not a tensor: no save_for_backward
        return x * alpha

    @staticmethod
    def backward(ctx, grad_output):
        # forward took TWO arguments, so backward must return TWO gradients.
        # alpha is not a tensor and cannot receive one -> None.
        return grad_output * ctx.alpha, None


class ReLUNoChainRule(Function):
    """BROKEN on purpose: returns the local derivative and drops grad_output."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (x > 0).to(grad_output.dtype)      # the missing `grad_output *`


class ReLUWrongAtZero(Function):
    """BROKEN in a way nothing will ever notice: derivative 1 at exactly x=0."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x >= 0)             # >= instead of >


# =========================================================================
# 1. do they match torch?
# =========================================================================
def match_torch():
    print("=" * 78)
    print("1. THE TWO FUNCTIONS vs THE BUILT-INS")
    print("=" * 78)

    x = torch.randn(2000, dtype=torch.float64, requires_grad=True)
    up = torch.randn(2000, dtype=torch.float64)      # a non-trivial upstream grad

    rows = []
    for name, mine, theirs in [("relu", MyReLU.apply, torch.relu),
                               ("sigmoid", MySigmoid.apply, torch.sigmoid)]:
        a = mine(x)
        (a * up).sum().backward()
        ga = x.grad.clone()
        x.grad = None
        b = theirs(x)
        (b * up).sum().backward()
        gb = x.grad.clone()
        x.grad = None
        fd = (a - b).abs().max().item()
        gd = (ga - gb).abs().max().item()
        rows.append((name, fd, gd))
        print(f"  {name:<9} forward max diff {fd:.2e}   backward max diff {gd:.2e}")
        rec(f"{name}_forward_diff", f"{fd:.3e}")
        rec(f"{name}_backward_diff", f"{gd:.3e}")

    # Scale, and what happens if backward returns the wrong number of things
    z = Scale.apply(x, 3.0)
    z.sum().backward()
    print(f"\n  Scale(x, 3.0): x.grad is all {x.grad.unique().tolist()} "
          f"(d(3x)/dx = 3) -- and backward returned (grad, None)")
    print("  The rule: backward returns exactly one value per forward argument,")
    print("  in the same order. Non-tensor arguments get None.")
    x.grad = None
    return rows


# =========================================================================
# 2. gradcheck
# =========================================================================
def do_gradcheck():
    print("\n" + "=" * 78)
    print("2. gradcheck: COMPARING YOUR BACKWARD TO A NUMERICAL ONE")
    print("=" * 78)
    print("  gradcheck nudges every input entry by +-eps, watches the output,")
    print("  and builds the Jacobian numerically. Then it compares that to the")
    print("  Jacobian your backward implies. 'grad' + 'check', nothing more.")
    print("  Cost: two forward passes per input element -- fine for a 10-element")
    print("  test tensor, hopeless for a real one. Test small, then trust it.\n")

    x64 = torch.randn(12, dtype=torch.float64, requires_grad=True)
    ok_relu = gradcheck(MyReLU.apply, (x64,), eps=1e-6, atol=1e-8)
    ok_sig = gradcheck(MySigmoid.apply, (x64,), eps=1e-6, atol=1e-8)
    print(f"  MyReLU    float64: {ok_relu}")
    print(f"  MySigmoid float64: {ok_sig}")
    rec("gradcheck_relu_float64", ok_relu)
    rec("gradcheck_sigmoid_float64", ok_sig)

    import warnings
    x32 = torch.randn(12, dtype=torch.float32, requires_grad=True)
    try:
        with warnings.catch_warnings():     # torch warns about float32 up front
            warnings.simplefilter("ignore")
            gradcheck(MySigmoid.apply, (x32,), eps=1e-6, atol=1e-8)
        msg32 = "passed"
    except Exception as e:
        first = str(e).strip().split("\n")[0]
        msg32 = f"FAILED -- {first[:70]}"
    print(f"  MySigmoid float32: {msg32}")
    rec("gradcheck_sigmoid_float32", msg32.split(" --")[0])
    print("\n  Same code, same maths, and float32 fails. The numerical Jacobian")
    print("  divides by eps = 1e-6, so it magnifies whatever rounding error is")
    print("  in the two forward passes. float32 keeps ~7 decimal digits; divide")
    print("  its rounding error by 1e-6 and the noise swamps the answer. float64")
    print("  keeps ~16, so there is room to spare. Always gradcheck in float64.")

    # what the noise actually looks like
    for dt in (torch.float32, torch.float64):
        xx = torch.randn(1, dtype=dt)
        eps = 1e-6
        f = lambda t: torch.sigmoid(t)
        num = ((f(xx + eps) - f(xx - eps)) / (2 * eps)).item()
        ana = (torch.sigmoid(xx) * (1 - torch.sigmoid(xx))).item()
        print(f"    {str(dt).replace('torch.', ''):<8} numerical {num:.10f}"
              f"   analytic {ana:.10f}   error {abs(num - ana):.2e}")
        rec(f"fd_noise_{str(dt).replace('torch.', '')}", f"{abs(num - ana):.3e}")
    return ok_relu, ok_sig, msg32


# =========================================================================
# 3. two broken backwards
# =========================================================================
def broken_backwards():
    print("\n" + "=" * 78)
    print("3. TWO BROKEN BACKWARDS")
    print("=" * 78)

    x = torch.randn(12, dtype=torch.float64, requires_grad=True)

    print("\n  (a) ReLUNoChainRule -- forgot to multiply by grad_output")
    try:
        gradcheck(ReLUNoChainRule.apply, (x,), eps=1e-6, atol=1e-8)
        a_msg = "passed"
    except Exception as e:
        a_msg = "caught by gradcheck"
    print(f"      gradcheck: {a_msg}")
    rec("gradcheck_no_chain_rule", a_msg)

    # ... and why it is invisible in a one-layer test
    xa = torch.randn(6, dtype=torch.float64, requires_grad=True)
    ya = ReLUNoChainRule.apply(xa)
    ya.sum().backward()                      # upstream grad is all ones!
    g_broken = xa.grad.clone()
    xa.grad = None
    yb = MyReLU.apply(xa)
    yb.sum().backward()
    g_ok = xa.grad.clone()
    same_under_sum = torch.equal(g_broken, g_ok)
    print(f"      but under `y.sum().backward()` the two agree: {same_under_sum}")
    print("      -- because .sum() sends an upstream gradient of exactly 1.0, and")
    print("      multiplying by 1.0 changes nothing. Test a custom Function with")
    print("      `(y * torch.randn_like(y)).sum().backward()`, never `.sum()`.")
    rec("no_chain_rule_hidden_by_sum", same_under_sum)

    xa.grad = None
    up = torch.randn(6, dtype=torch.float64)
    (ReLUNoChainRule.apply(xa) * up).sum().backward()
    g1 = xa.grad.clone()
    xa.grad = None
    (MyReLU.apply(xa) * up).sum().backward()
    g2 = xa.grad.clone()
    print(f"      with a random upstream grad: max diff {(g1 - g2).abs().max():.3f}")
    rec("no_chain_rule_diff_random_upstream",
        round((g1 - g2).abs().max().item(), 4))

    print("\n  (b) ReLUWrongAtZero -- derivative 1 at exactly x = 0")
    try:
        gradcheck(ReLUWrongAtZero.apply, (x,), eps=1e-6, atol=1e-8)
        b_msg = "passed"
    except Exception:
        b_msg = "caught by gradcheck"
    print(f"      gradcheck on random inputs: {b_msg}")
    xz = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    upz = torch.randn(4, dtype=torch.float64)
    (ReLUWrongAtZero.apply(xz) * upz).sum().backward()
    gz = xz.grad.clone()
    xz.grad = None
    (MyReLU.apply(xz) * upz).sum().backward()
    print(f"      at x = 0 exactly (same upstream grad for both):")
    print(f"        broken  {gz.numpy().round(4)}")
    print(f"        correct {xz.grad.numpy().round(4)}")
    rec("gradcheck_wrong_at_zero", b_msg)
    print("      gradcheck never sees it: torch.randn hits exactly 0.0 with")
    print("      probability ~0. Neither will your training data. relu is not")
    print("      differentiable at 0 -- there IS no right answer, only a")
    print("      convention. torch picks 0; matching the convention matters only")
    print("      when you compare implementations bit for bit.")
    return a_msg, b_msg


# =========================================================================
# 4. the payoff: saving fewer tensors
# =========================================================================
def count_saved(fn, *args):
    """Total bytes of every tensor autograd stashes for backward.

    saved_tensors_hooks fires on each save; dedup by data_ptr so a tensor saved
    twice is only counted once. This is how you measure activation memory on a
    CPU, where there is no torch.cuda.max_memory_allocated to ask.
    """
    seen, total = {}, [0]

    def pack(t):
        if t.data_ptr() not in seen:
            seen[t.data_ptr()] = t.numel() * t.element_size()
            total[0] += seen[t.data_ptr()]
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*args)
    return out, total[0], len(seen)


class FusedSwish(Function):
    """x * sigmoid(x), computed and differentiated in one node."""

    @staticmethod
    def forward(ctx, x):
        s = torch.sigmoid(x)
        ctx.save_for_backward(x, s)        # two tensors, and that is all
        return x * s

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        # d/dx [x*s(x)] = s + x*s*(1-s)
        return grad_output * (s + x * s * (1 - s))


def eager_swish(x):
    return x * torch.sigmoid(x)


def fusion_payoff():
    print("\n" + "=" * 78)
    print("4. WHY BOTHER: FEWER SAVED TENSORS")
    print("=" * 78)
    print("  torch already has relu and sigmoid, and its versions are as good as")
    print("  ours -- projects 1-3 above only proved we can match them. The reason")
    print("  to write a Function is that YOU choose what backward needs, so you")
    print("  choose what has to stay in memory until backward runs.\n")

    n = 1_000_000
    x = torch.randn(n, requires_grad=True)

    _, eager_bytes, eager_n = count_saved(eager_swish, x)
    _, fused_bytes, fused_n = count_saved(FusedSwish.apply, x)
    print(f"  swish(x) = x * sigmoid(x) on {n:,} floats ({n * 4 / 1e6:.1f} MB per tensor)")
    print(f"    eager (two ops):   {eager_n} tensors saved, {eager_bytes / 1e6:6.2f} MB")
    print(f"    fused Function:    {fused_n} tensors saved, {fused_bytes / 1e6:6.2f} MB")
    rec("swish_eager_saved_mb", round(eager_bytes / 1e6, 3))
    rec("swish_fused_saved_mb", round(fused_bytes / 1e6, 3))
    rec("swish_eager_saved_tensors", eager_n)
    rec("swish_fused_saved_tensors", fused_n)

    # and a version that recomputes sigmoid in backward instead of saving it
    class SwishRecompute(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)       # only x
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            s = torch.sigmoid(x)           # pay compute to save memory
            return grad_output * (s + x * s * (1 - s))

    _, recomp_bytes, recomp_n = count_saved(SwishRecompute.apply, x)
    print(f"    recompute variant: {recomp_n} tensor saved,  {recomp_bytes / 1e6:6.2f} MB")
    rec("swish_recompute_saved_mb", round(recomp_bytes / 1e6, 3))

    print("\n  Read that carefully: FUSING BOUGHT NOTHING. Eager already saved")
    print("  exactly two tensors -- the multiply's two operands -- and our fused")
    print("  version saves the same two. 'Fuse it into one Function to save")
    print("  memory' is not a law; it only pays when the eager expression leaves")
    print("  intermediates lying around. Two operations do not.")
    print("  What DID pay is the third variant: refusing to save sigmoid(x) and")
    print("  recomputing it in backward. Half the memory. That is gradient")
    print("  checkpointing in miniature -- project 10 applies it to whole layers.")

    print("\n  Now an expression with real intermediates -- tanh-approximated GELU:")
    print("    0.5 * x * (1 + tanh(0.7978 * (x + 0.044715 * x^3)))")

    C1, C2 = 0.7978845608028654, 0.044715

    def eager_gelu(t):
        return 0.5 * t * (1 + torch.tanh(C1 * (t + C2 * t * t * t)))

    class FusedGELU(Function):
        @staticmethod
        def forward(ctx, t):
            ctx.save_for_backward(t)       # just the input; everything else is
            inner = C1 * (t + C2 * t ** 3) # cheap to rebuild in backward
            return 0.5 * t * (1 + torch.tanh(inner))

        @staticmethod
        def backward(ctx, g):
            (t,) = ctx.saved_tensors
            inner = C1 * (t + C2 * t ** 3)
            u = torch.tanh(inner)
            dinner = C1 * (1 + 3 * C2 * t * t)
            return g * (0.5 * (1 + u) + 0.5 * t * (1 - u * u) * dinner)

    _, ge_bytes, ge_n = count_saved(eager_gelu, x)
    _, gf_bytes, gf_n = count_saved(FusedGELU.apply, x)
    print(f"    eager:  {ge_n} tensors saved, {ge_bytes / 1e6:6.2f} MB")
    print(f"    fused:  {gf_n} tensor saved,  {gf_bytes / 1e6:6.2f} MB"
          f"   ({ge_bytes / gf_bytes:.0f}x less)")
    rec("gelu_eager_saved_mb", round(ge_bytes / 1e6, 3))
    rec("gelu_fused_saved_mb", round(gf_bytes / 1e6, 3))
    rec("gelu_saved_tensor_ratio", round(ge_bytes / gf_bytes, 1))
    print("  Six intermediates versus one. THAT is when a custom Function pays,")
    print("  and it is why every serious library ships a fused GELU.")

    # correctness + timing of every variant, best of 5
    print(f"\n  {'variant':<20}{'saved':>9}{'fwd+bwd':>11}{'max grad diff vs eager':>26}")
    up = torch.randn(n)
    variants = [("swish eager", eager_swish, eager_bytes, "swish"),
                ("swish fused", FusedSwish.apply, fused_bytes, "swish"),
                ("swish recompute", SwishRecompute.apply, recomp_bytes, "swish"),
                ("gelu eager", eager_gelu, ge_bytes, "gelu"),
                ("gelu fused", FusedGELU.apply, gf_bytes, "gelu")]
    times, grads, refs = {}, {}, {}
    for name, f, byts, family in variants:
        best = 1e9
        for _ in range(5):
            x.grad = None
            t0 = time.perf_counter()
            (f(x) * up).sum().backward()
            best = min(best, time.perf_counter() - t0)
        times[name] = best * 1e3
        grads[name] = x.grad.clone()
        refs.setdefault(family, name)
        d = (grads[name] - grads[refs[family]]).abs().max().item()
        print(f"  {name:<20}{byts / 1e6:>7.1f}MB{best * 1e3:>9.1f}ms{d:>26.2e}")
        rec(f"{name.replace(' ', '_')}_ms", round(best * 1e3, 2))
        rec(f"{name.replace(' ', '_')}_grad_diff", f"{d:.3e}")
    x.grad = None
    print("\n  Note the swish rows: the fused Function is SLOWER than eager. On")
    print("  CPU, torch's built-in ops are already vectorised C++; our backward")
    print("  is more Python-level tensor operations, not fewer. A custom Function")
    print("  fuses the GRAPH, not the kernels. Real kernel fusion is phase 6.")
    return {"swish": (eager_bytes, fused_bytes, recomp_bytes),
            "gelu": (ge_bytes, gf_bytes)}, times


# =========================================================================
# 5. save_for_backward vs ctx.whatever
# =========================================================================
def the_safety_net():
    print("\n" + "=" * 78)
    print("5. save_for_backward IS NOT JUST A PLACE TO PUT THINGS")
    print("=" * 78)
    print("  You can write `ctx.x = x` and it works. Here is what you give up.\n")

    class SafeSquare(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return 2 * x * g

    class StashSquare(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.x = x                       # a plain attribute
            return x * x

        @staticmethod
        def backward(ctx, g):
            return 2 * ctx.x * g

    # the safe one: torch notices x changed and refuses to give a wrong answer
    x = torch.tensor([3.0], requires_grad=True)
    z = x * 1.0                             # a non-leaf we are allowed to mutate
    y = SafeSquare.apply(z)
    z.mul_(10)                              # sabotage: change the saved tensor
    try:
        y.backward()
        safe_msg = f"no error, x.grad = {x.grad.item()}"
    except RuntimeError as e:
        safe_msg = "RuntimeError: " + str(e).split(",")[0]
    print(f"  save_for_backward: {safe_msg}")
    rec("save_for_backward_after_inplace", safe_msg.split(":")[0])

    x2 = torch.tensor([3.0], requires_grad=True)
    z2 = x2 * 1.0
    y2 = StashSquare.apply(z2)
    z2.mul_(10)
    y2.backward()
    print(f"  ctx.x = x        : no error, x.grad = {x2.grad.item()}")
    print(f"  the true gradient at x = 3 is 2*3 = 6.0")
    rec("ctx_stash_after_inplace_grad", x2.grad.item())
    print("\n  save_for_backward records the tensor's VERSION COUNTER -- a number")
    print("  torch bumps on every in-place write. At backward time it compares,")
    print("  sees the mismatch, and raises. A plain attribute stores only the")
    print(f"  reference, so backward quietly used 30 instead of 3 and returned")
    print(f"  {x2.grad.item()} instead of 6.0. That is the same trap `.data` sets in")
    print("  project 2, from the other side of the fence.")
    print("\n  It also frees the tensor as soon as backward runs, and cooperates")
    print("  with saved_tensors_hooks (which is how section 4 could measure")
    print("  anything at all). A plain attribute does neither.")
    return safe_msg, x2.grad.item()


# =========================================================================
# figures
# =========================================================================
def fig_memory(byts, times):
    labels = ["swish\neager", "swish\nfused", "swish\nrecompute",
              "gelu\neager", "gelu\nfused"]
    keys = ["swish eager", "swish fused", "swish recompute",
            "gelu eager", "gelu fused"]
    mem = [b / 1e6 for b in byts["swish"]] + [b / 1e6 for b in byts["gelu"]]
    ms = [times[k] for k in keys]
    cols = [ps.SERIES[0], ps.SERIES[1], ps.SERIES[2], ps.SERIES[0], ps.SERIES[1]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, axis="y", color=ps.GRID, linewidth=0.8)
    axes[0].bar(labels, mem, color=cols, width=0.6)
    for i, v in enumerate(mem):
        axes[0].text(i, v + max(mem) * 0.03, f"{v:.0f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(mem) * 1.2)
    axes[0].set_title("MB held until backward runs (1M floats)", color=ps.INK,
                      fontsize=11, loc="left")
    axes[1].bar(labels, ms, color=cols, width=0.6)
    for i, v in enumerate(ms):
        axes[1].text(i, v + max(ms) * 0.03, f"{v:.0f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(ms) * 1.2)
    axes[1].set_title("Forward + backward, milliseconds", color=ps.INK,
                      fontsize=11, loc="left")
    ps.save(fig, os.path.join(OUT, "fusion_tradeoff.png"))


def fig_shapes():
    """What the three functions and their derivatives actually look like."""
    x = torch.linspace(-5, 5, 400, requires_grad=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (name, fn) in zip(axes, [("MyReLU", MyReLU.apply),
                                     ("MySigmoid", MySigmoid.apply),
                                     ("FusedSwish", FusedSwish.apply)]):
        ps.style_axes(ax)
        ax.grid(True, color=ps.GRID, linewidth=0.8)
        y = fn(x)
        y.sum().backward()
        ax.plot(x.detach(), y.detach(), color=ps.SERIES[0], lw=1.8, label="forward")
        ax.plot(x.detach(), x.grad.detach(), color=ps.SERIES[2], lw=1.5,
                ls="--", label="backward (derivative)")
        x.grad = None
        ax.set_title(name, color=ps.INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8)
    ps.save(fig, os.path.join(OUT, "function_shapes.png"))


# =========================================================================
def main():
    match_torch()
    do_gradcheck()
    broken_backwards()
    byts, times = fusion_payoff()
    the_safety_net()

    fig_memory(byts, times)
    fig_shapes()

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")


if __name__ == "__main__":
    main()
