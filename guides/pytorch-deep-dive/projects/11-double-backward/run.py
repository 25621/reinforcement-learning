"""Project 11 — Double backward.

Differentiate a gradient. `create_graph=True` tells autograd to record the
backward pass itself as a graph, so the gradient becomes just another tensor you
can differentiate.

  1. the smallest possible example, checked against the hand-derived answer
  2. the gradient penalty: nabla_theta ||nabla_x L||^2 on a real model
  3. why `.detach()` in the wrong place silently zeroes the second-order term
  4. Hessian-vector products: n backward passes vs one
  5. what create_graph costs in memory and time
  6. the ops that quietly refuse: once_differentiable, and the STE from
     project 09

Runs in about 10 seconds on CPU. No downloads.
"""

import csv
import os
import sys
import time
import weakref

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(4)
FINDINGS = {}


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# 1. the smallest example
# =========================================================================
def smallest_example():
    print("=" * 78)
    print("1. THE SMALLEST DOUBLE BACKWARD")
    print("=" * 78)
    print("  y = x^3 at x = 2.   dy/dx = 3x^2 = 12.   d2y/dx2 = 6x = 12.\n")

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3

    # First backward WITHOUT create_graph: g is a plain number, graph discarded
    g_plain = torch.autograd.grad(y, x)[0]
    print(f"  torch.autograd.grad(y, x)                  -> {g_plain.item()}"
          f"   grad_fn = {g_plain.grad_fn}")
    try:
        torch.autograd.grad(g_plain, x)
        msg = "worked"
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:66]
    print(f"  differentiating that again                 -> {msg}")
    rec("no_create_graph_error", msg)

    # With create_graph: g is a tensor that remembers how it was computed.
    # (y is rebuilt because the first grad() call already freed its graph --
    #  that is the `retain_graph=True` error most people meet first.)
    y = x ** 3
    g = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"\n  torch.autograd.grad(..., create_graph=True) -> {g.item()}"
          f"   grad_fn = {type(g.grad_fn).__name__}")
    g2 = torch.autograd.grad(g, x)[0]
    print(f"  differentiating THAT                        -> {g2.item()}"
          f"   (6x = {6 * 2.0})")
    rec("first_derivative", g.item())
    rec("second_derivative", g2.item())

    print("\n  The whole trick is in that grad_fn. Without create_graph the")
    print("  backward pass runs in 'just compute the numbers' mode and throws its")
    print("  own workings away; the result is a leaf with nothing behind it. With")
    print("  create_graph the backward pass is RECORDED like any forward pass, so")
    print("  the gradient is an ordinary node in a graph and backward works on it")
    print("  exactly as it does on a loss.")

    # a multi-variable check against the exact Hessian
    print("\n  A 3-variable check. f = sum(x^2 * y) + sum(x*y^2):")
    xv = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    yv = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    f = (xv ** 2 * yv).sum() + (xv * yv ** 2).sum()
    gx, gy = torch.autograd.grad(f, [xv, yv], create_graph=True)
    # d/dy of (df/dx) = d/dy (2xy + y^2) = 2x + 2y
    d_gx_dy = torch.autograd.grad(gx.sum(), yv)[0]
    exact = 2 * xv.detach() + 2 * yv.detach()
    print(f"    autograd  d(df/dx)/dy = {d_gx_dy.numpy()}")
    print(f"    by hand   2x + 2y     = {exact.numpy()}")
    print(f"    max difference        = {(d_gx_dy - exact).abs().max().item():.2e}")
    rec("mixed_partial_maxdiff", f"{(d_gx_dy - exact).abs().max().item():.3e}")


# =========================================================================
# the model used from here on
# =========================================================================
class Critic(nn.Module):
    def __init__(self, d=2, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Softplus(),
                                 nn.Linear(h, h), nn.Softplus(),
                                 nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def two_clouds(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(n, 2, generator=g) * 0.35 + torch.tensor([1.6, 0.0])
    fake = torch.randn(n, 2, generator=g) * 0.35 + torch.tensor([-1.6, 0.0])
    return real, fake


def grad_norm_of_output(model, x, create_graph):
    """||d model(x) / dx|| for every row of x."""
    x = x.detach().requires_grad_(True)
    out = model(x)
    g = torch.autograd.grad(out.sum(), x, create_graph=create_graph)[0]
    return g.norm(dim=1)


# =========================================================================
# 2. the gradient penalty
# =========================================================================
def gradient_penalty_study():
    print("\n" + "=" * 78)
    print("2. THE GRADIENT PENALTY")
    print("=" * 78)
    print("  A WGAN critic is only a valid critic if it is 1-LIPSCHITZ: moving")
    print("  its input by a distance d may change its output by at most d. Named")
    print("  after Rudolf Lipschitz, who wrote the condition down in the 1860s")
    print("  for a completely different purpose (proving differential equations")
    print("  have unique solutions). In plain terms: no cliffs. The slope is")
    print("  capped everywhere.")
    print("\n  For a scalar function, the slope in the steepest direction is the")
    print("  norm of its input-gradient. So '1-Lipschitz' is '||d out/d x|| <= 1")
    print("  everywhere', and you can PENALISE the violation:")
    print("      penalty = mean( (||d out/d x|| - 1)^2 )")
    print("  That penalty contains a gradient. Differentiating it to update the")
    print("  weights is a second derivative -- which is why WGAN-GP could not")
    print("  have been written before frameworks supported double backward.\n")

    real, fake = two_clouds()
    results = {}
    for use_gp in (False, True):
        torch.manual_seed(0)
        model = Critic()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        hist = {"loss": [], "lip": [], "wass": []}
        for step in range(300):
            opt.zero_grad(set_to_none=True)
            # the WGAN critic objective: push real high, fake low
            wass = model(real).mean() - model(fake).mean()
            loss = -wass
            if use_gp:
                # interpolate between the two clouds, penalise the slope there
                eps = torch.rand(len(real), 1)
                mid = eps * real + (1 - eps) * fake
                gn = grad_norm_of_output(model, mid, create_graph=True)
                loss = loss + 10.0 * ((gn - 1) ** 2).mean()
            loss.backward()
            opt.step()
            if step % 5 == 0:
                probe = torch.cat([real, fake,
                                   torch.rand(256, 1) * (real - fake) + fake])
                gn = grad_norm_of_output(model, probe, create_graph=False)
                hist["lip"].append(gn.max().item())
                hist["wass"].append(wass.item())
                hist["loss"].append(loss.item())
        results[use_gp] = hist
        tag = "with penalty" if use_gp else "no penalty  "
        print(f"  {tag}: final max ||d out/d x|| over the data = "
              f"{hist['lip'][-1]:8.2f}   critic gap = {hist['wass'][-1]:8.2f}")
        rec(f"gp_{use_gp}_final_lipschitz", round(hist["lip"][-1], 3))
        rec(f"gp_{use_gp}_final_gap", round(hist["wass"][-1], 3))

    ratio = results[False]["lip"][-1] / results[True]["lip"][-1]
    print(f"\n  The unconstrained critic's slope is {ratio:.0f}x larger, and its")
    print("  'distance' between the clouds is meaningless -- it just keeps")
    print("  scaling itself up, because nothing stops it. The penalised critic")
    print("  settles at a slope near 1 and reports a gap close to the actual")
    print(f"  separation of the two clouds ({(real.mean(0) - fake.mean(0)).norm():.2f}).")
    rec("gp_lipschitz_ratio", round(ratio, 1))
    return results, real, fake


# =========================================================================
# 3. the detach trap
# =========================================================================
def detach_trap():
    print("\n" + "=" * 78)
    print("3. THE ONE-CHARACTER BUG THAT ZEROES THE PENALTY")
    print("=" * 78)

    real, fake = two_clouds()
    torch.manual_seed(0)
    model = Critic()
    eps = torch.rand(len(real), 1)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)

    def full_loss(mode):
        """The realistic setting: a critic loss WITH the penalty added on."""
        base = -(model(real).mean() - model(fake).mean())
        if mode == "none":
            return base, float("nan")
        g = torch.autograd.grad(model(mid).sum(), mid,
                                create_graph=(mode != "no_create_graph"),
                                retain_graph=True)[0]
        if mode == "detached":
            g = g.detach()
        pen = ((g.norm(dim=1) - 1) ** 2).mean()
        return base + 10.0 * pen, pen.item()

    def weight_grads(mode):
        model.zero_grad(set_to_none=True)
        loss, pen = full_loss(mode)
        loss.backward()
        flat = torch.cat([q.grad.flatten() for q in model.parameters()])
        return flat, pen

    g_none, _ = weight_grads("none")
    g_ok, pen_ok = weight_grads("correct")
    g_det, pen_det = weight_grads("detached")

    print(f"  {'variant':<34}{'penalty value':>15}{'weight-grad norm':>19}")
    for label, g, pen in [("no penalty term at all", g_none, float('nan')),
                          ("penalty, create_graph=True", g_ok, pen_ok),
                          ("penalty, then g.detach()", g_det, pen_det)]:
        pv = "-" if pen != pen else f"{pen:.4f}"
        print(f"  {label:<34}{pv:>15}{g.norm().item():>19.6f}")
    rec("detach_penalty_value_correct", round(pen_ok, 4))
    rec("detach_penalty_value_detached", round(pen_det, 4))
    rec("detach_gradnorm_no_penalty", round(g_none.norm().item(), 6))
    rec("detach_gradnorm_correct", round(g_ok.norm().item(), 6))
    rec("detach_gradnorm_detached", round(g_det.norm().item(), 6))

    g_ncg, pen_ncg = weight_grads("no_create_graph")
    print(f"  {'penalty, create_graph=False':<34}{pen_ncg:>15.4f}"
          f"{g_ncg.norm().item():>19.6f}"
          f"   identical to no penalty: {torch.equal(g_ncg, g_none)}")
    rec("detach_gradnorm_no_create_graph", round(g_ncg.norm().item(), 6))
    rec("no_create_graph_equals_no_penalty", torch.equal(g_ncg, g_none))

    same = torch.equal(g_det, g_none)
    print(f"\n  detached-penalty gradient == no-penalty-at-all gradient: {same}")
    print(f"  correct-penalty gradient differs from no-penalty by "
          f"{(g_ok - g_none).norm().item():.4f}")
    rec("detached_equals_no_penalty", same)
    rec("correct_minus_none_gradnorm", round((g_ok - g_none).norm().item(), 4))

    print("\n  Read the last two rows. Both print exactly the right penalty VALUE")
    print("  -- the same number as the correct version, showing up happily in")
    print("  your training log, falling as the model improves -- and both produce")
    print("  a weight gradient that is bit-for-bit identical to not having the")
    print("  penalty at all. You pay the extra compute and buy nothing.")
    print("\n  Note what did NOT happen: no error. `create_graph=False` raises the")
    print("  famous 'does not require grad and does not have a grad_fn' only when")
    print("  the penalty is your ENTIRE loss. Add it to a normal loss -- which is")
    print("  what everyone does -- and the other term keeps the graph alive, so")
    print("  backward succeeds and the penalty is simply a constant. The loud")
    print("  version of this bug is the lucky version.")
    print("\n  Why a .detach() slips in: it is the standard fix for a memory leak,")
    print("  and here the tensor you would instinctively detach is the one whose")
    print("  history is the entire point. If you compute a quantity FROM a")
    print("  gradient, every op between the gradient and the loss has to stay in")
    print("  the graph.")
    print("\n  How to check, in one line: after loss.backward(), assert that some")
    print("  parameter's .grad is non-zero. That is the whole test, and it would")
    print("  have caught this. Better still, run it once with the penalty")
    print("  weight set to 0 and once at its real value: if the weight gradients")
    print("  match, the penalty is not connected.")
    return g_none, g_ok, g_det


# =========================================================================
# 4. Hessian-vector products
# =========================================================================
def hvp_study():
    print("\n" + "=" * 78)
    print("4. HESSIAN-VECTOR PRODUCTS: n BACKWARD PASSES, OR ONE")
    print("=" * 78)
    print("  The Hessian is the matrix of all second derivatives -- named after")
    print("  Ludwig Otto Hesse, who introduced it in 1857. For n parameters it")
    print("  has n^2 entries, so for anything real you cannot store it, let alone")
    print("  build it. But most algorithms that 'need the Hessian' only ever need")
    print("  H @ v for some vector v, and that is cheap:")
    print("      H v = d/dtheta ( (d L/d theta) . v )")
    print("  Differentiate the DOT PRODUCT of the gradient with v -- a scalar --")
    print("  and one extra backward pass gives you the whole matrix-vector")
    print("  product without ever forming the matrix.\n")

    torch.manual_seed(0)
    model = Critic(h=24)
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    real, fake = two_clouds(n=64)

    def loss_fn():
        return (model(real).mean() - model(fake).mean()) ** 2 + \
               0.1 * (model(real) ** 2).mean()

    def hvp(v):
        g = torch.autograd.grad(loss_fn(), params, create_graph=True)
        flat = torch.cat([t.flatten() for t in g])
        return torch.cat([t.flatten() for t in
                          torch.autograd.grad(flat @ v, params)])

    torch.manual_seed(1)
    v = torch.randn(n)
    t0 = time.perf_counter()
    hv = hvp(v)
    t_hvp = time.perf_counter() - t0

    t0 = time.perf_counter()
    H = torch.autograd.functional.hessian(
        lambda *ps: _loss_with(model, params, ps, real, fake), tuple(params))
    # stitch the block structure into one (n, n) matrix
    blocks = [[H[i][j].reshape(params[i].numel(), params[j].numel())
               for j in range(len(params))] for i in range(len(params))]
    Hfull = torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)
    t_full = time.perf_counter() - t0

    err = (Hfull @ v - hv).abs().max().item()
    print(f"  model has {n} parameters, so the Hessian is {n} x {n} "
          f"= {n * n:,} numbers")
    print(f"  H @ v via double backward : {t_hvp * 1000:8.1f} ms")
    print(f"  full Hessian, then H @ v  : {t_full * 1000:8.1f} ms"
          f"   ({t_full / t_hvp:.0f}x slower)")
    print(f"  max difference            : {err:.2e}")
    rec("hvp_params", n)
    rec("hvp_ms", round(t_hvp * 1000, 2))
    rec("full_hessian_ms", round(t_full * 1000, 1))
    rec("hvp_speedup", round(t_full / t_hvp, 1))
    rec("hvp_vs_full_maxdiff", f"{err:.3e}")
    print(f"\n  And the gap grows with the model: H @ v costs one extra backward")
    print(f"  pass no matter how big the model is, while the full matrix costs")
    print(f"  one per parameter. At {n} parameters that is already {t_full / t_hvp:.0f}x. At a million")
    print("  parameters the full matrix is 4 TB and the HVP still costs one pass.")
    print("  This is how conjugate-gradient natural-gradient methods (TRPO),")
    print("  influence functions, and second-order optimisers stay affordable.")
    return n, t_hvp, t_full


def _loss_with(model, params, new_params, real, fake):
    """Functional re-evaluation, for torch.autograd.functional.hessian."""
    names = [n for n, _ in model.named_parameters()]
    return torch.func.functional_call(
        model, dict(zip(names, new_params)),
        (real,)).mean().sub(
        torch.func.functional_call(
            model, dict(zip(names, new_params)), (fake,)).mean()) ** 2 + \
        0.1 * (torch.func.functional_call(
            model, dict(zip(names, new_params)), (real,)) ** 2).mean()


# =========================================================================
# 5. what create_graph costs
# =========================================================================
class Tracker:
    class Holder:
        __slots__ = ("t", "__weakref__")

        def __init__(self, t):
            self.t = t

    def __init__(self, ptrs):
        self.ptrs = ptrs
        self.live = self.peak = 0

    def pack(self, t):
        if t.data_ptr() in self.ptrs:
            return t
        nb = t.numel() * t.element_size()
        h = Tracker.Holder(t)
        self.live += nb
        self.peak = max(self.peak, self.live)
        weakref.finalize(h, self._rel, nb)
        return h

    def _rel(self, nb):
        self.live -= nb

    def unpack(self, h):
        return h.t if isinstance(h, Tracker.Holder) else h


def cost_of_create_graph():
    print("\n" + "=" * 78)
    print("5. WHAT create_graph COSTS")
    print("=" * 78)

    torch.manual_seed(0)
    model = Critic(d=2, h=256)
    real, fake = two_clouds(n=2048)
    ptrs = {p.data_ptr() for p in model.parameters()}

    def run(create_graph, penalise):
        tr = Tracker(ptrs)
        model.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(tr.pack, tr.unpack):
            loss = -(model(real).mean() - model(fake).mean())
            if penalise:
                gn = grad_norm_of_output(model, real, create_graph=create_graph)
                loss = loss + 10.0 * ((gn - 1) ** 2).mean()
            loss.backward()
        best = 1e9
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            l = -(model(real).mean() - model(fake).mean())
            if penalise:
                gn = grad_norm_of_output(model, real, create_graph=create_graph)
                l = l + 10.0 * ((gn - 1) ** 2).mean()
            l.backward()
            best = min(best, time.perf_counter() - t0)
        return tr.peak / 1e6, best * 1000

    m0, t_0 = run(False, False)
    m2, t_2 = run(True, True)
    print(f"  plain critic step                      "
          f"{m0:7.2f} MB peak   {t_0:6.1f} ms")
    print(f"  + gradient penalty (create_graph=True) "
          f"{m2:7.2f} MB peak   {t_2:6.1f} ms")
    print(f"  ratio                                  "
          f"{m2 / m0:7.2f}x         {t_2 / t_0:6.2f}x")
    rec("cost_plain_mb", round(m0, 2))
    rec("cost_double_mb", round(m2, 2))
    rec("cost_plain_ms", round(t_0, 1))
    rec("cost_double_ms", round(t_2, 1))
    rec("cost_mem_ratio", round(m2 / m0, 2))
    rec("cost_time_ratio", round(t_2 / t_0, 2))

    print("\n  Roughly: you are running a second forward pass (the recorded")
    print("  backward), and it needs its own saved tensors. That is the price of")
    print("  a second derivative, and it is why WGAN-GP training is noticeably")
    print("  slower than plain WGAN, and why gradient penalties are usually")
    print("  applied on a subset of the batch.")
    return m0, m2, t_0, t_2


# =========================================================================
# 6. ops that refuse
# =========================================================================
def ops_that_refuse():
    print("\n" + "=" * 78)
    print("6. THE OPS THAT QUIETLY REFUSE")
    print("=" * 78)
    print("  Double backward needs the BACKWARD to be differentiable too. Not")
    print("  every backward is.\n")

    class SquareOnce(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return 2 * x * g

    class SquareTwice(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return 2 * x * g       # written with differentiable ops -> reusable

    for name, fn in [("once_differentiable", SquareOnce.apply),
                     ("plain backward", SquareTwice.apply)]:
        x = torch.tensor(3.0, requires_grad=True)
        g = torch.autograd.grad(fn(x), x, create_graph=True)[0]
        try:
            g2 = torch.autograd.grad(g, x)[0].item()
            msg = f"second derivative = {g2}  (exact: 2.0)"
        except RuntimeError as e:
            msg = "RuntimeError: " + str(e).split("\n")[0][:52]
        print(f"  {name:<22} {msg}")
        rec(f"refuse_{name.replace(' ', '_')}", msg.split("  ")[0])

    print("\n  `once_differentiable` is a decorator that says 'my backward runs")
    print("  outside the graph' -- typically because it calls into C++ or numpy.")
    print("  It turns a silent wrong answer into a loud error, which is why it")
    print("  exists. A backward written in ordinary torch ops needs no decorator")
    print("  and differentiates as many times as you like.")

    print("\n  And the one from project 09. The straight-through estimator's")
    print("  backward returns the incoming gradient UNCHANGED -- a value with no")
    print("  dependence on x at all. Ask for its second derivative:\n")

    class RoundSTE(Function):
        @staticmethod
        def forward(ctx, x):
            return torch.round(x)

        @staticmethod
        def backward(ctx, g):
            return g

    def soft_round(t, T=0.1):
        f = torch.floor(t)
        return f + torch.sigmoid((t - f - 0.5) / T)

    for name, fn in [("round + STE", RoundSTE.apply),
                     ("soft round (T=0.1)", soft_round)]:
        x = torch.tensor(0.3, requires_grad=True)
        g = torch.autograd.grad(fn(x), x, create_graph=True)[0]
        try:
            g2 = torch.autograd.grad(g, x, allow_unused=True)[0]
            shown = "None" if g2 is None else f"{g2.item():.4f}"
        except RuntimeError:
            shown = "no graph"
        print(f"    {name:<20} first derivative {g.item():7.4f}"
              f"   grad_fn {str(type(g.grad_fn).__name__ if g.grad_fn else None):<16}"
              f" second derivative {shown:>9}")
        key = name.split()[0]
        rec(f"ste_{key}_first_derivative", round(g.item(), 4))
        rec(f"ste_{key}_second_derivative", shown)

    print("\n  The STE's first derivative has no grad_fn -- there is nothing")
    print("  behind it, so there is nothing to differentiate and torch says so.")
    print("  The soft version's slope genuinely varies with x, so it has a real")
    print("  second derivative.")
    print("\n  This is not a bug in the STE; it is what an STE IS. It replaces a")
    print("  staircase with a straight line, and a straight line has no curvature")
    print("  to report. But it means an STE cannot be combined with anything that")
    print("  needs a second derivative -- a gradient penalty, MAML-style")
    print("  meta-learning -- without silently losing exactly the term you were")
    print("  computing. If you need both, you need a smooth surrogate like the")
    print("  one above, not an STE.")


# =========================================================================
# figures
# =========================================================================
def fig_penalty(results, real, fake):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, color=ps.GRID, linewidth=0.8)
    xs = np.arange(len(results[True]["lip"])) * 5
    axes[0].plot(xs, results[False]["lip"], color=ps.SERIES[2], lw=1.7,
                 label="no penalty")
    axes[0].plot(xs, results[True]["lip"], color=ps.SERIES[1], lw=1.7,
                 label="gradient penalty")
    axes[0].axhline(1.0, color=ps.INK_MUTED, ls="--", lw=1.0)
    axes[0].text(xs[-1], 1.15, "the 1-Lipschitz target", ha="right",
                 fontsize=8.5, color=ps.INK_MUTED)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Steepest slope anywhere in the data", color=ps.INK,
                      fontsize=11, loc="left")
    axes[0].set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("max ‖∂out/∂x‖", color=ps.INK_SECONDARY, fontsize=10)

    axes[1].plot(xs, results[False]["wass"], color=ps.SERIES[2], lw=1.7,
                 label="no penalty")
    axes[1].plot(xs, results[True]["wass"], color=ps.SERIES[1], lw=1.7,
                 label="gradient penalty")
    true_gap = (real.mean(0) - fake.mean(0)).norm().item()
    axes[1].axhline(true_gap, color=ps.INK_MUTED, ls="--", lw=1.0)
    axes[1].text(xs[-1], true_gap * 1.15, f"actual separation = {true_gap:.1f}",
                 ha="right", fontsize=8.5, color=ps.INK_MUTED)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("What the critic reports as the gap", color=ps.INK,
                      fontsize=11, loc="left")
    axes[1].set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ps.save(fig, os.path.join(OUT, "gradient_penalty.png"))


def fig_landscape(results, real, fake):
    """The critic surface, with and without the penalty."""
    torch.manual_seed(0)
    fields = {}
    for use_gp in (False, True):
        torch.manual_seed(0)
        model = Critic()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            loss = -(model(real).mean() - model(fake).mean())
            if use_gp:
                eps = torch.rand(len(real), 1)
                mid = eps * real + (1 - eps) * fake
                gn = grad_norm_of_output(model, mid, create_graph=True)
                loss = loss + 10.0 * ((gn - 1) ** 2).mean()
            loss.backward()
            opt.step()
        xs = torch.linspace(-3, 3, 90)
        ys = torch.linspace(-2, 2, 60)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        with torch.no_grad():
            z = model(torch.stack([gx.flatten(), gy.flatten()], 1)).reshape(gx.shape)
        fields[use_gp] = (xs.numpy(), ys.numpy(), z.numpy())

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, use_gp, title in [(axes[0], False, "no penalty"),
                              (axes[1], True, "gradient penalty")]:
        ps.style_axes(ax)
        xs, ys, z = fields[use_gp]
        cs = ax.contourf(xs, ys, z, levels=18, cmap="RdBu_r")
        ax.contour(xs, ys, z, levels=10, colors="white", linewidths=0.5, alpha=0.5)
        ax.scatter(real[:, 0], real[:, 1], s=5, color="k", alpha=0.35)
        ax.scatter(fake[:, 0], fake[:, 1], s=5, color="k", alpha=0.35)
        ax.set_title(f"{title}  (range {z.min():.0f} to {z.max():.0f})",
                     color=ps.INK, fontsize=11, loc="left")
        ax.grid(False)
        fig.colorbar(cs, ax=ax, fraction=0.045)
    ps.save(fig, os.path.join(OUT, "critic_surface.png"))


def fig_cost(n, t_hvp, t_full, m0, m2, t0, t2):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, axis="y", color=ps.GRID, linewidth=0.8)
    axes[0].bar(["H @ v\n(double backward)", f"full Hessian\n({n}x{n})"],
                [t_hvp * 1000, t_full * 1000], color=[ps.SERIES[1], ps.SERIES[2]],
                width=0.5)
    axes[0].set_yscale("log")
    for i, v in enumerate([t_hvp * 1000, t_full * 1000]):
        axes[0].text(i, v * 1.25, f"{v:.1f} ms", ha="center", fontsize=9)
    axes[0].set_ylim(top=t_full * 1000 * 4)
    axes[0].set_title("Same answer, two costs", color=ps.INK, fontsize=11, loc="left")

    w = 0.35
    xs = np.arange(2)
    axes[1].bar(xs - w / 2, [m0, m2], w, color=ps.SERIES[0], label="peak MB")
    axes[1].bar(xs + w / 2, [t0, t2], w, color=ps.SERIES[3], label="ms")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["plain step", "+ gradient penalty"])
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("The price of create_graph=True", color=ps.INK,
                      fontsize=11, loc="left")
    ps.save(fig, os.path.join(OUT, "double_backward_cost.png"))


# =========================================================================
def main():
    t_start = time.perf_counter()
    smallest_example()
    results, real, fake = gradient_penalty_study()
    detach_trap()
    n, t_hvp, t_full = hvp_study()
    m0, m2, t0, t2 = cost_of_create_graph()
    ops_that_refuse()

    fig_penalty(results, real, fake)
    fig_landscape(results, real, fake)
    fig_cost(n, t_hvp, t_full, m0, m2, t0, t2)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")
    print(f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
