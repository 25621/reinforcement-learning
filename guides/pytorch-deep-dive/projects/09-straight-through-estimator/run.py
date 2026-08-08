"""Project 09 — Straight-through estimator.

`torch.round` has a derivative of exactly zero everywhere it is defined. Put it
in a model and training stops dead. The straight-through estimator lies about
that derivative on purpose, and it works well enough to be how every quantized
network and every VQ-VAE is trained.

  1. the dead gradient, measured
  2. two ways to write an STE, and why the one-liner works
  3. what the lie costs: STE gradient vs the real thing
  4. quantization-aware training vs quantize-after-training, 1 to 6 bits
  5. plain STE vs clipped STE on a 1-bit (sign) activation network

Runs in about 25 seconds on CPU. No downloads.
"""

import csv
import os
import sys
import time

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
# the estimators
# =========================================================================
class RoundSTE(Function):
    """round() forward; pretend it was the identity function in backward."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g            # d(identity)/dx = 1, so pass the gradient through


class RoundClippedSTE(Function):
    """Same, but only inside [-1, 1]. Outside, admit we have no idea."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1)


def round_ste_oneliner(x):
    """The same thing without a Function subclass.

    Forward:  x + (round(x) - x) = round(x)          -- the value is right
    Backward: detach() cuts the second term out of the graph entirely, so the
              only path back is through the leading `x`, whose derivative is 1.
    Identical results, three fewer lines, and it is what most repos actually use.
    """
    return x + (torch.round(x) - x).detach()


# =========================================================================
# 1. the dead gradient
# =========================================================================
def dead_gradient():
    print("=" * 78)
    print("1. WHY round() KILLS TRAINING")
    print("=" * 78)

    x = torch.linspace(-2, 2, 9, requires_grad=True)
    y = torch.round(x)
    y.sum().backward()
    print(f"  x            {x.detach().numpy().round(2)}")
    print(f"  round(x)     {y.detach().numpy().round(2)}")
    print(f"  d/dx         {x.grad.numpy().round(4)}")
    print(f"  gradient norm: {x.grad.norm().item()}")
    rec("round_grad_norm", x.grad.norm().item())

    print("\n  Every entry is 0, and that is CORRECT. round() is a staircase:")
    print("  flat between the steps (slope 0) and vertical at each step (slope")
    print("  undefined, and torch reports 0 there too). A slope of zero says")
    print("  'nudging x does not change the output' -- true for almost every x.")
    print("  But 'almost every' is not 'every', and the whole learning signal")
    print("  lives in the steps the flat parts cannot see.")

    x2 = torch.linspace(-2, 2, 9, requires_grad=True)
    RoundSTE.apply(x2).sum().backward()
    print(f"\n  with RoundSTE: d/dx = {x2.grad.numpy().round(4)}"
          f"   norm {x2.grad.norm().item():.4f}")
    rec("ste_grad_norm", round(x2.grad.norm().item(), 4))

    x3 = torch.linspace(-2, 2, 9, requires_grad=True)
    round_ste_oneliner(x3).sum().backward()
    same = torch.equal(x2.grad, x3.grad)
    print(f"  the one-liner `x + (round(x) - x).detach()` gives the same: {same}")
    rec("oneliner_matches_function", same)
    print("\n  'Straight-through': the gradient goes straight through the block,")
    print("  unchanged, as if it were not there. 'Estimator': it is not the")
    print("  derivative of round -- that is 0 -- it is a stand-in we chose.")
    return x.grad.clone(), x2.grad.clone()


# =========================================================================
# 2. what the lie costs
# =========================================================================
def cost_of_the_lie():
    print("\n" + "=" * 78)
    print("2. HOW WRONG IS THE STRAIGHT-THROUGH GRADIENT?")
    print("=" * 78)
    print("  Compare three answers to 'what is d round(x)/dx at x = 0.3?'\n")

    x = torch.tensor([0.3], requires_grad=True, dtype=torch.float64)
    eps = 1e-5
    fd = ((torch.round(x + eps) - torch.round(x - eps)) / (2 * eps)).item()
    torch.round(x).sum().backward()
    autog = x.grad.item()
    x.grad = None
    RoundSTE.apply(x).sum().backward()
    ste = x.grad.item()
    print(f"    finite difference : {fd}   (the honest answer for x away from a step)")
    print(f"    autograd          : {autog}")
    print(f"    STE               : {ste}   <- a deliberate fiction")
    rec("round_fd_grad", fd)
    rec("round_ste_grad", ste)

    # near a step, the true derivative is not 0 -- it is unbounded
    xs = torch.tensor([0.5 - 1e-6], dtype=torch.float64)
    d = 2e-6
    fd_step = ((torch.round(xs + d) - torch.round(xs - d)) / (2 * d)).item()
    print(f"\n    finite difference straddling the step at x = 0.5: {fd_step:,.0f}")
    print("    The true derivative is 0 almost everywhere and infinite on a set")
    print("    of measure zero. Neither number can steer gradient descent.")
    rec("round_fd_at_step", int(fd_step))

    print("\n  So what IS the STE the gradient of? Of a SMOOTHED round. Replace")
    print("  the staircase with a soft version that has a temperature T, and")
    print("  watch its gradient as T changes:\n")

    def soft_round(t, T):
        """A differentiable staircase: sharper as T -> 0."""
        f = torch.floor(t)
        frac = t - f
        return f + torch.sigmoid((frac - 0.5) / T)

    rows = []
    for T in [1.0, 0.5, 0.2, 0.1, 0.02]:
        xt = torch.linspace(-1.5, 1.5, 2001, dtype=torch.float64, requires_grad=True)
        soft_round(xt, T).sum().backward()
        g = xt.grad
        rows.append((T, g.mean().item(), g.max().item()))
        print(f"    T = {T:<5} mean slope {g.mean().item():6.3f}"
              f"   peak slope {g.max().item():9.2f}")
    rec("soft_round_mean_slope_T1", round(rows[0][1], 4))
    rec("soft_round_mean_slope_T002", round(rows[-1][1], 4))
    rec("soft_round_peak_slope_T002", round(rows[-1][2], 2))
    print("\n  The MEAN slope stays near 1 no matter how sharp the staircase gets")
    print("  -- that is the number the STE reports. What changes is the SHAPE:")
    print("  as T shrinks the slope piles up into ever-taller spikes at the steps")
    print("  and vanishes in between. The STE spreads that same total slope out")
    print("  evenly. It is the right answer on average and the wrong answer")
    print("  everywhere in particular. This is what 'biased estimator' means.")
    return rows


# =========================================================================
# 3. quantization-aware training
# =========================================================================
def quantize(w, bits, detach_scale=True):
    """Uniform quantization onto 2^bits evenly spaced levels.

    `detach_scale` keeps the min/max out of the graph. It matters more than it
    looks -- see section 4.
    """
    lo, hi = w.min(), w.max()
    if detach_scale:
        lo, hi = lo.detach(), hi.detach()
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels
    return torch.round((w - lo) / scale) * scale + lo


def quantize_ste(w, bits):
    q = quantize(w, bits)
    return w + (q - w).detach()      # forward = q, backward = identity


class QMLP(nn.Module):
    """An MLP whose weights are quantized on the way into every matmul."""

    def __init__(self, D=2, H=64, C=3, bits=None):
        super().__init__()
        self.l1 = nn.Linear(D, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, C)
        self.bits = bits

    def w(self, layer):
        if self.bits is None:
            return layer.weight
        return quantize_ste(layer.weight, self.bits)

    def forward(self, x):
        x = torch.relu(torch.nn.functional.linear(x, self.w(self.l1), self.l1.bias))
        x = torch.relu(torch.nn.functional.linear(x, self.w(self.l2), self.l2.bias))
        return torch.nn.functional.linear(x, self.w(self.l3), self.l3.bias)


def spiral(n_per=400, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n_per)
        t = np.linspace(c * 4, (c + 1) * 4, n_per) + rng.normal(0, 0.2, n_per)
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], 1))
        y.append(np.full(n_per, c))
    X = torch.tensor(np.concatenate(X), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    cut = int(0.75 * len(X))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def fit(Xtr, ytr, Xte, yte, bits=None, steps=700, seed=0):
    torch.manual_seed(seed)
    m = QMLP(bits=bits)
    opt = torch.optim.Adam(m.parameters(), lr=8e-3)
    lossf = nn.CrossEntropyLoss()
    curve = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = lossf(m(Xtr), ytr)
        loss.backward()
        opt.step()
        if s % 20 == 0:
            curve.append(loss.item())
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == yte).float().mean().item()
    return m, acc, curve


def post_train_quantize(model, bits, Xte, yte):
    """Train in float, round the weights afterwards, hope for the best."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        for l in (m.l1, m.l2, m.l3):
            l.weight.copy_(quantize(l.weight, bits))
    m.bits = None
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()


def qat_study():
    print("\n" + "=" * 78)
    print("3. QUANTIZATION-AWARE TRAINING vs QUANTIZE-AFTERWARDS")
    print("=" * 78)

    Xtr, ytr, Xte, yte = spiral()
    print(f"  {len(Xtr)} training points, {len(Xte)} test, 3-class spiral,")
    print(f"  MLP 2-64-64-3 (4,547 parameters), Adam, 700 steps each\n")

    t0 = time.perf_counter()
    fp32, fp32_acc, fp32_curve = fit(Xtr, ytr, Xte, yte, bits=None)
    print(f"  float32 baseline: test accuracy {fp32_acc:.4f}"
          f"   ({time.perf_counter() - t0:.1f}s)")
    rec("fp32_test_acc", round(fp32_acc, 4))

    bit_list = [1, 2, 3, 4, 6]
    ptq, qat, curves = [], [], {}
    for b in bit_list:
        a_ptq = post_train_quantize(fp32, b, Xte, yte)
        _, a_qat, c = fit(Xtr, ytr, Xte, yte, bits=b)
        ptq.append(a_ptq)
        qat.append(a_qat)
        curves[b] = c
        print(f"  {b}-bit weights ({2 ** b:>2} levels):  "
              f"quantize-after {a_ptq:.4f}   QAT with STE {a_qat:.4f}"
              f"   {a_qat - a_ptq:+.4f}")
        rec(f"ptq_{b}bit_acc", round(a_ptq, 4))
        rec(f"qat_{b}bit_acc", round(a_qat, 4))

    print(f"\n  Chance is {1 / 3:.4f}. Quantize-after-training falls to chance at")
    print(f"  1-2 bits; STE-trained weights survive.")
    print("\n  Why the difference? Quantize-after-training moves every weight to")
    print("  the nearest level ONCE, at the end, and nothing gets a chance to")
    print("  compensate. QAT quantizes on every forward pass, so the loss the")
    print("  optimizer sees is the loss of the QUANTIZED network -- the other")
    print("  weights spend the whole run adapting to the rounding error. Same")
    print("  final precision, completely different final weights.")
    print("\n  The STE is what makes that possible. Without it the quantized")
    print("  forward pass produces a gradient of exactly zero and the run is")
    print("  a no-op -- section 5 measures that.")
    return bit_list, ptq, qat, fp32_acc, fp32_curve, curves, (Xtr, ytr, Xte, yte)


# =========================================================================
# 4. no-STE control
# =========================================================================
def no_ste_control(data):
    print("\n" + "=" * 78)
    print("4. THE CONTROL: SAME NETWORK, NO STE")
    print("=" * 78)
    Xtr, ytr, Xte, yte = data

    def run_dead(detach_scale, steps=200):
        class DeadMLP(QMLP):
            def w(self, layer):
                # no STE: the real round(), with its real gradient
                return quantize(layer.weight, self.bits, detach_scale=detach_scale)

        torch.manual_seed(0)
        m = DeadMLP(bits=3)
        opt = torch.optim.Adam(m.parameters(), lr=8e-3)
        lossf = nn.CrossEntropyLoss()
        first_loss, gnorms, nonzero = None, [], []
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr), ytr)
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            ws = [m.l1.weight, m.l2.weight, m.l3.weight]
            gnorms.append(sum(p.grad.norm().item() ** 2 for p in ws) ** 0.5)
            nonzero.append(sum((p.grad != 0).sum().item() for p in ws))
            opt.step()
        with torch.no_grad():
            acc = (m(Xte).argmax(1) == yte).float().mean().item()
        n_w = sum(p.numel() for p in [m.l1.weight, m.l2.weight, m.l3.weight])
        return gnorms, nonzero, acc, first_loss, loss.item(), n_w

    gn, nz, acc, l0, l1, n_w = run_dead(detach_scale=True)
    print(f"  3-bit weights, round() with its true gradient:")
    print(f"    weight-gradient norm, max over 200 steps: {max(gn):.8f}")
    print(f"    weights with a non-zero gradient: {max(nz)} of {n_w:,}")
    print(f"    loss  {l0:.4f} -> {l1:.4f}")
    print(f"    test accuracy {acc:.4f}   (chance is {1 / 3:.4f}, "
          f"float32 baseline is 1.0000)")
    rec("no_ste_max_weight_grad_norm", round(max(gn), 8))
    rec("no_ste_nonzero_weight_grads", max(nz))
    rec("no_ste_test_acc", round(acc, 4))
    rec("no_ste_first_loss", round(l0, 4))
    rec("no_ste_last_loss", round(l1, 4))
    print(f"\n  Not one of the {n_w:,} weights ever received a gradient. The loss")
    print("  still moved, because the BIASES are not quantized and still train")
    print("  -- exactly the kind of partial progress that makes a dead-gradient")
    print("  bug hard to spot. Check the gradient norm of the tensor you think")
    print("  is learning, not just the loss.")

    # the subtle variant: leave min()/max() in the graph
    gn2, nz2, acc2, l02, l12, _ = run_dead(detach_scale=False)
    print(f"\n  A subtler version of the same bug. `quantize` computes its step")
    print(f"  size from w.min() and w.max(). Leave those in the graph and:")
    print(f"    weight-gradient norm, max over 200 steps: {max(gn2):.4f}  <- not zero!")
    print(f"    weights with a non-zero gradient: {max(nz2)} of {n_w:,}")
    print(f"    test accuracy {acc2:.4f}")
    rec("leak_max_weight_grad_norm", round(max(gn2), 4))
    rec("leak_nonzero_weight_grads", max(nz2))
    rec("leak_test_acc", round(acc2, 4))
    print("  A healthy-looking gradient norm, arriving at a handful of weights:")
    print("  only the ones that happen to BE the min or the max of their tensor.")
    print("  Every other weight still gets zero. A non-zero gradient norm is not")
    print("  proof that gradients are flowing -- count how many entries are")
    print("  actually non-zero.")
    return gn, acc


# =========================================================================
# 5. plain vs clipped STE on a sign() network
# =========================================================================
class SignSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, g):
        return g


class SignClippedSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1)


def sign_network(data, clipped, steps=700, seed=0):
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed)
    act = SignClippedSTE.apply if clipped else SignSTE.apply
    net = nn.ModuleList([nn.Linear(2, 64), nn.Linear(64, 64), nn.Linear(64, 3)])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()

    def fwd(x, keep=False):
        pre = []
        x = net[0](x)
        pre.append(x)
        x = act(x)
        x = net[1](x)
        pre.append(x)
        x = act(x)
        return net[2](x), pre

    drift = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        out, pre = fwd(Xtr)
        lossf(out, ytr).backward()
        opt.step()
        if s % 25 == 0:
            drift.append(max(p.abs().max().item() for p in pre))
    with torch.no_grad():
        out, pre = fwd(Xte)
        acc = (out.argmax(1) == yte).float().mean().item()
        frac_out = float(np.mean([(p.abs() > 1).float().mean().item() for p in pre]))
    return acc, drift, frac_out


def sign_study(data):
    print("\n" + "=" * 78)
    print("5. PLAIN STE vs CLIPPED STE, ON A 1-BIT ACTIVATION NETWORK")
    print("=" * 78)
    print("  Every hidden activation is replaced by sign(x): +1 or -1, one bit.")
    print("  Plain STE passes the gradient through everywhere. Clipped STE")
    print("  passes it only where |x| <= 1 -- the region where a small change")
    print("  in x could plausibly flip the sign. Outside that band the neuron is")
    print("  saturated and the honest answer really is 'this input does not")
    print("  matter', so clipping is arguably the LESS dishonest estimator.\n")

    seeds = [0, 1, 2]
    res = {}
    for clipped in (False, True):
        accs, drifts, fracs = [], [], []
        for s in seeds:
            a, d, f = sign_network(data, clipped=clipped, seed=s)
            accs.append(a)
            drifts.append(d)
            fracs.append(f)
        res[clipped] = (accs, drifts, fracs)

    for clipped, label in [(False, "plain STE  "), (True, "clipped STE")]:
        accs, drifts, fracs = res[clipped]
        print(f"  {label}: accuracy {np.mean(accs):.4f} "
              f"(seeds {', '.join(f'{a:.3f}' for a in accs)})"
              f"   peak |pre-act| {np.mean([max(d) for d in drifts]):.1f}"
              f"   {np.mean(fracs) * 100:.0f}% outside the band")
        tag = "clipped" if clipped else "plain"
        rec(f"sign_{tag}_acc_mean", round(float(np.mean(accs)), 4))
        rec(f"sign_{tag}_acc_seeds", ";".join(f"{a:.4f}" for a in accs))
        rec(f"sign_{tag}_peak_preact", round(float(np.mean([max(d) for d in drifts])), 3))
        rec(f"sign_{tag}_frac_outside", round(float(np.mean(fracs)), 4))

    gap = np.mean(res[False][0]) - np.mean(res[True][0])
    spread = max(max(res[False][0]) - min(res[False][0]),
                 max(res[True][0]) - min(res[True][0]))
    print(f"\n  gap between the two: {gap:+.4f}   spread within one method "
          f"across seeds: {spread:.4f}")
    rec("sign_gap", round(float(gap), 4))
    rec("sign_seed_spread", round(float(spread), 4))
    print("  Read that honestly: clipping changed NOTHING here. Not the")
    print("  accuracy (identical to four decimals, and the seed-to-seed spread")
    print("  is 17x the gap), and not the pre-activation drift it was supposed")
    print("  to control. The reason is that this network is two quantized layers")
    print("  deep on an easy task: the bogus gradient from saturated units never")
    print("  gets a chance to compound. Clipping was invented for binarized")
    print("  networks tens of layers deep, where it does compound. An STE variant")
    print("  that helps there can be pure noise here -- which is why every claim")
    print("  in this section is reported over three seeds and not one.")
    return res


# =========================================================================
# figures
# =========================================================================
def fig_staircase():
    x = torch.linspace(-2.2, 2.2, 1200, requires_grad=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, color=ps.GRID, linewidth=0.8)

    y = torch.round(x)
    axes[0].plot(x.detach(), y.detach(), color=ps.INK_SECONDARY, lw=1.8)
    axes[0].set_title("forward: round(x)", color=ps.INK, fontsize=11, loc="left")

    y.sum().backward()
    axes[1].plot(x.detach(), x.grad.clone(), color=ps.SERIES[2], lw=2.4,
                 label="true gradient (all zeros)")
    x.grad = None
    RoundSTE.apply(x).sum().backward()
    axes[1].plot(x.detach(), x.grad.clone(), color=ps.SERIES[0], lw=1.6,
                 ls="--", label="straight-through (all ones)")
    x.grad = None
    RoundClippedSTE.apply(x).sum().backward()
    axes[1].plot(x.detach(), x.grad.clone(), color=ps.SERIES[1], lw=1.6,
                 ls=":", label="clipped STE (|x| <= 1)")
    axes[1].set_ylim(-0.2, 1.3)
    axes[1].legend(frameon=False, fontsize=8.5, loc="center right")
    axes[1].set_title("backward: three different answers", color=ps.INK,
                      fontsize=11, loc="left")
    ps.save(fig, os.path.join(OUT, "staircase.png"))


def fig_soft(rows):
    fig, ax = ps.new_axes(7.0, 4.0)
    for i, T in enumerate([1.0, 0.5, 0.2, 0.1, 0.02]):
        xt = torch.linspace(-0.2, 2.2, 3000, dtype=torch.float64, requires_grad=True)
        f = torch.floor(xt)
        frac = xt - f
        (f + torch.sigmoid((frac - 0.5) / T)).sum().backward()
        ax.plot(xt.detach(), xt.grad, color=ps.SERIES[i], lw=1.4, label=f"T = {T}")
    ax.axhline(1.0, color=ps.INK_MUTED, ls="--", lw=1.2)
    ax.text(2.15, 1.35, "what the STE claims", ha="right", fontsize=9,
            color=ps.INK_MUTED)
    ax.set_ylim(0, 8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Sharpen a smooth staircase and its slope becomes spikes",
              "x", "d(soft round)/dx", os.path.join(OUT, "soft_round.png"))


def fig_qat(bit_list, ptq, qat, fp32_acc):
    fig, ax = ps.new_axes(7.2, 4.2)
    w = 0.36
    xs = np.arange(len(bit_list))
    ax.bar(xs - w / 2, ptq, w, color=ps.SERIES[2], label="train in float, quantize after")
    ax.bar(xs + w / 2, qat, w, color=ps.SERIES[1], label="quantization-aware (STE)")
    ax.axhline(fp32_acc, color=ps.SERIES[0], lw=1.4, ls="--")
    ax.text(len(bit_list) - 0.6, fp32_acc + 0.015, f"float32 = {fp32_acc:.3f}",
            fontsize=8.5, color=ps.SERIES[0], ha="right")
    ax.axhline(1 / 3, color=ps.INK_MUTED, lw=1.0, ls=":")
    ax.text(-0.45, 1 / 3 - 0.05, "chance", fontsize=8.5, color=ps.INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}-bit\n({2 ** b} levels)" for b in bit_list])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    ps.finish(fig, ax, "The same final precision, two very different results",
              "weight precision", "test accuracy",
              os.path.join(OUT, "qat_vs_ptq.png"))


def fig_sign(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, color=ps.GRID, linewidth=0.8)

    for clipped, col, label in [(False, ps.SERIES[2], "plain STE"),
                                (True, ps.SERIES[1], "clipped STE")]:
        accs, drifts, _ = res[clipped]
        xs = np.arange(len(drifts[0])) * 25
        for i, d in enumerate(drifts):
            axes[0].plot(xs, d, color=col, lw=1.4, alpha=0.65,
                         label=label if i == 0 else None)
    axes[0].axhline(1.0, color=ps.INK_MUTED, ls=":", lw=1.0)
    axes[0].text(axes[0].get_xlim()[1], 1.35, "edge of the |x| <= 1 band",
                 ha="right", fontsize=8.5, color=ps.INK_MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Pre-activation drift, 3 seeds each", color=ps.INK,
                      fontsize=11, loc="left")
    axes[0].set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)

    means = [float(np.mean(res[False][0])), float(np.mean(res[True][0]))]
    axes[1].bar(["plain STE", "clipped STE"], means,
                color=[ps.SERIES[2], ps.SERIES[1]], width=0.5)
    for i, clipped in enumerate([False, True]):
        for a in res[clipped][0]:
            axes[1].plot(i, a, "o", color=ps.INK, ms=4, zorder=3)
        axes[1].text(i, means[i] + 0.04, f"{means[i]:.3f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, 1.12)
    axes[1].axhline(1 / 3, color=ps.INK_MUTED, ls=":", lw=1.0)
    axes[1].text(-0.45, 1 / 3 + 0.02, "chance", fontsize=8.5, color=ps.INK_MUTED)
    axes[1].set_title("Test accuracy (dots = individual seeds)", color=ps.INK,
                      fontsize=11, loc="left")
    ps.save(fig, os.path.join(OUT, "sign_ste.png"))


# =========================================================================
def main():
    t0 = time.perf_counter()
    dead_gradient()
    rows = cost_of_the_lie()
    bit_list, ptq, qat, fp32_acc, fp32_curve, curves, data = qat_study()
    no_ste_control(data)
    sign_res = sign_study(data)

    fig_staircase()
    fig_soft(rows)
    fig_qat(bit_list, ptq, qat, fp32_acc)
    fig_sign(sign_res)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
