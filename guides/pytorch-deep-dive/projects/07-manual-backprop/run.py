"""Project 07 — Manual backprop.

Train a 2-layer MLP without ever calling `.backward()`. Every gradient is
written out by hand, then checked three ways:

  1. against autograd, tensor by tensor (including the intermediates)
  2. against central finite differences -- the ground truth that does not
     depend on either implementation being right
  3. by training: hand-gradient run vs autograd run, same seed, same weights

Plus the two mistakes that hand-written backprop makes most often:
a bias gradient that forgot to sum over the batch, and a loss that used
`sum` where it meant `mean`.

Runs in about 20 seconds on CPU. No downloads.
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(1)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # finite differences need the digits

FINDINGS = {}


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# data: three interleaved spiral arms, not separable by any straight line
# =========================================================================
def spiral(n_per=70, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n_per)
        t = np.linspace(c * 4, (c + 1) * 4, n_per) + rng.normal(0, 0.25, n_per)
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], 1))
        y.append(np.full(n_per, c))
    return (torch.tensor(np.concatenate(X)),
            torch.tensor(np.concatenate(y), dtype=torch.long))


def init_params(D, H, C, seed=1):
    g = torch.Generator().manual_seed(seed)
    # Kaiming-ish scaling: variance 1/fan_in keeps activations from exploding
    W1 = torch.randn(D, H, generator=g) * (1.0 / D) ** 0.5
    b1 = torch.zeros(H)
    W2 = torch.randn(H, C, generator=g) * (1.0 / H) ** 0.5
    b2 = torch.zeros(C)
    return [W1, b1, W2, b2]


# =========================================================================
# forward + the hand-written backward
# =========================================================================
def forward(params, x):
    """Keep every intermediate: backward needs them, that is the whole point."""
    W1, b1, W2, b2 = params
    z1 = x @ W1 + b1        # (N, H)
    h = torch.tanh(z1)      # (N, H)
    z2 = h @ W2 + b2        # (N, C)  -- these are the logits
    return z1, h, z2


def loss_from_logits(z2, y):
    """Mean cross-entropy, written out so nothing is hidden."""
    m = z2.max(dim=1, keepdim=True).values            # subtract the max first:
    logsumexp = m + (z2 - m).exp().sum(1, keepdim=True).log()   # stops exp() overflowing
    logp = z2 - logsumexp                             # (N, C)
    return -logp[torch.arange(len(y)), y].mean(), logp


def manual_backward(params, x, y, z1, h, z2, logp):
    """Every line here is one link of the chain rule."""
    W1, b1, W2, b2 = params
    N, C = z2.shape

    # --- softmax + cross-entropy, fused ----------------------------------
    # dL/dz2 = (p - onehot) / N.  Two derivatives collapse into one line:
    # the softmax's Jacobian and the log's 1/p cancel almost everything.
    p = logp.exp()                                   # (N, C) probabilities
    dz2 = p.clone()
    dz2[torch.arange(N), y] -= 1.0
    dz2 /= N                                         # because the loss used .mean()

    # --- second linear layer:  z2 = h @ W2 + b2 --------------------------
    # dL/dW2 = h^T @ dz2       (H,N)@(N,C) -> (H,C), matching W2
    # dL/db2 = dz2 summed over the batch: b2 was BROADCAST to N rows, and the
    #          backward of a broadcast is a sum over the stretched axis.
    # dL/dh  = dz2 @ W2^T      (N,C)@(C,H) -> (N,H), matching h
    dW2 = h.T @ dz2
    db2 = dz2.sum(0)
    dh = dz2 @ W2.T

    # --- tanh:  h = tanh(z1),  dh/dz1 = 1 - tanh(z1)^2 = 1 - h^2 ---------
    # Note it is written in terms of the OUTPUT h, not the input z1: cheaper,
    # and exactly what torch's TanhBackward does.
    dz1 = dh * (1.0 - h * h)

    # --- first linear layer ----------------------------------------------
    dW1 = x.T @ dz1
    db1 = dz1.sum(0)
    dx = dz1 @ W1.T

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "z1": dz1, "h": dh, "z2": dz2, "x": dx}


# =========================================================================
# 1. check every tensor against autograd
# =========================================================================
def check_vs_autograd(X, y, D, H, C):
    print("=" * 78)
    print("1. HAND-WRITTEN GRADIENTS vs AUTOGRAD, TENSOR BY TENSOR")
    print("=" * 78)

    params = init_params(D, H, C)
    xa = X.clone().requires_grad_(True)
    pa = [p.clone().requires_grad_(True) for p in params]
    z1, h, z2 = forward(pa, xa)
    for t in (z1, h, z2):
        t.retain_grad()          # intermediates are not leaves; ask for their .grad
    loss, logp = loss_from_logits(z2, y)
    loss.backward()

    mine = manual_backward(params, X, y, z1.detach(), h.detach(), z2.detach(),
                           logp.detach())

    # torch's own fused cross-entropy, as a third opinion on the loss value
    ref_loss = F.cross_entropy(z2.detach(), y)
    print(f"  loss  ours {loss.item():.15f}   F.cross_entropy {ref_loss.item():.15f}")
    rec("loss_vs_F_cross_entropy", f"{abs(loss.item() - ref_loss.item()):.3e}")

    names = ["W1", "b1", "W2", "b2", "z1", "h", "z2", "x"]
    autos = {"W1": pa[0].grad, "b1": pa[1].grad, "W2": pa[2].grad, "b2": pa[3].grad,
             "z1": z1.grad, "h": h.grad, "z2": z2.grad, "x": xa.grad}
    print(f"\n  {'tensor':<8}{'shape':<12}{'max |mine - autograd|':>24}"
          f"{'relative':>14}")
    worst = 0.0
    rows = []
    for n in names:
        d = (mine[n] - autos[n]).abs().max().item()
        scale = autos[n].abs().max().item()
        rel = d / scale if scale > 0 else 0.0
        worst = max(worst, rel)
        rows.append((n, d, rel))
        print(f"  {n:<8}{str(tuple(mine[n].shape)):<12}{d:>24.3e}{rel:>14.3e}")
    print(f"\n  worst relative disagreement: {worst:.2e}  (float64 rounding is ~1e-16)")
    rec("autograd_worst_relative_diff", f"{worst:.3e}")
    return rows, params


# =========================================================================
# 2. check against finite differences
# =========================================================================
def check_vs_finite_differences(X, y, D, H, C, n_probe=40):
    print("\n" + "=" * 78)
    print("2. ... AND AGAINST FINITE DIFFERENCES")
    print("=" * 78)
    print("  Autograd and my derivation could in principle share a mistake.")
    print("  Finite differences share nothing with either: just move a weight a")
    print("  little and watch the loss. It is slow and slightly inexact, which")
    print("  is exactly why it is only used as a check.\n")

    params = init_params(D, H, C)
    z1, h, z2 = forward(params, X)
    _, logp = loss_from_logits(z2, y)
    mine = manual_backward(params, X, y, z1, h, z2, logp)

    def loss_of(ps_):
        _, _, zz = forward(ps_, X)
        return loss_from_logits(zz, y)[0].item()

    rng = np.random.default_rng(0)
    eps = 1e-6
    rows = []
    for pi, name in enumerate(["W1", "b1", "W2", "b2"]):
        flat = params[pi].flatten()
        idxs = rng.choice(flat.numel(), size=min(n_probe, flat.numel()), replace=False)
        worst = 0.0
        for i in idxs:
            saved = flat[i].item()
            flat[i] = saved + eps
            lp = loss_of(params)
            flat[i] = saved - eps
            lm = loss_of(params)
            flat[i] = saved
            # CENTRAL difference: (f(x+e) - f(x-e)) / 2e. The one-sided version
            # (f(x+e) - f(x)) / e has error proportional to e; the central one
            # has error proportional to e^2, so it is far more accurate for free.
            num = (lp - lm) / (2 * eps)
            ana = mine[name].flatten()[i].item()
            worst = max(worst, abs(num - ana) / max(abs(ana), 1e-12))
        rows.append((name, worst))
        print(f"  {name:<4} {len(idxs)} random entries   worst relative error {worst:.2e}")
        rec(f"fd_{name}_worst_rel", f"{worst:.3e}")

    # one-sided vs central, on a single weight, to show the difference
    flat = params[0].flatten()
    i = 0
    saved = flat[i].item()
    l0 = loss_of(params)
    flat[i] = saved + eps
    lp = loss_of(params)
    flat[i] = saved - eps
    lm = loss_of(params)
    flat[i] = saved
    ana = mine["W1"].flatten()[i].item()
    one = (lp - l0) / eps
    cen = (lp - lm) / (2 * eps)
    print(f"\n  on W1[0,0]:  analytic {ana:.12f}")
    print(f"               one-sided {one:.12f}   error {abs(one - ana):.2e}")
    print(f"               central   {cen:.12f}   error {abs(cen - ana):.2e}")
    rec("fd_one_sided_error", f"{abs(one - ana):.3e}")
    rec("fd_central_error", f"{abs(cen - ana):.3e}")
    return rows


# =========================================================================
# 3. train it, by hand, and compare to autograd training
# =========================================================================
def train(X, y, D, H, C, steps=400, lr=1.0, mode="manual", loss_reduction="mean",
          lockstep=False):
    """One training run. `lockstep=True` additionally asks autograd for the
    gradient at every step and records how far it is from the hand-derived one
    -- both computed from the SAME weights, so nothing can drift."""
    params = init_params(D, H, C)
    curve, accs, agree = [], [], []
    for _ in range(steps):
        if lockstep:
            ps_ = [p.clone().requires_grad_(True) for p in params]
            zz1, hh, zz2 = forward(ps_, X)
            loss_a, _ = loss_from_logits(zz2, y)
            loss_a.backward()
        if mode == "manual":
            z1, h, z2 = forward(params, X)
            loss, logp = loss_from_logits(z2, y)
            if loss_reduction == "sum":
                loss = loss * len(y)
            g = manual_backward(params, X, y, z1, h, z2, logp)
            if loss_reduction == "sum":
                for k in g:
                    g[k] = g[k] * len(y)
            if lockstep:
                agree.append(max((g[n] - q.grad).abs().max().item()
                                 for n, q in zip(["W1", "b1", "W2", "b2"], ps_)))
            for p, name in zip(params, ["W1", "b1", "W2", "b2"]):
                p -= lr * g[name]
        else:
            ps_ = [p.clone().requires_grad_(True) for p in params]
            z1, h, z2 = forward(ps_, X)
            loss, _ = loss_from_logits(z2, y)
            if loss_reduction == "sum":
                loss = loss * len(y)
            loss.backward()
            with torch.no_grad():
                for p, q in zip(params, ps_):
                    p -= lr * q.grad
        curve.append(loss.item() / (len(y) if loss_reduction == "sum" else 1))
        with torch.no_grad():
            accs.append((forward(params, X)[2].argmax(1) == y).double().mean().item())
    return curve, accs, params, agree


def training_comparison(X, y, D, H, C):
    print("\n" + "=" * 78)
    print("3. TRAINING WITH HAND GRADIENTS ONLY")
    print("=" * 78)

    t0 = time.perf_counter()
    mc, ma, mp, agree = train(X, y, D, H, C, mode="manual", lockstep=True)
    tm = time.perf_counter() - t0
    t0 = time.perf_counter()
    ac, aa, ap, _ = train(X, y, D, H, C, mode="autograd")
    ta = time.perf_counter() - t0

    curve_diff = max(abs(a - b) for a, b in zip(mc, ac))
    param_diff = max((p - q).abs().max().item() for p, q in zip(mp, ap))
    print(f"  final loss     manual {mc[-1]:.12f}   autograd {ac[-1]:.12f}")
    print(f"  final accuracy manual {ma[-1]:.4f}          autograd {aa[-1]:.4f}")
    print(f"  wall clock     manual {tm:.2f}s (incl. the lockstep check)"
          f"   autograd {ta:.2f}s")

    print(f"\n  LOCKSTEP CHECK -- same weights, both gradients, every step:")
    print(f"    worst |manual - autograd| over all 400 steps: {max(agree):.2e}")
    print(f"    median: {sorted(agree)[len(agree) // 2]:.2e}")
    print("    The derivation is right at every single step, not just step 1.")

    print(f"\n  FREE-RUNNING -- two independent 400-step runs:")
    print(f"    worst loss disagreement  : {curve_diff:.2e}")
    print(f"    worst final-weight gap   : {param_diff:.2e}")
    print(f"    step 1 loss gap {abs(mc[0] - ac[0]):.2e}"
          f"   step 400 loss gap {abs(mc[-1] - ac[-1]):.2e}")
    print("\n  Read those two blocks together. Per step the two gradients agree to")
    print("  ~1e-16 -- the last bit of a float64. But gradient descent is a chaotic")
    print("  map: it feeds its own output back in 400 times, and a last-bit")
    print("  difference doubles every so often until it is visible. Both runs end")
    print("  at the same accuracy on different weights. This is why 'my rerun gave")
    print("  a slightly different loss' is almost never a bug -- and why comparing")
    print("  two implementations by their loss CURVES is the wrong test. Compare")
    print("  the gradients at fixed weights, like the lockstep check above.")
    rec("train_final_loss_manual", round(mc[-1], 12))
    rec("train_final_loss_autograd", round(ac[-1], 12))
    rec("train_final_acc_manual", round(ma[-1], 4))
    rec("train_final_acc_autograd", round(aa[-1], 4))
    rec("lockstep_worst_grad_diff", f"{max(agree):.3e}")
    rec("lockstep_median_grad_diff", f"{sorted(agree)[len(agree) // 2]:.3e}")
    rec("freerun_curve_max_diff", f"{curve_diff:.3e}")
    rec("freerun_param_max_diff", f"{param_diff:.3e}")
    rec("train_seconds_manual", round(tm, 2))
    rec("train_seconds_autograd", round(ta, 2))
    return mc, ac, ma, mp, agree


# =========================================================================
# 4. the two classic hand-backprop bugs
# =========================================================================
def classic_bugs(X, y, D, H, C):
    print("\n" + "=" * 78)
    print("4. THE TWO MISTAKES EVERY HAND-DERIVATION MAKES")
    print("=" * 78)

    params = init_params(D, H, C)
    z1, h, z2 = forward(params, X)
    _, logp = loss_from_logits(z2, y)
    g = manual_backward(params, X, y, z1, h, z2, logp)

    # --- bug A: forgetting the sum in the bias gradient -------------------
    print("\n  BUG A: db2 = dz2   (forgot to sum over the batch)")
    dz2 = g["z2"]
    print(f"    dz2 has shape {tuple(dz2.shape)}, but b2 has shape {tuple(params[3].shape)}.")
    try:
        params[3] - 0.1 * dz2
        shape_ok = True
    except RuntimeError as e:
        shape_ok = False
        print(f"    subtracting it raises: {str(e)[:60]}")
    print(f"    does `b2 - lr*dz2` raise? {not shape_ok}")
    print(f"    it does NOT: broadcasting turns b2 (3,) into (210, 3) and the")
    print(f"    update silently produces a {tuple((params[3] - 0.1 * dz2).shape)} 'bias'.")
    print("    Rule: the backward of a BROADCAST is a SUM over the axis that was")
    print("    stretched. b2 was copied to 210 rows in the forward pass, so 210")
    print("    gradients come back and must be added into one.")
    print(f"    sum-over-batch  db2 = {g['b2'].numpy().round(6)}")
    print(f"    mean-over-batch db2 = {(dz2.mean(0)).numpy().round(6)}  <- off by 1/N = 1/{len(y)}")
    rec("bug_bias_broadcast_shape", str(tuple((params[3] - 0.1 * dz2).shape)))
    rec("bug_bias_mean_vs_sum_ratio", round((g["b2"] / dz2.mean(0)).mean().item(), 3))

    # --- bug B: sum vs mean ----------------------------------------------
    print("\n  BUG B: loss = total instead of mean")
    N = len(y)
    print(f"    Every gradient is then exactly N = {N} times larger.")
    sum_c, sum_a, _, _ = train(X, y, D, H, C, steps=60, lr=1.0, loss_reduction="sum")
    mean_c, mean_a, _, _ = train(X, y, D, H, C, steps=60, lr=1.0, loss_reduction="mean")
    small_c, small_a, _, _ = train(X, y, D, H, C, steps=60, lr=1.0 / N, loss_reduction="sum")
    print(f"    mean loss, lr=1.0     : per-sample loss {mean_c[-1]:.6f}, acc {mean_a[-1]:.3f}")
    print(f"    sum  loss, lr=1.0     : per-sample loss {sum_c[-1]:.6f}, acc {sum_a[-1]:.3f}")
    print(f"    sum  loss, lr=1.0/{N} : per-sample loss {small_c[-1]:.6f}, acc {small_a[-1]:.3f}")
    match = max(abs(a - b) for a, b in zip(small_c, mean_c))
    print(f"    the last two curves agree to {match:.2e} -- a `sum` loss is not")
    print("    'wrong', it is the same run at a learning rate N times larger.")
    print("    That is why it usually shows up as 'my loss became nan' rather")
    print("    than as a shape error.")
    rec("sum_loss_final", round(sum_c[-1], 6))
    rec("mean_loss_final", round(mean_c[-1], 6))
    rec("sum_lr_scaled_final", round(small_c[-1], 6))
    rec("sum_vs_scaled_curve_diff", f"{match:.3e}")
    return mean_c, sum_c, small_c


# =========================================================================
# figures
# =========================================================================
def fig_agreement(rows):
    names = [r[0] for r in rows]
    rel = [max(r[2], 1e-19) for r in rows]
    fig, ax = ps.new_axes(7.0, 3.6)
    ax.bar(names, rel, color=ps.SERIES[0], width=0.55)
    ax.axhline(2.2e-16, color=ps.SERIES[2], lw=1.2, ls="--")
    ax.text(len(names) - 0.4, 2.6e-16, "float64 machine epsilon", ha="right",
            fontsize=8.5, color=ps.SERIES[2])
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 1e-13)
    ps.finish(fig, ax, "Hand-derived vs autograd: relative disagreement",
              "tensor", "max relative difference",
              os.path.join(OUT, "gradient_agreement.png"))


def fig_curves(mc, ac, ma):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, color=ps.GRID, linewidth=0.8)
    axes[0].plot(mc, color=ps.SERIES[0], lw=3.2, alpha=0.55, label="hand gradients")
    axes[0].plot(ac, color=ps.SERIES[2], lw=1.2, ls="--", label="loss.backward()")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Cross-entropy", color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].plot(ma, color=ps.SERIES[1], lw=1.6)
    axes[1].set_title("Training accuracy (hand gradients)", color=ps.INK,
                      fontsize=11, loc="left")
    axes[1].set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ps.save(fig, os.path.join(OUT, "training_curves.png"))


def fig_divergence(mc, ac, agree):
    fig, ax = ps.new_axes(7.2, 4.0)
    eps = 1e-18
    ax.plot([max(abs(a - b), eps) for a, b in zip(mc, ac)], color=ps.SERIES[2],
            lw=1.5, label="two free-running trainings: |loss difference|")
    ax.plot([max(v, eps) for v in agree], color=ps.SERIES[0], lw=1.5,
            label="same weights, both gradients: max |difference|")
    ax.axhline(2.2e-16, color=ps.INK_MUTED, lw=1.0, ls=":")
    ax.text(len(agree) * 0.99, 3.5e-16, "float64 machine epsilon", ha="right",
            fontsize=8.5, color=ps.INK_MUTED)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ps.finish(fig, ax, "The derivation stays exact; the trajectory does not",
              "step", "absolute difference (log scale)",
              os.path.join(OUT, "chaos.png"))


def fig_boundary(params, X, y):
    with torch.no_grad():
        xs = torch.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 120)
        ys = torch.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 120)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.flatten(), gy.flatten()], 1)
        Z = forward(params, grid)[2].argmax(1).reshape(gx.shape).numpy()
    fig, ax = ps.new_axes(5.4, 4.6)
    ax.contourf(xs.numpy(), ys.numpy(), Z, levels=[-.5, .5, 1.5, 2.5],
                colors=ps.SERIES[:3], alpha=0.15)
    for c in range(3):
        m = (y == c).numpy()
        ax.scatter(X[m, 0], X[m, 1], s=16, color=ps.SERIES[c],
                   edgecolor="white", lw=0.5)
    ax.grid(False)
    ps.finish(fig, ax, "Trained without ever calling .backward()",
              "x₀", "x₁", os.path.join(OUT, "decision_boundary.png"))


def fig_sum_vs_mean(mean_c, sum_c, small_c):
    fig, ax = ps.new_axes(7.0, 3.9)
    ax.plot(mean_c, color=ps.SERIES[0], lw=2.6, alpha=0.6, label="mean loss, lr=1.0")
    ax.plot(small_c, color=ps.SERIES[2], lw=1.2, ls="--", label="sum loss, lr=1.0/N")
    ax.plot(sum_c, color=ps.SERIES[3], lw=1.6, label="sum loss, lr=1.0")
    ax.legend(frameon=False, fontsize=9)
    ax.set_yscale("log")
    ps.finish(fig, ax, "`sum` instead of `mean` is a learning-rate bug, not a math bug",
              "step", "cross-entropy per sample (log scale)",
              os.path.join(OUT, "sum_vs_mean.png"))


# =========================================================================
def main():
    X, y = spiral()
    D, H, C = 2, 24, 3
    print(f"  data: {len(X)} points, {C} classes, MLP {D}-{H}-{C} "
          f"({D * H + H + H * C + C} parameters), float64\n")

    rows, _ = check_vs_autograd(X, y, D, H, C)
    check_vs_finite_differences(X, y, D, H, C)
    mc, ac, ma, mp, agree = training_comparison(X, y, D, H, C)
    mean_c, sum_c, small_c = classic_bugs(X, y, D, H, C)

    fig_agreement(rows)
    fig_curves(mc, ac, ma)
    fig_divergence(mc, ac, agree)
    fig_boundary(mp, X, y)
    fig_sum_vs_mean(mean_c, sum_c, small_c)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")


if __name__ == "__main__":
    main()
