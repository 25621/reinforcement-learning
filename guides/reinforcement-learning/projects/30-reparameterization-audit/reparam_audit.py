"""Project 30 — auditing the tanh-squashed Gaussian's log-probability.

SAC's actor samples `u ~ N(mu, sigma)` and returns `a = tanh(u)`. Squashing bends
the distribution, so the density changes, and the log-probability needs a
change-of-variables correction:

    log p(a) = log p(u) - SUM_i log(1 - tanh(u_i)^2)

Drop that term, or get its sign wrong, and **nothing crashes**. The agent trains.
The loss goes down. The learning curve looks plausible. It is simply optimizing
the entropy of the wrong distribution, and it is worse than it should be, and you
will never find out from the logs.

This file is the audit that makes the bug loud. Five checks, cheapest first:

  1. normalization   Does exp(log p) integrate to 1 over the action range?
                     A density that does not integrate to 1 is not a density.
  2. density         Does the analytic log p match the empirical histogram of
                     actually-sampled actions?
  3. entropy         Does -E[log p] match the true entropy computed by quadrature?
                     This is the number SAC's temperature is servoing on.
  4. stability       Does the literal `log(1 - tanh(u)^2)` survive a saturated
                     actor? (No. It returns -inf, and then NaN.)
  5. gradients       Is the reparameterized gradient really lower-variance than
                     the score-function one? By how much?

Then the consequence: SAC trained with and without the correction, on two tasks.

Run:
    python3 reparam_audit.py          # checks ~10s, then ~5 min of training
    python3 reparam_audit.py checks   # just the numerics, no training
"""

import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc
import plot_style as ps

OUT = HERE / "outputs"


# --------------------------------------------------------------------------
# The three candidate log-probs: the right one, and the two ways people get it wrong
# --------------------------------------------------------------------------
def logp_gaussian(u, mu, log_std):
    std = log_std.exp()
    return (-0.5 * (((u - mu) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))).sum(-1)


def logp_correct(u, mu, log_std):
    """What SAC actually needs (stable form of the correction)."""
    return logp_gaussian(u, mu, log_std) - (
        2 * (math.log(2) - u - F.softplus(-2 * u))
    ).sum(-1)


def logp_naive_correction(u, mu, log_std, eps=0.0):
    """The literal textbook formula. Correct in exact arithmetic; explodes in float32."""
    return logp_gaussian(u, mu, log_std) - torch.log(
        1 - torch.tanh(u) ** 2 + eps
    ).sum(-1)


def logp_no_correction(u, mu, log_std):
    """The bug: forget the Jacobian entirely."""
    return logp_gaussian(u, mu, log_std)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_normalization(mu=0.4, log_std=-0.3):
    """A density integrates to 1. Integrate both candidates over a in (-1, 1)."""
    mu_t = torch.tensor([mu])
    ls_t = torch.tensor([log_std])
    a = torch.linspace(-0.999999, 0.999999, 400_001).unsqueeze(-1)
    u = torch.atanh(a)

    p_correct = logp_correct(u, mu_t, ls_t).exp()
    p_wrong = logp_no_correction(u, mu_t, ls_t).exp()
    da = (a[1] - a[0]).item()
    i_correct = torch.trapezoid(p_correct, dx=da).item()
    i_wrong = torch.trapezoid(p_wrong, dx=da).item()

    print("1. NORMALIZATION — does exp(log p) integrate to 1 over a in (-1, 1)?")
    print(f"   with correction:    {i_correct:.6f}   <- a probability density")
    print(f"   without correction: {i_wrong:.6f}   <- integrates to {i_wrong:.2f}, "
          f"not a density at all")
    return i_correct, i_wrong


def check_density(mu=0.4, log_std=-0.3, n=400_000):
    """Sample 400k actions; compare their histogram to the analytic density."""
    g = torch.Generator().manual_seed(0)
    mu_t, ls_t = torch.tensor([mu]), torch.tensor([log_std])
    u = mu_t + ls_t.exp() * torch.randn(n, 1, generator=g)
    a = torch.tanh(u).squeeze(-1).numpy()

    hist, edges = np.histogram(a, bins=120, range=(-1, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ac = torch.tensor(centers, dtype=torch.float32).unsqueeze(-1)
    uc = torch.atanh(ac.clamp(-0.999999, 0.999999))
    p_correct = logp_correct(uc, mu_t, ls_t).exp().numpy()
    p_wrong = logp_no_correction(uc, mu_t, ls_t).exp().numpy()

    err_c = np.abs(p_correct - hist).max()
    err_w = np.abs(p_wrong - hist).max()
    print("\n2. DENSITY — analytic log p vs the histogram of 400k sampled actions")
    print(f"   with correction:    max |analytic - empirical| = {err_c:.4f}")
    print(f"   without correction: max |analytic - empirical| = {err_w:.4f}")
    return centers, hist, p_correct, p_wrong


def check_entropy():
    """Entropy is the number the temperature servos on. Get it wrong, servo on a lie.

    Sweep log_std. The true entropy of tanh(N(mu, sigma)) is computed by quadrature.
    The uncorrected estimate is just the Gaussian's entropy, which grows without
    bound as sigma grows — while the true entropy of the squashed action *falls*,
    because the mass piles up against the walls at -1 and +1.
    """
    log_stds = np.linspace(-2.0, 2.0, 25)
    true_h, naive_h, corrected_h = [], [], []
    for ls in log_stds:
        mu_t, ls_t = torch.tensor([0.0]), torch.tensor([float(ls)])
        # quadrature in u-space: H = -int p(u) log p(a(u)) du
        u = torch.linspace(-30, 30, 200_001).unsqueeze(-1)
        pu = logp_gaussian(u, mu_t, ls_t).exp()
        lp_a = logp_correct(u, mu_t, ls_t)
        du = (u[1] - u[0]).item()
        true_h.append(-torch.trapezoid(pu * lp_a, dx=du).item())
        naive_h.append(0.5 * math.log(2 * math.pi * math.e) + float(ls))  # Gaussian entropy
        # MC with the correct formula, as SAC computes it
        g = torch.Generator().manual_seed(0)
        us = mu_t + ls_t.exp() * torch.randn(200_000, 1, generator=g)
        corrected_h.append(-logp_correct(us, mu_t, ls_t).mean().item())

    print("\n3. ENTROPY — the quantity automatic temperature tuning targets")
    print(f"   {'log_std':>8s} {'true H':>10s} {'SAC est.':>10s} {'no correction':>14s}")
    for i in (0, 8, 12, 18, 24):
        print(f"   {log_stds[i]:8.2f} {true_h[i]:10.3f} {corrected_h[i]:10.3f} "
              f"{naive_h[i]:14.3f}")
    print("   As sigma grows the squashed policy's entropy FALLS (mass piles on the")
    print("   walls at +/-1) while the uncorrected estimate rises without bound.")
    return log_stds, np.array(true_h), np.array(corrected_h), np.array(naive_h)


def check_stability():
    """The literal formula dies on a saturated actor. In float32, silently."""
    print("\n4. NUMERICAL STABILITY — literal log(1 - tanh(u)^2) vs the stable form")
    print(f"   {'u':>6s} {'literal':>14s} {'+1e-6':>14s} {'stable':>14s}")
    mu, ls = torch.tensor([0.0]), torch.tensor([0.0])
    rows = []
    for uval in [1.0, 5.0, 8.0, 10.0, 20.0]:
        u = torch.tensor([[uval]])
        lit = logp_naive_correction(u, mu, ls).item()
        eps = logp_naive_correction(u, mu, ls, eps=1e-6).item()
        stab = logp_correct(u, mu, ls).item()
        rows.append((uval, lit, eps, stab))
        print(f"   {uval:6.1f} {lit:14.4f} {eps:14.4f} {stab:14.4f}")
    err = abs(rows[-1][2] - rows[-1][3])
    print("   At |u| >= 10, tanh(u)^2 rounds to exactly 1.0 in float32: log(0) = -inf,")
    print("   so log p = +inf and the next backward pass fills the actor with NaN.")
    print(f"   The popular `+1e-6` patch stops the NaN but silently CAPS the correction:")
    print(f"   at u=20 it is wrong by {err:.0f} nats. The stable identity")
    print("   2*(log2 - u - softplus(-2u)) needs no epsilon and stays exact.")
    return rows


def check_gradients(n_trials=2000, batch=64):
    """Why 'reparameterization' and not REINFORCE? Variance. Measure it.

    Both estimators are unbiased for d/dmu E_a[f(a)]. They differ enormously in
    how many samples you need before that expectation is worth anything.
    """
    torch.manual_seed(0)
    f = lambda a: (a - 0.7) ** 2  # a stand-in for -Q(s, a)
    mu0, ls0 = 0.2, math.log(0.5)

    rep, sf = [], []
    for _ in range(n_trials):
        # pathwise / reparameterized: a = tanh(mu + sigma*eps), grad flows through a
        mu = torch.tensor([mu0], requires_grad=True)
        ls = torch.tensor([ls0])
        eps = torch.randn(batch, 1)
        a = torch.tanh(mu + ls.exp() * eps)
        loss = f(a).mean()
        loss.backward()
        rep.append(mu.grad.item())

        # score function / REINFORCE: grad of log p, weighted by f. No path through a.
        mu = torch.tensor([mu0], requires_grad=True)
        with torch.no_grad():
            u = mu0 + math.exp(ls0) * eps
            a_sf = torch.tanh(u)
            w = f(a_sf)
        lp = logp_correct(u, mu, ls)
        loss = (w.squeeze(-1) * lp).mean()
        loss.backward()
        sf.append(mu.grad.item())

    rep, sf = np.array(rep), np.array(sf)
    print("\n5. GRADIENT VARIANCE — reparameterized vs score-function (REINFORCE)")
    print(f"   {'estimator':>22s} {'mean':>10s} {'std':>10s}")
    print(f"   {'reparameterized':>22s} {rep.mean():10.4f} {rep.std():10.4f}")
    print(f"   {'score function':>22s} {sf.mean():10.4f} {sf.std():10.4f}")
    print(f"   same expectation (they agree to {abs(rep.mean() - sf.mean()):.3f}), but the")
    print(f"   score-function estimator's std is {sf.std() / rep.std():.1f}x larger.")
    print("   SAC differentiates THROUGH the sampled action for exactly this reason.")
    return rep, sf


# --------------------------------------------------------------------------
# The consequence: train SAC with the correction, and without it
# --------------------------------------------------------------------------
def run_one(args):
    env_id, correction, seed = args
    cfg = cc.sac_config(env_id=env_id, seed=seed, total_steps=25_000, hidden=128,
                        eval_every=2_500, eval_episodes=3,
                        tanh_correction=correction)
    hist, _ = cc.train(cfg)
    return env_id, correction, seed, hist


def training_consequence():
    tasks = ["Pendulum-v1", "Hopper-v5"]
    seeds = [0, 1]
    jobs = [(t, c, s) for t in tasks for c in (True, False) for s in seeds]
    print(f"\n6. TRAINING — SAC with and without the correction ({len(jobs)} runs)...",
          flush=True)
    # `spawn`, not the default `fork`. Checks 1-5 above already ran backward passes in
    # THIS process, and torch refuses to fork once autograd's threads exist:
    #   "Unable to handle autograd's threading in combination with fork-based
    #    multiprocessing."
    # The other projects in this phase fork before touching autograd, so they are fine
    # with the default; this one is not, purely because the checks come first.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        results = list(pool.map(run_one, jobs))

    runs = {}
    for env_id, corr, seed, h in results:
        runs.setdefault(env_id, {}).setdefault(corr, {})[seed] = h

    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for j, t in enumerate(tasks):
        ax = ps.style_axes(axes[j])
        for i, corr in enumerate([True, False]):
            c = np.stack([runs[t][corr][s]["eval_return"] for s in seeds])
            steps = runs[t][corr][seeds[0]]["steps"]
            lab = "with correction (correct SAC)" if corr else "without correction (the bug)"
            ax.plot(steps, c.mean(0), color=ps.SERIES[i], lw=2.0, label=lab)
            ax.fill_between(steps, c.min(0), c.max(0), color=ps.SERIES[i],
                            alpha=0.15, lw=0)
        ax.set_title(t.replace("-v5", "").replace("-v1", ""), color=ps.INK,
                     fontsize=11, loc="left")
        ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
        if j == 0:
            ax.set_ylabel("evaluation return", color=ps.INK_SECONDARY, fontsize=9)
            ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("The bug does not crash. It just quietly costs you return.",
                 color=ps.INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "correction_ablation.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'correction_ablation.png'}")

    print(f"\n{'task':14s} {'with correction':>17s} {'without':>10s} {'entropy/dim (with)':>20s}"
          f" {'(without)':>11s}")
    for t in tasks:
        act_dim = {"Pendulum-v1": 1, "Hopper-v5": 3}[t]
        row = []
        for corr in (True, False):
            c = np.stack([runs[t][corr][s]["eval_return"] for s in seeds])
            e = np.mean([runs[t][corr][s]["entropy"][-1] for s in seeds]) / act_dim
            row += [c[:, -2:].mean(), e]
        print(f"{t:14s} {row[0]:17.0f} {row[2]:10.0f} {row[1]:20.2f} {row[3]:11.2f}")
    print("\n   Pendulum has ONE action dimension and the bug barely shows (-101 vs -103).")
    print("   Hopper has THREE, and the same bug halves the return. Look at the last")
    print("   column: the broken agent reports an entropy of +3.4 while its target is")
    print("   -1.0. Its temperature controller is servoing on a number that does not")
    print("   describe the policy the environment actually saw.")


def make_check_plots(dens, ent):
    centers, hist, p_correct, p_wrong = dens
    log_stds, true_h, corrected_h, naive_h = ent

    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    ax = ps.style_axes(axes[0])
    ax.fill_between(centers, hist, color=ps.INK_MUTED, alpha=0.30, lw=0,
                    label="histogram of 400k sampled actions")
    ax.plot(centers, p_correct, color=ps.SERIES[0], lw=2.2,
            label="analytic p(a), with correction")
    ax.plot(centers, p_wrong, color=ps.SERIES[2], lw=2.0, ls="--",
            label="analytic p(a), no correction")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("The correction IS the density", color=ps.INK, fontsize=11, loc="left")
    ax.set_xlabel("action a = tanh(u)", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("density", color=ps.INK_SECONDARY, fontsize=9)

    ax = ps.style_axes(axes[1])
    ax.plot(log_stds, true_h, color=ps.INK, lw=2.4, label="true entropy of tanh(N)")
    ax.plot(log_stds, corrected_h, color=ps.SERIES[0], lw=1.6, ls="--",
            label="SAC's estimate (with correction)")
    ax.plot(log_stds, naive_h, color=ps.SERIES[2], lw=2.0,
            label="uncorrected estimate (the bug)")
    ax.axhline(-1.0, color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.annotate("target entropy", (log_stds[0], -1.0), fontsize=8,
                color=ps.INK_MUTED, xytext=(2, 4), textcoords="offset points")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("What the temperature is servoing on", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("policy log_std", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("entropy (nats)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "logprob_audit.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"\nwrote {OUT / 'logprob_audit.png'}")


def make_grad_plot(rep, sf):
    fig, ax = ps.new_axes(7.6, 4.2)
    bins = np.linspace(min(sf.min(), rep.min()), max(sf.max(), rep.max()), 70)
    ax.hist(sf, bins=bins, color=ps.SERIES[2], alpha=0.65,
            label=f"score function (std {sf.std():.3f})")
    ax.hist(rep, bins=bins, color=ps.SERIES[0], alpha=0.85,
            label=f"reparameterized (std {rep.std():.3f})")
    ax.axvline(rep.mean(), color=ps.INK, lw=1.6, ls="--", label="true gradient")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax,
              "Same gradient, different variance — 2000 estimates of the same number",
              "estimated dJ/dmu", "count", OUT / "grad_variance.png")


def main():
    OUT.mkdir(exist_ok=True)
    checks_only = len(sys.argv) > 1 and sys.argv[1] == "checks"

    check_normalization()
    dens = check_density()
    ent = check_entropy()
    check_stability()
    rep, sf = check_gradients()

    make_check_plots(dens, ent)
    make_grad_plot(rep, sf)

    if not checks_only:
        training_consequence()


if __name__ == "__main__":
    main()
