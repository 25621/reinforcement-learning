r"""Project 40 — CQL: keep the dangerous max, but make the critic pay for optimism.

Project 39 established the disease. This is the first cure.

CQL adds ONE term to the ordinary Q-learning loss:

    loss = TD_error  +  alpha * [ logsumexp_a Q(s, a)  -  Q(s, a_data) ]
                          \_______________________/     \____________/
                            push DOWN every action        pull UP the one
                            the critic might dream of     that really happened

The whole project is a sweep over `alpha`, the size of that penalty, because `alpha`
is the knob that turns naive Q-learning into CQL continuously:

    alpha = 0     is EXACTLY project 39. Same code path, penalty multiplied by zero.
    alpha -> inf  the TD loss becomes irrelevant and only "stay on the data" remains,
                  which is behavior cloning wearing a critic as a hat.

So the sweep runs between two algorithms we have already measured, and the question
is what happens in between. Two of the three things it found were NOT what this
comment predicted before the run — see the README. The prediction that a huge alpha
would leave the critic "unable to tell a good action from a bad one" was wrong, and
the reason it was wrong is the most interesting thing in the project.

    python3 cql.py     # ~8 min: 5 values of alpha in parallel
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "38-bc-baseline-on-d4rl"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import offline_lib as ol  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
GRAD_STEPS = 20_000
ALPHAS = [0.0, 0.5, 5.0, 20.0, 100.0]
LEVEL = "medium"
BC_RETURN = 1384.7   # project 38, mean of 3 seeds — the bar CQL has to clear


def run_one(alpha):
    # alpha = 0 dispatches to `naive_q`, which is not a special case bolted on for
    # this sweep — it is literally the same function CQL calls, with the penalty
    # multiplied by zero. The continuity is the argument.
    cfg = ol.OfflineConfig(algo="cql" if alpha > 0 else "naive_q", level=LEVEL, seed=0,
                           grad_steps=GRAD_STEPS, cql_alpha=alpha,
                           eval_every=2_500, eval_episodes=10)
    hist, _, _ = ol.train(cfg)
    return alpha, hist


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    lo, hi = ol.score_bounds()
    q_ceiling = ol.q_ceiling()   # 756: the same number project 39 quotes

    print(f"CQL on the `{LEVEL}` dataset, sweeping the conservatism weight alpha")
    print(f"(a correct Q can never exceed {q_ceiling:.0f} — see project 39)\n", flush=True)

    with ProcessPoolExecutor(max_workers=len(ALPHAS)) as pool:
        results = dict(pool.map(run_one, ALPHAS))
    print(f"[{time.time() - t0:.0f}s] training done\n", flush=True)

    true_q = ol.true_value_of_the_data(LEVEL)
    print("=" * 92)
    print(f"{'alpha':>8s} {'':11s} {'final Q(data)':>14s} {'the TRUE value':>15s} "
          f"{'return':>10s} {'score':>8s}")
    print("-" * 92)
    for a in ALPHAS:
        h = results[a]
        tag = "(= naive Q)" if a == 0 else ""
        print(f"{a:8.1f} {tag:11s} {h['q_data'][-1]:14.1f} {true_q:15.1f} "
              f"{h['eval_return'][-1]:10.1f} {h['score'][-1]:8.1f}")
    print("-" * 92)
    print(f"{'':8s} {'BC (project 38):':22s} {'':>14s} {BC_RETURN:20.1f} {29.4:8.1f}")
    print(f"{'':8s} {'physical ceiling on Q:':22s} {q_ceiling:12.1f}")
    print(f"{'':8s} {'random / expert:':22s} {lo:12.1f} / {hi:.1f}")
    print("=" * 92)
    best = max(ALPHAS, key=lambda a: results[a]["eval_return"][-1])
    print(f"\n1. There is a THRESHOLD, not a sweet spot.")
    print(f"   alpha 0 and 0.5 are catastrophic ({results[0.0]['eval_return'][-1]:.0f}, "
          f"{results[0.5]['eval_return'][-1]:.0f}). Every alpha >= 5 works.")
    print(f"   best = {best} -> {results[best]['eval_return'][-1]:.0f} "
          f"(score {results[best]['score'][-1]:.1f}), which BEATS BC's {BC_RETURN:.0f}.")
    above = [results[a]["eval_return"][-1] for a in ALPHAS if a >= 5]
    print(f"\n2. Above the threshold, the exact value barely matters. A 20x range of")
    print(f"   alpha gives {min(above):.0f} to {max(above):.0f} — all of it above BC, and the")
    print(f"   spread is about the size of the seed noise project 38 measured (+/-95).")
    print(f"   There is NO second cliff: over-tightening does not break the policy.")
    print(f"\n3. Look at Q(data) against the true value ({true_q:.0f}). alpha=100 predicts")
    print(f"   {results[100.0]['q_data'][-1]:.0f} — wronger, in absolute terms, than naive Q's "
          f"{results[0.0]['q_data'][-1]:.0f} —")
    print(f"   and yet its policy is FINE. The values do not have to be right.")
    print(f"   They only have to be RANKED right.")
    print(f"[{time.time() - t0:.0f}s total]")

    # ---- plots ----
    fig, axes = ps.plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    # (1) The surprise: alpha does NOT control how big the values are. alpha=100 has
    #     the LARGEST values of all, above the physical ceiling — and a fine policy.
    ax = ps.style_axes(axes[0])
    for i, a in enumerate(ALPHAS):
        lbl = f"alpha = {a:g}" + ("  (= naive Q)" if a == 0 else "")
        ax.plot(results[a]["q_steps"], results[a]["q_data_all"],
                color=ps.SERIES[i], lw=1.8, label=lbl)
    ax.axhline(q_ceiling, color=ps.INK, ls="--", lw=1.6)
    ax.text(GRAD_STEPS * 0.02, q_ceiling + 90, f"physical ceiling ({q_ceiling:.0f})",
            color=ps.INK, fontsize=8.5)
    ax.axhline(true_q, color=ps.SERIES[1], lw=2.2)
    ax.text(GRAD_STEPS * 0.02, true_q - 260,
            f"the TRUE value of these actions ({true_q:.0f})", color=ps.SERIES[1], fontsize=8.5)
    ax.set_title("Alpha does NOT control the size of the values", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Q(s, a) for the data's own actions", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (2) ...and yet the returns are decided almost immediately.
    ax = ps.style_axes(axes[1])
    for i, a in enumerate(ALPHAS):
        ax.plot(results[a]["steps"], results[a]["eval_return"], marker="o", ms=3, lw=2,
                color=ps.SERIES[i], label=f"alpha = {a:g}")
    ax.axhline(BC_RETURN, color=ps.INK, ls="--", lw=1.6)
    ax.text(GRAD_STEPS * 0.02, BC_RETURN + 70, f"BC ({BC_RETURN:.0f})", color=ps.INK, fontsize=8.5)
    ax.axhline(lo, color=ps.INK_MUTED, ls=":", lw=1.4)
    ax.text(GRAD_STEPS * 0.02, lo + 70, f"random ({lo:.0f})", color=ps.INK_MUTED, fontsize=8.5)
    ax.set_title("...but it decides everything about the policy", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return in the real environment", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    # (3) The money panel. NOT a hump: a cliff, then a plateau that sags back to BC.
    ax = ps.style_axes(axes[2])
    rets = [results[a]["eval_return"][-1] for a in ALPHAS]
    x = np.arange(len(ALPHAS))
    ax.plot(x, rets, marker="o", ms=8, lw=2.4, color=ps.SERIES[1])
    ax.axhline(BC_RETURN, color=ps.INK, ls="--", lw=1.6)
    ax.text(0.02, BC_RETURN + 70, f"just cloning the data (BC): {BC_RETURN:.0f}",
            color=ps.INK, fontsize=8.5)
    ax.axhline(lo, color=ps.INK_MUTED, ls=":", lw=1.4)
    ax.text(0.02, lo + 70, f"shaking the joystick: {lo:.0f}", color=ps.INK_MUTED, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
    ax.annotate("too little pessimism:\nthis IS project 39", xy=(0.06, rets[0]),
                xytext=(0.10, 0.28), textcoords=("data", "axes fraction"),
                fontsize=8, color=ps.INK_SECONDARY,
                arrowprops=dict(arrowstyle="->", color=ps.INK_MUTED, lw=1))
    ax.annotate("a 20x range of alpha,\nand every value works.\nNo second cliff.",
                xy=(3, rets[3]), xytext=(0.52, 0.42), textcoords="axes fraction",
                fontsize=8, color=ps.INK_SECONDARY,
                arrowprops=dict(arrowstyle="->", color=ps.INK_MUTED, lw=1))
    ax.set_title("A cliff, then a plateau", color=ps.INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("conservatism weight alpha", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("final return", color=ps.INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "cql_alpha_sweep.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'cql_alpha_sweep.png'}")



if __name__ == "__main__":
    main()
