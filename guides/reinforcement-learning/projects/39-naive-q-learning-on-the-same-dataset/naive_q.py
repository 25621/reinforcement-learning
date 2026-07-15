"""Project 39 — Run ordinary Q-learning on a fixed dataset, and watch it fail.

The algorithm here is not a weak one. It is the agent from project 27 — the one
that learned to run on this exact robot — with ONE change: it may not collect data.

Two versions, so we can separate two questions that are easy to confuse.

    single critic   plain Q-learning. No protection of any kind.
    twin critics    TD3's fix: keep two Q-networks and always trust the more
                    PESSIMISTIC one. Online, this is THE cure for over-optimistic
                    values. Does it cure the offline disease too?

We log two numbers every 100 gradient steps:

    Q(s, a_data)  the value predicted for actions that ARE in the dataset
    Q(s, a_pi)    the value predicted for the action the policy has taught itself
                  to prefer — which is generally NOT in the dataset

and compare both against a hard ceiling: no state-action in this environment can be
worth more than `max_reward / (1 - gamma)`. That is arithmetic, not opinion. Any Q
above that line is not an overestimate — it is a number describing nothing.

    python3 naive_q.py     # ~6 min: 6 runs in parallel + a BC reference
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
GAMMA = 0.99

# The ground truth this whole project is built on. It lives in offline_lib because
# projects 40 and 41 hold their critics against the same number.
true_value_of_the_data = ol.true_value_of_the_data


def run_one(args):
    algo, level, twin = args
    cfg = ol.OfflineConfig(algo=algo, level=level, seed=0, grad_steps=GRAD_STEPS,
                           twin_critics=twin, eval_every=2_500, eval_episodes=10)
    hist, _, _ = ol.train(cfg)
    return algo, level, twin, hist


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    # The ceiling. HalfCheetah never pays more than ~7.6 in a single step, so with
    # gamma = 0.99 the most any state-action can be worth is 7.6 / (1 - 0.99) = 756.
    # This is not a tuning choice or an empirical observation — it is arithmetic, and
    # no correct Q-value can exceed it. Projects 40 and 41 quote the same number.
    r_max = max(float(ol.build_dataset(lv)["rew"].max()) for lv in ol.LEVELS)
    q_ceiling = ol.q_ceiling(GAMMA)
    print(f"largest single-step reward anywhere in the data: {r_max:.2f}")
    print(f"=> no state-action can be worth more than {q_ceiling:.0f}")

    # And we can do better than a loose ceiling: for the dataset's OWN actions we can
    # compute the true value exactly, by adding up what actually happened next.
    true_q = {lv: true_value_of_the_data(lv) for lv in ol.LEVELS}
    print("true value of the data's actions (computed from the recording, not predicted):")
    for lv in ol.LEVELS:
        print(f"    {lv:7s} {true_q[lv]:8.1f}")
    print(flush=True)

    jobs = ([("naive_q", lv, False) for lv in ol.LEVELS]
            + [("naive_q", lv, True) for lv in ol.LEVELS]
            + [("bc", "medium", True)])
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(run_one, jobs))
    runs = {(a, lv, tw): h for a, lv, tw, h in results}
    print(f"\n[{time.time() - t0:.0f}s] training done\n", flush=True)

    # ---- the table ----
    lo, hi = ol.score_bounds()
    print("=" * 100)
    print(f"{'dataset':9s} {'protection':22s} {'Q(data)':>10s} {'true value':>11s} "
          f"{'too high by':>13s} {'return':>9s} {'score':>7s}")
    print("-" * 100)
    for tw in (False, True):
        for lv in ol.LEVELS:
            h = runs[("naive_q", lv, tw)]
            name = "twin critics (TD3)" if tw else "none (single critic)"
            gap = h["q_data"][-1] - true_q[lv]
            # A RATIO only makes sense when the true value is positive. On `random` it
            # is negative (-25), and "predicted / true" would print a meaningless -12x.
            ratio = f" ({h['q_data'][-1] / true_q[lv]:.0f}x)" if true_q[lv] > 0 else ""
            print(f"{lv:9s} {name:22s} {h['q_data'][-1]:10.1f} {true_q[lv]:11.1f} "
                  f"{gap:8.0f}{ratio:>5s} {h['eval_return'][-1]:9.1f} {h['score'][-1]:7.1f}")
        print()
    hb = runs[("bc", "medium", True)]
    print(f"{'medium':9s} {'BC (project 38)':22s} {'—':>10s} {'—':>11s} {'—':>13s} "
          f"{hb['eval_return'][-1]:9.1f} {hb['score'][-1]:7.1f}")
    print("-" * 100)
    print(f"{'':9s} {'physical ceiling on any Q:':26s} {q_ceiling:6.1f}")
    print(f"{'':9s} {'random / expert teacher:':26s} {lo:6.1f} / {hi:.1f}")
    print("=" * 100)

    qm = runs[("naive_q", "medium", False)]["q_data"][-1]
    print("\n1. The critic is wrong about actions it HAS seen. On `medium` with a single")
    print(f"   critic it values the data's own actions at {qm:.0f}, when the recording")
    print(f"   says {true_q['medium']:.0f} — too high by a factor of {qm / true_q['medium']:.0f}.")
    print("   The rot is not confined to unseen actions. Bootstrapping spreads it.")
    print("\n2. Read the `return` column. EVERY run is worse than the uniform-random")
    print(f"   teacher ({lo:.0f}) — the policy that just shakes the joystick at random.")
    print("   Twin critics shrink the numbers. They do not make the policy work.")
    print(f"[{time.time() - t0:.0f}s total]")

    # ---- plots ----
    fig, axes = ps.plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    # (1) the values climb past what can physically exist — worst with no protection
    ax = ps.style_axes(axes[0])
    for i, lv in enumerate(ol.LEVELS):
        ax.plot(runs[("naive_q", lv, False)]["q_steps"],
                np.abs(runs[("naive_q", lv, False)]["q_pi_all"]),
                color=ps.SERIES[i], lw=2.0, label=f"{lv} — single critic")
        ax.plot(runs[("naive_q", lv, True)]["q_steps"],
                np.abs(runs[("naive_q", lv, True)]["q_pi_all"]),
                color=ps.SERIES[i], lw=1.5, ls=":", label=f"{lv} — twin critics")
    ax.axhline(q_ceiling, color=ps.INK, ls="--", lw=1.8)
    ax.text(GRAD_STEPS * 0.03, q_ceiling * 1.35,
            f"nothing can be worth more than {q_ceiling:.0f}", color=ps.INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Values climb out of the realm of the possible", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("|Q(s, a_policy)|  (log scale)", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (2) against GROUND TRUTH: the rot reaches the actions it has actually seen
    ax = ps.style_axes(axes[1])
    h = runs[("naive_q", "medium", False)]
    ax.plot(h["q_steps"], h["q_data_all"], color=ps.SERIES[0], lw=2.2,
            label="what the critic PREDICTS for the data's actions")
    ax.plot(h["q_steps"], h["q_pi_all"], color=ps.SERIES[2], lw=1.6, ls="--",
            label="what it predicts for the action it invents")
    ax.axhline(true_q["medium"], color=ps.SERIES[1], lw=2.2)
    ax.text(GRAD_STEPS * 0.30, true_q["medium"] + 60,
            f"the TRUE value of those same actions ({true_q['medium']:.0f}),\n"
            f"read straight off the recording",
            color=ps.SERIES[1], fontsize=8.5)
    ax.axhline(q_ceiling, color=ps.INK, ls="--", lw=1.5, label="physical ceiling")
    ax.fill_between(h["q_steps"], true_q["medium"], h["q_data_all"],
                    color=ps.SERIES[2], alpha=0.13)
    ax.set_title("It is wrong about the data it HAS seen", color=ps.INK,
                 fontsize=12, loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Q  (medium, single critic)", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (3) the only column that matters
    ax = ps.style_axes(axes[2])
    x = np.arange(len(ol.LEVELS))
    w = 0.35
    ax.bar(x - w / 2, [runs[("naive_q", lv, False)]["eval_return"][-1] for lv in ol.LEVELS],
           width=w, color=ps.SERIES[2], label="naive Q, single critic")
    ax.bar(x + w / 2, [runs[("naive_q", lv, True)]["eval_return"][-1] for lv in ol.LEVELS],
           width=w, color=ps.SERIES[3], label="naive Q, twin critics (TD3's fix)")
    ax.axhline(hb["eval_return"][-1], color=ps.SERIES[1], ls=":", lw=2)
    ax.text(-0.45, hb["eval_return"][-1] + 60, f"BC just copies the data: {hb['eval_return'][-1]:.0f}",
            color=ps.SERIES[1], fontsize=8.5, va="bottom")
    ax.axhline(lo, color=ps.INK, ls="--", lw=1.8)
    ax.text(-0.45, lo + 60, f"shaking the joystick at random: {lo:.0f}", color=ps.INK,
            fontsize=8.5, va="bottom")
    ax.axhline(0, color=ps.BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ol.LEVELS)
    ax.set_ylim(None, hb["eval_return"][-1] * 1.35)
    ax.set_title("Every bar is below random. Every one.", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("dataset it was trained on", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return in the real environment", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "naive_q_blowup.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'naive_q_blowup.png'}")


if __name__ == "__main__":
    main()
