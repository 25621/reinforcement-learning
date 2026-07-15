"""Project 43 — The same algorithms, three datasets. Who collected the data decides.

Phase 7 has so far run everything on `medium`. That hid the most practical fact in
offline RL: an algorithm's reputation depends almost entirely on the dataset it was
benchmarked on.

Three methods x three datasets, one fixed budget, nothing else changed:

    BC    copy the data                          (project 38)
    CQL   pessimistic values                     (project 40)
    IQL   never query an unseen action           (project 41)

    random   uniform joystick-shaking. Wide coverage, no skill.
    medium   a half-trained SAC. Some skill.
    expert   a fully-trained SAC. Skill.

And the question that actually matters, asked precisely: can any of them beat the
BEST SINGLE EPISODE in the data they were given? BC cannot — copying cannot exceed
the thing being copied. If an offline-RL method can, then it did something a copier
cannot do: it took good pieces from different mediocre episodes and joined them into
a run that nobody ever performed. That is called STITCHING, and it is the entire
argument for offline RL over imitation.

    python3 quality_study.py     # ~9 min: 9 runs in parallel
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
ALGOS = ["bc", "cql", "iql"]
NICE = {"bc": "BC", "cql": "CQL", "iql": "IQL"}


def run_one(job):
    algo, level = job
    cfg = ol.OfflineConfig(algo=algo, level=level, seed=0, grad_steps=GRAD_STEPS,
                           cql_alpha=5.0, expectile=0.7, awr_beta=3.0,
                           eval_every=2_500, eval_episodes=10)
    hist, _, _ = ol.train(cfg)
    return algo, level, hist


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    lo, hi = ol.score_bounds()

    # What is actually IN each dataset — the yardsticks every method is measured against.
    stats = {}
    for lv in ol.LEVELS:
        rets = ol.Dataset(lv).episode_returns()
        stats[lv] = {"mean": rets.mean(), "best": rets.max(), "n": len(rets)}
    print("the three datasets:")
    for lv in ol.LEVELS:
        s = stats[lv]
        print(f"  {lv:7s} {s['n']:3d} episodes   average {s['mean']:8.1f}   "
              f"best single episode {s['best']:8.1f}")
    print(f"\nteachers: random {lo:.0f}, expert {hi:.0f}\n", flush=True)

    jobs = [(a, lv) for lv in ol.LEVELS for a in ALGOS]
    with ProcessPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(run_one, jobs))
    R = {(a, lv): h for a, lv, h in results}
    print(f"[{time.time() - t0:.0f}s] training done\n", flush=True)

    # ---- the table ----
    print("=" * 96)
    print(f"{'':10s} " + " ".join(f"{NICE[a]:>18s}" for a in ALGOS) +
          f" {'data avg':>12s} {'data best':>12s}")
    print("-" * 96)
    for lv in ol.LEVELS:
        row = f"{lv:10s} "
        for a in ALGOS:
            h = R[(a, lv)]
            row += f" {h['eval_return'][-1]:9.1f} ({h['score'][-1]:5.1f})"
        row += f" {stats[lv]['mean']:12.1f} {stats[lv]['best']:12.1f}"
        print(row)
    print("=" * 96)
    print("(return, with the normalized score in brackets: 0 = random teacher, "
          "100 = expert teacher)\n")

    # ---- the stitching test ----
    print("Did anyone beat the best single episode in its own dataset?")
    print("-" * 68)
    for lv in ol.LEVELS:
        for a in ALGOS:
            r = R[(a, lv)]["eval_return"][-1]
            best = stats[lv]["best"]
            gap = r - best
            verdict = f"YES  (+{gap:.0f}, {r / best:.2f}x)" if gap > 0 else f"no   ({gap:+.0f})"
            print(f"  {NICE[a]:4s} on {lv:7s}:  got {r:8.1f}  vs best-in-data "
                  f"{best:8.1f}   {verdict}")
    print(f"\n[{time.time() - t0:.0f}s total]")

    # ---- plots ----
    fig, axes = ps.plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    # (1) the classic: return vs data quality
    ax = ps.style_axes(axes[0])
    x = np.arange(len(ol.LEVELS))
    for i, a in enumerate(ALGOS):
        ax.plot(x, [R[(a, lv)]["eval_return"][-1] for lv in ol.LEVELS], marker="o", ms=8,
                lw=2.4, color=ps.SERIES[i], label=NICE[a])
    ax.plot(x, [stats[lv]["mean"] for lv in ol.LEVELS], marker="_", ms=14, lw=1.6, ls="--",
            color=ps.INK_MUTED, label="the data's average episode")
    ax.set_xticks(x)
    ax.set_xticklabels(ol.LEVELS)
    ax.set_title("Return vs. who collected the data", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("dataset", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return in the real environment", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)

    # (2) grouped bars against the best episode in the data — the stitching test
    ax = ps.style_axes(axes[1])
    w = 0.26
    for i, a in enumerate(ALGOS):
        ax.bar(x + (i - 1) * w, [R[(a, lv)]["eval_return"][-1] for lv in ol.LEVELS],
               width=w, color=ps.SERIES[i], label=NICE[a])
    for j, lv in enumerate(ol.LEVELS):
        ax.plot([j - 1.6 * w, j + 1.6 * w], [stats[lv]["best"]] * 2, color=ps.INK, lw=2)
    ax.plot([], [], color=ps.INK, lw=2, label="best single episode in that dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(ol.LEVELS)
    ax.set_title("Anything above the black line was never demonstrated",
                 color=ps.INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("dataset", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (3) how much offline RL buys you OVER just copying — the decision a practitioner
    #     actually faces: "is the extra complexity worth it for MY data?"
    ax = ps.style_axes(axes[2])
    for i, a in enumerate([x for x in ALGOS if x != "bc"]):
        delta = [R[(a, lv)]["eval_return"][-1] - R[("bc", lv)]["eval_return"][-1]
                 for lv in ol.LEVELS]
        ax.plot(x, delta, marker="o", ms=8, lw=2.4, color=ps.SERIES[ALGOS.index(a)],
                label=f"{NICE[a]} − BC")
    ax.axhline(0, color=ps.INK, lw=1.6)
    ax.text(0.02, 0.94, "above 0: offline RL is worth the trouble", transform=ax.transAxes,
            fontsize=8.5, color=ps.INK_SECONDARY, va="top")
    ax.text(0.02, 0.06, "below 0: just clone the data", transform=ax.transAxes,
            fontsize=8.5, color=ps.INK_SECONDARY, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(ol.LEVELS)
    ax.set_title("When is offline RL worth it?", color=ps.INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("dataset", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return, minus BC's return", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "dataset_quality.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'dataset_quality.png'}")


if __name__ == "__main__":
    main()
