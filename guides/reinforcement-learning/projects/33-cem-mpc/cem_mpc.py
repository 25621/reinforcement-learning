"""Project 33 — CEM-MPC: the same model, a smarter search.

Project 32 changed the *model* and kept the search dumb. This project changes nothing
about the model and only changes the search, so any difference you see is attributable
to the search alone. That is the whole design: `mbrl_lib` is imported unchanged from
project 32, and the only edit is which `Planner` subclass gets passed to `run_mpc`.

To make the comparison honest, both planners get the **same budget**: 400 action
sequences rolled through the model per decision. Random shooting spends all 400 at
once, blindly. CEM spends them in 4 rounds of 100, each round aimed at where the last
round's winners landed. Same compute, same model — different answer.

Three experiments, and the first one comes out NULL — which is the most useful thing in
this project, so it is not being hidden:

  1. Learning curves at matched budget. On Pendulum, CEM and random shooting score the
     same. The better search does not produce a better agent.
  2. So did CEM do nothing? Isolate the search from the learning: freeze one trained
     model and ask both planners to find the best plan from the same states, at budgets
     from 50 to 3200 rollouts. Now the difference is enormous — and experiment 1's null
     result turns out to be a statement about Pendulum, not about CEM.
  3. Where the difference starts to matter for control. Sweep the planning horizon, which
     IS the dimension of the search space, and watch the gap open up.

The lesson is bigger than CEM: experiment 1 measures the wrong thing. It measures the
*return*, when the thing that changed is the *plan*. If you only ever run experiment 1,
you conclude "CEM is not worth it" and you are wrong.

  python3 cem_mpc.py     # ~5 min on 12 hyperthreads
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "32-pets-random-shooting-mpc"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import mbrl_lib as M  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"

SEEDS = [0, 1, 2]   # six cores; see project 32 on why more seeds would run slower
N_WARMUP, N_EPISODES = 1, 9

# THE MATCHED BUDGET. Both planners roll 400 action sequences through the model per
# decision. This is the number that makes the comparison a statement about search
# rather than a statement about compute.
BUDGET = 400

RS_PLAN = M.PlanConfig(horizon=15, n_candidates=BUDGET, n_iters=1)
CEM_PLAN = M.PlanConfig(
    horizon=15,
    n_candidates=BUDGET // 4,  # 100 per round...
    n_iters=4,                 # ...times 4 rounds = 400 rollouts. Same as above.
    n_elites=10,               # top 10% of each round survive to shape the next
    init_std=1.0,
    alpha=0.1,
)


def run(seed, planner_name, want_model=False):
    plan = RS_PLAN if planner_name == "random" else CEM_PLAN
    planner_cls = M.RandomShooting if planner_name == "random" else M.CEM
    cfg = M.MBRLConfig(
        seed=seed, n_members=5, n_warmup_episodes=N_WARMUP, n_episodes=N_EPISODES,
        plan=plan,
    )
    h = M.run_mpc(cfg, planner_cls)
    out = {
        "seed": seed,
        "planner": planner_name,
        "steps": h["steps"],
        "return": h["return"],
        "wall_total": h["wall_total"],
    }
    # Experiments 2 and 3 need a trained model to freeze. Reuse this one instead of
    # training another from scratch, which would cost a third of the whole budget.
    if want_model:
        out["state_dict"] = h["model"].state_dict()
        obs, _, _, _ = h["buffer"].arrays()
        out["obs"] = obs.copy()
    return out


def search_quality(model, states, budget, mode, horizon=15, seed=0):
    """Best predicted return each planner can find, given `budget` model rollouts.

    No environment here at all. The model is frozen and the question is purely: given a
    fixed function from action-sequences to predicted return, and a fixed number of
    times you may evaluate it, how good a sequence can you find? A planner is an
    optimizer, and this measures it as one.
    """
    gen = torch.Generator().manual_seed(seed + 31337)
    if mode == "random":
        cfg = M.PlanConfig(horizon=horizon, n_candidates=budget, n_iters=1)
        planner = M.RandomShooting(1, 2.0, cfg)
    else:
        iters = 4
        cfg = M.PlanConfig(
            horizon=horizon, n_candidates=max(budget // iters, 8), n_iters=iters,
            n_elites=max(budget // iters // 10, 2), init_std=1.0, alpha=0.1,
        )
        planner = M.CEM(1, 2.0, cfg)

    scores = []
    for s in states:
        planner.reset()  # no warm start: each state is judged cold, so the comparison
                         # measures the search, not the luck of the previous plan
        _, ret = planner.plan(model, torch.as_tensor(s, dtype=torch.float32), gen)
        scores.append(ret)
    return float(np.mean(scores))


def sq_job(args):
    model, states, budget, mode, horizon = args
    return budget, horizon, mode, search_quality(model, states, budget, mode, horizon)


def main():
    OUT.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    # ---- experiment 1: learning curves at matched budget ----
    print("=== random shooting vs CEM, 400 model rollouts per decision for both ===",
          flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run, s, p, p == "cem" and s == 0)
                for p in ("random", "cem") for s in SEEDS]
        res = [f.result() for f in futs]

    by = {p: [r for r in res if r["planner"] == p] for p in ("random", "cem")}
    fig, ax = ps.new_axes(7.4, 4.3)
    for i, (p, label) in enumerate(
        [("random", "random shooting (400 x 1)"), ("cem", "CEM (100 x 4 rounds)")]
    ):
        r = np.array([x["return"] for x in by[p]])
        st = np.array(by[p][0]["steps"])
        color = ps.SERIES[2] if p == "random" else ps.SERIES[0]
        ax.plot(st, r.mean(0), color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(st, r.min(0), r.max(0), color=color, alpha=0.15)
        wall = np.mean([x["wall_total"] for x in by[p]])
        print(f"  {label:28s} final {r[:, -1].mean():8.1f}  "
              f"(seeds: {', '.join(f'{v:.0f}' for v in r[:, -1])})  {wall:.0f}s")
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "Same model, same 400 rollouts — and the same score",
              "environment steps", "episode return", OUT / "learning_curves.png")

    # ---- the frozen model for the two pure-search experiments ----
    print("\n=== freezing the seed-0 CEM model for the search benchmarks ===", flush=True)
    src = next(r for r in res if "state_dict" in r)
    model = M.GaussianEnsemble(5, 3, 1, hidden=200, n_layers=3)
    model.load_state_dict(src["state_dict"])
    obs = src["obs"]
    rng = np.random.default_rng(0)
    states = obs[rng.choice(len(obs), size=24, replace=False)]

    # ---- experiment 2: search quality vs budget ----
    print("\n=== plan quality vs rollout budget (frozen model, no env) ===", flush=True)
    budgets = [50, 100, 200, 400, 800, 1600, 3200]
    with ProcessPoolExecutor(max_workers=10) as ex:
        jobs = [(model, states, b, m, 15) for m in ("random", "cem") for b in budgets]
        out = list(ex.map(sq_job, jobs))

    curve = {m: [next(o[3] for o in out if o[0] == b and o[2] == m) for b in budgets]
             for m in ("random", "cem")}
    for b, r, c in zip(budgets, curve["random"], curve["cem"]):
        print(f"  budget {b:5d}:  random {r:8.1f}   CEM {c:8.1f}   gap {c - r:6.1f}")

    # The number that makes the point: how many rollouts must random shooting spend to
    # match what CEM achieves with its smallest budget — and does it EVER match it?
    cem_cheapest = curve["cem"][0]
    rand_best = max(curve["random"])
    print(f"\n  CEM with {budgets[0]} rollouts scores {cem_cheapest:.1f}")
    print(f"  random shooting, given {budgets[-1]} rollouts ({budgets[-1] // budgets[0]}x more), "
          f"scores {curve['random'][-1]:.1f}")
    if rand_best < cem_cheapest:
        print("  -> random shooting never catches CEM's CHEAPEST setting, at any budget "
              "tested.")

    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(budgets, curve["random"], color=ps.SERIES[2], lw=2, marker="o", ms=5,
            label="random shooting")
    ax.plot(budgets, curve["cem"], color=ps.SERIES[0], lw=2, marker="o", ms=5, label="CEM")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "CEM finds a better plan with a fraction of the rollouts",
              "model rollouts allowed per decision (log scale)",
              "predicted return of the best plan found", OUT / "search_quality.png")

    # ---- experiment 3: the gap grows with the dimension of the search ----
    print("\n=== plan quality vs planning horizon (= dimension of the search space) ===",
          flush=True)
    horizons = [5, 10, 20, 30, 45, 60]
    with ProcessPoolExecutor(max_workers=10) as ex:
        jobs = [(model, states[:12], BUDGET, m, h) for m in ("random", "cem")
                for h in horizons]
        out = list(ex.map(sq_job, jobs))
    hcurve = {m: [next(o[3] for o in out if o[1] == h and o[2] == m) for h in horizons]
              for m in ("random", "cem")}
    for h, r, c in zip(horizons, hcurve["random"], hcurve["cem"]):
        print(f"  horizon {h:3d} ({h:3d}-dim search):  random {r:8.1f}   CEM {c:8.1f}"
              f"   gap {c - r:7.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
    axes[0].plot(horizons, hcurve["random"], color=ps.SERIES[2], lw=2, marker="o", ms=5,
                 label="random shooting")
    axes[0].plot(horizons, hcurve["cem"], color=ps.SERIES[0], lw=2, marker="o", ms=5,
                 label="CEM")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Longer plans are harder to guess", color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("planning horizon (dimensions to search)", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("predicted return of best plan", color=ps.INK_SECONDARY, fontsize=10)

    gap = np.array(hcurve["cem"]) - np.array(hcurve["random"])
    axes[1].plot(horizons, gap, color=ps.SERIES[4], lw=2, marker="o", ms=5)
    axes[1].axhline(0, color=ps.BASELINE, lw=1)
    axes[1].set_title("...so CEM's advantage widens", color=ps.INK, fontsize=11, loc="left")
    axes[1].set_xlabel("planning horizon", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("CEM's advantage over random", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "horizon_scaling.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'horizon_scaling.png'}")


if __name__ == "__main__":
    main()
