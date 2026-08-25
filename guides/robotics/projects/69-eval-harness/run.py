"""A nightly eval harness, and the five questions it has to survive.

  1. the dashboard        -- 50 tasks x 4 systems, with error bars
  2. how many seeds?      -- the power to detect a regression you care about
  3. what the aggregate hides
  4. fixed seeds vs fresh seeds
  5. where to spend the compute

About 7 minutes on 12 cores.
"""

import csv
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

# "spawn", not the Linux default "fork": torch keeps worker threads alive and
# forking a process that holds one of their locks deadlocks the child silently
# and forever.  The cost is that each worker starts a clean interpreter, which
# is why the policies are trained ONCE here and shipped to the workers as
# picklable objects rather than rebuilt inside them.
MP = multiprocessing.get_context("spawn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import harness as H
from harness import build_suite, run_task, run_task_expert, two_proportion_z, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []

N_SEEDS = 20
SUITE = build_suite(50)
BASE_SEEDS = np.arange(10_000, 10_000 + N_SEEDS)


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


# ---------------------------------------------------------------------------
# the systems under test, built once per worker process
# ---------------------------------------------------------------------------
SYSTEMS = ["expert (ceiling)", "bc-300", "domain randomised", "bc-100",
           "bc-300 + regression"]


def build_systems():
    """Train every system once, in this process."""
    print("training the systems under test...", flush=True)
    bc300 = H.train(300, seed=0)
    return {"expert (ceiling)": None,
            "bc-300": bc300,
            "bc-100": H.train(100, seed=0),
            "bc-300 + regression": bc300.rescaled(0.80),
            "domain randomised": H.train(300, seed=0,
                                         randomize={"mass_scale": (0.8, 1.9),
                                                    "gear": (0.75, 1.15),
                                                    "damp_scale": (0.7, 2.2),
                                                    "latency": (0, 2)})}


POLICIES = {}


def _job(args):
    system, policy, task_id, seeds = args
    task = SUITE[task_id]
    if policy is None:
        r = run_task_expert(task, seeds)
    else:
        r = run_task(policy, task, seeds)
    return system, task_id, r


def run_matrix(systems, seeds, workers=10, per_task=False):
    """Return {system: (n_tasks, n_seeds) array of 0/1}.

    ``per_task=False`` gives every task the SAME list of episode seeds, which
    is what almost every harness does and what experiment 4 is about: the 1000
    episodes are then only 20 distinct object placements, seen 50 times each.
    ``per_task=True`` offsets the seeds per task so the placements differ.
    """
    jobs = [(s, POLICIES[s], t["id"],
             seeds + (t["id"] * 977 if per_task else 0))
            for s in systems for t in SUITE]
    out = {s: np.zeros((len(SUITE), len(seeds)), dtype=int) for s in systems}
    with ProcessPoolExecutor(max_workers=workers, mp_context=MP) as ex:
        for system, tid, r in ex.map(_job, jobs, chunksize=2):
            out[system][tid] = r
    return out


# ---------------------------------------------------------------------------
# 1. the dashboard
# ---------------------------------------------------------------------------
def exp1_dashboard(M):
    print("\n=== 1. the dashboard " + "=" * 51)
    print("  system                  pass rate   95 % interval   tasks at 0")
    for s in SYSTEMS:
        a = M[s]
        k, n = int(a.sum()), a.size
        lo, hi = wilson(k, n)
        zero = int((a.sum(1) == 0).sum())
        print("  %-22s %9.3f   [%.3f, %.3f] %9d"
              % (s, k / n, lo, hi, zero))
        record("dashboard", system=s, pass_rate=k / n, lo=lo, hi=hi,
               episodes=n, tasks_all_fail=zero)

    fams = sorted({t["family"] for t in SUITE})
    print("\n  per-family pass rate:")
    short = {"expert (ceiling)": "expert", "bc-300": "bc-300",
             "domain randomised": "DR", "bc-100": "bc-100",
             "bc-300 + regression": "regressed"}
    print("  family          " + " ".join("%-9s" % short[s] for s in SYSTEMS))
    for fam in fams:
        idx = [t["id"] for t in SUITE if t["family"] == fam]
        line = "  %-15s" % fam
        for s in SYSTEMS:
            line += " %-9.3f" % M[s][idx].mean()
            record("family", family=fam, system=s, pass_rate=M[s][idx].mean())
        print(line)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4),
                           gridspec_kw={"width_ratios": [3, 1]})
    grid = np.array([M[s].mean(1) for s in SYSTEMS])
    im = ax[0].imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax[0].set_yticks(range(len(SYSTEMS)))
    ax[0].set_yticklabels(SYSTEMS, fontsize=8)
    ax[0].set_xticks(range(0, len(SUITE), 2))
    ax[0].set_xticklabels([SUITE[i]["name"] for i in range(0, len(SUITE), 2)],
                          rotation=90, fontsize=5)
    ax[0].set_title("pass rate per task (%d seeds each)" % N_SEEDS)
    fig.colorbar(im, ax=ax[0], fraction=0.02)

    means = [M[s].mean() for s in SYSTEMS]
    errs = np.array([[M[s].mean() - wilson(int(M[s].sum()), M[s].size)[0],
                      wilson(int(M[s].sum()), M[s].size)[1] - M[s].mean()]
                     for s in SYSTEMS]).T
    ax[1].barh(range(len(SYSTEMS)), means, xerr=errs, color="#1976d2",
               error_kw=dict(ecolor="#263238", capsize=3))
    ax[1].set_yticks(range(len(SYSTEMS)))
    ax[1].set_yticklabels(["" for _ in SYSTEMS])
    ax[1].invert_yaxis(); ax[1].set_xlabel("overall pass rate")
    ax[1].set_title("with 95 % intervals"); ax[1].grid(alpha=.3, axis="x")
    # imshow already draws row 0 at the top and barh was inverted to match, so
    # the two panels line up row for row.  Inverting one of them silently
    # mislabels every row in the heatmap against the bar beside it.
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dashboard.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. how many seeds do you need?
# ---------------------------------------------------------------------------
def exp2_power(M):
    print("\n=== 2. how many episodes to catch a regression? " + "=" * 24)
    a, b = M["bc-300"], M["bc-300 + regression"]
    print("  true difference in this suite: %.3f -> %.3f  (%.3f)"
          % (a.mean(), b.mean(), a.mean() - b.mean()))
    print("  episodes/night   detected   median |z|   (1000 resamples)")
    rng = np.random.default_rng(0)
    ns, powers = [], []
    for n_ep in (20, 50, 100, 200, 400, 1000):
        hits, zs = 0, []
        for _ in range(1000):
            ia = rng.integers(0, a.size, n_ep)
            ib = rng.integers(0, b.size, n_ep)
            ka, kb = int(a.ravel()[ia].sum()), int(b.ravel()[ib].sum())
            z = two_proportion_z(ka, n_ep, kb, n_ep)
            zs.append(z)
            hits += z > 1.96
        print("  %13d %10.1f %% %11.2f" % (n_ep, 100 * hits / 1000,
                                           np.median(zs)))
        record("power", episodes=n_ep, detect_pct=100 * hits / 1000,
               median_z=float(np.median(zs)))
        ns.append(n_ep); powers.append(100 * hits / 1000)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(ns, powers, "o-", color="#1976d2")
    ax.axhline(80, ls="--", color="#2e7d32", label="80 % detection")
    ax.set_xscale("log"); ax.set_xlabel("episodes per night")
    ax.set_ylabel("% of nights the regression is caught")
    ax.set_title("A %.0f-point regression needs this much evidence"
                 % (100 * (a.mean() - b.mean())))
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "power.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. what the aggregate hides
# ---------------------------------------------------------------------------
def exp3_aggregate(M):
    print("\n=== 3. what a single number hides " + "=" * 38)
    pairs = [("bc-100", "bc-300 + regression"), ("bc-300", "domain randomised")]
    for s1, s2 in pairs:
        p1, p2 = M[s1].mean(1), M[s2].mean(1)
        agg = abs(p1.mean() - p2.mean())
        flips = int((np.abs(p1 - p2) > 0.25).sum())
        big = int(((p1 > 0.7) & (p2 < 0.3)).sum() + ((p2 > 0.7) & (p1 < 0.3)).sum())
        print("  %-22s vs %-22s aggregate gap %.3f" % (s1, s2, agg))
        print("     tasks differing by >0.25 : %2d of %d" % (flips, len(SUITE)))
        print("     tasks one passes and the other fails outright : %d" % big)
        record("aggregate", a=s1, b=s2, aggregate_gap=agg, tasks_differ=flips,
               tasks_opposite=big)

    # the regression, seen per family instead of overall
    print("\n  where the injected regression actually landed:")
    fams = sorted({t["family"] for t in SUITE})
    base, reg = M["bc-300"], M["bc-300 + regression"]
    worst = []
    for fam in fams:
        idx = [t["id"] for t in SUITE if t["family"] == fam]
        d = base[idx].mean() - reg[idx].mean()
        worst.append((d, fam, base[idx].mean(), reg[idx].mean()))
    for d, fam, x, y in sorted(worst, reverse=True):
        print("     %-15s %.3f -> %.3f   (drop %.3f)" % (fam, x, y, d))
        record("regression_by_family", family=fam, before=x, after=y, drop=d)
    print("     overall %.3f -> %.3f   (drop %.3f)"
          % (base.mean(), reg.mean(), base.mean() - reg.mean()))


# ---------------------------------------------------------------------------
# 4. fixed seeds vs fresh seeds
# ---------------------------------------------------------------------------
N_REPEATS = 5


def exp4_seeds(M):
    print("\n=== 4. the error bar is wrong, and here is by how much " + "=" * 17)
    print("  Run the WHOLE suite %d more times, each with a different set of"
          % N_REPEATS)
    print("  %d episode seeds, and watch where the suite mean lands." % N_SEEDS)

    out = {}
    for mode, per_task in (("same seeds for every task", False),
                           ("different seeds per task", True)):
        means = []
        for r in range(N_REPEATS):
            seeds = np.arange(200_000 + r * 5_000, 200_000 + r * 5_000 + N_SEEDS)
            Mr = run_matrix(["bc-300"], seeds, per_task=per_task)
            means.append(Mr["bc-300"].mean())
        means = np.array(means)
        k = int(M["bc-300"].sum())
        lo, hi = wilson(k, M["bc-300"].size)
        wilson_half = (hi - lo) / 2
        emp = float(means.std(ddof=1))
        print("\n  %s" % mode)
        print("     the %d suite means : %s"
              % (N_REPEATS, " ".join("%.3f" % v for v in means)))
        print("     spread of the mean across seed sets : %.4f" % emp)
        print("     what the Wilson interval promised    : %.4f (half-width)"
              % wilson_half)
        print("     the interval is too narrow by        : %.1fx"
              % (emp * 1.96 / wilson_half))
        out[mode] = means
        record("seed_repeats", mode=mode, empirical_sd=emp,
               wilson_half=wilson_half, understated_by=emp * 1.96 / wilson_half,
               means=";".join("%.4f" % v for v in means))

    a = out["same seeds for every task"]
    b = out["different seeds per task"]
    print("\n  decorrelating the seeds shrank the run-to-run spread %.2fx"
          % (a.std(ddof=1) / max(b.std(ddof=1), 1e-9)))
    record("seed_decorrelation", correlated_sd=float(a.std(ddof=1)),
           decorrelated_sd=float(b.std(ddof=1)),
           improvement=float(a.std(ddof=1) / max(b.std(ddof=1), 1e-9)))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (mode, means) in enumerate(out.items()):
        ax.scatter(means, [i] * len(means), s=60, zorder=3,
                   color="#c62828" if i == 0 else "#2e7d32", label=mode)
    k = int(M["bc-300"].sum())
    lo, hi = wilson(k, M["bc-300"].size)
    ax.axvspan(lo, hi, color="#90caf9", alpha=.5,
               label="the 95 % interval one night reports")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["correlated", "decorrelated"],
                                              fontsize=8)
    ax.set_xlabel("suite pass rate, %d independent nightly runs" % N_REPEATS)
    ax.set_title("Where the suite mean actually lands", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seed_repeats.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. where to spend the compute
# ---------------------------------------------------------------------------
def exp5_allocation(M):
    print("\n=== 5. per-task claims need far more seeds than you think " + "=" * 14)
    p = M["bc-300"].mean(1)
    var = p * (1 - p)      # a task's binomial variance: maximal at 0.5

    widths = np.array([wilson(int(M["bc-300"][i].sum()), N_SEEDS)[1]
                       - wilson(int(M["bc-300"][i].sum()), N_SEEDS)[0]
                       for i in range(len(SUITE))])
    k = int(M["bc-300"].sum())
    lo, hi = wilson(k, M["bc-300"].size)
    print("  at %d seeds, a single task's 95 %% interval is %.3f wide on"
          % (N_SEEDS, widths.mean()))
    print("  average (worst %.3f).  The whole-suite interval is %.3f wide."
          % (widths.max(), hi - lo))
    print("  -> the suite mean is %.0fx sharper than any task in it."
          % (widths.mean() / (hi - lo)))
    record("intervals", mean_task_width=float(widths.mean()),
           worst_task_width=float(widths.max()), suite_width=hi - lo,
           ratio=float(widths.mean() / (hi - lo)))

    print("\n  seeds needed for a per-task interval of a given width (p=0.8):")
    for target in (0.40, 0.20, 0.10, 0.05):
        n = 4
        while n < 100_000:
            w = wilson(int(round(0.8 * n)), n)
            if w[1] - w[0] <= target:
                break
            n = int(n * 1.15) + 1
        print("     +-%.2f  needs %5d episodes on that one task"
              % (target / 2, n))
        record("task_seeds", target_width=target, episodes=n)

    # is it worth spending the episodes unevenly?
    total = len(SUITE) * N_SEEDS
    eq = np.full(len(SUITE), N_SEEDS, dtype=float)
    w = np.sqrt(var) + 1e-6
    prop = np.maximum(3, np.round(total * w / w.sum()))
    prop = prop * total / prop.sum()
    se_eq = np.sqrt((var / eq).sum()) / len(SUITE)
    se_prop = np.sqrt((var / prop).sum()) / len(SUITE)
    print("\n  spending the same total episodes where the variance is:")
    print("     %d each : %.4f      variance-weighted : %.4f  (%.2fx)"
          % (N_SEEDS, se_eq, se_prop, se_eq / se_prop))
    print("     -> a null result: p(1-p) is nearly flat for p between 0.5 and")
    print("        0.95, which is where almost every task sits.")
    record("allocation", equal_se=float(se_eq), proportional_se=float(se_prop),
           improvement=float(se_eq / se_prop))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    order = np.argsort(p)
    lo = np.array([wilson(int(M["bc-300"][i].sum()), N_SEEDS)[0] for i in order])
    hi = np.array([wilson(int(M["bc-300"][i].sum()), N_SEEDS)[1] for i in order])
    ax[0].fill_between(range(len(SUITE)), lo, hi, alpha=.3, color="#1976d2",
                       label="95 % interval")
    ax[0].plot(p[order], color="#0d47a1", label="pass rate")
    ax[0].set_xlabel("task, sorted by pass rate"); ax[0].set_ylabel("pass rate")
    ax[0].set_title("20 seeds is a wide interval"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].bar(range(len(SUITE)), var[order], color="#c62828")
    ax[1].set_xlabel("task, same order"); ax[1].set_ylabel("p(1-p)")
    ax[1].set_title("all the noise comes from the middle")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "variance.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    POLICIES.update(build_systems())
    print("suite: %d tasks x %d seeds x %d systems = %d episodes"
          % (len(SUITE), N_SEEDS, len(SYSTEMS), len(SUITE) * N_SEEDS * len(SYSTEMS)))
    M = run_matrix(SYSTEMS, BASE_SEEDS)
    exp1_dashboard(M)
    exp2_power(M)
    exp3_aggregate(M)
    exp4_seeds(M)
    exp5_allocation(M)

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
