"""Project 30 -- factor graphs: the back end, and how it survives being lied to.

Seven experiments:

  1. the graph, the solve, and where the information actually lives
  2. sparsity: why a 4000-variable problem is easier than it sounds
  3. outliers: how many lies before plain least squares gives up
  4. four robust kernels, and the outlier rate each one survives to
  5. switchable constraints and GNC against a fixed kernel
  6. initialization: Gauss-Newton is a local method and behaves like one
  7. marginalization: turning a smoother into a filter, and measuring the cost

Runs in about five minutes.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "29-2d-lidar-slam"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from posegraph import PoseGraph, chain, align_and_ate, KERNELS       # noqa: E402
from scanmatch import wrap, between, compose                         # noqa: E402
from factorgraph import (make_loop_trajectory, build_graph, gnc,      # noqa: E402
                         optimize_switchable, marginalize, linear_system)
from plot_style import COLORS, use_style, save                        # noqa: E402

import matplotlib.pyplot as plt                                       # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# =====================================================================  1
def exp1_the_graph(rng):
    banner("1. A factor graph, solved")

    truth, rels = make_loop_trajectory()
    g, outlier, init = build_graph(truth, rels, rng, n_good_lc=8)
    n_odom = len(rels)

    ate0, _ = align_and_ate(init, truth)
    t0 = time.time()
    hist, _ = g.optimize(iters=40)
    dur = time.time() - t0
    ate1, aligned = align_and_ate(g.poses(), truth)

    print(f"  {len(g.nodes)} pose variables ({3*len(g.nodes)} numbers)")
    print(f"  {n_odom} odometry factors + {len(g.edges)-n_odom} loop closures "
          f"= {3*len(g.edges)} equations")
    print(f"  solved in {dur*1000:.0f} ms, {len(hist)} Gauss-Newton iterations")
    print(f"  chi2 {hist[0]:.2f} -> {hist[-1]:.2f}")
    print(f"  trajectory error {ate0:.4f} m -> {ate1:.4f} m "
          f"({ate0/ate1:.1f}x better)")

    # Where does the improvement come from?  Re-solve with the loop closures
    # removed and with them alone, to see which factors carry the information.
    g_no = PoseGraph()
    for p in init:
        g_no.add_node(p)
    for k in range(n_odom):
        g_no.add_edge(*g.edges[k])
    g_no.optimize(iters=40)
    ate_no, _ = align_and_ate(g_no.poses(), truth)
    print(f"\n  optimizing the ODOMETRY FACTORS ALONE: {ate0:.4f} -> "
          f"{ate_no:.4f} m -- no change at all.")
    print("  That is not a bug and it is the most important structural fact in")
    print("  this project.  A chain of n-1 relative measurements between n poses")
    print("  has exactly as many equations as unknowns (after pinning the first")
    print("  pose), so there is exactly ONE trajectory that satisfies it perfectly:")
    print("  the one you get by chaining them.  Nothing is left to optimize.")
    print("  Every loop closure adds 3 more equations without adding any")
    print("  unknowns, and only then does 'best fit' start to mean anything.")

    print("\n  the same odometry, with more and more loop closures added:")
    for k in (0, 1, 2, 4, 8, 16, 32):
        # same seed every time, so only the loop-closure count changes
        gk, _, ik = build_graph(truth, rels, np.random.default_rng(77),
                                n_good_lc=k)
        gk.optimize(iters=40)
        a, _ = align_and_ate(gk.poses(), truth)
        a0, _ = align_and_ate(ik, truth)
        print(f"    {k:2d} loop closures: {a0:.4f} -> {a:.4f} m")
        record(1, "lc_count", n_lc=k, before=a0, after=a)

    record(1, "main", nodes=len(g.nodes), edges=len(g.edges),
           ate_before=ate0, ate_after=ate1, ate_odom_only=ate_no,
           seconds=dur)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax0.plot(truth[:, 0], truth[:, 1], "k--", lw=1.3, label="truth")
    ax0.plot(init[:, 0], init[:, 1], color=COLORS[1],
             label=f"odometry ({ate0:.3f} m)")
    ax0.plot(aligned[:, 0], aligned[:, 1], color=COLORS[0],
             label=f"optimized ({ate1:.3f} m)")
    for k in range(n_odom, len(g.edges)):
        i, j, _, _ = g.edges[k]
        ax0.plot([init[i, 0], init[j, 0]], [init[i, 1], init[j, 1]],
                 color=COLORS[2], lw=0.8, alpha=0.8)
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("Green lines are the only factors that add information")
    ax0.legend(fontsize=8)
    ax1.semilogy(hist, "o-", ms=3, color=COLORS[0])
    ax1.set_xlabel("Gauss-Newton iteration"); ax1.set_ylabel("$\\chi^2$")
    ax1.set_title("Quadratic convergence, once it is close")
    save(fig, os.path.join(OUT, "graph.png"))
    return truth, rels


# =====================================================================  2
def exp2_sparsity(truth, rels, rng):
    banner("2. Sparsity: the shape of the matrix is the algorithm")

    g, _, init = build_graph(truth, rels, rng, n_good_lc=8)
    H, b = linear_system(g)
    n = H.shape[0]
    nz = int((np.abs(H) > 1e-12).sum())
    print(f"  the normal-equation matrix H is {n} x {n} = {n*n} entries")
    print(f"  {nz} of them are non-zero ({100*nz/(n*n):.2f}%)")
    print(f"  a dense Cholesky of this costs about n^3/3 = {n**3//3:,} operations")
    print(f"  and the sparse one costs about the number of non-zeros, ~{nz:,}")
    print(f"  -> a factor of {n**3/3/nz:.0f} even at this toy size")

    # measure how it grows
    rows = []
    for n_side in (10, 20, 40, 80):
        tr, rl = make_loop_trajectory(n_side=n_side)
        gg, _, _ = build_graph(tr, rl, np.random.default_rng(0), n_good_lc=8)
        HH, _ = linear_system(gg)
        m = HH.shape[0]
        z = int((np.abs(HH) > 1e-12).sum())
        t0 = time.time()
        gg.optimize(iters=20)
        rows.append((len(gg.nodes), m, z, 100.0 * z / m ** 2, time.time() - t0))
    print(f"\n  {'poses':>7} {'H size':>8} {'non-zero':>9} {'density':>9} "
          f"{'solve (s)':>10}")
    for p, m, z, d, tm in rows:
        print(f"  {p:7d} {m:8d} {z:9d} {d:8.2f}% {tm:10.3f}")
    print(f"\n  Density falls as the problem grows: {rows[0][3]:.1f}% at "
          f"{rows[0][0]} poses, {rows[-1][3]:.2f}% at {rows[-1][0]}.")
    print("  The reason is structural, not accidental.  Each factor touches two")
    print("  poses, so it fills in a fixed 6x6 patch of H no matter how big the")
    print("  graph is.  Add a pose and you add one row and one factor: the number")
    print("  of non-zeros grows LINEARLY while the matrix area grows quadratically.")
    print("  This is why SLAM back ends scale, and it is entirely a property of")
    print("  the graph -- a robot only ever measures things near it in space and")
    print("  time, so it can never produce a dense problem.")
    print(f"  Our solver here uses a DENSE solve, and you can watch it lose: "
          f"{rows[-1][4]/rows[0][4]:.0f}x")
    print(f"  the time for {rows[-1][0]//rows[0][0]}x the poses.  A sparse solver "
          f"(what GTSAM, Ceres and g2o all ship) would be near-linear.")

    for p, m, z, d, tm in rows:
        record(2, "sparsity", poses=p, dim=m, nonzero=z, density_pct=d, seconds=tm)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax0.spy(np.abs(H) > 1e-12, markersize=0.6, color=COLORS[0])
    ax0.set_title("H: a band, plus a dot per loop closure", fontsize=9)
    ax1.loglog([r[0] for r in rows], [r[2] for r in rows], "o-", color=COLORS[0],
               label="non-zeros in H")
    ax1.loglog([r[0] for r in rows], [r[1] ** 2 for r in rows], "s--",
               color=COLORS[1], label="entries in H")
    ax1.set_xlabel("poses"); ax1.set_ylabel("count")
    ax1.set_title("Linear against quadratic")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "sparsity.png"))


# =====================================================================  3
def exp3_outliers(truth, rels):
    banner("3. How many lies does plain least squares survive?")

    rows = []
    for n_bad in (0, 1, 2, 3, 5, 8, 12, 20):
        ates = []
        for seed in range(8):
            r = np.random.default_rng(1000 + seed)
            g, out, init = build_graph(truth, rels, r, n_good_lc=8,
                                       n_bad_lc=n_bad)
            g.optimize(iters=40)
            ates.append(align_and_ate(g.poses(), truth)[0])
        rows.append((n_bad, float(np.median(ates)), float(np.mean(ates))))

    print(f"  8 true loop closures throughout; only the number of lies changes")
    print(f"  {'false':>6} {'as % of':>9} {'median ATE':>11} {'mean ATE':>10}")
    print(f"  {'edges':>6} {'closures':>9} {'(m)':>11} {'(m)':>10}")
    for nb, md, mn in rows:
        print(f"  {nb:6d} {100*nb/(8+nb):8.0f}% {md:11.4f} {mn:10.4f}")
    print(f"\n  ONE fabricated edge among nine takes the answer from "
          f"{rows[0][1]:.4f} m to {rows[1][1]:.4f} m")
    print(f"  -- {rows[1][1]/rows[0][1]:.0f}x worse, from 11% of the loop closures "
          f"being wrong.")
    print("  Least squares has no notion of a minority report.  It minimises the")
    print("  SUM OF SQUARED residuals, so an edge that disagrees by 20 times the")
    print("  expected amount contributes 400 times the pull of a good one.  Nine")
    print("  honest edges pulling with force 1 lose to one liar pulling with 400.")
    print("  Note also that the damage does not keep growing.  One lie already")
    print("  moves the answer as far as twenty do, because the first one is")
    print("  already enough to overwhelm the honest edges -- there is a cliff,")
    print("  not a slope, and you go over it at the first outlier.")

    for nb, md, mn in rows:
        record(3, "outliers", n_bad=nb, median_ate=md, mean_ate=mn)
    return rows


# =====================================================================  4
def exp4_kernels(truth, rels, l2_rows):
    banner("4. Four kernels, and where each one gives up")

    kernels = ["l2", "huber", "cauchy", "gm"]
    n_bads = [0, 1, 3, 6, 10, 16, 24]
    grid = {}
    for kern in kernels:
        for nb in n_bads:
            ates = []
            for seed in range(8):
                r = np.random.default_rng(1000 + seed)
                g, out, init = build_graph(truth, rels, r, n_good_lc=8,
                                           n_bad_lc=nb)
                g.optimize(iters=50, kernel=kern, delta=2.0)
                ates.append(align_and_ate(g.poses(), truth)[0])
            grid[(kern, nb)] = float(np.median(ates))

    print(f"  {'false':>6} " + "".join(f"{k:>12}" for k in kernels)
          + "     (median trajectory error, m)")
    for nb in n_bads:
        print(f"  {nb:6d} " + "".join(f"{grid[(k, nb)]:12.4f}" for k in kernels))

    clean = grid[("l2", 0)]
    print(f"\n  A clean graph solves to {clean:.4f} m.  Call a run 'survived' if")
    print(f"  it stays under {5*clean:.3f} m (5x that).")
    print(f"  {'kernel':>8} {'survives up to':>16} {'cost when clean':>17}")
    for k in kernels:
        surv = [nb for nb in n_bads if grid[(k, nb)] < 5 * clean]
        top = max(surv) if surv else 0
        print(f"  {k:>8} {top:12d} lies {100*(grid[(k,0)]/clean-1):+16.1f}%")
        record(4, "kernel_summary", kernel=k, survives=top,
               clean_cost_pct=100 * (grid[(k, 0)] / clean - 1))
    print("\n  Reading the row for a clean graph: every robust kernel costs")
    print("  something even when there is nothing to be robust to, because it")
    print("  down-weights the tail of a perfectly healthy noise distribution.")
    print("  That is the premium you pay for the insurance.")
    print("  Reading down a column: the kernels that switch an outlier fully OFF")
    print("  (Cauchy, Geman-McClure) stay FLAT no matter how many lies arrive,")
    print("  while Huber degrades steadily -- it stays inside the survival")
    print("  threshold but its error grows several-fold across the sweep.")
    print("  Huber's weight for a residual e is delta/e, so a hundred-sigma")
    print("  outlier still pulls with a hundredth of the force of a good edge --")
    print("  times its enormous disagreement, which is not nothing.  Cauchy's")
    print("  weight falls as 1/e^2 and Geman-McClure's as 1/e^4, so for them a")
    print("  far-out edge really is switched off rather than merely quietened.")

    for (k, nb), v in grid.items():
        record(4, "kernel_grid", kernel=k, n_bad=nb, ate=v)

    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for i, k in enumerate(kernels):
        ax.semilogy(n_bads, [grid[(k, nb)] for nb in n_bads], "o-",
                    color=COLORS[i], label=k)
    ax.axhline(5 * clean, ls=":", color="k", lw=1, label="survival threshold")
    ax.set_xlabel("fabricated loop closures (against 8 real ones)")
    ax.set_ylabel("median trajectory error (m)")
    ax.set_title("Where each kernel breaks")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "kernels.png"))
    return grid, clean, n_bads


# =====================================================================  5
def exp5_switchable_gnc(truth, rels, grid, clean, n_bads):
    banner("5. Letting the optimizer decide which edges to believe")

    rows = []
    for nb in n_bads:
        res = {}
        # switchable constraints
        for pw in (1.0, 10.0):
            ates, tp, fp, fn = [], 0, 0, 0
            for seed in range(6):
                r = np.random.default_rng(1000 + seed)
                g, out, init = build_graph(truth, rels, r, n_good_lc=8,
                                           n_bad_lc=nb)
                n_odom = len(rels)
                sw = optimize_switchable(g, n_odom, iters=25, prior_weight=pw)
                ates.append(align_and_ate(g.poses(), truth)[0])
                off = sw < 0.5
                tp += int(np.sum(off & out)); fp += int(np.sum(off & ~out))
                fn += int(np.sum(~off & out))
            res[f"switch pw={pw:g}"] = (float(np.median(ates)), tp, fp, fn)
        # GNC
        ates, tp, fp, fn = [], 0, 0, 0
        for seed in range(6):
            r = np.random.default_rng(1000 + seed)
            g, out, init = build_graph(truth, rels, r, n_good_lc=8, n_bad_lc=nb)
            w, mu = gnc(g, kernel="gm", delta=2.0)
            ates.append(align_and_ate(g.poses(), truth)[0])
            off = w < 0.1
            tp += int(np.sum(off & out)); fp += int(np.sum(off & ~out))
            fn += int(np.sum(~off & out))
        res["GNC (Geman-McClure)"] = (float(np.median(ates)), tp, fp, fn)
        rows.append((nb, res))

    names = list(rows[0][1].keys())
    print(f"  {'false':>6} " + "".join(f"{n:>22}" for n in names))
    for nb, res in rows:
        print(f"  {nb:6d} " + "".join(f"{res[n][0]:22.4f}" for n in names))
    print(f"\n  and how well each one identified WHICH edges were lies:")
    print(f"  {'method':>22} {'caught':>8} {'wrongly':>9} {'missed':>8} "
          f"{'precision':>10} {'recall':>8}")
    for n in names:
        tp = sum(r[1][n][1] for r in rows)
        fp = sum(r[1][n][2] for r in rows)
        fn = sum(r[1][n][3] for r in rows)
        print(f"  {n:>22} {tp:8d} {fp:9d} {fn:8d} "
              f"{tp/max(tp+fp,1):10.3f} {tp/max(tp+fn,1):8.3f}")
        record(5, "identification", method=n, tp=tp, fp=fp, fn=fn,
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    best_fixed = {nb: min(grid[(k, nb)] for k in ("cauchy", "gm"))
                  for nb in n_bads}
    print(f"\n  against the best fixed kernel from experiment 4:")
    print(f"  {'false':>6} {'best fixed':>12} {'best adaptive':>15}")
    for nb, res in rows:
        ba = min(v[0] for v in res.values())
        print(f"  {nb:6d} {best_fixed[nb]:12.4f} {ba:15.4f}")
    print("\n  The adaptive methods do not merely down-weight: they hand you a")
    print("  LIST of which measurements they refused, which a fixed kernel never")
    print("  does.  That list is the useful product -- a front end that keeps")
    print("  proposing the same false loop closure can be told to stop.")

    for nb, res in rows:
        for n, v in res.items():
            record(5, "adaptive", n_bad=nb, method=n, ate=v[0])

    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, n in enumerate(names):
        ax.semilogy(n_bads, [r[1][n][0] for r in rows], "o-", color=COLORS[i],
                    label=n)
    ax.semilogy(n_bads, [grid[("l2", nb)] for nb in n_bads], "s--",
                color=COLORS[6], label="plain least squares")
    ax.semilogy(n_bads, [best_fixed[nb] for nb in n_bads], "^--",
                color=COLORS[5], label="best fixed kernel")
    ax.set_xlabel("fabricated loop closures")
    ax.set_ylabel("median trajectory error (m)")
    ax.set_title("Deciding which edges to believe, instead of how much")
    ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "switchable.png"))


# =====================================================================  6
def exp6_initialization(truth, rels):
    banner("6. Gauss-Newton is a local method")

    rows = []
    for name in ("odometry chain", "all zeros", "random", "truth + noise"):
        ates, iters = [], []
        for seed in range(10):
            r = np.random.default_rng(2000 + seed)
            g, out, init = build_graph(truth, rels, r, n_good_lc=8)
            n = len(g.nodes)
            if name == "all zeros":
                g.nodes = [np.zeros(3) for _ in range(n)]
            elif name == "random":
                g.nodes = [np.array([r.uniform(-8, 8), r.uniform(-8, 8),
                                     r.uniform(-np.pi, np.pi)]) for _ in range(n)]
            elif name == "truth + noise":
                g.nodes = [truth[i] + np.array([0.2, 0.2, 0.05]) *
                           r.standard_normal(3) for i in range(n)]
            hist, _ = g.optimize(iters=100)
            ates.append(align_and_ate(g.poses(), truth)[0])
            iters.append(len(hist))
        rows.append((name, float(np.median(ates)), float(np.mean(iters))))

    print(f"  {'initial guess':>16} {'median ATE (m)':>16} {'iterations':>12}")
    for n, a, it in rows:
        print(f"  {n:>16} {a:16.4f} {it:12.1f}")
    good = rows[0][1]
    print(f"\n  From the odometry chain: {good:.4f} m.  From all zeros: "
          f"{rows[1][1]:.4f} m")
    print(f"  ({rows[1][1]/good:.0f}x worse).  From random poses: "
          f"{rows[2][1]:.4f} m.")
    print("  Gauss-Newton walks downhill from where you put it, and a pose graph")
    print("  has many valleys because rotations wrap around: a trajectory folded")
    print("  the wrong way can satisfy most of its edges quite well.")
    print("  This is why nobody ever initializes a SLAM back end from nothing.")
    print("  The odometry chain is not a convenience, it is the thing that puts")
    print("  the optimizer in the right valley -- exactly the role the closed-form")
    print("  steps played for the Levenberg-Marquardt refinement in project 16.")

    for n, a, it in rows:
        record(6, "initialization", init=n, ate=a, iterations=it)


# =====================================================================  7
def exp7_marginalization(truth, rels, rng):
    banner("7. Marginalization: a smoother becoming a filter")

    g, _, init = build_graph(truth, rels, rng, n_good_lc=8)
    g.optimize(iters=40)
    H, b = linear_system(g)
    n = len(g.nodes)

    rows = []
    for keep_last in (n, n // 2, n // 4, 10, 4):
        keep = np.arange(3 * (n - keep_last), 3 * n)
        drop = np.arange(0, 3 * (n - keep_last))
        if len(drop) == 0:
            Hm, bm = H, b
        else:
            Hm, bm = marginalize(H, b, keep, drop)
        dens = 100.0 * (np.abs(Hm) > 1e-9).sum() / max(Hm.size, 1)
        # solve the marginalized system and compare the kept poses
        Hf = H.copy()
        Hf[0:3, :] = 0.0; Hf[:, 0:3] = 0.0; Hf[0:3, 0:3] = np.eye(3)
        bf = b.copy(); bf[0:3] = 0.0
        dx_full = np.linalg.solve(Hf + 1e-9 * np.eye(3 * n), bf)
        try:
            dx_marg = np.linalg.solve(Hm + 1e-9 * np.eye(len(keep)), bm)
            diff = float(np.max(np.abs(dx_marg - dx_full[keep])))
        except np.linalg.LinAlgError:
            diff = float("nan")
        rows.append((keep_last, len(keep), dens, diff,
                     100.0 * (np.abs(H[np.ix_(keep, keep)]) > 1e-9).sum()
                     / max(len(keep) ** 2, 1)))

    print(f"  {'poses':>7} {'dim':>5} {'density BEFORE':>15} {'density AFTER':>14} "
          f"{'answer changed':>15}")
    print(f"  {'kept':>7} {'':>5} {'marginalizing':>15} {'marginalizing':>14} {'by':>15}")
    for kl, d, dens, diff, dens0 in rows:
        print(f"  {kl:7d} {d:5d} {dens0:14.1f}% {dens:13.1f}% {diff:15.2e}")
    print(f"\n  The 'answer changed by' column is the whole point: marginalizing")
    print(f"  is EXACT.  Keeping only the last {rows[-1][0]} poses gives numerically")
    print(f"  the same answer for those poses ({rows[-1][3]:.1e}) as solving all "
          f"{n} at once.")
    print("  No information was thrown away.  It was folded into a new, denser")
    print("  factor connecting whatever the dropped variables used to touch.")
    print(f"\n  And that is the cost: density goes {rows[0][2]:.1f}% -> "
          f"{rows[-1][2]:.1f}%.")
    print("  A filter is a smoother that marginalizes everything except the")
    print("  present.  It is exact, it is cheap per step, and it produces a small")
    print("  dense problem instead of a large sparse one.  The trap is that")
    print("  marginalizing also FREEZES the linearization point: the Jacobians")
    print("  used at the moment of marginalizing are baked in and can never be")
    print("  reconsidered.  A smoother can re-linearize its whole window when a")
    print("  loop closure changes its mind about where the robot was.  That is")
    print("  the real reason modern SLAM smooths -- not the sparsity, the ability")
    print("  to change its mind.")

    for kl, d, dens, diff, dens0 in rows:
        record(7, "marginalization", poses_kept=kl, dim=d, density_before=dens0,
               density_after=dens, max_diff=diff)

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, kl in zip(axes, (n, n // 4, 10)):
        keep = np.arange(3 * (n - kl), 3 * n)
        drop = np.arange(0, 3 * (n - kl))
        Hm = H if len(drop) == 0 else marginalize(H, b, keep, drop)[0]
        ax.spy(np.abs(Hm) > 1e-9, markersize=0.8, color=COLORS[0])
        ax.set_title(f"{kl} poses kept", fontsize=9)
    save(fig, os.path.join(OUT, "marginalization.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(31)
    truth, rels = exp1_the_graph(rng)
    exp2_sparsity(truth, rels, rng)
    l2_rows = exp3_outliers(truth, rels)
    grid, clean, n_bads = exp4_kernels(truth, rels, l2_rows)
    exp5_switchable_gnc(truth, rels, grid, clean, n_bads)
    exp6_initialization(truth, rels)
    exp7_marginalization(truth, rels, rng)

    path = os.path.join(OUT, "results.csv")
    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(RESULTS)
    print(f"\n  wrote {path}")
    print(f"\nTotal: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
