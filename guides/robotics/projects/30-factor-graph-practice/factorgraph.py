"""Factor graphs: the same problem as project 29's pose graph, said properly.

A FACTOR GRAPH has two kinds of node.  VARIABLES are the things you want to
know (robot poses here).  FACTORS are the things you were told (measurements,
priors).  An edge joins a factor to every variable it mentions.  The name is
literal: the posterior probability FACTORISES into a product, one term per
factor, and the graph is a picture of that product:

    p(x | z)  proportional to  prod_k  f_k( the variables factor k touches )

Taking the negative logarithm turns the product into a sum and the maximum into
a minimum, so finding the most probable trajectory becomes:

    minimise  sum_k  || r_k(x) ||^2_(information of k)

which is a nonlinear least-squares problem.  A pose graph is the special case
where every factor touches exactly two poses.  Everything below is a pose graph
too; what changes is that the machinery is now written in terms of factors, so
priors, robust kernels, switchable constraints and marginalization all fit into
the same frame.

This file imports project 29's PoseGraph for the solver core and adds:

  * GNC       -- graduated non-convexity, an outlier scheme that anneals
  * switchable constraints -- give every dubious edge its own on/off dial and
                              let the optimizer decide
  * marginalization -- turn old variables into a factor on the ones you keep,
                       which is the exact link between a smoother and a filter
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "29-2d-lidar-slam"))

from posegraph import PoseGraph, KERNELS, align_and_ate, chain     # noqa: E402
from scanmatch import wrap, between, compose, invert               # noqa: E402


# ------------------------------------------------------------ graph building
def make_loop_trajectory(n_side=25, side=8.0, rng=None):
    """A square lap, returned as (true poses, true relative poses)."""
    poses = [np.array([0.0, 0.0, 0.0])]
    rels = []
    step = side / n_side
    for lap_side in range(4):
        for k in range(n_side):
            r = np.array([step, 0.0, 0.0])
            if k == n_side - 1:
                r = np.array([step, 0.0, np.pi / 2])
            rels.append(r)
            poses.append(compose(poses[-1], r))
    return np.array(poses), rels


def build_graph(truth, rels, rng, odom_sigma=(0.05, 0.05, 0.025),
                n_good_lc=8, n_bad_lc=0, lc_sigma=(0.05, 0.05, 0.02),
                bad_scale=1.0):
    """A pose graph with noisy odometry, true loop closures, and lies.

    Returns (graph, list of which edges are outliers).
    """
    noisy_rels = [r + np.array(odom_sigma) * rng.standard_normal(3) for r in rels]
    init = chain(truth[0], noisy_rels)
    g = PoseGraph()
    for p in init:
        g.add_node(p)
    odom_info = np.diag([1.0 / odom_sigma[0] ** 2, 1.0 / odom_sigma[1] ** 2,
                         1.0 / odom_sigma[2] ** 2])
    for k, r in enumerate(noisy_rels):
        g.add_edge(k, k + 1, r, odom_info)
    lc_info = np.diag([1.0 / lc_sigma[0] ** 2, 1.0 / lc_sigma[1] ** 2,
                       1.0 / lc_sigma[2] ** 2])
    outlier = [False] * len(g.edges)

    n = len(truth)
    # good loop closures: pairs whose TRUE poses are genuinely close
    pool = [(i, j) for i in range(n) for j in range(i + n // 3, n)
            if np.linalg.norm(truth[i][:2] - truth[j][:2]) < 1.0]
    rng.shuffle(pool)
    for i, j in pool[:n_good_lc]:
        r = between(truth[i], truth[j]) + np.array(lc_sigma) * rng.standard_normal(3)
        g.add_edge(i, j, r, lc_info)
        outlier.append(False)
    # bad ones: a relative pose invented out of thin air, declared with the SAME
    # confidence as a real measurement.  That is what makes them dangerous --
    # not that they are wrong, but that they claim not to be.
    for _ in range(n_bad_lc):
        i = int(rng.integers(0, n - n // 3))
        j = int(rng.integers(i + n // 4, n))
        r = between(truth[i], truth[j]) + bad_scale * np.array(
            [rng.normal(0, 2.0), rng.normal(0, 2.0), rng.normal(0, 1.0)])
        g.add_edge(i, j, r, lc_info)
        outlier.append(True)
    return g, np.array(outlier), init


# --------------------------------------------------------------------- GNC
def gnc(graph, kernel="gm", mu0=None, factor=1.4, rounds=12, inner=6,
        delta=1.0):
    """Graduated Non-Convexity.

    The problem with a strong robust kernel is that it is not convex: it has
    many local minima, and which one you land in is decided by where you start.
    Start with a bad initial guess and a Geman-McClure kernel will confidently
    switch off the wrong edges and never reconsider.

    GNC dodges that by SOLVING A SEQUENCE of problems.  It begins with the
    kernel scale so wide that the kernel is effectively plain least squares --
    convex, one minimum, no way to get stuck.  Then it shrinks the scale a
    little and re-solves from the previous answer, and repeats.  Each problem is
    only slightly harder than the last and starts from that one's solution, so
    the optimizer is walked gently from the easy convex problem to the hard
    non-convex one it actually wanted to solve.

    "Graduated" as in graduated cylinder -- marked off in steps; "non-convexity"
    because the thing being introduced in steps is the non-convexity itself.
    """
    if mu0 is None:
        r = graph.residuals()
        e2 = np.array([float(rr @ info @ rr)
                       for rr, (_, _, _, info) in zip(r, graph.edges)])
        mu0 = float(np.sqrt(max(e2.max(), 1.0)))
    mu = mu0
    weights = None
    for _ in range(rounds):
        _, weights = graph.optimize(iters=inner, kernel=kernel, delta=mu)
        mu = max(mu / factor, delta)
    return weights, mu


# ------------------------------------------------- switchable constraints
def optimize_switchable(graph, n_odom, iters=40, prior_weight=1.0, lm=1e-6):
    """Give every loop-closure edge its own switch variable in [0, 1].

    Instead of a fixed rule for down-weighting a residual, each dubious edge
    gets an extra unknown `s` that multiplies its information, plus a prior
    pulling `s` towards 1.  The optimizer then TRADES: switching an edge off
    costs `prior_weight` but saves whatever residual that edge was contributing.
    An edge that disagrees with everything else by more than that price gets
    switched off, and one that merely disagrees a little does not.

    The appeal over a fixed kernel is that the threshold is not something you
    tune -- it emerges from the balance between the prior and the data.  The
    catch, which experiment 5 measures, is that `prior_weight` is a knob too;
    it has just moved somewhere less obvious.

    Implemented here as iteratively reweighted least squares with a closed-form
    switch update, which is what the joint solve reduces to for this prior.
    """
    m = len(graph.edges)
    sw = np.ones(m)
    for _ in range(iters):
        # 1. solve the poses with the current switches
        saved = [e[3].copy() for e in graph.edges]
        for k in range(m):
            i, j, rel, info = graph.edges[k]
            graph.edges[k] = (i, j, rel, saved[k] * sw[k] ** 2)
        graph.optimize(iters=3, kernel="l2", lm=lm)
        for k in range(m):
            i, j, rel, _ = graph.edges[k]
            graph.edges[k] = (i, j, rel, saved[k])
        # 2. solve each switch in closed form, holding the poses fixed:
        #    minimise  s^2 e2_k + prior_weight (1 - s)^2   ->   s = pw/(pw + e2)
        r = graph.residuals()
        e2 = np.array([float(rr @ info @ rr)
                       for rr, (_, _, _, info) in zip(r, graph.edges)])
        new = prior_weight / (prior_weight + e2)
        new[:n_odom] = 1.0                     # odometry is never switched off
        if np.max(np.abs(new - sw)) < 1e-6:
            sw = new
            break
        sw = new
    return sw


# ------------------------------------------------------- marginalization
def marginalize(H, b, keep, drop):
    """Remove variables from a linear system without losing their information.

    Split the normal equations into the block you keep and the block you drop:

        [ H_kk  H_kd ] [ x_k ]   [ b_k ]
        [ H_dk  H_dd ] [ x_d ] = [ b_d ]

    Solve the bottom row for x_d and substitute it into the top.  What is left
    is a smaller system in x_k alone:

        ( H_kk - H_kd H_dd^-1 H_dk ) x_k  =  b_k - H_kd H_dd^-1 b_d

    That subtracted term is the SCHUR COMPLEMENT, named after Issai Schur.  It
    is not an approximation: the marginalized system gives exactly the same
    answer for the variables you kept as the full system would have.

    The price is not accuracy but SPARSITY.  Every pair of kept variables that
    was connected through a dropped one becomes directly connected, so the
    matrix fills in.  Marginalize aggressively and you end up with a small dense
    problem instead of a large sparse one -- and a large sparse problem is often
    the cheaper of the two.  This single trade-off is the whole difference
    between a filter (marginalize everything but the present) and a smoother
    (keep the window, stay sparse), which is why modern SLAM chose smoothing.
    """
    Hkk = H[np.ix_(keep, keep)]
    Hkd = H[np.ix_(keep, drop)]
    Hdd = H[np.ix_(drop, drop)]
    bk, bd = b[keep], b[drop]
    Hdd_inv = np.linalg.pinv(Hdd)
    return Hkk - Hkd @ Hdd_inv @ Hkd.T, bk - Hkd @ Hdd_inv @ bd


def linear_system(graph, kernel="l2", delta=1.0):
    """Build the H and b of one Gauss-Newton step, without taking it."""
    from posegraph import _J_between
    n = len(graph.nodes)
    H = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    r_all = graph.residuals()
    e2 = np.array([float(r @ info @ r)
                   for r, (_, _, _, info) in zip(r_all, graph.edges)])
    w_all = KERNELS[kernel](e2, delta)
    for k, (i, j, rel, info) in enumerate(graph.edges):
        r = r_all[k]
        Ja, Jb = _J_between(graph.nodes[i], graph.nodes[j])
        Wi = w_all[k] * info
        si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
        H[si, si] += Ja.T @ Wi @ Ja
        H[si, sj] += Ja.T @ Wi @ Jb
        H[sj, si] += Jb.T @ Wi @ Ja
        H[sj, sj] += Jb.T @ Wi @ Jb
        b[si] -= Ja.T @ Wi @ r
        b[sj] -= Jb.T @ Wi @ r
    return H, b
