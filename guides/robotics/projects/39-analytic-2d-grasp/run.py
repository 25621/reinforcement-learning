"""Project 39 -- analytic 2D grasping: friction cones and force closure.

Seven experiments:

  1. one picture: a grasp that holds, a grasp that does not, and why
  2. every two-finger candidate on five shapes, scored
  3. the friction sweep -- how much of grasping is just rubber
  4. three force-closure tests, and which one is wrong
  5. what the quality number is actually summarising
  6. the arbitrary length hiding inside every grasp metric
  7. how many fingers you need, with and without friction

Runs in about one minute.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from grasp2d import (SHAPES, LAMBDA, antipodal, best_pair, centroid,          # noqa: E402
                     cone_generators, force_closure, force_closure_sampled,
                     max_resistible, perimeter_samples, quality, score_pairs)
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.patches import Polygon as MplPolygon                          # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
MU = 0.4          # a rubber pad on a plastic object
FN = 20.0         # newtons of total normal force the gripper can squeeze with
NAMES = list(SHAPES)


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<44s} {value}")


def draw_shape(ax, P, color="#c9d3dd", edge="#42505e"):
    ax.add_patch(MplPolygon(P, closed=True, fc=color, ec=edge, lw=1.4, zorder=1))


def draw_cone(ax, p, n, mu, L=0.022, color=COLORS[1]):
    a = np.arctan(mu)
    for s in (+1, -1):
        c, sn = np.cos(s * a), np.sin(s * a)
        d = np.array([c * n[0] - sn * n[1], sn * n[0] + c * n[1]])
        ax.plot([p[0], p[0] + L * d[0]], [p[1], p[1] + L * d[1]],
                color=color, lw=1.0, zorder=3)
    th = np.linspace(-a, a, 24)
    arc = np.stack([np.cos(th) * n[0] - np.sin(th) * n[1],
                    np.sin(th) * n[0] + np.cos(th) * n[1]], 1) * L
    ax.fill(np.concatenate([[p[0]], p[0] + arc[:, 0]]),
            np.concatenate([[p[1]], p[1] + arc[:, 1]]),
            color=color, alpha=0.18, zorder=2)


# ---------------------------------------------------------------------------
# 1. the picture
# ---------------------------------------------------------------------------

def exp1_picture():
    print("\n[1] friction cones, and the line between the fingers")
    P = SHAPES["rect"]
    ref = centroid(P)
    cases = [
        ("force closed", np.array([0.010, -0.020]), np.array([0.0, 1.0]),
         np.array([0.010, 0.020]), np.array([0.0, -1.0])),
        ("slips", np.array([0.036, -0.020]), np.array([0.0, 1.0]),
         np.array([-0.020, 0.020]), np.array([0.0, -1.0])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (label, p1, n1, p2, n2) in zip(axes, cases):
        W = cone_generators(np.stack([p1, p2]), np.stack([n1, n2]), MU, ref)
        fc = force_closure(W)
        q = quality(W)
        draw_shape(ax, P)
        draw_cone(ax, p1, n1, MU)
        draw_cone(ax, p2, n2, MU)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=COLORS[0], lw=1.6,
                zorder=4, label="line between contacts")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color=COLORS[0], ms=6, zorder=5)
        ax.plot(*ref, "x", color="#42505e", ms=7, zorder=5)
        ax.set_title(f"{label}\nforce closure = {fc},  Q = {q:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-0.07, 0.07)
        ax.set_ylim(-0.05, 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        record("1_picture", f"{label}: force_closure", fc)
        record("1_picture", f"{label}: quality", round(q, 4))
        record("1_picture", f"{label}: antipodal_test",
               antipodal(p1, n1, p2, n2, MU))
        ang = np.degrees(np.arccos(abs((p2 - p1) @ n1) / np.linalg.norm(p2 - p1)))
        record("1_picture", f"{label}: line vs normal (deg)", round(ang, 1))
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"friction cone half-angle atan(mu) = "
                 f"{np.degrees(np.arctan(MU)):.1f} deg   (mu = {MU})", y=1.02)
    save(fig, os.path.join(OUT, "cones.png"))


# ---------------------------------------------------------------------------
# 2. every candidate, scored
# ---------------------------------------------------------------------------

def exp2_landscape():
    print("\n[2] scoring every two-finger candidate")
    fig = plt.figure(figsize=(10.5, 5.6))
    for k, name in enumerate(NAMES):
        P = SHAPES[name]
        t0 = time.time()
        best, r = best_pair(P, MU)
        dt = time.time() - t0
        ax = fig.add_subplot(2, 3, k + 1)
        draw_shape(ax, P)
        # every force-closed candidate as a faint line, coloured by quality
        fcq = r["q"][r["fc"]]
        order = np.argsort(fcq)
        ii, jj = r["i"][r["fc"]][order], r["j"][r["fc"]][order]
        qq = fcq[order]
        norm = qq / max(qq.max(), 1e-9) if len(qq) else qq
        for a, b, u in zip(ii, jj, norm):
            ax.plot(*np.stack([r["pts"][a], r["pts"][b]]).T,
                    color=plt.cm.viridis(u), lw=0.6, alpha=0.5, zorder=2)
        if best is not None:
            a, b = r["i"][best], r["j"][best]
            ax.plot(*np.stack([r["pts"][a], r["pts"][b]]).T, color=COLORS[1],
                    lw=2.4, zorder=4)
            ax.plot(*np.stack([r["pts"][a], r["pts"][b]]).T, "o",
                    color=COLORS[1], ms=5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(P[:, 0].min() - 0.008, P[:, 0].max() + 0.008)
        ax.set_ylim(P[:, 1].min() - 0.008, P[:, 1].max() + 0.008)
        ax.grid(False)
        ax.set_title(f"{name}: {r['fc'].sum()}/{len(r['fc'])} closed, "
                     f"Q* = {r['q'].max():.3f}", fontsize=9)
        record("2_landscape", f"{name}: candidates", len(r["fc"]))
        record("2_landscape", f"{name}: force-closed fraction",
               round(float(r["fc"].mean()), 4))
        record("2_landscape", f"{name}: best quality", round(float(r["q"].max()), 4))
        record("2_landscape", f"{name}: quality spread (closed only)",
               f"{fcq.min():.2e} .. {fcq.max():.2e}" if len(fcq) else "none")
        record("2_landscape", f"{name}: search time (s)", round(dt, 3))
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=ax, fraction=0.35, pad=0.45, location="right")
    cb.set_label("quality, relative to the\nbest grasp on that shape", fontsize=8)
    ax.text(-0.10, 0.98, "each thin line is one\nforce-closed\ntwo-finger grasp\n\n"
                         "orange = the best\n\n"
                         "the triangle has NO\ntwo-finger grasp at\n"
                         "this friction: no two\nfaces come close\n"
                         "enough to facing\neach other",
            fontsize=8.5, va="top", transform=ax.transAxes)
    fig.suptitle(f"every two-finger candidate within a 75 mm gripper (mu = {MU})")
    save(fig, os.path.join(OUT, "landscape.png"))


# ---------------------------------------------------------------------------
# 3. the friction sweep
# ---------------------------------------------------------------------------

def exp3_friction():
    print("\n[3] how much of grasping is just rubber")
    mus = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2])
    frac = np.zeros((len(NAMES), len(mus)))
    bestq = np.zeros_like(frac)
    for a, name in enumerate(NAMES):
        for b, mu in enumerate(mus):
            r = score_pairs(SHAPES[name], mu, n_samples=70)
            frac[a, b] = r["fc"].mean()
            bestq[a, b] = r["q"].max()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for a, name in enumerate(NAMES):
        axes[0].plot(mus, 100 * frac[a], "o-", ms=3, label=name)
        axes[1].plot(mus, bestq[a], "o-", ms=3, label=name)
    axes[0].set_xlabel("friction coefficient mu")
    axes[0].set_ylabel("% of candidates force-closed")
    axes[0].set_title("a wider cone forgives more misalignment")
    axes[1].set_xlabel("friction coefficient mu")
    axes[1].set_ylabel("quality of the best grasp")
    axes[1].set_title("and makes the best grasp stronger")
    axes[1].legend(fontsize=8, ncol=2)
    save(fig, os.path.join(OUT, "friction.png"))
    for a, name in enumerate(NAMES):
        record("3_friction", f"{name}: closed % at mu=0", round(100 * frac[a, 0], 3))
        record("3_friction", f"{name}: closed % at mu=0.4",
               round(100 * frac[a, list(mus).index(0.4)], 2))
        record("3_friction", f"{name}: closed % at mu=1.0",
               round(100 * frac[a, list(mus).index(1.0)], 2))
    record("3_friction", "mean closed % at mu=0", round(100 * frac[:, 0].mean(), 3))
    record("3_friction", "mu 0.2 -> 0.4 multiplies closed % by",
           round(float(frac[:, list(mus).index(0.4)].mean() /
                       frac[:, list(mus).index(0.2)].mean()), 2))


# ---------------------------------------------------------------------------
# 4. three tests
# ---------------------------------------------------------------------------

def exp4_tests():
    print("\n[4] the exact test, the two-finger shortcut, and random sampling")
    rng = np.random.default_rng(0)
    n_ex = n_an = n_sa = 0
    agree_an = agree_sa = 0
    total = 0
    t_ex = t_sa = 0.0
    margins_missed = []
    for name in NAMES:
        P = SHAPES[name]
        ref = centroid(P)
        pts, nrm, _ = perimeter_samples(P, 60)
        idx = rng.integers(0, 60, size=(500, 2))
        for a, b in idx:
            if a == b:
                continue
            W = cone_generators(pts[[a, b]], nrm[[a, b]], MU, ref)
            t0 = time.time()
            ex = force_closure(W)
            t_ex += time.time() - t0
            an = antipodal(pts[a], nrm[a], pts[b], nrm[b], MU)
            t0 = time.time()
            sa = force_closure_sampled(W, n=2000, rng=rng)
            t_sa += time.time() - t0
            total += 1
            n_ex += ex
            n_an += an
            n_sa += sa
            agree_an += (ex == an)
            agree_sa += (ex == sa)
            if ex != sa:
                # how rare is the direction the sampler needed to find?
                d = rng.normal(size=(200000, 3))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                witness = float(((W @ d.T).max(axis=0) <= 0).mean())
                margins_missed.append((ex, sa, witness))
    record("4_tests", "grasps tested", total)
    record("4_tests", "exact: force-closed", n_ex)
    record("4_tests", "two-finger antipodal test agrees (%)",
           round(100 * agree_an / total, 3))
    record("4_tests", "random-direction test agrees (%)",
           round(100 * agree_sa / total, 3))
    record("4_tests", "random-direction disagreements", len(margins_missed))
    if margins_missed:
        fp = sum(1 for ex, sa, _ in margins_missed if sa and not ex)
        w = [x for _, _, x in margins_missed]
        record("4_tests", "  of which called an OPEN grasp closed", fp)
        record("4_tests", "  of which called a CLOSED grasp open",
               len(margins_missed) - fp)
        record("4_tests", "rarest witness direction (fraction of the sphere)",
               f"{min(w):.2e}")
        record("4_tests", "samples needed to catch it ~", int(1 / max(min(w), 1e-12)))
    record("4_tests", "exact test time per grasp (us)", round(1e6 * t_ex / total, 1))
    record("4_tests", "sampled test time per grasp (us)", round(1e6 * t_sa / total, 1))
    record("4_tests", "sampled / exact time", round(t_sa / t_ex, 1))


# ---------------------------------------------------------------------------
# 5. what the quality number summarises
# ---------------------------------------------------------------------------

def exp5_resistance():
    print("\n[5] what one quality number is hiding")
    P = SHAPES["ell"]
    r = score_pairs(P, MU)
    fcq = np.where(r["fc"], r["q"], -1.0)
    order = np.argsort(fcq)
    strong = int(order[-1])
    closed = np.flatnonzero(r["fc"])
    marginal = int(closed[np.argmin(r["q"][closed])])
    ref = r["ref"]

    th = np.linspace(0, 2 * np.pi, 181)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8),
                             subplot_kw=dict(projection="polar"))
    rows = []
    for ax, (label, k) in zip(axes, [("best grasp", strong),
                                     ("weakest force-closed grasp", marginal)]):
        a, b = r["i"][k], r["j"][k]
        W = cone_generators(r["pts"][[a, b]], r["nrm"][[a, b]], MU, ref)
        rad = np.array([max_resistible(W, [np.cos(t), np.sin(t), 0.0]) for t in th])
        ax.plot(th, FN * rad, color=COLORS[0])
        ax.fill(th, FN * rad, color=COLORS[0], alpha=0.15)
        q = quality(W)
        ax.plot(th, np.full_like(th, FN * q), "--", color=COLORS[1], lw=1.2)
        ax.set_title(f"{label}\nQ = {q:.4f}  ->  {FN * q:.2f} N worst case",
                     fontsize=9, pad=14)
        up = FN * max_resistible(W, [0.0, 1.0, 0.0])
        rows.append((label, q, up, FN * q))
        record("5_resistance", f"{label}: quality", round(q, 4))
        record("5_resistance", f"{label}: holds straight up (N)", round(up, 2))
        record("5_resistance", f"{label}: worst direction (N)", round(FN * q, 3))
        record("5_resistance", f"{label}: up / worst ratio", round(up / max(FN * q, 1e-9), 1))
    fig.tight_layout()
    fig.suptitle("pure-force disturbance a grasp resists, by direction "
                 f"(gripper squeezing with {FN:.0f} N)\n"
                 "dashed circle = the Ferrari-Canny radius, which also has to "
                 "cover torque", y=1.12, fontsize=9)
    save(fig, os.path.join(OUT, "resistance.png"))

    # and the ranking question: does passing force closure tell you anything?
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    up_all = []
    for k in closed:
        a, b = r["i"][k], r["j"][k]
        W = cone_generators(r["pts"][[a, b]], r["nrm"][[a, b]], MU, ref)
        up_all.append(FN * max_resistible(W, [0.0, 1.0, 0.0]))
    up_all = np.array(up_all)
    ax.scatter(FN * r["q"][closed], up_all, s=6, alpha=0.35, color=COLORS[0])
    ax.set_xscale("log")
    ax.set_xlabel("worst-direction strength (N)  =  Ferrari-Canny quality")
    ax.set_ylabel("straight-up strength (N)")
    ax.set_title("every point here PASSES force closure")
    save(fig, os.path.join(OUT, "ranking.png"))
    record("5_resistance", "force-closed grasps on 'ell'", len(closed))
    record("5_resistance", "their worst-direction strength spans (N)",
           f"{FN * r['q'][closed].min():.2e} .. {FN * r['q'][closed].max():.2e}")
    record("5_resistance", "orders of magnitude spanned",
           round(float(np.log10(r["q"][closed].max() / r["q"][closed].min())), 2))
    record("5_resistance", "closed grasps holding < 1 N in some direction (%)",
           round(100 * float((FN * r["q"][closed] < 1.0).mean()), 1))


# ---------------------------------------------------------------------------
# 6. the arbitrary length
# ---------------------------------------------------------------------------

def exp6_lambda():
    print("\n[6] the arbitrary length hiding inside the metric")
    lams = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    changes = {}
    for name in ("rect", "ell", "wedge"):
        P = SHAPES[name]
        winners, qs = [], []
        base = None
        ranks = []
        for lam in lams:
            r = score_pairs(P, MU, lam=lam)
            k = int(np.argmax(r["q"]))
            winners.append((r["i"][k], r["j"][k]))
            qs.append(r["q"][k])
            if base is None:
                base = k
                base_pair = (r["i"][k], r["j"][k])
            # where does the lambda=0.01 winner rank under this lambda?
            same = np.flatnonzero((r["i"] == base_pair[0]) & (r["j"] == base_pair[1]))
            ranks.append(int((r["q"] > r["q"][same[0]]).sum()) + 1)
        n_distinct = len(set(winners))
        changes[name] = (n_distinct, max(ranks))
        axes[0].plot(lams, qs, "o-", ms=3, label=name)
        axes[1].plot(lams, ranks, "o-", ms=3, label=name)
        record("6_lambda", f"{name}: distinct winners over lambda", n_distinct)
        record("6_lambda", f"{name}: worst rank of the lambda=0.01 winner", max(ranks))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("torque scale lambda (m)")
    axes[0].set_ylabel("quality of the best grasp")
    axes[0].set_title("the number moves")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rank of the lambda = 0.01 winner")
    axes[1].set_title("and so does the winner")
    axes[1].legend(fontsize=8)
    axes[0].axvline(LAMBDA, color="#8C8C8C", ls=":", lw=1)
    axes[1].axvline(LAMBDA, color="#8C8C8C", ls=":", lw=1)
    save(fig, os.path.join(OUT, "lambda.png"))


# ---------------------------------------------------------------------------
# 7. how many fingers
# ---------------------------------------------------------------------------

def exp7_fingers():
    print("\n[7] how many fingers you need")
    rng = np.random.default_rng(3)
    ks = [2, 3, 4, 5, 6]
    mus = [0.0, 0.1, 0.3, 0.6]
    frac = np.zeros((len(mus), len(ks)))
    P = SHAPES["hex"]
    ref = centroid(P)
    pts, nrm, _ = perimeter_samples(P, 120)
    for a, mu in enumerate(mus):
        for b, k in enumerate(ks):
            ok = 0
            for _ in range(600):
                idx = rng.choice(120, size=k, replace=False)
                W = cone_generators(pts[idx], nrm[idx], mu, ref)
                ok += force_closure(W)
            frac[a, b] = ok / 600
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for a, mu in enumerate(mus):
        ax.plot(ks, 100 * frac[a], "o-", ms=4, label=f"mu = {mu}")
    ax.set_xlabel("number of contacts, placed at random on a hexagon")
    ax.set_ylabel("% force-closed")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)
    ax.set_title("frictionless grasping needs four fingers; friction needs two")
    save(fig, os.path.join(OUT, "fingers.png"))
    for a, mu in enumerate(mus):
        for b, k in enumerate(ks):
            record("7_fingers", f"mu={mu}, k={k}: closed %", round(100 * frac[a, b], 2))


def main():
    use_style()
    t0 = time.time()
    exp1_picture()
    exp2_landscape()
    exp3_friction()
    exp4_tests()
    exp5_resistance()
    exp6_lambda()
    exp7_fingers()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
