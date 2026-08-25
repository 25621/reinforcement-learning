"""Project 42 -- a 6-DoF grasp pipeline you can actually execute.

Seven experiments:

  1. the whole pipeline on one scene, stage by stage
  2. the candidate funnel: which filter kills what
  3. four scorers, judged by whether the object comes up and stays up
  4. why the classical antipodal test struggles here
  5. the collision filter, and what happens without it
  6. clutter
  7. novel shapes and a noisy depth camera

Runs in about eight minutes on CPU.
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

import pick                                                                   # noqa: E402
import grasps                                                                 # noqa: E402
import net as gnet                                                            # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

N_TRAIN_SCENES = 300
N_EXEC = 10
N_EVAL = 110


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<52s} {value}")


def build_scene(rng, n=3, kinds=None, noise=0.0, stride=2):
    """A scene plus everything the pipeline derives from one camera shot."""
    model, data, n_obj = pick.make_scene(rng, n=n, kinds=kinds)
    cam = pick.Cam(model)
    P = pick.table_removed(cam.cloud(model, data, stride=stride, noise=noise,
                                     rng=rng))
    if len(P) < 30:
        cam.renderer.close()
        return None
    N = grasps.normals(P, viewpoint=data.cam_xpos[cam.cid])
    return dict(model=model, data=data, cam=cam, P=P, N=N,
                snap=pick.snapshot(model, data), n_obj=n_obj)


def close_scene(sc):
    sc["cam"].renderer.close()


def run_top(sc, cands, scores):
    """Execute the single best-scoring candidate and report what happened."""
    if not cands or len(scores) == 0:
        return dict(success=False, rise=0.0, lifted=-1)
    k = int(np.argmax(scores))
    pick.restore(sc["model"], sc["data"], sc["snap"])
    return pick.execute(sc["model"], sc["data"], cands[k])


def point_count_score(cands):
    """The oldest heuristic in grasp detection: grab where there is most stuff.

    "Most points between the fingers" is a real baseline (it is roughly GPD's
    coverage term).  It needs no normals, so unlike the antipodal test it does
    not care that the far side of every object is invisible.
    """
    return np.array([g["n_inside"] for g in cands], float)


# ---------------------------------------------------------------------------
# 1. the pipeline
# ---------------------------------------------------------------------------

def exp1_pipeline(state):
    print("\n[1] the pipeline, stage by stage")
    rng = np.random.default_rng(3)
    sc = build_scene(rng, n=3)
    t0 = time.time()
    cands, funnel = grasps.sample_candidates(sc["P"], sc["N"], rng)
    t_gen = time.time() - t0
    ap = np.array([grasps.antipodal_score(g, sc["N"], sc["P"]) for g in cands])
    k = int(np.argmax(point_count_score(cands)))
    g = cands[k]
    pick.restore(sc["model"], sc["data"], sc["snap"])
    r = pick.execute(sc["model"], sc["data"], g, cam=sc["cam"], record=True)

    fig = plt.figure(figsize=(11.4, 6.4))
    ax = fig.add_subplot(2, 3, 1)
    pick.restore(sc["model"], sc["data"], sc["snap"])
    pick.park(sc["model"], sc["data"])
    ax.imshow(sc["cam"].rgb(sc["model"], sc["data"]))
    ax.set_title("what the camera sees", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    P, N = sc["P"], sc["N"]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=P[:, 2], cmap="viridis")
    q = np.arange(0, len(P), 7)
    ax.quiver(P[q, 0], P[q, 1], P[q, 2], N[q, 0], N[q, 1], N[q, 2],
              length=0.012, color="#D55E00", linewidth=0.5)
    ax.set_title(f"{len(P)} points + normals", fontsize=9)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.5, c="#b8c2cc")
    for c in cands[::4]:
        a, cl, p = c["approach"], c["closing"], c["pos"]
        e = p - a * 0.03
        ax.plot(*np.stack([p, e]).T, color=COLORS[0], lw=0.5, alpha=0.5)
    ax.plot(*np.stack([g["pos"] - g["closing"] * 0.03,
                       g["pos"] + g["closing"] * 0.03]).T,
            color=COLORS[1], lw=2.4)
    ax.plot(*np.stack([g["pos"], g["pos"] - g["approach"] * 0.05]).T,
            color=COLORS[1], lw=2.4)
    ax.set_title(f"{len(cands)} candidates; the chosen one in orange", fontsize=9)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    if r["frames"]:
        idx = np.linspace(0, len(r["frames"]) - 1, 3).astype(int)
        for i, j in enumerate(idx):
            ax = fig.add_subplot(2, 3, 4 + i)
            ax.imshow(r["frames"][j])
            ax.set_title(["approach", "close and lift", "shake"][i], fontsize=9)
            ax.axis("off")
    fig.suptitle(f"one scene, end to end -- object lifted: {r['success']}")
    save(fig, os.path.join(OUT, "pipeline.png"))

    record("1_pipeline", "cloud points above the table", len(sc["P"]))
    record("1_pipeline", "candidates generated", len(cands))
    record("1_pipeline", "candidate generation (ms)", round(1000 * t_gen, 1))
    record("1_pipeline", "chosen grasp width (mm)", round(1000 * g["width"], 1))
    record("1_pipeline", "chosen grasp lifted the object", r["success"])
    state["funnel"] = funnel
    close_scene(sc)


# ---------------------------------------------------------------------------
# 2. the funnel + training data
# ---------------------------------------------------------------------------

def exp2_dataset(state):
    print("\n[2] the candidate funnel, and the training set")
    rng = np.random.default_rng(0)
    X, G, y = [], [], []
    tot = dict()
    t0 = time.time()
    for s in range(N_TRAIN_SCENES):
        sc = build_scene(rng, n=int(rng.integers(2, 5)))
        if sc is None:
            continue
        cands, funnel = grasps.sample_candidates(sc["P"], sc["N"], rng,
                                                 n_points=45, n_angles=6)
        for k, v in funnel.items():
            tot[k] = tot.get(k, 0) + v
        if cands:
            for k in rng.choice(len(cands), size=min(N_EXEC, len(cands)),
                                replace=False):
                g = cands[k]
                pick.restore(sc["model"], sc["data"], sc["snap"])
                r = pick.execute(sc["model"], sc["data"], g)
                x, gl = gnet.encode(g, rng)
                X.append(x)
                G.append(gl)
                y.append(r["success"])
        close_scene(sc)
        if (s + 1) % 60 == 0:
            print(f"    {s + 1}/{N_TRAIN_SCENES} scenes, {len(y)} grasps executed, "
                  f"{time.time() - t0:.0f}s")
    X = np.stack(X)
    G = np.stack(G)
    y = np.array(y, bool)
    record("2_dataset", "scenes", N_TRAIN_SCENES)
    record("2_dataset", "grasps executed in simulation", len(y))
    record("2_dataset", "of which succeeded", f"{y.mean():.3f}")
    record("2_dataset", "dataset time (s)", round(time.time() - t0))
    for k in ("sampled", "approach_too_flat", "empty", "too_wide",
              "finger_collision", "below_table", "kept"):
        record("2_dataset", f"funnel: {k}",
               f"{tot.get(k, 0)} ({100 * tot.get(k, 0) / max(tot['sampled'], 1):.1f}%)")

    net, curve = gnet.train(X, G, y, log=lambda s_: print(s_))
    state["net"] = net
    state["y"] = y

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    keys = ["approach_too_flat", "empty", "too_wide", "finger_collision",
            "below_table", "kept"]
    vals = [100 * tot.get(k, 0) / max(tot["sampled"], 1) for k in keys]
    axes[0].barh(["approach too flat", "nothing between\nthe fingers",
                  "too wide", "fingers would\nhit something",
                  "below the table", "KEPT"], vals,
                 color=[COLORS[6]] * 5 + [COLORS[2]])
    axes[0].set_xlabel("% of sampled poses")
    axes[0].set_title(f"{tot['sampled']} poses sampled", fontsize=9)
    axes[1].plot(range(1, len(curve) + 1), curve, "o-", ms=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("weighted BCE loss")
    axes[1].set_title(f"{len(y)} executed grasps, {100 * y.mean():.0f}% succeed",
                      fontsize=9)
    save(fig, os.path.join(OUT, "funnel.png"))


# ---------------------------------------------------------------------------
# 3. four scorers
# ---------------------------------------------------------------------------

def scorers(state, rng):
    net = state["net"]
    return {
        "random": lambda sc, c: rng.random(len(c)),
        "antipodal test (project 39, in 3D)":
            lambda sc, c: np.array([grasps.antipodal_score(g, sc["N"], sc["P"])
                                    for g in c]),
        "most points between the fingers": lambda sc, c: point_count_score(c),
        "learned PointNet": lambda sc, c: gnet.score(net, c, rng),
    }


def evaluate(state, n_scenes, seed, n=3, kinds=None, noise=0.0):
    rng = np.random.default_rng(seed)
    rnd = np.random.default_rng(seed + 500)
    meth = scorers(state, rnd)
    hits = {k: 0 for k in meth}
    other = {k: 0 for k in meth}
    tot = 0
    for _ in range(n_scenes):
        sc = build_scene(rng, n=n, kinds=kinds, noise=noise)
        if sc is None:
            continue
        cands, _ = grasps.sample_candidates(sc["P"], sc["N"], rng)
        if not cands:
            close_scene(sc)
            continue
        tot += 1
        for name, fn in meth.items():
            r = run_top(sc, cands, fn(sc, cands))
            hits[name] += r["success"]
        close_scene(sc)
    return {k: hits[k] / max(tot, 1) for k in meth}, tot


def exp3_compare(state):
    print("\n[3] four scorers")
    rates, tot = evaluate(state, N_EVAL, seed=101)
    for k, v in rates.items():
        record("3_compare", f"{k}: top-1 success", round(v, 4))
    record("3_compare", "scenes evaluated", tot)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ks = list(rates)
    ax.barh(ks, [100 * rates[k] for k in ks],
            color=[COLORS[6], COLORS[0], COLORS[4], COLORS[2]])
    ax.set_xlabel("top-1 grasp success (%), lifted AND survived a shake")
    save(fig, os.path.join(OUT, "compare.png"))
    state["rates"] = rates


# ---------------------------------------------------------------------------
# 4. why the antipodal test struggles
# ---------------------------------------------------------------------------

def exp4_visibility(state):
    """How well does each signal RANK grasps, and what does one camera cost?"""
    print("\n[4] what each score actually knows")
    rng = np.random.default_rng(202)
    rows = []
    for _ in range(45):
        sc = build_scene(rng, n=3)
        if sc is None:
            continue
        cands, _ = grasps.sample_candidates(sc["P"], sc["N"], rng, n_points=30)
        if not cands:
            close_scene(sc)
            continue
        learned = gnet.score(state["net"], cands, rng)
        cam_pos = sc["data"].cam_xpos[sc["cam"].cid]
        for k in rng.choice(len(cands), size=min(7, len(cands)), replace=False):
            g = cands[k]
            view = g["pos"] - cam_pos
            view = view / np.linalg.norm(view)
            # If the fingers close ALONG the line of sight, one of the two
            # contact faces is directly behind the object and the camera never
            # saw it -- so its "measured" normal is invented from the points
            # of some other surface.  This number is how much of that is going
            # on, from 0 (the fingers close across the view, both faces at
            # least partly visible) to 1 (one face completely hidden).
            align = float(abs(g["closing"] @ view))
            pick.restore(sc["model"], sc["data"], sc["snap"])
            r = pick.execute(sc["model"], sc["data"], g)
            rows.append((grasps.antipodal_score(g, sc["N"], sc["P"]),
                         float(g["n_inside"]), float(learned[k]),
                         -g["width"], align, float(r["success"])))
        close_scene(sc)
    A = np.array(rows, float)

    def auc(x, y):
        p, n = x[y > 0.5], x[y < 0.5]
        if len(p) == 0 or len(n) == 0:
            return np.nan
        return float((p[:, None] > n[None, :]).mean())

    names = ["antipodal test (project 39, in 3D)", "points between the fingers",
             "learned PointNet", "narrower is better"]
    aucs = [auc(A[:, i], A[:, 5]) for i in range(4)]
    for nm, a in zip(names, aucs):
        record("4_scores", f"AUC of {nm}", round(a, 3))
    record("4_scores", "grasps executed", len(A))
    record("4_scores", "of which succeeded", round(float(A[:, 5].mean()), 3))

    med = np.median(A[:, 4])
    lo = A[A[:, 4] <= med]
    hi = A[A[:, 4] > med]
    record("4_scores", "antipodal AUC, fingers close ACROSS the view",
           round(auc(lo[:, 0], lo[:, 5]), 3))
    record("4_scores", "antipodal AUC, fingers close ALONG the view "
                       "(far face hidden)", round(auc(hi[:, 0], hi[:, 5]), 3))
    record("4_scores", "median view alignment", round(float(med), 3))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.3))
    axes[0].barh(names, aucs, color=[COLORS[0], COLORS[4], COLORS[2], COLORS[6]])
    axes[0].axvline(0.5, color="#42505e", ls="--", lw=1)
    axes[0].text(0.505, -0.4, "chance", fontsize=8)
    axes[0].set_xlabel("AUC: does a good grasp outrank a bad one?")
    axes[0].set_xlim(0, 1)
    axes[0].tick_params(labelsize=8)
    axes[1].bar(["close ACROSS\nthe view", "close ALONG\nthe view"],
                [auc(lo[:, 0], lo[:, 5]), auc(hi[:, 0], hi[:, 5])], color=COLORS[0])
    axes[1].axhline(0.5, color="#42505e", ls="--", lw=1)
    axes[1].set_ylabel("AUC of the antipodal test")
    axes[1].set_ylim(0, 1)
    save(fig, os.path.join(OUT, "visibility.png"))


# ---------------------------------------------------------------------------
# 5. the collision filter
# ---------------------------------------------------------------------------

def exp5_collision(state):
    print("\n[5] the collision filter")
    rng = np.random.default_rng(303)
    rnd = np.random.default_rng(304)
    net = state["net"]
    res = {}
    for tag, use_filter in (("with the filter", True), ("without it", False)):
        hits = knocked = tot = 0
        r2 = np.random.default_rng(303)
        for _ in range(55):
            sc = build_scene(r2, n=4)
            if sc is None:
                continue
            cands, _ = grasps.sample_candidates(sc["P"], sc["N"], r2)
            if not use_filter:
                extra = _no_filter_candidates(sc, r2)
                cands = cands + extra
            if not cands:
                close_scene(sc)
                continue
            tot += 1
            s = gnet.score(net, cands, rnd)
            k = int(np.argmax(s))
            pick.restore(sc["model"], sc["data"], sc["snap"])
            before = sc["data"].xipos[pick._obj_body_ids(sc["model"])].copy()
            r = pick.execute(sc["model"], sc["data"], cands[k])
            after = sc["data"].xipos[pick._obj_body_ids(sc["model"])]
            moved = np.linalg.norm(after - before, axis=1) > 0.01
            if r["lifted"] >= 0:
                moved[r["lifted"]] = False
            hits += r["success"]
            knocked += bool(moved.any())
            close_scene(sc)
        res[tag] = (hits / max(tot, 1), knocked / max(tot, 1))
        record("5_collision", f"{tag}: top-1 success", round(res[tag][0], 4))
        record("5_collision", f"{tag}: disturbed a neighbouring object",
               round(res[tag][1], 4))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(2)
    ax.bar(x - 0.18, [100 * res[k][0] for k in res], 0.34, label="grasp succeeded",
           color=COLORS[2])
    ax.bar(x + 0.18, [100 * res[k][1] for k in res], 0.34,
           label="knocked a neighbour over", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(list(res))
    ax.set_ylabel("% of scenes")
    ax.legend(fontsize=8)
    ax.set_title("four objects on the table", fontsize=9)
    save(fig, os.path.join(OUT, "collision.png"))


def _no_filter_candidates(sc, rng):
    """The same generator with the collision check switched off."""
    saved = grasps.finger_collision
    grasps.finger_collision = lambda *a, **k: False
    try:
        c, _ = grasps.sample_candidates(sc["P"], sc["N"], rng)
    finally:
        grasps.finger_collision = saved
    return c


# ---------------------------------------------------------------------------
# 6. clutter
# ---------------------------------------------------------------------------

def exp6_clutter(state):
    print("\n[6] clutter")
    counts = [1, 2, 3, 4, 5]
    curves = {}
    for n in counts:
        rates, tot = evaluate(state, 40, seed=400 + n, n=n)
        for k, v in rates.items():
            curves.setdefault(k, []).append(v)
            record("6_clutter", f"{n} objects, {k}", round(v, 4))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for k, v in curves.items():
        ax.plot(counts, [100 * x for x in v], "o-", ms=4, label=k)
    ax.set_xlabel("objects on the table")
    ax.set_ylabel("top-1 grasp success (%)")
    ax.set_xticks(counts)
    ax.legend(fontsize=7.5)
    save(fig, os.path.join(OUT, "clutter.png"))


# ---------------------------------------------------------------------------
# 7. novel shapes and a noisy camera
# ---------------------------------------------------------------------------

def exp7_transfer(state):
    print("\n[7] novel shapes and depth noise")
    novel, _ = evaluate(state, 55, seed=700, kinds=pick.KINDS_TEST)
    for k, v in novel.items():
        record("7_transfer", f"novel shapes (L and T): {k}", round(v, 4))
    levels = [0.0, 0.001, 0.003, 0.006]
    curves = {}
    for lv in levels:
        rates, _ = evaluate(state, 40, seed=800, noise=lv)
        for k, v in rates.items():
            curves.setdefault(k, []).append(v)
            record("7_transfer", f"depth noise {1000 * lv:.0f} mm: {k}", round(v, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
    ks = list(novel)
    axes[0].barh(ks, [100 * novel[k] for k in ks],
                 color=[COLORS[6], COLORS[0], COLORS[4], COLORS[2]])
    axes[0].set_xlabel("top-1 success (%) on shapes never trained on")
    for k, v in curves.items():
        axes[1].plot([1000 * x for x in levels], [100 * y for y in v], "o-", ms=4,
                     label=k)
    axes[1].set_xlabel("depth noise (mm)")
    axes[1].set_ylabel("top-1 success (%)")
    axes[1].legend(fontsize=7)
    save(fig, os.path.join(OUT, "transfer.png"))


def main():
    use_style()
    t0 = time.time()
    state = {}
    exp1_pipeline(state)
    exp2_dataset(state)
    exp3_compare(state)
    exp4_visibility(state)
    exp5_collision(state)
    exp6_clutter(state)
    exp7_transfer(state)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
