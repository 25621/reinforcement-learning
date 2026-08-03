"""Project 49 -- AMCL: localizing on a map you already have, and paying for it.

Six experiments:
  1. one patrol lap: the cloud, the error, and the particle count over time
  2. KLD-sampling vs a fixed particle count -- the same accuracy, far cheaper
  3. global localization, and a map where it cannot work
  4. update thresholds: doing nothing while standing still
  5. closing the loop -- drive on the estimate, not on the truth
  6. kidnapping, and why an adaptive count is also an alarm
"""

import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "27-particle-filter"))
sys.path.insert(0, os.path.join(_PROJ, "46-pure-pursuit"))
sys.path.insert(0, os.path.join(_PROJ, "47-dwa-local-planner"))
sys.path.insert(0, os.path.join(_PROJ, "31-a-star-on-a-grid"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from amcl import AMCL, kld_bound, office_map, pose_error, scan   # noqa: E402
from gridmap import free_poses                                   # noqa: E402
from robot import DiffDrive, Path, pure_pursuit, wrap            # noqa: E402
from dwa import Costmap, astar_path                              # noqa: E402
from grid import search as grid_search                           # noqa: E402
from plot_style import COLORS, use_style, save                   # noqa: E402

import matplotlib.pyplot as plt                                  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []

N_BEAMS = 12
ANGLES = np.linspace(-math.pi, math.pi, N_BEAMS, endpoint=False)
DT = 0.1
WAYPOINTS = [(2.0, 3.5), (8.5, 3.5), (13.0, 3.5), (13.0, 10.5),
             (7.0, 7.0), (2.0, 10.5), (2.0, 3.5)]


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def build_route(gmap):
    """A patrol loop, planned once with A* on the inflated map."""
    cmap = Costmap(gmap.occ, res=gmap.res, robot_radius=0.35)
    legs = []
    for a, b in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        p = astar_path(cmap, np.array(a), np.array(b), grid_search)
        if p is None:
            raise RuntimeError(f"no route {a} -> {b}")
        legs.append(p if not legs else p[1:])
    return Path(np.vstack(legs), spacing=0.05, closed=True), cmap


# ------------------------------------------------------------------ episode
def run_episode(gmap, route, cmap, seed=0, steps=520, eps=0.05, n0=3000,
                fixed_n=None, d_thresh=0.2, drive_on="true", global_init=False,
                kidnap_at=None, augmented=False, inject=0.0, n_max=5000,
                keep_snapshots=(), v=0.7, L=0.9, n_min=50):
    """Drive the patrol loop while AMCL localizes; return the whole history.

    `drive_on` picks which pose the path tracker is given:
      "true"  -- cheating, the reference
      "amcl"  -- the real system
      "odom"  -- dead reckoning only, the control that shows what AMCL buys
    """
    rng = np.random.default_rng(seed)
    p0 = route.pts[0]
    th0 = math.atan2(route.pts[1][1] - p0[1], route.pts[1][0] - p0[0])
    rb = DiffDrive(p0[0], p0[1], th0, v_max=1.0, w_max=2.0, a_max=1.5,
                   alpha_max=3.0, rng=rng, slip_sigma=0.05)

    init = None
    if not global_init:
        # Local start: a tight cloud around the true pose, which is what a
        # real robot gets when you tell it roughly where it is.
        n_start = fixed_n if fixed_n else 500
        init = np.column_stack([
            rng.normal(p0[0], 0.3, n_start), rng.normal(p0[1], 0.3, n_start),
            rng.normal(th0, 0.15, n_start)])
    f = AMCL(gmap, ANGLES, n0=(fixed_n or n0) if global_init else 0,
             eps=(None if fixed_n else eps), rng=rng, d_thresh=d_thresh,
             sigma_z=0.3, n_max=n_max, init=init, augmented=augmented,
             inject=inject, n_min=n_min)

    i_hint = 0
    hist = {"err": [], "aerr": [], "n": [], "t": [], "true": [], "est": [],
            "odom_err": [], "xtrack": [], "snap": {}}
    t_pf = 0.0
    collided = 0

    for k in range(steps):
        true = rb.state
        odom = np.array([rb.ox, rb.oy, rb.oth])
        est = f.estimate()
        if drive_on == "true":
            pose = true
        elif drive_on == "odom":
            pose = odom
        else:
            pose = est
        i_hint = route.closest(pose[:2], hint=i_hint, window=300)
        _, w_cmd, _ = pure_pursuit(pose, route, i_hint, L, v)
        rb.step(v, w_cmd, DT)

        u = (rb.v, rb.w)                       # odometry: the COMMANDED motion
        t0 = time.perf_counter()
        f.predict(u, DT)
        if kidnap_at is not None and k == kidnap_at:
            # Pick the robot up and put it somewhere else, without telling
            # the filter.  The classic test: can the belief ever be wrong in
            # a way that resampling alone cannot fix?
            newp = free_poses(gmap, 1, rng)[0]
            rb.x, rb.y, rb.th = newp
        f.update(scan(gmap, rb.state, ANGLES, rng, sigma=0.05))
        t_pf += time.perf_counter() - t0

        e, ae = pose_error(f.estimate(), rb.state)
        oe, _ = pose_error(np.array([rb.ox, rb.oy, rb.oth]), rb.state)
        _, xt = route.cross_track(rb.state[:2], rb.state[2],
                                  route.closest(rb.state[:2]))
        hist["err"].append(e); hist["aerr"].append(ae); hist["n"].append(f.n)
        hist["odom_err"].append(oe); hist["true"].append(rb.state.copy())
        hist["est"].append(f.estimate()); hist["xtrack"].append(xt)
        if k in keep_snapshots:
            hist["snap"][k] = (f.parts.copy(), rb.state.copy())
        if float(cmap.clearance(rb.state[:2])) < 0.20:
            collided += 1

    for key in ("err", "aerr", "n", "odom_err", "xtrack"):
        hist[key] = np.asarray(hist[key])
    hist["true"] = np.asarray(hist["true"])
    hist["est"] = np.asarray(hist["est"])
    hist["ms_per_step"] = 1000.0 * t_pf / steps
    hist["updates"] = f.n_updates
    hist["skipped"] = f.n_skipped
    hist["collisions"] = collided
    return hist


def show_map(ax, gmap):
    ax.imshow(np.where(gmap.occ, 0.35, 1.0), cmap="gray", vmin=0, vmax=1,
              origin="lower", extent=list(gmap.extent()))
    ax.set_aspect("equal"); ax.grid(False)


def conv_step(err, tol=0.5, hold=5):
    """First step after which the error stays under `tol`."""
    ok = err < tol
    for i in range(len(ok) - hold):
        if ok[i:i + hold].all():
            return i
    return None


# ================================================================= 1. a lap
def exp1(gmap, route, cmap):
    print("[1] one patrol lap")
    snaps = (0, 6, 20, 120)
    h = run_episode(gmap, route, cmap, seed=0, global_init=True,
                    keep_snapshots=snaps, d_thresh=0.2, n_min=500, n0=5000)
    fig = plt.figure(figsize=(14, 6.2))
    for i, k in enumerate(snaps):
        ax = fig.add_subplot(2, 4, i + 1)
        show_map(ax, gmap)
        parts, tr = h["snap"][k]
        ax.plot(parts[:, 0], parts[:, 1], ".", color=COLORS[0], ms=1.2,
                alpha=0.5)
        ax.plot(tr[0], tr[1], "*", color=COLORS[1], ms=13)
        ax.set_title(f"step {k}: {len(parts)} particles", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 2, 3)
    show_map(ax, gmap)
    ax.plot(route.pts[:, 0], route.pts[:, 1], "--", color="0.6", lw=1.0)
    ax.plot(h["true"][:, 0], h["true"][:, 1], color=COLORS[0], lw=1.4,
            label="true")
    ax.plot(h["est"][:, 0], h["est"][:, 1], color=COLORS[1], lw=1.0,
            label="AMCL estimate")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("the patrol loop")
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(np.arange(len(h["err"])) * DT, h["err"], color=COLORS[0],
            label="AMCL position error")
    ax.plot(np.arange(len(h["err"])) * DT, h["odom_err"], color=COLORS[1],
            label="odometry-only error")
    ax.set_yscale("log"); ax.set_xlabel("t [s]"); ax.set_ylabel("error [m]")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(h["n"])) * DT, h["n"], color=COLORS[2], lw=1.0)
    ax2.set_ylabel("particles", color=COLORS[2]); ax2.grid(False)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("error (left axis) and particle count (right)")
    save(fig, os.path.join(OUT, "overview.png"))
    tail = slice(len(h["err"]) // 4, None)
    rec("1_lap", start_particles=int(h["n"][0]),
        end_particles=int(h["n"][-1]),
        median_particles=int(np.median(h["n"])),
        converge_s=round(conv_step(h["err"]) * DT, 1)
        if conv_step(h["err"]) is not None else None,
        settled_err_m=round(float(np.median(h["err"][tail])), 3),
        settled_heading_deg=round(float(np.degrees(np.median(h["aerr"][tail]))), 2),
        odom_final_err_m=round(float(h["odom_err"][-1]), 2),
        ms_per_step=round(h["ms_per_step"], 2),
        updates=h["updates"], skipped=h["skipped"])


# ================================================= 2. KLD vs a fixed count
def exp2(gmap, route, cmap):
    print("[2] KLD-sampling vs a fixed particle count (tracking regime)")
    seeds = range(5)
    fixed = [50, 100, 250, 500, 1000, 2500, 5000]
    epss = [0.02, 0.05, 0.10, 0.20, 0.40]
    pts_f, pts_k = [], []
    for n in fixed:
        e, nn, ms, cv = [], [], [], []
        for s in seeds:
            # Local start on purpose.  KLD's claim is about what you carry
            # while TRACKING a pose you already have; mixing in the global
            # search would measure a different thing and hide this one.
            h = run_episode(gmap, route, cmap, seed=s, fixed_n=n)
            tail = slice(len(h["err"]) // 4, None)
            e.append(np.median(h["err"][tail])); nn.append(np.mean(h["n"]))
            ms.append(h["ms_per_step"])
            cv.append(float(np.max(h["err"])))
        pts_f.append((np.mean(nn), np.mean(e), np.mean(ms)))
        rec("2_fixed", particles=n, mean_particles=round(float(np.mean(nn)), 0),
            median_err_m=round(float(np.mean(e)), 3),
            ms_per_step=round(float(np.mean(ms)), 2),
            worst_err_m=round(float(np.mean(cv)), 3))
    for ep in epss:
        e, nn, ms, cv = [], [], [], []
        for s in seeds:
            h = run_episode(gmap, route, cmap, seed=s, eps=ep, n0=5000)
            tail = slice(len(h["err"]) // 4, None)
            e.append(np.median(h["err"][tail])); nn.append(np.mean(h["n"]))
            ms.append(h["ms_per_step"])
            cv.append(float(np.max(h["err"])))
        pts_k.append((np.mean(nn), np.mean(e), np.mean(ms)))
        rec("2_kld", eps=ep, mean_particles=round(float(np.mean(nn)), 0),
            median_err_m=round(float(np.mean(e)), 3),
            ms_per_step=round(float(np.mean(ms)), 2),
            worst_err_m=round(float(np.mean(cv)), 3))

    pts_f, pts_k = np.asarray(pts_f), np.asarray(pts_k)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    axes[0].plot(pts_f[:, 0], pts_f[:, 1], "o-", color=COLORS[1],
                 label="fixed N")
    axes[0].plot(pts_k[:, 0], pts_k[:, 1], "s-", color=COLORS[0],
                 label="KLD-sampling")
    for x, y, n in zip(pts_f[:, 0], pts_f[:, 1], fixed):
        axes[0].annotate(str(n), (x, y), fontsize=6)
    for x, y, n in zip(pts_k[:, 0], pts_k[:, 1], epss):
        axes[0].annotate(f"eps={n}", (x, y), fontsize=6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("mean particles carried"); axes[0].set_ylabel("error [m]")
    axes[0].set_title("the same accuracy, fewer particles"); axes[0].legend()
    axes[1].plot(pts_f[:, 2], pts_f[:, 1], "o-", color=COLORS[1],
                 label="fixed N")
    axes[1].plot(pts_k[:, 2], pts_k[:, 1], "s-", color=COLORS[0],
                 label="KLD-sampling")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("ms per step"); axes[1].set_ylabel("error [m]")
    axes[1].set_title("the same picture in real time"); axes[1].legend()
    ks = np.arange(1, 600)
    for ep, c in zip([0.02, 0.05, 0.2], COLORS):
        axes[2].plot(ks, [kld_bound(k, ep) for k in ks], color=c,
                     label=f"eps = {ep}")
    axes[2].set_xlabel("occupied histogram cells k")
    axes[2].set_ylabel("particles the bound asks for")
    axes[2].set_yscale("log"); axes[2].legend()
    axes[2].set_title("the KLD bound itself")
    save(fig, os.path.join(OUT, "kld.png"))


# ================================================= 3. global localization
def exp3(gmap, route, cmap):
    print("[3] global localization: the collapse floor, and an ambiguous map")
    sym = office_map(symmetric=True)
    route_s, cmap_s = build_route(sym)
    # n_min is the floor KLD-sampling is never allowed to go below.  It looks
    # like a safety detail and it is actually the whole ball game during
    # global localization: the FIRST scan makes the weights extremely peaked,
    # so if the filter is allowed to shrink to 50 particles right then, every
    # hypothesis except one dies -- and if the survivor is the wrong one, no
    # amount of further evidence brings the right one back.  Resampling can
    # only ever copy particles that already exist.
    floors = [50, 100, 250, 500, 1000, 2000]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for g, rt, cm, c, lab in [(gmap, route, cmap, COLORS[0], "irregular map"),
                              (sym, route_s, cmap_s, COLORS[1],
                               "left-right symmetric map")]:
        rates, mirrors = [], []
        for fl in floors:
            ok, mirror, ms = 0, 0, []
            for s in range(16):
                h = run_episode(g, rt, cm, seed=s, global_init=True, eps=0.05,
                                n0=5000, n_min=fl)
                c_ = conv_step(h["err"])
                ok += int(c_ is not None)
                ms.append(h["ms_per_step"])
                # A run that never converges has settled CONFIDENTLY on a
                # wrong pose -- that is the failure worth counting, because a
                # filter that is merely uncertain still recovers, while one
                # that is certain and wrong does not.
                if c_ is None:
                    mirror += int(float(np.median(h["err"][-100:])) > 2.0)
            rates.append(ok / 16); mirrors.append(mirror / 16)
            rec("3_global", map=lab, n_min=fl, n=16, converged=ok,
                rate=round(ok / 16, 3),
                confidently_wrong_rate=round(mirror / 16, 3),
                ms_per_step=round(float(np.mean(ms)), 2))
        axes[0].plot(floors, rates, "o-", color=c, label=lab)
        axes[1].plot(floors, mirrors, "o-", color=c, label=lab)
    axes[0].set_ylabel("global localization success rate")
    axes[1].set_ylabel("fraction that settled confidently on a wrong pose")
    for ax in axes[:2]:
        ax.set_xscale("log"); ax.set_xlabel("particle floor n_min")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
    show_map(axes[2], sym)
    axes[2].set_title("the symmetric map: two identical alcoves")
    save(fig, os.path.join(OUT, "global.png"))


# ================================================= 4. update thresholds
def exp4(gmap, route, cmap):
    print("[4] update thresholds")
    ths = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    res = []
    for d in ths:
        e, ms, up = [], [], []
        for s in range(5):
            h = run_episode(gmap, route, cmap, seed=s, d_thresh=d, eps=0.05)
            tail = slice(len(h["err"]) // 4, None)
            e.append(np.median(h["err"][tail])); ms.append(h["ms_per_step"])
            up.append(h["updates"])
        res.append((d, np.mean(e), np.mean(ms), np.mean(up)))
        rec("4_thresh", d_thresh=d, median_err_m=round(float(np.mean(e)), 4),
            ms_per_step=round(float(np.mean(ms)), 3),
            updates=round(float(np.mean(up)), 1),
            skipped_frac=round(1 - float(np.mean(up)) / 520, 3))
    res = np.asarray(res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(res[:, 0], res[:, 1], "o-", color=COLORS[0])
    axes[0].set_ylabel("median position error [m]")
    axes[1].plot(res[:, 0], res[:, 2], "o-", color=COLORS[1])
    axes[1].set_ylabel("ms per control step")
    for ax in axes:
        ax.set_xlabel("update threshold [m of travel]")
    axes[0].set_title("accuracy"); axes[1].set_title("cost")
    save(fig, os.path.join(OUT, "thresholds.png"))


# ================================================= 5. closing the loop
def exp5(gmap, route, cmap):
    print("[5] driving on the estimate")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (mode, lab) in zip(axes, [("true", "true pose (cheating)"),
                                      ("amcl", "AMCL estimate"),
                                      ("odom", "odometry only")]):
        xt, coll, fin = [], [], []
        for s in range(5):
            # Three laps, not one.  Over 44 m the dead-reckoning error is
            # still small, so a one-lap test would report that odometry is
            # fine -- which is true for one lap and false for a shift.
            h = run_episode(gmap, route, cmap, seed=s, drive_on=mode,
                            eps=0.05, steps=1900, n_min=100)
            xt.append(float(np.mean(np.abs(h["xtrack"]))))
            coll.append(h["collisions"])
            fin.append(float(h["odom_err"][-1]))
            if s == 0:
                show_map(ax, gmap)
                ax.plot(route.pts[:, 0], route.pts[:, 1], "--", color="0.6",
                        lw=1.0)
                ax.plot(h["true"][:, 0], h["true"][:, 1], color=COLORS[0],
                        lw=1.4)
        rec("5_closed_loop", drive_on=lab, laps=3,
            mean_abs_xtrack_m=round(float(np.mean(xt)), 3),
            worst_seed_xtrack_m=round(float(np.max(xt)), 3),
            steps_in_collision=round(float(np.mean(coll)), 1),
            final_odometry_drift_m=round(float(np.mean(fin)), 2))
        ax.set_title(f"{lab}\nmean |cross-track| = {np.mean(xt):.2f} m",
                     fontsize=9)
    save(fig, os.path.join(OUT, "closed_loop.png"))


# ================================================= 6. kidnapping
def exp6(gmap, route, cmap):
    print("[6] kidnapping")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    arms = [(False, 0.0, COLORS[1], "no injection"),
            (False, 0.05, COLORS[2], "5% injected every update"),
            (True, 0.0, COLORS[0], "augmented MCL (self-tuning)")]
    for aug, inj, c, lab in arms:
        rec_err, rec_n, ok = [], [], 0
        for s in range(6):
            h = run_episode(gmap, route, cmap, seed=s, kidnap_at=200,
                            eps=0.05, augmented=aug, inject=inj, n_min=100)
            after = h["err"][200:]
            c_ = conv_step(after)
            ok += int(c_ is not None)
            rec_err.append(np.nan if c_ is None else c_ * DT)
            if s == 0:
                axes[0].plot(np.arange(len(h["err"])) * DT, h["err"],
                             color=c, lw=1.1, label=lab)
                axes[1].plot(np.arange(len(h["n"])) * DT, h["n"], color=c,
                             lw=1.1, label=lab)
            rec_n.append(float(np.mean(h["n"][200:260])))
        rec("6_kidnap", recovery=lab, n=6, recovered=ok, rate=round(ok / 6, 3),
            mean_recovery_s=round(float(np.nanmean(rec_err)), 1) if ok else None,
            mean_particles_after=round(float(np.mean(rec_n)), 0))
    axes[0].axvline(200 * DT, color="0.4", ls=":", lw=1.0)
    axes[1].axvline(200 * DT, color="0.4", ls=":", lw=1.0)
    axes[0].set_yscale("log"); axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("particles")
    for ax in axes:
        ax.set_xlabel("t [s]"); ax.legend(fontsize=7)
    axes[0].set_title("dotted = the robot is picked up and moved")
    axes[1].set_title("KLD raises N by itself when the belief spreads")
    save(fig, os.path.join(OUT, "kidnap.png"))


if __name__ == "__main__":
    t0 = time.time()
    gmap = office_map()
    route, cmap = build_route(gmap)
    print(f"route {route.length:.1f} m")
    exp1(gmap, route, cmap)
    exp2(gmap, route, cmap)
    exp3(gmap, route, cmap)
    exp4(gmap, route, cmap)
    exp5(gmap, route, cmap)
    exp6(gmap, route, cmap)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s")
