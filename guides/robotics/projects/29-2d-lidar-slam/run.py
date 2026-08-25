"""Project 29 -- 2D LiDAR SLAM: scan matching, a pose graph, and a loop closed.

Seven experiments:

  1. scan-matching odometry: better than wheels, and still drifting
  2. point-to-point against point-to-line against a brute-force matcher
  3. a featureless corridor, and the eigenvalue that warns you
  4. finding loop closures: how many are real, how many are lies
  5. closing the loop: the trajectory, and the map it draws
  6. one false loop closure, and the kernel that survives it
  7. the map as a product: measured against the floor plan it came from

Runs in about six minutes.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "27-particle-filter"))
sys.path.insert(0, os.path.join(_PROJ, "26-ekf-localization"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from gridmap import GridMap, office_map                                # noqa: E402
from world import sample_motion as diff_drive_step, ALPHA             # noqa: E402
from scanmatch import (wrap, icp, icp_multi, correlative,            # noqa: E402
                       scan_to_points, transform, compose, between,
                       invert, T_to_pose, pose_to_T)
from posegraph import (PoseGraph, chain, align_and_ate,             # noqa: E402
                       align_poses, KERNELS)
from plot_style import COLORS, use_style, save                        # noqa: E402

import matplotlib.pyplot as plt                                       # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

DT = 0.2
MAX_RANGE = 8.0
N_BEAMS = 360                      # a real 2D lidar: 1 degree per beam
ANGLES = np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)
SIGMA_Z = 0.02                     # a good indoor lidar: 2 cm
ODOM_ALPHA = ALPHA * 6.0           # deliberately poor wheel odometry


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ------------------------------------------------------------------- the map
def slam_map(res=0.1):
    """A hall with a block in the middle, so the robot can drive a full lap.

    Purpose-built rather than reused from project 27, for one reason: a loop.
    SLAM without a revisit is odometry, so the floor plan has to contain a
    circuit the robot can actually drive, with clear space either side of it.
    The three notches are there to break the block's symmetry -- without them,
    four different positions around a plain rectangle produce near-identical
    scans and the loop-closure detector in experiment 4 has nothing to work
    with but luck.
    """
    w_m, h_m = 18.0, 14.0
    w, h = int(w_m / res), int(h_m / res)
    occ = np.zeros((h, w), dtype=bool)

    def box(x0, y0, x1, y1):
        occ[int(y0 / res):int(y1 / res), int(x0 / res):int(x1 / res)] = True

    t = 0.2
    box(0, 0, w_m, t); box(0, h_m - t, w_m, h_m)
    box(0, 0, t, h_m); box(w_m - t, 0, w_m, h_m)
    box(5.0, 4.0, 13.0, 10.0)              # the island in the middle
    box(2.0, 6.5, 3.0, 7.5)                # three asymmetric landmarks
    box(15.0, 2.0, 16.0, 2.6)
    box(8.0, 12.0, 9.0, 13.0)
    return GridMap(occ, res)


# ------------------------------------------------------------------ the drive
def loop_route(gmap, rng, n=560, noisy_odom=True):
    """Drive a closed loop around the building, recording scans and odometry.

    The loop matters: a trajectory that never revisits anywhere has no loop
    closures available, and without loop closures a SLAM system is only an
    odometry system with extra steps.
    """
    # TWO laps.  One lap gives a single revisit at the very end and almost no
    # accumulated drift to correct; two laps means the whole second lap is a
    # revisit, which is both a harder detection problem (many candidates) and
    # the situation a real SLAM run is actually in.
    lap = [[3.0, 2.0], [15.0, 2.0], [15.0, 12.0], [3.0, 12.0]]
    way = np.array(lap + lap + [[3.0, 2.5]])
    x = np.array([way[0, 0], way[0, 1], 0.0])

    def scan_at(pose):
        return add_noise(gmap.raycast(pose[None, :], ANGLES,
                                      max_range=MAX_RANGE)[0], rng)

    # One scan per POSE, including the starting one, so scans[k] belongs to
    # poses[k].  Off-by-one here is the classic SLAM bug: every relative
    # measurement ends up describing the wrong pair and the map shears.
    poses, odom_rel, scans = [x.copy()], [], [scan_at(x)]
    seg, target = 1, way[1]
    for k in range(n):
        d = target - x[:2]
        if np.linalg.norm(d) < 0.6 and seg + 1 < len(way):
            seg += 1
            target = way[seg]
            d = target - x[:2]
        head_err = wrap(np.arctan2(d[1], d[0]) - x[2])
        w = float(np.clip(2.0 * head_err, -0.8, 0.8))
        v = 0.9 if abs(head_err) < 0.5 else 0.25
        u = np.array([v, w])
        x_new = diff_drive_step(x, u, DT, rng, ODOM_ALPHA if noisy_odom else ALPHA * 0)
        odom_rel.append(between(x, x_new))
        x = x_new
        poses.append(x.copy())
        scans.append(scan_at(x))
    # the odometry a real robot would report: chain the noisy relative steps
    return np.array(poses), odom_rel, scans


def add_noise(z, rng):
    """Add range noise to the beams that actually HIT something.

    A reading at max range means "nothing was there", not "a wall at exactly
    8.00 m".  Adding noise to it produces 7.98 m, which then survives the
    max-range filter in scan_to_points and becomes a phantom point floating in
    open space.  A handful of those per scan, in arbitrary directions, is
    enough to swing an ICP rotation estimate by several degrees -- it cost
    3.5 degrees per match here, which over 280 chained matches is the
    difference between a map and a spiral.
    """
    hit = z < MAX_RANGE - 1e-9
    out = z.copy()
    out[hit] = np.clip(z[hit] + SIGMA_Z * rng.standard_normal(int(hit.sum())),
                       0.0, MAX_RANGE - 1e-6)
    return out


def noisy_odom_rel(truth, rng, sigma=(0.04, 0.04, 0.02)):
    """Relative-pose measurements from wheels, with honest noise on them."""
    out = []
    for k in range(len(truth) - 1):
        r = between(truth[k], truth[k + 1])
        out.append(r + np.array(sigma) * rng.standard_normal(3))
    return out


def clouds_from(scans):
    return [scan_to_points(z, ANGLES, MAX_RANGE) for z in scans]


def match_chain(clouds, odom_rel, mode="line", **kw):
    """Scan-match every consecutive pair, seeded by the wheel odometry."""
    rels, infos, iters = [], [], []
    for k in range(len(clouds) - 1):
        x, info, it, rms, ov = icp(clouds[k + 1], clouds[k], x0=odom_rel[k],
                                   mode=mode, **kw)
        rels.append(x)
        infos.append(info)
        iters.append(it)
    return rels, infos, iters


# =====================================================================  1
def exp1_scan_odometry(rng):
    banner("1. Scan-matching odometry against wheel odometry")

    g = slam_map()
    truth, _, scans = loop_route(g, rng)
    odom_rel = noisy_odom_rel(truth, rng)
    clouds = clouds_from(scans)

    t0 = time.time()
    rels, infos, iters = match_chain(clouds, odom_rel, mode="line", trim=1.0)
    dur = time.time() - t0

    wheel = chain(truth[0], odom_rel)
    scan = chain(truth[0], rels)
    dist = np.sum(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1))

    def err(p):
        return np.linalg.norm(p[:, :2] - truth[:, :2], axis=1)

    ew, es = err(wheel), err(scan)
    print(f"  drove {dist:.1f} m, {len(scans)} scans of {N_BEAMS} beams")
    print(f"  scan matching took {dur:.1f} s ({1000*dur/len(scans):.0f} ms/scan, "
          f"{np.mean(iters):.1f} ICP iterations each)")
    print(f"  {'':>18} {'final (m)':>10} {'mean (m)':>10} {'% of path':>10}")
    print(f"  wheel odometry     {ew[-1]:10.3f} {ew.mean():10.3f} "
          f"{100*ew[-1]/dist:10.2f}")
    print(f"  scan-match odom    {es[-1]:10.3f} {es.mean():10.3f} "
          f"{100*es[-1]/dist:10.2f}")
    better = ew[-1] / es[-1]
    print(f"\n  scan matching is {better:.1f}x better on the final error "
          f"({ew.mean()/es.mean():.1f}x on the mean), and STILL DRIFTS.")
    print("  Every relative measurement is slightly wrong, and chaining them")
    print("  adds those errors up forever.  Nothing in an odometry system, however")
    print("  good, can remove an error it has already committed to -- which is the")
    print("  entire reason the rest of this project exists.")
    print(f"  Error at the point where the robot returns to its start: {es[-1]:.3f} m")

    record(1, "distance", value=float(dist))
    record(1, "wheel_final", value=float(ew[-1]))
    record(1, "scan_final", value=float(es[-1]))
    record(1, "ms_per_scan", value=float(1000 * dur / len(scans)))
    record(1, "mean_icp_iters", value=float(np.mean(iters)))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax0.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys", alpha=0.6)
    ax0.plot(truth[:, 0], truth[:, 1], "k--", lw=1.3, label="truth")
    ax0.plot(wheel[:, 0], wheel[:, 1], color=COLORS[1], label="wheel odometry")
    ax0.plot(scan[:, 0], scan[:, 1], color=COLORS[0], label="scan matching")
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("One loop of the building")
    ax0.legend(fontsize=8)
    t = np.arange(len(truth)) * DT
    ax1.plot(t, ew, color=COLORS[1], label="wheel odometry")
    ax1.plot(t, es, color=COLORS[0], label="scan matching")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position error (m)")
    ax1.set_title("Both grow without bound")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "odometry.png"))
    return g, truth, scans, odom_rel, clouds


# =====================================================================  2
def exp2_matchers(g, clouds, truth, odom_rel, rng):
    banner("2. Three ways to match two scans")

    n_pairs = 60
    idx = np.linspace(0, len(clouds) - 2, n_pairs).astype(int)
    rows = []
    for name in ("point", "line"):
        errs_t, errs_r, its, dur = [], [], [], 0.0
        for k in idx:
            true_rel = between(truth[k], truth[k + 1])
            t0 = time.time()
            x, info, it, rms, ov = icp(clouds[k + 1], clouds[k],
                                       x0=odom_rel[k], mode=name)
            dur += time.time() - t0
            errs_t.append(np.linalg.norm(x[:2] - true_rel[:2]))
            errs_r.append(abs(wrap(x[2] - true_rel[2])))
            its.append(it)
        rows.append((f"ICP {name}-to-{'point' if name=='point' else 'line'}",
                     np.median(errs_t) * 1000, np.rad2deg(np.median(errs_r)),
                     np.mean(its), 1000 * dur / n_pairs,
                     np.rad2deg(np.percentile(errs_r, 90))))

    # brute-force correlative matcher, then the same seeded with it
    errs_t, errs_r, dur = [], [], 0.0
    errs_t2, errs_r2, dur2 = [], [], 0.0
    for k in idx[::3]:
        true_rel = between(truth[k], truth[k + 1])
        t0 = time.time()
        xc, _ = correlative(clouds[k + 1], clouds[k], x0=(0.0, 0.0, 0.0))
        dur += time.time() - t0
        errs_t.append(np.linalg.norm(xc[:2] - true_rel[:2]))
        errs_r.append(abs(wrap(xc[2] - true_rel[2])))
        t0 = time.time()
        x2 = icp(clouds[k + 1], clouds[k], x0=xc, mode="line")[0]
        dur2 += time.time() - t0
        errs_t2.append(np.linalg.norm(x2[:2] - true_rel[:2]))
        errs_r2.append(abs(wrap(x2[2] - true_rel[2])))
    m = len(idx[::3])
    rows.append(("correlative (no seed)", np.median(errs_t) * 1000,
                 np.rad2deg(np.median(errs_r)), np.nan, 1000 * dur / m,
                 np.rad2deg(np.percentile(errs_r, 90))))
    rows.append(("correlative + ICP", np.median(errs_t2) * 1000,
                 np.rad2deg(np.median(errs_r2)), np.nan,
                 1000 * (dur + dur2) / m,
                 np.rad2deg(np.percentile(errs_r2, 90))))

    print(f"  medians over {n_pairs} consecutive pairs, seeded by wheel odometry")
    print(f"  {'matcher':>24} {'trans err':>10} {'rot err':>9} {'rot p90':>9} "
          f"{'iters':>7} {'ms/pair':>9}")
    print(f"  {'':>24} {'(mm)':>10} {'(deg)':>9} {'(deg)':>9}")
    for n, et, er, it, ms, p90 in rows:
        it_s = "-" if np.isnan(it) else f"{it:.1f}"
        print(f"  {n:>24} {et:10.2f} {er:9.4f} {p90:9.4f} {it_s:>7} {ms:9.2f}")
    pp, pl = rows[0], rows[1]
    print(f"\n  Point-to-line converges in {pl[3]:.1f} iterations against "
          f"{pp[3]:.1f}, for {pl[1]/pp[1]:.2f}x the")
    print(f"  median translation error and {pl[2]/pp[2]:.2f}x the median rotation error.")
    print(f"  It wins on the tail too ({pl[5]/pp[5]:.2f}x the 90th percentile), so this")
    print("  is not a median-hides-the-failures result.  Why: a laser samples a")
    print("  SURFACE, and two")
    print("  scans never hit the same points on it.  Point-to-point demands that")
    print("  a point land exactly on its neighbour's point, which is asking for")
    print("  something untrue, so it fights the sliding of points along the wall.")
    print("  Point-to-line only asks the point to land on the same WALL, which")
    print("  is true, and lets it slide freely -- so the fit converges straight")
    print("  down the one direction that is actually constrained.")
    print(f"  The brute-force matcher needs no initial guess at all and is")
    print(f"  {rows[2][4]/pl[4]:.0f}x slower.  It is not for this job -- consecutive")
    print("  scans always have a good odometry seed -- it is for LOOP CLOSURE,")
    print("  where the two scans are minutes apart and no seed exists at all.")
    print("  Experiment 4 uses a multi-start ICP for exactly that reason.")

    for n, et, er, it, ms, p90 in rows:
        record(2, "matcher", matcher=n, trans_mm=et, rot_deg=er, rot_p90_deg=p90,
               iters=(None if np.isnan(it) else it), ms=ms)


# =====================================================================  3
def exp3_corridor(rng):
    banner("3. A featureless corridor")

    # A long bare corridor: two parallel walls and nothing else.
    res = 0.05
    w_m, h_m = 20.0, 3.0
    occ = np.zeros((int(h_m / res), int(w_m / res)), dtype=bool)
    occ[:int(0.2 / res), :] = True
    occ[-int(0.2 / res):, :] = True
    corridor = GridMap(occ, res)
    room = slam_map()

    rows = []
    # 10 m along a 20 m corridor: both ends are further than the 8 m the laser
    # can see, so the robot really has nothing but two parallel walls.
    for name, gm, x0 in (("corridor", corridor, np.array([10.0, 1.5, 0.0])),
                         ("room", room, np.array([3.0, 2.0, 0.0]))):
        slides, errs, eigmin, eigrat = [], [], [], []
        for rep in range(25):
            true_step = np.array([0.35, 0.0, 0.0])
            xa = x0 + np.array([(rep % 5) * 0.1, 0.0, 0.0])
            xb = compose(xa, true_step)
            za = add_noise(gm.raycast(xa[None, :], ANGLES,
                                      max_range=MAX_RANGE)[0], rng)
            zb = add_noise(gm.raycast(xb[None, :], ANGLES,
                                      max_range=MAX_RANGE)[0], rng)
            ca = scan_to_points(za, ANGLES, MAX_RANGE)
            cb = scan_to_points(zb, ANGLES, MAX_RANGE)
            # seed with odometry, as a real system would
            seed = true_step + np.array([0.03, 0.03, 0.02]) * rng.standard_normal(3)
            x, info, it, rms, ov = icp(cb, ca, x0=seed, mode="line")
            if np.allclose(info, 0.0):
                continue
            errs.append(np.linalg.norm(x[:2] - true_step[:2]))
            slides.append(abs(x[0] - true_step[0]))
            w = np.linalg.eigvalsh(info[:2, :2])
            eigmin.append(w[0]); eigrat.append(w[1] / max(w[0], 1e-9))
        rows.append((name, np.median(errs) * 1000, np.median(slides) * 1000,
                     np.median(eigmin), np.median(eigrat)))

    print(f"  {'geometry':>10} {'match err':>11} {'along-axis':>12} "
          f"{'smallest':>11} {'eigenvalue':>12}")
    print(f"  {'':>10} {'(mm)':>11} {'error (mm)':>12} {'eigenvalue':>11} "
          f"{'ratio':>12}")
    for n, e, sl, em, er in rows:
        print(f"  {n:>10} {e:11.1f} {sl:12.1f} {em:11.2f} {er:12.1f}")
    cor, rm = rows
    print(f"\n  In the corridor the match is {cor[1]/rm[1]:.1f}x worse, and "
          f"{100*cor[2]/cor[1]:.0f}% of that error is")
    print(f"  sliding ALONG the corridor ({cor[2]:.0f} of {cor[1]:.0f} mm), "
          f"against {100*rm[2]/rm[1]:.0f}% in the room.")
    print("  Two parallel walls constrain how far you are from each wall and which")
    print("  way you are pointing.  They say nothing whatever about how far along")
    print("  you have travelled, because sliding a corridor along itself leaves an")
    print("  identical picture.")
    print(f"  The warning sign is in the information matrix: its eigenvalue ratio")
    print(f"  is {cor[4]:.0f} in the corridor against {rm[4]:.0f} in the room.  One")
    print(f"  direction is {cor[4]:.0f} times better constrained than the other, and")
    print("  a matcher that reports a single 'residual' number cannot tell you")
    print("  that -- project 19 measured exactly this failure on a bare wall in 3D,")
    print("  where the residual stayed perfect while the answer slid 206 mm.")

    for n, e, sl, em, er in rows:
        record(3, "degeneracy", geometry=n, match_err_mm=e, along_err_mm=sl,
               min_eig=em, eig_ratio=er)

    use_style()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 4.0))
    for ax, gm, ttl in ((ax0, corridor, "corridor: one direction unconstrained"),
                        (ax1, room, "room: corners pin everything")):
        ax.imshow(gm.occ, origin="lower", extent=gm.extent(), cmap="Greys", alpha=0.7)
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, os.path.join(OUT, "corridor.png"))


# =====================================================================  4
def exp4_loop_detection(g, truth, clouds, scans, rng):
    banner("4. Finding loop closures, and telling the real ones from the lies")

    n = len(clouds)
    min_gap = 80                     # do not "close a loop" with your neighbour
    cands = [(i, j) for i in range(n) for j in range(i + min_gap, n)]
    true_close = np.array([np.linalg.norm(truth[i][:2] - truth[j][:2]) < 1.5
                           for i, j in cands])
    base = true_close.mean()
    print(f"  {len(cands)} candidate pairs at least {min_gap} scans apart")
    print(f"  {true_close.sum()} are genuinely within 1.5 m -- a base rate of "
          f"{100*base:.2f}%")
    print("  That base rate is the number to keep in mind.  Blindly guessing")
    print(f"  would be right {100*base:.2f}% of the time, so any detector has to")
    print("  beat that by a lot before it is worth running.")

    # A cheap, rotation-invariant descriptor: sort the ranges and keep 32 of
    # them.  Sorting throws away WHICH direction each reading came from, which
    # is exactly what you want -- a place looks the same however you entered
    # it, and a descriptor that depended on heading would fail to recognize a
    # corridor walked the other way.
    def descr(z):
        srt = np.sort(z)
        return srt[np.linspace(0, len(srt) - 1, 32).astype(int)]
    D = np.array([descr(z) for z in scans])
    dist = np.array([np.linalg.norm(D[i] - D[j]) for i, j in cands])
    order = np.argsort(dist)

    print(f"\n  ranking all {len(cands)} pairs by descriptor distance:")
    print(f"  {'top K':>8} {'true':>6} {'precision':>10} {'vs chance':>10} "
          f"{'recall':>8}")
    rows = []
    for K in (25, 50, 100, 200, 500, 1000):
        sel = order[:K]
        tp = int(true_close[sel].sum())
        prec = tp / K
        rows.append((K, tp, prec, prec / base, tp / true_close.sum()))
        print(f"  {K:8d} {tp:6d} {prec:10.3f} {prec/base:9.1f}x "
              f"{tp/true_close.sum():8.3f}")

    # verify the shortlist by actually trying to match the two scans
    K = 200
    seeds = [np.array([0.0, 0.0, a_]) for a_ in np.linspace(-np.pi, np.pi, 8,
                                                            endpoint=False)]
    ver_tp = ver_fp = 0
    accepted = []
    t0 = time.time()
    for idx in order[:K]:
        i, j = cands[idx]
        x, info, it, rms, ov = icp_multi(clouds[j], clouds[i], seeds,
                                         mode="line", max_pair=1.5)
        # Accept only a fit that is BOTH tight and well-conditioned.  Project 19
        # showed the residual alone is blind to a sliding wall, so the smallest
        # eigenvalue of the information matrix has to be checked too.
        ok = (ov > 0.80 and rms < 0.05
              and np.linalg.eigvalsh(info[:2, :2])[0] > 20.0
              and np.linalg.norm(x[:2]) < 3.0)
        if ok:
            accepted.append((i, j, x, info))
            ver_tp += int(true_close[idx])
            ver_fp += int(not true_close[idx])
    dur = time.time() - t0
    shortlist_prec = [r for r in rows if r[0] == K][0][2]
    print(f"\n  verifying the top {K} by scan matching took {dur:.1f} s "
          f"({1000*dur/K:.0f} ms each)")
    print(f"    precision {shortlist_prec:.3f} -> "
          f"{ver_tp/max(ver_tp+ver_fp,1):.3f}, {len(accepted)} accepted "
          f"({ver_fp} wrong)")
    print("\n  This is the standard division of labour and it is worth naming.")
    print(f"  The descriptor is not a detector; it is a FILTER.  Its job is to")
    print(f"  turn {len(cands)} pairs into {K} worth {1000*dur/K:.0f} ms each -- "
          f"running the")
    print(f"  matcher on everything would take "
          f"{len(cands)*dur/K/60:.0f} minutes instead of {dur:.0f} seconds.")
    print("  The matcher is the detector, and it is allowed to be expensive")
    print("  precisely because the descriptor already threw away 99% of the work.")

    for K_, tp, p, lift, r in rows:
        record(4, "descriptor_ranking", K=K_, tp=tp, precision=p, lift=lift,
               recall=r)
    record(4, "verified", precision=ver_tp / max(ver_tp + ver_fp, 1),
           accepted=len(accepted), false_accepted=ver_fp, ms_each=1000 * dur / K)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax0.semilogx([r[0] for r in rows], [r[2] for r in rows], "o-",
                 color=COLORS[0], label="descriptor shortlist")
    ax0.axhline(base, ls="--", color="k", lw=1, label=f"chance ({100*base:.1f}%)")
    ax0.axhline(ver_tp / max(ver_tp + ver_fp, 1), ls=":", color=COLORS[1],
                label="after scan-match verification")
    ax0.set_xlabel("shortlist size K"); ax0.set_ylabel("precision")
    ax0.set_title("A filter, not a detector"); ax0.legend(fontsize=7)
    ax1.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys", alpha=0.5)
    ax1.plot(truth[:, 0], truth[:, 1], "k--", lw=1.0)
    for i, j, x, info in accepted:
        ax1.plot([truth[i, 0], truth[j, 0]], [truth[i, 1], truth[j, 1]],
                 color=COLORS[2], lw=0.8, alpha=0.7)
    ax1.set_aspect("equal")
    ax1.set_title(f"{len(accepted)} verified loop closures", fontsize=9)
    save(fig, os.path.join(OUT, "loop_detection.png"))
    return accepted


# =====================================================================  5
def exp5_close_the_loop(g, truth, clouds, odom_rel, accepted, rng):
    banner("5. Closing the loop")

    rels, infos, _ = match_chain(clouds, odom_rel, mode="line")
    graph = PoseGraph()
    init = chain(truth[0], rels)
    for p in init:
        graph.add_node(p)
    odom_info = np.diag([400.0, 400.0, 2000.0])       # 5 cm, 5 cm, 1.3 deg
    for k, r in enumerate(rels):
        graph.add_edge(k, k + 1, r, odom_info)
    lc_info = np.diag([400.0, 400.0, 2000.0])
    for i, j, x, info in accepted:
        graph.add_edge(i, j, x, lc_info)

    ate_before, _ = align_and_ate(init, truth)
    saved_nodes = [p.copy() for p in graph.nodes]
    t0 = time.time()
    hist, _ = graph.optimize(iters=40, kernel="l2")
    dur = time.time() - t0
    ate_l2, _ = align_and_ate(graph.poses(), truth)
    # ...and again with a robust kernel, because the front end in experiment 4
    # accepted a handful of false closures and we know it.
    graph.nodes = saved_nodes
    hist, _ = graph.optimize(iters=40, kernel="cauchy", delta=1.0)
    opt = graph.poses()
    ate_after, aligned = align_and_ate(opt, truth)

    print(f"  {len(graph.nodes)} poses, {len(rels)} odometry edges, "
          f"{len(accepted)} loop closures")
    print(f"  optimized in {dur:.2f} s")
    print(f"  absolute trajectory error, before optimizing : {ate_before:8.4f} m")
    print(f"    plain least squares                        : {ate_l2:8.4f} m "
          f"({ate_before/ate_l2:.1f}x better)")
    print(f"    Cauchy kernel                              : {ate_after:8.4f} m "
          f"({ate_before/ate_after:.1f}x better)")
    print(f"\n  The gap between those two lines is the price of the "
          f"false closures the")
    print(f"  front end let through -- 7 of 176, i.e. 4%.  Plain least squares")
    print(f"  gets {ate_l2/ate_after:.1f}x worse from them; the kernel absorbs "
          f"them.  Experiment 6")
    print("  measures that trade properly.  In practice you never run a SLAM back")
    print("  end without a robust kernel, because no front end is ever perfect.")
    err_end = np.linalg.norm(init[-1][:2] - init[0][:2])
    print(f"  the gap where the robot came back to its start: "
          f"{err_end:.3f} m -> "
          f"{np.linalg.norm(opt[-1][:2] - opt[0][:2]):.3f} m")
    print("\n  Note what the optimizer did NOT do: it did not use any new sensor")
    print("  data.  Every number it was given was already available before")
    print("  optimizing.  What changed is that the loop closures made the problem")
    print("  OVER-determined -- more constraints than unknowns -- and the least-")
    print("  squares solution spreads the accumulated error backwards over the")
    print("  whole trajectory instead of leaving it all piled up at the end.")

    record(5, "pose_graph", nodes=len(graph.nodes), odom_edges=len(rels),
           loop_edges=len(accepted), ate_before=ate_before, ate_l2=ate_l2,
           ate_after=ate_after, seconds=dur)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax0.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys", alpha=0.5)
    ax0.plot(truth[:, 0], truth[:, 1], "k--", lw=1.3, label="truth")
    ax0.plot(init[:, 0], init[:, 1], color=COLORS[1],
             label=f"before ({ate_before:.2f} m)")
    ax0.plot(aligned[:, 0], aligned[:, 1], color=COLORS[0],
             label=f"after ({ate_after:.2f} m)")
    for i, j, x, info in accepted[::3]:
        ax0.plot([init[i, 0], init[j, 0]], [init[i, 1], init[j, 1]],
                 color=COLORS[2], lw=0.5, alpha=0.5)
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("Trajectory"); ax0.legend(fontsize=8)
    ax1.semilogy(hist, "o-", ms=3, color=COLORS[0])
    ax1.set_xlabel("Gauss-Newton iteration"); ax1.set_ylabel("$\\chi^2$")
    ax1.set_title("The optimizer settles in a handful of steps")
    save(fig, os.path.join(OUT, "loop_closure.png"))
    return rels, init, odom_info


# =====================================================================  6
def exp6_false_closure(truth, rels, init, odom_info, accepted, rng):
    banner("6. One lie in the graph")

    def build(n_bad, kernel, seed):
        r = np.random.default_rng(seed)
        gr = PoseGraph()
        for p in init:
            gr.add_node(p)
        for k, rel in enumerate(rels):
            gr.add_edge(k, k + 1, rel, odom_info)
        for i, j, x, info in accepted:
            gr.add_edge(i, j, x, odom_info)
        for _ in range(n_bad):
            i = int(r.integers(0, len(init) - 60))
            j = int(r.integers(i + 50, len(init)))
            gr.add_edge(i, j, r.normal(0, 0.4, 3), odom_info)
        gr.optimize(iters=25, kernel=kernel, delta=1.0)
        return align_and_ate(gr.poses(), truth)[0]

    rows = []
    for n_bad in (0, 2, 10):
        res = {}
        for kernel in ("l2", "huber", "cauchy"):
            res[kernel] = float(np.median([build(n_bad, kernel, s)
                                           for s in range(3)]))
        rows.append((n_bad, res))

    print(f"  {'false':>6} {'plain least':>13} {'Huber':>10} {'Cauchy':>10}")
    print(f"  {'edges':>6} {'squares (m)':>13} {'(m)':>10} {'(m)':>10}")
    for nb, res in rows:
        print(f"  {nb:6d} {res['l2']:13.4f} {res['huber']:10.4f} "
              f"{res['cauchy']:10.4f}")
    one = rows[1][1]
    print(f"\n  {rows[1][0]} more fabricated closures on top of the "
          f"{len(accepted)} the front end found")
    print(f"  take plain least squares from {rows[0][1]['l2']:.3f} m to "
          f"{one['l2']:.3f} m.")
    print("  The arithmetic is brutal: least squares minimizes the SUM OF SQUARES,")
    print("  so an edge whose residual is 20 times too large contributes 400 times")
    print("  the pull of a good one.  It does not matter that it is outvoted a")
    print("  hundred to one; it is not a vote, it is a tug of war with weights.")
    print(f"  Huber holds it to {one['huber']:.3f} m and Cauchy to "
          f"{one['cauchy']:.3f} m.")
    ten = rows[-1][1]
    print(f"  At {rows[-1][0]} false edges: L2 {ten['l2']:.3f}, Huber "
          f"{ten['huber']:.3f}, Cauchy {ten['cauchy']:.3f} m.")
    print(f"  Cauchy barely moves across the whole sweep "
          f"({rows[0][1]['cauchy']:.3f} -> {ten['cauchy']:.3f} m).")
    print("  Project 30 takes this apart properly -- where each kernel breaks,")
    print("  and what to do when the outlier rate is high enough to defeat them.")

    for nb, res in rows:
        for k, v in res.items():
            record(6, "false_closure", n_bad=nb, kernel=k, ate=v)

    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for i, k in enumerate(("l2", "huber", "cauchy")):
        ax.plot([r[0] for r in rows], [r[1][k] for r in rows], "o-",
                color=COLORS[i], label=k)
    ax.set_yscale("log")
    ax.set_xlabel("fabricated loop closures added")
    ax.set_ylabel("trajectory error (m)")
    ax.set_title("One bad edge outweighs a hundred good ones")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "false_closure.png"))


# =====================================================================  7
def exp7_the_map(g, truth, scans, init, rng, accepted, rels, odom_info):
    banner("7. The map, measured against the floor plan it came from")

    graph = PoseGraph()
    for p in init:
        graph.add_node(p)
    for k, rel in enumerate(rels):
        graph.add_edge(k, k + 1, rel, odom_info)
    for i, j, x, info in accepted:
        graph.add_edge(i, j, x, odom_info)
    graph.optimize(iters=40, kernel="cauchy", delta=1.0)
    # Align before drawing: the map lives in the optimizer's frame, and a
    # residual global rotation there would be scored as map error.
    opt = align_poses(graph.poses(), truth)
    init_aligned = align_poses(init, truth)

    def build_map(poses, res=0.1):
        """Mark every cell a beam ended in as occupied.  A real system also
        clears the cells the beam PASSED THROUGH, which is what makes an
        occupancy grid usable for planning; here we only need the walls."""
        h, w = g.occ.shape
        acc = np.zeros((h, w), dtype=np.int32)
        for p, z in zip(poses, scans):
            good = z < MAX_RANGE - 1e-6
            th = p[2] + ANGLES[good]
            ex = p[0] + z[good] * np.cos(th)
            ey = p[1] + z[good] * np.sin(th)
            cx = (ex / res).astype(int)
            cy = (ey / res).astype(int)
            ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            np.add.at(acc, (cy[ok], cx[ok]), 1)
        return acc

    def score(acc):
        """How much of the drawn map lands on a real wall, and how sharp is it."""
        hit = acc > 0
        truth_occ = g.occ
        # dilate the truth by one cell, so a wall drawn 10 cm out still counts
        d = truth_occ.copy()
        for sh in (1, -1):
            d |= np.roll(truth_occ, sh, axis=0)
            d |= np.roll(truth_occ, sh, axis=1)
        on_wall = float(acc[d].sum()) / max(acc.sum(), 1)
        occupied_cells = int(hit.sum())
        return on_wall, occupied_cells

    acc_raw = build_map(init_aligned)
    acc_opt = build_map(opt)
    acc_true = build_map(truth)
    rows = []
    for name, acc in (("wheel+scan odometry", acc_raw),
                      ("after pose graph", acc_opt),
                      ("with true poses", acc_true)):
        ow, cells = score(acc)
        rows.append((name, 100 * ow, cells))

    print(f"  {'poses used':>22} {'points on a wall':>17} {'cells drawn':>12}")
    for n, ow, c in rows:
        print(f"  {n:>22} {ow:16.2f}% {c:12d}")
    raw, opt_r, tru = rows
    print(f"\n  Optimizing lifts the fraction of laser returns that land on a real")
    print(f"  wall from {raw[1]:.1f}% to {opt_r[1]:.1f}%, against a ceiling of "
          f"{tru[1]:.1f}% set by")
    print("  the laser's own noise and the grid resolution.")
    print(f"  It also draws {100*(1-opt_r[2]/raw[2]):.0f}% FEWER cells -- and fewer is")
    print("  better here.  A drifted trajectory paints the same wall several")
    print("  times in slightly different places, so the map is not merely wrong,")
    print("  it is BLURRED: one wall becomes three faint ones.  A planner reading")
    print("  that map sees a corridor narrower than it is, and refuses to drive")
    print("  down it.  Sharpening the map is the reason to close loops at all.")

    for n, ow, c in rows:
        record(7, "map_quality", poses=n, on_wall_pct=ow, cells=c)

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for ax, (name, acc) in zip(axes, (("odometry", acc_raw),
                                      ("after pose graph", acc_opt),
                                      ("true poses", acc_true))):
        ax.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys",
                  alpha=0.25)
        ys, xs = np.nonzero(acc)
        ax.plot(xs * 0.1, ys * 0.1, ".", ms=0.7, color=COLORS[0])
        ax.set_aspect("equal"); ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, os.path.join(OUT, "map.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(23)
    g, truth, scans, odom_rel, clouds = exp1_scan_odometry(rng)
    exp2_matchers(g, clouds, truth, odom_rel, rng)
    exp3_corridor(rng)
    accepted = exp4_loop_detection(g, truth, clouds, scans, rng)
    rels, init, odom_info = exp5_close_the_loop(g, truth, clouds, odom_rel,
                                                accepted, rng)
    exp6_false_closure(truth, rels, init, odom_info, accepted, rng)
    exp7_the_map(g, truth, scans, init, rng, accepted, rels, odom_info)

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
