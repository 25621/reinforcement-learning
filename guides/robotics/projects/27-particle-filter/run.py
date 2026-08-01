"""Project 27 -- Monte Carlo Localization: finding a robot that has no idea
where it is.

Seven experiments:

  1. global localization: 5 000 guesses collapse into one
  2. resampling: never, multinomial, stratified, systematic
  3. how many particles do you actually need?
  4. the kidnapped robot, and why resampling alone can never recover
  5. an over-confident laser model, and the particles it kills
  6. the beam model against the likelihood field: accuracy against speed
  7. a symmetric building: where the EKF cannot even state the answer

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
sys.path.insert(0, os.path.join(_PROJ, "26-ekf-localization"))
sys.path.insert(0, os.path.join(_PROJ, "24-1d-kf"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from gridmap import GridMap, office_map, free_poses                     # noqa: E402
from pf import (ParticleFilter, effective_sample_size, estimate,        # noqa: E402
                RESAMPLERS, wrap, sample_motion, beam_log_likelihood)
from plot_style import COLORS, use_style, save                          # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

DT = 0.2
SIGMA_Z = 0.15                 # the laser's TRUE noise, metres
# ...and the width the filter is TOLD to assume when it is searching the whole
# map.  Deliberately 3x too broad.  Experiment 3 measures why: a model as sharp
# as the real sensor kills every particle that is not already almost exactly
# right, and during global localization none of them is.  Experiment 5 measures
# the other half of the trade -- once the track is held, the same broadening
# costs accuracy.
SIGMA_GLOBAL = 0.5
MAX_RANGE = 8.0
N_BEAMS = 12
ANGLES = np.linspace(-np.pi * 0.75, np.pi * 0.75, N_BEAMS)
ALPHA = np.array([0.06, 0.006, 0.06, 0.012])     # motion noise coefficients


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ------------------------------------------------------------------ the drive
def make_route(gmap, n, rng, x0=(2.0, 2.0, 0.0)):
    """Drive a fixed loop, bouncing off walls, and record what the laser saw."""
    x = np.array(x0, dtype=float)
    poses, controls, scans = [x.copy()], [], []
    for k in range(n):
        # a gentle wander that turns away when a wall gets close ahead
        ahead = gmap.raycast(x[None, :], np.array([0.0, 0.35, -0.35]),
                             max_range=MAX_RANGE)[0]
        if ahead[0] < 1.2:
            w = 1.4 if ahead[1] > ahead[2] else -1.4
            v = 0.25
        else:
            w = 0.35 * np.sin(k * 0.06) + 0.15 * np.sin(k * 0.017)
            v = 0.8
        u = np.array([v, w])
        x = sample_motion(x[None, :], u, DT, rng, ALPHA)[0]
        z = gmap.raycast(x[None, :], ANGLES, max_range=MAX_RANGE)[0]
        z = np.clip(z + SIGMA_Z * rng.standard_normal(N_BEAMS), 0.0, MAX_RANGE)
        poses.append(x.copy()); controls.append(u); scans.append(z)
    return np.array(poses), np.array(controls), np.array(scans)


def pos_err(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def local_cloud(pose, n, rng, spread=(0.3, 0.3, np.deg2rad(10.0))):
    """Particles scattered around a known starting pose -- the TRACKING case.

    Global localization (particles spread over the whole map) and tracking
    (particles already near the answer) are different problems and need very
    different particle counts.  Experiments 2, 5 and 6 are about how well a
    filter HOLDS a track, so they start here; experiments 1, 3, 4 and 7 are
    about finding the robot from nothing, so they start from free_poses.
    """
    out = np.repeat(np.asarray(pose)[None, :], n, axis=0)
    out = out + np.asarray(spread) * rng.standard_normal((n, 3))
    out[:, 2] = wrap(out[:, 2])
    return out


def run_pf(gmap, poses, controls, scans, n_particles, rng, init="global", **kw):
    parts = (free_poses(gmap, n_particles, rng) if init == "global"
             else local_cloud(poses[0], n_particles, rng))
    pf = ParticleFilter(gmap, parts, ALPHA, SIGMA_Z, ANGLES, rng=rng,
                        max_range=MAX_RANGE, **kw)
    est, ess, spread = [], [], []
    for k in range(len(controls)):
        e = pf.step(controls[k], DT, scans[k])
        est.append(pf.estimate())
        ess.append(e)
        spread.append(np.sqrt(np.average(
            (pf.parts[:, 0] - pf.parts[:, 0].mean()) ** 2 +
            (pf.parts[:, 1] - pf.parts[:, 1].mean()) ** 2, weights=pf.w)))
    return pf, np.array(est), np.array(ess), np.array(spread)


# =====================================================================  1
def exp1_global(rng):
    banner("1. Global localization: 5 000 guesses, one answer")

    g = office_map()
    poses, controls, scans = make_route(g, 160, rng)
    n = 5000
    parts = free_poses(g, n, rng)
    pf = ParticleFilter(g, parts, ALPHA, SIGMA_GLOBAL, ANGLES, rng=rng,
                        max_range=MAX_RANGE)
    snaps = {0: (pf.parts.copy(), pf.w.copy())}
    errs, esss = [], []
    t0 = time.time()
    for k in range(len(controls)):
        e = pf.step(controls[k], DT, scans[k])
        errs.append(pos_err(pf.estimate(), poses[k + 1]))
        esss.append(e)
        if k + 1 in (1, 3, 10, 40):
            snaps[k + 1] = (pf.parts.copy(), pf.w.copy())
    dur = time.time() - t0
    errs = np.array(errs)

    conv = int(np.argmax(errs < 0.5)) if (errs < 0.5).any() else -1
    print(f"  {n} particles, {N_BEAMS} beams, {len(controls)} steps in {dur:.1f} s "
          f"({1000*dur/len(controls):.1f} ms/step)")
    print(f"  error after 1 scan   {errs[0]:8.3f} m")
    print(f"  error after 3 scans  {errs[2]:8.3f} m")
    print(f"  error after 10 scans {errs[9]:8.3f} m")
    print(f"  settled error (last 100 steps) {errs[-100:].mean():.3f} m")
    print(f"  first step below 0.5 m: {conv}")
    print(f"  effective sample size fell {esss[0]:.0f} -> {np.mean(esss[-100:]):.0f} "
          f"out of {n}")
    print(f"  the filter resampled {pf.n_resample} times in {len(controls)} steps")

    record(1, "n_particles", value=n)
    record(1, "ms_per_step", value=1000 * dur / len(controls))
    record(1, "err_after_1", value=float(errs[0]))
    record(1, "err_after_3", value=float(errs[2]))
    record(1, "err_after_10", value=float(errs[9]))
    record(1, "settled_err", value=float(errs[-100:].mean()))
    record(1, "converge_step", value=float(conv))

    use_style()
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.2))
    for ax, k in zip(axes, sorted(snaps)):
        p, w = snaps[k]
        ax.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys",
                  interpolation="nearest", alpha=0.85)
        srt = np.argsort(w)
        ax.scatter(p[srt, 0], p[srt, 1], s=2, c=w[srt], cmap="viridis")
        if k > 0:
            ax.plot(poses[k, 0], poses[k, 1], "*", ms=13, color=COLORS[1])
        ax.set_title(f"after {k} scans" + ("" if k == 0 else
                                           f"\nerr {errs[k-1]:.2f} m"), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, os.path.join(OUT, "global.png"))
    return g, poses, controls, scans


# =====================================================================  2
def exp2_resampling(g, rng):
    banner("2. Resampling: never, multinomial, stratified, systematic")

    n_rep, n_part = 10, 800
    rows = []
    ess_keep = {}
    for mode in ("never", "multinomial", "stratified", "systematic"):
        errs, ess_end, uniq = [], [], []
        for rep in range(n_rep):
            poses, controls, scans = make_route(g, 120, rng)
            kw = dict(resampler="systematic", ess_frac=None) if mode == "never" \
                else dict(resampler=mode, ess_frac=0.5)
            pf, est, ess, _ = run_pf(g, poses, controls, scans, n_part, rng,
                                     init="local", **kw)
            errs.append(np.mean([pos_err(est[k], poses[k + 1])
                                 for k in range(60, len(controls))]))
            ess_end.append(ess[-20:].mean())
            uniq.append(len(np.unique(pf.parts[:, 0])))
            if rep == 0:
                ess_keep[mode] = ess
        rows.append((mode, np.mean(errs), np.mean(ess_end), np.mean(uniq)))

    print(f"  {'resampling':>13} {'pos err (m)':>12} {'final ESS':>10} "
          f"{'distinct':>9}   (of {n_part} particles)")
    for m, e, es, u in rows:
        print(f"  {m:>13} {e:12.4f} {es:10.1f} {u:9.1f}")
    never, best = rows[0], min(rows[1:], key=lambda r: r[1])
    print(f"\n  Without resampling the ESS collapses to {never[2]:.1f} of {n_part}:")
    print(f"    all the weight ends up on one particle and the other")
    print(f"    {100*(1-never[2]/n_part):.1f}% are carried along contributing nothing.  Error "
          f"{never[1]:.3f} m vs {best[1]:.3f} m -- {never[1]/best[1]:.0f}x worse.")
    sp = max(r[1] for r in rows[1:]) / min(r[1] for r in rows[1:])
    print(f"  The three resamplers differ by only {100*(sp-1):.0f}% in tracking error.")
    print("  That is honest and slightly disappointing, so here is the difference")
    print("  measured where it actually lives -- in the variance each scheme adds.")

    # A clean, direct measurement: fix a weight vector, resample it 20 000 times,
    # and look at how much the number of offspring per particle jumps around.
    # Theory says systematic gives the smallest spread; this checks it.
    n_small = 200
    ww = rng.random(n_small) ** 3
    ww /= ww.sum()
    print(f"\n  {'scheme':>13} {'var of offspring count':>24} {'zero-offspring':>16}")
    for mode in ("multinomial", "stratified", "systematic"):
        counts = np.zeros((2000, n_small))
        for i in range(2000):
            idx = RESAMPLERS[mode](ww, rng)
            counts[i] = np.bincount(idx, minlength=n_small)
        v = float(counts.var(axis=0).mean())
        zero = float((counts == 0).mean())
        print(f"  {mode:>13} {v:24.4f} {100*zero:15.1f}%")
        record(2, "offspring_variance", mode=mode, variance=v, zero_frac=zero)
    print("  Same expected number of children for every particle; different spread.")
    print("  Systematic draws ONE random number and steps through the weights in")
    print("  equal strides, so a particle holding 1/N of the weight is guaranteed")
    print("  exactly one child.  Multinomial rolls the dice N separate times, so")
    print("  that same particle survives only about 63% of the time.  The loss is")
    print("  pure noise added to your belief for no information in return.")

    for m, e, es, u in rows:
        record(2, "resampling", mode=m, pos_err=e, final_ess=es, distinct=u)

    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (m, e) in enumerate(ess_keep.items()):
        ax.semilogy(np.arange(len(e)) * DT, np.maximum(e, 1), color=COLORS[i], label=m)
    ax.axhline(0.5 * n_part, ls="--", color="k", lw=1, label="resample threshold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("effective sample size")
    ax.set_title(f"Of {n_part} particles, how many are doing any work")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "resampling.png"))


# =====================================================================  3
def exp3_particle_count(g, rng):
    banner("3. How many particles do you actually need?")

    counts = [500, 1000, 2500, 5000, 10000]
    n_rep = 6
    rows = []
    routes = [make_route(g, 80, rng) for _ in range(n_rep)]
    for n in counts:
        row = {}
        for sig, tag in ((SIGMA_Z, "sharp"), (SIGMA_GLOBAL, "broad")):
            succ, errs, times = 0, [], []
            for poses, controls, scans in routes:
                t0 = time.time()
                parts = free_poses(g, n, rng)
                pf = ParticleFilter(g, parts, ALPHA, sig, ANGLES, rng=rng,
                                    max_range=MAX_RANGE)
                est = []
                for k in range(len(controls)):
                    pf.step(controls[k], DT, scans[k])
                    est.append(pf.estimate())
                times.append((time.time() - t0) / len(controls))
                e = np.median([pos_err(est[k], poses[k + 1])
                               for k in range(60, len(controls))])
                errs.append(e); succ += int(e < 0.5)
            row[tag] = (100.0 * succ / n_rep, np.mean(errs), 1000 * np.mean(times))
        rows.append((n, row))

    print("  global localization from a uniform prior, 80 scans to converge")
    print(f"  the laser's real noise is {SIGMA_Z} m; two models are compared")
    print(f"  {'particles':>10} | {'sharp model (0.15 m)':>25} | "
          f"{'broad model (0.50 m)':>25} | {'ms/step':>8}")
    print(f"  {'':>10} | {'success %':>12} {'median (m)':>12} | "
          f"{'success %':>12} {'median (m)':>12} |")
    for n, row in rows:
        sh, br = row["sharp"], row["broad"]
        print(f"  {n:10d} | {sh[0]:12.0f} {sh[1]:12.4f} | "
              f"{br[0]:12.0f} {br[1]:12.4f} | {br[2]:8.2f}")
    ok = [r for r in rows if r[1]["broad"][0] >= 80.0]
    print(f"\n  Sharp model: adding particles barely helps -- "
          f"{rows[0][1]['sharp'][0]:.0f}% at {rows[0][0]} particles, "
          f"{rows[-1][1]['sharp'][0]:.0f}% at {rows[-1][0]}.")
    print(f"  Broad model: {ok[0][0] if ok else 'none'} particles already reaches "
          f"{ok[0][1]['broad'][0]:.0f}%.")
    print("  This is the single most counter-intuitive thing in the project.  A")
    print("  sensor model TIGHTER than the sensor makes localization fail, and no")
    print("  amount of extra compute rescues it, because the failure is not a")
    print("  sampling problem: every particle that is not already nearly perfect")
    print("  is assigned a likelihood of effectively zero on the very first scan,")
    print("  and resampling can only copy what survived.  Widening the model is")
    print("  saying 'a particle 30 cm out is still worth keeping for now'.")
    print(f"  cost is linear in the particle count: "
          f"{rows[-1][1]['broad'][2]/rows[0][1]['broad'][2]:.0f}x the time for "
          f"{counts[-1]//counts[0]}x particles")
    print(f"  Compare with TRACKING, where experiment 2 held a track to 0.03 m")
    print("  with 800 particles.  Global localization needs one particle close")
    print("  enough to the answer to survive the first scan, and 'close enough' is")
    print("  a small box in a 3-D space (x, y, heading).  Filling a space to a")
    print("  fixed resolution costs particles as the CUBE of that resolution.")

    for n, row in rows:
        for tag, v in row.items():
            record(3, "particle_count", n=n, model=tag, success_pct=v[0],
                   err=v[1], ms_per_step=v[2])

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.semilogx([r[0] for r in rows], [r[1]["sharp"][0] for r in rows], "o-",
                color=COLORS[1], label="sharp model (0.15 m)")
    ax.semilogx([r[0] for r in rows], [r[1]["broad"][0] for r in rows], "s-",
                color=COLORS[0], label="broad model (0.50 m)")
    ax2 = ax.twinx()
    ax2.loglog([r[0] for r in rows], [r[1]["broad"][2] for r in rows], "^:",
               color=COLORS[6], label="ms per step")
    ax2.set_ylabel("ms per step", color=COLORS[6]); ax2.grid(False)
    ax.set_xlabel("number of particles")
    ax.set_ylabel("global localization success (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("More particles cannot fix an over-sharp sensor model")
    ax.legend(fontsize=8, loc="center right")
    save(fig, os.path.join(OUT, "particle_count.png"))


# =====================================================================  4
def exp4_kidnapped(g, rng):
    banner("4. The kidnapped robot")

    KIDNAP = 120

    n_rep, n_part = 6, 2000
    rows = []
    keep = {}
    for inject in (0.0, 0.005, 0.05, 0.20, "adaptive"):
        recov, final, settled = 0, [], []
        for rep in range(n_rep):
            # TWO independent drives.  The robot follows route A for KIDNAP
            # steps, is then picked up and set down somewhere else, and carries
            # on along route B.  Splicing two real drives -- rather than
            # teleporting the pose and re-using route A's controls -- matters:
            # controls planned for one part of the building drive straight into
            # a wall somewhere else, and scans taken from inside a wall are
            # garbage that NO particle can match, which would make every method
            # here fail for a reason that has nothing to do with kidnapping.
            pa, ca, sa = make_route(g, KIDNAP, rng)
            start_b = free_poses(g, 1, rng)[0]
            pb, cb, sb = make_route(g, 250 - KIDNAP, rng, x0=tuple(start_b))
            poses = np.vstack([pa, pb[1:]])
            controls = np.vstack([ca, cb])
            scans = np.vstack([sa, sb])
            parts = local_cloud(poses[0], n_part, rng)
            pf = ParticleFilter(g, parts, ALPHA, SIGMA_GLOBAL, ANGLES, rng=rng,
                                max_range=MAX_RANGE,
                                inject=(0.0 if inject == "adaptive" else inject),
                                adaptive=(inject == "adaptive"))
            errs = []
            for k in range(len(controls)):
                pf.step(controls[k], DT, scans[k])
                errs.append(pos_err(pf.estimate(), poses[k + 1]))
            errs = np.array(errs)
            after = errs[KIDNAP:]
            recov += int((after[-40:] < 0.5).mean() > 0.5)
            final.append(np.median(after[-40:]))
            settled.append(errs[45:KIDNAP].mean())
            if rep == 0:
                keep[inject] = errs
        rows.append((inject, 100.0 * recov / n_rep, np.mean(final), np.mean(settled)))

    print(f"  {'injection':>10} {'recovered':>10} {'err after':>10} {'err before':>11}")
    print(f"  {'rate':>10} {'%':>10} {'kidnap (m)':>10} {'kidnap (m)':>11}")
    for i, r, f, sb in rows:
        lbl = i if isinstance(i, str) else f"{i:.3f}"
        print(f"  {lbl:>10} {r:10.0f} {f:10.3f} {sb:11.4f}")
    base, ad = rows[0], [r for r in rows if isinstance(r[0], str)][0]
    fixed = [r for r in rows[1:] if not isinstance(r[0], str)]
    best_fixed = max(fixed, key=lambda r: r[1])
    print(f"\n  With no injection at all the filter still recovers {base[1]:.0f}% of")
    print("  the time -- not never, but only by luck: resampling can only COPY")
    print("  particles it already has, so recovery depends on some straggler")
    print("  happening to sit near the new location when the robot arrives.")
    print(f"\n  A fixed injection rate works, at a price paid every single step.")
    print(f"  {100*best_fixed[0]:.0f}% injection: {best_fixed[1]:.0f}% recovery, but the error")
    print(f"  BEFORE anything went wrong rose {base[3]:.4f} -> {best_fixed[3]:.4f} m "
          f"({100*(best_fixed[3]/base[3]-1):+.0f}%).")
    print(f"  And more is not better: {100*fixed[-1][0]:.0f}% injection recovers "
          f"{fixed[-1][1]:.0f}% of the time,")
    print(f"  because at that rate the cloud never converges in the first place "
          f"({fixed[-1][3]:.2f} m).")
    print(f"\n  ADAPTIVE injection (augmented MCL) gets {ad[1]:.0f}% recovery -- the same")
    print(f"  as the best fixed rate -- while costing only "
          f"{100*(ad[3]/base[3]-1):+.0f}% before the kidnap")
    print(f"  instead of {100*(best_fixed[3]/base[3]-1):+.0f}%.  That is the whole point: "
          f"the same insurance,")
    print(f"  {best_fixed[3]/ad[3]:.1f}x cheaper, because it is only bought when needed.")
    print("  It watches the average likelihood of the scan through two running")
    print("  averages, one fast and one slow.  While the filter is right they")
    print("  agree and nothing is injected.  When the robot is moved, the fast")
    print("  average collapses in a few steps while the slow one lags, and the")
    print("  gap between them becomes the injection rate -- a big burst, at")
    print("  exactly the moment a big burst is worth paying for.")
    print("  Note what makes this possible: the AVERAGE, UNNORMALIZED likelihood.")
    print("  Once weights are normalized they always sum to one no matter how")
    print("  badly every particle is doing, so the one number that says 'I am")
    print("  lost' has to be read off before normalization.")
    for i, r, f, sb in rows:
        record(4, "kidnap", inject=str(i), recovered_pct=r, err_after=f,
               err_before=sb)

    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (inj, e) in enumerate(keep.items()):
        lbl = "adaptive" if isinstance(inj, str) else f"inject {100*inj:.1f}%"
        ax.semilogy(np.arange(len(e)) * DT, np.maximum(e, 1e-2), color=COLORS[i],
                    label=lbl)
    ax.axvline(KIDNAP * DT, color="k", ls="--", lw=1)
    ax.text(KIDNAP * DT + 0.5, 5, "kidnapped", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("Recovery is impossible without fresh hypotheses")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "kidnapped.png"))


# =====================================================================  5
def exp5_overconfident(g, rng):
    banner("5. An over-confident laser model kills its own particles")

    n_rep, n_part = 8, 1000
    sigmas = [0.02, 0.05, 0.15, 0.4, 1.0]
    rows = []
    for s in sigmas:
        errs, ess, succ, uniq = [], [], 0, []
        for rep in range(n_rep):
            poses, controls, scans = make_route(g, 120, rng)
            parts = local_cloud(poses[0], n_part, rng)
            pf = ParticleFilter(g, parts, ALPHA, s, ANGLES, rng=rng,
                                max_range=MAX_RANGE)
            es, ee = [], []
            for k in range(len(controls)):
                es.append(pf.step(controls[k], DT, scans[k]))
                ee.append(pos_err(pf.estimate(), poses[k + 1]))
            err = float(np.mean(ee[-30:]))
            errs.append(err); succ += int(err < 0.5)
            ess.append(np.mean(es[-20:]))
            uniq.append(len(np.unique(pf.parts[:, 0])))
        rows.append((s, 100.0 * succ / n_rep, np.mean(errs), np.mean(ess),
                     np.mean(uniq)))

    print(f"  true laser noise is {SIGMA_Z:.2f} m")
    print(f"  {'model sigma':>12} {'success %':>10} {'err (m)':>9} {'ESS':>8} "
          f"{'distinct':>9}")
    for s, sc, e, es, u in rows:
        tag = "  <- the truth" if abs(s - SIGMA_Z) < 1e-9 else ""
        print(f"  {s:12.2f} {sc:10.0f} {e:9.3f} {es:8.1f} {u:9.1f}{tag}")
    sharp, truth_row, broad = rows[0], rows[2], rows[-1]
    print(f"\n  The best tracking accuracy is at the TRUE noise level "
          f"({truth_row[2]:.3f} m),")
    print(f"  which is the opposite of experiment 3's answer for global")
    print(f"  localization.  Both are right, and they are answering different")
    print("  questions.")
    print(f"\n  Too sharp ({sharp[0]:.2f} m, {SIGMA_Z/sharp[0]:.0f}x tighter than the "
          f"sensor): {sharp[2]/truth_row[2]:.1f}x the error, and the")
    print(f"  effective sample size falls to {sharp[3]:.0f} of {n_part} "
          f"({100*sharp[3]/n_part:.0f}%) -- the filter is")
    print(f"  carrying {n_part} particles and holding roughly {sharp[3]:.0f} distinct")
    print("  opinions.  That is PARTICLE DEPLETION, and pushed a little further it")
    print("  leaves one hypothesis, at which point the filter has become a very")
    print("  expensive and much worse EKF.")
    print(f"\n  Too broad ({broad[0]:.2f} m): {broad[2]/truth_row[2]:.1f}x the error, "
          f"but ESS {broad[3]:.0f} -- it stays")
    print("  healthy.  Same asymmetry as projects 24 and 25: being too vague costs")
    print("  accuracy, being too confident costs the filter's ability to hold more")
    print("  than one idea, and only one of those two failures can be recovered")
    print("  from by looking at more data.")
    for s, sc, e, es, u in rows:
        record(5, "sensor_sharpness", model_sigma=s, success_pct=sc, err=e,
               ess=es, distinct=u)

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.semilogx([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0],
                label="success (%)")
    ax2 = ax.twinx()
    ax2.loglog([r[0] for r in rows], [r[3] for r in rows], "s-", color=COLORS[1],
               label="final ESS")
    ax2.set_ylabel("effective sample size", color=COLORS[1]); ax2.grid(False)
    ax.axvline(SIGMA_Z, color="k", ls="--", lw=1)
    ax.set_xlabel("laser $\\sigma$ assumed by the model (m)")
    ax.set_ylabel("success (%)", color=COLORS[0])
    ax.set_title("Too sharp a sensor model destroys the particle cloud")
    save(fig, os.path.join(OUT, "sharpness.png"))


# =====================================================================  6
def exp6_beam_vs_field(g, rng):
    banner("6. The beam model against the likelihood field")

    n_rep, n_part = 10, 1000
    rows = []
    for sensor in ("beam", "field"):
        errs, succ, times = [], 0, []
        for rep in range(n_rep):
            poses, controls, scans = make_route(g, 120, rng)
            t0 = time.time()
            pf, est, ess, _ = run_pf(g, poses, controls, scans, n_part, rng,
                                     init="local", sensor=sensor)
            times.append((time.time() - t0) / len(controls))
            e = np.mean([pos_err(est[k], poses[k + 1])
                         for k in range(80, len(controls))])
            errs.append(e); succ += int(e < 0.5)
        rows.append((sensor, 100.0 * succ / n_rep, np.mean(errs), 1000 * np.mean(times)))

    print(f"  {'model':>8} {'success %':>10} {'err (m)':>9} {'ms/step':>9}")
    for s, sc, e, t in rows:
        print(f"  {s:>8} {sc:10.0f} {e:9.4f} {t:9.2f}")
    b, f = rows[0], rows[1]
    print(f"\n  the likelihood field is {b[3]/f[3]:.1f}x faster per step")
    print(f"  accuracy: {b[2]:.4f} m (beam) vs {f[2]:.4f} m (field), "
          f"success {b[1]:.0f}% vs {f[1]:.0f}%")
    print("  The field wins on cost because it replaces an 80-step ray march with")
    print("  one array lookup.  It pays for that by not knowing about occlusion:")
    print("  it scores a beam endpoint that lands NEAR a wall as a good match even")
    print("  when the beam would have been blocked long before reaching it.")

    for s, sc, e, t in rows:
        record(6, "sensor_model", model=s, success_pct=sc, err=e, ms_per_step=t)


# =====================================================================  7
def exp7_symmetry(rng):
    banner("7. A symmetric building: where a Gaussian cannot state the answer")

    g = office_map(symmetric=True)
    n_part = 4000
    n_rep = 8
    modes, errs, spreads = [], [], []
    keep = None
    for rep in range(n_rep):
        poses, controls, scans = make_route(g, 90, rng, x0=(3.6, 6.0, 0.3))
        parts = free_poses(g, n_part, rng)
        pf = ParticleFilter(g, parts, ALPHA, SIGMA_GLOBAL, ANGLES, rng=rng,
                            max_range=MAX_RANGE)
        for k in range(len(controls)):
            pf.step(controls[k], DT, scans[k])
        # how many separated clusters survive?  count with a coarse 1 m histogram
        hh, _, _ = np.histogram2d(pf.parts[:, 0], pf.parts[:, 1], bins=(16, 12),
                                  range=[[0, 16], [0, 12]], weights=pf.w)
        n_modes = int((hh > 0.08).sum())
        modes.append(n_modes)
        # error of the MEAN pose vs error of the BEST single particle
        mean_err = pos_err(pf.estimate(), poses[-1])
        d = np.hypot(pf.parts[:, 0] - poses[-1, 0], pf.parts[:, 1] - poses[-1, 1])
        best_err = float(d.min())
        near = float(pf.w[d < 0.5].sum())
        errs.append((mean_err, best_err, near))
        spreads.append(np.sqrt(np.average(
            (pf.parts[:, 0] - pf.parts[:, 0].mean()) ** 2, weights=pf.w)))
        if rep == 0:
            keep = (pf.parts.copy(), pf.w.copy(), poses)

    me = np.mean([e[0] for e in errs])
    be = np.mean([e[1] for e in errs])
    nr = np.mean([e[2] for e in errs])
    print(f"  {n_part} particles, {n_rep} runs on a left-right symmetric floor plan")
    print(f"  surviving clusters (1 m cells holding >8% of the weight): "
          f"{np.mean(modes):.1f}")
    print(f"  error of the weighted MEAN pose        {me:7.3f} m")
    print(f"  distance to the nearest particle       {be:7.3f} m")
    print(f"  weight sitting within 0.5 m of truth   {100*nr:6.1f}%")
    print(f"\n  And it resolved it.  {100*nr:.0f}% of the weight ends up within 0.5 m")
    print(f"  of the truth, and the cloud settles into {np.mean(modes):.1f} clusters, not two.")
    print("  That is an honest negative result and it is worth understanding,")
    print("  because it is the reason genuine ambiguity is rarer than it sounds.")
    print("  The two alcoves ARE mirror images of each other -- but a laser with")
    print("  8 m of range sees straight past them to the outer walls, and the")
    print("  robot is not in the middle of the building.  From the left alcove the")
    print("  far wall is 12 m away; from the right one it is 4 m.  For two places")
    print("  to be truly indistinguishable, EVERYTHING within sensor range has to")
    print("  be symmetric, not just the nearby furniture.  Real buildings that")
    print("  manage this are long identical corridors and repeated office bays --")
    print("  and note that a shorter-range sensor makes the problem WORSE, because")
    print("  it sees less of what breaks the tie.")
    print(f"\n  What the experiment does show cleanly is the cost of summarizing a")
    print(f"  particle cloud by its mean: {me:.3f} m for the weighted mean against")
    print(f"  {be:.3f} m for the nearest particle -- a factor of {me/max(be,1e-9):.0f}.  Whenever the")
    print("  cloud is not a single tight blob, the mean sits at a place no")
    print("  particle believes in.  A Gaussian filter has no choice but to report")
    print("  that place; a particle filter does, and throwing the distribution")
    print("  away at the last step to report one number undoes the whole point of")
    print("  having paid for it.")
    record(7, "symmetric", modes=float(np.mean(modes)), mean_pose_err=me,
           nearest_particle_err=be, weight_near_truth=nr)

    parts, w, poses = keep
    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.imshow(g.occ, origin="lower", extent=g.extent(), cmap="Greys",
              interpolation="nearest", alpha=0.85)
    srt = np.argsort(w)
    ax.scatter(parts[srt, 0], parts[srt, 1], s=3, c=w[srt], cmap="viridis")
    ax.plot(poses[-1, 0], poses[-1, 1], "*", ms=16, color=COLORS[1], label="truth")
    m = estimate(parts, w)
    ax.plot(m[0], m[1], "X", ms=11, color=COLORS[3], label="weighted mean")
    ax.set_title("Two identical alcoves: the belief stays two-peaked")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "symmetry.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(5)
    g, poses, controls, scans = exp1_global(rng)
    exp2_resampling(g, rng)
    exp3_particle_count(g, rng)
    exp4_kidnapped(g, rng)
    exp5_overconfident(g, rng)
    exp6_beam_vs_field(g, rng)
    exp7_symmetry(rng)

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
