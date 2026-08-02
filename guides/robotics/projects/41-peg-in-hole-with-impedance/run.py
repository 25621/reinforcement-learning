"""Project 41 -- peg-in-hole: why a stiff robot cannot do assembly.

Seven experiments:

  1. one insertion, from first touch to the bottom of the hole
  2. stiff position control vs impedance control
  3. how tight a hole you can still fill
  4. lateral stiffness: the search only works if it can out-pull friction
  5. no search / blind sweep / sweep that stops when it feels the drop
  6. the compliance centre, and why the 1980s bolted rubber to a wrist
  7. the chamfer: two millimetres of machining beats any controller

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
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from peg import (MU, Hole, Impedance, Peg, clearance, simulate)               # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.patches import Polygon as MplPolygon                          # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

PEG = Peg()
HOLE = Hole()
NOM = dict(kx=800.0, kz=1600.0, kth=1.0, cc=0.030, f_push=12.0)
# Experiment 6 deliberately softens the angular spring to KTH_SOFT.  A stiff
# angular servo straightens the peg on its own and hides the effect that
# experiment is measuring; a real RCC wrist really is this floppy in rotation.
KTH_SOFT = 0.15
STIFF = dict(kx=2.0e4, kz=2.0e4, kth=200.0, cc=0.0, f_push=1e9)
AMP = 0.012


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<50s} {value}")


def ctrl(**kw):
    d = dict(NOM)
    d.update(kw)
    return Impedance(mass=PEG.m, inertia=PEG.I, **d)


def run(x0=0.0, tilt0=0.0, strategy="triggered", hole=None, **kw):
    return simulate(PEG, hole or HOLE, ctrl(**kw), x0=x0, tilt0=tilt0,
                    strategy=strategy)


def sweep_offsets(offsets, **kw):
    return [run(x0=x, **kw) for x in offsets]


def draw_scene(ax, hole, q=None, peg=PEG, alpha=1.0):
    for P in hole.plates:
        ax.add_patch(MplPolygon(1000 * P, closed=True, fc="#b9c4cf",
                                ec="#42505e", lw=1.0))
    if q is not None:
        ax.add_patch(MplPolygon(1000 * peg.poly(q), closed=True, fc=COLORS[1],
                                ec="#7a3600", lw=1.0, alpha=alpha))


# ---------------------------------------------------------------------------
# 1. one insertion
# ---------------------------------------------------------------------------

def exp1_trace():
    print("\n[1] one insertion, traced")
    tr = run(x0=0.006, strategy="triggered")
    record("1_trace", "clearance (mm)", round(1000 * clearance(PEG, HOLE), 3))
    record("1_trace", "clearance ratio (clearance / peg half-width)",
           round(clearance(PEG, HOLE) / (PEG.w / 2), 4))
    record("1_trace", "start offset (mm)", 6.0)
    record("1_trace", "inserted", tr["success"])
    record("1_trace", "time to insert (s)", round(float(tr["t_insert"]), 2))
    record("1_trace", "peak contact force (N)", round(tr["peak_force"], 1))

    fig = plt.figure(figsize=(11.0, 3.6))
    ax = fig.add_subplot(1, 3, 1)
    for k, alpha in zip(np.linspace(0, len(tr["q"]) - 1, 5).astype(int),
                        [0.25, 0.35, 0.5, 0.7, 1.0]):
        ax.add_patch(MplPolygon(1000 * PEG.poly(tr["q"][k]), closed=True,
                                fc=COLORS[1], ec="none", alpha=alpha))
    draw_scene(ax, HOLE)
    ax.set_xlim(-35, 35); ax.set_ylim(-25, 55)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title("five moments of one insertion", fontsize=9)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(tr["t"], 1000 * tr["q"][:, 0], label="peg x (mm)")
    ax.plot(tr["t"], 1000 * tr["depth"], label="insertion depth (mm)")
    ax.axhline(0, color="#8C8C8C", lw=0.8)
    ax.set_xlabel("time (s)"); ax.legend(fontsize=8)
    ax.set_title("the sweep finds the hole, then stops", fontsize=9)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(tr["t"], tr["fz"], label="contact force, up (N)")
    ax.plot(tr["t"], tr["fx"], label="contact force, sideways (N)")
    ax.set_xlabel("time (s)"); ax.legend(fontsize=8)
    ax.set_title(f"peak {tr['peak_force']:.0f} N", fontsize=9)
    save(fig, os.path.join(OUT, "trace.png"))


# ---------------------------------------------------------------------------
# 2. stiff vs compliant
# ---------------------------------------------------------------------------

def exp2_stiff():
    print("\n[2] stiff position control vs impedance control")
    offs = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010])
    rows = {}
    rows["position control (stiff, no search)"] = sweep_offsets(
        offs, strategy="straight", **STIFF)
    rows["impedance, no search"] = sweep_offsets(offs, strategy="straight")
    rows["impedance + sweep search"] = sweep_offsets(offs, strategy="triggered")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for name, trs in rows.items():
        axes[0].plot(1000 * offs, [100 * t["success"] for t in trs], "o-", ms=4,
                     label=name)
        axes[1].plot(1000 * offs, [t["peak_force"] for t in trs], "o-", ms=4,
                     label=name)
        record("2_stiff", f"{name}: inserted",
               f"{sum(t['success'] for t in trs)}/{len(trs)}")
        record("2_stiff", f"{name}: peak force over all offsets (N)",
               round(max(t["peak_force"] for t in trs), 1))
    axes[0].set_ylabel("inserted (%)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("peak contact force (N)")
    axes[1].axhline(50, color="#8C8C8C", ls=":", lw=1)
    axes[1].text(0.2, 55, "50 N: a plastic connector starts to break", fontsize=7)
    for ax in axes:
        ax.set_xlabel("lateral offset of the hole (mm)")
    axes[0].legend(fontsize=7.5)
    save(fig, os.path.join(OUT, "stiff.png"))


# ---------------------------------------------------------------------------
# 3. clearance
# ---------------------------------------------------------------------------

def exp3_clearance():
    print("\n[3] how tight a hole, and how crooked a grasp")
    widths = [0.0202, 0.0204, 0.0210, 0.0220, 0.0240]
    tilts = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.045])
    grid = np.zeros((len(widths), len(tilts)))
    for i, w in enumerate(widths):
        h = Hole(width=w)
        for j, a in enumerate(tilts):
            grid[i, j] = run(x0=0.004, tilt0=a, hole=h)["success"]
        c = clearance(PEG, h)
        ok = [a for a, v in zip(tilts, grid[i]) if v]
        record("3_clearance", f"clearance {1000 * c:.2f} mm: worst crooked grasp "
                              f"still inserted (deg)",
               round(float(np.degrees(max(ok))), 2) if ok else 0.0)
        record("3_clearance", f"  ... geometric limit 2c/t (deg)",
               round(float(np.degrees(2 * c / h.t)), 2))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    im = axes[0].imshow(grid, origin="lower", cmap="RdYlGn", aspect="auto",
                        extent=[np.degrees(tilts[0]) - 0.15,
                                np.degrees(tilts[-1]) + 0.15, -0.5,
                                len(widths) - 0.5], vmin=0, vmax=1)
    axes[0].set_yticks(range(len(widths)))
    axes[0].set_yticklabels([f"{1000 * clearance(PEG, Hole(width=w)):.2f}"
                             for w in widths])
    axes[0].set_ylabel("clearance per side (mm)")
    axes[0].set_xlabel("how crooked the peg is held (deg)")
    axes[0].set_title("green = inserted", fontsize=9)
    axes[0].grid(False)
    cs = [clearance(PEG, Hole(width=w)) for w in widths]
    meas = []
    for i, c in enumerate(cs):
        ok = [a for a, v in zip(tilts, grid[i]) if v]
        meas.append(np.degrees(max(ok)) if ok else 0.0)
    axes[1].plot(1000 * np.array(cs), meas, "o-", ms=5, label="measured")
    axes[1].plot(1000 * np.array(cs),
                 [np.degrees(2 * c / HOLE.t) for c in cs], "s--", ms=4,
                 label="geometric limit 2 x clearance / depth")
    axes[1].set_xlabel("clearance per side (mm)")
    axes[1].set_ylabel("largest crooked grasp that still inserts (deg)")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "clearance.png"))


# ---------------------------------------------------------------------------
# 4. lateral stiffness
# ---------------------------------------------------------------------------

def exp4_stiffness():
    print("\n[4] lateral stiffness vs the friction holding the peg down")
    kxs = [200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
    offs = np.arange(0.0, 0.0141, 0.001)
    reach, peak, lag = [], [], []
    for kx in kxs:
        trs = sweep_offsets(offs, kx=kx)
        ok = [o for o, t in zip(offs, trs) if t["success"]]
        reach.append(1000 * max(ok) if ok else 0.0)
        peak.append(max(t["peak_force"] for t in trs))
        # how far behind its own command does the peg run while it is pressed?
        L = []
        for t in trs:
            touching = t["f"] > 1.0
            if touching.any():
                L.append(np.abs(t["ref"][touching, 0] -
                                t["q"][touching, 0]).max())
        lag.append(1000 * float(np.mean(L)) if L else 0.0)
        record("4_stiffness", f"kx={kx:.0f} N/m: largest offset recovered (mm)",
               round(reach[-1], 1))
        record("4_stiffness", f"kx={kx:.0f} N/m: measured lag while pressed (mm)",
               round(lag[-1], 2))
        record("4_stiffness", f"kx={kx:.0f} N/m: predicted lag mu*f_push/kx (mm)",
               round(1000 * MU * NOM["f_push"] / kx, 2))
    pred = [1000 * max(AMP - MU * NOM["f_push"] / kx, 0.0) for kx in kxs]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.3))
    axes[0].plot(kxs, lag, "o-", ms=5, label="measured")
    axes[0].plot(kxs, [1000 * MU * NOM["f_push"] / kx for kx in kxs], "s--", ms=4,
                 label="mu * f_push / kx")
    axes[0].set_ylabel("lag behind the command while pressed (mm)")
    axes[0].set_title("friction drags the peg behind its own reference", fontsize=9)
    axes[1].plot(kxs, reach, "o-", ms=5, label="measured")
    axes[1].plot(kxs, pred, "s--", ms=4, label="amplitude - lag")
    axes[1].set_ylabel("largest offset recovered (mm)")
    axes[2].plot(kxs, peak, "o-", ms=5, color=COLORS[1])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("peak contact force (N)")
    axes[2].set_title("and what stiffness costs on impact", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("lateral stiffness kx (N/m)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "stiffness.png"))
    record("4_stiffness", "sweep amplitude (mm)", 1000 * AMP)


# ---------------------------------------------------------------------------
# 5. search strategies
# ---------------------------------------------------------------------------

def exp5_search():
    print("\n[5] three search strategies")
    offs = np.array([0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
    names = {"straight": "no search", "sweep": "sweep, always",
             "triggered": "sweep until it feels the drop"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for s, label in names.items():
        trs = sweep_offsets(offs, strategy=s)
        axes[0].plot(1000 * offs, [100 * t["success"] for t in trs], "o-", ms=4,
                     label=label)
        tt = [t["t_insert"] if t["success"] else np.nan for t in trs]
        axes[1].plot(1000 * offs, tt, "o-", ms=4, label=label)
        ok = [t for t in trs if t["success"]]
        record("5_search", f"{label}: inserted", f"{len(ok)}/{len(trs)}")
        record("5_search", f"{label}: mean time when it works (s)",
               round(float(np.mean([t['t_insert'] for t in ok])), 2) if ok else "-")
        record("5_search", f"{label}: mean peak force (N)",
               round(float(np.mean([t['peak_force'] for t in trs])), 1))
    axes[0].set_ylabel("inserted (%)")
    axes[1].set_ylabel("time to insert (s)")
    for ax in axes:
        ax.set_xlabel("lateral offset (mm)")
        ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "search.png"))


# ---------------------------------------------------------------------------
# 6. the compliance centre
# ---------------------------------------------------------------------------

def exp6_rcc():
    print("\n[6] where the spring is attached")
    ccs = [-0.060, -0.030, 0.0, 0.015, 0.030, 0.045]
    offs = np.arange(0.0, 0.0101, 0.001)
    tilts = np.array([0.0, 0.005, 0.010, 0.020, 0.030, 0.040])
    ok_off, ok_tilt, maxrot = [], [], []
    for cc in ccs:
        trs = sweep_offsets(offs, cc=cc, kth=KTH_SOFT)
        ok_off.append(np.mean([t["success"] for t in trs]))
        maxrot.append(max(np.degrees(np.abs(t["q"][:, 2])).max() for t in trs))
        tt = [run(x0=0.004, tilt0=a, cc=cc, kth=KTH_SOFT) for a in tilts]
        ok_tilt.append(np.mean([t["success"] for t in tt]))
        record("6_rcc", f"compliance centre {1000 * cc:+.0f} mm below the peg "
                        f"centre: inserted (offset sweep)",
               f"{int(sum(t['success'] for t in trs))}/{len(trs)}")
        record("6_rcc", f"  ... worst tilt reached (deg)", round(maxrot[-1], 1))
        record("6_rcc", f"  ... inserted (crooked-grasp sweep)",
               f"{int(sum(t['success'] for t in tt))}/{len(tt)}")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    x = 1000 * np.array(ccs)
    axes[0].plot(x, 100 * np.array(ok_off), "o-", ms=5, label="offset sweep")
    axes[0].plot(x, 100 * np.array(ok_tilt), "s-", ms=5, label="crooked grasp")
    axes[0].axvline(1000 * PEG.L / 2, color="#8C8C8C", ls=":", lw=1)
    axes[0].text(1000 * PEG.L / 2 - 1, 8, "the peg tip", rotation=90, fontsize=7,
                 ha="right")
    axes[0].set_ylabel("inserted (%)")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, maxrot, "o-", ms=5, color=COLORS[1])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("worst peg tilt during the attempt (deg)")
    for ax in axes:
        ax.set_xlabel("compliance centre, mm below the peg's centre of mass")
    save(fig, os.path.join(OUT, "rcc.png"))


# ---------------------------------------------------------------------------
# 7. the chamfer
# ---------------------------------------------------------------------------

def exp7_chamfer():
    print("\n[7] the chamfer")
    chs = [0.0, 0.0005, 0.0015, 0.003, 0.005]
    offs = np.arange(0.0, 0.0071, 0.0005)
    reach = []
    for ch in chs:
        h = Hole(chamfer=ch)
        trs = sweep_offsets(offs, strategy="straight", hole=h)
        ok = [o for o, t in zip(offs, trs) if t["success"]]
        reach.append(1000 * max(ok) if ok else 0.0)
        record("7_chamfer", f"chamfer {1000 * ch:.1f} mm: capture range with NO "
                            f"search (mm)", round(reach[-1], 2))
    pred = [1000 * (c + clearance(PEG, HOLE)) for c in chs]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(1000 * np.array(chs), reach, "o-", ms=5, label="measured")
    ax.plot(1000 * np.array(chs), pred, "s--", ms=4,
            label="predicted: chamfer + clearance")
    ax.set_xlabel("chamfer on the hole (mm)")
    ax.set_ylabel("offset absorbed with no search at all (mm)")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "chamfer.png"))


def main():
    use_style()
    t0 = time.time()
    exp1_trace()
    exp2_stiff()
    exp3_clearance()
    exp4_stiffness()
    exp5_search()
    exp6_rcc()
    exp7_chamfer()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
