"""Five experiments on the managed lifecycle.

  1. cold start        -- what a node publishes before it is ready
  2. outage sweep      -- availability and contamination against outage length
  3. detection         -- how long before anybody knows
  4. bringing up four nodes at once
  5. error policy      -- the knob that fixes experiment 2's bad news

Everything is a deterministic millisecond-clock simulation, so it finishes in
under a second and every number is exactly reproducible.
"""

import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lifecycle as L
from lifecycle import (ACTIVE, Camera, Consumer, INACTIVE, ManagedPerception,
                       Supervisor, UnmanagedPerception, run_trial)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


# ---------------------------------------------------------------------------
# 1. cold start
# ---------------------------------------------------------------------------
def exp1_coldstart():
    print("\n=== 1. cold start: what goes out before the node is ready " + "=" * 14)
    print("  node        good  uncalibrated  stale   availability")
    for kind in ("unmanaged", "managed"):
        r = run_trial(kind)
        print("  %-10s %5d %13d %6d %13.3f"
              % (kind, r["good"], r["uncalibrated"], r["stale"],
                 r["availability"]))
        record("coldstart", **r)
    print("  -> the calibration file takes 200 ms to load.  Both nodes are")
    print("     alive from t=0; only one of them keeps quiet about it.")


# ---------------------------------------------------------------------------
# 2. the outage sweep
# ---------------------------------------------------------------------------
DURATIONS = [20, 40, 80, 150, 300, 600, 1200, 2000]
# The outage's phase relative to the 33 ms frame clock changes the answer by a
# frame or two.  Average over a whole frame period so the curves show the
# effect and not the alignment luck.
PHASES = range(0, L.FRAME_PERIOD, 3)


def avg_trial(kind, outage_ms, **kw):
    rs = [run_trial(kind, [(2000 + p, 2000 + p + outage_ms)], **kw)
          for p in PHASES]
    out = {k: float(np.mean([r[k] for r in rs]))
           for k in ("good", "uncalibrated", "stale", "bad", "availability",
                     "belief_wrong_ms", "detect_ms", "recoveries")}
    out["kind"] = kind
    return out


def exp2_outage():
    print("\n=== 2. one disconnect, of varying length " + "=" * 31)
    print("  outage    unmanaged: avail  uncal  stale     managed: avail  bad")
    curves = {}
    for kind in ("unmanaged", "managed"):
        curves[kind] = ([], [])
    for d in DURATIONS:
        rs = {}
        for kind in ("unmanaged", "managed"):
            r = avg_trial(kind, d)
            rs[kind] = r
            curves[kind][0].append(r["availability"])
            curves[kind][1].append(r["bad"])
            record("outage", outage_ms=d, **r)
        print("  %5d ms %16.3f %6.1f %6.1f %14.3f %4.1f"
              % (d, rs["unmanaged"]["availability"],
                 rs["unmanaged"]["uncalibrated"], rs["unmanaged"]["stale"],
                 rs["managed"]["availability"], rs["managed"]["bad"]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for kind, col in (("unmanaged", "#c62828"), ("managed", "#2e7d32")):
        ax[0].plot(DURATIONS, curves[kind][0], "o-", color=col, label=kind)
        ax[1].plot(DURATIONS, curves[kind][1], "o-", color=col, label=kind)
    ax[0].set_ylabel("availability (good frames / expected)")
    ax[1].set_ylabel("frames delivered that were WRONG")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("outage length (ms)")
        a.legend(fontsize=8); a.grid(alpha=.3)
    ax[0].set_title("who delivers more frames")
    ax[1].set_title("who delivers more lies")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "outage_sweep.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. how long before anybody knows
# ---------------------------------------------------------------------------
def exp3_detection():
    print("\n=== 3. detection: how long before the system knows " + "=" * 22)
    print("  node        detect (ms)   ms with a wrong belief about health")
    for kind in ("unmanaged", "managed"):
        r = run_trial(kind, [(2000, 3000)])
        print("  %-10s %11d %38d"
              % (kind, r["detect_ms"], r["belief_wrong_ms"]))
        record("detection", **r)
    print("  -> the unmanaged node is never silent, so the only way to notice")
    print("     is a timeout.  A timeout you can shorten costs false alarms.")


# ---------------------------------------------------------------------------
# 4. four nodes, one system
# ---------------------------------------------------------------------------
def fuse_trial(kind, seed, n_nodes=4, T=3000, fuse_period=33):
    """Four sensor nodes feed one fuser.  The fuser needs all four to be
    correct and it cannot tell whether they are.

    Each node takes a different, unpredictable time to get ready -- a
    calibration file on a slow SD card, a device that enumerates late, a
    network mount.  ``ros2 launch`` can order process *starts*; it cannot order
    *readiness*, because only the node knows when it is ready.

    Every ``fuse_period`` the fuser is classified:
      * WRONG   -- it produced an answer, and at least one input was
                   uncalibrated.  This is the dangerous one: it looks fine.
      * ABSENT  -- it produced nothing, because some input had not spoken yet.
      * GOOD    -- an answer, from four good inputs.
    """
    rng = random.Random(seed)
    cfgs = [rng.randint(60, 900) for _ in range(n_nodes)]
    cams = [Camera() for _ in range(n_nodes)]
    outs = [Consumer() for _ in range(n_nodes)]
    if kind == "managed":
        nodes = [ManagedPerception(c, configure_ms=g) for c, g in zip(cams, cfgs)]
        sups = [Supervisor(n, auto_activate=False) for n in nodes]
    else:
        nodes = [UnmanagedPerception(c, configure_ms=g)
                 for c, g in zip(cams, cfgs)]
        sups = []

    good = wrong = absent = 0
    for t in range(T):
        for s in sups:
            s.tick(t)
        if kind == "managed":
            # the coordinator's rule: nobody goes ACTIVE until EVERYBODY has
            # reached INACTIVE.  This two-phase bring-up is exactly why
            # "configure" and "activate" are two transitions and not one.
            ready = all(n.state in (INACTIVE, ACTIVE) for n in nodes)
            if ready:
                for n in nodes:
                    if n.state == INACTIVE and n.pending is None:
                        n.request("activate", t)
        for n, o in zip(nodes, outs):
            n.tick(t, o)
        if t % fuse_period == 0:
            kinds = [o.last_kind for o in outs]
            if any(k is None for k in kinds):
                absent += 1
            elif any(k != "good" for k in kinds):
                wrong += 1
            else:
                good += 1
    return good, wrong, absent


def exp4_bringup():
    print("\n=== 4. bringing up four nodes at once " + "=" * 34)
    print("  (60 random bring-ups; each node gets ready after 60-900 ms)")
    print("  node          good    silently WRONG    absent    %% wrong")
    for kind in ("unmanaged", "managed"):
        g = w = a = 0
        for seed in range(60):
            x, y, z = fuse_trial(kind, seed)
            g += x; w += y; a += z
        tot = g + w
        print("  %-12s %6d %16d %9d %10.1f"
              % (kind, g, w, a, 100 * w / max(tot, 1)))
        record("bringup", node=kind, good=g, wrong=w, absent=a,
               pct_wrong=100 * w / max(tot, 1))
    print("  -> the managed system is SILENT while it is not ready.")
    print("     The unmanaged one answers, and the answer is wrong.")


# ---------------------------------------------------------------------------
# 5. the knob that fixes experiment 2
# ---------------------------------------------------------------------------
def exp5_policy():
    print("\n=== 5. error policy: how twitchy should the node be? " + "=" * 19)
    tolerances = [0, 1, 3, 6, 12]
    print("  outage   " + "  ".join("tol=%-2d" % k for k in tolerances)
          + "   unmanaged")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    table = {k: [avg_trial("managed", d, tolerate=k)["availability"]
                 for d in DURATIONS] for k in tolerances}
    unm = [avg_trial("unmanaged", d)["availability"] for d in DURATIONS]
    for k in tolerances:
        ax.plot(DURATIONS, table[k], "o-", label="managed, tolerate %d" % k)
        for d, v in zip(DURATIONS, table[k]):
            record("policy", tolerate=k, outage_ms=d, availability=v)
    ax.plot(DURATIONS, unm, "s--", color="k", label="unmanaged")
    for i, d in enumerate(DURATIONS):
        print("  %5d ms  " % d + "  ".join("%.3f" % table[k][i]
                                           for k in tolerances)
              + "   %.3f" % unm[i])
        record("policy", tolerate=-1, outage_ms=d, availability=unm[i])
    ax.set_xscale("log"); ax.set_xlabel("outage length (ms)")
    ax.set_ylabel("availability")
    ax.set_title("Debouncing the error, without ever publishing a wrong frame")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_policy.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# a state-machine picture, drawn from the code so it cannot drift
# ---------------------------------------------------------------------------
def draw_machine():
    pos = {"unconfigured": (0, 0), "inactive": (2, 0), "active": (4, 0),
           "configuring": (1, 1), "activating": (3, 1), "cleaningup": (1, -1),
           "deactivating": (3, -1), "errorprocessing": (2, -2),
           "finalized": (4, -2)}
    fig, ax = plt.subplots(figsize=(8, 4))
    edges = [("unconfigured", "configuring"), ("configuring", "inactive"),
             ("inactive", "activating"), ("activating", "active"),
             ("active", "deactivating"), ("deactivating", "inactive"),
             ("inactive", "cleaningup"), ("cleaningup", "unconfigured"),
             ("active", "errorprocessing"), ("configuring", "errorprocessing"),
             ("errorprocessing", "unconfigured"),
             ("errorprocessing", "finalized")]
    for a, b in edges:
        x1, y1 = pos[a]; x2, y2 = pos[b]
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.3,
                                    shrinkA=31, shrinkB=31))
    for name, (x, y) in pos.items():
        primary = name in L.PRIMARY
        ax.scatter([x], [y], s=2600, zorder=3,
                   color="#1976d2" if primary else "#eceff1",
                   edgecolors="#455a64")
        ax.text(x, y, name.replace("processing", "\nprocessing"),
                ha="center", va="center", fontsize=7, zorder=4,
                color="white" if primary else "#263238")
    ax.set_xlim(-1, 5.2); ax.set_ylim(-3, 2)
    ax.axis("off")
    ax.set_title("The managed lifecycle: blue = a state you rest in,\n"
                 "grey = a state you are only passing through", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "state_machine.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    draw_machine()
    exp1_coldstart()
    exp2_outage()
    exp3_detection()
    exp4_bringup()
    exp5_policy()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
