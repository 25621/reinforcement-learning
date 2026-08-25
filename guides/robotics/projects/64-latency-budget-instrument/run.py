"""Measure the latency of a real pipeline, then find out what it costs.

  1. the budget      -- per-stage wait and work, and the end-to-end truth
  2. rate mismatch   -- latency you pay for doing nothing
  3. queue policy    -- under overload, the default choice is the worst one
  4. the tail        -- p50 is a comfort, p99 is the robot's actual experience
  5. the consequence -- tracking error and stability against dead time

Takes about 90 seconds, most of it deliberately spent waiting in real time.
"""

import os

# MUST come before numpy is imported anywhere.  A multi-threaded BLAS turns a
# 64x64 matmul into a thread-barrier benchmark: measured 7.2 ms per call with
# 12 threads versus 0.05 ms with one.  Every stage's "work" would then be
# quantised to 7 ms and the whole instrument would measure OpenMP.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline as P
from pipeline import DEFAULT, e2e_ms, pct, run_pipeline, stage_breakdown

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


# ---------------------------------------------------------------------------
# 1. the budget
# ---------------------------------------------------------------------------
def exp1_budget():
    print("\n=== 1. the photon-to-actuation budget " + "=" * 34)
    r = run_pipeline(10.0)
    e = e2e_ms(r["msgs"])
    bd = stage_breakdown(r["msgs"])

    print("  stage        wait p50   work p50   work p95    total p50")
    total_work = total_wait = 0.0
    for name, d in bd.items():
        w, k = np.median(d["wait"]), np.median(d["work"])
        total_wait += w
        total_work += k
        print("  %-12s %7.2f   %8.2f   %8.2f   %9.2f"
              % (name, w, k, pct(d["work"], 95), w + k))
        record("budget", stage=name, wait_p50=float(w), work_p50=float(k),
               work_p95=pct(d["work"], 95))
    print("  %-12s %7.2f   %8.2f" % ("TOTAL", total_wait, total_work))
    print()
    print("  sum of stage medians          : %6.2f ms" % (total_wait + total_work))
    print("  measured end-to-end p50       : %6.2f ms" % pct(e, 50))
    print("  of which actual computation   : %6.2f ms  (%.0f %%)"
          % (total_work, 100 * total_work / pct(e, 50)))
    print("  of which waiting              : %6.2f ms  (%.0f %%)"
          % (total_wait, 100 * total_wait / pct(e, 50)))
    print("  end-to-end p5 / p50 / p95 / max: %.1f / %.1f / %.1f / %.1f ms"
          % (pct(e, 5), pct(e, 50), pct(e, 95), e.max()))
    record("budget", stage="END-TO-END", p5=pct(e, 5), p50=pct(e, 50),
           p95=pct(e, 95), p99=pct(e, 99), maximum=float(e.max()),
           work_total=total_work, wait_total=total_wait,
           frames_captured=r["sent"], commands=len(e))

    # a stacked bar of where the time goes, plus the distribution
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))
    names = list(bd)
    waits = [np.median(bd[n]["wait"]) for n in names]
    works = [np.median(bd[n]["work"]) for n in names]
    y = np.arange(len(names))
    ax[0].barh(y, waits, color="#ef9a9a", label="waiting")
    ax[0].barh(y, works, left=waits, color="#1976d2", label="computing")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("milliseconds (median)")
    ax[0].set_title("where the %.0f ms goes" % pct(e, 50))
    ax[0].grid(alpha=.3, axis="x")

    ax[1].hist(e, bins=50, color="#455a64")
    for p, c in ((50, "#2e7d32"), (95, "#f9a825"), (99, "#c62828")):
        ax[1].axvline(pct(e, p), color=c, lw=2, label="p%d = %.0f ms" % (p, pct(e, p)))
    ax[1].set_xlabel("photon-to-actuation latency (ms)")
    ax[1].set_ylabel("commands"); ax[1].legend(fontsize=8)
    ax[1].set_title("the distribution, not the average")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "budget.png"), dpi=120)
    plt.close(fig)
    return e


# ---------------------------------------------------------------------------
# 2. rate mismatch
# ---------------------------------------------------------------------------
def exp2_rates():
    print("\n=== 2. a slow planner: take the newest, or take the next? " + "=" * 14)
    print("  (camera 30 Hz into a planner running slower; queue holds 10)")
    print("  drain    planner rate   plan wait p50   end-to-end p50   cmds/s")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for drain, col in (("latest", "#2e7d32"), ("fifo", "#c62828")):
        rates, lats = [], []
        for rate in (5, 10, 20, 30):
            r = run_pipeline(5.0, plan_rate=rate, plan_ms=2.0, hiccup=None,
                             maxsize=10, plan_drain=drain)
            e = e2e_ms(r["msgs"])
            bd = stage_breakdown(r["msgs"])
            w = float(np.median(bd["planner"]["wait"]))
            print("  %-8s %8d Hz %13.2f ms %14.2f ms %8.1f"
                  % (drain, rate, w, pct(e, 50), len(e) / 5.0))
            record("rates", drain=drain, plan_rate=rate, plan_wait_p50=w,
                   e2e_p50=pct(e, 50), e2e_p95=pct(e, 95), cmds_per_s=len(e) / 5.0)
            rates.append(rate); lats.append(pct(e, 50))
        ax.plot(rates, lats, "o-", color=col, label="take the %s" % drain)
    ax.axhline(1e3 / DEFAULT["cam_rate"], ls="--", color="#455a64",
               label="one camera period (%.0f ms)" % (1e3 / DEFAULT["cam_rate"]))
    ax.set_xlabel("planner rate (Hz)"); ax.set_ylabel("end-to-end p50 (ms)")
    ax.set_title("A queue you actually drain is latency you actually pay")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rate_mismatch.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. queue policy under overload
# ---------------------------------------------------------------------------
def exp3_queues():
    print("\n=== 3. queue policy when perception cannot keep up " + "=" * 22)
    print("  (camera 30 Hz = 33 ms per frame; perception needs 45 ms)")
    print("  policy         e2e p50   e2e p95     growth   delivered   dropped")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for policy, maxsize, col in (("unbounded", 10 ** 6, "#c62828"),
                                 ("drop_oldest", 2, "#2e7d32"),
                                 ("block", 2, "#1976d2")):
        r = run_pipeline(9.0, perc_ms=45.0, policy=policy, maxsize=maxsize,
                         hiccup=None)
        e = e2e_ms(r["msgs"])
        if len(e) < 4:
            continue
        t = np.array([m.t_capture for m in r["msgs"]])
        t = t - t.min()
        slope = float(np.polyfit(t, e, 1)[0])       # ms of latency per second
        print("  %-13s %7.1f %9.1f %8.1f/s %10d %9d  (captured %d)"
              % (policy, pct(e, 50), pct(e, 95), slope, len(e), r["dropped"],
                 r["sent"]))
        record("queues", policy=policy, e2e_p50=pct(e, 50), e2e_p95=pct(e, 95),
               growth_ms_per_s=slope, delivered=len(e), dropped=r["dropped"],
               captured=r["sent"])
        ax.plot(t, e, ".-", ms=3, lw=.8, color=col, label=policy)
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("photon-to-actuation latency (ms)")
    ax.set_title("Overloaded pipeline: an unbounded queue never recovers")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "queue_policy.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. where the tail comes from
# ---------------------------------------------------------------------------
def exp4_tail():
    print("\n=== 4. the tail " + "=" * 56)
    print("  configuration                p50     p95     p99     max")
    for name, over in (("as built (1 slow frame in 40)", {}),
                       ("no slow frames", {"hiccup": None}),
                       ("no slow frames, 12 Hz camera", {"hiccup": None,
                                                         "cam_rate": 12.0})):
        r = run_pipeline(10.0, **over)
        e = e2e_ms(r["msgs"])
        print("  %-28s %6.1f %7.1f %7.1f %7.1f"
              % (name, pct(e, 50), pct(e, 95), pct(e, 99), e.max()))
        record("tail", config=name, p50=pct(e, 50), p95=pct(e, 95),
               p99=pct(e, 99), maximum=float(e.max()),
               tail_ratio=e.max() / pct(e, 50))


# ---------------------------------------------------------------------------
# 5. what the latency costs
# ---------------------------------------------------------------------------
def tracking_error(delay_ms, kp=90.0, kd=14.0, T=12.0, dt=0.001):
    """A second-order plant chasing a moving target through a pure delay.

    "Dead time" is the control engineer's name for a delay in which nothing
    can be done: the measurement exists, the actuator exists, and the loop is
    simply blind for that long.  Feedback cannot correct an error it has not
    heard about yet, so the loop's usable gain falls as the delay grows.
    """
    n = int(T / dt)
    d = max(1, int(delay_ms * 1e-3 / dt))
    x = v = 0.0
    hist = np.zeros(d + 1)
    err = np.zeros(n)
    for k in range(n):
        t = k * dt
        ref = 0.35 * np.sin(2 * np.pi * 0.5 * t)
        dref = 0.35 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
        x_seen = hist[0]
        hist[:-1] = hist[1:]
        hist[-1] = x
        a = kp * (ref - x_seen) + kd * (dref - v)
        a = float(np.clip(a, -80, 80))
        v += dt * a
        x += dt * v
        err[k] = ref - x
        if not np.isfinite(x) or abs(x) > 10:
            return float("inf")
    return float(np.sqrt(np.mean(err[int(2 / dt):] ** 2)) * 1e3)   # mm


def exp5_cost(e):
    print("\n=== 5. what that latency costs the robot " + "=" * 31)
    delays = [0, 10, 20, 30, 44, 60, 77, 90, 112, 140, 180, 230, 290, 360, 440]
    errs = [tracking_error(d) for d in delays]
    for d, err in zip(delays, errs):
        record("cost", delay_ms=d, tracking_rms_mm=err)
    print("  delay (ms)   tracking RMS")
    for d, err in zip(delays, errs):
        print("  %8d     %s" % (d, "UNSTABLE" if not np.isfinite(err)
                                else "%8.2f mm" % err))
    for label, p in (("p50", 50), ("p95", 95), ("max", 100)):
        v = pct(e, p)
        print("  our measured %s = %5.1f ms -> %s" % (
            label, v, "UNSTABLE" if not np.isfinite(tracking_error(v))
            else "%.2f mm" % tracking_error(v)))
        record("cost", measured=label, delay_ms=v,
               tracking_rms_mm=tracking_error(v))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fin = [e_ for e_ in errs if np.isfinite(e_)]
    ax.plot(delays[:len(fin)], fin, "o-", color="#1976d2")
    for label, p, c in (("p50", 50, "#2e7d32"), ("p95", 95, "#f9a825"),
                        ("max", 100, "#c62828")):
        ax.axvline(pct(e, p), color=c, ls="--", lw=1.5,
                   label="measured %s = %.0f ms" % (label, pct(e, p)))
    if len(fin) < len(errs):
        ax.axvspan(delays[len(fin) - 1], delays[-1], color="#ffcdd2", alpha=.6)
        ax.text(delays[len(fin) - 1], max(fin) * .6, " unstable", fontsize=9,
                color="#b71c1c")
    ax.set_xlabel("dead time (ms)"); ax.set_ylabel("tracking RMS (mm)")
    ax.set_title("Latency is not a comfort metric")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency_cost.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("one calibrated matmul = %.3f ms" % (P.WORK.per_call * 1e3))
    e = exp1_budget()
    exp2_rates()
    exp3_queues()
    exp4_tail()
    exp5_cost(e)

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
