"""Five real-time drills.

  1. four ways to wait          -- drift, jitter, and the cost of each
  2. under load                 -- what the tail does when the machine is busy
  3. what the OS will give you  -- SCHED_FIFO, pinning, and who is allowed
  4. allocation in the hot path -- the "never malloc" rule, measured
  5. what jitter costs          -- and the two-line fix when you cannot remove it

About 3 minutes, most of it real elapsed time by construction.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
HZ = 1000.0
SECS = 6.0
STRATEGIES = ("sleep_rel", "sleep_abs", "spin", "hybrid")


def record(exp, **kw):
    kw.pop("periods", None)
    ROWS.append(dict(experiment=exp, **kw))


def header(r):
    return ("  %-11s %8.4f %8.4f %8.4f %8.4f %8d %11.1f"
            % (r["strategy"], r["p50"], r["p99"], r["p999"], r["worst"],
               r["overruns"], r["drift_ms"]))


COLS = "  strategy       p50      p99    p99.9    worst  overruns   drift(ms)"


# ---------------------------------------------------------------------------
# 1. four ways to wait
# ---------------------------------------------------------------------------
def exp1_strategies():
    print("\n=== 1. four ways to wait until it is time again " + "=" * 24)
    print("  (1 kHz, %.0f s, 0.2 ms of work per iteration)" % SECS)
    print(COLS)
    keep = {}
    for s in STRATEGIES:
        r = rt.run_loop(s, HZ, SECS)
        keep[s] = r["periods"]
        print(header(r))
        record("strategies", **r)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for s in STRATEGIES:
        p = keep[s]
        ax[0].plot(np.sort(p), np.linspace(0, 100, len(p)), lw=1.4, label=s)
        ax[1].plot(np.cumsum(p - 1e3 / HZ), lw=1.2, label=s)
    ax[0].set_xlim(0.7, 1.6); ax[0].set_xlabel("period (ms)")
    ax[0].set_ylabel("percentile"); ax[0].set_title("the distribution of periods")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("cumulative lateness (ms)")
    ax[1].set_title("drift: where the clock actually is")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "strategies.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. what rate can this stack actually hold?
# ---------------------------------------------------------------------------
def exp2_rate():
    print("\n=== 2. what rate can this stack actually hold? " + "=" * 25)
    print("  Absolute jitter is a property of the operating system; the RATIO")
    print("  to your period is what decides whether you can run there.")
    print("  strategy    rate   nominal   p99-nominal    ratio   verdict")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for strat, col in (("sleep_abs", "#c62828"), ("spin", "#2e7d32")):
        rates, ratios = [], []
        for hz in (100, 250, 500, 1000, 2000):
            r = rt.run_loop(strat, hz, 4.0)
            nom = 1e3 / hz
            excess = r["p99"] - nom
            ratio = r["p99"] / nom
            ok = "usable" if ratio < 1.2 else ("marginal" if ratio < 2 else "no")
            print("  %-10s %5d Hz %7.3f %11.4f ms %8.2f   %s"
                  % (strat, hz, nom, excess, ratio, ok))
            record("rate", strategy=strat, hz=hz, nominal_ms=nom,
                   p99=r["p99"], excess_ms=excess, worst=r["worst"],
                   overrun_pct=100 * r["overruns"] / (4.0 * hz),
                   p99_over_nominal=ratio)
            rates.append(hz); ratios.append(ratio)
        ax.plot(rates, ratios, "o-", color=col, label=strat)
    ax.axhline(1.2, ls="--", color="#455a64",
               label="p99 within 20 % of nominal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("loop rate (Hz)"); ax.set_ylabel("p99 period / nominal period")
    ax.set_title("CPython on a PREEMPT_DYNAMIC kernel, idle machine")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rate_ceiling.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. under load
# ---------------------------------------------------------------------------
def exp3_load():
    print("\n=== 3. the same loops on a busy machine " + "=" * 32)
    n_hogs = max(2, os.cpu_count())
    print("  (%d background processes spinning on %d cores)"
          % (n_hogs, os.cpu_count()))
    print(COLS)
    keep = {}
    for s in ("sleep_abs", "spin", "hybrid"):
        ps = rt.start_load(n_hogs, SECS + 3)
        try:
            r = rt.run_loop(s, HZ, SECS)
        finally:
            rt.stop_load(ps)
        keep[s] = r
        print(header(r))
        record("load", n_hogs=n_hogs, **r)

    idle = {x["strategy"]: x for x in ROWS if x["experiment"] == "strategies"}
    print("  strategy      p99 idle -> loaded    worst idle -> loaded")
    for s, r in keep.items():
        print("  %-11s %8.4f -> %-8.4f %10.4f -> %-8.4f"
              % (s, idle[s]["p99"], r["p99"], idle[s]["worst"], r["worst"]))
        record("load_ratio", strategy=s, p99_idle=idle[s]["p99"],
               p99_loaded=r["p99"], worst_idle=idle[s]["worst"],
               worst_loaded=r["worst"],
               p99_ratio=r["p99"] / idle[s]["p99"])


# ---------------------------------------------------------------------------
# 3. what the OS will give you
# ---------------------------------------------------------------------------
def exp3_os():
    print("\n=== 4. what this machine will actually give you " + "=" * 24)
    print("  kernel preemption model : %s" % rt.kernel_flavour())
    ok_rt, msg_rt = rt.try_realtime_priority()
    print("  SCHED_FIFO              : %s" % msg_rt)
    ok_pin, msg_pin = rt.try_pin()
    print("  CPU pinning             : %s" % msg_pin)
    record("os", kernel=rt.kernel_flavour(), sched_fifo=ok_rt,
           sched_fifo_msg=msg_rt, pinning=ok_pin)

    if ok_pin:
        ncpu = os.cpu_count()
        n_hogs = max(2, ncpu)
        others = list(range(1, ncpu))
        print("\n  hybrid loop under load, three placements:")
        print(COLS)
        cases = (("free to migrate", None, None),
                 ("pinned to a busy core", 0, None),
                 ("pinned, load kept off it", 0, others))
        for label, pin_cpu, load_cpus in cases:
            rt.unpin()
            if pin_cpu is not None:
                rt.try_pin(pin_cpu)
            ps = rt.start_load(n_hogs, SECS + 3, cpus=load_cpus)
            try:
                r = rt.run_loop("hybrid", HZ, SECS)
            finally:
                rt.stop_load(ps)
            r["strategy"] = label
            print(header(r))
            record("pinning", **r)
        rt.unpin()


# ---------------------------------------------------------------------------
# 4. allocation in the hot path
# ---------------------------------------------------------------------------
def exp4_allocation():
    print("\n=== 5. allocating inside the control thread " + "=" * 28)
    print(COLS)
    cases = (("preallocated", dict(allocate=False, collect_gc=True)),
             ("small objs + cycles", dict(allocate="small", collect_gc=True)),
             ("same, gc disabled", dict(allocate="small", collect_gc=False)),
             ("1.6 MB per iteration", dict(allocate="large", collect_gc=True)))
    for label, kw in cases:
        r = rt.run_loop("hybrid", HZ, 10.0, **kw)
        r["strategy"] = label
        print(header(r))
        record("allocation", **r)
    rows = [x for x in ROWS if x["experiment"] == "allocation"]
    base = rows[0]
    for r in rows[1:]:
        print("  %-22s p99 %.2fx, p99.9 %.2fx, worst %.2fx vs preallocated"
              % (r["strategy"], r["p99"] / base["p99"],
                 r["p999"] / base["p999"], r["worst"] / base["worst"]))

    # How big is the effect we are failing to see?  Measure it directly rather
    # than concluding "no effect" from a table that cannot resolve one.
    import gc as _gc
    n = 20000
    costs = {}
    t0 = rt.CLOCK()
    for k in range(n):
        pass
    costs["empty statement"] = (rt.CLOCK() - t0) / n
    t0 = rt.CLOCK()
    keep = []
    for k in range(n):
        d = {"q": [0.0] * 8, "t": k}
        d["self"] = d
        keep.append(d)
    costs["small object + cycle"] = (rt.CLOCK() - t0) / n
    del keep
    _gc.collect()
    t0 = rt.CLOCK()
    for k in range(2000):
        b = np.empty(200_000)
        b[0] = k
    costs["1.6 MB allocation"] = (rt.CLOCK() - t0) / 2000
    t0 = rt.CLOCK()
    _gc.collect()
    costs["one full gc.collect()"] = rt.CLOCK() - t0

    floor = base["p99"] - 1e3 / HZ
    print("\n  how big is the effect we failed to see?")
    for name, c in costs.items():
        print("     %-24s %8.4f ms   (%.0fx below this machine's %0.2f ms "
              "jitter floor)" % (name, c * 1e3, floor / max(c * 1e3, 1e-9),
                                 floor))
        record("alloc_cost", item=name, ms=c * 1e3, jitter_floor_ms=floor,
               ratio=floor / max(c * 1e3, 1e-9))


# ---------------------------------------------------------------------------
# 5. what jitter costs, and the fix
# ---------------------------------------------------------------------------
def pd_with_jitter(periods_ms, use_measured_dt, kp=6000.0, kd=160.0,
                   freq=1.0, nominal_dt=1e-3):
    """A PD controller on a second-order plant, stepped with the real periods.

    The controller does not get velocity handed to it -- almost none do.  It
    estimates velocity by differencing position, and that division needs a dt:

        v_est = (x - x_prev) / dt

    If the loop actually took 1.3 ms and the code divides by 1.0 ms, the
    velocity estimate is 30 % too small, so the damping term is 30 % too weak
    -- for that step only, then a different amount the next step.  **Jitter is
    a gain error that changes every iteration**, which is why it shows up as
    ringing rather than as a steady offset.

    ``use_measured_dt`` is the whole experiment: divide by the dt that actually
    elapsed instead of the one you wrote in the config file.
    """
    x = v = x_prev = 0.0
    err = np.zeros(len(periods_ms))
    t = 0.0
    w = 2 * np.pi * freq
    for k, p_ms in enumerate(periods_ms):
        dt_real = p_ms * 1e-3
        dt_used = dt_real if use_measured_dt else nominal_dt
        ref, dref = 0.25 * np.sin(w * t), 0.25 * w * np.cos(w * t)
        v_est = (x - x_prev) / dt_used
        a = float(np.clip(kp * (ref - x) + kd * (dref - v_est), -4000, 4000))
        x_prev = x
        v += dt_real * a           # the plant only knows real time
        x += dt_real * v
        err[k] = ref - x
        t += dt_real
        if not np.isfinite(x) or abs(x) > 50:
            return float("inf")
    return float(np.sqrt(np.mean(err[300:] ** 2)) * 1e3)


def exp5_cost():
    print("\n=== 6. what jitter costs the controller " + "=" * 32)
    n = 6000
    perfect = np.full(n, 1.0)
    base = pd_with_jitter(perfect, False)

    # part A: does jitter matter at all?  It depends on the controller.
    rng = np.random.default_rng(0)
    print("  A. the same jitter, four controllers (tracking RMS, mm)")
    print("     gains              0 us    100 us    200 us    350 us    500 us")
    gains = ((400, 28, "loose"), (2500, 90, "medium"),
             (6000, 160, "tight"), (10000, 200, "very tight"))
    spreads = (0.0, 0.10, 0.20, 0.35, 0.50)
    for kp, kd, tag in gains:
        vals = [pd_with_jitter(np.clip(1.0 + rng.normal(0, sd, n), 0.2, None),
                               False, kp, kd) for sd in spreads]
        print("     kp=%-6d %-10s" % (kp, tag)
              + " ".join("%9.2f" % v for v in vals))
        for sd, v in zip(spreads, vals):
            record("cost_gains", kp=kp, kd=kd, tag=tag, spread_us=sd * 1e3,
                   rms=v)

    # part B: the mechanism and the fix, on the tight controller
    print("  B. the tight controller, with and without the fix")
    print("     period spread   assumed dt      measured dt   extra error")
    levels, ea, eb = [], [], []
    for sd in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        p = np.clip(1.0 + rng.normal(0, sd, n), 0.2, None)
        a, b = pd_with_jitter(p, False), pd_with_jitter(p, True)
        print("     %6.0f us %14.3f mm %13.3f mm %11.3f mm"
              % (sd * 1e3, a, b, a - base))
        record("cost_synthetic", spread_us=sd * 1e3, rms_assumed_dt=a,
               rms_measured_dt=b, extra_mm=a - base)
        levels.append(sd * 1e3); ea.append(a); eb.append(b)

    # part C: the clocks we actually measured
    print("  C. the real clocks measured above")
    names = ["hybrid, idle", "sleep_abs, idle", "sleep_abs, loaded"]
    traces = {}
    for s in ("hybrid", "sleep_abs"):
        traces[s + ", idle"] = rt.run_loop(s, HZ, SECS)["periods"]
    ps = rt.start_load(max(2, os.cpu_count()), SECS + 3)
    try:
        traces["sleep_abs, loaded"] = rt.run_loop("sleep_abs", HZ, SECS)["periods"]
    finally:
        rt.stop_load(ps)
    print("     clock                spread   assumed dt   measured dt")
    for name in names:
        p = traces[name]
        a, b = pd_with_jitter(p, False), pd_with_jitter(p, True)
        print("     %-20s %5.0f us %8.3f mm %11.3f mm"
              % (name, np.std(p) * 1e3, a, b))
        record("cost_measured", clock=name, spread_us=float(np.std(p) * 1e3),
               rms_assumed_dt=a, rms_measured_dt=b, extra_mm=a - base)
    print("     perfect clock reference     0 us %8.3f mm" % base)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(levels, ea, "o-", color="#c62828",
            label="controller assumes dt = 1 ms")
    ax.plot(levels, eb, "s-", color="#2e7d32",
            label="controller uses the measured dt")
    for name, col in zip(names, ("#1976d2", "#7b1fa2", "#ef6c00")):
        p = traces[name]
        ax.scatter([np.std(p) * 1e3], [pd_with_jitter(p, False)], s=70,
                   color=col, zorder=5, label=name)
    ax.set_xlabel("standard deviation of the loop period (microseconds)")
    ax.set_ylabel("tracking RMS (mm)"); ax.legend(fontsize=7.5)
    ax.set_title("Jitter you cannot remove, you can still account for")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jitter_cost.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("work per iteration: %d matmuls" % rt.REPS)
    exp1_strategies()
    exp2_rate()
    exp3_load()
    exp3_os()
    exp4_allocation()
    exp5_cost()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
