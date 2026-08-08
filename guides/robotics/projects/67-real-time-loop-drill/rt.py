"""Four ways to run a loop at a fixed rate, and the machinery to measure them.

Every control loop in robotics is this:

    while running:
        read sensors; compute; write actuators; wait until it is time again

The last four words are where the project lives.  "Wait until it is time again"
has at least four implementations, they differ by orders of magnitude in
quality, and three of them look identical in a code review.

A note on the platform, stated up front because it changes what the numbers
mean: this machine runs a **PREEMPT_DYNAMIC** kernel, not PREEMPT_RT, and the
loop is written in Python.  Neither is a real-time system.  What survives that
is the *shape* of the results -- which strategy drifts, which one has a tail,
what background load does, what allocation does.  The absolute microseconds
would be 10-100x smaller in C on a PREEMPT_RT kernel; the ordering would not
change.
"""

import gc
import os
import time

import numpy as np

CLOCK = time.perf_counter


def work_calibrate(target_ms=0.30, n=48):
    """Size a fixed lump of numeric work to a chosen duration."""
    a = np.random.default_rng(0).standard_normal((n, n))
    t0 = CLOCK()
    for _ in range(500):
        a @ a
    per = (CLOCK() - t0) / 500
    reps = max(1, int(round(target_ms * 1e-3 / per)))
    return a, reps


A_MAT, REPS = work_calibrate()


def do_work():
    for _ in range(REPS):
        A_MAT @ A_MAT


# ---------------------------------------------------------------------------
# the four strategies
# ---------------------------------------------------------------------------
def run_loop(strategy, hz=1000.0, seconds=6.0, slack_ms=0.25, allocate=False,
             collect_gc=True):
    """Run the loop and return every measured period, plus the total drift.

    ``allocate`` makes the loop body create fresh objects, which is the
    "never malloc in the control thread" rule made testable.
    """
    period = 1.0 / hz
    n = int(seconds * hz)
    periods = np.zeros(n)
    if not collect_gc:
        gc.disable()
    junk = []
    t_start = CLOCK()
    next_t = t_start
    last = t_start
    try:
        for k in range(n):
            now = CLOCK()
            periods[k] = now - last
            last = now
            do_work()
            if allocate == "small":
                # small objects with reference cycles: what "just append the
                # sample to a list" looks like.  Cycles are what force CPython
                # past reference counting into an actual generational sweep.
                d = {"q": [0.0] * 8, "t": k}
                d["self"] = d
                junk.append(d)
                if len(junk) > 20000:
                    junk = junk[10000:]
            elif allocate == "large":
                # 1.6 MB per iteration.  glibc serves allocations above
                # ~128 kB with mmap and returns them with munmap, so this is
                # two system calls and a page-table walk inside the loop.
                buf = np.empty(200_000)
                buf[0] = k

            next_t += period
            if strategy == "sleep_rel":
                # the obvious one: "wait one period".  It waits one period
                # AFTER the work, so every iteration is late by the work time
                # and the lateness accumulates.
                d = period - (CLOCK() - now)
                if d > 0:
                    time.sleep(d)
            elif strategy == "sleep_abs":
                # aim at an absolute deadline computed from the start time, so
                # a late iteration does not push the next one
                d = next_t - CLOCK()
                if d > 0:
                    time.sleep(d)
            elif strategy == "spin":
                while CLOCK() < next_t:
                    pass
            elif strategy == "hybrid":
                # sleep for the bulk, spin for the last fraction of a
                # millisecond: sleep's granularity is coarse, spinning is
                # exact, and this buys the accuracy of one at most of the cost
                # of the other
                d = next_t - CLOCK() - slack_ms * 1e-3
                if d > 0:
                    time.sleep(d)
                while CLOCK() < next_t:
                    pass
            if strategy != "sleep_rel" and CLOCK() > next_t + period:
                next_t = CLOCK()          # we are hopelessly late: resynchronise
    finally:
        if not collect_gc:
            gc.enable()
    elapsed = CLOCK() - t_start
    p = periods[5:] * 1e3
    nominal = 1e3 / hz
    return dict(strategy=strategy,
                p50=float(np.percentile(p, 50)),
                p99=float(np.percentile(p, 99)),
                p999=float(np.percentile(p, 99.9)),
                worst=float(p.max()),
                jitter_p99=float(np.percentile(np.abs(p - nominal), 99)),
                overruns=int((p > 1.5 * nominal).sum()),
                drift_ms=float((elapsed - n * period) * 1e3),
                cpu_pct=100.0 if strategy == "spin" else float("nan"),
                periods=p)


# ---------------------------------------------------------------------------
# background load
# ---------------------------------------------------------------------------
def _hog(stop_at, cpus=None):
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    a = np.random.default_rng().standard_normal((160, 160))
    while time.time() < stop_at:
        a @ a


def start_load(n, seconds, cpus=None):
    """n processes competing for the CPU.  Processes, not threads: threads
    would fight over the interpreter lock and we want to test the *scheduler*.

    ``cpus`` confines the load to a subset of cores, which is how experiment 3
    builds an isolated core to pin the control loop to."""
    import multiprocessing as mp
    stop = time.time() + seconds
    ps = [mp.Process(target=_hog, args=(stop, cpus), daemon=True)
          for _ in range(n)]
    for p in ps:
        p.start()
    return ps


def stop_load(ps):
    for p in ps:
        p.terminate()
    for p in ps:
        p.join(2.0)


# ---------------------------------------------------------------------------
# what the OS will and will not let us have
# ---------------------------------------------------------------------------
def try_realtime_priority(prio=80):
    """Ask for SCHED_FIFO, the POSIX real-time scheduling policy.

    Under SCHED_FIFO a thread runs until it blocks or yields -- the scheduler
    will not preempt it for a normal task, whatever that task is doing.  That
    is the single biggest lever on jitter, and it needs CAP_SYS_NICE, which is
    to say root.  Reported honestly either way.
    """
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        return True, "SCHED_FIFO priority %d" % prio
    except (PermissionError, OSError) as e:
        return False, "denied: %s" % e


def try_pin(cpu=2):
    """Pin to one core.  Unlike SCHED_FIFO this needs no privileges."""
    try:
        os.sched_setaffinity(0, {cpu})
        return True, "pinned to CPU %d" % cpu
    except OSError as e:
        return False, str(e)


def unpin():
    os.sched_setaffinity(0, set(range(os.cpu_count())))


def kernel_flavour():
    try:
        v = open("/proc/version").read().strip()
    except OSError:
        return "unknown"
    for flag in ("PREEMPT_RT", "PREEMPT_DYNAMIC", "PREEMPT"):
        if flag in v:
            return flag
    return "no preemption flag"
