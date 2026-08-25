"""A real threaded robot pipeline, instrumented end to end.

Five stages, five threads, real queues, real CPU work:

    camera (30 Hz) -> perception -> estimator -> planner (10 Hz) -> controller
                                                                        |
                                                                   actuator

The one idea that makes the measurement honest: **the capture timestamp
travels inside the message.**  Every stage copies it forward untouched, so when
the controller finally writes a command it can subtract and get the true age of
the light that caused it.  If instead each stage timed itself and you added the
numbers up, you would measure only the time your code was *running* and none of
the time your data spent *waiting* -- which, as ``run.py`` shows, is most of it.

The work in each stage is real matrix multiplication, not ``sleep``.  Sleeping
gives the CPU back; a busy stage does not, and the difference shows up the
moment two stages want the same core.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # see the note at the top of run.py

import collections
import threading
import time

import numpy as np

CLOCK = time.perf_counter          # monotonic: never jumps, unlike wall clock


# ---------------------------------------------------------------------------
# calibrated CPU work
# ---------------------------------------------------------------------------
class Work:
    """Burn a requested number of milliseconds of real CPU.

    numpy's matmul releases Python's global interpreter lock while it runs, so
    two stages doing this genuinely execute at the same time on two cores.  A
    pure-Python spin loop would not, and every measurement would be about the
    interpreter instead of about the pipeline.
    """

    def __init__(self, n=64):
        self.a = np.random.default_rng(0).standard_normal((n, n))
        t0 = CLOCK()
        for _ in range(200):
            self.a @ self.a
        self.per_call = (CLOCK() - t0) / 200

    def __call__(self, ms):
        target = ms * 1e-3
        t0 = CLOCK()
        while CLOCK() - t0 < target:
            self.a @ self.a


WORK = Work()


# ---------------------------------------------------------------------------
# messages and links
# ---------------------------------------------------------------------------
class Msg:
    __slots__ = ("seq", "t_capture", "stamps")

    def __init__(self, seq, t_capture):
        self.seq = seq
        self.t_capture = t_capture     # the photon time, never modified
        self.stamps = []               # (stage, t_dequeued, t_done)


class Link:
    """A one-slot-to-many-slot connection between two stages.

    ``policy`` is the whole point of experiment 3:
      * ``block``       -- a bounded queue that makes the producer wait
      * ``drop_oldest`` -- a bounded queue that throws away stale data
      * ``unbounded``   -- what you get by default, and a trap
    """

    def __init__(self, policy="drop_oldest", maxsize=2):
        self.policy = policy
        self.maxsize = maxsize
        self.q = collections.deque()
        self.cv = threading.Condition()
        self.dropped = 0
        self.closed = False

    def put(self, m):
        with self.cv:
            if self.policy == "block":
                while len(self.q) >= self.maxsize and not self.closed:
                    self.cv.wait(0.5)
            elif self.policy == "drop_oldest":
                while len(self.q) >= self.maxsize:
                    self.q.popleft()
                    self.dropped += 1
            self.q.append(m)
            self.cv.notify_all()

    def get(self, timeout=0.5):
        with self.cv:
            while not self.q and not self.closed:
                if not self.cv.wait(timeout):
                    return None
            if not self.q:
                return None
            m = self.q.popleft()
            self.cv.notify_all()
            return m

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
class Stage(threading.Thread):
    """One processing step.

    ``period`` = None means "run as fast as data arrives" (event driven).
    A number means "wake up on your own clock and take whatever is there",
    which is how most planners and all controllers actually work -- and which
    is where a whole sampling period of latency comes from.
    """

    def __init__(self, name, work_ms, src=None, dst=None, period=None,
                 hiccup=None, sink=None, drain="latest"):
        super().__init__(daemon=True)
        self.name, self.work_ms = name, work_ms
        self.src, self.dst, self.period = src, dst, period
        self.hiccup = hiccup           # (every_n, multiplier) or None
        self.sink = sink
        # "latest": throw away everything but the newest message on each tick
        # "fifo":   take the next one in line, however old it is
        self.drain = drain
        self.stop_flag = threading.Event()
        self.n = 0

    def _work(self):
        ms = self.work_ms
        if self.hiccup and self.n % self.hiccup[0] == 0 and self.n:
            ms *= self.hiccup[1]
        WORK(ms)

    def run(self):
        next_tick = CLOCK()
        while not self.stop_flag.is_set():
            if self.period is None:
                m = self.src.get()
                if m is None:
                    continue
                t_in = CLOCK()
            else:
                next_tick += self.period
                dt = next_tick - CLOCK()
                if dt > 0:
                    time.sleep(dt)
                else:
                    next_tick = CLOCK()
                t_in = CLOCK()
                m = self.src.get(timeout=0.0) if self.src else None
                if self.src is not None:
                    if self.drain == "latest":
                        # keep our own rhythm and act on the freshest data
                        # available; everything older is thrown away unread
                        while True:
                            nxt = self.src.get(timeout=0.0)
                            if nxt is None:
                                break
                            m = nxt
                    if m is None:
                        continue
            self.n += 1
            self._work()
            m.stamps.append((self.name, t_in, CLOCK()))
            if self.dst is not None:
                self.dst.put(m)
            if self.sink is not None:
                self.sink.append(m)


class Camera(threading.Thread):
    """The only stage that creates messages, and the only clock that matters:
    ``t_capture`` is stamped the instant the (pretend) shutter closes."""

    def __init__(self, dst, rate=30.0, work_ms=1.0):
        super().__init__(daemon=True)
        self.dst, self.period, self.work_ms = dst, 1.0 / rate, work_ms
        self.stop_flag = threading.Event()
        self.seq = 0

    def run(self):
        next_tick = CLOCK()
        while not self.stop_flag.is_set():
            next_tick += self.period
            dt = next_tick - CLOCK()
            if dt > 0:
                time.sleep(dt)
            t_cap = CLOCK()
            WORK(self.work_ms)
            self.seq += 1
            self.dst.put(Msg(self.seq, t_cap))


# ---------------------------------------------------------------------------
# the standard pipeline
# ---------------------------------------------------------------------------
DEFAULT = dict(cam_rate=30.0, cam_ms=1.0, perc_ms=14.0, est_ms=2.0,
               plan_rate=10.0, plan_ms=6.0, ctrl_rate=200.0, ctrl_ms=0.4,
               policy="drop_oldest", maxsize=2, hiccup=(40, 5.0),
               plan_drain="latest")


def run_pipeline(duration=8.0, **over):
    cfg = dict(DEFAULT, **over)
    mk = lambda: Link(cfg["policy"], cfg["maxsize"])
    l1, l2, l3, l4 = mk(), mk(), mk(), mk()
    done = []

    cam = Camera(l1, cfg["cam_rate"], cfg["cam_ms"])
    stages = [
        Stage("perception", cfg["perc_ms"], l1, l2, hiccup=cfg["hiccup"]),
        Stage("estimator", cfg["est_ms"], l2, l3),
        Stage("planner", cfg["plan_ms"], l3, l4, period=1.0 / cfg["plan_rate"],
              drain=cfg["plan_drain"]),
        Stage("controller", cfg["ctrl_ms"], l4, None,
              period=1.0 / cfg["ctrl_rate"], sink=done),
    ]
    for s in stages:
        s.start()
    cam.start()
    time.sleep(duration)
    cam.stop_flag.set()
    for s in stages:
        s.stop_flag.set()
    for lnk in (l1, l2, l3, l4):
        lnk.close()
    cam.join(1.0)
    for s in stages:
        s.join(1.0)

    t_end = CLOCK()
    fresh = [m for m in done if m.t_capture < t_end - 0.5]
    return dict(msgs=fresh, sent=cam.seq,
                dropped=sum(l.dropped for l in (l1, l2, l3, l4)))


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def e2e_ms(msgs):
    """Photon to actuation, in milliseconds, one number per delivered command."""
    return np.array([1e3 * (m.stamps[-1][2] - m.t_capture) for m in msgs])


def stage_breakdown(msgs):
    """For each stage: how long it WAITED and how long it WORKED."""
    out = collections.OrderedDict()
    for m in msgs:
        prev_done = m.t_capture
        for name, t_in, t_out in m.stamps:
            w = out.setdefault(name, {"wait": [], "work": []})
            w["wait"].append(1e3 * (t_in - prev_done))
            w["work"].append(1e3 * (t_out - t_in))
            prev_done = t_out
    return out


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")
