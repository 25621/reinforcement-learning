"""The ROS 2 managed-node state machine, and two perception nodes to compare.

``rclpy`` is not installed here, so the state machine is written out rather
than imported.  That is not a workaround -- it is the point.  The lifecycle is
*a protocol*, about thirty lines of it, and once you have written those thirty
lines the rclpy API stops being magic: ``LifecycleNode``, ``on_configure``,
``TransitionEvent`` and the ``/change_state`` service are exactly the pieces
below with ROS names on them.  See the README for the line-by-line mapping.

Everything runs on an integer millisecond clock so that 200 trials finish
instantly and every trial is exactly reproducible.  The state machine itself is
real; only the wall clock is simulated.
"""

# ---------------------------------------------------------------------------
# the states, straight out of the ROS 2 design document
# ---------------------------------------------------------------------------
UNCONFIGURED = "unconfigured"
INACTIVE = "inactive"
ACTIVE = "active"
FINALIZED = "finalized"
PRIMARY = (UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED)

# transition states: the node is *in the middle of* a transition callback.
# They exist so that a supervisor asking "what are you doing?" during a slow
# 200 ms configure gets "configuring", not silence.
CONFIGURING = "configuring"
CLEANINGUP = "cleaningup"
ACTIVATING = "activating"
DEACTIVATING = "deactivating"
SHUTTINGDOWN = "shuttingdown"
ERRORPROCESSING = "errorprocessing"

SUCCESS, FAILURE, ERROR = "SUCCESS", "FAILURE", "ERROR"

# how long each transition callback takes, in milliseconds.  Opening a camera
# and loading a calibration file really is the slow one.
DURATION = {CONFIGURING: 200, CLEANINGUP: 50, ACTIVATING: 30,
            DEACTIVATING: 20, SHUTTINGDOWN: 30, ERRORPROCESSING: 20}

FRAME_PERIOD = 33          # ms, a 30 Hz camera
STALE_AFTER = 100          # ms, when a repeated frame counts as stale


# ---------------------------------------------------------------------------
# the hardware
# ---------------------------------------------------------------------------
class Camera:
    """A camera that can be unplugged.

    ``outages`` is a list of (start_ms, end_ms).  ``read`` raises during them,
    which is what a real driver does when the USB device vanishes.
    """

    def __init__(self, outages=()):
        self.outages = list(outages)
        self.opened = False

    def open(self, t):
        if self.down(t):
            raise IOError("no such device")
        self.opened = True

    def close(self):
        self.opened = False

    def down(self, t):
        return any(a <= t < b for a, b in self.outages)

    def read(self, t):
        if not self.opened or self.down(t):
            raise IOError("read failed")
        return ("frame", t)


# ---------------------------------------------------------------------------
# what the downstream consumer sees
# ---------------------------------------------------------------------------
class Consumer:
    """Counts what it was handed, sorted into three piles.

    * good          -- a fresh frame from a properly configured node
    * uncalibrated  -- a frame produced before the calibration was loaded;
                       geometrically wrong, and it looks completely normal
    * stale         -- the same frame republished because nothing new arrived
    """

    def __init__(self):
        self.good = self.uncalibrated = self.stale = 0
        self.last_good_t = 0
        self.last_kind = None        # what the most recent message was

    def take(self, t, kind):
        setattr(self, kind, getattr(self, kind) + 1)
        self.last_kind = kind
        if kind == "good":
            self.last_good_t = t

    @property
    def bad(self):
        return self.uncalibrated + self.stale


# ---------------------------------------------------------------------------
# node A: the way everyone writes it the first time
# ---------------------------------------------------------------------------
class UnmanagedPerception:
    """Opens the device in the constructor and publishes from the first tick.

    Nothing here is stupid.  Every line is the obvious line.  The calibration
    load is slow, so it happens in a background step; the publisher is created
    up front because that is where publishers go; a read failure is retried
    because retrying is right.  The problems are all *emergent*.
    """

    name = "unmanaged"

    def __init__(self, cam, t0=0, configure_ms=None):
        self.cam = cam
        cfg = DURATION[CONFIGURING] if configure_ms is None else configure_ms
        self.calib_ready_at = t0 + cfg                     # loads in background
        self.retry_at = None
        self.last_frame_t = None
        self.next_frame = t0
        try:
            self.cam.open(t0)
        except IOError:
            self.retry_at = t0 + 100
        # the supervisor can only ask "have I heard from you lately?"
        self.reported_state = "running"

    def tick(self, t, out):
        if self.retry_at is not None and t >= self.retry_at:
            try:
                self.cam.open(t)
                self.retry_at = None
            except IOError:
                self.retry_at = t + 100
        if t < self.next_frame:
            return
        self.next_frame += FRAME_PERIOD
        try:
            self.cam.read(t)
            self.last_frame_t = t
            fresh = True
        except IOError:
            self.retry_at = self.retry_at or t + 100
            fresh = False
        if fresh:
            kind = "good" if t >= self.calib_ready_at else "uncalibrated"
        else:
            # publish the last frame again -- "better than nothing", and the
            # single most damaging line in the file
            if self.last_frame_t is None:
                return
            kind = "stale" if t - self.last_frame_t > STALE_AFTER else "good"
        out.take(t, kind)

    def supervisor_view(self, t, consumer, timeout=STALE_AFTER):
        """All a supervisor can do: watch for silence.  And this node is never
        silent, so the only signal is *content*, which the supervisor cannot
        judge."""
        return "running" if t - consumer.last_good_t < timeout else "suspect"


# ---------------------------------------------------------------------------
# node B: the managed lifecycle node
# ---------------------------------------------------------------------------
class ManagedPerception:
    """The same node, wrapped in the ROS 2 lifecycle protocol.

    Two rules do all the work:
      1. it cannot publish unless it is ACTIVE;
      2. every state change is announced, so somebody else can decide.
    """

    name = "managed"

    def __init__(self, cam, t0=0, configure_ms=None, tolerate=0):
        self.cam = cam
        self.configure_ms = (DURATION[CONFIGURING] if configure_ms is None
                             else configure_ms)
        # how many consecutive failed reads to ride out before declaring an
        # error.  0 = strict.  This is POLICY, not protocol: the lifecycle
        # says what a node must announce, never how twitchy it should be.
        self.tolerate = tolerate
        self.misses = 0
        self.state = UNCONFIGURED
        self.pending = None          # (transition_state, finish_time)
        self.next_frame = None
        self.last_frame_t = None
        self.calibrated = False
        self.events = []             # what a /transition_event topic carries

    # -- the protocol ------------------------------------------------------
    def _begin(self, tstate, t):
        dur = self.configure_ms if tstate == CONFIGURING else DURATION[tstate]
        self.pending = (tstate, t + dur)
        self._emit(t, tstate)

    def _emit(self, t, s):
        self.state_or_transition = s
        self.events.append((t, s))

    def request(self, transition, t):
        """The external command: this is rclpy's /change_state service."""
        legal = {"configure": (UNCONFIGURED, CONFIGURING),
                 "cleanup": (INACTIVE, CLEANINGUP),
                 "activate": (INACTIVE, ACTIVATING),
                 "deactivate": (ACTIVE, DEACTIVATING),
                 "shutdown": (None, SHUTTINGDOWN)}
        need, tstate = legal[transition]
        if self.pending is not None:
            return False
        if need is not None and self.state != need:
            return False           # rejected: the whole point of a protocol
        self._begin(tstate, t)
        return True

    # -- the user callbacks ------------------------------------------------
    def _on_configure(self, t):
        try:
            self.cam.open(t)
        except IOError:
            return FAILURE          # not an ERROR: nothing is broken, the
        self.calibrated = True      # device just is not there yet
        return SUCCESS

    def _on_activate(self, t):
        self.next_frame = t
        self.misses = 0
        return SUCCESS

    def _on_deactivate(self, t):
        self.next_frame = None
        return SUCCESS

    def _on_cleanup(self, t):
        self.cam.close()
        self.calibrated = False
        self.last_frame_t = None
        return SUCCESS

    def _on_error(self, t):
        self.cam.close()
        self.calibrated = False
        self.last_frame_t = None
        return SUCCESS

    # -- the machine -------------------------------------------------------
    _NEXT = {
        CONFIGURING: {SUCCESS: INACTIVE, FAILURE: UNCONFIGURED,
                      ERROR: ERRORPROCESSING},
        ACTIVATING: {SUCCESS: ACTIVE, FAILURE: INACTIVE, ERROR: ERRORPROCESSING},
        DEACTIVATING: {SUCCESS: INACTIVE, FAILURE: ACTIVE, ERROR: ERRORPROCESSING},
        CLEANINGUP: {SUCCESS: UNCONFIGURED, FAILURE: ERRORPROCESSING,
                     ERROR: ERRORPROCESSING},
        ERRORPROCESSING: {SUCCESS: UNCONFIGURED, FAILURE: FINALIZED,
                          ERROR: FINALIZED},
        SHUTTINGDOWN: {SUCCESS: FINALIZED, FAILURE: FINALIZED,
                       ERROR: FINALIZED},
    }
    _CB = {CONFIGURING: "_on_configure", ACTIVATING: "_on_activate",
           DEACTIVATING: "_on_deactivate", CLEANINGUP: "_on_cleanup",
           ERRORPROCESSING: "_on_error", SHUTTINGDOWN: "_on_cleanup"}

    def tick(self, t, out):
        if self.pending is not None:
            tstate, done_at = self.pending
            if t >= done_at:
                res = getattr(self, self._CB[tstate])(t)
                self.pending = None
                self.state = self._NEXT[tstate][res]
                self._emit(t, self.state)
                if self.state == ERRORPROCESSING:
                    self._begin(ERRORPROCESSING, t)
            return
        if self.state != ACTIVE or t < self.next_frame:
            return
        self.next_frame += FRAME_PERIOD
        try:
            self.cam.read(t)
            self.last_frame_t = t
            self.misses = 0
            out.take(t, "good")     # ACTIVE implies configured implies calibrated
        except IOError:
            # a read failure in ACTIVE means the node cannot do its job.  It
            # publishes NOTHING -- silence is a truthful message -- and after
            # `tolerate` misses it declares an error instead of guessing.
            self.misses += 1
            if self.misses > self.tolerate:
                self.state = ERRORPROCESSING
                self._emit(t, ERRORPROCESSING)
                self._begin(ERRORPROCESSING, t)

    def supervisor_view(self, t, consumer, timeout=STALE_AFTER):
        """A supervisor subscribes to /transition_event and simply knows."""
        s = self.pending[0] if self.pending else self.state
        return "running" if s == ACTIVE else s


# ---------------------------------------------------------------------------
# the supervisor (rclpy: a separate node calling /change_state)
# ---------------------------------------------------------------------------
class Supervisor:
    """Drives a managed node up, and drives it back up after a failure.

    ``backoff`` is the wait before retrying a configure that failed.  It exists
    because a supervisor that retries instantly against an unplugged camera
    becomes a busy loop.
    """

    def __init__(self, node, backoff=100, auto_activate=True):
        self.node, self.backoff = node, backoff
        # off when a higher-level coordinator decides WHEN to go active --
        # experiment 4 needs all four nodes configured before any of them runs
        self.auto_activate = auto_activate
        self.next_try = 0
        self.recoveries = 0

    def tick(self, t):
        n = self.node
        if not isinstance(n, ManagedPerception) or n.pending is not None:
            return
        if t < self.next_try:
            return
        if n.state == UNCONFIGURED:
            if n.request("configure", t):
                self.next_try = t + self.backoff
                self.recoveries += 1
        elif n.state == INACTIVE and self.auto_activate:
            n.request("activate", t)


# ---------------------------------------------------------------------------
# one trial
# ---------------------------------------------------------------------------
def run_trial(kind, outages=(), T=6000, backoff=100, tolerate=0,
              configure_ms=None):
    cam = Camera(outages)
    out = Consumer()
    if kind == "managed":
        node = ManagedPerception(cam, configure_ms=configure_ms,
                                 tolerate=tolerate)
        sup = Supervisor(node, backoff)
    else:
        node = UnmanagedPerception(cam, configure_ms=configure_ms)
        sup = None

    belief_wrong = 0
    detect_ms = None
    fail_start = outages[0][0] if outages else None
    for t in range(T):
        if sup:
            sup.tick(t)
        node.tick(t, out)
        # is the supervisor's belief right?  "reality" = a good frame was
        # produced within the last frame period
        healthy = (t - out.last_good_t) <= FRAME_PERIOD + 1 and t > 250
        believed = node.supervisor_view(t, out) == "running"
        if t > 300 and healthy != believed:
            belief_wrong += 1
        if fail_start is not None and t >= fail_start and detect_ms is None \
                and not believed:
            detect_ms = t - fail_start

    expected = T // FRAME_PERIOD
    return dict(kind=kind, good=out.good, uncalibrated=out.uncalibrated,
                stale=out.stale, bad=out.bad,
                availability=out.good / expected,
                belief_wrong_ms=belief_wrong,
                detect_ms=detect_ms if detect_ms is not None else -1,
                recoveries=sup.recoveries - 1 if sup else 0)
