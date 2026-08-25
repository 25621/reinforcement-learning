"""Log a real robot run to MCAP, then find out what logging is worth.

  1. the log        -- write a run, read it back, plot it from the FILE
  2. the hot path   -- what logging does to a 1 kHz control loop
  3. the index      -- what the seek machinery buys
  4. the bill       -- bytes per second, per topic, and compression
  5. replayability  -- the property that is not about the format at all

Takes about 40 seconds.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"      # see project 64: before numpy loads
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import collections
import csv
import json
import queue
import sys
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mcap_io as M

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))
import arm as A            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "robot_run.mcap")
ROWS = []
NS = 1_000_000_000


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


# ---------------------------------------------------------------------------
# schemas.  Foxglove's well-known ones are used where they exist, so that the
# file opens in a 3D panel with no configuration at all.
# ---------------------------------------------------------------------------
POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "pose": {"type": "object", "properties": {
            "position": {"type": "object", "properties": {
                "x": {"type": "number"}, "y": {"type": "number"},
                "z": {"type": "number"}}},
            "orientation": {"type": "object", "properties": {
                "x": {"type": "number"}, "y": {"type": "number"},
                "z": {"type": "number"}, "w": {"type": "number"}}}}}}}

JOINT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "array", "items": {"type": "string"}},
                   "position": {"type": "array", "items": {"type": "number"}},
                   "velocity": {"type": "array", "items": {"type": "number"}},
                   "effort": {"type": "array", "items": {"type": "number"}}}}

TF_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}}},
        "rotation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}, "w": {"type": "number"}}}}}

DIAG_SCHEMA = {
    "type": "object",
    "properties": {"level": {"type": "integer"}, "name": {"type": "string"},
                   "distance_to_goal": {"type": "number"},
                   "action_norm": {"type": "number"}}}


def stamp(t):
    return {"sec": int(t), "nsec": int((t - int(t)) * 1e9)}


def tf_msg(t, parent, child, x, y, th):
    """A foxglove.FrameTransform: where `child` sits inside `parent`.

    Frames are logged as a *stream of relative transforms*, not as absolute
    positions, so that a viewer can rebuild the whole tree at any instant and
    attach anything it likes to any frame.  This is the same idea as ROS's TF
    tree (project 01)."""
    return {"timestamp": stamp(t), "parent_frame_id": parent,
            "child_frame_id": child,
            "translation": {"x": float(x), "y": float(y), "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": float(np.sin(th / 2)),
                         "w": float(np.cos(th / 2))}}


def pose_msg(t, frame, x, y, th=0.0):
    return {"timestamp": stamp(t), "frame_id": frame,
            "pose": {"position": {"x": float(x), "y": float(y), "z": 0.0},
                     "orientation": {"x": 0.0, "y": 0.0,
                                     "z": float(np.sin(th / 2)),
                                     "w": float(np.cos(th / 2))}}}


# ---------------------------------------------------------------------------
# 1. write a run
# ---------------------------------------------------------------------------
def exp1_write():
    print("\n=== 1. logging a robot run " + "=" * 45)
    rng = np.random.default_rng(4)
    env = A.PushEnv(rng)
    obs = env.reset()

    w = M.McapWriter(LOG, chunk_bytes=64 * 1024, profile="",
                     library="robotics-project-66")
    w.add_schema(1, "foxglove.PoseInFrame", "jsonschema", M.to_json(POSE_SCHEMA))
    w.add_schema(2, "robot.JointState", "jsonschema", M.to_json(JOINT_SCHEMA))
    w.add_schema(3, "robot.ControllerDiagnostic", "jsonschema",
                 M.to_json(DIAG_SCHEMA))
    w.add_schema(4, "foxglove.FrameTransform", "jsonschema", M.to_json(TF_SCHEMA))
    topics = {"/tip": (1, 1), "/puck": (2, 1), "/goal": (3, 1),
              "/joint_states": (4, 2), "/controller/diagnostics": (5, 3),
              "/tf": (6, 4)}
    for topic, (cid, sid) in topics.items():
        w.add_channel(cid, sid, topic, "json",
                      {"description": "project 66 demo log"})

    seq = collections.Counter()

    def emit(cid, t, payload):
        seq[cid] += 1
        w.write(cid, seq[cid], int(t * NS), M.to_json(payload))

    # 14 episodes back to back: one push takes about a second, and a log with
    # one chunk in it cannot demonstrate anything about seeking.
    t = 0.0
    for ep in range(14):
        env.reset()
        done = False
        while not done:
            a, _ = A.expert_action(env, side=1, rng=env.rng)
            tip = env.arm.tip(env.q)
            # 200 Hz joint states -- the inner servo's own rate, ten samples
            # per decision.  Logging only at the decision rate is experiment 5.
            for k in range(A.SUBSTEPS):
                emit(4, t + k * A.DT, {"name": ["j1", "j2"],
                                       "position": [float(v) for v in env.q],
                                       "velocity": [float(v) for v in env.qd],
                                       "effort": [0.0, 0.0]})
            pts = env.arm.points(env.q)
            emit(6, t, tf_msg(t, "world", "link1", pts[0][0], pts[0][1],
                              float(env.q[0])))
            emit(6, t, tf_msg(t, "link1", "link2", env.arm.l[0], 0.0,
                              float(env.q[1])))
            emit(1, t, pose_msg(t, "world", tip[0], tip[1]))
            emit(2, t, pose_msg(t, "world", env.puck[0], env.puck[1]))
            emit(3, t, pose_msg(t, "world", env.goal[0], env.goal[1]))
            emit(5, t, {"level": 0, "name": "push_controller",
                        "episode": ep,
                        "distance_to_goal":
                            float(np.linalg.norm(env.puck - env.goal)),
                        "action_norm": float(np.linalg.norm(a))})
            _, _, done, _ = env.step(a)
            t += A.CTRL_DT
    size = w.close()

    r = M.McapReader(LOG)
    msgs = list(r.messages())
    counts = collections.Counter(m["topic"] for m in msgs)
    print("  file            : %s (%.1f kB)" % (os.path.basename(LOG), size / 1024))
    print("  duration        : %.2f s   messages: %d   chunks: %d"
          % (t, r.stats["messages"], len(r.chunks)))
    for topic in topics:
        print("    %-26s %5d messages" % (topic, counts[topic]))
    assert r.stats["messages"] == len(msgs) == sum(seq.values())
    print("  round trip      : %d written, %d read back, all CRCs valid"
          % (sum(seq.values()), len(msgs)))
    record("write", bytes=size, seconds=t, messages=len(msgs),
           chunks=len(r.chunks), bytes_per_s=size / t)

    # ---- the plots are drawn from the FILE, not from the arrays above ----
    def xy(topic):
        return np.array([[json.loads(m["data"])["pose"]["position"]["x"],
                          json.loads(m["data"])["pose"]["position"]["y"]]
                         for m in msgs if m["topic"] == topic])

    diags = [json.loads(m["data"]) for m in msgs
             if m["topic"] == "/controller/diagnostics"]
    # pick a successful episode for the trajectory panel, straight from the log
    best = min(range(14), key=lambda e: min(
        d["distance_to_goal"] for d in diags if d["episode"] == e))
    ep0 = [i for i, d in enumerate(diags) if d["episode"] == best]
    tips, pucks, goals = xy("/tip"), xy("/puck"), xy("/goal")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    s = slice(ep0[0], ep0[-1] + 1)
    ax[0].plot(tips[s, 0], tips[s, 1], "-", color="#1976d2", label="tool tip")
    ax[0].plot(pucks[s, 0], pucks[s, 1], "-", lw=3, color="#f9a825", label="puck")
    ax[0].scatter(*goals[s][0], s=200, marker="*", color="#2e7d32", zorder=5,
                  label="goal")
    ax[0].scatter(pucks[s][0, 0], pucks[s][0, 1], s=60, color="#c62828",
                  label="puck start")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].set_title("episode %d, replayed from robot_run.mcap" % best)
    dist = np.array([d["distance_to_goal"] for d in diags])
    epi = np.array([d["episode"] for d in diags])
    ax[1].plot(np.arange(len(dist)) * A.CTRL_DT, 1e3 * dist, lw=.9,
               color="#455a64")
    for e in range(1, epi.max() + 1):
        ax[1].axvline(np.argmax(epi == e) * A.CTRL_DT, color="#cfd8dc", lw=.6)
    ax[1].axhline(1e3 * A.GOAL_TOL, ls="--", color="#2e7d32", label="success")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("puck to goal (mm)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("/controller/diagnostics, all 14 episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "replayed_run.png"), dpi=120)
    plt.close(fig)
    return r


# ---------------------------------------------------------------------------
# 2. what logging does to a 1 kHz control loop
# ---------------------------------------------------------------------------
def control_loop(mode, seconds=4.0, hz=1000.0, path=None, scan=360):
    """Run a fixed-rate loop and measure two different things.

    * ``cost``   -- the time the logging statement itself took, per iteration.
                    This is the direct question: what did I put in my loop?
    * ``period`` -- how well the loop then kept its 1 ms schedule.  This is the
                    consequence, and it is noisier because the operating
                    system's sleep is not perfect either -- which is exactly
                    why we measure the cost separately instead of inferring it.

    ``mode``:
      * ``off``    -- no logging at all, the reference
      * ``sync``   -- serialise and write inside the loop
      * ``queue``  -- drop the raw sample into a bounded queue; a background
                      thread serialises and writes
    """
    period = 1.0 / hz
    n = int(seconds * hz)
    clock = time.perf_counter
    w = q = th = None
    dropped = 0

    if mode != "off":
        w = M.McapWriter(path, chunk_bytes=256 * 1024)
        w.add_schema(1, "robot.JointState", "jsonschema", M.to_json(JOINT_SCHEMA))
        w.add_channel(1, 1, "/joint_states")
    if mode == "queue":
        q = queue.Queue(maxsize=1024)

        def drain():
            i = 0
            while True:
                item = q.get()
                if item is None:
                    return
                i += 1
                w.write(1, i, item[0], M.to_json(item[1]))
        th = threading.Thread(target=drain, daemon=True)
        th.start()

    periods, costs = np.zeros(n), np.zeros(n)
    ranges = [0.0] * scan          # a 360-point laser scan rides along
    next_t = clock()
    last = next_t
    x = 0.0
    for k in range(n):
        next_t += period
        d = next_t - clock()
        if d > 0:
            time.sleep(d)
        now = clock()
        periods[k] = now - last
        last = now
        x += 0.001 * np.sin(k * 0.01)                       # the "control law"
        sample = {"name": ["j1", "j2"], "position": [x, -x],
                  "velocity": [0.0, 0.0], "effort": [0.1, 0.2],
                  "ranges": ranges}
        t0 = clock()
        if mode == "sync":
            w.write(1, k + 1, int(now * NS), M.to_json(sample))
        elif mode == "queue":
            try:
                q.put_nowait((int(now * NS), sample))
            except queue.Full:
                dropped += 1        # bounded: we drop, we never block the loop
        costs[k] = clock() - t0
    if mode == "queue":
        q.put(None); th.join(10.0)
    if w is not None:
        w.close()
    p, c = periods[10:] * 1e3, costs[10:] * 1e3
    return dict(mode=mode, cost_p50=float(np.percentile(c, 50)),
                cost_p99=float(np.percentile(c, 99)),
                cost_max=float(c.max()),
                period_p99=float(np.percentile(p, 99)),
                overruns=int((p > 1.5 / hz * 1e3).sum()), dropped=dropped)


def exp2_hotpath():
    print("\n=== 2. logging from inside a 1 kHz control loop " + "=" * 24)
    print("  (each sample carries a 360-point laser scan)")
    print("  mode      in-loop p50  in-loop p99  in-loop max  period p99  drops")
    tmp = os.path.join(OUT, "_scratch.mcap")
    for mode in ("off", "sync", "queue"):
        r = control_loop(mode, path=tmp)
        print("  %-8s %10.4f ms %11.4f ms %11.4f ms %10.3f %6d"
              % (mode, r["cost_p50"], r["cost_p99"], r["cost_max"],
                 r["period_p99"], r["dropped"]))
        record("hotpath", **r)
    if os.path.exists(tmp):
        os.remove(tmp)


# ---------------------------------------------------------------------------
# 3. the index
# ---------------------------------------------------------------------------
def exp3_index(r):
    print("\n=== 3. what the index buys " + "=" * 45)
    t0, t1 = r.chunks[0]["t0"], r.chunks[-1]["t1"]
    a = t0 + int(0.45 * (t1 - t0))
    b = a + int(0.10 * (t1 - t0))
    print("  reading a 10 %% time window out of a %.1f s log" % ((t1 - t0) / NS))
    print("  method        messages   bytes read    time")
    for use_index in (True, False):
        t = time.perf_counter()
        n = sum(1 for _ in r.messages(t0=a, t1=b, use_index=use_index))
        el = (time.perf_counter() - t) * 1e3
        print("  %-12s %9d %11d %7.2f ms"
              % ("index" if use_index else "full scan", n, r.bytes_read, el))
        record("index", method="index" if use_index else "scan", messages=n,
               bytes_read=r.bytes_read, ms=el)
    rows = [x for x in ROWS if x["experiment"] == "index"]
    print("  -> the index read %.1fx fewer bytes for the same %d messages"
          % (rows[1]["bytes_read"] / rows[0]["bytes_read"], rows[0]["messages"]))


# ---------------------------------------------------------------------------
# 4. the bill
# ---------------------------------------------------------------------------
def exp4_bill(r):
    print("\n=== 4. the bill " + "=" * 56)
    per_topic = collections.Counter()
    for m in r.messages():
        per_topic[m["topic"]] += len(m["data"])
    dur = (r.chunks[-1]["t1"] - r.chunks[0]["t0"]) / NS
    total = sum(per_topic.values())
    print("  topic                        payload   share    per hour")
    for topic, b in per_topic.most_common():
        print("  %-26s %8.1f kB %6.1f %% %9.1f MB"
              % (topic, b / 1024, 100 * b / total, b / dur * 3600 / 1e6))
        record("bill", topic=topic, bytes=b, share_pct=100 * b / total,
               mb_per_hour=b / dur * 3600 / 1e6)
    size = os.path.getsize(LOG)
    gain = M.estimate_gain(LOG)
    print("  payload total %.1f kB, file %.1f kB (%.0f %% overhead: records,"
          % (total / 1024, size / 1024, 100 * (size - total) / total))
    print("  indexes, and the schema+channel preamble repeated in every chunk)")
    print("  compressing the whole file with zlib: %.2fx smaller" % gain)
    record("bill", topic="TOTAL", bytes=total, file_bytes=size,
           overhead_pct=100 * (size - total) / total, compression_est=gain)


# ---------------------------------------------------------------------------
# 5. replayability
# ---------------------------------------------------------------------------
def replay_trial(scheme):
    """Log one episode under a logging scheme, then try to recompute the
    controller's decisions from the log alone.

    This is the question a log exists to answer: *can I reproduce what the
    robot did?*  Nothing about MCAP makes the answer yes.  What you chose to
    put in it does.
    """
    rng = np.random.default_rng(11)
    env = A.PushEnv(rng)
    env.reset()
    path = os.path.join(OUT, "_replay.mcap")
    w = M.McapWriter(path, chunk_bytes=64 * 1024)
    w.add_schema(1, "robot.State", "jsonschema", M.to_json(JOINT_SCHEMA))
    w.add_channel(1, 1, "/state")
    w.add_channel(2, 1, "/action")

    truth, k, done = [], 0, False
    while not done:
        a, _ = A.expert_action(env, side=1, rng=env.rng)
        tip = env.arm.tip(env.q)
        if scheme == "poses only":
            payload = {"tip": [float(v) for v in tip],
                       "puck": [float(v) for v in env.puck]}
        elif scheme == "poses + joints, every 4th step":
            payload = None if k % 4 else {
                "q": [float(v) for v in env.q],
                "qd": [float(v) for v in env.qd],
                "puck": [float(v) for v in env.puck],
                "goal": [float(v) for v in env.goal]}
        else:                                    # "full state, every step"
            payload = {"q": [float(v) for v in env.q],
                       "qd": [float(v) for v in env.qd],
                       "puck": [float(v) for v in env.puck],
                       "goal": [float(v) for v in env.goal]}
        if payload is not None:
            w.write(1, k + 1, int(k * A.CTRL_DT * NS), M.to_json(payload))
            w.write(2, k + 1, int(k * A.CTRL_DT * NS),
                    M.to_json({"a": [float(v) for v in a]}))
        truth.append(a.copy())
        env.step(a)
        k += 1
        done = env.t >= A.EP_LEN or env.success
    size = w.close()

    # --- the replay -------------------------------------------------------
    rd = M.McapReader(path)
    states = [json.loads(m["data"]) for m in rd.messages(topics=["/state"])]
    actions = [json.loads(m["data"])["a"] for m in rd.messages(topics=["/action"])]
    rp = A.PushEnv(np.random.default_rng(0))
    rp.reset()
    ok, worst = 0, 0.0
    for s, a_logged in zip(states, actions):
        if "q" not in s:
            continue           # the log physically cannot answer the question
        rp.q = np.array(s["q"]); rp.qd = np.array(s["qd"])
        rp.puck = np.array(s["puck"]); rp.goal = np.array(s["goal"])
        a_re, _ = A.expert_action(rp, side=1, rng=np.random.default_rng(0))
        e = float(np.abs(np.array(a_re) - np.array(a_logged)).max())
        worst = max(worst, e)
        ok += e < 1e-9
    os.remove(path)
    return dict(scheme=scheme, decisions=len(truth), reproduced=ok,
                frac=ok / len(truth), worst_err=worst, bytes=size)


def exp5_replay():
    print("\n=== 5. replayability is not a property of the format " + "=" * 19)
    print("  what was logged                   bytes   decisions reproduced exactly")
    for scheme in ("poses only", "poses + joints, every 4th step",
                   "full state, every step"):
        r = replay_trial(scheme)
        print("  %-32s %6d %8d / %d  (%.0f %%)"
              % (scheme, r["bytes"], r["reproduced"], r["decisions"],
                 100 * r["frac"]))
        record("replay", **r)
    print("  -> the log format was identical in all three rows.")


if __name__ == "__main__":
    reader = exp1_write()
    exp2_hotpath()
    exp3_index(reader)
    exp4_bill(reader)
    exp5_replay()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
