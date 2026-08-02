"""Project 44 -- turning a block inside a two-finger hand, learned by search.

Seven experiments:

  1. the hand, the task, and one learned rollout
  2. the learning curve
  3. against a do-nothing baseline and the best open-loop push
  4. how far it can turn the block before it needs to let go
  5. the drop rate, and the trade it is making
  6. domain randomization, and what it costs in-domain
  7. can it work without seeing the block at all?

Runs in about seven minutes on CPU.
"""

import csv
import os
import sys
import time

import numpy as np

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import hand as H                                                              # noqa: E402
import ars                                                                    # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

ITERS = 150
N_EVAL = 40
# The block's pose sits in observation slots 8..13; zeroing them leaves the
# policy with joint angles, joint speeds and the goal only -- which is roughly
# what a real hand has when the camera is blocked by its own fingers.
BLIND_MASK = np.ones(18)
BLIND_MASK[8:14] = 0.0


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<52s} {value}")


def factory(**kw):
    def f(rng):
        return H.Hand(rng, **kw)
    return f


def random_factory(rng_outer=None):
    """Domain randomization: a different hand, physically, every episode."""
    def f(rng):
        return H.Hand(rng, mu=float(rng.uniform(0.7, 1.6)),
                      kp=float(rng.uniform(3.0, 5.5)),
                      block=float(rng.uniform(0.016, 0.020)),
                      density=float(rng.uniform(280, 620)))
    return f


def evaluate(pol, n=N_EVAL, seed=7, goal=None, **kw):
    rng = np.random.default_rng(seed)
    ok = dropped = 0
    errs = []
    for _ in range(n):
        h = H.Hand(rng, **kw)
        o = h.reset(goal)
        for _ in range(H.EP_LEN):
            o, _, done = h.step(pol(o))
            if done:
                break
        ok += h.success()
        dropped += h.dropped
        errs.append(abs(h.goal - h.angle()))
    return dict(success=ok / n, drop=dropped / n, err=float(np.mean(errs)))


def train(tag, fac, iters=ITERS, seed=0, **kw):
    t0 = time.time()

    def ev(W, norm):
        return evaluate(ars.policy_fn(W, norm, kw.get("obs_mask")), n=25)["success"]

    W, norm, curve, evals = ars.train(fac, 18, H.NU, iters=iters, seed=seed,
                                      log=lambda s: print(s), eval_every=25,
                                      eval_fn=ev, ep_len=H.EP_LEN, **kw)
    record("train", f"{tag}: training time (s)", round(time.time() - t0))
    return W, norm, curve, evals


# ---------------------------------------------------------------------------

def exp1_picture(state):
    print("\n[1] the hand and one learned episode")
    rng = np.random.default_rng(2)
    h = H.Hand(rng)
    cam = mujoco.Renderer(h.model, height=300, width=400)
    pol = ars.policy_fn(state["W"], state["norm"])
    goal = 0.35     # inside the range experiment 4 shows the hand can reach
    o = h.reset(goal)
    frames = [_shot(cam, h)]
    angles = [h.angle()]
    for k in range(H.EP_LEN):
        o, _, done = h.step(pol(o))
        angles.append(h.angle())
        if k % 34 == 0:
            frames.append(_shot(cam, h))
        if done:
            break
    frames.append(_shot(cam, h))
    cam.close()
    fig = plt.figure(figsize=(11.0, 5.0))
    for i, fr in enumerate(frames[:5]):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(fr)
        ax.set_title(f"step {i * 34}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(np.degrees(angles), label="block angle")
    ax.axhline(np.degrees(goal), ls="--", color=COLORS[1], label="goal")
    ax.axhspan(np.degrees(goal - 0.26), np.degrees(goal + 0.26), color=COLORS[2],
               alpha=0.12, label="counted as success")
    ax.set_xlabel("control step"); ax.set_ylabel("degrees")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "rollout.png"))
    record("1_picture", "goal for this episode (deg)", round(np.degrees(goal), 1))
    record("1_picture", "final angle (deg)", round(np.degrees(angles[-1]), 1))
    record("1_picture", "dropped", h.dropped)


def _shot(cam, h):
    cam.update_scene(h.data, camera="cam")
    return cam.render()


def exp2_curve(state):
    print("\n[2] the learning curve")
    curve, evals = state["curve"], state["evals"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    axes[0].plot(curve, lw=1.0)
    axes[0].set_xlabel("ARS iteration"); axes[0].set_ylabel("mean episode return")
    axes[1].plot([e[0] for e in evals], [100 * e[1] for e in evals], "o-", ms=4)
    axes[1].set_xlabel("ARS iteration"); axes[1].set_ylabel("success (%)")
    save(fig, os.path.join(OUT, "learning.png"))
    record("2_curve", "iterations", len(curve))
    record("2_curve", "episodes simulated", len(curve) * 8 * 2 * 2)
    record("2_curve", "policy parameters", 4 * 18)
    record("2_curve", "success at 25 iters", round(evals[0][1], 3))
    record("2_curve", "success at the end", round(evals[-1][1], 3))


def exp3_baselines(state):
    print("\n[3] baselines")
    rng = np.random.default_rng(5)
    zero = evaluate(lambda o: np.zeros(H.NU))
    # the best CONSTANT push, found by trying 150 of them -- this is the
    # control that says whether feedback is needed at all
    best, best_s = None, -1
    for _ in range(150):
        c = rng.uniform(-1, 1, H.NU)
        s = evaluate(lambda o, c=c: c, n=12, seed=3)["success"]
        if s > best_s:
            best, best_s = c, s
    const = evaluate(lambda o: best)
    learned = evaluate(ars.policy_fn(state["W"], state["norm"]))
    rows = {"do nothing (hold still)": zero,
            "best constant push (open loop)": const,
            "learned linear policy": learned}
    for k, v in rows.items():
        record("3_baselines", f"{k}: success", round(v["success"], 3))
        record("3_baselines", f"{k}: drop rate", round(v["drop"], 3))
        record("3_baselines", f"{k}: mean final error (deg)",
               round(np.degrees(v["err"]), 1))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ks = list(rows)
    ax.barh(ks, [100 * rows[k]["success"] for k in ks],
            color=[COLORS[6], COLORS[4], COLORS[2]])
    ax.set_xlabel("reached the goal angle within 15 deg (%)")
    save(fig, os.path.join(OUT, "baselines.png"))
    state["zero"] = zero


def exp45_cliff(state):
    print("\n[4+5] how far it can turn, and when it lets go")
    pol = ars.policy_fn(state["W"], state["norm"])
    goals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6])
    succ, drop, err = [], [], []
    for g in goals:
        r = evaluate(pol, n=24, seed=int(100 + 100 * g), goal=float(g))
        succ.append(r["success"])
        drop.append(r["drop"])
        err.append(r["err"])
        record("4_cliff", f"goal {np.degrees(g):.0f} deg: success",
               round(r["success"], 3))
        record("4_cliff", f"goal {np.degrees(g):.0f} deg: drop rate",
               round(r["drop"], 3))
        record("4_cliff", f"goal {np.degrees(g):.0f} deg: mean error (deg)",
               round(np.degrees(r["err"]), 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    axes[0].plot(np.degrees(goals), [100 * s for s in succ], "o-", ms=4,
                 label="reached the goal")
    axes[0].plot(np.degrees(goals), [100 * d for d in drop], "s-", ms=4,
                 label="dropped the block")
    axes[0].set_xlabel("goal rotation (deg)")
    axes[0].set_ylabel("% of 24 episodes")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.degrees(goals), np.degrees(err), "o-", ms=4,
                 label="error left over")
    axes[1].plot(np.degrees(goals), np.degrees(goals), "--", color="#8C8C8C",
                 lw=1, label="doing nothing at all")
    axes[1].set_xlabel("goal rotation (deg)")
    axes[1].set_ylabel("final error (deg)")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "cliff.png"))


def exp6_dr(state):
    print("\n[6] domain randomization")
    W2, n2, c2, e2 = train("randomized", random_factory(), seed=1, resample=True)
    state["W_dr"], state["norm_dr"] = W2, n2
    mus = [0.6, 0.8, 1.1, 1.4, 1.7]
    rows = {"trained on one hand": [], "trained on randomized hands": []}
    for mu in mus:
        rows["trained on one hand"].append(
            evaluate(ars.policy_fn(state["W"], state["norm"]), n=24, mu=mu)["success"])
        rows["trained on randomized hands"].append(
            evaluate(ars.policy_fn(W2, n2), n=24, mu=mu)["success"])
        for k in rows:
            record("6_dr", f"friction {mu}: {k}", round(rows[k][-1], 3))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for k, v in rows.items():
        ax.plot(mus, [100 * x for x in v], "o-", ms=4, label=k)
    ax.axvline(1.1, color="#8C8C8C", ls=":", lw=1)
    ax.text(1.12, 5, "value it trained on", fontsize=7.5)
    ax.set_xlabel("friction coefficient at the fingertips")
    ax.set_ylabel("success (%)")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "randomization.png"))


def exp7_blind(state):
    print("\n[7] without seeing the block")
    W3, n3, c3, e3 = train("blind", factory(), seed=2, obs_mask=BLIND_MASK)
    full = evaluate(ars.policy_fn(state["W"], state["norm"]))
    blind = evaluate(ars.policy_fn(W3, n3, BLIND_MASK))
    record("7_blind", "full observation: success", round(full["success"], 3))
    record("7_blind", "no block pose: success", round(blind["success"], 3))
    record("7_blind", "full observation: mean error (deg)",
           round(np.degrees(full["err"]), 1))
    record("7_blind", "no block pose: mean error (deg)",
           round(np.degrees(blind["err"]), 1))
    record("7_blind", "do nothing: mean error (deg)",
           round(np.degrees(state["zero"]["err"]), 1))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ks = ["do nothing", "no block pose\n(joints + goal only)", "full observation"]
    vs = [state["zero"]["success"], blind["success"], full["success"]]
    ax.barh(ks, [100 * v for v in vs], color=[COLORS[6], COLORS[4], COLORS[2]])
    ax.set_xlabel("success (%)")
    save(fig, os.path.join(OUT, "blind.png"))


def main():
    use_style()
    t0 = time.time()
    state = {}
    W, norm, curve, evals = train("nominal", factory())
    state.update(W=W, norm=norm, curve=curve, evals=evals)
    exp1_picture(state)
    exp2_curve(state)
    exp3_baselines(state)
    exp45_cliff(state)
    exp6_dr(state)
    exp7_blind(state)
    np.savez(os.path.join(_HERE, "policy.npz"), W=state["W"],
             mean=state["norm"].mean, m2=state["norm"].m2, n=state["norm"].n)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
