"""Behaviour cloning on a sim arm: six experiments.

  1. the task and the demonstrator          (is the expert actually good?)
  2. how much success one more demo buys    (the sample-efficiency curve)
  3. covariate shift, measured in two ways  (the reason BC is not enough)
  4. what BC failure actually looks like
  5. shaking the demonstrator's hand        (noise injection, DART-style)
  6. what the policy is allowed to see      (observation ablation)

Everything is written to outputs/results.csv; figures go to outputs/*.png.
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import arm as A
import nets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:34s} {value}  {unit}", flush=True)


def train_eval(O, Ac, seeds=(0, 1, 2), epochs=400, n_eval=60, feat="rel", **kw):
    """Train BC on the same data with several seeds; return per-seed results."""
    out = []
    for s in seeds:
        net, norm, hist = nets.train_bc(O, Ac, epochs=epochs, seed=s, **kw)
        r = A.evaluate(nets.make_policy(net, norm), n=n_eval, seed=999, feat=feat)
        out.append(dict(success=r["success"], err=r["err"], val=hist[-1][1],
                        net=net, norm=norm))
    return out


# ---------------------------------------------------------------------------
# 1. the task and the demonstrator
# ---------------------------------------------------------------------------
def exp1_task():
    rng = np.random.default_rng(3)
    env = A.PushEnv(rng)
    for side in (1, -1):
        res = [A.rollout(env, None, side=side) for _ in range(100)]
        record("1-demonstrator", f"expert success side {side:+d}",
               float(np.mean([r["success"] for r in res])))
        record("1-demonstrator", f"expert steps side {side:+d}",
               float(np.mean([r["steps"] for r in res])))

    # a picture of three demonstrations
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    rng = np.random.default_rng(11)
    env = A.PushEnv(rng)
    for ax in axes:
        r = A.rollout(env, None, side=1, record=True)
        for k, st in enumerate(r["states"][::8]):
            pts = env.arm.points(st[0])
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=plt.cm.viridis(k / 5),
                    lw=2, ms=3, alpha=0.8)
        ax.plot(r["tips"][:, 0], r["tips"][:, 1], "k-", lw=1, label="tip path")
        ax.plot(r["pucks"][:, 0], r["pucks"][:, 1], "r-", lw=2, label="puck path")
        ax.add_patch(plt.Circle(r["pucks"][0], A.R_PUCK, color="r", alpha=0.25))
        ax.add_patch(plt.Circle(env.goal, A.GOAL_TOL, color="g", alpha=0.25))
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 0.42)
        ax.set_ylim(-0.25, 0.32)
        ax.set_title(f"{r['steps']} steps, success={r['success']}")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Three expert demonstrations: circle behind the puck, then push")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task.png"), dpi=110)
    plt.close(fig)

    # what the demonstrations cost
    t = time.time()
    O, Ac, meta = A.collect_demos(50, seed=0, side_mode=1)
    record("1-demonstrator", "50 demos: transitions", len(O))
    record("1-demonstrator", "50 demos: wall clock", round(time.time() - t, 2), "s")
    record("1-demonstrator", "action |a| mean", round(float(np.abs(Ac).mean()), 3))
    record("1-demonstrator", "fraction of actions saturated",
           round(float((np.abs(Ac) > 0.99).mean()), 3))
    return O, Ac


# ---------------------------------------------------------------------------
# 2. sample efficiency
# ---------------------------------------------------------------------------
def exp2_data():
    counts = [10, 25, 50, 100, 200, 400]
    means, stds, vals, best = [], [], [], {}
    for nd in counts:
        O, Ac, _ = A.collect_demos(nd, seed=0, side_mode=1)
        res = train_eval(O, Ac)
        sc = [r["success"] for r in res]
        means.append(np.mean(sc))
        stds.append(np.std(sc))
        vals.append(np.mean([r["val"] for r in res]))
        record("2-sample-efficiency", f"{nd:3d} demos: success",
               f"{np.mean(sc):.3f} +- {np.std(sc):.3f}")
        record("2-sample-efficiency", f"{nd:3d} demos: val action MSE",
               round(float(np.mean([r["val"] for r in res])), 4))
        best[nd] = res[int(np.argmax(sc))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(counts, means, yerr=stds, marker="o", capsize=4)
    axes[0].axhline(1.0, ls="--", c="g", label="expert")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("demonstrations")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("More demonstrations, more success")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(vals, means, "o-")
    for c, v, m in zip(counts, vals, means):
        axes[1].annotate(str(c), (v, m), textcoords="offset points", xytext=(5, 4))
    axes[1].set_xlabel("validation action MSE (what training optimises)")
    axes[1].set_ylabel("success rate (what you want)")
    axes[1].set_title("The loss is a proxy, and a loose one")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sample_efficiency.png"), dpi=110)
    plt.close(fig)
    return best


# ---------------------------------------------------------------------------
# 3. covariate shift
# ---------------------------------------------------------------------------
def exp3_shift(best):
    """The gap between 'error on the expert's states' and 'error on mine'."""
    r50 = best[50]
    pol = nets.make_policy(r50["net"], r50["norm"])

    # held-out EXPERT states (the distribution training came from)
    Oe, Ae, _ = A.collect_demos(30, seed=555, side_mode=1)
    mse_expert = nets.action_mse(r50["net"], r50["norm"], Oe, Ae)

    # states the POLICY visits, labelled by asking the expert what it would do
    rng = np.random.default_rng(777)
    env = A.PushEnv(rng)
    Op, Ap, per_step = [], [], [[] for _ in range(A.EP_LEN)]
    for _ in range(30):
        obs = env.reset()
        for t in range(A.EP_LEN):
            a_exp, _ = A.expert_action(env, side=1)
            a_pol = pol(obs)
            Op.append(obs.copy())
            Ap.append(a_exp)
            per_step[t].append(float(((a_pol - a_exp) ** 2).mean()))
            obs, _, done, _ = env.step(a_pol)
            if done:
                break
    mse_policy = nets.action_mse(r50["net"], r50["norm"], np.array(Op), np.array(Ap))
    record("3-covariate-shift", "action MSE on expert states", round(mse_expert, 4))
    record("3-covariate-shift", "action MSE on policy states", round(mse_policy, 4))
    record("3-covariate-shift", "ratio", round(mse_policy / mse_expert, 2), "x")

    # how far from the training data does the policy drift?
    Otr, _, _ = A.collect_demos(50, seed=0, side_mode=1)
    mu, sd = Otr.mean(0), Otr.std(0) + 1e-3
    Ztr = (Otr - mu) / sd
    rng = np.random.default_rng(778)
    env = A.PushEnv(rng)
    nn_per_step = [[] for _ in range(A.EP_LEN)]
    for _ in range(15):
        obs = env.reset()
        for t in range(A.EP_LEN):
            z = (obs - mu) / sd
            nn_per_step[t].append(float(np.sqrt(((Ztr - z) ** 2).sum(1)).min()))
            obs, _, done, _ = env.step(pol(obs))
            if done:
                break
    steps = [np.mean(x) for x in per_step if x]
    nns = [np.mean(x) for x in nn_per_step if x]
    record("3-covariate-shift", "action error at step 1", round(steps[0], 4))
    record("3-covariate-shift", "action error at step 20",
           round(steps[min(19, len(steps) - 1)], 4))
    record("3-covariate-shift", "nearest-demo distance step 1", round(nns[0], 2))
    record("3-covariate-shift", "nearest-demo distance step 20",
           round(nns[min(19, len(nns) - 1)], 2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(["expert\nstates", "policy's own\nstates"], [mse_expert, mse_policy],
                color=["tab:green", "tab:red"])
    axes[0].set_ylabel("action MSE")
    axes[0].set_title(f"Same policy, {mse_policy / mse_expert:.1f}x the error")
    axes[1].plot(steps, "r-")
    axes[1].set_xlabel("step within the episode")
    axes[1].set_ylabel("action MSE vs expert")
    axes[1].set_title("Errors compound with time")
    axes[1].grid(alpha=0.3)
    axes[2].plot(nns, "b-")
    axes[2].set_xlabel("step within the episode")
    axes[2].set_ylabel("distance to nearest demo state")
    axes[2].set_title("...because the states drift away")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "covariate_shift.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. failure modes
# ---------------------------------------------------------------------------
def failure_modes(pol, n=60, seed=1234):
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    modes = dict(success=0, never_touched=0, pushed_away=0, ran_out_of_time=0)
    for _ in range(n):
        r = A.rollout(env, pol, record=True)
        d0 = float(np.linalg.norm(r["pucks"][0] - env.goal))
        moved = float(np.linalg.norm(r["pucks"][-1] - r["pucks"][0]))
        if r["success"]:
            modes["success"] += 1
        elif moved < 0.01:
            modes["never_touched"] += 1
        elif r["err"] > d0:
            modes["pushed_away"] += 1
        else:
            modes["ran_out_of_time"] += 1
    return {k: v / n for k, v in modes.items()}


def exp4_failures(best):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, nd in zip(axes, (50, 400)):
        r = best[nd]
        m = failure_modes(nets.make_policy(r["net"], r["norm"]))
        for k, v in m.items():
            record("4-failure-modes", f"{nd} demos: {k}", round(v, 3))
        ax.bar(list(m.keys()), list(m.values()),
               color=["tab:green", "tab:gray", "tab:red", "tab:orange"])
        ax.set_title(f"{nd} demonstrations")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("fraction of episodes")
    fig.suptitle("How behaviour cloning fails")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "failure_modes.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. noise injection
# ---------------------------------------------------------------------------
def exp5_noise():
    sigmas = [0.0, 0.05, 0.15, 0.30]
    means, stds, demo_ok = [], [], []
    for sg in sigmas:
        O, Ac, meta = A.collect_demos(50, seed=0, side_mode=1, noise=sg)
        res = train_eval(O, Ac)
        sc = [r["success"] for r in res]
        means.append(np.mean(sc))
        stds.append(np.std(sc))
        demo_ok.append(meta["n"] / meta["tries"])
        record("5-noise-injection", f"sigma {sg:.2f}: BC success",
               f"{np.mean(sc):.3f} +- {np.std(sc):.3f}")
        record("5-noise-injection", f"sigma {sg:.2f}: demos usable",
               round(meta["n"] / meta["tries"], 3))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=4, label="BC success")
    ax.plot(sigmas, demo_ok, "s--", c="gray", label="fraction of demos that succeed")
    ax.set_xlabel("noise added to the demonstrator's action (sigma)")
    ax.set_ylabel("rate")
    ax.set_title("A shakier demonstrator, a better student")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise_injection.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. observation ablation
# ---------------------------------------------------------------------------
def exp6_obs():
    out = {}
    for feat in ("rel", "raw"):
        O, Ac, _ = A.collect_demos(50, seed=0, side_mode=1, feat=feat)
        res = train_eval(O, Ac, feat=feat)
        sc = [r["success"] for r in res]
        out[feat] = np.mean(sc)
        record("6-observation", f"feat={feat}: success",
               f"{np.mean(sc):.3f} +- {np.std(sc):.3f}")
        record("6-observation", f"feat={feat}: val action MSE",
               round(float(np.mean([r["val"] for r in res])), 4))

    # Blind to joint velocity: is a still picture of the scene enough?
    # The mask has to be applied at TEST time as well as during training.
    # Zeroing the column only in the training data and then handing the policy
    # real velocities at run time does not measure "no velocity" -- it measures
    # a distribution mismatch that the policy has never seen, and it scores far
    # worse for that reason alone.
    def blind(o):
        o = np.array(o, np.float32)
        o[4:6] = 0.0
        return o

    O, Ac, _ = A.collect_demos(50, seed=0, side_mode=1)
    O_novel = O.copy()
    O_novel[:, 4:6] = 0.0
    sc = []
    for s in (0, 1, 2):
        net, norm, _ = nets.train_bc(O_novel, Ac, epochs=400, seed=s)
        pol = nets.make_policy(net, norm)
        sc.append(A.evaluate(lambda o: pol(blind(o)), n=60, seed=999)["success"])
    record("6-observation", "no joint velocity: success",
           f"{np.mean(sc):.3f} +- {np.std(sc):.3f}")
    out["no velocity"] = float(np.mean(sc))

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["rel", "raw", "no velocity"]
    ax.bar(["relative +\nabsolute", "absolute\nonly", "no joint\nvelocity"],
           [out[k] for k in keys], color=["tab:green", "tab:red", "tab:blue"])
    ax.set_ylabel("success rate")
    ax.set_title("50 demonstrations, same network, different inputs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "observations.png"), dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    exp1_task()
    best = exp2_data()
    exp3_shift(best)
    exp4_failures(best)
    exp5_noise()
    exp6_obs()

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s -> outputs/results.csv")


if __name__ == "__main__":
    main()
