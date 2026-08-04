"""Diffusion policy vs MLP behaviour cloning: six experiments.

  1. the multimodality, shown at a single state
  2. head-to-head on multimodal demonstrations
  3. the control: the SAME comparison on unimodal demonstrations
  4. how few denoising steps you can get away with (latency)
  5. chunk length and how much of the chunk to execute
  6. what it all costs per decision
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402
import dp                  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

N_DEMOS = 100
EPOCHS = 900
EVAL_N = 60
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:40s} {value}  {unit}", flush=True)


# ---------------------------------------------------------------------------
# one job = train one policy and evaluate it
# ---------------------------------------------------------------------------
def job(cfg):
    torch.set_num_threads(2)
    O, C, _ = dp.chunk_demos(N_DEMOS, horizon=cfg["horizon"], seed=cfg["seed"],
                             side_mode=cfg["side_mode"])
    t0 = time.time()
    if cfg["kind"] == "diffusion":
        model, norm, diff = dp.train_diffusion(O, C, epochs=EPOCHS, seed=cfg["seed"])
        pol = dp.ChunkPolicy("diffusion", model, norm, diff, cfg["horizon"],
                             cfg["exec_len"], cfg["n_steps"], seed=cfg["seed"])
    else:
        net, norm, h, ad = dp.train_mse_chunk(O, C, epochs=EPOCHS, seed=cfg["seed"])
        pol = dp.ChunkPolicy("mse", net, norm, None, cfg["horizon"],
                             cfg["exec_len"], seed=cfg["seed"])
    train_s = time.time() - t0

    t0 = time.time()
    ev = dp.evaluate_chunk(pol, n=EVAL_N, seed=999)
    ms = (time.time() - t0) / EVAL_N / A.EP_LEN * 1000
    out = dict(cfg)
    out.update(success=ev["success"], err=ev["err"], steps=ev["steps"],
               train_s=train_s, ms=ms)
    return out


def key(c):
    return (c["kind"], c["horizon"], c["exec_len"], c["n_steps"], c["side_mode"],
            c["seed"])


def cfg(kind, horizon=8, exec_len=None, n_steps=10, side_mode="random", seed=0):
    return dict(kind=kind, horizon=horizon,
                exec_len=horizon if exec_len is None else exec_len,
                n_steps=n_steps, side_mode=side_mode, seed=seed)


# ---------------------------------------------------------------------------
def exp1_picture():
    """What does 'the average of two good actions' actually look like?"""
    O, C, meta = dp.chunk_demos(60, horizon=8, seed=0, side_mode="random")
    model, norm, diff = dp.train_diffusion(O, C, epochs=EPOCHS, seed=0)
    net, mnorm, _, _ = dp.train_mse_chunk(O, C, epochs=EPOCHS, seed=0)

    # Find a state where the two circling directions genuinely disagree.
    # Not every state is ambiguous: once the tip is already behind the puck
    # both "sides" give the same push, and a scatter plot taken there shows one
    # blob and proves nothing.  The search below asks for a state where the two
    # expert actions are far apart -- which is exactly the definition of the
    # multimodality this project is about.
    rng = np.random.default_rng(5)
    env = A.PushEnv(rng)
    obs, best = env.reset(), -1.0
    for _ in range(60):
        env.reset()
        for _ in range(rng.integers(0, 5)):
            a, _ = A.expert_action(env, side=1)
            obs, _, _, _ = env.step(a * 0.3)
        ap, _ = A.expert_action(env, side=1)
        am, _ = A.expert_action(env, side=-1)
        gap = float(np.linalg.norm(ap - am))
        if gap > best:
            best, saved, obs_best = gap, env.state(), env.obs()
        if gap > 1.2:
            break
    env.set_state(saved)
    obs = obs_best
    a_plus, _ = A.expert_action(env, side=1)
    a_minus, _ = A.expert_action(env, side=-1)

    x = torch.tensor(norm(np.asarray(obs, np.float32))[None], dtype=torch.float32)
    xb = x.repeat(300, 1)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        samples = diff.sample(model, xb, 10, generator=gen)[:, 0].numpy()
        xm = torch.tensor(mnorm(np.asarray(obs, np.float32))[None], dtype=torch.float32)
        a_mse = net(xm)[0].numpy().reshape(8, -1)[0]

    d_modes = float(np.linalg.norm(a_plus - a_minus))
    record("1-multimodality", "distance between the two expert actions",
           round(d_modes, 3))
    record("1-multimodality", "MSE prediction to nearer expert action",
           round(float(min(np.linalg.norm(a_mse - a_plus),
                           np.linalg.norm(a_mse - a_minus))), 3))
    # how bimodal are the diffusion samples?  distance to each expert action
    dp_plus = np.linalg.norm(samples - a_plus, axis=1)
    dp_minus = np.linalg.norm(samples - a_minus, axis=1)
    frac_near = float(np.mean(np.minimum(dp_plus, dp_minus) < d_modes / 2))
    record("1-multimodality", "diffusion samples near one of the two modes",
           round(frac_near, 3))
    record("1-multimodality", "diffusion samples nearer the + mode",
           round(float(np.mean(dp_plus < dp_minus)), 3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.45,
               label="diffusion samples (300)")
    ax.scatter(*a_plus, s=160, marker="*", c="tab:green", label="expert, go left")
    ax.scatter(*a_minus, s=160, marker="*", c="tab:orange", label="expert, go right")
    ax.scatter(*a_mse, s=160, marker="X", c="tab:red", label="MSE prediction")
    ax.set_xlabel("action, joint 1")
    ax.set_ylabel("action, joint 2")
    ax.set_title("One state, two right answers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(np.minimum(dp_plus, dp_minus), bins=30, color="tab:blue",
            label="diffusion samples")
    ax.axvline(min(np.linalg.norm(a_mse - a_plus), np.linalg.norm(a_mse - a_minus)),
               c="tab:red", lw=2, label="MSE prediction")
    ax.set_xlabel("distance to the nearest expert action")
    ax.set_ylabel("count")
    ax.set_title("The average is far from both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multimodality.png"), dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    exp1_picture()

    jobs = []
    # 2. head to head on multimodal data (2 seeds)
    for s in (0, 1):
        jobs += [cfg("mse", horizon=1, seed=s), cfg("mse", horizon=8, seed=s),
                 cfg("diffusion", horizon=1, seed=s),
                 cfg("diffusion", horizon=8, seed=s)]
    # 3. the unimodal control
    for k, h in [("mse", 1), ("mse", 8), ("diffusion", 1), ("diffusion", 8)]:
        jobs.append(cfg(k, horizon=h, side_mode="1", seed=0))
    # 4. denoising steps
    for ns in (1, 2, 5, 20, 50):
        jobs.append(cfg("diffusion", horizon=8, n_steps=ns, seed=0))
    # 5. chunk length (open loop) and how much of the chunk to execute
    for h in (2, 4, 16):
        jobs.append(cfg("diffusion", horizon=h, seed=0))
    for ex in (1, 2, 4):
        jobs.append(cfg("diffusion", horizon=8, exec_len=ex, seed=0))
    jobs.append(cfg("mse", horizon=8, exec_len=1, seed=0))

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        res = list(ex.map(job, jobs))
    R = {key(r): r for r in res}
    print(f"[{time.time() - T0:6.1f}s] {len(res)} policies trained", flush=True)

    def get(**kw):
        c = cfg(**kw)
        return R[key(c)]

    def mean_over_seeds(**kw):
        v = [get(seed=s, **kw)["success"] for s in (0, 1)]
        return float(np.mean(v)), float(np.std(v))

    # -- 2. head to head ----------------------------------------------------
    names = [("mse", 1, "MLP, single action"),
             ("mse", 8, "MLP, 8-action chunk"),
             ("diffusion", 1, "diffusion, single action"),
             ("diffusion", 8, "diffusion, 8-action chunk")]
    multi = {}
    for kind, h, label in names:
        m, sd = mean_over_seeds(kind=kind, horizon=h)
        multi[(kind, h)] = m
        record("2-head-to-head", f"{label}: success", f"{m:.3f} +- {sd:.3f}")
        record("2-head-to-head", f"{label}: steps used when it succeeds",
               round(float(np.mean([get(seed=s, kind=kind, horizon=h)["steps"]
                                    for s in (0, 1)])), 1))

    # -- 3. unimodal control ------------------------------------------------
    uni = {}
    for kind, h in [("mse", 1), ("mse", 8), ("diffusion", 1), ("diffusion", 8)]:
        r = get(kind=kind, horizon=h, side_mode="1")
        uni[(kind, h)] = r["success"]
        record("3-unimodal-control", f"{kind} H={h} on one-sided demos: success",
               round(r["success"], 3))
    # Compare like with like.  Diffusion-with-a-chunk against MLP-without-one
    # changes two things at once, and the chunk turns out to matter more than
    # the distribution model, so that comparison answers the wrong question.
    record("3-unimodal-control", "diffusion - MLP at H=1, multimodal data",
           round(multi[("diffusion", 1)] - multi[("mse", 1)], 3))
    record("3-unimodal-control", "diffusion - MLP at H=1, unimodal data",
           round(uni[("diffusion", 1)] - uni[("mse", 1)], 3))
    record("3-unimodal-control", "diffusion - MLP at H=8, multimodal data",
           round(multi[("diffusion", 8)] - multi[("mse", 8)], 3))
    record("3-unimodal-control", "diffusion - MLP at H=8, unimodal data",
           round(uni[("diffusion", 8)] - uni[("mse", 8)], 3))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    lbl = ["MLP\n1 action", "MLP\n8-chunk", "diffusion\n1 action", "diffusion\n8-chunk"]
    axes[0].bar(lbl, [multi[("mse", 1)], multi[("mse", 8)],
                      multi[("diffusion", 1)], multi[("diffusion", 8)]],
                color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    axes[0].set_title("Demonstrations go both ways round (multimodal)")
    axes[0].set_ylabel("success rate")
    axes[1].bar(lbl, [uni[("mse", 1)], uni[("mse", 8)], uni[("diffusion", 1)],
                      uni[("diffusion", 8)]],
                color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    axes[1].set_title("Demonstrations always go the same way (unimodal)")
    fig.suptitle("The gap is caused by the DATA, not by the network")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "head_to_head.png"), dpi=110)
    plt.close(fig)

    # -- 4. denoising steps -------------------------------------------------
    steps, succ, msec = [], [], []
    for ns in (1, 2, 5, 10, 20, 50):
        r = get(kind="diffusion", horizon=8, n_steps=ns)
        steps.append(ns)
        succ.append(r["success"])
        msec.append(r["ms"])
        record("4-denoise-steps", f"{ns:2d} steps: success", round(r["success"], 3))
        record("4-denoise-steps", f"{ns:2d} steps: ms per decision", round(r["ms"], 3))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(steps, succ, "o-", label="success")
    ax.set_xscale("log")
    ax.set_xlabel("denoising steps at test time")
    ax.set_ylabel("success rate")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, msec, "s--", c="gray", label="ms/decision")
    ax2.set_ylabel("milliseconds per decision")
    ax.set_title("How many denoising steps does the robot need?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "denoise_steps.png"), dpi=110)
    plt.close(fig)

    # -- 5. chunk length ----------------------------------------------------
    hs, hsucc = [], []
    for h in (1, 2, 4, 8, 16):
        r = get(kind="diffusion", horizon=h) if h != 1 else get(kind="diffusion", horizon=1)
        hs.append(h)
        hsucc.append(r["success"])
        record("5-chunk", f"H={h:2d} (execute all): success", round(r["success"], 3))
    for ex in (1, 2, 4, 8):
        r = get(kind="diffusion", horizon=8, exec_len=ex)
        record("5-chunk", f"H=8, execute {ex} of the 8 actions: success",
               round(r["success"], 3))
    r_closed = get(kind="diffusion", horizon=8, exec_len=1)
    r_closed_mse = get(kind="mse", horizon=8, exec_len=1)
    record("5-chunk", "MLP H=8, execute only the first action: success",
           round(r_closed_mse["success"], 3))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(hs, hsucc, "o-", label="execute the whole chunk")
    ax.plot([8], [r_closed["success"]], "X", ms=12, c="tab:red",
            label="H=8, re-plan every step")
    ax.set_xlabel("chunk length H")
    ax.set_ylabel("success rate")
    ax.set_title("Committing to a plan beats re-deciding every step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chunk.png"), dpi=110)
    plt.close(fig)

    # -- 6. cost ------------------------------------------------------------
    record("6-cost", "MLP single action: ms/decision",
           round(get(kind="mse", horizon=1)["ms"], 3))
    record("6-cost", "diffusion 10 steps, chunk of 8: ms/decision",
           round(get(kind="diffusion", horizon=8)["ms"], 3))
    record("6-cost", "diffusion 10 steps, re-plan every step: ms/decision",
           round(r_closed["ms"], 3))
    record("6-cost", "training time MLP / diffusion",
           f"{get(kind='mse', horizon=8)['train_s']:.0f}s / "
           f"{get(kind='diffusion', horizon=8)['train_s']:.0f}s")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
