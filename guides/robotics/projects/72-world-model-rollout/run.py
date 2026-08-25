"""Plan by imagining video, and score the plan against a goal picture.

Six experiments:

1. the model -- one-step error, and what it looks like after six imagined steps
2. planning on raw pixel distance to the goal image
3. why it fails: count the pixels
4. three repairs -- reweight the puck, use the puck channel alone, use the
   model's own features
5. imagined score vs realised score: the planner exploiting its own model
6. the price: this against project 60's state-space planner

Run:  python3 run.py     (about 8 minutes; needs numpy, torch, matplotlib)
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import video as V          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
N_EPS = 12
HORIZON = 5
REPLAN = 5
POP = 24
ITERS = 2


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<50s} {value:>10}   {note}")


def run_planner(model, cost_name, cost_fn, n=N_EPS, seed=999, record_first=0):
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    ok, errs, frames = 0, [], []
    imagined, realised = [], []
    for ep in range(n):
        env.reset()
        goal = V.goal_image(env)
        plan = None
        for t in range(A.EP_LEN):
            frame = V.render(env)
            if t % REPLAN == 0:
                plan, _ = V.cem_plan(model, frame, goal, cost_fn,
                                     horizon=HORIZON, pop=POP, iters=ITERS,
                                     rng=rng)
                # what the model THINKS this plan achieves
                pred = V.imagine(model, frame,
                                 torch.tensor(plan[None]))
                imagined.append(float(cost_fn(pred, torch.tensor(goal))[0]))
            a = plan[t % REPLAN]
            _, _, done, info = env.step(a)
            if t % REPLAN == REPLAN - 1 and len(imagined) > len(realised):
                real = torch.tensor(V.render(env))[None]
                realised.append(float(cost_fn(real, torch.tensor(goal))[0]))
            if ep < record_first:
                frames.append(V.render(env))
            if done:
                break
        ok += info["success"]
        errs.append(info["err"])
    k = min(len(imagined), len(realised))
    return {"name": cost_name, "success": ok / n, "err": float(np.mean(errs)),
            "imagined": float(np.mean(imagined[:k])),
            "realised": float(np.mean(realised[:k])), "frames": frames}


def main():
    t0 = time.time()
    # Training likes many threads; PLANNING does not.  A CEM iteration is a
    # batch of 24 tiny 32x32 convolutions, and splitting that across twelve
    # cores spends more time at the thread barrier than on the arithmetic --
    # the same effect project 64 measured on a 64 x 64 matmul, and the reason
    # the first version of this script spent forty minutes in section 4.
    torch.set_num_threads(min(12, os.cpu_count()))

    # -- 1. the model --------------------------------------------------------
    print("\n[1] training the video model on play data")
    X, Aa, Y = V.collect_play(160, seed=0)
    record("data", "transitions of play", len(X),
           f"{X.nbytes / 1e6:.1f} MB of {V.IMG}x{V.IMG} frames")
    model, hist = V.train_model(X, Aa, Y, epochs=14, log="video")
    Xv, Av, Yv = V.collect_play(30, seed=77)
    with torch.no_grad():
        pred = model(torch.tensor(Xv), torch.tensor(Av))
        one_step = float(((pred - torch.tensor(Yv)) ** 2).mean())
        copy_in = float(((torch.tensor(Xv) - torch.tensor(Yv)) ** 2).mean())
    record("model", "one-step MSE", f"{one_step:.5f}",
           f"copy-the-input baseline {copy_in:.5f} "
           f"({copy_in / max(one_step, 1e-12):.1f}x worse)")

    # multi-step drift, measured on held-out play
    rng = np.random.default_rng(5)
    env = A.PushEnv(rng)
    drift = np.zeros(HORIZON + 2)
    cnt = 0
    for _ in range(30):
        env.reset()
        f = V.render(env)
        acts = rng.uniform(-1, 1, (HORIZON + 2, V.ACT_DIM)).astype(np.float32)
        x = torch.tensor(f)[None]
        for k in range(HORIZON + 2):
            with torch.no_grad():
                x = model(x, torch.tensor(acts[k])[None])
            env.step(acts[k])
            drift[k] += float(((x[0].numpy() - V.render(env)) ** 2).mean())
        cnt += 1
    drift /= cnt
    for k in (0, 2, HORIZON - 1, HORIZON + 1):
        record("model", f"imagined-vs-real MSE after {k + 1} steps",
               f"{drift[k]:.5f}")

    # -- 2/3/4. planning -----------------------------------------------------
    torch.set_num_threads(2)
    print("\n[2] planning on the raw picture")
    res = {}
    res["pixel"] = run_planner(model, "pixel L2", V.cost_pixel, record_first=1)
    record("plan", "raw pixel L2 to the goal image",
           round(res["pixel"]["success"], 3),
           f"final puck error {res['pixel']['err'] * 1000:.0f} mm")

    print("\n[3] counting the pixels")
    env.reset()
    fr = V.render(env)
    arm_px = float(fr[0].sum())
    puck_px = float(fr[1].sum())
    record("pixels", "arm pixels (channel 0)", round(arm_px, 1))
    record("pixels", "puck pixels (channel 1)", round(puck_px, 1))
    record("pixels", "arm : puck", round(arm_px / max(puck_px, 1e-9), 2),
           "how much louder the arm is in a pixel loss")
    # how much of the achievable L2 improvement is puck-related
    gi = V.goal_image(env)
    d = (fr - gi) ** 2
    record("pixels", "share of the pixel error in the puck channel",
           round(float(d[1].sum() / max(d.sum(), 1e-9)), 3))

    print("\n[4] three repairs")
    res["w10"] = run_planner(model, "puck x10",
                             lambda p, g: V.cost_pixel(p, g, w_puck=10.0))
    record("plan", "puck channel weighted 10x", round(res["w10"]["success"], 3),
           f"final puck error {res['w10']['err'] * 1000:.0f} mm")
    res["puck"] = run_planner(model, "puck only", V.cost_puck_only,
                              record_first=1)
    record("plan", "puck channel only", round(res["puck"]["success"], 3),
           f"final puck error {res['puck']['err'] * 1000:.0f} mm")
    res["latent"] = run_planner(model, "latent",
                                lambda p, g: V.cost_latent(model, p, g))
    record("plan", "distance in the model's own features",
           round(res["latent"]["success"], 3),
           f"final puck error {res['latent']['err'] * 1000:.0f} mm")

    # -- 5. imagined vs realised --------------------------------------------
    print("\n[5] what the planner believed, and what happened")
    for k, r in res.items():
        gapr = r["realised"] / max(r["imagined"], 1e-12)
        record("optimism", f"{r['name']}: imagined cost", f"{r['imagined']:.5f}",
               f"realised {r['realised']:.5f}  ({gapr:.2f}x)")

    # -- 6. the price --------------------------------------------------------
    print("\n[6] the price of thinking in pixels")
    t_pix = time.time()
    _ = run_planner(model, "timing", V.cost_puck_only, n=3, seed=31337)
    per_ep = (time.time() - t_pix) / 3
    record("cost", "seconds per episode of pixel planning", round(per_ep, 2),
           f"{POP} x {HORIZON} x {ITERS} model calls per replan")
    record("cost", "model parameters",
           sum(p.numel() for p in model.parameters()))
    record("cost", "total runtime (s)", round(time.time() - t0, 1))

    # ---------------- figures ----------------------------------------------
    env2 = A.PushEnv(np.random.default_rng(3))
    env2.reset()
    f0 = V.render(env2)
    acts = np.random.default_rng(1).uniform(-1, 1, (6, 2)).astype(np.float32)
    imgs_pred, imgs_true = [f0], [f0]
    x = torch.tensor(f0)[None]
    for k in range(6):
        with torch.no_grad():
            x = model(x, torch.tensor(acts[k])[None])
        imgs_pred.append(x[0].numpy())
        env2.step(acts[k])
        imgs_true.append(V.render(env2))
    fig, ax = plt.subplots(2, 7, figsize=(13, 4.1))
    for k in range(7):
        for r, imgs in enumerate([imgs_true, imgs_pred]):
            im = np.stack([imgs[k][0], imgs[k][1], np.zeros_like(imgs[k][0])], -1)
            ax[r, k].imshow(np.clip(im, 0, 1))
            ax[r, k].axis("off")
        ax[0, k].set_title(f"t+{k}", fontsize=8)
    ax[0, 0].set_ylabel("real")
    ax[1, 0].set_ylabel("imagined")
    fig.suptitle("six imagined steps (red = arm, green = puck); top row is the truth",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rollout.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
    names = [res[k]["name"] for k in ("pixel", "w10", "puck", "latent")]
    vals = [res[k]["success"] for k in ("pixel", "w10", "puck", "latent")]
    ax[0].bar(names, vals, color=["#adb5bd", "#e9c46a", "#2a9d8f", "#457b9d"])
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel(f"success over {N_EPS} episodes")
    ax[0].set_title("what you compare the goal image against")
    plt.setp(ax[0].get_xticklabels(), fontsize=8, rotation=12)
    ax[1].plot(np.arange(1, len(drift) + 1), drift, "o-", c="#d1495b")
    ax[1].set_xlabel("imagined steps")
    ax[1].set_ylabel("MSE against the real frame")
    ax[1].set_title("imagination drifts")
    ax[2].plot(hist, c="#264653")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("training MSE")
    ax[2].set_yscale("log")
    ax[2].set_title("video model training")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "planning.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
