"""Project 43 — train an agent that has never played the real game.

    python3 run.py --stage data       # ~1 min
    python3 run.py --stage world      # ~3 min
    python3 run.py --stage dream      # ~8 min   (three imagination horizons)
    python3 run.py --stage baseline   # ~2 min
    python3 run.py --stage figures    # ~1 min

Everything is measured against one number: coins collected per 100 steps of the
REAL game.  The dreaming agents never see it during training.

The experiment has a fixed budget of REAL_STEPS transitions of real experience.
Every method gets exactly that budget, so the comparison is about what you do
with your data, not how much of it you have.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import world_lib as W                                          # noqa: E402
import dream_lib as D                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.set_num_threads(12)

REAL_STEPS = 20000            # the shared budget of real experience
WORLD_STEPS, WORLD_BATCH, WORLD_LR = 3000, 256, 3e-4
DREAM_UPDATES, DREAM_BATCH, DREAM_LR = 400, 128, 3e-3
HORIZONS = [5, 15, 30]
EVAL_ENVS, EVAL_STEPS = 64, 200


def stage_data():
    t0 = time.time()
    f0, a, r, f1 = D.collect(REAL_STEPS, seed=43)
    torch.save(dict(f0=f0, a=a, r=r, f1=f1), CK / "buffer.pt")
    print(f"{len(f0)} real transitions, {float(r.sum()):.0f} coins "
          f"({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# stage: world — learn the simulator
# --------------------------------------------------------------------------

def stage_world():
    buf = torch.load(CK / "buffer.pt")
    f0, a, r, f1 = buf["f0"], buf["a"], buf["r"], buf["f1"]
    torch.manual_seed(43)
    world = D.DreamWorld()
    opt = torch.optim.AdamW(world.parameters(), lr=WORLD_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, WORLD_STEPS)
    g = torch.Generator().manual_seed(43)
    n = len(f0)
    n_train = int(n * 0.9)
    t0, log = time.time(), []
    for step in range(1, WORLD_STEPS + 1):
        i = torch.randint(0, n_train, (WORLD_BATCH,), generator=g)
        pred, rlog = world(f0[i], a[i])
        loss = F.mse_loss(pred, f1[i]) + 0.1 * F.binary_cross_entropy_with_logits(
            rlog, r[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log.append([step, f"{loss.item():.5f}"])
            print(f"  world {step:5d}/{WORLD_STEPS} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)")
    torch.save(world.state_dict(), CK / "world.pt")

    # how good is the dream, really?
    world.eval()
    with torch.no_grad():
        i = torch.arange(n_train, n)
        pred, rlog = world(f0[i], a[i])
        mse = float(F.mse_loss(pred, f1[i]))
        rp = (torch.sigmoid(rlog) > 0.5).float()
        rec = float(((rp == r[i]).float()).mean())
        prec = float((rp * r[i]).sum() / rp.sum().clamp(min=1))
        recall = float((rp * r[i]).sum() / r[i].sum().clamp(min=1))
    # agent-cell accuracy of a 1-step prediction and of a 15-step dream
    acc1 = _dream_accuracy(world, steps=1)
    acc15 = _dream_accuracy(world, steps=15)
    print(f"  held-out frame MSE {mse:.5f}   reward acc {rec:.3f} "
          f"(precision {prec:.3f}, recall {recall:.3f})")
    print(f"  player in the right cell: after 1 imagined step {acc1:.3f}, "
          f"after 15 {acc15:.3f}")
    with open(OUT / "world.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_mse", "reward_acc", "reward_precision",
                     "reward_recall", "agent_acc_1", "agent_acc_15"])
        wr.writerow([f"{mse:.5f}", f"{rec:.4f}", f"{prec:.4f}",
                     f"{recall:.4f}", f"{acc1:.4f}", f"{acc15:.4f}"])
    # a picture of a dream next to reality
    _dream_strip(world)


@torch.no_grad()
def _dream_accuracy(world, steps, n=128, seed=77):
    """Imagine `steps` random moves and check where the player ended up."""
    vec = D.VecGame(n, seed=seed)
    f = vec.reset()
    rng = np.random.default_rng(seed)
    truth = [(e.walls, e.agent) for e in vec.envs]
    fake = f.clone()
    for _ in range(steps):
        acts = torch.from_numpy(rng.integers(0, W.N_ACT, size=n))
        fake = world(fake, acts)[0].clamp(0, 1)
        new = []
        for i, (walls, ag) in enumerate(truth):
            dr, dc = W.DELTA[int(acts[i])]
            nr, nc = ag[0] + dr, ag[1] + dc
            new.append((walls, ag if walls[nr, nc] else (nr, nc)))
        truth = new
    ok = 0
    for i in range(n):
        _, ai, _, _ = W.read_frame(fake[i].numpy())
        ok += ((ai // W.GRID, ai % W.GRID) == truth[i][1])
    return ok / n


@torch.no_grad()
def _dream_strip(world, n_steps=12, seed=5):
    vec = D.VecGame(1, seed=seed)
    f = vec.reset()
    rng = np.random.default_rng(seed)
    real, fake = [f[0].numpy()], [f[0].numpy()]
    imagined = f.clone()
    for _ in range(n_steps):
        a = torch.from_numpy(rng.integers(0, W.N_ACT, size=1))
        imagined = world(imagined, a)[0].clamp(0, 1)
        f, _ = vec.step(a)
        real.append(f[0].numpy())
        fake.append(imagined[0].numpy())
    W.strip_image([real, fake], OUT / "dream_vs_real.png")
    W.write_gif(fake, OUT / "dream.gif", fps=3)
    W.write_gif(real, OUT / "real.gif", fps=3)
    print("dream_vs_real.png: top = the real game, bottom = the dream, "
          "same random buttons")


# --------------------------------------------------------------------------
# stage: dream — train the policy without touching the game
# --------------------------------------------------------------------------

def stage_dream():
    buf = torch.load(CK / "buffer.pt")
    starts = buf["f0"]
    world = D.DreamWorld()
    world.load_state_dict(torch.load(CK / "world.pt"))
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    rows, curves = [], {}
    for H in HORIZONS:
        torch.manual_seed(43)
        g = torch.Generator().manual_seed(43)
        policy = D.Policy()
        opt = torch.optim.AdamW(policy.parameters(), lr=DREAM_LR)
        t0, curve = time.time(), []
        for upd in range(1, DREAM_UPDATES + 1):
            i = torch.randint(0, len(starts), (DREAM_BATCH,), generator=g)
            loss, dreamed = D.imagine(world, policy, starts[i], H)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if upd % 50 == 0:
                real = D.evaluate(policy, EVAL_ENVS, EVAL_STEPS)
                curve.append([upd, dreamed / H * 100, real])
                print(f"  H={H:2d} update {upd:4d}  dreamed "
                      f"{dreamed / H * 100:5.2f}  real {real:5.2f} "
                      f"({time.time()-t0:.0f}s)")
        torch.save(policy.state_dict(), CK / f"policy_h{H}.pt")
        real = D.evaluate(policy, EVAL_ENVS, EVAL_STEPS)
        rows.append(dict(method=f"dream H={H}", real_coins_per_100=real,
                         dreamed_coins_per_100=curve[-1][1],
                         real_steps_used=REAL_STEPS))
        curves[f"h{H}"] = np.array(curve)
    np.savez(CK / "dream_curves.npz", **curves)
    _append_results(rows)


# --------------------------------------------------------------------------
# stage: baseline — the same budget spent the model-free way
# --------------------------------------------------------------------------

def _reinforce_real(total_steps, n_env=64, horizon=15, gamma=0.95, lr=3e-3,
                    seed=43, label=""):
    """The SAME actor-critic as the dream, but stepping the REAL game.

    This is the fair yardstick for the dreaming agents: identical policy,
    identical update rule, identical bootstrap.  The only thing that changes is
    where the transitions come from — the real environment instead of the world
    model — and how many of them the method is charged for.
    """
    torch.manual_seed(seed)
    policy = D.Policy()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    vec = D.VecGame(n_env, seed=seed + 7)
    f = vec.reset()
    used, t0 = 0, time.time()
    while used < total_steps:
        logps, ents, vals, rews = [], [], [], []
        for _ in range(horizon):
            a, logp, ent, v = policy.act(f)
            f, r = vec.step(a)
            logps.append(logp); ents.append(ent); vals.append(v); rews.append(r)
            used += n_env
        with torch.no_grad():
            _, boot = policy(f)
        logps = torch.stack(logps); ents = torch.stack(ents)
        vals = torch.stack(vals); rews = torch.stack(rews)
        rets, run = [], boot
        for t in reversed(range(horizon)):
            run = rews[t] + gamma * run
            rets.append(run)
        rets = torch.stack(rets[::-1])
        adv = (rets - vals).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = (-(logps * adv).mean() + 0.5 * F.mse_loss(vals, rets.detach())
                - 0.02 * ents.mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        f = f.detach()
    print(f"  {label}: {used} real steps in {time.time()-t0:.0f}s")
    return policy


def stage_baseline():
    rows = []
    rnd = D.evaluate(None, EVAL_ENVS, EVAL_STEPS)
    rows.append(dict(method="random buttons", real_coins_per_100=rnd,
                     dreamed_coins_per_100="", real_steps_used=0))
    print(f"  random: {rnd:.2f}")

    p_small = _reinforce_real(REAL_STEPS, label="model-free, same budget")
    torch.save(p_small.state_dict(), CK / "policy_mf_small.pt")
    v = D.evaluate(p_small, EVAL_ENVS, EVAL_STEPS)
    rows.append(dict(method="model-free, same budget",
                     real_coins_per_100=v, dreamed_coins_per_100="",
                     real_steps_used=REAL_STEPS))
    print(f"  model-free @ {REAL_STEPS}: {v:.2f}")

    big = REAL_STEPS * 20
    p_big = _reinforce_real(big, label="model-free, 20x the data")
    torch.save(p_big.state_dict(), CK / "policy_mf_big.pt")
    v = D.evaluate(p_big, EVAL_ENVS, EVAL_STEPS)
    rows.append(dict(method="model-free, 20x the data",
                     real_coins_per_100=v, dreamed_coins_per_100="",
                     real_steps_used=big))
    print(f"  model-free @ {big}: {v:.2f}")

    sc = D.evaluate_scripted(EVAL_ENVS, EVAL_STEPS)
    rows.append(dict(method="scripted (reads hidden state)",
                     real_coins_per_100=sc, dreamed_coins_per_100="",
                     real_steps_used=0))
    print(f"  scripted upper bound: {sc:.2f}")
    _append_results(rows)


RESULTS = None


def _append_results(rows):
    path = OUT / "results.csv"
    old = list(csv.DictReader(open(path))) if path.exists() else []
    keep = [r for r in old if r["method"] not in {x["method"] for x in rows}]
    allr = keep + [{k: (f"{v:.3f}" if isinstance(v, float) else v)
                    for k, v in r.items()} for r in rows]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["method", "real_coins_per_100",
                                           "dreamed_coins_per_100",
                                           "real_steps_used"])
        wr.writeheader()
        wr.writerows(allr)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

ORDER = ["random buttons", "model-free, same budget", "dream H=5",
         "dream H=15", "dream H=30", "model-free, 20x the data",
         "scripted (reads hidden state)"]


def stage_figures():
    res = {r["method"]: r for r in csv.DictReader(open(OUT / "results.csv"))}
    curves = np.load(CK / "dream_curves.npz")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax = axes[0]
    ps.style_axes(ax)
    meth = [m for m in ORDER if m in res]
    vals = [float(res[m]["real_coins_per_100"]) for m in meth]
    cols = []
    for m in meth:
        cols.append(ps.SERIES[1] if m.startswith("dream")
                    else (ps.BASELINE if "scripted" in m or "20x" in m
                          else ps.SERIES[0]))
    ax.barh(np.arange(len(meth)), vals, color=cols)
    ax.set_yticks(np.arange(len(meth)))
    ax.set_yticklabels(meth, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8,
                color=ps.INK_SECONDARY)
    ax.set_title("Coins per 100 real steps", color=ps.INK, fontsize=11,
                 loc="left")
    ax.set_xlabel("coins / 100 steps", color=ps.INK_SECONDARY, fontsize=9)

    ax = axes[1]
    ps.style_axes(ax)
    for i, key in enumerate(curves.files):
        c = curves[key]
        ax.plot(c[:, 0], c[:, 1], color=ps.SERIES[i], ls="--", lw=1.3,
                label=f"{key}: what it dreamed")
        ax.plot(c[:, 0], c[:, 2], color=ps.SERIES[i], lw=2.0,
                label=f"{key}: what it really got")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Dreamed reward vs. real reward", color=ps.INK, fontsize=11,
                 loc="left")
    ax.set_xlabel("policy updates", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("coins / 100 steps", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'results.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["data", "world", "dream", "baseline", "figures"])
    a = p.parse_args()
    {"data": stage_data, "world": stage_world, "dream": stage_dream,
     "baseline": stage_baseline, "figures": stage_figures}[a.stage]()


if __name__ == "__main__":
    main()
