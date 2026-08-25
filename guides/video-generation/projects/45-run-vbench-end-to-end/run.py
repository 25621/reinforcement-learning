"""Project 45 — Run a (miniature) VBench end to end.

Stages
    train      train the shared Phase-10 generator, save it for 46-50
    bench      score a roster of models on every axis; show one number hides it
    fragility  hold the model fixed, wobble the eval protocol, watch the number move
    figures    draw everything

Why a miniature VBench instead of the real one
----------------------------------------------
The real VBench downloads a multi-gigabyte T2V model and scores hundreds of
prompts on a GPU for hours.  The *lessons* it teaches — that one number hides
which axis a model fails on, and that "just reproduce the leaderboard" is
fragile — do not need that scale.  We reproduce them on the 16x16 sprite toy in
minutes, with axes that are exact instead of learned, so the point is not buried
under a black-box metric.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import eval_lib as E

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
def stage_train(args):
    torch.manual_seed(0)
    ds = E.make_dataset(3000, seed=1)
    net = E.VideoGen(base=32)
    print(f"generator: {E.count_params(net):,} params")
    t = time.time()
    losses = E.train(net, ds, steps=args.steps, batch=128, lr=2e-3, seed=0)
    print(f"trained in {time.time() - t:.0f}s")
    E.save_gen(net, "base", base=32)
    with open(OUT / "train_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for i, l in enumerate(losses):
            if i % 20 == 0:
                w.writerow([i, f"{l:.5f}"])


# ---------------------------------------------------------------------------
# the roster of "models": the trained generator plus deliberate degradations,
# each engineered to break exactly ONE axis, so we can watch the axes catch them.
# ---------------------------------------------------------------------------

def _gauss_blur(clips, sigma=1.8, r=3):
    k = torch.tensor([np.exp(-(i - r) ** 2 / (2 * sigma ** 2))
                      for i in range(2 * r + 1)], dtype=torch.float32)
    k = (k / k.sum()).view(1, 1, 1, 2 * r + 1)
    x = torch.as_tensor(clips)[:, :, None]                     # (B,T,1,h,w)
    B, T = x.shape[:2]
    x = x.reshape(B * T, 1, x.shape[-2], x.shape[-1])
    x = F.conv2d(x, k, padding=(0, r))
    x = F.conv2d(x, k.transpose(-1, -2), padding=(r, 0))
    return x.reshape(B, T, clips.shape[-2], clips.shape[-1]).numpy()


def _flicker(clips, rng, amp=0.7, frac=0.14):
    clips = np.array(clips)
    mask = rng.random(clips.shape) < frac
    return np.clip(clips + mask * rng.uniform(0, amp, clips.shape), 0, 1)


def _freeze(clips):
    clips = np.array(clips)
    return np.repeat(clips[:, :1], clips.shape[1], axis=1)      # hold frame 0


def build_roster(net, caps, rng):
    """Return {model_name: clips} for a fixed set of captions."""
    cap_t = torch.as_tensor(caps)
    base = E.sample(net, cap_t, steps=25, scale=2.0,
                    generator=torch.Generator().manual_seed(7)).numpy()
    base = np.clip(base, 0, 1)
    # "ignores the prompt": generate for a shuffled caption, score against true
    perm = rng.permutation(len(caps))
    wrong = E.sample(net, cap_t[perm], steps=25, scale=2.0,
                     generator=torch.Generator().manual_seed(7)).numpy()
    y0, x0 = _starts(caps, rng)
    oracle_ds = {"h": E.H, "w": E.W, "shape": caps[:, 0], "dir": caps[:, 1],
                 "speed": caps[:, 2], "y0": y0, "x0": x0}
    return {
        "real (oracle)": np.clip(
            E.render_batch(oracle_ds, np.arange(len(caps))).numpy(), 0, 1),
        "base generator": base,
        "blurry": _gauss_blur(base),
        "flickery": _flicker(base, rng),
        "frozen (no motion)": _freeze(base),
        "ignores prompt": np.clip(wrong, 0, 1),
    }


def _starts(caps, rng):
    r = np.random.default_rng(0)
    y = np.zeros(len(caps), np.float32)
    x = np.zeros(len(caps), np.float32)
    for i, c in enumerate(caps):
        y[i], x[i] = E.sample_start(c[0], c[1], c[2], r)
    return y, x


def stage_bench(args):
    net = E.load_gen("base")
    rng = np.random.default_rng(3)
    ds = E.make_dataset(400, seed=42)
    idx = np.arange(300)
    caps = E.caption_tensor(ds, idx).numpy()
    roster = build_roster(net, caps, rng)

    rows = []
    for name, clips in roster.items():
        s = E.vbench_score(clips, caps)
        s["model"] = name
        rows.append(s)
        print(f"{name:20s} " + "  ".join(
            f"{k[:4]}:{s[k]:.2f}" for k in E.AXES))
    axes = list(E.AXES) + ["overall"]
    with open(OUT / "bench.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + axes)
        for r in rows:
            w.writerow([r["model"]] + [f"{r[a]:.4f}" for a in axes])
    # a few example clips for the picture
    np.save(OUT / "_roster.npy",
            {n: c[:6] for n, c in roster.items()}, allow_pickle=True)
    np.save(OUT / "_caps.npy", caps[:6])


# ---------------------------------------------------------------------------
def stage_fragility(args):
    """Hold the model fixed; change only the eval protocol.  Watch the number.

    We report the aggregate VBench-style 'overall' score — the single number a
    leaderboard would print.  (Text-alignment alone saturates at 1.0 on this toy
    — itself a lesson: a maxed-out metric cannot rank anything — so the aggregate
    is the honest thing to track.)
    """
    net = E.load_gen("base")
    ds = E.make_dataset(4000, seed=99)
    rows = []

    def score(n_prompts, steps, scale, seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(ds["shape"]), size=n_prompts)
        caps = E.caption_tensor(ds, idx)
        gen = E.sample(net, caps, steps=steps, scale=scale,
                       generator=torch.Generator().manual_seed(seed)).numpy()
        return E.vbench_score(np.clip(gen, 0, 1), caps.numpy())["overall"]

    print("varying number of prompts (steps=25, scale=2, seed=0):")
    for n in [16, 48, 128, 384]:
        v = score(n, 25, 2.0, 0)
        rows.append(("n_prompts", n, v))
        print(f"  {n:4d} prompts -> {v:.3f}")

    print("varying random seed (n=128, steps=25, scale=2):")
    seed_vals = []
    for s in range(6):
        v = score(128, 25, 2.0, s)
        seed_vals.append(v)
        rows.append(("seed", s, v))
        print(f"  seed {s} -> {v:.3f}")

    print("varying sampling steps (n=128, scale=2, seed=0):")
    for st in [5, 10, 20, 40]:
        v = score(128, st, 2.0, 0)
        rows.append(("steps", st, v))
        print(f"  {st:2d} steps -> {v:.3f}")

    print("varying CFG scale (n=128, steps=25, seed=0):")
    for sc in [1.0, 1.5, 2.0, 3.0, 5.0]:
        v = score(128, 25, sc, 0)
        rows.append(("cfg", sc, v))
        print(f"  scale {sc} -> {v:.3f}")

    with open(OUT / "fragility.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["knob", "value", "overall_score"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}"])
    print(f"\nseed-only spread: {max(seed_vals) - min(seed_vals):.3f} "
          f"(the same model, same protocol, different luck)")


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- the toy: caption -> clip ----
    ds = E.make_dataset(40, seed=5)
    rows = []
    for i in range(4):
        rows.append(list(E.render_batch(ds, [i]).numpy()[0]))
        print("  " + E.caption(ds["shape"][i], ds["dir"][i], ds["speed"][i]))
    E.strip(rows, OUT / "the_toy.png", scale=7)

    # ---- per-axis bars ----
    if (OUT / "bench.csv").exists():
        names, data = [], {a: [] for a in list(E.AXES) + ["overall"]}
        with open(OUT / "bench.csv") as f:
            for row in csv.DictReader(f):
                names.append(row["model"])
                for a in data:
                    data[a].append(float(row[a]))
        axes = list(E.AXES)
        fig, ax = plt.subplots(figsize=(11, 4.6))
        x = np.arange(len(names))
        wgt = 0.15
        for j, a in enumerate(axes):
            ax.bar(x + (j - 2) * wgt, data[a], wgt, label=a.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("axis score (1 = perfect)")
        ax.set_title("Each degradation breaks a different axis — "
                     "a single number would hide which")
        ax.legend(fontsize=8, ncol=5, loc="lower center")
        ax.set_ylim(0, 1.15)
        fig.tight_layout()
        fig.savefig(OUT / "axes.png", dpi=110)
        plt.close(fig)

        # overall vs a human ranking cartoon
        fig, ax = plt.subplots(figsize=(8, 3.6))
        order = np.argsort(data["overall"])[::-1]
        ax.barh([names[i] for i in order][::-1],
                [data["overall"][i] for i in order][::-1], color="#c98a2b")
        ax.set_xlabel("single 'overall' number (mean of axes)")
        ax.set_title("Rank by one number and a frozen still image\n"
                     "outranks the real generator", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "single_number.png", dpi=110)
        plt.close(fig)

    # ---- roster example strip ----
    if (OUT / "_roster.npy").exists():
        roster = np.load(OUT / "_roster.npy", allow_pickle=True).item()
        rows = [[c[0][t] for t in range(E.T)] for c in roster.values()]
        E.strip(rows, OUT / "roster_clips.png", scale=5)
        for name in ["base generator", "frozen (no motion)", "flickery"]:
            if name in roster:
                E.write_gif(roster[name][0],
                            OUT / f"clip_{name.split()[0]}.gif")

    # ---- fragility ----
    if (OUT / "fragility.csv").exists():
        rows = list(csv.DictReader(open(OUT / "fragility.csv")))
        knobs = ["n_prompts", "seed", "steps", "cfg"]
        titles = ["# prompts", "random seed", "sampling steps", "CFG scale"]
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
        allv = [float(r["overall_score"]) for r in rows]
        for ax, kn, ti in zip(axs, knobs, titles):
            sub = [r for r in rows if r["knob"] == kn]
            xs = [float(r["value"]) for r in sub]
            ys = [float(r["overall_score"]) for r in sub]
            ax.plot(xs, ys, "o-", color="#2b6fc9")
            ax.set_title(ti, fontsize=10)
            ax.set_xlabel("value")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("aggregate 'leaderboard' score")
        axs[0].set_ylim(min(allv) - 0.03, max(allv) + 0.03)
        fig.suptitle("Same model, same weights — the leaderboard number moves "
                     "with the eval protocol alone", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "fragility.png", dpi=110)
        plt.close(fig)
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "bench", "fragility", "figures"])
    ap.add_argument("--steps", type=int, default=1800)
    stage = ap.parse_args()
    {"train": stage_train, "bench": stage_bench,
     "fragility": stage_fragility, "figures": stage_figures}[stage.stage](stage)
