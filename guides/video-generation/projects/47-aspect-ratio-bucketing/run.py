"""Project 47 — Aspect-ratio bucketing.

A batch of clips must be one tensor of one shape.  Two ways to get there:

  squish   resize every clip to one square shape (16x16), batch the squares,
           and at inference stretch the square output back to the target aspect
  bucket   sort clips by shape into buckets (tall / wide / square) and build
           each batch from a single bucket, so nothing is ever resized

We train one model each way and test both on portrait (24x12) and wide (12x24)
prompts.  The squish model has only ever seen circles that were secretly squashed
into ellipses, so it paints ellipses; the bucketed model paints round balls at
the right aspect.

Stages
    train    train the squish model and the bucketed model
    eval     score both on tall + wide test prompts
    figures  draw it
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "45-run-vbench-end-to-end"))
import eval_lib as E                                            # noqa: E402
import flow_lib as FL                                           # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CK = Path(__file__).resolve().parent / "checkpoints"
CK.mkdir(exist_ok=True)

BUCKETS = {"tall": (24, 12), "wide": (12, 24), "square": (16, 16)}
SQUARE = (16, 16)


def make_buckets(n_each, seed=0):
    return {name: E.make_dataset(n_each, seed=seed + i, h=h, w=w)
            for i, (name, (h, w)) in enumerate(BUCKETS.items())}


def _batch(ds, rng, batch):
    idx = rng.integers(0, len(ds["shape"]), size=batch)
    x = E.render_batch(ds, idx)
    cap = E.caption_tensor(ds, idx)
    return x, cap


def _step(net, flow, opt, x0, cap, g, drop=0.1):
    mask = torch.rand(cap.shape[0], generator=g) < drop
    if mask.any():
        cap = cap.clone()
        cap[mask] = E.null_caption(int(mask.sum()))
    noise = torch.randn(x0.shape, generator=g)
    t = flow.sample_t(x0.shape[0], generator=g)
    xt = flow.interpolate(x0, t, noise)
    pred = net(xt, t * flow.T_SCALE, cap)
    loss = F.mse_loss(pred, flow.target(x0, noise))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train_bucketed(buckets, steps, seed=0, batch=128):
    """Each step: pick one bucket, batch only from it (one tensor shape)."""
    net = E.VideoGen(base=32)
    flow = FL.RectifiedFlow()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    names = list(buckets)
    losses = []
    for s in range(steps):
        ds = buckets[names[s % len(names)]]        # round-robin the buckets
        x0, cap = _batch(ds, rng, batch)
        losses.append(_step(net, flow, opt, x0, cap, g))
        if s % 500 == 0:
            print(f"  bucket step {s:4d} loss {np.mean(losses[-100:]):.4f}")
    return net, losses


def train_squish(buckets, steps, seed=0, batch=128):
    """Each step: sample across buckets, resize every clip to 16x16, batch."""
    net = E.VideoGen(base=32)
    flow = FL.RectifiedFlow()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    names = list(buckets)
    losses = []
    for s in range(steps):
        # mixed batch: a third from each bucket, each resized to the square
        parts_x, parts_c = [], []
        for nm in names:
            x, c = _batch(buckets[nm], rng, batch // len(names))
            x = F.interpolate(x, size=SQUARE, mode="bilinear",
                              align_corners=False)
            parts_x.append(x)
            parts_c.append(c)
        x0 = torch.cat(parts_x)
        cap = torch.cat(parts_c)
        losses.append(_step(net, flow, opt, x0, cap, g))
        if s % 500 == 0:
            print(f"  squish step {s:4d} loss {np.mean(losses[-100:]):.4f}")
    return net, losses


# ---------------------------------------------------------------------------
def stage_train(args):
    torch.manual_seed(0)
    buckets = make_buckets(2500, seed=1)
    for name, fn in [("squish", train_squish), ("bucket", train_bucketed)]:
        print(f"=== {name} ===")
        t = time.time()
        net, _ = fn(buckets, args.steps)
        print(f"  {name} trained in {time.time() - t:.0f}s")
        E.save_gen(net, f"gen_{name}", base=32, where=CK)


# ---------------------------------------------------------------------------
def _gen_native(net, caps, h, w, squish=False):
    """Generate at aspect (h, w).  The squish model generates square then
    stretches — the naive fixed-resolution inference path."""
    cap_t = torch.from_numpy(caps).long()
    g = torch.Generator().manual_seed(0)
    if squish:
        sq = E.sample(net, cap_t, steps=25, scale=2.0, generator=g,
                      h=SQUARE[0], w=SQUARE[1]).clamp(0, 1)
        up = F.interpolate(sq, size=(h, w), mode="bilinear",
                           align_corners=False)
        return up.numpy()
    return E.sample(net, cap_t, steps=25, scale=2.0, generator=g,
                    h=h, w=w).clamp(0, 1).numpy()


def sprite_aspect(clips):
    """Height / width of the bright sprite region, averaged over frames.

    A correct round ball measures ~1.0.  The squish model generates a round ball
    at 16x16 and then stretches that square output to the target aspect, so on a
    tall frame the ball comes out taller than it is wide (ratio > 1) and on a
    wide frame flatter than it is tall (ratio < 1).  This reads the distortion
    off directly, where 1.0 is "still round".
    """
    vals = []
    for clip in clips:
        for f in np.asarray(clip, dtype=np.float32):
            ys, xs = np.where(f > 0.4)
            if len(ys) < 3:
                continue
            hgt = ys.max() - ys.min() + 1
            wid = xs.max() - xs.min() + 1
            vals.append(hgt / wid)
    return float(np.mean(vals)) if vals else 0.0


def stage_eval(args):
    rng = np.random.default_rng(7)
    rows = []
    saved = {}
    for aspect in ["tall", "wide"]:
        h, w = BUCKETS[aspect]
        # test on the ball prompts, where distortion is easiest to see
        ds = E.make_dataset(200, seed=100, h=h, w=w)
        idx = np.where(ds["shape"] == 0)[0][:120]              # balls only
        caps = E.caption_tensor(ds, idx).numpy()
        for model in ["squish", "bucket"]:
            net = E.load_gen(f"gen_{model}", where=CK)
            gen = _gen_native(net, caps, h, w, squish=(model == "squish"))
            al = E.text_alignment(gen, caps)
            asp = sprite_aspect(gen)
            print(f"{aspect:5s} {model:6s}  align {al['mean']:.2f}  "
                  f"shape {al['shape']:.2f}  ball h/w {asp:.2f} (1.0 = round)")
            rows.append([aspect, model, al["mean"], al["shape"],
                         al["speed"], asp])
            saved[(aspect, model)] = gen[:6]
    np.save(OUT / "_samples.npy", saved, allow_pickle=True)
    with open(OUT / "eval.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["aspect", "model", "align_mean", "shape", "speed",
                     "aspect_hw"])
        wr.writerows([[a, m, f"{x:.4f}", f"{d:.4f}", f"{s:.4f}", f"{r:.4f}"]
                      for a, m, x, d, s, r in rows])


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (OUT / "eval.csv").exists():
        rows = list(csv.DictReader(open(OUT / "eval.csv")))
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        labels = [f"{r['aspect']}\n{r['model']}" for r in rows]
        colors = ["#8a8f98" if r["model"] == "squish" else "#c98a2b"
                  for r in rows]
        # left: ball aspect ratio (1.0 = round)
        ax = axs[0]
        vals = [float(r["aspect_hw"]) for r in rows]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(1.0, ls="--", color="k", lw=1)
        ax.text(len(vals) - 0.5, 1.02, "round", fontsize=8, ha="right")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("ball height / width  (1.0 = round)", fontsize=10)
        # right: prompt following
        ax = axs[1]
        vals = [float(r["align_mean"]) for r in rows]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("prompt-following (mean of 3 attributes)", fontsize=10)
        fig.suptitle("Squish (grey) stretches balls into ellipses and follows "
                     "prompts worse;\nbucketing (amber) keeps them round",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "bucketing.png", dpi=110)
        plt.close(fig)

    if (OUT / "_samples.npy").exists():
        s = np.load(OUT / "_samples.npy", allow_pickle=True).item()
        for aspect in ["tall", "wide"]:
            rows = [list(s[(aspect, "squish")][0]),
                    list(s[(aspect, "bucket")][0])]
            E.strip(rows, OUT / f"{aspect}_squish_vs_bucket.png", scale=6)
            E.write_gif(s[(aspect, "squish")][0], OUT / f"{aspect}_squish.gif")
            E.write_gif(s[(aspect, "bucket")][0], OUT / f"{aspect}_bucket.gif")
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "eval", "figures"])
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    {"train": stage_train, "eval": stage_eval,
     "figures": stage_figures}[a.stage](a)
