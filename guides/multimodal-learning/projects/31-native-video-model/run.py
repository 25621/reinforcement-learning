"""Native video model: 3D patches instead of a pile of pictures.

Project 30 fed a language model 8 separately-encoded frames. This project throws
the language model away and asks a narrower question: if you build the video
model *natively* -- cutting the clip into boxes that span several frames at once
-- what does that buy, and what does it cost?

Same clips as project 30 (imported from its `video_lib.py`), three labels per
clip, five tokenisations, everything else held fixed.

    content    "does the clip contain a triangle?"   one frame is enough
    speed      "slow or fast?"                       needs 2+ frames, any order
    direction  "left / right / up / down?"           needs 2+ frames IN ORDER

Stages
    train   one arm (or `--arm all`)                 ~2 min each
    plot    accuracy against cost

Arms: framewise-pool, framewise-attend, tubelet2, tubelet4, tubelet8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "30-video-frame-vlm"))
import plot_style as ps  # noqa: E402
import tube  # noqa: E402
import video_lib as VD  # noqa: E402

OUT = HERE / "outputs"
N_CLIPS, N_VAL = 6000, 1000
SIZE = 32
STEPS, BS, LR = 800, 64, 1e-3
ARMS = {
    "framewise-pool": dict(mode="framewise-pool", tube_t=1),
    "framewise-attend": dict(mode="tubelet", tube_t=1),
    "tubelet2": dict(mode="tubelet", tube_t=2),
    "tubelet4": dict(mode="tubelet", tube_t=4),
    "tubelet8": dict(mode="tubelet", tube_t=8),
}
LABELS = {"direction": list(VD.DIRECTIONS), "speed": list(VD.SPEEDS),
          "content": ["no", "yes"]}


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def dataset():
    """Clips at half resolution plus the three labels.

    Downsampling 64 -> 32 by averaging 2x2 blocks costs nothing here (the
    objects are 12 pixels wide, so 6 pixels wide after) and quarters the number
    of tokens, which is the difference between a two-minute arm and an
    eight-minute one.
    """
    clips, labels = VD.make_dataset(N_CLIPS)      # ~5 s to render, no disk cache
    x = np.empty((N_CLIPS, VD.FRAMES, SIZE, SIZE, 3), dtype=np.float32)
    for i in range(0, N_CLIPS, 500):               # in chunks: the float copy of
        blk = clips[i:i + 500].reshape(-1, VD.FRAMES, SIZE, 2, SIZE, 2,
                                       3).astype(np.float32)   # all 6,000 clips
        x[i:i + 500] = blk.mean((3, 5)) / 127.5 - 1.0          # would be 2.4 GB
    x = torch.from_numpy(x)
    y = {
        "direction": torch.tensor([LABELS["direction"].index(l["direction"])
                                   for l in labels]),
        "speed": torch.tensor([LABELS["speed"].index(l["speed"]) for l in labels]),
        "content": torch.tensor([int("triangle" in (l["mover"] + " " + l["other"]))
                                 for l in labels]),
    }
    return x, y, labels


# left/right are one axis, up/down the other. Which axis a thing moves along is
# visible without knowing the order of the frames; which way along it is not.
AXIS = torch.tensor([0 if d in ("left", "right") else 1
                     for d in LABELS["direction"]])


def evaluate(model, x, y, bs=250):
    """Accuracy per task, plus the one number that separates two failure modes.

    `direction_axis` scores a prediction correct if it names the right *axis*
    (horizontal vs vertical), even with the wrong sign. A model that sees the
    frames as an unordered set can still reach 1.0 here while scoring 0.5 on
    direction itself, because the sign is the only part that needs order.
    """
    model.eval()
    hits = {k: 0 for k in y}
    hits["direction_axis"] = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out = model(x[i:i + bs])
            for k in y:
                hits[k] += int((out[k].argmax(1) == y[k][i:i + bs]).sum())
            pred = out["direction"].argmax(1)
            hits["direction_axis"] += int(
                (AXIS[pred] == AXIS[y["direction"][i:i + bs]]).sum())
    model.train()
    return {k: v / len(x) for k, v in hits.items()}


def stage_train(arm):
    torch.set_num_threads(12)
    x, y, _ = dataset()
    xtr, xte = x[:-N_VAL], x[-N_VAL:]
    ytr = {k: v[:-N_VAL] for k, v in y.items()}
    yte = {k: v[-N_VAL:] for k, v in y.items()}
    torch.manual_seed(0)
    model = tube.VideoViT(size=SIZE, frames=VD.FRAMES, **ARMS[arm])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(0)
    curve, t0 = [], time.time()
    for step in range(STEPS):
        i = torch.from_numpy(rng.integers(0, len(xtr), BS))
        out = model(xtr[i])
        loss = tube.multi_loss(out, {k: v[i] for k, v in ytr.items()})
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(float(loss.detach()))
        if step % 100 == 0:
            print(f"  step {step:3d}  loss {np.mean(curve[-50:]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    took = time.time() - t0
    acc = evaluate(model, xte, yte)
    # a second reading of the same weights: shuffle the frames at test time
    perm = torch.from_numpy(np.random.default_rng(1).permutation(VD.FRAMES))
    acc_shuf = evaluate(model, xte[:, perm], yte)
    row = {"arm": arm, "tokens": model.n_tokens,
           "attention_pairs": model.attention_pairs(),
           "params": model.n_params(), "s_per_step": took / STEPS,
           "seconds": took, "steps": STEPS,
           "acc": acc, "acc_shuffled": acc_shuf, "curve": curve}
    print(f"  {arm}: tokens {model.n_tokens}, {took/STEPS*1000:.0f} ms/step, "
          + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()))
    print("    shuffled frames: " + "  ".join(f"{k} {v:.3f}" for k, v in acc_shuf.items()))
    old = json.loads((OUT / "arms.json").read_text()) if (OUT / "arms.json").exists() else []
    _save("arms.json", [r for r in old if r["arm"] != arm] + [row])


def stage_baseline():
    """Majority-class accuracy for each task -- the real "chance" line."""
    _, y, _ = dataset()
    base = {k: float(torch.bincount(v[-N_VAL:]).max()) / N_VAL for k, v in y.items()}
    _save("baseline.json", {"majority_class": base, "n_val": N_VAL})
    print(" ", base)


def stage_plot():
    rows = json.loads((OUT / "arms.json").read_text())
    base = json.loads((OUT / "baseline.json").read_text())["majority_class"]
    order = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    rows = {r["arm"]: r for r in rows}
    tasks = ["content", "speed", "direction"]

    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ps.style_axes(axes[0])
    x = np.arange(len(order))
    w = 0.26
    for k, task in enumerate(tasks):
        vals = [rows[a]["acc"][task] for a in order]
        axes[0].bar(x + (k - 1) * w, vals, w - 0.02, color=ps.SERIES[k], label=task)
        for xi, v in zip(x + (k - 1) * w, vals):
            axes[0].text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7,
                         color=ps.INK_SECONDARY)
        axes[0].axhline(base[task], color=ps.BASELINE, ls="--", lw=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.replace("framewise-", "frame\n") for a in order],
                            fontsize=8)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    axes[0].set_title("accuracy by tokenisation (dashed = majority class)",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[0].set_ylabel("accuracy, 1,000 held-out clips", color=ps.INK_SECONDARY,
                       fontsize=10)

    ps.style_axes(axes[1])
    for i, a in enumerate(order):
        axes[1].scatter(rows[a]["s_per_step"] * 1000, rows[a]["acc"]["direction"],
                        s=70, color=ps.SERIES[i % len(ps.SERIES)], zorder=3)
        axes[1].annotate(f"{a} ({rows[a]['tokens']} tok)",
                         (rows[a]["s_per_step"] * 1000, rows[a]["acc"]["direction"]),
                         textcoords="offset points", xytext=(8, -3), fontsize=8,
                         color=ps.INK_SECONDARY)
    axes[1].axhline(base["direction"], color=ps.BASELINE, ls="--", lw=0.9)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("the motion question against training cost", color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("ms per training step", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("direction accuracy", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "arms.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/arms.png")

    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=110, sharey=True)
    fig.patch.set_facecolor(ps.SURFACE)
    x = np.arange(len(order))
    for ax, key, ttl, chance in [
            (axes[0], "direction", "which way (4 answers)", base["direction"]),
            (axes[1], "direction_axis",
             "which axis only (2 answers, no order needed)", 0.5)]:
        ps.style_axes(ax)
        for k, (field, name, colour) in enumerate([
                ("acc", "frames in order", ps.SERIES[0]),
                ("acc_shuffled", "frames shuffled at test time", ps.SERIES[2])]):
            vals = [rows[a][field][key] for a in order]
            ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=colour, label=name)
            for xi, v in zip(x + (k - 0.5) * 0.36, vals):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8,
                        color=ps.INK_SECONDARY)
        ax.axhline(chance, color=ps.BASELINE, ls="--", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("framewise-", "frame\n") for a in order],
                           fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(ttl, color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[0].set_ylabel("accuracy", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "shuffle.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/shuffle.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["train", "baseline", "plot"])
    p.add_argument("--arm", default="all", choices=list(ARMS) + ["all"])
    a = p.parse_args()
    if a.stage == "train":
        for arm in (ARMS if a.arm == "all" else [a.arm]):
            print(f"\n=== {arm} ===", flush=True)
            stage_train(arm)
    elif a.stage == "baseline":
        stage_baseline()
    else:
        stage_plot()
