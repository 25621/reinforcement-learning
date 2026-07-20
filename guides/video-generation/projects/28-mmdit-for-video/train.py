"""Project 28 — does joint text-video attention (MMDiT) beat one-way
cross-attention at actually obeying the prompt?

    python3 train.py --stage probe               # ~2 min  the judge
    python3 train.py --stage train --arm cross       # ~8 min
    python3 train.py --stage train --arm mmdit       # ~9 min
    python3 train.py --stage train --arm cross_wide  # ~9 min
    python3 train.py --stage figures                 # ~5 min

Both arms are trained with rectified flow (project 26) on project 25's cached
latents, with the digit and direction labels turned into four-word prompts.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
import plot_style as ps                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

import dit_lib as L                                          # noqa: E402
import flow_lib as FL                                        # noqa: E402
import fid_lib                                               # noqa: E402
import mmdit_lib as M                                        # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"

DIM, DEPTH, HEADS = 128, 5, 4
STEPS, BATCH, LR = 4500, 16, 6e-4
DROP_PROMPT = 0.1          # fraction of steps trained with the empty prompt
SAMPLE_STEPS = 30
SCALES = [1.0, 3.0]
# `cross_wide` is the control that keeps the comparison honest: MMDiT gives
# the text stream its own projections and MLP, which costs ~57% more
# parameters than plain cross-attention.  If MMDiT then wins, a fair reader
# asks whether the win came from the wiring or from the extra weights.  So the
# third arm is cross-attention widened (dim 160 instead of 128) until its
# parameter count matches MMDiT's to within 0.5%.
ARMS = ["cross", "mmdit", "cross_wide"]
ARM_DIM = {"cross": DIM, "mmdit": DIM, "cross_wide": 160}


def build(arm):
    mode = "mmdit" if arm == "mmdit" else "cross"
    return M.TextVideoDiT(mode, dim=ARM_DIM[arm], depth=DEPTH, heads=HEADS)


# --------------------------------------------------------------------------
# stage: probe — an independent judge of "which digit is that?"
# --------------------------------------------------------------------------

def probe():
    """Train a digit classifier on REAL clips, to grade generated ones.

    Project 23's feature network already classifies digits, so why train
    another one?  Because that one was built to be a *feature extractor* for
    the FID proxy — it stops at 42% accuracy, which is plenty for comparing
    two clouds of feature vectors but far too vague to answer "is this a 7?"
    about a single clip.  A grader needs to be right about individual items,
    so it gets more steps and is trained on exactly the one-digit clips this
    phase generates.
    """
    torch.manual_seed(0)
    net = fid_lib.FeatureNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(7)
    for step in range(1, 1501):
        clips, dig, _ = L.attr_batch(rng, 32, T=2, train=True)
        x = clips[:, :, 0]                                # (B,1,H,W)
        loss = F.cross_entropy(net(x), dig)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[probe] {step} loss {loss.item():.3f}", flush=True)
    net.eval()
    rng = np.random.default_rng(1234)
    correct = total = 0
    with torch.no_grad():
        for _ in range(8):
            clips, dig, _ = L.attr_batch(rng, 64, T=2, train=False)
            correct += int((net(clips[:, :, 0]).argmax(1) == dig).sum())
            total += len(dig)
    acc = correct / total
    torch.save({"state": net.state_dict(), "acc": acc}, CK / "probe.pt")
    print(f"[probe] held-out digit accuracy {acc:.1%}")

    # The direction judge needs no training at all: the digit is the only
    # bright object, so where its centre of mass ends up IS the answer.
    rng = np.random.default_rng(99)
    clips, _, dirs = L.attr_batch(rng, 128, train=False)
    pred = L.predicted_direction(clips)
    dacc = float((pred == dirs).float().mean())
    print(f"[probe] centroid direction judge accuracy on real clips: "
          f"{dacc:.1%}")
    (OUT / "probe.txt").write_text(
        f"digit classifier held-out accuracy: {acc:.3f}\n"
        f"centroid direction judge accuracy on real clips: {dacc:.3f}\n")


def load_probe():
    ck = torch.load(CK / "probe.pt", map_location="cpu", weights_only=False)
    net = fid_lib.FeatureNet()
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["acc"]


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(arm):
    torch.manual_seed(0)
    cache = L.load_latent_cache("latents", where=P25)
    data, digit, direction = (cache["latents"], cache["digit"],
                              cache["direction"])
    model = build(arm)
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    print(f"[{arm}] {L.count_params(model):,} params", flush=True)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        ids = M.prompt_batch(digit[idx], direction[idx])
        drop = torch.rand(BATCH, generator=g) < DROP_PROMPT
        ids[drop] = M.null_batch(int(drop.sum()))
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE, ids),
                          flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm,
                "elapsed": time.time() - t0,
                "params": L.count_params(model)}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time()-t0:.0f}s")


def load_arm(arm):
    ck = torch.load(CK / f"{arm}.pt", map_location="cpu", weights_only=False)
    m = build(arm)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

ALL_PROMPTS = [(d, k) for d in range(10) for k in range(4)]


@torch.no_grad()
def adherence(model, net, scale, repeats=2, seed=11):
    """Generate every (digit, direction) prompt and grade the result."""
    digits = torch.tensor([p[0] for p in ALL_PROMPTS] * repeats)
    dirs = torch.tensor([p[1] for p in ALL_PROMPTS] * repeats)
    ids = M.prompt_batch(digits, dirs)
    flow = FL.RectifiedFlow()
    g = torch.Generator().manual_seed(seed)
    z = M.cfg_sample(model, flow, (len(ids), 4, 4, 8, 8), ids, scale=scale,
                     steps=SAMPLE_STEPS, generator=g)
    vae, sc = L.load_vae("3d")
    clips = vae.decoder(z / sc).clamp(-1, 1)
    # digit: average the classifier over four frames, so one bad frame does
    # not decide the verdict
    probs = 0
    for f in (0, 5, 10, 15):
        probs = probs + F.softmax(net(clips[:, :, f]), dim=1)
    dig_ok = (probs.argmax(1) == digits).float()
    dir_ok = (L.predicted_direction(clips) == dirs).float()
    return dict(digit_acc=float(dig_ok.mean()),
                direction_acc=float(dir_ok.mean()),
                both_acc=float((dig_ok * dir_ok).mean())), clips, digits, dirs


@torch.no_grad()
def figures():
    net, probe_acc = load_probe()
    ev = L.load_latent_cache("latents_eval", where=P25)
    reals = ev["clips"][:80]      # match the 80 generated clips per arm
    fnet = fid_lib.load_features()

    rows, showcase = [], {}
    for arm in ARMS:
        model, ck = load_arm(arm)
        for scale in SCALES:
            t0 = time.time()
            acc, clips, digits, dirs = adherence(model, net, scale)
            row = dict(arm=arm, cfg_scale=scale, params=ck["params"],
                       digit_acc=round(acc["digit_acc"], 3),
                       direction_acc=round(acc["direction_acc"], 3),
                       both_acc=round(acc["both_acc"], 3),
                       fid_proxy=round(fid_lib.frechet(reals, clips, fnet), 2),
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(row, flush=True)
            if scale == 3.0:
                showcase[arm] = (clips, digits, dirs)

    rows.append(dict(arm="chance", cfg_scale="", params="",
                     digit_acc=0.1, direction_acc=0.25, both_acc=0.025,
                     fid_proxy="", seconds=""))
    rows.append(dict(arm="real clips (judge's ceiling)", cfg_scale="",
                     params="", digit_acc=round(probe_acc, 3),
                     direction_acc=1.0, both_acc="",
                     fid_proxy=round(fid_lib.frechet(
                         reals, ev["clips"][80:160], fnet), 2), seconds=""))
    with open(OUT / "adherence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig_bars(rows)
    fig_grid(showcase)
    fig_loss()
    print("wrote", OUT)


def fig_bars(rows):
    data = [r for r in rows if r["arm"] in ARMS]
    fig, ax = ps.new_axes(7.8, 4.4)
    keys = ["digit_acc", "direction_acc", "both_acc"]
    labels = ["names the right digit", "moves the right way", "gets both right"]
    x = np.arange(len(keys))
    width = 0.8 / len(data)
    for i, r in enumerate(data):
        ax.bar(x + (i - (len(data) - 1) / 2) * width, [r[k] for k in keys],
               width, color=ps.SERIES[i % len(ps.SERIES)],
               label=f"{r['arm']}, guidance {r['cfg_scale']}")
    for j, ch in enumerate([0.1, 0.25, 0.025]):
        ax.hlines(ch, x[j] - 0.42, x[j] + 0.42, color=ps.INK_MUTED, ls=":",
                  lw=1.2)
    ax.text(x[0] - 0.42, 0.12, "chance", color=ps.INK_MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ps.finish(fig, ax, "Prompt adherence: 40 prompts x 2 samples each", "",
              "fraction correct", OUT / "adherence.png")
    plt.close(fig)


def fig_grid(showcase):
    """One row per arm for the same four prompts."""
    picks = [(3, 2), (7, 0), (0, 3), (5, 1)]
    fig, axes = plt.subplots(len(picks), len(ARMS),
                             figsize=(9.6, 1.5 * len(picks)), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r, (d, k) in enumerate(picks):
        for c, arm in enumerate(ARMS):
            clips, digits, dirs = showcase[arm]
            hit = ((digits == d) & (dirs == k)).nonzero()[0, 0]
            ax = axes[r, c]
            ax.imshow(L.strip(clips[hit:hit + 1], n=6), cmap="gray",
                      vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"'{d} {L.DIRECTIONS[k]}'",
                              color=ps.INK_SECONDARY, fontsize=8)
            if r == 0:
                ax.set_title(arm, color=ps.INK, fontsize=10)
    fig.suptitle("Same prompt, both wirings (guidance 3.0)", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "prompt_grid.png", facecolor=ps.SURFACE)
    plt.close(fig)


def fig_loss():
    fig, ax = ps.new_axes(7.4, 4.0)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"log_{arm}.npy")
        k = 12
        sm = np.convolve(a[:, 2], np.ones(k) / k, mode="valid")
        ax.plot(a[k - 1:, 0], sm, color=ps.SERIES[i], lw=1.5, label=arm)
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Training loss (same objective, so comparable here)",
              "training step", "flow-matching MSE", OUT / "loss_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["probe", "train", "figures"])
    ap.add_argument("--arm", default="mmdit", choices=ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "probe":
        probe()
    elif args.stage == "train":
        train(args.arm)
    else:
        figures()
