"""Project 49 — Invisible watermarking.

We stamp a hidden signal into the generator's clips and build a detector that
reads it back.  Then we push on the central tension: a mark strong enough to
survive cropping and compression is harder to keep invisible.

The mark is a *spread-spectrum* watermark.  We fix one secret pattern of +1/-1
noise (the key) the size of a whole clip, and add a faint copy of it to every
output.  "Spread-spectrum" because the one-bit message ("this is watermarked")
is spread thin across thousands of pixels: each pixel is nudged by an amount too
small to see, but a detector that correlates the whole clip against the key adds
those thousands of tiny nudges back up into a clear signal.  Real footage, which
has no correlation with the secret key, stays near zero.

Stages
    embed    generate clips, watermark them at several strengths, detect
    attack   test whether the mark survives noise, blur, rescale, and crop
    figures  draw the imperceptibility-vs-robustness trade-off
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "45-run-vbench-end-to-end"))
import eval_lib as E                                            # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

KEY = None


def get_key(seed=1234):
    global KEY
    if KEY is None:
        g = np.random.default_rng(seed)
        KEY = (g.integers(0, 2, size=(E.T, E.H, E.W)) * 2 - 1).astype(np.float32)
    return KEY


def embed(clips, alpha):
    """Add alpha * key to each clip, then clip back into [0, 1]."""
    return np.clip(np.asarray(clips) + alpha * get_key()[None], 0.0, 1.0)


def _highpass(clip):
    """Remove the slow-varying sprite so only key-scale detail is left."""
    x = torch.as_tensor(np.asarray(clip, dtype=np.float32))[:, None]  # (T,1,h,w)
    k = torch.ones(1, 1, 3, 3) / 9.0
    blur = F.conv2d(x, k, padding=1)
    return (x - blur)[:, 0].numpy()


def detect(clip):
    """Correlation of the high-passed clip with the secret key.

    Watermarked clips score high (the faint key adds up); real footage scores
    near zero (its detail is unrelated to the key).
    """
    hp = _highpass(clip)
    key = get_key()
    return float((hp * key).mean() / (np.abs(hp).mean() + 1e-6))


def psnr(a, b):
    mse = np.mean((np.asarray(a) - np.asarray(b)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(1.0 / mse))


# ---------------------------------------------------------------------------
def stage_embed(args):
    net = E.load_gen("base")
    ds = E.make_dataset(600, seed=5)
    caps = E.caption_tensor(ds, np.arange(240))
    clips = E.sample(net, caps, steps=25, scale=2.0,
                     generator=torch.Generator().manual_seed(0)).clamp(0, 1).numpy()
    real = E.render_batch(ds, np.arange(240, 480)).numpy()     # unwatermarked

    rows = []
    for alpha in [0.02, 0.04, 0.08, 0.15, 0.30]:
        wm = embed(clips, alpha)
        ps = np.mean([psnr(clips[i], wm[i]) for i in range(len(clips))])
        wm_scores = np.array([detect(c) for c in wm])
        real_scores = np.array([detect(c) for c in real])
        # threshold halfway between the two clouds
        thr = 0.5 * (wm_scores.mean() + real_scores.mean())
        tpr = float((wm_scores > thr).mean())                  # caught marks
        fpr = float((real_scores > thr).mean())                # false alarms
        # robustness: does it survive a compression-style low-pass (blur)?
        wb = _attack(wm, "blur")
        rb = _attack(real, "blur")
        wbs = np.array([detect(c) for c in wb])
        rbs = np.array([detect(c) for c in rb])
        thr_b = 0.5 * (wbs.mean() + rbs.mean())
        robust = float((wbs > thr_b).mean())
        rows.append(dict(alpha=alpha, psnr=ps, tpr=tpr, fpr=fpr, robust=robust))
        print(f"alpha {alpha:.2f}  PSNR {ps:5.1f} dB  clean-detect {tpr:.2f}  "
              f"survives-blur {robust:.2f}  false-alarm {fpr:.2f}")
    with open(OUT / "embed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "psnr", "tpr", "fpr", "robust_blur"])
        for r in rows:
            w.writerow([r["alpha"], f"{r['psnr']:.2f}", f"{r['tpr']:.4f}",
                        f"{r['fpr']:.4f}", f"{r['robust']:.4f}"])
    # save a visual: clean, watermarked, and the amplified difference
    np.save(OUT / "_vis.npy",
            {"clean": clips[0], "wm_low": embed(clips[:1], 0.04)[0],
             "wm_high": embed(clips[:1], 0.30)[0]}, allow_pickle=True)


# ---------------------------------------------------------------------------
def _attack(clips, kind):
    x = torch.as_tensor(np.asarray(clips))
    if kind == "none":
        return clips
    if kind == "noise":
        return np.clip(clips + 0.05 * np.random.default_rng(0)
                       .standard_normal(clips.shape), 0, 1)
    if kind == "blur":                                         # compression proxy
        xx = x.reshape(-1, 1, E.H, E.W)
        k = torch.ones(1, 1, 3, 3) / 9.0
        xx = F.conv2d(xx, k, padding=1)
        return xx.reshape(clips.shape).numpy()
    if kind == "rescale":                                     # 16->8->16
        xx = x.reshape(-1, 1, E.H, E.W)
        xx = F.interpolate(xx, size=(8, 8), mode="bilinear", align_corners=False)
        xx = F.interpolate(xx, size=(E.H, E.W), mode="bilinear",
                           align_corners=False)
        return xx.reshape(clips.shape).numpy()
    if kind == "crop":                                        # drop 2px, resize back
        xx = x.reshape(-1, 1, E.H, E.W)[:, :, 2:-2, 2:-2]
        xx = F.interpolate(xx, size=(E.H, E.W), mode="bilinear",
                           align_corners=False)
        return xx.reshape(clips.shape).numpy()
    raise ValueError(kind)


def stage_attack(args):
    net = E.load_gen("base")
    ds = E.make_dataset(600, seed=5)
    caps = E.caption_tensor(ds, np.arange(240))
    clips = E.sample(net, caps, steps=25, scale=2.0,
                     generator=torch.Generator().manual_seed(0)).clamp(0, 1).numpy()
    real = E.render_batch(ds, np.arange(240, 480)).numpy()
    attacks = ["none", "noise", "blur", "rescale", "crop"]
    rows = []
    for alpha in [0.08, 0.30]:
        wm = embed(clips, alpha)
        for atk in attacks:
            wa = _attack(wm, atk)
            ra = _attack(real, atk)
            ws = np.array([detect(c) for c in wa])
            rs = np.array([detect(c) for c in ra])
            thr = 0.5 * (ws.mean() + rs.mean())
            tpr = float((ws > thr).mean())
            rows.append(dict(alpha=alpha, attack=atk, tpr=tpr,
                             margin=ws.mean() - rs.mean()))
            print(f"alpha {alpha:.2f}  {atk:8s}  survives {tpr:.2f}  "
                  f"margin {ws.mean() - rs.mean():.3f}")
    with open(OUT / "attack.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "attack", "tpr", "margin"])
        for r in rows:
            w.writerow([r["alpha"], r["attack"], f"{r['tpr']:.4f}",
                        f"{r['margin']:.4f}"])


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (OUT / "embed.csv").exists():
        rows = list(csv.DictReader(open(OUT / "embed.csv")))
        alpha = [float(r["alpha"]) for r in rows]
        ps = [float(r["psnr"]) for r in rows]
        robust = [float(r["robust_blur"]) for r in rows]
        fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
        ax2 = ax1.twinx()
        l1, = ax1.plot(alpha, ps, "o-", color="#2b6fc9")
        l2, = ax2.plot(alpha, robust, "s-", color="#c98a2b")
        ax1.set_xlabel("watermark strength alpha")
        ax1.set_ylabel("PSNR dB — higher = more invisible", color="#2b6fc9")
        ax2.set_ylabel("survives compression (blur)", color="#c98a2b")
        ax2.set_ylim(0, 1.05)
        ax1.legend([l1, l2], ["invisibility (PSNR)", "robustness (survives blur)"],
                   loc="center right", fontsize=9)
        ax1.set_title("The core tension pulls opposite ways: a louder mark\n"
                      "survives compression but is easier to see")
        fig.tight_layout()
        fig.savefig(OUT / "tradeoff.png", dpi=110)
        plt.close(fig)

    if (OUT / "attack.csv").exists():
        rows = list(csv.DictReader(open(OUT / "attack.csv")))
        attacks = ["none", "noise", "blur", "rescale", "crop"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = np.arange(len(attacks))
        for j, alpha in enumerate(["0.08", "0.3"]):
            sub = {r["attack"]: float(r["tpr"]) for r in rows
                   if abs(float(r["alpha"]) - float(alpha)) < 1e-6}
            vals = [sub.get(a, 0) for a in attacks]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38,
                   label=f"alpha = {alpha}",
                   color=["#8a8f98", "#c98a2b"][j])
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.set_ylabel("fraction of marks still detected")
        ax.set_ylim(0, 1.05)
        ax.set_title("Blur (a stand-in for compression) is the killer: it erases\n"
                     "the high-frequency key. A louder mark (amber) survives more.")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "attacks.png", dpi=110)
        plt.close(fig)

    if (OUT / "_vis.npy").exists():
        vis = np.load(OUT / "_vis.npy", allow_pickle=True).item()
        diff = np.clip(0.5 + 6 * (vis["wm_high"] - vis["clean"]), 0, 1)
        rows = [list(vis["clean"]), list(vis["wm_low"]), list(diff)]
        E.strip(rows, OUT / "visual.png", scale=6)
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["embed", "attack", "figures"])
    a = ap.parse_args()
    {"embed": stage_embed, "attack": stage_attack,
     "figures": stage_figures}[a.stage](a)
