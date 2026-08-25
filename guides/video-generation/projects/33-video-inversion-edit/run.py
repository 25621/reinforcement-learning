"""Project 33 — invert a real clip, change one word, keep everything else.

    python3 run.py --stage checks     # ~3 min  when does the round trip work?
    python3 run.py --stage figures    # ~9 min  the edit, and what it costs

No training: this project only uses project 30's trained T5 arm.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import dit_lib as L                                            # noqa: E402
import text_lib as T                                           # noqa: E402
import invert_lib as I                                         # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"

BASE_ARM = "t5"
N_EVAL = 32
STEPS = 60
EDIT_CFG = 3.0
SHIFT = 5                 # the edit: digit d becomes digit (d + 5) % 10


def setup(n=N_EVAL):
    model, _ = T.load_arm(BASE_ARM)
    bank = T.TextBank(BASE_ARM)
    ev = L.load_latent_cache("latents_eval", where=P25)
    z0, pix = ev["latents"][:n], ev["clips"][:n]
    dig, dr = ev["digit"][:n], ev["direction"][:n]
    new = (dig + SHIFT) % 10

    def txt(d):
        idx = torch.tensor([T.prompt_index(int(a), int(b), "short", 0)
                            for a, b in zip(d, dr)])
        return bank.get(idx)
    return (model, z0, pix, dig, dr, new, txt(dig), txt(new),
            bank.null(n))


# --------------------------------------------------------------------------
# stage: checks — how faithful is the round trip?
# --------------------------------------------------------------------------

def checks():
    model, z0, pix, dig, dr, new, text, new_text, null = setup(16)
    rows = []
    for steps in (10, 20, 40, 60):
        t0 = time.time()
        n = I.invert(model, z0, text, null, steps=steps, scale=1.0)
        back = I.denoise(model, n, text, null, steps=steps, scale=1.0)
        rows.append(dict(knob="sampling steps", value=steps,
                         roundtrip_mse=round(float((back - z0).pow(2).mean()), 5),
                         latent_var=round(float(z0.var()), 3),
                         seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    for scale in (1.0, 2.0, 3.0, 5.0):
        t0 = time.time()
        n = I.invert(model, z0, text, null, steps=STEPS, scale=scale)
        back = I.denoise(model, n, text, null, steps=STEPS, scale=scale)
        rows.append(dict(knob="guidance during BOTH directions", value=scale,
                         roundtrip_mse=round(float((back - z0).pow(2).mean()), 5),
                         latent_var=round(float(z0.var()), 3),
                         seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    with open(OUT / "roundtrip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, knob, xlabel in zip(
            axes, ("sampling steps", "guidance during BOTH directions"),
            ("Euler steps each way", "classifier-free guidance scale")):
        ps.style_axes(ax)
        sel = [r for r in rows if r["knob"] == knob]
        ax.plot([r["value"] for r in sel], [r["roundtrip_mse"] for r in sel],
                "-o", color=ps.SERIES[0], lw=1.8, ms=5)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=ps.INK_SECONDARY)
        ax.set_ylabel("round-trip error (MSE in latent space)",
                      color=ps.INK_SECONDARY, fontsize=9)
    axes[0].set_title("more steps, better round trip", color=ps.INK,
                      fontsize=11, loc="left")
    axes[1].set_title("guidance destroys it", color=ps.INK, fontsize=11,
                      loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "roundtrip.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "roundtrip.png")


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def score(clips, pix, new, dig, dr, judge):
    return dict(new_digit=round(T.grade(clips, new, dr, judge)[0], 3),
                old_digit=round(T.grade(clips, dig, dr, judge)[0], 3),
                direction=round(T.grade(clips, new, dr, judge)[1], 3),
                path_px=round(I.path_error(clips, pix), 2),
                flicker=round(I.flicker(clips), 4))


@torch.no_grad()
def figures():
    judge, _ = T.load_digit_judge()
    model, z0, pix, dig, dr, new, text, new_text, null = setup()
    g = torch.Generator().manual_seed(3)
    rows, showcase = [], {"real clip": pix}

    def add(name, clips, secs):
        r = dict(method=name, **score(clips, pix, new, dig, dr, judge),
                 seconds=round(secs, 1))
        rows.append(r)
        showcase[name] = clips
        print(r, flush=True)

    # 1. the obvious thing that does not work: just generate again
    t0 = time.time()
    z = T.cfg_sample(model, new_text, null, z0.shape, scale=EDIT_CFG,
                     steps=STEPS, generator=g)
    add("no inversion (fresh noise)", T.decode(z), time.time() - t0)

    # 2. the control: invert and put the SAME prompt back
    t0 = time.time()
    n = I.invert(model, z0, text, null, steps=STEPS, scale=1.0)
    z = I.denoise(model, n, text, null, steps=STEPS, scale=EDIT_CFG)
    add("invert + same prompt (control)", T.decode(z), time.time() - t0)

    # 3. the edit, at several inversion depths
    depths = [0.4, 0.6, 0.8, 1.0]
    sweep = []
    for tm in depths:
        t0 = time.time()
        n = I.invert(model, z0, text, null, steps=STEPS, scale=1.0, t_max=tm)
        z = I.denoise(model, n, new_text, null, steps=STEPS, scale=EDIT_CFG,
                      t_max=tm)
        clips = T.decode(z)
        add(f"edit, inverted to t={tm}", clips, time.time() - t0)
        sweep.append(rows[-1])

    # 4. the same edit, one frame at a time
    t0 = time.time()
    z = I.per_frame_edit(model, z0, text, null, new_text, steps=STEPS,
                         scale=EDIT_CFG)
    add("edit, every frame on its own", T.decode(z), time.time() - t0)
    t0 = time.time()
    z = I.per_frame_edit(model, z0, text, null, text, steps=STEPS,
                         scale=EDIT_CFG)
    add("frame-by-frame, same prompt (control)", T.decode(z),
        time.time() - t0)

    rows.append(dict(method="the real clips themselves", new_digit="",
                     old_digit=round(T.grade(pix, dig, dr, judge)[0], 3),
                     direction=round(T.grade(pix, dig, dr, judge)[1], 3),
                     path_px=0.0, flicker=round(I.flicker(pix), 4),
                     seconds=""))
    with open(OUT / "edits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig_tradeoff(sweep, rows, depths)
    fig_strips(showcase)
    print("wrote", OUT)


def fig_tradeoff(sweep, rows, depths):
    fig, ax = ps.new_axes(7.6, 4.4)
    ax.plot(depths, [r["new_digit"] for r in sweep], "-o", color=ps.SERIES[0],
            lw=1.9, ms=5, label="edit landed (new digit)")
    ax.plot(depths, [r["old_digit"] for r in sweep], "-o", color=ps.SERIES[2],
            lw=1.9, ms=5, label="old digit still there")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of clips", color=ps.INK_SECONDARY)
    ax2 = ax.twinx()
    ax2.plot(depths, [r["path_px"] for r in sweep], "-s", color=ps.SERIES[1],
             lw=1.9, ms=5, label="motion drifted (px)")
    ax2.set_ylabel("centroid path error vs the original (px)",
                   color=ps.SERIES[1], fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    base = next(r for r in rows if r["method"].startswith("no inversion"))
    ax2.axhline(base["path_px"], color=ps.INK_MUTED, ls="--", lw=1.2)
    ax2.text(0.42, base["path_px"] * 0.96, "no inversion at all", fontsize=8,
             color=ps.INK_MUTED)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="center left")
    ps.finish(fig, ax, "How deep to invert: edit strength against faithfulness",
              "inverted up to t =", "fraction of clips",
              OUT / "depth_tradeoff.png")


def fig_strips(showcase):
    pick = 1
    keys = list(showcase)
    fig, axes = plt.subplots(len(keys), 1, figsize=(8.8, 1.05 * len(keys)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, k in zip(axes, keys):
        ax.imshow(L.strip(showcase[k][pick:pick + 1], n=8), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(k, color=ps.INK_SECONDARY, fontsize=7.5, rotation=0,
                      ha="right", va="center")
    fig.suptitle("Same clip, same requested edit, different machinery",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "strips.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "strips.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["checks", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "checks":
        checks()
    else:
        figures()
