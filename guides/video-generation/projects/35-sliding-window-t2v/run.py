"""Project 35 — 64 frames out of a model that only makes 16.

    python3 run.py --stage reference            # ~1 min  what real long clips score
    python3 run.py --stage generate --mode indep_cut
    python3 run.py --stage generate --mode shared_cut
    python3 run.py --stage generate --mode shared_pixel
    python3 run.py --stage generate --mode shared_latent
    python3 run.py --stage generate --mode anchored
    python3 run.py --stage figures              # ~1 min

Five ways of joining seven overlapping windows into one video.  Same model,
same prompts, same shot list, same random numbers — only the joining changes.
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
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from PIL import Image                                          # noqa: E402

import long_lib as LL                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

N_CLIPS = 24               # digits 0-9, some twice
SEED = 35
EXTRA_STEPS, BATCH, LR = 4000, 16, 4e-4
DROP_PROMPT = 0.1


def prompt_digits():
    return torch.tensor([(i % 10) for i in range(N_CLIPS)])


# --------------------------------------------------------------------------
# stage: base — give the shared model a longer run before leaning on it
# --------------------------------------------------------------------------

def base():
    """Continue training project 30's t5 model for another 4000 steps.

    Why this stage exists, and why it is honest rather than cheating: project
    30 trained FOUR arms inside one time budget, so each got 2800 steps.  Every
    project in this phase stacks several generations on top of each other, and
    stacking multiplies whatever the model gets wrong in one window.  A model
    that is 24% right per window is not a fair test of a joining method — the
    joins would be blamed for the model's own noise.

    Nothing about the model changes: same weights, same data, same objective,
    same 2M parameters.  Only the number of optimiser steps.  The result is
    saved here rather than overwriting project 30's checkpoint, so project 30's
    published numbers stay reproducible.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"
    cache = LL.L.load_latent_cache("latents", where=P25)
    data, digit, direction = (cache["latents"], cache["digit"],
                              cache["direction"])
    bank = LL.T.TextBank("t5")
    model, ck = LL.T.load_arm("t5")
    model.train()
    flow = LL.FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    styles = LL.T.STYLES
    log, t0 = [], time.time()
    for step in range(1, EXTRA_STEPS + 1):
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        s = torch.randint(0, len(styles), (BATCH,), generator=g)
        f = torch.randint(0, LL.T.N_FILLER, (BATCH,), generator=g)
        pidx = torch.tensor([
            LL.T.prompt_index(int(digit[i]), int(direction[i]),
                              styles[int(si)], int(fi))
            for i, si, fi in zip(idx, s, f)])
        text = bank.get(pidx)
        drop = torch.rand(BATCH, generator=g) < DROP_PROMPT
        if drop.any():
            null = bank.null(BATCH)
            for k in text:
                seq, mask = text[k][0].clone(), text[k][1].clone()
                seq[drop] = null[k][0][drop][:, :seq.shape[1]]
                mask[drop] = null[k][1][drop][:, :mask.shape[1]]
                text[k] = (seq, mask)
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE, text),
                          flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[base] {step:5d} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    model.eval()
    torch.save({"state": model.state_dict(), "arm": "t5",
                "steps_here": EXTRA_STEPS, "steps_project30": 2800,
                "elapsed": time.time() - t0}, CK / "base.pt")
    np.save(OUT / "log_base.npy", np.array(log))
    print(f"[base] done {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: reference
# --------------------------------------------------------------------------

def measure(clips, judge=None):
    ratio, at, away = LL.seam_ratio(clips)
    pos, drift = LL.identity_drift(clips)
    follow = LL.direction_follow(clips)
    row = dict(seam_ratio=float(ratio.mean()),
               step_at_join=float(at.mean()),
               step_elsewhere=float(away.mean()),
               path_jerk=float(LL.path_jerk(clips).mean()),
               ink_spread=float(LL.ink_spread(clips).mean()),
               direction_follow=float(follow.mean()),
               identity_drift_end=float(drift[:, -1].mean()),
               identity_drift_mid=float(drift[:, drift.shape[1] // 2].mean()))
    if judge is not None:
        _, votes = LL.digit_votes(clips, judge)
        first = votes[:, :1]
        row["digit_stable"] = float((votes == first).float().mean())
    return row, (pos, drift)


@torch.no_grad()
def reference():
    """Two floors, not one.

    `real` is a genuine 64-frame clip: the best any method could possibly do.
    `real_vae` is that same clip pushed through the frozen VAE and back.  No
    generator in this phase can beat `real_vae`, because every one of them
    produces latents that must be decoded by that same VAE — its blur is
    already in the price.  Comparing a generated clip against `real` alone
    would blame the joining method for the VAE's losses.
    """
    t0 = time.time()
    judge, judge_acc = LL.T.load_digit_judge()
    rng = np.random.default_rng(SEED)
    digits = prompt_digits()
    clips, digits, _ = LL.long_real(rng, N_CLIPS, digits=digits)
    print(f"[reference] judge accuracy on real single frames: {judge_acc:.3f}")
    vae, scale = LL.L.load_vae("3d")
    mean, _ = vae.encode(clips)
    rt = LL.decode_long(mean * scale)
    for name, cl in (("real", clips), ("real_vae", rt)):
        row, (pos, drift) = measure(cl, judge)
        row["mode"] = name
        _, votes = LL.digit_votes(cl, judge)
        row["digit_acc"] = float((votes == digits[:, None]).float().mean())
        print(f"[reference] {row}", flush=True)
        torch.save({"row": row, "clips": cl[:8], "pos": pos, "drift": drift,
                    "digits": digits}, CK / f"gen_{name}.pt")
    print(f"[reference] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: window — how well does ONE window hold together?
# --------------------------------------------------------------------------

@torch.no_grad()
def window():
    """The ceiling nothing in this project can rise above.

    Before blaming a joining method for losing the character, measure what a
    SINGLE ordinary 16-frame generation already loses between its own first
    and last frame.  If the model cannot keep one handwriting for 16 frames,
    no amount of clever stitching will keep it for 64.
    """
    model, bank, _ = LL.load_base()
    digits = prompt_digits()
    dirs = torch.zeros(len(digits), dtype=torch.long)
    text = LL.text_for(bank, digits, dirs)
    g = torch.Generator().manual_seed(SEED)
    z = LL.sample_window(model, text, bank.null(len(digits)),
                         torch.randn((len(digits),) + LL.T.LATENT_SHAPE,
                                     generator=g))
    pix = LL.decode_long(z)
    cr = LL.glyph_crops(pix)
    rng = np.random.default_rng(SEED)
    real, _, _ = LL.long_real(rng, len(digits), schedule=[0],
                              digits=digits)
    vae, scale = LL.L.load_vae("3d")
    mean, _ = vae.encode(real)
    rcr = LL.glyph_crops(LL.decode_long(mean * scale))
    row = dict(generated_16f=float(LL.glyph_distance(cr[:, 15],
                                                     cr[:, 0]).mean()),
               real_vae_16f=float(LL.glyph_distance(rcr[:, 15],
                                                    rcr[:, 0]).mean()))
    print(f"[window] glyph drift across ONE 16-frame clip: {row}", flush=True)
    with open(OUT / "one_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow({k: round(v, 4) for k, v in row.items()})
    torch.save(row, CK / "one_window.pt")


# --------------------------------------------------------------------------
# stage: generate
# --------------------------------------------------------------------------

def generate(mode):
    t0 = time.time()
    model, bank, ck = LL.load_base()
    judge, _ = LL.T.load_digit_judge()
    digits = prompt_digits()
    lat, pix = LL.generate_long(model, bank, digits, mode, seed=SEED)
    gen_s = time.time() - t0
    row, (pos, drift) = measure(pix, judge)
    row["mode"] = mode
    row["seconds"] = round(gen_s, 1)
    _, votes = LL.digit_votes(pix, judge)
    row["digit_acc"] = float((votes == digits[:, None]).float().mean())
    print(f"[{mode}] {row}", flush=True)
    torch.save({"row": row, "clips": pix[:8], "pos": pos, "drift": drift,
                "latents": lat[:8]}, CK / f"gen_{mode}.pt")


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def save_gif(clip, path, scale=2, ms=80):
    x = ((clip.clamp(-1, 1) + 1) / 2)[0, 0].numpy()
    frames = [Image.fromarray((f * 255).astype(np.uint8)).resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in x]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=ms, loop=0)


def figures():
    rows, data = [], {}
    names = ["real", "real_vae"] + LL.MODES
    for m in names:
        p = CK / f"gen_{m}.pt"
        if not p.exists():
            raise SystemExit(f"missing {p} — run `--stage generate --mode {m}`"
                             if m in LL.MODES else
                             f"missing {p} — run `--stage reference`")
        d = torch.load(p, weights_only=False)
        d["row"]["ink_spread"] = float(LL.ink_spread(d["clips"]).mean())
        data[m] = d
        rows.append(d["row"])
    order = ["mode", "seam_ratio", "step_at_join", "step_elsewhere",
             "path_jerk", "ink_spread", "direction_follow", "digit_acc",
             "digit_stable",
             "identity_drift_mid", "identity_drift_end", "seconds"]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- 1. the centroid path, the picture that explains everything -------
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, key in zip(axes, ["real", "indep_cut", "anchored"]):
        ps.style_axes(ax)
        p = LL.centroid_path(data[key]["clips"])
        for b in range(4):
            ax.plot(p[b, :, 1], p[b, :, 0], lw=1.4, color=ps.SERIES[b],
                    alpha=0.9)
        for j in LL.joins():
            for b in range(4):
                ax.plot(p[b, j * LL.PIX_PER_LAT, 1],
                        p[b, j * LL.PIX_PER_LAT, 0], "o", ms=3.2,
                        color=ps.INK_MUTED)
        ax.set_title(key, fontsize=10, color=ps.INK)
        ax.set_xlabel("x of the digit")
        ax.invert_yaxis()
    axes[0].set_ylabel("y of the digit")
    fig.suptitle("Where the digit went over 64 frames (dots = window joins)",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "paths.png", dpi=150)
    plt.close(fig)

    # ---- 2. seam ratio ---------------------------------------------------
    fig, ax = ps.new_axes(7.4, 3.8)
    vals = [next(r["seam_ratio"] for r in rows if r["mode"] == n)
            for n in names]
    cols = [ps.INK_MUTED, ps.BASELINE] + [ps.SERIES[i % len(ps.SERIES)]
                                          for i in range(len(LL.MODES))]
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(vals[0], color=ps.INK_MUTED, ls="--", lw=1.2)
    ax.text(len(names) - 0.4, vals[0] + 0.05, "real clips",
            color=ps.INK_MUTED, fontsize=9, ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9,
                color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "How much bigger is the frame jump at a join?",
              "", "jump at a join / jump elsewhere", OUT / "seams.png")
    plt.close(fig)

    # ---- 3. identity drift over time -------------------------------------
    fig, ax = ps.new_axes(7.4, 4.0)
    for i, n in enumerate(names):
        d = data[n]
        pos = np.array(d["pos"]) if not torch.is_tensor(d["pos"]) \
            else np.array(d["pos"])
        ref = n.startswith("real")
        col = (ps.INK_MUTED if n == "real" else ps.BASELINE) if ref else \
            ps.SERIES[(i - 2) % len(ps.SERIES)]
        ax.plot(pos, d["drift"].mean(0).numpy(), lw=1.7, color=col, label=n,
                ls="--" if ref else "-")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Does it stay the same handwriting? (0 = identical)",
              "pixel frame", "glyph distance from frame 0",
              OUT / "identity_drift.png")
    plt.close(fig)

    # ---- 4. contact sheets ------------------------------------------------
    fig, axes = plt.subplots(len(names), 1,
                             figsize=(10.5, 1.35 * len(names)))
    for ax, n in zip(axes, names):
        sheet = LL.contact_sheet(data[n]["clips"][0:1], every=4)
        ax.imshow(sheet, cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=9,
                      color=ps.INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Every 4th frame of the same 64-frame video, one row per method",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "filmstrips.png", dpi=150)
    plt.close(fig)

    for n in names:
        save_gif(data[n]["clips"][0:1], OUT / f"long_{n}.gif")
    print("[figures] wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["base", "reference", "window", "generate",
                             "figures"])
    ap.add_argument("--mode", default="anchored", choices=LL.MODES)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "generate":
        generate(args.mode)
    else:
        globals()[args.stage]()
