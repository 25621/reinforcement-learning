"""Project 36 — draw the important frames first, fill in the rest afterwards.

    python3 run.py --stage cache      # ~1 min   encode clips once
    python3 run.py --stage train      # ~6 min   the interpolation model
    python3 run.py --stage compare    # ~4 min   three fillers, five gap sizes
    python3 run.py --stage long       # ~2 min   hierarchical vs sliding window
    python3 run.py --stage figures    # ~1 min

The comparison is against ground truth: every keyframe pair is cut out of a
REAL clip, so the frames the filler invents can be checked against the frames
that were actually there.
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
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from PIL import Image                                          # noqa: E402

import interp_lib as IP                                        # noqa: E402
LL = IP.LL
T = IP.T
L = IP.L

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

STEPS, BATCH, LR = 2500, 16, 6e-4
DROP_PROMPT = 0.1
N_EVAL = 96
SPEEDS = [0.7, 1.4, 2.1, 2.8, 4.2]      # training speed range is 1.4 - 2.2
SEED = 36


# --------------------------------------------------------------------------
# stage: cache
# --------------------------------------------------------------------------

def cache():
    t0 = time.time()
    tr = IP.build_cache(1024, seed=SEED, train=True, name="latents")
    print(f"[cache] train {tuple(tr['latents'].shape)} "
          f"scale {tr['scale']:.3f}", flush=True)
    for sp in SPEEDS:
        ev = IP.build_cache(N_EVAL, seed=900 + int(sp * 10), train=False,
                            speed=(sp, sp), name=f"eval_{sp}")
        gap = IP.anchor_gap(ev["clips"])
        print(f"[cache] speed {sp}: keyframes {gap.mean():.1f} px apart",
              flush=True)
    print(f"[cache] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train():
    torch.manual_seed(0)
    data = IP.load_cache("latents")
    lat, digit, direction = data["latents"], data["digit"], data["direction"]
    bank = T.TextBank("t5")
    model = IP.InterpDiT()
    flow = IP.FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    print(f"[train] {L.count_params(model):,} params", flush=True)
    mid = torch.tensor(IP.MIDDLE)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(lat), (BATCH,), generator=g)
        x0 = lat[idx]
        pidx = torch.tensor([T.prompt_index(int(digit[i]), int(direction[i]),
                                            "short", 0) for i in idx])
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
        pred = model(flow.interpolate(x0, t, noise), t * flow.T_SCALE, text,
                     cond=IP.anchor_cond(x0))
        # Loss only on the frames the model is actually asked to invent.
        # Grading it on the keyframes too would reward copying an input
        # straight to the output, which is free and teaches nothing.
        loss = F.mse_loss(pred[:, :, mid], flow.target(x0, noise)[:, :, mid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[train] {step:5d} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(),
                "elapsed": time.time() - t0}, CK / "interp.pt")
    np.save(OUT / "log_interp.npy", np.array(log))
    print(f"[train] done {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: compare
# --------------------------------------------------------------------------

@torch.no_grad()
def compare():
    t0 = time.time()
    base, bank, _ = LL.load_base()
    interp, _ = IP.load_interp()
    judge, _ = T.load_digit_judge()
    rows, keep = [], {}
    for sp in SPEEDS:
        ev = IP.load_cache(f"eval_{sp}")
        x0, clips = ev["latents"], ev["clips"]
        digits, dirs = ev["digit"], ev["direction"]
        truth_vae = LL.decode_long(x0)          # the VAE floor, not the raw clip
        text = LL.text_for(bank, digits, dirs)
        null = bank.null(len(digits))
        gap = float(IP.anchor_gap(clips).mean())
        for arm in IP.ARMS:
            g = torch.Generator().manual_seed(7)
            if arm == "linear":
                z = IP.fill_linear(x0)
            elif arm == "inpaint":
                z = IP.fill_inpaint(base, x0, text, null, generator=g)
            else:
                z = IP.fill_trained(interp, x0, text, null, generator=g)
            pred = LL.decode_long(z)
            row = dict(speed=sp, gap_px=round(gap, 2), arm=arm,
                       fill_error=float(IP.fill_error(pred, truth_vae).mean()),
                       path_error=float(IP.path_error(pred, truth_vae).mean()),
                       middle_ink=float(
                           ((IP.middle_slice(pred) + 1) / 2).mean()),
                       truth_ink=float(
                           ((IP.middle_slice(truth_vae) + 1) / 2).mean()))
            _, votes = LL.digit_votes(pred, judge)
            row["digit_acc"] = float((votes == digits[:, None]).float().mean())
            rows.append(row)
            print(f"[compare] {row}", flush=True)
            if sp in (1.4, 4.2):
                keep[f"{arm}_{sp}"] = pred[:4]
        row = dict(speed=sp, gap_px=round(gap, 2), arm="vae_floor",
                   fill_error=float(IP.fill_error(truth_vae,
                                                  truth_vae).mean()),
                   path_error=0.0,
                   middle_ink=float(((IP.middle_slice(truth_vae) + 1) / 2)
                                    .mean()),
                   truth_ink=float(((IP.middle_slice(truth_vae) + 1) / 2)
                                   .mean()))
        _, votes = LL.digit_votes(truth_vae, judge)
        row["digit_acc"] = float((votes == digits[:, None]).float().mean())
        rows.append(row)
        if sp in (1.4, 4.2):
            keep[f"truth_{sp}"] = truth_vae[:4]
    torch.save({"rows": rows, "keep": keep}, CK / "compare.pt")
    with open(OUT / "gap_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[compare] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: long — the two ways of getting to 40 frames
# --------------------------------------------------------------------------

KF_AT = [0, 3, 6, 9]           # where the coarse clip's frames land
LONG_LAT = 10                  # 40 pixel frames
LONG_DIR = 0                   # "right", for every window of both methods


@torch.no_grad()
def long():
    t0 = time.time()
    base, bank, _ = LL.load_base()
    interp, _ = IP.load_interp()
    judge, _ = T.load_digit_judge()
    n = 24
    digits = torch.tensor([i % 10 for i in range(n)])
    dirs = torch.full((n,), LONG_DIR)
    text = LL.text_for(bank, digits, dirs)
    null = bank.null(n)
    g = torch.Generator().manual_seed(SEED)

    # --- coarse: ONE ordinary 16-frame generation -------------------------
    coarse = LL.sample_window(
        base, text, null,
        torch.randn((n,) + T.LATENT_SHAPE, generator=g))
    C, _, Hl, Wl = T.LATENT_SHAPE
    # --- hierarchical: treat its 4 frames as keyframes 3 apart, then fill --
    lat = torch.zeros(n, C, LONG_LAT, Hl, Wl)
    for i, a in enumerate(KF_AT):
        lat[:, :, a] = coarse[:, :, i]
    for i in range(len(KF_AT) - 1):
        a, b = KF_AT[i], KF_AT[i + 1]
        win = lat[:, :, a:b + 1].clone()
        filled = IP.fill_trained(interp, win, text, null,
                                 generator=torch.Generator().manual_seed(9 + i))
        lat[:, :, a:b + 1] = filled
    hier = LL.decode_long(lat)

    # --- sliding window over the same 40 frames ---------------------------
    schedule = [LONG_DIR] * 4          # 4 windows = 10 latent frames
    _, slid = LL.generate_long(base, bank, digits, "anchored",
                               schedule=schedule, seed=SEED)
    coarse_pix = LL.decode_long(coarse)

    rows, keep = [], {}
    for name, clips, sched in (("coarse_16f", coarse_pix, [LONG_DIR]),
                               ("hierarchical", hier, None),
                               ("sliding_window", slid, schedule)):
        p = LL.centroid_path(clips)
        travel = (p - p[:, :1]).norm(dim=-1).max(1).values
        near_wall = ((p < 15) | (p > 49)).any(-1).float().mean(1)
        pos, drift = LL.identity_drift(clips)
        row = dict(method=name, frames=clips.shape[2],
                   travel_px=float(travel.mean()),
                   near_wall_frac=float(near_wall.mean()),
                   path_jerk=float(LL.path_jerk(clips).mean()),
                   identity_drift_end=float(drift[:, -1].mean()))
        if sched is not None:
            row["seam_ratio"] = float(LL.seam_ratio(clips, sched)[0].mean())
            row["direction_follow"] = float(
                LL.direction_follow(clips, sched).mean())
        _, votes = LL.digit_votes(clips, judge)
        row["digit_acc"] = float((votes == digits[:, None]).float().mean())
        row["digit_stable"] = float((votes == votes[:, :1]).float().mean())
        rows.append(row)
        keep[name] = clips[:4]
        print(f"[long] {row}", flush=True)
    torch.save({"rows": rows, "keep": keep}, CK / "long.pt")
    with open(OUT / "long_forms.csv", "w", newline="") as f:
        fields = ["method", "frames", "travel_px", "near_wall_frac",
                  "path_jerk", "seam_ratio", "direction_follow", "digit_acc",
                  "digit_stable", "identity_drift_end"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[long] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def save_gif(clip, path, scale=2, ms=90):
    x = ((clip.clamp(-1, 1) + 1) / 2)[0, 0].numpy()
    fr = [Image.fromarray((f * 255).astype(np.uint8)).resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in x]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=ms, loop=0)


def figures():
    comp = torch.load(CK / "compare.pt", weights_only=False)
    lng = torch.load(CK / "long.pt", weights_only=False)
    rows = comp["rows"]

    # ---- 1. error vs how far apart the keyframes are ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, key, lab in zip(axes, ["fill_error", "path_error"],
                            ["pixel error of invented frames",
                             "how far the digit is from the truth (px)"]):
        ps.style_axes(ax)
        for i, arm in enumerate(IP.ARMS + ["vae_floor"]):
            sel = [r for r in rows if r["arm"] == arm]
            col = ps.INK_MUTED if arm == "vae_floor" else ps.SERIES[i]
            ax.plot([r["gap_px"] for r in sel], [r[key] for r in sel], "o-",
                    lw=1.7, ms=4, color=col, label=arm,
                    ls="--" if arm == "vae_floor" else "-")
        ax.set_xlabel("distance between the two keyframes (pixels)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Keyframes far apart are harder to bridge", fontsize=11,
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "gap_sweep.png", dpi=150)
    plt.close(fig)

    # ---- 2. filmstrips at an easy and an impossible gap -------------------
    order = [("truth", "real (VAE)"), ("linear", "linear"),
             ("inpaint", "inpaint"), ("trained", "trained")]
    fig, axes = plt.subplots(len(order), 2, figsize=(10.6, 1.15 * len(order)))
    for c, sp in enumerate([1.4, 4.2]):
        for r, (key, lab) in enumerate(order):
            ax = axes[r, c]
            sheet = LL.contact_sheet(comp["keep"][f"{key}_{sp}"][0:1], every=2)
            ax.imshow(sheet, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, rotation=0, ha="right", va="center",
                              fontsize=9, color=ps.INK)
            if r == 0:
                ax.set_title(f"keyframes {'close' if sp == 1.4 else 'far'} "
                             f"apart (speed {sp})", fontsize=10, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "fills.png", dpi=150)
    plt.close(fig)

    # ---- 3. the two roads to 40 frames -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ps.style_axes(axes[0])
    for i, (name, clips) in enumerate(lng["keep"].items()):
        if name == "coarse_16f":
            continue
        p = LL.centroid_path(clips)
        for b in range(3):
            axes[0].plot(np.arange(p.shape[1]), p[b, :, 1].numpy(), lw=1.5,
                         color=ps.SERIES[i], alpha=0.85,
                         label=name if b == 0 else None)
    axes[0].axhline(49, color=ps.INK_MUTED, ls=":", lw=1.2)
    axes[0].text(1, 50.5, "canvas edge", fontsize=8, color=ps.INK_MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlabel("pixel frame")
    axes[0].set_ylabel("x of the digit")
    ps.style_axes(axes[1])
    names = [r["method"] for r in lng["rows"]]
    vals = [r["travel_px"] for r in lng["rows"]]
    axes[1].bar(range(len(names)), vals,
                color=[ps.SERIES[i] for i in range(len(names))])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=16, ha="right", fontsize=9)
    axes[1].set_ylabel("furthest the digit travelled (px)")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=9,
                     color=ps.INK_SECONDARY)
    fig.suptitle("Chaining adds events; interpolation adds detail",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "long_forms.png", dpi=150)
    plt.close(fig)

    for name, clips in lng["keep"].items():
        save_gif(clips[0:1], OUT / f"long_{name}.gif")
    print("[figures] wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "train", "compare", "long", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    globals()[args.stage]()
