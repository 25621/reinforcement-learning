"""Project 32 — an audio-driven talking head, and how to prove it is in sync.

    python3 run.py --stage data                 # ~1 min  the world + the ruler
    python3 run.py --stage train --arm generic  # ~7 min  eight speakers
    python3 run.py --stage train --arm finetune # ~2 min  one specific speaker
    python3 run.py --stage figures              # ~2 min

Every number here comes from the same three questions real talking-head papers
ask: is the mouth moving *with* the sound, is it moving *because* of the sound,
and is it still the same person's face afterwards?
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
import face_lib as FA                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

ARMS = ["generic", "finetune"]
STEPS, BATCH, LR = 900, 8, 2e-3
FT_STEPS, FT_LR = 400, 5e-4
SHIFTS = list(range(-4, 5))


def datasets():
    for name in ("train", "seen_eval", "held_eval", "ft_train"):
        if not (CK / f"{name}.pt").exists():
            raise SystemExit("run `python3 run.py --stage data` first")
    return {n: torch.load(CK / f"{n}.pt", map_location="cpu",
                          weights_only=False)
            for n in ("train", "seen_eval", "held_eval", "ft_train")}


# --------------------------------------------------------------------------
# stage: data
# --------------------------------------------------------------------------

def data():
    t0 = time.time()
    seen = list(range(FA.N_TRAIN_SPEAKERS))
    sets = {
        "train": FA.make_clips(seen, 40, seed=1),
        "seen_eval": FA.make_clips(seen, 4, seed=2),
        "held_eval": FA.make_clips([FA.HELD_OUT], 40, seed=3),
        "ft_train": FA.make_clips([FA.HELD_OUT], 20, seed=4),
    }
    for k, v in sets.items():
        torch.save(v, CK / f"{k}.pt")
        print(k, tuple(v["frames"].shape), flush=True)

    # ---- how good is the ruler? -----------------------------------------
    lines = []
    for k in ("seen_eval", "held_eval"):
        d = sets[k]
        c = FA.corr(FA.aperture(d["frames"][:, 0]), d["open"])
        lines.append(f"aperture-vs-truth correlation on REAL {k} frames: "
                     f"{c:.3f}")
    d = sets["held_eval"]
    lines.append(f"held-out speaker's personal max_open: "
                 f"{FA.speaker(FA.HELD_OUT)['max_open']:.2f} "
                 f"(training speakers: 0.75-1.00)")
    lines.append(f"real aperture swing, held-out speaker: "
                 f"{float((FA.aperture(d['frames'][:, 0]).amax(-1) - FA.aperture(d['frames'][:, 0]).amin(-1)).mean()):.4f}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "ruler.txt").write_text(txt + "\n")
    fig_world(sets["train"])
    print(f"{time.time()-t0:.0f}s")


def fig_world(d):
    i = 0
    fig = plt.figure(figsize=(9.6, 5.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.75])
    ax = fig.add_subplot(gs[0])
    ps.style_axes(ax)
    ax.plot(np.arange(FA.N_SAMPLES) / FA.SR, d["wav"][i].numpy(),
            color=ps.SERIES[0], lw=0.5)
    ax.set_title("the sound", color=ps.INK, fontsize=10, loc="left")
    ax.set_xlim(0, FA.CLIP_SEC)
    ax2 = fig.add_subplot(gs[1])
    ps.style_axes(ax2)
    ax2.imshow(d["mel"][i].numpy().T, aspect="auto", origin="lower",
               cmap="magma", extent=[0, FA.CLIP_SEC, 0, FA.N_MELS])
    ax2.set_title("what the model reads: 32 log-mel bands per frame",
                  color=ps.INK, fontsize=10, loc="left")
    ax2.grid(False)
    ax3 = fig.add_subplot(gs[2])
    ps.style_axes(ax3)
    ax3.imshow(np.concatenate(list(d["frames"][i, 0].numpy()), axis=1),
               cmap="gray", vmin=0, vmax=1)
    ax3.set_xticks([]), ax3.set_yticks([])
    ax3.grid(False)
    ax3.set_title("the frames it must produce", color=ps.INK, fontsize=10,
                  loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "world.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "world.png")


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(arm):
    torch.manual_seed(0)
    ds = datasets()
    if arm == "generic":
        d, steps, lr = ds["train"], STEPS, LR
        model = FA.TalkingHead()
    else:
        d, steps, lr = ds["ft_train"], FT_STEPS, FT_LR
        model = FA.TalkingHead()
        model.load_state_dict(torch.load(CK / "generic.pt", map_location="cpu",
                                         weights_only=False)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    n = len(d["frames"])
    print(f"[{arm}] {n} clips, "
          f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=g)
        pred = model(d["portrait"][idx], d["mel"][idx])
        loss = FA.loss_fn(pred, d["frames"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 200 == 0:
            print(f"[{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm,
                "elapsed": time.time() - t0}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time()-t0:.0f}s", flush=True)


def load(arm):
    m = FA.TalkingHead()
    m.load_state_dict(torch.load(CK / f"{arm}.pt", map_location="cpu",
                                 weights_only=False)["state"])
    m.eval()
    return m


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, d, mel=None, chunk=8):
    mel = d["mel"] if mel is None else mel
    out = [model(d["portrait"][i:i + chunk], mel[i:i + chunk])
           for i in range(0, len(mel), chunk)]
    pred = torch.cat(out)
    return pred, FA.aperture(pred[:, 0])


@torch.no_grad()
def figures():
    ds = datasets()
    rows, curves, showcase = [], {}, {}
    for arm in ARMS:
        model = load(arm)
        for setname in ("seen_eval", "held_eval"):
            if arm == "finetune" and setname == "seen_eval":
                pass          # kept: fine-tuning may cost accuracy elsewhere
            d = ds[setname]
            pred, ap = evaluate(model, d)
            real_ap = FA.aperture(d["frames"][:, 0])
            swing = float((ap.amax(-1) - ap.amin(-1)).mean())
            real_swing = float((real_ap.amax(-1) - real_ap.amin(-1)).mean())
            rows.append(dict(
                arm=arm, eval_set=setname,
                sync=round(FA.corr(ap, d["open"]), 3),
                identity_psnr=round(FA.identity_psnr(pred, d["frames"]), 2),
                mouth_swing=round(swing, 4),
                real_mouth_swing=round(real_swing, 4),
                swing_ratio=round(swing / real_swing, 2),
                jitter=round(FA.temporal_jitter(pred), 4)))
            print(rows[-1], flush=True)
            curves[(arm, setname)] = [FA.shifted_corr(ap, d["open"], k)
                                      for k in SHIFTS]
            showcase[(arm, setname)] = pred

    # ---- the control: does the AUDIO drive the mouth? --------------------
    d = ds["held_eval"]
    model = load("finetune")
    # roll rather than shuffle, so EVERY clip is guaranteed a different
    # utterance — a random permutation would leave some clips matched
    perm = torch.roll(torch.arange(len(d["frames"])), 1)
    pred_mis, ap_mis = evaluate(model, d, mel=d["mel"][perm])
    ctrl = [
        dict(measure="sync with the audio it was GIVEN",
             value=round(FA.corr(ap_mis, d["open"][perm]), 3)),
        dict(measure="sync with the audio of the PORTRAIT's own clip",
             value=round(FA.corr(ap_mis, d["open"]), 3)),
        dict(measure="sync, matched audio (for reference)",
             value=round(FA.corr(evaluate(model, d)[1], d["open"]), 3)),
        dict(measure="ceiling: the ruler on real frames",
             value=round(FA.corr(FA.aperture(d["frames"][:, 0]), d["open"]),
                         3)),
    ]
    for r in ctrl:
        print(r, flush=True)
    with open(OUT / "mismatch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["measure", "value"])
        w.writeheader()
        w.writerows(ctrl)

    with open(OUT / "sync.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "shift_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shift_frames"] + [f"{a}/{s}" for (a, s) in curves])
        for i, k in enumerate(SHIFTS):
            w.writerow([k] + [round(v[i], 3) for v in curves.values()])

    fig_shift(curves)
    fig_tracks(ds, showcase)
    fig_strips(ds, showcase, pred_mis)
    print("wrote", OUT)


def fig_shift(curves):
    fig, ax = ps.new_axes(7.6, 4.2)
    for i, (key, v) in enumerate(curves.items()):
        ax.plot(SHIFTS, v, "-o", color=ps.SERIES[i], lw=1.7, ms=4,
                label=f"{key[0]} on {key[1]}")
    ax.axvline(0, color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.legend(frameon=False, fontsize=8)
    ps.finish(fig, ax,
              "Sync curve: slide the audio, watch the agreement collapse",
              "audio shifted by (frames)", "correlation with mouth opening",
              OUT / "shift_curve.png")


def fig_tracks(ds, showcase):
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.0), dpi=110, sharex=True)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, setname in zip(axes, ("seen_eval", "held_eval")):
        ps.style_axes(ax)
        d = ds[setname]
        i = 0
        real = FA.aperture(d["frames"][:, 0])[i]
        ax.plot(real.numpy(), color=ps.INK_SECONDARY, lw=2.2,
                label="real clip")
        for j, arm in enumerate(ARMS):
            ap = FA.aperture(showcase[(arm, setname)][:, 0])[i]
            ax.plot(ap.numpy(), color=ps.SERIES[j], lw=1.7, label=arm)
        ax.set_title(setname, color=ps.INK, fontsize=10, loc="left")
        ax.set_ylabel("mouth aperture", color=ps.INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("frame", color=ps.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "tracks.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "tracks.png")


def fig_strips(ds, showcase, pred_mis):
    d = ds["held_eval"]
    i = 0
    rows = [("real clip", d["frames"][i, 0]),
            ("generic model", showcase[("generic", "held_eval")][i, 0]),
            ("fine-tuned on this speaker",
             showcase[("finetune", "held_eval")][i, 0]),
            ("fine-tuned, MISMATCHED audio", pred_mis[i, 0])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.2, 1.25 * len(rows)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (name, clip) in zip(axes, rows):
        ax.imshow(np.concatenate(list(clip.numpy()[::2]), axis=1), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(name, color=ps.INK_SECONDARY, fontsize=8, rotation=0,
                      ha="right", va="center")
    fig.suptitle("Held-out speaker, every other frame", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "strips.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "strips.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "train", "figures"])
    ap.add_argument("--arm", default="generic", choices=ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "data":
        data()
    elif args.stage == "train":
        train(args.arm)
    else:
        figures()
