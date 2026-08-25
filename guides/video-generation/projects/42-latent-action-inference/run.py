"""Project 42 — recovering the buttons from footage that never recorded them.

    python3 run.py --stage data       # ~1 min
    python3 run.py --stage train      # ~8 min   (five arms)
    python3 run.py --stage align      # ~1 min
    python3 run.py --stage control    # ~1 min
    python3 run.py --stage figures    # ~1 min

Five arms.  Four are trained on RANDOM play, where the button genuinely cannot
be guessed from the screen; the fifth is the twist.

    k4, k8, k16   a latent action model with 4 / 8 / 16 invented codes
    oracle        the same decoder handed the TRUE button — the ceiling
    greedy        k8, but trained on a skilled coin-seeker's play instead of
                  random play.  It looks like it should be easier.  It is not,
                  and why not is the most important thing this project teaches.
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
import lam_lib as LAM                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.set_num_threads(12)

STEPS, BATCH, LR = 2500, 64, 3e-4
# arm -> (number of codes, which recording to train on)
ARMS = {"k4": (4, "rand"), "k8": (8, "rand"), "k16": (16, "rand"),
        "oracle": (W.N_ACT, "rand"), "greedy": (8, "greedy")}
N_EVAL = 4096


def pairs(ep, batch, rng):
    """(frame_t, frame_t+1, true action, was-it-blocked) — labels for SCORING
    only.  The k* arms never see the action during training."""
    n_ep, T = ep["agent"].shape
    e = rng.integers(0, n_ep, size=batch)
    t = rng.integers(0, T - 1, size=batch)
    f0 = W.frames_of(ep, e, t)
    f1 = W.frames_of(ep, e, t + 1)
    a = ep["act"][e, t]
    blk = ep["blocked"][e, t]
    return (torch.from_numpy(f0), torch.from_numpy(f1),
            torch.from_numpy(a), torch.from_numpy(blk.astype(np.int64)))


def stage_data():
    # Random play (eps=1.0): the button is independent of the screen, so the
    # only way to know what happened is to look at the change.  This is the
    # setting a latent action model is actually FOR.
    W.save_episodes(W.record(2500, seed=42, eps=1.0), "train_rand", where=CK)
    W.save_episodes(W.record(400, seed=4242, eps=1.0), "test_rand", where=CK)
    # Greedy play: a good player, whose next move is largely predictable from
    # where the coin is.  Same game, very different footage.
    W.save_episodes(W.record(2500, seed=52, eps=0.1), "train_greedy", where=CK)
    W.save_episodes(W.record(400, seed=5252, eps=0.1), "test_greedy", where=CK)
    print("recorded random-play and greedy-play, 2500 train / 400 test each")


def train_ep(arm):
    return W.load_episodes(
        "train_rand" if ARMS[arm][1] == "rand" else "train_greedy", where=CK)


def test_ep(arm):
    return W.load_episodes(
        "test_rand" if ARMS[arm][1] == "rand" else "test_greedy", where=CK)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def stage_train():
    log = []
    for arm, (K, _) in ARMS.items():
        ep = train_ep(arm)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        model = LAM.LatentActionModel(n_codes=K)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        t0 = time.time()
        for step in range(1, STEPS + 1):
            f0, f1, a, _ = pairs(ep, BATCH, rng)
            if arm == "oracle":
                # Skip encoder and codebook: look the true button up directly.
                rec = model.dec(f0, model.vq.codebook()[a])
                code_loss = torch.zeros(())
            else:
                rec, idx, code_loss, _ = model(f0, f1)
            loss = F.mse_loss(rec, f1) + code_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0:
                log.append([arm, step, f"{loss.item():.5f}"])
                print(f"  {arm:7s} {step:5d}/{STEPS} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)")
        torch.save(model.state_dict(), CK / f"{arm}.pt")
        print(f"  {arm}: {W.count_params(model)} params, {time.time()-t0:.0f}s")
    with open(OUT / "loss.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "step", "loss"])
        wr.writerows(log)


def load_arm(arm):
    m = LAM.LatentActionModel(n_codes=ARMS[arm][0])
    m.load_state_dict(torch.load(CK / f"{arm}.pt"))
    m.eval()
    return m


# --------------------------------------------------------------------------
# align: do the invented codes mean the real buttons?
# --------------------------------------------------------------------------

def stage_align():
    rows, mats = [], {}
    for arm, (K, _) in ARMS.items():
        ep = test_ep(arm)
        rng = np.random.default_rng(7)
        f0, f1, a, blk = pairs(ep, N_EVAL, rng)
        m = load_arm(arm)
        with torch.no_grad():
            if arm == "oracle":
                codes = a.numpy()
                rec = m.dec(f0, m.vq.codebook()[a])
            else:
                codes = m.infer_code(f0, f1).numpy()
                rec = m(f0, f1)[0]
        acts = a.numpy()
        free = blk.numpy() == 0
        mat = LAM.confusion(codes, acts, K)
        mat_free = LAM.confusion(codes[free], acts[free], K)
        mat_blk = LAM.confusion(codes[~free], acts[~free], K)
        mats[arm] = mat
        mats[arm + "_free"] = mat_free
        rows.append(dict(
            arm=arm, n_codes=K, trained_on=ARMS[arm][1],
            codes_used=int((mat.sum(1) > 0).sum()),
            purity_free=LAM.purity(mat_free), nmi_free=LAM.nmi(mat_free),
            purity_blocked=LAM.purity(mat_blk),
            recon_mse=float(F.mse_loss(rec, f1))))
        print(f"{arm:7s} ({ARMS[arm][1]:6s}) codes {int((mat.sum(1)>0).sum()):2d}/"
              f"{K:2d}  free-purity {LAM.purity(mat_free):.3f}  "
              f"free-NMI {LAM.nmi(mat_free):.3f}  "
              f"blocked-purity {LAM.purity(mat_blk):.3f}  "
              f"recon {float(F.mse_loss(rec, f1)):.5f}")
    np.savez(CK / "confusion.npz", **mats)
    with open(OUT / "align.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items()})


# --------------------------------------------------------------------------
# control: drive the world with an invented code
# --------------------------------------------------------------------------

def stage_control():
    mats = np.load(CK / "confusion.npz")
    rows, drive = [], {}
    for arm, (K, _) in ARMS.items():
        ep = test_ep(arm)
        rng = np.random.default_rng(11)
        f0, f1, a, blk = pairs(ep, 512, rng)
        m = load_arm(arm)
        meaning = LAM.code_to_action(mats[arm + "_free"])   # code -> direction
        table = np.zeros((K, W.N_ACT + 1))                  # +1 = "did not move"
        ok = n = 0
        for c in range(K):
            code = torch.full((512,), c, dtype=torch.long)
            with torch.no_grad():
                out = m.apply_code(f0, code).numpy()
            for b in range(512):
                sym, ag, _, _ = W.read_frame(f0[b].numpy())
                walls = W.walls_from_symbols(sym)
                r0, c0 = ag // W.GRID, ag % W.GRID
                want = W.DELTA[meaning[c]]
                if walls[r0 + want[0], c0 + want[1]] == 1:
                    continue        # the code's direction is walled off here
                _, ai, _, _ = W.read_frame(out[b])
                r1, c1 = ai // W.GRID, ai % W.GRID
                mv = (r1 - r0, c1 - c0)
                j = W.DELTA.index(mv) if mv in W.DELTA else W.N_ACT
                table[c, j] += 1
                n += 1
                ok += (j == meaning[c])
        rows.append(dict(arm=arm, n_codes=K, trained_on=ARMS[arm][1],
                         control_accuracy=ok / max(n, 1)))
        drive[arm] = table
        print(f"{arm:7s} control accuracy {ok/max(n,1):.3f}  "
              f"({n} unblocked attempts)")
    np.savez(CK / "drive.npz", **drive)
    with open(OUT / "control.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items()})


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def stage_figures():
    mats = np.load(CK / "confusion.npz")
    align = {r["arm"]: r for r in csv.DictReader(open(OUT / "align.csv"))}
    control = {r["arm"]: r for r in csv.DictReader(open(OUT / "control.csv"))}

    # 1. the headline contrast: k8 on random vs greedy footage
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, key, title in [
            (axes[0], "k8_free", "8 codes, trained on RANDOM play\n"
                                 "(the button is unguessable from the screen)"),
            (axes[1], "greedy_free", "8 codes, trained on GREEDY play\n"
                                     "(a good player, next move predictable)")]:
        m = mats[key]
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(W.N_ACT))
        ax.set_xticklabels(W.ACT_NAMES, fontsize=9)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels([f"code {i}" for i in range(m.shape[0])], fontsize=8)
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left")
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(W.N_ACT):
                if m[i, j] > 0.05:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                            color="w" if m[i, j] < 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'confusion.png'}")

    # 2. purity / control across arms
    order = ["k4", "k8", "k16", "oracle", "greedy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    x = np.arange(len(order))
    cols = [ps.SERIES[0] if a != "greedy" else ps.SERIES[2] for a in order]
    ax = axes[0]
    ps.style_axes(ax)
    ax.bar(x, [float(align[a]["purity_free"]) for a in order], 0.6, color=cols)
    ax.axhline(0.25, color=ps.BASELINE, ls="--", lw=1)
    ax.text(0.0, 0.27, "chance (1 of 4 buttons)", color=ps.INK_MUTED,
            fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylim(0, 1.05)
    ax.set_title("Do the invented codes mean the real buttons?", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_ylabel("purity (unblocked moves)", color=ps.INK_SECONDARY,
                  fontsize=9)
    ax = axes[1]
    ps.style_axes(ax)
    ax.bar(x, [float(control[a]["control_accuracy"]) for a in order], 0.6,
           color=cols)
    ax.axhline(0.25, color=ps.BASELINE, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylim(0, 1.05)
    ax.set_title("Press an invented code — does the player obey?", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_ylabel("control accuracy", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "alignment.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'alignment.png'}")

    # 3. the invented vocabulary, drawn (k8 on random play)
    ep = W.load_episodes("test_rand", where=CK)
    rng = np.random.default_rng(3)
    f0, f1, a, blk = pairs(ep, 4, rng)
    m = load_arm("k8")
    strip = [[f0[b].numpy() for b in range(4)] + [None] +
             [f1[b].numpy() for b in range(4)]]
    for c in range(8):
        code = torch.full((4,), c, dtype=torch.long)
        with torch.no_grad():
            out = m.apply_code(f0, code).numpy()
        strip.append([out[b] for b in range(4)] + [None] * 5)
    W.strip_image(strip, OUT / "vocabulary.png")
    print("vocabulary.png: row 1 = 4 start frames, gap, then their 4 true next "
          "frames; rows 2-9 = the same starts driven by code 0..7")

    # 4. drive the world with a code sequence
    meaning = LAM.code_to_action(mats["k8_free"])
    want = [3, 3, 1, 1, 2, 2, 0, 0]
    seq = [int(np.nonzero(meaning == d)[0][0]) if (meaning == d).any() else 0
           for d in want]
    f = f0[:1]
    frames = [f[0].numpy()]
    for c in seq:
        with torch.no_grad():
            f = m.apply_code(f, torch.tensor([c]))
        frames.append(f[0].numpy())
    W.write_gif(frames, OUT / "driven.gif", fps=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["data", "train", "align", "control", "figures"])
    a = p.parse_args()
    {"data": stage_data, "train": stage_train, "align": stage_align,
     "control": stage_control, "figures": stage_figures}[a.stage]()


if __name__ == "__main__":
    main()
