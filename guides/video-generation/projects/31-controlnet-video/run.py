"""Project 31 — ControlNet for video: depth-style maps on every frame.

    python3 run.py --stage checks                  # ~1 min
    python3 run.py --stage train --arm temporal    # ~7 min
    python3 run.py --stage train --arm perframe    # ~6 min
    python3 run.py --stage train --arm scratch     # ~7 min
    python3 run.py --stage figures                 # ~8 min

The frozen base is project 30's T5 arm.  Only the control branch trains.
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
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402
import control_lib as C                                        # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"

BASE_ARM = "t5"
ARMS = ["temporal", "perframe", "scratch"]
STEPS, BATCH, LR = 1500, 16, 6e-4
DROP_CTRL = 0.1            # so the branch also learns "no control given"
SAMPLE_STEPS, CFG = 30, 3.0
N_EVAL = 48
JITTERS = [0.0, 0.1, 0.2, 0.3, 0.5]


def load_base():
    model, _ = T.load_arm(BASE_ARM)
    return model


def build(arm):
    return C.ControlledDiT(load_base(), mode=arm)


def prompt_text(bank, digits, dirs, style="short"):
    idx = torch.tensor([T.prompt_index(int(d), int(k), style, i % T.N_FILLER)
                        for i, (d, k) in enumerate(zip(digits, dirs))])
    return bank.get(idx), bank.null(len(idx))


# --------------------------------------------------------------------------
# stage: checks
# --------------------------------------------------------------------------

def checks():
    torch.manual_seed(0)
    base = load_base()
    bank = T.TextBank(BASE_ARM)
    ev = L.load_latent_cache("latents_eval", where=P25)
    clips, dig, dr = ev["clips"][:8], ev["digit"][:8], ev["direction"][:8]
    ctrl = C.depth_proxy(clips)
    text, _ = prompt_text(bank, dig, dr)
    x = torch.randn(8, *T.LATENT_SHAPE)
    t = torch.full((8,), 500.0)
    lines = []
    with torch.no_grad():
        ref = base(x, t, text)
        for arm in ARMS:
            m = C.ControlledDiT(load_base(), mode=arm)
            d = float((m(x, t, text, control=ctrl) - ref).abs().max())
            n_tr = sum(p.numel() for p in m.trainable())
            lines.append(f"{arm:<9} |output - frozen base| at init: {d:.2e}   "
                         f"trainable {n_tr:,} of {L.count_params(m):,} "
                         f"({100*n_tr/L.count_params(m):.1f}%)")
            assert d == 0.0, "zero-init connection is not identity!"

    # what does the control map actually carry?
    judge, _ = T.load_digit_judge()
    with torch.no_grad():
        real = ev["clips"][:128]
        up = C.upsampled_control(C.depth_proxy(real))
        acc_full = float((judge(real[:, :, 0]).argmax(1)
                          == ev["digit"][:128]).float().mean())
        acc_ctrl = float((judge(up[:, :, 0]).argmax(1)
                          == ev["digit"][:128]).float().mean())
    lines.append(f"digit judge on full-resolution frames: {acc_full:.1%}")
    lines.append(f"digit judge on the control map alone: {acc_ctrl:.1%} "
                 f"(chance 10%)")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "checks.txt").write_text(txt + "\n")


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(arm):
    torch.manual_seed(0)
    cache = L.load_latent_cache("latents", where=P25)
    lat, pix = cache["latents"], cache["clips"]
    digit, direction = cache["digit"], cache["direction"]
    bank = T.TextBank(BASE_ARM)
    model = build(arm)
    flow = FL.RectifiedFlow()
    params = model.trainable()
    opt = torch.optim.AdamW(params, lr=LR)
    g = torch.Generator().manual_seed(1)
    print(f"[{arm}] trainable {sum(p.numel() for p in params):,}", flush=True)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(lat), (BATCH,), generator=g)
        x0 = lat[idx]
        ctrl = C.depth_proxy(pix[idx])
        # a light dose of jitter during training, the way real control maps
        # arrive; both arms get exactly the same dose
        ctrl = C.jitter(ctrl, 0.05, generator=g)
        s = T.STYLES[int(torch.randint(0, len(T.STYLES), (1,), generator=g))]
        pidx = torch.tensor([
            T.prompt_index(int(digit[i]), int(direction[i]), s,
                           int(torch.randint(0, T.N_FILLER, (1,),
                                             generator=g)))
            for i in idx])
        text = bank.get(pidx)
        drop = torch.rand(BATCH, generator=g) < DROP_CTRL
        ctrl[drop] = 0.0
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE, text, control=ctrl),
                          flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 250 == 0:
            print(f"[{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("base.")}
    torch.save({"state": state, "arm": arm, "elapsed": time.time() - t0,
                "trainable": sum(p.numel() for p in params)}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time()-t0:.0f}s", flush=True)


def load_ctrl(arm):
    ck = torch.load(CK / f"{arm}.pt", map_location="cpu", weights_only=False)
    m = build(arm)
    m.load_state_dict(ck["state"], strict=False)
    m.eval()
    return m, ck


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def generate(model, text, null, control, seed=5):
    g = torch.Generator().manual_seed(seed)
    n = len(next(iter(text.values()))[0])
    z = T.cfg_sample(model, text, null, (n,) + T.LATENT_SHAPE, scale=CFG,
                     steps=SAMPLE_STEPS, generator=g, control=control)
    return T.decode(z)


@torch.no_grad()
def figures():
    judge, _ = T.load_digit_judge()
    ev = L.load_latent_cache("latents_eval", where=P25)
    pix, dig, dr = (ev["clips"][:N_EVAL], ev["digit"][:N_EVAL],
                    ev["direction"][:N_EVAL])
    bank = T.TextBank(BASE_ARM)
    text, null = prompt_text(bank, dig, dr)
    clean = C.depth_proxy(pix)

    rows = []
    # --- reference: the frozen base with no control at all ----------------
    base = load_base()
    t0 = time.time()
    gen = generate(base, text, null, None)
    rows.append(dict(arm="no control (frozen base)", jitter=0.0,
                     trainable=0,
                     track_px=round(C.tracking_error(gen, clean), 2),
                     flicker=round(C.flicker(gen), 4),
                     digit_acc=round(T.grade(gen, dig, dr, judge)[0], 3),
                     seconds=round(time.time() - t0, 1)))
    print(rows[-1], flush=True)
    rows.append(dict(arm="the real clips themselves", jitter=0.0, trainable="",
                     track_px=round(C.tracking_error(pix, clean), 2),
                     flicker=round(C.flicker(pix), 4), digit_acc="",
                     seconds=""))

    showcase = {"control": C.upsampled_control(clean), "real": pix,
                "no control": gen}
    sweep = {}
    for arm in ARMS:
        model, ck = load_ctrl(arm)
        vals = []
        for j in JITTERS:
            gj = torch.Generator().manual_seed(77)
            ctrl = C.jitter(clean, j, generator=gj)
            t0 = time.time()
            gen = generate(model, text, null, ctrl)
            row = dict(arm=arm, jitter=j, trainable=ck["trainable"],
                       track_px=round(C.tracking_error(gen, clean), 2),
                       flicker=round(C.flicker(gen), 4),
                       digit_acc=round(T.grade(gen, dig, dr, judge)[0], 3),
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            vals.append(row)
            print(row, flush=True)
            if j == 0.0:
                showcase[arm] = gen
            if j == 0.3:
                showcase[f"{arm} (jitter 0.3)"] = gen
        sweep[arm] = vals

    # --- does the prompt still decide WHAT? -------------------------------
    swap_rows = []
    model, _ = load_ctrl("temporal")
    wrong = (dig + 5) % 10
    text_w, null_w = prompt_text(bank, wrong, dr)
    gen_w = generate(model, text_w, null_w, clean)
    d_asked = T.grade(gen_w, wrong, dr, judge)[0]
    d_source = T.grade(gen_w, dig, dr, judge)[0]
    swap_rows.append(dict(measure="digit matches the PROMPT", value=round(d_asked, 3)))
    swap_rows.append(dict(measure="digit matches the CONTROL's source clip",
                          value=round(d_source, 3)))
    swap_rows.append(dict(measure="tracking error vs the control (px)",
                          value=round(C.tracking_error(gen_w, clean), 2)))
    swap_rows.append(dict(measure="chance", value=0.1))
    showcase["prompt swapped"] = gen_w
    with open(OUT / "swap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["measure", "value"])
        w.writeheader()
        w.writerows(swap_rows)
    for r in swap_rows:
        print(r, flush=True)

    with open(OUT / "control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_sweep(sweep, rows)
    fig_strips(showcase)
    fig_loss()
    print("wrote", OUT)


def fig_sweep(sweep, rows):
    base = next(r for r in rows if r["arm"].startswith("no control"))
    real = next(r for r in rows if r["arm"].startswith("the real"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    for i, arm in enumerate(ARMS):
        v = sweep[arm]
        axes[0].plot([r["jitter"] for r in v], [r["track_px"] for r in v],
                     "-o", color=ps.SERIES[i], lw=1.8, ms=4, label=arm)
        axes[1].plot([r["jitter"] for r in v], [r["flicker"] for r in v],
                     "-o", color=ps.SERIES[i], lw=1.8, ms=4, label=arm)
    axes[0].axhline(base["track_px"], color=ps.INK_MUTED, ls="--", lw=1.2)
    axes[0].text(0.0, base["track_px"] * 0.94, "no control", fontsize=8,
                 color=ps.INK_MUTED)
    axes[1].axhline(real["flicker"], color=ps.INK_MUTED, ls="--", lw=1.2)
    axes[1].text(0.0, real["flicker"] * 1.03, "real clips", fontsize=8,
                 color=ps.INK_MUTED)
    axes[0].set_title("Does the clip go where the control says?",
                      color=ps.INK, fontsize=11, loc="left")
    axes[1].set_title("Does the wobble get through to the video?",
                      color=ps.INK, fontsize=11, loc="left")
    axes[0].set_ylabel("tracking error (pixels)", color=ps.INK_SECONDARY)
    axes[1].set_ylabel("flicker", color=ps.INK_SECONDARY)
    for ax in axes:
        ax.set_xlabel("noise added to the control map", color=ps.INK_SECONDARY)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "jitter_sweep.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "jitter_sweep.png")


def fig_strips(showcase):
    keys = ["real", "control", "no control", "temporal", "perframe",
            "temporal (jitter 0.3)", "perframe (jitter 0.3)",
            "prompt swapped"]
    pick = 3
    fig, axes = plt.subplots(len(keys), 1, figsize=(8.6, 1.05 * len(keys)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, k in zip(axes, keys):
        ax.imshow(L.strip(showcase[k][pick:pick + 1], n=8), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(k, color=ps.INK_SECONDARY, fontsize=8, rotation=0,
                      ha="right", va="center")
    fig.suptitle("One control map, several ways of using it", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "strips.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "strips.png")


def fig_loss():
    fig, ax = ps.new_axes(7.4, 4.0)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"log_{arm}.npy")
        k = 10
        sm = np.convolve(a[:, 2], np.ones(k) / k, mode="valid")
        ax.plot(a[k - 1:, 0], sm, color=ps.SERIES[i], lw=1.5, label=arm)
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Control-branch training loss", "training step",
              "flow-matching MSE", OUT / "loss_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["checks", "train", "figures"])
    ap.add_argument("--arm", default="temporal", choices=ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "checks":
        checks()
    elif args.stage == "train":
        train(args.arm)
    else:
        figures()
