"""Project 34 — a video LoRA: one frozen model, several swappable styles.

    python3 run.py --stage data                        # ~1 min
    python3 run.py --stage clf                         # ~2 min  the style judge
    python3 run.py --stage train --style thick --arm r4    # ~3 min
    python3 run.py --stage train --style thick --arm r2
    python3 run.py --stage train --style thick --arm r8
    python3 run.py --stage train --style thick --arm full
    python3 run.py --stage train --style trail  --arm r4
    python3 run.py --stage figures                     # ~6 min

Only 50 clips per style, exactly as the project brief asks — the whole point
is that this is enough.
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
import fid_lib                                                 # noqa: E402
import text_lib as T                                           # noqa: E402
import lora_lib as LO                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

N_CLIPS = 50
STEPS, BATCH = 700, 8
LORA_LR, FULL_LR = 1e-3, 1e-4
SAMPLE_STEPS, CFG = 30, 3.0
ARMS = {"r2": 2, "r4": 4, "r8": 8, "full": None}
CLASSES = ["plain"] + LO.STYLES
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


# --------------------------------------------------------------------------
# stage: data
# --------------------------------------------------------------------------

def data():
    vae, scale = L.load_vae("3d")
    out = {}
    with torch.no_grad():
        for split, n, seed, train in (("train", N_CLIPS, 34, True),
                                      ("eval", 96, 77, False)):
            rng = np.random.default_rng(seed)
            clips, dig, dr = L.attr_batch(rng, n, train=train)
            for st in CLASSES:
                sx = LO.stylize(clips, st)
                lat = []
                for i in range(0, n, 16):
                    m, _ = vae.encode(sx[i:i + 16])
                    lat.append(m * scale)
                out[f"{split}_{st}"] = dict(clips=sx, latents=torch.cat(lat),
                                            digit=dig, direction=dr)
                print(split, st, tuple(out[f"{split}_{st}"]["latents"].shape),
                      flush=True)
        torch.save(out, CK / "data.pt")
        fig_styles(out)

        # Which styles can the frozen VAE still express?  A LoRA adapts the
        # GENERATOR; the tokenizer underneath it never changes.  If a style
        # disappears in the latent, no adapter can bring it back.
        rng = np.random.default_rng(3)
        clips, _, _ = L.attr_batch(rng, 4, train=False)
        rows, report = [], []
        for st in LO.CANDIDATES:
            sx = LO.stylize(clips, st)
            rec = vae.decoder(vae.encode(sx)[0]).clamp(-1, 1)
            plain = LO.stylize(clips, "plain")
            plain_rec = vae.decoder(vae.encode(plain)[0]).clamp(-1, 1)
            # "Is the change the style made still THERE, and still the SAME
            # change?"  Sizes alone would be misleading: a style can be
            # replaced by a different, equally large artefact.  So compare the
            # change maps by cosine similarity — 1.0 means the VAE kept
            # exactly the alteration that was asked for, 0 means whatever
            # survived has nothing to do with it.
            d_before = (sx - plain).flatten()
            d_after = (rec - plain_rec).flatten()
            same = float(F.cosine_similarity(d_before, d_after, dim=0))
            report.append(dict(style=st, note=LO.STYLE_HELP[st],
                               size_of_change=round(float(d_before.abs().mean()), 4),
                               size_after_vae=round(float(d_after.abs().mean()), 4),
                               survives_the_vae=round(same, 2)))
            print(report[-1], flush=True)
            rows.append((st, sx))
            rows.append((f"{st} through the VAE", rec))
    with open(OUT / "style_survival.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    fig_survival(rows)


def fig_survival(rows):
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, 0.92 * len(rows)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (name, clip) in zip(axes, rows):
        ax.imshow(L.strip(clip[1:2], n=8), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(name, color=ps.INK_SECONDARY, fontsize=7, rotation=0,
                      ha="right", va="center")
    fig.suptitle("A LoRA can only teach what the frozen VAE can still say",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "style_survival.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "style_survival.png")


def load_data():
    p = CK / "data.pt"
    if not p.exists():
        raise SystemExit("run `python3 run.py --stage data` first")
    return torch.load(p, map_location="cpu", weights_only=False)


def fig_styles(d):
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(8.6, 1.2 * len(CLASSES)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, st in zip(axes, CLASSES):
        ax.imshow(L.strip(d[f"train_{st}"]["clips"][0:1], n=8), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(f"{st}\n{LO.STYLE_HELP[st]}", color=ps.INK_SECONDARY,
                      fontsize=7.5, rotation=0, ha="right", va="center")
    fig.suptitle("The same clip in the three styles", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "styles.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "styles.png")


# --------------------------------------------------------------------------
# stage: clf — an independent judge of "is this the right style?"
# --------------------------------------------------------------------------

def clf():
    """Train the style judge on VAE ROUND-TRIPS, not on clean frames.

    The first version of this stage trained on the styled clips directly and
    scored only 65.6% when asked about reconstructions.  That is a
    train/test mismatch, and an avoidable one: a generated clip is *always* a
    VAE decode, never a clean render.  So the judge is trained on exactly the
    kind of picture it will be asked to grade.
    """
    torch.manual_seed(0)
    d = load_data()
    net = fid_lib.FeatureNet(n_classes=len(CLASSES))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(5)
    vae, scale = L.load_vae("3d")
    g = torch.Generator().manual_seed(0)

    n_pool = 192
    clips, _, _ = L.attr_batch(rng, n_pool, train=True)
    pool = {}
    with torch.no_grad():
        for st in CLASSES:
            s = LO.stylize(clips, st)
            out = [vae.decoder(vae.encode(s[i:i + 16])[0]).clamp(-1, 1)
                   for i in range(0, n_pool, 16)]
            pool[st] = torch.cat(out)
        print(f"[clf] built {n_pool} round-trips per class", flush=True)

    for step in range(1, 901):
        xs, ys = [], []
        for ci, st in enumerate(CLASSES):
            idx = torch.randint(0, n_pool, (8,), generator=g)
            f = torch.randint(0, pool[st].shape[2], (8,), generator=g)
            xs.append(torch.stack([pool[st][i, :, j] for i, j in zip(idx, f)]))
            ys.append(torch.full((8,), ci))
        loss = F.cross_entropy(net(torch.cat(xs)), torch.cat(ys))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"[clf] {step} loss {loss.item():.3f}", flush=True)
    net.eval()
    # graded on VAE round-trips, because that is what generated clips look like
    correct = total = 0
    with torch.no_grad():
        for ci, st in enumerate(CLASSES):
            ev = d[f"eval_{st}"]["clips"][:32]
            rec = vae.decoder(vae.encode(ev)[0]).clamp(-1, 1)
            for f in (0, 5, 10, 15):
                correct += int((net(rec[:, :, f]).argmax(1) == ci).sum())
                total += len(rec)
    acc = correct / total
    torch.save({"state": net.state_dict(), "acc": acc}, CK / "styleclf.pt")
    (OUT / "style_judge.txt").write_text(
        f"style judge accuracy on VAE-reconstructed real clips: {acc:.3f} "
        f"(chance {1/len(CLASSES):.3f})\n")
    print(f"[clf] accuracy on reconstructions {acc:.1%}")


def load_clf():
    ck = torch.load(CK / "styleclf.pt", map_location="cpu", weights_only=False)
    net = fid_lib.FeatureNet(n_classes=len(CLASSES))
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["acc"]


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def build(arm):
    if arm == "full":
        model, _ = T.load_arm("t5")
        model.requires_grad_(True)
        return model, list(model.parameters()), []
    model, names = LO.build_lora_model(r=ARMS[arm])
    return model, LO.lora_parameters(model), names


def train(style, arm):
    torch.manual_seed(0)
    d = load_data()[f"train_{style}"]
    lat, digit, direction = d["latents"], d["digit"], d["direction"]
    bank = T.TextBank("t5")
    model, params, names = build(arm)
    flow = FL.RectifiedFlow()
    lr = FULL_LR if arm == "full" else LORA_LR
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(1)
    n_tr = sum(p.numel() for p in params)
    print(f"[{style}/{arm}] {len(names)} adapted layers, {n_tr:,} trainable "
          f"({100*n_tr/L.count_params(model):.2f}% of the model)", flush=True)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(lat), (BATCH,), generator=g)
        x0 = lat[idx]
        s = T.STYLES[int(torch.randint(0, len(T.STYLES), (1,), generator=g))]
        pidx = torch.tensor([T.prompt_index(int(digit[i]), int(direction[i]),
                                            s, int(i) % T.N_FILLER)
                             for i in idx])
        text = bank.get(pidx)
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE, text),
                          flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 200 == 0:
            print(f"[{style}/{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    state = (model.state_dict() if arm == "full" else LO.lora_state(model))
    torch.save({"state": state, "arm": arm, "style": style,
                "trainable": n_tr, "elapsed": time.time() - t0},
               CK / f"{style}_{arm}.pt")
    np.save(OUT / f"log_{style}_{arm}.npy", np.array(log))
    kb = sum(v.numel() for v in state.values()) * 4 / 1024
    print(f"[{style}/{arm}] done in {time.time()-t0:.0f}s, "
          f"saved weights {kb:,.0f} KB", flush=True)


def load_trained(style, arm):
    ck = torch.load(CK / f"{style}_{arm}.pt", map_location="cpu",
                    weights_only=False)
    model, _, _ = build(arm)
    if arm == "full":
        model.load_state_dict(ck["state"])
    else:
        missing = model.load_state_dict(ck["state"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    model.eval()
    return model, ck


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

COMBOS = [(d, k) for d in range(10) for k in range(4)]


@torch.no_grad()
def sample(model, bank, seed=9):
    digits = torch.tensor([c[0] for c in COMBOS])
    dirs = torch.tensor([c[1] for c in COMBOS])
    pidx = torch.tensor([T.prompt_index(int(d), int(k), "short", i % T.N_FILLER)
                         for i, (d, k) in enumerate(zip(digits, dirs))])
    text, null = bank.get(pidx), bank.null(len(pidx))
    g = torch.Generator().manual_seed(seed)
    z = T.cfg_sample(model, text, null, (len(pidx),) + T.LATENT_SHAPE,
                     scale=CFG, steps=SAMPLE_STEPS, generator=g)
    return T.decode(z), digits, dirs


@torch.no_grad()
def style_score(clips, net, target):
    probs = 0
    for f in (0, 5, 10, 15):
        probs = probs + F.softmax(net(clips[:, :, f]), dim=1)
    return float((probs.argmax(1) == CLASSES.index(target)).float().mean())


@torch.no_grad()
def figures():
    judge, _ = T.load_digit_judge()
    snet, snet_acc = load_clf()
    bank = T.TextBank("t5")
    rows, showcase = [], {}

    def measure(name, model, target, extra=None):
        t0 = time.time()
        clips, dig, dr = sample(model, bank)
        d_acc, k_acc, _ = T.grade(clips, dig, dr, judge)
        row = dict(run=name, style_target=target,
                   style_score=round(style_score(clips, snet, target), 3),
                   plain_score=round(style_score(clips, snet, "plain"), 3),
                   digit_acc=round(d_acc, 3), direction_acc=round(k_acc, 3),
                   seconds=round(time.time() - t0, 1), **(extra or {}))
        rows.append(row)
        showcase[name] = clips
        print(row, flush=True)
        return row

    base, _ = T.load_arm("t5")
    measure("frozen base", base, "plain", dict(trainable=0, weights_kb=0))

    sizes = {}
    for arm in ARMS:
        model, ck = load_trained("thick", arm)
        kb = sum(v.numel() for v in ck["state"].values()) * 4 / 1024
        sizes[arm] = kb
        measure(f"thick / {arm}", model, "thick",
                dict(trainable=ck["trainable"], weights_kb=round(kb, 1)))
    model, ck = load_trained("trail", "r4")
    measure("trail / r4", model, "trail",
            dict(trainable=ck["trainable"],
                 weights_kb=round(sum(v.numel() for v in ck["state"].values())
                                  * 4 / 1024, 1)))

    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- the dial only LoRA gives you --------------------------------------
    model, _ = load_trained("thick", "r4")
    dial = []
    for s in SCALES:
        LO.set_scale(model, s)
        clips, dig, dr = sample(model, bank)
        d_acc, k_acc, _ = T.grade(clips, dig, dr, judge)
        dial.append(dict(lora_scale=s,
                         style_score=round(style_score(clips, snet, "thick"), 3),
                         plain_score=round(style_score(clips, snet, "plain"), 3),
                         digit_acc=round(d_acc, 3),
                         direction_acc=round(k_acc, 3)))
        showcase[f"thick r4, dial {s}"] = clips
        print(dial[-1], flush=True)
    LO.set_scale(model, 1.0)
    with open(OUT / "dial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dial[0]))
        w.writeheader()
        w.writerows(dial)

    fig_bars(rows, snet_acc)
    fig_dial(dial)
    fig_swap(showcase)
    print("wrote", OUT)


def fig_bars(rows, ceiling):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    names = [r["run"] for r in rows]
    x = np.arange(len(names))
    axes[0].bar(x, [r["style_score"] for r in rows],
                color=[ps.SERIES[i % len(ps.SERIES)] for i in range(len(rows))])
    axes[0].axhline(ceiling, color=ps.INK_MUTED, ls="--", lw=1.2)
    axes[0].text(0, ceiling + 0.02, "judge's ceiling on real clips",
                 fontsize=8, color=ps.INK_MUTED)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("did it learn the style?", color=ps.INK, fontsize=11,
                      loc="left")
    axes[1].bar(x - 0.2, [r["digit_acc"] for r in rows], 0.38,
                color=ps.SERIES[0], label="right digit")
    axes[1].bar(x + 0.2, [r["direction_acc"] for r in rows], 0.38,
                color=ps.SERIES[1], label="right direction")
    axes[1].axhline(0.25, color=ps.INK_MUTED, ls=":", lw=1.1)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("did it forget how to follow the prompt?", color=ps.INK,
                      fontsize=11, loc="left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "runs.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "runs.png")


def fig_dial(dial):
    fig, ax = ps.new_axes(7.4, 4.2)
    ax.plot([d["lora_scale"] for d in dial], [d["style_score"] for d in dial],
            "-o", color=ps.SERIES[0], lw=1.9, ms=5, label="looks like the thick style")
    ax.plot([d["lora_scale"] for d in dial], [d["direction_acc"] for d in dial],
            "-s", color=ps.SERIES[1], lw=1.9, ms=5,
            label="still moves the right way")
    ax.plot([d["lora_scale"] for d in dial], [d["digit_acc"] for d in dial],
            "-^", color=ps.SERIES[2], lw=1.9, ms=5,
            label="still draws the right digit")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "One trained adapter, turned up and down at will",
              "LoRA scale at inference time", "fraction correct",
              OUT / "dial.png")


def fig_swap(showcase):
    keys = ["frozen base", "thick / r4", "trail / r4",
            "thick r4, dial 0.0", "thick r4, dial 2.0"]
    pick = 12
    fig, axes = plt.subplots(len(keys), 1, figsize=(8.8, 1.15 * len(keys)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, k in zip(axes, keys):
        ax.imshow(L.strip(showcase[k][pick:pick + 1], n=8), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(k, color=ps.INK_SECONDARY, fontsize=8, rotation=0,
                      ha="right", va="center")
    fig.suptitle("Same frozen weights, same prompt, different sticky note",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "swap.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT / "swap.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "clf", "train", "figures"])
    ap.add_argument("--style", default="thick", choices=LO.STYLES)
    ap.add_argument("--arm", default="r4", choices=list(ARMS))
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "data":
        data()
    elif args.stage == "clf":
        clf()
    elif args.stage == "train":
        train(args.style, args.arm)
    else:
        figures()
