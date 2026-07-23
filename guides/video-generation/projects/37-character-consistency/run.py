"""Project 37 — the same character in every shot.

    python3 run.py --stage cache      # ~2 min  clips of 64 + 8 characters
    python3 run.py --stage ruler      # ~1 min  what the identity metric can read
    python3 run.py --stage ip         # ~7 min  train the IP-Adapter once
    python3 run.py --stage lora       # ~4 min  train one LoRA per character
    python3 run.py --stage evaluate   # ~3 min  a four-shot story per character
    python3 run.py --stage figures    # ~1 min

Every character used for evaluation comes from MNIST's TEST split, so no arm
has ever seen that handwriting during training.
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

import ident_lib as ID                                         # noqa: E402
LL, T, L, LR = ID.LL, ID.T, ID.L, ID.LR

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

IP_STEPS, BATCH, IP_LR = 2500, 16, 4e-4
LORA_STEPS, LORA_LR, LORA_R = 250, 1e-3, 4
DROP_REF = 0.1
REPEATS = 4                # 4 samples per (character, shot)
SEED = 37


# --------------------------------------------------------------------------
# stage: cache
# --------------------------------------------------------------------------

def cache():
    t0 = time.time()
    train_chars, test_chars = ID.character_sprites()
    tr = ID.build_cache(train_chars, train_split=True, seed=1, name="chars")
    te = ID.build_cache(test_chars, train_split=False, seed=2, name="test")
    print(f"[cache] train {tuple(tr['latents'].shape)} over "
          f"{len(train_chars)} characters; test {tuple(te['latents'].shape)} "
          f"over {len(test_chars)} unseen characters  "
          f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------
# stage: ruler — measure the metric before trusting it
# --------------------------------------------------------------------------

@torch.no_grad()
def ruler():
    """What does the glyph distance read on cases whose answer we know?

    Project 32 paid for this lesson: a score of 0.85 means nothing until you
    know that the ruler reads 0.87 on genuine data.  Here we need two numbers
    — the value for a perfect match and the value for a total miss — because
    every arm's score will be read against them.
    """
    te = ID.load_cache("test")
    clips, char = te["clips"], te["char"]
    lat = te["latents"]
    rt = LL.decode_long(lat)                      # VAE round-trip
    rows = []
    same, other, vae_rt = [], [], []
    for c in char.unique():
        mine = (char == c).nonzero().flatten()
        theirs = (char != c).nonzero().flatten()
        theirs = theirs[te["digit"][theirs] == te["digit"][mine[0]]]
        g = ID.glyph_of(clips[mine])
        same.append(LL.glyph_distance(g[1:], g[0:1]).mean())
        vae_rt.append(LL.glyph_distance(ID.glyph_of(rt[mine]),
                                        g).mean())
        if len(theirs):
            og = ID.glyph_of(clips[theirs[:8]])
            other.append(LL.glyph_distance(og, g[0:1]).mean())
    rows = [dict(case="same character, another clip",
                 distance=float(torch.stack(same).mean())),
            dict(case="same character through the VAE",
                 distance=float(torch.stack(vae_rt).mean())),
            dict(case="another person, same digit",
                 distance=float(torch.stack(other).mean()))]
    for r in rows:
        print(f"[ruler] {r['case']:<34} {r['distance']:.4f}", flush=True)
    with open(OUT / "ruler.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "distance"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    torch.save({"rows": rows}, CK / "ruler.pt")


# --------------------------------------------------------------------------
# stage: ip
# --------------------------------------------------------------------------

def ip():
    torch.manual_seed(0)
    data = ID.load_cache("chars")
    lat, digit, direction, char = (data["latents"], data["digit"],
                                   data["direction"], data["char"])
    clips = data["clips"]
    by_char = {int(c): (char == c).nonzero().flatten() for c in char.unique()}
    bank = T.TextBank("t5")
    model = ID.IPAdapterDiT()
    flow = ID.FL.RectifiedFlow()
    params = model.trainable()
    n_train = sum(p.numel() for p in params)
    n_base = sum(p.numel() for p in model.base.parameters())
    print(f"[ip] {n_train:,} trainable ({100*n_train/n_base:.1f}% of the "
          f"frozen model's {n_base:,})", flush=True)
    opt = torch.optim.AdamW(params, lr=IP_LR)
    g = torch.Generator().manual_seed(1)
    log, t0 = [], time.time()
    for step in range(1, IP_STEPS + 1):
        idx = torch.randint(0, len(lat), (BATCH,), generator=g)
        x0 = lat[idx]
        pidx = torch.tensor([T.prompt_index(int(digit[i]), int(direction[i]),
                                            "short", 0) for i in idx])
        text = bank.get(pidx)
        # reference = a frame from a DIFFERENT clip of the same character
        ref = torch.zeros(BATCH, 1, L.CANVAS, L.CANVAS)
        for b, i in enumerate(idx):
            pool = by_char[int(char[i])]
            pool = pool[pool != i]
            j = int(pool[torch.randint(0, len(pool), (1,), generator=g)])
            f = int(torch.randint(0, L.T_FRAMES, (1,), generator=g))
            ref[b, 0] = clips[j, 0, f]
        drop = torch.rand(BATCH, generator=g) < DROP_REF
        ref[drop] = -1.0                       # an empty frame = "no reference"
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        pred = model(flow.interpolate(x0, t, noise), t * flow.T_SCALE, text,
                     ref=ref)
        loss = F.mse_loss(pred, flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[ip] {step:5d} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"ref_enc": model.ref_enc.state_dict(),
                "img_attn": model.img_attn.state_dict(),
                "trainable": n_train, "base": n_base,
                "elapsed": time.time() - t0}, CK / "ip.pt")
    np.save(OUT / "log_ip.npy", np.array(log))
    print(f"[ip] done {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: lora — one small fine-tune per character
# --------------------------------------------------------------------------

def lora():
    te = ID.load_cache("test")
    lat, digit, direction, char = (te["latents"], te["digit"],
                                   te["direction"], te["char"])
    bank = T.TextBank("t5")
    flow = ID.FL.RectifiedFlow()
    store, t0 = {}, time.time()
    for c in char.unique().tolist():
        torch.manual_seed(100 + c)
        model, names = ID.build_lora_model(r=LORA_R)
        params = LR.lora_parameters(model)
        opt = torch.optim.AdamW(params, lr=LORA_LR)
        mine = (char == c).nonzero().flatten()
        g = torch.Generator().manual_seed(5 + c)
        for step in range(1, LORA_STEPS + 1):
            idx = mine[torch.randint(0, len(mine), (BATCH,), generator=g)]
            x0 = lat[idx]
            pidx = torch.tensor([T.prompt_index(int(digit[i]),
                                                int(direction[i]), "short", 0)
                                 for i in idx])
            text = bank.get(pidx)
            noise = torch.randn(x0.shape, generator=g)
            t = flow.sample_t(BATCH, generator=g)
            loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                    t * flow.T_SCALE, text),
                              flow.target(x0, noise))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        store[c] = LR.lora_state(model)
        n = sum(p.numel() for p in params)
        print(f"[lora] character {c}: {n:,} params, loss {loss.item():.4f}, "
              f"{time.time()-t0:.0f}s", flush=True)
    kb = sum(v.numel() for v in next(iter(store.values())).values()) * 4 / 1024
    torch.save({"store": store, "kb_each": kb, "steps": LORA_STEPS,
                "elapsed": time.time() - t0}, CK / "lora.pt")
    print(f"[lora] {kb:.0f} KB per character, {time.time()-t0:.0f}s",
          flush=True)


# --------------------------------------------------------------------------
# stage: dial — how loudly should the reference speak?
# --------------------------------------------------------------------------

IP_BEST = dict(scale=1.0, drop_ref_in_null=False)


@torch.no_grad()
def dial():
    """Sweep the image branch's strength and its treatment under guidance.

    Two knobs, neither of which needs retraining:

    * `scale` multiplies the image branch's output. 0 restores the base model
      exactly; larger values impose the reference harder.
    * `drop_ref_in_null` decides whether the unconditional branch of guidance
      also sees the reference. See `ident_lib.sample` for why this is not a
      detail: with the reference in both branches, guidance amplifies the
      prompt and leaves the character at strength 1.
    """
    te = ID.load_cache("test")
    clips, char, digit = te["clips"], te["char"], te["digit"]
    bank = T.TextBank("t5")
    judge, _ = T.load_digit_judge()
    ipm, _ = ID.load_ip()
    ids = char.unique().tolist()
    rows = []
    for drop in (False, True):
        for scale in (0.0, 1.0, 2.0, 3.0):
            ident, dacc = [], []
            for c in ids:
                mine = (char == c).nonzero().flatten()
                ref = clips[mine[0], :, 0:1].permute(1, 0, 2, 3)
                ref_glyph = ID.glyph_of(clips[mine[1:2]])
                digits = torch.full((len(ID.SHOTS),), int(digit[mine[0]]))
                dirs = torch.tensor(ID.SHOTS)
                text = LL.text_for(bank, digits, dirs)
                z = ID.sample(ipm, text, bank.null(len(digits)),
                              (len(digits),) + T.LATENT_SHAPE,
                              ref=ref.expand(len(digits), -1, -1, -1),
                              generator=torch.Generator().manual_seed(SEED + c),
                              drop_ref_in_null=drop, scale=scale)
                pix = LL.decode_long(z)
                ident.append(float(ID.identity_distance(pix,
                                                        ref_glyph).mean()))
                votes = F.softmax(judge(pix[:, :, 0]), 1).argmax(1)
                dacc.append(float((votes == digits).float().mean()))
            rows.append(dict(drop_ref_in_null=drop, scale=scale,
                             identity=float(np.mean(ident)),
                             digit_acc=float(np.mean(dacc))))
            print(f"[dial] {rows[-1]}", flush=True)
    with open(OUT / "dial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    torch.save({"rows": rows}, CK / "dial.pt")


# --------------------------------------------------------------------------
# stage: evaluate — a four-shot story per character
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate():
    t0 = time.time()
    te = ID.load_cache("test")
    clips, char, digit = te["clips"], te["char"], te["digit"]
    bank = T.TextBank("t5")
    judge, _ = T.load_digit_judge()
    ids = char.unique().tolist()
    # one reference frame per character, and the glyph it implies
    ref_img, ref_glyph, ref_digit = {}, {}, {}
    for c in ids:
        mine = (char == c).nonzero().flatten()
        ref_img[c] = clips[mine[0], :, 0:1].permute(1, 0, 2, 3)   # (1,1,H,W)
        ref_glyph[c] = ID.glyph_of(clips[mine[1:2]])              # other clip
        ref_digit[c] = int(digit[mine[0]])

    base, _, _ = LL.load_base()
    ipm, ipck = ID.load_ip()
    lck = torch.load(CK / "lora.pt", weights_only=False)

    rows, keep = [], {}
    for arm in ID.ARMS:
        for c in ids:
            digits = torch.full((len(ID.SHOTS) * REPEATS,), ref_digit[c])
            dirs = torch.tensor(ID.SHOTS * REPEATS)
            text = LL.text_for(bank, digits, dirs)
            null = bank.null(len(digits))
            shape = (len(digits),) + T.LATENT_SHAPE
            gen = torch.Generator().manual_seed(SEED + c)
            if arm == "text":
                z = ID.sample(base, text, null, shape, generator=gen)
            elif arm == "ip":
                ref = ref_img[c].expand(len(digits), -1, -1, -1)
                z = ID.sample(ipm, text, null, shape, ref=ref, generator=gen,
                              **IP_BEST)
            else:
                model, _ = ID.build_lora_model(r=LORA_R)
                sd = model.state_dict()
                sd.update(lck["store"][c])
                model.load_state_dict(sd)
                model.eval()
                z = ID.sample(model, text, null, shape, generator=gen)
            pix = LL.decode_long(z)
            d = ID.identity_distance(pix, ref_glyph[c])
            gl = ID.glyph_of(pix)
            shot_gl = gl.view(REPEATS, len(ID.SHOTS), 28, 28).mean(0)
            across = LL.glyph_distance(shot_gl[1:], shot_gl[0:1]).mean()
            votes = F.softmax(judge(pix[:, :, 0]), 1).argmax(1)
            dirs_ok = (L.predicted_direction(pix) == dirs).float().mean()
            rows.append(dict(arm=arm, character=c, digit=ref_digit[c],
                             identity=float(d.mean()),
                             shot_to_shot=float(across),
                             digit_acc=float((votes == digits).float().mean()),
                             direction_acc=float(dirs_ok)))
            if c == ids[0]:
                keep[arm] = pix[:len(ID.SHOTS)]
                keep[f"{arm}_ref"] = ref_img[c]
            print(f"[eval] {rows[-1]}", flush=True)
    # the floor: real clips of the character, seen through the VAE
    for c in ids:
        mine = (char == c).nonzero().flatten()[:len(ID.SHOTS)]
        rt = LL.decode_long(te["latents"][mine])
        votes = F.softmax(judge(rt[:, :, 0]), 1).argmax(1)
        rows.append(dict(arm="real_vae", character=c, digit=ref_digit[c],
                         identity=float(ID.identity_distance(
                             rt, ref_glyph[c]).mean()),
                         shot_to_shot=float(LL.glyph_distance(
                             ID.glyph_of(rt)[1:], ID.glyph_of(rt)[0:1]).mean()),
                         digit_acc=float((votes == ref_digit[c]).float()
                                         .mean()),
                         direction_acc=float("nan")))
        if c == ids[0]:
            keep["real_vae"] = rt[:len(ID.SHOTS)]
    torch.save({"rows": rows, "keep": keep}, CK / "eval.pt")
    with open(OUT / "identity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[eval] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def figures():
    ev = torch.load(CK / "eval.pt", weights_only=False)
    rl = torch.load(CK / "ruler.pt", weights_only=False)
    rows = ev["rows"]
    arms = ID.ARMS + ["real_vae"]
    lock = {r["case"]: r["distance"] for r in rl["rows"]}
    other = lock["another person, same digit"]

    summary = []
    for a in arms:
        sel = [r for r in rows if r["arm"] == a]
        summary.append(dict(
            arm=a,
            identity=float(np.mean([r["identity"] for r in sel])),
            identity_sd=float(np.std([r["identity"] for r in sel])),
            shot_to_shot=float(np.mean([r["shot_to_shot"] for r in sel])),
            digit_acc=float(np.mean([r["digit_acc"] for r in sel])),
            direction_acc=float(np.nanmean([r["direction_acc"]
                                            for r in sel]))))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(summary, flush=True)

    # ---- 1. identity, with both ends of the ruler drawn ------------------
    fig, ax = ps.new_axes(7.6, 4.0)
    vals = [s["identity"] for s in summary]
    errs = [s["identity_sd"] for s in summary]
    ax.bar(range(len(arms)), vals, yerr=errs, capsize=3,
           color=[ps.SERIES[0], ps.SERIES[1], ps.SERIES[2], ps.INK_MUTED])
    ax.axhline(other, color=ps.SERIES[2], ls=":", lw=1.4)
    ax.text(len(arms) - 0.45, other + 0.002, "a different person's digit",
            fontsize=8.5, color=ps.SERIES[2], ha="right")
    ax.axhline(lock["same character through the VAE"], color=ps.INK_MUTED,
               ls="--", lw=1.4)
    ax.text(len(arms) - 0.45, lock["same character through the VAE"] - 0.006,
            "best possible (VAE floor)", fontsize=8.5, color=ps.INK_MUTED,
            ha="right")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9,
                color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Distance from the reference character (lower = same)",
              "", "glyph distance", OUT / "identity.png")
    plt.close(fig)

    # ---- 2. a story per arm ----------------------------------------------
    order = ["real_vae"] + ID.ARMS
    fig, axes = plt.subplots(len(order), 1, figsize=(9.6, 1.3 * len(order)))
    for ax, a in zip(axes, order):
        cl = ev["keep"][a]
        sheet = np.concatenate(
            [LL.contact_sheet(cl[i:i + 1], every=5) for i in range(len(cl))],
            axis=1)
        ax.imshow(sheet, cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(a, rotation=0, ha="right", va="center", fontsize=9,
                      color=ps.INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("One character, four shots (right, down, left, up)",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "shots.png", dpi=150)
    plt.close(fig)

    # ---- 3. the reference image itself -----------------------------------
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(((ev["keep"]["text_ref"][0, 0] + 1) / 2).numpy(), cmap="gray",
              vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("the reference frame", fontsize=9, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "reference.png", dpi=150)
    plt.close(fig)

    # ---- 4. cost vs benefit ----------------------------------------------
    fig, ax = ps.new_axes(7.4, 3.8)
    labels = ["identity\n(lower better)", "shot-to-shot\n(lower better)",
              "digit acc", "direction acc"]
    xs = np.arange(len(labels))
    for i, s in enumerate(summary):
        v = [s["identity"] / other, s["shot_to_shot"] / other,
             s["digit_acc"], s["direction_acc"]]
        ax.bar(xs + (i - 1.5) * 0.2, v, 0.19, label=s["arm"],
               color=(ps.SERIES[i] if s["arm"] != "real_vae"
                      else ps.INK_MUTED))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=4)
    ps.finish(fig, ax,
              "Identity is not free: what each method costs elsewhere",
              "", "distances as a fraction of 'a different person'",
              OUT / "tradeoff.png")
    plt.close(fig)
    print("[figures] wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "ruler", "ip", "dial", "lora",
                             "evaluate", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    globals()[args.stage]()
