"""Discrete image tokens: train a VQ-VAE on faces and read the price list.

A VQ-VAE turns a picture into a small grid of whole numbers. This project
builds one, checks that the numbers really do carry the picture, and then
measures the three things nobody tells you up front:

  * what 1024 tokens per image buys you over 64,
  * how much of the codebook actually gets used, and
  * how the whole thing compares to JPEG at the same number of bytes.

Stages
    data      download 8,000 CelebA faces + their attribute captions   (~4 min once)
    train     the shared f=8 tokenizer used by projects 33-36          (~3 min)
    grid      f=2 / f=4 / f=8  ->  1024 / 256 / 64 tokens per image    (~4 min)
    collapse  plain VQ vs EMA vs EMA+restart, and codebook size        (~3 min)
    tokens    what one code actually means: montages and token maps    (~30 s)
    jpeg      the same byte budgets, spent on JPEG instead             (~20 s)

Run `--stage all` to do everything (~12 min after the download); every stage
writes JSON to outputs/ so the README's numbers are reproducible.
"""

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402
import vqvae as VQ  # noqa: E402

OUT = HERE / "outputs"
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def _grid_png(rows, path, labels=None, scale=2):
    """rows: list of (N,64,64,3) uint8 arrays, drawn one per row."""
    import matplotlib.pyplot as plt
    n = min(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.72 * scale / 2 * 2,
                                                    len(rows) * 0.78 * scale),
                             dpi=110)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(ps.SURFACE)
    for i, row in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(row[j])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(ps.BASELINE)
        if labels:
            axes[i, 0].set_ylabel(labels[i], color=ps.INK_SECONDARY, fontsize=9,
                                  rotation=0, ha="right", va="center", labelpad=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def stage_data():
    """Download the faces and look at them."""
    tr_i, tr_a, va_i, va_a = VQ.load_faces()
    print(f"train {tr_i.shape}  val {va_i.shape}")

    caps = VQ.all_captions(tr_a)
    uniq = {}
    for c in caps:
        uniq[c] = uniq.get(c, 0) + 1
    words = sorted({w for c in caps for w in c.split()})

    # a strip of faces with their captions underneath
    import textwrap
    import matplotlib.pyplot as plt
    n = 8
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 3.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for j in range(n):
        axes[j].imshow(tr_i[j])
        axes[j].set_xticks([]); axes[j].set_yticks([])
        for s in axes[j].spines.values():
            s.set_color(ps.BASELINE)
        axes[j].set_xlabel(textwrap.fill(caps[j], 20), fontsize=7,
                           color=ps.INK_SECONDARY, labelpad=6)
    fig.suptitle("CelebA faces at 64x64, captioned from their human attribute labels",
                 color=ps.INK, fontsize=11, x=0.02, y=1.0, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "faces.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'faces.png'}")

    _save("data.json", {
        "n_train": len(tr_i), "n_val": len(va_i), "resolution": VQ.IMG,
        "raw_bytes_per_image": VQ.IMG * VQ.IMG * 3,
        "n_distinct_captions": len(uniq),
        "vocab_words": len(words),
        "mean_caption_words": float(np.mean([len(c.split()) for c in caps])),
        "most_common": sorted(uniq.items(), key=lambda kv: -kv[1])[:8],
        "examples": caps[:12],
    })


# ---------------------------------------------------------------------------
def stage_train(steps=2500):
    """Train the f=8 tokenizer that projects 33-36 all import."""
    tr_i, _, va_i, _ = VQ.load_faces()
    cfg = dict(down=8, k=VQ.CODEBOOK, latent=VQ.LATENT_DIM, width=VQ.WIDTH,
               ema=True, restart=True)
    # seed BEFORE constructing: projects 33-36 cache the token rows this exact
    # tokenizer produces, so re-running must reproduce the same weights
    torch.manual_seed(0)
    model = VQ.VQVAE(**cfg)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"f=8 tokenizer: {model.n_tokens} tokens/image, {n_par/1e6:.2f}M params")
    hist = VQ.train_vqvae(model, tr_i, va_i, steps=steps)
    ev = VQ.evaluate(model, va_i)
    print(f"final: PSNR {ev['psnr']:.2f} dB, {ev['used']}/{model.k} codes used")
    VQ.save_tokenizer(model, cfg, extra={"eval": ev})
    _save("train_f8.json", {"config": cfg, "params": n_par,
                            "history": hist, "eval": ev})

    x = VQ.to_tensor(va_i[:10])
    with torch.no_grad():
        recon, _, _, _ = model(x)
    _grid_png([va_i[:10], VQ.to_uint8(recon)], OUT / "recon_f8.png",
              labels=["original", "64 tokens"])


# ---------------------------------------------------------------------------
def stage_grid(steps=800):
    """Same model, three compression factors: 1024 / 256 / 64 tokens."""
    tr_i, _, va_i, _ = VQ.load_faces()
    rows, results = [va_i[:8]], []
    labels = ["original"]
    for down in (2, 4, 8):
        cfg = dict(down=down, k=VQ.CODEBOOK, latent=VQ.LATENT_DIM, width=VQ.WIDTH,
                   ema=True, restart=True)
        torch.manual_seed(0)
        m = VQ.VQVAE(**cfg)
        print(f"\n--- f={down}: {m.grid}x{m.grid} = {m.n_tokens} tokens/image")
        t0 = time.time()
        VQ.train_vqvae(m, tr_i, va_i, steps=steps, log_every=350)
        secs = time.time() - t0
        ev = VQ.evaluate(m, va_i)
        ev.update(down=down, grid=m.grid, train_secs=secs,
                  ms_per_step=1000 * secs / steps,
                  params=sum(p.numel() for p in m.parameters()))
        ev["bytes_per_image"] = ev["tokens"] * ev["bits_per_token"] / 8
        ev["compression"] = VQ.IMG * VQ.IMG * 3 / ev["bytes_per_image"]
        results.append(ev)
        with torch.no_grad():
            recon, _, _, _ = m(VQ.to_tensor(va_i[:8]))
        rows.append(VQ.to_uint8(recon))
        labels.append(f"{m.n_tokens} tokens")
        # NB: a distinct name -- `vqvae_f8.pt` is the long-trained shared
        # tokenizer and must not be overwritten by this short matched run
        torch.save(m.state_dict(), HERE / "checkpoints" / f"grid_f{down}.pt")
    _grid_png(rows, OUT / "grid.png", labels=labels)
    _save("grid.json", {"steps": steps, "results": results})

    fig, ax = ps.new_axes(6.4, 4.0)
    toks = [r["tokens"] for r in results]
    ax.plot(toks, [r["psnr"] for r in results], "o-", color=ps.SERIES[0], lw=2)
    for r in results:
        ax.annotate(f"  {r['psnr']:.1f} dB\n  {r['compression']:.0f}x smaller",
                    (r["tokens"], r["psnr"]), fontsize=8.5, color=ps.INK_SECONDARY,
                    va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(toks); ax.set_xticklabels([str(t) for t in toks])
    gain = results[0]["psnr"] - results[-1]["psnr"]
    ps.finish(fig, ax, f"Sixteen times the tokens buys {gain:.1f} dB",
              "tokens per image", "reconstruction PSNR (dB)", OUT / "grid_curve.png")


# ---------------------------------------------------------------------------
def stage_collapse(steps=500):
    """Does the codebook actually get used? Three quantizers and three sizes."""
    tr_i, _, va_i, _ = VQ.load_faces()
    runs = []
    variants = [
        ("plain VQ", dict(ema=False, restart=False), VQ.CODEBOOK),
        ("EMA", dict(ema=True, restart=False), VQ.CODEBOOK),
        ("EMA + restart", dict(ema=True, restart=True), VQ.CODEBOOK),
    ]
    sizes = [(f"EMA + restart, K={k}", dict(ema=True, restart=True), k)
             for k in (64, 2048)]
    usage = {}
    for name, kw, k in variants + sizes:
        cfg = dict(down=8, k=k, latent=VQ.LATENT_DIM, width=VQ.WIDTH, **kw)
        torch.manual_seed(0)
        m = VQ.VQVAE(**cfg)
        print(f"\n--- {name}")
        h = VQ.train_vqvae(m, tr_i, va_i, steps=steps, log_every=300)
        ev = VQ.evaluate(m, va_i)
        ev.update(name=name, k=k, **kw)
        ev["frac_used"] = ev["used"] / k
        ev["effective_bits"] = float(np.log2(max(ev["perplexity"], 1e-9)))
        runs.append(ev)
        with torch.no_grad():
            counts = torch.zeros(k)
            for i in range(0, 512, 128):
                idx = m.encode_indices(VQ.to_tensor(va_i[i:i + 128]))
                counts += torch.bincount(idx.reshape(-1), minlength=k).float()
        usage[name] = sorted(counts.tolist(), reverse=True)
        print(f"    PSNR {ev['psnr']:.2f}  used {ev['used']}/{k}  ppl {ev['perplexity']:.1f}")
    _save("collapse.json", {"steps": steps, "runs": runs})

    fig, ax = ps.new_axes(7.0, 4.2)
    for i, (name, u) in enumerate(usage.items()):
        u = np.array(u) / max(sum(u), 1)
        ax.plot(np.arange(1, len(u) + 1), np.maximum(u, 1e-6),
                color=ps.SERIES[i % len(ps.SERIES)], lw=2, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "How often each codebook entry is used (sorted)",
              "codebook entry, most used first", "share of all tokens",
              OUT / "collapse.png")


# ---------------------------------------------------------------------------
def stage_tokens():
    """What does one code mean? Token maps and per-code image patches."""
    model = VQ.load_tokenizer()
    _, _, va_i, va_a = VQ.load_faces()
    x = VQ.to_tensor(va_i[:6])
    idx = model.encode_indices(x)              # (6, 8, 8)
    with torch.no_grad():
        recon, _, _, _ = model(x)

    # colour the token map by code id so repeated codes are visibly repeated
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    palette = rng.random((model.k, 3)) * 0.75 + 0.2
    maps = (palette[idx.numpy()] * 255).astype(np.uint8)
    maps = np.repeat(np.repeat(maps, 8, axis=1), 8, axis=2)
    _grid_png([va_i[:6], maps, VQ.to_uint8(recon)], OUT / "token_map.png",
              labels=["original", "the 64 codes", "decoded back"])

    # for the most-used codes, show the 8x8 image patches that chose them
    counts = torch.zeros(model.k)
    all_idx, patches = [], []
    for i in range(0, 1024, 128):
        chunk = va_i[i:i + 128]
        ii = model.encode_indices(VQ.to_tensor(chunk))
        counts += torch.bincount(ii.reshape(-1), minlength=model.k).float()
        all_idx.append(ii.numpy())
        p = chunk.reshape(len(chunk), 8, 8, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5)
        patches.append(p.reshape(-1, 8, 8, 3))
    all_idx = np.concatenate(all_idx).reshape(-1)
    patches = np.concatenate(patches)
    top = torch.topk(counts, 6).indices.tolist()
    rows, labels = [], []
    for c in top:
        where = np.where(all_idx == c)[0][:10]
        rows.append(patches[where])
        labels.append(f"code {c}")
    _grid_png(rows, OUT / "code_patches.png", labels=labels, scale=1.4)

    # position statistics: is a code tied to a place in the face?
    grid_of = np.stack([np.arange(64)] * (len(all_idx) // 64)).reshape(-1)
    pos_entropy = []
    for c in range(model.k):
        m = grid_of[all_idx == c]
        if len(m) < 20:
            continue
        p = np.bincount(m, minlength=64) / len(m)
        nz = p[p > 0]
        pos_entropy.append(float(-(nz * np.log2(nz)).sum()))
    _save("tokens.json", {
        "grid": model.grid, "tokens_per_image": model.n_tokens,
        "codebook": model.k,
        "top_codes": [{"code": int(c), "share": float(counts[c] / counts.sum())}
                      for c in top],
        "mean_position_entropy_bits": float(np.mean(pos_entropy)),
        "max_position_entropy_bits": 6.0,
        "n_codes_measured": len(pos_entropy),
    })


# ---------------------------------------------------------------------------
def stage_jpeg():
    """Spend the same number of bytes on JPEG and see who wins."""
    _, _, va_i, _ = VQ.load_faces()
    sub = va_i[:200]
    rows, out = [sub[:8]], []
    labels = ["original"]

    # the three trained VQ-VAEs
    for down in (2, 4, 8):
        p = HERE / "checkpoints" / f"grid_f{down}.pt"
        if not p.exists():
            print(f"  (skipping f={down}: run --stage grid first)")
            continue
        m = VQ.VQVAE(down=down, k=VQ.CODEBOOK, latent=VQ.LATENT_DIM, width=VQ.WIDTH)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(blob["state"] if "state" in blob else blob)
        m.eval()
        ev = VQ.evaluate(m, sub)
        nbytes = m.n_tokens * np.log2(VQ.CODEBOOK) / 8
        out.append({"codec": f"VQ-VAE f={down}", "tokens": m.n_tokens,
                    "bytes": float(nbytes), "psnr": float(ev["psnr"])})
        with torch.no_grad():
            r, _, _, _ = m(VQ.to_tensor(sub[:8]))
        rows.append(VQ.to_uint8(r))
        labels.append(f"VQ {m.n_tokens} tok\n{nbytes:.0f} B")

    # JPEG at a range of qualities
    for q in (1, 3, 5, 10, 20, 40, 75):
        tot_b, tot_p = 0, 0.0
        decoded = []
        for i, im in enumerate(sub):
            buf = BytesIO()
            Image.fromarray(im).save(buf, "JPEG", quality=q)
            tot_b += buf.tell()
            d = np.asarray(Image.open(BytesIO(buf.getvalue())).convert("RGB"))
            mse = ((d.astype(np.float32) - im.astype(np.float32)) / 127.5) ** 2
            tot_p += 10 * np.log10(4.0 / max(mse.mean(), 1e-12))
            if i < 8:
                decoded.append(d)
        out.append({"codec": f"JPEG q={q}", "tokens": None,
                    "bytes": tot_b / len(sub), "psnr": float(tot_p) / len(sub)})
        if q in (5, 20):
            rows.append(np.stack(decoded))
            labels.append(f"JPEG q={q}\n{tot_b/len(sub):.0f} B")

    for r in out:
        print(f"  {r['codec']:16s} {r['bytes']:7.0f} B  {r['psnr']:5.2f} dB")
    _save("jpeg.json", {"raw_bytes": VQ.IMG * VQ.IMG * 3, "results": out})
    _grid_png(rows, OUT / "jpeg.png", labels=labels)

    fig, ax = ps.new_axes(6.8, 4.2)
    vq = [r for r in out if r["codec"].startswith("VQ")]
    jp = [r for r in out if r["codec"].startswith("JPEG")]
    ax.plot([r["bytes"] for r in jp], [r["psnr"] for r in jp], "o-",
            color=ps.SERIES[1], lw=2, label="JPEG")
    ax.plot([r["bytes"] for r in vq], [r["psnr"] for r in vq], "s-",
            color=ps.SERIES[0], lw=2, label="VQ-VAE (this project)")
    for r in vq:
        ax.annotate(f" {r['tokens']} tok", (r["bytes"], r["psnr"]), fontsize=8.5,
                    color=ps.INK_SECONDARY)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Same bytes, different codec",
              "bytes per 64x64 image (log)", "PSNR (dB)", OUT / "jpeg_curve.png")


# ---------------------------------------------------------------------------
STAGES = {"data": stage_data, "train": stage_train, "grid": stage_grid,
          "collapse": stage_collapse, "tokens": stage_tokens, "jpeg": stage_jpeg}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (HERE / "checkpoints").mkdir(exist_ok=True)
    names = list(STAGES) if a.stage == "all" else [a.stage]
    for nm in names:
        print(f"\n=== {nm} " + "=" * (60 - len(nm)))
        fn = STAGES[nm]
        if a.steps and nm in ("train", "grid", "collapse"):
            fn(steps=a.steps)
        else:
            fn()
