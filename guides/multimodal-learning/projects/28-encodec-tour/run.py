"""EnCodec tour: turn sound into a short sequence of integers, and back.

EnCodec is a *neural* codec. An ordinary codec (MP3, Opus) is hand-designed
signal processing; EnCodec is an encoder/decoder pair trained on audio, with a
quantiser in the middle that snaps the encoder's output onto a fixed alphabet.
That alphabet is the point: once a second of sound is 150 to 1,200 integers, a
Transformer can model audio with exactly the machinery it uses for text.

Stages
    ladder    the same clip at 1.5 / 3 / 6 / 12 / 24 kbps: metrics + .wav files
    residual  what each successive codebook adds (this is the "R" in RVQ)
    usage     how much of the alphabet is really used, and how long the
              sequences get
    baseline  a non-learned codec at the same bitrate, for scale
    content   do the tokens carry meaning? classify spoken digits from codes
    plot      redraw the figures from the saved JSON

Audio: the 6.3-second steam-train whistle (real, wideband, 44.1 kHz) for the
quality work, and FSDD spoken digits for the classification work.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "06-mel-spectrogram-pipeline"))
sys.path.insert(0, str(PROJECTS / "26-mel-spectrogram-from-scratch"))
import audio_lib as A  # noqa: E402
import plot_style as ps  # noqa: E402
import spectro as S  # noqa: E402

OUT = HERE / "outputs"
MODEL = "facebook/encodec_24khz"
SR = 24000
BANDWIDTHS = [1.5, 3.0, 6.0, 12.0, 24.0]


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def load_codec():
    from transformers import EncodecModel
    torch.set_num_threads(12)
    m = EncodecModel.from_pretrained(MODEL).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def demo_clip(seconds=6.0):
    """The steam-train whistle at EnCodec's native 24 kHz.

    Native rate matters: hand a 24 kHz model 8 kHz audio and you are measuring
    the resampler as much as the codec.
    """
    S.showcase_clip()                              # ensures the download exists
    x, sr = S.read_wav(S.data_dir() / "steam.wav")
    x = S.resample(x, sr, SR)[: int(seconds * SR)]
    return (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float32)


@torch.no_grad()
def encode(model, x, bandwidth):
    """waveform -> integer codes of shape (n_codebooks, n_frames)."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None]
    enc = model.encode(t, bandwidth=bandwidth)
    return enc.audio_codes[0, 0], enc.audio_scales


@torch.no_grad()
def decode(model, codes, scales=None):
    out = model.decode(codes[None, None], scales if scales is not None else [None])
    return out[0][0, 0].numpy()


def save_clip(path, y, seconds=3.0):
    """Write a 3-second excerpt, not the whole 6 seconds.

    Every metric in this project is computed on the full clip; the .wav files
    exist only so you can listen, and half of them is enough to hear the
    difference at a fraction of the repository size.
    """
    S.write_wav(path, np.asarray(y)[: int(seconds * SR)], SR)


def si_snr(ref, est):
    """Scale-Invariant Signal-to-Noise Ratio, in dB.

    "Scale-invariant" means the reference is first rescaled to whatever volume
    best fits the estimate, so a codec is not punished for making the output
    slightly quieter -- only for changing its *shape*. Higher is better; 20 dB
    means the error is a hundredth of the signal's energy.
    """
    n = min(len(ref), len(est))
    r, e = ref[:n] - ref[:n].mean(), est[:n] - est[:n].mean()
    alpha = float(np.dot(e, r) / (np.dot(r, r) + 1e-12))
    tgt = alpha * r
    return float(10 * np.log10((tgt ** 2).sum() / (((e - tgt) ** 2).sum() + 1e-12)))


# ---------------------------------------------------------------------------
def stage_ladder():
    model = load_codec()
    x = demo_clip()
    save_clip(OUT / "orig_24k.wav", x)
    rows = []
    for bw in BANDWIDTHS:
        t0 = time.time()
        codes, scales = encode(model, x, bw)
        y = decode(model, codes, scales)
        took = time.time() - t0
        n_q, n_frames = codes.shape
        rows.append({
            "kbps": bw, "codebooks": int(n_q), "frames": int(n_frames),
            "frame_rate_hz": n_frames / (len(x) / SR),
            "tokens_per_second": n_q * n_frames / (len(x) / SR),
            "si_snr_db": si_snr(x, y),
            "lsd_db": S.log_spectral_distance(S.resample(x, SR, S.SR),
                                              S.resample(y, SR, S.SR)),
            "bytes_per_second": n_q * n_frames * 10 / 8 / (len(x) / SR),
            "compression_vs_pcm16": (SR * 2) / (n_q * n_frames * 10 / 8 / (len(x) / SR)),
            "seconds_to_run": took,
        })
        save_clip(OUT / f"encodec_{str(bw).replace('.', 'p')}kbps.wav", y)
        print(f"  {bw:5.1f} kbps  {n_q:2d} codebooks  SI-SNR {rows[-1]['si_snr_db']:5.2f} dB"
              f"  LSD {rows[-1]['lsd_db']:5.2f} dB", flush=True)
    _save("ladder.json", {"rows": rows, "clip_seconds": len(x) / SR,
                          "pcm16_kbps": SR * 16 / 1000})


def stage_residual():
    """Decode using only the first k codebooks of an 8-codebook encoding.

    Residual Vector Quantisation works like writing a number as a sum of coins:
    the first codebook picks the closest entry it can, then the *residual* --
    what is left over, the part it got wrong -- is quantised again by a second
    codebook, and so on. So codebook 1 is a coarse sketch and each later one is
    a correction. Dropping the tail is therefore graceful: you get a blurrier
    version, not a broken one. That is exactly why one trained codec covers a
    whole bitrate ladder.
    """
    model = load_codec()
    x = demo_clip()
    codes, scales = encode(model, x, 6.0)          # 8 codebooks
    rows = []
    for k in range(1, codes.shape[0] + 1):
        y = decode(model, codes[:k], scales)
        rows.append({"codebooks": k, "kbps": k * 75 * 10 / 1000,
                     "si_snr_db": si_snr(x, y),
                     "lsd_db": S.log_spectral_distance(S.resample(x, SR, S.SR),
                                                       S.resample(y, SR, S.SR))})
        if k == 1:      # k=2,4,8 are byte-identical to the 1.5/3/6 kbps files
            save_clip(OUT / f"residual_{k}cb.wav", y)
        print(f"  {k} codebooks: SI-SNR {rows[-1]['si_snr_db']:5.2f} dB", flush=True)
    _save("residual.json", {"rows": rows})


def stage_usage():
    """How much of the 1024-entry alphabet does each codebook actually use?"""
    model = load_codec()
    x = demo_clip()
    codes, _ = encode(model, x, 24.0)              # all 32 codebooks
    rows = []
    for k in range(codes.shape[0]):
        c = codes[k].numpy()
        counts = np.bincount(c, minlength=1024).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        ent = float(-(nz * np.log2(nz)).sum())
        rows.append({"codebook": k, "distinct_used": int((counts > 0).sum()),
                     "entropy_bits": ent, "perplexity": float(2 ** ent)})
    _save("usage.json", {
        "rows": rows, "codebook_size": 1024, "bits_per_code": 10,
        "frames_per_second": 75,
        "tokens_per_second": {str(bw): int(75 * int(bw / 0.75))
                              for bw in BANDWIDTHS},
        "text_tokens_per_second_speech": 3.0,
    })
    print("  codebook 0 entropy %.2f bits, codebook 31 entropy %.2f bits"
          % (rows[0]["entropy_bits"], rows[-1]["entropy_bits"]))


def mu_law(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def inv_mu_law(y, mu=255):
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def stage_baseline():
    """A hand-designed codec at the *same* bitrate, so the comparison is fair.

    mu-law companding is the classic telephone trick (it is what makes 8-bit
    phone audio listenable): squash the waveform with a logarithm before
    rounding, so quiet parts get fine steps and loud parts coarse ones, matching
    how hearing works. To hit EnCodec's 6 kbps we must also drop the sample rate
    hard -- 6000 bits/s / 4 bits = 1500 samples per second.
    """
    x = demo_clip()
    rows = []
    for kbps, bits in [(6.0, 4), (6.0, 8), (24.0, 8)]:
        sr_low = int(kbps * 1000 / bits)
        lo = S.resample(x, SR, sr_low)
        q = np.round((mu_law(lo) * 0.5 + 0.5) * (2 ** bits - 1))
        y = S.resample(inv_mu_law((q / (2 ** bits - 1) - 0.5) * 2), sr_low, SR)
        rows.append({"name": f"mu-law {bits}-bit @ {sr_low} Hz", "kbps": kbps,
                     "si_snr_db": si_snr(x, y),
                     "lsd_db": S.log_spectral_distance(S.resample(x, SR, S.SR),
                                                       S.resample(y, SR, S.SR))})
        save_clip(OUT / f"mulaw_{int(kbps)}kbps_{bits}bit.wav", y)
        print(f"  {rows[-1]['name']}: SI-SNR {rows[-1]['si_snr_db']:.2f} dB")
    _save("baseline.json", {"rows": rows})


# ---------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    """Codes -> digit. One embedding table per codebook, summed, then a CNN.

    Averaging the per-codebook embeddings mirrors what RVQ means: the codebooks
    are additive corrections to one vector, so combining their embeddings is the
    natural way to read them back. We average rather than sum so that the input
    magnitude does not grow with the number of codebooks -- otherwise the arms
    would differ in two ways at once.
    """

    def __init__(self, n_q, n_codes=1024, dim=48, n_classes=10):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n_codes, dim) for _ in range(n_q)])
        self.net = nn.Sequential(
            nn.Conv1d(dim, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU())
        self.head = nn.Linear(64, n_classes)

    def forward(self, codes):                      # (B, n_q, T)
        h = sum(e(codes[:, i]) for i, e in enumerate(self.emb)) / len(self.emb)
        h = self.net(h.transpose(1, 2))
        return self.head(h.mean(-1))


class MelClassifier(nn.Module):
    def __init__(self, n_mels=40, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU())
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).mean(-1))


def _fit(model, xtr, ytr, xte, yte, steps=400, bs=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = rng.integers(0, len(xtr), bs)
        loss = F.cross_entropy(model(xtr[i]), ytr[i])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        acc = float((model(xte).argmax(1) == yte).float().mean())
    model.train()
    return acc


def stage_content():
    """Are the codes a *representation* or just a compressed file?

    If a small classifier can read spoken digits straight off the integers,
    without ever seeing a waveform, then the codec has produced features and
    not merely bytes -- which is the claim that makes audio language models
    possible.
    """
    n_clips = 1500
    model = load_codec()
    xs, digits, spk = A.load_fsdd(PROJECTS / "06-mel-spectrogram-pipeline" / "data",
                                  sr_out=SR)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(xs))[:n_clips]
    xs, digits, spk = xs[idx], digits[idx], spk[idx]

    cache = HERE / "data" / f"codes_{n_clips}.npy"
    if cache.exists():
        codes = np.load(cache)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        chunks, t0 = [], time.time()
        with torch.no_grad():
            for j in range(0, len(xs), 32):
                t = torch.from_numpy(xs[j:j + 32])[:, None]
                chunks.append(model.encode(t, bandwidth=24.0).audio_codes[0].numpy())
                if j % 320 == 0:
                    print(f"    encoded {j}/{len(xs)} ({time.time()-t0:.0f}s)", flush=True)
        codes = np.concatenate(chunks).astype(np.int64)
        np.save(cache, codes)
    print("  codes", codes.shape)

    tr = spk != A.SPEAKERS.index(A.HELD_OUT_SPEAKER)
    y = torch.from_numpy(digits)
    mel = np.stack([np.asarray(A.log_mel(x, sr=SR, n_fft=1024, hop=256,
                                          n_mels=40)).reshape(-1, 40) for x in xs])
    mel = torch.from_numpy(((mel - mel.mean()) / mel.std()).astype(np.float32))
    C = torch.from_numpy(codes)

    rows = []
    for n_q in (1, 2, 4, 8, 32):
        acc = _fit(CodeClassifier(n_q), C[tr, :n_q], y[tr], C[~tr, :n_q], y[~tr])
        rows.append({"features": f"EnCodec codes, {n_q} codebook(s)",
                     "n_q": n_q, "accuracy": acc})
        print(f"  codes n_q={n_q}: {acc:.3f}", flush=True)
    acc = _fit(MelClassifier(), mel[tr], y[tr], mel[~tr], y[~tr])
    rows.append({"features": "log-mel, 40 bands", "n_q": 0, "accuracy": acc})
    print(f"  mel: {acc:.3f}")
    _save("content.json", {"rows": rows, "n_clips": n_clips,
                           "held_out_speaker": A.HELD_OUT_SPEAKER,
                           "test_n": int((~tr).sum())})


# ---------------------------------------------------------------------------
def stage_plot():
    lad = json.loads((OUT / "ladder.json").read_text())["rows"]
    base = json.loads((OUT / "baseline.json").read_text())["rows"]
    res = json.loads((OUT / "residual.json").read_text())["rows"]
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    axes[0].plot([r["kbps"] for r in lad], [r["si_snr_db"] for r in lad], "o-",
                 color=ps.SERIES[0], lw=2, label="EnCodec (learned)")
    axes[0].plot([r["kbps"] for r in base], [r["si_snr_db"] for r in base], "s",
                 color=ps.SERIES[2], ms=7, label="mu-law (hand-designed)")
    axes[0].set_xscale("log")
    axes[0].set_xticks([1.5, 3, 6, 12, 24])
    axes[0].set_xticklabels(["1.5", "3", "6", "12", "24"])
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("quality against bitrate", color=ps.INK, fontsize=11,
                      loc="left", pad=10)
    axes[0].set_xlabel("kbit/s", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("SI-SNR (dB, higher is better)", color=ps.INK_SECONDARY,
                       fontsize=10)
    axes[1].plot([r["codebooks"] for r in res], [r["si_snr_db"] for r in res],
                 "o-", color=ps.SERIES[1], lw=2)
    axes[1].set_title("each extra codebook corrects the last one's error",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("codebooks kept (of 8)", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("SI-SNR (dB)", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "quality.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/quality.png")

    # spectrograms of the ladder
    files = [("orig_24k.wav", "original")] + [
        (f"encodec_{str(b).replace('.', 'p')}kbps.wav", f"{b} kbps") for b in (1.5, 6.0, 24.0)]
    fig, axes = ps.plt.subplots(1, len(files), figsize=(12.0, 3.0), dpi=110,
                                sharey=True)
    fig.patch.set_facecolor(ps.SURFACE)
    ref = None
    for ax, (f, ttl) in zip(axes, files):
        y, sr = S.read_wav(OUT / f)
        P = np.log10(S.stft_power(S.resample(y, sr, S.SR), 512, 128) + 1e-10)
        ref = P.max() if ref is None else ref
        ax.imshow(P.T, origin="lower", aspect="auto", cmap="magma",
                  vmin=ref - 6, vmax=ref,
                  extent=[0, len(y) / sr, 0, S.SR / 2000])
        ax.set_title(ttl, color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("time (s)", color=ps.INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        ax.grid(False)
    axes[0].set_ylabel("frequency (kHz)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/spectra.png")

    use = json.loads((OUT / "usage.json").read_text())["rows"]
    con = json.loads((OUT / "content.json").read_text())["rows"]
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.2, 3.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    axes[0].bar([r["codebook"] for r in use], [r["entropy_bits"] for r in use],
                color=ps.SERIES[0])
    axes[0].axhline(10, color=ps.SERIES[2], ls="--", lw=1.2)
    axes[0].text(1, 10.15, "10 bits = every entry equally likely", fontsize=8,
                 color=ps.INK_SECONDARY)
    axes[0].set_title("later codebooks look almost random", color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[0].set_xlabel("codebook index", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("entropy (bits)", color=ps.INK_SECONDARY, fontsize=10)
    names = [r["features"].replace("EnCodec codes, ", "").replace("(s)", "")
             for r in con]
    axes[1].barh(range(len(con)), [r["accuracy"] for r in con],
                 color=[ps.SERIES[1] if r["n_q"] else ps.SERIES[3] for r in con])
    axes[1].set_yticks(range(len(con)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].axvline(0.1, color=ps.BASELINE, ls="--", lw=1.0)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("spoken digit read straight off the tokens", color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("accuracy, unheard speaker", color=ps.INK_SECONDARY,
                       fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "tokens.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/tokens.png")


STAGES = {"ladder": stage_ladder, "residual": stage_residual,
          "usage": stage_usage, "baseline": stage_baseline,
          "content": stage_content, "plot": stage_plot}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    a = p.parse_args()
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"\n=== {name} ===", flush=True)
        STAGES[name]()
