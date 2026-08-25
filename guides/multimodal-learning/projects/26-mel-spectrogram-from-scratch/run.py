"""Mel spectrogram from scratch: build it, check it, then try to undo it.

Stages
    verify   our DFT / STFT / inverse-STFT against NumPy and PyTorch  (~20 s)
    picture  the 10-second clip as waveform, spectrogram and mel spectrogram
    bank     what the mel filterbank actually looks like
    window   the time-versus-frequency trade-off, three window lengths
    invert   Griffin-Lim from the spectrogram and from the mel spectrogram,
             with .wav files you can listen to                        (~10 s)

`python3 run.py --stage all` runs the lot in about one minute
(plus a one-off 1.1 MB audio download the first time).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))
import plot_style as ps  # noqa: E402
import spectro as S  # noqa: E402

OUT = HERE / "outputs"
plt = ps.plt


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


# ---------------------------------------------------------------------------
def stage_verify():
    """Is our hand-written transform the same one the libraries use?"""
    import torch

    rng = np.random.default_rng(0)
    x = rng.standard_normal(S.N_FFT)

    # 1. the DFT matrix against NumPy's FFT (same maths, different algorithm)
    mine = S.rfft_naive(x)
    theirs = np.fft.rfft(x)
    dft_err = float(np.abs(mine - theirs).max())

    # 2. our framed+windowed STFT against torch.stft
    clip = S.showcase_clip()[: 4 * S.SR]
    ours = S.stft_manual(clip)
    t = torch.stft(torch.from_numpy(clip.astype(np.float64)), S.N_FFT,
                   hop_length=S.HOP, win_length=S.N_FFT,
                   window=torch.from_numpy(S.hann(S.N_FFT)), center=False,
                   return_complex=True).numpy().T
    n = min(len(ours), len(t))
    stft_err = float(np.abs(ours[:n] - t[:n]).max() / np.abs(t[:n]).max())

    # 3. Parseval's theorem (Marc-Antoine Parseval, 1799): the energy of a
    #    frame is the same whether you count it in samples or in frequencies.
    w = S.hann(S.N_FFT)
    fr = S.frame(clip)[100] * w
    spec = np.fft.rfft(fr)
    both = np.concatenate([spec, np.conj(spec[-2:0:-1])])
    parseval = float(abs((np.abs(both) ** 2).sum() / S.N_FFT - (fr ** 2).sum())
                     / (fr ** 2).sum())

    # 4. overlap-add really is the inverse of our STFT
    rec = S.istft_manual(np.fft.rfft(S.frame(clip) * w, axis=1))
    m = min(len(rec), len(clip))
    edge = S.N_FFT      # the first/last window cannot be normalised fully
    istft_err = float(np.abs(rec[edge:m - edge] - clip[edge:m - edge]).max())

    # 5. the filterbank: every row is a unit-area triangle, and every FFT bin
    #    inside [fmin, fmax] is covered by at least one row
    fb = S.mel_filterbank()
    freqs = np.linspace(0, S.SR / 2, S.N_FFT // 2 + 1)
    inside = (freqs >= S.FMIN) & (freqs <= S.FMAX)
    # participation ratio = "how many FFT bins does this band effectively use"
    part = (fb.sum(1) ** 2) / np.maximum((fb ** 2).sum(1), 1e-12)

    res = {
        "naive_dft_vs_numpy_fft_max_abs": dft_err,
        "our_stft_vs_torch_stft_rel": stft_err,
        "parseval_rel_error": parseval,
        "istft_roundtrip_max_abs": istft_err,
        "filterbank_row_area_min": float(fb.sum(1).min()),
        "filterbank_row_area_max": float(fb.sum(1).max()),
        "fft_bins_uncovered_in_band": int((fb[:, inside].sum(0) == 0).sum()),
        "bins_per_mel_band_lowest": float(part[0]),
        "bins_per_mel_band_highest": float(part[-1]),
        "bands_thinner_than_2_bins": int((part < 2).sum()),
        "n_fft": S.N_FFT, "hop": S.HOP, "n_mels": S.N_MELS, "sr": S.SR,
    }
    _save("verify.json", res)
    for k, v in res.items():
        print(f"  {k:36s} {v}")

    # speed: the FFT is the same answer, computed in n log n instead of n^2
    big = S.frame(clip)[:200]
    t0 = time.time(); _ = big @ S.dft_matrix(S.N_FFT)[:, :S.N_FFT // 2 + 1]
    slow = time.time() - t0
    t0 = time.time(); _ = np.fft.rfft(big, axis=1)
    fast = time.time() - t0
    _save("speed.json", {"naive_dft_s": slow, "fft_s": fast,
                         "speedup": slow / max(fast, 1e-9), "frames": 200})
    print(f"  200 frames: naive DFT {slow*1000:.0f} ms, FFT {fast*1000:.1f} ms "
          f"({slow/max(fast,1e-9):.0f}x)")


# ---------------------------------------------------------------------------
def _spec_axes(ax, M, sr=S.SR, hop=S.HOP, ylab="frequency (kHz)",
               extent_y=None, cmap="magma", dyn=6.0):
    # show a fixed dynamic range below the loudest cell, otherwise the near-
    # silent noise floor fills the colour map and hides everything above it
    im = ax.imshow(M.T, origin="lower", aspect="auto", cmap=cmap,
                   vmin=M.max() - dyn, vmax=M.max(),
                   extent=[0, M.shape[0] * hop / sr, 0,
                           extent_y if extent_y else sr / 2000])
    ax.set_ylabel(ylab, color=ps.INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
    ax.grid(False)
    return im


def stage_picture():
    x = S.showcase_clip()
    P = S.stft_power(x)
    L = S.log_mel(x)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), dpi=110,
                                        sharex=True)
    fig.patch.set_facecolor(ps.SURFACE)
    t = np.arange(len(x)) / S.SR
    axes[0].plot(t, x, lw=0.3, color=ps.SERIES[0])
    axes[0].set_ylabel("amplitude", color=ps.INK_SECONDARY, fontsize=9)
    ps.style_axes(axes[0])
    axes[0].set_title("10 s of audio: 6.3 s steam-train whistle, then six spoken digits",
                      color=ps.INK, fontsize=12, loc="left", pad=10)
    _spec_axes(axes[1], np.log10(P + 1e-10))
    axes[1].set_title("log power spectrogram — 257 linear frequency bins",
                      color=ps.INK_SECONDARY, fontsize=10, loc="left", pad=6)
    _spec_axes(axes[2], L, extent_y=S.N_MELS, ylab="mel band", dyn=14.0)
    axes[2].set_title(f"log-mel spectrogram — {S.N_MELS} perceptual bands",
                      color=ps.INK_SECONDARY, fontsize=10, loc="left", pad=6)
    axes[2].set_xlabel("time (s)", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "pipeline.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/pipeline.png")
    _save("shapes.json", {
        "samples": int(len(x)), "seconds": len(x) / S.SR,
        "stft_frames": int(P.shape[0]), "stft_bins": int(P.shape[1]),
        "mel_frames": int(L.shape[0]), "mel_bands": int(L.shape[1]),
        "numbers_waveform": int(len(x)), "numbers_stft": int(P.size),
        "numbers_mel": int(L.size)})


def stage_bank():
    fb = S.mel_filterbank()
    freqs = np.linspace(0, S.SR / 2, S.N_FFT // 2 + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    for j in range(0, S.N_MELS, 4):
        axes[0].plot(freqs, fb[j] / fb[j].max(), lw=1.0,
                     color=ps.SERIES[(j // 4) % len(ps.SERIES)])
    axes[0].set_title("every 4th mel filter (peak-normalised)", color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[0].set_xlabel("frequency (Hz)", color=ps.INK_SECONDARY, fontsize=10)
    hz = np.linspace(0, 8000, 400)
    axes[1].plot(hz, S.hz_to_mel(hz), lw=2.0, color=ps.SERIES[0], label="mel scale")
    axes[1].plot(hz, hz / 8000 * S.hz_to_mel(8000), lw=1.4, ls="--",
                 color=ps.BASELINE, label="a linear scale")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("hearing compresses high frequencies", color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("frequency (Hz)", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("mel", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "filterbank.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/filterbank.png")

    part = (fb.sum(1) ** 2) / np.maximum((fb ** 2).sum(1), 1e-12)
    edges = S.mel_to_hz(np.linspace(S.hz_to_mel(S.FMIN), S.hz_to_mel(S.FMAX),
                                    S.N_MELS + 2))
    rows = [{"band": j, "centre_hz": float(edges[j + 1]),
             "width_hz": float(edges[j + 2] - edges[j]),
             "fft_bins_used": float(part[j])}
            for j in (0, 20, 40, 60, S.N_MELS - 1)]
    _save("bank.json", {"rows": rows,
                        "bins_per_hz": float(S.SR / 2 / (S.N_FFT // 2))})


def stage_window():
    x = S.showcase_clip()
    seg = x[int(6.2 * S.SR): int(8.2 * S.SR)]        # the first spoken digits
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=110,
                                        sharey=True)
    fig.patch.set_facecolor(ps.SURFACE)
    rows = []
    for ax, n_fft in zip(axes, (128, 512, 2048)):
        hop = n_fft // 4
        P = np.log10(S.stft_power(seg, n_fft, hop) + 1e-10)
        ax.imshow(P.T, origin="lower", aspect="auto", cmap="magma",
                  vmin=P.max() - 6, vmax=P.max(),
                  extent=[0, len(seg) / S.SR, 0, S.SR / 2000])
        ax.set_title(f"{n_fft} samples = {1000*n_fft/S.SR:.0f} ms window",
                     color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("time (s)", color=ps.INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        ax.grid(False)
        rows.append({"n_fft": n_fft, "window_ms": 1000 * n_fft / S.SR,
                     "time_resolution_ms": 1000 * hop / S.SR,
                     "freq_resolution_hz": S.SR / n_fft,
                     "frames": int(P.shape[0]), "bins": int(P.shape[1])})
    axes[0].set_ylabel("frequency (kHz)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "window.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/window.png")
    _save("window.json", {"rows": rows})


def stage_invert():
    """How much of the sound survives each step? Rebuild it and measure."""
    x = S.showcase_clip()
    seg = x[int(5.0 * S.SR): int(8.0 * S.SR)]        # 3 s: whistle tail + speech
    fb = S.mel_filterbank()

    t0 = time.time()
    mag = np.sqrt(S.stft_power(seg))
    from_stft = S.griffin_lim(mag)                    # phase lost, nothing else
    mel = S.log_mel(seg, fb=fb)
    from_mel = S.griffin_lim(np.sqrt(S.mel_to_power(mel, fb)))
    # a control: keep the phase, so only the mel step is missing
    spec = np.fft.rfft(S.frame(seg) * S.hann(S.N_FFT), axis=1)
    keep_phase = S.istft_manual(np.sqrt(S.mel_to_power(mel, fb))
                                * spec / np.maximum(np.abs(spec), 1e-10))
    took = time.time() - t0

    for name, y in [("original", seg), ("from_stft", from_stft),
                    ("from_mel", from_mel), ("from_mel_true_phase", keep_phase)]:
        S.write_wav(OUT / f"recon_{name}.wav", y / (np.abs(y).max() + 1e-9) * 0.9)
    res = {
        "seconds": len(seg) / S.SR, "griffin_lim_iters": 48, "took_s": took,
        "lsd_db": {
            "from_stft (magnitude only, phase guessed)":
                S.log_spectral_distance(seg, from_stft),
            "from_mel (80 bands + phase guessed)":
                S.log_spectral_distance(seg, from_mel),
            "from_mel_true_phase (80 bands, phase given back)":
                S.log_spectral_distance(seg, keep_phase),
        },
        "numbers_per_second": {
            "waveform": S.SR,
            "stft_magnitude": (S.N_FFT // 2 + 1) * S.SR / S.HOP,
            "mel": S.N_MELS * S.SR / S.HOP,
        },
    }
    _save("invert.json", res)
    print(json.dumps(res["lsd_db"], indent=1))

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), dpi=110,
                                        sharey=True)
    fig.patch.set_facecolor(ps.SURFACE)
    ref = np.log10(S.stft_power(seg) + 1e-10).max()
    titles = ["original", "from spectrogram\n(phase guessed)",
              "from mel\n(phase guessed)", "from mel\n(phase given back)"]
    for ax, y, ttl in zip(axes, [seg, from_stft, from_mel, keep_phase], titles):
        P = np.log10(S.stft_power(y) + 1e-10)
        ax.imshow(P.T, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(y) / S.SR, 0, S.SR / 2000], vmin=ref - 6, vmax=ref)
        ax.set_title(ttl, color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("time (s)", color=ps.INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        ax.grid(False)
    axes[0].set_ylabel("frequency (kHz)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "invert.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/invert.png")


STAGES = {"verify": stage_verify, "picture": stage_picture, "bank": stage_bank,
          "window": stage_window, "invert": stage_invert}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    a = p.parse_args()
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"\n=== {name} ===", flush=True)
        STAGES[name]()
