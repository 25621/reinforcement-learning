"""Turn spoken digits into mel spectrograms, then classify them with a small CNN.

Stages:
  data      download FSDD and cache the waveforms
  figures   the pipeline, drawn one stage at a time, plus the window trade-off
  train     four ablations: representation, log, number of mel bins, window size
  report    redraw the result charts from the saved CSV

    python3 run.py --stage all      # ~5 min cold on a CPU
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "01-modality-survey"))
import plot_style as ps  # noqa: E402

from audio_lib import (CLIP_LEN, HELD_OUT_SPEAKER, SR, MelCNN, WaveCNN,  # noqa: E402
                       load_fsdd, log_mel, mel_filterbank, n_params,
                       speaker_split, stft_power, train_classifier)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

N_FFT, HOP, N_MELS = 256, 64, 40
EPOCHS = 20


def get_data():
    cache = DATA / "fsdd.npz"
    if not cache.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        x, d, s = load_fsdd(DATA)
        np.savez_compressed(cache, x=x, digit=d, speaker=s)
        print(f"[data] cached {x.shape} clips")
    z = np.load(cache)
    return z["x"], z["digit"], z["speaker"]


# ---------------------------------------------------------------------------
# figures that explain the pipeline
# ---------------------------------------------------------------------------
def fig_pipeline(x, digits):
    i = int(np.where(digits == 7)[0][0])
    w = x[i]
    p = stft_power(w, N_FFT, HOP)[0].numpy()
    fb, centres = mel_filterbank(SR, N_FFT, N_MELS)
    m = (fb @ p)
    lm = np.log(m + 1e-6)

    fig, axes = ps.plt.subplots(1, 5, figsize=(15.5, 3.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    t = np.arange(len(w)) / SR

    axes[0].plot(t, w, color=ps.SERIES[0], lw=0.5)
    axes[0].set_title(f"1. waveform\n{len(w)} samples @ {SR} Hz",
                      fontsize=9.5, color=ps.INK)
    axes[0].set_xlabel("seconds", fontsize=8, color=ps.INK_SECONDARY)
    ps.style_axes(axes[0])

    axes[1].imshow(np.log(p + 1e-6), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, len(w) / SR, 0, SR / 2])
    axes[1].set_title(f"2. |STFT|² (log-scaled)\n{p.shape[0]} freq bins × "
                      f"{p.shape[1]} frames", fontsize=9.5, color=ps.INK)
    axes[1].set_ylabel("Hz", fontsize=8, color=ps.INK_SECONDARY)

    for k in range(0, N_MELS, 2):
        axes[2].plot(np.linspace(0, SR / 2, fb.shape[1]), fb[k],
                     color=ps.SERIES[k % len(ps.SERIES)], lw=1.0)
    axes[2].set_title(f"3. {N_MELS} mel filters\n(narrow low, wide high)",
                      fontsize=9.5, color=ps.INK)
    axes[2].set_xlabel("Hz", fontsize=8, color=ps.INK_SECONDARY)
    ps.style_axes(axes[2])

    axes[3].imshow(m, origin="lower", aspect="auto", cmap="magma")
    axes[3].set_title(f"4. mel spectrogram\n{m.shape[0]} × {m.shape[1]} "
                      f"(no log yet)", fontsize=9.5, color=ps.INK)
    axes[3].set_ylabel("mel bin", fontsize=8, color=ps.INK_SECONDARY)

    axes[4].imshow(lm, origin="lower", aspect="auto", cmap="magma")
    axes[4].set_title("5. log-mel spectrogram\nwhat the CNN sees",
                      fontsize=9.5, color=ps.INK)
    axes[4].set_ylabel("mel bin", fontsize=8, color=ps.INK_SECONDARY)

    axes[1].set_xlabel("seconds", fontsize=8, color=ps.INK_SECONDARY)
    for ax in axes[3:]:
        ax.set_xlabel("time frame", fontsize=8, color=ps.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'pipeline.png'}")

    # Panel 4 vs 5 is the whole argument for the log, so quantify it.
    stats = {"mel_max_over_median": float(m.max() / (np.median(m) + 1e-12)),
             "logmel_range": float(lm.max() - lm.min()),
             "mel_p99_share_of_total": float(
                 np.sort(m.ravel())[-int(0.01 * m.size):].sum() / m.sum())}
    (OUT / "log_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  loudest 1% of mel cells hold "
          f"{stats['mel_p99_share_of_total'] * 100:.0f}% of all the energy")


def fig_window_tradeoff(x, digits):
    i = int(np.where(digits == 7)[0][0])
    w = x[i]
    fig, axes = ps.plt.subplots(1, 3, figsize=(11.5, 3.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, nf in zip(axes, (64, 256, 1024)):
        p = stft_power(w, nf, HOP)[0].numpy()
        ax.imshow(np.log(p + 1e-6), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(w) / SR, 0, SR / 2])
        ax.set_title(f"window {nf} samples = {nf / SR * 1000:.0f} ms\n"
                     f"{p.shape[0]} freq bins × {p.shape[1]} frames",
                     fontsize=9.5, color=ps.INK)
        ax.set_xlabel("seconds", fontsize=8, color=ps.INK_SECONDARY)
    axes[0].set_ylabel("Hz", fontsize=8, color=ps.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "window_tradeoff.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'window_tradeoff.png'}")


def fig_mel_coverage():
    """Ask how much *new* information each extra mel filter can possibly add.

    The spectrogram only has `n_fft//2+1` frequency bins to draw from. If a mel
    triangle is narrower than the spacing between those bins, it is averaging
    barely more than one number, and its neighbour is averaging almost the same
    one -- so asking for more mel bins there buys duplicated columns, not detail.

    The "effective bin count" below is the participation ratio
    (Σw)² / Σw², a standard way to ask "how many terms is this average really
    made of?": a filter that leans entirely on one bin scores 1, one that
    weights four bins equally scores 4.
    """
    rows = []
    for n_mels in (8, 20, 40, 80):
        fb, centres = mel_filterbank(SR, N_FFT, n_mels)
        eff = (fb.sum(1) ** 2) / (np.maximum((fb ** 2).sum(1), 1e-12))
        rows.append({"n_mels": n_mels, "fft_bins": int(fb.shape[1]),
                     "min_effective_bins": round(float(eff.min()), 2),
                     "median_effective_bins": round(float(np.median(eff)), 2),
                     "filters_under_2_bins": int((eff < 2).sum())})
    (OUT / "mel_coverage.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"  n_mels={r['n_mels']:3d}: {r['filters_under_2_bins']:2d} filters "
              f"average fewer than 2 FFT bins "
              f"(median {r['median_effective_bins']:.1f} of "
              f"{r['fft_bins']} available)")


# ---------------------------------------------------------------------------
# training ablations
# ---------------------------------------------------------------------------
def _feats(x, **kw):
    """Featurize in chunks; the full 3,000 x 129 x 128 tensor is ~200 MB."""
    outs = [log_mel(x[i:i + 256], **kw).numpy() for i in range(0, len(x), 256)]
    return np.concatenate(outs)


def _lin_spec(x, n_fft=N_FFT, hop=HOP):
    outs = [torch.log(stft_power(x[i:i + 256], n_fft, hop) + 1e-6).numpy()
            for i in range(0, len(x), 256)]
    return np.concatenate(outs)


def stage_train():
    x, digits, speakers = get_data()
    tr, te = speaker_split(speakers)
    print(f"[split] train {tr.sum()} clips (5 speakers), "
          f"test {te.sum()} clips (held-out '{HELD_OUT_SPEAKER}')")
    rows = []

    def run(tag, group, model, feats, note=""):
        best, final, hist = train_classifier(
            model, feats[tr], digits[tr], feats[te], digits[te],
            epochs=EPOCHS, label=tag)
        rows.append(dict(group=group, config=tag, best=round(best, 4),
                         final=round(final, 4), params=n_params(model),
                         input_shape="x".join(str(s) for s in feats.shape[1:]),
                         note=note))
        return hist

    # --- 1. which representation? ------------------------------------------
    print("\n=== representation")
    lm = _feats(x, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, log=True)
    run("log-mel (40 bins)", "representation", MelCNN(N_MELS), lm)
    ls = _lin_spec(x)
    run(f"log-spectrogram ({ls.shape[1]} bins)", "representation",
        MelCNN(ls.shape[1]), ls, "no mel warping, 3x taller")
    run("raw waveform", "representation", WaveCNN(), x,
        f"{CLIP_LEN} samples, 1D CNN")

    # --- 2. does the log matter? -------------------------------------------
    print("\n=== log compression")
    mel_nolog = _feats(x, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, log=False)
    run("mel, no log", "log", MelCNN(N_MELS), mel_nolog, "raw power")
    rows.append(dict(group="log", config="mel, log", best=rows[0]["best"],
                     final=rows[0]["final"], params=rows[0]["params"],
                     input_shape=rows[0]["input_shape"], note="same as above"))

    # --- 3. how many mel bins? ---------------------------------------------
    print("\n=== number of mel bins")
    for nm in (8, 20, 40, 80):
        if nm == N_MELS:
            rows.append(dict(group="n_mels", config="40", best=rows[0]["best"],
                             final=rows[0]["final"], params=rows[0]["params"],
                             input_shape=rows[0]["input_shape"],
                             note="same as above"))
            continue
        f = _feats(x, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=nm, log=True)
        run(str(nm), "n_mels", MelCNN(nm), f)

    # --- 4. how long a window? ---------------------------------------------
    print("\n=== STFT window length")
    for nf in (64, 1024):
        f = _feats(x, sr=SR, n_fft=nf, hop=HOP, n_mels=N_MELS, log=True)
        run(f"{nf} samples ({nf / SR * 1000:.0f} ms)", "window",
            MelCNN(N_MELS), f)
    rows.append(dict(group="window", config=f"{N_FFT} samples "
                     f"({N_FFT / SR * 1000:.0f} ms)", best=rows[0]["best"],
                     final=rows[0]["final"], params=rows[0]["params"],
                     input_shape=rows[0]["input_shape"], note="same as above"))

    with open(OUT / "ablations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT / 'ablations.csv'}")
    return rows


def fig_ablations():
    with open(OUT / "ablations.csv") as f:
        rows = list(csv.DictReader(f))
    groups = ["representation", "log", "n_mels", "window"]
    titles = {"representation": "Which input representation?",
              "log": "Take the log, or not?",
              "n_mels": "How many mel bins?",
              "window": "How long an STFT window?"}
    fig, axes = ps.plt.subplots(1, 4, figsize=(14.5, 3.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, g in zip(axes, groups):
        ps.style_axes(ax)
        rs = [r for r in rows if r["group"] == g]
        if g == "n_mels":
            rs = sorted(rs, key=lambda r: int(r["config"]))
        names = [r["config"] for r in rs]
        vals = [float(r["best"]) for r in rs]
        colors = [ps.SERIES[0] if v == max(vals) else ps.INK_MUTED for v in vals]
        ax.barh(range(len(rs)), vals, color=colors, height=0.6)
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.axvline(0.1, color=ps.BASELINE, ls="--", lw=1.1)
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8.5,
                    color=ps.INK_SECONDARY)
        ax.set_title(titles[g], color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("accuracy, unseen speaker", fontsize=8.5,
                      color=ps.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'ablations.png'}")


def main():
    torch.set_num_threads(12)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "figures", "train", "report"])
    a = ap.parse_args()
    if a.stage in ("all", "data", "figures", "train"):
        x, digits, speakers = get_data()
    if a.stage in ("all", "figures"):
        fig_pipeline(x, digits)
        fig_window_tradeoff(x, digits)
        fig_mel_coverage()
    if a.stage in ("all", "train"):
        stage_train()
    if a.stage in ("all", "report"):
        fig_ablations()


if __name__ == "__main__":
    main()
