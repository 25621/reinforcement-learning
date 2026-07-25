"""Throw away Whisper's decoder, freeze its encoder, and probe what it kept.

Whisper was trained to transcribe. We never run that task. We take the front
half -- the encoder -- freeze it, and ask what its hidden states are good for:

  * WHAT was said  (which digit)  -- tested on a speaker the model never heard
  * WHO said it    (which voice)  -- tested on held-out recordings

and compare against two honest baselines: the raw log-mel spectrogram the
encoder itself starts from, and project 06's CNN trained from scratch.

Stages:
  embed    frozen whisper-tiny encoder over 1,500 FSDD clips, every layer saved
  probe    linear probes: layer sweep, pooling ablation, label-efficiency curve
  figures  redraw the charts from the saved CSVs

    python3 run.py --stage all       # ~5 min cold, seconds once cached
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "06-mel-spectrogram-pipeline"))
import plot_style as ps  # noqa: E402

from audio_lib import (HELD_OUT_SPEAKER, SPEAKERS, SR, MelCNN, load_fsdd,  # noqa: E402
                       log_mel, n_params, resample_linear, speaker_split,
                       train_classifier)

OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

MODEL_ID = "openai/whisper-tiny"
WHISPER_SR = 16000
TAKES = 25                 # per (digit, speaker) -> 25 * 10 * 6 = 1,500 clips
CLIP_SECONDS = 8192 / SR   # 1.024 s, the fixed buffer audio_lib pads to

# Whisper's encoder emits one frame per 20 ms of audio, so our 1.024 s clip
# occupies the first 52 of the 1,500 output frames. The rest is padding.
VALID_FRAMES = int(np.ceil(CLIP_SECONDS / 0.02))
TOTAL_FRAMES = 1500
SHOTS = [1, 3, 10, 25]


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------
def _subset(digits, speakers, takes=TAKES, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for s in range(len(SPEAKERS)):
        for d in range(10):
            idx = np.where((speakers == s) & (digits == d))[0]
            keep.append(rng.permutation(idx)[:takes])
    return np.sort(np.concatenate(keep))


def stage_embed():
    from transformers import WhisperFeatureExtractor, WhisperModel

    DATA.mkdir(parents=True, exist_ok=True)
    # Load at 16 kHz: Whisper's front end is hard-wired to that rate, and FSDD
    # ships at 8 kHz. Feeding 8 kHz audio unchanged would make every word play
    # back at half speed as far as the model is concerned.
    x, digits, speakers = load_fsdd(DATA, sr_out=WHISPER_SR)
    sel = _subset(digits, speakers)
    x, digits, speakers = x[sel], digits[sel], speakers[sel]
    print(f"[data] {len(digits)} clips at {WHISPER_SR} Hz")

    fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    enc = WhisperModel.from_pretrained(MODEL_ID).encoder
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)          # frozen: measured, never trained
    n_layers = enc.config.encoder_layers
    print(f"[model] whisper-tiny encoder: {n_layers} layers, "
          f"{n_params(enc) / 1e6:.1f}M params (frozen)")

    # hidden_states has n_layers+1 entries: index 0 is the convolutional stem's
    # output (before any transformer block), then one per block.
    pooled = np.zeros((len(digits), n_layers + 1, enc.config.d_model), np.float32)
    pooled_all = np.zeros_like(pooled)
    bs = 16
    with torch.no_grad():
        for i in range(0, len(digits), bs):
            feats = fe(list(x[i:i + bs]), sampling_rate=WHISPER_SR,
                       return_tensors="pt").input_features
            hs = enc(feats, output_hidden_states=True).hidden_states
            for L, h in enumerate(hs):
                pooled[i:i + bs, L] = h[:, :VALID_FRAMES].mean(1).numpy()
                pooled_all[i:i + bs, L] = h.mean(1).numpy()
            if i % 320 == 0:
                print(f"  {i}/{len(digits)}", flush=True)

    # Baseline: mean-pool the plain log-mel spectrogram over time. Same pooling,
    # no learned encoder -- this is what the features are worth before Whisper
    # touches them.
    mel = np.concatenate([log_mel(x[i:i + 256], sr=WHISPER_SR, n_fft=400,
                                  hop=160, n_mels=80).numpy()
                          for i in range(0, len(x), 256)])
    mel_pooled = mel.mean(2)

    np.savez_compressed(DATA / "whisper_feats.npz", pooled=pooled,
                        pooled_all=pooled_all, mel=mel_pooled,
                        digit=digits, speaker=speakers)
    (OUT / "model.json").write_text(json.dumps(
        {"model": MODEL_ID, "layers": n_layers, "d_model": enc.config.d_model,
         "params_m": round(n_params(enc) / 1e6, 2),
         "valid_frames": VALID_FRAMES, "total_frames": TOTAL_FRAMES,
         "n_clips": int(len(digits))}, indent=2))
    print(f"wrote {DATA / 'whisper_feats.npz'}")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _probe(ftr, ytr, fte, yte, n_classes, epochs=300, lr=0.05, seed=0):
    """One linear layer on frozen features -> test accuracy."""
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = ftr.mean(0, keepdims=True), ftr.std(0, keepdims=True) + 1e-6
    xtr = torch.from_numpy((ftr - mu) / sd)
    xte = torch.from_numpy((fte - mu) / sd)
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    head = nn.Linear(xtr.shape[1], n_classes)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(head(xtr), ytr_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return float((head(xte).argmax(1) == yte_t).float().mean())


def _sub(f, y, mask, shots):
    """Training features/labels, optionally thinned to `shots` per class."""
    idx = np.where(mask)[0]
    if shots is not None:
        idx = np.concatenate([idx[y[idx] == c][:shots] for c in np.unique(y)])
    return f[idx], y[idx]


def _splits(digits, speakers, seed=0):
    """Two different splits, because the two tasks need different controls."""
    # Digit: hold out a whole voice, so the probe cannot cheat by memorizing it.
    d_tr, d_te = speaker_split(speakers, HELD_OUT_SPEAKER)
    # Speaker: every voice must appear in training or the task is impossible, so
    # split by recording instead -- different takes of the same digits.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(digits))
    cut = int(0.8 * len(perm))
    s_tr = np.zeros(len(digits), bool); s_tr[perm[:cut]] = True
    return (d_tr, d_te), (s_tr, ~s_tr)


def stage_probe():
    z = np.load(DATA / "whisper_feats.npz")
    meta = json.loads((OUT / "model.json").read_text())
    pooled, pooled_all, mel = z["pooled"], z["pooled_all"], z["mel"]
    digits, speakers = z["digit"], z["speaker"]
    (d_tr, d_te), (s_tr, s_te) = _splits(digits, speakers)
    n_layers = meta["layers"]

    # --- layer sweep -------------------------------------------------------
    # Run it twice. With every label available, both tasks sit near 100% and the
    # chart is a flat line that tells you nothing -- a saturated benchmark cannot
    # rank anything. Starving the probe of labels re-opens the gap, because a
    # weak representation needs many examples to become separable and a strong
    # one does not.
    for tag, shots in (("layers.csv", None), ("layers_lowshot.csv", 2)):
        rows = []
        print(f"\n=== which layer holds what?  "
              f"({'all labels' if shots is None else f'{shots} labels/class'})")
        for L in list(range(n_layers + 1)) + [-1]:
            f = mel if L == -1 else pooled[:, L]
            acc_d = _probe(*_sub(f, digits, d_tr, shots), f[d_te],
                           digits[d_te], 10)
            acc_s = _probe(*_sub(f, speakers, s_tr, shots), f[s_te],
                           speakers[s_te], len(SPEAKERS))
            name = ("log-mel (no encoder)" if L == -1
                    else "conv stem" if L == 0 else f"block {L}")
            rows.append(dict(layer=L, name=name, digit=round(acc_d, 4),
                             speaker=round(acc_s, 4)))
            print(f"  {name:20s}  digit {acc_d:.3f}   speaker {acc_s:.3f}")
        _write_csv(OUT / tag, rows)
        if shots is None:
            best = max(rows[:-1], key=lambda r: r["digit"])["layer"]
    print(f"\n[best digit layer] {best}")

    # --- pooling ablation --------------------------------------------------
    print("\n=== pooling over padding vs over real audio")
    prow = []
    for tag, arr in (("valid frames only", pooled), ("all 1500 frames", pooled_all)):
        f = arr[:, best]
        a = _probe(f[d_tr], digits[d_tr], f[d_te], digits[d_te], 10)
        prow.append(dict(pooling=tag, frames=VALID_FRAMES if "valid" in tag
                         else TOTAL_FRAMES, digit=round(a, 4)))
        print(f"  {tag:20s} digit {a:.3f}")
    _write_csv(OUT / "pooling.csv", prow)

    # --- label efficiency: frozen probe vs training a CNN from scratch -----
    print("\n=== how many labels does each approach need?")
    x_raw, dg, sp = _fsdd_8k()
    mel8 = np.concatenate([log_mel(x_raw[i:i + 256], sr=SR, n_fft=256, hop=64,
                                   n_mels=40).numpy()
                           for i in range(0, len(x_raw), 256)])
    erows = []
    for shots in SHOTS:
        idx = np.concatenate([np.where(d_tr & (digits == c))[0][:shots * 5]
                              for c in range(10)])
        f = pooled[:, best]
        a_probe = _probe(f[idx], digits[idx], f[d_te], digits[d_te], 10)
        # Same clips, but a CNN learning the whole feature extractor itself.
        m8_tr = np.concatenate([np.where(d_tr & (dg == c))[0][:shots * 5]
                                for c in range(10)])
        acc_cnn, _, _ = train_classifier(
            MelCNN(40), mel8[m8_tr], dg[m8_tr], mel8[d_te], dg[d_te],
            epochs=30, label=f"scratch CNN {shots * 5}/class")
        erows.append(dict(per_class=shots * 5, frozen_whisper=round(a_probe, 4),
                          scratch_cnn=round(acc_cnn, 4)))
        print(f"  {shots * 5:3d} clips/digit  frozen-probe {a_probe:.3f}  "
              f"scratch-CNN {acc_cnn:.3f}")
    _write_csv(OUT / "label_efficiency.csv", erows)


def _fsdd_8k():
    """The same 1,500 clips at 8 kHz, matching project 06's CNN input."""
    z = np.load(DATA / "whisper_feats.npz")
    x, digits, speakers = load_fsdd(DATA, sr_out=SR)
    sel = _subset(digits, speakers)
    assert np.array_equal(digits[sel], z["digit"])
    return x[sel], digits[sel], speakers[sel]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _read(path, ints=(), floats=()):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ints:
            r[k] = int(r[k])
        for k in floats:
            r[k] = float(r[k])
    return rows


def fig_layers():
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    panels = [("layers.csv", "All labels: both tasks saturate,\n"
                             "so the chart cannot rank anything", axes[0]),
              ("layers_lowshot.csv", "2 labels per class: the same features,\n"
                                     "now forced to show their quality", axes[1])]
    for fname, title, ax in panels:
        ps.style_axes(ax)
        rows = _read(OUT / fname, ("layer",), ("digit", "speaker"))
        enc = [r for r in rows if r["layer"] >= 0]
        base = [r for r in rows if r["layer"] == -1][0]
        xs = [r["layer"] for r in enc]
        ax.plot(xs, [r["digit"] for r in enc], color=ps.SERIES[0], lw=2.2,
                marker="o", ms=5, label="WHAT was said (digit, unseen speaker)")
        ax.plot(xs, [r["speaker"] for r in enc], color=ps.SERIES[3], lw=2.2,
                marker="s", ms=5, label="WHO said it (speaker)")
        ax.axhline(base["digit"], color=ps.SERIES[0], ls=":", lw=1.4)
        ax.axhline(base["speaker"], color=ps.SERIES[3], ls=":", lw=1.4)
        ax.text(0.03, base["digit"] + 0.02, "log-mel, no encoder (digit)",
                fontsize=7.5, color=ps.SERIES[0])
        ax.text(0.03, base["speaker"] + 0.02, "log-mel, no encoder (speaker)",
                fontsize=7.5, color=ps.SERIES[3])
        ax.set_xticks(xs)
        ax.set_xticklabels([r["name"] for r in enc], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("encoder depth", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("linear-probe accuracy", color=ps.INK_SECONDARY,
                       fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "layer_sweep.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'layer_sweep.png'}")


def fig_efficiency():
    rows = _read(OUT / "label_efficiency.csv", ("per_class",),
                 ("frozen_whisper", "scratch_cnn"))
    fig, ax = ps.new_axes(7.2, 4.2)
    xs = [r["per_class"] for r in rows]
    ax.plot(xs, [r["frozen_whisper"] for r in rows], color=ps.SERIES[0], lw=2.2,
            marker="o", ms=5, label="frozen Whisper encoder + linear probe")
    ax.plot(xs, [r["scratch_cnn"] for r in rows], color=ps.SERIES[2], lw=2.2,
            marker="s", ms=5, label="CNN trained from scratch")
    ax.axhline(0.1, color=ps.BASELINE, ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "Reusing a pretrained encoder is a data-efficiency win",
              "labelled clips per digit", "digit accuracy, unseen speaker",
              OUT / "label_efficiency.png")


def stage_figures():
    fig_layers()
    fig_efficiency()


def main():
    torch.set_num_threads(12)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "embed", "probe", "figures"])
    a = ap.parse_args()
    if a.stage in ("all", "embed"):
        stage_embed()
    if a.stage in ("all", "probe"):
        stage_probe()
    if a.stage in ("all", "figures"):
        stage_figures()


if __name__ == "__main__":
    main()
