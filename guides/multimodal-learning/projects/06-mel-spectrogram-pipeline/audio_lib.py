"""Waveform -> mel spectrogram -> small CNN, with every step written out.

No `torchaudio`, no `librosa`. The Short-Time Fourier Transform comes from
`torch.stft` and the mel filterbank is thirty lines of numpy, because the point
of the project is that the audio front end is not a black box.

Also holds the dataset: the Free Spoken Digit Dataset (FSDD) -- 3,000 recordings
of six people saying the digits 0-9, fifty times each. It is 15 MB, perfectly
balanced, and small enough to train on a CPU in seconds.

Project 07 (Whisper encoder reuse) imports this module for the loader and the
speaker-held-out split, so the two audio projects are graded on identical data.
"""

import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FSDD_URL = ("https://github.com/Jakobovski/free-spoken-digit-dataset"
            "/archive/refs/heads/master.tar.gz")
SR = 8000              # FSDD's native sample rate
CLIP_LEN = 8192        # 1.024 s -- longer than the longest clip (0.83 s)
SPEAKERS = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]
HELD_OUT_SPEAKER = "yweweler"


# ---------------------------------------------------------------------------
# 1. Reading audio
# ---------------------------------------------------------------------------
def read_wav(path):
    """Read a 16-bit PCM wav into a float32 array in [-1, 1].

    Python's standard library can do this, so there is no dependency to install.
    16-bit samples are integers from -32768 to 32767; dividing by 32768 puts them
    on the [-1, 1] scale every audio model expects.
    """
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
        sr = w.getframerate()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def fix_length(x, n=CLIP_LEN):
    """Centre the clip in a fixed-length buffer, zero-padding the rest.

    Batching needs every example the same shape. Centring (rather than padding
    only on the right) keeps the spoken word away from the edges, where a
    convolution sees half its window filled with silence.
    """
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    out = np.zeros(n, dtype=np.float32)
    s = (n - len(x)) // 2
    out[s:s + len(x)] = x
    return out


def resample_linear(x, sr_in, sr_out):
    """Change the sample rate by straight-line interpolation between samples.

    A production resampler filters first to avoid aliasing; linear interpolation
    is the cheap approximation. For the 8 kHz -> 16 kHz *up*sampling project 07
    needs it is nearly harmless, because upsampling invents no frequencies above
    the original 4 kHz ceiling -- there is nothing to alias.
    """
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x), dtype=np.float64)
    t_out = np.linspace(0, len(x) - 1, n_out)
    return np.interp(t_out, t_in, x).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. The mel scale
# ---------------------------------------------------------------------------
def hz_to_mel(f):
    """Hertz -> mels (the HTK formula).

    The *mel* scale, short for "melody", comes from 1937 listening experiments:
    people were played a tone and asked to tune a second one to "half as high".
    Their answers were not half the frequency. Below ~1 kHz pitch tracks
    frequency almost linearly; above it, you need ever-bigger jumps in Hz to
    hear the same step in pitch. This log-shaped curve is that finding.
    """
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """The inverse of `hz_to_mel` -- used to place filters evenly *in mels*."""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=256, n_mels=40, fmin=0.0, fmax=None):
    """Build the (n_mels, n_fft//2+1) matrix that squashes FFT bins into mel bins.

    Each row is a triangle: it rises from one mel-spaced centre to the next and
    falls again, so a filter's *output* is a weighted average of neighbouring
    FFT bins. Triangles overlap by half, which means no frequency falls into a
    gap between filters -- energy is redistributed, never discarded.

    Because the centres are evenly spaced in *mels* and mels are logarithmic in
    Hz, the low-frequency triangles come out narrow and the high-frequency ones
    wide: fine resolution where the ear is fine, coarse where the ear is coarse.
    """
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    # n_mels + 2 points because each triangle needs a left foot, a peak and a
    # right foot, and consecutive triangles share feet.
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (fft_freqs - left) / max(centre - left, 1e-9)
        falling = (right - fft_freqs) / max(right - centre, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0, None)
    return fb, hz_pts[1:-1]


# ---------------------------------------------------------------------------
# 3. STFT and log-mel
# ---------------------------------------------------------------------------
def stft_power(x, n_fft=256, hop=64):
    """|STFT|^2: how much energy sits at each frequency, in each time slice.

    A plain Fourier transform of the whole clip tells you *which* frequencies
    are present but not *when* -- it would give the same answer for "two-one"
    and "one-two". The fix is to chop the signal into short overlapping windows
    and transform each one, which is what "short-time" means.

    The window length is a genuine trade-off, not a tuning detail: a short window
    pins down *when* but blurs *which frequency*, and a long window does the
    opposite. You cannot have both at once.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x[None]
    # A Hann window tapers each chunk to zero at both ends. Without it, the hard
    # cut at a window edge looks like a click to the transform and smears energy
    # across every frequency ("spectral leakage").
    win = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                      center=True, return_complex=True)
    return spec.abs() ** 2                      # (B, n_fft//2+1, T)


def log_mel(x, sr=SR, n_fft=256, hop=64, n_mels=40, log=True, fb=None):
    """The full front end: power spectrogram -> mel filters -> log."""
    if fb is None:
        fb, _ = mel_filterbank(sr, n_fft, n_mels)
    p = stft_power(x, n_fft, hop)                       # (B, F, T)
    m = torch.from_numpy(fb) @ p                        # (B, n_mels, T)
    if not log:
        return m
    # Loudness is perceived logarithmically -- a whisper and a shout differ by a
    # factor of thousands in energy but only a few steps in perceived volume.
    # Taking the log compresses that range so a network does not spend its
    # capacity on the loud parts alone. The +1e-6 keeps log(0) finite.
    return torch.log(m + 1e-6)


# ---------------------------------------------------------------------------
# 4. FSDD
# ---------------------------------------------------------------------------
def fetch_fsdd(data_dir):
    root = Path(data_dir) / "free-spoken-digit-dataset-master" / "recordings"
    if root.exists():
        return root
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    tgz = Path(data_dir) / "fsdd.tar.gz"
    if not tgz.exists():
        print("[data] downloading FSDD (~15 MB) ...", flush=True)
        urllib.request.urlretrieve(FSDD_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(data_dir)
    return root


def load_fsdd(data_dir, sr_out=SR):
    """Return waveforms (N, CLIP_LEN'), digits (N,), speaker ids (N,).

    Filenames are `{digit}_{speaker}_{take}.wav`, so the labels come free.
    """
    root = fetch_fsdd(data_dir)
    files = sorted(root.glob("*.wav"))
    n = int(round(CLIP_LEN * sr_out / SR))
    xs = np.zeros((len(files), n), dtype=np.float32)
    digits = np.zeros(len(files), dtype=np.int64)
    speakers = np.zeros(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        digit, speaker, _ = f.stem.split("_")
        w, sr = read_wav(f)
        xs[i] = fix_length(resample_linear(w, sr, sr_out), n)
        digits[i] = int(digit)
        speakers[i] = SPEAKERS.index(speaker)
    return xs, digits, speakers


def speaker_split(speakers, held_out=HELD_OUT_SPEAKER):
    """Train on five voices, test on a sixth the model has never heard.

    A random split would put other recordings of the *same* speaker saying the
    *same* digit in both halves, and a model could score well by recognising the
    voice instead of the word. Holding a whole speaker out removes that shortcut,
    so the number you get is the one you actually care about.
    """
    h = SPEAKERS.index(held_out)
    return speakers != h, speakers == h


# ---------------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------------
class MelCNN(nn.Module):
    """A small 2D CNN -- the exact machinery built for images, on a mel image."""

    def __init__(self, n_mels=40, n_classes=10, width=32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(2 * c, n_classes)

    def forward(self, x):                       # x: (B, n_mels, T)
        return self.head(self.net(x[:, None]))


class WaveCNN(nn.Module):
    """The same idea applied to raw samples: a 1D CNN straight on the waveform.

    Deliberately given *more* parameters and aggressive striding, because 8,192
    raw samples is a much longer sequence than a 40x128 mel grid. It is the
    control that shows what the front end is worth.
    """

    def __init__(self, n_classes=10, width=32):
        super().__init__()
        c = width
        layers, in_c = [], 1
        for k, s, out_c in [(9, 4, c), (9, 4, 2 * c), (9, 4, 2 * c),
                            (9, 4, 4 * c), (9, 4, 4 * c)]:
            layers += [nn.Conv1d(in_c, out_c, k, stride=s, padding=k // 2),
                       nn.BatchNorm1d(out_c), nn.ReLU()]
            in_c = out_c
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_c, n_classes)

    def forward(self, x):                       # x: (B, samples)
        return self.head(self.net(x[:, None]))


def train_classifier(model, xtr, ytr, xte, yte, epochs=25, bs=64, lr=3e-3,
                     seed=0, label=""):
    """Plain AdamW + cosine schedule. Returns (best_acc, final_acc, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr_t = torch.as_tensor(xtr, dtype=torch.float32)
    xte_t = torch.as_tensor(xte, dtype=torch.float32)
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(ytr_t))
        for i in range(0, len(order), bs):
            idx = torch.from_numpy(order[i:i + bs])
            loss = F.cross_entropy(model(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(xte_t[i:i + 256]).argmax(1)
                               for i in range(0, len(yte_t), 256)])
        acc = (preds == yte_t).float().mean().item()
        hist.append(acc)
    print(f"  [{label}] best {max(hist):.4f}  final {hist[-1]:.4f}", flush=True)
    return max(hist), hist[-1], hist


def n_params(m):
    return sum(p.numel() for p in m.parameters())
