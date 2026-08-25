"""Every step from a raw waveform to a log-mel spectrogram, written by hand.

Phase 2's project 06 built a *working* front end with `torch.stft` plus a
hand-written mel filterbank, and used it to train a classifier. This module goes
one level lower: the Fourier transform itself is a matrix we build, the framing
is an explicit sliding window, and the inverse (spectrogram -> sound) is here
too, because the fastest way to see what a representation threw away is to try
to rebuild the audio from it and listen.

No `torchaudio`, no `librosa`, no `scipy` -- none of them are installed here,
which turns out to be a feature.

Audio sources (both real, both small):
  * a 6.3 s steam-train whistle, 44.1 kHz stereo, from PyTorch's own test assets
  * spoken digits from FSDD (8 kHz), reused from project 06

They are stitched into one 10-second 16 kHz clip because they look completely
different in a spectrogram: broadband hiss, a few pure whistle tones, then the
moving formants of speech.
"""

import urllib.request
import wave
from pathlib import Path

import numpy as np

SR = 16000            # everything is resampled to this
N_FFT = 512           # 32 ms at 16 kHz
HOP = 128             # 8 ms at 16 kHz
N_MELS = 80
FMIN, FMAX = 20.0, 8000.0

TRAIN_WAV_URL = ("https://raw.githubusercontent.com/pytorch/audio/v2.1.0/"
                 "test/torchaudio_unittest/assets/"
                 "steam-train-whistle-daniel_simon.wav")


def data_dir():
    return Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# 1. The Fourier transform, as a matrix
# ---------------------------------------------------------------------------
def dft_matrix(n):
    """The n x n matrix whose rows are the sine/cosine waves the DFT measures.

    The Discrete Fourier Transform asks one question per row: "how much of this
    particular wave is in my signal?" Row k is a complex wave that completes
    exactly k cycles across the n samples, and the answer is the dot product of
    that row with the signal. So the whole transform is one matrix multiply.

    exp(-2*pi*i*k*t/n) is just a compact way of writing
    cos(2*pi*k*t/n) - i*sin(2*pi*k*t/n): the cosine part measures how much of
    the signal lines up with a peak at t=0, the sine part how much lines up with
    a wave shifted a quarter cycle. Keeping both is what lets us recover *when*
    within its cycle each wave peaks -- the phase.
    """
    t = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(t, t) / n)


def naive_dft(x):
    """One frame -> its complex spectrum, by matrix multiply. O(n^2)."""
    return dft_matrix(len(x)) @ x


def rfft_naive(x):
    """Same thing, keeping only the first n/2+1 outputs.

    A real-valued signal makes the second half of the spectrum a mirror image of
    the first (bin n-k is the complex conjugate of bin k), so it carries no new
    information. Dropping it is what the "r" in `rfft` means. This is also why a
    512-point FFT of real audio gives 257 frequency bins and not 512.
    """
    return naive_dft(x)[: len(x) // 2 + 1]


# ---------------------------------------------------------------------------
# 2. Framing and windowing
# ---------------------------------------------------------------------------
def hann(n):
    """The Hann window (Julius von Hann, an Austrian meteorologist who used this
    smoothing shape on weather series long before anyone applied it to audio).

    It is one raised cosine hump: 0 at both ends, 1 in the middle.

    Why multiply a frame by it at all: cutting 512 samples out of a continuous
    recording leaves two hard edges, and the DFT reads a hard edge as a click --
    energy smeared across *every* frequency bin, which is called spectral
    leakage. Tapering both ends to zero removes the edges, so a pure tone shows
    up as one sharp line instead of a line plus a haze.

    `sym=False` (the "periodic" version) is what analysis code uses: the window
    is the first n points of an (n+1)-point symmetric window, so consecutive
    overlapping frames tile the signal evenly instead of double-counting the
    seam.
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def frame(x, n_fft=N_FFT, hop=HOP):
    """Slice the signal into overlapping chunks: (n_frames, n_fft).

    "Short-Time" Fourier Transform is named for exactly this step. A plain
    Fourier transform of a 10-second clip answers "which frequencies are in this
    recording" and completely loses *when* -- a whistle at second 2 and a
    whistle at second 8 give the same answer. Chopping the signal into short
    chunks first, and transforming each one, buys back time: one spectrum per
    chunk, laid side by side, is a picture of frequency over time.
    """
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def stft_manual(x, n_fft=N_FFT, hop=HOP, window=None):
    """Our own STFT: frame, window, then one DFT matrix multiply for all frames.

    Returns (n_frames, n_fft//2+1) complex numbers. This is a slow O(n_fft^2)
    transform on purpose -- the FFT is the same answer computed cleverly, and
    `run.py --stage verify` checks the two agree.
    """
    w = hann(n_fft) if window is None else window
    frames = frame(x, n_fft, hop) * w
    return frames.astype(np.float64) @ dft_matrix(n_fft)[:, : n_fft // 2 + 1]


def istft_manual(spec, n_fft=N_FFT, hop=HOP):
    """Complex spectrogram -> waveform, by overlap-add.

    Each frame is transformed back to samples, multiplied by the window a second
    time, and added into a running buffer at its own offset. Dividing at the end
    by the summed squared window undoes both multiplications exactly, so with
    unmodified frames this is a perfect inverse -- which is what makes it a fair
    place to *modify* frames (as Griffin-Lim does) and hear the result.
    """
    w = hann(n_fft)
    n_frames = spec.shape[0]
    full = np.concatenate([spec, np.conj(spec[:, -2:0:-1])], axis=1)
    frames = np.real(np.fft.ifft(full, axis=1))
    out = np.zeros(n_fft + hop * (n_frames - 1))
    norm = np.zeros_like(out)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * w
        norm[s:s + n_fft] += w ** 2
    return out / np.maximum(norm, 1e-8)


def stft_power(x, n_fft=N_FFT, hop=HOP):
    """Magnitude-squared spectrogram, the usual input to a mel filterbank.

    Squaring throws the phase away and keeps energy per bin. Everything after
    this point in the pipeline is deaf to phase -- the `invert` stage measures
    what that costs.
    """
    return np.abs(np.fft.rfft(frame(x, n_fft, hop) * hann(n_fft), axis=1)) ** 2


# ---------------------------------------------------------------------------
# 3. The mel scale and its filterbank
# ---------------------------------------------------------------------------
def hz_to_mel(f):
    """Hertz -> mel. "mel" is short for *melody*: the scale was built in the
    1930s by asking listeners to tune a tone until it sounded "twice as high",
    and 1000 mel was pinned to 1000 Hz as the reference point.

    Below ~500 Hz the two scales agree; above it, hearing compresses. The gap
    from 200 to 300 Hz is obvious; the gap from 7000 to 7100 Hz is inaudible,
    even though both are 100 Hz.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """The (n_mels, n_fft//2+1) matrix that folds FFT bins into mel bands.

    Recipe: place n_mels+2 points evenly on the *mel* axis, convert them back to
    Hz, and give band j a triangle that rises from point j to point j+1 and falls
    to point j+2. Neighbouring triangles overlap by half, so no frequency falls
    between two bands.

    Triangles (rather than hard on/off boxes) mean a tone drifting between two
    bands hands its energy over smoothly instead of jumping, which keeps the
    features stable under tiny pitch changes.

    Each row is normalised to unit area so a wide high-frequency band does not
    simply report a bigger number than a narrow low-frequency one for the same
    loudness ("slaney" normalisation in other libraries).
    """
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for j in range(n_mels):
        lo, mid, hi = pts[j], pts[j + 1], pts[j + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[j] = np.clip(np.minimum(up, down), 0, None)
        area = fb[j].sum()
        if area > 0:
            fb[j] /= area
    return fb


def log_mel(x, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, fb=None, eps=1e-10):
    """waveform -> log-mel spectrogram (n_frames, n_mels).

    The log is not decoration. Sound energy spans many orders of magnitude, and
    hearing is roughly logarithmic: the step from a whisper to speech feels like
    the step from speech to a shout, even though the second is a hundred times
    more energy. Without the log a handful of loud cells dominate every batch
    statistic -- project 06 measured that cost (0.72 -> 0.21 accuracy).
    """
    fb = mel_filterbank(sr, n_fft, n_mels) if fb is None else fb
    return np.log(stft_power(x, n_fft, hop) @ fb.T + eps)


def mel_to_power(mel_log, fb, eps=1e-10):
    """Undo the filterbank as far as it can be undone: least-squares.

    The filterbank is 80 rows by 257 columns, so it maps 257 numbers down to 80
    and cannot be inverted -- 177 dimensions were destroyed. The pseudo-inverse
    returns the *smallest* 257-vector consistent with the 80 we kept, which is
    the best guess available and audibly blurrier than the original.
    """
    mel = np.maximum(np.exp(mel_log) - eps, 0.0)
    return np.maximum(mel @ np.linalg.pinv(fb).T, 0.0)


# ---------------------------------------------------------------------------
# 4. Griffin-Lim: sound back out of a magnitude-only spectrogram
# ---------------------------------------------------------------------------
def griffin_lim(mag, n_fft=N_FFT, hop=HOP, iters=48, seed=0):
    """Guess the phase that a magnitude spectrogram lost (Griffin & Lim, 1984 --
    Daniel Griffin and Jae Lim, then at MIT).

    The idea is a loop between two facts that must both hold:
      1. the magnitudes are the ones we were given, and
      2. a real spectrogram of overlapping frames is *consistent* -- neighbouring
         frames share samples, so their phases cannot disagree.
    Start from random phase, rebuild a waveform (which forces consistency),
    re-analyse it (which gives a phase that is at least self-consistent), keep
    the phase, restore the true magnitudes, repeat. It converges to something
    that sounds close, never to the original: the phase really is gone.
    """
    rng = np.random.default_rng(seed)
    angle = np.exp(2j * np.pi * rng.random(mag.shape))
    x = istft_manual(mag * angle, n_fft, hop)
    for _ in range(iters):
        spec = np.fft.rfft(frame(x, n_fft, hop) * hann(n_fft), axis=1)
        angle = spec / np.maximum(np.abs(spec), 1e-10)
        n = min(len(mag), len(angle))
        x = istft_manual(mag[:n] * angle[:n], n_fft, hop)
    return x


def log_spectral_distance(a, b, n_fft=N_FFT, hop=HOP):
    """Distance between two clips in dB, averaged over every time-frequency cell.

    A plain sample-by-sample error is useless here: shift a waveform by one
    sample and it sounds identical but the error explodes. Comparing
    spectrograms measures what the ear cares about. "dB" is the decibel, one
    tenth of a bel, named after Alexander Graham Bell.
    """
    n = min(len(a), len(b))
    pa = np.log10(stft_power(a[:n], n_fft, hop) + 1e-10)
    pb = np.log10(stft_power(b[:n], n_fft, hop) + 1e-10)
    m = min(len(pa), len(pb))
    return float(np.sqrt((((pa[:m] - pb[:m]) * 10) ** 2).mean()))


# ---------------------------------------------------------------------------
# 5. Audio in and out
# ---------------------------------------------------------------------------
def read_wav(path):
    """16-bit PCM wav -> float32 in [-1, 1] (mono), plus the sample rate."""
    with wave.open(str(path)) as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    assert width == 2, "only 16-bit PCM"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def write_wav(path, x, sr=SR):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def lowpass(x, cutoff, sr, taps=127):
    """Windowed-sinc low-pass filter, applied by convolution.

    Needed before *downsampling*. A signal sampled at rate R can only represent
    frequencies below R/2 -- the Nyquist frequency, after Harry Nyquist, who
    proved the limit at Bell Labs in 1928. Anything above it does not disappear
    when you throw samples away: it *folds back* and reappears as a lower, wrong
    frequency (aliasing), the audio version of a spinning wheel looking like it
    turns backwards on film. So we delete those frequencies first.
    """
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / sr * n) * np.hanning(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def resample(x, sr_in, sr_out):
    """Change the sample rate by low-pass + linear interpolation."""
    if sr_in == sr_out:
        return x
    if sr_out < sr_in:
        x = lowpass(x, 0.45 * sr_out, sr_in)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n) * sr_in / sr_out,
                     np.arange(len(x)), x).astype(np.float32)


def _fsdd_clips(n=6):
    """A few spoken digits from project 06's FSDD download (8 kHz -> 16 kHz)."""
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "06-mel-spectrogram-pipeline"))
    import audio_lib as A
    root = A.fetch_fsdd(here.parent / "06-mel-spectrogram-pipeline" / "data")
    # filenames are `{digit}_{speaker}_{take}.wav`, so take digit 0..n-1 in order
    picks = [root / f"{d}_george_0.wav" for d in range(n)]
    out = []
    for f in picks:
        x, sr = read_wav(f)
        out.append(resample(x, sr, SR))
    return out


def showcase_clip(seconds=10.0):
    """Build (and cache) the 10-second demo clip: train whistle, then speech."""
    cache = data_dir() / "showcase.wav"
    if cache.exists():
        return read_wav(cache)[0]
    data_dir().mkdir(parents=True, exist_ok=True)
    raw = data_dir() / "steam.wav"
    if not raw.exists():
        urllib.request.urlretrieve(TRAIN_WAV_URL, raw)
    train, sr = read_wav(raw)
    train = resample(train, sr, SR)
    train = train / (np.abs(train).max() + 1e-9) * 0.7

    parts, digits = [train], _fsdd_clips(6)
    gap = np.zeros(int(0.12 * SR), dtype=np.float32)
    for d in digits:
        parts += [gap, (d / (np.abs(d).max() + 1e-9) * 0.7).astype(np.float32)]
    x = np.concatenate(parts)[: int(seconds * SR)]
    if len(x) < int(seconds * SR):
        x = np.pad(x, (0, int(seconds * SR) - len(x)))
    write_wav(cache, x)
    return x
