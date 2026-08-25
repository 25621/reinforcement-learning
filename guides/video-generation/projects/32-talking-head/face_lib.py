"""A miniature talking-head world: synthetic speakers, synthetic speech.

Why not just run EMO or Hallo?
------------------------------
Because they do not fit.  EMO is a Stable-Diffusion-scale UNet plus a
ReferenceNet plus an audio encoder, sampled with 25+ diffusion steps per frame;
one second of video is minutes of GPU time and tens of gigabytes of weights.
On the CPU this guide targets it is not minutes, it is hours per clip.

So this project builds the smallest thing that still contains every real
problem: a face whose mouth must move in time with a sound, an identity that
must survive the animation, and a speaker whose personal habits are not
visible in their photo.  Every measurement here — sync, the shift curve, the
mismatched-audio control — is exactly what talking-head papers report.

The pieces
----------
`speaker(sid)`   a person: head shape, skin tone, eye spacing, and — crucially
                 — a personal `max_open`, how far they drop their jaw.
`phones`         six sounds, each with a spectrum and a target mouth shape
                 (a "viseme": the *visible* counterpart of a phoneme).
`synth_audio`    turns a phone sequence into a waveform.
`log_mel`        turns a waveform into the per-frame features the model reads.
`render`         draws the frames.

Decoding two names
------------------
**Phoneme / viseme.** A *phoneme* is the smallest unit of sound that changes a
word's meaning; a *viseme* is the smallest unit of visible mouth shape.  They
are not the same and not one-to-one: "p", "b" and "m" all look identical from
outside (lips pressed shut), which is why lip-reading is hard and why a model
that predicts mouth shape can never fully recover the words.

**Mel scale.** Human hearing resolves low frequencies much more finely than
high ones — the gap between 100 Hz and 200 Hz is far more audible than the gap
between 5000 Hz and 5100 Hz.  The mel scale (from "melody") stretches the low
end and squashes the high end so that equal distances on the scale sound
equally far apart.  A mel spectrogram is therefore a spectrum re-binned the
way an ear would bin it — fewer numbers than a raw spectrum, and the numbers
that survive are the ones that matter for speech.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 64                 # frame is SIZE x SIZE, grayscale
T_FRAMES = 16
FPS = 8
SR = 8000                 # audio sample rate
CLIP_SEC = T_FRAMES / FPS
N_SAMPLES = int(SR * CLIP_SEC)
N_SLOTS = 8               # phones per clip (each spans two video frames)
N_MELS = 32
N_FFT, HOP = 256, 250     # 16000 / 250 = 64 spectrogram columns = 4 per frame

# ---------------------------------------------------------------------------
# sounds and the mouth shapes that go with them
# ---------------------------------------------------------------------------
# (name, formant frequencies, noisiness, loudness, mouth opening, mouth width)
PHONES = [
    ("sil", (0, 0), 0.0, 0.00, 0.00, 1.00),
    ("ah",  (730, 1100), 0.0, 1.00, 1.00, 1.05),
    ("ee",  (270, 2300), 0.0, 0.85, 0.35, 1.35),
    ("oo",  (300, 870), 0.0, 0.80, 0.55, 0.65),
    ("mm",  (250, 900), 0.0, 0.45, 0.05, 0.95),
    ("ss",  (0, 0), 1.0, 0.55, 0.20, 1.15),
]
PHONE_NAMES = [p[0] for p in PHONES]


def phone_sequence(rng, n=N_SLOTS):
    """A plausible little utterance: sounds, never two identical in a row."""
    seq = []
    for i in range(n):
        while True:
            p = int(rng.integers(len(PHONES)))
            if not seq or p != seq[-1]:
                break
        seq.append(p)
    return seq


def synth_audio(seq, rng, f0=None):
    """Turn a phone sequence into a waveform.

    Each sound is a couple of formants (resonances of the vocal tract) riding
    on a voice pitch, or filtered noise for "ss".  A short attack and decay
    stop every phone from starting with a click.
    """
    f0 = f0 or float(rng.uniform(90, 160))
    per = N_SAMPLES // len(seq)
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    t = np.arange(per) / SR
    env = np.minimum(np.minimum(t / 0.02, 1.0), np.minimum((per / SR - t) / 0.03, 1.0))
    env = np.clip(env, 0, 1)
    for i, p in enumerate(seq):
        _, (f1, f2), noisy, amp, _, _ = PHONES[p]
        if amp == 0:
            continue
        if noisy:
            sig = rng.normal(0, 1, per).astype(np.float32)
            # a crude high-pass: difference of neighbouring samples
            sig = np.diff(sig, prepend=sig[0])
        else:
            harm = np.zeros(per, dtype=np.float32)
            for k in range(1, 25):
                fk = f0 * k
                if fk > SR / 2:
                    break
                # louder near the two formants
                g = np.exp(-((fk - f1) / 180) ** 2) + \
                    0.7 * np.exp(-((fk - f2) / 280) ** 2) + 0.05
                harm += g * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28))
            sig = harm / (np.abs(harm).max() + 1e-6)
        out[i * per:(i + 1) * per] = amp * env * sig
    return out


def mouth_track(seq, max_open=1.0):
    """The ground-truth mouth opening and width, one value per video frame.

    The mouth does not snap between shapes; it slides.  Smoothing the step
    sequence is a one-line stand-in for coarticulation — the fact that how you
    shape a sound depends on the sounds either side of it.
    """
    per = T_FRAMES // len(seq)
    op = np.repeat([PHONES[p][4] for p in seq], per).astype(np.float32)
    wd = np.repeat([PHONES[p][5] for p in seq], per).astype(np.float32)
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    op = np.convolve(np.pad(op, 1, mode="edge"), k, mode="valid")
    wd = np.convolve(np.pad(wd, 1, mode="edge"), k, mode="valid")
    return op * max_open, wd


# ---------------------------------------------------------------------------
# audio features
# ---------------------------------------------------------------------------

def _mel_filters(n_mels=N_MELS, n_fft=N_FFT, sr=SR):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    edges = mel2hz(np.linspace(hz2mel(50), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        for j in range(lo, mid):
            fb[m, j] = (j - lo) / max(mid - lo, 1)
        for j in range(mid, hi):
            fb[m, j] = (hi - j) / max(hi - mid, 1)
    return torch.from_numpy(fb)


_FB = None


def log_mel(wav):
    """Waveform (B, N) -> per-video-frame log-mel features (B, T, N_MELS)."""
    global _FB
    if _FB is None:
        _FB = _mel_filters()
    spec = torch.stft(wav, N_FFT, HOP, window=torch.hann_window(N_FFT),
                      return_complex=True, center=True).abs()      # (B,F,C)
    mel = torch.einsum("mf,bfc->bmc", _FB, spec)
    mel = torch.log(mel + 1e-4)
    cols = mel.shape[-1]
    per = cols // T_FRAMES
    mel = mel[..., :per * T_FRAMES].reshape(mel.shape[0], N_MELS, T_FRAMES, per)
    mel = mel.mean(-1).transpose(1, 2)                             # (B,T,MELS)
    return (mel + 4.0) / 4.0                                       # roughly 0-centred


# ---------------------------------------------------------------------------
# speakers and rendering
# ---------------------------------------------------------------------------

N_TRAIN_SPEAKERS = 8
HELD_OUT = 8              # speaker id 8 is the one we fine-tune for


def speaker(sid):
    """One person's fixed appearance — plus one habit you cannot see."""
    rng = np.random.default_rng(1000 + sid)
    s = dict(
        hw=float(rng.uniform(19, 24)), hh=float(rng.uniform(24, 29)),
        skin=float(rng.uniform(0.68, 0.95)),
        eye_sep=float(rng.uniform(7.5, 10.5)),
        eye_y=float(rng.uniform(5.5, 8.0)),
        mouth_y=float(rng.uniform(9.0, 12.0)),
        mouth_w=float(rng.uniform(6.5, 9.0)),
        max_open=float(rng.uniform(0.75, 1.0)),
    )
    if sid == HELD_OUT:
        # The reason this project has a fine-tuning stage: this speaker barely
        # opens their mouth.  Nothing in a still photograph of a closed mouth
        # reveals that, so a model trained on other people will over-animate
        # them no matter how good the photo is.
        s["max_open"] = 0.40
        s["mouth_w"] = 9.5
    return s


def _soft(d, k=1.2):
    return 1.0 / (1.0 + np.exp(np.clip(d * k, -30, 30)))


def render(sp, opening, width, jitter_y=None):
    """Draw one clip: (T, SIZE, SIZE) in [0, 1]."""
    Tn = len(opening)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy = SIZE / 2, SIZE / 2 + 1.5
    out = np.zeros((Tn, SIZE, SIZE), dtype=np.float32)
    for t in range(Tn):
        dy = 0.0 if jitter_y is None else float(jitter_y[t])
        img = np.full((SIZE, SIZE), 0.06, dtype=np.float32)
        hh = sp["hh"] + 1.5 * opening[t]              # jaw drops as mouth opens
        head = _soft((((xs - cx) / sp["hw"]) ** 2
                      + ((ys - cy - dy) / hh) ** 2) - 1.0, 8.0)
        img = img + head * (sp["skin"] - 0.06)
        for sgn in (-1, 1):
            ex = cx + sgn * sp["eye_sep"]
            ey = cy - sp["eye_y"] + dy
            eye = _soft(((xs - ex) ** 2 + (ys - ey) ** 2) / 6.0 - 1.0, 6.0)
            img = img - eye * (sp["skin"] - 0.05)
        nose = _soft((((xs - cx) / 1.6) ** 2
                      + ((ys - cy - 1.5 - dy) / 3.0) ** 2) - 1.0, 6.0)
        img = img - nose * 0.10
        mw = sp["mouth_w"] * width[t]
        mh = 0.9 + 5.5 * opening[t]
        my = cy + sp["mouth_y"] + dy
        mouth = _soft((((xs - cx) / mw) ** 2 + ((ys - my) / mh) ** 2) - 1.0, 6.0)
        img = img - mouth * (sp["skin"] - 0.04)
        out[t] = np.clip(img, 0, 1)
    return out


def mouth_roi():
    """The box the aperture measure looks at (same for every speaker)."""
    cy = SIZE / 2 + 1.5
    y0 = int(cy + 5.0)
    return slice(y0, min(y0 + 18, SIZE)), slice(SIZE // 2 - 12, SIZE // 2 + 12)


def aperture(frames):
    """How open is the mouth, measured from pixels: (B, T).

    The mouth is the dark blob in the lower face, so "how much darkness is in
    the mouth box" rises and falls with the jaw.  It is a proxy, not a
    measurement of millimetres — which is fine, because every use of it here
    is a *correlation*, and correlation does not care about units.
    """
    ry, rx = mouth_roi()
    box = frames[..., ry, rx]
    return (1.0 - box).mean((-2, -1))


def portrait(sp):
    """A single still photo of the speaker, mouth closed."""
    return render(sp, np.zeros(1, dtype=np.float32),
                  np.ones(1, dtype=np.float32))[0]


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def make_clips(sids, per_speaker, seed=0, head_motion=True):
    """Clips of several speakers, with audio, features and ground truth."""
    rng = np.random.default_rng(seed)
    frames, wavs, ports, opens, sid_out = [], [], [], [], []
    for sid in sids:
        sp = speaker(sid)
        for _ in range(per_speaker):
            seq = phone_sequence(rng)
            wav = synth_audio(seq, rng)
            op, wd = mouth_track(seq, sp["max_open"])
            jy = None
            if head_motion:
                # a slow head sway, so "everything that moves is the mouth"
                # is not trivially true
                jy = 0.8 * np.sin(np.linspace(0, rng.uniform(2, 5), T_FRAMES)
                                  + rng.uniform(0, 6.28))
            frames.append(render(sp, op, wd, jy))
            wavs.append(wav)
            ports.append(portrait(sp))
            opens.append(op)
            sid_out.append(sid)
    x = torch.from_numpy(np.stack(frames))                 # (N,T,H,W)
    wav = torch.from_numpy(np.stack(wavs))
    return dict(frames=x.unsqueeze(1), wav=wav,
                portrait=torch.from_numpy(np.stack(ports)).unsqueeze(1),
                mel=log_mel(wav), open=torch.from_numpy(np.stack(opens)),
                sid=torch.tensor(sid_out))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class TalkingHead(nn.Module):
    """Portrait in, audio in, video out.

    The split that defines the task: the portrait decides *who*, the audio
    decides *how the face moves*.  The portrait is encoded once and reused for
    every frame; the audio produces one control vector per frame, which
    modulates the shared portrait features (FiLM — see project 13).

    Why does the audio encoder mix across time?
        Because a mouth shape is not a function of one instant of sound.
        Lips start closing before the "m" arrives and are still rounded after
        the "oo" ends — coarticulation.  A per-frame audio lookup would have
        to guess; a small 1D convolution over neighbouring frames does not.
    """

    def __init__(self, ch=64, adim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(),      # 64 -> 32
            nn.Conv2d(32, ch, 4, 2, 1), nn.SiLU(),     # 32 -> 16
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
        )
        self.aud = nn.Sequential(nn.Linear(N_MELS, adim), nn.SiLU(),
                                 nn.Linear(adim, adim), nn.SiLU())
        self.mix = nn.Sequential(nn.Conv1d(adim, adim, 5, 1, 2), nn.SiLU(),
                                 nn.Conv1d(adim, adim, 5, 1, 2), nn.SiLU())
        self.film = nn.Linear(adim, 2 * ch)
        self.dec = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, 32, 4, 2, 1), nn.SiLU(),   # 16 -> 32
            nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),   # 32 -> 64
        )
        # the portrait is also handed straight to the last layer, so sharp
        # identity detail does not have to survive the bottleneck
        self.out = nn.Conv2d(16 + 1, 1, 3, 1, 1)

    def forward(self, port, mel):
        B, _, H, W = port.shape
        Tn = mel.shape[1]
        f = self.enc(port)                                  # (B,ch,16,16)
        a = self.aud(mel)                                   # (B,T,adim)
        a = self.mix(a.transpose(1, 2)).transpose(1, 2)
        sc, sh = self.film(a).chunk(2, -1)                  # (B,T,ch) each
        f = f[:, None] * (1 + sc[..., None, None]) + sh[..., None, None]
        f = f.reshape(B * Tn, *f.shape[2:])
        h = self.dec(f)
        p = port[:, None].expand(B, Tn, 1, H, W).reshape(B * Tn, 1, H, W)
        y = torch.sigmoid(self.out(torch.cat([h, p], 1)))
        return y.reshape(B, Tn, 1, H, W).transpose(1, 2)    # (B,1,T,H,W)


def loss_fn(pred, target, mouth_weight=4.0):
    """L1 everywhere, extra weight on the mouth.

    Without the extra weight the mouth is a tiny fraction of the pixels, and a
    model can drive the average error down to almost nothing while animating
    nothing at all — the cheeks and background are most of the picture.
    Weighting the region we actually care about is the standard fix.
    """
    ry, rx = mouth_roi()
    base = (pred - target).abs().mean()
    mouth = (pred[..., ry, rx] - target[..., ry, rx]).abs().mean()
    return base + mouth_weight * mouth


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def corr(a, b):
    """Pearson correlation between two (B, T) tracks, averaged over clips."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = a.pow(2).sum(-1).sqrt() * b.pow(2).sum(-1).sqrt()
    return float((num / den.clamp(min=1e-6)).mean())


def shifted_corr(pred_track, true_track, k):
    """Correlation after sliding the truth by k frames — the sync curve."""
    if k > 0:
        a, b = pred_track[:, k:], true_track[:, :-k]
    elif k < 0:
        a, b = pred_track[:, :k], true_track[:, -k:]
    else:
        a, b = pred_track, true_track
    return corr(a, b)


def identity_psnr(pred, target):
    """Reconstruction quality OUTSIDE the mouth: did the face stay itself?"""
    ry, rx = mouth_roi()
    m = torch.ones_like(target)
    m[..., ry, rx] = 0
    mse = (((pred - target) ** 2) * m).sum() / m.sum()
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-12)))


def temporal_jitter(frames):
    return float((frames[:, :, 1:] - frames[:, :, :-1]).abs().mean())
