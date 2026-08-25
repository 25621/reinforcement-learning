"""Making a long video out of a model that only knows how to make a short one.

This file is the Phase-8 backbone: projects 36, 37, 38 and 39 import it, the
same way Phase 6 reused project 25's `dit_lib` and Phase 7 reused project 30's
`text_lib`.

The one fact everything here is built around
--------------------------------------------
Project 30's text-to-video model was trained on clips of exactly 16 pixel
frames, which the frozen 3D VAE stores as 4 latent frames.  Nothing in the
model forbids a longer input — it uses RoPE, so a longer grid still gets valid
positions — but project 29 measured what actually happens when you *ask* for a
longer grid than the model ever trained on: quality falls off a cliff.  So the
model stays at its trained shape and the LENGTH is assembled outside it.

That is the whole idea of sliding-window generation:

    window 0:  latent frames 0 1 2 3
    window 1:      latent frames 2 3 4 5          <- 2 frames of overlap
    window 2:          latent frames 4 5 6 7
    ...

Each window is one ordinary generation.  The overlap is the only place where
information can travel from one window to the next, so *how you handle the
overlap* decides whether you get one video or a pile of unrelated clips.

Why a long video cannot repeat one prompt here
----------------------------------------------
A beginner-reasonable question: why does the direction change every window
instead of "a 3 drifting right" for the whole minute?  Because of arithmetic,
not taste.  The training clips show a digit crossing most of a 64-pixel canvas
in 16 frames — roughly 1.75 pixels per frame.  Sixty-four frames of that is
110 pixels of travel on a canvas that only has ~36 pixels of room.  The digit
would be flattened against the wall for three quarters of the video, and the
model has never seen a wall.

Real long-form video has exactly the same property, which is easy to miss: a
30-second shot is not a 5-second shot repeated six times.  Something has to
change, or there is nothing to generate.  So our long timeline is a small
*shot list* — a direction per window — which is also precisely what project 38
asks a language model to write.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# the long timeline
# ---------------------------------------------------------------------------

WIN = 4              # latent frames the model was trained on
STRIDE = 2           # how many NEW latent frames each further window adds
OVERLAP = WIN - STRIDE
N_WIN = 7
TOTAL = WIN + (N_WIN - 1) * STRIDE          # 16 latent frames
PIX_PER_LAT = 4                             # the VAE's temporal compression
TOTAL_PIX = TOTAL * PIX_PER_LAT             # 64 pixel frames

# One direction per window: right, down, left, up, right, down, left.
# Turning every window keeps the digit near the middle of the canvas and puts
# a hard test (a change of direction) at every single join.
SCHEDULE = [0, 1, 2, 3, 0, 1, 2]
assert len(SCHEDULE) == N_WIN

BASE_ARM = "t5"      # project 30's winning arm: the encoder that reads it all
STYLE = "short"
CFG = 3.0
SAMPLE_STEPS = 30


def n_windows(schedule=SCHEDULE):
    return len(schedule)


def total_latent(schedule=SCHEDULE):
    """Latent frames a shot list of this length covers."""
    return WIN + (len(schedule) - 1) * STRIDE


def window_slices(schedule=SCHEDULE):
    """(start, stop) in latent frames for every window."""
    return [(k * STRIDE, k * STRIDE + WIN) for k in range(len(schedule))]


def joins(schedule=SCHEDULE):
    """Latent frame indices where a new window's contribution begins."""
    return [k * STRIDE + OVERLAP for k in range(1, len(schedule))]


def latent_direction_track(schedule=SCHEDULE):
    """The direction that is true of every latent frame of the long timeline.

    Window k owns the frames it is the first to reach — its last `STRIDE`
    frames.  Window 0 owns its whole span because nothing came before it.
    """
    track = np.zeros(total_latent(schedule), dtype=np.int64)
    track[:WIN] = schedule[0]
    for k in range(1, len(schedule)):
        s = k * STRIDE + OVERLAP
        track[s:s + STRIDE] = schedule[k]
    return track


# ---------------------------------------------------------------------------
# real long clips — the reference every measurement is compared against
# ---------------------------------------------------------------------------

def long_real(rng, batch, schedule=SCHEDULE, digits=None, speed=(1.4, 2.0),
              train=False, sprite_idx=None):
    """Real 64-frame clips of ONE digit following the whole shot list.

    Why bother rendering these?  Because every number this phase reports —
    "the seam is 3x bigger than a normal frame step", "the identity drifted by
    0.4" — is meaningless without knowing what a *genuine* long clip scores.
    Project 32 learned this the expensive way: a metric that reads 0.87 on
    real data is not allowed to be called a failure at 0.85.
    """
    sprites, labels = L._digit_sprites(train)
    D, H, W = L.DIGIT_PX, L.CANVAS, L.CANVAS
    track = latent_direction_track(schedule)
    per_pix = np.repeat(track, PIX_PER_LAT)
    n_pix = len(per_pix)
    clips = np.zeros((batch, n_pix, H, W), dtype=np.float32)
    digs = np.zeros(batch, dtype=np.int64)
    used = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        want = int(digits[b]) if digits is not None else int(rng.integers(10))
        pool = np.nonzero(labels == want)[0]
        pick = int(sprite_idx[b]) if sprite_idx is not None else \
            int(pool[rng.integers(len(pool))])
        sprite = sprites[pick]
        sp = float(rng.uniform(*speed))
        # walk the path once with a floating start, then shift it so the whole
        # path fits on the canvas — no clipping, so the digit never parks
        # against a wall in a way the model has never seen.
        ys, xs = [0.0], [0.0]
        for t in range(1, n_pix):
            dy, dx = L.DIR_VEC[L.DIRECTIONS[int(per_pix[t])]]
            ys.append(ys[-1] + dy * sp)
            xs.append(xs[-1] + dx * sp)
        ys, xs = np.array(ys), np.array(xs)
        lim_y, lim_x = H - D, W - D
        y0 = float(rng.uniform(0, max(lim_y - (ys.max() - ys.min()), 0))) \
            - ys.min()
        x0 = float(rng.uniform(0, max(lim_x - (xs.max() - xs.min()), 0))) \
            - xs.min()
        for t in range(n_pix):
            y = int(round(np.clip(y0 + ys[t], 0, lim_y)))
            x = int(round(np.clip(x0 + xs[t], 0, lim_x)))
            np.maximum(clips[b, t, y:y + D, x:x + D], sprite,
                       out=clips[b, t, y:y + D, x:x + D])
        digs[b], used[b] = want, pick
    x = torch.from_numpy(clips).unsqueeze(1)                # (B,1,T,H,W)
    return x * 2.0 - 1.0, torch.from_numpy(digs), torch.from_numpy(used)


# ---------------------------------------------------------------------------
# the base model and its prompts
# ---------------------------------------------------------------------------

_BASE = None


def base_path():
    return CK / "base.pt"


def fresh_base(arm=BASE_ARM):
    """A NEW copy of the shared model every call.

    `load_base` memoises one instance, which is right for read-only sampling
    and wrong for anything that modifies the model — freezing it, injecting
    LoRA, wrapping its blocks.  Those callers need their own copy.
    """
    p = base_path()
    model = T.build(arm)
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False)
    else:
        ck = torch.load(T.CK / f"{arm}.pt", map_location="cpu",
                        weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def load_base(arm=BASE_ARM):
    """The shared text-to-video model, plus its cached prompt bank.

    Prefers this project's longer-trained checkpoint (`--stage base`) and falls
    back to project 30's if it is not there yet.
    """
    global _BASE
    if _BASE is None:
        p = base_path()
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            model = T.build(arm)
            model.load_state_dict(ck["state"])
            model.eval()
        else:
            model, ck = T.load_arm(arm)
        _BASE = (model, T.TextBank(arm), ck)
    return _BASE


def text_for(bank, digits, dirs, style=STYLE, filler=0):
    """Context vectors for one caption per item: 'a 3 drifting right'."""
    idx = torch.tensor([T.prompt_index(int(d), int(k), style, filler)
                        for d, k in zip(digits, dirs)])
    return bank.get(idx)


def null_for(bank, n):
    return bank.null(n)


# ---------------------------------------------------------------------------
# generating one window, with or without an anchor
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_window(model, text, null, noise, anchor=None, anchor_noise=None,
                  steps=SAMPLE_STEPS, cfg=CFG, ctx=None, ctx_null=None):
    """Rectified-flow sampling of one 4-frame window.

    `anchor` is the already-decided beginning of this window (shape
    (B, C, OVERLAP, H, W)).  If given, every step of the walk overwrites those
    frames with the value they are *supposed* to have at the current noise
    level.  The model is left to invent only the rest.

    This is the same trick as image inpainting, and it works for the same
    reason: the model was trained on partly-noisy latents, so a latent whose
    first frames sit at exactly the right noise level looks completely normal
    to it.  Nothing here is fine-tuned or retrained.
    """
    flow = FL.RectifiedFlow()
    ctx = model.context(text) if ctx is None else ctx
    ctx_null = model.context(null) if ctx_null is None else ctx_null
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1)
    n_a = 0 if anchor is None else anchor.shape[2]
    for i in range(steps):
        if n_a:
            tt = ts[i]
            x[:, :, :n_a] = (1 - tt) * anchor + tt * anchor_noise
        t = ts[i].expand(x.shape[0]) * flow.T_SCALE
        v = model(x, t, None, ctx=ctx)
        if cfg != 1.0:
            v_n = model(x, t, None, ctx=ctx_null)
            v = v_n + cfg * (v - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
    if n_a:
        x[:, :, :n_a] = anchor                 # t = 0: the anchor exactly
    return x


# ---------------------------------------------------------------------------
# assembling the windows
# ---------------------------------------------------------------------------

def cross_fade(a, b, dim=2):
    """Linear cross-fade of two equal-length pieces along the time axis.

    Weight 1 -> 0 across `a` and 0 -> 1 across `b`, so the first frame is pure
    `a`, the last is pure `b`, and the change is gradual in between.
    """
    n = a.shape[dim]
    w = torch.linspace(1.0, 0.0, n + 2)[1:-1].view(
        *([1] * dim), n, *([1] * (a.dim() - dim - 1)))
    return w * a + (1 - w) * b


MODES = ["indep_cut", "shared_cut", "shared_pixel", "shared_latent",
         "anchored"]

MODE_HELP = {
    "indep_cut": "fresh noise per window, windows cut and stacked",
    "shared_cut": "one long noise field, windows cut and stacked",
    "shared_pixel": "one long noise field, overlaps cross-faded in pixels",
    "shared_latent": "one long noise field, overlaps cross-faded in latents",
    "anchored": "overlap pinned to what was already generated",
}


@torch.no_grad()
def generate_long(model, bank, digits, mode, schedule=SCHEDULE, seed=0,
                  steps=SAMPLE_STEPS, cfg=CFG, decode=True,
                  digit_schedule=None):
    """Build a 64-frame video out of seven 16-frame generations.

    Returns the long latent (B, C, TOTAL, H, W) and, unless `decode=False`,
    the decoded pixels (B, 1, TOTAL_PIX, 64, 64).
    """
    assert mode in MODES, mode
    B = len(digits)
    C, _, Hl, Wl = T.LATENT_SHAPE
    n_lat = total_latent(schedule)
    g = torch.Generator().manual_seed(seed)
    # ONE noise field for the whole timeline.  Window k reads its own slice of
    # it, so two windows that overlap start from the SAME random numbers in
    # the frames they share.  This is the core of FreeNoise's "noise
    # rescheduling": it costs nothing and it is what makes a cross-fade fuse
    # two pictures instead of dissolving between two different ones.
    long_noise = torch.randn((B, C, n_lat, Hl, Wl), generator=g)
    null = null_for(bank, B)
    ctx_null = model.context(null)

    lat = torch.zeros((B, C, n_lat, Hl, Wl))
    pix = None
    filled = 0
    prev_pix = None
    for k, (s, e) in enumerate(window_slices(schedule)):
        dirs = torch.full((B,), schedule[k], dtype=torch.long)
        # `digit_schedule` lets the SUBJECT change from shot to shot, which is
        # what project 38 needs: a shot list written by a language model may
        # (wrongly) introduce a new character halfway through, and we have to
        # be able to render that faithfully in order to measure it.
        who = digits if digit_schedule is None else digit_schedule[k]
        text = text_for(bank, who, dirs)
        if mode == "indep_cut":
            noise = torch.randn((B, C, WIN, Hl, Wl), generator=g)
        else:
            noise = long_noise[:, :, s:e].clone()
        anchor = anchor_noise = None
        if mode == "anchored" and k > 0:
            anchor = lat[:, :, s:s + OVERLAP].clone()
            anchor_noise = long_noise[:, :, s:s + OVERLAP]
        w = sample_window(model, text, null, noise, anchor=anchor,
                          anchor_noise=anchor_noise, steps=steps, cfg=cfg,
                          ctx_null=ctx_null)
        if k == 0:
            lat[:, :, s:e] = w
            filled = e
            if mode == "shared_pixel":
                pix = T.decode(w)
                prev_pix = pix
            continue
        if mode in ("indep_cut", "shared_cut"):
            lat[:, :, filled:e] = w[:, :, OVERLAP:]
        elif mode == "shared_latent":
            lat[:, :, s:s + OVERLAP] = cross_fade(
                lat[:, :, s:s + OVERLAP], w[:, :, :OVERLAP])
            lat[:, :, s + OVERLAP:e] = w[:, :, OVERLAP:]
        elif mode == "anchored":
            lat[:, :, s:e] = w
        elif mode == "shared_pixel":
            lat[:, :, filled:e] = w[:, :, OVERLAP:]     # for reference only
            new_pix = T.decode(w)
            ov = OVERLAP * PIX_PER_LAT
            head = cross_fade(prev_pix[:, :, -ov:], new_pix[:, :, :ov])
            pix = torch.cat([pix[:, :, :-ov], head, new_pix[:, :, ov:]], 2)
            prev_pix = new_pix
        filled = e
    if mode == "shared_pixel":
        return lat, pix
    return lat, (decode_long(lat) if decode else None)


@torch.no_grad()
def decode_long(lat, chunk=16):
    """Latents -> pixels.  The VAE decoder is a 3D convolution stack, so it
    accepts any number of latent frames; `chunk` exists only to bound memory.
    """
    out = []
    for s in range(0, lat.shape[2], chunk):
        out.append(T.decode(lat[:, :, s:s + chunk]))
    return torch.cat(out, 2)


# ---------------------------------------------------------------------------
# measuring a long clip
# ---------------------------------------------------------------------------

def frame_steps(clips):
    """Mean absolute change between consecutive frames: (B, T-1)."""
    d = (clips[:, :, 1:] - clips[:, :, :-1]).abs()
    return d.mean(dim=(1, 3, 4))


def seam_ratio(clips, schedule=SCHEDULE, join_frames=None, band=2):
    """How much bigger is the jump AROUND a join than a normal frame step?

    1.0 means the joins are invisible.  The `band` matters and is easy to get
    wrong: a discontinuity between two LATENT frames does not come out of the
    VAE as one bad pixel frame.  The decoder is a 3D convolution, so it spreads
    the disagreement over the four pixel frames each latent frame expands into
    and a little beyond.  Measuring only the single frame at the join therefore
    misses most of the damage — the first version of this function did exactly
    that and reported no seams at all.
    """
    if join_frames is None:
        join_frames = [j * PIX_PER_LAT for j in joins(schedule)]
    st = frame_steps(clips)                                  # (B, T-1)
    idx = sorted({min(max(j - 1 + o, 0), st.shape[1] - 1)
                  for j in join_frames for o in range(-band, band + 1)})
    if not idx:                       # a one-window "long" clip has no joins
        return torch.ones(len(st)), st.mean(1) * 0, st.mean(1)
    idx = torch.tensor(idx, dtype=torch.long)
    mask = torch.ones(st.shape[1], dtype=torch.bool)
    mask[idx] = False
    at = st[:, idx].mean(1)
    away = st[:, mask].mean(1)
    return (at / away.clamp(min=1e-6)), at, away


def centroid_path(clips):
    """Where the bright pixels are, frame by frame: (B, T, 2) as (y, x)."""
    x = ((clips.clamp(-1, 1) + 1) / 2).squeeze(1)             # (B,T,H,W)
    B, Tn, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    mass = x.sum((2, 3)).clamp(min=1e-4)
    return torch.stack([(x * ys).sum((2, 3)) / mass,
                        (x * xs).sum((2, 3)) / mass], -1)


def ink_spread(clips):
    """How spread out the bright pixels are, in pixels: (B,).

    A single digit sits in one place, so its ink has a small spread (~9 px).
    A *ghost* — the same digit visible in two places at once, which is what
    cross-fading two clips that disagree produces — doubles it.  This is the
    number that separates "smoothly blended" from "two pictures on top of each
    other", which the eye sees instantly and an averaged pixel error does not.
    """
    x = ((clips.clamp(-1, 1) + 1) / 2).squeeze(1)
    B, Tn, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    mass = x.sum((2, 3)).clamp(min=1e-4)
    cy = (x * ys).sum((2, 3)) / mass
    cx = (x * xs).sum((2, 3)) / mass
    d2 = (ys - cy[..., None, None]) ** 2 + (xs - cx[..., None, None]) ** 2
    return ((x * d2).sum((2, 3)) / mass).sqrt().mean(1)


def path_jerk(clips):
    """Mean size of the second difference of the centroid path.

    A smooth motion has a small second difference; a teleport at a seam is a
    huge one.  ("Jerk" is the everyday word for a sudden change of velocity —
    the same thing a passenger feels when a bus driver brakes late.)
    """
    p = centroid_path(clips)
    return (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]).norm(dim=-1).mean(1)


def direction_follow(clips, schedule=SCHEDULE):
    """Fraction of the timeline that moves the way its shot list asked.

    Measured per window from the centroid, not per frame, because a single
    frame's centroid is too noisy to call a direction.
    """
    p = centroid_path(clips)
    track = latent_direction_track(schedule)
    ok = []
    for k in range(len(schedule)):
        s_lat = k * STRIDE + (OVERLAP if k else 0)
        e_lat = k * STRIDE + WIN
        s, e = s_lat * PIX_PER_LAT, e_lat * PIX_PER_LAT
        d = p[:, e - 1] - p[:, s]
        want = torch.tensor([L.DIR_VEC[L.DIRECTIONS[int(track[e_lat - 1])]]])
        moved = (d * want).sum(-1)                # travel along the asked axis
        off = (d * torch.tensor([[want[0, 1], want[0, 0]]])).sum(-1).abs()
        ok.append(((moved > 1.5) & (moved > off)).float())
    return torch.stack(ok, 1)                     # (B, N_WIN)


# ---------------------------------------------------------------------------
# identity: is it still the same handwriting?
# ---------------------------------------------------------------------------

def glyph_crops(clips, size=28):
    """Cut a `size` x `size` box centred on the digit out of every frame.

    Identity has to be measured with the motion taken out, or every comparison
    would just be reporting that the digit moved.  Centring on the bright
    pixels removes the position and leaves the handwriting.
    """
    x = ((clips.clamp(-1, 1) + 1) / 2).squeeze(1)             # (B,T,H,W)
    B, Tn, H, W = x.shape
    p = centroid_path(clips)
    half = size // 2
    out = torch.zeros(B, Tn, size, size)
    pad = torch.nn.functional.pad(x, (half, half, half, half))
    for b in range(B):
        for t in range(Tn):
            cy = int(round(float(p[b, t, 0]))) + half
            cx = int(round(float(p[b, t, 1]))) + half
            cy = max(half, min(H + half - 1, cy))
            cx = max(half, min(W + half - 1, cx))
            out[b, t] = pad[b, t, cy - half:cy + half, cx - half:cx + half]
    return out


def glyph_distance(a, b):
    """Mean absolute difference between two centred glyph crops."""
    return (a - b).abs().mean(dim=(-1, -2))


def identity_drift(clips, ref=None, every=PIX_PER_LAT):
    """Glyph distance from the opening frame, sampled along the timeline.

    Returns (positions, distances) with distances shaped (B, len(positions)).
    """
    cr = glyph_crops(clips)
    base = cr[:, 0] if ref is None else ref
    pos = list(range(0, cr.shape[1], every))
    d = torch.stack([glyph_distance(cr[:, t], base) for t in pos], 1)
    return pos, d


@torch.no_grad()
def digit_votes(clips, judge, every=PIX_PER_LAT):
    """The judge's opinion of which digit is on screen, sampled along time."""
    import torch.nn.functional as F
    pos = list(range(0, clips.shape[2], every))
    votes = []
    for t in pos:
        votes.append(F.softmax(judge(clips[:, :, t]), dim=1).argmax(1))
    return pos, torch.stack(votes, 1)             # (B, len(pos))


def contact_sheet(clip, every=4, pad=1):
    """One row of frames from a single clip, as a (H, W') image in [0, 1]."""
    x = ((clip.clamp(-1, 1) + 1) / 2)[0, 0]                   # (T,H,W)
    picks = list(range(0, x.shape[0], every))
    H = x.shape[1]
    W = x.shape[2]
    sheet = np.ones((H, len(picks) * (W + pad) - pad), dtype=np.float32)
    for i, t in enumerate(picks):
        sheet[:, i * (W + pad):i * (W + pad) + W] = x[t].numpy()
    return sheet


__all__ = ["WIN", "STRIDE", "OVERLAP", "N_WIN", "TOTAL", "TOTAL_PIX",
           "PIX_PER_LAT", "SCHEDULE", "MODES", "MODE_HELP", "window_slices",
           "joins", "n_windows", "total_latent", "latent_direction_track", "long_real", "load_base",
           "text_for", "null_for", "sample_window", "generate_long",
           "decode_long", "cross_fade", "frame_steps", "seam_ratio",
           "centroid_path", "path_jerk", "ink_spread", "direction_follow",
           "glyph_crops",
           "glyph_distance", "identity_drift", "digit_votes", "contact_sheet",
           "T", "L", "FL"]
