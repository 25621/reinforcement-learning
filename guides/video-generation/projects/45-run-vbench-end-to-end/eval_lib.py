"""The Phase-10 backbone: a tiny captioned text-to-video model, plus the
machinery to *evaluate* it from many angles.

Projects 45, 46, 47, 48, 49 and 50 all import this file, the same way Phase 6
reused project 25's `dit_lib`, Phase 7 reused project 30's `text_lib`, Phase 8
reused project 35's `long_lib`, and Phase 9 reused project 40's `world_lib`.

Why Phase 10 starts from a new toy
----------------------------------
Phase 10 is not about a new architecture.  It is about *data, scale, and
evaluation* — the operational reality of shipping a video model.  Those lessons
need a generator we can (a) train in a couple of minutes on a CPU, (b) probe
along many independent quality axes, and (c) deliberately feed good or bad data
to.  Phase 9's coin-grid world is a *world model* (one action per frame), which
is the wrong shape: here we want a plain **text-to-video** model — say what you
want once, get a short clip.

The toy: captioned moving-sprite clips
--------------------------------------
A clip is 8 frames of a small grey sprite gliding across a dark square.  Three
attributes, named in the caption, decide what the clip looks like:

    shape      ball  | block            (a soft disk, or a soft square)
    direction  up | down | left | right (which way it launches)
    speed      slow | fast              (how many pixels it moves per frame)

So a caption is literally "a fast ball moving left".  The sprite bounces off the
four walls like a ball in a squash court — a simple, checkable *physical* rule
that project 50 probes.

Why a learned embedding table instead of a frozen CLIP text encoder
-------------------------------------------------------------------
Phase 7 fed captions through the REAL, frozen CLIP-L / T5 encoders, because its
captions were open-ended English and it needed a model that already knew what
"drifting" and "teal" mean.  Here the vocabulary is *closed*: exactly
2 x 4 x 2 = 16 possible captions.  A frozen encoder pretrained on the whole
internet would be enormous overkill and would slow every experiment down with a
model download.  When the vocabulary is a short fixed list, the natural encoder
is a small **learned embedding table** — one trainable vector per shape, per
direction, per speed.  It plays exactly the role CLIP played (turn words into a
conditioning vector) without the baggage a closed vocabulary does not need.

The generator itself is a small conditional flow-matching model — the same
rectified-flow recipe as project 26, steered by that caption vector through
FiLM, exactly like project 40's world model.  We work directly on the small
frames (no VAE) because at 16x16 there is nothing to compress.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
import flow_lib as FL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------

SHAPES = ["ball", "block"]
DIRS = ["up", "down", "left", "right"]
SPEEDS = ["slow", "fast"]
# per-attribute deltas.  bounce handles the walls.
DIR_VEC = {"up": (-1.0, 0.0), "down": (1.0, 0.0),
           "left": (0.0, -1.0), "right": (0.0, 1.0)}
SPEED_PX = {"slow": 1.4, "fast": 2.8}          # pixels per frame

T = 8                        # frames per clip
H = W = 16                   # default (square) frame size
RADIUS = 2.1                 # sprite half-size in pixels
MARGIN = 3.0                 # keep the start away from the wall

# "unknown" is an extra index every factor carries, so a caption can *omit* an
# attribute (project 46's bad captions) or drop it entirely (CFG's null prompt).
N_SHAPE, N_DIR, N_SPEED = len(SHAPES), len(DIRS), len(SPEEDS)
UNK = {"shape": N_SHAPE, "dir": N_DIR, "speed": N_SPEED}

COMBOS = [(s, d, v) for s in range(N_SHAPE)
          for d in range(N_DIR) for v in range(N_SPEED)]


def caption(shape, direction, speed):
    """Human-readable caption for a (shape, dir, speed) index tuple."""
    def word(lst, i):
        return "?" if i >= len(lst) else lst[i]
    return f"a {word(SPEEDS, speed)} {word(SHAPES, shape)} moving " \
           f"{word(DIRS, direction)}"


# ---------------------------------------------------------------------------
# rendering: turn attributes into pixels (the ground-truth "renderer")
# ---------------------------------------------------------------------------

def _soft_sprite(cy, cx, shape, h, w):
    """One frame: a soft ball or block centred at (cy, cx).

    Anti-aliased on purpose — a hard-edged sprite would make "motion smoothness"
    meaningless (every sub-pixel move would snap to the same pixels).  The soft
    edge lets the centre move continuously, so a jittery generation is visibly
    jittery.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    if shape == 0:                                   # ball: round (L2) distance
        dist = np.sqrt(dy * dy + dx * dx)
    else:                                            # block: square (Linf)
        dist = np.maximum(np.abs(dy), np.abs(dx))
    return np.clip(1.0 - (dist - RADIUS), 0.0, 1.0)


def render_clip(shape, direction, speed, y0, x0, h=H, w=W):
    """Full 8-frame clip.  Constant velocity, bounce off the walls.

    Returns (T, h, w) float32 in [0, 1].
    """
    vy, vx = DIR_VEC[DIRS[direction]]
    mag = SPEED_PX[SPEEDS[speed]]
    vy, vx = vy * mag, vx * mag
    lo, hi_y, hi_x = RADIUS, h - 1 - RADIUS, w - 1 - RADIUS
    cy, cx = float(y0), float(x0)
    frames = []
    for _ in range(T):
        frames.append(_soft_sprite(cy, cx, shape, h, w))
        cy, cx = cy + vy, cx + vx
        if cy < lo:
            cy, vy = 2 * lo - cy, -vy               # reflect off top
        if cy > hi_y:
            cy, vy = 2 * hi_y - cy, -vy             # reflect off bottom
        if cx < lo:
            cx, vx = 2 * lo - cx, -vx
        if cx > hi_x:
            cx, vx = 2 * hi_x - cx, -vx
    return np.stack(frames).astype(np.float32)


def sample_start(shape, direction, speed, rng, h=H, w=W):
    """A random, physically-valid start position for this caption.

    We push the start away from the wall it launches *towards*, so the first
    couple of frames are always bounce-free — that keeps the launch direction
    unambiguous for the read-back metric.
    """
    vy, vx = DIR_VEC[DIRS[direction]]
    y0 = rng.uniform(MARGIN, h - 1 - MARGIN)
    x0 = rng.uniform(MARGIN, w - 1 - MARGIN)
    if vy < 0:
        y0 = rng.uniform(MARGIN + 3, h - 1 - MARGIN)
    if vy > 0:
        y0 = rng.uniform(MARGIN, h - 1 - MARGIN - 3)
    if vx < 0:
        x0 = rng.uniform(MARGIN + 3, w - 1 - MARGIN)
    if vx > 0:
        x0 = rng.uniform(MARGIN, w - 1 - MARGIN - 3)
    return y0, x0


# ---------------------------------------------------------------------------
# a dataset of clips (stored as attribute tuples, rendered on the fly)
# ---------------------------------------------------------------------------

def make_dataset(n, seed=0, h=H, w=W):
    """n clips: pick a caption uniformly, a valid start, remember both."""
    rng = np.random.default_rng(seed)
    shape = rng.integers(0, N_SHAPE, size=n)
    direction = rng.integers(0, N_DIR, size=n)
    speed = rng.integers(0, N_SPEED, size=n)
    y0 = np.zeros(n, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        y0[i], x0[i] = sample_start(shape[i], direction[i], speed[i], rng, h, w)
    return dict(shape=shape, dir=direction, speed=speed, y0=y0, x0=x0, h=h, w=w)


def render_batch(ds, idx):
    """Render a set of clips from the dataset -> (B, T, h, w) tensor."""
    h, w = ds["h"], ds["w"]
    out = np.stack([render_clip(ds["shape"][i], ds["dir"][i], ds["speed"][i],
                                ds["y0"][i], ds["x0"][i], h, w) for i in idx])
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# the caption -> conditioning vector (our "text encoder")
# ---------------------------------------------------------------------------

class CaptionEncoder(nn.Module):
    """Three tiny embedding tables (shape, dir, speed) summed into one vector.

    Each table has an extra "unknown" row so a caption can leave an attribute
    unspecified.  This is the closed-vocabulary stand-in for a frozen CLIP text
    encoder: it turns words into a conditioning vector, and it is trained
    jointly with the generator (unlike CLIP, which stays frozen — but CLIP is
    frozen only to protect knowledge a closed 16-word vocabulary does not have).
    """

    def __init__(self, dim=128):
        super().__init__()
        self.e_shape = nn.Embedding(N_SHAPE + 1, dim)
        self.e_dir = nn.Embedding(N_DIR + 1, dim)
        self.e_speed = nn.Embedding(N_SPEED + 1, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))

    def forward(self, cap):
        """cap: (B, 3) long tensor of (shape, dir, speed) indices."""
        v = self.e_shape(cap[:, 0]) + self.e_dir(cap[:, 1]) \
            + self.e_speed(cap[:, 2])
        return self.mlp(v)


def null_caption(n):
    """The all-unknown caption — used as the unconditional prompt for CFG."""
    return torch.tensor([[UNK["shape"], UNK["dir"], UNK["speed"]]]) \
        .repeat(n, 1)


def caption_tensor(ds, idx):
    return torch.stack([torch.from_numpy(ds["shape"][idx]),
                        torch.from_numpy(ds["dir"][idx]),
                        torch.from_numpy(ds["speed"][idx])], dim=1).long()


# ---------------------------------------------------------------------------
# the generator: a small conditional flow-matching U-Net
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32)
                      / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class FiLMBlock(nn.Module):
    """Conv block steered by the conditioning vector (timestep + caption)."""

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.n2(h) * (1 + scale) + shift)
        return self.skip(x) + self.c2(h)


class VideoGen(nn.Module):
    """Denoises a whole 8-frame clip at once, conditioned on a caption.

    Frames live in the channel axis (T channels of one h x w image), so this is
    an ordinary 2D U-Net.  At 16x16 x 8 frames that is far cheaper than 3D
    convs and loses nothing — the caption vector, not spatial depth, carries all
    the structure.
    """

    def __init__(self, base=64, cond=128, frames=T):
        super().__init__()
        self.frames = frames
        self.cond_dim = cond
        self.caps = CaptionEncoder(cond)
        self.t_mlp = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(),
                                   nn.Linear(cond, cond))
        self.stem = nn.Conv2d(frames, base, 3, padding=1)
        self.d1 = FiLMBlock(base, base, cond)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.mid = FiLMBlock(base * 2, base * 2, cond)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = FiLMBlock(base * 2, base, cond)
        self.out_n = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, frames, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def cond_vector(self, t, cap):
        return self.t_mlp(timestep_embedding(t, self.cond_dim)) + self.caps(cap)

    def forward(self, x, t, cap):
        """x: (B, T, h, w) noisy clip.  cap: (B, 3) caption indices."""
        c = self.cond_vector(t, cap)
        h = self.d1(self.stem(x), c)
        m = self.mid(self.down(h), c)
        u = self.u1(torch.cat([self.up(m), h], dim=1), c)
        return self.out(F.silu(self.out_n(u)))


class _Wrap(nn.Module):
    """Adapts VideoGen to the (x, t) signature flow_lib.sample expects."""

    def __init__(self, net, cap):
        super().__init__()
        self.net, self.cap = net, cap

    def forward(self, x, t):
        return self.net(x, t, self.cap)


@torch.no_grad()
def sample(net, cap, steps=30, scale=1.0, generator=None, h=H, w=W,
           frames=T):
    """Generate clips for a batch of captions.  Returns (B, T, h, w).

    `scale` is the classifier-free-guidance strength: at scale 1 we just follow
    the caption; above 1 we push away from the null (all-unknown) caption to
    make the clip follow the words harder.
    """
    flow = FL.RectifiedFlow()
    n = cap.shape[0]
    shape = (n, frames, h, w)
    if scale == 1.0:
        return flow.sample(_Wrap(net, cap), shape, steps=steps,
                           generator=generator)
    null = null_caption(n)
    x = torch.randn(shape, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(n) * flow.T_SCALE
        v_c = net(x, t, cap)
        v_n = net(x, t, null)
        v = v_n + scale * (v_c - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(net, ds, steps=3000, batch=128, lr=2e-3, drop=0.1, seed=0,
          log_every=500, corrupt=None):
    """Flow-matching training.  Optionally corrupt captions (project 46).

    `corrupt` is a function ds, idx, rng -> (B, 3) caption tensor, letting a
    caller feed the model *different* captions than the true attributes (the
    whole point of the recaptioning experiment).  Default: the true caption.
    """
    flow = FL.RectifiedFlow()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(ds["shape"])
    losses = []
    net.train()
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        x0 = render_batch(ds, idx)
        cap = corrupt(ds, idx, rng) if corrupt else caption_tensor(ds, idx)
        # classifier-free dropout: sometimes show the null caption
        mask = torch.rand(batch, generator=g) < drop
        cap[mask] = null_caption(int(mask.sum())) if mask.any() else cap[mask]
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(batch, generator=g)
        xt = flow.interpolate(x0, t, noise)
        target = flow.target(x0, noise)
        pred = net(xt, t * flow.T_SCALE, cap)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}  loss {np.mean(losses[-100:]):.4f}")
    return losses


# ---------------------------------------------------------------------------
# reading a (possibly messy) generated clip back into attributes
# ---------------------------------------------------------------------------

def _centroid(frame, thresh=0.25):
    """Intensity-weighted centre of a frame, or None if the sprite vanished."""
    m = np.clip(frame - thresh, 0.0, None)
    total = m.sum()
    if total < 0.5:
        return None, 0.0
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    return (float((m * yy).sum() / total),
            float((m * xx).sum() / total)), float(frame.max())


def read_clip(clip):
    """Read attributes back out of a generated clip.

    Returns a dict with the estimated per-frame centres, the launch velocity,
    predicted (dir, speed, shape), plus quality proxies (snap error, subject
    mass).  Everything the Phase-10 metrics stand on is computed here, applied
    identically to a real clip and a generated one.
    """
    clip = np.asarray(clip, dtype=np.float32)
    centres, peaks, mass = [], [], []
    for f in clip:
        c, pk = _centroid(f)
        centres.append(c)
        peaks.append(pk)
        mass.append(float(np.clip(f, 0, None).sum()))
    # launch velocity from the first bounce-free step we can measure
    v0 = None
    for i in range(len(centres) - 1):
        if centres[i] is not None and centres[i + 1] is not None:
            v0 = (centres[i + 1][0] - centres[i][0],
                  centres[i + 1][1] - centres[i][1])
            break
    pred_dir, pred_speed = None, None
    if v0 is not None:
        vy, vx = v0
        if abs(vy) >= abs(vx):
            pred_dir = 0 if vy < 0 else 1          # up / down
        else:
            pred_dir = 2 if vx < 0 else 3          # left / right
        mag = (vy * vy + vx * vx) ** 0.5
        mid = 0.5 * (SPEED_PX["slow"] + SPEED_PX["fast"])
        pred_speed = 0 if mag < mid else 1
    # shape from corner energy: a block fills its corners, a ball does not
    pred_shape = _read_shape(clip, centres)
    # snap error: how far pixels sit from a clean {0, 1} value -> blur / noise
    snap = float(np.minimum(np.abs(clip), np.abs(clip - 1.0)).mean())
    return dict(centres=centres, peaks=peaks, mass=mass, v0=v0,
                pred_dir=pred_dir, pred_speed=pred_speed, pred_shape=pred_shape,
                snap=snap)


def _read_shape(clip, centres, r=2):
    """Corner-vs-edge energy at the sprite centre, averaged over frames.

    A block lights the four diagonal corners of its bounding box as brightly as
    its edges; a ball's corners are dark.  So corner/edge ratio separates them.
    """
    ratios = []
    for f, c in zip(clip, centres):
        if c is None:
            continue
        cy, cx = int(round(c[0])), int(round(c[1]))
        h, w = f.shape
        if cy - r < 0 or cy + r >= h or cx - r < 0 or cx + r >= w:
            continue
        corners = (f[cy - r, cx - r] + f[cy - r, cx + r]
                   + f[cy + r, cx - r] + f[cy + r, cx + r]) / 4
        edges = (f[cy - r, cx] + f[cy + r, cx]
                 + f[cy, cx - r] + f[cy, cx + r]) / 4
        if edges > 0.1:
            ratios.append(corners / (edges + 1e-6))
    if not ratios:
        return None
    return 1 if np.mean(ratios) > 0.5 else 0        # 1 = block, 0 = ball


# ---------------------------------------------------------------------------
# the evaluation axes (a miniature VBench)
# ---------------------------------------------------------------------------

def text_alignment(clips, caps):
    """Per-attribute prompt-following: did the clip show what the caption said?

    Returns a dict of accuracies for direction, speed, shape, and their mean.
    This is the "does it obey the words" axis — VBench's text-alignment family.
    """
    caps = np.asarray(caps)
    dir_ok = speed_ok = shape_ok = n = 0
    for clip, cap in zip(clips, caps):
        r = read_clip(clip)
        n += 1
        dir_ok += int(r["pred_dir"] == cap[1])
        speed_ok += int(r["pred_speed"] == cap[2])
        shape_ok += int(r["pred_shape"] == cap[0])
    d = dict(direction=dir_ok / n, speed=speed_ok / n, shape=shape_ok / n)
    d["mean"] = (d["direction"] + d["speed"] + d["shape"]) / 3
    return d


def subject_consistency(clips):
    """Does the sprite keep its brightness/size across frames?

    A model that lets the subject fade or throb scores low.  We use the sprite's
    total mass per frame; a steady subject has near-constant mass.
    """
    out = []
    for clip in clips:
        m = np.array([np.clip(f, 0, None).sum() for f in clip])
        if m.mean() < 1e-3:
            out.append(0.0)
        else:
            out.append(float(np.clip(1.0 - m.std() / (m.mean() + 1e-6),
                                     0.0, 1.0)))
    return float(np.mean(out))


def motion_smoothness(clips):
    """Is the trajectory smooth, or does the centre jitter frame-to-frame?

    The path is straight lines with occasional bounces, so the *second*
    difference of the centre should be near zero.  Jitter shows up as large
    second differences.  We report 1 - normalised jitter.
    """
    out = []
    for clip in clips:
        pts = [p for p in read_clip(clip)["centres"] if p is not None]
        if len(pts) < 3:
            out.append(0.0)
            continue
        arr = np.array(pts)
        acc = np.abs(np.diff(arr, n=2, axis=0)).mean()
        out.append(float(np.clip(1.0 - acc / 3.0, 0.0, 1.0)))
    return float(np.mean(out))


def imaging_quality(clips):
    """Sharpness: are pixels clean (near 0 or 1), or smeared to grey?

    A blurry or noisy generator leaves lots of mid-grey pixels; a crisp one does
    not.  We report 1 - 3 * mean snap error (the x3 spreads the range so a small
    absolute blur is visible), so higher is sharper.
    """
    out = [1.0 - 3.0 * read_clip(clip)["snap"] for clip in clips]
    return float(np.clip(np.mean(out), 0.0, 1.0))


def background_stability(clips):
    """Temporal flicker: does the background stay dark, or twinkle?

    We take the pixels that are dark in every frame's *median* and measure how
    much they flicker over time.  A hallucinating generator sparkles here.
    """
    out = []
    for clip in clips:
        clip = np.asarray(clip, dtype=np.float32)
        med = np.median(clip, axis=0)
        bg = med < 0.15
        if bg.sum() < 4:
            out.append(0.0)
            continue
        flick = clip[:, bg].std(axis=0).mean()
        out.append(float(np.clip(1.0 - flick * 4.0, 0.0, 1.0)))
    return float(np.mean(out))


AXES = {
    "text_alignment": lambda clips, caps: text_alignment(clips, caps)["mean"],
    "subject_consistency": lambda clips, caps: subject_consistency(clips),
    "motion_smoothness": lambda clips, caps: motion_smoothness(clips),
    "imaging_quality": lambda clips, caps: imaging_quality(clips),
    "background_stability": lambda clips, caps: background_stability(clips),
}


def vbench_score(clips, caps):
    """Run every axis and return the per-axis dict plus the simple mean."""
    scores = {name: fn(clips, caps) for name, fn in AXES.items()}
    scores["overall"] = float(np.mean(list(scores.values())))
    return scores


# ---------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------

def colorize(clip):
    """(T, h, w) grey -> (T, h, w, 3) warm-on-dark RGB for display."""
    clip = np.clip(np.asarray(clip, dtype=np.float32), 0.0, 1.0)
    lo = np.array([0.05, 0.06, 0.10])
    hi = np.array([0.99, 0.82, 0.42])
    return lo + clip[..., None] * (hi - lo)


def write_gif(clip, path, scale=10, fps=6):
    from PIL import Image
    rgb = colorize(clip)
    imgs = [Image.fromarray((np.repeat(np.repeat(f, scale, 0), scale, 1)
                             * 255).astype(np.uint8), "RGB") for f in rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"wrote {path}")


def strip(rows, path, scale=6, gap=2):
    """A grid of frames as one PNG.  rows: list of lists of (h, w) frames."""
    from PIL import Image
    tiles = [[colorize(f) if f is not None else None for f in row]
             for row in rows]
    th = max(t.shape[0] for row in tiles for t in row if t is not None) * scale
    tw = max(t.shape[1] for row in tiles for t in row if t is not None) * scale
    ny, nx = len(tiles), max(len(r) for r in tiles)
    canvas = np.full((ny * (th + gap) - gap, nx * (tw + gap) - gap, 3), 0.7,
                     dtype=np.float32)
    for i, row in enumerate(tiles):
        for j, t in enumerate(row):
            if t is None:
                continue
            px = np.repeat(np.repeat(t, scale, 0), scale, 1)
            y, x = i * (th + gap), j * (tw + gap)
            canvas[y:y + px.shape[0], x:x + px.shape[1]] = px
    Image.fromarray((canvas * 255).astype(np.uint8), "RGB").save(path)
    print(f"wrote {path}")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# checkpoints (project 45 trains the shared base model; 48/49/50 load it)
# ---------------------------------------------------------------------------

def save_gen(net, name="base", base=32, where=CK):
    torch.save({"state": net.state_dict(), "base": base}, Path(where) / f"{name}.pt")
    print(f"saved {name}.pt")


def load_gen(name="base", where=CK):
    p = Path(where) / f"{name}.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run project 45's `train` stage first")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = VideoGen(base=ck["base"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net
