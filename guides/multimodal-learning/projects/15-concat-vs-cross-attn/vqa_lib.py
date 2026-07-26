"""Shared Phase-4 toy world: a tiny Visual Question Answering (VQA) task.

Projects 15 (fusion comparison) and 18 (Perceiver IO) both need the *same*
question-answering task, so it lives here and project 18 imports it. When 18
says "Perceiver IO answers five questions in one forward pass", the task it is
answering is bit-for-bit the task project 15 used.

The world
---------
A 64x64 image on a black background holds 5 objects. Each object has

  * one of 3 shapes  (square, circle, triangle)
  * one of 6 colours (red, green, blue, yellow, cyan, magenta) -- all distinct
    inside one image, so "the red object" always names exactly one thing
  * a position on a 4x4 grid of cells, jittered by a few pixels

Every image comes with 5 questions, one of each type:

  0  what shape is the <c> object                        -> square/circle/triangle
  1  is the <c> object in the top or bottom half         -> top/bottom
  2  what shape is the object nearest the <c> object     -> square/circle/triangle
  3  what colour is the object nearest the <c> object    -> one of 6 colours
  4  how many objects have the same shape as <c>         -> one..five

Types 0 and 1 are *non-relational*: look at one object, read off a property.
Types 2, 3 and 4 are *relational*: you must find the anchor object first, then
compare it with the others. That split is the whole point of the task -- a
fusion method that only ever sees one summary vector of the image can memorise
"the picture as a whole", but it cannot look up "the object next to THIS one",
because which object matters is decided by the question.

Why synthetic instead of a real VQA dataset: every answer here is generated
from the scene, so there is no annotator noise, no language prior to exploit
("what colour is the banana?" -> "yellow" without looking), and we can ask for
exactly the skill we want to measure. Real VQA benchmarks are full of questions
answerable from the text alone, which would hide the differences we are testing.

The frozen vision encoder
-------------------------
Real VLMs do not train their image encoder from scratch alongside the fusion
module -- they borrow a pretrained one and freeze it. We do the same, in
miniature: `pretrain_vision` trains a small ViT ONCE on a pretext task (name
the colour and shape sitting in each patch), then every fusion variant reads
the same frozen patch tokens. Two reasons this matters here:

  * fairness -- any difference between the fusion variants must come from the
    fusion, because the pixels reached all of them through identical weights;
  * cost -- the encoder runs once over the whole dataset and the result is
    cached, so a fusion experiment is minutes instead of tens of minutes.

The pretext task deliberately teaches only *local* facts (what is in this
patch). Relations -- which object is nearest, how many share a shape -- are
never taught to the encoder, so they remain the fusion module's job.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
IMG = 64
N_OBJ = 5
GRID = 4                       # objects sit on a 4x4 grid of cells
CELL = IMG // GRID             # 16 px per cell
RADIUS = 5                     # objects are ~11 px across
PATCH = CELL                   # one patch per cell, so a patch holds one object

COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta"]
RGB = np.array([
    [230, 40, 40], [40, 200, 60], [50, 90, 240],
    [240, 220, 40], [40, 220, 220], [220, 60, 220],
], dtype=np.uint8)
SHAPES = ["square", "circle", "triangle"]

# One flat answer vocabulary shared by all five question types, so every model
# is a single 16-way classifier and the accuracies are directly comparable.
ANSWERS = SHAPES + ["top", "bottom"] + COLORS + ["one", "two", "three", "four", "five"]
A2I = {a: i for i, a in enumerate(ANSWERS)}

QUESTION_TEMPLATES = [
    "what shape is the {c} object",
    "is the {c} object in the top or bottom half",
    "what shape is the object nearest the {c} object",
    "what color is the object nearest the {c} object",
    "how many objects have the same shape as the {c} object",
]
QUESTION_KIND = ["property", "position", "relational", "relational", "counting"]
N_QUESTIONS = len(QUESTION_TEMPLATES)

_WORDS = sorted({w for t in QUESTION_TEMPLATES for w in t.replace("{c}", "").split()}
                | set(COLORS))
PAD = 0
VOCAB = {"<pad>": PAD, **{w: i + 1 for i, w in enumerate(_WORDS)}}
Q_LEN = max(len(t.format(c="red").split()) for t in QUESTION_TEMPLATES)


def _draw(canvas, shape, cy, cx, color):
    """Paint one object onto a (IMG, IMG, 3) uint8 canvas."""
    y0, y1 = cy - RADIUS, cy + RADIUS + 1
    x0, x1 = cx - RADIUS, cx + RADIUS + 1
    yy, xx = np.mgrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1]
    if shape == 0:                                   # square
        mask = np.ones_like(yy, dtype=bool)
    elif shape == 1:                                 # circle
        mask = yy ** 2 + xx ** 2 <= RADIUS ** 2
    else:                                            # triangle, apex at the top
        mask = 2 * np.abs(xx) <= yy + RADIUS
    canvas[y0:y1, x0:x1][mask] = color


def make_scenes(n, seed=0):
    """Render n scenes. Returns images plus the per-object facts we need later.

    images  uint8 (n, 64, 64, 3)
    shapes  int64 (n, 5)     shape id of each object
    colors  int64 (n, 5)     colour id of each object (distinct within a scene)
    cells   int64 (n, 5)     which of the 16 grid cells each object occupies
    yx      int64 (n, 5, 2)  object centres in pixels
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    shapes = rng.integers(0, len(SHAPES), size=(n, N_OBJ))
    colors = np.stack([rng.permutation(len(COLORS))[:N_OBJ] for _ in range(n)])
    cells = np.stack([rng.permutation(GRID * GRID)[:N_OBJ] for _ in range(n)])
    jitter = rng.integers(-2, 3, size=(n, N_OBJ, 2))
    cy = (cells // GRID) * CELL + CELL // 2 + jitter[..., 0]
    cx = (cells % GRID) * CELL + CELL // 2 + jitter[..., 1]
    lo, hi = RADIUS, IMG - RADIUS - 1
    cy, cx = np.clip(cy, lo, hi), np.clip(cx, lo, hi)
    for i in range(n):
        for o in range(N_OBJ):
            _draw(images[i], shapes[i, o], cy[i, o], cx[i, o], RGB[colors[i, o]])
    return images, shapes, colors, cells, np.stack([cy, cx], axis=-1)


def make_questions(shapes, colors, yx):
    """Every scene gets all 5 question types, asked about a random present object.

    Returns tokens int64 (n, 5, Q_LEN) and answers int64 (n, 5).
    """
    n = len(shapes)
    rng = np.random.default_rng(12345)
    anchor = rng.integers(0, N_OBJ, size=(n, N_QUESTIONS))
    tokens = np.zeros((n, N_QUESTIONS, Q_LEN), dtype=np.int64)
    answers = np.zeros((n, N_QUESTIONS), dtype=np.int64)

    # pairwise distances between the 5 objects, with self-distance set to +inf
    d = np.linalg.norm((yx[:, :, None, :] - yx[:, None, :, :]).astype(np.float64),
                       axis=-1)
    d[:, np.arange(N_OBJ), np.arange(N_OBJ)] = np.inf
    nearest = d.argmin(axis=2)                                   # (n, 5)
    same_shape = (shapes[:, :, None] == shapes[:, None, :]).sum(axis=2)   # (n, 5)
    rows = np.arange(n)

    for q, template in enumerate(QUESTION_TEMPLATES):
        a = anchor[:, q]
        cid = colors[rows, a]
        for i in range(n):
            words = template.format(c=COLORS[cid[i]]).split()
            tokens[i, q, :len(words)] = [VOCAB[w] for w in words]
        if q == 0:
            ans = shapes[rows, a]
        elif q == 1:
            ans = np.where(yx[rows, a, 0] < IMG // 2, A2I["top"], A2I["bottom"])
        elif q == 2:
            ans = shapes[rows, nearest[rows, a]]
        elif q == 3:
            ans = A2I["red"] + colors[rows, nearest[rows, a]]
        else:
            ans = A2I["one"] + same_shape[rows, a] - 1
        answers[:, q] = ans
    return tokens, answers


def patch_labels(colors, cells, shapes):
    """Pretext targets: what colour and what shape sits in each of the 16 cells.

    Label 0 means "this cell is empty", so colours are 1..6 and shapes are 1..3.
    """
    n = len(colors)
    col = np.zeros((n, GRID * GRID), dtype=np.int64)
    shp = np.zeros((n, GRID * GRID), dtype=np.int64)
    rows = np.repeat(np.arange(n), N_OBJ)
    col[rows, cells.reshape(-1)] = colors.reshape(-1) + 1
    shp[rows, cells.reshape(-1)] = shapes.reshape(-1) + 1
    return col, shp


def pixels(raw):
    """uint8 (B,64,64,3) -> float (B,3,64,64) in roughly [-1, 1]."""
    arr = raw.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))


# ---------------------------------------------------------------------------
# building blocks (shared by every fusion variant so the comparison is fair)
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Pre-norm self-attention. We hand-roll QKV and call the fused kernel:
    nn.MultiheadAttention is about 5x slower on this CPU."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).split(D, dim=-1)
        shape = (B, T, self.h, D // self.h)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.proj(a.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Queries come from one stream, keys/values from the other.

    `gate=True` adds Flamingo's trick: the branch is multiplied by tanh of a
    parameter initialised at 0, so at step 0 this layer outputs exactly its
    input. Project 19 is built entirely around that property.
    """

    def __init__(self, d, heads, gate=False):
        super().__init__()
        self.h = heads
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.gate = nn.Parameter(torch.zeros(1)) if gate else None

    def forward(self, x, ctx):
        B, T, D = x.shape
        N = ctx.shape[1]
        q = self.q(self.nq(x)).view(B, T, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(self.nkv(ctx)).split(D, dim=-1)
        k, v = (t.view(B, N, self.h, D // self.h).transpose(1, 2) for t in (k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        out = self.proj(a.transpose(1, 2).reshape(B, T, D))
        if self.gate is not None:
            out = out * torch.tanh(self.gate)
        return x + out


class MLP(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                 nn.Linear(mult * d, d))

    def forward(self, x):
        return x + self.net(self.norm(x))


class AttentionPool(nn.Module):
    """Squeeze a sequence of tokens into ONE vector with a learned query.

    Late fusion needs a single image vector. Averaging the patches would be the
    lazy choice and would blur the objects together; this instead lets the model
    learn *what to look at* when it compresses. It is the strongest single
    vector a trainable pooler can produce from these patches -- which is what
    makes the concat baseline in project 15 a fair opponent rather than a
    strawman.
    """

    def __init__(self, d, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = CrossAttention(d, heads)
        self.mlp = MLP(d)

    def forward(self, tokens):
        q = self.query.expand(len(tokens), -1, -1)
        return self.mlp(self.attn(q, tokens))[:, 0]


class ImageTower(nn.Module):
    """A small ViT. Returns the patch-token sequence (one token per grid cell)."""

    def __init__(self, d=128, layers=3, heads=4, patch=PATCH):
        super().__init__()
        n = (IMG // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [SelfAttention(d, heads), MLP(d)]
        self.norm = nn.LayerNorm(d)

    def forward(self, px):
        x = self.patch(px).flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TextTower(nn.Module):
    """A small bidirectional text encoder over the question words."""

    def __init__(self, d=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, Q_LEN, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [SelfAttention(d, heads), MLP(d)]
        self.norm = nn.LayerNorm(d)

    def forward(self, tok):
        x = self.emb(tok) + self.pos[:, : tok.shape[1]]
        pad = (tok == PAD)[:, None, None, :]
        mask = torch.zeros_like(pad, dtype=x.dtype).masked_fill(pad, float("-inf"))
        for layer in self.layers:
            x = layer(x, mask) if isinstance(layer, SelfAttention) else layer(x)
        x = self.norm(x)
        keep = (~pad[:, 0, 0])[..., None].to(x.dtype)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return x, pooled                   # word tokens, mean-pooled vector


# ---------------------------------------------------------------------------
# stage 1: pretrain the vision encoder, then freeze it
# ---------------------------------------------------------------------------
class PretextVision(nn.Module):
    """Image tower + per-patch heads.

    `teach_position` adds a third head that asks each patch token to name its
    own cell. That sounds pointless -- the tower already adds a positional
    embedding at its input, so the information is *available* -- but available
    is not the same as preserved. Nothing in a colour-and-shape objective needs
    position, so the tower is free to discard it on the way out, and it does.
    Project 15 measures the consequence: without this head, every question that
    depends on *where* something is collapses to chance for every fusion
    method. A frozen encoder can only pass on what its pretraining forced it to
    keep.
    """

    def __init__(self, d=128, layers=3, teach_position=True):
        super().__init__()
        self.tower = ImageTower(d, layers)
        self.color_head = nn.Linear(d, len(COLORS) + 1)
        self.shape_head = nn.Linear(d, len(SHAPES) + 1)
        self.pos_head = nn.Linear(d, GRID * GRID) if teach_position else None

    def forward(self, px):
        h = self.tower(px)
        pos = self.pos_head(h) if self.pos_head is not None else None
        return self.color_head(h), self.shape_head(h), pos


def pretrain_vision(images, colors, cells, shapes, steps=700, batch=128, lr=1e-3,
                    seed=0, teach_position=True, verbose=True):
    """Teach the encoder local facts only. Returns (model, accuracy dict)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    col, shp = patch_labels(colors, cells, shapes)
    cell_id = torch.arange(GRID * GRID)
    model = PretextVision(teach_position=teach_position)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    n_train = len(images) - 500
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = rng.integers(0, n_train, size=batch)
        pc, ps, pp = model(pixels(images[ids]))
        loss = (F.cross_entropy(pc.reshape(-1, pc.shape[-1]),
                                torch.from_numpy(col[ids]).reshape(-1))
                + F.cross_entropy(ps.reshape(-1, ps.shape[-1]),
                                  torch.from_numpy(shp[ids]).reshape(-1)))
        if pp is not None:
            loss = loss + F.cross_entropy(
                pp.reshape(-1, pp.shape[-1]),
                cell_id.repeat(len(ids)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"    pretext step {step:4d}  loss {float(loss.detach()):.4f}",
                  flush=True)
    model.eval()
    with torch.no_grad():
        ids = np.arange(n_train, len(images))
        pc, ps, pp = model(pixels(images[ids]))
        occupied = torch.from_numpy(col[ids]) > 0
        acc = dict(
            color=float((pc.argmax(-1) == torch.from_numpy(col[ids]))[occupied]
                        .float().mean()),
            shape=float((ps.argmax(-1) == torch.from_numpy(shp[ids]))[occupied]
                        .float().mean()),
            position=(float((pp.argmax(-1) == cell_id).float().mean())
                      if pp is not None else None))
    return model, acc


@torch.no_grad()
def cache_features(tower, images, batch=256):
    """Run the frozen encoder over every scene once and keep the patch tokens."""
    tower.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(tower(pixels(images[i:i + batch])).numpy().astype(np.float16))
    return np.concatenate(out)


class FeatureVQA:
    """Cached patch tokens + flat (question, answer) pairs."""

    def __init__(self, feats, tokens, answers):
        self.feats = feats
        n, q = answers.shape
        self.img_id = np.repeat(np.arange(n), q)
        self.tokens = tokens.reshape(n * q, Q_LEN)
        self.answers = answers.reshape(-1)
        self.kinds = np.tile(np.arange(q), (n, 1)).reshape(-1)

    def __len__(self):
        return len(self.answers)

    def batch(self, ids):
        f = torch.from_numpy(self.feats[self.img_id[ids]].astype(np.float32))
        return f, torch.from_numpy(self.tokens[ids]), torch.from_numpy(self.answers[ids])


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------
def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def cosine_lr(step, total, base, warmup=100):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + np.cos(np.pi * p))


def train(model, data, steps, batch=128, lr=1e-3, seed=0, log_every=200,
          verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05,
                            betas=(0.9, 0.98), eps=1e-6)
    history = []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = rng.integers(0, len(data), size=batch)
        feats, tok, ans = data.batch(ids)
        loss = F.cross_entropy(model(feats, tok), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:5d}  loss {np.mean(history[-50:]):.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate(model, data, batch=500):
    """Overall accuracy plus accuracy per question type."""
    model.eval()
    correct = np.zeros(len(data), dtype=bool)
    for i in range(0, len(data), batch):
        ids = np.arange(i, min(i + batch, len(data)))
        feats, tok, ans = data.batch(ids)
        correct[ids] = (model(feats, tok).argmax(-1) == ans).numpy()
    per_kind = {q: float(correct[data.kinds == q].mean()) for q in range(N_QUESTIONS)}
    return float(correct.mean()), per_kind
