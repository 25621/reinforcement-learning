"""Shared Phase-3 backbone: a small CLIP you can train from scratch on a CPU.

Projects 10 (tiny CLIP), 12 (hard negatives), 13 (temperature) and 14 (data
filtering) all need the *same* three things:

  1. a few thousand real (image, caption) pairs,
  2. a small image tower + a small text tower, and
  3. one InfoNCE training loop.

They live here so the four projects cannot drift apart -- when project 12 says
"hard negatives beat random negatives", the only thing that changed between the
two runs is the batch sampler.

Everything is cached on disk under ``10-tiny-clip/data/``:

    coco_64.npz        5,000 COCO images, 72x72 uint8, plus their 5 captions each
    vocab.json         the word vocabulary built from those captions

Sizing notes (12 CPU threads):
    image tower  ViT, 64x64 image / 16px patches -> 17 tokens, d=128, 4 layers
    text tower   20 tokens, d=128, 2 layers
    ~1.5M parameters total, ~145 ms per step at batch 128 (1.1 ms per pair).

Why 16-pixel patches on a 64-pixel image (a 4x4 grid, which sounds far too
coarse): the same wall-clock buys 3x more optimizer steps than 8px patches, and
on a fixed CPU budget those extra steps are worth much more than the extra
detail. Measured here: 8px patches for 400 steps reach R@1 0.006, 16px patches
for 1,200 steps reach 0.040. This is Phase 2 project 08's finding again --
match the patch to the task, do not minimize it.
"""

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
STORE = 72          # images are cached slightly larger than they are used ...
IMG = 64            # ... so training can take random 64x64 crops as augmentation
PATCH = 16
CTX = 20            # max caption length in word tokens
N_IMAGES = 5000     # the whole COCO test split
N_TEST = 500        # held-out images, never trained on

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=clip-benchmark%2Fwds_mscoco_captions"
    "&config=default&split=test&offset={offset}&length={length}"
)


def data_dir():
    """The one cache directory, shared by projects 10/12/13/14."""
    return Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _square(img, size=STORE):
    """Shortest side -> size, then center-crop. Same recipe real CLIP uses."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _get(url, tries=10, base=3.0):
    """GET with exponential backoff -- both the listing API and the image CDN
    answer HTTP 429 ('slow down') if you ask too fast."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(min(base * 2 ** attempt, 60.0))
    raise RuntimeError


def _list_rows(n, verbose=True):
    """Page through the datasets-server listing (capped at 100 rows per call).

    The listing endpoint is rate-limited far more aggressively than the image
    CDN, so every page is appended to a cache file immediately: an interrupted
    run resumes where it stopped instead of starting the whole listing again.
    """
    cache = data_dir() / "rows.json"
    rows = json.loads(cache.read_text()) if cache.exists() else []
    if len(rows) >= n:
        return rows[:n]
    while len(rows) < n:
        batch = min(100, n - len(rows))
        payload = _get(_ROWS_URL.format(offset=len(rows), length=batch))
        rows += [{"src": r["row"]["jpg"]["src"], "txt": r["row"]["txt"]}
                 for r in json.loads(payload)["rows"]]
        cache.write_text(json.dumps(rows))
        if verbose and len(rows) % 500 == 0:
            print(f"    listed {len(rows)}/{n} rows", flush=True)
        time.sleep(1.0)
    return rows


def fetch_coco(n=N_IMAGES, workers=12, verbose=True):
    """Download n COCO (image, 5 captions) pairs into one npz. Idempotent."""
    out = data_dir() / "coco_64.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _list_rows(n, verbose)

    def grab(i_row):
        i, row = i_row
        arr = np.asarray(_square(Image.open(BytesIO(_get(row["src"])))), dtype=np.uint8)
        caps = [c.strip() for c in row["txt"].split("\n") if c.strip()]
        return i, arr, caps

    imgs = np.zeros((len(rows), STORE, STORE, 3), dtype=np.uint8)
    caps = [None] * len(rows)
    with ThreadPoolExecutor(workers) as pool:
        for done, (i, arr, c) in enumerate(pool.map(grab, enumerate(rows)), 1):
            imgs[i], caps[i] = arr, c
            if verbose and done % 500 == 0:
                print(f"    fetched {done}/{len(rows)} images", flush=True)

    # Five captions per image, padded to a rectangular array of strings so numpy
    # can store them; the pad is an empty string we never sample.
    width = max(len(c) for c in caps)
    caps = [c + [""] * (width - len(c)) for c in caps]
    np.savez_compressed(out, images=imgs, captions=np.array(caps, dtype=object))
    return out


def load_coco(n=N_IMAGES):
    """-> (images uint8 (N,72,72,3), captions list[list[str]])."""
    z = np.load(fetch_coco(n), allow_pickle=True)
    caps = [[s for s in row if s] for row in z["captions"]]
    return z["images"][:n], caps[:n]


# ---------------------------------------------------------------------------
# a word-level tokenizer (CLIP uses BPE; at this scale words are plenty)
# ---------------------------------------------------------------------------
PAD, UNK, SOT, EOT = 0, 1, 2, 3
_WORD = re.compile(r"[a-z]+")


def build_vocab(captions, min_count=5):
    """Words seen >= min_count times get an id; everything else becomes <unk>."""
    path = data_dir() / "vocab.json"
    if path.exists():
        return json.loads(path.read_text())
    counts = {}
    for caps in captions:
        for c in caps:
            for w in _WORD.findall(c.lower()):
                counts[w] = counts.get(w, 0) + 1
    words = sorted([w for w, c in counts.items() if c >= min_count])
    vocab = {"<pad>": PAD, "<unk>": UNK, "<sot>": SOT, "<eot>": EOT}
    for w in words:
        vocab[w] = len(vocab)
    path.write_text(json.dumps(vocab))
    return vocab


def tokenize(texts, vocab, ctx=CTX):
    """-> int64 (N, ctx). Every row is <sot> words... <eot> <pad>..."""
    out = np.zeros((len(texts), ctx), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [SOT] + [vocab.get(w, UNK) for w in _WORD.findall(t.lower())][: ctx - 2]
        ids.append(EOT)
        out[i, : len(ids)] = ids
    return out


def eot_positions(tokens):
    """Index of the <eot> token in each row -- the text tower reads its state there."""
    return (tokens == EOT).float().argmax(dim=-1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block with fused attention (5x faster than
    nn.MultiheadAttention on this CPU)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        shape = (B, T, self.h, D // self.h)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class ImageTower(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, out=128, patch=PATCH):
        super().__init__()
        n = (IMG // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, patch)     # split + project in one op
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, out, bias=False)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.proj(self.norm(x[:, 0]))


class TextTower(nn.Module):
    def __init__(self, vocab_size, d=128, layers=2, heads=4, out=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(1, CTX, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, out, bias=False)
        # Causal mask, exactly like real CLIP's text encoder: each word may only
        # look left, so the <eot> state is a summary of the whole caption.
        self.register_buffer("mask", torch.ones(CTX, CTX).tril().bool()[None, None])

    def forward(self, tok):
        x = self.emb(tok) + self.pos[:, : tok.shape[1]]
        m = self.mask[..., : tok.shape[1], : tok.shape[1]]
        for b in self.blocks:
            x = b(x, m)
        x = self.norm(x)
        eot = eot_positions(tok).long()
        return self.proj(x[torch.arange(len(tok)), eot])


class TinyCLIP(nn.Module):
    def __init__(self, vocab_size, d=128, out=128, img_layers=4, txt_layers=2,
                 temperature=0.07, learn_temperature=True, patch=PATCH):
        super().__init__()
        self.visual = ImageTower(d, img_layers, out=out, patch=patch)
        self.text = TextTower(vocab_size, d, txt_layers, out=out)
        # CLIP stores log(1/tau) rather than tau: it keeps the value unconstrained
        # for the optimizer (any real number is a valid log) and exponentiating
        # guarantees the scale stays positive.
        scale = torch.tensor(np.log(1.0 / temperature), dtype=torch.float32)
        self.logit_scale = nn.Parameter(scale, requires_grad=learn_temperature)

    def encode(self, pixels, tokens):
        v = F.normalize(self.visual(pixels), dim=-1)
        t = F.normalize(self.text(tokens), dim=-1)
        return v, t

    def forward(self, pixels, tokens):
        v, t = self.encode(pixels, tokens)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * v @ t.T


def clip_loss(logits):
    """Symmetric InfoNCE -- see project 09 for the derivation and gradient check."""
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_pixels(raw, rng=None):
    """uint8 (B,72,72,3) -> float (B,3,64,64). rng=None means center crop, no flip."""
    b = len(raw)
    if rng is None:
        off = np.full((b, 2), (STORE - IMG) // 2)
        flip = np.zeros(b, dtype=bool)
    else:
        off = rng.integers(0, STORE - IMG + 1, size=(b, 2))
        flip = rng.random(b) < 0.5
    crops = np.empty((b, IMG, IMG, 3), dtype=np.uint8)
    for i in range(b):
        y, x = off[i]
        c = raw[i, y:y + IMG, x:x + IMG]
        crops[i] = c[:, ::-1] if flip[i] else c
    arr = crops.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))


class Pairs:
    """The (image, caption) pool. One image has 5 captions; we resample which one
    is used every time the image appears, which is free extra text variety."""

    def __init__(self, images, captions, vocab, index):
        self.images, self.captions, self.vocab = images, captions, vocab
        self.index = np.asarray(index)

    def batch(self, ids, rng=None):
        ids = np.asarray(ids)
        raw = self.images[ids]
        if rng is None:
            texts = [self.captions[i][0] for i in ids]
        else:
            texts = [self.captions[i][rng.integers(len(self.captions[i]))] for i in ids]
        tok = torch.from_numpy(tokenize(texts, self.vocab))
        return to_pixels(raw, rng), tok


def splits(n=N_IMAGES, n_test=N_TEST, seed=0):
    """Deterministic train/test split over image ids."""
    order = np.random.default_rng(seed).permutation(n)
    return order[n_test:], order[:n_test]


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def cosine_lr(step, total, base, warmup=50):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + np.cos(np.pi * p))


def train(model, pool, steps, batch=128, lr=3e-4, seed=0, sampler=None,
          log_every=100, weight_decay=0.1, verbose=True):
    """One InfoNCE training run. `sampler(step, rng) -> image ids` overrides the
    default uniform-random batch (project 12 swaps in a hard-negative sampler)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.98), eps=1e-6)
    history, taus = [], []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = (sampler(step, rng) if sampler is not None
               else rng.choice(pool.index, size=batch, replace=False))
        px, tok = pool.batch(ids, rng)
        loss = clip_loss(model(px, tok))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():                        # CLIP clamps tau >= 0.01
            model.logit_scale.clamp_(max=np.log(100.0))
        history.append(float(loss.detach()))
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:5d}  loss {float(loss):.4f}"
                  f"  tau {1 / model.logit_scale.exp().item():.4f}", flush=True)
    return history


@torch.no_grad()
def embed_all(model, pool, ids, caption_index=0, batch=250):
    """Encode a whole split once, center-cropped and deterministic."""
    model.eval()
    vs, ts = [], []
    ids = np.asarray(ids)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        px = to_pixels(pool.images[chunk])
        texts = [pool.captions[j][caption_index % len(pool.captions[j])] for j in chunk]
        tok = torch.from_numpy(tokenize(texts, pool.vocab))
        v, t = model.encode(px, tok)
        vs.append(v.numpy())
        ts.append(t.numpy())
    return np.concatenate(vs), np.concatenate(ts)


def recall_at_k(sim, ks=(1, 5, 10)):
    """sim[i, j] scores query i against gallery j; the right answer is j == i."""
    order = np.argsort(-sim, axis=1)
    rank = np.argmax(order == np.arange(len(sim))[:, None], axis=1)
    return {k: float((rank < k).mean()) for k in ks}


def evaluate(model, pool, test_ids):
    """Retrieval both ways over the held-out gallery, plus geometry stats."""
    v, t = embed_all(model, pool, test_ids)
    sim = v @ t.T
    i2t = recall_at_k(sim)
    t2i = recall_at_k(sim.T)
    matched = float(np.mean(np.diag(sim)))
    off = sim[~np.eye(len(sim), dtype=bool)]
    return {
        "i2t_r1": i2t[1], "i2t_r5": i2t[5], "i2t_r10": i2t[10],
        "t2i_r1": t2i[1], "t2i_r5": t2i[5], "t2i_r10": t2i[10],
        "matched_cos": matched,
        "mismatched_cos": float(off.mean()),
        "alignment": float(np.mean(np.sum((v - t) ** 2, axis=1))),
        "uniformity_img": _uniformity(v),
        "uniformity_txt": _uniformity(t),
        "gap": float(np.linalg.norm(v.mean(0) - t.mean(0))),
    }


def _uniformity(x, t=2.0):
    """Wang & Isola's uniformity: log of the average exp(-t * squared distance).
    More negative = points are spread more evenly over the sphere."""
    d = np.sum((x[:, None] - x[None]) ** 2, axis=-1)
    iu = np.triu_indices(len(x), 1)
    return float(np.log(np.mean(np.exp(-t * d[iu]))))


def make_pool(vocab=None, n=N_IMAGES):
    """Load everything and return (pool_factory, train_ids, test_ids, vocab)."""
    images, captions = load_coco(n)
    vocab = vocab or build_vocab(captions)
    tr, te = splits(len(images))
    return images, captions, vocab, tr, te


def setup(threads=12):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
