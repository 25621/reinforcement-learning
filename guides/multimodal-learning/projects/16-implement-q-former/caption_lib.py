"""Shared Phase-4 captioning stack: frozen CLIP features + a small caption LM.

Projects 16 (Q-Former) and 19 (gated cross-attention) both need the same three
things, so they live here and project 19 imports this file:

  1. real COCO images encoded ONCE by a frozen CLIP ViT-B/32 into 50 patch
     tokens of 768 numbers each (49 patches on a 7x7 grid, plus CLS),
  2. the matching captions and a small word-level vocabulary,
  3. a small causal Transformer that writes captions.

Why cache the CLIP features instead of running CLIP inside the training loop:
CLIP is frozen in both projects, so its output for a given image never changes.
Running it once costs ~24 ms per image; running it every step would cost that
again for every epoch. Caching turns a 4-minute-per-run cost into a one-off.

Why a *frozen real* CLIP rather than a from-scratch encoder: the point of both
projects is the piece we bolt on (queries, gated layers). Borrowing an encoder
that already sees the world properly means a bad caption is the fusion module's
fault, not the encoder's -- which is exactly how BLIP-2 and Flamingo were built.

Cache layout, all under ``16-implement-q-former/data/``:

    rows.json      the COCO listing (image URL + 5 captions per row)
    clip_feats.npy (N, 50, 768) float16   frozen CLIP patch tokens
    thumbs.npy     (N, 64, 64, 3) uint8   small copies, only for README figures
    captions.json  the 5 captions per image
    vocab.json     word -> id
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

N_IMAGES = 3000
N_VAL = 400
CTX = 24                 # caption length in word tokens, including <sot>/<eot>
CLIP_TOKENS = 50         # 1 CLS + 7x7 patches
CLIP_DIM = 768

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=clip-benchmark%2Fwds_mscoco_captions"
    "&config=default&split=test&offset={offset}&length={length}"
)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def data_dir():
    return Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# download + frozen-CLIP encoding
# ---------------------------------------------------------------------------
def _get(url, tries=10, base=3.0):
    """GET with exponential backoff. The Hugging Face listing endpoint answers
    HTTP 429 ('slow down') after ~25 quick calls, far sooner than the image
    CDN does, so every retry waits twice as long as the last."""
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
    """Page through the listing, 100 rows at a time, caching after every page so
    an interrupted run resumes instead of restarting."""
    cache = data_dir() / "rows.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(cache.read_text()) if cache.exists() else []
    while len(rows) < n:
        batch = min(100, n - len(rows))
        payload = _get(_ROWS_URL.format(offset=len(rows), length=batch))
        rows += [{"src": r["row"]["jpg"]["src"], "txt": r["row"]["txt"]}
                 for r in json.loads(payload)["rows"]]
        cache.write_text(json.dumps(rows))
        if verbose and len(rows) % 500 == 0:
            print(f"    listed {len(rows)}/{n} rows", flush=True)
        time.sleep(1.0)
    return rows[:n]


def _square(img, size):
    """Shortest side -> size, then centre-crop. CLIP's own preprocessing."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def build_cache(n=N_IMAGES, workers=12, chunk=250, verbose=True):
    """Download n COCO images, run frozen CLIP over them, keep only the features.

    Images are processed in chunks so we never hold 3,000 full-size JPEGs in
    memory: fetch 250, encode 250, throw the pixels away, repeat.
    """
    feats_path = data_dir() / "clip_feats.npy"
    if feats_path.exists():
        return
    rows = _list_rows(n, verbose)

    from transformers import CLIPModel
    torch.set_num_threads(12)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.eval()
    for p in clip.parameters():
        p.requires_grad_(False)

    feats = np.zeros((n, CLIP_TOKENS, CLIP_DIM), dtype=np.float16)
    thumbs = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    captions = [None] * n

    def grab(i):
        img = Image.open(BytesIO(_get(rows[i]["src"])))
        big = np.asarray(_square(img, 224), dtype=np.uint8)
        small = np.asarray(_square(img, 64), dtype=np.uint8)
        caps = [c.strip() for c in rows[i]["txt"].split("\n") if c.strip()]
        return i, big, small, caps

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, n, chunk):
            idx = list(range(start, min(start + chunk, n)))
            buf = np.zeros((len(idx), 224, 224, 3), dtype=np.uint8)
            for i, big, small, caps in pool.map(grab, idx):
                buf[i - start] = big
                thumbs[i] = small
                captions[i] = caps
            x = buf.astype(np.float32) / 255.0
            x = (x - _MEAN) / _STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
            with torch.no_grad():
                for j in range(0, len(idx), 64):
                    out = clip(pixel_values=x[j:j + 64]).last_hidden_state
                    feats[start + j:start + j + len(out)] = out.numpy().astype(np.float16)
            if verbose:
                print(f"    encoded {min(start + chunk, n)}/{n} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    np.save(feats_path, feats)
    np.save(data_dir() / "thumbs.npy", thumbs)
    (data_dir() / "captions.json").write_text(json.dumps(captions))


# ---------------------------------------------------------------------------
# word-level tokenizer (real CLIP uses BPE; at 3,000 captions words are plenty)
# ---------------------------------------------------------------------------
PAD, UNK, SOT, EOT = 0, 1, 2, 3
_WORD = re.compile(r"[a-z]+")


def build_vocab(captions, min_count=5):
    path = data_dir() / "vocab.json"
    if path.exists():
        return json.loads(path.read_text())
    counts = {}
    for caps in captions:
        for c in caps:
            for w in _WORD.findall(c.lower()):
                counts[w] = counts.get(w, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK, "<sot>": SOT, "<eot>": EOT}
    for w in sorted(w for w, c in counts.items() if c >= min_count):
        vocab[w] = len(vocab)
    path.write_text(json.dumps(vocab))
    return vocab


def tokenize(texts, vocab, ctx=CTX):
    out = np.zeros((len(texts), ctx), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [SOT] + [vocab.get(w, UNK) for w in _WORD.findall(t.lower())][: ctx - 2]
        ids.append(EOT)
        out[i, : len(ids)] = ids
    return out


def detokenize(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    words = []
    for i in ids:
        if i == EOT:
            break
        if i not in (PAD, SOT):
            words.append(inv.get(int(i), "<unk>"))
    return " ".join(words)


# ---------------------------------------------------------------------------
# the data pool
# ---------------------------------------------------------------------------
class CocoFeats:
    """Frozen CLIP features + tokenized captions, split into train/val."""

    def __init__(self, n=N_IMAGES, n_val=N_VAL, seed=0):
        build_cache(n)
        self.feats = np.load(data_dir() / "clip_feats.npy", mmap_mode="r")[:n]
        self.thumbs = np.load(data_dir() / "thumbs.npy", mmap_mode="r")[:n]
        self.captions = json.loads((data_dir() / "captions.json").read_text())[:n]
        self.vocab = build_vocab(self.captions)
        # tokenize all 5 captions of every image once; picking a random one per
        # step is free extra text variety, exactly like project 10 did
        self.tokens = np.stack([tokenize(c[:5] + c[:1] * (5 - len(c[:5])), self.vocab)
                                for c in self.captions])
        order = np.random.default_rng(seed).permutation(n)
        self.val_ids, self.train_ids = order[:n_val], order[n_val:]

    def batch(self, ids, rng=None, caption=0):
        ids = np.asarray(ids)
        f = torch.from_numpy(np.asarray(self.feats[ids], dtype=np.float32))
        pick = rng.integers(0, 5, len(ids)) if rng is not None else np.full(len(ids), caption)
        tok = torch.from_numpy(self.tokens[ids, pick])
        return f, tok


# ---------------------------------------------------------------------------
# the caption language model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm decoder block. `xattn_ctx` is optional: project 16 passes image
    tokens as a *prefix* instead, project 19 passes them here."""

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


class CaptionLM(nn.Module):
    """A small causal Transformer over caption words.

    `n_prefix` reserves that many leading positions for image tokens supplied
    by the caller. With n_prefix=0 this is an ordinary text-only language model
    -- which is exactly what project 19 freezes.
    """

    def __init__(self, vocab_size, d=256, layers=4, heads=4, n_prefix=0, ctx=CTX):
        super().__init__()
        self.d, self.n_prefix, self.ctx = d, n_prefix, ctx
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(1, n_prefix + ctx, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.emb.weight               # weight tying
        # PyTorch initialises nn.Embedding from N(0, 1). Harmless for an input
        # table, disastrous once it is *tied* to the output layer: the logits
        # then start with a standard deviation of sqrt(d) ~ 16, so the loss at
        # step 0 is in the hundreds instead of ln(vocab) ~ 7.7. GPT-style 0.02
        # is what every real implementation uses, for exactly this reason.
        nn.init.normal_(self.emb.weight, std=0.02)

    def causal_mask(self, total, n_prefix, device):
        """Words may look left; every word may look at all prefix tokens. The
        prefix positions attend among themselves only (they are not text)."""
        m = torch.ones(total, total, dtype=torch.bool, device=device).tril()
        if n_prefix:
            m[:n_prefix, :n_prefix] = True
        return m[None, None]

    def forward(self, tok, prefix=None, hooks=None):
        """hooks: optional list of callables, one per block, applied to the
        hidden states BEFORE that block. Project 19 injects gated
        cross-attention through this without touching the frozen weights."""
        x = self.emb(tok)
        n_prefix = 0
        if prefix is not None:
            n_prefix = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        mask = self.causal_mask(x.shape[1], n_prefix, x.device)
        for i, blk in enumerate(self.blocks):
            if hooks is not None:
                x = hooks[i](x)
            x = blk(x, mask)
        return self.head(self.norm(x)), n_prefix


def caption_loss(logits, tok, n_prefix):
    """Next-word cross-entropy on the caption only.

    The prefix positions predict nothing we care about, and padding after <eot>
    is not language, so both are dropped. Predicting them would spend capacity
    on a target the model cannot and should not learn.
    """
    logits = logits[:, n_prefix:]                    # align to caption positions
    pred = logits[:, :-1]
    target = tok[:, 1:]
    keep = target != PAD
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1])[keep.reshape(-1)],
                           target.reshape(-1)[keep.reshape(-1)])


def cosine_lr(step, total, base, warmup=100):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + np.cos(np.pi * p))
