r"""Shared Phase-7 backbone: ONE vocabulary, ONE transformer, ONE loss.

This is the Chameleon recipe in miniature, and projects 34, 35 and 36 import it
so that nothing but the thing under test changes between them.

The whole idea fits in one picture:

    vocabulary = [ specials | text words | image codes | audio codes ]
                   0..5       6..~40       ~40..~550     ~550..~810

    a training example is one flat row of integers, e.g.

        <bos> a smiling young woman <boi> 391 12 508 ... 77 <eoi> <eos>
              \________ text _______/     \____ 64 image codes ___/

    the loss is next-token prediction over the whole row, exactly the loss a
    text-only language model uses. Nothing in the model knows which stretch is
    "the image".

What lives here
    `Vocab`            builds and owns the shared integer alphabet
    `build_sequences`  turns (caption, image-token-row) pairs into padded rows
    `UnifiedLM`        a small causal transformer (SDPA attention, pre-norm)
    `train_lm`         one training loop with per-modality loss tracking
    `sample`           autoregressive generation with top-k / temperature
    `AttrProbe`        a small CNN that reads attributes off a face, used to
                       grade generated images (projects 33 and 36)

Sizing on this CPU (6 threads): d=192, 4 layers, 4 heads, sequence 88,
batch 32 -> 1.9M parameters and ~113 ms per step, so 1,500 steps is about
3 minutes. Doubling to d=256/6 layers costs 2.4x the time for a model that is
still far too small to matter -- on a fixed CPU budget the extra steps are
worth more, which is the same trade project 10 measured for patch size.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, BOI, EOI, BOA, EOA = range(7)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<boi>", "<eoi>", "<boa>", "<eoa>"]
N_SPECIAL = len(SPECIALS)
TEXT_CTX = 20          # longest caption in words


# ---------------------------------------------------------------------------
# the shared vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    """One integer alphabet covering specials + words + image codes (+ audio).

    Keeping the three blocks inside a single `nn.Embedding` is the whole point
    of a unified model: the transformer sees integers, not modalities. The
    block boundaries are only used by *us*, to report a per-modality loss.
    """

    def __init__(self, words, n_image_codes, n_audio_codes=0):
        self.words = list(words)
        self.n_image = n_image_codes
        self.n_audio = n_audio_codes
        self.word_base = N_SPECIAL
        self.image_base = self.word_base + len(self.words)
        self.audio_base = self.image_base + n_image_codes
        self.size = self.audio_base + n_audio_codes
        self.w2i = {w: self.word_base + i for i, w in enumerate(self.words)}

    # --- encoding -----------------------------------------------------------
    def text_ids(self, caption):
        return [self.w2i[w] for w in caption.split() if w in self.w2i]

    def image_ids(self, codes):
        return (np.asarray(codes, dtype=np.int64) + self.image_base).tolist()

    def audio_ids(self, codes):
        return (np.asarray(codes, dtype=np.int64) + self.audio_base).tolist()

    # --- decoding -----------------------------------------------------------
    def decode_text(self, ids):
        out = []
        for i in ids:
            if self.word_base <= i < self.image_base:
                out.append(self.words[i - self.word_base])
        return " ".join(out)

    def decode_image(self, ids):
        return np.array([i - self.image_base for i in ids], dtype=np.int64)

    def kind(self, ids):
        """Per-token modality label: 0 special, 1 text, 2 image, 3 audio."""
        ids = np.asarray(ids)
        k = np.zeros_like(ids)
        k[(ids >= self.word_base) & (ids < self.image_base)] = 1
        k[(ids >= self.image_base) & (ids < self.audio_base)] = 2
        k[ids >= self.audio_base] = 3
        return k

    def to_json(self):
        return {"words": self.words, "n_image": self.n_image,
                "n_audio": self.n_audio, "size": self.size,
                "image_base": self.image_base, "audio_base": self.audio_base}


def build_vocab(captions, n_image_codes, n_audio_codes=0):
    words = sorted({w for c in captions for w in c.split()})
    return Vocab(words, n_image_codes, n_audio_codes)


# ---------------------------------------------------------------------------
# the (image tokens, caption) corpus, shared by projects 33, 34, 35 and 36
# ---------------------------------------------------------------------------
def _vq():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "32-discrete-image-tokens"))
    import vqvae
    return vqvae


def load_pairs():
    """Every CelebA face as 64 image codes, with its caption. Cached once.

    Returns a dict with train/val token rows, captions, raw images and the raw
    attribute labels (the last two are only needed by the referee probe).
    """
    VQ = _vq()
    import time as _t
    d = Path(__file__).resolve().parent / "data"
    d.mkdir(exist_ok=True)
    cache = d / "tokens.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tok = VQ.load_tokenizer()
    tr_i, tr_a, va_i, va_a = VQ.load_faces()
    t0 = _t.time()
    out = dict(tr_codes=VQ.tokenize_all(tok, tr_i), va_codes=VQ.tokenize_all(tok, va_i),
               tr_caps=np.array(VQ.all_captions(tr_a)),
               va_caps=np.array(VQ.all_captions(va_a)),
               tr_imgs=tr_i, tr_attrs=tr_a, va_imgs=va_i, va_attrs=va_a)
    print(f"  tokenized {len(tr_i) + len(va_i)} faces in {_t.time() - t0:.0f}s "
          f"-> {out['tr_codes'].shape[1]} codes each", flush=True)
    np.savez_compressed(cache, **out)
    return out


def pair_vocab(pairs, n_audio_codes=0):
    VQ = _vq()
    return build_vocab(list(pairs["tr_caps"]) + list(pairs["va_caps"]),
                       VQ.CODEBOOK, n_audio_codes)


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------
def pair_sequence(vocab, caption, img_codes, order="t2i"):
    """One interleaved row. `order` decides which modality comes first.

    t2i: <bos> text <boi> image <eoi> <eos>     -- learn to draw from words
    i2t: <bos> <boi> image <eoi> text <eos>     -- learn to describe a picture

    Training on both orders in the same model is what makes it any-to-any:
    the generation direction is chosen at sampling time by what you put in the
    prompt, not by which model you load.
    """
    t = vocab.text_ids(caption)[:TEXT_CTX]
    im = vocab.image_ids(img_codes)
    if order == "t2i":
        return [BOS] + t + [BOI] + im + [EOI, EOS]
    return [BOS, BOI] + im + [EOI] + t + [EOS]


def pad_batch(rows, length=None, pad=PAD):
    length = length or max(len(r) for r in rows)
    out = np.full((len(rows), length), pad, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r[:length]
    return out


def build_sequences(vocab, captions, img_codes, orders=("t2i", "i2t"), seed=0,
                    length=None):
    """One row per (caption, image), with the order drawn at random."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, ic in zip(captions, img_codes):
        o = orders[rng.integers(len(orders))] if len(orders) > 1 else orders[0]
        rows.append(pair_sequence(vocab, c, ic, o))
    return pad_batch(rows, length)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads, ffn=None, mlp_factory=None):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        ffn = ffn or 4 * d
        # mlp_factory lets project 35 swap in a Mixture-of-Experts layer
        self.mlp = (mlp_factory(d) if mlp_factory else
                    nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d)))

    def forward(self, x, aux=None):
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, t, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, t, self.heads, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(b, t, d))
        m = self.mlp(self.n2(x))
        if isinstance(m, tuple):            # MoE returns (output, router info)
            m, info = m
            if aux is not None:
                aux.append(info)
        return x + m


class UnifiedLM(nn.Module):
    """A plain decoder-only transformer. It has no idea what a picture is."""

    def __init__(self, vocab_size, d=256, layers=6, heads=4, ctx=96,
                 mlp_factory=None):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_factory=mlp_factory)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids, aux=None):
        """Everything except the final vocabulary projection.

        Project 36 needs this: it bolts a *second* output head onto the same
        hidden states, so the two heads must share the body exactly.
        """
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x, aux)
        return self.norm(x)

    def forward(self, ids, aux=None):
        return self.head(self.hidden(ids, aux))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def modality_losses(logits, targets, kinds, ignore=PAD):
    """Cross-entropy split by what kind of token was being predicted.

    This is the diagnostic the whole phase turns on: one number per modality,
    from one loss. `kinds` is the modality label of the *target*.
    """
    lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         targets.reshape(-1), reduction="none")
    valid = (targets.reshape(-1) != ignore)
    k = kinds.reshape(-1)
    out = {}
    for name, code in (("special", 0), ("text", 1), ("image", 2), ("audio", 3)):
        m = valid & (k == code)
        out[name] = float(lp[m].mean()) if m.any() else float("nan")
        out[name + "_tokens"] = int(m.sum())
    out["all"] = float(lp[valid].mean())
    return out


def train_lm(model, seqs, vocab, val_seqs=None, steps=2400, batch=32, lr=3e-3,
             seed=0, log_every=200, weights=None, aux_loss_w=0.0, verbose=True,
             sampler=None):
    """One training loop, shared by projects 33/34/35/36.

    `weights`  optional per-modality loss weights, e.g. {"image": 3.0} -- this
               is the knob project 34 turns.
    `sampler`  optional callable(rng, batch) -> row indices, so project 34 can
               change the data mixture instead of the loss.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    seqs_t = torch.from_numpy(seqs)
    kinds_all = torch.from_numpy(vocab.kind(seqs))
    hist, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        pick = sampler(rng, batch) if sampler else rng.integers(0, len(seqs), batch)
        ids = seqs_t[pick]
        x, y = ids[:, :-1], ids[:, 1:]
        kk = kinds_all[pick][:, 1:]
        aux = [] if aux_loss_w else None
        logits = model(x, aux)
        lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                             reduction="none")
        valid = (y.reshape(-1) != PAD).float()
        if weights:
            w = torch.ones_like(valid)
            for name, code in (("text", 1), ("image", 2), ("audio", 3)):
                if name in weights:
                    w = torch.where(kk.reshape(-1) == code,
                                    torch.full_like(w, weights[name]), w)
            loss = (lp * valid * w).sum() / (valid * w).sum().clamp_min(1)
        else:
            loss = (lp * valid).sum() / valid.sum().clamp_min(1)
        if aux:
            loss = loss + aux_loss_w * torch.stack([a["balance"] for a in aux]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "train_loss": float(loss.detach()),
                   "secs": time.time() - t0}
            if val_seqs is not None:
                rec.update({"val_" + k: v for k, v in
                            evaluate_lm(model, val_seqs, vocab).items()})
                model.train()
            hist.append(rec)
            if verbose:
                msg = f"  step {step:5d}  loss {float(loss.detach()):.3f}"
                if val_seqs is not None:
                    msg += (f"  val text {rec['val_text']:.3f}"
                            f"  val image {rec['val_image']:.3f}")
                print(msg + f"  {time.time() - t0:5.0f}s", flush=True)
    model.eval()
    return hist


@torch.no_grad()
def evaluate_lm(model, seqs, vocab, batch=64):
    """Per-modality validation loss over a whole set of rows."""
    model.eval()
    tot = {}
    seqs_t = torch.from_numpy(seqs)
    kinds = torch.from_numpy(vocab.kind(seqs))
    for i in range(0, len(seqs), batch):
        ids = seqs_t[i:i + batch]
        logits = model(ids[:, :-1])
        r = modality_losses(logits, ids[:, 1:], kinds[i:i + batch, 1:])
        for k, v in r.items():
            if k.endswith("_tokens"):
                tot[k] = tot.get(k, 0) + v
            elif not np.isnan(v):
                n = r[k + "_tokens"] if k + "_tokens" in r else 1
                tot.setdefault("_w" + k, 0.0)
                tot["_w" + k] += v * n
                tot.setdefault("_n" + k, 0)
                tot["_n" + k] += n
    out = {}
    for name in ("text", "image", "audio", "special"):
        n = tot.get("_n" + name, 0)
        out[name] = tot["_w" + name] / n if n else float("nan")
        out[name + "_tokens"] = tot.get(name + "_tokens", 0)
    return out


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, prompts, n_new, temperature=1.0, top_k=None, seed=0,
           allow=None, stop=None):
    """Continue every prompt for `n_new` tokens. `prompts` is a list of lists.

    `allow` optionally restricts the output to one block of the vocabulary
    (e.g. only image codes) -- the honest way to keep a half-trained model from
    wandering into the wrong modality mid-picture.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.from_numpy(pad_batch(prompts))
    for _ in range(n_new):
        logits = model(ids[:, -model.ctx:])[:, -1]
        if allow is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allow[0]:allow[1]] = 0
            logits = logits + mask
        logits = logits / max(temperature, 1e-6)
        if top_k:
            kth = logits.topk(top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
        if stop is not None and bool((nxt == stop).all()):
            break
    return ids


# ---------------------------------------------------------------------------
# the referee: a small attribute classifier used to grade generated faces
# ---------------------------------------------------------------------------
GRADED_ATTRS = ["Male", "Smiling", "Blond_Hair", "Eyeglasses", "Young",
                "Black_Hair", "Wearing_Hat", "Mouth_Slightly_Open"]


class AttrProbe(nn.Module):
    """Tiny CNN: 64x64 face -> one logit per graded attribute.

    Why a separate model at all -- doesn't the generator already know the
    attributes? It knows them only as *input words*. The question we want to
    answer is whether the picture it drew actually shows them, and a generator
    cannot be its own judge: asking it would just report what it intended. The
    probe is trained on REAL faces and never sees a generated one during
    training, so it is an independent referee.
    """

    def __init__(self, n_out=len(GRADED_ATTRS), w=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1), nn.SiLU(),          # 32
            nn.Conv2d(w, w * 2, 4, 2, 1), nn.SiLU(),      # 16
            nn.Conv2d(w * 2, w * 4, 4, 2, 1), nn.SiLU(),  # 8
            nn.Conv2d(w * 4, w * 4, 4, 2, 1), nn.SiLU(),  # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(w * 4, n_out))

    def forward(self, x):
        return self.net(x)


def probe_targets(attrs, names=GRADED_ATTRS):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "32-discrete-image-tokens"))
    import vqvae as VQ
    ix = [VQ.ATTRS.index(n) for n in names]
    return torch.from_numpy((attrs[:, ix] > 0).astype(np.float32))


def train_probe(train_u8, train_attrs, val_u8, val_attrs, steps=700, batch=64,
                lr=2e-3, seed=0, verbose=True):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "32-discrete-image-tokens"))
    import vqvae as VQ
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    probe = AttrProbe()
    y_tr, y_va = probe_targets(train_attrs), probe_targets(val_attrs)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    for step in range(1, steps + 1):
        p = rng.integers(0, len(train_u8), batch)
        x = VQ.to_tensor(train_u8[p])
        loss = F.binary_cross_entropy_with_logits(probe(x), y_tr[p])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    probe.eval()
    with torch.no_grad():
        acc = []
        for i in range(0, len(val_u8), 128):
            pr = probe(VQ.to_tensor(val_u8[i:i + 128])) > 0
            acc.append((pr == (y_va[i:i + 128] > 0.5)).float().mean(0))
        acc = torch.stack(acc).mean(0)
    if verbose:
        for n, a in zip(GRADED_ATTRS, acc.tolist()):
            print(f"    probe {n:22s} {a:.3f}")
    return probe, {n: float(a) for n, a in zip(GRADED_ATTRS, acc)}
