"""Shared Phase-7 image tokenizer: a small VQ-VAE trained on CelebA faces.

Every Phase-7 project needs the same thing -- a way to turn a picture into a
short row of whole numbers that a language model can read -- so it lives here
and projects 33, 34, 35 and 36 import this file.

What this module provides:

  1. `fetch_celeba()`  -- N real CelebA faces, center-cropped to 64x64, cached
     as one npz, together with their 40 binary attributes.
  2. `attr_caption()`  -- turns those attributes into a short English caption
     ("a smiling young woman with blond hair"), so every image comes with real
     paired text. No captioning model is involved; the labels are human-made.
  3. `VQVAE`           -- encoder / codebook / decoder, straight-through
     gradients, optional EMA codebook and dead-code restarts.
  4. `train_vqvae()`   -- one training loop, used for every configuration.
  5. `load_tokenizer()`-- the frozen f=8 tokenizer the downstream projects use,
     plus `encode_indices()` / `decode_indices()`.

Sizing notes (this CPU, 6 torch threads), batch 32:
    f=8 (64 tokens/image)    80 ms per step   -> 2,500 steps in 199 s
    f=4 (256 tokens/image)  103 ms per step
    f=2 (1024 tokens/image) 170 ms per step -- its feature maps stay large all
                                               the way through, so every layer
                                               costs more even though the model
                                               has FEWER parameters.

Why faces and not COCO: a tiny autoregressive model downstream has to *predict*
these tokens. Faces are aligned and low-variety, so 64 tokens per image really
can carry a recognisable face; COCO at the same budget carries a colour smear.
The tokenizer code is identical either way -- the dataset is chosen so the
downstream projects have something visible to show.
"""

import json
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
IMG = 64                 # every face is stored and used at 64x64
N_IMAGES = 8000
N_VAL = 800
CODEBOOK = 512           # entries in the shared palette
LATENT_DIM = 32          # width of one code vector (deliberately small, see below)
WIDTH = 32               # channel count after the first stride-2 stage
DOWN = 8                 # the shared tokenizer's compression factor -> 8x8 = 64 tokens
THREADS = 12

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=tpremoli%2FCelebA-attrs"
    "&config=default&split=train&offset={offset}&length={length}"
)

# the 40 CelebA attribute names, in dataset order
ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young",
]
_IX = {a: i for i, a in enumerate(ATTRS)}


def data_dir():
    """The one cache directory, shared by projects 32/33/34/35/36."""
    return Path(__file__).resolve().parent / "data"


def ckpt_dir():
    return Path(__file__).resolve().parent / "checkpoints"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _get(url, tries=8, base=2.0):
    """GET with exponential backoff -- the listing endpoint answers HTTP 429
    ('slow down') if you ask for many pages in a row."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(base * (2 ** attempt))


def _crop(img):
    """CelebA frames are 178x218 with a lot of hair and background. The standard
    crop keeps the 148x148 box around the face, then we resize to 64x64."""
    img = img.convert("RGB")
    left, top = (178 - 148) // 2, 40
    return img.crop((left, top, left + 148, top + 148)).resize((IMG, IMG), Image.BICUBIC)


def fetch_celeba(n=N_IMAGES, verbose=True):
    """Download `n` CelebA faces + attributes once; return (uint8 images, attrs).

    images: (n, 64, 64, 3) uint8
    attrs:  (n, 40) int8, +1 / -1 exactly as CelebA labels them
    """
    data_dir().mkdir(parents=True, exist_ok=True)
    cache = data_dir() / f"celeba_{n}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["images"], z["attrs"]

    rows = []
    page = 100
    for off in range(0, n, page):
        d = json.loads(_get(_ROWS_URL.format(offset=off, length=min(page, n - off))))
        rows.extend(d["rows"])
        if verbose and off % 1000 == 0:
            print(f"  listing {off + len(d['rows'])}/{n}", flush=True)

    def one(r):
        raw = _get(r["row"]["image"]["src"])
        return np.asarray(_crop(Image.open(BytesIO(raw))), dtype=np.uint8)

    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        images = list(ex.map(one, rows))
    if verbose:
        print(f"  downloaded {len(images)} faces in {time.time() - t0:.0f}s", flush=True)

    images = np.stack(images)
    attrs = np.stack([
        np.array([r["row"][a] for a in ATTRS], dtype=np.int8) for r in rows
    ])
    np.savez_compressed(cache, images=images, attrs=attrs)
    return images, attrs


def load_faces(n=N_IMAGES):
    """(train_imgs, train_attrs, val_imgs, val_attrs) as uint8 / int8 arrays."""
    imgs, attrs = fetch_celeba(n)
    return imgs[:-N_VAL], attrs[:-N_VAL], imgs[-N_VAL:], attrs[-N_VAL:]


def to_tensor(u8):
    """uint8 HWC in [0,255] -> float CHW in [-1,1] (the range the decoder's tanh
    can actually reach)."""
    x = torch.from_numpy(np.ascontiguousarray(u8)).float().div_(127.5).sub_(1.0)
    return x.permute(0, 3, 1, 2).contiguous()


def to_uint8(x):
    """float CHW in [-1,1] -> uint8 HWC, for saving pictures."""
    x = x.clamp(-1, 1).add(1).mul(127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()


# ---------------------------------------------------------------------------
# attributes -> a short English caption
# ---------------------------------------------------------------------------
def attr_caption(a):
    """One CelebA attribute row (+1/-1) -> a short caption, e.g.
    'a smiling young woman with blond hair and heavy makeup'.

    Only attributes that are visible at 64x64 and reasonably reliable are used;
    the rest are dropped so the text stays short and honest.
    """
    on = lambda k: a[_IX[k]] > 0

    words = ["a"]
    if on("Smiling"):
        words.append("smiling")
    if on("Young"):
        words.append("young")
    else:
        words.append("older")
    words.append("man" if on("Male") else "woman")

    extras = []
    hair = None
    for k, name in (("Bald", "no hair"), ("Blond_Hair", "blond hair"),
                    ("Gray_Hair", "gray hair"), ("Black_Hair", "black hair"),
                    ("Brown_Hair", "brown hair")):
        if on(k):
            hair = name
            break
    if hair:
        extras.append(hair)
    if on("Bangs"):
        extras.append("bangs")
    if on("Eyeglasses"):
        extras.append("glasses")
    if on("Wearing_Hat"):
        extras.append("a hat")
    if on("Heavy_Makeup"):
        extras.append("heavy makeup")
    if on("No_Beard") is False:
        extras.append("a beard")
    if on("Mouth_Slightly_Open"):
        extras.append("an open mouth")

    if extras:
        words.append("with")
        words.append(" and ".join(extras))
    return " ".join(words)


def all_captions(attrs):
    return [attr_caption(a) for a in attrs]


# ---------------------------------------------------------------------------
# the VQ-VAE
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection. Cheap way to add depth
    without changing the resolution."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(), nn.Conv2d(c, c, 1),
        )

    def forward(self, x):
        return x + self.net(x)


def _widths(down, width):
    """Channel count after each stride-2 stage: w, 2w, 2w, ... (capped).

    The very first layer already has stride 2. Doing the cheap work at full
    64x64 resolution and only then widening is what keeps this trainable on a
    CPU -- a 64-channel convolution at 64x64 costs 16x what the same
    convolution costs at 16x16.
    """
    n = int(np.log2(down))
    return [min(width * 2 ** min(i, 1), 4 * width) for i in range(n)]


class Encoder(nn.Module):
    """64x64x3 -> (64/down) x (64/down) x latent_dim, by stride-2 convolutions."""

    def __init__(self, down=DOWN, width=WIDTH, latent=LATENT_DIM):
        super().__init__()
        chans = _widths(down, width)
        layers, c = [], 3
        for nxt in chans:
            layers += [nn.Conv2d(c, nxt, 4, stride=2, padding=1), nn.SiLU()]
            c = nxt
        layers += [ResBlock(c), ResBlock(c), nn.SiLU(), nn.Conv2d(c, latent, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """The mirror image of the encoder: latent grid -> 64x64x3 in [-1,1]."""

    def __init__(self, down=DOWN, width=WIDTH, latent=LATENT_DIM):
        super().__init__()
        chans = _widths(down, width)[::-1]
        c = chans[0]
        layers = [nn.Conv2d(latent, c, 1), ResBlock(c), ResBlock(c)]
        for nxt in chans[1:] + [width]:
            layers += [nn.SiLU(), nn.ConvTranspose2d(c, nxt, 4, stride=2, padding=1)]
            c = nxt
        layers += [nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Quantizer(nn.Module):
    """The codebook: snap every latent vector to its nearest of K entries.

    `ema=True` updates the entries with an exponential moving average of the
    vectors assigned to them instead of by gradient descent -- this is what the
    original VQ-VAE-2 does, and it is much more stable.

    `restart=True` re-seeds any entry that went unused for a while onto a real
    encoder output, which is the standard cure for codebook collapse.
    """

    def __init__(self, k=CODEBOOK, dim=LATENT_DIM, beta=0.25, ema=True,
                 decay=0.99, restart=True, dead_after=200):
        super().__init__()
        self.k, self.dim, self.beta = k, dim, beta
        self.ema, self.decay, self.restart, self.dead_after = ema, decay, restart, dead_after
        emb = torch.randn(k, dim) * 0.1
        if ema:
            self.register_buffer("embedding", emb)
            self.register_buffer("cluster_size", torch.ones(k))
            self.register_buffer("ema_w", emb.clone())
        else:
            self.embedding = nn.Parameter(emb)
        self.register_buffer("idle", torch.zeros(k))       # steps since last use
        self.register_buffer("hits", torch.zeros(k))       # lifetime usage count

    def _emb(self):
        return self.embedding

    def forward(self, z):
        """z: (B, D, H, W) -> (quantized z, loss, indices, stats)."""
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)                  # (BHW, D)
        e = self._emb()
        # squared distance to every entry, without materialising the difference
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ e.t()
                + e.pow(2).sum(1)[None, :])
        idx = dist.argmin(1)                                          # (BHW,)
        q = e[idx].view(b, h, w, d).permute(0, 3, 1, 2)

        if self.ema:
            # the codebook is not trained by the optimizer, so only the
            # commitment term (encoder -> code) contributes a gradient
            loss = self.beta * F.mse_loss(z, q.detach())
        else:
            loss = (F.mse_loss(q, z.detach())                        # codebook loss
                    + self.beta * F.mse_loss(z, q.detach()))         # commitment loss

        with torch.no_grad():
            counts = torch.bincount(idx, minlength=self.k).float()
            self.hits += counts
            self.idle = torch.where(counts > 0, torch.zeros_like(self.idle), self.idle + 1)
            if self.ema and self.training:
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                onehot_sum = torch.zeros_like(self.ema_w).index_add_(0, idx, flat)
                self.ema_w.mul_(self.decay).add_(onehot_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + 1e-5) / (n + self.k * 1e-5) * n
                self.embedding.copy_(self.ema_w / cs.unsqueeze(1))
            if self.restart and self.training:
                dead = (self.idle > self.dead_after).nonzero(as_tuple=True)[0]
                if len(dead) > 0:
                    pick = torch.randint(0, flat.shape[0], (len(dead),))
                    self.embedding[dead] = flat[pick]
                    if self.ema:
                        self.ema_w[dead] = flat[pick]
                        self.cluster_size[dead] = 1.0
                    self.idle[dead] = 0
            probs = counts / counts.sum()
            nz = probs[probs > 0]
            stats = {
                "used": int((counts > 0).sum()),
                "perplexity": float(torch.exp(-(nz * nz.log()).sum())),
            }

        # straight-through: forward uses q, backward pretends q == z
        q = z + (q - z).detach()
        return q, loss, idx.view(b, h, w), stats


class VQVAE(nn.Module):
    def __init__(self, down=DOWN, k=CODEBOOK, latent=LATENT_DIM, width=WIDTH,
                 ema=True, restart=True, beta=0.25):
        super().__init__()
        self.down, self.k, self.grid = down, k, IMG // down
        self.n_tokens = self.grid * self.grid
        self.enc = Encoder(down, width, latent)
        self.dec = Decoder(down, width, latent)
        self.vq = Quantizer(k, latent, beta=beta, ema=ema, restart=restart)

    def forward(self, x):
        z = self.enc(x)
        q, vq_loss, idx, stats = self.vq(z)
        return self.dec(q), vq_loss, idx, stats

    @torch.no_grad()
    def encode_indices(self, x):
        """images (B,3,64,64) -> token grid (B, grid, grid) of ints."""
        _, _, idx, _ = self.vq(self.enc(x))
        return idx

    @torch.no_grad()
    def decode_indices(self, idx):
        """token grid (B, grid, grid) of ints -> images (B,3,64,64) in [-1,1]."""
        b, h, w = idx.shape
        q = self.vq._emb()[idx.reshape(-1)].view(b, h, w, -1).permute(0, 3, 1, 2)
        return self.dec(q)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def psnr(a, b):
    """Peak signal-to-noise ratio in dB, on images in [-1,1] (peak = 2)."""
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def train_vqvae(model, train_u8, val_u8, steps=2000, batch=32, lr=3e-4,
                seed=0, log_every=250, verbose=True):
    """One training loop for every configuration. Returns a history list."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    val = to_tensor(val_u8[:256])
    n = len(train_u8)
    hist, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        pick = np.random.randint(0, n, batch)
        x = to_tensor(train_u8[pick])
        if np.random.rand() < 0.5:                      # horizontal flip: faces
            x = torch.flip(x, dims=[3])                 # are symmetric, free data
        recon, vq_loss, _, stats = model(x)
        rec = F.mse_loss(recon, x)
        loss = rec + vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vrec, _, _, vstats = model(val)
                v = psnr(vrec, val)
            model.train()
            hist.append({"step": step, "train_mse": rec.item(), "val_psnr": v,
                         "used": vstats["used"], "perplexity": vstats["perplexity"],
                         "secs": time.time() - t0})
            if verbose:
                print(f"  step {step:5d}  mse {rec.item():.4f}  val PSNR {v:5.2f} dB  "
                      f"codes used {vstats['used']:3d}/{model.k}  "
                      f"ppl {vstats['perplexity']:6.1f}  {time.time() - t0:5.0f}s",
                      flush=True)
    model.eval()
    return hist


@torch.no_grad()
def evaluate(model, val_u8, batch=128):
    """PSNR and codebook usage over the whole validation set."""
    model.eval()
    tot, n, counts = 0.0, 0, torch.zeros(model.k)
    for i in range(0, len(val_u8), batch):
        x = to_tensor(val_u8[i:i + batch])
        recon, _, idx, _ = model(x)
        mse = F.mse_loss(recon.clamp(-1, 1), x, reduction="none").mean((1, 2, 3))
        tot += (10 * torch.log10(4.0 / mse.clamp_min(1e-12))).sum().item()
        n += len(x)
        counts += torch.bincount(idx.reshape(-1), minlength=model.k).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    return {
        "psnr": tot / n,
        "used": int((counts > 0).sum()),
        "perplexity": float(torch.exp(-(nz * nz.log()).sum())),
        "bits_per_token": float(np.log2(model.k)),
        "tokens": model.n_tokens,
    }


# ---------------------------------------------------------------------------
# the shared frozen tokenizer used by projects 33-36
# ---------------------------------------------------------------------------
TOKENIZER_PATH = "vqvae_f8.pt"


def load_tokenizer(strict=True):
    """Load the frozen f=8 tokenizer (8x8 = 64 tokens per face).

    Projects 33/34/35/36 call this. Run
    `python3 run.py --stage train` inside 32-discrete-image-tokens first.
    """
    path = ckpt_dir() / TOKENIZER_PATH
    if not path.exists():
        if not strict:
            return None
        raise FileNotFoundError(
            f"{path} is missing. Run:\n"
            f"    cd {Path(__file__).resolve().parent}\n"
            f"    python3 run.py --stage train\n"
            "(about 3 minutes; it downloads CelebA on first use.)")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = VQVAE(**blob["config"])
    model.load_state_dict(blob["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_tokenizer(model, config, extra=None):
    ckpt_dir().mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "config": config, "extra": extra or {}},
               ckpt_dir() / TOKENIZER_PATH)


@torch.no_grad()
def tokenize_all(model, images_u8, batch=128, verbose=False):
    """(N,64,64,3) uint8 -> (N, n_tokens) int16 flat token rows in raster order."""
    out = []
    for i in range(0, len(images_u8), batch):
        idx = model.encode_indices(to_tensor(images_u8[i:i + batch]))
        out.append(idx.reshape(len(idx), -1).to(torch.int16).numpy())
        if verbose and i % 2048 == 0:
            print(f"  tokenized {i}/{len(images_u8)}", flush=True)
    return np.concatenate(out)
