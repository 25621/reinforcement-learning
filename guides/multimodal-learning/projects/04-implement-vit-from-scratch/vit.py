"""A Vision Transformer written from scratch, plus the CIFAR-10 plumbing to train it.

Every layer here is hand-written on top of plain tensor ops. Nothing is imported
from `timm` or `torchvision.models`, because the whole point of the project is
that a ViT is short enough to read in one sitting.

The one shortcut: `Attention` can call `F.scaled_dot_product_attention` instead
of its own softmax. That is a *speed* switch, not a logic switch -- `run.py
--stage verify` checks the two paths agree to ~1e-6, so the fast path is only
the slow path with a faster kernel underneath.

Project 08 (patch-size study) imports this module, so the ViT it sweeps is
literally the same code trained here.
"""

import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# Channel means/stds of the CIFAR-10 training set. Normalizing with the *dataset's
# own* statistics puts every input channel at roughly mean 0, std 1, which is the
# range LayerNorm and the optimizer are implicitly tuned for.
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Patchification
# ---------------------------------------------------------------------------
def patchify_unfold(x, patch):
    """The literal reading of 'cut the image into squares'.

    Returns (B, N, patch*patch*C): one flat row per square, read in raster order
    (left to right, then top to bottom -- the order a printer scans a page).

    This is the slow, obvious version. `PatchEmbed` below does the same thing
    with one strided convolution; `run.py --stage verify` proves they match.
    """
    B, C, H, W = x.shape
    # unfold slides a patch x patch window with stride = patch, i.e. no overlap.
    cols = F.unfold(x, kernel_size=patch, stride=patch)   # (B, C*patch*patch, N)
    return cols.transpose(1, 2)                            # (B, N, C*patch*patch)


class PatchEmbed(nn.Module):
    """Cut the image into patches and project each one to `dim` numbers.

    A convolution whose kernel size *equals* its stride never lets two windows
    overlap, so it visits exactly the same squares `patchify_unfold` would --
    and applying the kernel is exactly the linear projection we wanted anyway.
    One op does both jobs.
    """

    def __init__(self, img_size=32, patch=4, in_chans=3, dim=128):
        super().__init__()
        assert img_size % patch == 0, "patch size must divide the image size"
        self.patch = patch
        self.grid = img_size // patch
        self.n_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                    # (B, dim, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, n_patches, dim), raster order

    def as_linear(self, x):
        """The unfold-then-matmul route, using this layer's own weights.

        Only used by the verification stage. Conv2d stores its weight as
        (dim, C, patch, patch); flattening the last three axes gives exactly the
        (dim, C*patch*patch) matrix that multiplies one flattened patch.
        """
        cols = patchify_unfold(x, self.patch)                 # (B, N, C*p*p)
        w = self.proj.weight.flatten(1)                       # (dim, C*p*p)
        return cols @ w.t() + self.proj.bias


# ---------------------------------------------------------------------------
# 2. Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head self-attention, written out.

    Every token proposes a query ("what am I looking for?"), a key ("what do I
    advertise?") and a value ("what do I hand over if picked"). Scores are
    query-dot-key, softmaxed into weights that sum to 1, and the output is the
    weighted average of the values.
    """

    def __init__(self, dim, heads=4, fast=True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        # One matrix produces query, key and value together: cheaper than three
        # separate Linears doing the same work.
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.fast = fast

    def _split(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4)      # (3, B, heads, N, head_dim)

    def forward(self, x, return_attn=False):
        B, N, D = x.shape
        q, k, v = self._split(x)
        if self.fast and not return_attn:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Divide by sqrt(head_dim): without it the dot products grow with
            # dimension, softmax saturates, and gradients vanish.
            scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
            attn = scores.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)            # merge heads back
        out = self.proj(out)
        if return_attn:
            return out, attn
        return out


class Block(nn.Module):
    """One transformer block: attention, then a per-token MLP, both residual.

    Pre-norm (LayerNorm *before* each sub-layer) keeps a clean identity path from
    input to output, which is what makes deep stacks trainable without warmup.
    """

    def __init__(self, dim, heads=4, mlp_ratio=4, fast=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, fast=fast)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------
# 3. The ViT
# ---------------------------------------------------------------------------
class ViT(nn.Module):
    """Patchify -> add CLS + positions -> transformer blocks -> classify.

    Args worth knowing:
      pool     'cls'  read the class token's final vector
               'mean' average all patch tokens instead
      use_pos  False removes the positional embedding, which makes the model
               provably blind to *where* each patch was (see run.py verify).
    """

    def __init__(self, img_size=32, patch=4, dim=128, depth=4, heads=4,
                 n_classes=10, pool="cls", use_pos=True, fast=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 3, dim)
        n = self.patch_embed.n_patches
        self.pool = pool
        self.use_pos = use_pos

        # A learned token that belongs to no patch. It starts identical for every
        # image, so whatever it ends up holding was *gathered* from the patches
        # by attention -- which is exactly what makes it a whole-image summary.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # One learned vector per slot. Attention itself is order-blind, so this
        # is the only thing that tells the model patch 0 sits top-left.
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))

        self.blocks = nn.ModuleList([Block(dim, heads, fast=fast)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def tokens(self, x):
        t = self.patch_embed(x)
        cls = self.cls_token.expand(t.size(0), -1, -1)
        t = torch.cat([cls, t], dim=1)
        if self.use_pos:
            t = t + self.pos_embed
        return t

    def forward(self, x, return_attn=False):
        t = self.tokens(x)
        attn = None
        for i, blk in enumerate(self.blocks):
            if return_attn and i == len(self.blocks) - 1:
                out, attn = blk.attn(blk.norm1(t), return_attn=True)
                t = t + out
                t = t + blk.mlp(blk.norm2(t))
            else:
                t = blk(t)
        t = self.norm(t)
        feat = t[:, 0] if self.pool == "cls" else t[:, 1:].mean(dim=1)
        logits = self.head(feat)
        return (logits, attn) if return_attn else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4. CIFAR-10
# ---------------------------------------------------------------------------
def cifar_data(root):
    """Load CIFAR-10 as uint8 arrays, downloading and caching once if needed.

    Kept as uint8 (150 MB) rather than float32 (600 MB) and converted per batch;
    the conversion is far cheaper than the memory traffic of holding floats.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        f = root / f"cifar10_{split}.npz"
        if not f.exists():
            _download_cifar(f, split)
        d = np.load(f)
        out[split] = (d["x"], d["y"])
    return out


def _download_cifar(dest, split):
    """Pull one CIFAR-10 split from the Hugging Face parquet mirror.

    The canonical cs.toronto.edu tarball is unusably slow from many networks;
    the HF CDN serves the same 50k/10k images in a few seconds.
    """
    import huggingface_hub as hf
    import pyarrow.parquet as pq
    from PIL import Image

    print(f"[data] downloading CIFAR-10 {split} split ...", flush=True)
    p = hf.hf_hub_download("uoft-cs/cifar10",
                           f"plain_text/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    tb = pq.read_table(p)
    imgs = tb.column("img").to_pylist()
    y = np.array(tb.column("label").to_pylist(), dtype=np.int64)
    x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                  for d in imgs])
    np.savez_compressed(dest, x=x, y=y)
    print(f"[data] cached {dest.name}: {x.shape}", flush=True)


def to_tensor(x_uint8):
    """(N,32,32,3) uint8 -> (N,3,32,32) normalized float32."""
    x = torch.from_numpy(x_uint8).float().div_(255.0)
    x = (x - torch.from_numpy(CIFAR_MEAN)) / torch.from_numpy(CIFAR_STD)
    return x.permute(0, 3, 1, 2).contiguous()


def augment(x, gen):
    """Random horizontal flip + random 4-pixel shift.

    Both are label-preserving: a mirrored cat is still a cat. They cost almost
    nothing and stop a small ViT from memorizing the 50k training images, which
    it otherwise starts doing within a couple of epochs.
    """
    B = x.size(0)
    flip = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dx = torch.randint(0, 9, (2,), generator=gen)
    return pad[:, :, dx[0]:dx[0] + 32, dx[1]:dx[1] + 32]


@torch.no_grad()
def evaluate(model, x, y, bs=500):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        logits = model(x[i:i + bs])
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(y)


def train_vit(model, data, steps=700, bs=128, lr=1e-3, seed=0,
              eval_every=100, n_eval=2000, label="", max_seconds=None,
              schedule="cosine"):
    """Train with AdamW and a warmup-then-`schedule` learning rate.

    Returns a history dict with, for every eval point, the step number, the
    seconds of wall clock spent so far, and the test accuracy. Project 08 reads
    both the step axis and the time axis off this same history.

    `schedule="const"` holds the learning rate flat after warmup. Project 08
    needs that: with a cosine decay, a run is only "finished" at its own final
    step, so reading an early slice off the curve would compare a half-annealed
    model against a fully-annealed one. A flat rate makes every prefix of the
    run a legitimate "trained for k steps" checkpoint.

    `max_seconds` stops training once the budget is spent, which is how project
    08 gives every patch size the same amount of *compute* rather than the same
    number of steps.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 2)

    xtr_u8, ytr_np = data["train"]
    xte_u8, yte_np = data["test"]
    # Index the uint8 array and convert only the 128 images actually used this
    # step. Converting all 50k up front would build a 600 MB float tensor and
    # make every batch a gather over four times more memory.
    ytr = torch.from_numpy(ytr_np)
    xte = to_tensor(xte_u8[:n_eval])
    yte = torch.from_numpy(yte_np[:n_eval])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = max(1, steps // 20)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        if schedule == "const":
            return lr
        p = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + np.cos(np.pi * p))

    hist = {"step": [], "sec": [], "acc": [], "loss": []}
    model.train()
    running, t0, eval_sec = 0.0, time.time(), 0.0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        idx = rng.integers(0, len(ytr), bs)
        xb = augment(to_tensor(xtr_u8[idx]), gen)
        loss = F.cross_entropy(model(xb), ytr[torch.from_numpy(idx)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()

        if (s + 1) % eval_every == 0 or s == steps - 1:
            te = time.time()
            acc = evaluate(model, xte, yte)
            eval_sec += time.time() - te     # don't bill evaluation to training
            hist["step"].append(s + 1)
            hist["sec"].append(time.time() - t0 - eval_sec)
            hist["acc"].append(acc)
            hist["loss"].append(running / eval_every)
            print(f"  [{label}] step {s + 1:4d}  {hist['sec'][-1]:6.1f}s  "
                  f"loss {hist['loss'][-1]:.3f}  test-acc {acc:.4f}", flush=True)
            running = 0.0
            if max_seconds is not None and hist["sec"][-1] >= max_seconds:
                print(f"  [{label}] budget of {max_seconds}s spent at "
                      f"step {s + 1}", flush=True)
                break
    return hist
