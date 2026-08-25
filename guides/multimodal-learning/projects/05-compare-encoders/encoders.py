"""Four frozen image encoders behind one interface, plus the Imagenette loader.

The four towers were trained in three genuinely different ways, which is the
whole reason to line them up:

  resnet50   convolutional, supervised on ImageNet-1k labels
  vit-b16    transformer,   supervised on ImageNet-21k labels
  siglip     transformer,   contrastive on web image-text pairs (no labels)
  dinov2     transformer,   self-supervised on images alone (no text, no labels)

Each one exposes exactly one method -- `embed(images) -> (N, D) float32` -- so
`run.py` can treat them interchangeably and the comparison stays honest.
"""

import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_SIZE = 224

# ImageNet's channel statistics. ResNet, ViT and DINOv2 were all trained with
# these; SigLIP uses a simpler [-1, 1] scaling instead. Feeding a model the
# wrong constants does not crash -- it just quietly costs you accuracy.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# Imagenette is 10 ImageNet classes picked to be easy to tell apart. The folder
# names are the raw WordNet ids ImageNet ships; these are the human names.
WNID_TO_NAME = {
    "n01440764": "tench",          "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church",         "n03394916": "French horn",
    "n03417042": "garbage truck",  "n03425413": "gas pump",
    "n03445777": "golf ball",      "n03888257": "parachute",
}
CLASSES = sorted(WNID_TO_NAME)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fetch_imagenette(data_dir):
    root = Path(data_dir) / "imagenette2-160"
    if root.exists():
        return root
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    tgz = Path(data_dir) / "imagenette2-160.tgz"
    if not tgz.exists():
        print("[data] downloading Imagenette (~95 MB) ...", flush=True)
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
    print("[data] extracting ...", flush=True)
    with tarfile.open(tgz) as t:
        t.extractall(data_dir)
    return root


def _load(path):
    """Shortest side -> 224, then center crop. The standard evaluation recipe."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, round(w * s)), max(IMG_SIZE, round(h * s))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    return np.array(img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE)))


def load_split(root, split, per_class, seed=0):
    """Return (N,224,224,3) uint8 images and (N,) int labels, class-balanced."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for ci, wnid in enumerate(CLASSES):
        files = sorted((Path(root) / split / wnid).glob("*.JPEG"))
        pick = rng.permutation(len(files))[:per_class]
        for i in pick:
            xs.append(_load(files[i]))
            ys.append(ci)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def normalize(x_uint8, mode="imagenet"):
    x = torch.from_numpy(x_uint8).float().div_(255.0)
    if mode == "imagenet":
        x = (x - torch.from_numpy(IMAGENET_MEAN)) / torch.from_numpy(IMAGENET_STD)
    else:                                   # SigLIP: plain [-1, 1]
        x = x * 2 - 1
    return x.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
class Encoder:
    """Common wrapper: hold a frozen tower, hand back one vector per image."""

    def __init__(self, key, name, how, norm_mode="imagenet"):
        self.key, self.name, self.how = key, name, how
        self.norm_mode = norm_mode
        self.model = None

    def load(self):
        raise NotImplementedError

    def _forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, x_uint8, bs=16, verbose=True):
        if self.model is None:
            t = time.time()
            self.load()
            self.model.eval()
            # Freeze everything. These towers are measured, never trained.
            for p in self.model.parameters():
                p.requires_grad_(False)
            if verbose:
                print(f"  [{self.key}] loaded in {time.time() - t:.0f}s", flush=True)
        out, t0 = [], time.time()
        for i in range(0, len(x_uint8), bs):
            xb = normalize(x_uint8[i:i + bs], self.norm_mode)
            out.append(self._forward(xb).float().numpy())
        feats = np.concatenate(out)
        self.ms_per_image = (time.time() - t0) / len(x_uint8) * 1000
        if verbose:
            print(f"  [{self.key}] {feats.shape} at "
                  f"{self.ms_per_image:.0f} ms/image", flush=True)
        return feats

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


class ResNet50(Encoder):
    def __init__(self):
        super().__init__("resnet50", "ResNet-50",
                         "supervised on ImageNet-1k labels")

    def load(self):
        import torchvision
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the 1000-way classifier: we want the features underneath it, not
        # its opinion about ImageNet classes.
        m.fc = nn.Identity()
        self.model = m

    def _forward(self, x):
        return self.model(x)          # (B, 2048) global-average-pooled


class HFEncoder(Encoder):
    def __init__(self, key, name, how, hf_id, pool="cls", norm_mode="imagenet"):
        super().__init__(key, name, how, norm_mode)
        self.hf_id, self.pool = hf_id, pool

    def load(self):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(self.hf_id)
        self.model = getattr(m, "vision_model", m)

    def _forward(self, x):
        o = self.model(pixel_values=x)
        if self.pool == "cls":
            # Token 0 is the CLS token: the slot the model was trained to use
            # as a whole-image summary.
            return o.last_hidden_state[:, 0]
        # SigLIP has no CLS token; it ends in a learned attention-pooling head,
        # and `pooler_output` is that head's result.
        return o.pooler_output


def all_encoders():
    return [
        ResNet50(),
        HFEncoder("vit_b16", "ViT-B/16", "supervised on ImageNet-21k labels",
                  "google/vit-base-patch16-224-in21k", pool="cls"),
        HFEncoder("siglip", "SigLIP-B/16",
                  "contrastive on web image-text pairs (no labels)",
                  "google/siglip-base-patch16-224", pool="pooler",
                  norm_mode="siglip"),
        HFEncoder("dinov2", "DINOv2-B/14",
                  "self-supervised on images alone (no labels, no text)",
                  "facebook/dinov2-base", pool="cls"),
    ]


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def linear_probe(ftr, ytr, fte, yte, epochs=400, lr=0.1, wd=1e-4, seed=0):
    """Fit one linear layer on frozen features; return test accuracy.

    Features are standardized first (subtract the training mean, divide by the
    training std) purely so one shared learning rate works for towers whose
    outputs live on very different scales -- ResNet's 2048 post-ReLU numbers are
    all non-negative, DINOv2's are roughly centred. Statistics come from the
    *training* split only; using the test split's mean would leak.
    """
    torch.manual_seed(seed)
    mu, sd = ftr.mean(0, keepdims=True), ftr.std(0, keepdims=True) + 1e-6
    xtr = torch.from_numpy((ftr - mu) / sd)
    xte = torch.from_numpy((fte - mu) / sd)
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    head = nn.Linear(xtr.shape[1], len(CLASSES))
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):                     # full-batch: the data is tiny
        loss = nn.functional.cross_entropy(head(xtr), ytr_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return (head(xte).argmax(1) == yte_t).float().mean().item()


def knn_probe(ftr, ytr, fte, yte, k=10):
    """Training-free probe: label a test image by its k nearest training images.

    Uses cosine similarity, so only the *direction* of each feature vector
    matters, not its length. Where the linear probe asks "can a plane separate
    these classes?", this asks "do same-class images simply land near each
    other?" -- a stricter, more local question.
    """
    a = ftr / (np.linalg.norm(ftr, axis=1, keepdims=True) + 1e-8)
    b = fte / (np.linalg.norm(fte, axis=1, keepdims=True) + 1e-8)
    sim = b @ a.T
    nn_idx = np.argsort(-sim, axis=1)[:, :k]
    votes = ytr[nn_idx]
    pred = np.array([np.bincount(v, minlength=len(CLASSES)).argmax()
                     for v in votes])
    return float((pred == yte).mean())
