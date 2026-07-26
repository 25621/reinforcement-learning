"""Project 17 -- teach a frozen CLIP to read depth maps with a small adapter.

A depth map is a grayscale picture where brightness means distance, not colour.
CLIP has never seen one: it was trained on photographs with captions. So we
have a backbone that knows a great deal about the visual world and an input it
cannot parse. The question this project answers is how few *new* weights it
takes to bridge that gap, and where those weights should go.

Five conditions, all classifying the same 10 Imagenette categories:

  rgb-ceiling         frozen CLIP on the ORIGINAL photo          (upper bound)
  naive-frozen        frozen CLIP on the depth map, no adapter   (lower bound)
  input-adapter       a tiny conv stem in front of frozen CLIP
  bottleneck-adapter  small Houlsby adapters inside all 12 blocks
  scratch-cnn         a small CNN trained on depth from scratch

Where the depth maps come from: a real monocular depth estimator
(Depth-Anything-V2-Small) run once over real photographs. These are *estimated*
depth maps, not laser measurements, which is worth saying out loud -- but they
are genuine geometry read off real scenes, and they throw away colour and
texture entirely, which is what makes them a different modality for our purpose.

Every condition is finally scored the same way -- extract features, then fit a
converged linear probe -- so that a slow classifier head cannot masquerade as a
bad adapter. See the README for what that correction changed.

Usage:
    python3 run.py --stage data      # download + estimate depth (~4 min, cached)
    python3 run.py --stage train     # the five conditions (~15 min)
    python3 run.py --stage plot
"""

import argparse
import csv
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(12)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
RES = 224
N_TRAIN, N_VAL = 1000, 500
EPOCHS = 6
BATCH = 32
LR = 2e-3
CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church",
           "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
# the depth estimator was trained with the ImageNet statistics, not CLIP's
_DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DEPTH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONDITIONS = ["rgb-ceiling", "naive-frozen", "input-adapter",
              "bottleneck-adapter", "scratch-cnn"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _square(img, size=RES):
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    return img.crop(((w - size) // 2, (h - size) // 2,
                     (w - size) // 2 + size, (h - size) // 2 + size))


def _download():
    tgz = DATA / "imagenette2-160.tgz"
    root = DATA / "imagenette2-160"
    if root.exists():
        return root
    DATA.mkdir(parents=True, exist_ok=True)
    if not tgz.exists():
        print("  downloading Imagenette ...", flush=True)
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA)
    return root


def stage_data(_args):
    """Sample images, estimate depth for each, cache depth + CLIP RGB features."""
    if (DATA / "depth.npy").exists():
        print("cache already built")
        return
    root = _download()
    wnids = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    rng = np.random.default_rng(0)

    def sample(split, per_class):
        files, labels = [], []
        for c, w in enumerate(wnids):
            paths = sorted((root / split / w).glob("*.JPEG"))
            pick = rng.permutation(len(paths))[:per_class]
            files += [paths[i] for i in pick]
            labels += [c] * len(pick)
        return files, np.array(labels)

    tr_files, tr_y = sample("train", N_TRAIN // 10)
    va_files, va_y = sample("val", N_VAL // 10)
    files = tr_files + va_files
    labels = np.concatenate([tr_y, va_y])
    print(f"  {len(files)} images ({len(tr_files)} train / {len(va_files)} val)",
          flush=True)

    from transformers import CLIPModel, AutoModelForDepthEstimation
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf").eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    depth = np.zeros((len(files), RES, RES), dtype=np.uint8)
    rgb_feats = np.zeros((len(files), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        arr = np.stack([np.asarray(_square(Image.open(p)), dtype=np.uint8)
                        for p in chunk])
        x = torch.from_numpy(np.ascontiguousarray(
            ((arr.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(0, 3, 1, 2)))
        xd = torch.from_numpy(np.ascontiguousarray(
            ((arr.astype(np.float32) / 255.0 - _DEPTH_MEAN) / _DEPTH_STD)
            .transpose(0, 3, 1, 2)))
        with torch.no_grad():
            d = depth_model(pixel_values=xd).predicted_depth
            d = F.interpolate(d[:, None], (RES, RES), mode="bilinear")[:, 0]
            # per-image min-max so "bright = near" means the same in every map
            lo = d.amin(dim=(1, 2), keepdim=True)
            hi = d.amax(dim=(1, 2), keepdim=True)
            depth[i:i + len(chunk)] = ((d - lo) / (hi - lo + 1e-6) * 255).byte().numpy()
            out = clip.get_image_features(pixel_values=x)
            rgb_feats[i:i + len(chunk)] = (
                out.pooler_output if hasattr(out, "pooler_output") else out).numpy()
        if (i + 16) % 160 == 0:
            print(f"    {i + 16}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)

    np.save(DATA / "depth.npy", depth)
    np.save(DATA / "rgb_feats.npy", rgb_feats)
    np.save(DATA / "labels.npy", labels)
    (DATA / "meta.json").write_text(json.dumps(dict(
        n_train=len(tr_files), n_val=len(va_files),
        files=[str(p) for p in files])))
    print("  cached", DATA / "depth.npy", flush=True)


def load():
    meta = json.loads((DATA / "meta.json").read_text())
    depth = np.load(DATA / "depth.npy")
    y = np.load(DATA / "labels.npy")
    rgb = np.load(DATA / "rgb_feats.npy")
    n = meta["n_train"]
    return depth, y, rgb, np.arange(n), np.arange(n, len(y))


def source_files():
    return json.loads((DATA / "meta.json").read_text())["files"]


def depth_pixels(raw, jitter=None):
    """uint8 (B,224,224) -> float (B,1,224,224), CLIP-normalised.

    The depth map is a single channel. We normalise it with the *average* of
    CLIP's three per-channel statistics, so a mid-grey depth map lands where a
    mid-grey photo would: the frozen backbone is expecting inputs in that range
    and behaves oddly outside it.
    """
    x = raw.astype(np.float32) / 255.0
    x = (x - _MEAN.mean()) / _STD.mean()
    t = torch.from_numpy(x)[:, None]
    if jitter is not None and jitter.random() < 0.5:
        t = torch.flip(t, dims=[3])
    return t


# ---------------------------------------------------------------------------
# the three ways to attach new weights
# ---------------------------------------------------------------------------
class InputAdapter(nn.Module):
    """A tiny conv net in FRONT of the frozen encoder.

    It starts as an exact copy of the naive baseline: the depth map is repeated
    into three channels, and the learned correction is added on top with a
    zero-initialised final layer. So step 0 of this run reproduces the
    `naive-frozen` condition exactly, and every point of improvement afterwards
    is attributable to the adapter. Project 19 uses the same "start as the old
    model" idea with a gate instead of a zero-init weight.
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x.expand(-1, 3, -1, -1) + self.net(x)


class Bottleneck(nn.Module):
    """Houlsby-style adapter: squeeze 768 numbers down to 32, transform, expand.

    Why a squeeze at all, when a full 768x768 layer would be more expressive:
    the whole point is to add as few weights as possible. Down-up through 32
    costs 2 * 768 * 32 = 49k weights instead of 590k, and empirically that
    narrow path is enough to *steer* a strong backbone rather than rebuild it.
    The up-projection is zero-initialised, so the adapter starts as a no-op.
    """

    def __init__(self, d=768, bottleneck=32):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


class DepthCLIP(nn.Module):
    """Frozen CLIP vision tower + optionally an input adapter and/or in-block
    adapters + a linear classifier."""

    def __init__(self, mode, n_classes=10):
        super().__init__()
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision = clip.vision_model
        self.post = clip.visual_projection
        for p in self.parameters():
            p.requires_grad_(False)                       # freeze everything
        self.stem = InputAdapter() if mode == "input-adapter" else None
        self.adapters = None
        self.handles = []
        if mode == "bottleneck-adapter":
            self.adapters = nn.ModuleList(
                Bottleneck() for _ in self.vision.encoder.layers)
            for adapter, layer in zip(self.adapters, self.vision.encoder.layers):
                self.handles.append(layer.register_forward_hook(_hook(adapter)))
        self.head = nn.Linear(512, n_classes)

    def features(self, x):
        """The 512-number image vector CLIP would hand to a classifier."""
        x = self.stem(x) if self.stem is not None else x.expand(-1, 3, -1, -1)
        return self.post(self.vision(pixel_values=x).pooler_output)

    def forward(self, x):
        return self.head(self.features(x))


def _hook(adapter):
    """Insert an adapter after a frozen encoder block without editing it.

    A forward hook receives that block's output and may replace it. This is how
    you add adapters to a pretrained model you did not write: no subclassing,
    no copied source, and the original weights stay exactly where they were.
    """
    def fn(module, inputs, output):
        if isinstance(output, tuple):
            return (adapter(output[0]),) + output[1:]
        return adapter(output)
    return fn


class ScratchCNN(nn.Module):
    """A small convnet trained on depth from scratch -- the 'no transfer' control."""

    def __init__(self, n_classes=10, w=48):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (w, w * 2, w * 2, w * 4, w * 4):
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                       nn.GroupNorm(8, c_out), nn.GELU()]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x):
        return self.head(self.body(x).mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def linear_probe(feats, y, tr, va, steps=600, lr=1e-2, seed=0):
    """A plain logistic regression on frozen features -- used by the two
    conditions where nothing before the classifier is trainable."""
    torch.manual_seed(seed)
    f = torch.from_numpy(feats)
    f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    head = nn.Linear(f.shape[1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        ids = rng.integers(0, len(tr), size=128)
        loss = F.cross_entropy(head(f[tr][ids]), yt[tr][ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(f[va]).argmax(-1) == yt[va]).float().mean())
    return acc, sum(p.numel() for p in head.parameters())


@torch.no_grad()
def frozen_features(model, depth, ids, batch=48):
    """Run whatever is in front of the classifier over every image once.

    Every condition is finally scored the same way -- extract features, then
    fit a converged linear probe. Without this the comparison would be unfair:
    the no-adapter baseline gets a 600-step probe on cached features while an
    adapter's head only sees the ~190 joint steps, so a slow head would look
    like a bad adapter.
    """
    model.eval()
    outs = []
    for i in range(0, len(ids), batch):
        outs.append(model.features(depth_pixels(depth[ids[i:i + batch]])).numpy())
    return np.concatenate(outs)


def train_model(model, depth, y, tr, va, epochs=EPOCHS, lr=LR, seed=0):
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(seed)
    steps = epochs * (len(tr) // BATCH)
    hist = []
    step = 0
    for _ in range(epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order) - BATCH + 1, BATCH):
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * step / steps))
            ids = order[i:i + BATCH]
            x = depth_pixels(depth[ids], rng)
            model.train()
            loss = F.cross_entropy(model(x), yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hist.append(float(loss.detach()))
            step += 1
            if step % 20 == 0:
                print(f"    step {step:4d}/{steps}  loss {np.mean(hist[-20:]):.3f}",
                      flush=True)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(va), 48):
            ids = va[i:i + 48]
            correct += int((model(depth_pixels(depth[ids])).argmax(-1)
                            == yt[ids]).sum())
    return correct / len(va), sum(p.numel() for p in params), hist, model


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    depth, y, rgb, tr, va = load()
    rows = []
    for cond in (args.only or CONDITIONS):
        print(f"\n=== {cond}", flush=True)
        t0 = time.time()
        joint = None
        if cond == "rgb-ceiling":
            acc, n = linear_probe(rgb, y, tr, va)
        elif cond == "naive-frozen":
            model = DepthCLIP("none")
            feats = frozen_features(model, depth, np.arange(len(y)))
            acc, n = linear_probe(feats, y, tr, va)
        elif cond == "scratch-cnn":
            # no frozen feature space to probe -- the whole net is the model
            acc, n, _, _ = train_model(ScratchCNN(), depth, y, tr, va)
            joint = acc
        else:
            joint, n, _, model = train_model(DepthCLIP(cond), depth, y, tr, va)
            feats = frozen_features(model, depth, np.arange(len(y)))
            acc, _ = linear_probe(feats, y, tr, va)
        secs = time.time() - t0
        rows.append(dict(condition=cond, acc=acc,
                         joint_head_acc=("" if joint is None else round(joint, 4)),
                         trainable_params=n, seconds=secs))
        print(f"  probe acc {acc:.3f}  (joint head {joint})  "
              f"trainable {n:,}  ({secs:.0f}s)", flush=True)

    path = OUT / "adapters.csv"
    old = {r["condition"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    old.update({r["condition"]: r for r in rows})
    ordered = [old[c] for c in CONDITIONS if c in old]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "acc", "joint_head_acc",
                                          "trainable_params", "seconds"])
        w.writeheader()
        w.writerows(ordered)
    print("\nwrote", path)


def stage_plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(OUT / "adapters.csv")))
    depth, y, _, tr, va = load()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    colors = {"rgb-ceiling": "#333333", "naive-frozen": "#c44e52",
              "input-adapter": "#4c72b0", "bottleneck-adapter": "#55a868",
              "scratch-cnn": "#b07aa1"}
    x = np.arange(len(rows))
    ax.bar(x, [float(r["acc"]) for r in rows],
           color=[colors[r["condition"]] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, float(r["acc"]) + 0.012,
                f"{float(r['acc']):.3f}\n{int(r['trainable_params']):,} w",
                ha="center", fontsize=8)
    ax.axhline(0.1, color="k", ls=":", lw=1)
    ax.text(len(rows) - 0.5, 0.115, "chance", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([r["condition"] for r in rows], rotation=18, fontsize=8)
    ax.set_ylabel("validation accuracy (10 classes)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Reading depth maps with a frozen CLIP")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for r in rows:
        if r["condition"] == "rgb-ceiling":
            continue
        ax.scatter(int(r["trainable_params"]), float(r["acc"]), s=110,
                   color=colors[r["condition"]])
        ax.annotate(r["condition"], (int(r["trainable_params"]), float(r["acc"])),
                    textcoords="offset points", xytext=(7, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable weights (log scale)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy per trainable weight")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "adapters.png", dpi=130)

    # what the new modality actually looks like -- photo above, its depth below
    files = source_files()
    fig2, axes2 = plt.subplots(2, 5, figsize=(13, 5.4))
    pick = np.random.default_rng(3).choice(va, 5, replace=False)
    for j, idx in enumerate(pick):
        axes2[0, j].imshow(_square(Image.open(files[idx])))
        axes2[0, j].set_title(f"photo: {CLASSES[y[idx]]}", fontsize=8)
        axes2[1, j].imshow(depth[idx], cmap="magma")
        axes2[1, j].set_title("estimated depth", fontsize=8)
        for a in (axes2[0, j], axes2[1, j]):
            a.axis("off")
    fig2.suptitle("The new modality: colour and texture are gone, geometry remains",
                  fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / "modality.png", dpi=130)
    print("wrote", OUT / "adapters.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="train", choices=["data", "train", "plot"])
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    {"data": stage_data, "train": stage_train, "plot": stage_plot}[a.stage](a)
