"""Shared model, data, and measurement helpers for Phase 8 (deployment), projects 42-47.

Everything the deployment projects need in one place:

* `load_cifar()`   - CIFAR-10 as plain float32 tensors, read from the local
                     Hugging Face parquet cache (no network after the first run).
* `SmallCNN`       - a deliberately quantization-friendly CNN (Conv -> BN -> ReLU
                     blocks, no exotic ops) small enough to train on a CPU in
                     about a minute.
* `get_trained_cnn()` - train once, cache the weights, reuse from every project.
* timing helpers   - percentiles, and an *interleaved* A/B timer, because this
                     machine is shared and back-to-back timings drift.

Projects 43-47 import this file by adding project 42's directory to `sys.path`.
"""

from __future__ import annotations

import glob
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 channel statistics, used to normalize inputs the usual way.
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _parquet_path(split: str) -> str:
    pattern = (
        "/home/canary/.cache/huggingface/hub/datasets--uoft-cs--cifar10/"
        f"snapshots/*/plain_text/{split}-*.parquet"
    )
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[0]
    # Not in the local cache -> download it once.
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="uoft-cs/cifar10",
        filename=f"plain_text/{split}-00000-of-00001.parquet",
        repo_type="dataset",
    )


def load_cifar(split: str = "test", n: int | None = None, normalize: bool = True):
    """Return (images NCHW float32, labels int64) for `split` in {'train','test'}.

    Decoded once into an .npz cache under `data/`; later calls just memory-load it.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"cifar_{split}.npz")
    if not os.path.exists(cache):
        import pyarrow.parquet as pq
        from PIL import Image

        table = pq.read_table(_parquet_path(split))
        imgs = table.column("img").to_pylist()
        labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
        arr = np.zeros((len(imgs), 32, 32, 3), dtype=np.uint8)
        for i, rec in enumerate(imgs):
            arr[i] = np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
        np.savez(cache, x=arr, y=labels)
        print(f"cached {len(imgs)} {split} images -> {cache}")
    blob = np.load(cache)
    x, y = blob["x"], blob["y"]
    if n is not None:
        x, y = x[:n], y[:n]
    x = x.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    if normalize:
        x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Three Conv-BN-ReLU blocks then a linear classifier. ~290k parameters.

    Every layer here has an int8 kernel in PyTorch's CPU quantization backend,
    which matters for projects 44 and 45: an "unsupported" layer silently stays
    in float32 and quietly removes the speedup you were measuring.
    """

    def __init__(self, width: int = 32, n_classes: int = 10):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 32 -> 16
            nn.Conv2d(w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16 -> 8
            nn.Conv2d(2 * w, 4 * w, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),               # 8 -> 1
        )
        self.classifier = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, 1)
        return self.classifier(h)


def train_cnn(epochs: int = 4, n_train: int = 20000, batch: int = 128, seed: int = 0,
              lr: float = 3e-3, verbose: bool = True) -> SmallCNN:
    torch.manual_seed(seed)
    xtr, ytr = load_cifar("train", n_train)
    model = SmallCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = xtr[idx]
            if torch.rand(()) < 0.5:               # cheap horizontal-flip augmentation
                xb = torch.flip(xb, dims=[3])
            loss = F.cross_entropy(model(xb), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        if verbose:
            print(f"  epoch {ep + 1}/{epochs}  loss {loss.item():.3f}  "
                  f"({time.time() - t0:.0f}s)")
    model.eval()
    return model


def get_trained_cnn(force: bool = False) -> SmallCNN:
    """Train the CNN once and cache it in `42/checkpoints/small_cnn.pt`."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, "small_cnn.pt")
    model = SmallCNN()
    if os.path.exists(path) and not force:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    print("training SmallCNN (cached afterwards) ...")
    model = train_cnn()
    torch.save(model.state_dict(), path)
    acc = accuracy(model, *load_cifar("test", 2000))
    with open(os.path.join(CKPT_DIR, "small_cnn.json"), "w") as fh:
        json.dump({"test_acc_2000": acc}, fh)
    print(f"  test accuracy (2000 images): {acc:.4f}")
    return model


@torch.no_grad()
def predict(model, x, batch: int = 256) -> torch.Tensor:
    model.eval()
    out = [model(x[i:i + batch]) for i in range(0, len(x), batch)]
    return torch.cat(out)


@torch.no_grad()
def accuracy(model, x, y, batch: int = 256) -> float:
    return (predict(model, x, batch).argmax(1) == y).float().mean().item()


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def percentiles(samples, qs=(50, 95, 99)) -> dict:
    a = np.sort(np.asarray(samples, dtype=np.float64))
    out = {f"p{q}": float(np.percentile(a, q)) for q in qs}
    out["mean"] = float(a.mean())
    out["min"] = float(a[0])
    out["max"] = float(a[-1])
    out["n"] = int(a.size)
    return out


def time_calls(fn, n: int = 50, warmup: int = 5):
    """Call `fn` n times, return the per-call wall time in milliseconds."""
    for _ in range(warmup):
        fn()
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t0) * 1e3
    return out


def interleaved(variants: dict, rounds: int = 7, calls: int = 10, warmup: int = 3):
    """Time several callables by rotating between them, round by round.

    On a shared machine another process can grab the cores half-way through a
    measurement. Timing A fully and then B fully would charge that entirely to
    B. Rotating A,B,A,B,... spreads any drift over both, so the *comparison*
    survives even when the absolute numbers wobble. Returns
    {name: {"median_ms":..., "per_round":[...]}}.
    """
    names = list(variants)
    for name in names:
        for _ in range(warmup):
            variants[name]()
    per_round = {name: [] for name in names}
    for _ in range(rounds):
        for name in names:
            t0 = time.perf_counter()
            for _ in range(calls):
                variants[name]()
            per_round[name].append((time.perf_counter() - t0) * 1e3 / calls)
    return {
        name: {"median_ms": float(np.median(per_round[name])),
               "min_ms": float(np.min(per_round[name])),
               "max_ms": float(np.max(per_round[name])),
               "per_round": per_round[name]}
        for name in names
    }


def file_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def write_csv(path: str, rows, header):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")
