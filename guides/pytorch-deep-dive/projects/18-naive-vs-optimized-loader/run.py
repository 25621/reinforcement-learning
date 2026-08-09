"""Project 18 — Naive vs optimized loader.

Measures what each DataLoader knob is actually worth, on a real on-disk image
dataset with a real per-sample decode+augment cost:

  1. throughput vs num_workers (0, 1, 2, 4, 8)
  2. the same sweep on a *cheap* dataset, where workers buy nothing
  3. the overlap ceiling: min(loader-only, model-only) vs what we measured
  4. persistent_workers: the per-epoch worker startup tax
  5. prefetch_factor and pin_memory
  6. fork vs spawn start methods
  7. where the CPU actually goes (loader wall time vs model wall time)

CPU-only by design: this box has a GTX 1070 Ti that this PyTorch build cannot
use (sm_61 < sm_70), so "the GPU is starved" becomes "the main process is
starved", which is the same measurement with cheaper hardware.

Runtime ~4 min. Needs torch, torchvision, numpy, matplotlib, Pillow.
"""

import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

SEED = 0
N_IMAGES = 3000
IMG_SIZE = 96
CROP = 64
BATCH = 32
WORKER_COUNTS = [0, 1, 2, 4, 8]

# Keep the main process single-threaded so that "the trainer" is a fixed amount
# of work and only the loader configuration changes between rows.
torch.set_num_threads(1)


# ----------------------------------------------------------------------------
# 0. the dataset on disk
# ----------------------------------------------------------------------------
def build_corpus():
    """Write N_IMAGES small JPEGs. Real files, real decode cost."""
    if DATA.exists() and len(list(DATA.glob("*.jpg"))) == N_IMAGES:
        return sorted(DATA.glob("*.jpg"))
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    for i in range(N_IMAGES):
        cls = i % 4
        base = rng.integers(60, 200, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # a class-dependent blob so the task is learnable at all
        cy, cx = rng.integers(20, IMG_SIZE - 20, size=2)
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < 14**2
        base[m] = np.array([[40, 220][cls & 1], [40, 220][(cls >> 1) & 1], 128], dtype=np.uint8)
        Image.fromarray(base).save(DATA / f"{i:05d}_{cls}.jpg", quality=88)
    return sorted(DATA.glob("*.jpg"))


class JpegDataset(Dataset):
    """The realistic case: open a file, decode it, augment it, make a tensor."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        # a per-sample random crop + flip + a colour jitter, all in PIL/NumPy
        rng = np.random.default_rng(i)
        top, left = rng.integers(0, IMG_SIZE - CROP + 1, size=2)
        img = img.crop((int(left), int(top), int(left) + CROP, int(top) + CROP))
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = np.clip(a * (0.8 + 0.4 * rng.random()) + 0.1 * (rng.random() - 0.5), 0, 1)
        a = (a - 0.45) / 0.25
        x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        y = int(path.stem.split("_")[1])
        return x, y


class HeavyJpegDataset(JpegDataset):
    """The starved case: the same pipeline plus an expensive blur, so one
    process cannot keep up with the model."""

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        a = x.numpy()
        # a separable 9-tap blur written the slow way, three times over —
        # a stand-in for the heavy augmentation real pipelines actually run
        k = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1], dtype=np.float32)
        k /= k.sum()
        for _ in range(8):
            p = np.pad(a, ((0, 0), (4, 4), (4, 4)), mode="reflect")
            a = sum(k[t] * p[:, t:t + CROP, 4:4 + CROP] for t in range(9))
            p = np.pad(a, ((0, 0), (4, 4), (4, 4)), mode="reflect")
            a = sum(k[t] * p[:, 4:4 + CROP, t:t + CROP] for t in range(9))
        return torch.from_numpy(np.ascontiguousarray(a)), y


class CheapDataset(Dataset):
    """The control: same shapes, no decode, no augment. Already in RAM."""

    def __init__(self, n):
        g = torch.Generator().manual_seed(SEED)
        self.x = torch.randn(n, 3, CROP, CROP, generator=g)
        self.y = torch.randint(0, 4, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4),
    )


# ----------------------------------------------------------------------------
# 1. the measurement harness
# ----------------------------------------------------------------------------
def run_epoch(dataset, *, num_workers, pin_memory=False, prefetch_factor=None,
              persistent_workers=False, epochs=1, train=True, ctx=None):
    """Return (samples_per_sec, loader_seconds, model_seconds)."""
    kw = dict(batch_size=BATCH, shuffle=True, num_workers=num_workers,
              pin_memory=pin_memory, drop_last=True)
    if num_workers > 0:
        kw["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kw["prefetch_factor"] = prefetch_factor
        if ctx is not None:
            kw["multiprocessing_context"] = ctx
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, generator=g, **kw)

    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    lossf = nn.CrossEntropyLoss()

    n_seen = 0
    t_loader = 0.0
    t_model = 0.0
    t0 = time.perf_counter()
    for _ in range(epochs):
        t_mark = time.perf_counter()
        for x, y in loader:
            t_loader += time.perf_counter() - t_mark
            t_mark = time.perf_counter()
            if train:
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
            t_model += time.perf_counter() - t_mark
            n_seen += x.shape[0]
            t_mark = time.perf_counter()
    wall = time.perf_counter() - t0
    del loader
    return n_seen / wall, t_loader, t_model, wall


def main():
    files = build_corpus()
    print(f"corpus: {len(files)} jpegs in {DATA}")
    half = files[: len(files) // 2]
    suites = [
        ("cheap", CheapDataset(len(files)), "already in RAM"),
        ("medium", JpegDataset(files), "jpeg decode + augment"),
        ("heavy", HeavyJpegDataset(half), "jpeg + expensive blur"),
    ]
    rows = []
    tp_by_suite = {}

    # --- 1/2. the sweep, three per-sample costs ---------------------------
    for name, ds, label in suites:
        print(f"\n== throughput vs num_workers — {name} ({label}, {len(ds)} samples) ==")
        tp_by_suite[name] = {}
        for nw in WORKER_COUNTS:
            # best of 2: CPU timings on a shared box bounce, and the best run is
            # the one least polluted by whatever else the machine was doing
            tp, tl, tm, wall = max((run_epoch(ds, num_workers=nw) for _ in range(2)),
                                   key=lambda r: r[0])
            tp_by_suite[name][nw] = tp
            rows.append(dict(section=f"workers_{name}", config=f"num_workers={nw}",
                             samples_per_sec=round(tp, 1), loader_s=round(tl, 2),
                             model_s=round(tm, 2), wall_s=round(wall, 2)))
            print(f"  num_workers={nw:<2} {tp:8.1f} samples/s   waiting on data {tl:5.2f}s"
                  f"   model {tm:5.2f}s   wall {wall:5.2f}s")

    # --- 3. the overlap ceiling ------------------------------------------
    print("\n== the overlap ceiling: predicted vs measured ==")
    _, _, tm_c, _ = run_epoch(suites[0][1], num_workers=0, train=True)
    model_rate = (len(suites[0][1]) // BATCH * BATCH) / tm_c
    print(f"  model alone (rate M)                     : {model_rate:8.1f} samples/s\n")
    print(f"  {'suite':<8}{'L (loader alone)':>18}{'1/(1/L+1/M)':>14}{'nw=0':>9}"
          f"{'min(4L,M)':>12}{'nw=4':>9}")
    for name, ds, _label in suites:
        L, _, _, _ = run_epoch(ds, num_workers=0, train=False)
        serial = 1 / (1 / L + 1 / model_rate)
        ceiling = min(4 * L, model_rate)
        print(f"  {name:<8}{L:18.1f}{serial:14.1f}{tp_by_suite[name][0]:9.1f}"
              f"{ceiling:12.1f}{tp_by_suite[name][4]:9.1f}")
        for cfg, val in (("loader_alone_1proc", L), ("serial_prediction", serial),
                         ("overlap_ceiling_4w", ceiling)):
            rows.append(dict(section=f"ceiling_{name}", config=cfg,
                             samples_per_sec=round(val, 1), loader_s="", model_s="", wall_s=""))
    rows.append(dict(section="ceiling", config="model_alone",
                     samples_per_sec=round(model_rate, 1), loader_s="", model_s="", wall_s=""))

    heavy = suites[2][1]
    # --- 4. persistent_workers -------------------------------------------
    print("\n== 4. persistent_workers over 10 short epochs (300 images each) ==")
    small = JpegDataset(files[:300])
    for persist in (False, True):
        tp, tl, tm, wall = max((run_epoch(small, num_workers=4, persistent_workers=persist,
                                          epochs=10) for _ in range(2)), key=lambda r: r[0])
        rows.append(dict(section="persistent", config=f"persistent_workers={persist}",
                         samples_per_sec=round(tp, 1), loader_s=round(tl, 2),
                         model_s=round(tm, 2), wall_s=round(wall, 2)))
        print(f"  persistent_workers={str(persist):<5} {tp:8.1f} samples/s   wall {wall:5.2f}s"
              f"   waiting on data {tl:5.2f}s")

    # --- 5. prefetch_factor and pin_memory --------------------------------
    print("\n== 5. prefetch_factor and pin_memory (4 workers, best of 3) ==")
    print("     the spread column is the point: read it before believing a row")
    for pf in (1, 2, 4, 8):
        reps = [run_epoch(heavy, num_workers=4, prefetch_factor=pf)[0] for _ in range(3)]
        rows.append(dict(section="prefetch", config=f"prefetch_factor={pf}",
                         samples_per_sec=round(max(reps), 1), loader_s="",
                         model_s="", wall_s=f"spread {round(max(reps)-min(reps), 1)}"))
        print(f"  prefetch_factor={pf:<2} best {max(reps):7.1f} samples/s"
              f"   spread across 3 repeats {max(reps)-min(reps):6.1f}")
    for pin in (False, True):
        reps = [run_epoch(heavy, num_workers=4, pin_memory=pin)[0] for _ in range(3)]
        rows.append(dict(section="pin", config=f"pin_memory={pin}",
                         samples_per_sec=round(max(reps), 1), loader_s="",
                         model_s="", wall_s=f"spread {round(max(reps)-min(reps), 1)}"))
        print(f"  pin_memory={str(pin):<5} best {max(reps):7.1f} samples/s"
              f"   spread across 3 repeats {max(reps)-min(reps):6.1f}   (no usable GPU here)")

    # --- 6. fork vs spawn --------------------------------------------------
    print("\n== 6. start method: fork vs spawn (4 workers, 300 images, 2 epochs) ==")
    for name in ("fork", "spawn"):
        ctx = mp.get_context(name)
        tp, tl, tm, wall = run_epoch(small, num_workers=4, epochs=2, ctx=ctx)
        rows.append(dict(section="startmethod", config=f"start_method={name}",
                         samples_per_sec=round(tp, 1), loader_s=round(tl, 2),
                         model_s=round(tm, 2), wall_s=round(wall, 2)))
        print(f"  {name:<5} {tp:8.1f} samples/s   wall {wall:5.2f}s")

    # --- write ------------------------------------------------------------
    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "config", "samples_per_sec",
                                          "loader_s", "model_s", "wall_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'}")

    # --- figures ----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    for i, (name, _ds, label) in enumerate(suites):
        ax.plot(WORKER_COUNTS, [tp_by_suite[name][n] for n in WORKER_COUNTS], "o-",
                color=ps.SERIES[i], lw=2, label=label)
    ax.axhline(model_rate, color=ps.INK_MUTED, ls="--", lw=1.5,
               label="model-only ceiling")
    ax.set_title("Throughput vs num_workers", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("num_workers", color=ps.INK_SECONDARY)
    ax.set_ylabel("samples / s", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    idx = np.arange(len(WORKER_COUNTS))
    lo = [r["loader_s"] for r in rows if r["section"] == "workers_heavy"]
    mo = [r["model_s"] for r in rows if r["section"] == "workers_heavy"]
    ax.set_title("Where one epoch goes (heavy suite)", color=ps.INK, fontsize=12, loc="left")
    ax.bar(idx, lo, 0.6, color=ps.SERIES[3], label="main process waiting for data")
    ax.bar(idx, mo, 0.6, bottom=lo, color=ps.SERIES[4], label="main process training")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in WORKER_COUNTS])
    ax.set_xlabel("num_workers", color=ps.INK_SECONDARY)
    ax.set_ylabel("seconds", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    labels, vals = [], []
    for r in rows:
        if r["section"] in ("persistent", "pin", "startmethod"):
            labels.append(r["config"].split("=")[0][:9] + "\n" + r["config"].split("=")[1])
            vals.append(r["samples_per_sec"])
    ax.bar(np.arange(len(vals)), vals, 0.6,
           color=[ps.SERIES[i // 2 % len(ps.SERIES)] for i in range(len(vals))])
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("The other knobs", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("samples / s", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "loader_throughput.png")


if __name__ == "__main__":
    main()
