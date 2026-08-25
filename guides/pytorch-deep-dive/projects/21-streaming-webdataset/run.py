"""Project 21 — Streaming WebDataset.

Trains from sharded `.tar` archives without unpacking them, and measures the
four things that actually differ from a map-style dataset:

  1. loose files vs tar shards: read throughput and what the OS is doing
  2. sharding across workers — with and without a worker splitter
  3. the shuffle buffer: streaming can only shuffle a window, and window size
     changes what a batch looks like (and what the model learns)
  4. what you lose: `len()`, indexing, a clean epoch boundary, even shard splits

Runtime ~1 min. Needs torch, numpy, matplotlib, Pillow, webdataset.
"""

import csv
import io
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
LOOSE = DATA / "loose"
SHARDS = DATA / "shards"
OUT.mkdir(exist_ok=True)

SEED = 0
N_IMAGES = 4000
N_SHARDS = 8
IMG = 48
N_CLASSES = 4
BATCH = 32

torch.set_num_threads(4)


# ----------------------------------------------------------------------------
# 0. build the corpus twice: as loose files, and as tar shards
# ----------------------------------------------------------------------------
def render(i, rng):
    """A tiny image whose class is legible from its colour and blob position.

    Classes are laid out in BLOCKS inside each shard (125 of class 0, then 125
    of class 1, ...). Real crawled datasets look like this — a shard is usually
    one source, one crawl batch, one category — and it is exactly the layout
    that makes the shuffle buffer matter.
    """
    cls = (i % (N_IMAGES // N_SHARDS)) // (N_IMAGES // N_SHARDS // N_CLASSES)
    a = rng.integers(70, 190, size=(IMG, IMG, 3), dtype=np.uint8)
    cy, cx = rng.integers(10, IMG - 10, size=2)
    yy, xx = np.ogrid[:IMG, :IMG]
    a[(yy - cy) ** 2 + (xx - cx) ** 2 < 8**2] = np.array(
        [[30, 235][cls & 1], [30, 235][(cls >> 1) & 1], 120], dtype=np.uint8)
    return a, cls


def build():
    if (SHARDS / f"train-{N_SHARDS-1:04d}.tar").exists():
        return
    if DATA.exists():
        shutil.rmtree(DATA)
    LOOSE.mkdir(parents=True)
    SHARDS.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    per_shard = N_IMAGES // N_SHARDS
    i = 0
    for s in range(N_SHARDS):
        tar_path = SHARDS / f"train-{s:04d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for _ in range(per_shard):
                a, cls = render(i, rng)
                buf = io.BytesIO()
                Image.fromarray(a).save(buf, format="JPEG", quality=88)
                raw = buf.getvalue()
                key = f"{i:06d}"
                # a WebDataset "sample" = consecutive members sharing a basename
                info = tarfile.TarInfo(f"{key}.jpg")
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
                label = str(cls).encode()
                info = tarfile.TarInfo(f"{key}.cls")
                info.size = len(label)
                tar.addfile(info, io.BytesIO(label))
                (LOOSE / f"{key}.jpg").write_bytes(raw)
                (LOOSE / f"{key}.cls").write_bytes(label)
                i += 1


class LooseFiles(Dataset):
    """The map-style baseline: two file opens per sample."""

    def __init__(self):
        self.keys = sorted(p.stem for p in LOOSE.glob("*.jpg"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        img = Image.open(LOOSE / f"{k}.jpg").convert("RGB")
        y = int((LOOSE / f"{k}.cls").read_text())
        return to_tensor(img), y


def to_tensor(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def decode_sample(sample):
    img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    return to_tensor(img), int(sample["cls"])


def shard_urls():
    return str(SHARDS / f"train-{{0000..{N_SHARDS-1:04d}}}.tar")


def make_wds(shuffle_buf=0, workersplitter=wds.split_by_worker, shardshuffle=False):
    ds = wds.WebDataset(shard_urls(), shardshuffle=shardshuffle, seed=SEED,
                        workersplitter=workersplitter, empty_check=False)
    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf)
    return ds.map(decode_sample)


def tag_worker(sample):
    """Return which worker process produced this sample."""
    info = torch.utils.data.get_worker_info()
    return -1 if info is None else info.id


def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, N_CLASSES))


def main():
    build()
    loose_bytes = sum(p.stat().st_size for p in LOOSE.iterdir())
    n_loose = len(list(LOOSE.iterdir()))
    shard_bytes = sum(p.stat().st_size for p in SHARDS.iterdir())
    rows = []
    print(f"loose : {n_loose} files, {loose_bytes/1e6:.2f} MB")
    print(f"shards: {N_SHARDS} files, {shard_bytes/1e6:.2f} MB "
          f"({N_IMAGES//N_SHARDS} samples each)")
    rows.append(dict(section="corpus", config="loose_files", value=n_loose,
                     note=f"{loose_bytes/1e6:.2f} MB"))
    rows.append(dict(section="corpus", config="shard_files", value=N_SHARDS,
                     note=f"{shard_bytes/1e6:.2f} MB"))

    # --- 1. read throughput ------------------------------------------------
    print("\n== 1. reading the same 4000 samples, two ways ==")
    speeds = {}
    for nw in (0, 4):
        loader = DataLoader(LooseFiles(), batch_size=BATCH, shuffle=True,
                            num_workers=nw, drop_last=True,
                            generator=torch.Generator().manual_seed(SEED))
        best = min(timed(loader) for _ in range(2))
        speeds[f"loose nw={nw}"] = N_IMAGES / best
        print(f"  loose files, num_workers={nw}   {best:5.2f}s  "
              f"{N_IMAGES/best:8.1f} samples/s")
        rows.append(dict(section="throughput", config=f"loose_nw{nw}",
                         value=round(N_IMAGES / best, 1), note=f"{best:.2f}s"))
    for nw in (0, 4):
        loader = wds.WebLoader(make_wds().batched(BATCH), batch_size=None,
                               num_workers=nw)
        best = min(timed(loader) for _ in range(2))
        speeds[f"wds nw={nw}"] = N_IMAGES / best
        print(f"  tar shards,  num_workers={nw}   {best:5.2f}s  "
              f"{N_IMAGES/best:8.1f} samples/s")
        rows.append(dict(section="throughput", config=f"wds_nw{nw}",
                         value=round(N_IMAGES / best, 1), note=f"{best:.2f}s"))
    print(f"  -> streaming is {speeds['wds nw=0']/speeds['loose nw=0']:.2f}x the "
          f"single-process rate of loose files")

    print("\n== 1b. the part that does not depend on the page cache ==")
    keys = sorted(p.stem for p in LOOSE.glob("*.jpg"))
    t0 = time.perf_counter()
    for k in keys:
        (LOOSE / f"{k}.jpg").read_bytes()
        (LOOSE / f"{k}.cls").read_bytes()
    t_open = time.perf_counter() - t0
    t0 = time.perf_counter()
    for s_ in range(N_SHARDS):
        (SHARDS / f"train-{s_:04d}.tar").read_bytes()
    t_tar = time.perf_counter() - t0
    print(f"  open+read {n_loose} loose files : {t_open*1000:7.1f} ms "
          f"({t_open/n_loose*1e6:.1f} us each)")
    print(f"  open+read {N_SHARDS} tar shards    : {t_tar*1000:7.1f} ms")
    print(f"  -> {t_open/t_tar:.1f}x more time in file-open overhead alone, "
          f"and {n_loose//N_SHARDS}x more filesystem lookups")
    rows.append(dict(section="open_overhead", config="loose_files_ms",
                     value=round(t_open * 1000, 1), note=f"{n_loose} files"))
    rows.append(dict(section="open_overhead", config="tar_shards_ms",
                     value=round(t_tar * 1000, 1), note=f"{N_SHARDS} files"))

    # --- 2. sharding across workers ---------------------------------------
    print("\n== 2. who reads what: sharding across workers ==")
    for label, splitter in (("split_by_worker (default)", wds.split_by_worker),
                            ("workersplitter=None", None)):
        ds = make_wds(workersplitter=splitter)
        loader = wds.WebLoader(ds, batch_size=None, num_workers=4)
        keys = [int(y) for _, y in loader]
        seen = len(keys)
        print(f"  {label:<26} yielded {seen:5d} samples for a {N_IMAGES}-sample "
              f"epoch  ({seen/N_IMAGES:.1f}x)")
        rows.append(dict(section="sharding", config=label, value=seen,
                         note=f"{seen/N_IMAGES:.2f}x the dataset"))
    print(f"  how the {N_SHARDS} shards divide, counted per worker:")
    for nworkers in (2, 3, 5, 12):
        ds = wds.WebDataset(shard_urls(), shardshuffle=False, seed=SEED,
                            empty_check=False).map(tag_worker)
        loader = wds.WebLoader(ds, batch_size=None, num_workers=nworkers)
        per = Counter(int(w) for w in loader)
        counts = [per.get(i, 0) for i in range(nworkers)]
        idle = sum(1 for c in counts if c == 0)
        print(f"    {nworkers:2d} workers -> {counts}"
              f"   total {sum(counts)}"
              + (f"   {idle} worker(s) got NOTHING" if idle else ""))
        rows.append(dict(section="shard_split", config=f"{N_SHARDS}shards_{nworkers}workers",
                         value=sum(counts), note=f"per worker {counts}"))

    # --- 3. the shuffle buffer --------------------------------------------
    print("\n== 3. the shuffle buffer: how mixed is a batch? ==")
    mix = {}
    for buf in (0, 32, 128, 512, 2000):
        ds = make_wds(shuffle_buf=buf, shardshuffle=N_SHARDS)
        loader = wds.WebLoader(ds.batched(BATCH), batch_size=None, num_workers=0)
        distinct = []
        for _, y in loader:
            distinct.append(len(set(y.tolist())))
        mix[buf] = float(np.mean(distinct))
        print(f"  shuffle buffer {buf:<5} mean distinct classes per batch of "
              f"{BATCH}: {mix[buf]:.2f}  (max {N_CLASSES})")
        rows.append(dict(section="shuffle_buffer", config=f"buffer={buf}",
                         value=round(mix[buf], 3), note=f"batch size {BATCH}"))

    print("\n== 3b. and what that does to training ==")
    accs = {}
    val = LooseFiles()
    vx = torch.stack([val[i][0] for i in range(0, len(val), 7)])
    vy = torch.tensor([val[i][1] for i in range(0, len(val), 7)])
    for buf in (0, 128, 2000):
        ds = make_wds(shuffle_buf=buf, shardshuffle=N_SHARDS)
        loader = wds.WebLoader(ds.batched(BATCH), batch_size=None, num_workers=0)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        lossf = nn.CrossEntropyLoss()
        for _ in range(2):
            for x, y in loader:
                opt.zero_grad(set_to_none=True)
                lossf(model(x), y).backward()
                opt.step()
        with torch.no_grad():
            accs[buf] = float((model(vx).argmax(1) == vy).float().mean())
        print(f"  shuffle buffer {buf:<5} test accuracy after 2 epochs: {accs[buf]:.3f}")
        rows.append(dict(section="shuffle_training", config=f"buffer={buf}",
                         value=round(accs[buf], 4), note="2 epochs"))

    # --- 4. what streaming takes away --------------------------------------
    print("\n== 4. what an IterableDataset cannot do ==")
    ds = make_wds()
    for what, fn in (("len(dataset)", lambda: len(ds)),
                     ("dataset[0]", lambda: ds[0])):
        try:
            fn()
            print(f"  {what:<14} -> works (unexpected)")
        except Exception as e:
            print(f"  {what:<14} -> {type(e).__name__}: {str(e).splitlines()[0][:70]}")
            rows.append(dict(section="no_random_access", config=what,
                             value=type(e).__name__, note=str(e).splitlines()[0][:90]))
    sized = make_wds().with_epoch(1000)
    print(f"  .with_epoch(n) gives the loader a length again: "
          f"{sum(1 for _ in sized)} samples per 'epoch'")
    rows.append(dict(section="no_random_access", config="with_epoch",
                     value=1000, note="declared, not derived"))

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "config", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'}")

    # --- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    names = list(speeds)
    ax.bar(np.arange(len(names)), [speeds[n] for n in names], 0.6,
           color=[ps.SERIES[0], ps.SERIES[0], ps.SERIES[1], ps.SERIES[1]])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_title("Read throughput", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("samples / s", color=ps.INK_SECONDARY)

    ax = axes[1]
    bufs = list(mix)
    ax.plot(np.arange(len(bufs)), [mix[b] for b in bufs], "o-", lw=2,
            color=ps.SERIES[0])
    ax.axhline(N_CLASSES, color=ps.INK_MUTED, ls="--", lw=1.2,
               label=f"all {N_CLASSES} classes")
    ax.set_xticks(np.arange(len(bufs)))
    ax.set_xticklabels([str(b) for b in bufs])
    ax.set_ylim(0, N_CLASSES + 0.3)
    ax.set_title("Distinct classes per batch", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("shuffle buffer size", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ks = list(accs)
    ax.bar(np.arange(len(ks)), [accs[k] for k in ks], 0.6,
           color=[ps.SERIES[2] if accs[k] < 0.6 else ps.SERIES[1] for k in ks])
    ax.axhline(1 / N_CLASSES, color=ps.INK_MUTED, ls="--", lw=1.2, label="chance")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"buffer\n{k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy after 2 streamed epochs", color=ps.INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ps.save(fig, OUT / "webdataset.png")


def timed(loader):
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
