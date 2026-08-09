"""Project 22 — Memory-mapped tokens.

Tokenizes a text corpus into one flat `.bin` file and trains a small
[GPT-style] language model straight out of `np.memmap`, measuring:

  1. the token file: dtype choice, size, and the uint16 overflow trap
  2. resident memory: full load vs memmap, before and after touching pages
  3. cold vs warm reads (using posix_fadvise to drop the page cache for real)
  4. the nanoGPT memory "leak", and why re-opening the memmap fixes it
  5. random-window sampling: why a memmap dataset needs no shuffling
  6. an actual training run, memmap vs in-RAM, same steps

Runtime ~4 min. Needs torch, numpy, matplotlib.
"""

import csv
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

SEED = 0
VOCAB_CAP = 8000
BLOCK = 64
BATCH = 32
BIG_TOKENS = 150_000_000       # 300 MB as uint16 — big enough to see page faults
TRAIN_STEPS = 400

torch.set_num_threads(4)


# ----------------------------------------------------------------------------
# 0. build the token files
# ----------------------------------------------------------------------------
def corpus_text():
    """Real English prose: every markdown file in this repository."""
    root = Path(__file__).resolve().parents[4]
    parts = []
    for p in sorted(root.rglob("*.md")):
        if "node_modules" in p.parts or ".git" in p.parts:
            continue
        try:
            parts.append(p.read_text(errors="ignore"))
        except OSError:
            pass
    return "\n".join(parts)


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    meta = DATA / "vocab.npy"
    if (DATA / "train.bin").exists() and meta.exists():
        vocab = np.load(meta, allow_pickle=True).item()
        return vocab
    text = corpus_text()
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    common = [w for w, _ in Counter(words).most_common(VOCAB_CAP - 1)]
    vocab = {w: i + 1 for i, w in enumerate(common)}     # 0 = <unk>
    ids = np.array([vocab.get(w, 0) for w in words], dtype=np.int64)
    n_val = len(ids) // 20
    ids[: len(ids) - n_val].astype(np.uint16).tofile(DATA / "train.bin")
    ids[len(ids) - n_val:].astype(np.uint16).tofile(DATA / "val.bin")
    np.save(meta, vocab, allow_pickle=True)
    return vocab


def build_big():
    """A file too big to want in RAM, for the page-cache measurements."""
    path = DATA / "big.bin"
    if path.exists() and path.stat().st_size == BIG_TOKENS * 2:
        return path
    rng = np.random.default_rng(SEED)
    chunk = 10_000_000
    with open(path, "wb") as f:
        for _ in range(BIG_TOKENS // chunk):
            rng.integers(0, VOCAB_CAP, size=chunk, dtype=np.uint16).tofile(f)
    return path


def rss_mb():
    """Resident set size: how much RAM this process is actually holding."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def drop_cache(path):
    """Ask the kernel to forget this file's cached pages. No root needed."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------
# 1. the model
# ----------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d=96, heads=4, layers=2):
        super().__init__()
        torch.manual_seed(SEED)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(BLOCK, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t))
        causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        h = self.blocks(h, mask=causal, is_causal=True)
        return self.head(self.ln(h))


def get_batch(tokens, batch=BATCH, block=BLOCK, rng=None):
    """The nanoGPT pattern: pick random start offsets, slice, shift by one."""
    rng = rng or np.random.default_rng(SEED)
    ix = rng.integers(0, len(tokens) - block - 1, size=batch)
    x = np.stack([tokens[i: i + block] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1: i + 1 + block] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    vocab = build()
    train_path, val_path = DATA / "train.bin", DATA / "val.bin"
    n_train = train_path.stat().st_size // 2
    rows = []

    # --- 1. the token file -------------------------------------------------
    print("== 1. the token file ==")
    print(f"  vocabulary          : {len(vocab)+1} ids (0 = <unk>)")
    print(f"  train.bin           : {n_train:,} tokens, "
          f"{train_path.stat().st_size/1e6:.2f} MB as uint16")
    for name, dt in (("uint16", np.uint16), ("int32", np.int32), ("int64", np.int64)):
        mb = n_train * np.dtype(dt).itemsize / 1e6
        print(f"    as {name:<7}: {mb:8.2f} MB  ({np.dtype(dt).itemsize} bytes/token)")
        rows.append(dict(section="dtype_size", config=name, value=round(mb, 2),
                         note=f"{np.dtype(dt).itemsize} bytes per token"))
    print("  the uint16 ceiling  : 65535. GPT-2's vocab is 50257, so it fits;")
    print("                        a 100k-token vocab does not.")
    over = np.array([65535, 65536, 65540, 131076], dtype=np.int64).astype(np.uint16)
    print("    np.array([65535, 65536, 65540, 131076]).astype(np.uint16) -> "
          f"{[int(v) for v in over]}")
    try:
        a = np.zeros(1, dtype=np.uint16)
        a[0] = 70000
        print("    direct assignment of 70000: no error (unexpected)")
    except Exception as e:
        print(f"    but direct assignment of 70000 -> {type(e).__name__}")
        rows.append(dict(section="dtype_size", config="uint16_overflow",
                         value=str([int(v) for v in over]),
                         note=f"astype wraps silently; assignment raises {type(e).__name__}"))
    t = torch.from_numpy(np.arange(4, dtype=np.uint16))
    try:
        (t + 1)
        print("    torch arithmetic on uint16: works (unexpected)")
    except Exception as e:
        print(f"    torch.from_numpy(uint16) works ({t.dtype}) but t+1 -> "
              f"{type(e).__name__}: {str(e)[:52]}")
        rows.append(dict(section="dtype_size", config="torch_uint16",
                         value=type(e).__name__, note=str(e)[:80]))

    # --- 2. resident memory ------------------------------------------------
    print("\n== 2. RAM: loading vs mapping a 300 MB token file ==")
    big = build_big()
    size_mb = big.stat().st_size / 1e6
    base = base0 = rss_mb()
    print(f"  file on disk                        : {size_mb:.1f} MB")
    print(f"  baseline process RSS                : {base:7.1f} MB")
    arr = np.fromfile(big, dtype=np.uint16)
    after_load = rss_mb()
    print(f"  after np.fromfile (read it all)     : {after_load:7.1f} MB   "
          f"(+{after_load-base:.1f})")
    del arr
    mm = np.memmap(big, dtype=np.uint16, mode="r")
    after_map = rss_mb()
    print(f"  after np.memmap (map it)            : {after_map:7.1f} MB   "
          f"(+{after_map-base:.1f})")
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        i = int(rng.integers(0, len(mm) - BLOCK))
        _ = int(mm[i: i + BLOCK].sum())
    after_touch = rss_mb()
    print(f"  after reading 200 random windows    : {after_touch:7.1f} MB   "
          f"(+{after_touch-base:.1f})")
    _ = int(np.asarray(mm[::4096]).sum())
    after_all = rss_mb()
    print(f"  after touching one byte per 8 KB    : {after_all:7.1f} MB   "
          f"(+{after_all-base:.1f})")
    for cfg, val in (("baseline", base), ("np.fromfile", after_load),
                     ("np.memmap", after_map), ("after_200_windows", after_touch),
                     ("after_striding_whole_file", after_all)):
        rows.append(dict(section="rss", config=cfg, value=round(val, 1),
                         note=f"file {size_mb:.1f} MB"))
    del mm

    # --- 3. cold vs warm ----------------------------------------------------
    print("\n== 3. cold cache vs warm cache (posix_fadvise DONTNEED) ==")
    timings = {}
    for label in ("cold", "warm"):
        if label == "cold":
            drop_cache(big)
        mm = np.memmap(big, dtype=np.uint16, mode="r")
        rng = np.random.default_rng(1)
        ix = rng.integers(0, len(mm) - BLOCK - 1, size=2000)
        t0 = time.perf_counter()
        s = 0
        for i in ix:
            s += int(mm[i: i + BLOCK].sum())
        dt = time.perf_counter() - t0
        timings[label] = dt
        print(f"  2000 random windows, {label:<4} cache : {dt*1000:8.1f} ms  "
              f"({dt/2000*1e6:6.1f} us per window)")
        rows.append(dict(section="cold_warm", config=f"random_{label}",
                         value=round(dt * 1000, 1), note="2000 windows of 64 tokens"))
        del mm
    print(f"  -> a cold random read costs {timings['cold']/timings['warm']:.1f}x "
          f"a warm one; the page cache is doing the real work")
    drop_cache(big)
    mm = np.memmap(big, dtype=np.uint16, mode="r")
    t0 = time.perf_counter()
    s = int(np.asarray(mm[: 2000 * BLOCK]).sum())
    seq = time.perf_counter() - t0
    print(f"  the same {2000*BLOCK:,} tokens read SEQUENTIALLY, cold: {seq*1000:8.1f} ms"
          f"  ({timings['cold']/seq:.1f}x cheaper than random)")
    rows.append(dict(section="cold_warm", config="sequential_cold",
                     value=round(seq * 1000, 1), note=f"{2000*BLOCK} tokens"))
    del mm

    # --- 4. the nanoGPT "leak" ---------------------------------------------
    print("\n== 4. the growing-RSS problem, and the one-line fix ==")
    leak_curve, fix_curve = [], []
    base = rss_mb()
    mm = np.memmap(big, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(2)
    for step in range(3000):
        get_batch(mm, rng=rng)
        if step % 300 == 0:
            leak_curve.append(rss_mb() - base)
    print(f"  one memmap held for 3000 batches : RSS +{leak_curve[-1]:7.1f} MB")
    del mm
    base = rss_mb()
    rng = np.random.default_rng(2)
    for step in range(3000):
        mm = np.memmap(big, dtype=np.uint16, mode="r")   # re-open every batch
        get_batch(mm, rng=rng)
        del mm
        if step % 300 == 0:
            fix_curve.append(rss_mb() - base)
    print(f"  re-opened every batch            : RSS +{fix_curve[-1]:7.1f} MB")
    rows.append(dict(section="leak", config="held_memmap_3000_batches",
                     value=round(leak_curve[-1], 1), note="MB of RSS growth"))
    rows.append(dict(section="leak", config="reopened_each_batch",
                     value=round(fix_curve[-1], 1), note="MB of RSS growth"))

    # --- 5. random windows as the shuffle ----------------------------------
    print("\n== 5. random offsets replace the sampler ==")
    tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    n_windows = len(tokens) - BLOCK - 1
    print(f"  train.bin holds {len(tokens):,} tokens -> {n_windows:,} distinct "
          f"windows of {BLOCK}")
    print(f"  a map-style dataset would need an index list of {n_windows*8/1e6:.1f} MB; "
          f"random offsets need 0")
    rng = np.random.default_rng(3)
    starts = rng.integers(0, n_windows, size=20000)
    dup = 1 - len(set(starts.tolist())) / 20000
    print(f"  20000 random starts: {len(set(starts.tolist())):,} distinct, "
          f"repeat rate {dup:.4f} (birthday-paradox estimate "
          f"{20000/(2*n_windows):.4f})")
    rows.append(dict(section="windows", config="distinct_windows", value=n_windows,
                     note=f"{len(tokens)} tokens, block {BLOCK}, repeat rate "
                          f"{dup:.4f} in 20000 draws"))

    # --- 6. train ----------------------------------------------------------
    print("\n== 6. training the same model two ways ==")
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
    losses = {}
    for label in ("memmap", "in RAM"):
        src = tokens if label == "memmap" else np.array(tokens)
        model = TinyGPT(VOCAB_CAP)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        rng = np.random.default_rng(SEED)
        curve = []
        t0 = time.perf_counter()
        for step in range(TRAIN_STEPS):
            x, y = get_batch(src, rng=rng)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(-1, VOCAB_CAP), y.reshape(-1))
            loss.backward()
            opt.step()
            if step % 20 == 0:
                curve.append(loss.item())
        dt = time.perf_counter() - t0
        model.eval()
        with torch.no_grad():
            vr = np.random.default_rng(99)
            vl = np.mean([float(F.cross_entropy(
                model(vx).reshape(-1, VOCAB_CAP), vy.reshape(-1)))
                for vx, vy in (get_batch(val, rng=vr) for _ in range(10))])
        losses[label] = curve
        print(f"  {label:<7} {TRAIN_STEPS} steps in {dt:5.2f}s "
              f"({dt/TRAIN_STEPS*1000:5.1f} ms/step)  final train loss "
              f"{curve[-1]:.3f}  val loss {vl:.3f}  (perplexity {math.exp(vl):.1f})")
        rows.append(dict(section="training", config=label, value=round(dt, 2),
                         note=f"{dt/TRAIN_STEPS*1000:.1f} ms/step, train {curve[-1]:.4f}, "
                              f"val {vl:.4f}, ppl {math.exp(vl):.2f}"))
    print(f"  uniform-guess loss over {VOCAB_CAP} tokens would be "
          f"{math.log(VOCAB_CAP):.3f}")

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
    labels = ["np.fromfile", "np.memmap", "+200\nwindows", "+whole\nfile"]
    vals = [after_load - base0, after_map - base0, after_touch - base0,
            after_all - base0]
    ax.bar(np.arange(4), vals, 0.6,
           color=[ps.SERIES[2], ps.SERIES[1], ps.SERIES[1], ps.SERIES[1]])
    ax.axhline(size_mb, color=ps.INK_MUTED, ls="--", lw=1.2,
               label=f"file size {size_mb:.0f} MB")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Extra RSS for a 300 MB token file", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_ylabel("MB above baseline", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    xs = np.arange(len(leak_curve)) * 300
    ax.plot(xs, leak_curve, "o-", lw=2, color=ps.SERIES[2], label="one memmap held open")
    ax.plot(xs, fix_curve, "s-", lw=2, color=ps.SERIES[1], label="re-opened each batch")
    ax.set_title("RSS while sampling random batches", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("batches", color=ps.INK_SECONDARY)
    ax.set_ylabel("MB above baseline", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    styles = [dict(lw=4, alpha=0.35), dict(lw=1.6, ls="--")]
    for i, (label, curve) in enumerate(losses.items()):
        ax.plot(np.arange(len(curve)) * 20, curve, color=ps.SERIES[i],
                label=label, **styles[i])
    ax.axhline(math.log(VOCAB_CAP), color=ps.INK_MUTED, ls="--", lw=1.2,
               label="uniform guess")
    ax.set_title("Training from a memory-mapped corpus", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY)
    ax.set_ylabel("loss", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ps.save(fig, OUT / "memmap.png")


if __name__ == "__main__":
    main()
