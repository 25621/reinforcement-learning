"""Project 03 — Manual indexing.

Given (shape, stride, offset), compute the flat storage index of [i, j, k]
with pen-and-paper arithmetic, and check every single answer against
`.data_ptr()`.

Then use the formula in the other direction: hand PyTorch a stride we invented
ourselves (`torch.as_strided`) and get a sliding-window view of a signal that
costs no memory at all.

Runs in a few seconds. No downloads, no training.
"""

import csv
import itertools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# The formula, written out by hand.
# --------------------------------------------------------------------------
def flat_index(stride, offset, idx):
    """storage_index = offset + sum(idx[d] * stride[d]).

    That is the entire indexing machinery of PyTorch, in one line.
    """
    total = offset
    for i, s in zip(idx, stride):
        total += i * s
    return total


def unravel(stride, offset, flat, shape):
    """The inverse: flat storage index -> logical [i, j, k].

    Only well-defined when the strides are strictly decreasing and gap-free
    (a contiguous tensor); for a transposed or expanded tensor several logical
    positions can map to the same storage cell, so there is no unique answer.
    """
    rest = flat - offset
    idx = []
    for s in stride:
        idx.append(rest // s)
        rest = rest % s
    return tuple(idx)


# --------------------------------------------------------------------------
# 1. Check the formula against the real memory addresses.
# --------------------------------------------------------------------------
def verify_formula():
    base = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    img = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    views = [
        ("base", base),
        ("base.transpose(0, 2)", base.transpose(0, 2)),
        ("base.permute(2, 0, 1)", base.permute(2, 0, 1)),
        ("base[1:3, ::2, 2:]", base[1:3, ::2, 2:]),
        ("base[:, :, ::3]", base[:, :, ::3]),
        ("base[2]", base[2]),
        ("img", img),
        ("img.permute(0, 2, 3, 1)", img.permute(0, 2, 3, 1)),
        ("img.to(channels_last)", img.to(memory_format=torch.channels_last)),
        ("base[:, :1].expand(4, 5, 6)", base[:, :1].expand(4, 5, 6)),
    ]

    print(f"{'view':<30}{'shape':<18}{'stride':<22}{'checks':>8}{'mismatch':>10}")
    print("-" * 88)
    rows, total_checks, total_bad = [], 0, 0
    for name, v in views:
        itemsize = v.element_size()
        stride, offset = tuple(v.stride()), v.storage_offset()
        # `base_ptr` is the address of storage element 0, which is NOT the same
        # as v.data_ptr() whenever storage_offset() is non-zero.
        base_ptr = v.untyped_storage().data_ptr()

        bad = 0
        checks = 0
        for idx in itertools.product(*[range(s) for s in v.shape]):
            predicted = flat_index(stride, offset, idx)
            actual = (v[idx].data_ptr() - base_ptr) // itemsize
            bad += predicted != actual
            checks += 1
        print(f"{name:<30}{str(tuple(v.shape)):<18}{str(stride):<22}"
              f"{checks:>8}{bad:>10}")
        rows.append({"view": name, "shape": tuple(v.shape), "stride": stride,
                     "offset": offset, "checks": checks, "mismatches": bad})
        total_checks += checks
        total_bad += bad

    print(f"\n{total_checks} element addresses predicted by hand, "
          f"{total_bad} mismatches.\n")
    return rows, total_checks, total_bad


# --------------------------------------------------------------------------
# 2. A tensor class in 40 lines, to prove the model is complete.
# --------------------------------------------------------------------------
class MiniTensor:
    """A tensor is (storage, shape, stride, offset). Nothing else is needed
    to index it, transpose it, or slice it."""

    def __init__(self, storage, shape, stride=None, offset=0):
        self.storage = storage                        # a flat python list
        self.shape = tuple(shape)
        self.offset = offset
        self.stride = tuple(stride) if stride is not None \
            else MiniTensor.contiguous_stride(shape)

    @staticmethod
    def contiguous_stride(shape):
        """Row-major: each dimension's stride is the product of all the
        dimensions to its right — i.e. the size of the block it steps over."""
        stride, acc = [], 1
        for s in reversed(shape):
            stride.append(acc)
            acc *= s
        return tuple(reversed(stride))

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self.storage[flat_index(self.stride, self.offset, idx)]

    def transpose(self, a, b):
        shape, stride = list(self.shape), list(self.stride)
        shape[a], shape[b] = shape[b], shape[a]
        stride[a], stride[b] = stride[b], stride[a]
        return MiniTensor(self.storage, shape, stride, self.offset)

    def slice_dim(self, dim, start, stop, step=1):
        shape, stride = list(self.shape), list(self.stride)
        shape[dim] = len(range(start, stop, step))
        offset = self.offset + start * stride[dim]
        stride[dim] = stride[dim] * step
        return MiniTensor(self.storage, shape, stride, offset)

    def is_contiguous(self):
        return self.stride == MiniTensor.contiguous_stride(self.shape)


def verify_minitensor():
    print("=" * 88)
    print("A 40-line tensor class, checked element by element against PyTorch")
    print("=" * 88)
    t = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    m = MiniTensor(t.flatten().tolist(), (3, 4, 5))

    pairs = [
        ("as built", t, m),
        ("transpose(0, 2)", t.transpose(0, 2), m.transpose(0, 2)),
        ("[:, 1:4:2]", t[:, 1:4:2], m.slice_dim(1, 1, 4, 2)),
        ("transpose then slice", t.transpose(1, 2)[:, 2:5],
         m.transpose(1, 2).slice_dim(1, 2, 5)),
    ]
    bad = 0
    for name, tt, mm in pairs:
        assert tuple(tt.shape) == mm.shape, (name, tt.shape, mm.shape)
        assert tuple(tt.stride()) == mm.stride, (name, tt.stride(), mm.stride)
        for idx in itertools.product(*[range(s) for s in mm.shape]):
            bad += float(tt[idx]) != mm[idx]
        print(f"  {name:<24} shape {str(mm.shape):<12} stride {str(mm.stride):<14}"
              f" contiguous={mm.is_contiguous()}")
    print(f"\n  element mismatches vs PyTorch: {bad}")
    print("  -> shape + stride + offset really is the whole story.\n")
    return bad


# --------------------------------------------------------------------------
# 3. Inventing a stride: sliding windows for free.
# --------------------------------------------------------------------------
def sliding_windows():
    print("=" * 88)
    print("as_strided: a stride we chose ourselves")
    print("=" * 88)

    signal = torch.arange(10, dtype=torch.float32)
    win = 4
    # n windows, each `win` long. Step 1 along dim 0 (next window starts one
    # sample later) and 1 along dim 1 (next sample inside the window).
    windows = torch.as_strided(signal, (len(signal) - win + 1, win), (1, 1))
    print(f"signal  {signal.tolist()}")
    print(f"windows shape {tuple(windows.shape)} stride {tuple(windows.stride())}")
    print(windows.numpy())
    same = torch.equal(windows, signal.unfold(0, win, 1))
    print(f"matches signal.unfold(0, {win}, 1): {same}\n")

    # The memory argument, on a realistic size.
    n, w = 1_000_000, 256
    big = torch.arange(n, dtype=torch.float32)
    view = torch.as_strided(big, (n - w + 1, w), (1, 1))
    view_bytes = big.untyped_storage().nbytes()
    copy_bytes = view.numel() * big.element_size()
    print(f"{n:,}-sample signal, {w}-sample windows:")
    print(f"  strided view   : {view_bytes / 1e6:8.1f} MB "
          f"({view.numel():,} logical elements)")
    print(f"  materialised   : {copy_bytes / 1e9:8.1f} GB")
    print(f"  saving         : {copy_bytes / view_bytes:8.0f}x\n")

    # And it is not just theoretical: a moving average, same job both ways.
    k = 20_000
    t0 = time.perf_counter()
    ma = view[:k].mean(dim=1)
    strided_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    ref = torch.stack([big[i:i + w] for i in range(k)]).mean(dim=1)
    loop_ms = (time.perf_counter() - t0) * 1e3
    ok = torch.allclose(ma, ref)
    print(f"  moving average of {k:,} windows, strided view : {strided_ms:7.1f} ms")
    print(f"  moving average of {k:,} windows, python loop : {loop_ms:7.1f} ms "
          f"({loop_ms / strided_ms:.0f}x slower)")
    print(f"  the two results agree: {ok}\n")

    return {"view_bytes": view_bytes, "copy_bytes": copy_bytes,
            "unfold_matches": bool(same), "loop_matches": bool(ok),
            "strided_ms": strided_ms, "loop_ms": loop_ms, "windows_timed": k}


# --------------------------------------------------------------------------
# 4. The sharp edge: as_strided does not check bounds.
# --------------------------------------------------------------------------
def in_bounds(t, shape, stride, offset):
    """Highest storage element the (shape, stride, offset) triple can reach."""
    last = offset + sum((s - 1) * st for s, st in zip(shape, stride))
    capacity = t.untyped_storage().nbytes() // t.element_size()
    return last < capacity, last, capacity


def bounds_demo():
    print("=" * 88)
    print("as_strided trusts you completely")
    print("=" * 88)
    x = torch.arange(20, dtype=torch.float32)
    y = x[5:10]                                   # a 5-element window of x
    print(f"y = x[5:10] -> {y.tolist()}   (numel {y.numel()})")

    over = torch.as_strided(y, (8,), (1,))        # ask for 8 from a 5-long view
    print(f"as_strided(y, (8,), (1,)) -> {over.tolist()}")
    print("  y only has 5 elements. The extra 3 came from x, which y happens")
    print("  to share storage with. No error was raised.")

    ok, last, cap = in_bounds(y, (8,), (1,), y.storage_offset())
    print(f"  bounds check against the STORAGE: last element {last}, "
          f"capacity {cap}, in bounds = {ok}")
    ok2, last2, cap2 = in_bounds(y, (40,), (1,), y.storage_offset())
    print(f"  asking for 40 instead: last element {last2}, capacity {cap2}, "
          f"in bounds = {ok2}  <- this one would read past the buffer\n")
    return {"reachable_8": last, "capacity": cap, "safe_8": ok, "safe_40": ok2}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_windows():
    signal = torch.arange(12, dtype=torch.float32)
    win, step = 4, 1
    n = (len(signal) - win) // step + 1
    windows = torch.as_strided(signal, (n, win), (step, 1))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax.set_facecolor(ps.SURFACE)
    ax.set_xlim(-0.6, 12)
    ax.set_ylim(n + 0.6, -1.4)
    ax.axis("off")

    for k in range(12):
        ax.text(k + 0.5, -1.0, str(k), ha="center", va="center", fontsize=9,
                color=ps.INK_MUTED)
        ax.add_patch(plt.Rectangle((k + 0.06, -0.55), 0.88, 0.5,
                                   facecolor="#ffffff", edgecolor=ps.BASELINE))
        ax.text(k + 0.5, -0.30, str(k), ha="center", va="center", fontsize=8.5,
                color=ps.INK_SECONDARY)
    ax.text(-0.6, -1.0, "storage", ha="left", va="center", fontsize=9,
            color=ps.INK_MUTED)
    ax.text(-0.6, -0.30, "signal", ha="left", va="center", fontsize=9,
            color=ps.INK_MUTED)

    for r in range(n):
        start = r * step
        for c in range(win):
            k = start + c
            ax.add_patch(plt.Rectangle((k + 0.06, r + 0.12), 0.88, 0.72,
                                       facecolor=ps.SERIES[0], alpha=0.80,
                                       edgecolor="none"))
            ax.text(k + 0.5, r + 0.48, str(int(windows[r, c])), ha="center",
                    va="center", fontsize=8.5, color="white")
        ax.text(-0.6, r + 0.48, f"row {r}", ha="left", va="center", fontsize=8.5,
                color=ps.INK_SECONDARY)

    ax.set_title("as_strided(signal, (9, 4), (1, 1)) — 36 logical elements, "
                 "12 real ones\nEvery row overlaps its neighbour by 3 cells "
                 "that exist only once in memory",
                 color=ps.INK, fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sliding_windows.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/sliding_windows.png")


def fig_memory(w):
    fig, ax = ps.new_axes(6.6, 3.6)
    labels = ["strided view\n(as_strided)", "materialised copy\n(stack of windows)"]
    vals = [w["view_bytes"] / 1e6, w["copy_bytes"] / 1e6]
    bars = ax.bar(labels, vals, color=[ps.SERIES[1], ps.SERIES[2]], width=0.55)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        txt = f"{v:.0f} MB" if v < 1000 else f"{v/1000:.1f} GB"
        ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom",
                fontsize=10, color=ps.INK)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax,
              "1M-sample signal, 256-sample sliding windows (log scale)",
              "", "megabytes", os.path.join(OUT, "window_memory.png"))


# --------------------------------------------------------------------------
def main():
    print("=" * 88)
    print("Predicting every memory address by hand")
    print("=" * 88)
    rows, checks, bad = verify_formula()
    verify_minitensor()
    w = sliding_windows()
    b = bounds_demo()

    with open(os.path.join(OUT, "verification.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {OUT}/verification.csv")

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["finding", "value"])
        wr.writerow(["addresses_predicted", checks])
        wr.writerow(["mismatches", bad])
        for k, v in {**w, **b}.items():
            wr.writerow([k, v])
    print(f"wrote {OUT}/findings.csv")

    fig_windows()
    fig_memory(w)


if __name__ == "__main__":
    main()
