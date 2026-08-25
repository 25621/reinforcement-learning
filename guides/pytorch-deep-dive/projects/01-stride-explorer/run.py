"""Project 01 — Stride explorer.

Print (shape, stride, storage_offset, is_contiguous, data_ptr) after every
reshape / transpose / permute / slice / expand, and draw the flat storage
buffer that all of those views share.

Everything here is metadata arithmetic, so the whole script runs in about a
second. No training, no downloads.
"""

import csv
import os

import numpy as np
import torch

import plot_style as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# The probe: everything PyTorch will tell you about a tensor's layout.
# --------------------------------------------------------------------------
def probe(name, t, base):
    """One row of the layout table.

    `base` is the tensor we started from. Comparing `data_ptr()` against the
    base tells us whether `t` still points into the same storage buffer.
    """
    itemsize = t.element_size()
    # data_ptr() is the address of element [0,0,...]. Subtracting the base
    # address and dividing by the element size gives storage_offset() back —
    # a nice cross-check that we understand what the number means.
    byte_delta = t.data_ptr() - base.data_ptr()
    return {
        "expression": name,
        "shape": tuple(t.shape),
        "stride": tuple(t.stride()),
        "offset": t.storage_offset(),
        "contiguous": bool(t.is_contiguous()),
        "elements": t.numel(),
        "storage_elems": t.untyped_storage().nbytes() // itemsize,
        "shares_storage": t.untyped_storage().data_ptr()
        == base.untyped_storage().data_ptr(),
        "offset_from_ptr": byte_delta // itemsize,
    }


def show(rows):
    head = (f"{'expression':<36}{'shape':<16}{'stride':<22}"
            f"{'off':>6}{'contig':>8}{'shares':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['expression']:<36}{str(r['shape']):<16}{str(r['stride']):<22}"
            f"{r['offset']:>6}{str(r['contiguous']):>8}{str(r['shares_storage']):>8}"
        )
    print()


# --------------------------------------------------------------------------
# 1. The classic tour: one 3x4 tensor, many views.
# --------------------------------------------------------------------------
def tour_2d():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rows = [
        probe("x = arange(12).reshape(3,4)", x, x),
        probe("x.t()", x.t(), x),
        probe("x.permute(1,0)", x.permute(1, 0), x),
        probe("x.reshape(4,3)", x.reshape(4, 3), x),
        probe("x.view(2,6)", x.view(2, 6), x),
        probe("x.flatten()", x.flatten(), x),
        probe("x[1]", x[1], x),
        probe("x[:, 2]", x[:, 2], x),
        probe("x[:, 1:3]", x[:, 1:3], x),
        probe("x[::2]", x[::2], x),
        probe("x[:, ::2]", x[:, ::2], x),
        probe("x.unsqueeze(0)", x.unsqueeze(0), x),
        probe("x.t().contiguous()", x.t().contiguous(), x),
        probe("x.t().reshape(12)", x.t().reshape(12), x),
        probe("x.diagonal()", x.diagonal(), x),
        probe("x.clone()", x.clone(), x),
    ]
    show(rows)
    return x, rows


# --------------------------------------------------------------------------
# 2. Broadcasting is stride 0. This is the single most surprising row.
# --------------------------------------------------------------------------
def tour_expand():
    col = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    rows = [
        probe("col = arange(3).reshape(3,1)", col, col),
        probe("col.expand(3, 5)", col.expand(3, 5), col),
        probe("col.expand(3, 1000)", col.expand(3, 1000), col),
        probe("col.repeat(1, 5)", col.repeat(1, 5), col),
        probe("col.broadcast_to((4,3,5))", col.broadcast_to((4, 3, 5)), col),
    ]
    show(rows)

    big = col.expand(3, 1_000_000)
    print(f"col.expand(3, 1_000_000): {big.numel():,} logical elements, "
          f"{big.untyped_storage().nbytes()} bytes of real storage")
    print(f"col.repeat(1, 1_000_000): would need "
          f"{col.repeat(1, 1_000_000).untyped_storage().nbytes():,} bytes\n")
    return rows


# --------------------------------------------------------------------------
# 3. A 4-D tensor, the shape every vision model actually uses.
# --------------------------------------------------------------------------
def tour_nchw():
    img = torch.zeros(8, 3, 32, 32)
    rows = [
        probe("img (N,C,H,W)", img, img),
        probe("img.permute(0,2,3,1)  # NHWC", img.permute(0, 2, 3, 1), img),
        probe("img.to(memory_format=channels_last)",
              img.to(memory_format=torch.channels_last), img),
        probe("img.flatten(2)", img.flatten(2), img),
        probe("img[:, 0]", img[:, 0], img),
        probe("img[2:4]", img[2:4], img),
    ]
    show(rows)

    # channels_last has NCHW *shape* but NHWC *strides*: the shape you index
    # with is unchanged, only the memory order moved.
    cl = img.to(memory_format=torch.channels_last)
    print(f"channels_last shape  {tuple(cl.shape)}   stride {tuple(cl.stride())}")
    print(f"permute(0,2,3,1)     {tuple(img.permute(0,2,3,1).shape)}   "
          f"stride {tuple(img.permute(0,2,3,1).stride())}")
    print("  -> same memory order, different logical shape.\n")
    return rows


# --------------------------------------------------------------------------
# 4. Which operations can .view() survive?
# --------------------------------------------------------------------------
def view_survival():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    cases = {
        "x": x,
        "x.transpose(1,2)": x.transpose(1, 2),
        "x.permute(2,0,1)": x.permute(2, 0, 1),
        "x[:, :, ::2]": x[:, :, ::2],       # every 2nd column: a regular lattice
        "x[:, 1:]": x[:, 1:],               # drops rows: leaves gaps of 2 different sizes
        "x[:, :1].expand(2,3,4)": x[:, :1].expand(2, 3, 4),   # a real stride-0 expand
    }
    print(f"{'expression':<26}{'stride':<20}{'contiguous':>12}{'view(-1) works':>18}")
    print("-" * 76)
    out = []
    for name, t in cases.items():
        try:
            t.view(-1)
            ok = True
        except RuntimeError:
            ok = False
        print(f"{name:<26}{str(tuple(t.stride())):<20}"
              f"{str(t.is_contiguous()):>12}{str(ok):>18}")
        out.append((name, tuple(t.stride()), t.is_contiguous(), ok))
    print("\n  Note row 4: NOT contiguous, yet .view(-1) succeeds. Contiguity is a")
    print("  sufficient condition for view, not a necessary one.\n")
    return out


# --------------------------------------------------------------------------
# 5. Does non-contiguity actually cost anything? Measure it.
# --------------------------------------------------------------------------
def contiguity_cost(n=2048, reps=20):
    import time

    # Pin to one thread. With many threads the number bounces around by 2x
    # depending on what else the machine is doing, and we are trying to measure
    # a memory-access pattern, not the OS scheduler.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    a = torch.randn(n, n)
    b = a.t()                      # non-contiguous view
    bc = b.contiguous()            # same numbers, contiguous

    def timed(fn, trials=7):
        # Best-of-N, not mean-of-N. A shared CPU only ever makes a run *slower*
        # (another process steals a core), never faster, so the minimum is the
        # closest thing to the true cost.
        fn()                       # warm up caches / lazy init
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            best = min(best, (time.perf_counter() - t0) / reps * 1e3)
        return best                                       # ms per call

    ms_c = timed(lambda: a.sum(dim=1))
    ms_nc = timed(lambda: b.sum(dim=1))
    ms_fix = timed(lambda: bc.sum(dim=1))
    ms_copy = timed(lambda: b.contiguous())

    torch.set_num_threads(old_threads)

    print(f"(single-threaded, best of 7 x {reps} calls)")
    print(f"row-sum over contiguous {n}x{n}          {ms_c:7.2f} ms")
    print(f"row-sum over transposed view             {ms_nc:7.2f} ms   "
          f"({ms_nc / ms_c:.1f}x slower)")
    print(f"row-sum after .contiguous()              {ms_fix:7.2f} ms")
    print(f"cost of the .contiguous() copy itself    {ms_copy:7.2f} ms")
    breakeven = ms_copy / max(ms_nc - ms_fix, 1e-9)
    print(f"-> copying pays for itself after {breakeven:.1f} reuses\n")
    return {"contig": ms_c, "noncontig": ms_nc, "after_copy": ms_fix,
            "copy": ms_copy, "breakeven_reuses": breakeven}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_storage_map(x):
    """Draw the flat buffer once, then show which cell each view reads."""
    views = [
        ("x  (3,4) stride (4,1)", x),
        ("x.t()  (4,3) stride (1,4)", x.t()),
        ("x[:, 1:3]  (3,2) stride (4,1) off 1", x[:, 1:3]),
        ("x[::2]  (2,4) stride (8,1)", x[::2]),
    ]
    fig, axes = plt.subplots(len(views), 1, figsize=(8.6, 6.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    for ax, (title, v) in zip(axes, views):
        ax.set_facecolor(ps.SURFACE)
        ax.set_xlim(-2.1, 11.7)
        ax.set_ylim(-0.35, 1.0)
        ax.axis("off")
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left", pad=4)

        # Walk the view in the order Python would (row-major over the *view*)
        # and record where each step lands in storage.
        order = {}
        rows, cols = v.shape
        for step, (i, j) in enumerate(
            (i, j) for i in range(rows) for j in range(cols)
        ):
            order[v.storage_offset() + i * v.stride(0) + j * v.stride(1)] = step

        for k in range(12):
            hit = k in order
            ax.add_patch(
                Rectangle(
                    (k - 0.42, 0.1), 0.84, 0.62,
                    facecolor=ps.SERIES[0] if hit else "#ffffff",
                    edgecolor=ps.BASELINE, linewidth=1.0,
                    alpha=0.88 if hit else 1.0,
                )
            )
            if hit:
                ax.text(k, 0.41, str(order[k]), ha="center", va="center",
                        fontsize=10, color="white", weight="bold")
            ax.text(k, -0.18, str(k), ha="center", va="center", fontsize=8,
                    color=ps.INK_MUTED)
        ax.text(-2.0, -0.18, "storage index", fontsize=8, color=ps.INK_MUTED,
                ha="left", va="center")

    fig.suptitle("One 12-element buffer, four views.  Number in the box = the step at which\n"
                 "that view reads that cell;  grey number below = the flat storage index.",
                 color=ps.INK, fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT, "storage_map.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/storage_map.png")


def fig_stride_grid(x):
    """Show the logical grid of a tensor and of its transpose, labelled with
    the flat index each cell maps to."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    for ax, (title, v) in zip(axes, [("x  stride (4,1)", x), ("x.t()  stride (1,4)", x.t())]):
        ax.set_facecolor(ps.SURFACE)
        rows, cols = v.shape
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ps.style_axes(ax)
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left", pad=8)
        for i in range(rows):
            for j in range(cols):
                flat = v.storage_offset() + i * v.stride(0) + j * v.stride(1)
                ax.add_patch(
                    Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                              facecolor=ps.SERIES[1], alpha=0.10 + 0.06 * (flat % 4),
                              edgecolor=ps.BASELINE)
                )
                ax.text(j, i - 0.12, f"[{i},{j}]", ha="center", va="center",
                        fontsize=8, color=ps.INK_SECONDARY)
                ax.text(j, i + 0.20, f"→ {flat}", ha="center", va="center",
                        fontsize=9, color=ps.INK)

    fig.suptitle("Same numbers, same memory — only the index formula changed",
                 color=ps.INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "stride_grid.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/stride_grid.png")


def fig_cost(costs):
    fig, ax = ps.new_axes(6.6, 3.6)
    names = ["contiguous\nrow-sum", "transposed view\nrow-sum", "copy first,\nthen row-sum",
             "the .contiguous()\ncopy alone"]
    vals = [costs["contig"], costs["noncontig"], costs["after_copy"], costs["copy"]]
    colors = [ps.SERIES[1], ps.SERIES[2], ps.SERIES[0], ps.SERIES[3]]
    bars = ax.bar(names, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                va="bottom", fontsize=9, color=ps.INK)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax, "Strides are free; reading against them is not (2048x2048 float32)",
              "", "milliseconds per call", os.path.join(OUT, "contiguity_cost.png"))


# --------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("1. One 3x4 tensor, sixteen ways to look at it")
    print("=" * 78)
    x, rows2d = tour_2d()

    print("=" * 78)
    print("2. expand() = stride 0 = broadcasting, with no memory cost")
    print("=" * 78)
    rows_exp = tour_expand()

    print("=" * 78)
    print("3. A (N,C,H,W) image batch")
    print("=" * 78)
    rows_nchw = tour_nchw()

    print("=" * 78)
    print("4. When does .view() refuse?")
    print("=" * 78)
    view_survival()

    print("=" * 78)
    print("5. What non-contiguity costs")
    print("=" * 78)
    costs = contiguity_cost()

    all_rows = rows2d + rows_exp + rows_nchw
    with open(os.path.join(OUT, "layout_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {OUT}/layout_table.csv")

    with open(os.path.join(OUT, "timings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "ms_per_call"])
        for k, v in costs.items():
            w.writerow([k, f"{v:.3f}"])
    print(f"wrote {OUT}/timings.csv")

    fig_storage_map(x)
    fig_stride_grid(x)
    fig_cost(costs)


if __name__ == "__main__":
    main()
