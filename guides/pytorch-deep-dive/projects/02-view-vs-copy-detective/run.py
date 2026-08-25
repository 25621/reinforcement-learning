"""Project 02 — View vs copy detective.

For ~25 common operations, answer two separate questions:

  1. does the result point at the same storage as its parent?
  2. if I write into the result, does the parent change?

Then three consequences: a silently corrupted dataset, a conditional operation
that is a view on Monday and a copy on Tuesday, and a gradient that comes out
wrong with no error message at all.

Runs in a couple of seconds. No downloads, no training.
"""

import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------
def investigate(name, make):
    """Run `make(parent)` and report how the result relates to its parent.

    `shares`   - same underlying storage buffer (cheap test, metadata only)
    `writes`   - writing into the result actually changes the parent
    `is_view`  - PyTorch's own bookkeeping flag (`_is_view()`)
    """
    parent = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    before = parent.clone()
    child = make(parent)

    shares = (child.untyped_storage().data_ptr()
              == parent.untyped_storage().data_ptr())
    is_view = child._is_view()

    # Try to write. Some results are read-only (a stride-0 expand), and PyTorch
    # refuses rather than corrupting several logical cells at once.
    try:
        with torch.no_grad():
            child.add_(100.0)
        writable = True
        writes = not torch.equal(parent, before)
    except RuntimeError:
        writable = False
        writes = False

    return {
        "expression": name,
        "shares_storage": shares,
        "is_view_flag": is_view,
        "writable": writable,
        "write_reaches_parent": writes,
    }


CASES = [
    ("x.view(6, 4)",              lambda x: x.view(6, 4)),
    ("x.reshape(6, 4)",           lambda x: x.reshape(6, 4)),
    ("x.t()",                     lambda x: x.t()),
    ("x.transpose(0, 1)",         lambda x: x.transpose(0, 1)),
    ("x.permute(1, 0)",           lambda x: x.permute(1, 0)),
    ("x.flatten()",               lambda x: x.flatten()),
    ("x.squeeze()",               lambda x: x.squeeze()),
    ("x.unsqueeze(0)",            lambda x: x.unsqueeze(0)),
    ("x[0]",                      lambda x: x[0]),
    ("x[0:2]",                    lambda x: x[0:2]),
    ("x[:, 1:3]",                 lambda x: x[:, 1:3]),
    ("x[::2]",                    lambda x: x[::2]),
    ("x[[0, 2]]        # fancy",  lambda x: x[[0, 2]]),
    ("x[x > 5]         # mask",   lambda x: x[x > 5]),
    ("x.index_select(0, idx)",    lambda x: x.index_select(0, torch.tensor([0, 2]))),
    ("x.narrow(0, 1, 2)",         lambda x: x.narrow(0, 1, 2)),
    ("x.select(0, 1)",            lambda x: x.select(0, 1)),
    ("x.diagonal()",              lambda x: x.diagonal()),
    ("x.expand_as(x)",            lambda x: x.expand_as(x)),
    ("x[:, :1].expand(4, 6)",     lambda x: x[:, :1].expand(4, 6)),
    ("x.repeat(1, 1)",            lambda x: x.repeat(1, 1)),
    ("x.contiguous()",            lambda x: x.contiguous()),
    ("x.t().contiguous()",        lambda x: x.t().contiguous()),
    ("x.clone()",                 lambda x: x.clone()),
    ("x.detach()",                lambda x: x.detach()),
    ("x.to(torch.float32)",       lambda x: x.to(torch.float32)),
    ("x.to(torch.float64)",       lambda x: x.to(torch.float64)),
    ("x.float()",                 lambda x: x.float()),
    ("x + 0",                     lambda x: x + 0),
    ("x.split(2)[0]",             lambda x: x.split(2)[0]),
    ("x.chunk(2)[0]",             lambda x: x.chunk(2)[0]),
    ("torch.as_strided(x,(2,2),(1,1))",
     lambda x: torch.as_strided(x, (2, 2), (1, 1))),
]


def run_table():
    rows = [investigate(name, fn) for name, fn in CASES]
    head = (f"{'expression':<34}{'shares':>8}{'_is_view':>10}"
            f"{'writable':>10}{'write hits parent':>20}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['expression']:<34}{str(r['shares_storage']):>8}"
              f"{str(r['is_view_flag']):>10}{str(r['writable']):>10}"
              f"{str(r['write_reaches_parent']):>20}")
    print()
    return rows


# --------------------------------------------------------------------------
# 1. The conditional operation: reshape is a view OR a copy
# --------------------------------------------------------------------------
def conditional_reshape():
    print("=" * 78)
    print("reshape(): the same call, two different behaviours")
    print("=" * 78)
    # Two independent tensors so the two writes cannot contaminate each other.
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)      # contiguous
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()  # non-contiguous

    ra = a.reshape(12)
    rb = b.reshape(12)
    print(f"a.is_contiguous()={a.is_contiguous()}  -> a.reshape(12) shares storage: "
          f"{ra.data_ptr() == a.data_ptr()}")
    print(f"b.is_contiguous()={b.is_contiguous()} -> b.reshape(12) shares storage: "
          f"{rb.data_ptr() == b.data_ptr()}")

    ra[0] = -1.0
    rb[0] = -1.0
    print(f"after writing -1 into each result:  a[0,0]={a[0,0].item():6.1f}   "
          f"b[0,0]={b[0,0].item():6.1f}")
    print("  -> identical code. The write escaped into `a` and vanished for `b`.\n")
    return {"contig_shares": bool(ra.data_ptr() == a.data_ptr()),
            "noncontig_shares": bool(rb.data_ptr() == b.data_ptr())}


# --------------------------------------------------------------------------
# 2. dtype conversion that isn't a conversion
# --------------------------------------------------------------------------
def noop_cast():
    print("=" * 78)
    print(".to(dtype): a no-op cast returns the SAME tensor")
    print("=" * 78)
    x = torch.ones(4)
    same = x.to(torch.float32)          # already float32
    other = x.to(torch.float64)
    print(f"x.to(torch.float32) shares storage: {same.data_ptr() == x.data_ptr()}")
    print(f"x.to(torch.float64) shares storage: {other.data_ptr() == x.data_ptr()}")
    same += 99
    print(f"after 'same += 99',  x = {x.tolist()}")
    print("  -> `.to()` only copies when it has to. A defensive `.to(dtype)`")
    print("     is NOT a defensive copy; use .clone() when you mean a copy.\n")
    return {"same_dtype_shares": bool(same.data_ptr() == x.data_ptr())}


# --------------------------------------------------------------------------
# 3. NumPy shares memory with PyTorch (and that surprises people)
# --------------------------------------------------------------------------
def numpy_bridge():
    print("=" * 78)
    print("The NumPy bridge: three constructors, two of them alias")
    print("=" * 78)
    arr = np.ones(5, dtype=np.float32)
    t_from = torch.from_numpy(arr)
    t_as = torch.as_tensor(arr)
    t_ctor = torch.tensor(arr)

    t_from[0] = 7.0
    t_as[1] = 8.0
    t_ctor[2] = 9.0
    print(f"numpy array after writing through all three tensors: {arr.tolist()}")
    print("  torch.from_numpy -> alias (index 0 changed)")
    print("  torch.as_tensor  -> alias (index 1 changed)")
    print("  torch.tensor     -> copy  (index 2 unchanged)\n")
    return {"from_numpy_alias": float(arr[0]) == 7.0,
            "as_tensor_alias": float(arr[1]) == 8.0,
            "tensor_copies": float(arr[2]) == 1.0}


# --------------------------------------------------------------------------
# 4. The silent dataset corruption
# --------------------------------------------------------------------------
def dataset_corruption():
    print("=" * 78)
    print("A realistic bug: normalising a batch corrupts the dataset")
    print("=" * 78)
    dataset = torch.arange(20, dtype=torch.float32).reshape(4, 5)  # 4 samples
    original = dataset.clone()

    def get_batch_buggy(i):
        return dataset[i]                 # a VIEW into the dataset

    def get_batch_safe(i):
        return dataset[i].clone()         # an independent copy

    for i in range(2):
        b = get_batch_buggy(i)
        b -= b.mean()                     # in-place normalisation
    drift_buggy = (dataset - original).abs().sum().item()

    dataset.copy_(original)
    for i in range(2):
        b = get_batch_safe(i)
        b -= b.mean()
    drift_safe = (dataset - original).abs().sum().item()

    print(f"total change to the dataset, view version : {drift_buggy:.1f}")
    print(f"total change to the dataset, clone version: {drift_safe:.1f}")
    print("  -> epoch 2 trains on data epoch 1 quietly rewrote. No error,")
    print("     no warning, and the loss curve still goes down.\n")
    return {"drift_view": drift_buggy, "drift_clone": drift_safe}


# --------------------------------------------------------------------------
# 5. In-place + autograd: one loud failure, one silent one
# --------------------------------------------------------------------------
def autograd_traps():
    print("=" * 78)
    print("In-place writes and autograd")
    print("=" * 78)

    # Reference: the correct gradient of sum(exp(x)) is exp(x).
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    z = x.exp()
    z.sum().backward()
    correct = x.grad.clone()
    print(f"correct  d/dx sum(exp(x)) = {correct.tolist()}")

    # (a) The loud failure: PyTorch's version counter catches the edit.
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    z = x.exp()
    try:
        z.mul_(2.0)          # exp saved its OUTPUT for backward; we just changed it
        z.sum().backward()
        loud = "no error (unexpected)"
    except RuntimeError as e:
        loud = str(e).split("\n")[0][:96]
    print(f"z.mul_(2.0)      -> {loud}")

    # (b) The silent failure: `.data` writes behind autograd's back.
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    z = x.exp()
    z.data.mul_(2.0)         # same edit, version counter never notified
    z.sum().backward()
    silent = x.grad.clone()
    print(f"z.data.mul_(2.0) -> no error, gradient = {[round(v, 4) for v in silent.tolist()]}")
    print(f"                    correct would be     {[round(v, 4) for v in correct.tolist()]}")
    ratio = (silent / correct).mean().item()
    print(f"  -> every gradient is {ratio:.1f}x too large, silently.")
    print("     This is why `.data` is discouraged: it is `.detach()` with the")
    print("     safety check removed.\n")
    return {"correct_grad": correct.tolist(), "silent_grad": silent.tolist(),
            "ratio": ratio, "loud_error": loud}


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fig_table(rows):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.30 * n + 1.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax.set_facecolor(ps.SURFACE)
    ax.set_xlim(0, 3)
    ax.set_ylim(n, -1.2)
    ax.axis("off")

    cols = ["shares\nstorage", "writable", "write reaches\nparent"]
    keys = ["shares_storage", "writable", "write_reaches_parent"]
    for c, name in enumerate(cols):
        ax.text(c + 0.5, -0.75, name, ha="center", va="center", fontsize=8.5,
                color=ps.INK_SECONDARY)

    for r, row in enumerate(rows):
        ax.text(-0.12, r + 0.5, row["expression"], ha="right", va="center",
                fontsize=8.5, color=ps.INK, family="monospace")
        for c, k in enumerate(keys):
            val = row[k]
            ax.add_patch(Rectangle((c + 0.06, r + 0.08), 0.88, 0.84,
                                   facecolor=ps.SERIES[1] if val else ps.SERIES[2],
                                   alpha=0.75, edgecolor="none"))
            ax.text(c + 0.5, r + 0.5, "yes" if val else "no", ha="center",
                    va="center", fontsize=8.5, color="white")

    ax.set_title("Which operations alias, and which quietly copy",
                 color=ps.INK, fontsize=12, loc="left", pad=14, x=-0.30)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "view_vs_copy.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/view_vs_copy.png")


def fig_grad(traps):
    fig, ax = ps.new_axes(6.6, 3.4)
    idx = np.arange(3)
    w = 0.38
    ax.bar(idx - w / 2, traps["correct_grad"], w, color=ps.SERIES[1],
           label="correct gradient")
    ax.bar(idx + w / 2, traps["silent_grad"], w, color=ps.SERIES[2],
           label="after z.data.mul_(2.0)  — no error raised")
    ax.set_xticks(idx)
    ax.set_xticklabels(["x[0]=0.0", "x[1]=1.0", "x[2]=2.0"])
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(ps.INK_SECONDARY)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax, "`.data` writes past autograd's safety check: 2x wrong, silently",
              "", "d/dx of sum(exp(x))", os.path.join(OUT, "silent_wrong_gradient.png"))


# --------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Every operation, two questions: does it share, and does a write escape?")
    print("=" * 78)
    rows = run_table()

    findings = {}
    findings.update(conditional_reshape())
    findings.update(noop_cast())
    findings.update(numpy_bridge())
    findings.update(dataset_corruption())
    traps = autograd_traps()

    with open(os.path.join(OUT, "view_vs_copy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/view_vs_copy.csv")

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in findings.items():
            w.writerow([k, v])
        w.writerow(["silent_grad_ratio", f"{traps['ratio']:.3f}"])
    print(f"wrote {OUT}/findings.csv")

    fig_table(rows)
    fig_grad(traps)


if __name__ == "__main__":
    main()
