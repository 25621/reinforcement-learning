"""Project 05 — Broadcasting bug hunt.

Five expressions that run without an error and compute something other than
what was meant. For each one: the rule that was violated, the shape that gave
it away, and the fix.

Then the cost of bug #1, measured: a linear model trained with the buggy loss
converges to predicting the batch mean for every input.

Runs in a few seconds. No downloads.
"""

import csv
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# A helper that explains a broadcast instead of just doing it.
# --------------------------------------------------------------------------
def explain(shape_a, shape_b):
    """Right-align the two shapes and report what happens dimension by
    dimension. This is the whole broadcasting algorithm, written out."""
    n = max(len(shape_a), len(shape_b))
    a = (1,) * (n - len(shape_a)) + tuple(shape_a)   # missing dims are 1s
    b = (1,) * (n - len(shape_b)) + tuple(shape_b)
    out, notes = [], []
    for da, db in zip(a, b):
        if da == db:
            out.append(da)
            notes.append("match")
        elif da == 1:
            out.append(db)
            notes.append(f"left stretched 1->{db}")
        elif db == 1:
            out.append(da)
            notes.append(f"right stretched 1->{da}")
        else:
            return None, a, b, [f"INCOMPATIBLE {da} vs {db}"]
    return tuple(out), a, b, notes


def report(label, shape_a, shape_b, expected):
    out, a, b, notes = explain(shape_a, shape_b)
    print(f"  {label}")
    print(f"    {str(tuple(shape_a)):>18}  ->  padded to {a}")
    print(f"    {str(tuple(shape_b)):>18}  ->  padded to {b}")
    print(f"    result {out}   expected {expected}   "
          f"{'OK' if out == expected else '<-- WRONG'}")
    for i, note in enumerate(notes):
        if note != "match":
            print(f"      dim {i}: {note}")
    print()
    return out


# --------------------------------------------------------------------------
# BUG 1 — the (N,) / (N,1) loss
# --------------------------------------------------------------------------
def bug1():
    print("=" * 88)
    print("BUG 1  A column of predictions minus a row of targets")
    print("=" * 88)
    N = 8
    target = torch.randn(N)                       # labels loaded as a flat vector
    pred = (target + 0.1 * torch.randn(N)).unsqueeze(1)   # a GOOD model, (N,1)

    report("(pred - target)", pred.shape, target.shape, (N,))

    bad = ((pred - target) ** 2).mean()
    good = ((pred.squeeze(1) - target) ** 2).mean()
    print(f"    buggy loss {bad.item():.4f}   correct loss {good.item():.4f}   "
          f"ratio {bad.item() / good.item():.1f}x")
    print(f"    (target variance is {target.var(unbiased=False).item():.4f} - the")
    print("     buggy loss cannot go below roughly 2x that, no matter how good")
    print("     the model gets. A bad model hides the bug; a good one exposes it.)")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        F.mse_loss(pred, target)
        msg = str(w[0].message).split(".")[0] if w else "(no warning)"
    print(f"    F.mse_loss warns: {msg}\n")
    return {"bug1_buggy_loss": bad.item(), "bug1_correct_loss": good.item(),
            "bug1_ratio": bad.item() / good.item()}


# --------------------------------------------------------------------------
# BUG 2 — mean() without keepdim
# --------------------------------------------------------------------------
def bug2():
    print("=" * 88)
    print("BUG 2  Row-normalising with a mean that lost its dimension")
    print("=" * 88)
    # Square on purpose: this bug is SILENT only when rows == columns.
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    report("x - x.mean(dim=1)", x.shape, x.mean(dim=1).shape, (4, 4))

    bad = x - x.mean(dim=1)                       # (3,4) - (3,) -> (3,4), wrong axis
    good = x - x.mean(dim=1, keepdim=True)
    print(f"    row means after the buggy version : "
          f"{[round(v, 3) for v in bad.mean(dim=1).tolist()]}")
    print(f"    row means after the correct version: "
          f"{[round(v, 3) for v in good.mean(dim=1).tolist()]}")
    print("    The buggy version subtracted the mean of row j from column j.")
    print("    Every row still has a non-zero mean, which is the thing we asked")
    print("    it to remove - and no error was raised.\n")
    return {"bug2_row_means_buggy": [round(v, 4) for v in bad.mean(dim=1).tolist()],
            "bug2_row_means_correct": [round(v, 4)
                                       for v in good.mean(dim=1).tolist()]}


# --------------------------------------------------------------------------
# BUG 3 — a per-channel bias that lands on the width axis
# --------------------------------------------------------------------------
def bug3():
    print("=" * 88)
    print("BUG 3  Per-channel scaling applied to the width of an image")
    print("=" * 88)
    img = torch.zeros(1, 3, 8, 3)                 # N,C,H,W with W == C == 3
    img[:, 0] = 0.2
    img[:, 1] = 0.5
    img[:, 2] = 0.8
    gain = torch.tensor([2.0, 1.0, 0.5])          # one gain per COLOUR channel

    report("img * gain", img.shape, gain.shape, "per-channel gain")

    bad = img * gain                              # aligns with W, not C
    good = img * gain.view(1, 3, 1, 1)
    print(f"    channel means, buggy : "
          f"{[round(v, 3) for v in bad.mean(dim=(0, 2, 3)).tolist()]}")
    print(f"    channel means, correct: "
          f"{[round(v, 3) for v in good.mean(dim=(0, 2, 3)).tolist()]}")
    print("    Both ran. The buggy one multiplied column 0 of every channel by")
    print("    2.0 instead of multiplying the red channel by 2.0.")
    print("    This is only silent because W happens to equal C. Change the")
    print("    image to 8x4 and the same line raises a shape error.\n")

    wide = torch.zeros(1, 3, 8, 4)
    try:
        wide * gain
        crashes = False
    except RuntimeError:
        crashes = True
    print(f"    with W=4 instead of 3, does `img * gain` raise? {crashes}\n")
    return {"bug3_channel_means_buggy":
            [round(v, 4) for v in bad.mean(dim=(0, 2, 3)).tolist()],
            "bug3_channel_means_correct":
            [round(v, 4) for v in good.mean(dim=(0, 2, 3)).tolist()],
            "bug3_raises_when_W_differs": crashes,
            "bug3_bad_img": bad, "bug3_good_img": good}


# --------------------------------------------------------------------------
# BUG 4 — a pairwise distance matrix that is only a diagonal
# --------------------------------------------------------------------------
def bug4():
    print("=" * 88)
    print("BUG 4  Pairwise distances that are not pairwise")
    print("=" * 88)
    a = torch.randn(5, 3)
    b = torch.randn(5, 3)

    report("(a - b).pow(2).sum(-1)", a.shape, b.shape, (5, 5))

    bad = (a - b).pow(2).sum(-1)                     # (5,) - elementwise pairs
    good = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
    print(f"    buggy result shape {tuple(bad.shape)}, correct {tuple(good.shape)}")
    print(f"    the buggy vector equals the diagonal of the correct matrix: "
          f"{torch.allclose(bad, good.diagonal())}")
    print("    You asked for 25 distances and got the 5 that happen to line up.")
    print("    With 5 queries and 7 keys the same line raises - so the bug hides")
    print("    exactly when your two sets are the same size, which in a")
    print("    self-attention or retrieval setting is most of the time.\n")
    return {"bug4_buggy_shape": tuple(bad.shape),
            "bug4_correct_shape": tuple(good.shape),
            "bug4_is_diagonal": bool(torch.allclose(bad, good.diagonal()))}


# --------------------------------------------------------------------------
# BUG 5 — an attention mask that masks the wrong axis
# --------------------------------------------------------------------------
def bug5():
    print("=" * 88)
    print("BUG 5  A padding mask applied to queries instead of keys")
    print("=" * 88)
    B = T = 4                                       # the coincidence that hides it
    scores = torch.zeros(B, 1, T, T)
    mask = torch.zeros(B, T)                        # 1 = padding
    mask[0, 3] = 1.0
    mask[1, 2:] = 1.0

    report("scores + mask * -1e9", scores.shape, mask.shape, (B, 1, T, T))
    print("    Note the shape check says OK. It is supposed to: the result really")
    print("    is (B, 1, T, T). Broadcasting checked the sizes and had nothing to")
    print("    say about which axis MEANS what - that part is only in your head.\n")

    bad = scores + mask * -1e9                      # (B,1,T,T) vs (B,T) -> aligns
    good = scores + mask[:, None, None, :] * -1e9   # the intended broadcast

    bad_masked = (bad < -1e8)[0, 0]
    good_masked = (good < -1e8)[0, 0]
    print(f"    batch item 0, positions masked (buggy) :\n{bad_masked.int().numpy()}")
    print(f"    batch item 0, positions masked (correct):\n{good_masked.int().numpy()}")
    print("    The correct version blanks a COLUMN (nobody may attend TO the pad).")
    print("    The buggy version used the batch dimension as if it were a query")
    print("    position, so every item in the batch got a different, wrong mask.")
    print("    It is silent only because batch size happens to equal sequence")
    print("    length - a coincidence that is common in toy tests and rare in")
    print("    production, which is how this ships.\n")
    return {"bug5_buggy_mask": bad_masked.int().tolist(),
            "bug5_correct_mask": good_masked.int().tolist(),
            "bug5_bad": bad_masked, "bug5_good": good_masked}


# --------------------------------------------------------------------------
# What bug 1 costs: train the same model twice.
# --------------------------------------------------------------------------
def training_cost(steps=400):
    print("=" * 88)
    print("What bug 1 costs: the same model, the same data, two loss lines")
    print("=" * 88)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 1, generator=g)
    y = (3.0 * x.squeeze(1) + 1.0 + 0.1 * torch.randn(256, generator=g))

    results = {}
    for tag in ["correct", "buggy"]:
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        curve = []
        for _ in range(steps):
            pred = model(x)                          # (256, 1)
            if tag == "correct":
                loss = ((pred.squeeze(1) - y) ** 2).mean()
            else:
                loss = ((pred - y) ** 2).mean()      # (256,1) vs (256,) -> (256,256)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                true_mse = ((model(x).squeeze(1) - y) ** 2).mean().item()
            curve.append(true_mse)
        w = model.weight.item()
        b = model.bias.item()
        print(f"  {tag:<8} slope {w:7.4f} (true 3.0)   intercept {b:7.4f} "
              f"(true 1.0)   real MSE {curve[-1]:8.4f}")
        results[tag] = {"curve": curve, "w": w, "b": b, "mse": curve[-1],
                        "pred": model(x).squeeze(1).detach()}

    print(f"\n  y has mean {y.mean().item():.4f} and the buggy model's intercept is")
    print(f"  {results['buggy']['b']:.4f} with slope {results['buggy']['w']:.4f}:")
    print("  it learned to predict the batch mean and ignore the input entirely.")
    print("  Reason: averaging over all N*N pairs turns the loss into")
    print("  var(y) + mean_i (pred_i - mean(y))^2, and the only way to lower that")
    print("  is to push every prediction to mean(y). The loss went down the whole")
    print("  time; the model learned nothing.\n")
    return results, x, y


# --------------------------------------------------------------------------
# Defences
# --------------------------------------------------------------------------
def defences():
    print("=" * 88)
    print("Three ways to make these bugs loud")
    print("=" * 88)
    a, b = (8, 1), (8,)
    print(f"  1. Ask first:   torch.broadcast_shapes({a}, {b}) = "
          f"{tuple(torch.broadcast_shapes(a, b))}")

    def strict_sub(u, v):
        if u.shape != v.shape:
            raise ValueError(f"strict_sub: {tuple(u.shape)} != {tuple(v.shape)}")
        return u - v

    try:
        strict_sub(torch.zeros(8, 1), torch.zeros(8))
        out = "no error"
    except ValueError as e:
        out = str(e)
    print(f"  2. Refuse to broadcast in the places that matter: {out}")

    x = torch.zeros(8, 1)
    y = torch.zeros(8)
    ok = F.mse_loss(x, y.unsqueeze(1))
    print(f"  3. Give the loss matching shapes on purpose: "
          f"F.mse_loss(pred, target.unsqueeze(1)) = {ok.item():.4f}, no warning\n")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_training(results, x, y):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    ax = axes[0]
    ps.style_axes(ax)
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.plot(results["correct"]["curve"], color=ps.SERIES[1], label="correct loss line")
    ax.plot(results["buggy"]["curve"], color=ps.SERIES[2], label="broadcast bug")
    ax.set_yscale("log")
    ax.set_title("True MSE during training", color=ps.INK, fontsize=11, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("mean squared error", color=ps.INK_SECONDARY, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ps.INK_SECONDARY)

    ax = axes[1]
    ps.style_axes(ax)
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    xs = x.squeeze(1).numpy()
    order = np.argsort(xs)
    ax.scatter(xs, y.numpy(), s=9, color=ps.INK_MUTED, alpha=0.5, label="data")
    ax.plot(xs[order], results["correct"]["pred"].numpy()[order],
            color=ps.SERIES[1], lw=2, label="correct model")
    ax.plot(xs[order], results["buggy"]["pred"].numpy()[order],
            color=ps.SERIES[2], lw=2, label="bugged model")
    ax.set_title("What each model learned", color=ps.INK, fontsize=11, loc="left")
    ax.set_xlabel("x", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("y", color=ps.INK_SECONDARY, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ps.INK_SECONDARY)

    ps.save(fig, os.path.join(OUT, "training_cost.png"))


def fig_masks(b5):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (title, m) in zip(axes, [("buggy: scores + mask * -1e9", b5["bug5_bad"]),
                                     ("correct: mask[:, None, None, :]",
                                      b5["bug5_good"])]):
        ax.set_facecolor(ps.SURFACE)
        ax.imshow(m.numpy(), cmap="Reds", vmin=0, vmax=1)
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("key position", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel("query position", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "X" if m[i, j] else ".", ha="center", va="center",
                        fontsize=10, color=ps.INK)
    fig.suptitle("Batch item 0: which attention scores got blanked out",
                 color=ps.INK, fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "mask_axes.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/mask_axes.png")


def fig_channels(b3):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    imgs = [("original", torch.stack([torch.full((8, 3), v) for v in
                                      (0.2, 0.5, 0.8)]).unsqueeze(0)),
            ("buggy: img * gain", b3["bug3_bad_img"]),
            ("correct: gain.view(1,3,1,1)", b3["bug3_good_img"])]
    for ax, (title, im) in zip(axes, imgs):
        ax.set_facecolor(ps.SURFACE)
        rgb = im[0].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(rgb, interpolation="nearest", aspect="auto")
        ax.set_title(title, color=ps.INK, fontsize=9.5, loc="left", pad=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["W0", "W1", "W2"], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(colors=ps.INK_MUTED)
    fig.suptitle("A 3-wide RGB image. The bug brightens a COLUMN; the fix "
                 "brightens a CHANNEL.",
                 color=ps.INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT, "channel_axes.png"), facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/channel_axes.png")


# --------------------------------------------------------------------------
def main():
    print("=" * 88)
    print("The rule: line the shapes up from the RIGHT. Each pair of dimensions")
    print("must be equal, or one of them must be 1. Missing dimensions count as 1.")
    print("=" * 88 + "\n")

    findings = {}
    findings.update(bug1())
    findings.update(bug2())
    b3 = bug3()
    findings.update({k: v for k, v in b3.items() if not torch.is_tensor(v)})
    findings.update(bug4())
    b5 = bug5()
    findings.update({k: v for k, v in b5.items() if not torch.is_tensor(v)})
    results, x, y = training_cost()
    findings["train_correct_slope"] = round(results["correct"]["w"], 4)
    findings["train_buggy_slope"] = round(results["buggy"]["w"], 4)
    findings["train_correct_mse"] = round(results["correct"]["mse"], 4)
    findings["train_buggy_mse"] = round(results["buggy"]["mse"], 4)
    findings["y_mean"] = round(y.mean().item(), 4)
    defences()

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in findings.items():
            w.writerow([k, v])
    print(f"wrote {OUT}/findings.csv")

    fig_training(results, x, y)
    fig_masks(b5)
    fig_channels(b3)


if __name__ == "__main__":
    main()
