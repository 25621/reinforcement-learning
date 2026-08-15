"""Project 33 -- Format sweep.

Train one tiny transformer seven times, once per numeric format, and measure what
each format costs in quality and in speed. Runs in about 6 minutes on 12 CPU
threads.
"""

import json
import math
import os
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import formats as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.path.dirname(OUT), "data", "input.txt")
os.makedirs(os.path.dirname(DATA), exist_ok=True)
results = {}


def log(*a):
    print(*a, flush=True)


# ============================================================ A. verification
def section_a():
    log("\n=== A. Does our from-scratch caster agree with the hardware format? ===")
    torch.manual_seed(0)
    x = torch.cat([torch.randn(50000) * 3, torch.randn(50000) * 1e-3,
                   torch.randn(50000) * 1e3])
    rows = []
    for fmt in [F.BF16, F.FP16, F.E4M3, F.E5M2]:
        ref = x.to(fmt.torch_dtype).float()
        mine = fmt.cast(x)
        same = (ref == mine) | (ref.isnan() & mine.isnan())
        rows.append({"format": fmt.name, "values": x.numel(),
                     "mismatches": int((~same).sum())})
        log(f"  {fmt.name:10s} mismatches vs torch.{fmt.torch_dtype}: "
            f"{int((~same).sum())} / {x.numel()}")
    results["verification"] = rows


# ================================================= B. measured format anatomy
def section_b():
    log("\n=== B. Format anatomy, measured rather than quoted ===")
    rows = [f.describe() for f in F.ALL]
    for r in rows:
        log(f"  {r['name']:10s} {r['bits']:2d} bits  1+{r['exp']}+{r['mant']:2d}  "
            f"max={r['max']:.3g}  min_normal={r['min_normal']:.3g}  "
            f"eps={r['eps']:.3g}  steps in [1,2)={r['values_in_[1,2)']}")
    results["anatomy"] = rows

    log("\n  Bit layout of 0.1 in FP32 (sign exponent mantissa):")
    for v in [0.1, 1.0, -2.5]:
        s, e, m, bits = F.bits_of(v)
        log(f"    {v:>5} -> {bits}   (exponent field {e} = 2^{e - 127}, "
            f"mantissa {m}/2^23)")

    log("\n  What happens at the edges of FP8 E4M3 (max = 448):")
    probe = torch.tensor([0.3, 1e-4, 448.0, 500.0])
    edge = {}
    for fmt in [F.BF16, F.FP16, F.E4M3, F.E5M2]:
        vals = fmt.cast(probe).tolist()
        edge[fmt.name] = vals
        log(f"    {fmt.name:10s} {['%.6g' % v for v in vals]}")
    results["edges"] = {"inputs": probe.tolist(), "outputs": edge}

    log("\n  Every value FP4 E2M1 can represent (positives):", F.grid(F.E2M1))
    results["fp4_grid"] = F.grid(F.E2M1)

    # A float format's *relative* error does not depend on how big the numbers
    # are -- until they fall off one end of the exponent range, at which point it
    # goes to 100%. Sweeping the same tensor across 13 orders of magnitude makes
    # that flat-then-cliff shape visible, and it is the whole argument for
    # keeping a scale factor next to an FP8 tensor.
    log("\n  Relative error of the same tensor at different magnitudes:")
    torch.manual_seed(1)
    base = torch.randn(20000)
    sweep = {}
    exps = list(range(-10, 6))
    for fmt in [F.BF16, F.FP16, F.E4M3, F.E5M2]:
        errs = []
        for k in exps:
            x = base * (10.0 ** k)
            q = fmt.cast(x)
            q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
            errs.append(float((x - q).abs().sum() / x.abs().sum()))
        sweep[fmt.name] = errs
        log(f"    {fmt.name:10s} " +
            " ".join(f"{e:.3f}" for e in errs))
    log(f"    magnitudes: " + " ".join(f"1e{k:<3d}" for k in exps))
    results["scale_sweep"] = {"exponents": exps, "rel_error": sweep}

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, errs in sweep.items():
        ax.plot(exps, errs, marker="o", markersize=3, label=name)
    ax.set_xlabel("tensor magnitude (values are N(0,1) x 10^k)")
    ax.set_ylabel("mean relative error")
    ax.set_title("Precision is flat inside the range and 100% outside it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/scale_sweep.png", dpi=120)
    plt.close(fig)


# =============================================== C. the value grid, as a plot
def section_c():
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.2), constrained_layout=True)
    for ax, (fmt, hi) in zip(axes, [(F.E2M1, 6.5), (F.E4M3, 6.5), (F.FP16, 6.5)]):
        pts = F.grid(fmt, limit=hi)
        ax.plot(pts, [0] * len(pts), "|", markersize=18, color="#1f77b4")
        ax.set_yticks([])
        ax.set_xlim(-0.2, hi)
        ax.set_title(f"{fmt.name}: {len(pts)} representable values in [0, {hi}]",
                     fontsize=10)
    axes[-1].set_xlabel("value")
    fig.suptitle("Low-precision formats are dense near zero and sparse far from it",
                 fontsize=12)
    fig.savefig(f"{OUT}/value_grid.png", dpi=120)
    plt.close(fig)
    log(f"\n  wrote {OUT}/value_grid.png")


# ============================================= D. what the silicon actually does
def section_d():
    log("\n=== D. Matmul speed per dtype on this CPU ===")
    n = 1024
    rows = []
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        a = torch.randn(n, n, dtype=dtype)
        b = torch.randn(n, n, dtype=dtype)
        t = time.perf_counter()
        a @ b
        one = time.perf_counter() - t
        reps = max(2, min(30, int(1.5 / max(one, 1e-4))))
        t = time.perf_counter()
        for _ in range(reps):
            a @ b
        dt = (time.perf_counter() - t) / reps
        gflops = 2 * n ** 3 / dt / 1e9
        rows.append({"dtype": str(dtype), "ms": dt * 1e3, "gflops": gflops})
        log(f"  {str(dtype):16s} {dt * 1e3:7.2f} ms   {gflops:7.1f} GFLOP/s")
    fastest = max(rows, key=lambda r: r["gflops"])
    log(f"  fastest: {fastest['dtype']} "
        f"({fastest['gflops'] / rows[0]['gflops']:.2f}x vs fp32)")
    results["matmul_speed"] = rows


# ================================================================= the model
class QLinear(nn.Linear):
    """A Linear layer whose weights and inputs are pushed through a format."""
    fmt = None
    scaled = False

    def forward(self, x):
        if QLinear.fmt is None or QLinear.fmt is F.FP32:
            return super().forward(x)
        w = F.ste_cast(self.weight, QLinear.fmt, QLinear.scaled)
        xq = F.ste_cast(x, QLinear.fmt, QLinear.scaled)
        return Fn.linear(xq, w, self.bias)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = QLinear(d, 3 * d, bias=False)
        self.proj = QLinear(d, d, bias=False)
        self.fc1 = QLinear(d, 4 * d, bias=False)
        self.fc2 = QLinear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        shape = (B, T, self.h, D // self.h)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(Fn.gelu(self.fc1(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d=96, h=4, layers=3, block=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                targets.reshape(-1))
        return logits, loss


def get_data():
    if not os.path.exists(DATA):
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, DATA)
    text = open(DATA).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:], len(chars)


def batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


# =========================================== E. same model, six numeric formats
CONFIGS = [
    ("FP32", F.FP32, False),
    ("BF16", F.BF16, False),
    ("FP16", F.FP16, False),
    ("FP8 E4M3 (no scale)", F.E4M3, False),
    ("FP8 E4M3 (scaled)", F.E4M3, True),
    ("FP4 E2M1 (no scale)", F.E2M1, False),
    ("FP4 E2M1 (scaled)", F.E2M1, True),
]
STEPS, BS, BLOCK = 600, 32, 64


def train_one(label, fmt, scaled, train, val, vocab):
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(99)
    model = GPT(vocab, block=BLOCK)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    QLinear.fmt, QLinear.scaled = fmt, scaled
    curve, t0 = [], time.perf_counter()
    for step in range(STEPS):
        x, y = batch(train, BS, BLOCK, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == STEPS - 1:
            curve.append((step, float(loss)))
    dt = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        vgen = torch.Generator().manual_seed(7)
        vloss = sum(float(model(*batch(val, BS, BLOCK, vgen))[1])
                    for _ in range(10)) / 10
    QLinear.fmt = None
    log(f"  {label:22s} val loss {vloss:.5f}   ppl {math.exp(vloss):6.2f}   "
        f"{dt:5.1f}s")
    return {"label": label, "val_loss": vloss, "ppl": math.exp(vloss),
            "seconds": dt, "curve": curve}, model


def section_e():
    log("\n=== E. Training the same tiny transformer in six formats ===")
    train, val, vocab = get_data()
    rows, models = [], {}
    for label, fmt, scaled in CONFIGS:
        row, model = train_one(label, fmt, scaled, train, val, vocab)
        rows.append(row)
        models[label] = model
    results["training"] = rows

    plot_training(rows)
    return models["FP32"], train


# ============================================ F. why FP16 needs a loss scale
def plot_training(rows):
    """Draw the section-E figure; also callable from `run.py --plot`."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    for row in rows:
        xs = [c[0] for c in row["curve"]]
        ys = [c[1] for c in row["curve"]]
        axes[0].plot(xs, ys, label=row["label"], linewidth=1.6)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Loss curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = [r["label"] for r in rows]
    losses = [r["val_loss"] for r in rows]
    axes[1].barh(labels, losses,
                 color=["#d62728" if v > losses[0] * 1.05 else "#1f77b4"
                        for v in losses])
    axes[1].axvline(losses[0], ls="--", color="k", lw=1)
    axes[1].set_xlim(1.8, max(losses) * 1.03)
    axes[1].set_xlabel(f"final validation loss after {STEPS} steps "
                       f"(lower is better; dashed = FP32)")
    axes[1].set_title("Only FP4 without a scale is visibly worse")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3, axis="x")
    for label, v in zip(labels, losses):
        axes[1].annotate(f"{v:.5f}", (v, label), fontsize=8,
                         xytext=(5, -3), textcoords="offset points")
    fig.suptitle("Same model, same seed, same data -- "
                 "only the number format changes")
    fig.savefig(f"{OUT}/format_sweep.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/format_sweep.png")


def section_f(model, train):
    log("\n=== F. Gradient underflow: the reason FP16 ships with a loss scaler ===")
    QLinear.fmt = None
    gen = torch.Generator().manual_seed(5)
    x, y = batch(train, BS, BLOCK, gen)
    rows = []
    for scale in [1.0, 1024.0]:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        (loss * scale).backward()
        grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                           if p.grad is not None]) / scale
        for fmt in [F.BF16, F.FP16, F.E4M3]:
            # A loss scaler multiplies the loss (and so every gradient) by a
            # constant before the backward pass, casts, then divides it out.
            # Only the *storage* is scaled, so this is free -- it just moves the
            # gradients up into the part of the format that has resolution.
            g = fmt.cast(grads * scale)
            zeros = float(((g == 0) & (grads != 0)).float().mean())
            rows.append({"scale": scale, "format": fmt.name,
                         "flushed_to_zero": zeros})
            log(f"  loss scale {scale:6.0f}  {fmt.name:10s} "
                f"gradients flushed to zero: {zeros * 100:6.3f}%")
    results["gradient_underflow"] = rows
    smallest = float(grads.abs()[grads != 0].min())
    log(f"  smallest non-zero gradient magnitude: {smallest:.3g}")
    log(f"  FP16 smallest subnormal: {F.FP16.min_subnormal:.3g}   "
        f"BF16: {F.BF16.min_subnormal:.3g}   E4M3: {F.E4M3.min_subnormal:.3g}")
    results["smallest_grad"] = smallest


def main():
    torch.set_num_threads(12)
    torch.manual_seed(0)
    t0 = time.time()
    section_a()
    section_b()
    section_c()
    section_d()
    model, train = section_e()
    section_f(model, train)
    results["total_seconds"] = time.time() - t0
    with open(f"{OUT}/findings.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"\nwrote {OUT}/findings.json   total {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:   # redraw section E from the committed findings.json
        torch.set_num_threads(1)
        plot_training(json.load(open(f"{OUT}/findings.json"))["training"])
    else:
        main()
