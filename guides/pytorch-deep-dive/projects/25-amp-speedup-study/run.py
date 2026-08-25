"""Project 25 — AMP speedup study.

Mixed precision on a machine with no 16-bit hardware. The headline speed result
inverts, and the parts of AMP that are about *numbers* rather than about silicon
all still reproduce exactly:

  1. fp32 vs autocast(bf16) vs autocast(fp16): the honest timing on this CPU
  2. what autocast actually casts — the per-operation policy, read off dtypes
  3. activation memory: what 16-bit storage really saves
  4. range: fp16 overflows at 65504, bf16 does not
  5. underflow: how many gradients fp16 flushes to zero, and what a scale fixes
  6. GradScaler for real — the scale trajectory, and a skipped step
  7. accuracy: 400 training steps in emulated fp32 / bf16 / fp16

Runtime ~4 min. Needs torch, numpy, matplotlib.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24-profile-a-training-step"))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

rows = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<40} {value:>14}  {note}")


# ---------------------------------------------------------------------------
# 1. the timing everyone expects to be a win
# ---------------------------------------------------------------------------
def timing_study():
    print("[1] fp32 vs autocast, forward + backward (batch 4 x seq 64)")
    model = P.new_model()
    x, y = P.make_batch(batch=4, seq=64)

    def step(dtype=None):
        def go():
            if dtype is None:
                loss = P.loss_fn(model(x), y)
            else:
                with torch.autocast("cpu", dtype=dtype):
                    loss = P.loss_fn(model(x), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        return go

    fp32, sp32 = P.best_of(step(None), repeats=5, warmup=2)
    record("timing", "fp32 fwd+bwd (ms)", f"{fp32:.1f}", f"spread {sp32:.1f}")
    for dt, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        ms, sp = P.best_of(step(dt), repeats=2, warmup=1)
        record("timing", f"autocast {name} fwd+bwd (ms)", f"{ms:.1f}", f"spread {sp:.1f}")
        record("timing", f"autocast {name} speedup", f"{fp32/ms:.2f}x",
               "below 1.0 means slower")

    # pure 16-bit, no autocast — separates "autocast overhead" from "the CPU has
    # no 16-bit arithmetic"
    m16 = P.new_model().bfloat16()

    def step16():
        loss = P.loss_fn(m16(x), y)
        loss.backward()
        m16.zero_grad(set_to_none=True)
    ms, sp = P.best_of(step16, repeats=2, warmup=1)
    record("timing", "model.bfloat16() fwd+bwd (ms)", f"{ms:.1f}",
           f"{fp32/ms:.2f}x vs fp32 — no autocast involved")

    # the raw matmul, so the cause is unambiguous
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    flops = 2 * 1024 ** 3
    for dt, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        aa, bb = a.to(dt), b.to(dt)
        reps = 5 if dt == torch.float32 else 2
        ms, _ = P.best_of(lambda: aa @ bb, repeats=reps, warmup=1)
        record("matmul", f"1024^3 matmul {name} (ms)", f"{ms:.1f}",
               f"{flops / (ms / 1e3) / 1e9:.1f} GFLOP/s")
    return fp32


# ---------------------------------------------------------------------------
# 2. autocast's per-operation policy, read off the dtypes it produces
# ---------------------------------------------------------------------------
def policy_study():
    print("\n[2] what autocast casts, and what it refuses to cast")
    model = P.new_model()
    x, y = P.make_batch()
    seen = {}

    def hook(mod, inp, out):
        if torch.is_tensor(out):
            seen[type(mod).__name__] = out.dtype
    handles = [m.register_forward_hook(hook) for m in model.modules()]

    a = torch.randn(64, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = P.loss_fn(logits, y)
        probes = {
            "matmul (fp32 inputs)": (a @ a).dtype,
            "softmax (fp32 input)": torch.softmax(a, dim=-1).dtype,
            "log (fp32 input)": torch.log(a.abs()).dtype,
            "layer_norm (fp32 input)": F.layer_norm(a, (64,)).dtype,
            "gelu (fp32 input)": F.gelu(a).dtype,
        }
    for h in handles:
        h.remove()

    for name, dt in seen.items():
        record("policy", f"{name} output dtype", str(dt).replace("torch.", ""),
               "cast to 16-bit" if dt == torch.bfloat16 else "kept in fp32")
    record("policy", "logits dtype", str(logits.dtype).replace("torch.", ""))
    record("policy", "cross_entropy loss dtype", str(loss.dtype).replace("torch.", ""),
           "reductions stay fp32")
    for name, dt in probes.items():
        record("policy", name, str(dt).replace("torch.", ""),
               "16-bit list" if dt == torch.bfloat16 else "fp32 list / follows input")
    for p in model.parameters():
        record("policy", "parameter dtype", str(p.dtype).replace("torch.", ""),
               "master weights are never cast")
        break


# ---------------------------------------------------------------------------
# 3. what 16-bit really saves: activation bytes
# ---------------------------------------------------------------------------
def memory_study():
    print("\n[3] activation memory")
    model = P.new_model()
    x, y = P.make_batch()
    out = {}
    for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
        model.zero_grad(set_to_none=True)
        with P.ActivationBytes(model) as tracker:
            if dt is None:
                loss = P.loss_fn(model(x), y)
            else:
                with torch.autocast("cpu", dtype=dt):
                    loss = P.loss_fn(model(x), y)
            peak = tracker.peak
        loss.backward()
        out[name] = peak
        record("memory", f"saved activations {name}", P.human(peak))
    record("memory", "bf16 / fp32", f"{out['bf16']/out['fp32']:.2f}x",
           "half would be 0.50")
    return out


# ---------------------------------------------------------------------------
# 4-5. range and underflow: the two failure modes, measured
# ---------------------------------------------------------------------------
def numerics_study():
    print("\n[4] range")
    record("range", "fp16 largest finite", f"{torch.finfo(torch.float16).max:.0f}")
    record("range", "bf16 largest finite", f"{torch.finfo(torch.bfloat16).max:.3e}")
    record("range", "fp32 largest finite", f"{torch.finfo(torch.float32).max:.3e}")
    record("range", "fp16 smallest normal", f"{torch.finfo(torch.float16).tiny:.3e}")
    record("range", "bf16 smallest normal", f"{torch.finfo(torch.bfloat16).tiny:.3e}")
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16),
                     ("fp32", torch.float32)):
        eps = torch.finfo(dt).eps
        record("range", f"{name} eps (gap next to 1.0)", f"{eps:.3e}",
               f"= 2^-{int(round(-np.log2(eps)))} → {int(round(-np.log2(eps)))} mantissa bits")

    big = torch.full((512, 512), 300.0)
    prod = (big @ big / 512)
    record("range", "300x300 accumulated in fp16",
           "inf" if torch.isinf(big.half() @ big.half() / 512).any() else "finite",
           f"true value {prod[0,0].item():.0f}")
    record("range", "same in bf16",
           "inf" if torch.isinf(big.bfloat16() @ big.bfloat16() / 512).any() else "finite")

    print("\n[5] underflow, and what a loss scale fixes")
    model = P.new_model()
    x, y = P.make_batch()
    # a small loss is the realistic case: late in training, or any auxiliary term
    (P.loss_fn(model(x), y) * 1e-4).backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    nz = g != 0
    record("underflow", "gradient elements", f"{g.numel()}")
    record("underflow", "median |grad|", f"{g[nz].abs().median():.3e}")
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        lost = ((g.to(dt) == 0) & nz).sum().item()
        record("underflow", f"flushed to zero in {name}",
               f"{100*lost/nz.sum().item():.2f} %", f"{lost} of {nz.sum().item()}")
    scales = [1, 2 ** 8, 2 ** 12, 2 ** 16]
    curve = []
    for s in scales:
        lost = (((g * s).to(torch.float16) == 0) & nz).sum().item()
        pct = 100 * lost / nz.sum().item()
        curve.append(pct)
        record("underflow", f"fp16 zeros at scale 2^{int(np.log2(s))}", f"{pct:.2f} %")
    return scales, curve


# ---------------------------------------------------------------------------
# 6. GradScaler for real: the scale trajectory and a skipped step
# ---------------------------------------------------------------------------
def scaler_study():
    print("\n[6] GradScaler")
    model = P.new_model(seed=1, d=128, n_layer=2, seq=64)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", init_scale=2. ** 16, growth_interval=8)
    gen = torch.Generator().manual_seed(7)
    trajectory, skipped = [], []
    before = None
    for step in range(40):
        x, y = P.make_batch(batch=8, seq=64, gen=gen)
        loss = P.loss_fn(model(x), y)
        if step == 20:
            # plant a gradient of 1e35: finite on its own, but inf once the
            # scaler multiplies it — exactly the situation the scaler exists for
            loss = loss + model.head.weight.sum() * 1e35
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        trajectory.append(scaler.get_scale())
        w = model.head.weight.detach().clone()
        scaler.step(opt)
        scaler.update()
        moved = (model.head.weight - w).abs().max().item()
        if moved == 0.0:
            skipped.append(step)
        if step == 20:
            before = moved
    record("scaler", "initial scale", f"{trajectory[0]:.0f}")
    record("scaler", "scale after 8 clean steps", f"{trajectory[9]:.0f}",
           "growth_interval=8 → doubles")
    record("scaler", "scale at the overflow step", f"{trajectory[20]:.0f}")
    record("scaler", "scale after the overflow", f"{trajectory[21]:.0f}", "halved")
    record("scaler", "weight change on that step", f"{before:.1e}", "step skipped")
    record("scaler", "skipped steps", str(skipped))
    return trajectory


# ---------------------------------------------------------------------------
# 7. does 16-bit arithmetic actually cost accuracy?
# ---------------------------------------------------------------------------
class FakeCastLinear(nn.Module):
    """A Linear whose matmul inputs are rounded to `dtype` and back.

    This is what a Tensor Core does: multiply 16-bit inputs, accumulate in fp32.
    Rounding in fp32 storage keeps the arithmetic at full CPU speed while
    reproducing the precision loss exactly.
    """

    def __init__(self, lin, dtype):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.dtype = dtype

    def forward(self, x):
        if self.dtype is None:
            return F.linear(x, self.weight, self.bias)
        return F.linear(x.to(self.dtype).float(), self.weight.to(self.dtype).float(),
                        self.bias)


def swap_linears(module, dtype):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeCastLinear(child, dtype))
        else:
            swap_linears(child, dtype)


def accuracy_study(steps=400):
    print("\n[7] training accuracy under emulated precision")
    curves = {}
    for name, dt in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        model = P.new_model(seed=0, d=128, n_layer=2, seq=64)
        swap_linears(model, dt)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(0)
        losses = []
        t0 = time.perf_counter()
        for _ in range(steps):
            x, y = P.make_batch(batch=16, seq=64, gen=gen)
            loss = P.loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves[name] = losses
        tail = float(np.mean(losses[-20:]))
        record("accuracy", f"{name} final loss (mean last 20)", f"{tail:.6f}",
               f"{time.perf_counter()-t0:.0f} s")
    base = float(np.mean(curves["fp32"][-20:]))
    for name in ("bf16", "fp16"):
        gap = float(np.mean(curves[name][-20:])) - base
        record("accuracy", f"{name} - fp32 (final)", f"{gap:+.6f}")
        step_gap = np.abs(np.array(curves[name]) - np.array(curves["fp32"]))
        record("accuracy", f"{name} largest single-step gap", f"{step_gap.max():.4f}",
               f"at step {int(step_gap.argmax())} — the curves do differ")
    return curves


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | CPU only\n")
    timing_study()
    policy_study()
    memory_study()
    scales, curve = numerics_study()
    trajectory = scaler_study()
    curves = accuracy_study()

    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    ax.plot(np.arange(len(scales)), curve, "o-", color=ps.SERIES[2], lw=1.8)
    ax.set_xticks(np.arange(len(scales)))
    ax.set_xticklabels([f"2^{int(np.log2(s))}" for s in scales])
    ax.set_title("fp16 gradients lost to underflow", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("loss scale", color=ps.INK_SECONDARY)
    ax.set_ylabel("% of gradients flushed to zero", color=ps.INK_SECONDARY)

    ax = axes[1]
    ax.semilogy(trajectory, color=ps.SERIES[0], lw=1.8)
    ax.axvline(20, color=ps.SERIES[2], ls="--", lw=1.2)
    ax.text(20.6, max(trajectory) * 0.6, "overflow", color=ps.SERIES[2], fontsize=9)
    ax.set_title("GradScaler finds the largest safe scale", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY)
    ax.set_ylabel("loss scale", color=ps.INK_SECONDARY)

    ax = axes[2]
    for i, (name, losses) in enumerate(curves.items()):
        sm = np.convolve(losses, np.ones(20) / 20, mode="valid")
        ax.plot(sm, color=ps.SERIES[i], lw=1.6, label=name)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Precision does not move the loss curve", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY)
    ax.set_ylabel("loss (20-step mean)", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "amp_speedup_study.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
