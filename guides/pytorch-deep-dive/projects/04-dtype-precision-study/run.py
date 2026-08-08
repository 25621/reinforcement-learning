"""Project 04 — dtype precision study.

Add up one million small numbers in float16, bfloat16, float32 and float64,
and take the error apart into the two things people usually conflate:

  * representation error - what the dtype does to each number before you add
  * accumulation  error - what the dtype does to the running total

Then the consequences: why a `GradScaler` exists, and why bfloat16 is the
default on modern hardware even though it is *less* precise than float16.

Runs in about a minute (the naive summation loops are deliberately
element-by-element Python). No downloads, no training.
"""

import csv
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
N = 1_000_000
VALUE = 0.001                       # true sum = 1000.0 exactly


# --------------------------------------------------------------------------
# bfloat16 is literally the top half of a float32, so we can do it in numpy.
# --------------------------------------------------------------------------
def to_bf16(x):
    """Round a float32 to bfloat16 precision, round-to-nearest-even.

    A float32 is 1 sign + 8 exponent + 23 mantissa bits. A bfloat16 is 1 + 8 + 7:
    the same exponent field, with 16 mantissa bits chopped off the bottom. So
    "convert to bfloat16" = "keep the top 16 bits of the float32".
    """
    u = np.float32(x).view(np.uint32) if np.isscalar(x) else \
        np.asarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even before truncating: add half an ulp, plus the
    # low bit of the surviving mantissa to break ties towards even.
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))
    u = u & np.uint32(0xFFFF0000)
    return u.view(np.float32)


# --------------------------------------------------------------------------
# 1. What each dtype can actually represent.
# --------------------------------------------------------------------------
def dtype_facts():
    print("=" * 92)
    print("What each dtype can hold")
    print("=" * 92)
    rows = []
    specs = [
        (torch.float64, 11, 52), (torch.float32, 8, 23),
        (torch.bfloat16, 8, 7), (torch.float16, 5, 10),
    ]
    head = (f"{'dtype':<12}{'bytes':>6}{'exp':>5}{'mant':>6}{'eps':>12}"
            f"{'max':>12}{'min normal':>13}{'decimal digits':>16}")
    print(head)
    print("-" * len(head))
    for dt, exp_bits, mant_bits in specs:
        fi = torch.finfo(dt)
        digits = np.log10(2 ** (mant_bits + 1))
        print(f"{str(dt).replace('torch.',''):<12}{fi.bits // 8:>6}{exp_bits:>5}"
              f"{mant_bits:>6}{fi.eps:>12.3e}{fi.max:>12.3e}"
              f"{fi.smallest_normal:>13.3e}{digits:>16.1f}")
        rows.append({"dtype": str(dt), "bytes": fi.bits // 8,
                     "exponent_bits": exp_bits, "mantissa_bits": mant_bits,
                     "eps": fi.eps, "max": fi.max,
                     "smallest_normal": fi.smallest_normal,
                     "decimal_digits": round(digits, 2)})
    print()
    print("float16 and bfloat16 are both 2 bytes. float16 spends 5 bits on range")
    print("and 10 on precision; bfloat16 spends 8 and 7. Same size, opposite bet.\n")
    return rows


# --------------------------------------------------------------------------
# 2. Representation error: the damage done before a single addition.
# --------------------------------------------------------------------------
def representation_error():
    print("=" * 92)
    print(f"Storing {VALUE} one million times, before adding anything")
    print("=" * 92)
    rows = []
    for dt in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
        stored = torch.full((1,), VALUE, dtype=dt).double().item()
        exact_total = stored * N
        rel = abs(stored - VALUE) / VALUE
        print(f"  {str(dt).replace('torch.',''):<10} stores it as {stored:.12f}  "
              f"(relative error {rel:9.2e})  -> a perfect sum would be "
              f"{exact_total:10.4f}")
        rows.append({"dtype": str(dt), "stored_value": stored,
                     "relative_error": rel, "perfect_sum": exact_total})
    print("\n  Even with a flawless adder, bfloat16 cannot reach 1000.0: the value")
    print("  it is adding is not 0.001. This error has nothing to do with summing.\n")
    return rows


# --------------------------------------------------------------------------
# 3. Accumulation error: one number at a time, the way a beginner would.
# --------------------------------------------------------------------------
def naive_sum(dtype_name, n=N, value=VALUE, checkpoints=None):
    """Add `value` to a running total `n` times, keeping the total in a
    low-precision type the whole way. Records partial sums as it goes."""
    checkpoints = checkpoints or []
    cp, out = set(checkpoints), {}

    if dtype_name == "bfloat16":
        acc = np.float32(0.0)
        v = to_bf16(np.float32(value))
        for i in range(1, n + 1):
            acc = to_bf16(acc + v)
            if i in cp:
                out[i] = float(acc)
    else:
        np_dt = {"float16": np.float16, "float32": np.float32,
                 "float64": np.float64}[dtype_name]
        acc = np_dt(0.0)
        v = np_dt(value)
        for i in range(1, n + 1):
            acc = np_dt(acc + v)
            if i in cp:
                out[i] = float(acc)
    return float(acc), out


def kahan_sum(n=N, value=VALUE):
    """Kahan compensated summation in float16.

    Named after William Kahan, the numerical analyst who designed IEEE 754.
    The idea: after `total + value` rounds, work out how much was thrown away
    and carry it into the next addition, so the lost crumbs are not lost.
    """
    total = np.float16(0.0)
    comp = np.float16(0.0)          # the running crumb
    v = np.float16(value)
    for _ in range(n):
        y = np.float16(v - comp)
        t = np.float16(total + y)
        comp = np.float16(np.float16(t - total) - y)   # what rounding ate
        total = t
    return float(total)


def accumulation_study():
    print("=" * 92)
    print(f"Adding {VALUE} to a running total {N:,} times (true answer: 1000.0)")
    print("=" * 92)
    cps = [10 ** k for k in range(1, 7)]
    curves, rows = {}, []
    for name in ["float16", "bfloat16", "float32", "float64"]:
        t0 = time.perf_counter()
        total, partials = naive_sum(name, checkpoints=cps)
        dt = time.perf_counter() - t0
        err = abs(total - 1000.0) / 1000.0
        print(f"  naive loop, {name:<9} -> {total:12.4f}   "
              f"relative error {err:9.2e}   ({dt:.1f}s)")
        curves[name] = partials
        rows.append({"method": f"naive loop ({name})", "result": total,
                     "rel_error": err})
    print()

    # Same dtype, better algorithm.
    t0 = time.perf_counter()
    k = kahan_sum()
    print(f"  Kahan loop, float16   -> {k:12.4f}   "
          f"relative error {abs(k - 1000.0) / 1000.0:9.2e}   "
          f"({time.perf_counter() - t0:.1f}s)")
    rows.append({"method": "Kahan loop (float16)", "result": k,
                 "rel_error": abs(k - 1000.0) / 1000.0})

    # What PyTorch actually does.
    print()
    for dt in [torch.float16, torch.bfloat16, torch.float32]:
        x = torch.full((N,), VALUE, dtype=dt)
        s = x.sum().double().item()
        s32 = x.sum(dtype=torch.float32).double().item()
        name = str(dt).replace("torch.", "")
        print(f"  torch .sum(), {name:<9} -> {s:12.4f}   "
              f"relative error {abs(s - 1000.0) / 1000.0:9.2e}")
        print(f"  .sum(dtype=float32),  {name:<9} -> {s32:12.4f}   "
              f"relative error {abs(s32 - 1000.0) / 1000.0:9.2e}")
        rows.append({"method": f"torch.sum ({name})", "result": s,
                     "rel_error": abs(s - 1000.0) / 1000.0})
        rows.append({"method": f"torch.sum ({name}, acc=float32)", "result": s32,
                     "rel_error": abs(s32 - 1000.0) / 1000.0})
    print()
    return curves, rows


def stall_point():
    """Where does a float16 running total stop growing?"""
    print("=" * 92)
    print("Why the naive float16 total freezes")
    print("=" * 92)
    eps = torch.finfo(torch.float16).eps
    for v in [1.0, 0.001]:
        acc = np.float16(0.0)
        val = np.float16(v)
        steps = 0
        while steps < 5_000_000:
            new = np.float16(acc + val)
            if new == acc:
                break
            acc, steps = new, steps + 1
        predicted = 2 * v / eps
        print(f"  adding {v:<7} -> total freezes at {float(acc):10.4f} "
              f"after {steps:,} additions   (rule of thumb 2*value/eps = "
              f"{predicted:.1f})")
    print("\n  A float16 has ~3 decimal digits. Once the total is 1000x bigger than")
    print("  the thing being added, the addition rounds to nothing and the loop")
    print("  runs forever for free.\n")
    return {"eps_fp16": eps}


# --------------------------------------------------------------------------
# 4. Underflow, and the reason GradScaler exists.
# --------------------------------------------------------------------------
def underflow_study():
    print("=" * 92)
    print("Underflow: gradients that vanish on the way into float16")
    print("=" * 92)
    g = torch.randn(200_000).abs() * 1e-7      # a plausible small-gradient bucket
    for dt in [torch.float16, torch.bfloat16]:
        lost = (g.to(dt) == 0).float().mean().item()
        print(f"  {str(dt).replace('torch.',''):<10} loses "
              f"{lost * 100:5.1f}% of them to zero")

    scale = 1024.0
    scaled = (g * scale).to(torch.float16)
    lost_scaled = (scaled == 0).float().mean().item()
    recovered = (scaled.float() / scale)
    rel = ((recovered - g).abs() / g).mean().item()
    print(f"\n  float16 after multiplying by {scale:.0f} first: "
          f"{lost_scaled * 100:5.1f}% lost, "
          f"mean relative error {rel:.2e} after dividing back")
    print("\n  That multiply-then-divide IS what torch.cuda.amp.GradScaler does.")
    print("  bfloat16 needs no scaler because its exponent range is float32's.\n")
    return {"fp16_lost_pct": (g.to(torch.float16) == 0).float().mean().item() * 100,
            "bf16_lost_pct": (g.to(torch.bfloat16) == 0).float().mean().item() * 100,
            "fp16_scaled_lost_pct": lost_scaled * 100,
            "fp16_scaled_rel_err": rel}


# --------------------------------------------------------------------------
# 5. The trade the two 16-bit formats make, on the same two numbers.
# --------------------------------------------------------------------------
def range_vs_precision():
    print("=" * 92)
    print("Same 16 bits, opposite bets")
    print("=" * 92)
    probes = [1e-8, 1e-5, 1.001, 1.0001, 65000.0, 1e10]
    head = f"{'value':>12}{'float16':>18}{'bfloat16':>18}   verdict"
    print(head)
    print("-" * (len(head) + 22))
    rows = []
    for p in probes:
        a = torch.tensor([p], dtype=torch.float16).double().item()
        b = torch.tensor([p], dtype=torch.bfloat16).double().item()
        ea = abs(a - p) / p if p else 0.0
        eb = abs(b - p) / p if p else 0.0
        verdict = "float16 better" if ea < eb else ("bfloat16 better" if eb < ea
                                                   else "tie")
        print(f"{p:>12.6g}{a:>18.8g}{b:>18.8g}   {verdict}")
        rows.append({"value": p, "float16": a, "bfloat16": b,
                     "fp16_rel_err": ea, "bf16_rel_err": eb})
    print("\n  float16 wins on the numbers near 1. bfloat16 wins on everything")
    print("  far from 1 - and 'far from 1' is where gradients and activations live.\n")
    return rows


# --------------------------------------------------------------------------
# 6. Does any of this survive a real operation?
# --------------------------------------------------------------------------
def matmul_error():
    print("=" * 92)
    print("A 512x512 matmul, each dtype against a float64 reference")
    print("=" * 92)
    a = torch.randn(512, 512, dtype=torch.float64)
    b = torch.randn(512, 512, dtype=torch.float64)
    ref = a @ b
    rows = []
    for dt in [torch.float32, torch.bfloat16, torch.float16]:
        got = (a.to(dt) @ b.to(dt)).double()
        rel = ((got - ref).norm() / ref.norm()).item()
        name = str(dt).replace("torch.", "")
        print(f"  {name:<10} relative error {rel:.3e}")
        rows.append({"dtype": name, "matmul_rel_error": rel})
    print("\n  512 products accumulated per output element, and float16 is still")
    print("  ~4x more accurate than bfloat16 here - because these inputs are all")
    print("  near 1, which is float16's home turf.\n")
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_curves(curves):
    fig, ax = ps.new_axes(7.4, 4.4)
    for (name, partials), color in zip(curves.items(), ps.SERIES):
        ns = sorted(partials)
        errs = [max(abs(partials[n] - n * VALUE) / (n * VALUE), 1e-17) for n in ns]
        ax.plot(ns, errs, marker="o", ms=4, color=color, label=f"naive loop, {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(ps.INK_SECONDARY)
    ps.finish(fig, ax,
              "One number at a time: error grows with how many you have added",
              "numbers summed so far", "relative error of the running total",
              os.path.join(OUT, "accumulation_error.png"))


def fig_methods(rows):
    keep = [r for r in rows if r["method"] in (
        "naive loop (float16)", "torch.sum (float16)", "Kahan loop (float16)",
        "naive loop (bfloat16)", "torch.sum (bfloat16)",
        "torch.sum (float16, acc=float32)", "naive loop (float32)")]
    fig, ax = ps.new_axes(8.0, 4.2)
    names = [r["method"] for r in keep]
    vals = [max(r["rel_error"], 1e-9) for r in keep]
    colors = [ps.SERIES[2] if "naive" in n and "16" in n else
              (ps.SERIES[1] if ("float32" in n or "Kahan" in n) else ps.SERIES[0])
              for n in names]
    bars = ax.barh(range(len(keep)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for b, v, r in zip(bars, vals, keep):
        label = "0.0  (exact, by luck)" if r["rel_error"] == 0 else f"{v:.1e}"
        ax.text(v * 1.25, b.get_y() + b.get_height() / 2, label,
                va="center", fontsize=8.5, color=ps.INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    ps.finish(fig, ax, "Same million numbers, seven ways of adding them up",
              "relative error (log scale)", "",
              os.path.join(OUT, "summation_methods.png"))


def fig_probe(rows):
    fig, ax = ps.new_axes(7.6, 4.2)
    xs = np.arange(len(rows))
    w = 0.38
    EXACT, BROKEN = 1e-9, 2.0     # floor for "no error", ceiling for "destroyed"

    def clamp(v):
        return min(max(v, EXACT), BROKEN)

    fp = [clamp(r["fp16_rel_err"]) for r in rows]
    bf = [clamp(r["bf16_rel_err"]) for r in rows]
    ax.bar(xs - w / 2, fp, w, color=ps.SERIES[0], label="float16")
    ax.bar(xs + w / 2, bf, w, color=ps.SERIES[3], label="bfloat16")
    ax.set_yscale("log")
    ax.set_ylim(EXACT / 2, 300)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['value']:g}" for r in rows], fontsize=9)

    # Name the two catastrophes and the two exact hits.
    for i, r in enumerate(rows):
        for off, key, val in ((-w / 2, "fp16_rel_err", r["float16"]),
                              (w / 2, "bf16_rel_err", r["bfloat16"])):
            e = r[key]
            if e >= 1.0 or not np.isfinite(e):
                tag = "-> inf" if not np.isfinite(val) else "-> 0"
                ax.text(i + off, BROKEN * 1.15, tag, ha="center", fontsize=8.5,
                        color=ps.SERIES[2])
            elif e <= 0:
                ax.text(i + off, EXACT * 1.3, "exact", ha="center", fontsize=8,
                        color=ps.INK_MUTED, rotation=90)

    leg = ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    for t in leg.get_texts():
        t.set_color(ps.INK_SECONDARY)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax, "Storing six numbers in the two 16-bit formats",
              "value being stored", "relative error (log scale)",
              os.path.join(OUT, "range_vs_precision.png"))


# --------------------------------------------------------------------------
def main():
    facts = dtype_facts()
    rep = representation_error()
    curves, methods = accumulation_study()
    stall = stall_point()
    under = underflow_study()
    probes = range_vs_precision()
    mm = matmul_error()

    with open(os.path.join(OUT, "dtype_facts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(facts[0].keys()))
        w.writeheader()
        w.writerows(facts)
    with open(os.path.join(OUT, "summation_methods.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(methods[0].keys()))
        w.writeheader()
        w.writerows(methods)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for r in rep:
            w.writerow([f"stored_{r['dtype']}", r["stored_value"]])
        for k, v in {**stall, **under}.items():
            w.writerow([k, v])
        for r in mm:
            w.writerow([f"matmul_rel_error_{r['dtype']}", r["matmul_rel_error"]])
    print(f"wrote {OUT}/dtype_facts.csv, summation_methods.csv, findings.csv")

    fig_curves(curves)
    fig_methods(methods)
    fig_probe(probes)


if __name__ == "__main__":
    main()
