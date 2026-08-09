"""Project 24 — Profile a training step.

Capture one forward + backward + optimizer step with `torch.profiler`, then read
the table properly:

  1. where the step's time goes (forward / backward / optimizer)
  2. the top operators by SELF CPU time, and why self != total
  3. the same table grouped by input shape — which matmul is the expensive one
  4. which operators allocate the memory
  5. warm-up: what the first step measures that the tenth does not
  6. the observer effect: how much the profiler itself costs
  7. a Chrome trace you can open in Perfetto

Runtime ~1 min. Needs torch, numpy, matplotlib.
"""

import csv
import gzip
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

STEPS = 5
rows = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<38} {value:>14}  {note}")


def train_step(model, opt, x, y, tag=True):
    if tag:
        with record_function("## forward"):
            logits = model(x)
            loss = P.loss_fn(logits, y)
        with record_function("## backward"):
            loss.backward()
        with record_function("## optimizer"):
            opt.step()
            opt.zero_grad(set_to_none=True)
    else:
        loss = P.loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"cuda available: {torch.cuda.is_available()}")

    model = P.new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x, y = P.make_batch()
    n_params = P.count_params(model)
    print(f"model: {n_params/1e6:.2f} M parameters, batch {x.shape[0]}x{x.shape[1]}\n")

    # -----------------------------------------------------------------
    # 5 (first): warm-up — the first step is not like the others
    # -----------------------------------------------------------------
    print("[warm-up] the first step, and the first step at a new shape")
    warm = P.new_model()
    wopt = torch.optim.AdamW(warm.parameters(), lr=3e-4)
    per_step = []
    for i in range(10):
        t0 = time.perf_counter()
        train_step(warm, wopt, x, y, tag=False)
        per_step.append((time.perf_counter() - t0) * 1e3)
    for i, ms in enumerate(per_step[:2]):
        record("warmup", f"step {i} of the process (ms)", f"{ms:.1f}")
    steady = float(np.median(per_step[4:]))
    record("warmup", "steady state at this shape (ms)", f"{steady:.1f}", "median of steps 4-9")
    record("warmup", "step 0 / steady", f"{per_step[0]/steady:.2f}x",
           "process start-up: varies with what the OS has cached")

    # the reproducible half: a shape this process has never seen. PyTorch picks
    # a matmul algorithm per shape and allocates fresh buffers for it.
    ratios = []
    for b in (12, 10, 6):
        xb, yb = P.make_batch(batch=b)
        t0 = time.perf_counter()
        train_step(warm, wopt, xb, yb, tag=False)
        first = (time.perf_counter() - t0) * 1e3
        later, _ = P.best_of(lambda: train_step(warm, wopt, xb, yb, tag=False),
                             repeats=5, warmup=2)
        ratios.append(first / later)
        record("warmup", f"first step at batch {b}", f"{first/later:.2f}x",
               f"{first:.1f} ms vs {later:.1f} ms once warm")
    record("warmup", "median new-shape penalty", f"{np.median(ratios):.2f}x",
           "why every benchmark needs warm-up iterations")
    print()

    # -----------------------------------------------------------------
    # 1-4: the profiled capture
    # -----------------------------------------------------------------
    print("[profile] capturing 5 steps")
    for _ in range(3):                       # warm up before measuring
        train_step(model, opt, x, y, tag=False)

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True, profile_memory=True) as prof:
        for _ in range(STEPS):
            train_step(model, opt, x, y)
    wall_profiled = (time.perf_counter() - t0) * 1e3 / STEPS

    ka = prof.key_averages()
    table = ka.table(sort_by="self_cpu_time_total", row_limit=25)
    (OUT / "profile_table.txt").write_text(table)
    print(table[:1400])

    # --- section split, read off the record_function regions
    regions = {e.key: e for e in ka if e.key.startswith("## ")}
    total_us = sum(regions[k].cpu_time_total for k in regions)
    section_share = {}
    for k, e in regions.items():
        share = 100.0 * e.cpu_time_total / total_us
        section_share[k[3:]] = share
        record("split", f"{k[3:]} share of step", f"{share:.1f} %",
               f"{e.cpu_time_total/STEPS/1000:.1f} ms/step")
    record("split", "backward / forward",
           f"{regions['## backward'].cpu_time_total / regions['## forward'].cpu_time_total:.2f}x",
           "theory says ~2x")

    # --- top ops by self CPU time
    ops = [e for e in ka if not e.key.startswith("## ") and e.self_cpu_time_total > 0]
    ops.sort(key=lambda e: -e.self_cpu_time_total)
    self_total = sum(e.self_cpu_time_total for e in ops)
    top = ops[:8]
    print()
    for e in top[:5]:
        record("top-ops", e.key, f"{100*e.self_cpu_time_total/self_total:.1f} %",
               f"{e.count//STEPS} calls/step, {e.self_cpu_time_total/STEPS/1000:.2f} ms/step")
    record("top-ops", "top-3 share of self CPU",
           f"{100*sum(e.self_cpu_time_total for e in ops[:3])/self_total:.1f} %")
    record("top-ops", "distinct operators", f"{len(ops)}")
    record("top-ops", "total operator calls / step", f"{sum(e.count for e in ops)//STEPS}")

    # --- self vs total: find the op with the biggest gap
    gap = max(ops, key=lambda e: e.cpu_time_total - e.self_cpu_time_total)
    record("self-vs-total", gap.key,
           f"{gap.self_cpu_time_total/1000:.1f} / {gap.cpu_time_total/1000:.1f} ms",
           "self / total — the rest is its children")

    # --- grouped by input shape
    ka_shapes = prof.key_averages(group_by_input_shape=True)
    mms = [e for e in ka_shapes if e.key in ("aten::mm", "aten::addmm", "aten::bmm")]
    mms.sort(key=lambda e: -e.self_cpu_time_total)
    print()
    for e in mms[:4]:
        record("by-shape", f"{e.key} {e.input_shapes}",
               f"{e.self_cpu_time_total/STEPS/1000:.2f} ms/step", f"{e.count//STEPS} calls/step")

    # --- memory
    allocs = [e for e in ka if e.self_cpu_memory_usage > 0]
    allocs.sort(key=lambda e: -e.self_cpu_memory_usage)
    print()
    for e in allocs[:4]:
        record("memory", e.key, P.human(e.self_cpu_memory_usage / STEPS), "allocated / step")
    record("memory", "total allocated / step",
           P.human(sum(e.self_cpu_memory_usage for e in allocs) / STEPS))

    # -----------------------------------------------------------------
    # 6: the observer effect
    # -----------------------------------------------------------------
    print()
    wall_plain, spread = P.best_of(lambda: train_step(model, opt, x, y, tag=False),
                                   repeats=7, warmup=2)
    record("overhead", "step without profiler (ms)", f"{wall_plain:.1f}",
           f"spread {spread:.1f}")
    record("overhead", "step with profiler (ms)", f"{wall_profiled:.1f}")
    record("overhead", "profiler overhead", f"{wall_profiled/wall_plain:.2f}x")
    prof_total_ms = sum(e.self_cpu_time_total for e in ops) / STEPS / 1000
    record("overhead", "sum of self CPU (ms/step)", f"{prof_total_ms:.1f}")
    record("overhead", "sum of self CPU / wall clock", f"{prof_total_ms/wall_profiled:.2f}",
           "CPU work is synchronous: these agree")

    # -----------------------------------------------------------------
    # 7: a trace for Perfetto
    # -----------------------------------------------------------------
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof2:
        train_step(model, opt, x, y)
    raw = OUT / "trace_step.json"
    prof2.export_chrome_trace(str(raw))
    with open(raw, "rb") as f_in, gzip.open(OUT / "trace_step.json.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    size = raw.stat().st_size
    raw.unlink()
    record("trace", "trace_step.json.gz",
           P.human((OUT / "trace_step.json.gz").stat().st_size),
           f"{P.human(size)} before gzip")

    # -----------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------
    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    names = list(section_share.keys())
    vals = [section_share[k] for k in names]
    ax.barh(np.arange(len(names)), vals, color=ps.SERIES[:len(names)])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("Where one training step goes", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("share of step (%)", color=ps.INK_SECONDARY)

    ax = axes[1]
    labels = [e.key.replace("aten::", "") for e in top][::-1]
    shares = [100 * e.self_cpu_time_total / self_total for e in top][::-1]
    ax.barh(np.arange(len(labels)), shares, color=ps.SERIES[0])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Top operators by self CPU time", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("share of self CPU time (%)", color=ps.INK_SECONDARY)

    ax = axes[2]
    ax.plot(np.arange(len(per_step)), per_step, "o-", color=ps.SERIES[2], lw=1.8)
    ax.axhline(steady, color=ps.BASELINE, ls="--", lw=1.2)
    ax.text(4.5, steady * 1.06, "steady state", color=ps.INK_MUTED, fontsize=9)
    ax.set_title("The first step is not a measurement", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("step", color=ps.INK_SECONDARY)
    ax.set_ylabel("wall time (ms)", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "profile_a_training_step.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
