"""Project 47 - measuring latency properly: percentiles, batch size, and tails.

Sections:
  1. one number is not enough: the distribution of 3000 identical calls
  2. batch size: the latency / throughput trade-off, and how to pick from a budget
  3. how many samples does a p99 need before it means anything?
  4. warm-up: the first call is not like the others
  5. where tails come from - a noisy neighbour, measured
  6. percentiles do not add up: what happens when a request needs several calls

Run:  python3 run.py       (~3 minutes, plus ~2.5 min the first time to train the CNN)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "42-export-to-onnx"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import deploy_lib as D  # noqa: E402
from plot_style import SERIES, style_axes  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

FINDINGS: list[tuple] = []
RNG = np.random.default_rng(0)


def note(section, name, value):
    FINDINGS.append((section, name, value))
    print(f"    {name:<52} {value}")


def pct(a, q):
    return float(np.percentile(np.asarray(a), q))


@torch.no_grad()
def measure(model, x, n, warmup=20):
    """n timed calls of `model(x)`, in milliseconds."""
    for _ in range(warmup):
        model(x)
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        model(x)
        out[i] = (time.perf_counter() - t0) * 1e3
    return out


class Hog:
    """A noisy neighbour: threads that do nothing but burn CPU."""

    def __init__(self, n_threads=6):
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._burn, daemon=True)
                        for _ in range(n_threads)]

    def _burn(self):
        a = np.random.randn(256, 256)
        while not self.stop.is_set():
            a @ a                                    # releases the GIL, uses a core

    def __enter__(self):
        for t in self.threads:
            t.start()
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=2)


# ==========================================================================
def main():
    t_start = time.time()
    model = D.get_trained_cnn()
    x, _ = D.load_cifar("test", 256)
    one = x[:1].contiguous()

    # ------------------------------------------------------------------ [1]
    print("\n[1] one number is not enough")
    lat = measure(model, one, 3000)
    stats = D.percentiles(lat, qs=(50, 90, 95, 99, 99.9))
    for k in ("p50", "p90", "p95", "p99", "p99.9"):
        note(1, f"{k}", f"{stats[k]:.3f} ms")
    note(1, "mean", f"{stats['mean']:.3f} ms")
    note(1, "max", f"{stats['max']:.3f} ms")
    note(1, "mean / p50", f"{stats['mean'] / stats['p50']:.2f}x  "
                          f"(the average is pulled up by the tail)")
    note(1, "p99 / p50", f"{stats['p99'] / stats['p50']:.2f}x")
    note(1, "max / p50", f"{stats['max'] / stats['p50']:.2f}x")
    note(1, "calls slower than the mean",
         f"{100 * float((lat > stats['mean']).mean()):.1f}%  "
         f"(not 50% - the distribution is skewed)")

    # ------------------------------------------------------------------ [2]
    print("\n[2] batch size: latency against throughput")
    batch_rows = []
    batches = (1, 2, 4, 8, 16, 32, 64)
    # Rotate through the batch sizes round by round. Measuring each size to
    # completion in turn would charge any burst of CPU contention entirely to
    # whichever size was unlucky - and the p99 column is exactly where that shows.
    samples: dict[int, list] = {b: [] for b in batches}
    for _ in range(12):
        for b in batches:
            xb = x[:b].contiguous()
            samples[b].append(measure(model, xb, max(24, int(320 / b)), warmup=2))
    for b in batches:
        lb = np.concatenate(samples[b])
        row = {"batch": b, "n": int(lb.size), "p50": pct(lb, 50), "p95": pct(lb, 95),
               "p99": pct(lb, 99), "per_image_p50": pct(lb, 50) / b,
               "throughput": b * 1000 / pct(lb, 50)}
        batch_rows.append(row)
        note(2, f"batch {b:>2}",
             f"p50 {row['p50']:7.2f} ms  p95 {row['p95']:7.2f} ms  "
             f"p99 {row['p99']:7.2f} ms  "
             f"{row['per_image_p50']:6.3f} ms/image  {row['throughput']:7.0f} img/s  "
             f"(n={row['n']})")
    budget = 10.0
    # Selected on p95, not p99: see section 5 - on this shared machine the p99
    # column is measuring the neighbours, and it is not even monotone in batch size.
    ok = [r for r in batch_rows if r["p95"] <= budget]
    best = max(ok, key=lambda r: r["throughput"]) if ok else None
    note(2, f"largest batch whose p95 fits a {budget:.0f} ms budget",
         f"batch {best['batch']} at {best['throughput']:.0f} img/s"
         if best else "none")
    note(2, "throughput gained from batch 1 to that batch",
         f"{best['throughput'] / batch_rows[0]['throughput']:.2f}x" if best else "-")

    # ------------------------------------------------------------------ [3]
    print("\n[3] how many samples does a p99 need?")
    truth = pct(lat, 99)
    note(3, "p99 from all 3000 samples (treated as truth)", f"{truth:.3f} ms")
    sample_rows = []
    for n in (20, 100, 300, 1000, 3000):
        estimates = np.array([pct(RNG.choice(lat, n, replace=True), 99)
                              for _ in range(400)])
        lo, hi = np.percentile(estimates, [2.5, 97.5])
        sample_rows.append({"n": n, "lo": float(lo), "hi": float(hi),
                            "width_pct": float(100 * (hi - lo) / truth)})
        note(3, f"p99 estimated from {n:>4} samples",
             f"95% of estimates land in [{lo:.2f}, {hi:.2f}] ms  "
             f"= +/-{50 * (hi - lo) / truth:5.1f}% of the true value")
    note(3, "samples strictly above the p99, at n=20",
         f"{20 * 0.01:.1f} expected - the estimate is an extrapolation, not a measurement")

    # ------------------------------------------------------------------ [4]
    print("\n[4] warm-up")
    cold_start = json.loads(__import__("subprocess").run(
        [sys.executable, os.path.join(HERE, "cold_start.py")],
        capture_output=True, text=True).stdout.strip().splitlines()[-1])
    note(4, "fresh process: import torch", f"{cold_start['import_s']:.2f} s")
    note(4, "fresh process: build the model and load weights",
         f"{cold_start['load_ms']:.0f} ms")
    note(4, "fresh process: first inference", f"{cold_start['first_ms']:.1f} ms")
    note(4, "fresh process: second inference", f"{cold_start['second_ms']:.2f} ms")
    note(4, "total time before the first answer",
         f"{cold_start['total_s']:.2f} s  = "
         f"{1000 * cold_start['total_s'] / cold_start['second_ms']:.0f}x one warm call")
    fresh = D.SmallCNN().eval()
    fresh.load_state_dict(model.state_dict())
    cold = []
    with torch.no_grad():
        for _ in range(12):
            t0 = time.perf_counter()
            fresh(one)
            cold.append((time.perf_counter() - t0) * 1e3)
    note(4, "first 6 calls of a fresh model (ms)",
         "  ".join(f"{v:.2f}" for v in cold[:6]))
    note(4, "call 1 vs the steady state",
         f"{cold[0]:.2f} ms vs {stats['p50']:.2f} ms  "
         f"= {cold[0] / stats['p50']:.1f}x")
    note(4, "calls needed to reach 1.2x the steady state",
         next((i + 1 for i, v in enumerate(cold) if v < 1.2 * stats["p50"]), ">12"))
    note(4, "p99 including the first 12 calls",
         f"{pct(np.concatenate([cold, lat[:988]]), 99):.3f} ms  "
         f"vs {pct(lat[:1000], 99):.3f} ms without")

    # ------------------------------------------------------------------ [5]
    print("\n[5] where tails come from: a noisy neighbour")
    with Hog(6):
        lat_hog = measure(model, one, 1000, warmup=10)
    quiet = D.percentiles(lat[:1000], qs=(50, 99))
    noisy = D.percentiles(lat_hog, qs=(50, 99))
    note(5, "p50: alone / with a neighbour",
         f"{quiet['p50']:.3f} / {noisy['p50']:.3f} ms   "
         f"{noisy['p50'] / quiet['p50']:.2f}x")
    note(5, "p99: alone / with a neighbour",
         f"{quiet['p99']:.3f} / {noisy['p99']:.3f} ms   "
         f"{noisy['p99'] / quiet['p99']:.2f}x")
    note(5, "max: alone / with a neighbour",
         f"{quiet['max']:.3f} / {noisy['max']:.3f} ms")
    note(5, "which percentile noticed the neighbour more",
         "p99" if noisy["p99"] / quiet["p99"] > noisy["p50"] / quiet["p50"] else "p50")

    # ------------------------------------------------------------------ [6]
    print("\n[6] percentiles do not add up")
    comp_rows = []
    for k in (1, 2, 5, 10):
        # A "request" that makes k sequential model calls, resampled from the
        # measured distribution 20000 times.
        totals = RNG.choice(lat, (20000, k), replace=True).sum(axis=1)
        p99_sum = pct(totals, 99)
        naive = k * stats["p99"]
        slow = float((RNG.choice(lat, (20000, k), replace=True)
                      > stats["p99"]).any(axis=1).mean())
        comp_rows.append({"k": k, "p99": p99_sum, "naive": naive, "any_slow": slow})
        note(6, f"{k:>2} sequential calls: real p99 / k x p99",
             f"{p99_sum:7.2f} / {naive:7.2f} ms   "
             f"({100 * p99_sum / naive:.0f}% of the naive bound)")
        note(6, f"  chance at least one of the {k:>2} calls is a p99 outlier",
             f"{100 * slow:.1f}%   (1 - 0.99^{k} = {100 * (1 - 0.99 ** k):.1f}%)")

    # --------------------------------------------------------------- output
    summary = {"stats": stats, "batches": batch_rows, "samples": sample_rows,
               "cold": cold, "quiet": quiet, "noisy": noisy, "composite": comp_rows}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.save(os.path.join(OUT, "latencies_ms.npy"), lat.astype(np.float32))
    D.write_csv(os.path.join(OUT, "findings.csv"), FINDINGS, ["section", "name", "value"])
    D.write_csv(os.path.join(OUT, "batch_sweep.csv"),
                [(r["batch"], f"{r['p50']:.3f}", f"{r['p95']:.3f}", f"{r['p99']:.3f}",
                  f"{r['per_image_p50']:.4f}", f"{r['throughput']:.0f}")
                 for r in batch_rows],
                ["batch", "p50_ms", "p95_ms", "p99_ms", "ms_per_image",
                 "images_per_s"])
    figure(summary, lat, lat_hog)
    print(f"\ntotal {time.time() - t_start:.0f}s")


def figure(summary, lat, lat_hog):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    s = summary["stats"]
    ax = axes[0]
    hi = np.percentile(lat, 99.7)
    ax.hist(np.clip(lat, None, hi), bins=70, color=SERIES[0])
    for key, color in [("p50", SERIES[1]), ("mean", SERIES[3]), ("p99", SERIES[2])]:
        ax.axvline(s[key], color=color, linestyle="--", linewidth=1.4,
                   label=f"{key} {s[key]:.2f} ms")
    ax.set_yscale("log")
    ax.set_xlabel("ms"); ax.set_ylabel("calls (log)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("3000 identical calls", loc="left", fontsize=11)

    ax = axes[1]
    b = summary["batches"]
    ax.plot([r["throughput"] for r in b], [r["p95"] for r in b], "o-",
            color=SERIES[0])
    for r in b:
        ax.annotate(f"{r['batch']}", (r["throughput"], r["p95"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(10.0, color=SERIES[2], linestyle="--", linewidth=1.3,
               label="10 ms p95 budget")
    ax.set_xlabel("throughput (images / s)"); ax.set_ylabel("p95 latency (ms)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("the trade-off, labelled by batch size", loc="left", fontsize=11)

    ax = axes[2]
    sr = summary["samples"]
    xs = np.arange(len(sr))
    truth = np.percentile(lat, 99)
    ax.errorbar(xs, [(r["lo"] + r["hi"]) / 2 for r in sr],
                yerr=[[(r["lo"] + r["hi"]) / 2 - r["lo"] for r in sr],
                      [r["hi"] - (r["lo"] + r["hi"]) / 2 for r in sr]],
                fmt="o", color=SERIES[0], capsize=5)
    ax.axhline(truth, color=SERIES[2], linestyle="--", linewidth=1.3,
               label=f"true p99 {truth:.2f} ms")
    ax.set_xticks(xs); ax.set_xticklabels([r["n"] for r in sr])
    ax.set_xlabel("samples used"); ax.set_ylabel("estimated p99 (ms)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("95% of p99 estimates land here", loc="left", fontsize=11)

    ax = axes[3]
    bins = np.linspace(0, np.percentile(np.concatenate([lat, lat_hog]), 99), 60)
    ax.hist(lat, bins=bins, color=SERIES[0], alpha=0.75, label="alone")
    ax.hist(lat_hog, bins=bins, color=SERIES[2], alpha=0.6, label="noisy neighbour")
    ax.set_yscale("log")
    ax.set_xlabel("ms"); ax.set_ylabel("calls (log)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("same model, busier machine", loc="left", fontsize=11)

    fig.tight_layout()
    path = os.path.join(OUT, "latency.png")
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
