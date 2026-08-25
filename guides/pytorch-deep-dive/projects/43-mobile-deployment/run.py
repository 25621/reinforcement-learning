"""Project 43 - take the CNN to ExecuTorch, PyTorch's runtime for phones.

This file runs in the guide's normal environment. It

  1. makes sure the trained CNN and the CIFAR cache exist,
  2. runs `et_bench.py` inside the ExecuTorch virtual environment,
  3. turns the JSON it produced into the tables and the figure.

    python3 run.py --setup     # create ~/.venvs/executorch (about 1.5 GB, once)
    python3 run.py             # ~2 minutes

Why a separate environment: ExecuTorch 1.4.0 requires torch >= 2.13, and this guide
runs on torch 2.10. Installing it into the main environment upgrades PyTorch under
your feet — see section 0 of the README, which reports exactly what broke.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "42-export-to-onnx")
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import deploy_lib as D  # noqa: E402
from plot_style import SERIES, style_axes  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
VENV = os.path.expanduser("~/.venvs/executorch")
VPY = os.path.join(VENV, "bin", "python")


def setup_venv():
    print(f"creating {VENV} (torch 2.13 CPU + executorch, about 1.5 GB) ...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV])
    subprocess.check_call([os.path.join(VENV, "bin", "pip"), "install", "-q",
                           "torch==2.13.0", "--index-url",
                           "https://download.pytorch.org/whl/cpu"])
    subprocess.check_call([os.path.join(VENV, "bin", "pip"), "install", "-q", "executorch"])
    print("done")


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print("    " + line)
    print("    " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def figure(res):
    import matplotlib.pyplot as plt

    lat = res["latency_ms_batch1"]
    order = ["portable", "pytorch_eager", "xnnpack", "xnnpack_int8"]
    labels = ["ExecuTorch\nportable", "PyTorch\neager", "ExecuTorch\n+XNNPACK",
              "ExecuTorch\n+XNNPACK int8"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    ax = axes[0]
    vals = [lat[k] for k in order]
    colors = [SERIES[2], SERIES[0], SERIES[1], SERIES[3]]
    ax.bar(range(4), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.2f}", ha="center", fontsize=8.5, color="#0b0b0b")
    ax.set_ylabel("ms per image (log)")
    ax.set_title("latency, batch 1", loc="left", fontsize=11)

    ax = axes[1]
    sizes = res["sizes_mb"]
    keys = [k for k in ["state_dict .pt", "onnx (graph + sidecar)", "pte portable",
                        "pte xnnpack", "pte xnnpack_int8"] if sizes.get(k)]
    ax.barh(range(len(keys)), [sizes[k] for k in keys], color=SERIES[0])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("pte ", ".pte ") for k in keys], fontsize=8.5)
    ax.invert_yaxis()
    for i, k in enumerate(keys):
        ax.text(sizes[k] + 0.012, i, f"{sizes[k]:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(sizes[k] for k in keys) * 1.25)
    ax.set_xlabel("MB")
    ax.set_title("file size on the device", loc="left", fontsize=11)

    ax = axes[2]
    progs = res["programs"]
    names = ["portable", "xnnpack", "xnnpack_dyn"]
    arena = [progs[n]["arena_bytes"] / 1024 for n in names]
    ax.bar(range(3), arena, color=[SERIES[2], SERIES[1], SERIES[4]])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["portable", "XNNPACK\nstatic", "XNNPACK\ndynamic"], fontsize=8.5)
    for i, v in enumerate(arena):
        ax.text(i, v + max(arena) * 0.02, f"{v:.0f} KB", ha="center", fontsize=8.5)
    ax.set_ylabel("KB")
    ax.set_title("memory planned ahead of time", loc="left", fontsize=11)

    fig.tight_layout()
    path = os.path.join(OUT, "executorch.png")
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="create the ExecuTorch venv")
    ap.add_argument("--reuse", action="store_true",
                    help="skip the benchmark, re-read outputs/et_results.json")
    args = ap.parse_args()

    if args.setup:
        setup_venv()
    if not os.path.exists(VPY):
        sys.exit(f"no ExecuTorch environment at {VENV}\nrun:  python3 run.py --setup")

    # Make sure the shared artefacts exist while we are still in an environment
    # that has pyarrow and pillow for decoding CIFAR.
    D.get_trained_cnn()
    D.load_cifar("test", 8)

    result_path = os.path.join(OUT, "et_results.json")
    if not args.reuse:
        print(f"running et_bench.py under {VPY}\n")
        proc = subprocess.run([VPY, os.path.join(HERE, "et_bench.py")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            sys.exit(proc.stderr[-3000:])
        # the runtime prints ARM cpuinfo probes on stderr; keep them, they are evidence
        with open(os.path.join(OUT, "et_bench.log"), "w") as fh:
            fh.write(proc.stderr)
    res = json.load(open(result_path))

    print(f"[venv] torch {res['torch']}\n")

    print("[1] three ways to build the same .pte")
    rows = []
    for name in ("portable", "xnnpack", "xnnpack_int8"):
        p = res["programs"][name]
        rows.append([name, f"{p['mb']:.4f}", p["graph_call_nodes"], p["operators"],
                     p["delegates"], f"{p['arena_bytes'] / 1024:.1f}"])
    table(rows, ["program", "MB", "graph nodes", "operators", "delegates", "arena KB"])
    print(f"    lowering time: portable {res['portable_lower_s']:.2f}s | "
          f"xnnpack {res['xnnpack_lower_s']:.2f}s | int8 {res['int8_lower_s']:.2f}s")

    print("\n[2] does the .pte compute the same thing?")
    rows = []
    for name, r in res["runs"].items():
        rows.append([name, r["n_images"], f"{r['max_abs_diff']:.3e}",
                     f"{r['agree_pct']:.2f}%", f"{r['accuracy']:.4f}"])
    table(rows, ["program", "images", "max |diff|", "same prediction", "accuracy"])

    print("\n[3] latency, one image at a time")
    lat = res["latency_ms_batch1"]
    base = lat["xnnpack"]
    rows = [[k, f"{v:.3f}", f"{v / base:.1f}x"] for k, v in sorted(lat.items(),
                                                                  key=lambda kv: -kv[1])]
    table(rows, ["runtime", "ms", "vs XNNPACK"])

    print("\n[4] dynamic shapes are bounded")
    rows = [[b, res["static_batches"][b], res["dynamic_batches"][b]]
            for b in ("1", "8", "15", "16")]
    table(rows, ["batch", "static .pte", f"dynamic .pte (declared max "
                                         f"{res['declared_max_batch']})"])
    print(f"    arena: static {res['programs']['xnnpack']['arena_bytes'] / 1024:.1f} KB"
          f"  ->  dynamic {res['programs']['xnnpack_dyn']['arena_bytes'] / 1024:.1f} KB")
    print(f"    declaring max=32 instead accepts batches "
          f"{res['declared_max_32_accepts']}")

    print("\n[5] what you would copy to the phone")
    table([[k, f"{v:.4f}"] for k, v in res["sizes_mb"].items() if v],
          ["artefact", "MB"])

    figure(res)
    rows = []
    for sec, d in [("program", res["programs"]), ("run", res["runs"])]:
        for name, vals in d.items():
            for k, v in vals.items():
                rows.append((sec, name, k, v))
    for k, v in res["latency_ms_batch1"].items():
        rows.append(("latency_ms", k, "batch1", f"{v:.4f}"))
    for k, v in res["sizes_mb"].items():
        if v:
            rows.append(("size_mb", k, "", f"{v:.4f}"))
    for b in ("1", "8", "15", "16"):
        rows.append(("shape", "static", b, res["static_batches"][b]))
        rows.append(("shape", "dynamic", b, res["dynamic_batches"][b]))
    D.write_csv(os.path.join(OUT, "findings.csv"), rows,
                ["section", "item", "metric", "value"])


if __name__ == "__main__":
    main()
