"""Project 45 - static (post-training) int8 quantization of a CNN, and what
calibration is actually for.

Sections:
  1. why project 44's dynamic quantization does nothing here
  2. the PTQ flow: observers in, calibrate, convert - and what the model became
  3. accuracy, size, latency against float32
  4. how much calibration data is enough - and what wrong data does
  5. the observer matters more than the amount of data
  6. static vs dynamic on the same layers, head to head

Run:  python3 run.py       (~3 minutes, plus ~2.5 min the first time to train the CNN)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "42-export-to-onnx"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import deploy_lib as D  # noqa: E402
from plot_style import SERIES, style_axes  # noqa: E402

from torch.ao.quantization import (  # noqa: E402
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QConfigMapping,
    default_dynamic_qconfig,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

FINDINGS: list[tuple] = []


def note(section, name, value):
    FINDINGS.append((section, name, value))
    print(f"    {name:<52} {value}")


def module_mb(model) -> float:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def kind_counts(model) -> dict:
    from collections import Counter

    return dict(Counter(type(m).__name__ for m in model.modules()
                        if len(list(m.children())) == 0))


def ptq(model, calib_x, qconfig_mapping=None, example=None):
    """The whole post-training-quantization recipe in five lines."""
    mapping = qconfig_mapping or get_default_qconfig_mapping("x86")
    example = example if example is not None else calib_x[:1]
    prepared = prepare_fx(copy.deepcopy(model).eval(), mapping, (example,))
    with torch.no_grad():
        for i in range(0, len(calib_x), 64):        # 2. calibration
            prepared(calib_x[i:i + 64])
    return convert_fx(prepared)


# ==========================================================================
def main():
    t_start = time.time()
    print(f"torch {torch.__version__} | quantized backend "
          f"{torch.backends.quantized.engine}")

    fp32 = D.get_trained_cnn()
    xte, yte = D.load_cifar("test", 2000)
    xtr, _ = D.load_cifar("train", 1024)            # calibration pool, never trained on
    acc32 = D.accuracy(fp32, xte, yte)
    mb32 = module_mb(fp32)

    # ------------------------------------------------------------------ [1]
    print("\n[1] why dynamic quantization does nothing to a CNN")
    dyn_conv = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Conv2d}, dtype=torch.qint8)
    dyn_lin = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Linear}, dtype=torch.qint8)
    n_q_conv = sum(1 for m in dyn_conv.modules()
                   if type(m).__module__.startswith("torch.ao.nn.quantized"))
    n_q_lin = sum(1 for m in dyn_lin.modules()
                  if type(m).__module__.startswith("torch.ao.nn.quantized"))
    conv_params = sum(m.weight.numel() for m in fp32.modules() if isinstance(m, nn.Conv2d))
    lin_params = sum(m.weight.numel() for m in fp32.modules() if isinstance(m, nn.Linear))
    note(1, "asking for Conv2d: modules actually quantized", n_q_conv)
    note(1, "asking for Linear: modules actually quantized", n_q_lin)
    note(1, "weights in Conv2d / in Linear",
         f"{conv_params / 1e3:.1f}k / {lin_params / 1e3:.1f}k "
         f"({100 * conv_params / (conv_params + lin_params):.1f}% is convolution)")
    note(1, "size: float32 / dynamic-quantized",
         f"{mb32:.3f} MB / {module_mb(dyn_conv):.3f} MB")
    note(1, "accuracy unchanged", f"{D.accuracy(dyn_conv, xte, yte):.4f}")

    # ------------------------------------------------------------------ [2]
    print("\n[2] the static flow: observe, calibrate, convert")
    mapping = get_default_qconfig_mapping("x86")
    prepared = prepare_fx(copy.deepcopy(fp32).eval(), mapping, (xte[:1],))
    n_obs = sum(1 for m in prepared.modules()
                if "Observer" in type(m).__name__ or "FakeQuantize" in type(m).__name__)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, 512, 64):
            prepared(xtr[i:i + 64])
    t_calib = time.perf_counter() - t0
    int8 = convert_fx(prepared)

    note(2, "observers inserted by prepare_fx", n_obs)
    note(2, "calibration: 512 images", f"{t_calib:.2f}s")
    note(2, "leaf modules, float32", str(kind_counts(fp32)))
    note(2, "leaf modules, int8", str(kind_counts(int8)))
    note(2, "quantize / dequantize nodes in the graph",
         sum(1 for n in int8.graph.nodes
             if "quantize_per_tensor" in str(n.target) or "dequantize" in str(n.target)))

    # ------------------------------------------------------------------ [3]
    print("\n[3] accuracy, size, latency")
    acc8 = D.accuracy(int8, xte, yte)
    mb8 = module_mb(int8)
    agree = float((D.predict(int8, xte).argmax(1)
                   == D.predict(fp32, xte).argmax(1)).float().mean())
    lat = {}
    for b in (1, 32):
        xb = xte[:b].contiguous()
        res = D.interleaved({"fp32": lambda xb=xb: fp32(xb),
                             "int8": lambda xb=xb: int8(xb)},
                            rounds=9, calls=max(3, 64 // b))
        lat[b] = (res["fp32"]["median_ms"], res["int8"]["median_ms"])
    note(3, "accuracy: float32 / int8", f"{acc32:.4f} / {acc8:.4f}  "
                                        f"({100 * (acc8 - acc32):+.2f} points)")
    note(3, "predictions identical to float32", f"{100 * agree:.2f}%")
    note(3, "size: float32 / int8", f"{mb32:.3f} MB / {mb8:.3f} MB  "
                                    f"({mb32 / mb8:.2f}x)")
    for b, (a, c) in lat.items():
        note(3, f"latency batch {b}: float32 / int8 (ms)",
             f"{a:.3f} / {c:.3f}   {a / c:.2f}x")

    # ------------------------------------------------------------------ [4]
    print("\n[4] how much calibration data, and what kind")
    calib_rows = []
    for n in (1, 4, 16, 64, 256, 1024):
        model_n = ptq(fp32, xtr[:n])
        a = D.accuracy(model_n, xte, yte)
        calib_rows.append((f"{n} images", a))
        note(4, f"calibrated on {n:>4} real images", f"accuracy {a:.4f}")
    torch.manual_seed(0)
    wrong = {
        "gaussian noise": torch.randn(256, 3, 32, 32),
        "all zeros": torch.zeros(256, 3, 32, 32),
        "one image, repeated": xtr[:1].repeat(256, 1, 1, 1),
        "images x 20 (wrong scale)": xtr[:256] * 20,
    }
    for label, data in wrong.items():
        a = D.accuracy(ptq(fp32, data), xte, yte)
        calib_rows.append((label, a))
        note(4, f"calibrated on {label}", f"accuracy {a:.4f}")

    # ------------------------------------------------------------------ [5]
    print("\n[5] the observer, the granularity, and one flag that matters more")
    weight_obs = {
        "per-tensor": MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        "per-channel": PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    }
    obs_rows = []
    for reduce_range in (False, True):
        for w_name, w_obs in weight_obs.items():
            act = HistogramObserver.with_args(dtype=torch.quint8,
                                              qscheme=torch.per_tensor_affine,
                                              reduce_range=reduce_range)
            qmap = QConfigMapping().set_global(QConfig(activation=act, weight=w_obs))
            acc = D.accuracy(ptq(fp32, xtr[:256], qmap), xte, yte)
            obs_rows.append((f"reduce_range={reduce_range}", w_name, acc))
            note(5, f"reduce_range={str(reduce_range):<5} + {w_name:<12} weights",
                 f"accuracy {acc:.4f}")

    act_obs = {
        "MinMax": MinMaxObserver,
        "MovingAverageMinMax": MovingAverageMinMaxObserver,
        "Histogram": HistogramObserver,
    }
    obs_rows2 = []
    for a_name, cls in act_obs.items():
        qmap = QConfigMapping().set_global(QConfig(
            activation=cls.with_args(dtype=torch.quint8,
                                     qscheme=torch.per_tensor_affine,
                                     reduce_range=True),
            weight=weight_obs["per-channel"]))
        acc = D.accuracy(ptq(fp32, xtr[:256], qmap), xte, yte)
        obs_rows2.append((a_name, acc))
        note(5, f"activation observer: {a_name:<20} (reduce_range=True)",
             f"accuracy {acc:.4f}")

    # ------------------------------------------------------------------ [6]
    print("\n[6] static vs dynamic on layers where both are legal")
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).eval()
    torch.manual_seed(0)
    for p in mlp.parameters():
        nn.init.normal_(p, std=0.02) if p.dim() > 1 else nn.init.zeros_(p)
    mlp_dyn = torch.ao.quantization.quantize_dynamic(
        mlp, {nn.Linear: default_dynamic_qconfig}, dtype=torch.qint8)
    mlp_static = ptq(mlp, xtr[:256], example=xte[:32])
    xb = xte[:32].contiguous()
    res = D.interleaved({"float32": lambda: mlp(xb),
                         "dynamic int8": lambda: mlp_dyn(xb),
                         "static int8": lambda: mlp_static(xb)},
                        rounds=9, calls=20)
    ref = mlp(xb)
    for name, r in res.items():
        note(6, f"MLP batch 32, {name}", f"{r['median_ms']:.3f} ms")
    note(6, "dynamic vs static speed-up over float32",
         f"{res['float32']['median_ms'] / res['dynamic int8']['median_ms']:.2f}x / "
         f"{res['float32']['median_ms'] / res['static int8']['median_ms']:.2f}x")
    note(6, "max |output difference| vs float32: dynamic / static",
         f"{(mlp_dyn(xb) - ref).abs().max():.4f} / "
         f"{(mlp_static(xb) - ref).abs().max():.4f}")

    # --------------------------------------------------------------- output
    summary = {
        "acc": {"fp32": acc32, "int8": acc8}, "mb": {"fp32": mb32, "int8": mb8},
        "latency": {str(k): v for k, v in lat.items()},
        "calibration": calib_rows, "observers": obs_rows,
        "activation_observers": obs_rows2,
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    D.write_csv(os.path.join(OUT, "findings.csv"), FINDINGS, ["section", "name", "value"])
    figure(summary, acc32)
    torch.jit.save(torch.jit.script(int8), os.path.join(OUT, "small_cnn_int8.pt"))
    print(f"    wrote {os.path.join(OUT, 'small_cnn_int8.pt')} "
          f"({D.file_mb(os.path.join(OUT, 'small_cnn_int8.pt')):.3f} MB)")
    print(f"\ntotal {time.time() - t_start:.0f}s")


def figure(summary, acc32):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    ax = axes[0]
    rows = summary["calibration"]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [SERIES[1]] * 6 + [SERIES[2]] * (len(rows) - 6)
    ax.barh(range(len(rows)), vals, color=colors)
    ax.axvline(acc32, color=SERIES[0], linestyle="--", linewidth=1.3, label="float32")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, max(max(vals), acc32) * 1.15)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("calibration data: how much, and what", loc="left", fontsize=11)

    ax = axes[1]
    obs = summary["observers"]
    acts = ["reduce_range=False", "reduce_range=True"]
    width = 0.36
    for i, w in enumerate(["per-tensor", "per-channel"]):
        vals = [next(r[2] for r in obs if r[0] == a and r[1] == w) for a in acts]
        ax.bar(np.arange(len(acts)) + (i - 0.5) * width, vals, width,
               color=SERIES[i], label=f"{w} weights")
    ax.axhline(acc32, color=SERIES[2], linestyle="--", linewidth=1.3, label="float32")
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([a.replace("=", "=\n") for a in acts], fontsize=8.5)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(acc32, max(r[2] for r in obs)) * 1.25)
    ax.legend(frameon=False, fontsize=8.5, ncol=1)
    ax.set_title("one flag decides whether per-channel helps", loc="left", fontsize=11)

    ax = axes[2]
    bs = sorted(int(k) for k in summary["latency"])
    idx = np.arange(len(bs))
    f = [summary["latency"][str(b)][0] for b in bs]
    q = [summary["latency"][str(b)][1] for b in bs]
    ax.bar(idx - 0.19, f, 0.36, color=SERIES[0], label="float32")
    ax.bar(idx + 0.19, q, 0.36, color=SERIES[1], label="static int8")
    for i, (a, c) in enumerate(zip(f, q)):
        ax.text(i, max(a, c) * 1.03, f"{a / c:.2f}x", ha="center", fontsize=9)
    ax.set_xticks(idx); ax.set_xticklabels([f"batch {b}" for b in bs])
    ax.set_ylabel("ms per call")
    ax.set_ylim(0, max(f + q) * 1.2)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("latency", loc="left", fontsize=11)

    fig.tight_layout()
    path = os.path.join(OUT, "static_quant.png")
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
