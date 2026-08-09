"""Project 42 - Export a CNN to ONNX and check it really is the same model.

Six sections:

  1. the two exporters (legacy TorchScript tracing vs the torch.export one)
  2. numerical verification against ONNX Runtime on real test images
  3. the tracing trap: control flow gets baked in, silently
  4. fixed vs dynamic batch dimension
  5. latency: PyTorch eager vs ONNX Runtime at batch 1 / 8 / 32
  6. what the exported graph actually contains (BatchNorm has vanished)

Run:  python3 run.py        (~3 minutes, plus ~2.5 min the first time to train the CNN)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)

import onnx
import onnxruntime as ort

import deploy_lib as D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))
from plot_style import SERIES, style_axes  # noqa: E402

OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

FINDINGS: list[tuple] = []


def note(section, name, value):
    FINDINGS.append((section, name, value))
    print(f"    {name:<44} {value}")


def onnx_ops(path):
    graph = onnx.load(path).graph
    return Counter(n.op_type for n in graph.node)


def ort_session(path, threads=6):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    return ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])


# ==========================================================================
def section1(model, sample):
    print("\n[1] two exporters")
    legacy = os.path.join(DATA, "cnn_legacy.onnx")
    dynamo = os.path.join(DATA, "cnn_dynamo.onnx")

    t0 = time.perf_counter()
    torch.onnx.export(model, (sample,), legacy, input_names=["x"],
                      output_names=["logits"], dynamo=False)
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    torch.onnx.export(model, (sample,), dynamo, input_names=["x"],
                      output_names=["logits"], dynamo=True)
    t_dynamo = time.perf_counter() - t0

    note(1, "legacy (tracing) export time (s)", f"{t_legacy:.2f}")
    note(1, "dynamo (torch.export) export time (s)", f"{t_dynamo:.2f}")
    note(1, "legacy file (MB)", f"{D.file_mb(legacy):.3f}")
    note(1, "dynamo file (MB)", f"{D.file_mb(dynamo):.3f}")
    side = dynamo + ".data"
    has_side = os.path.exists(side)
    note(1, "dynamo wrote a separate weights file", str(has_side))
    if has_side:
        note(1, "  that sidecar file (MB)", f"{D.file_mb(side):.3f}")

    ops_l, ops_d = onnx_ops(legacy), onnx_ops(dynamo)
    note(1, "legacy nodes", f"{sum(ops_l.values())}  {dict(ops_l)}")
    note(1, "dynamo nodes", f"{sum(ops_d.values())}  {dict(ops_d)}")
    note(1, "opset", onnx.load(legacy).opset_import[0].version)
    return legacy, dynamo


# ==========================================================================
def section2(model, legacy, dynamo, x, y):
    print("\n[2] does the exported graph compute the same thing?")
    torch_logits = D.predict(model, x).numpy()
    rows = []
    for name, path in [("legacy", legacy), ("dynamo", dynamo)]:
        sess = ort_session(path)
        # both were exported with batch 1, so feed one image at a time
        outs = np.concatenate([sess.run(None, {"x": x[i:i + 1].numpy()})[0]
                               for i in range(len(x))])
        diff = np.abs(outs - torch_logits)
        agree = float((outs.argmax(1) == torch_logits.argmax(1)).mean())
        acc = float((outs.argmax(1) == y.numpy()).mean())
        rows.append((name, diff.max(), diff.mean(), agree, acc))
        note(2, f"{name}: max |logit difference|", f"{diff.max():.3e}")
        note(2, f"{name}: mean |logit difference|", f"{diff.mean():.3e}")
        note(2, f"{name}: predictions identical to PyTorch", f"{agree * 100:.2f}%")
        note(2, f"{name}: accuracy on {len(x)} images", f"{acc:.4f}")
    torch_acc = float((torch_logits.argmax(1) == y.numpy()).mean())
    note(2, "PyTorch accuracy on the same images", f"{torch_acc:.4f}")

    # How big is "the same input, twice" in float32? A yardstick for the numbers above.
    a = D.predict(model, x[:256]).numpy()
    b = D.predict(model, x[:256], batch=7).numpy()
    note(2, "PyTorch vs PyTorch at a different batch size", f"{np.abs(a - b).max():.3e}")
    return rows, np.abs(
        np.concatenate([ort_session(dynamo).run(None, {"x": x[i:i + 1].numpy()})[0]
                        for i in range(512)]) - torch_logits[:512])


# ==========================================================================
class Branchy(nn.Module):
    """Doubles inputs that are positive on average, halves the rest."""

    def forward(self, x):
        if x.mean() > 0:
            return x * 2.0
        return x * 0.5


class Cond(nn.Module):
    """The same rule written so the exporter can see both branches."""

    def forward(self, x):
        return torch.cond(x.mean() > 0, lambda t: t * 2.0, lambda t: t * 0.5, (x,))


def section3():
    print("\n[3] the tracing trap: control flow")
    pos, neg = torch.ones(1, 4), -torch.ones(1, 4)
    ref_neg = Branchy()(neg)[0, 0].item()

    traced = os.path.join(DATA, "branch_traced.onnx")
    torch.onnx.export(Branchy().eval(), (pos,), traced, input_names=["x"], dynamo=False)
    got = ort_session(traced).run(None, {"x": neg.numpy()})[0][0, 0]
    note(3, "traced with a positive input, run on negative", f"{got:+.2f}")
    note(3, "  PyTorch on the same negative input", f"{ref_neg:+.2f}")
    note(3, "  traced graph contains a branch node",
         str(any(n.op_type == "If" for n in onnx.load(traced).graph.node)))

    try:
        torch.onnx.export(Branchy().eval(), (pos,), os.path.join(DATA, "branch_dyn.onnx"),
                          input_names=["x"], dynamo=True)
        note(3, "torch.export on the same model", "succeeded (unexpected)")
    except Exception as exc:
        note(3, "torch.export on the same model", f"refused: {type(exc).__name__}")

    cond = os.path.join(DATA, "branch_cond.onnx")
    torch.onnx.export(Cond().eval(), (pos,), cond, input_names=["x"], dynamo=True)
    sess = ort_session(cond)
    p = sess.run(None, {"x": pos.numpy()})[0][0, 0]
    n = sess.run(None, {"x": neg.numpy()})[0][0, 0]
    note(3, "torch.cond version: positive input", f"{p:+.2f}")
    note(3, "torch.cond version: negative input", f"{n:+.2f}")
    note(3, "  its graph", " ".join(x.op_type for x in onnx.load(cond).graph.node))
    return got, ref_neg


# ==========================================================================
def section4(model, sample):
    print("\n[4] fixed vs dynamic batch dimension")
    fixed = os.path.join(DATA, "cnn_fixed.onnx")
    dynb = os.path.join(DATA, "cnn_dynbatch.onnx")
    torch.onnx.export(model, (sample,), fixed, input_names=["x"],
                      output_names=["logits"], dynamo=True)
    batch = torch.export.Dim("batch")
    torch.onnx.export(model, (sample,), dynb, input_names=["x"], output_names=["logits"],
                      dynamo=True, dynamic_shapes={"x": {0: batch}})

    for name, path in [("fixed", fixed), ("dynamic", dynb)]:
        sess = ort_session(path)
        note(4, f"{name}: declared input shape", str(sess.get_inputs()[0].shape))
        for b in (1, 8):
            try:
                out = sess.run(None, {"x": np.random.randn(b, 3, 32, 32).astype(np.float32)})[0]
                note(4, f"  {name}: batch {b}", f"ok, output {out.shape}")
            except Exception as exc:
                msg = str(exc).split("Please fix")[0].strip().replace("\n", " ")
                note(4, f"  {name}: batch {b}", f"FAILED: {msg[-90:]}")
    return dynb


# ==========================================================================
def section5(model, dynb, x):
    print("\n[5] latency: PyTorch eager vs ONNX Runtime")
    sess = ort_session(dynb)
    rows = []
    for b in (1, 8, 32):
        xb = x[:b].contiguous()
        nb = xb.numpy()

        def torch_call(xb=xb):
            with torch.no_grad():
                model(xb)

        def ort_call(nb=nb):
            sess.run(None, {"x": nb})

        # equal work per round (~64 images), so every batch size is measured
        # over a comparable amount of computation
        res = D.interleaved({"pytorch": torch_call, "onnxruntime": ort_call},
                            rounds=9, calls=max(3, 64 // b))
        t_pt = res["pytorch"]["median_ms"]
        t_or = res["onnxruntime"]["median_ms"]
        rows.append((b, t_pt, t_or, b * 1000 / t_pt, b * 1000 / t_or))
        note(5, f"batch {b:>2}: PyTorch / ORT (ms)",
             f"{t_pt:7.2f} / {t_or:7.2f}   ORT is {t_pt / t_or:.2f}x")
        note(5, f"batch {b:>2}: images per second",
             f"{b * 1000 / t_pt:8.0f} / {b * 1000 / t_or:8.0f}")
    return rows


# ==========================================================================
def section6(model, legacy):
    print("\n[6] what survived the export")
    n_modules = sum(1 for _ in model.modules())
    kinds = Counter(type(m).__name__ for m in model.modules())
    graph = onnx.load(legacy).graph
    ops = Counter(n.op_type for n in graph.node)
    note(6, "PyTorch modules", f"{n_modules}  {dict(kinds)}")
    note(6, "ONNX nodes", f"{len(graph.node)}  {dict(ops)}")
    note(6, "BatchNorm layers in the module tree", kinds.get("BatchNorm2d", 0))
    note(6, "BatchNormalization nodes in the graph", ops.get("BatchNormalization", 0))
    note(6, "initializers (weight tensors) in the file", len(graph.initializer))

    # A conv whose weights were rescaled by the folded BatchNorm no longer
    # matches the eager weights.
    conv0 = model.features[0].weight.detach()
    init = {i.name: i for i in graph.initializer}
    fold = [onnx.numpy_helper.to_array(v) for v in init.values()
            if onnx.numpy_helper.to_array(v).shape == tuple(conv0.shape)]
    if fold:
        note(6, "|folded conv0 weight - eager conv0 weight| max",
             f"{np.abs(fold[0] - conv0.numpy()).max():.3e}")
    return kinds, ops


# ==========================================================================
def figure(lat_rows, diff, ops_kinds):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    bs = [r[0] for r in lat_rows]
    idx = np.arange(len(bs))
    ax = axes[0]
    ax.bar(idx - 0.19, [r[1] for r in lat_rows], 0.36, color=SERIES[0], label="PyTorch")
    ax.bar(idx + 0.19, [r[2] for r in lat_rows], 0.36, color=SERIES[1], label="ONNX Runtime")
    ax.set_xticks(idx); ax.set_xticklabels([f"batch {b}" for b in bs])
    ax.set_title("latency per call", loc="left", fontsize=11)
    ax.set_ylabel("ms"); ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(bs, [r[3] for r in lat_rows], "o-", color=SERIES[0], label="PyTorch")
    ax.plot(bs, [r[4] for r in lat_rows], "o-", color=SERIES[1], label="ONNX Runtime")
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_title("throughput", loc="left", fontsize=11)
    ax.set_xlabel("batch size"); ax.set_ylabel("images / second")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.hist(diff.ravel(), bins=40, color=SERIES[2])
    ax.set_yscale("log")
    ax.set_title("|ONNX logit - PyTorch logit|", loc="left", fontsize=11)
    ax.set_xlabel("absolute difference"); ax.set_ylabel("count (log)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "onnx_export.png"), facecolor="#fcfcfb",
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(OUT, 'onnx_export.png')}")


def main():
    t_start = time.time()
    print(f"torch {torch.__version__} | onnx {onnx.__version__} | "
          f"onnxruntime {ort.__version__} | threads {torch.get_num_threads()}")
    model = D.get_trained_cnn()
    x, y = D.load_cifar("test", 2000)
    sample = x[:1].contiguous()

    legacy, dynamo = section1(model, sample)
    _, diff = section2(model, legacy, dynamo, x, y)
    section3()
    dynb = section4(model, sample)
    lat_rows = section5(model, dynb, x)
    kinds, _ = section6(model, legacy)
    figure(lat_rows, diff, kinds)

    D.write_csv(os.path.join(OUT, "findings.csv"), FINDINGS,
                ["section", "name", "value"])
    D.write_csv(os.path.join(OUT, "latency.csv"),
                [(b, f"{a:.3f}", f"{c:.3f}", f"{d:.1f}", f"{e:.1f}")
                 for b, a, c, d, e in lat_rows],
                ["batch", "pytorch_ms", "ort_ms", "pytorch_img_s", "ort_img_s"])
    # a copy of the exported model, small enough to keep in the repo
    import shutil
    shutil.copy(os.path.join(DATA, "cnn_dynbatch.onnx"),
                os.path.join(OUT, "small_cnn.onnx"))
    side = os.path.join(DATA, "cnn_dynbatch.onnx.data")
    if os.path.exists(side):
        shutil.copy(side, os.path.join(OUT, "small_cnn.onnx.data"))
    print(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
