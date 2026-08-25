"""The half of project 43 that must run inside the ExecuTorch virtual environment.

ExecuTorch pins its own PyTorch version (1.4.0 needs torch >= 2.13), so it cannot
share the interpreter the rest of this guide uses. `run.py` launches this file with
the venv's python and reads the JSON it writes.

Produces, in `outputs/`:
  cnn_portable.pte      every operator executed by ExecuTorch's own reference kernels
  cnn_xnnpack.pte       the whole graph handed to the XNNPACK backend
  cnn_xnnpack_int8.pte  the same, quantized to int8
  cnn_xnnpack_dyn.pte   a dynamic (bounded) batch dimension
  et_results.json       everything measured
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "42-export-to-onnx")
sys.path.insert(0, LIB)

import deploy_lib as D  # noqa: E402

OUT = os.path.join(HERE, "outputs")
PTE = os.path.join(HERE, "data")          # the .pte files, all but one gitignored
os.makedirs(OUT, exist_ok=True)
os.makedirs(PTE, exist_ok=True)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: E402
    XnnpackPartitioner,
)
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

R = {"torch": torch.__version__}
N_EVAL = 512


def plan_stats(prog):
    plan = prog.executorch_program.execution_plan[0]
    graph = prog.exported_program().graph
    return {
        "operators": len(plan.operators),
        "delegates": len(plan.delegates),
        "arena_bytes": int(max(plan.non_const_buffer_sizes)),
        "graph_call_nodes": sum(1 for n in graph.nodes if n.op == "call_function"),
    }


def save(prog, name):
    path = os.path.join(PTE, name)
    with open(path, "wb") as fh:
        fh.write(prog.buffer)
    return path


def bench(method, inputs, n=100, warmup=10):
    for _ in range(warmup):
        method.execute(inputs)
    t0 = time.perf_counter()
    for _ in range(n):
        method.execute(inputs)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    torch.set_num_threads(6)
    model = D.get_trained_cnn()
    model.eval()
    x, y = D.load_cifar("test", N_EVAL)
    sample = (x[:1].contiguous(),)

    # ---------------------------------------------------------------- exports
    t0 = time.perf_counter()
    ep = torch.export.export(model, sample)
    R["export_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prog_p = to_edge(ep).to_executorch()
    R["portable_lower_s"] = time.perf_counter() - t0
    p_portable = save(prog_p, "cnn_portable.pte")

    t0 = time.perf_counter()
    prog_x = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    R["xnnpack_lower_s"] = time.perf_counter() - t0
    p_xnn = save(prog_x, "cnn_xnnpack.pte")

    # int8, the way a phone build would actually do it (PT2E quantization,
    # then the same XNNPACK partitioner)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    t0 = time.perf_counter()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True))
    gm = prepare_pt2e(torch.export.export(model, sample).module(), quantizer)
    with torch.no_grad():                      # calibration: 128 real images
        for i in range(0, 128, 32):
            gm(x[i:i + 32])
    gm = convert_pt2e(gm)
    prog_q = to_edge_transform_and_lower(
        torch.export.export(gm, sample), partitioner=[XnnpackPartitioner()]).to_executorch()
    R["int8_lower_s"] = time.perf_counter() - t0
    p_int8 = save(prog_q, "cnn_xnnpack_int8.pte")

    R["programs"] = {}
    for name, prog, path in [("portable", prog_p, p_portable),
                             ("xnnpack", prog_x, p_xnn),
                             ("xnnpack_int8", prog_q, p_int8)]:
        R["programs"][name] = dict(plan_stats(prog), mb=D.file_mb(path))

    # ------------------------------------------------------- run and verify
    runtime = Runtime.get()
    ref = D.predict(model, x).numpy()
    R["runs"] = {}
    for name, path in [("portable", p_portable), ("xnnpack", p_xnn),
                       ("xnnpack_int8", p_int8)]:
        method = runtime.load_program(path).load_method("forward")
        n_acc = N_EVAL if name != "portable" else 64   # portable is ~600x slower
        outs = []
        t0 = time.perf_counter()
        for i in range(n_acc):
            outs.append(method.execute([x[i:i + 1].contiguous()])[0])
        acc_s = time.perf_counter() - t0
        outs = torch.cat(outs).numpy()
        R["runs"][name] = {
            "max_abs_diff": float(abs(outs - ref[:n_acc]).max()),
            "agree_pct": float((outs.argmax(1) == ref[:n_acc].argmax(1)).mean() * 100),
            "accuracy": float((outs.argmax(1) == y[:n_acc].numpy()).mean()),
            "n_images": n_acc,
            "eval_s": acc_s,
        }

    # ------------------------------------------------------------- latency
    inputs = [x[:1].contiguous()]
    methods = {name: runtime.load_program(p).load_method("forward")
               for name, p in [("portable", p_portable), ("xnnpack", p_xnn),
                               ("xnnpack_int8", p_int8)]}

    def eager():
        with torch.no_grad():
            model(x[:1])

    # The three fast runtimes are timed interleaved (see deploy_lib.interleaved):
    # this machine is shared, and back-to-back timings reordered the ranking
    # between runs until we rotated them.
    fast = D.interleaved({"xnnpack": lambda: methods["xnnpack"].execute(inputs),
                          "xnnpack_int8": lambda: methods["xnnpack_int8"].execute(inputs),
                          "pytorch_eager": eager},
                         rounds=9, calls=50)
    lat = {name: v["median_ms"] for name, v in fast.items()}
    lat["portable"] = bench(methods["portable"], inputs, n=20, warmup=2)
    R["latency_ms_batch1"] = lat
    R["latency_spread"] = {name: [v["min_ms"], v["max_ms"]] for name, v in fast.items()}

    # ------------------------------------------------- bounded dynamic shape
    dim = torch.export.Dim("b", min=1, max=16)
    ep_dyn = torch.export.export(model, (x[:2].contiguous(),), dynamic_shapes={"x": {0: dim}})
    prog_d = to_edge_transform_and_lower(ep_dyn,
                                         partitioner=[XnnpackPartitioner()]).to_executorch()
    p_dyn = save(prog_d, "cnn_xnnpack_dyn.pte")
    R["programs"]["xnnpack_dyn"] = dict(plan_stats(prog_d), mb=D.file_mb(p_dyn))
    R["declared_max_batch"] = 16

    method_d = runtime.load_program(p_dyn).load_method("forward")
    method_s = methods["xnnpack"]
    R["dynamic_batches"] = {}
    R["static_batches"] = {}
    for b in (1, 8, 15, 16):
        for tag, meth in [("dynamic_batches", method_d), ("static_batches", method_s)]:
            try:
                out = meth.execute([x[:b].contiguous()])[0]
                R[tag][str(b)] = f"ok {tuple(out.shape)}"
            except Exception as exc:                       # noqa: BLE001
                R[tag][str(b)] = f"FAILED ({type(exc).__name__})"

    # Does asking for a larger maximum actually get you one?
    dim32 = torch.export.Dim("b", min=1, max=32)
    prog_d32 = to_edge_transform_and_lower(
        torch.export.export(model, (x[:2].contiguous(),), dynamic_shapes={"x": {0: dim32}}),
        partitioner=[XnnpackPartitioner()]).to_executorch()
    p_d32 = save(prog_d32, "cnn_xnnpack_dyn32.pte")
    method_d32 = runtime.load_program(p_d32).load_method("forward")
    accepted = []
    for b in (1, 8, 15, 16, 24, 32):
        try:
            method_d32.execute([x[:b].contiguous()])
            accepted.append(b)
        except Exception:                                  # noqa: BLE001
            pass
    R["declared_max_32_accepts"] = accepted
    R["programs"]["xnnpack_dyn32"] = dict(plan_stats(prog_d32), mb=D.file_mb(p_d32))

    # ------------------------------------------------------------ file sizes
    ckpt = os.path.join(LIB, "checkpoints", "small_cnn.pt")
    R["sizes_mb"] = {
        "state_dict .pt": D.file_mb(ckpt) if os.path.exists(ckpt) else None,
        "onnx (graph + sidecar)": None,
    }
    onnx_dir = os.path.join(LIB, "outputs")
    o1 = os.path.join(onnx_dir, "small_cnn.onnx")
    o2 = o1 + ".data"
    if os.path.exists(o1):
        R["sizes_mb"]["onnx (graph + sidecar)"] = D.file_mb(o1) + (
            D.file_mb(o2) if os.path.exists(o2) else 0.0)
    for name in ("portable", "xnnpack", "xnnpack_int8"):
        R["sizes_mb"][f"pte {name}"] = R["programs"][name]["mb"]

    # the one artefact small enough to keep in the repo: the int8 mobile build
    import shutil

    shutil.copy(p_int8, os.path.join(OUT, "cnn_xnnpack_int8.pte"))

    with open(os.path.join(OUT, "et_results.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
