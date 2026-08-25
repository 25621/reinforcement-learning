"""Measure what a brand-new process pays before it can answer one request.

Printed as JSON on the last line so `run.py` can read it.
"""
import json
import os
import sys
import time

t0 = time.perf_counter()
import torch                                                     # noqa: E402

t_import = time.perf_counter() - t0

torch.set_num_threads(6)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "42-export-to-onnx"))
import deploy_lib as D                                           # noqa: E402

t1 = time.perf_counter()
model = D.SmallCNN().eval()
model.load_state_dict(torch.load(
    os.path.join(D.CKPT_DIR, "small_cnn.pt"), map_location="cpu"))
t_load = time.perf_counter() - t1

x = torch.zeros(1, 3, 32, 32)
with torch.no_grad():
    t2 = time.perf_counter()
    model(x)
    t_first = time.perf_counter() - t2
    t3 = time.perf_counter()
    model(x)
    t_second = time.perf_counter() - t3

print(json.dumps({"import_s": t_import, "load_ms": t_load * 1e3,
                  "first_ms": t_first * 1e3, "second_ms": t_second * 1e3,
                  "total_s": time.perf_counter() - t0}))
