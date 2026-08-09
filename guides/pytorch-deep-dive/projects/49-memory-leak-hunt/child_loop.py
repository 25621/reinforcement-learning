"""One training loop, in a fresh process, printing its RSS growth as JSON.

Section 6 of `run.py` needs to change glibc's behaviour, and glibc reads its
tuning environment variables **once, at process start**. Setting them from
inside a running Python process is too late — so the measurement has to happen
in a child.
"""

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "48-nan-forensics"))
import debug_lib as D  # noqa: E402

torch.set_num_threads(4)
MODE = sys.argv[1]
STEPS = int(sys.argv[2])
D_MODEL, DEPTH, BATCH = 1024, 6, 256

torch.manual_seed(0)
model = nn.Sequential(*sum([[nn.Linear(D_MODEL, D_MODEL), nn.ReLU()]
                            for _ in range(DEPTH)], []))
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.randn(BATCH, D_MODEL)

keep = []
base = D.rss_mb()
for _ in range(STEPS):
    out = model(x)
    loss = out.pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if MODE == "append loss":
        keep.append(loss)
    else:
        keep.append(loss.item())

n, byts = D.live_tensor_bytes()
print(json.dumps({"mode": MODE, "steps": STEPS, "rss": D.rss_mb() - base,
                  "census_mb": byts / 1e6,
                  "mmap_threshold": os.environ.get("MALLOC_MMAP_THRESHOLD_", "default"),
                  "trim_threshold": os.environ.get("MALLOC_TRIM_THRESHOLD_", "default")}))
