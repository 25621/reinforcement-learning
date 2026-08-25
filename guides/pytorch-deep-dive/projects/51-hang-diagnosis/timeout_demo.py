"""Two ranks; rank 1 never reaches the second collective. Rank 0 reports.

The point of this script is the `timeout=` argument. Without it the process
group waits for gloo's default (30 minutes) and prints nothing in the meantime,
so a hang produces no information at all. With it, the wait ends in an
exception that names the collective, the rank, and the elapsed time.

Usage:  MASTER_ADDR=... MASTER_PORT=... PG_TIMEOUT_S=15 python3 timeout_demo.py <rank>
"""

import datetime
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

RANK = int(sys.argv[1])
TIMEOUT = float(os.environ.get("PG_TIMEOUT_S", "15"))
# "sleep": rank 1 stays alive and never arrives (a true hang, ended by the
# timeout). "crash": rank 1 exits, which closes its sockets and produces a fast
# error instead. The difference is the point of section 7.
STRAGGLER = os.environ.get("STRAGGLER_MODE", "sleep")

torch.set_num_threads(1)
dist.init_process_group(
    "gloo", rank=RANK, world_size=2,
    timeout=datetime.timedelta(seconds=TIMEOUT),
)

t = torch.ones(4)
dist.all_reduce(t)                       # both ranks arrive here
print(f"rank {RANK} passed collective 1", flush=True)

if RANK == 1:
    if STRAGGLER == "crash":
        raise SystemExit("rank 1 hit a bad batch and exited")
    time.sleep(3600)                     # ...and rank 1 wanders off, still alive

t0 = time.time()
try:
    dist.all_reduce(t)                   # rank 0 alone
    print(f"rank {RANK} passed collective 2", flush=True)
except Exception as exc:                 # noqa: BLE001 - we want whatever it is
    print(f"TIMEOUT after {time.time() - t0:.1f}s: "
          f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}", flush=True)
