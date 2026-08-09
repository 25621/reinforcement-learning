"""A process that hangs (or pretends to) in one of five ways, on demand.

Usage:  python3 victim.py <mode> <stack_file> [extra]

Every mode prints `READY <pid>` on stdout when it is about to enter the bad
region, then never returns. `run.py` waits for that line, lets the situation
settle, and points `triage.py` at the process.

The two lines at the top of `main()` are the most important lines in this
project:

    faulthandler.register(signal.SIGUSR1, file=..., all_threads=True)

They cost nothing, they run at startup, and they are what makes a hang
*answerable later*. You cannot add them to a process that is already stuck.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODE = sys.argv[1]
STACK_FILE = sys.argv[2]
EXTRA = sys.argv[3] if len(sys.argv) > 3 else ""


def ready():
    print(f"READY {os.getpid()}", flush=True)


# ---------------------------------------------------------------------------
# 1. A DataLoader whose workers inherit a held lock
# ---------------------------------------------------------------------------

def dataloader_fork():
    """The classic fork deadlock.

    `num_workers > 0` starts worker processes with `fork` by default. `fork`
    copies the parent's *memory*, including the state of every lock — but only
    the calling thread. If some other thread was holding a lock at the instant
    of the fork, the child wakes up with a lock that is permanently held by a
    thread that does not exist in the child. The first worker that tries to
    take it waits forever.

    Real versions of this involve logging, CUDA contexts, OpenMP thread pools,
    HDF5 handles, database connections, and OpenCV. The mechanism is identical.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    lock = threading.Lock()

    class DS(Dataset):
        def __len__(self):
            return 256

        def __getitem__(self, i):
            with lock:                       # held by `holder` at fork time
                return torch.zeros(8)

    def holder():
        lock.acquire()
        time.sleep(3600)

    threading.Thread(target=holder, daemon=True).start()
    time.sleep(0.3)
    ready()
    dl = DataLoader(DS(), batch_size=16, num_workers=2)
    for _ in dl:
        pass


# ---------------------------------------------------------------------------
# 2. One rank of a 2-rank job dies; the other waits for it
# ---------------------------------------------------------------------------

def ddp_straggler():
    """Rank 1 is alive but never arrives. Rank 0 waits at `all_reduce`, forever.

    This is the shape of most real distributed hangs, and its signature is
    misleading: the rank you find stuck is the *innocent* one, and its stack
    points at a perfectly correct line of your code.

    Note the deliberate choice of "alive but stuck" over "crashed". A rank that
    *exits* closes its TCP connections, and gloo notices immediately — you get
    `Connection closed by peer` in seconds, which is unpleasant but is not a
    hang. A rank that is alive and simply not participating produces no signal
    at all until the timeout fires. `run.py` measures both.
    """
    import torch
    import torch.distributed as dist

    rank = int(EXTRA)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    t = torch.ones(4) * (rank + 1)
    dist.all_reduce(t)                       # step 1: both ranks arrive
    if rank == 1:
        ready()
        time.sleep(3600)                     # alive, busy elsewhere, never arrives
    ready()
    for _ in range(1000):
        dist.all_reduce(t)                   # step 2: rank 0 alone. forever.
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# 3. A loop that never ends because the comparison is against a NaN
# ---------------------------------------------------------------------------

def nan_loop():
    """`while loss > tol` where loss is NaN.

    Every comparison against a NaN is False, so `loss > tol` is False and
    `not (loss > tol)` is True... but written the other way round —
    `while not (loss < tol)` — the loop never exits. This process is not stuck;
    it is running as fast as it can, forever. Project 48 is about where the NaN
    came from; this is about what it does to your control flow.
    """
    import torch

    torch.set_num_threads(2)
    x = torch.randn(600, 600)
    loss = torch.tensor(float("nan"))
    ready()
    it = 0
    while not (loss < 1e-3):                 # NaN < 1e-3 is False, forever
        x = (x @ x).tanh()
        it += 1


# ---------------------------------------------------------------------------
# 4. Not hung at all: too many threads for the cores
# ---------------------------------------------------------------------------

def oversubscribed():
    """Real progress, at a crawl.

    Four processes each asking for every core means the operating system spends
    its time swapping threads on and off cores instead of doing arithmetic. The
    job still moves; it just looks dead from the outside, which is exactly why
    it belongs in a hang-triage project.
    """
    import torch

    torch.set_num_threads(int(EXTRA or 12))
    x = torch.randn(1400, 1400)
    ready()
    beat = STACK_FILE + ".beat"
    for i in range(100000):
        x = (x @ x).tanh()
        with open(beat, "w") as fh:          # a heartbeat: proof of progress
            fh.write(str(i))


# ---------------------------------------------------------------------------
# 5. Two threads, two locks, opposite orders
# ---------------------------------------------------------------------------

def lock_order():
    """The textbook deadlock, in one process.

    Thread A takes lock 1 then wants lock 2. Thread B takes lock 2 then wants
    lock 1. Neither will let go. Nothing about this is PyTorch-specific, and
    that is the point: not every hang in a training script is a training bug.
    """
    a, b = threading.Lock(), threading.Lock()

    def worker(first, second, name):
        with first:
            time.sleep(0.5)
            with second:
                pass

    t1 = threading.Thread(target=worker, args=(a, b, "A"), daemon=True)
    t2 = threading.Thread(target=worker, args=(b, a, "B"), daemon=True)
    t1.start(); t2.start()
    time.sleep(1.0)
    ready()
    t1.join(); t2.join()


MODES = {
    "dataloader_fork": dataloader_fork,
    "ddp_straggler": ddp_straggler,
    "nan_loop": nan_loop,
    "oversubscribed": oversubscribed,
    "lock_order": lock_order,
}


def main():
    # Two lines. Register them at startup in every long-running job you own.
    fh = open(STACK_FILE, "w")
    faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True)
    MODES[MODE]()


if __name__ == "__main__":
    main()
