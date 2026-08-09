"""Shared helpers for Phase 7 (distributed training) — projects 36-41.

Everything here exists because this machine has no usable GPU (a GTX 1070 Ti,
compute capability sm_61, which this PyTorch build has no kernels for). So every
project runs *real* `torch.distributed` code with the **gloo** backend over
loopback TCP, one OS process per "device". The code you write is byte-for-byte
what you would write for NCCL on 8 GPUs; only the backend string changes.

Two rules this file enforces, both learned the hard way:

1. **spawn, never fork.** A forked child inherits PyTorch's thread pools in a
   broken state and deadlocks. `mp.get_context("spawn")` starts a fresh
   interpreter.
2. **`CUDA_VISIBLE_DEVICES=""` in every child.** Otherwise `torch` sees the
   unusable GPU, and anything that auto-picks an "accelerator" (FSDP does)
   crashes with `no kernel image is available for execution on the device`.

And one rule about *returning* results: a torch tensor sent through a
`multiprocessing.Queue` is not copied — torch passes a shared-memory file
descriptor instead. If the child exits before the parent reads it, the parent
gets `EOFError`. So `_entry` serialises the whole result to plain bytes with
`torch.save` first; the bytes are self-contained and the child can exit freely.
"""

from __future__ import annotations

import io
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# process-group plumbing
# ---------------------------------------------------------------------------


def free_port() -> int:
    """Ask the OS for a port nobody is using, then release it.

    Every process group needs a rendezvous address. Hard-coding 29500 means a
    leftover process from a previous run silently steals the port, so we ask
    for a fresh one each time.
    """
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank, world, port, threads, fn, args, extra_env, q):
    """Runs inside each spawned process: set env, join the group, call `fn`."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    for k, v in (extra_env or {}).items():
        os.environ[k] = str(v)

    torch.set_num_threads(threads)
    result = None
    try:
        # PG_TIMEOUT_S turns an infinite wait into an error with a message.
        # Project 40 relies on it; everyone else gets the 30-minute default.
        kw = {}
        if os.environ.get("PG_TIMEOUT_S"):
            import datetime

            kw["timeout"] = datetime.timedelta(seconds=float(os.environ["PG_TIMEOUT_S"]))
        dist.init_process_group(backend="gloo", rank=rank, world_size=world, **kw)
        result = fn(rank, world, *args)
    except BaseException as exc:  # noqa: BLE001 - report, never hang the parent
        import traceback

        result = {"__error__": f"{type(exc).__name__}: {exc}",
                  "__traceback__": traceback.format_exc()[-2000:]}
    finally:
        try:
            buf = io.BytesIO()
            torch.save(result, buf)          # -> plain bytes, no shared memory
            q.put((rank, buf.getvalue()))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def launch(fn, world_size, threads=2, args=(), timeout=600.0, env=None,
           raise_on_error=True):
    """Run `fn(rank, world_size, *args)` in `world_size` fresh processes.

    `fn` must be a module-level function (spawn pickles it by name) and should
    return something picklable. Returns the list of results ordered by rank.
    """
    ctx = mp.get_context("spawn")
    port = free_port()
    q = ctx.Queue()
    procs = [ctx.Process(target=_entry,
                         args=(r, world_size, port, threads, fn, args, env, q))
             for r in range(world_size)]
    for p in procs:
        p.start()

    results = {}
    deadline = time.time() + timeout
    try:
        for _ in range(world_size):
            rank, blob = q.get(timeout=max(1.0, deadline - time.time()))
            results[rank] = torch.load(io.BytesIO(blob), weights_only=False)
    except Exception as exc:  # queue timeout == someone hung or died
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        raise RuntimeError(
            f"launch({fn.__name__}, world_size={world_size}) did not finish: {exc}"
        ) from exc

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    ordered = [results[r] for r in range(world_size)]
    if raise_on_error:
        for r, res in enumerate(ordered):
            if isinstance(res, dict) and "__error__" in res:
                raise RuntimeError(f"rank {r} failed: {res['__error__']}\n"
                                   f"{res.get('__traceback__', '')}")
    return ordered


# ---------------------------------------------------------------------------
# timing on a shared machine
# ---------------------------------------------------------------------------


def interleaved(candidates, rounds=3, warmup=1):
    """Time several closures fairly on a machine other people are also using.

    Running A ten times and then B ten times measures whatever else the machine
    was doing during each block. Round-robin (A B A B A B ...) spreads that
    noise evenly across candidates. We report the *median* of each candidate's
    per-round times, and also the minimum, which is the closest thing to an
    interference-free measurement.

    `candidates` maps a name to a zero-argument callable that does the work.
    """
    for _ in range(warmup):
        for fn in candidates.values():
            fn()

    times = {name: [] for name in candidates}
    for _ in range(rounds):
        for name, fn in candidates.items():
            t0 = time.perf_counter()
            fn()
            times[name].append(time.perf_counter() - t0)
    return {name: {"median": statistics.median(ts), "min": min(ts), "all": ts}
            for name, ts in times.items()}


# ---------------------------------------------------------------------------
# memory accounting (no CUDA allocator, so we count bytes ourselves)
# ---------------------------------------------------------------------------


def tensor_bytes(t) -> int:
    """Bytes actually stored on this rank for one tensor.

    For a normal tensor that is `numel * itemsize`. For a **DTensor** (what FSDP
    turns parameters into) the `.shape` is the *global* shape — the whole
    unsharded parameter — while `.to_local()` gives only this rank's slice. We
    must count the local slice, or sharding will look like it saved nothing.
    """
    if t is None:
        return 0
    local = t.to_local() if hasattr(t, "to_local") else t
    return local.numel() * local.element_size()


def model_state_bytes(model, optimizer=None):
    """Per-rank bytes held by parameters, gradients, and optimizer state."""
    p_bytes = sum(tensor_bytes(p) for p in model.parameters())
    g_bytes = sum(tensor_bytes(p.grad) for p in model.parameters())
    o_bytes = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    o_bytes += tensor_bytes(v)
    return {"params": p_bytes, "grads": g_bytes, "optim": o_bytes,
            "total": p_bytes + g_bytes + o_bytes}


def rss_peak_mb() -> float:
    """Peak resident memory of this process, in MB, from /proc/self/status.

    `VmHWM` ("high water mark") is the largest resident-set size the kernel has
    ever seen for this process. It never goes down, so it survives Python
    freeing objects — which is exactly what we want when comparing two training
    setups.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


# ---------------------------------------------------------------------------
# a tiny synthetic task, shared by several projects
# ---------------------------------------------------------------------------


def make_teacher_data(n, in_dim=64, n_classes=8, seed=0):
    """A learnable classification dataset with no download and no disk.

    A fixed random "teacher" matrix labels random inputs. The task is genuinely
    learnable (loss falls), deterministic given the seed, and identical on every
    rank — so any difference between ranks comes from the distributed code, not
    from the data.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    teacher = torch.randn(in_dim, n_classes, generator=g)
    y = (x @ teacher).argmax(dim=1)
    return x, y


def fmt_bytes(n) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"
