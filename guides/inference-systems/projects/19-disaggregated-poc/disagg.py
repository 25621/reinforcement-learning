"""Two processes, one KV cache: a prefill worker and a decode worker.

The point of splitting them is that the two phases want different machines.
Prefill is a big matrix multiply -- it wants FLOPs. Decode is a long series of
tiny ones that each drag the whole weight matrix out of memory -- it wants
bandwidth. Put them in one process and every long prompt freezes every
streaming answer; put them in two and the decode loop never sees a prefill.

What has to move between them is the **KV cache of the prompt**, which is the
only thing prefill produces that decode needs. That is the entire engineering
problem of disaggregated serving, and this file makes it concrete:

    prefill worker ── writes KV into shared memory ──► decode worker
                   ── sends (rid, len, first token) on a queue ──►

Shared memory stands in for the RDMA / NVLink transfer a real cluster uses.
It is the same operation -- bytes copied from one address space to another
without going through the model -- at a different speed, and section D turns
this machine's measured speed into the numbers for real links.

Processes are started with the **spawn** start method, not fork: a forked
child inherits PyTorch's thread pool in an undefined state and deadlocks on
the first parallel op.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB16 = os.path.join(HERE, "..", "16-static-vs-continuous")


def _setup(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, LIB16)
    import torch
    torch.set_num_threads(threads)
    return torch


def kv_bytes_per_token(n_layers, n_kv_heads, d_head, elem=4):
    """2 (K and V) x layers x kv-heads x head width x bytes -- Phase 2's
    formula, and the size of everything this project moves."""
    return 2 * n_layers * n_kv_heads * d_head * elem


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def pack_kv(torch, pool, slot, length, buf, offset):
    """Copy one sequence's KV out of the pool into a flat byte buffer.

    Real systems avoid this copy: their transfer engine registers the cache
    pages directly and the NIC reads them in place. We cannot, so the cost is
    measured separately in section A and reported as what it is -- an artifact
    of doing this in Python, not part of the transfer itself.
    """
    t0 = time.perf_counter()
    parts = []
    for layer in range(pool.n_layers):
        parts.append(pool.k[layer][slot, :, :length, :].reshape(-1))
        parts.append(pool.v[layer][slot, :, :length, :].reshape(-1))
    flat = torch.cat(parts)
    pack_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = flat.numel() * 4
    view = torch.frombuffer(buf, dtype=torch.float32, count=flat.numel(),
                            offset=offset)
    view.copy_(flat)
    copy_s = time.perf_counter() - t0
    return n, pack_s, copy_s


def unpack_kv(torch, pool, slot, length, buf, offset, n_kv_heads, d_head):
    """The mirror image, in the decode worker."""
    per = n_kv_heads * length * d_head
    t0 = time.perf_counter()
    total = pool.n_layers * 2 * per
    flat = torch.frombuffer(buf, dtype=torch.float32, count=total,
                            offset=offset).clone()
    copy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    i = 0
    for layer in range(pool.n_layers):
        pool.k[layer][slot, :, :length, :] = flat[i:i + per].view(
            n_kv_heads, length, d_head)
        i += per
        pool.v[layer][slot, :, :length, :] = flat[i:i + per].view(
            n_kv_heads, length, d_head)
        i += per
    unpack_s = time.perf_counter() - t0
    return copy_s, unpack_s


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


def prefill_worker(jobs, out_q, ready_q, shm_name, threads, slot_bytes, max_len):
    from multiprocessing import shared_memory

    torch = _setup(threads)
    import batchlib
    from batchlib import SlotKV

    runner, _ = batchlib.load_runner(n_threads=threads)
    shm = shared_memory.SharedMemory(name=shm_name)
    pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head, max_len)
    ready_q.put(("ready", time.time()))

    while True:
        job = jobs.get()
        if job is None:
            break
        rid, ids, slot_idx, arrive = job
        t0 = time.perf_counter()
        x = torch.tensor(ids).view(1, -1)
        logits, _ = runner.prefill(pool, [0], x, [len(ids)], count=False)
        first = int(logits.argmax(-1))
        prefill_s = time.perf_counter() - t0
        nbytes, pack_s, copy_s = pack_kv(
            torch, pool, 0, len(ids), shm.buf, slot_idx * slot_bytes)
        out_q.put(("kv", rid, len(ids), first, slot_idx, nbytes,
                   prefill_s, pack_s, copy_s, time.time()))
    shm.close()
    out_q.put(("done", None))


def decode_worker(kv_q, res_q, shm_name, threads, slot_bytes, max_len,
                  n_slots, max_new):
    from multiprocessing import shared_memory

    torch = _setup(threads)
    import batchlib
    from batchlib import SlotKV

    runner, _ = batchlib.load_runner(n_threads=threads)
    shm = shared_memory.SharedMemory(name=shm_name)
    pool = SlotKV(runner.n_layers, n_slots, runner.n_kv_heads, runner.d_head,
                  max_len)
    res_q.put(("ready", time.time()))

    running = []          # dicts: rid, slot, cur_len, last, n_out, times
    finished = 0
    expected = None
    while True:
        # ---- admit everything that has arrived, without blocking ----------
        while True:
            try:
                msg = kv_q.get_nowait()
            except Exception:
                break
            if msg[0] == "count":
                expected = msg[1]
                continue
            if msg[0] in ("ready", "done"):
                continue
            _, rid, plen, first, slot_idx, nbytes, p_s, pack_s, copy_s, t_sent = msg
            slot = pool.acquire()
            c_s, u_s = unpack_kv(torch, pool, slot, plen, shm.buf,
                                 slot_idx * slot_bytes, runner.n_kv_heads,
                                 runner.d_head)
            now = time.time()
            running.append({"rid": rid, "slot": slot, "cur_len": plen,
                            "last": first, "n_out": 1, "times": [now]})
            res_q.put(("arrived", rid, t_sent, now, nbytes, p_s, pack_s,
                       copy_s, c_s, u_s))

        if not running:
            if expected is not None and finished >= expected:
                break
            time.sleep(0.002)
            continue

        # ---- one decode step for everyone still running -------------------
        slots = [r["slot"] for r in running]
        toks = [r["last"] for r in running]
        lens = [r["cur_len"] for r in running]
        logits, dt = runner.decode_step(pool, slots, toks, lens, count=False)
        nxt = logits.argmax(-1).tolist()
        now = time.time()
        done = []
        for j, r in enumerate(running):
            r["cur_len"] += 1
            r["last"] = nxt[j]
            r["n_out"] += 1
            r["times"].append(now)
            if r["n_out"] >= max_new:
                done.append(r)
        for r in done:
            pool.release(r["slot"])
            running.remove(r)
            finished += 1
            res_q.put(("finished", r["rid"], r["times"]))
    shm.close()
    res_q.put(("done", None))
