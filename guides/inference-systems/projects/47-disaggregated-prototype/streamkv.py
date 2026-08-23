"""The KV-cache data plane: workers, transports, and layer-streaming.

Project 19 (Phase 3) asked *whether* to disaggregate -- is shipping the
cache cheaper than recomputing it (160x yes), and what does the split cost
on one box. This project builds the part 19 waved at: the TRANSFER itself.

Three ideas live here:

  1. **Transports.** The same bytes can cross an address-space boundary
     through shared memory, a TCP socket, or a multiprocessing pipe. On one
     machine the socket is a stand-in for the NIC path a real cluster uses
     (Mooncake/NIXL over RDMA); measuring all three shows how much of the
     cost is the wire and how much is the copies around it.

  2. **Blocking vs streamed handoff.** The naive prototype prefills the
     whole prompt, THEN ships the whole cache: the transfer sits fully
     inside TTFT. But layer i's KV for the prompt is FINAL the moment layer
     i's prefill finishes -- nothing later ever rewrites it (the model only
     APPENDS to the cache; it never edits). So a sender thread can ship
     layer i while the CPU computes layer i+1, and when prefill ends only
     the LAST layer's bytes are still in flight. That is the overlap trick
     production transfer engines use.

  3. **A throttled sender.** Loopback moves GB/s, so on this machine the
     transfer hides trivially. To see what a real cross-node link changes,
     the sender can pace itself to a fixed budget (e.g. 100 MB/s); the
     pacing loop is honest about being a simulation and is labelled one.

Workers are spawned (never forked -- a forked child inherits PyTorch's
thread pool mid-state and deadlocks, project 19's trap).
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from queue import Queue

HERE = os.path.dirname(os.path.abspath(__file__))
LIB16 = os.path.join(os.path.dirname(HERE), "16-static-vs-continuous")

HDR = struct.Struct("<iq")     # layer index, payload bytes
PORT = 8770


def _setup(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    sys.path.insert(0, LIB16)
    import torch
    torch.set_num_threads(threads)
    return torch


def kv_layer_bytes(pool, layer, length):
    """One layer's K and V for the first `length` positions, as bytes.
    The reshape+numpy round trip COPIES; that is the pack cost a real
    engine avoids by letting the NIC read the cache pages in place."""
    k = pool.k[layer][0, :, :length, :].reshape(-1).numpy().tobytes()
    v = pool.v[layer][0, :, :length, :].reshape(-1).numpy().tobytes()
    return k + v


def unpack_layer(torch, pool, layer, length, data, n_kv_heads, d_head):
    n = len(data) // 2
    k = torch.frombuffer(bytearray(data[:n]), dtype=torch.float32)
    v = torch.frombuffer(bytearray(data[n:]), dtype=torch.float32)
    pool.k[layer][0, :, :length, :] = k.view(n_kv_heads, length, d_head)
    pool.v[layer][0, :, :length, :] = v.view(n_kv_heads, length, d_head)


class PacedSender(threading.Thread):
    """Ships (layer, bytes) messages over a socket, optionally paced to
    `rate` bytes/second. An unbounded queue means enqueueing NEVER stalls
    the compute thread -- a slow link grows the backlog instead."""

    def __init__(self, sock, rate=None):
        super().__init__(daemon=True)
        self.sock = sock
        self.rate = rate
        self.q = Queue()
        self.sent_at = {}

    def run(self):
        credit_t = time.perf_counter()
        while True:
            item = self.q.get()
            if item is None:
                break
            layer, data = item
            if self.rate:
                # pacing: each chunk "costs" len/rate seconds of link time
                now = time.perf_counter()
                credit_t = max(credit_t, now) + len(data) / self.rate
                self.sock.sendall(HDR.pack(layer, len(data)))
                self.sock.sendall(data)
                wait = credit_t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            else:
                self.sock.sendall(HDR.pack(layer, len(data)))
                self.sock.sendall(data)
            self.sent_at[layer] = time.time()

    def send(self, layer, data):
        self.q.put((layer, data))

    def close(self):
        self.q.put(None)
        self.join()


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed")
        got += r
    return bytes(buf)


# ---------------------------------------------------------------------------
# the two model workers
# ---------------------------------------------------------------------------


def prefill_worker(cmd_q, out_q, threads):
    """Loads the model, then serves jobs:
       ("prefill", plen, mode, rate)  -> layer-wise prefill, ship KV
       ("unified", plen)              -> prefill + one decode step, no ship
    """
    torch = _setup(threads)
    import batchlib
    from batchlib import SlotKV

    runner, _ = batchlib.load_runner(n_threads=threads)
    out_q.put(("ready",))

    while True:
        job = cmd_q.get()
        if job is None:
            break
        kind = job[0]

        if kind == "unified":
            _, plen = job
            pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head,
                          plen + 32)
            ids = torch.randint(1000, 12000, (1, plen))
            t0 = time.perf_counter()
            logits, _ = runner.prefill(pool, [0], ids, [plen], count=False)
            t_pf = time.perf_counter() - t0
            tok = int(logits.argmax(-1))
            t0 = time.perf_counter()
            runner.decode_step(pool, [0], [tok], [plen], count=False)
            t_step = time.perf_counter() - t0
            out_q.put(("unified", plen, t_pf, t_step))
            continue

        _, plen, mode, rate = job          # "prefill"
        pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head,
                      plen + 32)
        sock = socket.create_connection(("127.0.0.1", PORT))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sender = PacedSender(sock, rate=rate)
        sender.start()

        g = torch.Generator().manual_seed(plen)
        ids = torch.randint(1000, 12000, (1, plen), generator=g)

        # -- the layer-by-layer prefill (batchlib.prefill, opened up) -------
        slots_t = torch.as_tensor([0])
        lens = torch.as_tensor([plen])
        q_pos = torch.arange(0, plen).view(1, -1)
        kv_len = lens.clamp(max=plen)
        layer_done = []
        with torch.inference_mode():
            t0 = time.perf_counter()
            x = runner.embed[ids]
            for i in range(runner.n_layers):
                x = runner._layer(i, x, pool, slots_t, q_pos, kv_len, False)
                layer_done.append(time.perf_counter() - t0)
                if mode == "streamed":
                    sender.send(i, kv_layer_bytes(pool, i, plen))
            from batchlib import rms_norm
            x = rms_norm(x, runner.norm, runner.eps)
            logits = x[:, -1, :] @ runner.lm_head.T
            t_prefill = time.perf_counter() - t0
        prefill_done_wall = time.time()

        if mode == "blocking":
            for i in range(runner.n_layers):
                sender.send(i, kv_layer_bytes(pool, i, plen))
        sender.close()
        sock.close()
        out_q.put(("prefilled", plen, mode, rate, t_prefill,
                   prefill_done_wall, int(logits.argmax(-1)), layer_done))


def decode_worker(cmd_q, out_q, tok_q, threads):
    """Receives KV layer by layer, then decodes 8 greedy tokens. `tok_q` is
    a dedicated queue for the first token (an mp.Queue cannot be passed
    inside a message on another queue -- only inherited at spawn time)."""
    torch = _setup(threads)
    import batchlib
    from batchlib import SlotKV

    runner, _ = batchlib.load_runner(n_threads=threads)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(4)
    out_q.put(("ready",))

    while True:
        job = cmd_q.get()
        if job is None:
            break
        _, plen = job
        pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head,
                      plen + 32)
        conn, _ = srv.accept()
        recv_at = {}
        for _ in range(runner.n_layers):
            layer, n = HDR.unpack(recv_exact(conn, HDR.size))
            data = recv_exact(conn, n)
            unpack_layer(torch, pool, layer, plen, data,
                         runner.n_kv_heads, runner.d_head)
            recv_at[layer] = time.time()
        conn.close()
        all_recv_wall = time.time()

        tok = tok_q.get()                  # from the prefill worker's logits
        t0 = time.perf_counter()
        cur, toks = plen, [tok]
        for _ in range(8):
            logits, _ = runner.decode_step(pool, [0], [toks[-1]], [cur],
                                           count=False)
            cur += 1
            toks.append(int(logits.argmax(-1)))
        t_decode8 = time.perf_counter() - t0
        out_q.put(("decoded", plen, all_recv_wall, t_decode8, toks))
