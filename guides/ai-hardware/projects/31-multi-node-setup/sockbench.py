"""Raw TCP measurements - no PyTorch, no NCCL, just two processes and a socket.

This is the honest floor under every "cross-node bandwidth" number: whatever
NCCL or gloo achieves between two nodes, it is built on top of exactly this.
Here both processes live on one machine, so what we measure is the loopback
path (kernel memory copies, no wire, no switch). Section A of the README says
which parts of the result do and do not carry over to a real network.
"""

from __future__ import annotations

import socket
import time

import torch.multiprocessing as mp


def _reps(n: int) -> int:
    """Fewer round trips for big messages. Module level, not a lambda, because
    spawn has to pickle everything it hands to the child process."""
    return 200 if n <= (1 << 14) else (40 if n <= (1 << 20) else 8)


def _server(port, sizes, nodelay, q):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    q.put("ready")
    conn, _ = srv.accept()
    if nodelay:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for n in sizes:
        buf = bytearray(n)
        view = memoryview(buf)
        for _ in range(_reps(n)):
            got = 0
            while got < n:
                k = conn.recv_into(view[got:], n - got)
                if k == 0:
                    return
                got += k
            conn.sendall(view[:8])          # 8-byte acknowledgement
    conn.close()
    srv.close()


def pingpong(port=41777, nodelay=True):
    """Round-trip a message of each size and report one-way time and bandwidth.

    A ping-pong (send, wait for the reply) is the only way to measure latency
    without synchronised clocks on the two ends: you time the round trip on one
    clock and halve it.
    """
    sizes = [8, 1 << 10, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_server, args=(port, sizes, nodelay, q))
    p.start()
    q.get()
    cli = socket.create_connection(("127.0.0.1", port))
    if nodelay:
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    out = []
    for n in sizes:
        payload = bytes(n)
        r = _reps(n)
        cli.sendall(payload)                      # one warm-up round trip
        cli.recv(8)
        t0 = time.perf_counter()
        for _ in range(r - 1):
            cli.sendall(payload)
            cli.recv(8)
        rtt = (time.perf_counter() - t0) / (r - 1)
        out.append(dict(bytes=n, rtt_us=rtt * 1e6, oneway_us=rtt * 5e5,
                        GBs=n / (rtt / 2) / 1e9))
    cli.close()
    p.join(timeout=30)
    return out


def _split_server(port, reps, nodelay, q):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    q.put("ready")
    conn, _ = srv.accept()
    if nodelay:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(reps):
        got = 0
        while got < 8:                     # the reply is sent only once both
            k = conn.recv(8 - got)         # halves have arrived
            if k == 0:
                return
            got += len(k)
        conn.sendall(b"12345678")
    conn.close()
    srv.close()


def two_write(port=41800, nodelay=True, reps=300):
    """The pattern that actually exposes Nagle: *two* small writes before the
    peer answers. Nagle holds the second one back until the first is
    acknowledged, so the round trip picks up the acknowledgement delay. A plain
    ping-pong cannot show this, because it never has a second small write
    outstanding -- a benchmark can only measure an effect it lets happen.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_split_server, args=(port, reps, nodelay, q))
    p.start()
    q.get()
    cli = socket.create_connection(("127.0.0.1", port))
    if nodelay:
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    cli.sendall(b"1234")
    cli.sendall(b"5678")
    cli.recv(8)
    t0 = time.perf_counter()
    for _ in range(reps - 1):
        cli.sendall(b"1234")
        cli.sendall(b"5678")
        cli.recv(8)
    rtt = (time.perf_counter() - t0) / (reps - 1)
    cli.close()
    p.join(timeout=30)
    return dict(rtt_us=rtt * 1e6, nodelay=nodelay)
