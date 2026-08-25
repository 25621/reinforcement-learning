"""Project 47 -- the disaggregated data plane: moving KV well.

  A. Transport ladder: the same KV payload through shared memory, a TCP
     socket, and a multiprocessing pipe (vs recomputing it).
  B. Blocking vs layer-streamed handoff, on the raw loopback and on a
     paced 100 MB/s "cross-node link".
  C. What the user feels: TTFT decomposition, unified vs the two handoffs.
  D. The arithmetic at real scale: what streaming saves on real links.

    python3 run.py           # ~6 minutes on 3+3 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import sys
import time
from multiprocessing import shared_memory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import streamkv  # noqa: E402

F = {}
KV_PER_TOK = 24576              # bytes, Qwen2.5-0.5B fp32 (Phase 2 formula)
THREADS = 3
LINK = 100e6                    # the paced "cross-node" sender, bytes/s


# ---------------------------------------------------------------------------
# A. transport ladder (bytes only, no model)
# ---------------------------------------------------------------------------


def _child_shm(name, n, ready_q, go_q, done_q, reps):
    shm = shared_memory.SharedMemory(name=name)
    for _ in range(reps):
        ready_q.put(1)
        go_q.get()                       # the producer has finished writing
        data = bytes(shm.buf[:n])        # the consumer's read
        done_q.put((time.time(), len(data)))
    shm.close()


def _child_sock(port, n, ready_q, done_q, reps):
    for _ in range(reps):
        s = socket.create_connection(("127.0.0.1", port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ready_q.put(1)
        got = 0
        while got < n:
            b = s.recv(1 << 20)
            if not b:
                break
            got += len(b)
        done_q.put((time.time(), got))
        s.close()


def _child_pipe(conn, ready_q, done_q, reps):
    for _ in range(reps):
        ready_q.put(1)
        data = conn.recv_bytes()
        done_q.put((time.time(), len(data)))


def transport_ladder(payload_tokens, reps=5):
    """Time the same payload across three transports.

    Every arm waits for a `ready` from the child before starting its clock:
    a spawned interpreter takes ~1 s to boot, and a transport that happens to
    absorb that boot inside its timed window looks 50x slower than one that
    does not. The clock covers exactly producer-write -> consumer-has-bytes.
    """
    ctx = mp.get_context("spawn")
    n = payload_tokens * KV_PER_TOK
    blob = os.urandom(n)
    rows = {}

    # -- shared memory: write, signal, read --------------------------------
    shm = shared_memory.SharedMemory(create=True, size=n)
    ready, go, done = ctx.Queue(), ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_child_shm,
                    args=(shm.name, n, ready, go, done, reps))
    p.start()
    best = 1e9
    for _ in range(reps):
        ready.get()
        t0 = time.time()
        shm.buf[:n] = blob               # producer's write
        go.put(1)
        t_end, _ = done.get()
        best = min(best, t_end - t0)
    p.join()
    shm.close()
    shm.unlink()
    rows["shared_memory"] = best

    # -- tcp socket ---------------------------------------------------------
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    ready, done = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_child_sock, args=(port, n, ready, done, reps))
    p.start()
    best = 1e9
    for _ in range(reps):
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ready.get()
        t0 = time.time()
        conn.sendall(blob)
        t_end, _ = done.get()
        best = min(best, t_end - t0)
        conn.close()
    p.join()
    srv.close()
    rows["tcp_socket"] = best

    # -- multiprocessing pipe ------------------------------------------------
    a, b = ctx.Pipe()
    ready, done = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_child_pipe, args=(b, ready, done, reps))
    p.start()
    best = 1e9
    for _ in range(reps):
        ready.get()
        t0 = time.time()
        a.send_bytes(blob)
        t_end, _ = done.get()
        best = min(best, t_end - t0)
    p.join()
    rows["mp_pipe"] = best

    return {"tokens": payload_tokens, "bytes": n,
            **{k: {"s": v, "gb_s": n / v / 1e9} for k, v in rows.items()}}


# ---------------------------------------------------------------------------
# B/C. the two model workers
# ---------------------------------------------------------------------------


def run_handoffs():
    ctx = mp.get_context("spawn")
    pf_cmd, pf_out = ctx.Queue(), ctx.Queue()
    dc_cmd, dc_out = ctx.Queue(), ctx.Queue()
    tok_q = ctx.Queue()
    pf = ctx.Process(target=streamkv.prefill_worker,
                     args=(pf_cmd, pf_out, THREADS))
    dc = ctx.Process(target=streamkv.decode_worker,
                     args=(dc_cmd, dc_out, tok_q, THREADS))
    pf.start()
    dc.start()
    assert pf_out.get(timeout=300)[0] == "ready"
    assert dc_out.get(timeout=300)[0] == "ready"

    trials = []
    for plen in (256, 1024):
        for mode in ("blocking", "streamed"):
            for rate in (None, LINK):
                dc_cmd.put(("decode", plen))
                pf_cmd.put(("prefill", plen, mode, rate))
                (_, _, _, _, t_prefill, pf_done_wall, first_tok,
                 layer_done) = pf_out.get(timeout=600)
                tok_q.put(first_tok)
                _, _, all_recv_wall, t_decode8, toks = dc_out.get(timeout=600)
                tail = all_recv_wall - pf_done_wall
                trials.append({
                    "plen": plen, "mode": mode,
                    "link": "loopback" if rate is None else f"{LINK/1e6:.0f}MB/s",
                    "prefill_s": t_prefill,
                    "handoff_tail_s": tail,
                    "decode_step_s": t_decode8 / 8,
                    "kv_bytes": plen * KV_PER_TOK,
                    "layer_s": (layer_done[-1] - layer_done[0]) / 23,
                    "toks": toks[:4],
                })
                print(f"[B] plen {plen:5d} {mode:9s} {trials[-1]['link']:8s} "
                      f"prefill {t_prefill:6.2f} s  tail {tail*1e3:8.1f} ms",
                      flush=True)

        # unified reference at this plen (prefill + first step, one process)
        pf_cmd.put(("unified", plen))
        _, _, t_pf, t_step = pf_out.get(timeout=600)
        trials.append({"plen": plen, "mode": "unified", "link": "-",
                       "prefill_s": t_pf, "handoff_tail_s": 0.0,
                       "decode_step_s": t_step})
        print(f"[C] plen {plen:5d} unified   prefill {t_pf:6.2f} s  "
              f"step {t_step*1e3:.0f} ms", flush=True)

    pf_cmd.put(None)
    dc_cmd.put(None)
    pf.join(timeout=30)
    dc.join(timeout=30)
    for p in (pf, dc):
        if p.is_alive():
            p.kill()
    return trials


# ---------------------------------------------------------------------------
# D. arithmetic on real links
# ---------------------------------------------------------------------------


def link_arithmetic():
    """Handoff tail added to TTFT: whole cache (blocking) vs last layer
    (streamed), for real models and links. Arithmetic, clearly labelled --
    none of this hardware exists here."""
    models = [
        ("Llama-3.1-8B fp16", 32, 8 * 128 * 2 * 2),      # layers, kv B/tok/layer
        ("Llama-3.1-70B fp16", 80, 8 * 128 * 2 * 2),
        ("Qwen2.5-0.5B fp32 (ours)", 24, 2 * 64 * 4 * 2),
    ]
    links = [("100GbE / RDMA", 12.5e9), ("400G IB", 50e9), ("NVLink4", 900e9)]
    rows = []
    for name, layers, per_tok_layer in models:
        for plen in (2048, 16384, 131072):
            kv = layers * per_tok_layer * plen
            row = {"model": name, "plen": plen, "kv_gb": kv / 1e9}
            for lname, bw in links:
                row[lname] = {"blocking_ms": kv / bw * 1e3,
                              "streamed_ms": kv / layers / bw * 1e3}
            rows.append(row)
    return rows


def main():
    F["ladder"] = [transport_ladder(256), transport_ladder(1024)]
    for row in F["ladder"]:
        print(f"[A] {row['tokens']} tok ({row['bytes']/1e6:.1f} MB): " +
              "  ".join(f"{k} {row[k]['gb_s']:.2f} GB/s"
                        for k in ("shared_memory", "tcp_socket", "mp_pipe")),
              flush=True)
    F["handoffs"] = run_handoffs()
    F["arith"] = link_arithmetic()
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    lad = f["ladder"]
    keys = ["shared_memory", "tcp_socket", "mp_pipe"]
    w = .35
    for j, row in enumerate(lad):
        ax[0].bar([i + (j - .5) * w for i in range(3)],
                  [row[k]["gb_s"] for k in keys], w,
                  color=["#2471a3", "#c0392b"][j],
                  label=f"{row['tokens']} tok ({row['bytes']/1e6:.1f} MB)")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(["shared\nmemory", "TCP\nsocket", "mp\npipe"])
    ax[0].set_ylabel("GB/s (one way, incl. copies)")
    ax[0].set_title("A. transports on one box")
    ax[0].legend(fontsize=8)

    hs = [t for t in f["handoffs"] if t["mode"] != "unified"]
    # A LINEAR axis, on purpose: the streamed tails are NEGATIVE (the cache
    # lands before prefill ends), and a log axis cannot draw a negative bar --
    # it would silently delete the entire result this panel exists to show.
    combos = [(m, l) for m in ("blocking", "streamed")
              for l in ("loopback", "100MB/s")]
    labels = [f"{'block' if m == 'blocking' else 'stream'}\n"
              f"{'loop' if l == 'loopback' else '100MB/s'}" for m, l in combos]
    for j, plen in enumerate((256, 1024)):
        vals = []
        for m, l in combos:
            t = next(x for x in hs if x["plen"] == plen and x["mode"] == m
                     and x["link"] == l)
            vals.append(t["handoff_tail_s"] * 1e3)
        ax[1].bar([i + (j - .5) * w for i in range(4)], vals, w,
                  color=["#2471a3", "#c0392b"][j], label=f"plen {plen}")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("handoff tail after prefill, ms")
    ax[1].set_title("B. what decode waits for\n(below zero = nothing)")
    ax[1].legend(fontsize=8)

    plen = 1024
    modes = ["unified", "blocking", "streamed"]
    rows = []
    for m in modes:
        cand = [x for x in f["handoffs"] if x["plen"] == plen and
                (x["mode"] == m and (m == "unified" or x["link"] == "100MB/s"))]
        rows.append(cand[0])
    bot = [0, 0, 0]
    for part, col, lab in (("prefill_s", "#2471a3", "prefill"),
                           ("handoff_tail_s", "#f39c12", "KV handoff"),
                           ("decode_step_s", "#27ae60", "first decode step")):
        vals = [r[part] for r in rows]
        ax[2].bar(range(3), vals, .5, bottom=bot, color=col, label=lab)
        bot = [b + v for b, v in zip(bot, vals)]
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(["unified", "disagg\nblocking", "disagg\nstreamed"])
    ax[2].set_ylabel("TTFT parts, s (plen 1024, 100 MB/s link)")
    ax[2].set_title("C. where the handoff sits in TTFT")
    ax[2].legend(fontsize=8)

    ar = [r for r in f["arith"] if r["model"].startswith("Llama-3.1-70B")]
    xs = [r["plen"] for r in ar]
    for lname, col in (("100GbE / RDMA", "#c0392b"), ("400G IB", "#f39c12"),
                       ("NVLink4", "#27ae60")):
        ax[3].plot(xs, [r[lname]["blocking_ms"] for r in ar], "o-", color=col,
                   label=f"{lname} blocking")
        ax[3].plot(xs, [r[lname]["streamed_ms"] for r in ar], "o--", color=col,
                   alpha=.5, label=f"{lname} streamed")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("prompt tokens (Llama-3.1-70B)")
    ax[3].set_ylabel("TTFT added, ms (arithmetic)")
    ax[3].set_title("D. streaming at real scale")
    ax[3].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "streamed_kv.png"), dpi=110)
    print("wrote outputs/streamed_kv.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
