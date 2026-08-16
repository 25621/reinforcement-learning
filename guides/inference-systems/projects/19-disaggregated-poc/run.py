"""Project 19 -- a disaggregated prefill/decode proof of concept.

  A. What actually has to move, and how fast can this machine move it?
  B. Two processes vs one, on the same traffic and the same total CPU.
     Throughput, TTFT, and the number disaggregation is really sold on:
     inter-token latency that a long prompt cannot disturb.
  C. Ship the cache or recompute it? The break-even.
  D. What link do you need in a real cluster? The arithmetic, per model size.

    python3 run.py           # ~7 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from multiprocessing import shared_memory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))

from disagg import kv_bytes_per_token, pack_kv, unpack_kv  # noqa: E402
import disagg  # noqa: E402

F = {}

N_REQ = 12
PROMPT_LEN = 256
MAX_NEW = 32
MAX_LEN = 320
N_SLOTS = 8
TOTAL_THREADS = 6


def make_jobs(seed=0):
    rng = random.Random(seed)
    return [(i, [rng.randrange(1000, 12000) for _ in range(PROMPT_LEN)])
            for i in range(N_REQ)]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))]


# ---------------------------------------------------------------------------
# A. what moves
# ---------------------------------------------------------------------------


def section_a():
    import torch

    import batchlib
    from batchlib import SlotKV

    print("\n=== A. what has to move ===")
    runner, _ = batchlib.load_runner(n_threads=TOTAL_THREADS)
    bpt = kv_bytes_per_token(runner.n_layers, runner.n_kv_heads, runner.d_head)
    pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head, MAX_LEN)
    rows = []
    shm = shared_memory.SharedMemory(create=True, size=MAX_LEN * bpt + 64)
    try:
        for plen in [64, 128, 256]:
            ids = torch.randint(1000, 12000, (1, plen))
            t0 = time.perf_counter()
            runner.prefill(pool, [0], ids, [plen], count=False)
            prefill_s = time.perf_counter() - t0
            best = None
            for _ in range(3):
                n, pack_s, copy_s = pack_kv(torch, pool, 0, plen, shm.buf, 0)
                c_s, u_s = unpack_kv(torch, pool, 0, plen, shm.buf, 0,
                                     runner.n_kv_heads, runner.d_head)
                cand = (pack_s, copy_s, c_s, u_s)
                if best is None or sum(cand) < sum(best):
                    best = cand
            pack_s, copy_s, rcopy_s, unpack_s = best
            move_s = copy_s + rcopy_s
            rows.append({
                "prompt_len": plen, "kv_MB": round(n / 1e6, 2),
                "prefill_s": round(prefill_s, 3),
                "pack_ms": round(pack_s * 1e3, 2),
                "shm_write_ms": round(copy_s * 1e3, 2),
                "shm_read_ms": round(rcopy_s * 1e3, 2),
                "unpack_ms": round(unpack_s * 1e3, 2),
                "raw_GB_s": round(2 * n / 1e9 / move_s, 1),
                "transfer_vs_prefill": round(move_s / prefill_s, 5),
                "all_overhead_vs_prefill": round(
                    (pack_s + copy_s + rcopy_s + unpack_s) / prefill_s, 4),
            })
            print(f"  {plen:4d} tok = {rows[-1]['kv_MB']:5.2f} MB   "
                  f"prefill {prefill_s*1e3:7.1f} ms   move {move_s*1e3:5.2f} ms "
                  f"({rows[-1]['raw_GB_s']} GB/s)   "
                  f"transfer/prefill = {rows[-1]['transfer_vs_prefill']:.4%}")
    finally:
        shm.close()
        shm.unlink()
    F["A"] = {"kv_bytes_per_token": bpt, "rows": rows,
              "n_layers": runner.n_layers, "n_kv_heads": runner.n_kv_heads,
              "d_head": runner.d_head}
    return runner, bpt


# ---------------------------------------------------------------------------
# B. unified baseline (one process does both)
# ---------------------------------------------------------------------------


def run_unified(runner, jobs):
    """Prefill and decode share one engine. A prefill takes a whole forward
    pass, so every running request's token clock stops while it happens --
    the interference disaggregation exists to remove."""
    import torch

    from batchlib import SlotKV

    pool = SlotKV(runner.n_layers, N_SLOTS, runner.n_kv_heads, runner.d_head,
                  MAX_LEN)
    queue = list(jobs)
    running = []
    t_start = time.time()
    ttfts, per_req_times = [], {}
    while queue or running:
        if queue and pool.n_free() > 0:
            rid, ids = queue.pop(0)
            slot = pool.acquire()
            x = torch.tensor(ids).view(1, -1)
            logits, _ = runner.prefill(pool, [slot], x, [len(ids)], count=False)
            now = time.time()
            running.append({"rid": rid, "slot": slot, "cur_len": len(ids),
                            "last": int(logits.argmax(-1)), "n": 1})
            ttfts.append(now - t_start)
            per_req_times[rid] = [now]
            continue
        slots = [r["slot"] for r in running]
        logits, _ = runner.decode_step(pool, slots, [r["last"] for r in running],
                                       [r["cur_len"] for r in running],
                                       count=False)
        nxt = logits.argmax(-1).tolist()
        now = time.time()
        done = []
        for j, r in enumerate(running):
            r["cur_len"] += 1
            r["last"] = nxt[j]
            r["n"] += 1
            per_req_times[r["rid"]].append(now)
            if r["n"] >= MAX_NEW:
                done.append(r)
        for r in done:
            pool.release(r["slot"])
            running.remove(r)
    wall = time.time() - t_start
    itls = [b - a for ts in per_req_times.values() for a, b in zip(ts, ts[1:])]
    return wall, ttfts, itls


def run_disaggregated(jobs, bpt, threads_each):
    ctx = mp.get_context("spawn")
    slot_bytes = MAX_LEN * bpt + 4096
    shm = shared_memory.SharedMemory(create=True, size=slot_bytes * N_SLOTS)
    job_q, kv_q, res_q, ready_q = (ctx.Queue() for _ in range(4))
    p = ctx.Process(target=disagg.prefill_worker,
                    args=(job_q, kv_q, ready_q, shm.name, threads_each,
                          slot_bytes, MAX_LEN))
    d = ctx.Process(target=disagg.decode_worker,
                    args=(kv_q, res_q, shm.name, threads_each, slot_bytes,
                          MAX_LEN, N_SLOTS, MAX_NEW))
    p.start()
    d.start()
    ready_q.get()                       # prefill worker up
    while True:
        m = res_q.get()
        if m[0] == "ready":
            break
    kv_q.put(("count", len(jobs)))

    t_start = time.time()
    for i, (rid, ids) in enumerate(jobs):
        job_q.put((rid, ids, i % N_SLOTS, time.time()))
    job_q.put(None)

    arrivals, per_req_times, xfer = [], {}, []
    finished = 0
    while finished < len(jobs):
        m = res_q.get()
        if m[0] == "arrived":
            _, rid, t_sent, t_recv, nbytes, p_s, pack_s, copy_s, c_s, u_s = m
            arrivals.append(t_recv - t_start)
            xfer.append({"rid": rid, "bytes": nbytes,
                         "queue_hop_ms": round((t_recv - t_sent) * 1e3, 2),
                         "prefill_ms": round(p_s * 1e3, 1),
                         "pack_ms": round(pack_s * 1e3, 2),
                         "write_ms": round(copy_s * 1e3, 2),
                         "read_ms": round(c_s * 1e3, 2),
                         "unpack_ms": round(u_s * 1e3, 2)})
        elif m[0] == "finished":
            per_req_times[m[1]] = m[2]
            finished += 1
    wall = time.time() - t_start
    job_q.close()
    p.join(timeout=30)
    d.join(timeout=30)
    for proc in (p, d):
        if proc.is_alive():
            proc.terminate()
    shm.close()
    shm.unlink()
    itls = [b - a for ts in per_req_times.values() for a, b in zip(ts, ts[1:])]
    return wall, arrivals, itls, xfer


def section_b(runner, bpt):
    print("\n=== B. one process vs two, same total CPU ===")
    jobs = make_jobs()
    print(f"  {N_REQ} requests, {PROMPT_LEN}-token prompts, {MAX_NEW} tokens out")

    w1, ttft1, itl1 = run_unified(runner, jobs)
    del runner                                  # free 2 GB before forking two
    print(f"  unified        wall {w1:6.2f}s  ttft p50 {pct(ttft1,50):5.2f}s"
          f"  ITL p50 {pct(itl1,50)*1e3:6.1f} ms  p99 {pct(itl1,99)*1e3:7.1f} ms")

    w2, ttft2, itl2, xfer = run_disaggregated(jobs, bpt, TOTAL_THREADS // 2)
    print(f"  disaggregated  wall {w2:6.2f}s  ttft p50 {pct(ttft2,50):5.2f}s"
          f"  ITL p50 {pct(itl2,50)*1e3:6.1f} ms  p99 {pct(itl2,99)*1e3:7.1f} ms")

    F["B"] = {
        "unified": {"wall_s": round(w1, 2), "ttft_p50": round(pct(ttft1, 50), 3),
                    "ttft_p99": round(pct(ttft1, 99), 3),
                    "itl_p50_ms": round(pct(itl1, 50) * 1e3, 1),
                    "itl_p99_ms": round(pct(itl1, 99) * 1e3, 1),
                    "itl_jitter": round(pct(itl1, 99) / pct(itl1, 50), 2),
                    "tok_s": round(N_REQ * MAX_NEW / w1, 1)},
        "disaggregated": {"wall_s": round(w2, 2), "ttft_p50": round(pct(ttft2, 50), 3),
                          "ttft_p99": round(pct(ttft2, 99), 3),
                          "itl_p50_ms": round(pct(itl2, 50) * 1e3, 1),
                          "itl_p99_ms": round(pct(itl2, 99) * 1e3, 1),
                          "itl_jitter": round(pct(itl2, 99) / pct(itl2, 50), 2),
                          "tok_s": round(N_REQ * MAX_NEW / w2, 1),
                          "threads_each": TOTAL_THREADS // 2},
        "transfers": xfer,
        "median_hop_ms": round(statistics.median(x["queue_hop_ms"] for x in xfer), 2),
        "median_prefill_ms": round(statistics.median(x["prefill_ms"] for x in xfer), 1),
        "throughput_ratio": round((N_REQ * MAX_NEW / w2) / (N_REQ * MAX_NEW / w1), 3),
        "itl_jitter_ratio": round((pct(itl1, 99) / pct(itl1, 50)) /
                                  (pct(itl2, 99) / pct(itl2, 50)), 2),
    }
    print(f"  --> throughput {F['B']['throughput_ratio']:.2f}x, "
          f"ITL jitter (p99/p50) {F['B']['itl_jitter_ratio']:.2f}x better disaggregated")


# ---------------------------------------------------------------------------
# C / D. the arithmetic
# ---------------------------------------------------------------------------


def section_c():
    print("\n=== C. ship the cache, or recompute it? ===")
    a = F["A"]["rows"]
    rows = []
    for r in a:
        # recompute = redo the prefill on the decode box; ship = move the bytes
        rows.append({
            "prompt_len": r["prompt_len"],
            "recompute_ms": round(r["prefill_s"] * 1e3, 1),
            "ship_ms": round(r["shm_write_ms"] + r["shm_read_ms"], 2),
            "ship_plus_serdes_ms": round(r["pack_ms"] + r["shm_write_ms"] +
                                         r["shm_read_ms"] + r["unpack_ms"], 2),
        })
        rows[-1]["speedup"] = round(rows[-1]["recompute_ms"] /
                                    rows[-1]["ship_plus_serdes_ms"], 1)
        print(f"  {r['prompt_len']:4d} tok: recompute {rows[-1]['recompute_ms']:7.1f} ms"
              f"   ship {rows[-1]['ship_plus_serdes_ms']:6.2f} ms"
              f"   -> {rows[-1]['speedup']}x cheaper to ship")
    F["C"] = rows


def section_d():
    print("\n=== D. what link does a real cluster need? ===")
    models = [
        ("Llama-3.1-8B  (32L, 8kv, 128d)", 32, 8, 128, 8000),
        ("Llama-3.1-70B (80L, 8kv, 128d)", 80, 8, 128, 1500),
        ("Llama-2-13B   (40L, 40kv, 128d, MHA)", 40, 40, 128, 5000),
    ]
    links = [("PCIe 4.0 x16", 32e9), ("100 Gb InfiniBand", 12.5e9),
             ("400 Gb InfiniBand", 50e9), ("NVLink 4", 900e9)]
    rows = []
    for name, L, kv, d, prefill_tok_s in models:
        bpt = 2 * L * kv * d * 2          # BF16 in production
        need = bpt * prefill_tok_s        # bytes/s the prefill pool emits
        row = {"model": name, "kv_B_per_token": bpt,
               "prefill_tok_s": prefill_tok_s,
               "kv_bytes_per_s": round(need / 1e9, 2)}
        for lname, bw in links:
            row[lname] = round(bw / need, 2)
        rows.append(row)
        print(f"  {name:38s} {bpt/1024:6.1f} KB/tok -> {need/1e9:6.2f} GB/s   "
              + "  ".join(f"{ln}:{row[ln]:6.2f}x" for ln, _ in links))
    F["D"] = {"rows": rows, "links": [[n, b] for n, b in links],
              "note": "x = link bandwidth / KV bytes the prefill pool produces. "
                      "Below 1.0 the link, not the GPU, sets your prefill rate."}


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    a = f["A"]["rows"]
    x = range(len(a))
    ax[0].bar([i - .2 for i in x], [r["prefill_s"] * 1e3 for r in a], .4,
              color="#c0392b", label="prefill (compute)")
    ax[0].bar([i + .2 for i in x],
              [r["pack_ms"] + r["shm_write_ms"] + r["shm_read_ms"] + r["unpack_ms"]
               for r in a], .4, color="#2471a3", label="move the KV")
    ax[0].set_yscale("log")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([r["prompt_len"] for r in a])
    ax[0].set_xlabel("prompt tokens")
    ax[0].set_ylabel("ms (log)")
    ax[0].set_title("A/C. moving vs recomputing")
    ax[0].legend(fontsize=8)

    b = f["B"]
    names = ["unified", "disaggregated"]
    ax[1].bar([0, 1], [b[n]["tok_s"] for n in names], color=["#c0392b", "#2471a3"])
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("output tokens / s")
    ax[1].set_title(f"B. throughput ({b['throughput_ratio']:.2f}x)")
    for i, n in enumerate(names):
        ax[1].text(i, b[n]["tok_s"], f"{b[n]['tok_s']:.1f}", ha="center", va="bottom")

    w = .35
    for j, n in enumerate(names):
        ax[2].bar([0 + (j - .5) * w, 1 + (j - .5) * w],
                  [b[n]["itl_p50_ms"], b[n]["itl_p99_ms"]], w,
                  color=["#c0392b", "#2471a3"][j], label=n)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["ITL p50", "ITL p99"])
    ax[2].set_ylabel("ms")
    ax[2].set_title("B. the token clock")
    ax[2].legend(fontsize=8)

    d = f["D"]
    links = [ln for ln, _ in d["links"]]
    xs = range(len(d["rows"]))
    for j, ln in enumerate(links):
        ax[3].bar([i + (j - 1.5) * .2 for i in xs],
                  [r[ln] for r in d["rows"]], .2, label=ln)
    ax[3].axhline(1.0, color="k", ls="--", lw=1)
    ax[3].set_yscale("log")
    ax[3].set_xticks(list(xs))
    ax[3].set_xticklabels([r["model"].split()[0] for r in d["rows"]], fontsize=8)
    ax[3].set_ylabel("link headroom (x)")
    ax[3].set_title("D. can the wire keep up?")
    ax[3].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "disaggregated.png"), dpi=110)
    print("wrote outputs/disaggregated.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    runner, bpt = section_a()
    section_b(runner, bpt)
    section_c()
    section_d()
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()


if __name__ == "__main__":
    main()
