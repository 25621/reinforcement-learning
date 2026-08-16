"""Project 07 — Request-lifecycle tracer.

Takes project 02's server, switches on its per-stage trace, drives it with a
mixed workload, and answers one question with numbers instead of intuition:
for the slowest request in a load test, where did the time actually go?

  A. no load        — the stage breakdown when nobody else is waiting
  B. under load     — the same breakdown at concurrency 8, mixed prompt sizes
  C. the slow one   — a waterfall for the worst request, versus the median
  D. unaccounted    — server-side total vs what the client measured

Run:  python3 run.py          (~4 min)
      python3 run.py --plot   (redraw from outputs/findings.json)
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
P02 = os.path.join(os.path.dirname(HERE), "02-streaming-server")
sys.path.insert(0, HERE)
sys.path.insert(0, P02)

from loadgen import one_stream  # noqa: E402

PORT = 8123
URL = f"http://127.0.0.1:{PORT}"
THREADS = 3
MAX_NEW = 24
TRACE = os.path.join(OUT, "trace.jsonl")

SHORT = "In one sentence: what is a KV cache?"
LONG = ("Here is a transcript to summarise.\n" +
        "A user asked the assistant about inference servers and the assistant "
        "explained prefill and decode in detail. " * 40 +
        "\nSummarise it in one sentence.")


def start_server():
    env = dict(os.environ)
    env["TRACE_PATH"] = TRACE
    proc = subprocess.Popen(
        [sys.executable, os.path.join(P02, "server.py"),
         "--port", str(PORT), "--threads", str(THREADS)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            if httpx.get(URL + "/health", timeout=2.0).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError("server did not start")


def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


async def _drive(prompts, concurrency, max_new):
    sem = asyncio.Semaphore(concurrency)
    results = []
    async with httpx.AsyncClient(timeout=900.0) as client:
        async def one(p):
            async with sem:
                r = await one_stream(client, URL, p, max_new, 1)
                r["prompt_kind"] = "long" if len(p) > 400 else "short"
                results.append(r)
        t0 = time.perf_counter()
        await asyncio.gather(*[one(p) for p in prompts])
        wall = time.perf_counter() - t0
    return results, wall


def drive(prompts, concurrency, max_new=MAX_NEW):
    return asyncio.run(_drive(prompts, concurrency, max_new))


def read_traces(since_index=0):
    rows = []
    with open(TRACE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[since_index:]


def stages(rec):
    """Split one request's wall clock into named, non-overlapping stages."""
    tokenize = rec["t_tokenized"] - rec["t_admit"]
    queue = rec["t_scheduled"] - rec["t_tokenized"]
    prefill = rec["prefill_s"]
    decode = sum(rec["decode_step_s"])
    total = rec["t_done"] - rec["t_admit"]
    other = total - (tokenize + queue + prefill + decode)
    return {"request_id": rec.get("request_id"),
            "prompt_tokens": rec.get("prompt_tokens"),
            "output_tokens": rec.get("n_output_tokens"),
            "tokenize_s": tokenize, "queue_s": queue, "prefill_s": prefill,
            "decode_s": decode, "other_s": other, "total_s": total,
            "decode_step_s": rec["decode_step_s"]}


def summarise(rows, label):
    tot = sum(r["total_s"] for r in rows)
    out = {"label": label, "n_requests": len(rows),
           "total_s": round(tot, 3),
           "share_pct": {k: round(100 * sum(r[k + "_s"] for r in rows) / tot, 1)
                         for k in ("tokenize", "queue", "prefill", "decode",
                                   "other")},
           "median_total_s": round(statistics.median(
               [r["total_s"] for r in rows]), 3),
           "max_total_s": round(max(r["total_s"] for r in rows), 3),
           "median_queue_s": round(statistics.median(
               [r["queue_s"] for r in rows]), 3),
           "max_queue_s": round(max(r["queue_s"] for r in rows), 3)}
    print(f"  {label}: {len(rows)} requests | shares " +
          " ".join(f"{k}={v}%" for k, v in out["share_pct"].items()))
    print(f"     median total {out['median_total_s']}s, worst "
          f"{out['max_total_s']}s, worst queue {out['max_queue_s']}s")
    return out


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 4.6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    keys = ["tokenize", "queue", "prefill", "decode", "other"]
    cols = ["tab:grey", "tab:red", "tab:orange", "tab:blue", "tab:purple"]

    # 1. share of total time, no load vs under load
    lo, hi = f["A_no_load"], f["B_under_load"]
    bottom_lo = bottom_hi = 0
    for k, c in zip(keys, cols):
        ax0.bar(0, lo["share_pct"][k], bottom=bottom_lo, color=c, width=.6,
                label=k)
        ax0.bar(1, hi["share_pct"][k], bottom=bottom_hi, color=c, width=.6)
        bottom_lo += lo["share_pct"][k]
        bottom_hi += hi["share_pct"][k]
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["concurrency 1", "concurrency 8"])
    ax0.set_ylabel("% of total request time")
    ax0.set_title("A/B. the same server, two loads.\nQueue is not a stage — "
                  "it is the answer.")
    ax0.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, .5))
    ax0.grid(alpha=.3, axis="y")

    # 2. waterfall: every request under load, sorted by total
    rows = sorted(f["B_requests"], key=lambda r: r["total_s"])
    y = range(len(rows))
    left = [0.0] * len(rows)
    for k, c in zip(keys, cols):
        vals = [r[k + "_s"] for r in rows]
        ax1.barh(list(y), vals, left=left, color=c, height=.75)
        left = [a + b for a, b in zip(left, vals)]
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([f"{r['prompt_tokens']}tok" for r in rows], fontsize=6)
    ax1.set_xlabel("seconds from admission")
    ax1.set_title("C. one bar per request at concurrency 8\n"
                  "(sorted by total; label = prompt length)")
    ax1.grid(alpha=.3, axis="x")

    # 3. the slowest request, stage by stage, vs the median request
    s, m = f["C_slowest"], f["C_median"]
    idx = range(len(keys))
    ax2.bar([i - .2 for i in idx], [m[k + "_s"] for k in keys], width=.4,
            color="tab:green", label=f"median request ({m['total_s']:.1f} s)")
    ax2.bar([i + .2 for i in idx], [s[k + "_s"] for k in keys], width=.4,
            color="tab:red", label=f"slowest request ({s['total_s']:.1f} s)")
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels(keys)
    ax2.set_ylabel("seconds")
    ax2.set_title("C. where the extra time went\n"
                  f"({f['C_attribution']['queue_share_of_gap_pct']}% of the gap "
                  "is queue)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT, "lifecycle.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return

    t_start = time.time()
    if os.path.exists(TRACE):
        os.remove(TRACE)
    f = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "threads_per_server": THREADS,
         "max_new_tokens": MAX_NEW}

    print("starting traced server ...")
    proc = start_server()
    try:
        drive([SHORT], 1, max_new=4)                     # warm-up
        n_warm = len(read_traces())

        print("A. concurrency 1 ...")
        client_a, wall_a = drive([SHORT] * 4 + [LONG] * 2, 1)
        rows_a = [stages(r) for r in read_traces(n_warm)]
        f["A_no_load"] = summarise(rows_a, "A (concurrency 1)")
        f["A_requests"] = rows_a
        n_after_a = n_warm + len(rows_a)

        print("B. concurrency 8, mixed prompt sizes ...")
        prompts = ([SHORT] * 10 + [LONG] * 6)
        client_b, wall_b = drive(prompts, 8)
        rows_b = [stages(r) for r in read_traces(n_after_a)]
        f["B_under_load"] = summarise(rows_b, "B (concurrency 8)")
        f["B_under_load"]["wall_s"] = round(wall_b, 3)
        f["B_requests"] = rows_b
    finally:
        stop_server(proc)

    # C. attribution for the slowest request.
    by_total = sorted(rows_b, key=lambda r: r["total_s"])
    slow = by_total[-1]
    med = by_total[len(by_total) // 2]
    gap = slow["total_s"] - med["total_s"]
    f["C_slowest"] = slow
    f["C_median"] = med
    f["C_attribution"] = {
        "gap_s": round(gap, 3),
        "queue_delta_s": round(slow["queue_s"] - med["queue_s"], 3),
        "prefill_delta_s": round(slow["prefill_s"] - med["prefill_s"], 3),
        "decode_delta_s": round(slow["decode_s"] - med["decode_s"], 3),
        "other_delta_s": round(slow["other_s"] - med["other_s"], 3),
        "queue_share_of_gap_pct": round(
            100 * (slow["queue_s"] - med["queue_s"]) / gap, 1) if gap else 0.0,
        "slow_prompt_tokens": slow["prompt_tokens"],
        "median_prompt_tokens": med["prompt_tokens"]}
    print(f"  C: slowest {slow['total_s']:.2f}s vs median {med['total_s']:.2f}s "
          f"— gap {gap:.2f}s, of which queue "
          f"{f['C_attribution']['queue_delta_s']:.2f}s "
          f"({f['C_attribution']['queue_share_of_gap_pct']}%)")

    # D. what the client saw vs what the server could account for.
    client_e2e = sorted(r["e2e_s"] for r in client_b)
    server_tot = sorted(r["total_s"] for r in rows_b)
    med_client = statistics.median(client_e2e)
    med_server = statistics.median(server_tot)
    client_ttft = sorted(r["ttft_s"] for r in client_b if r["ttft_s"])
    server_ttft = sorted(r["queue_s"] + r["prefill_s"] + r["tokenize_s"]
                         for r in rows_b)
    med_c_ttft = statistics.median(client_ttft)
    med_s_ttft = statistics.median(server_ttft)
    f["D_client_vs_server"] = {
        "client_median_e2e_s": round(med_client, 3),
        "server_median_total_s": round(med_server, 3),
        "unexplained_s": round(med_client - med_server, 3),
        "unexplained_pct": round(100 * (med_client - med_server) / med_client, 1),
        "client_median_ttft_s": round(med_c_ttft, 3),
        "server_median_ttft_s": round(med_s_ttft, 3),
        "ttft_unexplained_ms": round(1000 * (med_c_ttft - med_s_ttft), 1),
        "client_max_e2e_s": round(client_e2e[-1], 3),
        "server_max_total_s": round(server_tot[-1], 3),
        "median_other_stage_pct": f["B_under_load"]["share_pct"]["other"]}
    print(f"  D: client median E2E {med_client:.2f}s vs server-accounted "
          f"{med_server:.2f}s — {f['D_client_vs_server']['unexplained_pct']}% "
          f"lives outside the engine's own trace")
    print(f"     client TTFT {med_c_ttft:.3f}s vs server queue+prefill "
          f"{med_s_ttft:.3f}s — "
          f"{f['D_client_vs_server']['ttft_unexplained_ms']} ms of HTTP")

    # E. Does the decode cadence itself degrade under load?
    def step_stats(rows):
        st = [x for r in rows for x in r["decode_step_s"]]
        st_sorted = sorted(st)
        return {"n_steps": len(st),
                "p50_ms": round(1000 * statistics.median(st), 1),
                "p99_ms": round(1000 * st_sorted[int(.99 * len(st)) - 1], 1),
                "max_ms": round(1000 * st_sorted[-1], 1)}

    f["E_decode_cadence"] = {"no_load": step_stats(rows_a),
                             "under_load": step_stats(rows_b)}
    print(f"  E: decode step p50/p99 — no load "
          f"{f['E_decode_cadence']['no_load']['p50_ms']}/"
          f"{f['E_decode_cadence']['no_load']['p99_ms']} ms, under load "
          f"{f['E_decode_cadence']['under_load']['p50_ms']}/"
          f"{f['E_decode_cadence']['under_load']['p99_ms']} ms")

    f["wall_clock_s"] = round(time.time() - t_start, 1)
    json.dump(f, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        for sec in ("A_no_load", "B_under_load"):
            for k, v in f[sec]["share_pct"].items():
                fh.write(f"{sec},share_pct_{k},{v}\n")
            fh.write(f"{sec},median_total_s,{f[sec]['median_total_s']}\n")
            fh.write(f"{sec},max_total_s,{f[sec]['max_total_s']}\n")
        for k, v in f["C_attribution"].items():
            fh.write(f"C,{k},{v}\n")
        for k, v in f["D_client_vs_server"].items():
            fh.write(f"D,{k},{v}\n")
        for cond, st in f["E_decode_cadence"].items():
            for k, v in st.items():
                fh.write(f"E,{cond}_{k},{v}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
