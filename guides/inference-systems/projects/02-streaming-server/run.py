"""Project 02 — Streaming server: TTFT vs ITL under real concurrent load.

  A. streaming vs buffered  — same total work, very different first byte
  B. chunk_every sweep      — how often you flush changes what the user feels
  C. concurrency sweep      — what happens with no batching in the engine
  D. the batching headroom  — what project 01's batch curve says we left behind

Run:  python3 run.py          (~6 min)
      python3 run.py --plot   (redraw from outputs/findings.json)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, HERE)

from loadgen import run_load  # noqa: E402

PORT = 8117
URL = f"http://127.0.0.1:{PORT}"
THREADS = 3
MAX_NEW = 24
PROMPT = ("You are a helpful assistant. Explain, in a few sentences, why a "
          "language model server sends tokens to the user one at a time "
          "instead of waiting for the whole answer.")


def start_server(env_extra=None):
    env = dict(os.environ)
    env.pop("TRACE_PATH", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "server.py"),
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
    raise RuntimeError("server did not come up")


def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    a = f["A_streaming_vs_buffered"]
    names = ["streaming\n(SSE, flush each token)", "buffered\n(one JSON at the end)"]
    ttfts = [a["streaming"]["ttft_p50_s"], a["buffered"]["ttft_p50_s"]]
    e2es = [a["streaming"]["e2e_p50_s"], a["buffered"]["e2e_p50_s"]]
    x = range(2)
    ax[0].bar([i - .18 for i in x], ttfts, width=.36, label="TTFT (p50)")
    ax[0].bar([i + .18 for i in x], e2es, width=.36, label="end-to-end (p50)")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(names, fontsize=9)
    ax[0].set_ylabel("seconds")
    ax[0].set_title(f"A. same total time, {a['ttft_speedup']}x sooner\n"
                    "to the first visible word")
    ax[0].legend()
    ax[0].grid(alpha=.3, axis="y")

    b = f["B_chunk_every"]
    ax[1].plot([r["chunk_every"] for r in b], [r["itl_p50_s"] * 1000 for r in b],
               "o-", label="ITL p50 (ms)")
    ax[1].plot([r["chunk_every"] for r in b], [r["itl_p99_s"] * 1000 for r in b],
               "s--", label="ITL p99 (ms)")
    ax[1].set_xlabel("tokens buffered before each flush")
    ax[1].set_ylabel("gap between visible updates (ms)")
    ax[1].set_title("B. flushing less often makes\nthe stream lumpier, not faster")
    ax[1].legend()
    ax[1].grid(alpha=.3)

    c = f["C_concurrency"]
    cc = [r["concurrency"] for r in c]
    ax2 = ax[2]
    ax2.plot(cc, [r["ttft_p50_s"] for r in c], "o-", color="tab:red",
             label="TTFT p50 (s)")
    ax2.plot(cc, [r["itl_p50_s"] for r in c], "^-", color="tab:orange",
             label="ITL p50 (s)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("concurrent requests")
    ax2.set_ylabel("seconds")
    ax2.grid(alpha=.3)
    ax3 = ax2.twinx()
    ax3.plot(cc, [r["throughput_tok_s"] for r in c], "s--", color="tab:green",
             label="throughput (tok/s)")
    ax3.set_ylabel("tokens / second", color="tab:green")
    ax3.set_ylim(0, max(r["throughput_tok_s"] for r in c) * 1.8)
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax2.set_title("C. 8x the users, "
                  f"{c[-1]['throughput_tok_s'] / c[0]['throughput_tok_s']:.2f}x "
                  "the throughput\n(no batching in the engine)")
    fig.tight_layout()
    p = os.path.join(OUT, "streaming.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return

    t_start = time.time()
    f = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "threads_per_server": THREADS,
         "max_new_tokens": MAX_NEW, "prompt_chars": len(PROMPT)}

    print("starting server ...")
    proc = start_server()
    try:
        # Warm the model once so no measurement pays first-touch costs.
        run_load(URL, PROMPT, concurrency=1, n_requests=1, max_new=4)

        print("A. streaming vs buffered (concurrency 1) ...")
        s = run_load(URL, PROMPT, concurrency=1, n_requests=3, max_new=MAX_NEW,
                     stream=True)
        b = run_load(URL, PROMPT, concurrency=1, n_requests=3, max_new=MAX_NEW,
                     stream=False)
        f["A_streaming_vs_buffered"] = {
            "streaming": s, "buffered": b,
            "ttft_speedup": round(b["ttft_p50_s"] / s["ttft_p50_s"], 1),
            "e2e_ratio": round(b["e2e_p50_s"] / s["e2e_p50_s"], 2)}
        print(f"   TTFT {s['ttft_p50_s']}s vs {b['ttft_p50_s']}s "
              f"({f['A_streaming_vs_buffered']['ttft_speedup']}x), "
              f"E2E {s['e2e_p50_s']}s vs {b['e2e_p50_s']}s")

        print("B. chunk_every sweep ...")
        rows = []
        for ce in (1, 2, 4, 8):
            r = run_load(URL, PROMPT, concurrency=1, n_requests=2,
                         max_new=MAX_NEW, stream=True, chunk_every=ce)
            rows.append(r)
            print(f"   chunk_every={ce}: TTFT {r['ttft_p50_s']}s  "
                  f"ITL p50 {r['itl_p50_s']}s  p99 {r['itl_p99_s']}s")
        f["B_chunk_every"] = rows

        print("C. concurrency sweep ...")
        rows = []
        for cc in (1, 2, 4, 8):
            r = run_load(URL, PROMPT, concurrency=cc, n_requests=2 * cc,
                         max_new=MAX_NEW, stream=True)
            rows.append(r)
            print(f"   c={cc}: TTFT p50 {r['ttft_p50_s']}s p99 {r['ttft_p99_s']}s "
                  f"| ITL p50 {r['itl_p50_s']}s | {r['throughput_tok_s']} tok/s")
        f["C_concurrency"] = rows
    finally:
        stop_server(proc)

    # C2. The other way to handle 8 users without batching: let all 8 decode
    # loops run at once and fight over the CPU (SERIALIZE=0).
    print("C2. concurrency 8 without the model lock ...")
    proc = start_server({"SERIALIZE": "0"})
    try:
        run_load(URL, PROMPT, concurrency=1, n_requests=1, max_new=4)
        r = run_load(URL, PROMPT, concurrency=8, n_requests=16, max_new=MAX_NEW,
                     stream=True)
    finally:
        stop_server(proc)
    f["C2_no_lock_c8"] = r
    locked8 = [x for x in f["C_concurrency"] if x["concurrency"] == 8][0]
    f["C2_vs_C"] = {
        "ttft_p50_s": {"serialized": locked8["ttft_p50_s"], "interleaved": r["ttft_p50_s"]},
        "itl_p50_s": {"serialized": locked8["itl_p50_s"], "interleaved": r["itl_p50_s"]},
        "throughput_tok_s": {"serialized": locked8["throughput_tok_s"],
                             "interleaved": r["throughput_tok_s"]}}
    print(f"   TTFT p50 {locked8['ttft_p50_s']}s -> {r['ttft_p50_s']}s | "
          f"ITL p50 {locked8['itl_p50_s']}s -> {r['itl_p50_s']}s | "
          f"{locked8['throughput_tok_s']} -> {r['throughput_tok_s']} tok/s")

    # D. What a batching engine would have got, from project 01's measurement.
    p01 = os.path.join(os.path.dirname(HERE), "01-manual-inference-loop",
                       "outputs", "findings.json")
    if os.path.exists(p01):
        d = json.load(open(p01))["D_decode_batch_scaling"]
        by_b = {r["batch"]: r for r in d}
        c8 = [r for r in f["C_concurrency"] if r["concurrency"] == 8][0]
        c1 = f["C_concurrency"][0]
        f["D_batching_headroom"] = {
            "measured_throughput_gain_1_to_8": round(
                c8["throughput_tok_s"] / c1["throughput_tok_s"], 2),
            "project01_batched_gain_1_to_8": by_b[8]["throughput_vs_b1"],
            "note": ("this server runs one forward pass per request; a batching "
                     "engine folds 8 requests into one forward pass")}
        print(f"D. throughput gain 1->8 users: measured "
              f"{f['D_batching_headroom']['measured_throughput_gain_1_to_8']}x, "
              f"batched ceiling {by_b[8]['throughput_vs_b1']}x")

    f["wall_clock_s"] = round(time.time() - t_start, 1)
    json.dump(f, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        a = f["A_streaming_vs_buffered"]
        fh.write(f"A,ttft_streaming_s,{a['streaming']['ttft_p50_s']}\n")
        fh.write(f"A,ttft_buffered_s,{a['buffered']['ttft_p50_s']}\n")
        fh.write(f"A,ttft_speedup,{a['ttft_speedup']}\n")
        for r in f["B_chunk_every"]:
            fh.write(f"B,itl_p50_s@chunk{r['chunk_every']},{r['itl_p50_s']}\n")
        for r in f["C_concurrency"]:
            fh.write(f"C,ttft_p50_s@c{r['concurrency']},{r['ttft_p50_s']}\n")
            fh.write(f"C,tok_per_s@c{r['concurrency']},{r['throughput_tok_s']}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
