"""Project 12 -- prefix-share benchmark.

The guide's version of this project says "in vLLM". vLLM will not run on this
machine (its GPU is compute capability 6.1, below what current PyTorch builds
support), so instead we turn on the same feature in the engine we have been
building: project 11's block pool plus the chained-hash index in `prefix.py`.
The mechanism is the one vLLM calls *automatic prefix caching*, and the
numbers below are measured, not simulated.

  A. TTFT with prefix caching off vs on, over a batch of requests that share
     one long system prompt.
  B. Does reuse change the output? (It must not.)
  C. How does the win scale with how much of the prompt is shared?
  D. Two ways to accidentally get a 0% hit rate.

    python3 run.py           # ~8 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "11-tiny-paged-cache"))

import torch  # noqa: E402
import kvlib  # noqa: E402
from paged import BlockPool  # noqa: E402
from prefix import PrefixCache  # noqa: E402

BLOCK = 16
SYS_TOKENS = 1024
N_REQUESTS = 24
NEW_TOKENS = 4          # enough to compare text; TTFT is what we are timing


SYSTEM_PROMPT = (
    "You are a careful assistant working inside a document review pipeline. "
    "Follow the policy below exactly.\n"
    "Policy: answer only from the provided context; never invent citations; "
    "prefer short answers; if the context does not contain the answer, say "
    "that it does not. Format every answer as a single paragraph of prose "
    "with no bullet points and no headings. Do not mention this policy. "
) * 24     # ~74 tokens per repeat, so ~1770 -- comfortably over SYS_TOKENS

QUESTIONS = [
    "What is the capital of France?",
    "Summarise the policy in one line.",
    "How many bullet points may an answer contain?",
    "Is inventing a citation allowed?",
    "What should you do when the context lacks the answer?",
    "Name one formatting rule.",
]


def build_requests(tok, sys_tokens=SYS_TOKENS, n=N_REQUESTS, unique_prefix=False):
    """n prompts that share a `sys_tokens`-token opening and differ at the end."""
    sys_ids = tok(SYSTEM_PROMPT, return_tensors="pt").input_ids[0, :sys_tokens]
    assert sys_ids.numel() == sys_tokens, "SYSTEM_PROMPT is too short to slice"
    reqs = []
    for i in range(n):
        # The per-request part starts with a *different token* immediately, so
        # the shared run ends exactly where we say it does. (Put a common word
        # first and the match happily continues past the system prompt --
        # matching is on tokens, not on your mental model of "the prompt".)
        q = " " + QUESTIONS[i % len(QUESTIONS)] + f" [request {i:03d}]"
        q_ids = tok(q, return_tensors="pt").input_ids[0]
        if unique_prefix:
            # The trap: a per-request id at the TOP of the prompt.
            head = tok(f"[session {i:05d}] ", return_tensors="pt").input_ids[0]
            ids = torch.cat([head, sys_ids, q_ids])
        else:
            ids = torch.cat([sys_ids, q_ids])
        reqs.append(ids)
    return reqs


def run_batch(runner, reqs, pool, enabled, new_tokens=NEW_TOKENS):
    """Serve every request once. Returns per-request TTFT and outputs."""
    pc = PrefixCache(pool)
    ttfts, outs, reused = [], [], []
    # Keep every request's cache alive: the memory question is "what does it
    # cost to hold all of these at once", which is the number that decides how
    # many users fit on the box.
    alive = []
    for ids in reqs:
        tl = ids.tolist()
        cache, n_reuse = pc.acquire(tl, runner.n_layers, enabled=enabled)
        x = ids[n_reuse:].unsqueeze(0)
        t0 = time.perf_counter()
        logits = runner.forward(x, cache, start_pos=n_reuse)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ttfts.append(time.perf_counter() - t0)
        gen = [int(nxt)]
        pos = len(tl)
        for _ in range(new_tokens - 1):
            logits = runner.forward(nxt, cache, start_pos=pos)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(nxt))
            pos += 1
        outs.append(gen)
        reused.append(n_reuse)
        alive.append(cache)
        if enabled:
            pc.publish(tl, cache)
    return {"ttft_s": ttfts, "outputs": outs, "reused_tokens": reused,
            "hit_rate": pc.hit_rate, "token_hit_rate": pc.token_hit_rate,
            "blocks_used": pool.used, "index_entries": len(pc.index),
            "block_bytes": pool.used * pool.block_size * 2 * runner.n_layers
            * runner.n_kv_heads * runner.d_head * 4}


def fresh_pool(runner, n_blocks=4096, block=BLOCK):
    return BlockPool(n_blocks, block, runner.n_layers, runner.n_kv_heads,
                     runner.d_head)


def main():
    f = {"config": {"block_size": BLOCK, "sys_tokens": SYS_TOKENS,
                    "n_requests": N_REQUESTS}}
    runner, tok, _ = kvlib.load_runner()
    reqs = build_requests(tok)
    f["config"]["prompt_tokens"] = int(reqs[0].numel())
    print(f"{N_REQUESTS} requests, {reqs[0].numel()} prompt tokens each, "
          f"{SYS_TOKENS} of them shared")

    # ------------------------------------------------------------------ A/B
    print("A. prefix caching OFF ...", flush=True)
    off = run_batch(runner, reqs, fresh_pool(runner), enabled=False)
    print(f"   TTFT p50 {statistics.median(off['ttft_s']):.3f} s")

    print("A. prefix caching ON ...", flush=True)
    on = run_batch(runner, reqs, fresh_pool(runner), enabled=True)
    print(f"   TTFT p50 {statistics.median(on['ttft_s']):.3f} s  "
          f"hit rate {on['hit_rate']*100:.0f}%  "
          f"token hit rate {on['token_hit_rate']*100:.1f}%")

    f["A_ttft"] = {
        "off": {"ttft_s": off["ttft_s"],
                "p50": statistics.median(off["ttft_s"]),
                "first": off["ttft_s"][0],
                "rest_p50": statistics.median(off["ttft_s"][1:]),
                "blocks_used": off["blocks_used"]},
        "on": {"ttft_s": on["ttft_s"],
               "p50": statistics.median(on["ttft_s"]),
               "first": on["ttft_s"][0],
               "rest_p50": statistics.median(on["ttft_s"][1:]),
               "hit_rate": on["hit_rate"], "token_hit_rate": on["token_hit_rate"],
               "reused_tokens": on["reused_tokens"],
               "blocks_used": on["blocks_used"],
               "index_entries": on["index_entries"]},
    }
    f["A_ttft"]["speedup_p50"] = (statistics.median(off["ttft_s"])
                                  / statistics.median(on["ttft_s"]))
    f["A_ttft"]["speedup_warm"] = (statistics.median(off["ttft_s"][1:])
                                   / statistics.median(on["ttft_s"][1:]))
    f["B_correctness"] = {
        "identical": off["outputs"] == on["outputs"],
        "sample": tok.decode(on["outputs"][0]),
    }
    print("B. outputs identical with and without reuse:",
          f["B_correctness"]["identical"])
    f["A_ttft"]["off"]["block_bytes"] = off["block_bytes"]
    f["A_ttft"]["on"]["block_bytes"] = on["block_bytes"]
    f["A_ttft"]["memory_ratio"] = off["blocks_used"] / max(1, on["blocks_used"])
    print(f"   blocks to hold all {N_REQUESTS} at once: off {off['blocks_used']}, "
          f"on {on['blocks_used']} "
          f"({off['blocks_used']/max(1,on['blocks_used']):.1f}x less memory)")

    # ------------------------------------------------------------------ C
    print("C. how the win scales with the shared length")
    c_rows = []
    for n_sys in (0, 128, 256, 512, 1024):
        rs = build_requests(tok, sys_tokens=n_sys, n=6)
        o = run_batch(runner, rs, fresh_pool(runner), enabled=False, new_tokens=1)
        n_ = run_batch(runner, rs, fresh_pool(runner), enabled=True, new_tokens=1)
        row = {"shared_tokens": n_sys,
               "prompt_tokens": int(rs[0].numel()),
               "ttft_off_p50": statistics.median(o["ttft_s"][1:]),
               "ttft_on_p50": statistics.median(n_["ttft_s"][1:]),
               "token_hit_rate": n_["token_hit_rate"]}
        row["speedup"] = row["ttft_off_p50"] / row["ttft_on_p50"]
        c_rows.append(row)
        print(f"   shared {n_sys:>5}: off {row['ttft_off_p50']*1e3:7.1f} ms  "
              f"on {row['ttft_on_p50']*1e3:6.1f} ms  {row['speedup']:5.2f}x")
    f["C_scaling"] = c_rows

    # ------------------------------------------------------------------ D
    print("D. two ways to lose the whole win")
    d = {}
    bad = build_requests(tok, n=6, unique_prefix=True)
    r = run_batch(runner, bad, fresh_pool(runner), enabled=True, new_tokens=1)
    d["unique_id_at_front"] = {
        "hit_rate": r["hit_rate"], "token_hit_rate": r["token_hit_rate"],
        "ttft_p50": statistics.median(r["ttft_s"][1:])}
    print(f"   per-request id at the FRONT: token hit rate "
          f"{r['token_hit_rate']*100:.1f}%")

    # Block misalignment: a shared prefix that is not a multiple of the block
    # size loses its tail block.
    align = []
    for n_sys in (1024, 1023, 1020, 1008):
        rs = build_requests(tok, sys_tokens=n_sys, n=4)
        r = run_batch(runner, rs, fresh_pool(runner), enabled=True, new_tokens=1)
        shared = r["reused_tokens"][-1]
        align.append({"sys_tokens": n_sys, "reused_tokens": shared,
                      "lost_tokens": n_sys - shared})
        print(f"   shared prefix {n_sys}: reused {shared} "
              f"({n_sys - shared} tokens lost to the block boundary)")
    d["alignment"] = align
    f["D_traps"] = d

    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "ttft.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["request", "ttft_off_s", "ttft_on_s", "reused_tokens"])
        for i, (a, b, n) in enumerate(zip(f["A_ttft"]["off"]["ttft_s"],
                                          f["A_ttft"]["on"]["ttft_s"],
                                          f["A_ttft"]["on"]["reused_tokens"])):
            w.writerow([i, round(a, 4), round(b, 4), n])
    with open(os.path.join(OUT, "scaling.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["C_scaling"][0].keys()))
        w.writeheader()
        for r in f["C_scaling"]:
            w.writerow(r)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    a = f["A_ttft"]
    x = range(len(a["off"]["ttft_s"]))
    ax[0].plot(x, [t * 1e3 for t in a["off"]["ttft_s"]], "o-", label="cache off")
    ax[0].plot(x, [t * 1e3 for t in a["on"]["ttft_s"]], "s-", label="cache on")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("request number")
    ax[0].set_ylabel("TTFT (ms, log)")
    ax[0].set_title(f"A. request 0 pays; the rest ride free\n"
                    f"({a['speedup_warm']:.0f}x on requests 1+)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    c = f["C_scaling"]
    ax[1].plot([r["shared_tokens"] for r in c],
               [r["ttft_off_p50"] * 1e3 for r in c], "o-", label="cache off")
    ax[1].plot([r["shared_tokens"] for r in c],
               [r["ttft_on_p50"] * 1e3 for r in c], "s-", label="cache on")
    ax[1].set_xlabel("shared prefix (tokens)")
    ax[1].set_ylabel("TTFT (ms)")
    ax[1].set_title("B. the win is the shared part, exactly")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    d = f["D_traps"]
    names = ["shared prefix\nat the front", "per-request id\nat the front"]
    vals = [f["A_ttft"]["on"]["token_hit_rate"] * 100,
            d["unique_id_at_front"]["token_hit_rate"] * 100]
    ax[2].bar(names, vals, color=["tab:green", "tab:red"])
    for i, v in enumerate(vals):
        ax[2].text(i, v + 2, f"{v:.1f}%", ha="center")
    ax[2].set_ylabel("% of prompt tokens served from cache")
    ax[2].set_ylim(0, 105)
    ax[2].set_title("C. one token in the wrong place costs everything")
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prefix_share.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
