"""Project 11 -- a tiny paged KV cache.

Four questions:

  A. Does paging change the model's output? (It must not.)
  B. What does contiguous allocation actually cost on a realistic mix of
     request lengths -- and how much of the loss is fragmentation rather than
     genuine demand?
  C. Block size is a dial. Which way does it trade?
  D. What does paging cost per decode step when you *don't* have a custom
     kernel to read the blocks in place?

    python3 run.py           # ~5 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

import torch  # noqa: E402
import kvlib  # noqa: E402
from paged import BlockPool, PagedCache  # noqa: E402

# A realistic serving shape to reason about: Llama-3.1-8B, bf16 cache.
PER_TOKEN_BYTES = 2 * 32 * 8 * 128 * 2       # 128 KB/token
MAX_MODEL_LEN = 8192
ARENA_BYTES = 12e9                            # cache arena, sized so it is busy


# ---------------------------------------------------------------------------
# B. the allocation simulator
# ---------------------------------------------------------------------------


def workload(n=600, seed=0):
    """Request lengths with the long tail real traffic has.

    Most chat turns are short; a few carry a pasted document. A lognormal
    reproduces that shape: the *logarithm* of the length is normal, so the
    right tail is heavy while nothing goes negative.
    """
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        total = int(min(MAX_MODEL_LEN, max(64, rng.lognormvariate(6.4, 1.05))))
        # Arrival times: Poisson process -> exponential gaps.
        reqs.append({"id": i, "len": total, "gap": rng.expovariate(1 / 0.15)})
    t = 0.0
    for r in reqs:
        t += r["gap"]
        r["arrive"] = t
        # Hold time scales with generated tokens; ~40 tokens/s per request.
        r["depart"] = t + r["len"] / 40.0
    return reqs


class ContiguousArena:
    """First-fit allocator over one flat arena, with free-list coalescing.

    This is what a KV cache looked like before PagedAttention: every sequence
    needs one unbroken run of addresses.
    """

    def __init__(self, size):
        self.size = size
        self.free = [(0, size)]           # (start, length), sorted, disjoint

    def alloc(self, n):
        for i, (s, ln) in enumerate(self.free):
            if ln >= n:
                if ln == n:
                    self.free.pop(i)
                else:
                    self.free[i] = (s + n, ln - n)
                return s
        return None

    def release(self, start, n):
        self.free.append((start, n))
        self.free.sort()
        merged = []
        for s, ln in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + ln)
            else:
                merged.append((s, ln))
        self.free = merged

    def free_bytes(self):
        return sum(ln for _, ln in self.free)

    def largest_hole(self):
        return max((ln for _, ln in self.free), default=0)


def simulate(policy, reqs, block_size=16):
    """Run the trace under one allocation policy. No queueing: if a request
    cannot be admitted on arrival it is rejected, so the number admitted is a
    clean measure of capacity."""
    events = []
    for r in reqs:
        events.append((r["arrive"], 0, r["id"]))
        events.append((r["depart"], 1, r["id"]))
    events.sort()
    by_id = {r["id"]: r for r in reqs}

    admitted, rejected, frag_rejects = 0, 0, 0
    admitted_tokens, rejected_tokens = 0, 0
    # Counting *requests* alone flatters a broken allocator: dropping one
    # 8000-token request makes room for a dozen short ones. So also split
    # rejections by request size.
    long_cut = sorted(r["len"] for r in reqs)[int(0.75 * len(reqs))]
    rej_long, n_long = 0, sum(1 for r in reqs if r["len"] >= long_cut)
    live = {}
    peak_live = 0
    util = []          # fraction of the arena *reserved*
    useful = []        # fraction of the arena actually holding tokens
    live_tokens = 0
    arena_tokens = ARENA_BYTES / PER_TOKEN_BYTES

    if policy == "paged":
        n_blocks = int(ARENA_BYTES // (PER_TOKEN_BYTES * block_size))
        free_blocks = n_blocks
    else:
        arena = ContiguousArena(int(ARENA_BYTES))

    for t, kind, rid in events:
        r = by_id[rid]
        if kind == 0:
            if policy == "paged":
                need = math.ceil(r["len"] / block_size)
                if need <= free_blocks:
                    free_blocks -= need
                    live[rid] = need
                    admitted += 1
                    admitted_tokens += r["len"]
                    live_tokens += r["len"]
                else:
                    rejected += 1
                    rejected_tokens += r["len"]
                    rej_long += r["len"] >= long_cut
            else:
                # reserve_max cannot know the final length, so it books the
                # worst case; reserve_actual is an oracle upper bound.
                n = (MAX_MODEL_LEN if policy == "reserve_max" else r["len"])
                n *= PER_TOKEN_BYTES
                enough_total = arena.free_bytes() >= n
                start = arena.alloc(n)
                if start is None:
                    rejected += 1
                    rejected_tokens += r["len"]
                    rej_long += r["len"] >= long_cut
                    if enough_total:
                        frag_rejects += 1
                else:
                    live[rid] = (start, n)
                    admitted += 1
                    admitted_tokens += r["len"]
                    live_tokens += r["len"]
            peak_live = max(peak_live, len(live))
            if policy == "paged":
                util.append(1 - free_blocks / n_blocks)
            else:
                util.append(1 - arena.free_bytes() / ARENA_BYTES)
            useful.append(live_tokens / arena_tokens)
        else:
            if rid in live:
                live_tokens -= r["len"]
                if policy == "paged":
                    free_blocks += live.pop(rid)
                else:
                    arena.release(*live.pop(rid))
    return {"policy": policy, "block_size": block_size, "admitted": admitted,
            "rejected": rejected, "frag_rejects": frag_rejects,
            "admitted_tokens": admitted_tokens,
            "rejected_tokens": rejected_tokens,
            "long_reject_pct": 100 * rej_long / max(1, n_long),
            "long_cut": long_cut,
            "peak_concurrent": peak_live,
            "mean_util": statistics.mean(util) if util else 0.0,
            "mean_useful_util": statistics.mean(useful) if useful else 0.0}


# ---------------------------------------------------------------------------


def main():
    f = {}
    reqs = workload()
    f["workload"] = {
        "n": len(reqs),
        "median_len": statistics.median(r["len"] for r in reqs),
        "mean_len": statistics.mean(r["len"] for r in reqs),
        "p99_len": sorted(r["len"] for r in reqs)[int(0.99 * len(reqs))],
        "max_model_len": MAX_MODEL_LEN,
        "per_token_bytes": PER_TOKEN_BYTES,
        "arena_gb": ARENA_BYTES / 1e9,
    }
    print(f"workload: {len(reqs)} requests, median {f['workload']['median_len']:.0f} "
          f"tokens, p99 {f['workload']['p99_len']}")

    # ------------------------------------------------------------------ B
    print("B. allocation policies on the same trace")
    sims = [simulate("reserve_max", reqs),
            simulate("reserve_actual", reqs),
            simulate("paged", reqs, 16)]
    for s in sims:
        print(f"   {s['policy']:>14}: admitted {s['admitted']:>4}/{len(reqs)} "
              f"({s['admitted_tokens']/1000:6.1f}k tokens)  "
              f"frag-rejects {s['frag_rejects']:>3}  "
              f"long-req rejects {s['long_reject_pct']:5.1f}%  "
              f"reserved {s['mean_util']*100:5.1f}%  "
              f"actually holding tokens {s['mean_useful_util']*100:5.1f}%")
    f["B_policies"] = sims

    # ------------------------------------------------------------------ C
    print("C. block-size sweep (internal waste on the same lengths)")
    sweep = []
    lens = [r["len"] for r in reqs]
    for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        reserved = sum(math.ceil(L / bs) * bs for L in lens)
        used = sum(lens)
        s = simulate("paged", reqs, bs)
        sweep.append({"block_size": bs,
                      "waste_pct": 100 * (reserved - used) / used,
                      "blocks_per_seq": statistics.mean(math.ceil(L / bs) for L in lens),
                      "admitted": s["admitted"]})
        print(f"   block {bs:>4}: waste {sweep[-1]['waste_pct']:5.2f}%  "
              f"table entries/seq {sweep[-1]['blocks_per_seq']:7.1f}  "
              f"admitted {s['admitted']}")
    f["C_block_sweep"] = sweep

    # ------------------------------------------------------------------ A, D
    print("A/D. correctness and cost on the real model")
    runner, tok, model = kvlib.load_runner()
    body = ("Paging the key-value cache removes fragmentation because a "
            "sequence no longer needs one unbroken run of memory. ") * 60
    prompt = tok(body, return_tensors="pt").input_ids[:, :256]

    cont = kvlib.ContiguousCache(runner.n_layers)
    ids_c, pf_c, st_c = runner.generate(prompt, cont, max_new_tokens=16,
                                        stop_on_eos=False)

    pool = BlockPool(512, 16, runner.n_layers, runner.n_kv_heads, runner.d_head)
    pcache = PagedCache(pool, runner.n_layers)
    ids_p, pf_p, st_p = runner.generate(prompt, pcache, max_new_tokens=16,
                                        stop_on_eos=False)

    f["A_correctness"] = {
        "tokens_identical": ids_c == ids_p,
        "text": tok.decode(ids_p),
        "blocks_used": pcache.blocks_used(),
        "tokens_held": pcache.length,
        "internal_waste_tokens": pcache.internal_waste_tokens(),
    }
    print("   identical tokens:", f["A_correctness"]["tokens_identical"],
          "| blocks", pcache.blocks_used(), "| waste",
          pcache.internal_waste_tokens(), "token slots")

    # D: per-step cost, paged vs contiguous, at two context lengths. The
    # gather cost is proportional to the cache size, so it only becomes
    # visible once the cache is big.
    long_prompt = tok(body * 6, return_tensors="pt").input_ids[:, :2048]
    d_rows = []
    for ctx, pr in (("256", prompt), ("2048", long_prompt)):
        cc = kvlib.ContiguousCache(runner.n_layers)
        _, _, stc = runner.generate(pr, cc, max_new_tokens=8, stop_on_eos=False)
        d_rows.append({"ctx": ctx, "cache": "contiguous", "block_size": None,
                       "median_step_s": statistics.median(stc)})
        for bs in (16, 64, 256):
            pool = BlockPool(max(8, 8192 // bs), bs, runner.n_layers,
                             runner.n_kv_heads, runner.d_head)
            pc = PagedCache(pool, runner.n_layers)
            _, _, st = runner.generate(pr, pc, max_new_tokens=8, stop_on_eos=False)
            d_rows.append({"ctx": ctx, "cache": "paged", "block_size": bs,
                           "median_step_s": statistics.median(st)})
            print(f"   ctx {ctx:>4} paged bs={bs:>3}: "
                  f"{statistics.median(st)*1e3:6.1f} ms  vs contiguous "
                  f"{statistics.median(stc)*1e3:6.1f} ms  "
                  f"({statistics.median(st)/statistics.median(stc):.2f}x)")
    f["D_cost"] = d_rows

    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "block_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["C_block_sweep"][0].keys()))
        w.writeheader()
        for r in f["C_block_sweep"]:
            w.writerow(r)
    with open(os.path.join(OUT, "policies.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["B_policies"][0].keys()))
        w.writeheader()
        for r in f["B_policies"]:
            w.writerow(r)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    p = f["B_policies"]
    names = ["reserve\nmax length", "reserve\nactual (oracle)", "paged\n(bs 16)"]
    tot = sum(r["admitted_tokens"] + r["rejected_tokens"] for r in p) / len(p)
    ax[0].bar(names, [r["admitted_tokens"] / 1000 for r in p], color="tab:blue",
              label="tokens served")
    ax[0].bar(names, [r["rejected_tokens"] / 1000 for r in p],
              bottom=[r["admitted_tokens"] / 1000 for r in p], color="tab:red",
              label="tokens rejected")
    for i, r in enumerate(p):
        ax[0].text(i, r["admitted_tokens"] / 2000,
                   f"{100*r['admitted_tokens']/tot:.0f}%", ha="center",
                   color="w", fontsize=10)
    ax[0].set_ylabel("thousands of tokens")
    ax[0].set_title(f"A. same {f['workload']['arena_gb']:.0f} GB, same traffic")
    ax[0].legend(fontsize=8)

    s = f["C_block_sweep"]
    ax[1].plot([r["block_size"] for r in s], [r["waste_pct"] for r in s], "o-",
               color="tab:red")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("block size (tokens)")
    ax[1].set_ylabel("internal waste (% of useful tokens)", color="tab:red")
    ax2 = ax[1].twinx()
    ax2.plot([r["block_size"] for r in s], [r["blocks_per_seq"] for r in s], "s--",
             color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_ylabel("block-table entries per sequence", color="tab:blue")
    ax[1].set_title("B. the block-size trade")
    ax[1].grid(alpha=.3)

    d = f["D_cost"]
    ctxs = sorted({r["ctx"] for r in d}, key=int)
    labels = ["contiguous", "paged bs=16", "paged bs=64", "paged bs=256"]
    w = 0.35
    for ci, ctx in enumerate(ctxs):
        rows = [r for r in d if r["ctx"] == ctx]
        ax[2].bar([i + ci * w for i in range(len(rows))],
                  [r["median_step_s"] * 1e3 for r in rows], width=w,
                  label=f"context {ctx}")
    ax[2].set_xticks([i + w / 2 for i in range(len(labels))])
    ax[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax[2].set_ylabel("median decode step (ms)")
    ax[2].set_title("C. the price of a gather with no paged kernel")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paged_cache.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
