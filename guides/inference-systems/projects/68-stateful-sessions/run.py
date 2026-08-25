#!/usr/bin/env python3
"""Project 68 — sessions that live across requests, and what happens when they
have to leave memory.

  A. What a live session is worth: a real 6-turn agent conversation, served
     with and without a kept KV cache.
  B. Evicting is a choice: **drop** (recompute later) or **offload** (copy the
     cache out and copy it back).  Both are measured, and they cross over.
  C. Forking a session: agents branch, branches share a prefix.
  D. Tool stalls: an agent session is idle most of its life.  What holding its
     cache during the stall costs, and the rule that follows.
  E. A fleet of 48 agent sessions under a memory budget, four policies.

  python3 run.py           # ~2 minutes
  python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sesslib as S                                            # noqa: E402
S.add_ctxlib_to_path()
import ctxlib                                                  # noqa: E402

OUT = os.path.join(HERE, "outputs")
FINDINGS = os.path.join(OUT, "findings.json")

LAYERS = 8                 # of 24 — project 57's reason: 48 live caches in RAM
PCIE_GBS = 25.0            # a real H100's usable host<->device bandwidth
TURN_TOKENS = 12           # tokens generated per agent turn


# ---------------------------------------------------------------------------
# a small agent-shaped conversation
# ---------------------------------------------------------------------------

TOOLS = [
    ("search", 2.5), ("read_file", 0.4), ("run_tests", 12.0),
    ("http_get", 1.2), ("db_query", 0.8),
]

STEPS = [
    "Find where the retry limit is configured.",
    "The tool returned three files. Open the most likely one.",
    "That file sets it to 3. Check whether any test depends on that value.",
    "One test does. Change the value to 5 and update the test.",
    "Run the test suite and report the result.",
    "Summarise what you changed and why.",
]


def turn_text(i: int) -> str:
    """One agent turn: the step, plus a chunk of tool output pasted back in.

    Tool results are why agent sessions grow so fast — the model writes 12
    tokens and the tool writes 200 back into the context.
    """
    tool, _ = TOOLS[i % len(TOOLS)]
    filler = (f"[{tool} result] " + "line of tool output; " * 26)
    return f"{STEPS[i % len(STEPS)]}\n{filler}"


def build_history(tok, n_turns: int) -> list[list[int]]:
    """Cumulative token ids after each turn of the conversation."""
    text = ""
    outs = []
    for i in range(n_turns):
        text += f"<|im_start|>user\n{turn_text(i)}<|im_end|>\n" \
                f"<|im_start|>assistant\n"
        outs.append(tok(text, add_special_tokens=False).input_ids)
        text += "ok, done.<|im_end|>\n"
    return outs


# ---------------------------------------------------------------------------
# A: warm vs cold turns
# ---------------------------------------------------------------------------

def section_A(model, tok):
    hist = build_history(tok, len(STEPS))
    cold, warm, rows = [], [], []

    # cold: every turn re-prefills the entire conversation so far
    for i, ids in enumerate(hist):
        _, s = S.prefill(model, ids)
        cold.append(s)

    # warm: the session keeps its cache, each turn prefills only what is new
    past, prev = None, 0
    for i, ids in enumerate(hist):
        fresh = ids[prev:]
        past, s = S.prefill(model, fresh, past=past)
        warm.append(s)
        rows.append(dict(turn=i + 1, ctx=len(ids), fresh=len(fresh),
                         cold_ms=round(cold[i] * 1000, 1),
                         warm_ms=round(s * 1000, 1),
                         speedup=round(cold[i] / max(s, 1e-9), 2)))
        prev = len(ids)

    kv_bytes = S.cache_bytes(past)
    return dict(rows=rows, layers=LAYERS,
                cold_total_ms=round(sum(cold) * 1000, 1),
                warm_total_ms=round(sum(warm) * 1000, 1),
                total_speedup=round(sum(cold) / sum(warm), 2),
                ctx_final=len(hist[-1]),
                kv_bytes=kv_bytes,
                bytes_per_token=round(kv_bytes / len(hist[-1]), 1)), past, hist


# ---------------------------------------------------------------------------
# B: drop or offload?
# ---------------------------------------------------------------------------

def section_B(model, tok, hist):
    """Measure the two ways of bringing an evicted session back."""
    tok_counts = [len(h) for h in hist]
    rows = []
    for n in tok_counts:
        ids = hist[tok_counts.index(n)]
        past, prefill_s = S.prefill(model, ids)           # = recompute cost
        park, park_s = S.timed(S.Offloaded, past)         # copy out
        _, restore_s = S.timed(park.restore)              # copy back
        nb = park.nbytes
        rows.append(dict(
            ctx=n, bytes=nb,
            recompute_ms=round(prefill_s * 1000, 2),
            offload_out_ms=round(park_s * 1000, 2),
            restore_ms=round(restore_s * 1000, 2),
            copy_gbs=round(nb / max(restore_s, 1e-9) / 1e9, 2),
            pcie_restore_ms=round(nb / (PCIE_GBS * 1e9) * 1000, 3),
            ratio=round(prefill_s / max(restore_s, 1e-9), 2)))
        del past, park
    return dict(rows=rows, pcie_gbs=PCIE_GBS,
                mean_copy_gbs=round(statistics.mean(
                    r["copy_gbs"] for r in rows), 2))


# ---------------------------------------------------------------------------
# C: forking a session
# ---------------------------------------------------------------------------

def section_C(model, tok, hist):
    """An agent that tries three tools in parallel forks its session."""
    base_ids = hist[2]
    past, base_s = S.prefill(model, base_ids)
    base_bytes = S.cache_bytes(past)

    branch_texts = ["Try the search tool.", "Try reading the file.",
                    "Try querying the database."]
    branches = []
    for t in branch_texts:
        ids = tok(t, add_special_tokens=False).input_ids
        cp, clone_s = S.timed(S.clone_cache, past)
        cp, ext_s = S.prefill(model, ids, past=cp)
        branches.append(dict(text=t, new_tokens=len(ids),
                             clone_ms=round(clone_s * 1000, 2),
                             extend_ms=round(ext_s * 1000, 2),
                             bytes=S.cache_bytes(cp)))
        del cp

    n = len(branches)
    copied = sum(b["bytes"] for b in branches)
    shared = base_bytes + sum(b["bytes"] - base_bytes for b in branches)
    return dict(base_ctx=len(base_ids), base_bytes=base_bytes,
                base_prefill_ms=round(base_s * 1000, 2),
                branches=branches, n_branches=n,
                copied_bytes=copied, shared_bytes=shared,
                saving_x=round(copied / shared, 2),
                clone_vs_prefill=round(
                    statistics.mean(b["clone_ms"] for b in branches)
                    / (base_s * 1000), 3))


# ---------------------------------------------------------------------------
# E: a fleet of agent sessions under a memory budget
# ---------------------------------------------------------------------------

def fleet(policy, cal, n_sessions=48, turns=6, budget_mb=24.0, seed=5,
          pressure=0.7):
    """Virtual-time simulation of many agent sessions sharing one GPU.

    Calibrated with the numbers measured in A and B: seconds per prefill token,
    seconds per restored byte, bytes per token.  The simulation is what lets us
    ask about 48 concurrent sessions on a machine that can hold six.
    """
    rng = random.Random(seed)
    budget = budget_mb * 1e6
    bpt = cal["bytes_per_token"]
    s_per_tok = cal["prefill_s_per_token"]
    s_per_byte_restore = cal["restore_s_per_byte"]
    s_per_byte_offload = cal["offload_s_per_byte"]
    decode_s = cal["decode_s_per_turn"]

    ev = []
    for sid in range(n_sessions):
        t = rng.expovariate(1 / 6.0)
        ctx = 0
        for k in range(turns):
            ctx += cal["tokens_per_turn"]
            tool, stall = TOOLS[(sid + k) % len(TOOLS)]
            ev.append((t, sid, k, ctx, tool))
            t += stall * rng.uniform(0.6, 1.6) + decode_s
    ev.sort()

    live: dict[int, dict] = {}
    parked: dict[int, dict] = {}
    per_turn: dict[int, list] = {}
    used = 0.0
    if policy == "unlimited":
        budget = float("inf")
    lat, recomputed, restored, evictions, mem_seconds = [], 0, 0, 0, 0.0
    last_t = ev[0][0]

    def free_for(need, now, protect):
        nonlocal used, evictions, mem_seconds
        while used + need > budget and live:
            vic = min((s for s in live.values() if s["sid"] != protect),
                      key=lambda s: s["last"], default=None)
            if vic is None:
                break
            used -= vic["bytes"]
            evictions += 1
            if policy in ("offload-lru", "stall-aware"):
                parked[vic["sid"]] = dict(bytes=vic["bytes"], ctx=vic["ctx"])
            live.pop(vic["sid"])

    for (t, sid, k, ctx, tool) in ev:
        mem_seconds += used * max(0.0, t - last_t)
        last_t = t
        need = ctx * bpt
        cost = 0.0
        s = live.get(sid)
        if s is not None:
            cost += cal["tokens_per_turn"] * s_per_tok      # new tokens only
            used += need - s["bytes"]
        else:
            free_for(need, t, sid)
            p = parked.pop(sid, None)
            if p is not None:
                cost += p["bytes"] * s_per_byte_restore
                restored += 1
                cost += cal["tokens_per_turn"] * s_per_tok
            else:
                cost += ctx * s_per_tok                    # full re-prefill
                recomputed += ctx if k > 0 else 0
            used += need
        # a stall-aware server parks the session *now*, before the tool runs,
        # but only when the memory is actually contended: parking a session on
        # an empty machine pays the copy for nothing.
        live[sid] = dict(sid=sid, bytes=need, ctx=ctx, last=t)
        if policy == "no-cache":
            used -= need
            live.pop(sid)
        elif policy == "stall-aware":
            stall = dict(TOOLS)[tool]
            round_trip = need * (s_per_byte_offload + s_per_byte_restore)
            if used > pressure * budget and stall > round_trip:
                cost += need * s_per_byte_offload
                parked[sid] = dict(bytes=need, ctx=ctx)
                used -= need
                live.pop(sid)
                evictions += 1
        lat.append(cost + decode_s)
        per_turn.setdefault(k, []).append(cost + decode_s)
    return dict(policy=policy,
                mean_ms=round(1000 * statistics.mean(lat), 1),
                p95_ms=round(1000 * S.pct(lat, 95), 1),
                p99_ms=round(1000 * S.pct(lat, 99), 1),
                evictions=evictions, restored=restored,
                recomputed_tokens=recomputed,
                peak_sessions_held=None,
                last_turn_ms=round(1000 * statistics.mean(
                    per_turn[max(per_turn)]), 1),
                first_turn_ms=round(1000 * statistics.mean(per_turn[0]), 1),
                mem_gb_seconds=round(mem_seconds / 1e9, 2))


# ---------------------------------------------------------------------------

def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C, D, E = F["A"], F["B"], F["C"], F["D"], F["E"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle("Stateful sessions: keeping a KV cache alive between turns, "
                 "and what eviction really costs", fontsize=13)

    p = ax[0][0]
    xs = [r["turn"] for r in A["rows"]]
    p.plot(xs, [r["cold_ms"] for r in A["rows"]], "o-", color="#c0504d",
           label="cold: re-prefill the whole history")
    p.plot(xs, [r["warm_ms"] for r in A["rows"]], "s-", color="#4a6fa5",
           label="warm: session keeps its cache")
    p.set_xlabel("agent turn")
    p.set_ylabel("prefill time (ms)")
    p.set_title(f"A. A live session pays only for what is new\n"
                f"whole conversation {A['total_speedup']}x faster")
    p.legend(fontsize=8)
    p.grid(alpha=0.3)

    p = ax[0][1]
    rows = B["rows"]
    xs = [r["ctx"] for r in rows]
    p.plot(xs, [r["recompute_ms"] for r in rows], "o-", color="#c0504d",
           label="drop → recompute (measured)")
    p.plot(xs, [r["restore_ms"] for r in rows], "s-", color="#4c9f70",
           label=f"offload → restore (measured, {B['mean_copy_gbs']} GB/s)")
    p.plot(xs, [r["pcie_restore_ms"] for r in rows], "^--", color="#8d9db6",
           label=f"restore over a {B['pcie_gbs']} GB/s PCIe link (arithmetic)")
    p.set_xlabel("session context (tokens)")
    p.set_ylabel("milliseconds to bring the session back (log)")
    p.set_yscale("log")
    p.set_title("B. Drop or offload? Both measured on the same caches")
    p.legend(fontsize=8)
    p.grid(alpha=0.3)

    p = ax[1][0]
    names = list(D["rows"].keys())
    vals = [D["rows"][n]["hold_cost_ratio"] for n in names]
    p.bar(range(len(names)), vals, color="#4a6fa5")
    p.axhline(1.0, color="k", lw=1)
    p.set_xticks(range(len(names)))
    p.set_xticklabels(names, rotation=15, fontsize=8)
    p.set_ylabel("stall length / round-trip cost (log)")
    p.set_yscale("log")
    p.axhline(1.0, color="k", lw=1)
    p.set_title("D. During a tool stall, is the cache worth its seat?\n"
                "above 1 = the copy pays for itself")

    p = ax[1][1]
    pol = [r["policy"] for r in E]
    base = E[0]["mean_ms"]
    rel = [r["mean_ms"] / base for r in E]
    rel_gpu = [r["mean_ms"] / F["E_gpu"][0]["mean_ms"] for r in F["E_gpu"]]
    w = 0.38
    p.bar([i - w / 2 for i in range(len(pol))], rel, w, color="#4a6fa5",
          label="measured (this CPU, 0.5B)")
    p.bar([i + w / 2 for i in range(len(pol))], rel_gpu, w, color="#4c9f70",
          label="arithmetic (8B on an H100)")
    p.axhline(1.0, color="k", lw=1)
    p.set_xticks(range(len(pol)))
    p.set_xticklabels(pol, rotation=15, fontsize=8)
    p.set_ylabel("mean turn latency / no-cache")
    p.set_title("E. 48 agent sessions, one budget, five policies")
    p.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, "sessions.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    print(f"loading Qwen2.5-0.5B-Instruct ({LAYERS} of 24 blocks) ...",
          flush=True)
    tok, model = ctxlib.load(layers=LAYERS, threads=6)

    print("A. warm vs cold turns ...", flush=True)
    A, past, hist = section_A(model, tok)
    print(f"   {A['total_speedup']}x over the conversation, "
          f"{A['bytes_per_token']} bytes/token", flush=True)

    print("B. drop vs offload ...", flush=True)
    Bs = section_B(model, tok, hist)

    print("C. forking ...", flush=True)
    Cs = section_C(model, tok, hist)

    # --- calibration for the fleet simulation -----------------------------
    last = A["rows"][-1]
    cal = dict(
        bytes_per_token=A["bytes_per_token"],
        tokens_per_turn=int(A["ctx_final"] / len(A["rows"])),
        prefill_s_per_token=statistics.mean(
            r["recompute_ms"] / 1000 / r["ctx"] for r in Bs["rows"]),
        restore_s_per_byte=statistics.mean(
            r["restore_ms"] / 1000 / r["bytes"] for r in Bs["rows"]),
        offload_s_per_byte=statistics.mean(
            r["offload_out_ms"] / 1000 / r["bytes"] for r in Bs["rows"]),
        decode_s_per_turn=0.0,
        park_gain=1.0,
    )
    # one real decode measurement for the per-turn generation cost
    t0 = time.perf_counter()
    with torch.inference_mode():
        o = model(torch.tensor([hist[-1][-1:]]), past_key_values=past,
                  use_cache=True, logits_to_keep=1)
        S.decode(model, o.past_key_values, o.logits, TURN_TOKENS)
    cal["decode_s_per_turn"] = time.perf_counter() - t0
    del past

    # --- D: the stall arithmetic ------------------------------------------
    ctx = A["ctx_final"]
    nb = ctx * A["bytes_per_token"]
    round_trip_measured = nb * (cal["offload_s_per_byte"]
                                + cal["restore_s_per_byte"])
    round_trip_pcie = 2 * nb / (PCIE_GBS * 1e9)
    D = dict(ctx=ctx, bytes=nb,
             round_trip_measured_ms=round(round_trip_measured * 1000, 2),
             round_trip_pcie_ms=round(round_trip_pcie * 1000, 3),
             rows={})
    for tool, stall in TOOLS:
        D["rows"][tool] = dict(
            stall_s=stall,
            hold_cost_ratio=round(stall / max(round_trip_measured, 1e-9), 1),
            hold_cost_ratio_pcie=round(stall / max(round_trip_pcie, 1e-9), 1),
            mb_seconds_if_held=round(nb / 1e6 * stall, 1))

    print("E. fleet simulation ...", flush=True)
    POLICIES = ("no-cache", "drop-lru", "offload-lru", "stall-aware",
                "unlimited")
    E = [fleet(p, cal) for p in POLICIES]

    # the same fleet on hardware this machine does not have: an 8B model on an
    # H100, its KV cache offloaded to host memory over PCIe.  Every number in
    # `gpu_cal` is arithmetic from published figures, and it exists to answer
    # the obvious objection to section B -- that a CPU with no PCIe link makes
    # offloading look artificially good.
    gpu_cal = dict(
        bytes_per_token=131072,          # 8B, GQA 8 KV heads, 32 layers, fp16
        tokens_per_turn=cal["tokens_per_turn"],
        prefill_s_per_token=1 / 10000.0,  # ~10k tok/s prefill on one H100
        restore_s_per_byte=1 / 25e9,      # 25 GB/s usable PCIe
        offload_s_per_byte=1 / 25e9,
        decode_s_per_turn=TURN_TOKENS / 60.0,   # ~60 tok/s decode
        park_gain=1.0)
    E_gpu = [fleet(p, gpu_cal, budget_mb=24.0 * gpu_cal["bytes_per_token"]
                   / cal["bytes_per_token"]) for p in POLICIES]

    # where the drop-vs-offload decision actually flips
    def crossover(bytes_per_token, prefill_tok_s):
        return bytes_per_token * prefill_tok_s        # bytes/s the link needs
    Bs["crossover"] = dict(
        note=("offload beats recompute when the link is faster than "
              "bytes_per_token x prefill_tokens_per_second"),
        rows={
            "this CPU, 0.5B (8 layers), measured":
                dict(bytes_per_token=cal["bytes_per_token"],
                     prefill_tok_s=round(1 / cal["prefill_s_per_token"], 1),
                     link_needed_gb_s=round(crossover(
                         cal["bytes_per_token"],
                         1 / cal["prefill_s_per_token"]) / 1e9, 3),
                     link_available_gb_s=Bs["mean_copy_gbs"]),
            "8B on an H100 over PCIe (arithmetic)":
                dict(bytes_per_token=131072, prefill_tok_s=10000,
                     link_needed_gb_s=round(crossover(131072, 10000) / 1e9, 3),
                     link_available_gb_s=25.0),
            "8B over 10 Gb Ethernet (arithmetic)":
                dict(bytes_per_token=131072, prefill_tok_s=10000,
                     link_needed_gb_s=round(crossover(131072, 10000) / 1e9, 3),
                     link_available_gb_s=1.25),
            "70B MHA on 8xH100 over PCIe (arithmetic)":
                dict(bytes_per_token=2621440, prefill_tok_s=4000,
                     link_needed_gb_s=round(crossover(2621440, 4000) / 1e9, 3),
                     link_available_gb_s=25.0),
        })

    F = dict(A=A, B=Bs, C=Cs, D=D, E=E, E_gpu=E_gpu, gpu_cal=gpu_cal,
             cal={k: (round(v, 12) if isinstance(
        v, float) else v) for k, v in cal.items()})
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- A. warm vs cold ----------------------------------------------")
    for r in A["rows"]:
        print(f"turn {r['turn']}: ctx {r['ctx']:5d}  fresh {r['fresh']:4d}  "
              f"cold {r['cold_ms']:7.1f} ms  warm {r['warm_ms']:6.1f} ms  "
              f"{r['speedup']}x")
    print("\n--- B. drop vs offload -------------------------------------------")
    for r in Bs["rows"]:
        print(f"ctx {r['ctx']:5d}  {r['bytes'] / 1e6:6.2f} MB  recompute "
              f"{r['recompute_ms']:7.2f} ms  restore {r['restore_ms']:6.2f} ms"
              f"  ({r['ratio']}x)   PCIe restore {r['pcie_restore_ms']} ms")
    print("\n--- C. forking ---------------------------------------------------")
    print(f"base {Cs['base_ctx']} tokens = {Cs['base_bytes'] / 1e6:.2f} MB; "
          f"{Cs['n_branches']} branches copied {Cs['copied_bytes'] / 1e6:.2f} MB"
          f" vs {Cs['shared_bytes'] / 1e6:.2f} MB shared "
          f"({Cs['saving_x']}x)")
    print(f"a clone costs {Cs['clone_vs_prefill']}x the prefill it replaces")
    print("\n--- D. tool stalls -----------------------------------------------")
    print(f"a {D['ctx']}-token session is {D['bytes'] / 1e6:.2f} MB; "
          f"park+restore {D['round_trip_measured_ms']} ms measured, "
          f"{D['round_trip_pcie_ms']} ms over PCIe")
    for tool, r in D["rows"].items():
        print(f"  {tool:<10} stall {r['stall_s']:5.1f}s -> "
              f"{r['hold_cost_ratio']:7.1f}x the round trip; holding costs "
              f"{r['mb_seconds_if_held']} MB-seconds")
    print("\n--- B2. where the choice flips ------------------------------------")
    for k, v in Bs["crossover"]["rows"].items():
        verdict = ("offload" if v["link_available_gb_s"] >= v["link_needed_gb_s"]
                   else "recompute")
        print(f"  {k:<42} needs {v['link_needed_gb_s']:8.3f} GB/s, has "
              f"{v['link_available_gb_s']:6.2f} -> {verdict}")
    print("\n--- E. the fleet -------------------------------------------------")
    for r in E:
        print(f"{r['policy']:<13} mean {r['mean_ms']:7.1f} ms  p95 "
              f"{r['p95_ms']:8.1f} ms  p99 {r['p99_ms']:8.1f} ms  evictions "
              f"{r['evictions']:4d}  restored {r['restored']:4d}  recomputed "
              f"{r['recomputed_tokens']:6d} tok  {r['mem_gb_seconds']} GB-s")
    print("   the same fleet, 8B on an H100 (arithmetic calibration):")
    for r in E_gpu:
        print(f"{r['policy']:<13} mean {r['mean_ms']:7.1f} ms  p95 "
              f"{r['p95_ms']:8.1f} ms  turn 1 {r['first_turn_ms']:7.1f} ms  "
              f"turn 6 {r['last_turn_ms']:7.1f} ms  recomputed "
              f"{r['recomputed_tokens']:6d} tok")
    plot(F)


if __name__ == "__main__":
    main()
