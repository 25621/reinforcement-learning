"""Project 28 -- speculation inside a continuous batch.

Everything in projects 23-27 ran one sequence at a time. A real server runs
dozens at once, and speculation changes the shape of every forward pass: the
batch now carries k+1 query tokens per row, and after verification each row
has advanced by a *different* amount.

  A. Correctness first: a heterogeneous speculative batch must produce exactly
     what each request would have produced on its own.
  B. The mechanics of raggedness -- how uneven is the batch, and what does
     padding the drafts cost?
  C. The scaling law. Speculation trades spare compute for latency, and a
     bigger batch has less spare compute. Measure where it runs out.
  D. End to end: throughput with and without speculation, at two batch widths.
  E. The tempting shortcut -- advance every row by the batch minimum so the
     rows stay aligned. Correct, and expensive.

    python3 run.py           # ~5 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))

import torch  # noqa: E402

import batchlib  # noqa: E402
import specbatch as SB  # noqa: E402

K = 3
MAX_NEW = 32
MAX_LEN = 640
F = {}

PASSAGE = (
    "The Antikythera mechanism is an ancient Greek hand-powered device that "
    "has been described as the oldest known analogue computer. It was used to "
    "predict astronomical positions and eclipses decades in advance."
)

# Four copy-heavy prompts of differing difficulty. They have to be copy-heavy
# because the drafter here is prompt lookup ([project 25]) -- on prompts with
# nothing to copy every row proposes nothing, alpha sits at 1.0, and there is
# no raggedness left to study. They differ from each other so that the rows
# accept *different* amounts, which is the whole point of section B.
PROMPTS = [
    "Repeat the following paragraph exactly, changing only the word "
    "\"Greek\" to \"Hellenic\". Output the paragraph and nothing else.\n\n"
    + PASSAGE,
    "Rewrite this function so the default timeout is 60 instead of 30. "
    "Output the whole function.\n\n"
    "```python\ndef load_config(path):\n    cfg = read(path)\n"
    "    cfg['timeout'] = int(cfg.get('timeout', 30))\n    return cfg\n```",
    "Repeat this list exactly, twice: apple, banana, cherry, date, "
    "elderberry, fig, grape.",
    "Echo this JSON object exactly, then echo it again:\n"
    "{\"name\": \"Lyon\", \"country\": \"France\", \"population\": 513275}",
]


def chat_ids(tok, msg):
    text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# the batched loops
# ---------------------------------------------------------------------------


def run_batch(runner, prompts, k=0, max_new=MAX_NEW, n_slots=8, sync=False,
              max_len=MAX_LEN):
    """A saturated continuous batch, optionally speculative.

    `k = 0` is the ordinary one-token-per-row decode loop (Phase 3's engine).
    `k > 0` proposes k n-gram drafts per row and verifies them all in one pass.
    `sync = True` is section E's shortcut: every row advances by the *minimum*
    number of tokens any row accepted, keeping the rows aligned.

    Time is virtual -- the clock advances by the measured duration of each
    forward pass, never by `time.time()`. On a shared machine that is the only
    way two configurations minutes apart can be compared.
    """
    pool = SB.SpecSlotKV(runner.n_layers, n_slots, runner.n_kv_heads,
                         runner.d_head, max_len)
    rows = []
    for i, p in enumerate(prompts):
        rows.append({"rid": i, "prompt": list(p), "tokens": list(p),
                     "out": [], "slot": None, "done": False})

    clock = 0.0
    stats = {"prefill_s": 0.0, "decode_s": 0.0, "draft_s": 0.0,
             "passes": 0, "verified_slots": 0, "accepted_tokens": 0,
             "bonus_tokens": 0, "pad_slots": 0, "proposed": 0,
             "accept_hist": [], "step_rows": [], "sync_lost": 0}

    pending = list(rows)
    live = []
    done_rows = []

    def admit():
        """Fill every free slot from the queue, then prefill the newcomers in
        one rectangular pass. Every batch width therefore serves the *same*
        set of requests -- a narrow server just takes more waves to do it.
        Comparing a 2-slot server on two easy prompts against a 16-slot server
        on all sixteen would measure the workload, not the batch width."""
        nonlocal clock
        new = []
        while pending and pool.n_free() and len(live) + len(new) < n_slots:
            r = pending.pop(0)
            r["slot"] = pool.acquire()
            new.append(r)
        if not new:
            return
        lens = [len(r["prompt"]) for r in new]
        ids = torch.zeros(len(new), max(lens), dtype=torch.long)
        for j, r in enumerate(new):
            ids[j, :len(r["prompt"])] = torch.tensor(r["prompt"])
        logits, dt = runner.prefill(pool, [r["slot"] for r in new], ids, lens,
                                    count=False)
        stats["prefill_s"] += dt
        stats["prefill_passes"] = stats.get("prefill_passes", 0) + 1
        clock += dt
        for j, r in enumerate(new):
            nxt = int(logits[j].argmax())
            r["tokens"].append(nxt)
            r["out"].append(nxt)
        live.extend(new)

    # -- decode -------------------------------------------------------------
    admit()
    while live:
        act = [r for r in live if not r["done"]]
        slots = [r["slot"] for r in act]
        lengths = [len(r["tokens"]) for r in act]

        if k == 0:
            toks = [r["tokens"][-1] for r in act]
            # `decode_step` wants "tokens already owned", which is one less
            # than the length of the token list under our invariant.
            logits, dt = runner.decode_step(
                pool, slots, toks, [x - 1 for x in lengths], count=False)
            stats["decode_s"] += dt
            clock += dt
            stats["passes"] += 1
            stats["verified_slots"] += len(act)
            stats["step_rows"].append(len(act))
            for j, r in enumerate(act):
                nxt = int(logits[j].argmax())
                r["tokens"].append(nxt)
                r["out"].append(nxt)
                stats["bonus_tokens"] += 1
        else:
            t0 = time.perf_counter()
            block = torch.zeros(len(act), k + 1, dtype=torch.long)
            real_k = []
            for j, r in enumerate(act):
                d = SB.ngram_propose(r["tokens"], k)
                padded, n_real = SB.pad_drafts(d, k)
                real_k.append(n_real)
                block[j, 0] = r["tokens"][-1]
                block[j, 1:] = torch.tensor(padded)
            stats["draft_s"] += time.perf_counter() - t0
            stats["proposed"] += k * len(act)
            stats["pad_slots"] += sum(k - n for n in real_k)

            logits, dt = runner.verify_step(pool, slots, block, lengths,
                                            count=False)
            stats["decode_s"] += dt
            clock += dt
            stats["passes"] += 1
            stats["verified_slots"] += len(act) * (k + 1)
            stats["step_rows"].append(len(act))

            preds = logits.argmax(-1)
            gains = []
            for j, r in enumerate(act):
                n_acc = 0
                for i in range(real_k[j]):
                    if int(block[j, i + 1]) == int(preds[j, i]):
                        n_acc += 1
                    else:
                        break
                gains.append(n_acc)
            if sync:
                m = min(gains)
                stats["sync_lost"] += sum(gains) - m * len(gains)
                gains = [m] * len(gains)
            for j, r in enumerate(act):
                n_acc = gains[j]
                new = [int(block[j, i + 1]) for i in range(n_acc)]
                new.append(int(preds[j, n_acc]))
                r["tokens"].extend(new)
                r["out"].extend(new)
                stats["accepted_tokens"] += n_acc
                stats["bonus_tokens"] += 1
                stats["accept_hist"].append(n_acc)

        for r in act:
            if len(r["out"]) >= max_new:
                r["done"] = True
                pool.release(r["slot"])
                done_rows.append(r)
        live[:] = [r for r in live if not r["done"]]
        admit()

    done_rows.sort(key=lambda r: r["rid"])
    out_tokens = sum(min(len(r["out"]), max_new) for r in done_rows)
    stats["wall_s"] = clock
    stats["out_tokens"] = out_tokens
    stats["tok_per_s"] = out_tokens / clock
    stats["outputs"] = [r["out"][:max_new] for r in done_rows]
    return stats


# ---------------------------------------------------------------------------
# A + B
# ---------------------------------------------------------------------------


def section_ab(runner, tok, prompt_ids):
    print("\n== A. a speculative batch must not change any row ==")
    solo = [run_batch(runner, [p], k=0, n_slots=1)["outputs"][0]
            for p in prompt_ids]
    batched = run_batch(runner, prompt_ids, k=0, n_slots=4)
    spec = run_batch(runner, prompt_ids, k=K, n_slots=4)
    rows = []
    for i, p in enumerate(prompt_ids):
        rows.append({
            "row": i,
            "prompt_tokens": len(p),
            "batched_identical": solo[i] == batched["outputs"][i],
            "spec_identical": solo[i] == spec["outputs"][i],
            "text": tok.decode(solo[i]),
        })
        print(f"  row {i}: batched={rows[-1]['batched_identical']}  "
              f"speculative={rows[-1]['spec_identical']}")
    F["A"] = {"rows": rows,
              "all_identical": all(r["spec_identical"] for r in rows)}

    print("\n== B. how ragged is the batch? ==")
    hist = spec["accept_hist"]
    per = [hist.count(i) / len(hist) for i in range(K + 1)]
    F["B"] = {
        "k": K, "rows": 4,
        "passes": spec["passes"],
        "accept_hist": [round(x, 4) for x in per],
        "mean_accepted": round(sum(hist) / len(hist), 3),
        "proposed": spec["proposed"],
        "pad_slots": spec["pad_slots"],
        "pad_frac": round(spec["pad_slots"] / spec["proposed"], 4),
        "accepted_tokens": spec["accepted_tokens"],
        "verified_slots": spec["verified_slots"],
        "useful_frac": round(
            (spec["accepted_tokens"] + spec["bonus_tokens"])
            / spec["verified_slots"], 4),
        "tok_per_s_nospec": round(batched["tok_per_s"], 3),
        "tok_per_s_spec": round(spec["tok_per_s"], 3),
        "speedup": round(spec["tok_per_s"] / batched["tok_per_s"], 3),
    }
    print(f"  accepted-per-row histogram: "
          f"{[round(x,2) for x in F['B']['accept_hist']]}")
    print(f"  {F['B']['pad_frac']:.1%} of proposals were padding "
          f"(a row with nothing to propose)")
    print(f"  {F['B']['useful_frac']:.1%} of verified token-slots became "
          f"output tokens")
    print(f"  batch of 4: {F['B']['speedup']:.2f}x")


# ---------------------------------------------------------------------------
# C. the scaling law
# ---------------------------------------------------------------------------


def section_c(runner):
    print("\n== C. what a verification pass costs as the batch grows ==")
    rows = []
    for b in (1, 2, 4, 8, 16, 32):
        pool = SB.SpecSlotKV(runner.n_layers, b, runner.n_kv_heads,
                             runner.d_head, 256)
        slots = list(range(b))
        ids = torch.randint(1000, 12000, (b, 64))
        runner.prefill(pool, slots, ids, [64] * b, count=False)
        lengths = [64] * b
        blk1 = torch.randint(1000, 12000, (b, 1))
        blkk = torch.randint(1000, 12000, (b, K + 1))

        def dec():
            runner.decode_step(pool, slots, blk1[:, 0].tolist(),
                               [x - 1 for x in lengths], count=False)

        def ver():
            runner.verify_step(pool, slots, blkk, lengths, count=False)

        t = batchlib_interleaved({"decode": dec, "verify": ver}, rounds=3)
        rows.append({
            "batch": b,
            "decode_ms": round(t["decode"] * 1000, 2),
            "verify_ms": round(t["verify"] * 1000, 2),
            "tax": round(t["verify"] / t["decode"], 4),
        })
        print(f"  B={b:2d}  decode {rows[-1]['decode_ms']:7.1f} ms  "
              f"verify(k+1={K+1}) {rows[-1]['verify_ms']:7.1f} ms  "
              f"tax {rows[-1]['tax']:.2f}x")
    F["C"] = {"k": K, "rows": rows}


def batchlib_interleaved(fns, rounds=3, warmup=1):
    for _ in range(warmup):
        for f in fns.values():
            f()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for name, f in fns.items():
            t0 = time.perf_counter()
            f()
            best[name] = min(best[name], time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# D + E. end to end
# ---------------------------------------------------------------------------


def section_de(runner, prompt_ids):
    print("\n== D. end to end, two batch widths ==")
    work = (prompt_ids * 4)[:16]          # the same 16 requests every time
    rows = []
    tax = {r["batch"]: r["tax"] for r in F["C"]["rows"]}
    for n_slots in (2, 4, 8, 16):
        base = run_batch(runner, work, k=0, n_slots=n_slots)
        spec = run_batch(runner, work, k=K, n_slots=n_slots)
        alpha = ((spec["accepted_tokens"] + spec["bonus_tokens"])
                 / sum(spec["step_rows"]))
        rows.append({
            "n_slots": n_slots,
            "nospec_tok_s": round(base["tok_per_s"], 3),
            "spec_tok_s": round(spec["tok_per_s"], 3),
            "speedup": round(spec["tok_per_s"] / base["tok_per_s"], 3),
            "nospec_passes": base["passes"],
            "spec_passes": spec["passes"],
            "alpha": round(alpha, 3),
            "tax_at_this_batch": tax.get(n_slots),
        })
        print(f"  slots={n_slots}: {rows[-1]['nospec_tok_s']:.1f} -> "
              f"{rows[-1]['spec_tok_s']:.1f} tok/s = "
              f"{rows[-1]['speedup']:.2f}x   (alpha {alpha:.2f}, passes "
              f"{base['passes']} -> {spec['passes']})")
    F["D"] = {"rows": rows}

    print("\n== E. the 'keep the rows aligned' shortcut ==")
    ragged = run_batch(runner, work, k=K, n_slots=8)
    synced = run_batch(runner, work, k=K, n_slots=8, sync=True)
    F["E"] = {
        "ragged_tok_s": round(ragged["tok_per_s"], 3),
        "synced_tok_s": round(synced["tok_per_s"], 3),
        "ratio": round(ragged["tok_per_s"] / synced["tok_per_s"], 3),
        "tokens_thrown_away": synced["sync_lost"],
        "synced_accepted_kept": synced["accepted_tokens"],
        "synced_bonus": synced["bonus_tokens"],
        "accepted_if_ragged": ragged["accepted_tokens"],
        "ragged_bonus": ragged["bonus_tokens"],
        "identical": ragged["outputs"] == synced["outputs"],
        "ragged_passes": ragged["passes"],
        "synced_passes": synced["passes"],
    }
    print(f"  ragged {F['E']['ragged_tok_s']:.1f} tok/s vs synced "
          f"{F['E']['synced_tok_s']:.1f} tok/s = {F['E']['ratio']:.2f}x")
    print(f"  the shortcut threw away {F['E']['tokens_thrown_away']} already-"
          f"verified tokens; output identical: {F['E']['identical']}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    b = f["B"]
    ax[0].bar(range(K + 1), b["accept_hist"], color="#e67e22")
    ax[0].set_xticks(range(K + 1))
    ax[0].set_xlabel("tokens accepted by one row in one pass")
    ax[0].set_ylabel("fraction of (row, pass) pairs")
    ax[0].set_title(f"B. ragged: mean {b['mean_accepted']:.2f}, "
                    f"but rarely the mean")

    c = f["C"]["rows"]
    bs = [r["batch"] for r in c]
    ax[1].plot(bs, [r["decode_ms"] for r in c], "o-", color="#7f8c8d",
               label="plain decode step")
    ax[1].plot(bs, [r["verify_ms"] for r in c], "s-", color="#2471a3",
               label=f"verify {K+1} tokens/row")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(bs)
    ax[1].set_xticklabels(bs)
    ax[1].set_xlabel("batch size")
    ax[1].set_ylabel("milliseconds per pass")
    ax[1].set_title("C. the pass costs")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    alpha = f["D"]["rows"][-1]["alpha"]
    ax[2].plot(bs, [r["tax"] for r in c], "s-", color="#c0392b",
               label="speculation tax (verify ÷ decode)")
    ax[2].axhline(alpha, color="#27ae60", ls="--",
                  label=f"measured alpha = {alpha:.2f}")
    ax[2].fill_between(bs, [r["tax"] for r in c], alpha, alpha=0.15,
                       color="#27ae60")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xticks(bs)
    ax[2].set_xticklabels(bs)
    ax[2].set_xlabel("batch size")
    ax[2].set_ylabel("x a plain decode step")
    ax[2].set_title("C. speculation pays while alpha > tax")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    d = f["D"]["rows"]
    xs = range(len(d))
    ax[3].bar([i - 0.2 for i in xs], [r["nospec_tok_s"] for r in d], 0.4,
              color="#7f8c8d", label="no speculation")
    ax[3].bar([i + 0.2 for i in xs], [r["spec_tok_s"] for r in d], 0.4,
              color="#2471a3", label=f"speculative (k={K})")
    ax[3].set_xticks(list(xs))
    ax[3].set_xticklabels([f"{r['n_slots']} slots" for r in d])
    ax[3].set_ylabel("output tokens / s")
    ax[3].set_title("D. end to end")
    ax[3].legend(fontsize=8)
    for i, r in enumerate(d):
        ax[3].text(i + 0.2, r["spec_tok_s"], f"{r['speedup']:.2f}x",
                   ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speculation_batching.png"), dpi=110)
    print("wrote outputs/speculation_batching.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    runner, tok = batchlib.load_runner()
    # `load_runner` returns a plain `BatchedRunner`. Re-class it in place
    # rather than loading 2 GB of weights a second time -- the subclass adds
    # methods only, no new state.
    runner.__class__ = SB.SpecBatchedRunner
    prompt_ids = [chat_ids(tok, p) for p in PROMPTS]
    F["setup"] = {"model": batchlib.MODEL_ID, "k": K, "max_new": MAX_NEW,
                  "threads": batchlib.N_THREADS,
                  "prompt_tokens": [len(p) for p in prompt_ids]}
    section_ab(runner, tok, prompt_ids)
    section_c(runner)
    section_de(runner, prompt_ids)
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
