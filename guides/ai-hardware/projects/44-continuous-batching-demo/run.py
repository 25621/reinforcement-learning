"""Project 44 -- Continuous batching demo.

Two schedulers, one workload, one model.

  * static batching: fill a batch, run it until the *last* sequence in it
    finishes, then start the next batch.  Every sequence that finished early
    keeps its seat -- and the hardware keeps computing its row.
  * continuous batching: after every single decode step, drop whoever finished
    and admit whoever is waiting.

The workload is deliberately uneven (short and long generations mixed) and the
requests arrive over time, because that is exactly the situation static batching
handles badly.

Runs in about 4 minutes on 12 CPU threads.
"""

import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "39-deploy-with-vllm"))
import servelib as S  # noqa: E402

OUT = S.outdir(__file__)
N_REQ = 24
CAPS = [4, 8, 16]
ARRIVAL_RATE = 4.0        # requests per second
results = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------- workload
def make_workload(w, n=N_REQ, seed=0):
    """Uneven prompts and uneven generation lengths, with Poisson arrivals."""
    rng = random.Random(seed)
    base = ("Explain, in one paragraph, why memory bandwidth rather than "
            "arithmetic is the limit for language model inference. ")
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(ARRIVAL_RATE)
        plen = rng.choice([16, 24, 32, 64, 96])
        olen = rng.choice([8, 12, 16, 24, 48, 64])
        reqs.append(dict(rid=i, arrival=t, prompt_len=plen, out_len=olen,
                         ids=S.prompt_ids(w.tok, base, plen)))
    return reqs


class Req:
    def __init__(self, spec):
        self.spec = spec
        self.seq = S.Sequence(spec["rid"], spec["ids"], max_new=spec["out_len"])
        self.ttft = None
        self.finish = None
        self.produced = 0


# ---------------------------------------------------------------- schedulers
def run_static(eng, w, specs, cap):
    """Form a batch, run it to completion, then form the next one."""
    t0 = time.perf_counter()
    pending = [Req(s) for s in specs]
    done, timeline, wasted, useful, steps = [], [], 0, 0, 0
    while pending:
        # wait until at least one request has arrived
        while not any(time.perf_counter() - t0 >= r.spec["arrival"] for r in pending):
            time.sleep(0.005)
        ready = [r for r in pending if time.perf_counter() - t0 >= r.spec["arrival"]]
        batch = ready[:cap]
        for r in batch:
            pending.remove(r)
        lg = eng.forward([r.seq for r in batch], [r.seq.prompt_ids for r in batch])
        now = time.perf_counter() - t0
        for r, g in zip(batch, lg):
            r.seq.out_ids.append(S.greedy(g))
            r.produced = 1
            r.ttft = now
        # the batch runs until its LONGEST member is done
        while any(r.produced < r.spec["out_len"] for r in batch):
            lg = eng.decode_step([r.seq for r in batch])
            steps += 1
            now = time.perf_counter() - t0
            timeline.append((now, sum(1 for r in batch
                                      if r.produced < r.spec["out_len"]), len(batch)))
            live = 0
            for r, g in zip(batch, lg):
                if r.produced < r.spec["out_len"]:
                    r.seq.out_ids.append(S.greedy(g))
                    r.produced += 1
                    useful += 1
                    live += 1
                    if r.produced == r.spec["out_len"]:
                        r.finish = now
            # Seats that produced nothing this step: held by a request that has
            # already finished, or never filled because the batch was short.
            wasted += cap - live
        for r in batch:
            eng.free(r.seq)
            done.append(r)
    return summarize(done, time.perf_counter() - t0, timeline, wasted, useful,
                     steps, cap, "static")


def run_continuous(eng, w, specs, cap, max_admit=None):
    """Admit and evict at every step boundary."""
    t0 = time.perf_counter()
    waiting = [Req(s) for s in specs]
    active, done, timeline = [], [], []
    wasted = useful = steps = 0
    while waiting or active:
        now = time.perf_counter() - t0
        arrived = [r for r in waiting if now >= r.spec["arrival"]]
        room = cap - len(active)
        admit = arrived[:room if max_admit is None else min(room, max_admit)]
        for r in admit:
            waiting.remove(r)
        if admit:
            lg = eng.forward([r.seq for r in admit],
                             [r.seq.prompt_ids for r in admit])
            now = time.perf_counter() - t0
            for r, g in zip(admit, lg):
                r.seq.out_ids.append(S.greedy(g))
                r.produced = 1
                r.ttft = now
                active.append(r)
        if not active:
            time.sleep(0.005)
            continue
        lg = eng.decode_step([r.seq for r in active])
        steps += 1
        now = time.perf_counter() - t0
        timeline.append((now, len(active), cap))
        still = []
        for r, g in zip(active, lg):
            r.seq.out_ids.append(S.greedy(g))
            r.produced += 1
            useful += 1
            if r.produced >= r.spec["out_len"]:
                r.finish = now
                eng.free(r.seq)
                done.append(r)
            else:
                still.append(r)
        wasted += cap - len(active)      # seats that produced nothing this step
        active = still
    name = "continuous" if max_admit is None else f"continuous (admit<={max_admit})"
    return summarize(done, time.perf_counter() - t0, timeline, wasted, useful,
                     steps, cap, name)


def summarize(done, makespan, timeline, wasted, useful, steps, cap, name):
    done = sorted(done, key=lambda r: r.spec["rid"])
    lat = sorted(r.finish - r.spec["arrival"] for r in done)
    ttft = sorted(r.ttft - r.spec["arrival"] for r in done)
    tokens = sum(r.produced for r in done)
    return dict(
        name=name, cap=cap, makespan=makespan, steps=steps,
        tokens=tokens, throughput=tokens / makespan,
        mean_latency=sum(lat) / len(lat), p99_latency=lat[int(0.99 * (len(lat) - 1))],
        max_latency=lat[-1],
        mean_ttft=sum(ttft) / len(ttft), p99_ttft=ttft[int(0.99 * (len(ttft) - 1))],
        # Slot efficiency counts decode steps only: of the cap x steps seats the
        # hardware paid for, how many produced a token?  Both schedulers are
        # measured the same way, so an empty seat and a seat held by a finished
        # request cost exactly the same -- which they do.
        wasted_slot_steps=wasted, useful_slot_steps=useful,
        slot_efficiency=useful / max(1, cap * steps),
        timeline=timeline,
        latencies=lat, ttfts=ttft)


# -------------------------------------------------------------------- figures
def make_plots(res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    runs = res["runs"]
    caps = sorted({r["cap"] for r in runs})
    kinds = ["static", "continuous"]
    width = 0.35
    for j, kind in enumerate(kinds):
        ys = [next(r["throughput"] for r in runs
                   if r["cap"] == c and r["name"] == kind) for c in caps]
        ax[0].bar([i + (j - 0.5) * width for i in range(len(caps))], ys, width,
                  label=kind, color=["#d62728", "#1f77b4"][j])
        for i, y in enumerate(ys):
            ax[0].text(i + (j - 0.5) * width, y, f"{y:.0f}", ha="center",
                       va="bottom", fontsize=8)
    ax[0].set_xticks(range(len(caps)))
    ax[0].set_xticklabels([f"cap {c}" for c in caps])
    ax[0].set_ylabel("tokens / s")
    ax[0].set_title("Throughput")
    ax[0].legend(fontsize=8)

    for j, kind in enumerate(kinds):
        ys = [next(r["mean_latency"] for r in runs
                   if r["cap"] == c and r["name"] == kind) for c in caps]
        y2 = [next(r["p99_latency"] for r in runs
                   if r["cap"] == c and r["name"] == kind) for c in caps]
        ax[1].bar([i + (j - 0.5) * width for i in range(len(caps))], ys, width,
                  label=f"{kind} mean", color=["#d62728", "#1f77b4"][j])
        ax[1].plot([i + (j - 0.5) * width for i in range(len(caps))], y2, "k_",
                   ms=18, label="p99" if j == 0 else None)
    ax[1].set_xticks(range(len(caps)))
    ax[1].set_xticklabels([f"cap {c}" for c in caps])
    ax[1].set_ylabel("seconds")
    ax[1].set_title("End-to-end latency (bar = mean, tick = p99)")
    ax[1].legend(fontsize=8)

    best = res["timeline_cap"]
    for kind, color in [("static", "#d62728"), ("continuous", "#1f77b4")]:
        tl = next(r["timeline"] for r in runs
                  if r["cap"] == best and r["name"] == kind)
        ax[2].step([p[0] for p in tl], [p[1] for p in tl], where="post",
                   color=color, label=f"{kind}: live sequences")
    ax[2].axhline(best, ls=":", color="k", label=f"batch cap {best}")
    ax[2].set_xlabel("seconds")
    ax[2].set_ylabel("sequences being decoded")
    ax[2].set_title("Who is in the batch, over time")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.savefig(f"{OUT}/continuous_batching.png", dpi=130)
    log(f"   wrote {OUT}/continuous_batching.png")


def main():
    S.setup()
    t0 = time.time()
    w = S.Weights(S.SMALL)
    specs = make_workload(w)
    total_out = sum(s["out_len"] for s in specs)
    log(f"workload: {len(specs)} requests, prompts "
        f"{min(s['prompt_len'] for s in specs)}-{max(s['prompt_len'] for s in specs)} "
        f"tokens, outputs {min(s['out_len'] for s in specs)}-"
        f"{max(s['out_len'] for s in specs)} tokens ({total_out} in total), "
        f"arriving over {specs[-1]['arrival']:.1f} s")
    results["workload"] = [dict(rid=s["rid"], arrival=s["arrival"],
                                prompt_len=s["prompt_len"], out_len=s["out_len"])
                           for s in specs]

    runs = []
    for cap in CAPS:
        blocks = cap * 16 + 64
        for fn, kw in [(run_static, {}), (run_continuous, {})]:
            eng = S.Engine(w, num_blocks=blocks)
            r = fn(eng, w, specs, cap, **kw)
            runs.append(r)
            log(f"   cap {cap:2d} {r['name']:12s}: {r['makespan']:6.1f} s makespan, "
                f"{r['throughput']:6.1f} tok/s, mean latency {r['mean_latency']:5.1f} s, "
                f"p99 {r['p99_latency']:5.1f} s, TTFT {r['mean_ttft']:5.2f} s, "
                f"slot efficiency {r['slot_efficiency']:.0%}")
            del eng
    results["runs"] = runs
    results["timeline_cap"] = CAPS[1]

    for cap in CAPS:
        st = next(r for r in runs if r["cap"] == cap and r["name"] == "static")
        co = next(r for r in runs if r["cap"] == cap and r["name"] == "continuous")
        log(f"   -> cap {cap}: continuous batching is "
            f"{co['throughput'] / st['throughput']:.2f}x the throughput and "
            f"{st['mean_latency'] / co['mean_latency']:.2f}x lower mean latency; "
            f"static wasted {st['wasted_slot_steps']} slot-steps on requests that "
            f"had already finished")

    # Does admitting fewer requests per step help the users already running?
    log("\nadmission control: one new request per step vs all of them")
    eng = S.Engine(w, num_blocks=CAPS[1] * 16 + 64)
    limited = run_continuous(eng, w, specs, CAPS[1], max_admit=1)
    results["limited_admission"] = limited
    full = next(r for r in runs if r["cap"] == CAPS[1] and r["name"] == "continuous")
    log(f"   admit<=1: {limited['throughput']:.1f} tok/s, mean TTFT "
        f"{limited['mean_ttft']:.2f} s (vs {full['throughput']:.1f} tok/s, "
        f"{full['mean_ttft']:.2f} s when admitting everyone at once)")

    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("scheduler,cap,makespan_s,throughput_tok_s,mean_latency_s,"
                "p99_latency_s,mean_ttft_s,slot_efficiency,wasted_slot_steps\n")
        for r in runs:
            f.write(f"{r['name']},{r['cap']},{r['makespan']:.2f},"
                    f"{r['throughput']:.2f},{r['mean_latency']:.2f},"
                    f"{r['p99_latency']:.2f},{r['mean_ttft']:.3f},"
                    f"{r['slot_efficiency']:.4f},{r['wasted_slot_steps']}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
