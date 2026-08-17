"""Project 42 -- Stream-overlap audit.

Find the serial gap in a generation loop -- the CPU work that stalls the GPU
and the synchronisation that stalls the CPU -- pipeline it, and measure.

  python3 run.py          # full run, ~7 minutes
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "37-roofline-plot-for-your-engine"))

import torch  # noqa: E402

import enginelib as E  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CTX = 512
NTOK = 48
STOPS = ["\n\n", "</s>", "```", "User:"]


# ------------------------------------------------------------ the CPU side
class Detokeniser:
    """What a serving loop really does between two decode steps.

    Per sequence, per token: turn the id into text, append it to that
    sequence's buffer, and scan the tail for any stop string.  This is the
    work that the GPU is waiting for in a naive loop.
    """

    def __init__(self, tok, batch: int, work: int = 1):
        self.tok = tok
        self.buf = [""] * batch
        self.work = work          # repeat factor, to sweep the CPU cost
        self.hits = 0

    def step(self, ids) -> None:
        for _ in range(self.work):
            for s, i in enumerate(ids):
                piece = self.tok.convert_ids_to_tokens(int(i))
                self.buf[s] += piece
                tail = self.buf[s][-16:]
                for st in STOPS:
                    if st in tail:
                        self.hits += 1


class GrammarMask:
    """A second, much more expensive kind of per-step CPU work.

    Constrained decoding (JSON mode, a grammar) has to build a mask over the
    whole vocabulary for every sequence at every step, on the CPU, before the
    next token can be sampled.  Unlike detokenisation, this is real work: the
    vocabulary here is Qwen's 151,643 tokens.
    """

    def __init__(self, vocab: int, batch: int):
        import numpy as np
        self.np = np
        self.vocab = vocab
        self.batch = batch
        self.allowed = np.arange(0, vocab, 7)      # stand-in for "what the
        self.mask = np.zeros((batch, vocab), dtype=np.bool_)  # grammar permits"

    def step(self, ids) -> None:
        for s, i in enumerate(ids):
            self.mask[s].fill(False)
            self.mask[s][self.allowed] = True
            self.mask[s][int(i) % self.vocab] = True


def load_tokeniser():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


# ------------------------------------------------------------- the loops
def loop_serial(step, eng, det, B, n=NTOK) -> float:
    """The obvious implementation: run a step, wait for it, post-process it."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
        torch.cuda.synchronize()          # <-- the stall
        ids = eng.tok_host[:B].copy_(eng.tok[:B]).tolist()
        det.step(ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def loop_pipelined(step, eng, det, B, n=NTOK) -> float:
    """Issue step i+1 BEFORE post-processing step i, so they overlap."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    for i in range(n):
        torch.cuda.synchronize()
        ids = eng.tok_host[:B].copy_(eng.tok[:B]).tolist()
        if i + 1 < n:
            step()                        # the GPU starts token i+1 now
        det.step(ids)                     # ... while the CPU handles token i
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def loop_gpu_only(step, eng, det=None, B=1, n=NTOK) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def cpu_only(det, ids, n=NTOK) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        det.step(ids)
    return (time.perf_counter() - t0) / n * 1e3


# ------------------------------------------------------------- the copy
def copy_costs(eng, B: int) -> dict:
    """Getting the tokens off the device: pageable, pinned, and asynchronous."""
    pageable = torch.empty(B, dtype=torch.int32)
    pinned = torch.empty(B, dtype=torch.int32, pin_memory=True)
    big_page = torch.empty(B * 4096, dtype=torch.float32)
    big_pin = torch.empty(B * 4096, dtype=torch.float32, pin_memory=True)
    src = torch.empty(B * 4096, device=E.DEV)

    def t(fn, reps=200):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e6

    out = {
        "tokens_pageable_us": t(lambda: pageable.copy_(eng.tok[:B])),
        "tokens_pinned_us": t(lambda: pinned.copy_(eng.tok[:B])),
        "tokens_pinned_async_us": t(lambda: pinned.copy_(eng.tok[:B], non_blocking=True)),
        "logits_pageable_us": t(lambda: big_page.copy_(src), reps=50),
        "logits_pinned_us": t(lambda: big_pin.copy_(src), reps=50),
        "logits_pinned_async_us": t(lambda: big_pin.copy_(src, non_blocking=True), reps=50),
        "logits_mib": src.numel() * 4 / 2 ** 20,
    }
    return out


# ---------------------------------------------------------------- sections
def audit(eng, tok, B: int) -> list:
    """Four loops: eager or graphed, serial or pipelined."""
    eng.set_len(CTX - 1)
    g = E.Graph(lambda: eng.decode_step(B, advance=True))
    rows = []
    for launch, step in (("eager", lambda: eng.decode_step(B, advance=True)),
                         ("graph", g.replay)):
        for name, fn in (("serial", loop_serial), ("pipelined", loop_pipelined)):
            eng.set_len(CTX - 1)
            det = Detokeniser(tok, B)
            ms = fn(step, eng, det, B)
            rows.append({"launch": launch, "loop": name, "batch": B, "ms": ms,
                         "tok_s": B / ms * 1e3})
            print(f"   {launch:6s} {name:10s} B={B:3d}  {ms:7.3f} ms/step  "
                  f"{B/ms*1e3:8.0f} tok/s")
    eng.set_len(CTX - 1)
    det = Detokeniser(tok, B)
    gpu_only = loop_gpu_only(g.replay, eng, det, B)
    eng.set_len(CTX - 1)
    gpu_only_eager = loop_gpu_only(lambda: eng.decode_step(B, advance=True), eng, det, B)
    ids = [1000 + i for i in range(B)]
    cpu = cpu_only(Detokeniser(tok, B), ids)
    g.close()
    for r in rows:
        r["gpu_only_ms"] = gpu_only
        r["gpu_only_eager_ms"] = gpu_only_eager
        r["cpu_only_ms"] = cpu
    print(f"   floors: GPU alone (graph) {gpu_only:.3f} ms, "
          f"(eager) {gpu_only_eager:.3f} ms, CPU alone {cpu:.3f} ms")
    return rows


def sweep_cpu_work(eng, tok, B: int) -> list:
    """How much CPU work per token does it take before overlap matters?"""
    eng.set_len(CTX - 1)
    g = E.Graph(lambda: eng.decode_step(B, advance=True))
    ids = [1000 + i for i in range(B)]
    rows = []
    for work in (1, 4, 16, 64, 160):
        eng.set_len(CTX - 1)
        s = loop_serial(g.replay, eng, Detokeniser(tok, B, work), B)
        eng.set_len(CTX - 1)
        p = loop_pipelined(g.replay, eng, Detokeniser(tok, B, work), B)
        c = cpu_only(Detokeniser(tok, B, work), ids)
        eng.set_len(CTX - 1)
        gpu = loop_gpu_only(g.replay, eng, None, B)
        rows.append({"work": work, "batch": B, "cpu_ms": c, "gpu_ms": gpu,
                     "serial_ms": s, "pipelined_ms": p, "speedup": s / p,
                     "predicted": (gpu + c) / max(gpu, c)})
        print(f"   work x{work:4d}  cpu {c:7.3f} ms  gpu {gpu:6.3f} ms  "
              f"serial {s:7.3f} -> pipelined {p:7.3f}  ({s/p:.2f}x, "
              f"predicted {(gpu+c)/max(gpu,c):.2f}x)")
    g.close()
    return rows


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    rows = f["audit_b1"]
    labels = [f"{r['launch']}\n{r['loop']}" for r in rows]
    vals = [r["ms"] for r in rows]
    colors = ["#c0392b" if r["loop"] == "serial" else "#1f6f8b" for r in rows]
    a.bar(labels, vals, color=colors)
    a.axhline(rows[0]["gpu_only_ms"], color="0.35", ls="--", lw=1.2,
              label=f"GPU work alone ({rows[0]['gpu_only_ms']:.2f} ms)")
    for i, v in enumerate(vals):
        a.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
    a.set_ylabel("ms per token (batch 1)")
    a.set_title("Where the serial gap is")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y", lw=0.4)

    a = ax[1]
    rows = f["grammar"]
    idx = list(range(len(rows)))
    w = 0.27
    a.bar([i - w for i in idx], [r["gpu_ms"] for r in rows], w, color="0.6",
          label="GPU work alone")
    a.bar(idx, [r["serial_ms"] for r in rows], w, color="#c0392b", label="serial")
    a.bar([i + w for i in idx], [r["pipelined_ms"] for r in rows], w,
          color="#1f6f8b", label="pipelined")
    for i, r in enumerate(rows):
        a.text(i + w, r["pipelined_ms"] + 0.12, f"{r['speedup']:.2f}x",
               ha="center", fontsize=9)
    a.set_xticks(idx)
    a.set_xticklabels([f"batch {r['batch']}\n(mask {r['cpu_ms']:.2f} ms)" for r in rows])
    a.set_ylabel("ms per step")
    a.set_title("Constrained decoding: CPU work worth overlapping")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y", lw=0.4)

    a = ax[2]
    rows = f["sweep"]
    x = [r["cpu_ms"] / r["gpu_ms"] for r in rows]
    a.plot(x, [r["speedup"] for r in rows], "o-", color="#c0392b",
           label="measured")
    a.plot(x, [r["predicted"] for r in rows], "s--", color="0.45",
           label="(GPU + CPU) / max(GPU, CPU)")
    a.axvline(1.0, color="0.7", ls=":", lw=1)
    a.annotate("CPU work = GPU work", (1.0, 1.05), fontsize=8, color="0.4",
               rotation=90, va="bottom")
    a.set_xscale("log")
    a.set_xlabel("CPU post-processing per step / GPU work per step")
    a.set_ylabel("speedup from pipelining")
    a.set_title("Overlap is worth at most 2x, exactly when the two are equal")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    fig.suptitle("Project 42 - overlapping the CPU and the GPU in a generation loop",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "stream_overlap.png"), dpi=125)
    print("wrote", os.path.join(OUT, "stream_overlap.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    print("loading the Qwen2.5 tokeniser (real detokenisation, not a stand-in)")
    tok = load_tokeniser()
    cfg = E.Config()
    eng = E.Engine(cfg, max_batch=32, max_seq=CTX, max_tokens=32)
    eng.ctx_hint = CTX
    eng.tok_host = torch.empty(32, dtype=torch.int32)

    print("A. audit at batch 1")
    a1 = audit(eng, tok, 1)
    print("B. audit at batch 32")
    a32 = audit(eng, tok, 32)
    print("C. how expensive must the CPU work be? (batch 32)")
    sweep = sweep_cpu_work(eng, tok, 32)
    print("D. a realistic expensive case: constrained decoding")
    grammar = []
    for B in (1, 8, 32):
        eng.set_len(CTX - 1)
        g = E.Graph(lambda: eng.decode_step(B, advance=True))
        ids = [1000 + i for i in range(B)]
        gm = GrammarMask(151643, B)
        c = cpu_only(gm, ids)
        eng.set_len(CTX - 1)
        gpu = loop_gpu_only(g.replay, eng, None, B)
        eng.set_len(CTX - 1)
        ser = loop_serial(g.replay, eng, GrammarMask(151643, B), B)
        eng.set_len(CTX - 1)
        pipe = loop_pipelined(g.replay, eng, GrammarMask(151643, B), B)
        g.close()
        grammar.append({"batch": B, "cpu_ms": c, "gpu_ms": gpu,
                        "serial_ms": ser, "pipelined_ms": pipe,
                        "speedup": ser / pipe,
                        "predicted": (gpu + c) / max(gpu, c)})
        print(f"   B={B:3d}  mask {c:6.3f} ms  gpu {gpu:6.3f} ms  "
              f"serial {ser:7.3f} -> pipelined {pipe:7.3f}  ({ser/pipe:.2f}x)")

    print("E. the device-to-host copy")
    cc = copy_costs(eng, 32)
    for k, v in cc.items():
        print(f"   {k:26s} {v:8.2f}")

    f = {"device": E.device_info(), "ctx": CTX, "ntok": NTOK,
         "audit_b1": a1, "audit_b32": a32, "sweep": sweep,
         "grammar": grammar, "copy": cc}
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
