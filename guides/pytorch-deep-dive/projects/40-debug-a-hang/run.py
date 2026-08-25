"""Project 40 — Make a distributed job hang on purpose, then find out how.

Run:  python3 run.py           (~3 minutes; several sections wait for a timeout)

Every experiment runs in real child processes. When one is supposed to hang, we
give it a short process-group timeout so it dies with a message instead of
freezing this script forever. The captured messages are saved verbatim under
outputs/ — reading them is the point of the project.

Sections
  1. the classic: one rank skips a collective
  2. what the timeout message actually tells you (and what it does not)
  3. monitored_barrier: gloo names the rank that never arrived
  4. faulthandler: the stack trace of every rank, on demand
  5. mismatched *order* of collectives - worse than a hang
  6. mismatched *shapes*, and what TORCH_DISTRIBUTED_DEBUG=DETAIL adds
  7. uneven data: the hang that only happens on the last epoch, and two fixes
"""

from __future__ import annotations

import csv
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "36-two-gpu-ddp"))
import dist_lib as D  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<50} {value}")


def capture_to(path):
    """Redirect this process's file descriptors 1 and 2 into a file.

    Not `contextlib.redirect_stderr` — the messages we want are printed by
    PyTorch's C++ layer straight to the file descriptor, and Python-level
    redirection never sees them.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        f = open(path, "w", buffering=1)
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)
            f.close()

    return _ctx()


def run_maybe_hanging(fn, world, args=(), timeout=45.0, pg_timeout=10.0, threads=1):
    """Launch, and report whether it finished or had to be killed."""
    t0 = time.perf_counter()
    try:
        res = D.launch(fn, world, threads=threads, args=args, timeout=timeout,
                       env={"PG_TIMEOUT_S": pg_timeout}, raise_on_error=False)
        dt = time.perf_counter() - t0
        errs = [r.get("__error__") for r in res
                if isinstance(r, dict) and "__error__" in r]
        return {"finished": True, "secs": dt, "results": res, "errors": errs}
    except RuntimeError as exc:
        return {"finished": False, "secs": time.perf_counter() - t0,
                "results": [], "errors": [str(exc)[:400]]}


# ---------------------------------------------------------------------------
# 1 + 2. one rank skips a collective
# ---------------------------------------------------------------------------

def w_skip(rank, world, mode):
    """The bug in one line: a collective inside a rank-dependent `if`.

    mode='all'   every rank reduces - correct
    mode='exit'  rank 1 skips the reduction and leaves the program
    mode='busy'  rank 1 skips the reduction and carries on working, which is
                 what really happens in a training loop
    """
    log = os.path.join(OUT, f"skip_{mode}_rank{rank}.log")
    t = torch.ones(1024) * (rank + 1)
    with capture_to(log):
        try:
            for step in range(3):
                # rank 0 has "extra work" - say, computing a validation metric
                # that needs a mean over all ranks. It reduces. Nobody else
                # does. This is the whole bug.
                if mode == "all" or rank == 0:
                    dist.all_reduce(t)
            if mode == "busy" and rank != 0:
                time.sleep(40)          # rank 1 has moved on to the next epoch
            err = ""
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
    return {"err": err, "value": float(t[0])}


def section_1_2():
    print("\n[1] one rank calls all_reduce, the others do not")
    good = run_maybe_hanging(w_skip, 2, args=("all",), pg_timeout=10, timeout=60)
    gone = run_maybe_hanging(w_skip, 2, args=("exit",), pg_timeout=10, timeout=60)
    busy = run_maybe_hanging(w_skip, 2, args=("busy",), pg_timeout=12, timeout=90)

    record("skip", "all ranks reduce: finished?",
           f"{good['finished']} in {good['secs']:.1f}s")
    record("skip", "all ranks reduce: value on rank 0 after 3 reductions",
           good["results"][0]["value"] if good["finished"] else "-",
           "1+2=3, doubled twice more = 12")

    def first_err(r):
        for x in r["results"]:
            if isinstance(x, dict) and x.get("err"):
                return x["err"]
        return (r["errors"] or ["(killed)"])[0]

    record("skip", "rank 1 skips it and exits: rank 0 gets",
           first_err(gone)[:150],
           "the peer's socket closed, so this looks like a network fault")
    record("skip", "rank 1 skips it and keeps working: rank 0 gets",
           first_err(busy)[:150])
    record("skip", "  how long rank 0 waited", f"{busy['secs']:.1f}s",
           "exactly the process-group timeout we set (12s); the default is 10 min")

    print("\n[2] what the message does and does not tell you")
    with open(os.path.join(OUT, "hang_messages.txt"), "w") as f:
        f.write("rank 1 skipped the collective and exited:\n  "
                + first_err(gone) + "\n\nrank 1 skipped it and kept working:\n  "
                + first_err(busy) + "\n")
    record("skip", "does the message name the rank that failed to show up?",
           "no - it only says THIS rank waited",
           "which is why sections 3 and 4 exist")
    record("skip", "does it name the collective that was mismatched?",
           "no - just a recv that timed out")
    return busy, good


# ---------------------------------------------------------------------------
# 3. monitored_barrier
# ---------------------------------------------------------------------------

def w_monitored(rank, world):
    """`monitored_barrier` is a gloo-only barrier that reports *who* is missing.

    Rank 0 collects an acknowledgement from every other rank, so when the
    timeout fires it knows exactly which ranks never sent one.
    """
    log = os.path.join(OUT, f"monitored_rank{rank}.log")
    err = ""
    with capture_to(log):
        try:
            if rank == 1:
                time.sleep(30)            # rank 1 is stuck in a long data load
            dist.monitored_barrier(timeout=__import__("datetime").timedelta(seconds=6))
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"err": err}


def section_3():
    print("\n[3] monitored_barrier names the guilty rank")
    r = run_maybe_hanging(w_monitored, 3, timeout=90, pg_timeout=60)
    msgs = [x.get("err", "") for x in r["results"] if isinstance(x, dict)]
    named = next((m for m in msgs if "rank" in m.lower() and m), "")
    record("monitored", "rank 0's error", (named or "(none)")[:200])
    record("monitored", "does it name the missing rank?",
           "yes" if "1" in named else "unclear")
    return r


# ---------------------------------------------------------------------------
# 4. faulthandler
# ---------------------------------------------------------------------------

def w_faulthandler(rank, world):
    """Every rank promises to dump its Python stack after N seconds.

    This is the single most useful trick for a hang you cannot reproduce
    quickly: you get a stack per rank, and the odd one out is your bug. In a
    real job you would send SIGABRT instead, or set
    `TORCH_NCCL_DUMP_ON_TIMEOUT=1` and read the flight recorder.
    """
    import faulthandler

    log = os.path.join(OUT, f"stacks_rank{rank}.log")
    err = ""
    with capture_to(log):
        f = open(log + ".stack", "w")
        faulthandler.dump_traceback_later(5, file=f, exit=False)
        t = torch.ones(64)
        try:
            if rank == 0:
                dist.all_reduce(t)        # rank 0 waits here
            else:
                time.sleep(8)             # rank 1 is busy elsewhere
                dist.all_reduce(t)
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
        faulthandler.cancel_dump_traceback_later()
        f.close()
    text = open(log + ".stack").read() if os.path.exists(log + ".stack") else ""
    return {"err": err, "stack": text[-1200:]}


def section_4():
    print("\n[4] faulthandler: what is each rank doing right now?")
    r = run_maybe_hanging(w_faulthandler, 2, timeout=90, pg_timeout=60)
    for i, res in enumerate(r["results"]):
        stack = res.get("stack", "") if isinstance(res, dict) else ""
        # faulthandler prints "most recent call first", so the deepest frame -
        # the line that is actually stuck - is the FIRST one.
        lines = [ln.strip() for ln in stack.splitlines() if ln.strip().startswith("File")]
        record("stacks", f"rank {i}, innermost frame",
               lines[0][:120] if lines else "(none)")
    with open(os.path.join(OUT, "stacks_rank0.log.stack"), "a"):
        pass
    return r


# ---------------------------------------------------------------------------
# 5. mismatched order
# ---------------------------------------------------------------------------

def w_order(rank, world, swap):
    """Rank 1 does the same two collectives in the opposite order."""
    log = os.path.join(OUT, f"order_rank{rank}.log")
    a = torch.ones(8) * (rank + 1)          # to be all-reduced -> expect 3
    b = torch.zeros(8) + rank * 10          # to be broadcast from rank 0 -> expect 0
    err = ""
    with capture_to(log):
        try:
            if swap and rank == 1:
                dist.broadcast(b, src=0)
                dist.all_reduce(a)
            else:
                dist.all_reduce(a)
                dist.broadcast(b, src=0)
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {"a": float(a[0]), "b": float(b[0]), "err": err}


def section_5():
    print("\n[5] the same two collectives, in a different order on rank 1")
    ok = run_maybe_hanging(w_order, 2, args=(False,), pg_timeout=10, timeout=60)
    swapped = run_maybe_hanging(w_order, 2, args=(True,), pg_timeout=20, timeout=60)
    record("order", "correct order: all_reduce result / broadcast result",
           f"{ok['results'][0]['a']:.0f} / {ok['results'][1]['b']:.0f}"
           if ok["finished"] else "hung")
    if swapped["finished"]:
        r1 = swapped["results"][1]
        record("order", "swapped order: finished?", f"yes in {swapped['secs']:.1f}s")
        record("order", "swapped order: rank 1 all_reduce result (expect 3)",
               f"{r1['a']:.0f}")
        record("order", "swapped order: rank 1 broadcast result (expect 0)",
               f"{r1['b']:.0f}")
        record("order", "swapped order: any exception?", r1["err"] or "none")
    else:
        record("order", "swapped order", f"HUNG, killed after {swapped['secs']:.1f}s")
    return ok, swapped


# ---------------------------------------------------------------------------
# 6. mismatched shapes
# ---------------------------------------------------------------------------

def w_shape(rank, world, detail):
    log = os.path.join(OUT, f"shape_rank{rank}{'_detail' if detail else ''}.log")
    n = 16 if rank == 0 else 32              # different sizes on purpose
    t = torch.ones(n) * (rank + 1)
    err = ""
    with capture_to(log):
        try:
            dist.all_reduce(t)
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"err": err, "sum": float(t.sum()), "n": n}


def w_ddp_mismatch(rank, world):
    """A model whose shape depends on the rank — a real and common accident,
    e.g. a hidden size derived from `len(dataset)` when the shards differ."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    log = os.path.join(OUT, f"ddpshape_rank{rank}.log")
    hidden = 64 if rank == 0 else 96
    err = ""
    with capture_to(log):
        try:
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(), nn.Linear(hidden, 8))
            ddp = DDP(m)
            F.cross_entropy(ddp(torch.randn(4, 32)), torch.zeros(4, dtype=torch.long)).backward()
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:400]}"
    return {"err": err, "hidden": hidden}


def section_6():
    print("\n[6] mismatched shapes")
    raw = run_maybe_hanging(w_shape, 2, args=(False,), pg_timeout=12, timeout=30)
    if raw["finished"]:
        e = next((x.get("err") for x in raw["results"] if x.get("err")), "")
        record("shape", "raw all_reduce, 16 floats vs 32 floats",
               (e or "no error - it 'worked'")[:170])
    else:
        record("shape", "raw all_reduce, 16 floats vs 32 floats",
               f"no error and no result: hung, killed after {raw['secs']:.0f}s",
               "a size mismatch does not raise - the ranks simply never agree")

    plain = run_maybe_hanging(w_ddp_mismatch, 2, pg_timeout=20, timeout=60)
    for i, x in enumerate(plain["results"]):
        record("shape", f"DDP, different hidden size per rank: rank {i}",
               (x.get("err") or "no error")[:180] if isinstance(x, dict) else "-")

    t0 = time.perf_counter()
    try:
        res = D.launch(w_ddp_mismatch, 2, threads=1, timeout=60,
                       env={"PG_TIMEOUT_S": 20, "TORCH_DISTRIBUTED_DEBUG": "DETAIL"},
                       raise_on_error=False)
        for i, x in enumerate(res):
            record("shape", f"the same with TORCH_DISTRIBUTED_DEBUG=DETAIL: rank {i}",
                   (x.get("err") or "no error")[:220] if isinstance(x, dict) else "-")
    except RuntimeError:
        record("shape", "the same run with TORCH_DISTRIBUTED_DEBUG=DETAIL",
               f"hung, killed after {time.perf_counter() - t0:.0f}s")
    return raw, plain


# ---------------------------------------------------------------------------
# 7. uneven data
# ---------------------------------------------------------------------------

def w_uneven(rank, world, fix):
    """Rank 0 has 6 batches, rank 1 has 4. DDP all-reduces once per backward,
    so rank 0's 5th backward waits for a partner that has already left the loop.

    fix='none'  -> hangs
    fix='trim'  -> agree on the smallest number of batches first (one all-reduce)
    fix='join'  -> torch.distributed.algorithms.Join: the finished rank keeps
                   answering collectives with dummy work until the others finish
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    n_batches = 6 if rank == 0 else 4
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
    x, y = D.make_teacher_data(64 * 8, 32, 8, seed=2)

    if fix == "trim":
        t = torch.tensor([n_batches])
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_batches = int(t.item())

    log = os.path.join(OUT, f"uneven_{fix}_rank{rank}.log")
    err, done = "", 0
    with capture_to(log):
        try:
            if fix == "join":
                from torch.distributed.algorithms.join import Join
                ctx = Join([ddp])
            else:
                import contextlib
                ctx = contextlib.nullcontext()
            with ctx:
                for s in range(n_batches):
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(ddp(x[s * 8:(s + 1) * 8]), y[s * 8:(s + 1) * 8]).backward()
                    opt.step()
                    done += 1
            if fix == "none" and n_batches == 4:
                # the rank that runs out of data does NOT exit: it moves on to
                # validation, logging, the next epoch. Meanwhile rank 0 is still
                # waiting inside backward() for a gradient all-reduce partner.
                time.sleep(40)
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
    w = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return {"err": err, "steps_done": done, "n_batches": n_batches,
            "wsum": float(w.sum())}


def section_7():
    print("\n[7] uneven number of batches per rank")
    none = run_maybe_hanging(w_uneven, 2, args=("none",), pg_timeout=15, timeout=70)
    trim = run_maybe_hanging(w_uneven, 2, args=("trim",), pg_timeout=15, timeout=70)
    join = run_maybe_hanging(w_uneven, 2, args=("join",), pg_timeout=15, timeout=70)

    record("uneven", "no fix: finished?", f"{none['finished']} after {none['secs']:.1f}s")
    if none["finished"]:
        record("uneven", "no fix: steps completed rank0 / rank1",
               f"{none['results'][0]['steps_done']} / {none['results'][1]['steps_done']}")
        record("uneven", "no fix: error on rank 0",
               (none["results"][0]["err"] or "none")[:170])
    record("uneven", "trim to the minimum: steps rank0 / rank1",
           f"{trim['results'][0]['steps_done']} / {trim['results'][1]['steps_done']}"
           if trim["finished"] else "hung")
    record("uneven", "trim: are the replicas still identical?",
           f"{abs(trim['results'][0]['wsum'] - trim['results'][1]['wsum']):.3e}"
           if trim["finished"] else "-")
    record("uneven", "Join(): steps rank0 / rank1",
           f"{join['results'][0]['steps_done']} / {join['results'][1]['steps_done']}"
           if join["finished"] else "hung")
    record("uneven", "Join: are the replicas still identical?",
           f"{abs(join['results'][0]['wsum'] - join['results'][1]['wsum']):.3e}"
           if join["finished"] else "-")
    return none, trim, join


def main():
    t0 = time.time()
    section_1_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader(); w.writerows(FINDINGS)
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
