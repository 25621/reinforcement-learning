# Debug a Hang

---

> A distributed hang is almost always one rank waiting for a call the others already made.

---

## Key Insight

Every [rank](/shared/glossary/#rank) in a distributed job must take part in each [collective operation](/shared/glossary/#collective-operation); if one rank skips it, the others wait forever and the job hangs. Setting `TORCH_NCCL_BLOCKING_WAIT=1` and `NCCL_DEBUG=INFO` makes [NCCL](/shared/glossary/#nccl) report which collective is stuck and on which rank.

## Why This Matters

Hangs are the most common and most confusing distributed failure. Learning to read these signals turns a silent freeze into a quick, fixable mismatch.

---

**This is project 40.** Every project so far was careful to call each collective on
every rank. This one breaks that rule six different ways on purpose, catches what
comes out, and saves the messages verbatim in `outputs/`.

The backend here is [gloo](/shared/glossary/#gloo), not NCCL — there is no usable GPU
on this machine. The NCCL environment variables the guide mentions
(`TORCH_NCCL_BLOCKING_WAIT=1`, `NCCL_DEBUG=INFO`) therefore do nothing, and the section
below explains their gloo equivalents. The *[deadlock](/shared/glossary/#deadlock)s
themselves are identical* — they are a property of the collective contract, not of the
library implementing it.

What `run.py` finds:

- a hang looks different depending on whether the missing rank **exited** (`Connection
  closed by peer` — which reads like a network fault) or is **still working**
  (`Timed out waiting 12000ms for recv`)
- neither message names the guilty rank; `monitored_barrier` does, in one line:
  **`[Rank 0]: Rank 1 failed to pass monitoredBarrier in 6000 ms`**
- `faulthandler` names the guilty *line*: rank 0 stuck in `all_reduce`, rank 1
  somewhere else entirely
- calling two collectives in a **different order** is worse than a hang: rank 1 got
  **2 where the answer was 3** and **10 where the answer was 0**, and *then* timed out
- an `all_reduce` with **mismatched shapes** produces no error and no result — it hangs
  until killed — while DDP catches the same mistake at construction with a precise
  message
- and the classic uneven-data hang, with two fixes that both leave the replicas
  identical to **0.000e+00**

---

## Files

| file | what it is |
|---|---|
| `run.py` | all seven experiments |
| [`../36-two-gpu-ddp/dist_lib.py`](../36-two-gpu-ddp/dist_lib.py) | shared launcher (it also supplies the `PG_TIMEOUT_S` knob used here) |
| `outputs/hang_messages.txt` | the two hang messages, verbatim |
| `outputs/stacks_rank*.log.stack` | the per-rank stack dumps from section 4 |
| `outputs/findings.csv` | every number quoted on this page |

```bash
python3 run.py          # ~3.5 minutes: several sections deliberately wait for a timeout
```

---

## How to make a hang debuggable at all

A hang, by definition, does not end. Every experiment here therefore runs in real child
processes with **two** deadlines:

1. **A process-group timeout.** `init_process_group(..., timeout=timedelta(seconds=12))`
   turns an infinite wait into an exception with a message. The default is **10
   minutes**, which is a sensible production value and a terrible debugging one.
2. **A launcher timeout.** If the children ignore even that, the parent kills them and
   records "hung, killed after Ns".

Lowering the process-group timeout is the single most useful thing you can do to a job
that hangs. It converts "nothing happens, forever" into a stack trace at the moment of
failure.

---

## 1-2. One rank skips a collective

The bug, in the form you will actually write it:

```python
for step in range(3):
    if rank == 0:                 # "only rank 0 needs the validation metric"
        dist.all_reduce(t)
```

Rank 0 waits for partners who never come. What rank 0 *sees* depends on what rank 1
does next, and the difference matters:

| what rank 1 does after skipping | what rank 0 gets |
|---|---|
| exits the program | `RuntimeError: Connection closed by peer [192.168.200.105]:6575` |
| keeps working (sleeps 40 s) | `RuntimeError: Timed out waiting 12000ms for recv operation to complete` |
| (control) also calls `all_reduce` | finishes in 3.0 s, value 12.0 — correct |

The first row is a trap in itself. `Connection closed by peer` looks like a network
problem, and people go and check their firewall. It is not a network problem: it is
your own rank 1 having reached the end of `main()` while rank 0 was still waiting. The
second row — a plain timeout — is what the same bug looks like when the other rank is
still alive, which is the common case in a training loop.

What neither message tells you:

| question | answer |
|---|---|
| which rank failed to show up? | **no** — only that *this* rank waited |
| which collective was mismatched? | **no** — just "a recv timed out" |

Hence the next two sections.

> **Why not just print from every rank and compare?** You can, and it works for small
> jobs — but the print has to be *before* the collective that hangs, you need it on
> every rank, and with 256 ranks you are reading 256 interleaved logs. The two tools
> below get you the same answer without editing the training loop.

---

## 3. `monitored_barrier` names the rank

```python
dist.monitored_barrier(timeout=timedelta(seconds=6))
```

With rank 1 stuck in a long data load, rank 0 reports:

```
RuntimeError: [Rank 0]: Rank 1 failed to pass monitoredBarrier in 6000 ms
```

That is the whole diagnosis in one line. It works because `monitored_barrier` is not a
symmetric collective: rank 0 collects an acknowledgement from every other rank
individually, so when the deadline passes it knows exactly which ones never sent one.

Two caveats. It is **gloo-only** — there is no NCCL equivalent, because NCCL's
collectives do not have a per-rank acknowledgement to check. And it is a diagnostic,
not something to leave in a hot loop: it costs a round trip per rank. Drop it into the
suspicious part of your loop when a job is hanging, find the rank, take it out again.

---

## 4. `faulthandler`: what is every rank doing *right now*

```python
import faulthandler
faulthandler.dump_traceback_later(5, file=open(f"stack_rank{rank}.log", "w"))
```

Every rank promises to print its Python stack in 5 seconds' time, whatever it happens
to be doing. Run it, wait, then read the odd one out:

| rank | innermost frame |
|---|---|
| 0 | `torch/distributed/distributed_c10d.py", line 3014 in all_reduce` |
| 1 | `run.py` — in the training loop, not in a collective at all |

Rank 0 is inside `all_reduce`; rank 1 is not. There is your mismatch, with a line
number, and you did not have to guess where to put a `print`.

Two details that make this actually work:

- `faulthandler` prints **"most recent call first"**, so the frame you care about — the
  line that is stuck — is the **first** one, not the last. Read from the top.
- Write to a **file per rank**. Every rank dumping to a shared stderr produces
  interleaved, unreadable output.

On a NCCL job the equivalent, higher-powered tool is the *flight recorder*
(`TORCH_NCCL_DEBUG_INFO_TEMP_FILE` plus `TORCH_NCCL_DUMP_ON_TIMEOUT=1`), which dumps
every collective each rank has started and finished — so you can see not just where a
rank is, but that rank 3 is on collective #4197 while everyone else is on #4198.

---

## 5. The same collectives in a different order

Rank 0 does `all_reduce(a)` then `broadcast(b)`. Rank 1 does them the other way round.
Both ranks call both collectives, so the counts match. This is the failure everyone
expects to be a hang, and it is worse:

| | rank 1's result | expected |
|---|---|---|
| `all_reduce(a)` | **2** | 3 |
| `broadcast(b)` | **10** | 0 |
| then | `RuntimeError: Timed out waiting 20000ms for recv operation to complete` | |

Look at those two numbers before the timeout. `a` came back as 2, which is rank 1's own
un-reduced value; `b` came back as 10, which is rank 1's own value instead of rank 0's.
**The data was silently wrong before anything failed.** gloo matched rank 0's send
against rank 1's mismatched receive, produced garbage, and only got stuck later.

The lesson is uncomfortable and important: a hang is the *good* outcome. A mismatched
collective order can corrupt your data first and hang second, and if the run happens to
finish before the timeout you get a trained model with no error at all. Collectives
must be called in the **same order on every rank** — which in practice means: never put
one inside a branch that depends on rank, on the data, or on the loss.

---

## 6. Mismatched shapes

| | result |
|---|---|
| raw `all_reduce`, 16 floats on rank 0 vs 32 on rank 1 | **no error and no result — hung, killed after 30 s** |
| DDP with hidden size 64 on rank 0, 96 on rank 1, rank 1 | `RuntimeError: [1]: params[0] in this process with sizes [96, 32] appears not to match sizes of the same param in process 0.` |
| the same DDP run with `TORCH_DISTRIBUTED_DEBUG=DETAIL`, rank 1 | *the identical message* |

Three things worth separating here.

**A raw collective does not check shapes.** `all_reduce` on tensors of different sizes
does not raise; the ranks simply never agree on how many bytes are in flight, and the
job stops. If you are all-reducing something whose size depends on the data — a count
of examples, a variable-length buffer — check the sizes yourself first (all-reduce the
*length* before the payload).

**DDP does check**, at construction time, and its message is exactly what you want: the
parameter, the shape it found, and the rank it disagrees with. This catches the
realistic version of the bug — a model whose width is derived from the local dataset,
or a checkpoint loaded on one rank only.

**`TORCH_DISTRIBUTED_DEBUG=DETAIL` added nothing here**, because DDP's shape check runs
regardless. Its extra value is elsewhere: it logs unused parameters, records collective
call sequences, and turns some silent mismatches into errors. It is worth setting when
you are stuck, but it is not a magic switch — the honest result on this machine is that
the message with it was byte-for-byte the message without it.

---

## 7. Uneven data: the hang that arrives at the end of the epoch

Rank 0 has 6 batches, rank 1 has 4. This happens whenever the shards are not exactly
equal — a filtered dataset, a `drop_last=False` loader, a stream that ends when it ends.

DDP all-reduces gradients inside `backward()`, so rank 0's 5th backward pass waits for
a partner that has already left the loop:

| | result |
|---|---|
| no fix | rank 0 completed **4 of 6** steps, then `Timed out waiting 15000ms for recv` after 43.9 s |

Note *when* it broke: after four perfectly normal steps. A job like this trains happily
through the whole epoch and dies at the end of it — or at the end of epoch 37, if the
shard lengths only differ occasionally.

### Fix A: agree on the smallest count first

```python
t = torch.tensor([n_batches])
dist.all_reduce(t, op=dist.ReduceOp.MIN)     # one extra collective, once
n_batches = int(t.item())
```

| | value |
|---|---|
| steps completed, rank 0 / rank 1 | 4 / 4 |
| max \|weights(rank 0) − weights(rank 1)\| | **0.000e+00** |

Simple, predictable, and it throws away rank 0's two extra batches. That is usually the
right trade: two batches per epoch is nothing, and every rank now runs the same number
of steps.

### Fix B: `Join` — let the finished rank keep answering

```python
from torch.distributed.algorithms.join import Join
with Join([ddp]):
    for xb, yb in loader:
        ...
```

| | value |
|---|---|
| steps completed, rank 0 / rank 1 | **6 / 4** |
| max \|weights(rank 0) − weights(rank 1)\| | **0.000e+00** |

Rank 0 got to use all six batches. `Join` works by having the finished ranks keep
participating in the collectives with dummy (zero) contributions until the last rank is
done, so no one is ever left waiting. Use it when the data really is uneven and you do
not want to throw any away.

Both fixes leave the replicas bit-identical, which is the property that matters.

---

## The NCCL environment variables, and what to use instead

The guide names two, and they belong to NCCL:

| variable | what it does | works here? |
|---|---|---|
| `NCCL_DEBUG=INFO` | NCCL logs its topology, algorithm choice, and each collective | **no** — no NCCL on this machine |
| `TORCH_NCCL_BLOCKING_WAIT=1` | makes a NCCL collective raise on timeout instead of spinning | **no** |
| `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | dumps the flight recorder: every collective each rank started | **no** |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | backend-independent extra checks and logging | yes (added nothing here) |
| `init_process_group(timeout=...)` | turns an infinite wait into a message | **yes — the one that mattered** |
| `dist.monitored_barrier()` | names the rank that did not arrive | **yes**, gloo only |
| `faulthandler.dump_traceback_later` | the stuck line, per rank | **yes**, backend-independent |

On a GPU cluster, set the NCCL ones first — they are purpose-built. Everywhere else,
the last three do the same job.

---

## What to remember

1. **Every rank, every collective, same order.** A collective inside `if rank == 0:` or
   `if loss > threshold:` is the bug.
2. **Lower the process-group timeout when debugging.** Ten minutes of silence teaches
   you nothing; twelve seconds and a stack trace teaches you everything.
3. **`Connection closed by peer` is usually not a network fault** — it is another rank
   that already exited.
4. **`monitored_barrier` names the rank; `faulthandler` names the line.** Two tools,
   ten seconds, no code changes.
5. **A mismatched *order* can corrupt data before it hangs.** The hang is the lucky
   outcome.
6. **Raw collectives do not check shapes; DDP does.** All-reduce a length before you
   all-reduce a variable-length payload.
7. **Uneven shards hang at the end of the epoch.** Trim to the minimum, or use `Join`.

---

*Next: [project 41](../41-multi-node-setup/README.md) — the same job, spread across two
machines, where a new set of things can fail before training even starts.*
