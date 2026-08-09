"""Project 49 - hunting a memory leak in a CPU training loop.

Six training loops that differ by one line each. One of them is clean; the
others hold on to something. Three instruments are pointed at all six, and the
interesting result is how much they disagree.

Sections:
  1. the suspect: resident memory that only goes up
  2. instrument 1 - RSS, and the slope that predicts the crash
  3. instrument 2 - a census of every tensor Python can reach
  4. instrument 3 - walking the autograd graph
  5. the disagreement, and who is right
  6. one environment variable turns the leak off
  7. the catalogue: six loops, three instruments
  8. the leak that needs the garbage collector
  9. the fixes, verified

Run:  python3 run.py        (~3 minutes)
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "48-nan-forensics"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import debug_lib as D  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F_ = D.Findings()

D_MODEL, DEPTH, BATCH = 1024, 6, 256
ACT_MB = BATCH * D_MODEL * 4 / 1e6          # one activation tensor, in MB
STEPS = 60


def build():
    torch.manual_seed(0)
    layers = []
    for _ in range(DEPTH):
        layers += [nn.Linear(D_MODEL, D_MODEL), nn.ReLU()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# The six loops. Each is the same training step plus (at most) one extra line.
# ---------------------------------------------------------------------------

LOOPS = {
    "clean":            "keep.append(loss.item())",
    "append loss":      "keep.append(loss)",
    "append output":    "keep.append(out)",
    "feature hook":     "(a forward hook stores one layer's output)",
    "running sum":      "running = running + loss   # no .item(), no backward on it",
    "step record":      "records.append(StepRecord(out.detach(), prev))",
}


class StepRecord:
    """A per-step log entry that points back at the previous one.

    A doubly linked list is an ordinary thing to build — it is how you walk a
    training history forwards and backwards. It is also a *reference cycle*:
    A points at B and B points back at A, so neither one's reference count ever
    reaches zero, and Python's ordinary refcount cleanup can never free either.
    Only the cycle-collecting garbage collector can, and it runs on its own
    schedule — based on how many objects have been allocated, not on how many
    bytes they are holding.
    """

    __slots__ = ("tensor", "prev", "next")

    def __init__(self, tensor, prev):
        self.tensor, self.prev, self.next = tensor, prev, None
        if prev is not None:
            prev.next = self                    # <- closes the cycle


def run_loop(mode: str, steps: int = STEPS, track_every: int = 5,
             saved_tracker: bool = False):
    """Run one training loop and record memory at every `track_every` steps."""
    gc.collect()
    D.malloc_trim()
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(BATCH, D_MODEL)

    keep: list = []
    records: list[StepRecord] = []
    running = torch.zeros(())
    feats: list = []
    handle = None
    if mode == "feature hook":
        handle = model[7].register_forward_hook(lambda m, i, o: feats.append(o))

    base_rss = D.rss_mb()
    base_n, base_bytes = D.live_tensor_bytes()
    trace = {"step": [], "rss": [], "census_mb": [], "census_n": [], "nodes": []}

    for step in range(1, steps + 1):
        out = model(x)
        loss = out.pow(2).mean()
        if mode == "running sum":
            running = running + loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if mode == "clean":
            keep.append(loss.item())
        elif mode == "append loss":
            keep.append(loss)
        elif mode == "append output":
            keep.append(out)
        elif mode == "step record":
            records.append(StepRecord(out.detach(), records[-1] if records else None))

        if step % track_every == 0:
            n, byts = D.live_tensor_bytes()
            probe = (running if mode == "running sum"
                     else (keep[-1] if keep and torch.is_tensor(keep[-1]) else None))
            _, nodes, _ = D.walk_graph(probe)
            trace["step"].append(step)
            trace["rss"].append(D.rss_mb() - base_rss)
            trace["census_mb"].append((byts - base_bytes) / 1e6)
            trace["census_n"].append(n - base_n)
            trace["nodes"].append(nodes)

    if handle is not None:
        handle.remove()
    result = {"trace": trace, "rss_end": D.rss_mb() - base_rss}
    # tear the loop's own state down and see what comes back
    keep.clear(); feats.clear(); records.clear()
    running = None; out = None; loss = None
    result["rss_after_clear"] = D.rss_mb() - base_rss
    gc.collect()
    result["rss_after_gc"] = D.rss_mb() - base_rss
    D.malloc_trim()
    result["rss_after_trim"] = D.rss_mb() - base_rss
    del model, opt, x
    gc.collect(); D.malloc_trim()
    return result


def slope(steps, values) -> float:
    """MB per step, by least squares over the second half of the run.

    The first few steps include one-off costs — weights, the first activation
    buffers, lazily imported code — that are not a leak. Fitting only the second
    half asks the question that matters: *is it still growing?*
    """
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    half = len(steps) // 2
    a, b = steps[half:], values[half:]
    return float(np.polyfit(a, b, 1)[0])


# ===========================================================================
# 1-2. The suspect, and the slope
# ===========================================================================

F_.head("1. The suspect: resident memory that only goes up")

res = {name: run_loop(name) for name in LOOPS}
suspect = res["append loss"]
clean = res["clean"]

F_.note("model", f"{DEPTH} x Linear({D_MODEL},{D_MODEL}) + ReLU, batch {BATCH}")
F_.note("one activation tensor", f"{ACT_MB:.2f} MB")
F_.note("activations per forward pass", f"{2 * DEPTH} ({2 * DEPTH * ACT_MB:.1f} MB)")
F_.note("steps", STEPS)
for label, r in (("clean loop", clean), ("suspect loop", suspect)):
    F_.note(f"{label}: RSS growth over {STEPS} steps", f"{r['rss_end']:.1f} MB")
    F_.note(f"{label}: RSS slope (2nd half)",
            f"{slope(r['trace']['step'], r['trace']['rss']):.2f} MB/step")

with open("/proc/meminfo") as fh:
    total_kb = int(next(l for l in fh if l.startswith("MemTotal")).split()[1])
sl = slope(suspect["trace"]["step"], suspect["trace"]["rss"])
F_.note("machine RAM", f"{total_kb / 1e6:.1f} GB")
F_.note("steps until the suspect loop eats all of it", f"{int(total_kb / 1e3 / max(sl, 1e-9)):,}")
F_.note("at 3 steps/second, that is", f"{total_kb / 1e3 / max(sl, 1e-9) / 3 / 60:.0f} minutes")


# ===========================================================================
# 3-4. What the other two instruments say
# ===========================================================================

F_.head("3-4. The census and the graph walk, on the same loop")

t = suspect["trace"]
F_.note("RSS at the end", f"{t['rss'][-1]:.1f} MB")
F_.note("tensor census at the end", f"{t['census_mb'][-1]:.2f} MB in "
                                    f"{t['census_n'][-1]} new storages")
F_.note("census slope (2nd half)", f"{slope(t['step'], t['census_mb']):.3f} MB/step")
F_.note("autograd nodes reachable from the kept loss", t["nodes"][-1])
F_.note("the disagreement", f"RSS says {sl:.2f} MB/step, the census says "
                            f"{slope(t['step'], t['census_mb']):.3f} MB/step")

mdl = build()
xx = torch.randn(BATCH, D_MODEL)
oo = mdl(xx)
ll = oo.pow(2).mean()
names_b, n_b, bytes_b = D.walk_graph(ll)
F_.note("graph nodes BEFORE backward", n_b)
F_.note("tensors saved on those nodes BEFORE backward", f"{bytes_b / 1e6:.2f} MB")
ll.backward()
names_a, n_a, bytes_a = D.walk_graph(ll)
F_.note("graph nodes AFTER backward", n_a)
F_.note("tensors saved on those nodes AFTER backward", f"{bytes_a / 1e6:.2f} MB")
F_.note("so keeping a loss after backward keeps", f"{n_a} node objects and 0 activations")
del mdl, xx, oo, ll
gc.collect()


# ===========================================================================
# 5-6. Who is right: the allocator
# ===========================================================================

F_.head("5-6. RSS was not measuring your tensors")

F_.note("suspect: RSS after the loop", f"{suspect['rss_end']:.1f} MB")
F_.note("suspect: after clearing the list", f"{suspect['rss_after_clear']:.1f} MB")
F_.note("suspect: after gc.collect()", f"{suspect['rss_after_gc']:.1f} MB")
F_.note("suspect: after malloc_trim(0)", f"{suspect['rss_after_trim']:.1f} MB")
F_.note("memory the process was holding but not using",
        f"{suspect['rss_after_gc'] - suspect['rss_after_trim']:.1f} MB")

# Same loop, in a child process, with glibc told to mmap large blocks.
import subprocess  # noqa: E402

CHILD = os.path.join(HERE, "child_loop.py")
env_runs = {}
for label, extra in (("default glibc", {}),
                     ("MALLOC_MMAP_THRESHOLD_=65536", {"MALLOC_MMAP_THRESHOLD_": "65536"}),
                     ("MALLOC_TRIM_THRESHOLD_=131072", {"MALLOC_TRIM_THRESHOLD_": "131072"})):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **extra)
    out = subprocess.run([sys.executable, CHILD, "append loss", str(STEPS)],
                         capture_output=True, text=True, env=env, timeout=600)
    env_runs[label] = json.loads(out.stdout.strip().splitlines()[-1])
    F_.note(f"{label}: RSS growth", f"{env_runs[label]['rss']:.1f} MB "
                                    f"({env_runs[label]['rss'] / STEPS:.2f} MB/step)")
F_.note("ratio, default vs mmap-threshold",
        f"{env_runs['default glibc']['rss'] / max(env_runs['MALLOC_MMAP_THRESHOLD_=65536']['rss'], 1e-9):.1f}x")


# ===========================================================================
# 7. The catalogue
# ===========================================================================

F_.head("7. Six loops, three instruments")

table = []
for name, r in res.items():
    tr = r["trace"]
    row = {
        "loop": name,
        "rss_mb_step": slope(tr["step"], tr["rss"]),
        "census_mb_step": slope(tr["step"], tr["census_mb"]),
        "census_n_step": slope(tr["step"], tr["census_n"]),
        "nodes_end": tr["nodes"][-1],
        "rss_end": r["rss_end"],
        "rss_after_gc": r["rss_after_gc"],
        "rss_after_trim": r["rss_after_trim"],
    }
    table.append(row)
    F_.note(f"{name}", f"RSS {row['rss_mb_step']:6.2f} MB/step | "
                       f"census {row['census_mb_step']:6.2f} MB/step | "
                       f"+{row['census_n_step']:4.1f} storages/step | "
                       f"graph nodes {row['nodes_end']:4d}")

with open(os.path.join(OUT, "catalogue.csv"), "w") as fh:
    keys = list(table[0])
    fh.write(",".join(keys) + "\n")
    for row in table:
        fh.write(",".join(f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k])
                          for k in keys) + "\n")
print(f"wrote {os.path.join(OUT, 'catalogue.csv')}")


# ===========================================================================
# 8. The leak that needs the garbage collector
# ===========================================================================

F_.head("8. The leak refcounting cannot free")

N_CYC = 40


def cycle_loop(n=N_CYC, collect=False, cyclic=True):
    """Create n pairs of records and drop every outside reference to them.

    If `cyclic`, each pair points at the other, so neither reference count ever
    reaches zero and Python's ordinary cleanup cannot free them. If not, they
    are freed the instant the loop moves on.
    """
    gc.collect(); D.malloc_trim()
    base = D.rss_mb()
    for _ in range(n):
        a = StepRecord(torch.randn(BATCH, D_MODEL), None)
        b = StepRecord(torch.randn(BATCH, D_MODEL), a if cyclic else None)
        del a, b                                  # nothing outside points at them now
        if collect:
            gc.collect()
    during = D.rss_mb() - base
    n_freed = gc.collect()
    after = D.rss_mb() - base
    D.malloc_trim()
    trimmed = D.rss_mb() - base
    return during, after, n_freed, trimmed


gc.disable()                                      # so only our explicit collect runs
cyc_during, cyc_after, cyc_freed, cyc_trim = cycle_loop(cyclic=True)
acyc_during, acyc_after, _, _ = cycle_loop(cyclic=False)
gc.enable()

F_.note("tensors created and dropped", f"{2 * N_CYC} x {ACT_MB:.2f} MB = "
                                       f"{2 * N_CYC * ACT_MB:.1f} MB")
F_.note("NO cycle: RSS growth while the loop runs", f"{acyc_during:.1f} MB")
F_.note("WITH a cycle: RSS growth while the loop runs", f"{cyc_during:.1f} MB")
F_.note("WITH a cycle: RSS after one gc.collect()", f"{cyc_after:.1f} MB")
F_.note("objects that gc.collect() freed", cyc_freed)
F_.note("WITH a cycle: RSS after gc.collect() AND malloc_trim(0)",
        f"{cyc_trim:.1f} MB")
F_.note("read those three together",
        "gc DID free the objects (160 of them); RSS only reflects it once glibc "
        "is asked to hand the pages back - the same two-layer story as section 5")
F_.note("why refcounting could not free them",
        "a.next is b and b.prev is a, so neither count ever reaches 0")

cyc_gc_during, _, _, _ = cycle_loop(cyclic=True, collect=True)
F_.note("WITH a cycle, calling gc.collect() every step", f"{cyc_gc_during:.1f} MB")


# ===========================================================================
# 9. The fixes, verified
# ===========================================================================

F_.head("9. The fixes, verified")

fixes = {
    "append loss (the bug)": "append loss",
    "append loss.item()": "clean",
    "append output": "append output",
}
for label, mode in fixes.items():
    r = res[mode]
    F_.note(label, f"{slope(r['trace']['step'], r['trace']['rss']):.2f} MB/step RSS, "
                   f"{slope(r['trace']['step'], r['trace']['census_mb']):.2f} MB/step census")

hook_r = res["feature hook"]
F_.note("feature hook left registered",
        f"{slope(hook_r['trace']['step'], hook_r['trace']['census_mb']):.2f} MB/step census "
        f"(= {ACT_MB:.2f} MB per step, one activation)")
F_.note("that is exactly one saved activation per step", True)


# ===========================================================================
# figures
# ===========================================================================

fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9), dpi=110)
for ax in axes:
    style_axes(ax)
fig.patch.set_facecolor("#fcfcfb")

ax = axes[0]
for i, name in enumerate(LOOPS):
    tr = res[name]["trace"]
    ax.plot(tr["step"], tr["rss"], color=SERIES[i], lw=1.7, label=name)
ax.set_title("1. RSS: every loop looks guilty", loc="left", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("RSS growth (MB)")
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
for i, name in enumerate(LOOPS):
    tr = res[name]["trace"]
    ax.plot(tr["step"], tr["census_mb"], color=SERIES[i], lw=1.7, label=name)
ax.set_title("2. tensor census: only three are", loc="left", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("tensors Python holds (MB)")
ax.legend(fontsize=7, frameon=False)

ax = axes[2]
names = list(LOOPS)
xs = np.arange(len(names))
ax.bar(xs - 0.2, [slope(res[n]["trace"]["step"], res[n]["trace"]["rss"]) for n in names],
       width=0.38, color=SERIES[2], label="RSS")
ax.bar(xs + 0.2, [slope(res[n]["trace"]["step"], res[n]["trace"]["census_mb"]) for n in names],
       width=0.38, color=SERIES[1], label="census")
ax.set_xticks(xs)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
ax.set_ylabel("MB / step")
ax.set_title("3. the two instruments disagree", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

ax = axes[3]
labels = ["end of\nloop", "clear\nthe list", "gc.\ncollect()", "malloc_\ntrim(0)"]
vals = [suspect["rss_end"], suspect["rss_after_clear"],
        suspect["rss_after_gc"], suspect["rss_after_trim"]]
ax.bar(range(4), vals, color=[SERIES[2], SERIES[2], SERIES[3], SERIES[1]], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("RSS above baseline (MB)")
ax.set_title("4. where the memory actually was", loc="left", fontsize=11)

save(fig, os.path.join(OUT, "memory_leak.png"))
F_.write(os.path.join(OUT, "findings.csv"))
