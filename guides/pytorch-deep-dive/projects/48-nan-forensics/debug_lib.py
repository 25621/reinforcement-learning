"""Shared helpers for Phase 9 (debugging hard problems) — projects 48-52.

Everything in this phase is about *instruments*: small pieces of code that make
an invisible failure visible. This file holds the ones more than one project
needs.

Three of them are worth reading before you use them, because each hides a trap
that cost a debugging session to find:

* :func:`walk_graph` — walking the autograd graph looks like a five-line
  breadth-first search. It is, but `id(node)` is **not** a stable identity for a
  `grad_fn`. Python builds a fresh wrapper object every time you touch
  `.grad_fn` or `.next_functions`, and if you drop your reference the address is
  immediately recycled for the next wrapper. A `seen` set keyed on `id()`
  therefore reports collisions that never happened and the walk stops after two
  nodes. The fix is one line: keep the wrappers alive in a list.

* :func:`live_tensor_bytes` — a census of every tensor Python can still reach.
  It sees tensors held by *your* variables and lists. It does **not** see
  tensors held by the autograd graph, because those live in C++ objects that
  Python's garbage collector never enumerates. Knowing what an instrument is
  blind to is half of knowing what its answer means.

* :func:`malloc_trim` — asks glibc to hand free memory back to the operating
  system. You need it because resident memory going up is not the same claim as
  "your program is holding tensors", and this is the cheapest way to tell those
  two apart. Project 49 is largely about that distinction.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import os
import statistics
import time

import torch

# ---------------------------------------------------------------------------
# findings bookkeeping — every number a project quotes goes through here
# ---------------------------------------------------------------------------


class Findings:
    """Collects (section, name, value) rows, prints them, writes one CSV.

    Every number that appears in a project README comes out of this list, so a
    reader can diff the page against `outputs/findings.csv` and catch us if the
    text drifts from the measurement.
    """

    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []
        self.section = "-"

    def head(self, title: str):
        self.section = title
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def note(self, name: str, value) -> None:
        if isinstance(value, float):
            value = f"{value:.6g}"
        self.rows.append((self.section, name, str(value)))
        print(f"    {name:<56} {value}")

    def write(self, path: str) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["section", "name", "value"])
            w.writerows(self.rows)
        print(f"\nwrote {path}  ({len(self.rows)} rows)")


# ---------------------------------------------------------------------------
# memory instruments
# ---------------------------------------------------------------------------

_LIBC = None


def malloc_trim() -> bool:
    """Ask glibc to return free heap pages to the OS. True if it did anything.

    C's `free()` gives memory back to the *allocator*, not to the kernel. glibc
    keeps it to serve the next allocation quickly. `malloc_trim(0)` walks the
    free lists and releases what it can. Linux + glibc only; a no-op elsewhere.
    """
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        return bool(_LIBC.malloc_trim(0))
    except OSError:
        return False


def rss_mb() -> float:
    """Resident set size: how much physical memory this process occupies, in MB.

    "Resident" means actually in RAM right now, as opposed to promised-but-never-
    touched address space. It is the number the OOM killer looks at, which is
    why we track it — and, as project 49 shows, it is *not* the number your
    tensors add up to.
    """
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def live_tensor_bytes() -> tuple[int, int]:
    """(number of distinct storages, total bytes) of every tensor Python can reach.

    Two tensors that are views of each other share one *storage* — one buffer of
    numbers. Counting tensors would double-count them, so we deduplicate on the
    storage's address.

    Blind spot, on purpose: tensors that only the autograd graph holds are owned
    by C++ and never appear in `gc.get_objects()`. See the module docstring.
    """
    seen, total = set(), 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                st = obj.untyped_storage()
                ptr = st.data_ptr()
                if ptr and ptr not in seen:
                    seen.add(ptr)
                    total += st.nbytes()
        except Exception:  # a half-built object can raise on isinstance
            pass
    return len(seen), total


# ---------------------------------------------------------------------------
# autograd-graph instruments
# ---------------------------------------------------------------------------


def walk_graph(tensor) -> tuple[list[str], int, int]:
    """Walk the autograd graph backwards from `tensor`.

    Returns (node type names, number of nodes, bytes of tensors still saved on
    those nodes).

    "Backwards" is the direction the gradient will travel: from the loss towards
    the inputs. Each node is one operation that was recorded during the forward
    pass, and `next_functions` are the nodes whose outputs it consumed.

    See the module docstring for why `keep` exists — without it this function
    silently returns 2 instead of 32.
    """
    if tensor is None or tensor.grad_fn is None:
        return [], 0, 0
    keep, seen, stack = [], set(), [tensor.grad_fn]
    names, storages, total = [], set(), 0
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        keep.append(fn)               # <- the whole trick: pin the wrapper
        seen.add(id(fn))
        names.append(type(fn).__name__)
        for attr in dir(fn):
            if not attr.startswith("_saved_"):
                continue
            try:
                val = getattr(fn, attr)
            except Exception:
                continue              # freed already: backward() ran
            if torch.is_tensor(val):
                st = val.untyped_storage()
                if st.data_ptr() and st.data_ptr() not in storages:
                    storages.add(st.data_ptr())
                    total += st.nbytes()
        for nxt, _ in getattr(fn, "next_functions", ()):
            if nxt is not None:
                stack.append(nxt)
    return names, len(names), total


# ---------------------------------------------------------------------------
# reproducibility instruments
# ---------------------------------------------------------------------------


def fingerprint(*tensors) -> str:
    """A short hash of the exact bits of some tensors.

    Comparing floats with `==` answers "are these close?". Hashing the raw bytes
    answers "are these the same number, bit for bit?" — which is the only
    question a determinism audit is allowed to ask.
    """
    h = hashlib.md5()
    for t in tensors:
        t = t.detach().contiguous() if torch.is_tensor(t) else torch.as_tensor(t)
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def state_fingerprint(module) -> str:
    """Bit-exact hash of a module's parameters and buffers, in a fixed order."""
    items = sorted(module.state_dict().items())
    return fingerprint(*[v for _, v in items])


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def interleaved(variants: dict, rounds: int = 5, calls: int = 20, warmup: int = 2):
    """Time several callables by rotating between them, round by round.

    This machine is shared. If you time variant A for ten seconds and then
    variant B for ten seconds, and something else on the box wakes up in the
    middle, the result is a fact about the neighbour and not about A or B.
    Rotating spreads any such interference over every variant.

    Returns {name: {"best": s, "median": s, "spread": max/min}}. Report `best`:
    the fastest run is the one with the fewest interruptions, so it is the
    cleanest estimate of the work itself.
    """
    names = list(variants)
    for name in names:
        for _ in range(warmup):
            variants[name]()
    samples: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(rounds):
        for name in names:
            t0 = time.perf_counter()
            for _ in range(calls):
                variants[name]()
            samples[name].append((time.perf_counter() - t0) / calls)
    out = {}
    for name in names:
        s = sorted(samples[name])
        out[name] = {"best": s[0], "median": statistics.median(s),
                     "spread": s[-1] / s[0] if s[0] else float("nan")}
    return out
