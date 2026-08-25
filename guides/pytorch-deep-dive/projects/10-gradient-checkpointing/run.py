"""Project 10 — Gradient checkpointing.

Throw the activations away during the forward pass, recompute them during the
backward pass, and trade compute for memory.

We build checkpointing ourselves first, as a `torch.autograd.Function` -- it is
about fifteen lines -- because a custom Function is the only way to make the
memory actually measurable here, and because writing it is the fastest way to
understand it.

  1. what a forward pass actually keeps, and how much of it is even activations
  2. checkpointing in fifteen lines; gradients IDENTICAL to the uncheckpointed
     run and to torch's implementation -- this is not an approximation
  3. a segment sweep: peak activation memory and step time, against sqrt(L)
  4. dropout: our naive version gets it wrong, and how torch fixes it
  5. use_reentrant, and the silent no-op it causes

Runs in about 45 seconds on CPU. No downloads.
"""

import csv
import os
import sys
import time
import weakref

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint as torch_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(4)
DEPTH, DIM, BATCH = 48, 512, 256

FINDINGS = {}


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# checkpointing, from scratch
# =========================================================================
class CheckpointNaive(Function):
    """The obvious fifteen lines. It is also subtly, silently broken -- see
    section 5. Kept because the bug it has is the bug the real API had."""

    @staticmethod
    def forward(ctx, fn, rng_state, x):
        ctx.fn, ctx.rng_state = fn, rng_state
        ctx.save_for_backward(x)
        with torch.no_grad():          # <- the whole point: record nothing
            return fn(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        if ctx.rng_state is not None:
            torch.set_rng_state(ctx.rng_state)
        with torch.enable_grad():      # now DO record a graph, for this piece only
            y = ctx.fn(x)
        torch.autograd.backward(y, grad_out)   # accumulates into every leaf
        return None, None, x.grad      # one return per forward argument


class Checkpoint(Function):
    """The same idea, with the segment's parameters passed in as real inputs."""

    @staticmethod
    def forward(ctx, fn, rng_state, x, *params):
        # Passing `params` looks pointless -- fn already closes over the very
        # same Parameter objects and can reach them without our help. It is not
        # pointless: autograd only creates a graph node for this Function if at
        # least one TENSOR ARGUMENT requires grad. `fn` is a Python closure and
        # `x` is often plain data, so without the params in the argument list
        # this call is invisible to the graph and the whole segment silently
        # trains nothing. Section 5 measures exactly that.
        ctx.fn, ctx.rng_state = fn, rng_state
        ctx.save_for_backward(x, *params)
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, *params = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        if ctx.rng_state is not None:
            torch.set_rng_state(ctx.rng_state)     # replay the same randomness
        with torch.enable_grad():
            y = ctx.fn(x)
        # torch.autograd.grad RETURNS the gradients instead of accumulating them
        # into .grad. That matters: we hand them back to the engine, which does
        # the accumulating. Calling .backward() here as well would count every
        # gradient twice.
        grads = torch.autograd.grad(y, [x] + list(params), grad_out,
                                    allow_unused=True)
        return (None, None) + tuple(grads)


def our_checkpoint(fn, x, params, preserve_rng=False, naive=False):
    state = torch.get_rng_state() if preserve_rng else None
    if naive:
        return CheckpointNaive.apply(fn, state, x)
    return Checkpoint.apply(fn, state, x, *params)


# =========================================================================
# the model
# =========================================================================
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


class DeepNet(nn.Module):
    """`segments` = how many groups the blocks are split into.

    0        -> no checkpointing (the baseline)
    1        -> one checkpoint around everything
    DEPTH    -> checkpoint every single block
    """

    def __init__(self, depth=DEPTH, d=DIM, segments=0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.head = nn.Linear(d, 1)
        self.segments = segments

    def _run(self, blocks, x):
        for b in blocks:
            x = b(x)
            if self.drop is not None:
                x = self.drop(x)
        return x

    def forward(self, x, impl="ours", preserve_rng=False, use_reentrant=False,
                naive=False):
        if self.segments == 0:
            x = self._run(self.blocks, x)
        else:
            per = max(1, len(self.blocks) // self.segments)
            for i in range(0, len(self.blocks), per):
                g = self.blocks[i:i + per]
                fn = lambda t, g=g: self._run(g, t)
                if impl == "ours":
                    params = [p for b in g for p in b.parameters()]
                    x = our_checkpoint(fn, x, params,
                                       preserve_rng=preserve_rng, naive=naive)
                else:
                    x = torch_checkpoint(fn, x, use_reentrant=use_reentrant,
                                         preserve_rng_state=preserve_rng)
        return self.head(x).mean()


# =========================================================================
# measuring: a live byte counter for whatever autograd is holding
# =========================================================================
class Tracker:
    """Counts bytes autograd is holding for backward, as they come and go.

    `saved_tensors_hooks` fires the moment a tensor is stashed for backward. We
    wrap it in a tiny `Holder`; when the graph node owning the Holder is
    released, CPython's refcounting drops it at once and the weakref finalizer
    fires, so the counter goes back DOWN. Running forward and backward inside
    the context therefore traces the real curve, including the tensors a
    recomputed segment saves and then frees.

    Parameters are excluded by data_ptr. A Linear's backward does "save" its
    weight, but the weight was already resident and checkpointing cannot move
    it -- counting it would flatter every configuration equally.
    """

    class Holder:
        __slots__ = ("t", "__weakref__")

        def __init__(self, t):
            self.t = t

    def __init__(self, param_ptrs):
        self.param_ptrs = param_ptrs
        self.live = self.peak = self.after_forward = 0
        self.total_packed = self.n_packed = self.n_params = 0

    def pack(self, t):
        if t.data_ptr() in self.param_ptrs:
            self.n_params += 1
            return t
        nb = t.numel() * t.element_size()
        h = Tracker.Holder(t)
        self.live += nb
        self.n_packed += 1
        self.total_packed += nb
        self.peak = max(self.peak, self.live)
        weakref.finalize(h, self._release, nb)
        return h

    def _release(self, nb):
        self.live -= nb

    def unpack(self, h):
        return h.t if isinstance(h, Tracker.Holder) else h


def measure_memory(net, x, **fwd):
    ptrs = {p.data_ptr() for p in net.parameters()}
    tr = Tracker(ptrs)
    net.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(tr.pack, tr.unpack):
        out = net(x, **fwd)
        tr.after_forward = tr.live
        out.backward()
    return {"held_mb": tr.after_forward / 1e6, "peak_mb": tr.peak / 1e6,
            "packed_mb": tr.total_packed / 1e6, "n_packed": tr.n_packed,
            "n_params_saved": tr.n_params}


def time_step(net, x, reps=3, warmup=1, **fwd):
    """Best of `reps`. Best, not mean: the fastest run is the one with the
    fewest interruptions from everything else on the machine, so it is the
    cleanest estimate of the work itself."""
    for _ in range(warmup):
        net.zero_grad(set_to_none=True)
        net(x, **fwd).backward()
    best = 1e9
    for _ in range(reps):
        net.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        net(x, **fwd).backward()
        best = min(best, time.perf_counter() - t0)
    return best


# =========================================================================
# 1. what a forward pass keeps
# =========================================================================
def what_is_kept():
    print("=" * 78)
    print("1. WHAT ONE FORWARD PASS LEAVES BEHIND")
    print("=" * 78)
    torch.manual_seed(0)
    net = DeepNet(segments=0)
    x = torch.randn(BATCH, DIM)
    r = measure_memory(net, x)
    r["seconds"] = time_step(net, x, reps=3)
    per_act = BATCH * DIM * 4
    pbytes = sum(p.numel() * p.element_size() for p in net.parameters())
    print(f"  {DEPTH} blocks of Linear({DIM},{DIM}) + GELU, batch {BATCH}")
    print(f"  one activation tensor = {BATCH} x {DIM} x 4 B = {per_act / 1e6:.2f} MB\n")
    print(f"  activations autograd held : {r['held_mb']:.1f} MB "
          f"in {r['n_packed']} tensors ({r['n_packed'] / DEPTH:.0f} per block)")
    print(f"  parameters it ALSO saved  : {r['n_params_saved']} references, "
          f"{pbytes / 1e6:.1f} MB -- already resident, not counted again")
    print(f"  step time                 : {r['seconds'] * 1000:.0f} ms")
    rec("baseline_activation_mb", round(r["held_mb"], 2))
    rec("baseline_activation_tensors", r["n_packed"])
    rec("baseline_param_saves", r["n_params_saved"])
    rec("param_mb", round(pbytes / 1e6, 2))
    rec("activation_mb_each", round(per_act / 1e6, 3))

    print("\n  Two activation tensors per block, and neither is optional. Linear's")
    print("  backward needs its INPUT to compute dW; GELU's backward needs ITS")
    print("  input to compute the local slope. The chain rule genuinely wants")
    print("  both numbers. Checkpointing does not make them unnecessary -- it")
    print("  makes them arrive later, from a recomputation, instead of sitting")
    print("  in memory the whole time.")
    print(f"\n  Activations ({r['held_mb']:.0f} MB) and weights ({pbytes / 1e6:.0f} MB) are neck and neck at")
    print("  this size -- but only the activations grow when you raise the batch")
    print("  size or add layers. Double the batch and the weights do not move")
    print("  while the activations double. That is why 'lower the batch size' is")
    print("  the reflex answer to an out-of-memory error, and checkpointing is")
    print("  how you avoid having to.")
    return r


# =========================================================================
# 2. does our version agree with torch's?
# =========================================================================
def agreement():
    print("\n" + "=" * 78)
    print("2. FIFTEEN LINES vs torch.utils.checkpoint")
    print("=" * 78)
    grads = {}
    for tag, kw in [("plain", dict(segments=0)),
                    ("ours", dict(segments=6, impl="ours")),
                    ("torch", dict(segments=6, impl="torch"))]:
        segs = kw.pop("segments")
        torch.manual_seed(0)
        net = DeepNet(depth=12, d=128, segments=segs)
        torch.manual_seed(1)
        xx = torch.randn(64, 128)
        net.zero_grad(set_to_none=True)
        net(xx, **kw).backward()
        grads[tag] = torch.cat([p.grad.flatten() for p in net.parameters()])
    d_ours = (grads["ours"] - grads["plain"]).abs().max().item()
    d_torch = (grads["torch"] - grads["plain"]).abs().max().item()
    print(f"  our Checkpoint    vs no checkpointing: max |diff| {d_ours:.3e}")
    print(f"  torch.checkpoint  vs no checkpointing: max |diff| {d_torch:.3e}")
    rec("ours_vs_plain_maxdiff", f"{d_ours:.3e}")
    rec("torch_vs_plain_maxdiff", f"{d_torch:.3e}")
    print("\n  Both exact. Ours is missing the production details -- multiple")
    print("  inputs and outputs, non-tensor arguments, autocast state -- and it")
    print("  handles dropout only if you ask (section 4). The mechanism is")
    print("  complete; the edge cases are what the other 400 lines of")
    print("  torch/utils/checkpoint.py are for.")


# =========================================================================
# 3. the sweep
# =========================================================================
def sweep():
    print("\n" + "=" * 78)
    print("3. SEGMENT SWEEP")
    print("=" * 78)
    x = torch.randn(BATCH, DIM)
    configs = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
    rows = []
    for s in configs:
        torch.manual_seed(0)
        net = DeepNet(segments=s)
        r = measure_memory(net, x)
        r["segments"] = s
        r["seconds"] = 1e9
        rows.append(r)
        del net

    # Time every configuration once per round, then take the best across
    # rounds. Interleaving matters: a background process that slows down one
    # round slows every configuration in it, instead of picking a victim.
    for _ in range(4):
        for r in rows:
            torch.manual_seed(0)
            net = DeepNet(segments=r["segments"])
            r["seconds"] = min(r["seconds"], time_step(net, x))
            del net

    base = rows[0]
    print(f"  {'segments':>9}{'blocks/seg':>11}{'held after fwd':>16}"
          f"{'PEAK':>9}{'sec/step':>10}{'peak vs baseline':>20}")
    for r in rows:
        s = r["segments"]
        bs = "-" if s == 0 else f"{max(1, DEPTH // s)}"
        tag = ("baseline" if s == 0
               else f"{base['peak_mb'] / r['peak_mb']:.1f}x less")
        print(f"  {('none' if s == 0 else s):>9}{bs:>11}{r['held_mb']:>13.1f} MB"
              f"{r['peak_mb']:>7.1f}MB{r['seconds']:>10.3f}{tag:>20}")

    ratios = sorted(r["seconds"] / base["seconds"] for r in rows[1:])
    print(f"\n  A word on the time column before anything else. Theory says every")
    print(f"  checkpointed run should cost about 1.33x: forward + recompute +")
    print(f"  backward is 1+1+2 units of work instead of 1+2. Measured here:")
    print(f"    min {ratios[0]:.2f}x   median {ratios[len(ratios) // 2]:.2f}x"
          f"   max {ratios[-1]:.2f}x")
    print(f"  The spread between configurations is as large as the effect itself,")
    print(f"  because this is a shared CPU and these steps take ~100 ms. Take the")
    print(f"  time column as 'roughly a third more, give or take'; the memory")
    print(f"  columns are exact byte counts and do not move between runs.")
    rec("time_ratio_min", round(ratios[0], 2))
    rec("time_ratio_median", round(ratios[len(ratios) // 2], 2))
    rec("time_ratio_max", round(ratios[-1], 2))

    best = min(rows[1:], key=lambda r: r["peak_mb"])
    print(f"\n  Lowest peak: {best['segments']} segments of "
          f"{DEPTH // best['segments']} blocks -> {best['peak_mb']:.1f} MB, "
          f"{base['peak_mb'] / best['peak_mb']:.1f}x less than the "
          f"{base['peak_mb']:.1f} MB baseline,")
    print(f"  at {best['seconds'] / base['seconds']:.2f}x the step time. "
          f"sqrt(depth) = sqrt({DEPTH}) = {DEPTH ** 0.5:.1f}.")
    rec("best_segments", best["segments"])
    rec("best_peak_mb", round(best["peak_mb"], 2))
    rec("best_time_ratio", round(best["seconds"] / base["seconds"], 3))
    rec("baseline_peak_mb", round(base["peak_mb"], 2))
    rec("baseline_seconds", round(base["seconds"], 4))
    for r in rows:
        rec(f"seg{r['segments']}_held_mb", round(r["held_mb"], 2))
        rec(f"seg{r['segments']}_peak_mb", round(r["peak_mb"], 2))
        rec(f"seg{r['segments']}_seconds", round(r["seconds"], 4))

    print("\n  Read the two memory columns together -- they pull in OPPOSITE")
    print("  directions and that is the entire story.")
    print("    'held after fwd' = one saved input per segment boundary.")
    print("                       More segments -> more boundaries -> MORE memory.")
    print("    'PEAK'           = boundaries, plus the activations of the ONE")
    print("                       segment currently being recomputed.")
    print("                       Bigger segments -> a bigger recompute -> MORE.")
    print("  With k segments of L/k blocks each, peak is roughly")
    print("      k  +  L/k     (in units of one activation tensor)")
    print(f"  which is smallest at k = sqrt(L) = sqrt({DEPTH}) = {DEPTH ** 0.5:.1f}. "
          f"Total memory drops")
    print(f"  from O(L) to O(sqrt(L)) -- and that is where the rule of thumb")
    print("  'checkpoint every sqrt(depth) layers' comes from. It is not folklore;")
    print("  it is the minimum of k + L/k.")
    print("\n  One checkpoint around everything (k=1) holds almost nothing between")
    print("  forward and backward and is still the WORST peak, because backward")
    print("  has to rebuild all 48 blocks at once. 'Just checkpoint the whole")
    print("  model' is the intuitive move and the wrong one.")
    return rows


# =========================================================================
# 4. dropout and the RNG
# =========================================================================
def rng_trap():
    print("\n" + "=" * 78)
    print("4. THE DROPOUT TRAP")
    print("=" * 78)
    print("  Recomputation replays the forward pass. Anything RANDOM in it --")
    print("  a dropout mask above all -- has to come out the same the second")
    print("  time, or backward differentiates a network that never ran.\n")

    def run(**kw):
        torch.manual_seed(0)
        net = DeepNet(depth=12, d=128, segments=4, dropout=0.3)
        net.train()
        torch.manual_seed(1)
        xx = torch.randn(64, 128)
        torch.manual_seed(7)
        net.zero_grad(set_to_none=True)
        net(xx, **kw).backward()
        return torch.cat([p.grad.flatten() for p in net.parameters()])

    torch.manual_seed(0)
    plain = DeepNet(depth=12, d=128, segments=0, dropout=0.3)
    plain.train()
    torch.manual_seed(1)
    xx = torch.randn(64, 128)
    torch.manual_seed(7)
    plain.zero_grad(set_to_none=True)
    plain(xx).backward()
    ref = torch.cat([p.grad.flatten() for p in plain.parameters()])

    cases = [("ours, no RNG handling", dict(impl="ours", preserve_rng=False)),
             ("ours, RNG restored", dict(impl="ours", preserve_rng=True)),
             ("torch, preserve_rng_state=True", dict(impl="torch", preserve_rng=True)),
             ("torch, preserve_rng_state=False", dict(impl="torch", preserve_rng=False))]
    for label, kw in cases:
        g = run(**kw)
        d = (g - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(g, ref, dim=0).item()
        print(f"  {label:<32} max |diff| {d:>9.3e}   cosine similarity {cos:.4f}")
        key = label.replace(",", "").replace(" ", "_").replace("=", "")
        rec(f"dropout_{key}_maxdiff", f"{d:.3e}")
        rec(f"dropout_{key}_cosine", round(cos, 4))

    print("\n  Our fifteen-line version is exactly right without dropout and")
    print("  quietly wrong with it. Restoring the RNG state before recomputing")
    print("  -- three lines -- makes it exact again. That single line is why")
    print("  torch's `checkpoint` has a `preserve_rng_state` argument at all,")
    print("  and why it defaults to True.")
    print("\n  Notice the shape of the failure: cosine similarity stays high, so")
    print("  the gradient still points roughly the right way. Nothing raises,")
    print("  nothing is nan, the loss still falls. It just falls more slowly,")
    print("  forever. Correctness bugs that only cost you a little are the")
    print("  expensive kind.")


# =========================================================================
# 5. use_reentrant
# =========================================================================
def reentrant():
    print("\n" + "=" * 78)
    print("5. use_reentrant=True: THE SILENT NO-OP")
    print("=" * 78)
    import warnings
    torch.manual_seed(0)
    net = DeepNet(depth=8, d=64, segments=4)
    x = torch.randn(32, 64)          # plain data: requires_grad is False

    def probe(label, **kw):
        net.zero_grad(set_to_none=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net(kw.pop("x", x), **kw).backward()
        missing = sum(1 for p in net.parameters() if p.grad is None)
        gn = sum(p.grad.norm().item() ** 2
                 for p in net.blocks[0].parameters() if p.grad is not None) ** 0.5
        print(f"  {label:<46} first-block grad norm {gn:>9.5f}"
              f"   params with no grad: {missing}")
        key = label.replace(" ", "_").replace(",", "").replace("=", "")
        rec(f"reentrant_{key}_grad_norm", round(gn, 5))
        rec(f"reentrant_{key}_params_without_grad", missing)
        return gn, missing

    probe("ours, params passed in", impl="ours")
    probe("ours, naive (params not passed in)", impl="ours", naive=True)
    probe("torch, use_reentrant=False", impl="torch", use_reentrant=False)
    probe("torch, use_reentrant=True", impl="torch", use_reentrant=True)
    probe("torch, use_reentrant=True, input.requires_grad_()",
          impl="torch", use_reentrant=True,
          x=x.clone().requires_grad_(True))

    print("\n  Two of those five silently trained nothing at all in the")
    print("  checkpointed blocks, for the same reason:")
    print("\n    Autograd builds a graph node for a Function only if at least one")
    print("    of its TENSOR ARGUMENTS requires grad.")
    print("\n  The input to the first checkpointed segment is raw data, with")
    print("  requires_grad=False. The weights inside the segment do require it,")
    print("  but autograd cannot see them -- they arrive through a Python")
    print("  closure, not through the argument list. So no node, no backward,")
    print("  no gradients, no error message.")
    print("\n  Two ways out, and they are not equally good:")
    print("    * put the parameters in the argument list, so autograd can see")
    print("      them (our Checkpoint, and torch's reentrant path when it can)")
    print("    * mark the INPUT as requiring grad -- the line-5 workaround. It")
    print("      works by making the check pass, and it also makes torch compute")
    print("      and keep a gradient for your data that you will never use.")
    print("  use_reentrant=False replaces the rule instead of working around it:")
    print("  it uses saved-tensor hooks and never asks the question. That is why")
    print("  torch now warns when you do not pass the argument, and why the")
    print("  answer is always False.")


# =========================================================================
# figures
# =========================================================================
def figures(rows):
    base = rows[0]
    segs = [r["segments"] for r in rows[1:]]
    held = [r["held_mb"] for r in rows[1:]]
    peak = [r["peak_mb"] for r in rows[1:]]
    secs = [r["seconds"] for r in rows[1:]]
    act = BATCH * DIM * 4 / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
        a.grid(True, color=ps.GRID, linewidth=0.8)

    axes[0].plot(segs, peak, "o-", color=ps.SERIES[2], lw=2.0, ms=5,
                 label="peak (what actually limits you)")
    axes[0].plot(segs, held, "s--", color=ps.SERIES[0], lw=1.4, ms=4,
                 label="held between forward and backward")
    theory = [(k + DEPTH / k) * act * 2 for k in segs]
    axes[0].plot(segs, theory, ":", color=ps.INK_MUTED, lw=1.4,
                 label="theory: k + L/k")
    axes[0].axhline(base["peak_mb"], color=ps.SERIES[1], ls="--", lw=1.2)
    axes[0].text(segs[-1], base["peak_mb"] * 1.05, "no checkpointing",
                 ha="right", fontsize=8.5, color=ps.SERIES[1])
    axes[0].axvline(DEPTH ** 0.5, color=ps.INK_MUTED, ls=":", lw=1.2)
    axes[0].text(DEPTH ** 0.5 * 1.1, max(peak) * 0.75, "√48", fontsize=10,
                 color=ps.INK_MUTED)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(segs)
    axes[0].set_xticklabels(segs)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].set_title("Activation memory", color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("checkpoint segments (k)", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("MB (log scale)", color=ps.INK_SECONDARY, fontsize=10)

    ratios = [s / base["seconds"] for s in secs]
    axes[1].plot(segs, ratios, "o", color=ps.SERIES[1], ms=7)
    axes[1].axhline(4 / 3, color=ps.SERIES[3], ls="-", lw=1.6)
    axes[1].text(segs[-1], 4 / 3 + 0.03, "theory: 1.33x (one extra forward)",
                 ha="right", fontsize=9, color=ps.SERIES[3])
    axes[1].axhline(1.0, color=ps.INK_MUTED, ls="--", lw=1.0)
    axes[1].text(segs[-1], 1.01, "no checkpointing", ha="right", fontsize=8.5,
                 color=ps.INK_MUTED)
    axes[1].set_xscale("log")
    axes[1].set_xticks(segs)
    axes[1].set_xticklabels(segs)
    axes[1].set_ylim(0.9, max(1.7, max(ratios) * 1.05))
    axes[1].set_title("Step time ratio — the scatter IS the measurement noise",
                      color=ps.INK, fontsize=11, loc="left")
    axes[1].set_xlabel("checkpoint segments (k)", color=ps.INK_SECONDARY, fontsize=10)
    ps.save(fig, os.path.join(OUT, "memory_time_tradeoff.png"))

    fig, ax = ps.new_axes(6.8, 4.3)
    ax.plot([base["seconds"] * 1000], [base["peak_mb"]], "*", ms=18,
            color=ps.SERIES[2], label="no checkpointing", zorder=3)
    ax.plot([s * 1000 for s in secs], peak, "o-", color=ps.SERIES[0], lw=1.4,
            ms=5, label="checkpointed")
    for r in rows[1:]:
        if r["segments"] in (1, 2, 6, 12, 48):
            ax.annotate(f"k={r['segments']}", (r["seconds"] * 1000, r["peak_mb"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=8.5,
                        color=ps.INK_SECONDARY)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Pick a point on the curve",
              "milliseconds per forward+backward", "peak activation memory (MB, log)",
              os.path.join(OUT, "pareto.png"))


# =========================================================================
def main():
    t0 = time.perf_counter()
    what_is_kept()
    agreement()
    rows = sweep()
    rng_trap()
    reentrant()
    figures(rows)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
