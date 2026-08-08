"""Project 17 — Reproducible training.

Train the same model twice and get bit-identical weights. Then find every knob
that quietly breaks that, and measure what each one costs.

  1. the baseline: two runs, zero difference
  2. one missing seed at a time
  3. the DataLoader: shuffle order, worker seeds, and num_workers
  4. thread count changes the arithmetic itself
  5. what torch.use_deterministic_algorithms(True) does -- and does not do
  6. resuming bit-exactly needs the RNG state, not just the weights
  7. what determinism costs, and what it still does not buy you

Runs in about 15 seconds on CPU. No downloads.
"""

import copy
import csv
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

FINDINGS = OrderedDict()
STEPS = 60


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# the training problem: a DataLoader with a numpy-based augmentation
# =========================================================================
class Noisy(Dataset):
    """40 features, 3 classes, plus a numpy random jitter applied per sample."""

    def __init__(self, n=1024):
        g = torch.Generator().manual_seed(1234)          # the DATA never changes
        self.X = torch.randn(n, 40, generator=g)
        W = torch.randn(40, 3, generator=g)
        self.y = (self.X @ W).argmax(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        jitter = torch.from_numpy(np.random.randn(40).astype(np.float32)) * 0.1
        return self.X[i] + jitter, self.y[i]


def seed_all(seed=0, torch_seed=True, numpy_seed=True, python_seed=True):
    if python_seed:
        random.seed(seed)
    if numpy_seed:
        np.random.seed(seed)
    if torch_seed:
        torch.manual_seed(seed)


def train(seed=0, num_workers=0, threads=1, deterministic=False,
          shuffle_generator=True, steps=STEPS, **seed_kw):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    seed_all(seed, **seed_kw)

    model = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    gen = None
    if shuffle_generator:
        gen = torch.Generator()
        gen.manual_seed(seed)
    loader = DataLoader(Noisy(), batch_size=64, shuffle=True,
                        num_workers=num_workers, generator=gen,
                        persistent_workers=num_workers > 0)

    losses, step = [], 0
    while step < steps:
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
    torch.use_deterministic_algorithms(False)
    return model, np.array(losses)


def wdiff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


# =========================================================================
# 1. the baseline
# =========================================================================
def baseline():
    print("=" * 78)
    print("1. THE BASELINE: TWO RUNS, ZERO DIFFERENCE")
    print("=" * 78)

    a, la = train()
    b, lb = train()
    d = wdiff(a, b)
    print(f"  identical loss curves     : {np.array_equal(la, lb)}")
    print(f"  max |weight difference|   : {d:.3e}")
    print(f"  final loss                : {la[-1]:.6f} and {lb[-1]:.6f}")
    print()
    print("  Everything random in this run is seeded:")
    print("    torch.manual_seed  the model's initialization")
    print("    np.random.seed     the augmentation inside __getitem__")
    print("    random.seed        anything using Python's own random module")
    print("    DataLoader(generator=g)   the shuffle order")
    print("    torch.set_num_threads(1)  the arithmetic itself (section 4)")
    print()
    print("  Note the DATA is built with its own torch.Generator, not from the")
    print("  global seed. A dataset that changes when you change a seed makes every")
    print("  comparison below meaningless -- fix the data first, then vary the run.")
    print()
    rec("baseline_weight_diff", d)
    rec("baseline_curves_identical", bool(np.array_equal(la, lb)))
    return la


# =========================================================================
# 2. one missing seed at a time
# =========================================================================
def missing_seeds(ref_losses):
    print("=" * 78)
    print("2. ONE MISSING SEED AT A TIME")
    print("=" * 78)

    ref, _ = train()
    cases = [
        ("everything seeded", dict()),
        ("no torch seed", dict(torch_seed=False)),
        ("no numpy seed", dict(numpy_seed=False)),
        ("no python random seed", dict(python_seed=False)),
        ("no DataLoader generator", dict()),
    ]
    rows = []
    print(f"  {'what is missing':<26}{'max |weight diff|':>20}{'first step where':>19}")
    print(f"  {'':<26}{'':>20}{'losses differ':>19}")
    for label, kw in cases:
        sg = label != "no DataLoader generator"
        m, ls = train(shuffle_generator=sg, **kw)
        d = wdiff(ref, m)
        first = next((i for i, (u, v) in enumerate(zip(ref_losses, ls)) if u != v), None)
        rows.append((label, d, first))
        print(f"  {label:<26}{d:>20.3e}{str(first) if first is not None else 'never':>19}")
        rec(f"missing_{label.replace(' ', '_')}", d)
    print()
    print("  Reading the last column: 'never' means the two runs produced the same")
    print("  loss at every one of the 60 steps.")
    print()
    print("  - `torch.manual_seed` controls the model's initialization, so leaving")
    print("    it out changes step 0 and everything after it.")
    print("  - `np.random.seed` controls the augmentation in __getitem__. The model")
    print("    starts identical and the curves separate at step 0 anyway, because")
    print("    the very first batch has different noise on it.")
    print("  - `random.seed` changes nothing here, because nothing in this script")
    print("    uses Python's `random`. Seed it anyway: torchvision transforms and")
    print("    most augmentation libraries do use it, and the failure is silent.")
    print("  - the DataLoader generator controls the shuffle ORDER. Without one it")
    print("    draws from the global torch RNG, which we did seed -- so this run is")
    print("    still reproducible. It stops being reproducible the moment anything")
    print("    else consumes the global RNG between epochs.")
    print()
    print("  The rule that follows: seed all three, pass an explicit generator, and")
    print("  do not rely on 'the global seed covers it'. It covers it until someone")
    print("  adds a dropout layer or a random crop above your loader.")
    print()
    return rows


# =========================================================================
# 3. the DataLoader
# =========================================================================
class Peek(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        return (i,
                np.random.randint(0, 10 ** 6),
                torch.randint(0, 10 ** 6, (1,)).item(),
                random.randint(0, 10 ** 6))


def dataloader_section():
    print("=" * 78)
    print("3. THE DATALOADER: WORKERS HAVE THEIR OWN RANDOMNESS")
    print("=" * 78)

    def peek(nw, seed=0):
        seed_all(seed)
        dl = DataLoader(Peek(), batch_size=8, num_workers=nw, shuffle=False)
        return [tuple(b[1].tolist()) for b in dl]

    a0, b0 = peek(0), peek(0)
    a4, b4 = peek(4), peek(4)
    overlap = len(set(sum(a0, ())) & set(sum(a4, ())))
    print(f"  num_workers=0, run twice : identical  {a0 == b0}")
    print(f"  num_workers=4, run twice : identical  {a4 == b4}")
    print(f"  num_workers=0 vs 4       : identical  {a0 == a4}   "
          f"values in common: {overlap} of 32")
    print()
    print("  Both settings are perfectly reproducible, and they reproduce DIFFERENT")
    print("  runs. Each worker gets `base_seed + worker_id` and seeds torch, numpy")
    print("  and Python's random with it, so the number of workers determines how")
    print("  many independent streams there are and which sample draws from which.")
    print()
    print("  So `num_workers` is part of your seed. Change it for speed and your")
    print("  'reproducible' run reproduces something else -- with no warning, and")
    print("  usually while you are changing it for an unrelated reason.")
    print()
    rec("workers_0_reproducible", a0 == b0)
    rec("workers_4_reproducible", a4 == b4)
    rec("workers_0_vs_4_overlap", overlap)

    # the famous bug, and its current status
    seed_all(0)
    dl = DataLoader(Peek(), batch_size=4, num_workers=4, shuffle=False)
    batches = [tuple(b[1].tolist()) for b in dl][:4]
    dup = len(set(batches)) == 1
    print(f"  the classic 'every worker draws the same numpy numbers' bug: "
          f"{'PRESENT' if dup else 'not present'}")
    print(f"    first four batches, one per worker: {batches[0]}")
    print(f"                                        {batches[1]}")
    print()
    print("  This is worth knowing because half the DataLoader boilerplate on the")
    print("  internet exists to fix it. Under `fork`, workers inherit the parent's")
    print("  numpy state, so historically all of them produced the SAME augmentation")
    print("  sequence -- your batch of 32 'random crops' was 4 crops repeated 8")
    print("  times. Modern PyTorch seeds numpy and Python's random per worker, so")
    print("  the hand-written worker_init_fn is no longer needed. Copying it does no")
    print("  harm; believing you still need it does, because it hides the real")
    print("  remaining issue, which is the paragraph above.")
    print()
    rec("classic_numpy_worker_bug_present", dup)


# =========================================================================
# 4. threads
# =========================================================================
def thread_section():
    print("=" * 78)
    print("4. THE THREAD COUNT CHANGES THE ARITHMETIC")
    print("=" * 78)

    torch.manual_seed(0)
    x = torch.randn(1_000_000)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    print(f"  {'threads':>8}{'x.sum()':>20}{'max |A@B - (A@B at 1 thread)|':>34}")
    ref = None
    sums = {}
    for n in (1, 2, 4, 8, 12):
        torch.set_num_threads(n)
        s = x.sum().item()
        m = a @ b
        if ref is None:
            ref = m
        sums[n] = s
        print(f"  {n:>8}{s:>20.7f}{(m - ref).abs().max().item():>34.3e}")
    torch.set_num_threads(1)
    print()
    print(f"  distinct sums across thread counts: {len(set(sums.values()))} of {len(sums)}")
    print()
    print("  Floating point addition is not associative: (a+b)+c and a+(b+c) can")
    print("  differ in the last bits. A multi-threaded reduction splits the array")
    print("  into one chunk per thread, adds each chunk, then combines -- so the")
    print("  GROUPING depends on how many threads there are, and the answer changes.")
    print()
    print("  Nothing here is random. Each thread count is perfectly reproducible on")
    print("  its own. But the number of threads defaults to the number of cores, so")
    print("  the same script on the same seed gives different numbers on a laptop")
    print("  and on a server -- and different numbers again when another process is")
    print("  using half the machine and you set OMP_NUM_THREADS to compensate.")
    print()

    m1, l1 = train(threads=1)
    m12, l12 = train(threads=12)
    d = wdiff(m1, m12)
    first = next((i for i, (u, v) in enumerate(zip(l1, l12)) if u != v), None)
    print(f"  a full training run at 1 vs 12 threads: max |weight diff| {d:.3e}, "
          f"curves differ from step {first if first is not None else 'never'}")
    print()
    print("  Small, because this model is small -- but not zero, and it is the kind")
    print("  of difference that grows: project 7 measured two runs 1e-16 apart")
    print("  reaching 0.357 after 150 steps.")
    print()
    rec("thread_sum_distinct", len(set(sums.values())))
    rec("thread_train_weight_diff", d)
    rec("thread_train_first_diff_step", -1 if first is None else first)
    return sums


# =========================================================================
# 5. use_deterministic_algorithms
# =========================================================================
def deterministic_section():
    print("=" * 78)
    print("5. WHAT torch.use_deterministic_algorithms(True) ACTUALLY DOES")
    print("=" * 78)

    torch.use_deterministic_algorithms(True)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    torch.set_num_threads(1)
    m1 = a @ b
    torch.set_num_threads(12)
    m12 = a @ b
    torch.set_num_threads(1)
    print(f"  with the flag ON, matmul at 1 vs 12 threads still differs by "
          f"{(m1 - m12).abs().max().item():.3e}")
    print()

    tests = {
        "index_add_": lambda: torch.zeros(10).index_add_(0, torch.tensor([1, 1, 2]), torch.ones(3)),
        "scatter_add_": lambda: torch.zeros(10).scatter_add_(0, torch.tensor([1, 1, 2]), torch.ones(3)),
        "index_put_(accumulate=True)": lambda: torch.zeros(10).index_put_(
            (torch.tensor([1, 1]),), torch.ones(2), accumulate=True),
        "interpolate bilinear, backward": lambda: F.interpolate(
            torch.randn(1, 1, 8, 8, requires_grad=True), scale_factor=2,
            mode="bilinear").sum().backward(),
        "grid_sample, backward": lambda: F.grid_sample(
            torch.randn(1, 1, 8, 8, requires_grad=True),
            torch.rand(1, 4, 4, 2) * 2 - 1, align_corners=False).sum().backward(),
        "embedding_bag, backward": lambda: F.embedding_bag(
            torch.tensor([1, 2, 3]), torch.randn(10, 4, requires_grad=True),
            torch.tensor([0])).sum().backward(),
    }
    raised = 0
    print("  operations that are famous for being non-deterministic, on CPU:")
    for name, fn in tests.items():
        try:
            fn()
            print(f"    {name:<32} runs")
        except RuntimeError as e:
            raised += 1
            print(f"    {name:<32} RAISES: {str(e)[:44]}")
    torch.use_deterministic_algorithms(False)
    print()
    print(f"  {raised} of {len(tests)} raised.")
    print()
    print("  On CPU this flag is close to a no-op, and that is not a disappointment")
    print("  -- it is the point. The flag does not MAKE anything deterministic. It")
    print("  turns on a check: 'if I am about to use an implementation whose result")
    print("  depends on scheduling, raise instead'. Nearly every CPU kernel already")
    print("  has a deterministic implementation, so nothing fires.")
    print()
    print("  On CUDA the same six operations mostly do raise, because their fast")
    print("  implementations use atomicAdd, and atomic additions land in whatever")
    print("  order the blocks happen to finish. There the flag is essential, and it")
    print("  comes with a companion:")
    print()
    print("    CUBLAS_WORKSPACE_CONFIG=:4096:8   (an environment variable, because")
    print("    cuBLAS picks its reduction split when the workspace is created --")
    print("    before any Python you could call has run)")
    print()
    print("  And as the first line shows, the flag says nothing about thread counts.")
    print("  Determinism has to be pinned at three levels: the seeds (section 2),")
    print("  the algorithms (this flag), and the parallel layout (section 4).")
    print()
    rec("det_flag_raised_on_cpu", raised)


# =========================================================================
# 6. resuming
# =========================================================================
def resume_section():
    print("=" * 78)
    print("6. RESUMING BIT-EXACTLY NEEDS THE RNG STATE (AND MORE)")
    print("=" * 78)

    torch.set_num_threads(1)
    ds = Noisy()
    per_epoch = len(ds) // 64
    EPOCHS = 4
    print(f"  {len(ds)} samples / batch 64 = {per_epoch} steps per epoch, "
          f"{EPOCHS} epochs. We interrupt after epoch 2 and resume.")
    print()

    def fresh():
        seed_all(0)
        m = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 3))
        o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
        g = torch.Generator()
        g.manual_seed(0)
        return m, o, g

    def snapshot(model, opt, gen):
        return {"model": copy.deepcopy(model.state_dict()),
                "optim": copy.deepcopy(opt.state_dict()),
                "torch_rng": torch.get_rng_state().clone(),
                "numpy_rng": copy.deepcopy(np.random.get_state()),
                "loader_gen": gen.get_state().clone()}

    # The reference: four epochs straight through. Both checkpoints are taken
    # DURING this run, so each one is a true prefix of it.
    model, opt, gen = fresh()
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=gen)
    ref_curve, ck_boundary, ck_mid = [], None, None
    for e in range(EPOCHS):
        for i, (x, y) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            ref_curve.append(loss.item())
            if e == 2 and i + 1 == per_epoch // 2:
                ck_mid = snapshot(model, opt, gen)
                mid_len = len(ref_curve)
        if e == 1:
            ck_boundary = snapshot(model, opt, gen)
            boundary_len = len(ref_curve)
    ref_model, ref_curve = model, np.array(ref_curve)
    head_b = list(ref_curve[:boundary_len])
    head_m = list(ref_curve[:mid_len])

    def resume(head, ckpt, epochs_done, restore_rng):
        m2 = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 3))
        m2.load_state_dict(ckpt["model"])
        o2 = torch.optim.SGD(m2.parameters(), lr=0.05, momentum=0.9)
        # deepcopy: Optimizer.load_state_dict does NOT copy the state tensors when
        # dtype and device already match, so a second load from the same
        # checkpoint would see buffers the first run had already stepped.
        o2.load_state_dict(copy.deepcopy(ckpt["optim"]))
        g2 = torch.Generator()
        if restore_rng:
            torch.set_rng_state(ckpt["torch_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            g2.set_state(ckpt["loader_gen"])
        else:
            seed_all(0)
            g2.manual_seed(0)
        l2 = DataLoader(ds, batch_size=64, shuffle=True, generator=g2)
        rest = []
        for e in range(EPOCHS - epochs_done):
            for x, y in l2:
                o2.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m2(x), y)
                loss.backward()
                o2.step()
                rest.append(loss.item())
        full = np.array(head + rest)[:len(ref_curve)]
        first = next((i for i, (u, v) in enumerate(zip(ref_curve, full)) if u != v), None)
        return wdiff(ref_model, m2), first, full

    cases = [
        ("weights + optimizer only", head_b, ck_boundary, 2, False),
        ("+ RNG state (torch, numpy, loader)", head_b, ck_boundary, 2, True),
        ("+ RNG state, interrupted mid-epoch", head_m, ck_mid, 2, True),
    ]
    results = OrderedDict()
    print(f"  {'what was restored':<40}{'max |weight diff|':>19}{'curves differ from':>21}")
    for label, head, ck, done, rng in cases:
        d, first, curve = resume(head, ck, done, rng)
        results[label] = (d, first, curve)
        print(f"  {label:<40}{d:>19.3e}"
              f"{(str(first) if first is not None else 'never'):>21}")
    print()
    print(f"  Every row is compared against the same run trained straight through")
    print(f"  for {EPOCHS} epochs -- which is what a resume is supposed to reproduce.")
    print()
    print("  Line 1: a correct model, and a different run. Without the RNG state the")
    print("  shuffle order and the augmentation noise after the restart are whatever")
    print("  seed 0 produces from a standing start -- which is exactly what epoch 1")
    print("  saw. The run silently replays its first epoch's random choices.")
    print()
    print("  Line 2: bit-identical, to 0.000e+00. Weights, optimizer state, torch")
    print("  RNG, numpy RNG and the loader's generator, all restored on an epoch")
    print("  boundary. This is what a complete checkpoint buys you.")
    print()
    print("  Line 3 is the one nobody expects. Same five things restored, but the")
    print("  interruption happened half way through epoch 3 -- and it diverges from")
    print("  the moment of the restart.")
    print()
    print("  A DataLoader cannot be resumed mid-epoch. Iterating it again starts a")
    print("  NEW epoch: a fresh permutation, batch 0 first. The batches you had not")
    print("  reached get shuffled back in, and the ones you had already trained on")
    print("  come round a second time. Nothing is corrupted -- the sample ORDER is")
    print("  simply not the one the uninterrupted run would have used. On a dataset")
    print("  where one epoch is a day of training, that is most of your run.")
    print()
    print("  Three ways out, in increasing order of effort: checkpoint only on epoch")
    print("  boundaries (what most training scripts quietly do), save the batch index")
    print("  and skip that many batches on resume (correct, and slow), or use a")
    print("  stateful loader that serializes its own position.")
    print()
    print("  One more trap, found while writing this section. Restoring the same")
    print("  checkpoint twice in one process gave a different answer the second")
    print("  time, and the cause is that `Optimizer.load_state_dict` does NOT copy")
    print("  the state tensors when the dtype and device already match -- the")
    print("  momentum buffers you load ARE the checkpoint's tensors, and the first")
    print("  run steps them in place. `model.load_state_dict` copies (it uses")
    print("  copy_), so the asymmetry is easy to miss. Deep-copy the optimizer state")
    print("  if you intend to reuse a checkpoint.")
    print()
    print("  Five things to save, and none of them is optional:")
    print("    torch.get_rng_state()        (plus torch.cuda.get_rng_state_all())")
    print("    np.random.get_state()")
    print("    random.getstate()")
    print("    the DataLoader generator's get_state()")
    print("    where you are inside the epoch")
    print()
    for i, (label, v) in enumerate(results.items()):
        rec(f"resume_{i}_diff", v[0])
    return results, ref_curve


# =========================================================================
# 7. cost
# =========================================================================
def cost_section():
    print("=" * 78)
    print("7. WHAT IT COSTS")
    print("=" * 78)

    times = {}
    for label, kw in (("1 thread", dict(threads=1)),
                      ("8 threads", dict(threads=8)),
                      ("1 thread + deterministic flag", dict(threads=1, deterministic=True))):
        best = min(_time_one(kw) for _ in range(3))
        times[label] = best
        print(f"  {label:<32}{best:>8.2f} s for {STEPS} steps")
    torch.set_num_threads(1)
    print()
    print("  On this problem the deterministic flag is free, and threads barely")
    print("  help because the model is tiny. Neither number generalizes: on a real")
    print("  model the flag can cost 10-30% (it replaces fast atomicAdd kernels with")
    print("  slower ordered ones) and threads are the whole reason CPU training is")
    print("  bearable at all.")
    print()
    print("  The honest summary of this project is a hierarchy, not a switch:")
    print()
    print("    same process, same seeds            -> bit-identical")
    print("    same machine, same threads, same    -> bit-identical")
    print("      num_workers, same torch build")
    print("    different thread count              -> different in the last bits")
    print("    different CPU, GPU, or torch build  -> different, sometimes visibly")
    print()
    print("  'Reproducible' in a paper almost never means the top line. It means")
    print("  'the seeds are fixed, so the conclusion does not depend on luck'. That")
    print("  is the useful property, and it is why the practical advice is to seed")
    print("  everything, report the variance over several seeds, and reach for")
    print("  bit-exactness only when you are BISECTING -- comparing two versions of")
    print("  your own code, where any difference at all is a signal.")
    print()
    for k, v in times.items():
        rec(f"time_{k.replace(' ', '_')}", v)


def _time_one(kw):
    t = time.perf_counter()
    train(**kw)
    return time.perf_counter() - t


# =========================================================================
# figure
# =========================================================================
def figure(rows, resume, ref_losses, sums):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    labels = [r[0] for r in rows]
    vals = [max(r[1], 1e-18) for r in rows]
    colors = [ps.SERIES[1] if v < 1e-12 else ps.SERIES[2] for v in vals]
    ax.barh(range(len(labels))[::-1], vals, color=colors, height=0.55)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e-18, 10)
    ax.grid(True, axis="x", color=ps.GRID, linewidth=0.8)
    ax.set_title("Max weight difference from the reference run", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_xlabel("max |weight difference|  (log; leftmost bar is exactly 0)",
                  color=ps.INK_SECONDARY, fontsize=9)

    ax = axes[1]
    ax.plot(ref_losses, color=ps.INK_MUTED, linewidth=3.0, label="uninterrupted")
    for i, (label, v) in enumerate(resume.items()):
        ax.plot(v[2], color=ps.SERIES[i], linewidth=1.5,
                linestyle=["-", "--", ":"][i], label=label)
    ax.axvline(32, color=ps.INK_MUTED, linewidth=0.9, linestyle=":")
    ax.text(33, max(ref_losses) * 0.9, "restart", color=ps.INK_SECONDARY, fontsize=9)
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Resuming: with and without the RNG state", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("training loss", color=ps.INK_SECONDARY, fontsize=10)

    ps.save(fig, os.path.join(OUT, "reproducibility.png"))


def main():
    ref_losses = baseline()
    rows = missing_seeds(ref_losses)
    dataloader_section()
    sums = thread_section()
    deterministic_section()
    resume, ref2 = resume_section()
    cost_section()
    figure(rows, resume, ref2, sums)

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
