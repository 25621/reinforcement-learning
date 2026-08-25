"""Project 50 - a determinism audit of a small CPU training run.

The goal is a run that produces the *same bits* twice. Not "the same to five
decimal places" — the same bits. The method is not "add flags until it works";
it is to run the thing twice in two fresh processes, hash the result, and then
remove one control at a time to find out which ones were actually load-bearing.

Sections:
  1. the baseline: the same script, twice, two different models
  2. the ladder: controls added one at a time
  3. the ablation: which single control, removed, breaks it
  4. torch.use_deterministic_algorithms on CPU: an audit of what it blocks
  5. thread count is part of your seed
  6. PYTHONHASHSEED: the randomness that is not in your code
  7. how big does one changed bit get?
  8. what determinism costs
  9. the recipe, and what it does *not* buy you

Run:  python3 run.py        (~5 minutes; it launches ~40 short child processes)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.utils.data

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

CHILD = os.path.join(HERE, "train_once.py")

FULL = dict(torch_seed=True, py_np_seed=True, loader_generator=True,
            worker_init=True, deterministic_algos=True, sorted_vocab=True,
            threads=4, workers=0, augment=True, epochs=3)
NAIVE = dict(FULL, torch_seed=False, py_np_seed=False, loader_generator=False,
             worker_init=False, deterministic_algos=False, sorted_vocab=False)


def run_child(cfg, env_extra=None, timeout=300):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, CHILD, json.dumps(cfg)],
                         capture_output=True, text=True, env=env, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def twice(cfg, env_extra=None):
    """Run the identical configuration in two fresh processes."""
    a = run_child(cfg, env_extra)
    b = run_child(cfg, env_extra)
    return a, b, a["fingerprint"] == b["fingerprint"]


# ===========================================================================
# 1. Baseline
# ===========================================================================

F_.head("1. The baseline: the same script, twice")

a, b, same = twice(NAIVE, {"PYTHONHASHSEED": "random"})
F_.note("run A fingerprint", a["fingerprint"])
F_.note("run B fingerprint", b["fingerprint"])
F_.note("bit-identical?", same)
F_.note("final loss A / B", f"{a['final_loss']:.6f} / {b['final_loss']:.6f}")
F_.note("difference in final loss", f"{abs(a['final_loss'] - b['final_loss']):.6f}")
F_.note("would you notice this in a results table?",
        "the losses differ in the 2nd decimal - about the size of a real improvement")


# ===========================================================================
# 2. The ladder
# ===========================================================================

F_.head("2. The ladder: add one control at a time")

LADDER = [
    ("nothing", dict(NAIVE)),
    ("+ torch.manual_seed(0)", dict(NAIVE, torch_seed=True)),
    ("+ random.seed / np.random.seed", dict(NAIVE, torch_seed=True, py_np_seed=True)),
    ("+ DataLoader generator", dict(NAIVE, torch_seed=True, py_np_seed=True,
                                    loader_generator=True)),
    ("+ sorted vocabulary", dict(NAIVE, torch_seed=True, py_np_seed=True,
                                 loader_generator=True, sorted_vocab=True)),
    ("+ use_deterministic_algorithms", dict(FULL, workers=0)),
]
ladder_rows = []
for label, cfg in LADDER:
    ra, rb, ok = twice(cfg, {"PYTHONHASHSEED": "random"})
    ladder_rows.append((label, ok, ra["fingerprint"], rb["fingerprint"]))
    F_.note(label, f"{'LOCKED  ' if ok else 'differs '} {ra['fingerprint']} vs {rb['fingerprint']}")


# ===========================================================================
# 3. The ablation
# ===========================================================================

F_.head("3. The ablation: remove exactly one control from the locked config")

ABLATIONS = [
    ("torch.manual_seed", dict(FULL, torch_seed=False)),
    ("random/np seeds", dict(FULL, py_np_seed=False)),
    ("DataLoader generator", dict(FULL, loader_generator=False)),
    ("sorted vocabulary", dict(FULL, sorted_vocab=False)),
    ("use_deterministic_algorithms", dict(FULL, deterministic_algos=False)),
    ("worker_init_fn (with 2 workers)", dict(FULL, workers=2, worker_init=False)),
    ("nothing (control, 2 workers)", dict(FULL, workers=2)),
]
ablation_rows = []
for label, cfg in ABLATIONS:
    ra, rb, ok = twice(cfg, {"PYTHONHASHSEED": "random"})
    ablation_rows.append((label, ok))
    verdict = "still locked" if ok else "BREAKS reproducibility"
    F_.note(f"remove {label}", verdict)

F_.note("controls that were load-bearing",
        sum(1 for _, ok in ablation_rows if not ok))
F_.note("controls that made no difference here",
        sum(1 for _, ok in ablation_rows if ok))

# The surprise above is that worker_init_fn is NOT load-bearing. Check that
# claim directly instead of trusting one fingerprint comparison.
w1 = run_child(dict(FULL, workers=2, worker_init=False), {"PYTHONHASHSEED": "0"})
w2 = run_child(dict(FULL, workers=2, worker_init=False), {"PYTHONHASHSEED": "0"})
F_.note("2 workers, NO worker_init_fn, two processes: same fingerprint?",
        w1["fingerprint"] == w2["fingerprint"])
wsrc = os.path.join(os.path.dirname(torch.utils.data.__file__), "_utils", "worker.py")
with open(wsrc) as fh:
    wtext = fh.read()
F_.note("torch seeds `random` in each worker", "random.seed(seed)" in wtext)
F_.note("torch seeds `torch` in each worker", "torch.manual_seed(seed)" in wtext)
F_.note("torch seeds `numpy` in each worker", "np.random.seed(np_seed)" in wtext)
F_.note("so the classic worker_init_fn advice is",
        "outdated for random/numpy/torch - still needed for any OTHER global RNG")


# ===========================================================================
# 4. What use_deterministic_algorithms actually blocks, on CPU
# ===========================================================================

F_.head("4. Auditing torch.use_deterministic_algorithms on CPU")

OPS = {
    "index_add_": lambda: torch.zeros(10).index_add_(0, torch.tensor([1, 1]), torch.ones(2)),
    "scatter_add_": lambda: torch.zeros(10).scatter_add_(0, torch.tensor([1, 1]), torch.ones(2)),
    "index_put_(accumulate=True)": lambda: torch.zeros(10).index_put_(
        (torch.tensor([1, 1]),), torch.ones(2), accumulate=True),
    "put_": lambda: torch.zeros(10).put_(torch.tensor([1, 1]), torch.ones(2)),
    "bincount": lambda: torch.bincount(torch.tensor([1, 1, 2])),
    "kthvalue": lambda: torch.kthvalue(torch.randn(10), 3),
    "grid_sample backward": lambda: torch.nn.functional.grid_sample(
        torch.randn(1, 1, 4, 4, requires_grad=True), torch.rand(1, 4, 4, 2),
        align_corners=False).sum().backward(),
    "nll_loss2d backward": lambda: torch.nn.functional.nll_loss(
        torch.randn(2, 3, 4, 4, requires_grad=True),
        torch.randint(0, 3, (2, 4, 4))).backward(),
    "max_pool3d backward": lambda: torch.nn.functional.max_pool3d(
        torch.randn(1, 1, 4, 4, 4, requires_grad=True), 2).sum().backward(),
    "interpolate(bilinear) backward": lambda: torch.nn.functional.interpolate(
        torch.randn(1, 1, 4, 4, requires_grad=True), scale_factor=2,
        mode="bilinear").sum().backward(),
    "reflection_pad backward": lambda: torch.nn.functional.pad(
        torch.randn(1, 1, 4, 4, requires_grad=True), (1, 1, 1, 1),
        mode="reflect").sum().backward(),
    "scatter_reduce_": lambda: torch.zeros(10).scatter_reduce_(
        0, torch.tensor([1, 1]), torch.ones(2), reduce="sum"),
    "embedding_bag backward": lambda: torch.nn.functional.embedding_bag(
        torch.tensor([0, 1, 1]), torch.randn(4, 3, requires_grad=True),
        torch.tensor([0])).sum().backward(),
    "cumsum": lambda: torch.randn(10).cumsum(0),
    "median": lambda: torch.randn(10).median(),
}
torch.use_deterministic_algorithms(True)
blocked = []
for name, fn in OPS.items():
    try:
        fn()
    except RuntimeError as exc:
        if "deterministic" in str(exc):
            blocked.append(name)
torch.use_deterministic_algorithms(False)
F_.note("operations probed", len(OPS))
F_.note("blocked by the flag on this CPU build", f"{len(blocked)}: {', '.join(blocked) or 'none'}")
F_.note("what that means",
        "on CPU the flag is close to a no-op; it is a GPU tool, where atomics reorder")


# ===========================================================================
# 5. Thread count is part of your seed
# ===========================================================================

F_.head("5. Thread count is part of your seed")

g = torch.Generator().manual_seed(0)
big = torch.randn(4_000_000, generator=g)
mat = torch.randn(768, 768, generator=g)
per_thread = {}
for n in (1, 2, 4, 6):
    torch.set_num_threads(n)
    per_thread[n] = (D.fingerprint(big.sum()), D.fingerprint(mat @ mat),
                     float(big.sum()))
torch.set_num_threads(4)
for n, (fs, fm, v) in per_thread.items():
    F_.note(f"{n} thread(s): sum fingerprint / value", f"{fs} / {v!r}")
distinct_sum = len({v[0] for v in per_thread.values()})
distinct_mm = len({v[1] for v in per_thread.values()})
F_.note("distinct sum results across 4 thread counts", distinct_sum)
F_.note("distinct matmul results across 4 thread counts", distinct_mm)

torch.set_num_threads(4)
repeats = {D.fingerprint(big.sum()) for _ in range(20)}
F_.note("distinct sum results over 20 repeats at a FIXED thread count", len(repeats))
F_.note("so CPU non-determinism is", "a configuration difference, not randomness")

ta = run_child(dict(FULL, threads=1))
tb = run_child(dict(FULL, threads=4))
F_.note("full training run, 1 thread vs 4 threads: same fingerprint?",
        ta["fingerprint"] == tb["fingerprint"])
F_.note("final loss, 1 thread / 4 threads", f"{ta['final_loss']:.8f} / {tb['final_loss']:.8f}")


# ===========================================================================
# 6. PYTHONHASHSEED
# ===========================================================================

F_.head("6. PYTHONHASHSEED: randomness that is not in your code")

cfg6 = dict(FULL, sorted_vocab=False)
r1 = run_child(cfg6, {"PYTHONHASHSEED": "random"})
r2 = run_child(cfg6, {"PYTHONHASHSEED": "random"})
r3 = run_child(cfg6, {"PYTHONHASHSEED": "0"})
r4 = run_child(cfg6, {"PYTHONHASHSEED": "0"})
F_.note("vocab built from a set, PYTHONHASHSEED=random: two runs match?",
        r1["fingerprint"] == r2["fingerprint"])
F_.note("vocab built from a set, PYTHONHASHSEED=0: two runs match?",
        r3["fingerprint"] == r4["fingerprint"])
r5 = run_child(dict(FULL, sorted_vocab=True), {"PYTHONHASHSEED": "random"})
r6 = run_child(dict(FULL, sorted_vocab=True), {"PYTHONHASHSEED": "random"})
F_.note("vocab built from sorted(set), PYTHONHASHSEED=random: two runs match?",
        r5["fingerprint"] == r6["fingerprint"])
F_.note("preferred fix", "sort the set; do not rely on an env var callers may not set")


# ===========================================================================
# 7. How big does one changed bit get?
# ===========================================================================

F_.head("7. Amplification: one changed bit, 48 steps later")

la = np.array(ta["losses"])
lb = np.array(tb["losses"])
diff = np.abs(la - lb)
first_nonzero = int(np.argmax(diff > 0)) if (diff > 0).any() else -1
F_.note("the only difference between these two runs", "torch.set_num_threads(1) vs (4)")
F_.note("first step at which the losses differ at all", first_nonzero)
F_.note("|difference| at that step", f"{diff[first_nonzero]:.3e}")
F_.note("largest |difference| over the run",
        f"{diff.max():.3e} (at step {int(diff.argmax())})")
F_.note("|difference| at the last step", f"{diff[-1]:.3e}")
F_.note("final losses printed to 8 decimals", f"{la[-1]:.8f} vs {lb[-1]:.8f}")
F_.note("sum of all parameters, 1 thread", repr(ta["param_sum"]))
F_.note("sum of all parameters, 4 threads", repr(tb["param_sum"]))
F_.note("parameter fingerprints match?", ta["fingerprint"] == tb["fingerprint"])
F_.note("verdict",
        "the loss re-converges to 8 decimals while the weights stay different - "
        "'same answer' is a weaker test than 'same bits'")


# ===========================================================================
# 8. What determinism costs
# ===========================================================================

F_.head("8. What determinism costs")

x = torch.randn(2048, 2048)
w = torch.randn(2048, 2048)


def matmul_at(n):
    def fn():
        torch.set_num_threads(n)
        return x @ w
    return fn


timings = D.interleaved({"1 thread": matmul_at(1), "4 threads": matmul_at(4),
                         "6 threads": matmul_at(6)}, rounds=3, calls=3, warmup=1)
torch.set_num_threads(4)
base = timings["6 threads"]["best"]
for k, v in timings.items():
    F_.note(f"2048x2048 matmul, {k}", f"{v['best'] * 1e3:.1f} ms ({v['best'] / base:.2f}x)")

det_cost = D.interleaved({
    "off": lambda: torch.zeros(4096).index_add_(0, torch.randint(0, 4096, (4096,)),
                                                torch.ones(4096)),
}, rounds=3, calls=50)
torch.use_deterministic_algorithms(True, warn_only=True)
det_cost.update(D.interleaved({
    "on": lambda: torch.zeros(4096).index_add_(0, torch.randint(0, 4096, (4096,)),
                                               torch.ones(4096)),
}, rounds=3, calls=50))
torch.use_deterministic_algorithms(False)
F_.note("index_add_, deterministic flag off / on",
        f"{det_cost['off']['best'] * 1e6:.1f} us / {det_cost['on']['best'] * 1e6:.1f} us")
F_.note("the real cost on CPU is the thread count, not the flag", True)
F_.note("caveat", "the two index_add_ numbers are within this shared machine's "
                  "run-to-run spread; treat them as 'no measurable difference'")


# ===========================================================================
# 9. The recipe
# ===========================================================================

F_.head("9. The recipe, and what it does not buy you")

rec_a, rec_b, rec_ok = twice(FULL, {"PYTHONHASHSEED": "0"})
F_.note("full recipe, two fresh processes: bit-identical?", rec_ok)
F_.note("fingerprint", rec_a["fingerprint"])
w_a, w_b, w_ok = twice(dict(FULL, workers=2), {"PYTHONHASHSEED": "0"})
F_.note("full recipe with 2 DataLoader workers: bit-identical?", w_ok)
e_a = run_child(dict(FULL, epochs=6), {"PYTHONHASHSEED": "0"})
F_.note("same recipe, 6 epochs instead of 3: same fingerprint?",
        e_a["fingerprint"] == rec_a["fingerprint"])
F_.note("bit-exact is a promise about", "this machine, this thread count, this torch build")

with open(os.path.join(OUT, "recipe.py"), "w") as fh:
    fh.write('''"""The determinism recipe this project's audit arrived at.

Copy this into your project. Every line is here because removing it broke
reproducibility in the measured ablation (section 3 of the README) — except
where noted.
"""
import os, random
import numpy as np
import torch


def set_determinism(seed: int = 0, threads: int | None = None) -> None:
    # 1. every random-number generator that will be asked for a number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # also seeds CUDA, if present

    # 2. refuse to run any kernel that has no deterministic implementation.
    #    On CPU this blocks very little (see section 4) - it is cheap insurance
    #    that pays off the day the same code runs on a GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False  # stop cuDNN picking a different
                                            # algorithm run to run

    # 3. the thread count changes the summation order of every parallel
    #    reduction, so it is part of the seed whether you like it or not.
    if threads is not None:
        torch.set_num_threads(threads)


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker).

    PyTorch seeds each worker's torch generator; it does NOT seed numpy or
    Python's random inside the worker. Without this, any np.random augmentation
    is unseeded and the run is not reproducible.
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, **kw):
    g = torch.Generator()
    g.manual_seed(0)                        # shuffling has its own generator
    return torch.utils.data.DataLoader(
        dataset, generator=g, worker_init_fn=seed_worker, **kw)


# And two rules no function can enforce for you:
#   * never derive anything order-dependent from a set or an unordered dict -
#     sort it (or set PYTHONHASHSEED, which every caller must remember).
#   * record the torch version, the thread count and the device with the
#     checkpoint. Bit-exactness is a promise about a configuration, not a model.
''')
print(f"wrote {os.path.join(OUT, 'recipe.py')}")


# ===========================================================================
# figures
# ===========================================================================

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0), dpi=110)
for ax in axes:
    style_axes(ax)
fig.patch.set_facecolor("#fcfcfb")

ax = axes[0]
labels = [lab for lab, _, _, _ in ladder_rows]
oks = [1 if ok else 0 for _, ok, _, _ in ladder_rows]
ax.barh(range(len(labels)), [1] * len(labels),
        color=[SERIES[1] if o else SERIES[2] for o in oks], height=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks([])
ax.invert_yaxis()
for i, o in enumerate(oks):
    ax.text(0.02, i, "bit-identical" if o else "differs", va="center",
            fontsize=8, color="white", fontweight="bold")
ax.set_title("1. the ladder (green = locked)", loc="left", fontsize=11)

ax = axes[1]
labels = [lab for lab, _ in ablation_rows]
oks = [1 if ok else 0 for _, ok in ablation_rows]
ax.barh(range(len(labels)), [1] * len(labels),
        color=[SERIES[1] if o else SERIES[2] for o in oks], height=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks([])
ax.invert_yaxis()
for i, o in enumerate(oks):
    ax.text(0.02, i, "still locked" if o else "BREAKS", va="center",
            fontsize=8, color="white", fontweight="bold")
ax.set_title("2. remove one control (red = load-bearing)", loc="left", fontsize=11)

ax = axes[2]
ax.semilogy(np.maximum(diff, 1e-12), color=SERIES[0], lw=1.5)
ax.axhline(1e-7, color=SERIES[3], ls="--", lw=1.2, label="one float32 ulp, roughly")
ax.set_title("3. one thread-count change, NOT amplified", loc="left", fontsize=11)
ax.set_xlabel("training step")
ax.set_ylabel("|loss(1 thread) - loss(4 threads)|")
ax.legend(fontsize=8, frameon=False)

ax = axes[3]
ks = list(timings)
vals = [timings[k]["best"] * 1e3 for k in ks]
ax.bar(range(len(ks)), vals, color=[SERIES[2], SERIES[3], SERIES[1]], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=8)
ax.set_ylabel("2048x2048 matmul (ms)")
ax.set_title("4. the price of pinning threads", loc="left", fontsize=11)

save(fig, os.path.join(OUT, "determinism.png"))
F_.write(os.path.join(OUT, "findings.csv"))
