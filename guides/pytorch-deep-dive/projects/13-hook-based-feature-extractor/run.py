"""Project 13 — Hook-based feature extractor.

Pull intermediate activations out of torchvision's pretrained ResNet-18 without
editing a single line of torchvision, using `register_forward_hook`.

  1. what a hook gives you: every stage's shape, channels and bytes
  2. the in-place trap: the tensor you captured is not the tensor that was there
  3. the reuse trap: one module, two calls, one surviving capture
  4. `model.forward(x)` runs the model and skips every hook
  5. the leak: a stored activation that still has a grad_fn keeps the graph
  6. backward hooks: gradient norms per stage, for free
  7. the payoff: which layer should you actually tap? (a linear probe says
     it is not the last one)

Runs in about 30 seconds on CPU. Downloads ResNet-18 weights (45 MB) once.
"""

import csv
import os
import sys
import weakref
from collections import OrderedDict, defaultdict

import types

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(4)
torch.manual_seed(0)
FINDINGS = OrderedDict()

TAPS = ["relu", "layer1", "layer2", "layer3", "layer4", "fc"]


def rec(k, v):
    FINDINGS[k] = v
    return v


def plain_block_forward(self, x):
    """torchvision's BasicBlock.forward with `out += identity` deleted."""
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return self.relu(out)


def load_model():
    return torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()


# =========================================================================
# 1. what a hook gives you
# =========================================================================
def what_a_hook_gives(model):
    print("=" * 78)
    print("1. SIX LINES TO SEE INSIDE A MODEL YOU DID NOT WRITE")
    print("=" * 78)

    feats = {}

    def make_hook(name):
        def hook(module, inputs, output):
            feats[name] = output.detach().clone()
        return hook

    handles = [model.get_submodule(n).register_forward_hook(make_hook(n)) for n in TAPS]

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)

    print(f"  input {tuple(x.shape)}   ->   output {tuple(logits.shape)}\n")
    print(f"  {'tap':<9}{'output shape':<24}{'channels':>9}{'map':>9}{'MB / image':>12}")
    rows = []
    for n in TAPS:
        t = feats[n]
        mb = t[0].numel() * 4 / 1e6
        shape = tuple(t.shape)
        ch = shape[1] if t.dim() == 4 else shape[1]
        mp = f"{shape[2]}x{shape[3]}" if t.dim() == 4 else "-"
        print(f"  {n:<9}{str(shape):<24}{ch:>9}{mp:>9}{mb:>12.3f}")
        rows.append((n, ch, mp, mb))
    print()
    print("  Every stage halves the spatial map and doubles the channels. Halving")
    print("  height and width divides the map by 4 while doubling channels multiplies")
    print("  by 2, so each stage costs HALF the bytes of the one before: 3.21 MB after")
    print("  the stem, 0.10 MB at layer4 -- 32x smaller. A deep feature is small AND")
    print("  abstract, which is why people cache 512 numbers per image, not the image.")
    print()
    for h in handles:
        h.remove()
    print(f"  handles removed: len(model.relu._forward_hooks) == {len(model.relu._forward_hooks)}")
    print("  The handle is the only way back. Lose it and the hook fires forever,")
    print("  including inside somebody else's evaluation loop.")
    print()
    rec("stem_mb_per_image", rows[0][3])
    rec("layer1_mb_per_image", rows[1][3])
    rec("layer4_mb_per_image", rows[4][3])
    return rows


# =========================================================================
# 2. the in-place trap
# =========================================================================
def inplace_trap(model):
    print("=" * 78)
    print("2. THE TENSOR YOU CAPTURED IS NOT THE TENSOR THAT WAS THERE")
    print("=" * 78)

    naive, safe = {}, {}
    bn = model.get_submodule("layer1.0.bn1")
    h1 = bn.register_forward_hook(lambda m, i, o: naive.__setitem__("bn1", o))
    h2 = bn.register_forward_hook(lambda m, i, o: safe.__setitem__("bn1", o.detach().clone()))

    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    h1.remove()
    h2.remove()

    n, s = naive["bn1"], safe["bn1"]
    frac_naive = (n < 0).float().mean().item()
    frac_safe = (s < 0).float().mean().item()
    diff = (n - s).abs().max().item()

    print("  hook on layer1.0.bn1 -- a BatchNorm, whose output is roughly zero-mean:")
    print(f"    stored the reference       : {100 * frac_naive:5.1f}% of values are negative")
    print(f"    stored .detach().clone()   : {100 * frac_safe:5.1f}% of values are negative")
    print(f"    max |difference|           : {diff:.3f}")
    print()
    print(f"  torchvision builds its ReLU as nn.ReLU(inplace={model.layer1[0].relu.inplace}). The very next")
    print("  line of BasicBlock.forward overwrites bn1's output buffer in place. The")
    print("  hook fired first and stored a *reference*; by the time you look at it,")
    print("  the negatives have been clipped to zero and you are holding post-ReLU")
    print("  values under the name 'bn1'.")
    print()
    print("  Nothing raises. The shape is right, the dtype is right, and the numbers")
    print("  are the wrong layer's. Always `.detach().clone()` in a hook.")
    print()
    rec("inplace_frac_negative_naive", frac_naive)
    rec("inplace_frac_negative_cloned", frac_safe)
    rec("inplace_max_diff", diff)


# =========================================================================
# 3. the reuse trap
# =========================================================================
def reuse_trap(model):
    print("=" * 78)
    print("3. ONE MODULE, TWO CALLS, ONE SURVIVING CAPTURE")
    print("=" * 78)

    last = {}
    every = defaultdict(list)
    tap = model.get_submodule("layer1.0.relu")
    h1 = tap.register_forward_hook(lambda m, i, o: last.__setitem__("relu", o.detach().clone()))
    h2 = tap.register_forward_hook(lambda m, i, o: every["relu"].append(o.detach().clone()))

    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    h1.remove()
    h2.remove()

    print(f"  hook on layer1.0.relu fired {len(every['relu'])} times in one forward pass")
    a, b = every["relu"]
    print(f"    call 1 mean {a.mean():.4f}   max {a.max():.3f}   (after bn1)")
    print(f"    call 2 mean {b.mean():.4f}   max {b.max():.3f}   (after the residual add)")
    print(f"    the dict-style hook kept call 2: {torch.equal(last['relu'], b)}")
    print()
    print("  This is project 12's finding arriving in practice: BasicBlock creates one")
    print("  `self.relu` and calls it twice. `features[name] = output` silently keeps")
    print("  whichever call happened last -- and the two are different activations")
    print("  from different depths.")
    print()
    print("  Append to a list, or hook a module that only runs once.")
    print()
    rec("relu_fired", len(every["relu"]))
    rec("relu_call1_mean", a.mean().item())
    rec("relu_call2_mean", b.mean().item())


# =========================================================================
# 4. forward() skips hooks
# =========================================================================
def forward_skips_hooks(model):
    print("=" * 78)
    print("4. `model.forward(x)` RUNS THE MODEL AND SKIPS EVERY HOOK")
    print("=" * 78)

    seen = []
    h = model.register_forward_hook(lambda m, i, o: seen.append(1))
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        y1 = model(x)              # __call__
        n_call = len(seen)
        y2 = model.forward(x)      # forward, directly
        n_fwd = len(seen) - n_call
    h.remove()

    print("  a hook on the top-level model:")
    print(f"    model(x)          -> hook fired {n_call} time(s)")
    print(f"    model.forward(x)  -> hook fired {n_fwd} time(s)")
    print(f"    same output either way: {torch.equal(y1, y2)}")
    print()
    print("  `nn.Module.__call__` is not a synonym for `forward`. It is a wrapper that")
    print("  runs the pre-forward hooks, calls forward, runs the forward hooks, and")
    print("  arranges the backward hooks. Call `forward` yourself and you skip all of")
    print("  that machinery -- while the model still returns exactly the right answer,")
    print("  so nothing looks broken.")
    print()
    print("  Note the hook has to be on the module you bypassed. Hooks on *inner*")
    print("  modules still fire, because ResNet's own forward reaches them the normal")
    print("  way, through their __call__. That is what makes this bug so slippery:")
    print("  most of your feature dict fills in, and one entry is missing.")
    print()
    print("  This is the reason the docs say 'never call .forward() directly'.")
    print()
    rec("hook_fires_on_call", n_call)
    rec("hook_fires_on_forward", n_fwd)


# =========================================================================
# 5. the leak
# =========================================================================
class ByteTracker:
    """Count the bytes autograd is currently holding for backward.

    Same trick as projects 8/10/11: the pack hook fires when a tensor is saved,
    and a weakref finalizer fires when the graph lets it go, so the counter goes
    both up and down.
    """

    def __init__(self, params):
        self.live = 0
        self.peak = 0
        self.skip = {p.data_ptr() for p in params}

    class Holder:
        __slots__ = ("t", "__weakref__")

        def __init__(self, t):
            self.t = t

    def _release(self, nbytes):
        self.live -= nbytes

    def pack(self, t):
        h = ByteTracker.Holder(t)
        if t.data_ptr() not in self.skip:
            nb = t.numel() * t.element_size()
            self.live += nb
            self.peak = max(self.peak, self.live)
            weakref.finalize(h, self._release, nb)
        return h

    def unpack(self, h):
        return h.t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        return self._ctx.__exit__(*a)


def graph_size(t, param_ptrs):
    """How many graph nodes are reachable from this tensor?

    `keep` is not decoration: torch hands out a fresh Python wrapper for a
    grad_fn on every access, and CPython recycles ids, so a walk that does not
    hold references undercounts badly.
    """
    if t.grad_fn is None:
        return 0
    keep, seen, stack, nodes = [], set(), [t.grad_fn], 0
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        keep.append(fn)
        seen.add(id(fn))
        nodes += 1
        for nxt, _ in fn.next_functions:
            stack.append(nxt)
    return nodes


def the_leak(model):
    print("=" * 78)
    print("5. THE HOOK THAT KEEPS THE WHOLE GRAPH ALIVE")
    print("=" * 78)

    params = list(model.parameters())
    param_ptrs = {p.data_ptr() for p in params}
    x = torch.randn(2, 3, 224, 224)

    bag = {}
    h = model.layer2.register_forward_hook(
        lambda m, i, o: bag.update(naive=o, detached=o.detach()))
    with ByteTracker(params) as tr:
        model(x)                       # grad mode ON: this builds a graph
    h.remove()

    saved_mb = tr.peak / 1e6
    feat_mb = bag["naive"].numel() * 4 / 1e6
    print(f"  one forward pass of batch 2 saves {saved_mb:.1f} MB of activations for backward")
    print(f"  the feature you wanted (layer2 output) is {feat_mb:.2f} MB\n")

    print(f"  {'what you stored':<20}{'grad_fn':<20}{'graph nodes reachable':>22}")
    for k in ("naive", "detached"):
        t = bag[k]
        gf = type(t.grad_fn).__name__ if t.grad_fn is not None else "None"
        n = graph_size(t, param_ptrs)
        print(f"  {k:<20}{gf:<20}{n:>22}")
        rec(f"graph_nodes_{k}", n)
    print()
    print(f"  Same numbers, same storage, same {feat_mb:.2f} MB of values. One of them also")
    print(f"  has {graph_size(bag['naive'], param_ptrs)} live autograd nodes hanging off it, and those nodes hold")
    print(f"  the {saved_mb:.1f} MB of saved activations this forward pass produced.")
    print()
    n_batches = 200
    print(f"  Loop over {n_batches} batches collecting features and the arithmetic is:")
    print(f"    features you meant to keep : {n_batches * feat_mb:8.1f} MB")
    print(f"    graphs you also kept       : {n_batches * saved_mb:8.1f} MB   "
          f"({saved_mb / feat_mb:.0f}x more)")
    print()
    print("  A tensor produced inside a forward pass carries a `grad_fn`, and a")
    print("  grad_fn references everything the backward pass would need. Keeping the")
    print("  number keeps the machine that made it. This is the 'OOM at step 50 but")
    print("  not step 1' from the debugging table, and it is the same bug as keeping")
    print("  `total_loss += loss` instead of `loss.item()`.")
    print()
    print("  `.detach()` returns a tensor sharing the same storage -- no copy, not one")
    print("  extra byte of values -- with no grad_fn, so the graph is free to go.")
    print()
    print("  If you also keep the tensor past the next forward pass, add `.clone()`")
    print("  for the reason in section 2: `.detach()` shares storage, so it does not")
    print("  protect you from an in-place overwrite.")
    print()
    rec("activation_mb_per_forward", saved_mb)
    rec("feature_mb", feat_mb)
    return saved_mb, feat_mb


# =========================================================================
# 6. backward hooks
# =========================================================================
def backward_hooks(model):
    print("=" * 78)
    print("6. BACKWARD HOOKS: A GRADIENT HEALTH CHECK IN EIGHT LINES")
    print("=" * 78)

    stages = ["maxpool", "layer1", "layer2", "layer3", "layer4"]

    def measure(plain):
        m = load_model()
        if plain:                       # same weights, one line of forward removed
            for mod in m.modules():
                if type(mod).__name__ == "BasicBlock":
                    mod.forward = types.MethodType(plain_block_forward, mod)
        g = {}
        handles = [m.get_submodule(n).register_full_backward_hook(
            lambda mod, gi, go, n=n: g.__setitem__(n, go[0].detach().norm().item()))
            for n in stages]
        torch.manual_seed(1)
        m(torch.randn(4, 3, 224, 224)).pow(2).mean().backward()
        for h in handles:
            h.remove()
        return g

    grads = measure(plain=False)
    plain = measure(plain=True)

    print("  ||dL/d(stage output)||, read in the order backward visits them:")
    print(f"  {'stage':<9}{'residual':>12}{'no residual':>14}")
    for k in reversed(stages):
        print(f"  {k:<9}{grads[k]:>12.4f}{plain[k]:>14.4f}")
    print()
    print(f"  stem / layer4 ratio:  residual {grads['maxpool'] / grads['layer4']:.1f}x   "
          f"plain {plain['maxpool'] / plain['layer4']:.1f}x")
    print()
    print("  Eight lines of hook, and you can see whether the gradient survives the")
    print("  trip. Here it does: it enters at layer4 and is *larger* by the time it")
    print("  reaches the stem. Nothing is vanishing.")
    print()
    print("  The control is the interesting part. The second column is the same model")
    print("  with the residual connections deleted -- literally one line, `out +=")
    print("  identity`, removed from every block. The gradients barely move.")
    print()
    print("  At 18 layers with BatchNorm everywhere, skip connections are not what is")
    print("  keeping this gradient alive; normalization already is. The textbook")
    print("  vanishing-gradient picture needs much more depth, no normalization, or")
    print("  both. Measure your own network rather than assuming the folklore applies")
    print("  to it -- which is exactly what the hook is for.")
    print()
    print("  Two practical notes:")
    print("   - `register_full_backward_hook`, not `register_backward_hook`. The old")
    print("     one fires per tensor operation rather than per module and reports")
    print("     partial gradients; it is deprecated for that reason.")
    print("   - a full backward hook on an in-place module raises. Try it on")
    print("     `model.relu` (nn.ReLU(inplace=True)) and you get:")
    try:
        m2 = load_model()
        h = m2.relu.register_full_backward_hook(lambda mod, gi, go: None)
        m2(torch.randn(1, 3, 224, 224))
        print("     (no error -- this torch version allows it)")
    except RuntimeError as e:
        print(f"       RuntimeError: {str(e).split('.')[0]}.")
    print("     The hook needs the module's real output to hand back; an in-place op")
    print("     overwrites it. Same root cause as section 2, raised loudly this time.")
    print()
    for k, v in grads.items():
        rec(f"gradnorm_{k}", v)
    for k, v in plain.items():
        rec(f"gradnorm_noresidual_{k}", v)
    return grads, plain


# =========================================================================
# 7. the payoff — which layer should you tap?
# =========================================================================
S = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def make_shapes(n, rng):
    """Small pale square / circle / triangle on a cluttered background."""
    X = np.zeros((n, 3, S, S), np.float32)
    y = np.zeros(n, np.int64)
    yy, xx = np.mgrid[0:S, 0:S]
    colors = np.array([[0.62, 0.55, 0.50], [0.55, 0.62, 0.50], [0.55, 0.50, 0.62]], np.float32)
    for i in range(n):
        s, c = rng.integers(3), rng.integers(3)
        bg = rng.uniform(0.15, 0.45, (3, 1, 1)).astype(np.float32) + np.zeros((3, S, S), np.float32)
        for _ in range(6):                                   # distractor blobs
            bx, by = rng.integers(0, S, 2)
            br = rng.integers(8, 22)
            m = ((xx - bx) ** 2 + (yy - by) ** 2) < br * br
            cc = rng.uniform(0.1, 0.6, 3).astype(np.float32)
            for ch in range(3):
                bg[ch][m] = cc[ch]
        cx, cy = rng.integers(30, S - 30, 2)
        r = rng.integers(9, 15)
        if s == 0:
            mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
        elif s == 1:
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        else:
            mask = (np.abs(xx - cx) < (r - (cy - yy + r) / 2)) & (yy > cy - r) & (yy < cy + r)
        for ch in range(3):
            bg[ch][mask] = colors[c, ch]
        X[i] = bg + rng.normal(0, 0.03, (3, S, S)).astype(np.float32)
        y[i] = s
    return torch.from_numpy((X - IMAGENET_MEAN) / IMAGENET_STD), torch.from_numpy(y)


def ridge_probe(ftr, ytr, fte, yte, lam=10.0):
    """Closed-form linear classifier. No training loop, no learning rate."""
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    Xa = torch.cat([a, torch.ones(len(a), 1)], 1).double()
    Y = torch.nn.functional.one_hot(ytr, 3).double() * 2 - 1
    W = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], dtype=torch.float64), Xa.T @ Y)
    Xb = torch.cat([b, torch.ones(len(b), 1)], 1).double()
    return ((Xb @ W).argmax(1) == yte).float().mean().item()


def which_layer(model):
    print("=" * 78)
    print("7. WHICH LAYER SHOULD YOU ACTUALLY TAP?")
    print("=" * 78)
    print("  Task: square vs circle vs triangle, 15-pixel shapes in three pale")
    print("  colours on a cluttered background. Chance is 0.333. ResNet-18 has never")
    print("  seen this task; we only read its features and fit a linear classifier.")
    print()

    feats = {}

    def make(name):
        def hook(m, i, o):
            feats[name] = o.detach() if o.dim() == 2 else o.detach().mean((2, 3))
        return hook

    handles = [model.get_submodule(n).register_forward_hook(make(n)) for n in TAPS]

    def extract(X):
        out = {n: [] for n in TAPS}
        with torch.no_grad():
            for i in range(0, len(X), 32):
                model(X[i:i + 32])
                for n in TAPS:
                    out[n].append(feats[n].clone())
        return {n: torch.cat(v) for n, v in out.items()}

    per_seed = defaultdict(list)
    dims = {}
    lam_table = defaultdict(dict)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Xtr, ytr = make_shapes(400, rng)
        Xte, yte = make_shapes(200, rng)
        Ftr, Fte = extract(Xtr), extract(Xte)
        for n in TAPS:
            dims[n] = Ftr[n].shape[1]
            per_seed[n].append(ridge_probe(Ftr[n], ytr, Fte[n], yte))
        if seed == 0:
            for lam in (1.0, 10.0, 100.0, 1000.0):
                for n in TAPS:
                    lam_table[lam][n] = ridge_probe(Ftr[n], ytr, Fte[n], yte, lam)
    for h in handles:
        h.remove()

    print(f"  {'tap':<9}{'dim':>6}{'accuracy':>11}{'+- (3 seeds)':>15}")
    for n in TAPS:
        a = np.array(per_seed[n])
        print(f"  {n:<9}{dims[n]:>6}{a.mean():>11.3f}{a.std():>15.3f}")
        rec(f"probe_{n}", a.mean())
        rec(f"probe_{n}_std", a.std())
    print()

    best = max(TAPS, key=lambda n: np.mean(per_seed[n]))
    print(f"  best tap: {best} at {np.mean(per_seed[best]):.3f}, against layer4's "
          f"{np.mean(per_seed['layer4']):.3f}")
    print()
    print("  The curve goes UP and then DOWN. The last convolutional stage -- the one")
    print("  everybody grabs, the one `avgpool` feeds -- is 0.20 WORSE than the middle")
    print("  of the network on this task.")
    print()
    print("  Nothing is broken. layer4's features are the ones ImageNet training")
    print("  rewarded: 'is this a golden retriever or a beagle'. Whether an outline is")
    print("  a triangle or a square is not an ImageNet distinction, so the deep layers")
    print("  are free to throw it away, and they do. The middle of the network still")
    print("  represents shape because it has to, in order to build categories later.")
    print()
    print("  Is that just the regularizer? Same ranking at every lambda:")
    print(f"  {'lambda':<9}" + "".join(f"{n:>9}" for n in TAPS))
    for lam in (1.0, 10.0, 100.0, 1000.0):
        print(f"  {lam:<9.0f}" + "".join(f"{lam_table[lam][n]:>9.3f}" for n in TAPS))
    print()
    print("  One honest wrinkle: `fc` scores above layer4, and it cannot possibly")
    print("  contain information layer4 does not -- the logits are a fixed linear map")
    print("  of exactly those pooled features. The gap is the probe's regularizer")
    print("  reacting to a different scaling, not extra information. Probe numbers")
    print("  always measure the probe as well as the features; compare taps under one")
    print("  fixed probe, and do not read small gaps as meaning.")
    print()
    return per_seed, dims, lam_table


# =========================================================================
# figures
# =========================================================================
def figures(rows, per_seed, grads, plain):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    names = [r[0] for r in rows]
    mb = [r[3] for r in rows]
    ax.bar(range(len(names)), mb, color=ps.SERIES[0], width=0.6)
    for i, v in enumerate(mb):
        ax.text(i, v + max(mb) * 0.02, f"{v:.3f}", ha="center",
                color=ps.INK_SECONDARY, fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8)
    ax.set_ylim(0, max(mb) * 1.2)
    ax.set_title("Activation bytes per image at each tap", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_ylabel("MB per image (float32)", color=ps.INK_SECONDARY, fontsize=10)

    ax = axes[1]
    m = [float(np.mean(per_seed[n])) for n in TAPS]
    sd = [float(np.std(per_seed[n])) for n in TAPS]
    ax.errorbar(range(len(TAPS)), m, yerr=sd, color=ps.SERIES[0], linewidth=2.0,
                marker="o", markersize=5, capsize=3, label="linear probe accuracy")
    ax.axhline(1 / 3, color=ps.SERIES[2], linewidth=1.4, linestyle="--", label="chance (0.333)")
    best = int(np.argmax(m))
    ax.annotate(f"best: {TAPS[best]}  {m[best]:.3f}", xy=(best, m[best]),
                xytext=(best - 0.2, m[best] + 0.07), color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(TAPS)))
    ax.set_xticklabels(TAPS, fontsize=9)
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.set_ylim(0.2, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Deeper is not better: shape probe by tap point", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_ylabel("test accuracy", color=ps.INK_SECONDARY, fontsize=10)

    ps.save(fig, os.path.join(OUT, "taps.png"))

    fig, ax = ps.new_axes(7.2, 4.0)
    keys = ["layer4", "layer3", "layer2", "layer1", "maxpool"]
    ax.plot(range(len(keys)), [grads[k] for k in keys], color=ps.SERIES[1],
            linewidth=2.0, marker="o", markersize=5, label="ResNet-18 (residual)")
    ax.plot(range(len(keys)), [plain[k] for k in keys], color=ps.SERIES[3],
            linewidth=2.0, marker="s", markersize=5, linestyle="--",
            label="same weights, residual add deleted")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "Gradient norm arriving at each stage, in backward order",
              "stage (backward visits them left to right)", "||dL / d output||  (log)",
              os.path.join(OUT, "gradient_norms.png"))


def main():
    model = load_model()
    rows = what_a_hook_gives(model)
    inplace_trap(model)
    reuse_trap(model)
    forward_skips_hooks(model)
    the_leak(model)
    grads, plain = backward_hooks(model)
    per_seed, dims, lam_table = which_layer(model)
    figures(rows, per_seed, grads, plain)

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")

    path = os.path.join(OUT, "probe_by_tap.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tap", "dim", "acc_mean", "acc_std"] + [f"lambda_{l:g}" for l in (1, 10, 100, 1000)])
        for n in TAPS:
            w.writerow([n, dims[n], f"{np.mean(per_seed[n]):.4f}", f"{np.std(per_seed[n]):.4f}"]
                       + [f"{lam_table[l][n]:.4f}" for l in (1.0, 10.0, 100.0, 1000.0)])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
