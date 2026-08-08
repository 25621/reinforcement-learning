"""Project 12 — Module introspection.

Walk a real pretrained model (torchvision's ResNet-18) with `named_modules()`
and answer, with numbers, the questions everybody eventually asks about a model
they did not write:

  1. what is the tree actually shaped like
  2. where do the 11.7 million parameters really live
  3. parameters vs buffers vs state_dict — why the three counts differ
  4. how `nn.Module` registers things, and the two ways to hide a layer from it
  5. two ways of counting that disagree: reused modules and tied weights
  6. trainable vs frozen, and the memory arithmetic of a training step

Runs in about 10 seconds on CPU. Downloads ResNet-18 weights (45 MB) once.
"""

import csv
import os
import sys
from collections import Counter, OrderedDict

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


def rec(k, v):
    FINDINGS[k] = v
    return v


def human(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} k"
    return str(n)


def load_model():
    return torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)


# =========================================================================
# 1. the shape of the tree
# =========================================================================
def the_tree(model):
    print("=" * 78)
    print("1. THE TREE")
    print("=" * 78)

    children = list(model.named_children())
    modules = list(model.named_modules())
    leaves = [(n, m) for n, m in modules if not list(m.children())]
    containers = [(n, m) for n, m in modules if list(m.children())]

    print(f"  named_children()  {len(children):>4}   direct children only, one level deep")
    print(f"  named_modules()   {len(modules):>4}   the whole tree, including the model itself")
    print(f"  ...of which leaves{len(leaves):>4}   modules with no children = the ones that compute")
    print(f"  ...of which inner {len(containers):>4}   containers: they only hold other modules")
    print()
    rec("named_children", len(children))
    rec("named_modules", len(modules))
    rec("leaf_modules", len(leaves))

    print("  the ten direct children (this is `print(model)` without the noise):")
    for n, m in children:
        own = sum(p.numel() for p in m.parameters())
        print(f"    {n:<10} {type(m).__name__:<20} {human(own):>9} params")
    print()

    print("  what the leaves are made of:")
    for t, c in Counter(type(m).__name__ for _, m in leaves).most_common():
        print(f"    {t:<20} {c:>3}")
    print()

    depths = [n.count(".") + 1 for n, _ in modules[1:]]
    print(f"  deepest path: {max(depths)} levels, e.g. "
          f"{max((n for n, _ in modules), key=lambda s: s.count('.'))}")
    rec("max_depth", max(depths))

    # A name is a path you can walk back down.
    name = "layer4.1.conv2"
    got = model.get_submodule(name)
    print(f"  get_submodule('{name}') -> {type(got).__name__}, weight {tuple(got.weight.shape)}")
    print()
    return leaves


# =========================================================================
# 2. where the parameters live
# =========================================================================
def where_the_parameters_are(model):
    print("=" * 78)
    print("2. WHERE THE 11.7 M PARAMETERS ACTUALLY LIVE")
    print("=" * 78)

    total = sum(p.numel() for p in model.parameters())
    by_type = Counter()
    for name, p in model.named_parameters():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        by_type[type(owner).__name__] += p.numel()

    print(f"  total: {human(total)} parameters = {total * 4 / 1e6:.1f} MB in float32\n")
    print(f"  {'layer type':<16}{'params':>12}{'share':>9}")
    for t, c in by_type.most_common():
        print(f"  {t:<16}{human(c):>12}{100 * c / total:>8.2f}%")
    print()
    rec("total_params", total)
    for t, c in by_type.items():
        rec(f"params_{t}", c)

    # Per stage, so the figure has something to say.
    stages = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    per_stage = []
    for s in stages:
        mod = model.get_submodule(s)
        per_stage.append((s, sum(p.numel() for p in mod.parameters())))
    print("  per stage (each `layer` doubles the channels and quarters the map):")
    for s, c in per_stage:
        bar = "#" * max(1, round(40 * c / total))
        print(f"    {s:<8}{human(c):>10}  {bar}")
    print()
    print("  The last stage alone holds more than half the model: doubling channels")
    print("  quadruples a conv kernel's parameter count (in_ch x out_ch), while the")
    print("  feature map it runs on only shrinks by 4x. Parameters pile up at the")
    print("  deep end; compute stays roughly flat across stages.")
    print()

    fc = sum(p.numel() for p in model.fc.parameters())
    print(f"  the classifier head `fc` (512 -> 1000): {human(fc)} = {100 * fc / total:.1f}% of the model")
    print("  -- one Linear layer costs as much as an entire early conv stage.")
    print()
    rec("fc_params", fc)
    return per_stage, by_type


# =========================================================================
# 3. parameters vs buffers vs state_dict
# =========================================================================
def parameters_buffers_statedict(model):
    print("=" * 78)
    print("3. PARAMETERS, BUFFERS, STATE_DICT: THREE COUNTS, THREE MEANINGS")
    print("=" * 78)

    n_par = len(list(model.named_parameters()))
    n_buf = len(list(model.named_buffers()))
    n_sd = len(model.state_dict())
    print(f"  named_parameters()  {n_par:>4} tensors   trained by the optimizer")
    print(f"  named_buffers()     {n_buf:>4} tensors   part of the model, NOT trained")
    print(f"  state_dict()        {n_sd:>4} tensors   = {n_par} + {n_buf}, everything you must save")
    print()
    rec("n_param_tensors", n_par)
    rec("n_buffer_tensors", n_buf)
    rec("n_state_dict_keys", n_sd)

    bn = model.get_submodule("layer1.0.bn1")
    print("  one BatchNorm2d, opened up:")
    for n, p in bn.named_parameters():
        print(f"    parameter  {n:<20} {str(tuple(p.shape)):<8} {p.dtype}  requires_grad={p.requires_grad}")
    for n, b in bn.named_buffers():
        print(f"    buffer     {n:<20} {str(tuple(b.shape)):<8} {b.dtype}")
    print()
    print("  `weight` and `bias` are learned from gradients.")
    print("  `running_mean` / `running_var` are the mean and variance of everything")
    print("  the layer has seen, updated by *counting*, not by gradient descent.")
    print("  `num_batches_tracked` is a single int64 -- a non-float in your checkpoint.")
    print()

    buf_elems = sum(b.numel() for b in model.buffers())
    print(f"  buffers hold {human(buf_elems)} numbers -- {100 * buf_elems / FINDINGS['total_params']:.3f}% of the")
    print("  parameter count, and leaving them out of a checkpoint destroys the")
    print("  model anyway (project 16 measures exactly how much).")
    print()
    rec("buffer_elements", buf_elems)

    # Why a buffer and not a parameter, demonstrated rather than asserted.
    x = torch.randn(8, 64, 8, 8)
    bn2 = nn.BatchNorm2d(64)
    before = bn2.running_mean.clone()
    bn2.train()
    bn2(x)
    moved_train = (bn2.running_mean - before).abs().max().item()
    bn2.eval()
    before2 = bn2.running_mean.clone()
    bn2(x)
    moved_eval = (bn2.running_mean - before2).abs().max().item()
    print(f"  running_mean change after one forward in train(): {moved_train:.6f}")
    print(f"  running_mean change after one forward in eval() : {moved_eval:.6f}")
    print("  No optimizer was involved either time. That is what makes it a buffer:")
    print("  state the *forward pass* writes, so no gradient can reach it.")
    print()
    rec("bn_running_mean_moved_train", moved_train)
    rec("bn_running_mean_moved_eval", moved_eval)

    # eval() vs train() is a flag, and you can see it on the tree.
    model.eval()
    flags_eval = set(m.training for m in model.modules())
    model.train()
    flags_train = set(m.training for m in model.modules())
    model.eval()
    print(f"  model.eval()  -> .training is {flags_eval} on all {len(list(model.modules()))} modules")
    print(f"  model.train() -> .training is {flags_train}")
    print("  `.eval()` is not a mode switch inside PyTorch. It sets one boolean on")
    print("  every module in the tree, and the few layers that care (BatchNorm,")
    print("  Dropout) read it themselves in their own forward.")
    print()


# =========================================================================
# 4. how registration works, and the two ways to lose a layer
# =========================================================================
class Broken(nn.Module):
    """Three sub-layers in a plain Python list, and one plain tensor."""

    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(16, 16) for _ in range(3)]   # invisible
        self.scale = torch.ones(16)                            # invisible

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x * self.scale


class Fixed(nn.Module):
    """The same thing, registered properly."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
        self.register_buffer("scale", torch.ones(16))

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x * self.scale


def the_registry():
    print("=" * 78)
    print("4. `nn.Module` IS A REGISTRY, AND IT ONLY SEES WHAT YOU HAND IT")
    print("=" * 78)

    lin = nn.Linear(4, 3)
    print("  what a Linear keeps in its three private dicts:")
    print(f"    _parameters {list(lin._parameters)}")
    print(f"    _buffers    {list(lin._buffers)}")
    print(f"    _modules    {list(lin._modules)}")
    print()
    print("  `nn.Module.__setattr__` looks at what you assign. An nn.Parameter goes")
    print("  into _parameters, an nn.Module into _modules, and anything registered")
    print("  with register_buffer into _buffers. Everything else is set as an")
    print("  ordinary Python attribute and is invisible to the framework.")
    print()

    b, f = Broken(), Fixed()
    for label, m in (("Broken (plain list + plain tensor)", b), ("Fixed (ModuleList + buffer)", f)):
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {label:<36} parameters(): {n_p:>5}   state_dict(): {len(m.state_dict()):>2} keys")
    print()
    print("  Both models run. Both give the same answer. But `Broken` reports zero")
    print("  parameters, so:")
    print("    - the optimizer gets an empty list and nothing ever trains")
    print("    - state_dict() saves nothing, and the checkpoint is an empty shell")
    print("    - .to('cuda') moves nothing, and the forward pass crashes on device mismatch")
    print()
    rec("broken_params", sum(p.numel() for p in b.parameters()))
    rec("fixed_params", sum(p.numel() for p in f.parameters()))

    try:
        torch.optim.SGD(b.parameters(), lr=0.1)
        print("  torch.optim.SGD(Broken().parameters()) -> no error raised")
    except ValueError as e:
        print(f"  torch.optim.SGD(Broken().parameters()) raises: {e}")
    print("  -- this one PyTorch does catch. The silent version is a model where")
    print("  *some* layers are in a plain list: the optimizer gets a non-empty list,")
    print("  training looks fine, and those layers never move.")
    print()


# =========================================================================
# 5. two counts that disagree
# =========================================================================
class Tied(nn.Module):
    """A tiny language model with the classic input/output weight tying."""

    def __init__(self, vocab=1000, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.out.weight = self.emb.weight        # one tensor, two names

    def forward(self, idx):
        return self.out(self.emb(idx))


def counts_that_disagree(model, leaves):
    print("=" * 78)
    print("5. TWO WAYS OF COUNTING THAT DISAGREE")
    print("=" * 78)

    # (a) a module can be called more than once per forward
    calls = Counter()
    handles = [m.register_forward_hook(lambda mod, i, o, n=n: calls.update([n]))
               for n, m in leaves]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for h in handles:
        h.remove()

    repeats = {n: c for n, c in calls.items() if c > 1}
    print(f"  leaf modules in the tree      : {len(leaves)}")
    print(f"  leaf modules that ran         : {len(calls)}")
    print(f"  forward calls that happened   : {sum(calls.values())}")
    print(f"  modules called more than once : {len(repeats)}  (all of them ReLU)")
    print()
    print("  Every BasicBlock builds ONE `self.relu` in __init__ and calls it TWICE")
    print("  in forward -- once after bn1, once after the residual add. ReLU has no")
    print("  parameters, so reusing it is free and torchvision does exactly that.")
    print()
    print("  Consequence: the tree tells you what a model *owns*, never what it")
    print("  *does*. If you need the sequence of operations, you have to run it")
    print("  (project 13 does; a forward hook on layer1.0.relu fires twice).")
    print()
    rec("leaf_calls_total", sum(calls.values()))
    rec("modules_called_twice", len(repeats))

    # (b) two names, one tensor
    t = Tied()
    p_count = sum(p.numel() for p in t.parameters())
    sd_count = sum(v.numel() for v in t.state_dict().values())
    shared = t.emb.weight.data_ptr() == t.out.weight.data_ptr()
    print(f"  a weight-tied model:  emb.weight and out.weight share storage: {shared}")
    print(f"    sum over parameters()          : {p_count:>7}")
    print(f"    sum over state_dict().values() : {sd_count:>7}   <- counts it twice")
    print(f"    parameter tensors              : {len(list(t.parameters())):>7}   one object")
    print(f"    state_dict keys                : {len(t.state_dict()):>7}   {list(t.state_dict())}")
    print()
    print("  `parameters()` de-duplicates by object identity, because handing the")
    print("  same tensor to an optimizer twice would apply every update twice.")
    print("  `state_dict()` does not de-duplicate: it is a *file format*, and a")
    print("  loader that only got one of the two keys could not fill in the other.")
    print()
    print(f"  So this model is honestly {human(p_count)} parameters, and a script that adds up")
    print(f"  state_dict entries will report {human(sd_count)} -- a 2x error on the headline number.")
    print()
    rec("tied_parameters_sum", p_count)
    rec("tied_state_dict_sum", sd_count)


# =========================================================================
# 6. trainable vs frozen, and what a training step really costs
# =========================================================================
def trainable_and_memory(model):
    print("=" * 78)
    print("6. TRAINABLE VS FROZEN, AND THE MEMORY OF A TRAINING STEP")
    print("=" * 78)

    total = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc.parameters():
        p.requires_grad_(True)

    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_n = total - train_n
    print("  a linear probe: freeze the trunk, train the head only")
    print(f"    trainable {human(train_n):>9}  ({100 * train_n / total:.1f}%)")
    print(f"    frozen    {human(frozen_n):>9}  ({100 * frozen_n / total:.1f}%)")
    print()
    print("  This is the one-line filter every fine-tuning script has:")
    print("    optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=...)")
    print("  Leave the filter out and the optimizer allocates state for 11.7 M")
    print("  parameters whose .grad stays None forever.")
    print()
    rec("probe_trainable", train_n)

    print(f"  {'what':<28}{'bytes':>12}")
    print(f"  {'parameters (fp32)':<28}{total * 4 / 1e6:>10.1f} MB")
    print(f"  {'+ gradients':<28}{total * 4 / 1e6:>10.1f} MB    one per trainable parameter")
    print(f"  {'+ AdamW state (2 moments)':<28}{2 * total * 4 / 1e6:>10.1f} MB")
    print(f"  {'= full fine-tune':<28}{4 * total * 4 / 1e6:>10.1f} MB    before a single activation")
    print(f"  {'= linear probe':<28}{(total + 3 * train_n) * 4 / 1e6:>10.1f} MB")
    print()
    print("  Four copies of the model, and the activations are extra (project 27).")
    print("  That factor of four is why 'it fits for inference' says nothing about")
    print("  whether it fits for training.")
    print()
    rec("full_finetune_mb", 4 * total * 4 / 1e6)
    rec("probe_mb", (total + 3 * train_n) * 4 / 1e6)

    for p in model.parameters():
        p.requires_grad_(True)


# =========================================================================
# figures
# =========================================================================
def figures(model, per_stage, by_type):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    names = [s for s, _ in per_stage]
    vals = [c / 1e6 for _, c in per_stage]
    ax.bar(range(len(names)), vals, color=ps.SERIES[0], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", color=ps.INK_SECONDARY, fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8)
    ax.set_title("Parameters per stage (millions)", color=ps.INK, fontsize=11, loc="left", pad=10)
    ax.set_ylabel("million parameters", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)

    ax = axes[1]
    order = by_type.most_common()
    labels = [t for t, _ in order]
    vals2 = [c for _, c in order]
    ax.barh(range(len(labels))[::-1], vals2, color=[ps.SERIES[1], ps.SERIES[2], ps.SERIES[3]][:len(labels)],
            height=0.55)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    for i, v in enumerate(vals2):
        ax.text(v * 1.15, len(labels) - 1 - i, human(v), va="center",
                color=ps.INK_SECONDARY, fontsize=8)
    ax.grid(True, axis="x", color=ps.GRID, linewidth=0.8)
    ax.set_xlim(1e3, max(vals2) * 12)
    ax.set_title("Parameters by layer type (log scale)", color=ps.INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_xlabel("parameters", color=ps.INK_SECONDARY, fontsize=10)

    ps.save(fig, os.path.join(OUT, "parameter_budget.png"))

    # cumulative parameters along the walk
    fig, ax = ps.new_axes(7.2, 4.2)
    cum, running, tick_names, tick_pos = [], 0, [], []
    leaves = [(n, m) for n, m in model.named_modules() if not list(m.children())]
    for i, (n, m) in enumerate(leaves):
        running += sum(p.numel() for p in m.parameters(recurse=False))
        cum.append(running / 1e6)
    ax.plot(range(len(cum)), cum, color=ps.SERIES[0], linewidth=2.0)
    ax.fill_between(range(len(cum)), 0, cum, color=ps.SERIES[0], alpha=0.12)
    for stage, color in zip(["layer1", "layer2", "layer3", "layer4", "fc"], ps.SERIES[1:]):
        idx = [i for i, (n, _) in enumerate(leaves) if n.startswith(stage)]
        if idx:
            ax.axvline(idx[0], color=ps.INK_MUTED, linewidth=0.8, linestyle=":")
            ax.text(idx[0] + 0.4, max(cum) * 0.06, stage, color=ps.INK_SECONDARY,
                    fontsize=8, rotation=90)
    ps.finish(fig, ax, "Cumulative parameters as you walk the 52 leaf modules",
              "leaf module, in named_modules() order", "million parameters cumulative",
              os.path.join(OUT, "cumulative_parameters.png"))


def main():
    model = load_model()
    leaves = the_tree(model)
    per_stage, by_type = where_the_parameters_are(model)
    parameters_buffers_statedict(model)
    the_registry()
    counts_that_disagree(model, leaves)
    trainable_and_memory(model)
    figures(model, per_stage, by_type)

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")

    # A committed, readable dump of the whole walk.
    path = os.path.join(OUT, "resnet18_modules.txt")
    with open(path, "w") as fh:
        fh.write(f"{'name':<26}{'type':<20}{'own params':>12}  shapes\n")
        fh.write("-" * 100 + "\n")
        for n, m in model.named_modules():
            own = list(m.named_parameters(recurse=False))
            shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in own)
            fh.write(f"{(n or '<root>'):<26}{type(m).__name__:<20}"
                     f"{sum(v.numel() for _, v in own):>12}  {shapes}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
