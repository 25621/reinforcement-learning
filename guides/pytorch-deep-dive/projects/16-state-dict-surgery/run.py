"""Project 16 — State dict surgery.

Load torchvision's pretrained ResNet-18 weights into a model with completely
different attribute names, and a smaller head, and verify it numerically.

  1. what a state_dict is: keys, shapes, and the two dtypes in it
  2. the rename: 122 keys remapped, output bit-identical
  3. strict=False, and the 60 keys it will not tell you about unless you look
  4. buffers: what dropping running_mean/running_var actually costs
  5. head surgery: 1000 classes -> 10, exactly
  6. the mapping bug shapes cannot catch
  7. the checkpoint hygiene list

Runs in about 15 seconds on CPU. Downloads ResNet-18 weights (45 MB) once.
"""

import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock

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

PROBE = torch.randn(24, 3, 160, 160)          # fixed inputs for every comparison


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# the target architecture: same computation, different names
# =========================================================================
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x))))


def make_stage(cin, cout, stride):
    down = None
    if stride != 1 or cin != cout:
        down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                             nn.BatchNorm2d(cout))
    return nn.Sequential(BasicBlock(cin, cout, stride, down),
                         BasicBlock(cout, cout))


class MyResNet(nn.Module):
    """ResNet-18 with our own names: stem / stages.N / head."""

    def __init__(self, n_classes=1000):
        super().__init__()
        self.stem = Stem()
        self.stages = nn.ModuleList([
            make_stage(64, 64, 1),
            make_stage(64, 128, 2),
            make_stage(128, 256, 2),
            make_stage(256, 512, 2),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return self.head(self.pool(x).flatten(1))


def remap(src):
    """torchvision key -> our key. Four rules, in order."""
    out = OrderedDict()
    for k, v in src.items():
        if k.startswith("conv1."):
            nk = k.replace("conv1.", "stem.conv.", 1)
        elif k.startswith("bn1."):
            nk = k.replace("bn1.", "stem.norm.", 1)
        elif k.startswith("layer"):
            i = int(k[5]) - 1
            nk = f"stages.{i}." + k.split(".", 1)[1]
        elif k.startswith("fc."):
            nk = k.replace("fc.", "head.", 1)
        else:
            nk = k
        out[nk] = v
    return out


def load_ref():
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    return m.eval()


def agreement(a, b, x=PROBE):
    """How often do two models pick the same class, and how far apart are the logits?"""
    with torch.no_grad():
        pa, pb = a(x), b(x)
    top1 = (pa.argmax(1) == pb.argmax(1)).float().mean().item()
    return top1, (pa - pb).abs().max().item()


# =========================================================================
# 1. what a state_dict is
# =========================================================================
def what_it_is(ref):
    print("=" * 78)
    print("1. WHAT A state_dict IS")
    print("=" * 78)

    sd = ref.state_dict()
    print(f"  type      : {type(sd).__name__}")
    print(f"  entries   : {len(sd)}")
    print(f"  dtypes    : {sorted({str(v.dtype) for v in sd.values()})}")
    print(f"  total MB  : {sum(v.numel() * v.element_size() for v in sd.values()) / 1e6:.1f}")
    print()
    for k in list(sd)[:6]:
        print(f"    {k:<34}{str(tuple(sd[k].shape)):<20}{sd[k].dtype}")
    print("    ...")
    print()
    print("  A plain ordered dict of tensors, keyed by the dotted path from")
    print("  project 12. There is no architecture in here: no layer types, no")
    print("  connection order, no forward pass. A checkpoint cannot tell you what")
    print("  model it came from -- only what shapes it expects to be poured into.")
    print()
    print("  That is exactly why surgery is possible at all, and also why a")
    print("  successful load proves nothing about correctness (section 6).")
    print()
    rec("state_dict_entries", len(sd))
    return sd


# =========================================================================
# 2. the rename
# =========================================================================
def the_rename(ref, sd):
    print("=" * 78)
    print("2. THE RENAME")
    print("=" * 78)

    mine = MyResNet().eval()
    mapped = remap(sd)
    print("  four rules:")
    print("    conv1.*   -> stem.conv.*")
    print("    bn1.*     -> stem.norm.*")
    print("    layerN.*  -> stages.{N-1}.*")
    print("    fc.*      -> head.*")
    print()
    same = set(mapped) == set(mine.state_dict())
    print(f"  key sets identical after remapping: {same}")
    result = mine.load_state_dict(mapped, strict=True)
    print(f"  load_state_dict(strict=True) -> missing {len(result.missing_keys)}, "
          f"unexpected {len(result.unexpected_keys)}")

    top1, mx = agreement(ref, mine)
    print(f"  top-1 agreement with torchvision : {top1:.3f}")
    print(f"  max |logit difference|           : {mx:.3e}")
    print()
    print("  Bit-identical. Not 'close' -- the same weights in the same order")
    print("  through the same operations. The names were never part of the model.")
    print()
    print("  Note what did NOT have to match: our Stem is one module where")
    print("  torchvision has four separate attributes, our stages live in a")
    print("  ModuleList instead of being four named attributes, and our head is")
    print("  called head. The state_dict only cares about the final dotted strings")
    print("  and the shapes behind them.")
    print()
    rec("rename_top1", top1)
    rec("rename_max_logit_diff", mx)
    return mine, mapped


# =========================================================================
# 3. strict=False
# =========================================================================
def strict_false(ref, mapped):
    print("=" * 78)
    print("3. strict=False, AND THE KEYS IT WILL NOT MENTION UNLESS YOU LOOK")
    print("=" * 78)

    # A realistic bug: someone forgot the stem rule.
    broken = OrderedDict()
    for k, v in mapped.items():
        broken[k.replace("stem.", "") if k.startswith("stem.") else k] = v

    mine = MyResNet().eval()
    result = mine.load_state_dict(broken, strict=False)
    print(f"  a mapping that forgot the stem rule:")
    print(f"    missing_keys    : {len(result.missing_keys)}   e.g. {result.missing_keys[:2]}")
    print(f"    unexpected_keys : {len(result.unexpected_keys)}   e.g. {result.unexpected_keys[:2]}")
    top1, mx = agreement(ref, mine)
    print(f"    top-1 agreement with torchvision: {top1:.3f}")
    print(f"    max |logit difference|          : {mx:.2f}")
    print()
    print("  It ran. It returned. It produced a thousand confident numbers per")
    print("  image, and it agrees with the real model on almost nothing, because")
    print("  the first convolution -- the one every pixel goes through -- is still")
    print("  at its random initialization.")
    print()
    print("  `strict=False` does not mean 'be lenient about small differences'. It")
    print("  means 'do not raise'. The information is all in the return value, and")
    print("  the return value is what nobody assigns:")
    print()
    print("    result = model.load_state_dict(sd, strict=False)")
    print("    assert not result.missing_keys, result.missing_keys")
    print()
    print("  Two lines. Use strict=True whenever you can, and when you cannot,")
    print("  assert on exactly the keys you meant to skip -- not on none of them.")
    print()
    rec("strict_false_missing", len(result.missing_keys))
    rec("strict_false_top1", top1)

    # shapes are checked either way
    bad = OrderedDict(mapped)
    bad["head.weight"] = torch.randn(10, 512)
    try:
        MyResNet().load_state_dict(bad, strict=False)
        print("  a shape mismatch under strict=False: accepted (!)")
    except RuntimeError as e:
        first = str(e).strip().split("\n")[1].strip() if "\n" in str(e) else str(e)
        print("  a shape mismatch under strict=False still RAISES:")
        print(f"    {first[:96]}")
    print()
    print("  So the guarantee is narrow but real: `strict` controls which KEYS must")
    print("  be present; shapes are always checked. Every silent failure in this")
    print("  project is a key problem or a semantic one, never a shape one.")
    print()
    return top1


# =========================================================================
# 4. buffers
# =========================================================================
def buffers_matter(ref, mapped):
    print("=" * 78)
    print("4. WHAT DROPPING THE BATCHNORM BUFFERS COSTS")
    print("=" * 78)

    rows = []
    variants = {
        "everything (122 keys)": lambda k: True,
        "drop num_batches_tracked": lambda k: not k.endswith("num_batches_tracked"),
        "drop running_mean/var": lambda k: "running_" not in k,
        "parameters only (62 keys)": lambda k: not ("running_" in k or k.endswith("num_batches_tracked")),
    }
    for vi, (label, keep) in enumerate(variants.items()):
        sub = OrderedDict((k, v) for k, v in mapped.items() if keep(k))
        mine = MyResNet().eval()
        res = mine.load_state_dict(sub, strict=False)
        top1, mx = agreement(ref, mine)
        print(f"  {label:<28} loaded {len(sub):>4} keys   top-1 {top1:>6.3f}   "
              f"max |logit diff| {mx:>8.2f}")
        rows.append((label, len(sub), top1, mx))
        rec(f"buffers_{vi}_{label.split()[0]}_top1", top1)
    print()
    print("  Dropping `num_batches_tracked` costs exactly nothing: in eval() it is")
    print("  never read, and even in train() it only matters if momentum=None.")
    print()
    print("  Dropping `running_mean` and `running_var` destroys the model, and it")
    print("  is worth being precise about why. In eval() BatchNorm computes")
    print("  (x - running_mean) / sqrt(running_var + eps) * weight + bias. A fresh")
    print("  BatchNorm has running_mean=0 and running_var=1, so a layer whose real")
    print("  activations are centred at, say, 4.0 with variance 0.01 gets left at 4.0")
    print("  instead of being pulled to 0 -- and then the learned `weight` and `bias`,")
    print("  which were fitted for the normalized version, are applied on top.")
    print()
    print("  60 tensors out of 122, 9.6k numbers out of 11.7M -- 0.08% of the")
    print("  checkpoint -- and without them the pretrained weights are worthless.")
    print("  This is the concrete answer to project 12's 'why is a buffer not just")
    print("  a plain attribute': a plain attribute would not be in this file.")
    print()
    return rows


# =========================================================================
# 5. head surgery
# =========================================================================
def head_surgery(ref, mapped):
    print("=" * 78)
    print("5. HEAD SURGERY: 1000 CLASSES -> 10, EXACTLY")
    print("=" * 78)

    keep = [207, 281, 285, 340, 388, 417, 555, 817, 949, 963]   # ten ImageNet classes
    small = MyResNet(n_classes=10).eval()

    sub = OrderedDict((k, v) for k, v in mapped.items() if not k.startswith("head."))
    sub["head.weight"] = mapped["head.weight"][keep].clone()
    sub["head.bias"] = mapped["head.bias"][keep].clone()
    res = small.load_state_dict(sub, strict=True)

    with torch.no_grad():
        full = ref(PROBE)[:, keep]
        cut = small(PROBE)
    print(f"  kept classes            : {keep}")
    print(f"  head.weight  1000x512 -> {tuple(sub['head.weight'].shape)}")
    print(f"  max |logit difference| vs the same 10 columns of the full model: "
          f"{(full - cut).abs().max().item():.3e}")
    print(f"  argmax-over-10 agreement: "
          f"{(full.argmax(1) == cut.argmax(1)).float().mean().item():.3f}")
    print()
    print("  A Linear layer's weight is (out_features, in_features), so row i is")
    print("  the entire recipe for class i. Selecting ten rows selects ten classes")
    print("  and changes nothing else -- the trunk never knew how many classes")
    print("  there were.")
    print()
    print("  The usual version of this operation throws the head away instead:")
    print()
    print("    sd = {k: v for k, v in sd.items() if not k.startswith('head.')}")
    print("    model.load_state_dict(sd, strict=False)   # head stays random")
    print()
    print("  Both are legitimate. Slicing keeps a working classifier for a subset;")
    print("  dropping gives you a fresh head for new classes. What you must not do")
    print("  is leave the 1000-way head in the file and hope -- that is a shape")
    print("  mismatch, and section 3 showed it raises.")
    print()
    rec("head_surgery_max_diff", (full - cut).abs().max().item())


# =========================================================================
# 6. the mapping bug shapes cannot catch
# =========================================================================
def semantic_bug(ref, mapped):
    print("=" * 78)
    print("6. THE MAPPING BUG THAT LOADS PERFECTLY")
    print("=" * 78)

    swapped = OrderedDict(mapped)
    a, b = "stages.3.0.conv2.weight", "stages.3.1.conv2.weight"
    print(f"  swap two keys of identical shape {tuple(mapped[a].shape)}:")
    print(f"    {a}  <->  {b}")
    swapped[a], swapped[b] = mapped[b], mapped[a]

    mine = MyResNet().eval()
    res = mine.load_state_dict(swapped, strict=True)
    top1, mx = agreement(ref, mine)
    print(f"  load_state_dict(strict=True) -> missing {len(res.missing_keys)}, "
          f"unexpected {len(res.unexpected_keys)}")
    print(f"  top-1 agreement with torchvision : {top1:.3f}")
    print(f"  max |logit difference|           : {mx:.2f}")
    print()
    print("  strict=True passed. Every key present, every shape correct, no")
    print("  warning of any kind -- and two 3x3x512x512 convolutions are doing each")
    print("  other's job.")
    print()
    print("  This is the failure mode that matters, because it is the one your")
    print("  tooling cannot see. Off-by-one in a layer index, a reversed enumerate,")
    print("  a regex that matched `layer1` inside `layer10` -- all of them produce")
    print("  shape-compatible nonsense.")
    print()
    print("  There is exactly one defence, and it is the whole point of this")
    print("  project: after any surgery, RUN BOTH MODELS ON THE SAME INPUT AND")
    print("  COMPARE. Section 2 got 0.000e+00. Anything else is a bug you have not")
    print("  found yet.")
    print()
    rec("swap_top1", top1)
    rec("swap_max_logit_diff", mx)
    return top1


# =========================================================================
# 7. hygiene
# =========================================================================
def hygiene(mine):
    print("=" * 78)
    print("7. CHECKPOINT HYGIENE")
    print("=" * 78)

    path = os.path.join(OUT, "demo_checkpoint.pt")
    torch.save({"model": mine.state_dict(), "arch": "MyResNet", "epoch": 0}, path)
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    print(f"  torch.save/torch.load(weights_only=True) round trip: keys {list(ckpt)}")
    os.remove(path)
    print()
    print("  `weights_only=True` is the default from torch 2.6 onwards, and the")
    print("  reason is that the old default used Python's pickle, which can execute")
    print("  arbitrary code while loading. A checkpoint from the internet was a")
    print("  program, not data. weights_only=True restricts the loader to tensors")
    print("  and plain containers.")
    print()
    print("  `map_location='cpu'` matters because a state_dict remembers which")
    print("  device each tensor was on. A checkpoint saved from cuda:3 tries to")
    print("  restore itself to cuda:3, on a machine that may have one GPU.")
    print()
    print("  A checkpoint worth keeping contains:")
    print("    model.state_dict()       the weights AND buffers (section 4)")
    print("    optimizer.state_dict()   momentum / Adam moments (project 14)")
    print("    scheduler.state_dict()   where you are on the LR curve")
    print("    the step or epoch number")
    print("    the RNG state            (project 17)")
    print("    the config that built the architecture")
    print()
    print("  That last line is the one people skip, and it is why old checkpoints")
    print("  become unloadable: the file has 122 anonymous tensors in it and no")
    print("  record of what produces those 122 shapes.")
    print()


# =========================================================================
# figure
# =========================================================================
def figure(rows, strict_top1, swap_top1, rename_top1):
    by = dict((r[0], r[2]) for r in rows)
    labels = ["correct rename\n(strict=True)", "forgot the stem rule\n(strict=False)",
              "swapped two conv\nweights (strict=True)", "no running mean/var\n(strict=False)",
              "no num_batches_tracked\n(strict=False)"]
    vals = [rename_top1, strict_top1, swap_top1,
            by["drop running_mean/var"], by["drop num_batches_tracked"]]

    fig, ax = ps.new_axes(9.0, 4.4)
    colors = [ps.SERIES[1] if v > 0.99 else ps.SERIES[2] for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, width=0.58)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ps.finish(fig, ax, "Every one of these loaded without raising an error",
              "", "top-1 agreement with the original model",
              os.path.join(OUT, "surgery_outcomes.png"))


def main():
    ref = load_ref()
    sd = what_it_is(ref)
    mine, mapped = the_rename(ref, sd)
    strict_top1 = strict_false(ref, mapped)
    rows = buffers_matter(ref, mapped)
    head_surgery(ref, mapped)
    swap_top1 = semantic_bug(ref, mapped)
    hygiene(mine)
    figure(rows, strict_top1, swap_top1, FINDINGS["rename_top1"])

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")

    path = os.path.join(OUT, "key_mapping.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["torchvision_key", "our_key", "shape"])
        for k, v in sd.items():
            w.writerow([k, list(remap({k: v}))[0], list(v.shape)])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
