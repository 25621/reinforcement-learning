"""Project 54 - reading `native_functions.yaml`, and testing what it claims.

The file is the table of contents for ATen. This project (a) finds the copy that
already exists on your disk, (b) reads five entries in full, (c) takes a census
of all of them, and then (d) does the part nobody does: checks the file's claims
against the library that is actually running.

Sections:
  1. the file is already on your machine
  2. anatomy of one entry, field by field
  3. the census: 3184 operators, sorted by how they are written
  4. five ops you use every day
  5. invariant 1: a composite op must not have a hand-written derivative
  6. invariant 2: what the file declares vs what the dispatcher loaded
  7. from a kernel name to the file that defines it
  8. answering a real question with tags

Run:  python3 run.py        (~1 minute)
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "53-trace-one-op-end-to-end"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import source_lib as S  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F = S.Findings()

from torchgen.model import DispatchKey  # noqa: E402

fns, backend_indices = S.native_functions()
by_name = S.native_functions_by_name()


# ===========================================================================
# 1. The file is already on your machine
# ===========================================================================
F.head("1. finding the file")

path = S.native_functions_yaml_path()
F.note("path", path.replace(os.path.expanduser("~"), "~"))
F.note("size (KB)", os.path.getsize(path) / 1024)
raw_text = open(path).read()
F.note("lines", raw_text.count("\n") + 1)

raw = yaml.safe_load(raw_text)
F.note("entries physically in the file", len(raw))
F.note("operators torchgen produces from it", len(fns))
generated_tag = sum(1 for f in fns if any(str(t) == "generated" for t in f.tags))
F.note("operators created by 'autogen:' lines", generated_tag)
F.note("entries carrying an autogen: line", sum(1 for f in fns if f.autogen))
F.note("check: entries + generated == operators", len(raw) + generated_tag == len(fns))

# What else ships alongside it.
for label, p in [
    ("tags.yaml", S.tags_yaml_path()),
    ("derivatives.yaml", S.derivatives_yaml_path()),
]:
    F.note(f"{label} present", os.path.exists(p))
    if os.path.exists(p):
        F.note(f"{label} size (KB)", os.path.getsize(p) / 1024)


# ===========================================================================
# 2. Anatomy of one entry
# ===========================================================================
F.head("2. anatomy of add.out")


def raw_entry(func_prefix: str) -> dict:
    for e in raw:
        if e["func"].startswith(func_prefix):
            return e
    raise KeyError(func_prefix)


anatomy = raw_entry("add.out(")
with open(os.path.join(OUT, "entry_add_out.yaml"), "w") as fh:
    yaml.safe_dump(anatomy, fh, sort_keys=False)
for field, value in anatomy.items():
    F.note(f"add.out field: {field}", str(value).replace("\n", " ")[:90])

parsed = by_name["add.out"]
F.note("parsed: structured", parsed.structured)
F.note("parsed: inherits", str(parsed.structured_inherits))
F.note("parsed: CPU kernel (filled in by ufunc_inner_loop)",
       backend_indices[DispatchKey.CPU].get_kernel(parsed).kernel)
F.note("dispatch: block mentions CPU", "CPU" in str(anatomy.get("dispatch", "")).split(",")[0])


# ===========================================================================
# 3. The census
# ===========================================================================
F.head("3. the census")

kinds = Counter(f.func.kind().name for f in fns)
for k, v in kinds.most_common():
    F.note(f"variant kind: {k}", v)
F.note("distinct base names", len({f.func.name.name.base for f in fns}))

style = Counter()
for f in fns:
    if f.has_composite_implicit_autograd_kernel:
        style["CompositeImplicitAutograd"] += 1
    elif f.has_composite_explicit_autograd_kernel:
        style["CompositeExplicitAutograd"] += 1
    else:
        style["backend-specific"] += 1
for k, v in style.most_common():
    F.note(f"written as: {k}", v)

structured = Counter()
for f in fns:
    if f.structured:
        structured["structured (the base)"] += 1
    elif f.structured_delegate:
        structured["delegates to a structured op"] += 1
    else:
        structured["unstructured"] += 1
for k, v in structured.most_common():
    F.note(f"style: {k}", v)

key_hist = Counter()
for f in fns:
    for key, index in backend_indices.items():
        if index.has_kernel(f):
            key_hist[str(key)] += 1
for k, v in key_hist.most_common(12):
    F.note(f"declared kernels for key {k}", v)

tag_hist = Counter()
for f in fns:
    for t in f.tags:
        tag_hist[str(t)] += 1
for k, v in tag_hist.most_common(12):
    F.note(f"tag {k}", v)


# ===========================================================================
# 4. Five ops you use every day
# ===========================================================================
F.head("4. five daily ops")

derivatives = yaml.safe_load(open(S.derivatives_yaml_path()))
# derivatives.yaml identifies an op by its FULL schema, overload included:
#   "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
# Matching on the bare base name ("add") would merge `add.Tensor` with
# `add.Scalar` and report derivatives that do not exist for that overload.
deriv_names = {entry["name"].split("(")[0] for entry in derivatives}

DAILY = ["add.Tensor", "linear", "matmul", "relu", "softmax.int", "conv2d"]
table = []
for name in DAILY:
    f = by_name[name]
    cpu = backend_indices[DispatchKey.CPU].get_kernel(f)
    composite = (
        "implicit"
        if f.has_composite_implicit_autograd_kernel
        else "explicit" if f.has_composite_explicit_autograd_kernel else "no"
    )
    has_deriv = str(f.func.name) in deriv_names
    row = {
        "op": name,
        "composite": composite,
        "structured_delegate": str(f.structured_delegate),
        "cpu_kernel": cpu.kernel if cpu else "-",
        "in derivatives.yaml": has_deriv,
        "runtime kernels": len(S.dispatch_rows(f"aten::{name}")),
    }
    table.append(row)
    F.note(f"{name:14s} composite", composite)
    F.note(f"{name:14s} own CPU kernel", row["cpu_kernel"])
    F.note(f"{name:14s} hand-written derivative", has_deriv)
    F.note(f"{name:14s} kernels loaded at runtime", row["runtime kernels"])


# ===========================================================================
# 5. Invariant 1: composite implicit => no hand-written derivative
# ===========================================================================
F.head("5. testing a plausible rule about composite ops")

# The rule, stated before looking: a CompositeImplicitAutograd op is written
# purely in terms of other ops, so autograd can differentiate it by
# differentiating the ops it calls. Writing a derivative formula for it as well
# should therefore be unnecessary - so no such op should appear in
# derivatives.yaml. That is a claim, and claims can be tested.
composites = [f for f in fns if f.has_composite_implicit_autograd_kernel
              and f.func.kind().name == "functional"]
both = [f for f in composites if str(f.func.name) in deriv_names]
F.note("CompositeImplicitAutograd functional ops", len(composites))
F.note("of those, ALSO in derivatives.yaml", len(both))
F.note("rule holds for", f"{len(composites) - len(both)}/{len(composites)}")
for f in both[:10]:
    F.note("  breaks the rule", str(f.func.name))

# So the rule is wrong. What happens when both exist? Ask the dispatcher: if a
# derivative formula was written, the generator emits an Autograd kernel for
# the op, and that kernel sits ABOVE the composite decomposition, so it wins.
# If no formula was written there is no Autograd kernel and the decomposition
# is what autograd sees.
with_formula = without_formula = 0
with_formula_has_autograd = without_formula_has_autograd = 0
odd_ones = []
for f in composites[:400]:
    keys = S.dispatch_keys(f"aten::{f.func.name}")
    has_autograd = any(k.startswith("Autograd") for k in keys)
    if str(f.func.name) in deriv_names:
        with_formula += 1
        with_formula_has_autograd += int(has_autograd)
    else:
        without_formula += 1
        without_formula_has_autograd += int(has_autograd)
        if has_autograd:
            odd_ones.append(str(f.func.name))
F.note("composites WITH a formula, checked", with_formula)
F.note("  ... that have an Autograd kernel loaded", with_formula_has_autograd)
F.note("composites WITHOUT a formula, checked", without_formula)
F.note("  ... that have an Autograd kernel loaded", without_formula_has_autograd)
F.note("  ... which ones", " ".join(odd_ones[:8]) if odd_ones else "(none)")

# The two ops from section 4 make the point concretely.
for name in ["linear", "softmax.int"]:
    keys = sorted(S.dispatch_keys(f"aten::{name}"))
    F.note(f"{name}: in derivatives.yaml", name in deriv_names)
    F.note(f"{name}: Autograd keys loaded",
           " ".join(k for k in keys if k.startswith("Autograd")) or "(none)")


# ===========================================================================
# 6. Invariant 2: the file vs the running dispatcher
# ===========================================================================
F.head("6. declared vs loaded")

ALIAS_COVERS_CPU = [
    DispatchKey.CompositeImplicitAutograd,
    DispatchKey.CompositeExplicitAutograd,
    DispatchKey.CompositeExplicitAutogradNonFunctional,
]

sample = [f for f in fns if f.func.kind().name in ("functional", "out")]
rng = np.random.default_rng(0)
idx = rng.choice(len(sample), size=600, replace=False)
probe = [sample[int(i)] for i in idx]

agree_naive = disagree_naive = 0
agree_cpu = disagree_cpu = missing_op = 0
meta_from_python = []
meta_declared_but_python = 0
runtime_only_meta = 0
disagreements = []
for f in probe:
    name = f"aten::{f.func.name}"
    try:
        rows = S.dispatch_rows(name)
    except RuntimeError:
        missing_op += 1
        continue
    if not rows:
        missing_op += 1
        continue
    runtime_keys = {k for k, _ in rows}
    # The naive reading: "the entry names a CPU kernel, or an alias key that
    # covers CPU". This is what you would write after reading the file once.
    naive_cpu = backend_indices[DispatchKey.CPU].has_kernel(f) or any(
        backend_indices[k].has_kernel(f) for k in ALIAS_COVERS_CPU
    )
    # The correct reading adds one rule: an op with `structured_delegate:` has
    # no kernel of its own, but the generator still emits a CPU registration
    # that forwards to the delegate. Project 53's `add.Tensor` is exactly this.
    declared_cpu = naive_cpu or bool(f.structured_delegate)
    runtime_cpu = bool(
        runtime_keys
        & {"CPU", "CompositeImplicitAutograd", "CompositeExplicitAutograd",
           "CompositeExplicitAutogradNonFunctional"}
    )
    if naive_cpu == runtime_cpu:
        agree_naive += 1
    else:
        disagree_naive += 1
    if declared_cpu == runtime_cpu:
        agree_cpu += 1
    else:
        disagree_cpu += 1
        disagreements.append((str(f.func.name), declared_cpu, runtime_cpu))

    declared_meta = backend_indices[DispatchKey.Meta].has_kernel(f)
    where_meta = dict(rows).get("Meta")
    if where_meta and ".py" in where_meta:
        meta_from_python.append(str(f.func.name))
        if declared_meta:
            meta_declared_but_python += 1
    if where_meta and not declared_meta and not any(
        backend_indices[k].has_kernel(f) for k in ALIAS_COVERS_CPU
    ):
        runtime_only_meta += 1

F.note("operators probed", len(probe))
F.note("not found in the running dispatcher", missing_op)
F.note("naive reading agrees with the runtime", agree_naive)
F.note("naive reading disagrees", disagree_naive)
F.note("after adding the structured_delegate rule, agrees", agree_cpu)
F.note("after adding the structured_delegate rule, disagrees", disagree_cpu)
for d in disagreements[:8]:
    F.note("  disagreement (op, declared, runtime)", str(d))
F.note("Meta kernels registered from a .py file", len(meta_from_python))
F.note("  ... of those, also declared in the YAML", meta_declared_but_python)
F.note("Meta kernels the YAML never mentions", runtime_only_meta)
for n in sorted(meta_from_python)[:8]:
    F.note("  python meta kernel", n)


# ===========================================================================
# 7. From a kernel name to the file that defines it
# ===========================================================================
F.head("7. following a kernel name")

# The YAML gives you a C++ function name, not a file. The generated declaration
# header - which ships in the wheel - gives you the namespace and signature, and
# the name is unique enough to grep for in aten/src/ATen/native/.
inc = os.path.join(S.torch_root(), "include", "ATen", "ops")
F.note("generated per-operator headers shipped in the wheel", len(os.listdir(inc)))
for op, kernel in [("add", "ufunc_add_CPU"), ("relu", "relu"), ("linear", "linear")]:
    header = os.path.join(inc, f"{op}_native.h")
    if not os.path.exists(header):
        continue
    text = open(header).read()
    hits = [ln.strip() for ln in text.splitlines() if kernel in ln]
    F.note(f"{op}_native.h lines naming {kernel}", len(hits))
    for h in hits[:2]:
        F.note("  ", h[:100])

sizes = {}
for label, sub in [("declaration headers (ATen/ops)", "ATen/ops"), ("all headers", "")]:
    root = os.path.join(S.torch_root(), "include", sub) if sub else os.path.join(S.torch_root(), "include")
    total = 0
    count = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith((".h", ".hpp")):
                total += os.path.getsize(os.path.join(dirpath, fn))
                count += 1
    sizes[label] = (count, total / 1e6)
    F.note(f"{label}: files", count)
    F.note(f"{label}: MB", total / 1e6)


# ===========================================================================
# 8. Answering a real question with tags
# ===========================================================================
F.head("8. tags answer real questions")

nondet = sorted({str(f.func.name.name.base) for f in fns
                 if any(str(t) == "nondeterministic_seeded" for t in f.tags)})
F.note("ops tagged nondeterministic_seeded (base names)", len(nondet))
F.note("  a few", ", ".join(nondet[:12]))
with open(os.path.join(OUT, "nondeterministic_ops.txt"), "w") as fh:
    fh.write("\n".join(nondet) + "\n")

# Verify the tag on one op the reader can check by hand.
torch.manual_seed(0)
a1 = torch.rand(4)
torch.manual_seed(0)
a2 = torch.rand(4)
F.note("rand reproduces under manual_seed", bool(torch.equal(a1, a2)))
F.note("'rand' carries the tag", "rand" in nondet)

core = sorted({str(f.func.name.name.base) for f in fns
               if any(str(t) == "core" for t in f.tags)})
F.note("ops tagged core", len(core))
F.note("  a few", ", ".join(core[:12]))

pointwise = {str(f.func.name.name.base) for f in fns
             if any(str(t) == "pointwise" for t in f.tags)}
F.note("ops tagged pointwise", len(pointwise))
view_ops = {str(f.func.name.name.base) for f in fns
            if any(str(t) == "view_copy" for t in f.tags)}
F.note("ops tagged view_copy", len(view_ops))


# ===========================================================================
# figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=110)
fig.patch.set_facecolor("#fcfcfb")
axes = axes.ravel()
for ax in axes:
    style_axes(ax)

ax = axes[0]
labs = list(kinds)
vals = [kinds[k] for k in labs]
ax.bar(range(len(labs)), vals, color=SERIES[0], width=0.6)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("operators")
ax.set_title(f"1. {len(fns)} operators by variant kind", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

ax = axes[1]
labs = list(style)
vals = [style[k] for k in labs]
ax.barh(range(len(labs)), vals, color=[SERIES[1], SERIES[2], SERIES[3]][: len(labs)], height=0.6)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels([l.replace("Composite", "Composite\n") for l in labs], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("operators")
ax.set_title("2. how each op is written", loc="left", fontsize=11)
for i, v in enumerate(vals):
    ax.text(v, i, f" {v}", va="center", fontsize=8)

ax = axes[2]
top = key_hist.most_common(10)
ax.barh([t[0] for t in top], [t[1] for t in top], color=SERIES[0], height=0.6)
ax.invert_yaxis()
ax.tick_params(labelsize=8)
ax.set_xlabel("operators declaring a kernel")
ax.set_title("3. dispatch keys in the file", loc="left", fontsize=11)

ax = axes[3]
top = tag_hist.most_common(9)
ax.barh([t[0] for t in top], [t[1] for t in top], color=SERIES[4], height=0.6)
ax.invert_yaxis()
ax.tick_params(labelsize=8)
ax.set_xlabel("operators carrying the tag")
ax.set_title("4. tags", loc="left", fontsize=11)

fig.tight_layout()
save(fig, os.path.join(OUT, "native_functions_census.png"))
F.write(os.path.join(OUT, "findings.csv"))
