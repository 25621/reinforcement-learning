"""Shared helpers for Phase 10 (reading the PyTorch source) — projects 53-57.

Every project in this phase reads PyTorch's own source material. Two pieces of
that material turn out to be sitting on your disk already, inside the wheel you
installed with `pip install torch`, and this file is mostly about finding them:

* :func:`native_functions` — parses `native_functions.yaml`, the master list of
  every built-in operation. The surprise is that you do **not** need to clone
  the PyTorch repository to read it: the wheel ships a copy under
  `torchgen/packaged/`, because the code generator (`torchgen`) is shipped too
  and a generator without its input file is useless. So the "source" half of
  this phase is a local file read, not a download.

* :func:`dispatch_table` — asks the *running* library which kernels are actually
  registered for an operator. This is the runtime counterpart to the YAML file:
  the YAML says what was *declared* at build time, the table says what is
  *loaded* right now. Projects 54 and 56 both live on the gap between the two.

Everything else here is small shared plumbing: parsing the dispatch dump into
rows, and finding the installed wheel's directories.
"""

from __future__ import annotations

import functools
import os
import re
import sys

import torch

# Phase 9's `Findings` bookkeeper is reused unchanged: every number quoted in a
# README of this phase goes through it and lands in outputs/findings.csv.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "48-nan-forensics")
)
from debug_lib import Findings, interleaved  # noqa: E402,F401


# ---------------------------------------------------------------------------
# where the installed wheel keeps things
# ---------------------------------------------------------------------------


def torch_root() -> str:
    """Directory of the installed `torch` package (the unpacked wheel)."""
    return os.path.dirname(os.path.abspath(torch.__file__))


def torchgen_root() -> str:
    """Directory of the installed `torchgen` package — the code generator.

    `torchgen` is a normal Python package that pip installs next to `torch`.
    It is the program that turns `native_functions.yaml` into C++ during a
    PyTorch build, and it ships the YAML files it needs to do that.
    """
    import torchgen

    return os.path.dirname(os.path.abspath(torchgen.__file__))


def packaged_aten() -> str:
    """Path to the packaged `ATen/native` directory holding the YAML files."""
    return os.path.join(torchgen_root(), "packaged", "ATen", "native")


def native_functions_yaml_path() -> str:
    return os.path.join(packaged_aten(), "native_functions.yaml")


def tags_yaml_path() -> str:
    return os.path.join(packaged_aten(), "tags.yaml")


def derivatives_yaml_path() -> str:
    return os.path.join(torchgen_root(), "packaged", "autograd", "derivatives.yaml")


# ---------------------------------------------------------------------------
# native_functions.yaml
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def native_functions():
    """Parse `native_functions.yaml` with PyTorch's own parser.

    Using `torchgen`'s parser rather than a plain `yaml.safe_load` matters:
    the YAML has defaults and shorthands (a missing `dispatch:` block means
    something specific, `structured_delegate` redirects an entry to another
    one). `torchgen` applies exactly the rules the build applies, so what you
    read here is what the build read.

    Returns `(functions, backend_indices)` where `functions` is a list of
    `NativeFunction` objects and `backend_indices` maps a `DispatchKey` to the
    kernels declared for it.
    """
    from torchgen.gen import parse_native_yaml

    parsed = parse_native_yaml(native_functions_yaml_path(), tags_yaml_path())
    return parsed.native_functions, parsed.backend_indices


@functools.lru_cache(maxsize=1)
def native_functions_by_name() -> dict:
    """`{"add.Tensor": NativeFunction, ...}` for direct lookup."""
    fns, _ = native_functions()
    return {str(f.func.name): f for f in fns}


def is_structured(op_name: str):
    """Is this `.out` entry a *structured* kernel? None if the op is unknown.

    "Structured" is PyTorch's name for an op written in the modern style: the
    author writes only the maths, and the code generator writes the
    surroundings — shape checking, output allocation, output *resizing*. An
    unstructured op has all of that hand-written inside the kernel, which is
    why unstructured ops are where the inconsistencies of project 57 live.
    """
    f = native_functions_by_name().get(op_name)
    if f is None:
        return None
    return bool(f.structured or f.structured_delegate)


# ---------------------------------------------------------------------------
# the live dispatch table
# ---------------------------------------------------------------------------

# A dump line looks like one of:
#   CPU: registered at /pytorch/build/.../RegisterCPU_0.cpp:1309 :: (...) -> ...
#   CompositeImplicitAutograd[alias]: registered at /pytorch/build/... :: ...
#   Meta (inactive): registered at ...
# The `[alias]` suffix is easy to forget, and forgetting it silently drops every
# composite operator from the results — the parse quietly succeeds and the op
# looks like it has no implementation at all.
_DUMP_ROW = re.compile(
    r"^([A-Za-z0-9_]+)(\[alias\])?( \(inactive\))?: registered at (.*?)(?: ::|$)"
)


def dispatch_table(op: str) -> str:
    """Raw text of the dispatch table for e.g. `"aten::add.Tensor"`.

    This is a debugging hook of the C++ dispatcher exposed to Python. It prints
    one line per registered kernel: the dispatch key, and the source file the
    kernel was registered from. Those file paths are real paths inside the
    machine that built your wheel, which is how you learn the name of the file
    to open on GitHub.
    """
    return torch._C._dispatch_dump(op)


def dispatch_rows(op: str) -> list[tuple[str, str]]:
    """Parse :func:`dispatch_table` into `[(dispatch_key, source_location)]`.

    Alias keys such as `CompositeImplicitAutograd[alias]` are kept, with the
    `[alias]` marker stripped: they are real registrations, they just stand for
    a whole family of keys instead of one.

    Two kinds of line are dropped. Keys marked `(inactive)` are registrations
    for a library that is compiled in but not loaded. And the `debug:` line is
    not a kernel at all — it records where the operator's *schema* was declared.
    Counting either one would overstate how many kernels can actually run.
    """
    rows = []
    for line in dispatch_table(op).splitlines():
        m = _DUMP_ROW.match(line.strip())
        if not m:
            continue
        key, _alias, inactive, where = m.groups()
        if inactive or key == "debug":
            continue
        rows.append((key, where.strip()))
    return rows


def dispatch_keys(op: str) -> set[str]:
    """Just the set of dispatch keys with a live kernel for `op`."""
    return {k for k, _ in dispatch_rows(op)}


def key_set(t: torch.Tensor) -> list[str]:
    """The dispatch keys a tensor carries, highest priority first.

    A tensor does not "have a type" as far as the dispatcher is concerned. It
    carries a *set of keys* — CPU, AutogradCPU, and so on — and the dispatcher
    runs the kernel for the highest-priority key present. Printing the set is
    the fastest way to see why a particular kernel was chosen.
    """
    text = str(torch._C._dispatch_key_set(t))
    inner = text[text.index("(") + 1 : text.rindex(")")]
    return [k.strip() for k in inner.split(",") if k.strip()]
