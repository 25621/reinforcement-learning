# PyTorch Deep Dive: From User to Power User

A comprehensive guide to understanding PyTorch as a *system* — not just as the framework you `import torch` from. This guide assumes you have written some PyTorch code before (a few training scripts, a few `nn.Module` subclasses) and want to understand what is actually happening underneath, how to make it fast, how to scale it across many GPUs, and how to debug it when things go wrong.

> **Why this guide exists.** Every other modern AI guide — text generation, image generation, LLMs, robotics, multimodal — assumes PyTorch fluency. None teaches PyTorch as its own subject. The result is that most practitioners can write `loss.backward()` but cannot explain what the autograd engine is doing, profile a training step, or fix a hang in distributed training. This guide closes that gap.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Tensors and the Storage Model](#phase-1-tensors-and-the-storage-model)
3. [Phase 2: Autograd, Inside Out](#phase-2-autograd-inside-out)
4. [Phase 3: `nn.Module`, Optimizers, and the Training Loop](#phase-3-nnmodule-optimizers-and-the-training-loop)
5. [Phase 4: Data Loading and Input Pipelines](#phase-4-data-loading-and-input-pipelines)
6. [Phase 5: Performance — Profiling, Mixed Precision, and `torch.compile`](#phase-5-performance--profiling-mixed-precision-and-torchcompile)
7. [Phase 6: Custom Kernels — C++, CUDA, and Triton Extensions](#phase-6-custom-kernels--c-cuda-and-triton-extensions)
8. [Phase 7: Distributed Training — DDP, FSDP, and Beyond](#phase-7-distributed-training--ddp-fsdp-and-beyond)
9. [Phase 8: Deployment — TorchScript, ExecuTorch, and ONNX](#phase-8-deployment--torchscript-executorch-and-onnx)
10. [Phase 9: Debugging Hard Problems](#phase-9-debugging-hard-problems)
11. [Phase 10: Reading the PyTorch Source](#phase-10-reading-the-pytorch-source)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Additional Resources](#additional-resources)
15. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

You do not need everything in this list mastered, but if more than a couple are unfamiliar, slow down here before continuing.

### Concepts to Know

- **Python**: classes, decorators, context managers, generators, `__init__` vs `__new__`, the GIL (at least intuitively)
- **NumPy**: array shapes, broadcasting, views vs copies, `dtype`, strides (yes — strides)
- **Linear algebra**: matrix multiplication, the difference between an element-wise and a contracted operation, why `A @ B` is `O(n³)`
- **Basic calculus**: chain rule, partial derivatives
- **Shell and git**: you will read and clone a lot of repos
- **Some prior PyTorch**: you have at least trained an MLP or CNN. If not, do [PyTorch's 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) first.

### The One Sentence Everything Comes Back To

```
A PyTorch tensor is a (storage, shape, stride, dtype, device, requires_grad) tuple,
and autograd is a graph of operations recorded on the forward pass that gets
traversed backward to compute gradients.
```

If that sentence is fuzzy now, it will be sharp by the end of Phase 2.

### Resources

- [PyTorch official docs](https://pytorch.org/docs/stable/index.html) — your single most important reference
- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd) — autograd in 100 lines of Python
- [Edward Z. Yang — PyTorch internals blog post](http://blog.ezyang.com/2019/05/pytorch-internals/) — still the best single read on this topic
- [PyTorch Developer Podcast](https://pytorch-dev-podcast.simplecast.com/) — short episodes by the core team

---

## Phase 1: Tensors and the Storage Model

The tensor is the foundational object. Most surprising performance bugs in PyTorch come from misunderstanding how tensors are laid out in memory.

### Concepts to Learn

- **Storage** vs **Tensor**: a tensor is a *view* into a 1-D storage buffer
- **Shape**, **stride**, **offset**: how multidimensional indexing maps to a flat buffer
- **Contiguous vs non-contiguous tensors**: when `.contiguous()` is needed and why
- **`view` vs `reshape` vs `permute` vs `transpose`**: which copy, which don't
- **dtype**: `float32`, `float16`, `bfloat16`, `int8`, `bool`, when each is appropriate
- **Device**: CPU vs CUDA vs MPS vs XLA; pinned memory; `non_blocking=True`
- **Broadcasting rules** and the silent bugs they cause
- **In-place operations** (`add_`, `mul_`) and when they break autograd

### The Mental Model

```
┌─────────────────────────────────────────────────────┐
│  Tensor: shape=(2, 3), stride=(3, 1), offset=0      │
│                                                     │
│  Storage (1-D buffer of 6 floats):                  │
│  ┌──┬──┬──┬──┬──┬──┐                                │
│  │a │b │c │d │e │f │                                │
│  └──┴──┴──┴──┴──┴──┘                                │
│   0  1  2  3  4  5                                  │
│                                                     │
│  Element [i, j] → storage[offset + i*stride[0]      │
│                                  + j*stride[1]]     │
└─────────────────────────────────────────────────────┘
```

- **`shape=(2, 3)`** — 2 rows, 3 columns (the logical grid seen by Python code).
- **`stride=(3, 1)`** — moving one step along dim-0 (rows) jumps **3** positions in
  the flat storage buffer; moving one step along dim-1 (columns) jumps **1** position.
  For example, `[0, 0] → storage[0]`, `[1, 0] → storage[3]`, `[0, 2] → storage[2]`.

After calling `.transpose(0, 1)` on this tensor (a separate operation — not part of
the indexing formula above):

```
┌─────────────────────────────────────────────────────┐
│  .transpose(0, 1) → shape=(3, 2), stride=(1, 3)     │
│  (same storage, different view — NOT contiguous)    │
└─────────────────────────────────────────────────────┘
```

- **`.transpose(0, 1)`** — swaps axis 0 and axis 1. The logical grid flips from
  2×3 to 3×2 and the strides swap accordingly: `stride=(1, 3)`. No data is copied;
  PyTorch just updates the shape and stride metadata. Because the new stride order
  no longer matches the row-major memory layout, the result is **not contiguous**
  and cannot be passed directly to `.view()`.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Stride explorer](projects/01-stride-explorer/README.md) | Print `.shape`, `.stride()`, `.storage_offset()`, `.is_contiguous()` after every reshape/transpose/permute on a few tensors | ⭐ |
| [View vs copy detective](projects/02-view-vs-copy-detective/README.md) | Modify a tensor through a view, observe the original; find operations that silently copy | ⭐⭐ |
| [Manual indexing](projects/03-manual-indexing/README.md) | Given `(shape, stride, offset)`, compute the flat storage index for `[i, j, k]` by hand and check with `.data_ptr()` | ⭐⭐ |
| [dtype precision study](projects/04-dtype-precision-study/README.md) | Compare `float32`, `float16`, `bfloat16` on a sum of 1M small numbers; observe loss-of-precision and underflow | ⭐⭐ |
| [Broadcasting bug hunt](projects/05-broadcasting-bug-hunt/README.md) | Construct 5 expressions where broadcasting produces a result you didn't intend; write the rule you violated | ⭐⭐ |

### Sample Code: Strides in Action

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x.shape, x.stride())          # (3, 4) (4, 1)
print(x.is_contiguous())            # True

y = x.t()                           # transpose
print(y.shape, y.stride())          # (4, 3) (1, 4)
print(y.is_contiguous())            # False — same storage, different view

# This will raise: "view size is not compatible with input tensor's size and stride"
# y.view(12)

# This works because reshape copies if needed:
z = y.reshape(12)
print(z.data_ptr() == x.data_ptr()) # False — new storage
```

### Key Insight

`view` requires a contiguous tensor and gives you a guaranteed no-copy alias. `reshape` will copy if it has to. `permute` and `transpose` *never* copy — they just rewrite strides. This is why `x.transpose(0, 1).view(-1)` fails: the transposed tensor isn't contiguous.

### Resources

- [PyTorch docs: Tensor Views](https://pytorch.org/docs/stable/tensor_view.html)
- [Edward Z. Yang — PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) (the tensor section)
- [NumPy internals: strides](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray) — same idea, simpler text

---

## Phase 2: Autograd, Inside Out

Autograd is the heart of PyTorch. Most users treat it as a black box; the goal of this phase is to make it transparent.

### Concepts to Learn

- The **dynamic computation graph** (DAG): nodes are `Function`s, edges are tensors
- `requires_grad`, `grad_fn`, `is_leaf`, `retain_grad`
- The **backward pass**: how `loss.backward()` traverses the graph
- **Accumulating gradients**: why `.grad` accumulates, why `optimizer.zero_grad()` exists
- **`torch.no_grad()`** vs **`detach()`** vs **`.data`** — what each does and what it breaks
- **Custom `autograd.Function`**: writing your own forward and backward
- **Gradient checking** with `torch.autograd.gradcheck`
- **Higher-order derivatives** with `create_graph=True`
- **Gradient checkpointing** — trading compute for memory
- **Memory traps**: holding onto graph references, retaining intermediate activations

### The Backward Pass, Visualized

```
Forward:
    x  ──[matmul]── h ──[relu]── a ──[matmul]── y ──[mse]── loss
    W₁ ─────┘                    W₂ ─────┘             ↑
                                                       │
Backward (autograd walks this in reverse):              │
                                                       ▼
    ∂loss/∂y → ∂loss/∂a, ∂loss/∂W₂ → ∂loss/∂h → ∂loss/∂x, ∂loss/∂W₁

Each [op] saved whatever it needed (often the inputs) during forward,
and provides a backward() method that consumes the upstream gradient
and produces gradients for its inputs.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Micrograd in PyTorch style](projects/06-micrograd-in-pytorch-style/README.md) | Reimplement a scalar autograd engine with `Value` objects that build a DAG and traverse it backward | ⭐⭐ |
| [Manual backprop](projects/07-manual-backprop/README.md) | Train a 2-layer MLP without ever calling `.backward()` — compute gradients by hand and compare to autograd | ⭐⭐⭐ |
| [Custom `autograd.Function`](projects/08-custom-autograd-function/README.md) | Implement a `ReLU` and a `Sigmoid` as `torch.autograd.Function` subclasses with `forward` and `backward` | ⭐⭐ |
| [Straight-through estimator](projects/09-straight-through-estimator/README.md) | Implement a non-differentiable `round` operation that passes gradients through as if it were identity | ⭐⭐⭐ |
| [Gradient checkpointing](projects/10-gradient-checkpointing/README.md) | Manually checkpoint a deep network with `torch.utils.checkpoint` and measure the memory/time tradeoff | ⭐⭐⭐ |
| [Double backward](projects/11-double-backward/README.md) | Compute ∇(‖∇L‖²) — the gradient of the gradient norm — for a small model | ⭐⭐⭐⭐ |

### Sample Code: A Custom Autograd Function

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)        # save what backward will need
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x).sum()
y.backward()
print(x.grad)

# Verify against autograd:
assert torch.autograd.gradcheck(MyReLU.apply, (torch.randn(5, dtype=torch.double, requires_grad=True),))
```

### Key Insight

`torch.no_grad()` disables graph construction for everything inside it. `detach()` returns a new tensor that *shares storage* but is disconnected from the graph. `.data` is the same as `detach()` but skips version-counter checks — which is why it's dangerous and largely deprecated.

### The Three Most Common Autograd Bugs

1. **Forgetting `optimizer.zero_grad()`** → gradients accumulate across steps and your loss diverges mysteriously
2. **Modifying a tensor in-place after using it in a graph** → "one of the variables needed for gradient computation has been modified by an inplace operation"
3. **Keeping a reference to `loss` across training steps** → the entire computation graph stays alive in memory until you `del loss`

### Resources

- [PyTorch docs: Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — read this twice
- [Karpathy — micrograd YouTube lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [PyTorch docs: Extending autograd](https://pytorch.org/docs/stable/notes/extending.html)

---

## Phase 3: `nn.Module`, Optimizers, and the Training Loop

These are the parts of PyTorch you use every day. Most people use them correctly without understanding them; the goal here is to understand.

### Concepts to Learn

- **`nn.Module`**: `__init__`, `forward`, `parameters()`, `named_parameters()`, `buffers`, `state_dict`
- The difference between a **parameter** (`nn.Parameter`) and a **buffer** (`register_buffer`) — why BatchNorm running stats are buffers, not parameters
- **`module.train()` vs `module.eval()`** — what actually changes (Dropout, BatchNorm)
- **Hooks**: forward hooks, backward hooks, module hooks, tensor hooks — for inspection and intervention
- **Initialization schemes**: Xavier/Glorot, Kaiming/He, why they matter, where they live (`nn.init`)
- **Optimizers**: SGD, momentum, Adam, AdamW — what state each holds, what `step()` does
- **Learning rate schedulers**: `StepLR`, `CosineAnnealingLR`, warmup
- **`state_dict`**: how saving and loading actually works; `strict=False`; weight remapping
- **Reproducibility**: seeding, `torch.use_deterministic_algorithms`, cuBLAS workspace env vars

### The Anatomy of a Training Step

```python
# Five lines that cost more than they look:
optimizer.zero_grad()      # 1. clear .grad on every parameter
output = model(x)          # 2. forward pass — builds the autograd graph
loss = criterion(output, y)# 3. scalar loss — root of the graph
loss.backward()            # 4. traverse graph in reverse, populate .grad
optimizer.step()           # 5. read each .grad, update parameters in place
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Module introspection](projects/12-module-introspection/README.md) | Walk a pretrained model with `named_modules()` and print its full parameter shape and count | ⭐ |
| [Hook-based feature extractor](projects/13-hook-based-feature-extractor/README.md) | Use a forward hook to grab intermediate activations from a ResNet without modifying it | ⭐⭐ |
| [Custom optimizer](projects/14-custom-optimizer/README.md) | Implement SGD-with-momentum as an `optim.Optimizer` subclass | ⭐⭐ |
| [Implement AdamW from scratch](projects/15-implement-adamw-from-scratch/README.md) | Including bias correction and decoupled weight decay | ⭐⭐⭐ |
| [State dict surgery](projects/16-state-dict-surgery/README.md) | Load weights from one architecture into a slightly different one, mapping keys manually | ⭐⭐⭐ |
| [Reproducible training](projects/17-reproducible-training/README.md) | Train the same model twice with full determinism and verify bit-exact outputs | ⭐⭐⭐ |

### Sample Code: A Custom Optimizer

```python
import torch
from torch.optim import Optimizer

class MySGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]
                v.mul_(mu).add_(p.grad)         # v = mu*v + g
                p.add_(v, alpha=-lr)            # p = p - lr*v
```

### Key Insight

`nn.Module` is essentially a registry. When you assign `self.conv1 = nn.Conv2d(...)` in `__init__`, the parent `__setattr__` notices it's a Module and registers it in `_modules`. Similarly for `nn.Parameter` → `_parameters` and tensors registered via `register_buffer` → `_buffers`. `state_dict()` flattens all of these into a dict. Knowing this makes weight surgery, model merging, and custom serialization possible.

### Resources

- [PyTorch docs: nn.Module source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html) — surprisingly readable
- [PyTorch docs: optim source](https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html) — AdamW in ~100 lines
- [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — debugging discipline

---

## Phase 4: Data Loading and Input Pipelines

The most underrated source of slow training is the data loader. A starved GPU is a wasted GPU.

### Concepts to Learn

- `Dataset` (map-style) vs `IterableDataset` (stream-style)
- `DataLoader`: `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`
- **Worker processes**: how multiprocessing actually works, the fork vs spawn distinction, and why workers can deadlock with CUDA
- **Collate functions**: handling variable-length sequences, padding, custom batching
- **Samplers**: `RandomSampler`, `WeightedRandomSampler`, `DistributedSampler`
- **Memory mapping**: streaming large datasets without loading them into RAM (`numpy.memmap`, Apache Arrow, WebDataset)
- **Pipeline overlap**: keeping CPU and GPU busy at the same time
- **`torch.utils.data.IterableDataset` + sharding**: for datasets that don't fit on disk

### The Bottleneck Diagnosis

```
GPU utilization low + CPU pegged at 100% on one core  → data loader bound, increase num_workers
GPU utilization low + all CPU idle                    → CPU-GPU transfer bound, enable pin_memory
GPU utilization low + I/O wait high                   → disk bound, switch to streaming/mmap/WebDataset
GPU utilization high + slow training                  → compute bound, optimize the model itself
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Naive vs optimized loader](projects/18-naive-vs-optimized-loader/README.md) | Train ResNet on a subset of ImageNet with `num_workers=0`, then 4, 8, 16; plot throughput | ⭐⭐ |
| [Custom collate](projects/19-custom-collate/README.md) | Write a collate_fn that pads variable-length token sequences to the longest in the batch | ⭐⭐ |
| [Weighted sampler](projects/20-weighted-sampler/README.md) | Implement class-balanced sampling on an imbalanced classification dataset | ⭐⭐ |
| [Streaming WebDataset](projects/21-streaming-webdataset/README.md) | Load a sharded `.tar` dataset with WebDataset, train without unpacking | ⭐⭐⭐ |
| [Memory-mapped tokens](projects/22-memory-mapped-tokens/README.md) | Tokenize a large text corpus to a single `.bin` file, train an LLM from `np.memmap` | ⭐⭐⭐ |
| [Profile and fix](projects/23-profile-and-fix/README.md) | Take a slow training script, profile the data loader with the PyTorch profiler, fix it | ⭐⭐⭐ |

### Sample Code: A Streaming Token Dataset

```python
import numpy as np
import torch

class TokenStream(torch.utils.data.Dataset):
    """Sample fixed-length chunks from a memory-mapped uint16 token file."""
    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, i):
        # Re-open memmap inside workers to avoid fork/sharing issues
        chunk = torch.from_numpy(self.tokens[i : i + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
```

### Key Insight

The DataLoader spawns worker processes. On Linux with `fork`, each worker inherits the parent's memory. This is why a 100 GB dataset object created in the main process doesn't get copied 16 times — until a worker writes to it (copy-on-write). On macOS and Windows with `spawn`, each worker re-runs your script. This is why `if __name__ == "__main__":` guards matter.

### Resources

- [PyTorch docs: Data loading](https://pytorch.org/docs/stable/data.html)
- [WebDataset](https://github.com/webdataset/webdataset) — for large-scale streaming
- [Hugging Face `datasets`](https://huggingface.co/docs/datasets) — Arrow-backed, integrates with PyTorch
- [Karpathy — nanoGPT data pipeline](https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py) — the memmap pattern in production

---

## Phase 5: Performance — Profiling, Mixed Precision, and `torch.compile`

This is where you stop guessing and start measuring.

### Concepts to Learn

- The **PyTorch Profiler** (`torch.profiler`) — recording traces, viewing in Chrome's trace viewer or Perfetto
- **CUDA streams** and **asynchronous execution** — why `time.time()` around CUDA ops lies, and why you need `torch.cuda.synchronize()`
- **Mixed precision**: `torch.cuda.amp.autocast`, `GradScaler`, `bfloat16` vs `float16`
- **Kernel fusion** and why launch overhead matters for small ops
- **`torch.compile`** (PT 2.x): TorchDynamo, AOTAutograd, Inductor — what each does
- **Memory profiling**: `torch.cuda.memory_summary()`, `torch.cuda.max_memory_allocated()`, memory snapshots
- **Activation memory** vs **parameter memory** vs **optimizer state memory** — the three-way breakdown of training memory
- **Gradient accumulation** — bigger effective batch without bigger memory

### The Mixed Precision Mental Model

```
fp32:  exponent 8 bits, mantissa 23 bits   — old reliable, 4 bytes/element
fp16:  exponent 5 bits, mantissa 10 bits   — narrow range, overflows easily
bf16:  exponent 8 bits, mantissa 7 bits    — same range as fp32, less precision
                                              → almost always the right choice on A100/H100/MI300
```

`bfloat16` does not need a `GradScaler`. `float16` does, because its range is too narrow to represent the small gradients that appear in some layers.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Profile a training step](projects/24-profile-a-training-step/README.md) | Capture a profiler trace of one forward+backward+step, identify the top 3 kernels by time | ⭐⭐ |
| [AMP speedup study](projects/25-amp-speedup-study/README.md) | Train the same model in fp32, fp16+GradScaler, bf16; compare throughput and final accuracy | ⭐⭐ |
| [`torch.compile` test](projects/26-torch-compile-test/README.md) | Compile a transformer block, measure forward+backward time vs eager | ⭐⭐ |
| [Memory breakdown](projects/27-memory-breakdown/README.md) | For a transformer, compute the expected memory for params + grads + optimizer state + activations; verify with `memory_summary` | ⭐⭐⭐ |
| [Gradient accumulation](projects/28-gradient-accumulation/README.md) | Train with effective batch size 4× larger than fits in memory; verify gradients are identical to a large-batch run | ⭐⭐⭐ |
| [Bottleneck fix](projects/29-bottleneck-fix/README.md) | Take a model where `torch.compile` makes things *slower*; find out why (usually graph breaks) and fix it | ⭐⭐⭐⭐ |

### Sample Code: Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()                       # only needed for fp16

for x, y in loader:
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

    with autocast(dtype=torch.bfloat16):    # bf16 → no scaler needed
        logits = model(x)
        loss = loss_fn(logits, y)

    loss.backward()                         # if fp16: scaler.scale(loss).backward()
    optimizer.step()                        # if fp16: scaler.step(optimizer); scaler.update()
```

### Key Insight

CUDA is asynchronous. When you write `y = model(x)`, the CPU queues kernels onto a CUDA stream and returns immediately. The GPU executes them later. This is why `time.time()` around a forward pass measures *kernel launch time*, not *execution time*. Profile with the PyTorch profiler, not with timers, unless you `torch.cuda.synchronize()` first.

### Resources

- [PyTorch Profiler tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Perfetto UI](https://ui.perfetto.dev/) — best trace viewer
- [PyTorch docs: torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — the best single read on this topic

---

## Phase 6: Custom Kernels — C++, CUDA, and Triton Extensions

You will need this when an existing op is too slow, doesn't exist, or you want to fuse several into one.

### Concepts to Learn

- **C++ extensions** via `torch.utils.cpp_extension` — `load` (JIT) vs `setup.py` (ahead-of-time)
- **The Dispatcher** — how PyTorch routes calls like `torch.add(a, b)` to the right backend (CPU/CUDA/etc.) and dtype
- **CUDA basics**: grid, block, thread, shared memory, warps, occupancy
- **Triton** — Python-flavored GPU kernel language; how it compares to raw CUDA
- **Kernel fusion in practice**: implementing FlashAttention-style tiled algorithms
- **Pybind11** — how Python and C++ talk; `TORCH_LIBRARY` for registering custom ops
- **`torch.library.custom_op`** (modern API for registering ops that play nicely with `torch.compile`)

### When to Write a Custom Kernel

- An op you need doesn't exist at all
- An op exists but is slow because of memory bandwidth (you can fuse several into one)
- You want a numerically-different implementation (e.g., a stable softmax variant)
- You're implementing a research idea that isn't expressible in standard ops

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [C++ extension for elementwise add](projects/30-cpp-extension-for-elementwise-add/README.md) | Write `add_cuda` as a C++ extension, register it, call from Python | ⭐⭐ |
| [Triton softmax](projects/31-triton-softmax/README.md) | Implement softmax in Triton, compare against `F.softmax` on a 4096-wide tensor | ⭐⭐⭐ |
| [Triton matmul](projects/32-triton-matmul/README.md) | Tile-based matmul in Triton; aim for >50% of cuBLAS throughput | ⭐⭐⭐⭐ |
| [Fused MLP](projects/33-fused-mlp/README.md) | One Triton kernel: matmul → bias → gelu → matmul; compare against unfused | ⭐⭐⭐⭐ |
| [Mini FlashAttention](projects/34-mini-flashattention/README.md) | Tiled, online-softmax attention kernel; verify numerical match with eager attention | ⭐⭐⭐⭐⭐ |
| [Custom op registration](projects/35-custom-op-registration/README.md) | Wrap your kernel as a `torch.library.custom_op` so `torch.compile` can use it | ⭐⭐⭐ |

### Sample Code: A Triton Vector Add

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")
torch.testing.assert_close(add(a, b), a + b)
```

### Key Insight

In modern deep learning, most kernels are memory-bound, not compute-bound. The peak FLOPs of an A100 is ~312 TFLOPs (bf16), but HBM bandwidth is only ~2 TB/s. For a kernel that reads X bytes and does Y FLOPs, the *arithmetic intensity* Y/X determines whether you're bound by compute or memory. This is why FlashAttention works: it does the same FLOPs as standard attention but reads memory dramatically fewer times.

### Resources

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — start with vector add, end with FlashAttention
- [PyTorch docs: Custom C++ and CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [PyTorch docs: torch.library](https://pytorch.org/docs/stable/library.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135) — read after the Triton tutorials, not before
- [GPU Mode (formerly CUDA MODE) Discord and YouTube](https://github.com/cuda-mode/lectures) — best community for kernel work

---

## Phase 7: Distributed Training — DDP, FSDP, and Beyond

Once a model doesn't fit on one GPU, or trains too slowly on one, you need to go multi-GPU and eventually multi-node.

### Concepts to Learn

- **Communication primitives**: AllReduce, AllGather, ReduceScatter, Broadcast, AllToAll
- **NCCL** — the library that does these on NVIDIA GPUs; how it uses NVLink and InfiniBand
- **Data Parallelism (DDP)**: replicate the model, split the batch, all-reduce gradients
- **Tensor Parallelism**: split individual layers across GPUs (Megatron-style)
- **Pipeline Parallelism**: split layers across GPUs and pipeline microbatches
- **FSDP (Fully Sharded Data Parallel)**: shard parameters, gradients, and optimizer state — the modern default for training large models on commodity clusters
- **DeepSpeed ZeRO** stages 1, 2, 3 — comparable to FSDP
- **3D Parallelism** — combining data + tensor + pipeline
- **`torchrun`** and the rendezvous protocol; `LOCAL_RANK`, `RANK`, `WORLD_SIZE`
- **Common failure modes**: hangs (mismatched calls across ranks), OOM on rank 0, NCCL timeouts

### The Decision Tree

```
Model fits on one GPU, dataset is big                → DDP
Model barely fits on one GPU                         → FSDP (shard optim state and grads)
Model doesn't fit on one GPU                         → FSDP (full shard) or DeepSpeed ZeRO-3
Model doesn't fit even with FSDP                     → add tensor parallelism
Sequence length is the bottleneck                    → sequence/context parallelism
You have hundreds of GPUs across nodes               → 3D parallelism (Megatron-DeepSpeed)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Two-GPU DDP](projects/36-two-gpu-ddp/README.md) | Train a ResNet on 2 GPUs with `torchrun`, observe near-linear speedup | ⭐⭐ |
| [Implement gradient AllReduce](projects/37-implement-gradient-allreduce/README.md) | From scratch with `torch.distributed.all_reduce`, verify it matches DDP | ⭐⭐⭐ |
| [FSDP a transformer](projects/38-fsdp-a-transformer/README.md) | Train a 1B-parameter LLM with FSDP, verify it works on hardware that can't fit it under DDP | ⭐⭐⭐⭐ |
| [Tensor parallel attention](projects/39-tensor-parallel-attention/README.md) | Split a multi-head attention layer column-wise across 2 GPUs (Megatron-style) | ⭐⭐⭐⭐ |
| [Debug a hang](projects/40-debug-a-hang/README.md) | Intentionally introduce a rank-imbalanced call, watch it hang, fix it; use `TORCH_NCCL_BLOCKING_WAIT=1` and `NCCL_DEBUG=INFO` | ⭐⭐⭐ |
| [Multi-node setup](projects/41-multi-node-setup/README.md) | Set up a 2-node cluster (cloud or two boxes on a LAN), run a DDP job that crosses node boundaries | ⭐⭐⭐⭐ |

### Sample Code: A Minimal DDP Script

```python
# Run with: torchrun --nproc_per_node=2 train.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for x, y in loader:                  # use a DistributedSampler in practice
        x, y = x.cuda(), y.cuda()
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()                  # DDP all-reduces gradients here
        optim.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Key Insight

In DDP, every GPU has a full copy of the model. The all-reduce in the backward pass averages gradients across GPUs, so the gradient update is mathematically equivalent to training on a batch `world_size×` larger. FSDP differs fundamentally: each GPU holds only a shard of the parameters, gathering full layers temporarily during forward and backward — this lets you train models far bigger than any one GPU's memory.

### Resources

- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) — tensor parallelism
- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) — wrapper around all of the above

---

## Phase 8: Deployment — TorchScript, ExecuTorch, and ONNX

Training is half the job. The other half is shipping the model somewhere it can actually run.

### Concepts to Learn

- **`torch.jit.script`** vs **`torch.jit.trace`** — the trade-offs (control flow vs simplicity)
- **TorchScript** — the older deployment path; still in wide use but increasingly legacy
- **`torch.export`** — the modern graph-capture API, the foundation for the new deployment story
- **ExecuTorch** — PyTorch's mobile/edge runtime
- **ONNX** — the cross-framework graph format; when it's a win and when it's a trap
- **Quantization**: dynamic, static (PTQ), QAT (quantization-aware training); `int8`, `int4`
- **TensorRT** and **OpenVINO** — vendor-specific accelerators
- **Serving frameworks**: TorchServe, Triton Inference Server, vLLM (for LLMs)
- **Latency vs throughput**: batching strategies, continuous batching for LLMs

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Export to ONNX](projects/42-export-to-onnx/README.md) | Export a CNN to ONNX, run it with `onnxruntime`, verify numerical match with PyTorch | ⭐⭐ |
| [Mobile deployment](projects/43-mobile-deployment/README.md) | Use ExecuTorch to run a small model on an Android device | ⭐⭐⭐⭐ |
| [Dynamic quantization](projects/44-dynamic-quantization/README.md) | Quantize an LLM to int8 with dynamic quantization, measure speedup and quality loss | ⭐⭐⭐ |
| [Static quantization (PTQ)](projects/45-static-quantization-ptq/README.md) | Calibrate static int8 quantization on a CNN, compare to dynamic | ⭐⭐⭐ |
| [Build a Triton server](projects/46-build-a-triton-server/README.md) | Wrap a model in NVIDIA Triton Inference Server, query it over HTTP | ⭐⭐⭐ |
| [Latency profiling](projects/47-latency-profiling/README.md) | Measure p50/p95/p99 latency for batch sizes 1, 8, 32 on the same model | ⭐⭐ |

### Sample Code: Export + ONNX Inference

```python
import torch
import onnxruntime as ort

model = MyModel().eval()
dummy = torch.randn(1, 3, 224, 224)

# New (preferred) export API:
exported = torch.export.export(model, (dummy,))
torch.onnx.export(exported, dummy, "model.onnx", dynamo=True)

session = ort.InferenceSession("model.onnx")
out_onnx = session.run(None, {"x": dummy.numpy()})[0]
out_torch = model(dummy).detach().numpy()

import numpy as np
np.testing.assert_allclose(out_onnx, out_torch, rtol=1e-3, atol=1e-3)
```

### Key Insight

The deployment story in PyTorch is in transition. TorchScript was the old answer; `torch.export` + ExecuTorch / AOTInductor is the new answer. ONNX is a useful intermediate when you need to deploy on hardware whose vendor has good ONNX support but weak PyTorch support. If you're shipping LLMs, none of this matters: use vLLM or TensorRT-LLM directly.

### Resources

- [PyTorch docs: torch.export](https://pytorch.org/docs/stable/export.html)
- [ExecuTorch docs](https://pytorch.org/executorch/)
- [PyTorch quantization tutorial](https://pytorch.org/tutorials/recipes/quantization.html)
- [vLLM](https://docs.vllm.ai/) — for LLM serving specifically

---

## Phase 9: Debugging Hard Problems

When things go wrong in PyTorch, the error messages are sometimes a paragraph long and sometimes one line that means nothing. This phase is about learning to read the signs.

### Concepts to Learn

- **NaN debugging**: `torch.autograd.set_detect_anomaly(True)`, finding the layer that first produces NaN
- **Shape mismatches**: reading stack traces back to the offending op
- **Memory leaks**: how to spot them with `torch.cuda.memory_allocated()` snapshots over time
- **Slow training**: data loader vs forward vs backward vs optimizer vs comm
- **Non-determinism**: tracking down which op is non-deterministic
- **Distributed hangs**: which rank is stuck, on which collective, and why
- **Numerical mismatch**: between fp32 and bf16, between eager and compiled, between two CUDA versions

### The Debug Toolbox

| Symptom | First thing to try |
|---------|-------------------|
| Loss is NaN | `torch.autograd.set_detect_anomaly(True)` for one step |
| OOM at step 50 but not step 1 | You're holding onto graph references; `del loss` and re-check |
| Training slow, GPU at 30% | Profile with PyTorch profiler, look for gaps between kernels |
| DDP hangs | `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| `torch.compile` makes things slower | `TORCH_LOGS="graph_breaks"` to find graph breaks |
| Results differ on rerun | Set seeds, then `torch.use_deterministic_algorithms(True)` |
| Numerical mismatch with old version | Pin the version, bisect across versions |

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [NaN forensics](projects/48-nan-forensics/README.md) | Take a model that NaNs; use anomaly mode to find the layer; explain why | ⭐⭐⭐ |
| [Memory leak hunt](projects/49-memory-leak-hunt/README.md) | Write a training loop with a subtle leak; find it with memory snapshots | ⭐⭐⭐ |
| [Determinism audit](projects/50-determinism-audit/README.md) | Make a training run bit-exact reproducible; document every flag needed | ⭐⭐⭐ |
| [Hang diagnosis](projects/51-hang-diagnosis/README.md) | Intentionally hang a 2-rank DDP job; diagnose with NCCL_DEBUG | ⭐⭐⭐ |
| [Eager vs compile diff](projects/52-eager-vs-compile-diff/README.md) | Find a case where eager and compiled outputs differ; narrow it to a single op | ⭐⭐⭐⭐ |

### Key Insight

The PyTorch error stack is a contract. Most "this doesn't work" bugs are shape, dtype, or device mismatches that the framework caught for you. The hard bugs are silent: a NaN that propagated, a hang waiting on a collective, a gradient that's wrong by a factor of 2 because you forgot `mean` vs `sum`. For those, the right answer is almost always: print more things, in more places, sooner.

### Resources

- [PyTorch docs: Notes — Faq](https://pytorch.org/docs/stable/notes/faq.html)
- [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — half this article is debugging discipline
- [PyTorch forums](https://discuss.pytorch.org/) — search before you ask; the answer often exists

---

## Phase 10: Reading the PyTorch Source

The final unlock. PyTorch's source code is large but well-organized. Once you can read it, you can answer any question about PyTorch that the docs don't.

### Concepts to Learn

- The **repo layout**: `aten/`, `c10/`, `torch/`, `torch/csrc/`, `torch/distributed/`
- **ATen** — the tensor library that does the actual math
- **c10** — the core abstractions (DeviceType, ScalarType, TensorImpl)
- **The Dispatcher** — `aten/src/ATen/native/native_functions.yaml` is the table of contents for every op
- **`torchgen`** — the code generator that takes YAML and produces C++ bindings
- **`torch/csrc/autograd/`** — the engine that does backward
- **Pybind11 glue** — how Python sees C++ classes

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Trace one op end to end](projects/53-trace-one-op-end-to-end/README.md) | Pick `torch.add`; trace from Python call → dispatcher → CPU kernel | ⭐⭐⭐⭐ |
| [Read `native_functions.yaml`](projects/54-read-native-functions-yaml/README.md) | Find five ops you use daily; understand their dispatch entries | ⭐⭐⭐ |
| [Build PyTorch from source](projects/55-build-pytorch-from-source/README.md) | Run the build's codegen stage for real; measure what the rest would cost | ⭐⭐⭐⭐ |
| [Patch and rebuild](projects/56-patch-and-rebuild/README.md) | Replace a live kernel with your own and watch your `printf` fire | ⭐⭐⭐⭐ |
| [Fix a "good first issue"](projects/57-fix-a-good-first-issue/README.md) | Hunt a real bug in your installed PyTorch, then fix and test it | ⭐⭐⭐⭐⭐ |

### Key Insight

PyTorch is a Python frontend over a C++ library. The Python you write is converted (via the dispatcher and codegen) into calls into ATen. ATen contains the actual kernels — for CPU, CUDA, MPS, and so on. Autograd lives in C++ too (`torch/csrc/autograd/`) and operates on `Variable`s, which are now just `Tensor`s with a `grad_fn`. The whole stack is more readable than its size suggests, once you know where to look.

### Resources

- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [Edward Z. Yang — PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — still the canonical map
- [PyTorch Developer Wiki](https://github.com/pytorch/pytorch/wiki) — start with the "Codebase structure" page
- [PyTorch Developer Podcast](https://pytorch-dev-podcast.simplecast.com/) — short episodes by core devs

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | Comfortable in Python, NumPy, basic PyTorch |
| 1. Tensors | 1 week | Can explain views, strides, contiguity |
| 2. Autograd | 1–2 weeks | Wrote a custom `autograd.Function`; manual backprop matches autograd |
| 3. nn.Module, optim | 1 week | Custom optimizer; state-dict surgery |
| 4. Data loading | 1 week | Profiled and fixed a data-bound training run |
| 5. Performance | 2 weeks | Profiler trace; mixed-precision; `torch.compile` working |
| 6. Custom kernels | 2–4 weeks | Triton softmax matches `F.softmax`; mini FlashAttention |
| 7. Distributed | 2–3 weeks | FSDP a model that doesn't fit on one GPU |
| 8. Deployment | 1–2 weeks | Exported a model; quantized one; served one |
| 9. Debugging | Ongoing | Diagnosed NaN, memory leak, DDP hang on your own |
| 10. Source | Ongoing | Can trace a PyTorch op from Python down to its CUDA kernel |

**Total to "power user":** ~3 months of focused study, longer if combined with real projects (recommended).

---

## Key Advice

1. **Read the source.** PyTorch's source is more accessible than its size suggests. Most "how does X work" questions are answered by 30 lines of C++.
2. **Profile before optimizing.** You will be wrong about where the time is going. Always.
3. **Distinguish memory-bound from compute-bound.** Most deep learning kernels are memory-bound. This single distinction explains FlashAttention, mixed precision, kernel fusion, and most modern speedups.
4. **Always `torch.cuda.synchronize()` before timing.** CUDA is asynchronous; naive timing is meaningless.
5. **`set_to_none=True` for `zero_grad`.** It's faster and the modern default.
6. **Prefer `bfloat16` over `float16`** on hardware that supports it (Ampere onward). No `GradScaler` needed.
7. **In distributed code, every rank must call every collective.** A hang almost always means one rank skipped a call the others made.
8. **`torch.compile` is not free.** It's a big win for some workloads and a small loss for others. Always measure.
9. **Determinism has a cost.** Setting `torch.use_deterministic_algorithms(True)` is slower; only do it when you need bit-exact reproducibility.
10. **Write tests, even for ML code.** Especially for custom kernels: `torch.testing.assert_close` against a reference implementation is non-negotiable.

---

## Common Pitfalls to Avoid

- ❌ Timing CUDA ops with `time.time()` without `torch.cuda.synchronize()`
- ❌ Forgetting `optimizer.zero_grad()` and wondering why loss diverges
- ❌ Calling `.cuda()` inside the training loop instead of once
- ❌ Holding references to `loss` across iterations and leaking the graph
- ❌ Using `float16` without a `GradScaler` and seeing silent NaNs
- ❌ Setting `num_workers` higher than the number of CPU cores
- ❌ Comparing eager and compiled outputs with `==` instead of `assert_close`
- ❌ Mismatched calls across ranks in distributed code (one rank logs, others don't)
- ❌ Re-implementing an op that already exists somewhere in `torch.nn.functional`
- ❌ Skipping the profiler step and "optimizing" by guesswork

---

## Additional Resources

### Books and Long-Form Reading
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (free PDF from Manning)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) — free, runnable, PyTorch version available

### Talks and Lectures
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — the best ramp into PyTorch internals
- [GPU Mode lectures](https://github.com/cuda-mode/lectures) — CUDA + PyTorch at the kernel level

### Reference Material
- [PyTorch docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Developer Wiki](https://github.com/pytorch/pytorch/wiki)
- [PyTorch forums](https://discuss.pytorch.org/)

### Tools You Should Know
- **Profiler / Perfetto** — for traces
- **Weights & Biases** or **TensorBoard** — for training curves
- **`py-spy`** — for sampling Python profiles of running processes
- **`nsys`** and **`ncu`** — NVIDIA's low-level profilers, for when the PyTorch profiler isn't enough

### Communities
- [GPU Mode Discord](https://github.com/cuda-mode/lectures) — kernel-level work
- [PyTorch forums](https://discuss.pytorch.org/) — official Q&A
- r/MachineLearning — broader but useful

---

## Quick Start Checklist

- [ ] Can explain what `.stride()` returns and why `.view()` sometimes fails
- [ ] Can implement a `torch.autograd.Function` with custom forward and backward
- [ ] Have done manual backprop through a 2-layer MLP and matched autograd
- [ ] Have written a custom `optim.Optimizer` subclass
- [ ] Have captured and read a PyTorch profiler trace
- [ ] Know the difference between `bfloat16` and `float16` and which to use when
- [ ] Have used `torch.compile` and know how to find graph breaks
- [ ] Have written at least one Triton kernel
- [ ] Have trained a model with DDP on more than one GPU
- [ ] Have trained a model with FSDP that didn't fit under DDP
- [ ] Have exported a model to ONNX or via `torch.export`
- [ ] Have diagnosed a NaN, a memory leak, and a DDP hang at least once each
- [ ] Have read a piece of PyTorch C++ source and understood it

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
