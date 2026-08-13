"""A Triton kernel, wrapped as a PyTorch operator.

The kernel is SwiGLU's activation, `silu(a) * b`, which is a real fused kernel:
it is in every Llama-family feed-forward block, and writing it by hand saves
two full-size intermediates.

Four objects live here:

  raw_silu_mul      the kernel called directly from Python. Correct, fast, and
                    invisible to everything PyTorch does around it.
  aihw::silu_mul    the same kernel registered as a custom operator, with a
                    CPU kernel, a fake (shape-only) implementation, and a
                    backward rule.
  aihw::double_*    a pair of tiny operators, identical except that one
                    declares that it mutates its input and the other lies.
  aihw::relu_*fake  operators with a missing and a wrong fake implementation.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------------------------------ kernels

@triton.jit
def _silu_mul_fwd(A, B, O, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    tl.store(O + offs, a * tl.sigmoid(a) * b, mask=mask)


@triton.jit
def _silu_mul_bwd(A, B, GO, GA, GB, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    go = tl.load(GO + offs, mask=mask)
    s = tl.sigmoid(a)
    # d/da [a*sigmoid(a)*b] = b * sigmoid(a) * (1 + a*(1 - sigmoid(a)))
    tl.store(GA + offs, go * b * (s * (1 + a * (1 - s))), mask=mask)
    tl.store(GB + offs, go * (a * s), mask=mask)


def raw_silu_mul(a, b):
    """The kernel, called directly. Nothing knows this function exists."""
    o = torch.empty_like(a)
    n = a.numel()
    _silu_mul_fwd[(triton.cdiv(n, 1024),)](a, b, o, n, BLOCK=1024)
    return o


# ------------------------------------------------------- the registered op

@torch.library.custom_op("aihw::silu_mul", mutates_args=())
def silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    o = torch.empty_like(a)
    n = a.numel()
    _silu_mul_fwd[(triton.cdiv(n, 1024),)](a, b, o, n, BLOCK=1024)
    return o


@silu_mul.register_kernel("cpu")
def _(a, b):
    """A CPU path, so the op is usable (and testable) without a GPU."""
    return F.silu(a) * b


@silu_mul.register_fake
def _(a, b):
    """Shapes and dtypes only - no data. This is what the compiler traces
    through, so it must agree with the real kernel about the output's shape,
    dtype, and device."""
    return torch.empty_like(a)


@torch.library.custom_op("aihw::silu_mul_bwd", mutates_args=())
def silu_mul_bwd(a: torch.Tensor, b: torch.Tensor,
                 go: torch.Tensor) -> list[torch.Tensor]:
    ga, gb = torch.empty_like(a), torch.empty_like(a)
    n = a.numel()
    _silu_mul_bwd[(triton.cdiv(n, 1024),)](a, b, go, ga, gb, n, BLOCK=1024)
    return [ga, gb]


@silu_mul_bwd.register_kernel("cpu")
def _(a, b, go):
    s = torch.sigmoid(a)
    return [go * b * (s * (1 + a * (1 - s))), go * (a * s)]


@silu_mul_bwd.register_fake
def _(a, b, go):
    return [torch.empty_like(a), torch.empty_like(a)]


def _setup_context(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    g = silu_mul_bwd(a, b, grad.contiguous())
    return g[0], g[1]


silu_mul.register_autograd(_backward, setup_context=_setup_context)


# --------------------------------------------- operators that get it wrong

@torch.library.custom_op("aihw::double_lie", mutates_args=())
def double_lie(x: torch.Tensor) -> None:
    """Mutates its input, and declares that it does not."""
    x.mul_(2.0)


@double_lie.register_fake
def _(x):
    return None


@torch.library.custom_op("aihw::double_honest", mutates_args={"x"})
def double_honest(x: torch.Tensor) -> None:
    """Identical, honestly declared."""
    x.mul_(2.0)


@double_honest.register_fake
def _(x):
    return None


@torch.library.custom_op("aihw::relu_badfake", mutates_args=())
def relu_badfake(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@relu_badfake.register_fake
def _(x):
    """Wrong on purpose: claims half the length."""
    return torch.empty(x.shape[0] // 2, device=x.device, dtype=x.dtype)


@torch.library.custom_op("aihw::relu_nofake", mutates_args=())
def relu_nofake(x: torch.Tensor) -> torch.Tensor:
    """No fake implementation registered at all."""
    return torch.relu(x)
