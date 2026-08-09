# What this kernel looks like in Triton (cannot run here: needs sm_70+, this
# GPU is sm_61). Compare it line by line with softmax_3pass above.

import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, o_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # <- one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                   # <- the same max subtraction
    e = tl.exp(x)
    tl.store(o_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)

def softmax(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)        # the whole row in one block
    softmax_kernel[(rows,)](x, out, x.stride(0), cols, BLOCK=BLOCK, num_warps=8)
    return out

# The three passes are invisible because `x` lives in SRAM the whole time:
# tl.max, tl.exp and tl.sum all read the same on-chip copy. That is exactly
# what the C++ version gets from the L1 cache. The kernel is one program per
# row in both cases, and both are limited by how fast the row arrives from
# main memory, not by the arithmetic.
