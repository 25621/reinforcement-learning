# The fused MLP as one Triton kernel (needs sm_70+, this GPU is sm_61).
#
# The structure to notice: the k loop over the FIRST matmul finishes, the
# bias+GELU runs on the accumulator while it is still in registers, and only
# then does the second matmul start -- all inside one kernel launch, so the
# hidden activation never reaches HBM.

import triton
import triton.language as tl

@triton.jit
def fused_mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, y_ptr,
                     M, D, H, O,
                     BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_h = tl.arange(0, BH)

    # ---- matmul 1: x @ W1 ------------------------------------------------
    acc = tl.zeros((BM, BH), dtype=tl.float32)
    for k in range(0, D, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None] * D + offs_k[None, :])
        w = tl.load(w1_ptr + offs_k[:, None] * H + offs_h[None, :])
        acc += tl.dot(a, w)

    # ---- epilogue: bias + GELU, on the accumulator, in registers ---------
    acc += tl.load(b1_ptr + offs_h)[None, :]
    acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.70710678))

    # ---- matmul 2: h @ W2 ------------------------------------------------
    offs_o = tl.arange(0, BH)
    out = tl.zeros((BM, BH), dtype=tl.float32)
    for k in range(0, H, BK):
        w = tl.load(w2_ptr + (k + tl.arange(0, BK))[:, None] * O + offs_o[None, :])
        out += tl.dot(tl.trans(tl.trans(acc)[k:k + BK]), w)   # schematic

    out += tl.load(b2_ptr + offs_o)[None, :]
    tl.store(y_ptr + offs_m[:, None] * O + offs_o[None, :], out)

# This is schematic on purpose -- a real fused MLP kernel needs the whole
# hidden row (H values) live at once to start matmul 2, so BH must equal H and
# the tile must fit in registers. That constraint is the real reason production
# kernels fuse the EPILOGUE into the matmul and leave the two matmuls separate,
# which is exactly what project 33 measures on the CPU.
