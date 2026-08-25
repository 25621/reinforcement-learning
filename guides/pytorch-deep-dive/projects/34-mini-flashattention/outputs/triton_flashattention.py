# FlashAttention in Triton (needs sm_70+, this GPU is sm_61). The loop
# structure is identical to flash_attention in run.py -- compare them.

import triton
import triton.language as tl

@triton.jit
def flash_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, T, D: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr):
    start = tl.program_id(0) * BR            # this program owns BR query rows
    offs_m = start + tl.arange(0, BR)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])   # stays in SRAM

    m_i = tl.full((BR,), -float("inf"), tl.float32)   # running max
    l_i = tl.zeros((BR,), tl.float32)                 # running denominator
    acc = tl.zeros((BR, D), tl.float32)               # running output

    for j in range(0, T, BC):
        offs_n = j + tl.arange(0, BC)
        k = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])

        s = tl.dot(q, tl.trans(k)) * scale            # the BR x BC tile
        m_new = tl.maximum(m_i, tl.max(s, 1))
        corr = tl.exp(m_i - m_new)                    # <- the rescaling
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * corr + tl.sum(p, 1)
        acc = acc * corr[:, None] + tl.dot(p, v)
        m_i = m_new

    tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], acc / l_i[:, None])

# The GPU version has one advantage the CPU version cannot have: q, k, v tiles
# are moved into SRAM by an explicit tl.load, and acc lives in registers for the
# whole loop. On the CPU we rely on the cache to keep the same values close.
# The algorithm -- and the memory it saves -- is the same.
