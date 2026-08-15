"""How much faster is memory that lives on the chip?

Tenstorrent's bet is that the expensive part of inference is the distance the
weights travel, so it gives every core its own SRAM and asks the compiler to
keep the weights there. There is no Tensix grid here to test that on -- but a
GPU's L2 cache is the same idea with a different name (fast memory physically
on the die), so the *ratio* between on-die and off-die can be measured
directly by choosing a working set that fits, or does not fit, in L2.

Two details make this measurable rather than merely plausible:

* **The re-reading happens inside the kernel, not by launching it again.**
  A 1 MB buffer at on-die speed takes under 2 us to read, which is about the
  same as a kernel launch, so a loop of launches would measure the launch.
* **Each pass starts at a rotated offset.** Reading the same addresses in the
  same order every pass lets the compiler notice the loads are identical and
  hoist them out of the loop -- at which point the "bandwidth test" does one
  pass and 511 empty iterations. Rotating the start defeats that.

Sizes are chosen as whole multiples of `GRID * BLOCK` so that every pass reads
exactly the whole buffer and the byte count needs no correction.
"""

import torch
import triton
import triton.language as tl

GRID = 256
BLOCK = 512
STRIDE_ELEMS = GRID * BLOCK          # 512 KB in fp32


@triton.jit
def _read_reduce(X, OUT, n, REPS, GRID: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for rep in range(REPS):
        i = ((pid + rep) % GRID) * BLOCK
        while i < n:
            off = i + tl.arange(0, BLOCK)
            acc += tl.load(X + off, mask=off < n, other=0.0)
            i += GRID * BLOCK
    tl.store(OUT + pid, tl.sum(acc, axis=0))


def read_reduce(x, out, reps):
    _read_reduce[(GRID,)](x, out, x.numel(), reps, GRID=GRID, BLOCK=BLOCK,
                          num_warps=4)


def reps_for(n_elems, target_bytes=256 << 20):
    """Enough passes that every size moves roughly the same total traffic."""
    return max(2, int(target_bytes // (n_elems * 4)))
