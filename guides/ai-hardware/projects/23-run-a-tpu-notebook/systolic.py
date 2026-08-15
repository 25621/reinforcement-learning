"""A cycle-by-cycle simulator of a weight-stationary systolic array (a TPU MXU).

The array is a grid of R rows by C columns of multiply-accumulate cells.

  * Every cell holds one weight, loaded once and then left alone
    ("weight-stationary").
  * Activations enter on the LEFT and shuffle one column to the right per
    cycle.
  * Partial sums enter at the TOP, gain one product per cell, and fall one row
    per cycle. They leave at the BOTTOM as finished outputs.

Nothing is fetched from memory in the middle of this. Each activation is read
into the edge of the array once and is then reused by every column it passes
through; that reuse is the whole reason the design exists.

Index convention for `X @ W` with X of shape (M, K) and W of shape (K, N):

    array row r    <-> the contraction index k
    array column c <-> the output index n
    time step      <-> which row m of X is being fed

The skew is what makes it line up. Activation X[m, k] is pushed into the left
edge of row k at cycle m + k, so it reaches cell (k, n) at cycle m + k + n --
exactly the cycle at which the partial sum for output (m, n) arrives there
from the cell above. Every cell is busy on a different (m, n) at the same
instant, which is where the throughput comes from.
"""

import numpy as np


class SystolicArray:
    """R x C weight-stationary array. Simulated cell by cell, cycle by cycle."""

    def __init__(self, rows=128, cols=128):
        self.R = rows
        self.C = cols

    # ------------------------------------------------------------------ pass
    def one_pass(self, X, W, count_only=False):
        """Run one tile that fits: X is (M, K<=R), W is (K<=R, N<=C).

        Returns (Y, stats). With `count_only=True` the arithmetic is skipped
        and only the cycle counts are produced, which is what the sweeps use.
        """
        M, K = X.shape
        K2, N = W.shape
        assert K == K2 and K <= self.R and N <= self.C

        # cycles: M rows enter back-to-back, then the last one has to walk
        # (K - 1) rows down and (N - 1) columns right before it falls out.
        cycles = M + K + N - 2 + 1
        stats = dict(M=M, K=K, N=N, cycles=cycles,
                     useful_macs=M * K * N,
                     cell_cycles=cycles * self.R * self.C)
        if count_only:
            return None, stats

        # --- the actual hardware ---------------------------------------
        # a_reg[r, c] : the activation currently sitting in cell (r, c)
        # p_reg[r, c] : the partial sum currently sitting in cell (r, c)
        a_reg = np.zeros((self.R, self.C), dtype=np.float64)
        p_reg = np.zeros((self.R, self.C), dtype=np.float64)
        w_grid = np.zeros((self.R, self.C), dtype=np.float64)
        w_grid[:K, :N] = W                       # weights loaded, then frozen

        Y = np.zeros((M, N), dtype=np.float64)

        for t in range(cycles):
            # 1. what falls out of the bottom of the array this cycle
            #    the psum leaving column n at cycle t belongs to row m = t - K - n + 1
            out_row = p_reg[K - 1, :N].copy()
            m = t - (K - 1) - 1
            for n in range(N):
                mm = m - n
                if 0 <= mm < M:
                    Y[mm, n] = out_row[n]

            # 2. shift: activations one column right, partial sums one row down
            a_reg[:, 1:] = a_reg[:, :-1]
            p_reg[1:, :] = p_reg[:-1, :]
            p_reg[0, :] = 0.0

            # 3. inject this cycle's activations at the left edge.
            #    X[m, k] enters row k at cycle m + k  ->  m = t - k
            a_reg[:, 0] = 0.0
            for k in range(K):
                mm = t - k
                if 0 <= mm < M:
                    a_reg[k, 0] = X[mm, k]

            # 4. every cell multiplies and accumulates, all at once
            p_reg[:K, :N] += a_reg[:K, :N] * w_grid[:K, :N]

        return Y, stats

    # ------------------------------------------------------------------ tiled
    def matmul_cost(self, M, K, N, weight_load_overlapped=True):
        """Cycle cost of a full (M,K)x(K,N) matmul, tiling over the array.

        A K or N bigger than the array is chopped into ceil(K/R) x ceil(N/C)
        tiles. Every tile pays for the padding of its own leftovers: a tile
        with only 5 of 128 columns filled still occupies the whole array for
        the cycles it runs.
        """
        kt = -(-K // self.R)                       # ceil
        nt = -(-N // self.C)
        cycles = 0
        for i in range(kt):
            k = min(self.R, K - i * self.R)
            for j in range(nt):
                n = min(self.C, N - j * self.C)
                cycles += M + k + n - 2 + 1
                if not weight_load_overlapped:
                    cycles += k                    # push weights in first
        useful = M * K * N
        return dict(M=M, K=K, N=N, tiles=kt * nt, cycles=cycles,
                    useful_macs=useful,
                    cell_cycles=cycles * self.R * self.C,
                    utilization=useful / (cycles * self.R * self.C))
