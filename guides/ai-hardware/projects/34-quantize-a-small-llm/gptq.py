"""GPTQ, written from scratch (Frantar et al., 2022 -- arxiv 2210.17323).

The idea in one paragraph
------------------------
Round-to-nearest quantizes every weight independently, so every weight gets its
own small error and the errors just pile up. GPTQ instead quantizes the columns
of W one at a time and, after each column, *edits the columns it has not
quantized yet* so that they partially cancel the error it just made. "How much
should I edit the others?" is answered by the Hessian of the layer's output
error, which for a linear layer is exactly H = 2 X^T X where X is the stack of
calibration activations. No gradients and no labels are involved -- only the
inputs the layer actually sees.

Cost note: the inverse-Hessian factor is computed once per layer with a Cholesky
factorization, which is why GPTQ is minutes rather than hours.
"""

import torch


@torch.no_grad()
def gptq_quantize(W, H, bits=4, group_size=None, sym=True,
                  blocksize=128, percdamp=0.01):
    """Quantize W (out, in) using the Hessian H (in, in). Returns the fake-quantized W."""
    W = W.clone().float()
    H = H.clone().float()
    out_f, in_f = W.shape

    # Columns that no calibration sample ever excited carry no information;
    # pin them so the Cholesky stays well defined and zero the weights.
    diag = torch.arange(in_f)
    dead = torch.diag(H) == 0
    H[diag[dead], diag[dead]] = 1.0
    W[:, dead] = 0.0

    # Dampening: add a small multiple of the mean diagonal. Real activation
    # covariances are near-singular (features are correlated), and the inverse of
    # a near-singular matrix is enormous -- the update would explode.
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp

    # Hinv as an upper-triangular Cholesky factor of H^-1. Working with the
    # factor rather than the full inverse is what makes the column-by-column
    # update a simple outer product.
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    Q = torch.zeros_like(W)
    qmax = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
    scale = zero = None

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(i2 - i1):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]

            # (Re)fit the scale at the start of each group, using the weights as
            # they stand now -- they already carry the compensation from earlier
            # columns, so fitting on the original values would be wrong.
            if group_size is None:
                if scale is None:
                    scale, zero = _fit(W, bits, sym)
            elif col % group_size == 0:
                scale, zero = _fit(W[:, col: col + group_size], bits, sym)

            if sym:
                q = torch.clamp(torch.round(w / scale.squeeze(1)),
                                -qmax - 1, qmax)
                deq = q * scale.squeeze(1)
            else:
                q = torch.clamp(torch.round(w / scale.squeeze(1)) + zero.squeeze(1),
                                0, qmax)
                deq = (q - zero.squeeze(1)) * scale.squeeze(1)

            Q1[:, j] = deq
            err = (w - deq) / d
            # Push the error onto the remaining columns of this block.
            W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            E1[:, j] = err

        Q[:, i1:i2] = Q1
        # Lazy batch update: one matmul pays for the whole block's effect on the
        # columns after it, instead of `blocksize` separate rank-1 updates.
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    return Q


def _fit(block, bits, sym):
    """Per-output-row scale (and zero-point) for one group of columns."""
    if sym:
        qmax = 2 ** (bits - 1) - 1
        amax = block.abs().amax(dim=1, keepdim=True)
        return (amax / qmax).clamp(min=1e-8), torch.zeros_like(amax)
    qmax = 2 ** bits - 1
    lo = block.amin(dim=1, keepdim=True)
    hi = block.amax(dim=1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    return scale, torch.round(-lo / scale)


class HessianCollector:
    """Accumulate H = sum over tokens of x x^T for every Linear in one block."""

    def __init__(self, block):
        self.h = {}
        self.n = {}
        self.handles = []
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.h[name] = torch.zeros(mod.in_features, mod.in_features)
                self.n[name] = 0
                self.handles.append(
                    mod.register_forward_pre_hook(self._make(name)))

    def _make(self, name):
        def hook(mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            self.h[name] += 2.0 * (x.T @ x)
            self.n[name] += x.shape[0]
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        return {k: v / max(self.n[k], 1) for k, v in self.h.items()}


@torch.no_grad()
def gptq_model(model, batches, bits=4, group_size=None, sym=True,
               blocksize=128, verbose=True, quantlib=None):
    """Run block-sequential GPTQ over a whole causal LM, in place.

    "Sequential" matters: block k+1's Hessian is collected from the activations
    produced by the *already quantized* block k. If you collected every Hessian
    from the clean fp32 model instead, each layer would be compensating for an
    input distribution that no longer exists by the time it runs.
    """
    ql = quantlib
    hidden, kwargs = ql.block_inputs(model, batches)
    blocks = model.model.layers
    for bi, block in enumerate(blocks):
        collector = HessianCollector(block)
        for h in hidden:
            ql.run_block(block, h, kwargs)
        hess = collector.close()
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data = gptq_quantize(
                    mod.weight.data, hess[name], bits=bits,
                    group_size=group_size, sym=sym, blocksize=blocksize)
        # Re-run with the quantized weights so the next block sees reality.
        hidden = [ql.run_block(block, h, kwargs) for h in hidden]
        if verbose:
            print(f"  block {bi + 1}/{len(blocks)} quantized", flush=True)
    return model
