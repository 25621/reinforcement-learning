"""NF4 ("4-bit NormalFloat") and LoRA, built from scratch.

Why NF4 instead of INT4
-----------------------
INT4's 16 levels are evenly spaced. Neural-network weights are not evenly
distributed -- they pile up near zero and thin out in the tails, close to a bell
curve. Evenly spaced levels therefore waste most of their resolution on the
tails, where almost no weights live. NF4 instead places its 16 levels at the
*quantiles* of a standard normal distribution: equally many weights land in each
bucket. The name says exactly that -- a 4-bit float whose grid is fitted to the
Normal distribution.
"""

import math

import torch


def normal_ppf(p):
    """Inverse CDF of the standard normal, from torch's erfinv.

    "ppf" = percent point function, the inverse of the cumulative distribution:
    ppf(0.9) is the value below which 90% of a standard normal lies.
    """
    return math.sqrt(2.0) * torch.erfinv(2.0 * torch.as_tensor(p) - 1.0)


def nf4_levels():
    """Derive the 16 NF4 levels rather than copying the published table.

    16 levels cannot be both symmetric around zero *and* contain zero exactly --
    that would need an odd count. QLoRA resolves it by taking 8 quantiles on the
    positive side, 7 on the negative side, and one exact 0.0. Zero is worth the
    asymmetry: padding, masks and pruned weights are exactly zero, and a grid
    that could only *approximate* zero would give every one of them a small bias.
    """
    offset = 1 - 0.5 * (1 / 32 + 1 / 30)   # 0.9677083; keeps the tails finite
    pos = normal_ppf(torch.linspace(offset, 0.5, 9))[:-1]     # 8 values
    neg = -normal_ppf(torch.linspace(offset, 0.5, 8))[:-1]    # 7 values
    vals = torch.cat([neg, torch.zeros(1), pos]).sort().values
    vals = vals / vals.abs().max()              # normalize into [-1, 1]
    assert vals.numel() == 16, vals.numel()
    return vals


NF4 = nf4_levels()


def nf4_quantize(W, block=64, levels=NF4):
    """Block-wise NF4: one absmax scale per `block` weights, then nearest level.

    Returns (codes, scales). `codes` are indices 0..15, so two weights fit in one
    byte in a real implementation.
    """
    flat = W.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, block)
    scale = blocks.abs().amax(1, keepdim=True).clamp(min=1e-8)
    normed = blocks / scale
    # Nearest level by binary search on the midpoints between levels. The
    # obvious alternative -- comparing every weight against all 16 levels at
    # once -- would build a tensor 16x the size of the model.
    bounds = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize(normed, bounds)
    return codes, scale, (W.shape, pad)


def nf4_dequantize(codes, scale, meta, levels=NF4):
    shape, pad = meta
    out = (levels[codes] * scale).reshape(-1)
    if pad:
        out = out[:-pad]
    return out.reshape(shape)


def nf4_fake_quant(W, block=64):
    return nf4_dequantize(*nf4_quantize(W, block))


def double_quantize_scales(scale, block2=256):
    """Quantize the FP32 block scales themselves to INT8.

    With block=64, every 64 weights carry one FP32 scale -- that is 32/64 = 0.5
    extra bits per weight, which is a lot when the weight itself is only 4 bits.
    Storing those scales as INT8 with their own (much rarer) FP32 scale drops the
    overhead to about 0.127 bits per weight. This is QLoRA's "double
    quantization": quantizing the quantization constants.
    """
    s = scale.reshape(-1)
    pad = (-s.numel()) % block2
    if pad:
        s = torch.cat([s, torch.zeros(pad)])
    g = s.reshape(-1, block2)
    mean = g.mean(1, keepdim=True)
    amax = (g - mean).abs().amax(1, keepdim=True).clamp(min=1e-8)
    q = torch.round((g - mean) / amax * 127).clamp(-127, 127)
    deq = (q / 127 * amax + mean).reshape(-1)
    return deq[:scale.numel()].reshape(scale.shape)


# ---------------------------------------------------------------------- LoRA
class LoRALinear(torch.nn.Module):
    """Frozen base weight + a trainable rank-r correction B @ A.

    Why a low rank is enough: fine-tuning moves a pretrained weight matrix only a
    little, and that small movement turns out to be well approximated by a matrix
    of low rank. Storing it as two thin matrices (in_f x r and r x out_f) makes
    the trainable parameter count ~r/min(in,out) of the original -- and, more
    importantly, the optimizer state shrinks by the same factor.
    """

    def __init__(self, base, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.zeros(r, base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B starts at zero so the adapter contributes nothing at step 0 and the
        # model begins fine-tuning from exactly the pretrained function.
        self.scaling = alpha / r
        self.r = r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def inject_lora(model, targets=("q_proj", "v_proj"), r=16, alpha=32):
    """Wrap the named Linear layers of every transformer block with a LoRA adapter."""
    # Collect first: mutating the module tree while walking it makes named_modules
    # descend into the freshly inserted wrappers.
    todo = []
    for bi, block in enumerate(model.model.layers):
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in targets:
                todo.append((block, name, mod))
    for block, name, mod in todo:
        parent = block
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(mod, r=r, alpha=alpha))
    return len(todo)


class ActivationBytes:
    """Count the bytes autograd keeps alive for the backward pass.

    There is no CUDA memory API on this CPU-only box, so we hook the save/restore
    path that autograd itself uses: every tensor stashed for backward passes
    through `pack`. De-duplicating by storage pointer avoids counting a tensor
    twice when several ops save the same one.
    """

    def __init__(self):
        self.total = 0
        self.seen = set()

    def __enter__(self):
        def pack(t):
            key = t.untyped_storage().data_ptr()
            if key not in self.seen:
                self.seen.add(key)
                self.total += t.untyped_storage().nbytes()
            return t
        self.ctx = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)
