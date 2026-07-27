"""A Mixture-of-Experts feed-forward layer, small enough to watch.

A normal transformer block ends with one MLP that every token goes through.
An MoE block keeps E copies of that MLP ("experts") plus a tiny linear "router"
that scores the experts for each token and sends the token to the best `top_k`
of them. Two consequences that are easy to mix up:

  * total parameters go up by roughly E times,
  * compute per token goes up by only top_k times.

That is the entire selling point: capacity you pay for in memory, not in
arithmetic. It is *conditional* computation -- which weights run depends on the
token, which a dense layer can never do.

The router is a `Linear(d, E)` with no bias. For every token it produces E
scores; softmax turns them into weights; the top-k experts run and their
outputs are combined with those weights. Because the weights multiply the
expert outputs, the router receives a gradient and learns which expert helps.

Why a load-balancing loss is needed at all. Nothing in the main objective
prevents the router from sending every token to expert 0 -- and early in
training that is exactly what happens, because whichever expert is randomly
slightly better gets more tokens, gets better, and attracts still more. The
auxiliary loss (from Shazeer et al. 2017, used unchanged in Switch Transformer
and Mixtral) multiplies "fraction of tokens routed to expert i" by "average
router probability for expert i" and sums; it is smallest when both are flat,
so it pushes the router towards using everybody.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Drop-in replacement for the MLP in `unified.Block`.

    Returns `(output, info)`; `info["balance"]` is the auxiliary loss and
    `info["counts"]` records how many tokens each expert received, which is the
    measurement project 35 exists to make.
    """

    def __init__(self, d, n_experts=8, top_k=2, ffn=None):
        super().__init__()
        ffn = ffn or 4 * d
        self.n_experts, self.top_k, self.d = n_experts, top_k, d
        self.router = nn.Linear(d, n_experts, bias=False)
        # all experts in two batched tensors so one einsum runs them together
        self.w_in = nn.Parameter(torch.randn(n_experts, d, ffn) * (d ** -0.5))
        self.w_out = nn.Parameter(torch.randn(n_experts, ffn, d) * (ffn ** -0.5))
        self.register_buffer("last_expert", torch.zeros(1, dtype=torch.long),
                             persistent=False)

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)                                   # (N, d)
        logits = self.router(flat)                                # (N, E)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)             # (N, k)
        top_w = top_w / top_w.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            # which (token, slot) pairs chose expert e
            hit = (top_i == e)
            if not hit.any():
                continue
            tok = hit.any(-1).nonzero(as_tuple=True)[0]
            w = (top_w * hit).sum(-1)[tok]                        # its gate weight
            h = F.gelu(flat[tok] @ self.w_in[e]) @ self.w_out[e]
            out.index_add_(0, tok, h * w[:, None])
            counts[e] = len(tok)

        # Shazeer's load-balancing loss: E * sum_i (frac_i * mean_prob_i)
        frac = counts / max(counts.sum().item(), 1)
        balance = self.n_experts * (frac * probs.mean(0)).sum()
        info = {"balance": balance, "counts": counts.detach(),
                "assign": top_i.detach().reshape(b, t, self.top_k)}
        return out.reshape(b, t, d), info


def moe_factory(n_experts=8, top_k=2, ffn=None):
    return lambda d: MoEFFN(d, n_experts=n_experts, top_k=top_k, ffn=ffn)


def dense_factory(ffn_mult=4):
    def make(d):
        h = int(ffn_mult * d)
        return nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    return make


@torch.no_grad()
def routing_table(model, seqs, kinds, batch=32, max_rows=None):
    """For every layer, count how many tokens of each modality went to each
    expert. Returns an array of shape (layers, n_modalities, n_experts).

    This is the whole experiment: if experts specialise by modality, the rows
    of this table will look very different from one another.

    `max_rows` defaults to "all of them" on purpose. The validation set is
    ordered -- every image row, then every audio row -- so any cap short of the
    full length silently drops a whole modality and leaves its row of the table
    empty.
    """
    import numpy as np
    model.eval()
    max_rows = len(seqs) if max_rows is None else min(len(seqs), max_rows)
    tables, n_exp = None, None
    for i in range(0, max_rows, batch):
        ids = torch.from_numpy(seqs[i:i + batch])
        kk = torch.from_numpy(kinds[i:i + batch])
        aux = []
        model(ids, aux)
        if tables is None:
            n_exp = aux[0]["assign"].max().item() + 1
            n_exp = max(n_exp, model.blocks[0].mlp.n_experts)
            tables = np.zeros((len(aux), 4, n_exp))
        for li, info in enumerate(aux):
            a = info["assign"]                                    # (B, T, k)
            for m in range(4):
                sel = a[kk == m]                                  # (n_tok, k)
                if sel.numel():
                    tables[li, m] += np.bincount(sel.reshape(-1).numpy(),
                                                 minlength=n_exp)
    return tables


def specialisation(table):
    """One number per layer: how far routing depends on the modality.

    We compare P(expert | modality) with P(expert), using the *mutual
    information* between "which modality is this token" and "which expert did
    it get", in bits. Zero means the router ignores the modality entirely; the
    maximum is log2(number of modalities present).

    Only the three real modalities count -- the marker tokens (<bos>, <boi>,
    ...) are dropped, because they are a handful of positions whose routing
    would otherwise inflate the number without saying anything about text,
    images or audio.
    """
    import numpy as np
    out = []
    for layer in table[:, 1:4]:
        joint = layer / max(layer.sum(), 1e-9)
        pm = joint.sum(1, keepdims=True)
        pe = joint.sum(0, keepdims=True)
        nz = joint > 0
        mi = float((joint[nz] * np.log2(joint[nz] / (pm @ pe)[nz])).sum())
        out.append(mi)
    return out
