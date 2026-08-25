"""InfoNCE, written out by hand -- plus the reference implementations we check it against.

Nothing here calls a library loss function except `reference_*`, which exists
only so the hand-written version has something to be compared with.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the loss itself
# ---------------------------------------------------------------------------
def similarity_matrix(image_emb, text_emb, tau):
    """L2-normalize both sides, then one matmul gives every cosine similarity.

    Row i, column j is the cosine similarity between image i and caption j,
    divided by the temperature tau. The diagonal is the matched pairs.
    """
    v = image_emb / image_emb.norm(dim=-1, keepdim=True)
    t = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return (v @ t.T) / tau


def infonce_manual(image_emb, text_emb, tau=0.07, direction="both"):
    """Symmetric InfoNCE with the softmax and the log written out explicitly.

    For one row, the loss is  -log( exp(S_ii) / sum_j exp(S_ij) )
                            =  -S_ii + logsumexp_j(S_ij).
    That is exactly cross-entropy against the label "the correct column is i".
    """
    s = similarity_matrix(image_emb, text_emb, tau)
    n = s.shape[0]
    idx = torch.arange(n)

    # Rows: each IMAGE ranks all N captions.
    row_loss = (-s[idx, idx] + torch.logsumexp(s, dim=1)).mean()
    # Columns: each CAPTION ranks all N images.
    col_loss = (-s[idx, idx] + torch.logsumexp(s, dim=0)).mean()

    if direction == "rows":
        return row_loss
    if direction == "cols":
        return col_loss
    return 0.5 * (row_loss + col_loss)


def reference_infonce(image_emb, text_emb, tau=0.07, direction="both"):
    """The same thing built out of torch's cross_entropy, for comparison."""
    s = similarity_matrix(image_emb, text_emb, tau)
    labels = torch.arange(s.shape[0])
    row = F.cross_entropy(s, labels)
    col = F.cross_entropy(s.T, labels)
    if direction == "rows":
        return row
    if direction == "cols":
        return col
    return 0.5 * (row + col)


# ---------------------------------------------------------------------------
# the gradient, derived on paper
# ---------------------------------------------------------------------------
def analytic_grad_wrt_logits(s, direction="both"):
    """d(loss)/d(similarity matrix), worked out by hand.

    For the row half, differentiating  -S_ii + logsumexp_j(S_ij)  gives

        dL/dS_ij = (P_ij - [i == j]) / N,      P = softmax over the row.

    Read that as: every entry is pushed DOWN in proportion to how much
    probability the model currently gives it, and the true pair is pulled UP by
    a full unit. A negative the model already scores near zero gets a near-zero
    gradient -- it teaches nothing. That single fact is the whole motivation for
    hard-negative mining (project 12) and for temperature (project 13).
    """
    n = s.shape[0]
    eye = torch.eye(n, dtype=s.dtype)
    g_row = (torch.softmax(s, dim=1) - eye) / n
    g_col = (torch.softmax(s, dim=0) - eye) / n
    if direction == "rows":
        return g_row
    if direction == "cols":
        return g_col
    return 0.5 * (g_row + g_col)


def push_weights(s):
    """For each row, the softmax probabilities on the NEGATIVES only, sorted
    from hardest (highest score) to easiest, renormalized to sum to 1.

    This is literally 'what fraction of the pushing-apart force does each wrong
    caption receive'.
    """
    p = torch.softmax(s, dim=1).clone()
    n = s.shape[0]
    p[torch.arange(n), torch.arange(n)] = 0.0
    p = p / p.sum(dim=1, keepdim=True)
    return torch.sort(p, dim=1, descending=True).values


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def in_batch_accuracy(s):
    """Fraction of rows whose highest-scoring column is the correct one.
    This is the quantity InfoNCE is a soft relaxation of."""
    return float((s.argmax(dim=1) == torch.arange(s.shape[0])).float().mean())


def finite_difference(fn, x, n_probes=40, eps=1e-3, seed=0):
    """Nudge single entries of x and see whether the loss moved by the amount
    the gradient predicted. The oldest and bluntest correctness check there is."""
    rng = np.random.default_rng(seed)
    x = x.detach().clone().double().requires_grad_(True)
    loss = fn(x)
    loss.backward()
    grad = x.grad.detach().clone()

    rows = rng.integers(0, x.shape[0], n_probes)
    cols = rng.integers(0, x.shape[1], n_probes)
    worst = 0.0
    for i, j in zip(rows, cols):
        plus = x.detach().clone()
        plus[i, j] += eps
        minus = x.detach().clone()
        minus[i, j] -= eps
        numeric = (fn(plus).item() - fn(minus).item()) / (2 * eps)
        worst = max(worst, abs(numeric - grad[i, j].item()))
    return worst
