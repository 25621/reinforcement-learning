"""Medusa heads: self-speculation without a second model.

A Medusa head is a tiny add-on to the *target* model that predicts a token
two, three or four positions ahead from the same hidden state the target
already computed for the next one. Three heads give you three draft tokens per
target forward pass, and the extra cost is three small matrix multiplies --
no second model, no second KV cache, no second tokenizer.

    base head (already in the model):   h_t  ->  token t+1
    medusa head 1 (trained here):       h_t  ->  token t+2
    medusa head 2:                      h_t  ->  token t+3
    medusa head 3:                      h_t  ->  token t+4

Each head is `lm_head(h + W2 · SiLU(W1 · h))` -- a residual block feeding the
model's *existing, frozen* output projection. Writing it that way is what makes
it trainable on a CPU: a fresh output projection would be 1536 x 151,936 =
233M parameters per head, against 4.7M for the residual block.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "23-greedy-speculative-decoding"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import speclib as S  # noqa: E402  (also puts project 09 on the path)
import kvlib  # noqa: E402


# ---------------------------------------------------------------------------
# reading the target's hidden states
# ---------------------------------------------------------------------------


def final_hidden(runner, ids, cache, start_pos=0):
    """The vector just before the output projection, for every position.

    `Qwen2Runner.forward` returns logits, which is one matmul too far: the
    Medusa heads consume the *hidden state*, not the vocabulary distribution.
    This repeats the forward pass and stops one step early.
    """
    t = ids.shape[1]
    pos = torch.arange(start_pos, start_pos + t)
    kv_pos = torch.arange(0, start_pos + t)
    x = runner.embed[ids]
    for i in range(runner.n_layers):
        x = runner._layer(i, x, cache, pos, kv_pos)
    return kvlib.rms_norm(x, runner.norm, runner.eps)


# ---------------------------------------------------------------------------
# the heads
# ---------------------------------------------------------------------------


class MedusaHeads(torch.nn.Module):
    """`n_heads` residual blocks sharing the target's frozen `lm_head`."""

    def __init__(self, d_model: int, lm_head: torch.Tensor, n_heads: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.lm_head = lm_head              # frozen, not a Parameter
        self.w1 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(d_model, d_model) * (d_model ** -0.5))
            for _ in range(n_heads)])
        # W2 starts at zero, so every head begins life as an exact copy of the
        # base next-token head. Training only has to learn the *difference*
        # between "next token" and "the token after that", which is a much
        # smaller thing to learn than a distribution from scratch.
        self.w2 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(d_model, d_model))
            for _ in range(n_heads)])

    def features(self, h, i):
        return h + F.silu(h @ self.w1[i]) @ self.w2[i]

    def logits(self, h, i):
        return self.features(h, i) @ self.lm_head.T

    def propose(self, h):
        """One hidden state -> n_heads draft token ids (greedy)."""
        return [int(self.logits(h, i).argmax()) for i in range(self.n_heads)]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# data collection: hidden states + two kinds of label
# ---------------------------------------------------------------------------


CHAT_INSTRUCTION = "Write an encyclopedia paragraph about the following topic."


def collect(runner, tok, n_tokens=16384, chunk=220, max_len=512,
            chat_wrap=True):
    """Run the frozen target over real text and keep, for each position:

      h[t]        the hidden state
      gt[t]       the *ground-truth* next token (what the corpus says)
      sd[t]       the model's own argmax at that position (self-distillation)

    Both label sets fall out of the same forward pass, which is what makes the
    comparison in section C essentially free. The difference matters: at
    serving time a Medusa head has to predict what *the target model* will
    emit, not what a human wrote -- so training on the corpus is training on
    the wrong target, slightly.

    `chat_wrap` puts each passage inside the model's chat template, as an
    assistant turn. Without it the heads are trained on raw encyclopedia
    prose and then asked, at serving time, to predict tokens inside a chat
    reply -- a domain gap that costs more acceptance than any amount of extra
    training makes back. It costs nothing to fix, so it is on by default.
    """
    text = kvlib.wikitext_lines(400_000)
    ids = tok(text, return_tensors="pt").input_ids[0]
    cache = S.make_cache(runner, max_len=max_len)
    hs, gts, sds = [], [], []
    got, i = 0, 0
    t0 = time.perf_counter()
    while got < n_tokens and i + chunk + 1 < len(ids):
        piece = tok.decode(ids[i:i + chunk])
        if chat_wrap:
            head = " ".join(piece.split()[:8])
            wrapped = tok.apply_chat_template(
                [{"role": "user", "content": f"{CHAT_INSTRUCTION} {head}"},
                 {"role": "assistant", "content": piece}], tokenize=False)
        else:
            wrapped = piece
        block = tok(wrapped, return_tensors="pt").input_ids[:, :max_len - 1]
        cache.reset()
        h = final_hidden(runner, block, cache, start_pos=0)[0]
        logits = h @ runner.lm_head.T
        # Position t's hidden state predicts the token at t+1, so the last
        # position has no label and is dropped.
        hs.append(h[:-1].detach().clone())
        gts.append(block[0, 1:].clone())
        sds.append(logits[:-1].argmax(-1).detach().clone())
        got += h.shape[0] - 1
        i += chunk
    return {
        "h": torch.cat(hs),               # (N, d)
        "gt": torch.cat(gts),             # (N,) ground-truth token at t+1
        "sd": torch.cat(sds),             # (N,) model's own token at t+1
        "n": got,
        "chat_wrap": chat_wrap,
        "collect_s": round(time.perf_counter() - t0, 2),
    }


def make_targets(labels, n_heads):
    """Head i predicts the token i+1 positions further ahead than the base
    head does. Position t's target for head i is therefore `labels[t + i + 1]`,
    so the usable range shrinks by n_heads at the end."""
    n = labels.numel() - n_heads
    return torch.stack([labels[i + 1:i + 1 + n] for i in range(n_heads)], 1)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_heads(data, label_key, d_model, lm_head, n_heads=3, steps=250,
                batch=24, lr=1e-3, seed=0, holdout=512, log_every=50):
    torch.manual_seed(seed)
    heads = MedusaHeads(d_model, lm_head, n_heads)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=0.0)

    tgt = make_targets(data[label_key], n_heads)
    H = data["h"][:tgt.shape[0]]
    n_train = H.shape[0] - holdout
    g = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=g)
        h = H[idx]
        loss = 0.0
        for i in range(n_heads):
            loss = loss + F.cross_entropy(heads.logits(h, i), tgt[idx, i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append({"step": step,
                          "loss": round(loss.detach().item() / n_heads, 4)})
    train_s = time.perf_counter() - t0

    # held-out top-1 accuracy, per head, against both label sets
    acc = {}
    with torch.no_grad():
        hv = H[n_train:]
        for key in ("gt", "sd"):
            t = make_targets(data[key], n_heads)[n_train:n_train + hv.shape[0]]
            acc[key] = [round(float((heads.logits(hv, i).argmax(-1) == t[:, i])
                                    .float().mean()), 4)
                        for i in range(n_heads)]
    return heads, {"curve": curve, "train_s": round(train_s, 1),
                   "holdout_top1": acc, "n_params": heads.n_params(),
                   "steps": steps, "batch": batch, "lr": lr,
                   "label_key": label_key}


# ---------------------------------------------------------------------------
# the speculative loop, self-speculation flavour
# ---------------------------------------------------------------------------


@torch.inference_mode()
def medusa_decode(target, heads, cache, prompt_ids, max_new=48):
    """Speculative decoding where the drafter lives inside the target.

    The one structural difference from `speclib.speculative_greedy`: the
    drafter needs the target's *hidden state*, so it cannot run before the
    target does. Each round therefore reuses the hidden state produced by the
    previous round's verification pass -- specifically the one at the last
    *accepted* position, because the hidden states computed for rejected draft
    tokens were conditioned on tokens that no longer exist.
    """
    cache.reset()
    prompt = prompt_ids[0].tolist()
    t0 = time.perf_counter()
    h = final_hidden(target, prompt_ids, cache, start_pos=0)[0]
    prefill_s = time.perf_counter() - t0
    logits = h[-1] @ target.lm_head.T
    tokens = prompt + [int(logits.argmax())]
    h_use = h[-1]
    cache.truncate(len(tokens) - 1)

    k = heads.n_heads
    proposed = accepted = tested = iters = 0
    verify_s = draft_s = 0.0
    accept_run, per_pos_hits, iter_s, iter_n = [], [0] * k, [], []

    while len(tokens) - len(prompt) < max_new:
        t_iter = time.perf_counter()
        t1 = time.perf_counter()
        drafts = heads.propose(h_use)
        draft_s += time.perf_counter() - t1

        block = torch.tensor([[tokens[-1]] + drafts])
        t1 = time.perf_counter()
        hh = final_hidden(target, block, cache, start_pos=len(tokens) - 1)[0]
        preds = (hh @ target.lm_head.T).argmax(-1).tolist()
        verify_s += time.perf_counter() - t1

        n_acc = 0
        for i in range(k):
            tested += 1
            if drafts[i] == preds[i]:
                n_acc += 1
                per_pos_hits[i] += 1
            else:
                break
        tokens.extend(drafts[:n_acc])
        tokens.append(preds[n_acc])
        h_use = hh[n_acc]                  # the hidden state that produced it
        proposed += k
        accepted += n_acc
        iters += 1
        accept_run.append(n_acc)
        cache.truncate(len(tokens) - 1)
        iter_s.append(time.perf_counter() - t_iter)
        iter_n.append(n_acc + 1)

    produced = len(tokens) - len(prompt)
    return {
        "tokens": tokens[len(prompt):][:max_new],
        "produced": produced,
        "prefill_s": prefill_s,
        "draft_s": draft_s,
        "verify_s": verify_s,
        "decode_s": draft_s + verify_s,
        "iters": iters,
        "proposed": proposed,
        "accepted": accepted,
        "tested": tested,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "conditional_acceptance": accepted / tested if tested else 0.0,
        "tokens_per_iter": produced / iters if iters else 0.0,
        "accept_run": accept_run,
        "per_pos_hits": per_pos_hits,
        "iter_s": iter_s,
        "itl": S.itl_from_bursts(iter_s, iter_n),
    }
