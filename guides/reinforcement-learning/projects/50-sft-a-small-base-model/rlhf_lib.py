"""Shared post-training stack for RL Phase 9 — a tiny LLM on a verifiable task.

Every project in this phase (SFT, reward model, PPO-RLHF, DPO, GRPO, RLVR,
length-bias, reward-hacking) needs the same three ingredients:

  1. a small autoregressive language model we can train on CPU in minutes,
  2. an "instruction" task whose answers a program can check exactly,
  3. helpers to generate completions and score their log-probabilities.

This module provides all three around a char-level addition task:

    prompt      "37+8="        (the "user turn")
    completion  "45;"          (the "assistant turn"; ';' is the end token)

Because a verifier (`is_correct`) can grade any answer exactly, this single task
supports the whole phase: SFT demonstrations, preference pairs (correct vs
wrong), a reward model, PPO/GRPO rollouts, and RLVR. Operands are 0-50 so a
4-layer char GPT reaches ~90% with a full SFT budget (~1500 steps) — and,
stopped early (650 steps, ~36%), yields a *partial* policy with clear headroom
for RL to improve.

The GPT itself is a minimal modern decoder (RMSNorm, RoPE, SwiGLU) — the same
skeleton used across these guides, inlined here so the RL phase is
self-contained. Imported by projects 51-57 via sys.path.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(12)
DEVICE = torch.device("cpu")   # the box's GPU (sm_61) is unsupported by torch

# ------------------------------------------------------------------- tokenizer
MAXN = 50                       # operands in [0, MAXN]
CHARS = "0123456789+=;"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
BLOCK = 16
END = STOI[";"]


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


# ------------------------------------------------------------------- the model
@dataclass
class Config:
    vocab_size: int = VOCAB
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = BLOCK


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(d_head, T, device):
    inv = 1.0 / 10000.0 ** (torch.arange(0, d_head, 2, device=device).float() / d_head)
    ang = torch.outer(torch.arange(T, device=device).float(), inv)
    return torch.cos(ang).repeat(1, 2), torch.sin(ang).repeat(1, 2)   # (T, d_head)


def apply_rope(x, cos, sin):        # x: (B, h, T, d_head)
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.n_head
        self.dh = cfg.n_embd // cfg.n_head
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.o = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, rope):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = int(8 / 3 * d)
        self.w = nn.Linear(d, hidden, bias=False)
        self.g = nn.Linear(d, hidden, bias=False)
        self.o = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.w(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(self, x, rope):
        x = x + self.attn(self.n1(x), rope)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    """Minimal decoder-only transformer. forward(idx) -> logits (B, T, vocab)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok.weight = self.head.weight              # weight tying
        self._rope = None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def _rope_for(self, T, device):
        if self._rope is None or self._rope[0] < T:
            self._rope = (T, rope_tables(self.cfg.n_embd // self.cfg.n_head, T, device))
        cos, sin = self._rope[1]
        return cos[:T], sin[:T]

    def body(self, idx):
        """Final hidden states (B, T, d) — used by the reward model head."""
        B, T = idx.shape
        x = self.tok(idx)
        rope = self._rope_for(T, idx.device)
        for blk in self.blocks:
            x = blk(x, rope)
        return self.norm_f(x)

    def forward(self, idx):
        return self.head(self.body(idx))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def new_model(seed=0):
    torch.manual_seed(seed)     # seed BEFORE init, or the net is irreproducible
    return GPT(Config())


# ---------------------------------------------------------------- task / data
def sample_problem(rng):
    a, b = rng.randint(0, MAXN), rng.randint(0, MAXN)
    return a, b, a + b


def prompt_str(a, b):
    return f"{a}+{b}="


def is_correct(a, b, completion):
    """The verifier: does the completion state the right sum before ';'?"""
    return completion.split(";")[0] == str(a + b)


def sft_batch(bs, rng, corrupt=False, full_loss=False):
    """A padded SFT batch. Loss covers only answer tokens unless full_loss.

    corrupt=True writes a wrong answer — a stand-in for noisy web pretraining.
    """
    xs, ys, masks = [], [], []
    for _ in range(bs):
        a, b, c = sample_problem(rng)
        ans = rng.randint(0, 2 * MAXN) if corrupt else c
        s = f"{a}+{b}={ans};"
        ids = encode(s) + [END] * (BLOCK + 1 - len(s))
        eq = s.index("=")
        # target at position i is s[i+1]; it is an answer token when i >= eq
        m = [1.0] * BLOCK if full_loss else [1.0 if i >= eq else 0.0 for i in range(BLOCK)]
        xs.append(ids[:BLOCK]); ys.append(ids[1:BLOCK + 1]); masks.append(m)
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(masks))


def masked_ce(logits, y, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           reduction="none")
    return (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)


def train_sft(model, steps, lr=3e-3, corrupt=False, full_loss=False, seed=0,
              log_every=0, eval_every=0, curve=None):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        x, y, mask = sft_batch(64, rng, corrupt=corrupt, full_loss=full_loss)
        logits = model(x)
        loss = masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  sft step {step}/{steps} loss {loss.item():.3f}", flush=True)
        if eval_every and curve is not None and (step + 1) % eval_every == 0:
            curve.append((step + 1, accuracy(model, n=400)))
    return model


def base_model(seed=0, steps=150):
    """A 'pretrained base': fluent in the format but trained on wrong answers.

    Stands in for a raw web-pretrained model — it continues text plausibly
    (digits, then ';') without ever answering correctly.
    """
    m = new_model(seed=seed)
    return train_sft(m, steps=steps, corrupt=True, seed=seed)


def sft_model(steps=650, seed=0, lr=3e-3):
    """A fresh model supervised-fine-tuned to *partial* accuracy — the shared
    starting policy for the RL/preference projects (they improve on it)."""
    m = new_model(seed=seed)
    return train_sft(m, steps=steps, seed=seed, lr=lr)


# ------------------------------------------------------------------ generation
@torch.no_grad()
def sample_batch(model, ab_list, max_new=4, temperature=0.0):
    """Batched completions for many prompts, bucketed by prompt length so each
    batch is a clean rectangle (no padding artifacts). Returns completion
    strings aligned to ab_list. temperature=0 means greedy decoding."""
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, (a, b) in enumerate(ab_list):
        by_len[len(prompt_str(a, b))].append(i)
    comps = [None] * len(ab_list)
    for idxs in by_len.values():
        cur = torch.tensor([encode(prompt_str(*ab_list[i])) for i in idxs])
        B = len(idxs)
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new):
            logits = model(cur[:, -BLOCK:])[:, -1]
            if temperature and temperature > 0:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            for j in range(B):
                if not done[j]:
                    t = int(nxt[j]); gen[j].append(t)
                    if t == END:
                        done[j] = True
            cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
            if done.all():
                break
        for j, i in enumerate(idxs):
            comps[i] = decode(gen[j])
    return comps


def eval_problems(n=400, seed=1234):
    rng = random.Random(seed)
    return [(rng.randint(0, MAXN), rng.randint(0, MAXN)) for _ in range(n)]


@torch.no_grad()
def accuracy(model, n=400, seed=1234, temperature=0.0):
    probs = eval_problems(n, seed)
    gens = sample_batch(model, probs, temperature=temperature)
    return sum(is_correct(a, b, g) for (a, b), g in zip(probs, gens)) / n


# --------------------------------------------------- sequence log-probs (RL/DPO)
def seq_from(a, b, completion):
    """Token ids for prompt+completion, plus a 0/1 mask over completion tokens."""
    p = encode(prompt_str(a, b))
    c = encode(completion)
    return p + c, [0] * len(p) + [1] * len(c)


def pad_batch(seqs, masks, pad=END):
    """Right-pad ragged sequences into rectangles. Padded positions carry a
    zero mask, so the pad id never touches any loss."""
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), pad, dtype=torch.long)
    m = torch.zeros(B, T)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(mk)] = torch.tensor(mk, dtype=torch.float)
    return x, m


def completion_token_logps(model, seqs, masks):
    """Per-token log-probs of the completion under `model`.

    Returns (tok_lp (B,T-1), mask (B,T-1)) — the workhorse for DPO/PPO/GRPO.
    """
    x, m = pad_batch(seqs, masks)
    logits = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, T-1)
    return tok_lp, m[:, 1:]


def completion_logprobs(model, seqs, masks):
    """Summed completion log-prob per sequence — a (B,) tensor."""
    tok_lp, m = completion_token_logps(model, seqs, masks)
    return (tok_lp * m).sum(-1)


# ---------------------------------------------------------------- reward model
class RewardModel(nn.Module):
    """A GPT body whose next-token head is swapped for a 1-number score head.

    The transformer reads the whole (prompt + completion) sequence; the scalar
    is taken at the sequence's final token, where the model has seen everything.
    """

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.gpt = GPT(Config())
        self.score = nn.Linear(Config().n_embd, 1, bias=False)

    def reward(self, seqs):
        """Scores for a list of token-id sequences (ragged ok) — a (B,) tensor."""
        x, _ = pad_batch(seqs, [[1] * len(s) for s in seqs])
        h = self.gpt.body(x)                                   # (B, T, d)
        last = torch.tensor([len(s) - 1 for s in seqs])
        return self.score(h[torch.arange(len(seqs)), last]).squeeze(-1)


def preference_pairs(n, rng, kind="random"):
    """(a, b, chosen, rejected) tuples. chosen is always the correct answer.

    kind="random":   rejected is a random wrong number (easy to tell apart)
    kind="nearmiss": rejected is off by one (hard to tell apart)
    kind="mixed":    half of each
    """
    out = []
    for i in range(n):
        a, b, c = sample_problem(rng)
        k = kind if kind != "mixed" else ("random" if i % 2 == 0 else "nearmiss")
        if k == "random":
            w = c
            while w == c:
                w = rng.randint(0, 2 * MAXN)
        else:
            w = c + rng.choice([-1, 1])
            w = max(w, 0)
        out.append((a, b, f"{c};", f"{w};"))
    return out


def pair_seqs(pairs):
    """Token sequences for the chosen and rejected sides of preference pairs."""
    ch = [seq_from(a, b, cho)[0] for a, b, cho, _ in pairs]
    rj = [seq_from(a, b, rej)[0] for a, b, _, rej in pairs]
    return ch, rj


def train_reward_model(pairs, epochs=6, lr=1e-3, seed=0, bs=64, log_every=0):
    """Bradley-Terry training: push chosen scores above rejected scores."""
    rm = RewardModel(seed=seed)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(epochs):
        order = list(range(len(pairs)))
        random.Random(seed * 1000 + ep).shuffle(order)
        for i in range(0, len(order), bs):
            batch = [pairs[j] for j in order[i:i + bs]]
            ch, rj = pair_seqs(batch)
            loss = -F.logsigmoid(rm.reward(ch) - rm.reward(rj)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if log_every:
            print(f"  rm epoch {ep} loss {loss.item():.4f}", flush=True)
    return rm


@torch.no_grad()
def rm_pair_accuracy(rm, pairs):
    """Fraction of held-out pairs where the RM ranks chosen above rejected."""
    ch, rj = pair_seqs(pairs)
    return float((rm.reward(ch) > rm.reward(rj)).float().mean())
