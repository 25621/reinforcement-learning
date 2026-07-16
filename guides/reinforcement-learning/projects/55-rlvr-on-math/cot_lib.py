"""A harder math task where showing your work pays: sum of four numbers.

Projects 55 (RLVR) and 56 (length-bias audit) need a task where completions
can be SHORT or LONG for the same prompt, so that completion length is a real
behavior the model can change. This module provides it:

    prompt            "3+7+9+2="
    direct answer     "21;"                                (short, hard)
    chain of thought  "3+7=10,10+9=19,19+2=21;"            (long, easy)
    verbose CoT       "3+7=10,10+9=19,19+2=21,21=21;"      (long + padding)

Adding four numbers in one shot is hard for a tiny model (~25-30% correct),
but each scratchpad step is a two-number addition it can nail (~90%). The
"verbose" style restates the final answer — pure filler that adds no
information, used by project 56 to measure length bias.

The verifier reads the LAST number before ';' so all three styles are graded
by the same rule. Reuses the GPT and padding helpers from project 50's
rlhf_lib; only the vocabulary and task differ.
"""

import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "50-sft-a-small-base-model"))
import rlhf_lib as L  # noqa: E402

K = 4                            # operands per problem
MAXOP = 20                       # each operand in [0, MAXOP]
CHARS = "0123456789+=,;"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
BLOCK = 48
END = STOI[";"]
MAX_NEW = 36


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def new_model(seed=0):
    torch.manual_seed(seed)
    return L.GPT(L.Config(vocab_size=VOCAB, block_size=BLOCK))


# ---------------------------------------------------------------- task / data
def problem(rng):
    return [rng.randint(0, MAXOP) for _ in range(K)]


def prompt_str(ops):
    return "+".join(str(o) for o in ops) + "="


def direct_str(ops):
    return f"{sum(ops)};"


def cot_str(ops, verbose=False):
    """Running partial sums: 3+7=10,10+9=19,19+2=21;  (the scratchpad)."""
    acc = ops[0]
    steps = []
    for o in ops[1:]:
        steps.append(f"{acc}+{o}={acc + o}")
        acc += o
    if verbose:
        steps.append(f"{acc}={acc}")          # filler: restate the answer
    return ",".join(steps) + ";"


def final_answer(completion):
    nums = re.findall(r"\d+", completion.split(";")[0])
    return int(nums[-1]) if nums else None


def is_correct(ops, completion):
    return final_answer(completion) == sum(ops)


def style_of(completion):
    """'direct' (no scratchpad), 'cot', or 'verbose' (CoT + restated answer)."""
    body = completion.split(";")[0]
    if "=" not in body:
        return "direct"
    return "verbose" if re.search(r"(\d+)=\1$", body) else "cot"


# ------------------------------------------------------------------------ SFT
def sft_batch(bs, rng, p_cot=0.15, p_verbose=0.0):
    """Mixed-style demonstrations, loss-masked to the completion."""
    xs, ys, masks = [], [], []
    for _ in range(bs):
        ops = problem(rng)
        u = rng.random()
        if u < p_cot:
            comp = cot_str(ops)
        elif u < p_cot + p_verbose:
            comp = cot_str(ops, verbose=True)
        else:
            comp = direct_str(ops)
        s = prompt_str(ops) + comp
        ids = encode(s) + [END] * (BLOCK + 1 - len(s))
        cut = len(prompt_str(ops)) - 1
        m = [1.0 if i >= cut else 0.0 for i in range(BLOCK)]
        xs.append(ids[:BLOCK]); ys.append(ids[1:BLOCK + 1]); masks.append(m)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(masks)


def train_sft(model, steps, lr=3e-3, p_cot=0.15, p_verbose=0.0, seed=0,
              log_every=0):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        x, y, mask = sft_batch(64, rng, p_cot=p_cot, p_verbose=p_verbose)
        logits = model(x)
        loss = L.masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  sft step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return model


# ------------------------------------------------------------------ generation
@torch.no_grad()
def sample_batch(model, ops_list, max_new=MAX_NEW, temperature=0.0):
    """Batched completions, bucketed by prompt length (same idea as project
    50's sampler, retargeted to this vocabulary)."""
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, ops in enumerate(ops_list):
        by_len[len(prompt_str(ops))].append(i)
    comps = [None] * len(ops_list)
    for idxs in by_len.values():
        cur = torch.tensor([encode(prompt_str(ops_list[i])) for i in idxs])
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


def eval_problems(n=300, seed=1234):
    rng = random.Random(seed)
    return [problem(rng) for _ in range(n)]


@torch.no_grad()
def evaluate(model, n=300, seed=1234, temperature=0.0):
    """Accuracy, mean completion length, and per-style rates in one pass."""
    probs = eval_problems(n, seed)
    comps = sample_batch(model, probs, temperature=temperature)
    acc = sum(is_correct(o, c) for o, c in zip(probs, comps)) / n
    length = sum(len(c) for c in comps) / n
    styles = {"direct": 0, "cot": 0, "verbose": 0}
    for c in comps:
        styles[style_of(c)] += 1 / n
    return dict(acc=acc, length=length, **styles)


# --------------------------------------------------------- log-probs (RL/DPO)
def seq_from(ops, completion):
    p = encode(prompt_str(ops))
    c = encode(completion)
    return p + c, [0] * len(p) + [1] * len(c)


def completion_token_logps(model, seqs, masks):
    x, m = L.pad_batch(seqs, masks, pad=END)
    logits = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp, m[:, 1:]


def completion_logprobs(model, seqs, masks):
    tok_lp, m = completion_token_logps(model, seqs, masks)
    return (tok_lp * m).sum(-1)
