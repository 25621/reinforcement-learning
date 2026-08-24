"""Shared pieces for project 67 — serving a reasoning model.

The model is Qwen3-0.6B, a real *hybrid* reasoning model: its chat template has
a switch (`enable_thinking`) that either lets the model write a long
chain-of-thought inside a `<think> ... </think>` block before answering, or
closes that block immediately so it answers straight away.  That switch is what
makes this project possible on a CPU: the same weights give us both a "chat"
workload and a "reasoning" workload, so every difference we measure comes from
the thinking, not from a different model.

Everything here is greedy (do_sample=False), which matters for the budget
experiment in run.py: with greedy decoding a prefix determines the whole
continuation, so truncating a recorded thinking trace at B tokens and appending
`</think>` reproduces *exactly* what a server with a B-token thinking budget
would have produced.
"""

from __future__ import annotations

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
N_THREADS = 6            # 6 beats 12 on this box; see project 24
MAX_THINK = 1024         # hard cap on a single generation
CHAT_CAP = 256           # cap for the no-thinking arm

BRIEF = ("Reason in at most three short sentences, then give the answer.")

# ---------------------------------------------------------------- problems ---
# 24 problems with a checkable numeric answer, in three difficulty tiers.
# "Difficulty" here is defined operationally: how many arithmetic steps a
# correct solution needs.  That is also what drives thinking length, which is
# the quantity this project is about.

PROBLEMS = [
    # --- tier "easy": one step ---------------------------------------------
    ("easy", "What is 17 * 4?", 68),
    ("easy", "What is 144 divided by 12?", 12),
    ("easy", "What is 250 - 87?", 163),
    ("easy", "What is 15% of 300?", 45),
    ("easy", "A box holds 24 pencils. How many pencils are in 5 boxes?", 120),

    # --- tier "medium": two or three steps ---------------------------------
    ("medium", "A shop sells pens for 3 dollars and notebooks for 7 dollars. "
               "Tom buys 4 pens and 3 notebooks. How much does he pay?", 33),
    ("medium", "A train travels 60 km in 45 minutes. At the same speed, "
               "how many km does it travel in 2 hours?", 160),
    ("medium", "Alice is twice as old as Bob. In 5 years the sum of their ages "
               "will be 40. How old is Alice now?", 20),
    ("medium", "A rectangle is 3 times as long as it is wide. Its perimeter "
               "is 64 cm. What is its area in square cm?", 192),
    ("medium", "A shirt costs 40 dollars. It is discounted 25%, then the "
               "discounted price rises 20%. What is the final price?", 36),

    # --- tier "hard": four or more steps, or a trap -------------------------
    ("hard", "Three friends split a bill. Ann pays twice what Ben pays, and "
             "Cal pays 5 dollars more than Ben. The bill is 45 dollars. How "
             "many dollars does Ann pay?", 20),
    ("hard", "A worker is paid 18 dollars per hour for the first 40 hours in a "
             "week and 1.5 times that rate after 40 hours. She worked 47 "
             "hours. What is her pay in dollars?", 909),
    ("hard", "A number is 4 more than 3 times another number. Their sum is 60. "
             "What is the larger number?", 46),
    ("hard", "Machine A makes 12 parts per hour, machine B makes 18. They work "
             "together for 3 hours, then B stops and A works 2 more hours. "
             "How many parts in total?", 114),
    ("hard", "A book has 240 pages. Sam reads 1/4 of it on Monday, 1/3 of the "
             "remainder on Tuesday, and 40 pages on Wednesday. How many pages "
             "are left?", 80),
]

SUFFIX = " Give the final answer as a number after ####."


def load(model_id: str = MODEL_ID, threads: int = N_THREADS):
    """Load the model on CPU in float32 and return (tokenizer, model)."""
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"          # batch generate needs left padding
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def prompt_ids(tok, question: str, thinking: bool,
               system: str | None = None) -> list[int]:
    """Chat-template a question. `thinking=False` closes <think> immediately.

    `system` is how a serving stack *asks* for shorter thinking — a polite
    instruction in the prompt.  Section C measures how much that is worth
    against a hard runtime cap, which is the same request made with authority.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": question + SUFFIX}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def batch_generate(model, tok, id_lists, max_new: int, batch: int = 8):
    """Greedy-generate for a list of tokenised prompts.

    Returns (list_of_new_token_lists, wall_seconds).  Sequences are cut at the
    first EOS, so the returned lengths are the *real* output lengths — which is
    the whole quantity this project measures.
    """
    out, t0 = [], time.time()
    eos = {tok.eos_token_id}
    for i in range(0, len(id_lists), batch):
        chunk = id_lists[i:i + batch]
        width = max(len(x) for x in chunk)
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (width - len(x)) + x for x in chunk])
        att = torch.tensor([[0] * (width - len(x)) + [1] * len(x)
                            for x in chunk])
        gen = model.generate(input_ids=inp, attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad)
        for row in gen[:, width:].tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos:
                    cut = j
                    break
            out.append(row[:cut])
    return out, time.time() - t0


# ------------------------------------------------------- thinking surgery ---

def think_split(tok, new_ids: list[int]) -> tuple[list[int], list[int], bool]:
    """Split a generation into (thinking tokens, answer tokens, closed?).

    Qwen3 opens with a `<think>` token and closes with `</think>`.  The opening
    tag is stripped here so that "thinking tokens" counts only the reasoning
    itself.  If the cap was hit before the model closed the block, `closed` is
    False and there is no answer at all — the request burned its whole budget
    and produced nothing a user can read.  That case is the reason thinking
    budgets exist.
    """
    open_id = tok.convert_tokens_to_ids("<think>")
    close = tok.convert_tokens_to_ids("</think>")
    body = new_ids[1:] if new_ids and new_ids[0] == open_id else list(new_ids)
    if close in body:
        k = body.index(close)
        return body[:k], body[k + 1:], True
    return body, [], False


def budget_prompt(tok, base_ids: list[int], think_ids: list[int],
                  budget: int) -> list[int]:
    """Prompt that a server with a `budget`-token thinking cap would decode from.

    Take the thinking the model *did* produce, keep the first `budget` tokens of
    it, then force the closing tag.  Under greedy decoding this is byte-exact:
    the model would have generated those same tokens, and the server would have
    interrupted it at the same place.
    """
    open_id = tok.convert_tokens_to_ids("<think>")
    keep = think_ids[:budget]
    tail = tok("\n</think>\n\n", add_special_tokens=False).input_ids
    return base_ids + [open_id] + keep + tail


# --------------------------------------------------------------- grading ---

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract(text: str):
    """Pull the answer number out of a completion.

    Prefer what follows '####' (we asked for it); otherwise take the last number
    in the text, which is what every GSM8K-style grader does.
    """
    if "####" in text:
        text = text.split("####")[-1]
    hits = _NUM.findall(text.replace("$", " "))
    if not hits:
        return None
    try:
        return float(hits[-1].replace(",", ""))
    except ValueError:
        return None


def correct(text: str, answer: float) -> bool:
    got = extract(text)
    return got is not None and abs(got - answer) < 1e-6


def pct(xs, p: float) -> float:
    """Nearest-rank percentile (same convention as the rest of the guide)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])
