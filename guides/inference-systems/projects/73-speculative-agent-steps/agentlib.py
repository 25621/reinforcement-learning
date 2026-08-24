"""Shared pieces for project 73 — speculating the next *agent step*.

Speculative decoding guesses the next token, checks it with the big model, and
keeps it if it was right.  This project moves the same trick one level up: while
the model is still deciding which tool to call, start running the tool we think
it will pick.  If the guess was right the tool is already finished (or partly
finished) when the decision arrives; if it was wrong, throw the work away.

Two things make this honest rather than a toy:

* the tool choices come from a **real** model (Qwen2.5-0.5B-Instruct) reading a
  real growing conversation, and every model latency in the project is measured
  on this machine;
* the tools have **real, different latencies**, and some of them have side
  effects, which is what decides whether speculation is allowed at all.
"""

from __future__ import annotations

import os
import re
import sys
import time

import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 6

# name -> (latency seconds, read_only?)
TOOLS = {
    "list_dir":   (0.10, True),     # a directory listing: nearly free
    "read_file":  (0.30, True),
    "search":     (1.50, True),     # a repository-wide grep or a web search
    "run_tests":  (4.00, True),     # the expensive one
    "write_file": (0.60, False),    # changes the world: not speculatable
    "git_commit": (0.80, False),    # changes the world, visibly
}
TOOL_NAMES = list(TOOLS)

SYSTEM = (
    "You are a coding agent. At each step you call exactly one tool.\n"
    "Available tools: " + ", ".join(TOOL_NAMES) + ".\n"
    "Reply with one line only, in the form: TOOL: <tool_name>")

# 8 short tasks; each is a plausible agent episode.  Every one of them ends in
# an edit, because an agent that only reads is not the interesting case for
# section E: the question there is what happens when a *speculated* step would
# have changed something.
_END = " Finish by writing the change to the file and committing it."
TASKS = [
    "Find where the retry limit is set in this repository and raise it to 5."
    + _END,
    "A test named test_parser is failing. Diagnose it and fix the parser."
    + _END,
    "Add a --verbose flag to the CLI and make sure the tests still pass." + _END,
    "The README is out of date about installation. Correct it." + _END,
    "Remove the unused helper function in utils." + _END,
    "Find every place that hardcodes port 8080 and make it configurable." + _END,
    "The build is slow. Find the slowest test and mark it as optional." + _END,
    "A user reports a crash on empty input. Reproduce it and fix it." + _END,
]

# what a tool "returns", so the conversation grows like a real agent's
RESULTS = {
    "list_dir": "src/ tests/ README.md pyproject.toml",
    "read_file": "def retry(n=3): ...  # 40 lines omitted",
    "search": "3 matches in src/client.py, tests/test_retry.py, docs/faq.md",
    "run_tests": "12 passed, 1 failed: tests/test_retry.py::test_limit",
    "write_file": "wrote 24 lines to src/client.py",
    "git_commit": "created commit a91f3c2",
}


def load(model_id: str = MODEL_ID, threads: int = N_THREADS):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    model.eval()
    return tok, model


def build_prompt(tok, task: str, history: list[tuple[str, str]]) -> list[int]:
    lines = [f"Task: {task}"]
    for tool, res in history:
        lines.append(f"Called {tool} -> {res}")
    lines.append("Which tool do you call next?")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


_TOOL_RE = re.compile("|".join(re.escape(t) for t in TOOL_NAMES))


@torch.inference_mode()
def decide(model, tok, ids: list[int], max_new: int = 20):
    """One real agent decision. Returns (tool, seconds, n_tokens, text).

    The model writes a short line; we read the first tool name out of it.  If it
    writes something unparseable we fall back to the highest-scoring tool name
    under the model's own next-token distribution — a serving system needs a
    decision for every request, not an exception.
    """
    t0 = time.perf_counter()
    inp = torch.tensor([ids])
    out = model(inp, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    first_logits = out.logits[:, -1, :]
    toks = [int(nxt)]
    for _ in range(max_new - 1):
        if toks[-1] == tok.eos_token_id:
            break
        out = model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(int(nxt))
    secs = time.perf_counter() - t0
    text = tok.decode(toks)
    m = _TOOL_RE.search(text)
    if m:
        return m.group(0), secs, len(toks), text
    best, bs = TOOL_NAMES[0], -1e9
    for name in TOOL_NAMES:
        tid = tok(" " + name, add_special_tokens=False).input_ids[0]
        s = float(first_logits[0, tid])
        if s > bs:
            best, bs = name, s
    return best, secs, len(toks), text


# ------------------------------------------------------------- predictors ---

class FreqPredictor:
    """Guess the next tool from the previous one (a first-order Markov table).

    "First-order" means it looks one step back and nothing more — the cheapest
    predictor that can possibly be better than a constant guess.  It is trained
    on episodes the evaluation never sees.
    """

    def __init__(self):
        self.table: dict[str, dict[str, int]] = {}
        self.base: dict[str, int] = {}

    def fit(self, episodes: list[list[str]]):
        for ep in episodes:
            prev = "<start>"
            for tool in ep:
                self.table.setdefault(prev, {})
                self.table[prev][tool] = self.table[prev].get(tool, 0) + 1
                self.base[tool] = self.base.get(tool, 0) + 1
                prev = tool
        return self

    def predict(self, prev: str | None) -> str:
        row = self.table.get(prev or "<start>")
        if row:
            return max(row, key=row.get)
        return max(self.base, key=self.base.get) if self.base else TOOL_NAMES[0]


class ConstantPredictor:
    """Always guess the most common tool overall."""

    def __init__(self):
        self.best = TOOL_NAMES[0]

    def fit(self, episodes):
        c = {}
        for ep in episodes:
            for t in ep:
                c[t] = c.get(t, 0) + 1
        if c:
            self.best = max(c, key=c.get)
        return self

    def predict(self, prev=None):
        return self.best


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])
