"""Speculative decoding on real weights -- Phase 4's shared toy.

Phase 2 (`09/kvlib.py`) asked "what does the cache hold?" and ran one sequence
at a time. Phase 3 (`16/batchlib.py`) asked "who shares each forward pass?" and
ran many sequences at once. Phase 4 asks a third question that neither of them
can express: **what happens when a forward pass turns out to have been wrong?**

Speculation guesses several tokens ahead, checks them all in one pass, and
throws away the guesses that were wrong. Throwing them away means the KV cache
has to *shrink* -- entries written for a rejected token must disappear as if
they had never been written. Phase 2's `ContiguousCache` grows with
`torch.cat` and has no way back, so this file replaces it with a preallocated
buffer plus a length counter:

    cache.truncate(n)      # rejecting 3 tokens = `self.length -= 3`

That one line is the whole mechanical difference between ordinary decoding and
speculative decoding. Everything else here is bookkeeping around it.

The model arithmetic is *not* rewritten -- `Qwen2Runner` from project 09 is
imported unchanged, because its attention loop already reads through a cache
object we control. Only the cache is new.

Shared by projects 23, 24, 25, 26, 27 and 29.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

import torch  # noqa: E402

import kvlib  # noqa: E402
from kvlib import Qwen2Runner, interleaved, wikitext_lines  # noqa: E402,F401

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 6


# ---------------------------------------------------------------------------
# A cache you can take tokens back out of
# ---------------------------------------------------------------------------


class SpecCache(kvlib.KVCache):
    """Preallocated KV storage with a movable end marker.

    Why not reuse `ContiguousCache`? Because it stores each layer as a tensor
    that grows by `torch.cat`. Undoing an append would mean re-slicing 28
    tensors every time a draft token is rejected -- which happens on most
    iterations. Here the storage is allocated once at `max_len`, `append`
    writes into it, and *rollback is a subtraction*:

        length = 7  ->  truncate(5)  ->  length = 5

    The bytes for the two rejected tokens are still sitting in the buffer, but
    nothing can read them: attention only ever sees `[:, :, :length, :]`, and
    the next append overwrites them. That is exactly how a production engine
    handles rejection too -- the KV blocks stay allocated, the sequence length
    field moves back.
    """

    def __init__(self, n_layers, n_kv_heads, d_head, max_len=2048,
                 dtype=torch.float32):
        self.n_layers = n_layers
        self.max_len = max_len
        self.k = [torch.zeros(1, n_kv_heads, max_len, d_head, dtype=dtype)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(1, n_kv_heads, max_len, d_head, dtype=dtype)
                  for _ in range(n_layers)]
        self.length = 0

    def append(self, layer, k, v):
        t = k.shape[2]
        start = self.length              # same for every layer of one pass
        end = start + t
        if end > self.max_len:
            raise RuntimeError(f"SpecCache overflow: {end} > {self.max_len}")
        self.k[layer][:, :, start:end, :] = k
        self.v[layer][:, :, start:end, :] = v
        if layer == self.n_layers - 1:
            # Only advance once the whole forward pass has been written, so
            # every layer of one pass agrees on where it starts.
            self.length = end
        return self.k[layer][:, :, :end, :], self.v[layer][:, :, :end, :]

    def truncate(self, n: int):
        """Forget everything past position n. This is the rollback."""
        self.length = min(self.length, max(0, n))

    def reset(self):
        self.length = 0

    def n_tokens(self) -> int:
        return self.length

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def make_cache(runner: Qwen2Runner, max_len: int = 2048) -> SpecCache:
    return SpecCache(runner.n_layers, runner.n_kv_heads, runner.d_head, max_len)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_one(model_id: str, n_threads: int = N_THREADS):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return Qwen2Runner(model, tok), tok, model


def load_pair(target_id: str = TARGET_ID, draft_id: str = DRAFT_ID,
              n_threads: int = N_THREADS):
    """Both models share one tokenizer.

    This is not a detail -- it is the hard constraint on picking a draft.
    Verification compares *token ids*, so draft and target must agree on what
    id 9707 means. Qwen2.5-0.5B and Qwen2.5-1.5B ship the identical 151,936
    entry vocabulary, which is why they can be paired at all. A Llama draft in
    front of a Qwen target would need a detokenize-retokenize bridge on every
    single proposal, and the bridge costs more than the speculation saves.
    """
    target, tok, tmodel = load_one(target_id, n_threads)
    draft, dtok, dmodel = load_one(draft_id, n_threads)
    assert target.lm_head.shape[0] == draft.lm_head.shape[0], "vocab mismatch"
    return target, draft, tok, (tmodel, dmodel)


def chat_ids(tok, user_msg: str) -> torch.Tensor:
    """Wrap a message in Qwen's chat template, as a served request would."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids


# ---------------------------------------------------------------------------
# Baseline: ordinary greedy decoding
# ---------------------------------------------------------------------------


@torch.inference_mode()
def greedy_decode(runner: Qwen2Runner, cache: SpecCache, prompt_ids,
                  max_new: int = 48):
    """One token per forward pass. The thing speculation has to beat."""
    cache.reset()
    t0 = time.perf_counter()
    logits = runner.forward(prompt_ids, cache, start_pos=0)
    prefill_s = time.perf_counter() - t0

    tokens, step_s = [], []
    nxt = int(logits[0, -1].argmax())
    tokens.append(nxt)
    while len(tokens) < max_new:
        t0 = time.perf_counter()
        logits = runner.forward(torch.tensor([[nxt]]), cache,
                                start_pos=cache.length)
        step_s.append(time.perf_counter() - t0)
        nxt = int(logits[0, -1].argmax())
        tokens.append(nxt)
    return {
        "tokens": tokens,
        "prefill_s": prefill_s,
        "decode_s": sum(step_s),
        "steps": len(step_s) + 1,
        "tok_per_s": len(tokens) / (sum(step_s) + prefill_s),
    }


# ---------------------------------------------------------------------------
# Drafters
# ---------------------------------------------------------------------------


class ModelDrafter:
    """The classic drafter: a smaller model of the same family.

    It keeps its *own* KV cache, at its own length, and has to be rolled back
    on rejection exactly like the target -- a detail every from-scratch
    implementation gets wrong once. If you forget, the draft silently keeps
    conditioning on tokens the target already threw away.
    """

    name = "model"

    def __init__(self, runner: Qwen2Runner, max_len: int = 2048):
        self.runner = runner
        self.cache = make_cache(runner, max_len)
        self.draft_s = 0.0
        self.draft_passes = 0

    def reset(self):
        self.cache.reset()
        self.draft_s = 0.0
        self.draft_passes = 0

    def propose(self, tokens: list, k: int):
        """Return up to k proposed token ids continuing `tokens`."""
        t0 = time.perf_counter()
        out = []
        # Catch up: feed whatever the draft has not seen yet.
        missing = tokens[self.cache.length:]
        logits = self.runner.forward(torch.tensor([missing]), self.cache,
                                     start_pos=self.cache.length)
        self.draft_passes += 1
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        for _ in range(k - 1):
            logits = self.runner.forward(torch.tensor([[nxt]]), self.cache,
                                         start_pos=self.cache.length)
            self.draft_passes += 1
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
        self.draft_s += time.perf_counter() - t0
        return out

    def rollback(self, n_tokens: int):
        self.cache.truncate(n_tokens)


class NgramDrafter:
    """Prompt-lookup drafting: the "model" is the text itself.

    Take the last `n` tokens generated, search backwards through everything
    written so far for the same n tokens, and propose whatever followed them
    last time. Cost: a list scan. No weights, no cache, no second model.

    Project 25 is built entirely on this class.
    """

    name = "ngram"

    def __init__(self, max_n: int = 4, min_n: int = 2):
        self.max_n = max_n
        self.min_n = min_n
        self.draft_s = 0.0
        self.draft_passes = 0
        self.hits = 0
        self.misses = 0

    def reset(self):
        self.draft_s = 0.0
        self.draft_passes = 0
        self.hits = 0
        self.misses = 0

    def propose(self, tokens: list, k: int):
        t0 = time.perf_counter()
        out = []
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(tokens) <= n:
                continue
            pat = tokens[-n:]
            # Search the *most recent* match first: later text is a better
            # predictor of what comes next than the top of the document.
            for i in range(len(tokens) - n - 1, -1, -1):
                if tokens[i:i + n] == pat:
                    out = tokens[i + n:i + n + k]
                    break
            if out:
                break
        self.draft_s += time.perf_counter() - t0
        if out:
            self.hits += 1
        else:
            self.misses += 1
        return out

    def rollback(self, n_tokens: int):
        pass


# ---------------------------------------------------------------------------
# The speculative loop
# ---------------------------------------------------------------------------


@torch.inference_mode()
def speculative_greedy(target: Qwen2Runner, drafter, t_cache: SpecCache,
                       prompt_ids, k: int = 4, max_new: int = 48):
    """Greedy speculative decoding.

    The invariant that makes the bookkeeping simple:

        target cache length  ==  len(tokens) - 1

    i.e. the target has processed everything except the newest token. Each
    iteration therefore feeds `[last_token] + drafts` -- k+1 positions -- and
    gets back k+1 next-token predictions, one per position. The prediction
    made *at* the last accepted position is the free "bonus" token: the target
    computed it anyway, so even a completely useless drafter still yields one
    real token per pass and speculation can never be slower than 1 token per
    target forward pass.
    """
    t_cache.reset()
    drafter.reset()

    t0 = time.perf_counter()
    logits = target.forward(prompt_ids, t_cache, start_pos=0)
    prefill_s = time.perf_counter() - t0

    prompt = prompt_ids[0].tolist()
    tokens = prompt + [int(logits[0, -1].argmax())]
    t_cache.truncate(len(tokens) - 1)      # restore the invariant

    proposed = accepted = iters = tested = 0
    verify_s = 0.0
    accept_run = []                        # accepted-per-iteration histogram
    per_pos_hits = [0] * k                 # how often position i survived

    iter_s, iter_n = [], []                # duration and yield of each round

    while len(tokens) - len(prompt) < max_new:
        t_iter = time.perf_counter()
        drafts = drafter.propose(tokens, k)
        kk = len(drafts)
        tested_this = 0

        block = torch.tensor([[tokens[-1]] + drafts])
        t1 = time.perf_counter()
        logits = target.forward(block, t_cache, start_pos=len(tokens) - 1)
        verify_s += time.perf_counter() - t1
        preds = logits[0].argmax(-1).tolist()   # kk+1 predictions

        n_acc = 0
        for i in range(kk):
            tested_this += 1
            if drafts[i] == preds[i]:
                n_acc += 1
                per_pos_hits[i] += 1
            else:
                break
        tested += tested_this

        tokens.extend(drafts[:n_acc])
        tokens.append(preds[n_acc])             # the bonus token
        proposed += kk
        accepted += n_acc
        iters += 1
        accept_run.append(n_acc)

        t_cache.truncate(len(tokens) - 1)
        drafter.rollback(len(tokens) - 1)
        iter_s.append(time.perf_counter() - t_iter)
        iter_n.append(n_acc + 1)

    produced = len(tokens) - len(prompt)   # may exceed max_new: the last
    new_tokens = tokens[len(prompt):][:max_new]   # round emits up to k+1
    total_s = prefill_s + verify_s + drafter.draft_s
    return {
        "tokens": new_tokens,
        "produced": produced,
        "prefill_s": prefill_s,
        "verify_s": verify_s,
        "draft_s": drafter.draft_s,
        "decode_s": verify_s + drafter.draft_s,
        "iters": iters,
        "proposed": proposed,
        "accepted": accepted,
        "tested": tested,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "conditional_acceptance": accepted / tested if tested else 0.0,
        "tokens_per_iter": len(tokens[len(prompt):]) / iters if iters else 0.0,
        "accept_run": accept_run,
        "per_pos_hits": per_pos_hits,
        "iter_s": iter_s,
        "iter_n": iter_n,
        "itl": itl_from_bursts(iter_s, iter_n),
        "tok_per_s": len(new_tokens) / total_s,
    }


def itl_from_bursts(iter_s, iter_n):
    """Inter-token latencies as a *client* sees them.

    Speculation emits tokens in bursts: nothing arrives for the whole
    iteration, then several tokens land at once. So one iteration of duration
    d that yielded n tokens produces one gap of d and n-1 gaps of ~0. Averaging
    them gives the familiar mean ITL; the *tail* is what changes with k, and
    it is invisible unless you model the burst shape like this.
    """
    out = []
    for d, n in zip(iter_s, iter_n):
        out.append(d)
        out.extend([0.0] * max(0, n - 1))
    return out


# ---------------------------------------------------------------------------
# Sampling mode (project 24)
# ---------------------------------------------------------------------------


def probs_from(logits: torch.Tensor, temperature: float = 1.0,
               top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """Turn one row of logits into the distribution we will actually sample.

    This *is* the distribution the speculative-sampling guarantee is about.
    Whatever transforms you apply here -- temperature, top-k, top-p -- become
    part of "the model's output distribution", and both the draft and the
    target must be described by whatever they each actually sampled from.
    Project 24 section E shows what happens when they are not.
    """
    if temperature <= 0:
        out = torch.zeros_like(logits)
        out[int(logits.argmax())] = 1.0
        return out
    logits = logits.float() / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        srt, idx = torch.sort(probs, descending=True)
        keep = (torch.cumsum(srt, 0) - srt) < top_p    # always keeps the top 1
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx[keep]] = True
        probs = probs * mask
        probs = probs / probs.sum()
    return probs


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The distribution to draw from after a rejection: norm(max(0, p - q)).

    Intuition: the draft over-proposed some tokens (where q > p) and
    under-proposed others (where p > q). Accepting with probability p/q
    already trimmed the over-proposed ones down to size. What is missing from
    the output is exactly the part of p that q never covered -- the positive
    part of `p - q` -- so that is what a rejection must supply.
    """
    r = (p - q).clamp(min=0)
    s = r.sum()
    return r / s if s > 0 else p.clone()


def tv_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Total-variation distance: half the L1 gap between two distributions.

    Reads as "the largest disagreement about the probability of any event".
    0 means identical; 1 means they never agree. The factor of 1/2 is what
    makes the range [0, 1] instead of [0, 2].
    """
    return float(0.5 * (a - b).abs().sum())


def draw_many(p: torch.Tensor, q: torch.Tensor, n: int, mode: str,
              seed: int = 0, q_test: torch.Tensor | None = None
              ) -> torch.Tensor:
    """Draw n tokens through one speculative-verification rule, vectorised.

    Modes:
      rejection    the correct rule: accept with min(1, p/q), else draw from
                   the residual
      resample_p   the classic bug: on rejection draw from p
      always       accept everything (i.e. output the draft's distribution q)
      greedy_check greedy verification bolted onto random sampling: accept
                   only if the draft's token is the target's argmax

    `q_test` lets project 24 build the mismatch bug: the token is *drawn* from
    `q` but *scored* against `q_test`. In a correct implementation they are
    the same object.
    """
    g = torch.Generator().manual_seed(seed)
    qt = q if q_test is None else q_test
    x = torch.multinomial(q, n, replacement=True, generator=g)
    if mode == "always":
        return x
    if mode == "greedy_check":
        am = int(p.argmax())
        return torch.where(x == am, x, torch.full_like(x, am))
    r = torch.rand(n, generator=g)
    ratio = (p[x] / qt[x].clamp(min=1e-30)).clamp(max=1.0)
    accept = r < ratio
    fallback = p if mode == "resample_p" else residual(p, qt)
    y = torch.multinomial(fallback, n, replacement=True, generator=g)
    return torch.where(accept, x, y)


def empirical(samples: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(samples, minlength=vocab).float() / samples.numel()


@torch.inference_mode()
def speculative_sampling(target: Qwen2Runner, draft: Qwen2Runner,
                         t_cache: SpecCache, d_cache: SpecCache, prompt_ids,
                         k: int = 4, max_new: int = 32, temperature: float = 1.0,
                         top_p: float = 1.0, seed: int = 0,
                         mode: str = "rejection", draft_top_p: float | None = None,
                         test_top_p: float | None = None):
    """Speculative decoding for random sampling.

    Greedy verification asks "is the draft's token the one the target would
    pick?" -- a yes/no question. Random sampling has no single right answer,
    so the question becomes "how do I keep or replace this token so that the
    tokens coming out are distributed *exactly* as the target would have
    sampled them?" The answer is rejection sampling, and getting the
    replacement step wrong is silent: the text still reads fine.

    `draft_top_p` / `test_top_p` exist only so project 24 can build the
    mismatch bug on purpose: sample the draft from one distribution and score
    it with another.
    """
    g = torch.Generator().manual_seed(seed)
    t_cache.reset()
    d_cache.reset()
    dp = top_p if draft_top_p is None else draft_top_p
    tp = dp if test_top_p is None else test_top_p

    def draw(pr):
        return int(torch.multinomial(pr, 1, generator=g))

    t0 = time.perf_counter()
    logits = target.forward(prompt_ids, t_cache, start_pos=0)
    prefill_s = time.perf_counter() - t0
    prompt = prompt_ids[0].tolist()
    tokens = prompt + [draw(probs_from(logits[0, -1], temperature, top_p))]
    t_cache.truncate(len(tokens) - 1)

    proposed = accepted = iters = tested = 0
    draft_s = verify_s = 0.0
    tvs, accept_run = [], []

    while len(tokens) - len(prompt) < max_new:
        t1 = time.perf_counter()
        missing = tokens[d_cache.length:]
        dl = draft.forward(torch.tensor([missing]), d_cache,
                           start_pos=d_cache.length)
        qs_sample = [probs_from(dl[0, -1], temperature, dp)]
        qs_test = [probs_from(dl[0, -1], temperature, tp)]
        drafts = [draw(qs_sample[0])]
        for _ in range(k - 1):
            dl = draft.forward(torch.tensor([[drafts[-1]]]), d_cache,
                               start_pos=d_cache.length)
            qs_sample.append(probs_from(dl[0, -1], temperature, dp))
            qs_test.append(probs_from(dl[0, -1], temperature, tp))
            drafts.append(draw(qs_sample[-1]))
        draft_s += time.perf_counter() - t1

        t1 = time.perf_counter()
        block = torch.tensor([[tokens[-1]] + drafts])
        tl = target.forward(block, t_cache, start_pos=len(tokens) - 1)
        verify_s += time.perf_counter() - t1
        ps = [probs_from(tl[0, i], temperature, top_p) for i in range(k + 1)]

        n_acc = 0
        for i in range(k):
            x = drafts[i]
            tested += 1                    # positions actually looked at
            tvs.append(tv_distance(ps[i], qs_test[i]))
            if mode == "greedy_check":
                ok = x == int(ps[i].argmax())
            else:
                ratio = min(1.0, float(ps[i][x] / max(float(qs_test[i][x]), 1e-30)))
                ok = float(torch.rand((), generator=g)) < ratio
            if ok:
                n_acc += 1
            else:
                break

        if n_acc == k:
            nxt = draw(ps[k])
        elif mode == "resample_p":
            nxt = draw(ps[n_acc])
        elif mode == "greedy_check":
            nxt = int(ps[n_acc].argmax())
        else:
            nxt = draw(residual(ps[n_acc], qs_test[n_acc]))

        tokens.extend(drafts[:n_acc])
        tokens.append(nxt)
        proposed += k
        accepted += n_acc
        iters += 1
        accept_run.append(n_acc)
        t_cache.truncate(len(tokens) - 1)
        d_cache.truncate(len(tokens) - 1)

    new_tokens = tokens[len(prompt):][:max_new]
    return {
        "tokens": new_tokens,
        "prefill_s": prefill_s,
        "draft_s": draft_s,
        "verify_s": verify_s,
        "decode_s": draft_s + verify_s,
        "iters": iters,
        "proposed": proposed,
        "accepted": accepted,
        "tested": tested,
        # Two different questions. `acceptance_rate` divides by everything
        # proposed, including positions verification never reached because it
        # stops at the first mismatch. `conditional_acceptance` divides by the
        # positions actually looked at -- that is the one the theory predicts.
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "conditional_acceptance": accepted / tested if tested else 0.0,
        "tokens_per_iter": len(tokens[len(prompt):]) / iters if iters else 0.0,
        "accept_run": accept_run,
        "mean_tv": sum(tvs) / len(tvs) if tvs else 0.0,
        "one_minus_tv": 1 - (sum(tvs) / len(tvs)) if tvs else 0.0,
    }


@torch.inference_mode()
def sample_decode(runner: Qwen2Runner, cache: SpecCache, prompt_ids,
                  max_new: int = 32, temperature: float = 1.0,
                  top_p: float = 1.0, seed: int = 0):
    """Plain (non-speculative) random sampling -- the baseline for project 24."""
    g = torch.Generator().manual_seed(seed)
    cache.reset()
    t0 = time.perf_counter()
    logits = runner.forward(prompt_ids, cache, start_pos=0)
    prefill_s = time.perf_counter() - t0
    tokens, step_s = [], []
    pr = probs_from(logits[0, -1], temperature, top_p)
    nxt = int(torch.multinomial(pr, 1, generator=g))
    tokens.append(nxt)
    while len(tokens) < max_new:
        t0 = time.perf_counter()
        logits = runner.forward(torch.tensor([[nxt]]), cache,
                                start_pos=cache.length)
        step_s.append(time.perf_counter() - t0)
        pr = probs_from(logits[0, -1], temperature, top_p)
        nxt = int(torch.multinomial(pr, 1, generator=g))
        tokens.append(nxt)
    return {"tokens": tokens, "prefill_s": prefill_s,
            "decode_s": sum(step_s), "steps": len(step_s) + 1}


# ---------------------------------------------------------------------------
# Small helpers used by several projects
# ---------------------------------------------------------------------------


def speedup_model(alpha_len: float, k: int, cost_ratio: float,
                  verify_overhead: float = 0.0) -> float:
    """Predicted wall-clock speedup of speculation, from three numbers.

    alpha_len       tokens produced per target forward pass (>= 1)
    k               drafts proposed per iteration
    cost_ratio      one draft forward pass / one target forward pass
    verify_overhead extra cost of a k+1-wide target pass vs a 1-wide one,
                    as a fraction (0.18 means "18% more expensive")

    Baseline spends 1 target pass per token. Speculation spends
    (1 + verify_overhead) target passes plus k * cost_ratio target-equivalents
    per alpha_len tokens.
    """
    per_iter = (1.0 + verify_overhead) + k * cost_ratio
    return alpha_len / per_iter


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[i]


WORKLOADS = {
    "chat": (
        "Explain in a short paragraph why the sky looks blue during the day."
    ),
    "summarize": (
        "Summarise the following passage in two sentences.\n\n"
        "The Antikythera mechanism is an ancient Greek hand-powered device "
        "that has been described as the oldest known analogue computer. It "
        "was used to predict astronomical positions and eclipses decades in "
        "advance, and to track the four-year cycle of athletic games. The "
        "artefact was recovered in 1901 from a shipwreck off the coast of the "
        "Greek island of Antikythera, and its complexity was not matched by "
        "any known device for well over a thousand years."
    ),
    "code": (
        "Complete this Python function. Repeat the whole function in your "
        "answer.\n\n"
        "def normalise_scores(scores):\n"
        "    total = sum(scores)\n"
        "    if total == 0:\n"
        "        return [0.0 for s in scores]\n"
        "    return [\n"
    ),
    "json": (
        "Return a JSON object with the keys \"name\", \"city\", \"country\" "
        "and \"population\" for the city of Lyon, France. Output JSON only."
    ),
}
