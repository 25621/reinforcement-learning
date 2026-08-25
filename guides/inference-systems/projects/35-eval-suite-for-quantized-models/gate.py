"""A deploy gate for quantized models -- and the machinery to audit the gate.

A gate is easy to write and easy to get wrong. The failure is not that it
crashes; it is that it *has no resolution*. If a metric moves by +-4 points
purely because you drew a different 120 questions, then a rule saying "block if
MMLU drops by more than 3 points" is a coin flip wearing a lab coat: it blocks
healthy models and waves damaged ones through, and it does both silently.

So this module keeps two things apart:

  `EvalResult`  -- the *per-item* record (was question 47 right? what was the
                   loss on window 3? did token 219 match production?). Aggregate
                   scores are computed from it, never stored instead of it.
  `Gate`        -- a set of named checks with thresholds, run against two
                   `EvalResult`s.

Keeping the per-item record is what makes the audit possible. Once you have it,
resampling the eval set costs microseconds, so "how often would this gate be
wrong?" becomes a question you can answer instead of a thing you assume.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Everything one model scored, at the finest granularity available."""
    name: str
    mmlu_correct: list = field(default_factory=list)   # 0/1 per question
    mmlu_choice: list = field(default_factory=list)    # 0..3 per question
    window_nll: list = field(default_factory=list)     # mean nll per window
    window_ntok: list = field(default_factory=list)
    token_match: list = field(default_factory=list)    # 0/1 vs baseline, per token
    gen_divergence: list = field(default_factory=list)  # first differing token
    gen_len: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)        # cost of each eval

    # -- aggregate scores ----------------------------------------------------

    def mmlu(self, idx=None):
        c = self.mmlu_correct if idx is None else [self.mmlu_correct[i] for i in idx]
        return sum(c) / len(c)

    def ppl(self, idx=None):
        rng = range(len(self.window_nll)) if idx is None else idx
        tot = sum(self.window_nll[i] * self.window_ntok[i] for i in rng)
        n = sum(self.window_ntok[i] for i in rng)
        return math.exp(tot / n)

    def agreement(self, idx=None):
        m = self.token_match if idx is None else [self.token_match[i] for i in idx]
        return sum(m) / len(m) if m else 1.0

    def gen_identical(self):
        return sum(1 for d, L in zip(self.gen_divergence, self.gen_len)
                   if d >= L) / max(len(self.gen_len), 1)


# ---------------------------------------------------------------------------


# What a team writes on day one: round numbers that sound careful. Every one of
# them is a guess, and section E measures what the guesses cost.
NAIVE_CHECKS = {
    "perplexity": dict(kind="ratio", max=1.05),
    "mmlu": dict(kind="drop_pts", max=3.0),
    "shadow_agreement": dict(kind="min", min=0.98),
    "generation_identical": dict(kind="min", min=0.50),
}

# The same checks with thresholds derived from two measurements instead of
# intuition: each metric's own noise floor (section B) and what a candidate
# known to be harmless actually scores (INT8 per-channel). A threshold tighter
# than the noise floor decides at random; a threshold tighter than the harmless
# reference blocks models that are fine.
CALIBRATED_CHECKS = {
    "perplexity": dict(kind="ratio", max=1.05),
    "mmlu": dict(kind="drop_pts", max=8.4),        # 2 s.d. at n = 140
    "shadow_agreement": dict(kind="min", min=0.95),
    "generation_identical": dict(kind="min", min=0.30),
}

DEFAULT_CHECKS = NAIVE_CHECKS


class Gate:
    """`verdict(baseline, candidate)` -> (allowed, per-check detail)."""

    def __init__(self, checks=None):
        self.checks = dict(checks or DEFAULT_CHECKS)

    def verdict(self, base: EvalResult, cand: EvalResult, idx=None):
        idx = idx or {}
        detail, fails = {}, []
        for name, spec in self.checks.items():
            if name == "perplexity":
                v = cand.ppl(idx.get("window")) / base.ppl(idx.get("window"))
                ok = v <= spec["max"]
            elif name == "mmlu":
                v = 100 * (base.mmlu(idx.get("mmlu")) - cand.mmlu(idx.get("mmlu")))
                ok = v <= spec["max"]
            elif name == "shadow_agreement":
                v = cand.agreement(idx.get("token"))
                ok = v >= spec["min"]
            elif name == "generation_identical":
                v = cand.gen_identical()
                ok = v >= spec["min"]
            else:
                raise KeyError(name)
            detail[name] = {"value": v, "pass": ok, **spec}
            if not ok:
                fails.append(name)
        return (not fails), detail, fails


# ---------------------------------------------------------------------------
# auditing the gate
# ---------------------------------------------------------------------------


def bootstrap_spread(values, n_boot=2000, seed=0, stat=None):
    """Resample the per-item scores with replacement and report the spread.

    This is the *bootstrap*, named after "pulling yourself up by your own
    bootstraps": you have one sample and no way to draw another from the world,
    so you draw new samples from the sample you already have. If accuracy on
    120 questions swings by +-4 points across those redraws, then 4 points is
    what your eval cannot see, no matter how carefully you set the threshold."""
    rng = random.Random(seed)
    n = len(values)
    stat = stat or (lambda v: sum(v) / len(v))
    out = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        out.append(stat(s))
    out.sort()
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    return {"mean": mean, "sd": math.sqrt(var),
            "p2.5": out[int(0.025 * len(out))],
            "p97.5": out[int(0.975 * len(out))]}


def split_half_verdicts(gate, base, cand, n_trials=400, seed=0):
    """Run the gate `n_trials` times on random halves of the eval pool.

    Each trial is a legitimate way the eval set could have been drawn. The
    fraction of trials that BLOCK is the gate's decision noise: on a candidate
    that is genuinely identical to the baseline, every block is a false block."""
    rng = random.Random(seed)
    blocks, per_check = 0, {}
    nm, nw, nt = (len(base.mmlu_correct), len(base.window_nll),
                  len(base.token_match))
    for _ in range(n_trials):
        idx = {"mmlu": rng.sample(range(nm), nm // 2),
               "window": rng.sample(range(nw), max(nw // 2, 1)),
               "token": rng.sample(range(nt), nt // 2) if nt else None}
        ok, _, fails = gate.verdict(base, cand, idx)
        blocks += 0 if ok else 1
        for f in fails:
            per_check[f] = per_check.get(f, 0) + 1
    return {"block_rate": blocks / n_trials,
            "per_check_block_rate": {k: v / n_trials for k, v in per_check.items()},
            "n_trials": n_trials}


def separation(base: EvalResult, cands, metric):
    """Signal-to-noise of one metric: how far it moves per unit of its own
    sampling noise. A metric with separation < 1 cannot resolve the damage it
    is being asked to detect, however severe that damage looks on a chart."""
    if metric == "mmlu":
        noise = bootstrap_spread(base.mmlu_correct)["sd"] * 100
        vals = [100 * (base.mmlu() - c.mmlu()) for c in cands]
    elif metric == "perplexity":
        noise = bootstrap_spread(base.window_nll)["sd"]
        vals = [math.log(c.ppl() / base.ppl()) for c in cands]
        noise = noise or 1e-9
    elif metric == "shadow_agreement":
        # A truly identical model agrees on every token, so the null has zero
        # variance and the separation would be infinite -- true, and useless as
        # a number. The honest quantity is the standard error of the *estimate*
        # itself: with n scored tokens and an observed disagreement rate p,
        # that is sqrt(p(1-p)/n). Averaged over the candidates so the figure is
        # one number per metric, like the others.
        n = max(len(base.token_match), 1)
        ps = [1.0 - c.agreement() for c in cands]
        noise = sum(math.sqrt(max(p * (1 - p), 1e-12) / n) for p in ps) / len(ps)
        vals = ps
    else:
        raise KeyError(metric)
    return {"noise_sd": noise, "shifts": vals,
            "separation": [abs(v) / max(noise, 1e-9) for v in vals]}
