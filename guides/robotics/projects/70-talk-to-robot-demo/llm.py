"""Score every skill sentence under a real language model, quickly.

The model is SmolLM2-360M-Instruct, downloaded once and run on the CPU.  It is
small, and that is deliberate: everything this project measures -- length bias,
the affordance gap, the difference between generating a plan and scoring one --
shows up more sharply on a small model, and you can re-run the whole thing in
minutes instead of renting a GPU.

The one performance trick worth knowing
---------------------------------------
SayCan scores ALL skills at every step.  With 26 skills and a 90-token prompt,
the naive way runs the 90-token prompt through the network 26 times.  Those 26
passes compute exactly the same thing.

So we run the prompt once, keep the network's internal summary of it (the
"KV cache" -- the keys and values every attention layer computed for those
tokens), copy that summary 26 times, and then push only the ~6 tokens of each
skill sentence through.  Same numbers, checked to the last decimal against the
slow version in ``run.py``; about 3x less work.
"""

import os
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


class Scorer:
    """log P(skill sentence | prompt) for a whole list of skills at once."""

    def __init__(self, model_id=MODEL_ID, threads=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32).eval()
        self._tok_cache = {}
        self.calls = 0
        self.score_seconds = 0.0
        self.gen_seconds = 0.0

    def _ids(self, text):
        if text not in self._tok_cache:
            self._tok_cache[text] = self.tok(text, add_special_tokens=False).input_ids
        return self._tok_cache[text]

    @torch.no_grad()
    def score(self, prompt, options):
        """Return (total_logprob, n_tokens) for each option.

        ``total_logprob`` is the sum of log-probabilities of the option's
        tokens.  It is NOT a probability of the sentence being a good idea; it
        is how unsurprised the model is to read that sentence next.  The token
        count comes back too, because section 3 of the README needs it.
        """
        self.calls += 1
        t_start = time.time()
        pid = torch.tensor([self.tok(prompt).input_ids])
        pre = self.model(pid, use_cache=True)
        cache = pre.past_key_values
        first = F.log_softmax(pre.logits[0, -1].float(), -1)

        toks = [self._ids(" " + o) for o in options]
        B, L = len(toks), max(len(t) for t in toks)
        P = pid.shape[1]
        ids = torch.full((B, L), self.tok.pad_token_id, dtype=torch.long)
        keep = torch.zeros(B, L, dtype=torch.bool)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = torch.tensor(t)
            keep[i, :len(t)] = True
        cache.batch_repeat_interleave(B)
        am = torch.cat([torch.ones(B, P, dtype=torch.long), keep.long()], 1)
        pos = torch.arange(P, P + L)[None].expand(B, L)
        out = self.model(ids, attention_mask=am, position_ids=pos,
                         past_key_values=cache, use_cache=False)
        lp = F.log_softmax(out.logits.float(), -1)

        scores, lens = [], []
        for i, t in enumerate(toks):
            s = float(first[t[0]])
            if len(t) > 1:
                idx = torch.tensor(t[1:])
                s += float(lp[i, :len(t) - 1].gather(1, idx[:, None]).sum())
            scores.append(s)
            lens.append(len(t))
        self.score_seconds += time.time() - t_start
        return scores, lens

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=90):
        """Let the model write a plan in its own words (the section-2 baseline)."""
        t_start = time.time()
        pid = torch.tensor([self.tok(prompt).input_ids])
        out = self.model.generate(pid, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        self.gen_seconds += time.time() - t_start
        return self.tok.decode(out[0, pid.shape[1]:], skip_special_tokens=True)
