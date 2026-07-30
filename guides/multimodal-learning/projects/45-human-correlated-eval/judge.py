"""Do the automatic judges agree with people -- and how well do people agree
with each other?

The honest problem this project has to solve first: we have no human raters. The
way out is to notice that MS-COCO already contains something better than a
rating panel -- **five independent people describing the same photograph**. Any
two of those descriptions disagree about what is worth mentioning, which is
exactly the disagreement a rating panel has.

So three "human raters" are constructed here, one per annotator: rater k scores
a caption by how well its content lines up with annotator k's own sentence,
using one fixed rule. The rule is mechanical, but the *disagreement between the
three raters is genuinely human* -- it comes from three people looking at one
picture and choosing different things to say about it. That disagreement is the
ceiling: no automatic judge should be expected to agree with the panel better
than the panel agrees with itself.

What this proxy panel is and is not
-----------------------------------
It IS a real measurement of annotator disagreement, and a real upper bound.
It is NOT a measurement of what people think "quality" means -- a human rater
would also notice fluency, detail and tone, and our rule is blind to all three.
Every number below should be read as "agreement about *content*".
"""

import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "42-run-a-vlm-evaluation-harness"))
sys.path.insert(0, str(PROJECTS / "40-multimodal-dpo"))
sys.path.insert(0, str(PROJECTS / "37-mini-laion-pipeline"))
import dpo_lib as D  # noqa: E402
import harness as H  # noqa: E402
import pipeline_lib as P  # noqa: E402

SCALE = (1, 5)


# ---------------------------------------------------------------------------
# the outputs to be rated
# ---------------------------------------------------------------------------
def systems():
    return ["smolvlm-256m", "smolvlm-500m", "blip", "swapped", "truncated"]


@torch.no_grad()
def blip_captions(images, batch=8, max_new=20):
    """Project 37's recaptioner (BLIP-base) run straight, without its
    id-keyed cache -- these photos are not that project's crawl records."""
    model, proc = P.load_captioner()
    out = []
    for k in range(0, len(images), batch):
        pil = [Image.fromarray(a) for a in images[k:k + batch]]
        gen = model.generate(**proc(images=pil, return_tensors="pt"),
                             max_new_tokens=max_new, num_beams=1)
        out += [t.strip() for t in proc.batch_decode(gen, skip_special_tokens=True)]
    return out


def make_outputs(bank, ids, cache=None):
    """One caption per (system, image). Five systems x 20 photos = 100 outputs.

    The five are chosen to span the failure modes a judge has to tell apart:
    two real VLMs, a real captioner, a *fluent but wrong* caption (a human
    caption of a different photo -- the hallucination case), and a real caption
    cut to three words (correct as far as it goes, but useless). A judge that
    cannot separate those is not measuring quality.
    """
    import json
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = {}
    docs = [{"image": int(i),
             "question": "Describe the image in one short sentence."}
            for i in ids]
    task = H.CaptionGen()
    images = np.stack([bank.images[i] for i in ids])
    for repo, name in (("HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm-256m"),
                       ("HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm-500m")):
        m = H.SmolVLMModel(repo)
        out[name] = [t.strip() for t in m.predict(task, docs, images,
                                                  verbose=False)]
        del m
    out["blip"] = blip_captions(images)
    rng = np.random.default_rng(3)
    shift = rng.permutation(len(ids))
    out["swapped"] = [bank.captions[int(ids[(k + 1 + shift[k] % 3) % len(ids)])][0]
                      for k in range(len(ids))]
    out["truncated"] = [" ".join(bank.captions[int(i)][0].split()[:3])
                        for i in ids]
    if cache:
        Path(cache).write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# the human panel
# ---------------------------------------------------------------------------
def human_rating(caption, reference):
    """Score `caption` against one annotator's sentence, on 1-5.

    hits    objects the two agree on
    extra   objects the caption claims that this annotator did not mention
    missed  objects the annotator mentioned that the caption ignores

    The scale starts at 3 ("says nothing wrong, says nothing much"), rises with
    agreement and falls with unsupported claims. `extra` is punished harder than
    `missed`, because inventing an object is a worse error than skipping one --
    the same asymmetry project 40's hallucination meter uses.
    """
    c = set(D.objects_in([caption]))
    r = set(D.objects_in([reference]))
    hits, extra, missed = len(c & r), len(c - r), len(r - c)
    score = 3 + 1.25 * hits - 1.0 * extra - 0.5 * missed
    if not c:
        score = min(score, 2.0)          # an output naming nothing cannot be a 5
    return int(np.clip(round(score), *SCALE))


def human_panel(bank, ids, outputs, n_raters=3):
    """-> ratings[rater][system] = list over images."""
    panel = {}
    for k in range(n_raters):
        panel[f"human-{k+1}"] = {
            s: [human_rating(outputs[s][j],
                             bank.captions[int(i)][k % len(bank.captions[int(i)])])
                for j, i in enumerate(ids)]
            for s in outputs}
    return panel


# ---------------------------------------------------------------------------
# the automatic judges
# ---------------------------------------------------------------------------
JUDGE_PROMPT = ("Here is a caption of the image:\n\"{cap}\"\n"
                "How well does the caption describe the image? "
                "Reply with a single number from 1 (bad) to 5 (perfect).")


def parse_score(text):
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else 3       # unparsed -> the neutral middle


class VlmJudge:
    """A real VLM shown the photo and the caption, asked for a number.

    This is *LLM-as-judge* in miniature. It is the cheap substitute the field
    reaches for when open-ended outputs need grading, and the whole point of
    this project is to find out whether the substitute is any good.
    """

    def __init__(self, repo="HuggingFaceTB/SmolVLM-256M-Instruct", blind=False):
        self.blind = blind
        self.name = ("judge-blind" if blind else
                     "judge-" + repo.split("/")[-1].replace("-Instruct", "").lower())
        self.m = H.SmolVLMModel(repo)

    @torch.no_grad()
    def rate(self, bank, ids, captions, bs=8):
        out = []
        for k in range(0, len(ids), bs):
            caps = captions[k:k + bs]
            idx = ids[k:k + bs]
            # A blind judge sees a blank grey card instead of the photo. Same
            # model, same prompt, same number of image tokens -- the only thing
            # removed is the evidence.
            pil = [Image.fromarray(np.full_like(bank.images[0], 128) if self.blind
                                   else bank.images[int(i)]) for i in idx]
            text = [self.m._prompt(JUDGE_PROMPT.format(cap=c)) for c in caps]
            inp = self.m.proc(text=text, images=[[p] for p in pil],
                              padding=True, return_tensors="pt")
            gen = self.m.model.generate(**inp, max_new_tokens=4, do_sample=False)
            dec = self.m.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            out += [parse_score(t) for t in dec]
        return out


class ClipScoreJudge:
    """CLIPScore: no references, no generation -- just how close the caption sits
    to the photo in a frozen CLIP's shared space, rescaled onto 1-5.

    It is in the panel because it is the cheapest automatic judge that exists
    and is widely used for exactly this. Whether "close in CLIP space" tracks
    "a person would call this a good caption" is the question."""

    name = "clipscore"

    def __init__(self):
        self.model, self.tok = P.load_clip()

    def rate(self, bank, ids, captions, bs=64):
        imgs = np.stack([bank.images[int(i)] for i in ids])
        s, _ = P.clip_scores(imgs, list(captions), self.model, self.tok,
                             verbose=False)
        return list(np.asarray(s))


class CiderJudge:
    """CIDEr against the five human captions: the reference-based metric the
    captioning benchmarks are actually scored with. It sees the answer key, so
    it should have an unfair advantage over every other judge here."""

    name = "cider"

    def __init__(self, bank):
        self.bank = bank

    def rate(self, bank, ids, captions, **kw):
        refs = [bank.captions[int(i)] for i in ids]
        _, per = H.cider(list(captions), refs)
        return per


# ---------------------------------------------------------------------------
# agreement statistics, implemented here because scipy is not installed
# ---------------------------------------------------------------------------
def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def rankdata(x):
    """Average ranks, ties shared -- what Spearman's correlation needs."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    """Pearson's correlation computed on *ranks* instead of values.

    Named after Charles Spearman, who introduced it in 1904. Ranks are the right
    currency for ratings: they only assume that a 5 is better than a 4, not that
    the step from 4 to 5 is the same size as the step from 1 to 2 -- which no
    rating scale, human or model, actually guarantees.
    """
    return pearson(rankdata(a), rankdata(b))


def cohen_kappa(a, b, levels=None):
    """Chance-corrected agreement between two raters (Jacob Cohen, 1960).

    Raw agreement flatters everyone: if two raters both hand out 4s most of the
    time they will agree often by luck alone. Kappa (the Greek letter kappa)
    subtracts that luck:

        kappa = (observed agreement - agreement expected by chance)
                / (1 - agreement expected by chance)

    1.0 is perfect, 0.0 is "no better than two people guessing independently
    with the same habits", and negative means they disagree more than chance.
    """
    a, b = np.asarray(a), np.asarray(b)
    levels = levels if levels is not None else sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(levels)}
    m = np.zeros((len(levels), len(levels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    m /= max(m.sum(), 1)
    po = float(np.trace(m))
    pe = float((m.sum(0) * m.sum(1)).sum())
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def fleiss_kappa(ratings, levels):
    """The same idea for a whole panel (Joseph Fleiss, 1971): how much do N
    raters agree on each item, above what their overall habits predict?"""
    counts = np.zeros((len(ratings[0]), len(levels)))
    idx = {v: i for i, v in enumerate(levels)}
    for r in ratings:
        for j, v in enumerate(r):
            counts[j, idx[v]] += 1
    n = counts.sum(1)[0]
    if n < 2:
        return 0.0
    p_item = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_bar = float(p_item.mean())
    p_e = float(((counts.sum(0) / counts.sum()) ** 2).sum())
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 0.0


def bin3(x, lo, hi):
    """Squash a continuous score onto three bands so a kappa can be computed
    against 1-5 human ratings. Kappa needs categories; correlations do not."""
    x = np.asarray(x, dtype=float)
    return np.digitize(x, [lo, hi]) + 1


def pairwise(matrix):
    """Mean pairwise Spearman among a list of rating vectors."""
    vals = [spearman(a, b) for a, b in combinations(matrix, 2)]
    return float(np.mean(vals)) if vals else 0.0
