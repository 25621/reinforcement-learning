"""Contamination detection: does a benchmark's test set already sit inside the
corpus a model was pretrained on?

The method is the one project 37 used for data filtering, turned around. There,
defects were injected into a crawl so every filter could be given a real
precision and recall. Here, *known* benchmark leaks are injected into a
pretraining shard, in four different disguises, so every detector can be given a
real precision and recall too. Without injected ground truth you can only report
"the detector fired 41 times" and have no idea how many it missed.

The four disguises, from easiest to hardest to catch:

    verbatim     the exact benchmark question and answer, copy-pasted into a
                 web page (the situation everybody imagines)
    reworded     the same question in different words -- a forum post, a
                 translation, a quiz site that retyped it
    answer-key   the answer sheet with no question ("12. B, 13. yes, ...")
    image-only   *no benchmark text at all*, but the benchmark's photograph is
                 in the corpus with somebody else's caption

The last one is why this project belongs in a multimodal guide. A text n-gram
scan -- the standard tool -- cannot see it, because there is no matching text to
find. The leak arrives as pixels.
"""

import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "42-run-a-vlm-evaluation-harness"))
sys.path.insert(0, str(PROJECTS / "37-mini-laion-pipeline"))
sys.path.insert(0, str(PROJECTS / "40-multimodal-dpo"))
import dpo_lib as D  # noqa: E402
import harness as H  # noqa: E402
import pipeline_lib as P  # noqa: E402

FLAVORS = ["verbatim", "reworded", "answer-key", "image-only"]


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def toks(text):
    return norm(text).split()


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# the benchmark and the pretraining shard
# ---------------------------------------------------------------------------
def make_benchmark(bank, n_pope=40, n_mcq=30):
    """A small mixed benchmark: yes/no existence questions plus caption
    multiple choice, both taken from project 42's harness so the questions are
    literally the ones a model would be scored on."""
    rng = np.random.default_rng(11)
    items = []
    pope = H.Pope("random")
    for d in pope.build(bank, np.arange(0, n_pope), rng)[:n_pope]:
        items.append({"id": len(items), "kind": "pope", "image": d["image"],
                      "question": d["question"], "answer": d["target"]})
    mcq = H.CaptionMCQ(circular=False, n_items=n_mcq)
    for d in mcq.build(bank, np.arange(0, 90), rng):
        items.append({"id": len(items), "kind": "mcq", "image": d["image"],
                      "question": d["question"], "answer": d["target"]})
    return items


SYNONYMS = {"is": "can we see", "there": "any", "image": "picture",
            "photo": "picture", "describes": "matches", "which": "what",
            "caption": "description", "answer": "reply"}


def reword(text, rng):
    """A crude paraphrase: swap a few words for synonyms and move the answer
    instruction to the front. It is deliberately crude -- if a detector cannot
    survive *this* much rewriting it will not survive a real forum post."""
    out = []
    for w in text.split():
        low = w.strip("?.,").lower()
        out.append(SYNONYMS.get(low, w) if low in SYNONYMS and rng.random() < 0.85
                   else w)
    s = " ".join(out)
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    if len(parts) > 1:
        parts = parts[1:] + parts[:1]
    return " ".join(parts)


def build_shard(bank, benchmark, leak_rate=0.5, n_clean=2400, seed=0):
    """-> (docs, leaked_ids)

    docs are {"text", "image" (or None), "leak" (flavor or None), "item" (or
    None)}. The clean half is real MS-COCO captions from photos the benchmark
    never asks about -- ordinary web-ish image-text pairs.
    """
    rng = np.random.default_rng(seed)
    bench_imgs = {b["image"] for b in benchmark}
    docs = []
    for i in range(bank.n_photos, bank.n_photos + n_clean):
        if i >= len(bank.captions):
            break
        docs.append({"text": bank.captions[i][0], "image": None,
                     "leak": None, "item": None})
    # a few hundred clean *image*-text pairs, so the image detector has
    # something to produce false positives on
    for i in range(90, bank.n_photos):
        if i not in bench_imgs:
            docs.append({"text": bank.captions[i][0], "image": int(i),
                         "leak": None, "item": None})

    leaked = {}
    order = rng.permutation(len(benchmark))
    n_leak = int(round(leak_rate * len(benchmark)))
    for k, idx in enumerate(order[:n_leak]):
        b = benchmark[int(idx)]
        flavor = FLAVORS[k % len(FLAVORS)]
        if flavor == "verbatim":
            text = f"Q: {b['question']} A: {b['answer']}"
            image = None
        elif flavor == "reworded":
            text = reword(f"{b['question']} The correct reply: {b['answer']}", rng)
            image = None
        elif flavor == "answer-key":
            text = f"Answer key, item {b['id']}: {b['answer']}"
            image = None
        else:                                   # image-only
            caps = bank.captions[b["image"]]
            text = caps[2 % len(caps)]          # a different annotator's words
            image = int(b["image"])
        docs.append({"text": text, "image": image, "leak": flavor,
                     "item": b["id"]})
        leaked[b["id"]] = flavor
    rng.shuffle(docs)
    return docs, leaked


# ---------------------------------------------------------------------------
# detectors -- each returns the set of benchmark item ids it flags
# ---------------------------------------------------------------------------
def detect_exact(benchmark, docs, **kw):
    """The strictest possible test: is the question, character for character
    (after lowercasing), a document in the corpus? Catches copy-paste and
    nothing else."""
    have = {norm(d["text"]) for d in docs}
    return {b["id"] for b in benchmark if norm(b["question"]) in have}


def _index(docs, n):
    idx = {}
    for k, d in enumerate(docs):
        for g in ngrams(toks(d["text"]), n):
            idx.setdefault(g, []).append(k)
    return idx


def detect_ngram(benchmark, docs, n=13, mask=None, **kw):
    """GPT-3's rule: a test item is contaminated if it shares **any** n-gram of
    n consecutive words with any document in the corpus.

    Why 13 and not 5: short word sequences repeat in ordinary English all the
    time ("in front of a"), so a small n flags everything and means nothing.
    Long ones essentially never repeat by chance, so a match is evidence of
    copying. The `--stage order` sweep in run.py measures exactly where that
    changeover happens on this corpus.

    `mask` removes a shared template from the benchmark question before
    matching -- see `template_ngrams`.
    """
    idx = _index(docs, n)
    hits = set()
    for b in benchmark:
        q = toks(b["question"])
        for g in ngrams(q, n):
            if mask and g in mask:
                continue
            if g in idx:
                hits.add(b["id"])
                break
    return hits


def template_ngrams(benchmark, n, min_share=0.25):
    """n-grams that appear in at least `min_share` of the benchmark's own
    questions -- that is, the boilerplate the benchmark itself repeats.

    This is the trap the project is built around. Every POPE question ends
    "in the image? Answer yes or no.", so any two POPE questions already share
    long n-grams with each other. Let one such question into the corpus and a
    plain n-gram scan flags the *whole* benchmark as contaminated.
    """
    c = Counter()
    for b in benchmark:
        c.update(ngrams(toks(b["question"]), n))
    need = max(2, int(min_share * len(benchmark)))
    return {g for g, v in c.items() if v >= need}


def _shingles(words, k=5):
    return ngrams(words, k) or {tuple(words)}


def minhash(words, n_hash=64, k=5, seed=0):
    """A 64-number sketch of a document's 5-word shingles.

    A MinHash is built by hashing every shingle 64 different ways and keeping
    the smallest value each time. The useful fact: for two documents, the
    probability that a given one of those 64 minimums agrees is exactly their
    *Jaccard similarity* -- the size of the shingle overlap divided by the size
    of the union. So counting agreements estimates the overlap without ever
    comparing the documents directly.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 31, n_hash, dtype=np.int64)
    b = rng.integers(0, 2 ** 31, n_hash, dtype=np.int64)
    # Python's built-in `hash` of a string is salted per process, so using it
    # here would give a different sketch on every run. CRC32 is stable.
    base = np.array([zlib.crc32(" ".join(s).encode()) for s in _shingles(words, k)],
                    dtype=np.int64)
    if len(base) == 0:
        return np.zeros(n_hash, dtype=np.int64)
    return ((a[:, None] * base[None, :] + b[:, None]) % (2 ** 61 - 1)).min(1)


def detect_minhash(benchmark, docs, thresh=0.4, n_hash=64, bands=16, **kw):
    """Near-duplicate detection: flag a test item whose estimated shingle
    overlap with some corpus document is above `thresh`.

    Unlike an n-gram scan this survives small edits, because dropping a word
    only removes a few shingles out of many. It is the standard deduplication
    tool (project 37 used the image version of the same idea) and it is the
    natural answer to a reworded leak.
    """
    sig_docs = np.stack([minhash(toks(d["text"]), n_hash) for d in docs])
    buckets = [{} for _ in range(bands)]
    rows = n_hash // bands
    for k in range(len(docs)):
        for bnd in range(bands):
            key = sig_docs[k, bnd * rows:(bnd + 1) * rows].tobytes()
            buckets[bnd].setdefault(key, []).append(k)
    hits = set()
    for b in benchmark:
        s = minhash(toks(b["question"]), n_hash)
        cand = set()
        for bnd in range(bands):
            cand |= set(buckets[bnd].get(
                s[bnd * rows:(bnd + 1) * rows].tobytes(), []))
        for k in cand:
            if float((sig_docs[k] == s).mean()) >= thresh:
                hits.add(b["id"])
                break
    return hits


def detect_embedding(benchmark, docs, model=None, tok=None, thresh=0.92,
                     batch=256, **kw):
    """Meaning-level matching: encode every question and every document with a
    frozen text encoder and flag anything whose nearest neighbour is close.

    This is the only text detector that can, in principle, catch a paraphrase
    with no shared wording. Its problem is the mirror image of the n-gram
    scan's: it also thinks two *different* questions in the same template are
    nearly identical, so its threshold has to be set very high, and even then
    it confuses "similar" with "copied".
    """
    import torch
    if model is None:
        model, tok = P.load_clip()

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), batch):
            e = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=77, return_tensors="pt")
            v = P._pooled(model.get_text_features(**e))
            out.append((v / v.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out)

    dv = enc([d["text"] for d in docs])
    qv = enc([b["question"] for b in benchmark])
    sims = qv @ dv.T
    return {b["id"] for b, s in zip(benchmark, sims.max(1)) if s >= thresh}


def detect_image(benchmark, docs, bank, max_dist=6, **kw):
    """Perceptual-hash lookup on the *pictures*.

    Project 37's `phash`: shrink to 32x32 grey, keep the 8x8 lowest-frequency
    DCT coefficients, record which are above their median. Two files of the
    same photo -- resized, re-compressed, re-saved -- land within a few bits of
    each other, so "the same picture" becomes a small integer distance. This is
    the only detector here that can see a leak carrying no benchmark text.
    """
    corpus = [(P.phash(bank.images[d["image"]]), d) for d in docs
              if d["image"] is not None]
    hits = set()
    for b in benchmark:
        h = P.phash(bank.images[b["image"]])
        for hc, _ in corpus:
            if P.hamming(h, hc) <= max_dist:
                hits.add(b["id"])
                break
    return hits


# ---------------------------------------------------------------------------
# scoring the detectors
# ---------------------------------------------------------------------------
def expand_image_leaks(benchmark, leaked):
    """Contamination is a property of the *photo*, not of the sentence.

    Our injection sheet records which benchmark item we leaked, but POPE asks
    two questions about each photo. If the photo lands in the corpus, both
    questions are compromised, even though only one of them is on the sheet.
    Without this correction the image detector looks like it is producing false
    positives when it is in fact right and the label sheet is wrong -- a good
    reminder that "is this item contaminated?" needs a definition before it
    needs a detector.
    """
    by_image = {}
    for b in benchmark:
        by_image.setdefault(b["image"], []).append(b["id"])
    out = dict(leaked)
    id2img = {b["id"]: b["image"] for b in benchmark}
    for item, flavor in leaked.items():
        if flavor == "image-only":
            for sib in by_image.get(id2img[item], []):
                out.setdefault(sib, "image-only")
    return out


def score_detector(flagged, leaked, benchmark, flavors=FLAVORS):
    truth = set(leaked)
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    per = {}
    for f in flavors:
        want = {i for i, v in leaked.items() if v == f}
        per[f] = len(flagged & want) / max(len(want), 1)
    return {"flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec,
            "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
            "recall_by_flavor": per}


# ---------------------------------------------------------------------------
# does contamination actually inflate a score?
# ---------------------------------------------------------------------------
class LookupModel:
    """A "model" whose entire ability is recall.

    It answers a benchmark question by finding the corpus document with the
    largest word overlap and copying any answer it can see there; if nothing is
    close enough, it guesses. It has no vision and no reasoning at all, so
    every point it scores above chance is contamination made visible. Sweeping
    the leak rate turns "contamination inflates scores" from a worry into a
    curve you can read a number off.
    """

    def __init__(self, docs, thresh=0.55, seed=0):
        self.docs = docs
        self.thresh = thresh
        self.rng = np.random.default_rng(seed)
        self.index = {}
        for k, d in enumerate(docs):
            for g in ngrams(toks(d["text"]), 4):
                self.index.setdefault(g, []).append(k)

    def _best(self, question):
        q = ngrams(toks(question), 4)
        cand = Counter()
        for g in q:
            for k in self.index.get(g, ()):
                cand[k] += 1
        if not cand:
            return None, 0.0
        k, hit = cand.most_common(1)[0]
        return self.docs[k], hit / max(len(q), 1)

    def answer(self, item):
        doc, sim = self._best(item["question"])
        if doc is not None and sim >= self.thresh:
            m = re.search(r"(?:A:|reply:|item \d+:)\s*([A-Da-d]|yes|no)\b",
                          doc["text"], re.IGNORECASE)
            if m:
                return m.group(1).upper() if len(m.group(1)) == 1 \
                    else m.group(1).lower()
        return ("yes" if self.rng.integers(2) else "no") if item["kind"] == "pope" \
            else "ABCD"[int(self.rng.integers(4))]


def bank():
    """Project 42's photo bank, shared rather than copied."""
    return H.Bank()
