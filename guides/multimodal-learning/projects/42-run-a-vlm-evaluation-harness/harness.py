"""A miniature evaluation harness for vision-language models.

Real harnesses (`lmms-eval`, `VLMEvalKit`) are thousands of lines, but the shape
of the thing is small and worth building once by hand:

    Task    knows how to turn a dataset into questions, and how to grade one
            answer.  It never knows which model is answering.
    Model   knows how to turn (image, question) into a string.  It never knows
            which benchmark it is sitting.
    run()   walks every (task, model) pair and writes one row per cell.

That separation is the whole product. Once a task is written down as data plus a
grading function, every model is asked *exactly* the same thing and graded by
*exactly* the same parser, which is the only way two numbers from two papers can
be compared at all.

Eight tasks live here, chosen to cover the same ground as the suite in the
guide's Phase 9 table:

    pope-random       object existence, easy negatives      (hallucination)
    pope-adversarial  object existence, co-occurring negatives
    mmbench-mini      4-way multiple choice + circular eval (general QA)
    caption-match     4-way multiple choice over sentences  (sentence grounding)
    ocr-mini          read a word painted onto the photo    (OCR / documents)
    count-mini        count the shapes on a canvas          (structured reading)
    spatial-mini      left or right, on a two-photo collage (grounding)
    caption-gen       write a caption, graded by BLEU/CIDEr (open-ended)

Three of them (`ocr-mini`, `count-mini`, `spatial-mini`) draw their own images.
That is deliberate: it is the only way to get exact ground truth for reading,
counting and left/right on a CPU budget, and it is honest as long as we say so.
Their absolute scores are not comparable to DocVQA or RefCOCO; what transfers is
the *protocol* -- a free-form answer, a parser, and a number you can defend.
"""

import json
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
sys.path.insert(0, str(PROJECTS / "40-multimodal-dpo"))
import dpo_lib as D  # noqa: E402
import vlm_lib as V  # noqa: E402

IMG = 336                 # every image the models see is IMG x IMG
N_PHOTOS = 200            # size of the photo bank we download
N_STATS = 3000            # corpus the "popular"/"co-occurring" tables count over
LETTERS = "ABCD"


def data_dir():
    d = HERE / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# the dataset: MS-COCO photos and their five human captions
# ---------------------------------------------------------------------------
def _get(url, tries=8, base=2.0):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(min(base * 2 ** attempt, 30.0))
    raise RuntimeError


class Bank:
    """The photo bank plus the caption corpus.

    Two different sizes are in play on purpose. We *ask questions about* 200
    photos, because every question costs a forward pass through a VLM. We
    *count statistics over* 3,000 captions, because "which objects are common"
    and "which objects keep company with which" are facts about the corpus, and
    estimating them from 200 photos would mostly measure sampling noise.
    """

    def __init__(self, n_photos=N_PHOTOS, n_stats=N_STATS, workers=12):
        rows = V._list_rows(n_stats, verbose=False)
        self.captions = [[c.strip() for c in r["txt"].split("\n") if c.strip()]
                         for r in rows]
        self.n_photos = n_photos
        cache = data_dir() / f"photos_{IMG}.npz"
        if cache.exists():
            z = np.load(cache)
            if len(z["images"]) >= n_photos:
                self.images = z["images"][:n_photos]
                return
        out = np.zeros((n_photos, IMG, IMG, 3), dtype=np.uint8)

        def grab(i):
            img = Image.open(BytesIO(_get(rows[i]["src"])))
            return i, np.asarray(V.square(img, IMG), dtype=np.uint8)

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            for i, arr in pool.map(grab, range(n_photos)):
                out[i] = arr
        print(f"    fetched {n_photos} photos ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(cache, images=out)
        self.images = out

    def objects(self, i):
        return sorted(D.objects_in(self.captions[i]))

    def stats(self):
        """Corpus-level frequency and co-occurrence tables, computed once."""
        if getattr(self, "_stats", None) is None:
            freq, co = Counter(), {}
            for caps in self.captions:
                present = sorted(D.objects_in(caps))
                freq.update(present)
                for a in present:
                    for b in present:
                        if a != b:
                            co.setdefault(a, Counter())[b] += 1
            self._stats = (freq, co)
        return self._stats


# ---------------------------------------------------------------------------
# drawing the three synthetic tasks
# ---------------------------------------------------------------------------
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


WORDS = ("MARKET RIVER PLANET CANDLE FOREST TICKET SILVER GARDEN WINTER BRIDGE "
         "COFFEE MIRROR PUZZLE ROCKET ANCHOR YELLOW ORANGE PEPPER VELVET COPPER "
         "TUNNEL BASKET LANTERN HARBOR MEADOW").split()
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 160, 60),
          "yellow": (235, 195, 40)}


def draw_ocr(photo, word, rng, size=None, plaque=True):
    """Paint a word on a white plaque over a real photo.

    The plaque is there so the default task tests *reading*, not contrast
    handling: black-on-white letters are legible for anything that can read at
    all, so a failure is a failure of OCR rather than of the lighting.
    `plaque=False` drops it and paints straight onto the photo, which is what
    turns the same task into a contrast problem.
    """
    img = Image.fromarray(photo).copy()
    d = ImageDraw.Draw(img)
    size = int(rng.integers(34, 46)) if size is None else int(size)
    font = _font(size)
    box = d.textbbox((0, 0), word, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = 14
    x = int(rng.integers(pad, max(pad + 1, IMG - w - pad)))
    y = int(rng.integers(pad, max(pad + 1, IMG - h - 3 * pad)))
    if plaque:
        d.rectangle([x - pad, y - pad, x + w + pad, y + h + 2 * pad],
                    fill=(255, 255, 255))
    d.text((x - box[0], y - box[1]), word, font=font, fill=(15, 15, 15))
    return np.asarray(img, dtype=np.uint8)


def draw_count(n, color, rng):
    """A plain canvas with n identical shapes on it.

    No photo underneath: counting on top of a busy street scene would confuse
    "can it count" with "can it segment", and only the first question is being
    asked here.
    """
    img = Image.new("RGB", (IMG, IMG), (245, 245, 240))
    d = ImageDraw.Draw(img)
    placed = []
    r = 34
    while len(placed) < n:
        cx = int(rng.integers(r + 8, IMG - r - 8))
        cy = int(rng.integers(r + 8, IMG - r - 8))
        if all((cx - px) ** 2 + (cy - py) ** 2 > (2.4 * r) ** 2 for px, py in placed):
            placed.append((cx, cy))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color],
                      outline=(30, 30, 30), width=3)
    return np.asarray(img, dtype=np.uint8)


def draw_pair(left_photo, right_photo):
    """Two photos side by side with a black divider -- a collage whose left/right
    ground truth is known by construction, which no real photo gives us."""
    img = Image.new("RGB", (IMG, IMG), (0, 0, 0))
    half = (IMG - 6) // 2
    for k, p in enumerate((left_photo, right_photo)):
        # Centre-crop each square photo to a tall half-width strip so the
        # collage fills the canvas: a shrunken thumbnail floating in black
        # would make the task partly a test of seeing small things.
        src = Image.fromarray(p)
        w = int(round(src.width * half / IMG))
        strip = src.crop(((src.width - w) // 2, 0,
                          (src.width - w) // 2 + w, src.height))
        img.paste(strip.resize((half, IMG), Image.BICUBIC), (k * (half + 6), 0))
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "no": 0}


def parse_yesno(text):
    for w in _norm(text):
        if w in ("yes", "yeah", "yep"):
            return "yes"
        if w in ("no", "nope"):
            return "no"
    return "?"


def parse_letter(text, n=4, strict=False):
    """Pull a multiple-choice letter out of whatever the model said.

    Models answer "B", "B.", "(B)", "Answer: B" and "b" all for the same thing,
    so a usable parser has to accept them. This function is exactly the kind of
    code that quietly moves published scores: `strict=True` accepts only a bare
    leading letter, which is what a first-draft harness usually does, and it
    marks every "Answer: B" wrong. The two settings are reported side by side in
    this project's results because the difference is enormous.
    """
    t = text.strip()
    m = re.match(r"^\W*([A-Da-d])\b", t)
    if m:
        i = LETTERS.index(m.group(1).upper())
        return LETTERS[i] if i < n else "?"
    if strict:
        return "?"
    m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?\s*([A-Da-d])\b",
                  t, re.IGNORECASE)
    if m:
        i = LETTERS.index(m.group(1).upper())
        return LETTERS[i] if i < n else "?"
    m = re.search(r"\b([A-D])\b", t)
    return m.group(1) if m else "?"


def parse_number(text):
    for w in _norm(text):
        if w.isdigit():
            return int(w)
        if w in NUMBER_WORDS:
            return NUMBER_WORDS[w]
    return None


def parse_side(text):
    t = " ".join(_norm(text))
    left, right = "left" in t, "right" in t
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "?"


def prf(records, preds, positive="yes"):
    tp = fp = fn = tn = 0
    for r, a in zip(records, preds):
        if r["target"] == positive:
            tp += a == positive
            fn += a != positive
        else:
            fp += a == positive
            tn += a != positive
    p = tp / max(tp + fp, 1)
    q = tp / max(tp + fn, 1)
    return {"precision": p, "recall": q,
            "f1": 0.0 if p + q == 0 else 2 * p * q / (p + q),
            "yes_rate": (tp + fp) / max(len(records), 1)}


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hyps, refs_list, n=4):
    """Corpus BLEU-n with the standard brevity penalty.

    BLEU = *BiLingual Evaluation Understudy*, from machine translation (Papineni
    et al., 2002): an "understudy" because it was pitched as a stand-in that
    could rehearse the human judge's part cheaply. It counts how many of the
    candidate's word n-grams appear in the references, clipped so repeating a
    word cannot inflate the count, and multiplies by a penalty for being shorter
    than the reference (nothing else stops a one-word answer from scoring 1.0 on
    precision).
    """
    clipped = [0] * n
    total = [0] * n
    hyp_len = ref_len = 0
    for hyp, refs in zip(hyps, refs_list):
        h = _norm(hyp)
        rs = [_norm(r) for r in refs]
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]
        for k in range(1, n + 1):
            hc = _ngrams(h, k)
            mx = Counter()
            for r in rs:
                for g, c in _ngrams(r, k).items():
                    mx[g] = max(mx[g], c)
            clipped[k - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            total[k - 1] += max(sum(hc.values()), 0)
    if min(total) == 0 or min(clipped) == 0:
        return 0.0
    logp = sum(np.log(clipped[k] / total[k]) for k in range(n)) / n
    bp = 1.0 if hyp_len > ref_len else np.exp(1 - ref_len / max(hyp_len, 1))
    return float(bp * np.exp(logp))


def cider(hyps, refs_list, n=4):
    """CIDEr-D-flavoured consensus score.

    CIDEr = *Consensus-based Image Description Evaluation* (Vedantam et al.,
    2015). The idea BLEU misses: not every matching word is equally impressive.
    Matching "a" is worthless because every caption contains it; matching
    "surfboard" is strong evidence. So each n-gram is weighted by TF-IDF -- term
    frequency times *inverse document frequency*, i.e. divided by how many
    images use it anywhere -- and the score is the cosine similarity between the
    candidate's weighted n-gram vector and the references'.
    """
    docs = [[_norm(r) for r in refs] for refs in refs_list]
    dfs = []
    for k in range(1, n + 1):
        df = Counter()
        for rs in docs:
            seen = set()
            for r in rs:
                seen |= set(_ngrams(r, k))
            df.update(seen)
        dfs.append(df)
    N = len(docs)

    def vec(tokens, k):
        c = _ngrams(tokens, k)
        out = {}
        for g, v in c.items():
            idf = np.log(max(N, 1) / max(dfs[k - 1].get(g, 0), 1.0))
            out[g] = v * idf
        return out

    def cos(a, b):
        num = sum(v * b.get(g, 0.0) for g, v in a.items())
        na = np.sqrt(sum(v * v for v in a.values()))
        nb = np.sqrt(sum(v * v for v in b.values()))
        return 0.0 if na == 0 or nb == 0 else num / (na * nb)

    scores = []
    for hyp, rs in zip(hyps, docs):
        h = _norm(hyp)
        per_n = []
        for k in range(1, n + 1):
            hv = vec(h, k)
            sims = []
            for r in rs:
                s = cos(hv, vec(r, k))
                # CIDEr-D's length gate: a candidate far from the reference
                # length is damped, which stops a caption that dumps every rare
                # word in the vocabulary from scoring well.
                s *= float(np.exp(-((len(h) - len(r)) ** 2) / (2 * 6.0 ** 2)))
                sims.append(s)
            per_n.append(np.mean(sims) if sims else 0.0)
        scores.append(10.0 * float(np.mean(per_n)))
    return float(np.mean(scores)), scores


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
MCQ_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


class Task:
    """One benchmark. `build` makes the questions, `grade` turns raw model
    strings into a metric dict. `kind` tells a model adapter what shape of
    answer is wanted, which is how a matcher like CLIP can declare that it
    cannot sit a generative task instead of silently scoring zero."""

    kind = "freeform"
    primary = "accuracy"

    def images(self, bank, docs):
        return np.stack([bank.images[d["image"]] for d in docs])


class Pope(Task):
    kind = "yesno"

    def __init__(self, negatives="random"):
        self.negatives = negatives
        self.name = f"pope-{negatives}"

    def build(self, bank, ids, rng):
        freq, co = bank.stats()
        popular = [g for g, _ in freq.most_common()]
        docs = []
        for i in ids:
            present = bank.objects(i)
            absent = [g for g in D.GROUPS if g not in present]
            if not present or not absent:
                continue
            yes_obj = present[int(rng.integers(len(present)))]
            if self.negatives == "random":
                neg = absent[int(rng.integers(len(absent)))]
            else:
                partners = Counter()
                for a in present:
                    for b, c in co.get(a, {}).items():
                        if b in absent:
                            partners[b] += c
                neg = (partners.most_common(1)[0][0] if partners
                       else next(g for g in popular if g in absent))
            for obj, ans in ((yes_obj, "yes"), (neg, "no")):
                docs.append({
                    "image": int(i), "target": ans, "object": obj,
                    "question": f"Is there {D.art(obj)} in the image? "
                                f"Answer yes or no.",
                })
        return docs

    def grade(self, docs, raw):
        preds = [parse_yesno(t) for t in raw]
        acc = float(np.mean([p == d["target"] for p, d in zip(preds, docs)]))
        out = {"n": len(docs), "accuracy": acc,
               "unparsed": sum(p == "?" for p in preds)}
        out.update(prf(docs, preds))
        return out, preds


class MCQ(Task):
    """A 4-way multiple-choice task with MMBench's *circular evaluation*.

    Circular evaluation asks the same question four times, rotating which slot
    the right answer sits in, and gives credit only if all four are right.
    MMBench introduced it because models have a position habit -- some reach for
    "A", some for the last option -- and a model that always answers "B" already
    gets 25% on a normal run. Rotating the options makes that strategy worth
    0%, because it cannot be right in all four rotations at once.
    """

    kind = "mcq"

    def __init__(self, name, circular=True):
        self.name = name
        self.circular = circular

    def _rotate(self, base_docs):
        docs = []
        for d in base_docs:
            rots = range(len(d["options"])) if self.circular else [0]
            for r in rots:
                opts = d["options"][r:] + d["options"][:r]
                gold = (d["options"].index(d["answer_text"]) - r) % len(opts)
                docs.append({**d, "options": opts, "rotation": r,
                             "target": LETTERS[gold],
                             "question": d["stem"] + "\n" + "\n".join(
                                 f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
                             + MCQ_SUFFIX})
        return docs

    def grade(self, docs, raw):
        preds = [parse_letter(t, 4) for t in raw]
        strict = [parse_letter(t, 4, strict=True) for t in raw]
        ok = [p == d["target"] for p, d in zip(preds, docs)]
        acc = float(np.mean(ok))
        groups = {}
        for d, o in zip(docs, ok):
            groups.setdefault(d["item"], []).append(o)
        circ = float(np.mean([all(v) for v in groups.values()]))
        chosen = Counter(p for p in preds if p != "?")
        out = {"n": len(docs), "accuracy": acc, "circular_accuracy": circ,
               "unparsed": sum(p == "?" for p in preds),
               "strict_accuracy": float(np.mean(
                   [p == d["target"] for p, d in zip(strict, docs)])),
               "strict_unparsed": sum(p == "?" for p in strict),
               "letter_bias": {k: chosen.get(k, 0) / max(len(preds), 1)
                               for k in LETTERS}}
        return out, preds


class ObjectMCQ(MCQ):
    def __init__(self, circular=True):
        super().__init__("mmbench-mini", circular)

    def build(self, bank, ids, rng):
        freq, _ = bank.stats()
        popular = [g for g, _ in freq.most_common(40)]
        base = []
        for n, i in enumerate(ids):
            present = bank.objects(i)
            absent = [g for g in popular if g not in present]
            if not present or len(absent) < 3:
                continue
            gold = present[int(rng.integers(len(present)))]
            distract = list(rng.choice(absent, 3, replace=False))
            opts = [gold] + [str(x) for x in distract]
            base.append({"image": int(i), "item": n, "options": opts,
                         "answer_text": gold,
                         "stem": "Which of these is in the image?"})
        return self._rotate(base)


WORD2GROUP = {w: g for g, syn in D.GROUPS.items() for w in syn.split()}
RELATIONS = {"on": "under", "under": "on", "above": "below", "below": "above",
             "behind": "in front of", "left": "right", "right": "left",
             "inside": "outside", "over": "beneath", "beneath": "over",
             "next": "far", "holding": "avoiding", "with": "without"}
COUNTS = {"one": "three", "two": "four", "three": "one", "four": "two",
          "a": "three", "an": "three", "several": "one", "many": "one"}


class CaptionMCQ(MCQ):
    """Pick the sentence that describes the picture, out of four -- where the
    three wrong sentences are **edits of the right one**.

    `hard=True` is not decoration; it is a repair. The first version of this
    task drew the three wrong captions at random from the corpus, and a frozen
    CLIP scored **1.000**. When the distractors describe unrelated scenes,
    "which caption fits" collapses into plain topic matching, and a task every
    model aces measures nothing.

    The hard version builds each distractor by changing **one thing** in the
    true caption: swapping two of its nouns ("a cat on a table" -> "a table on
    a cat"), replacing one noun with an object the picture does not contain, or
    flipping a relation or a number word. These are *minimal pairs*, the design
    behind ARO and SugarCrepe, and they target a known weakness: a
    contrastively-trained matcher largely treats a sentence as a bag of words,
    so the swapped version looks nearly identical to it while a reader can see
    the difference immediately.
    """

    def __init__(self, circular=True, hard=True, n_items=30):
        super().__init__("caption-match", circular)
        self.hard = hard
        self.n_items = n_items

    @staticmethod
    def _nouns(words):
        return [k for k, w in enumerate(words) if w.strip(".,").lower() in WORD2GROUP]

    def _edits(self, caption, absent, rng):
        """Three one-change rewrites of `caption`, or fewer if it is too plain."""
        words = caption.split()
        pos = self._nouns(words)
        out = []
        if len(pos) >= 2:                     # swap two nouns: same words, new meaning
            a, b = pos[0], pos[1]
            w = list(words)
            # Swap only the word itself and leave each slot's punctuation where
            # it was, or the sentence ends up with a full stop in the middle and
            # the distractor becomes detectable without looking at the picture.
            ca, ta = words[a].rstrip(".,;"), words[a][len(words[a].rstrip(".,;")):]
            cb, tb = words[b].rstrip(".,;"), words[b][len(words[b].rstrip(".,;")):]
            w[a], w[b] = cb + ta, ca + tb
            out.append(" ".join(w))
        if pos and absent:                    # replace a noun with an absent object
            w = list(words)
            k = pos[int(rng.integers(len(pos)))]
            core = w[k].rstrip(".,;")
            w[k] = absent[int(rng.integers(len(absent)))] + w[k][len(core):]
            out.append(" ".join(w))
        for table in (RELATIONS, COUNTS):     # flip a relation or a number word
            w = list(words)
            for k, x in enumerate(w):
                low = x.strip(".,").lower()
                if low in table:
                    w[k] = x.replace(x.strip(".,"), table[low])
                    out.append(" ".join(w))
                    break
        seen, uniq = {caption}, []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def build(self, bank, ids, rng):
        base = []
        pool = [int(x) for x in ids]
        for i in pool:
            if len(base) >= self.n_items:
                break
            gold = bank.captions[i][0]
            if not self.hard:
                others = [j for j in pool if j != i]
                picks = rng.choice(others, 3, replace=False)
                opts = [gold] + [bank.captions[int(j)][0] for j in picks]
            else:
                present = bank.objects(i)
                absent = [g for g in D.GROUPS if g not in present]
                edits = self._edits(gold, absent, rng)
                if len(edits) < 3:
                    continue
                opts = [gold] + edits[:3]
            base.append({"image": i, "item": len(base), "options": opts,
                         "answer_text": gold,
                         "stem": "Which caption describes the image?"})
        return self._rotate(base)


class Ocr(Task):
    kind = "freeform"

    name = "ocr-mini"

    def __init__(self, size=None, plaque=True, nonwords=False):
        # Three difficulty knobs, all off by default. `size` fixes the font
        # height; `plaque` removes the white background; `nonwords` replaces the
        # English word list with random letter strings. The last one is the
        # interesting one: an English word can be *guessed* from a blurry shape
        # plus a language prior, a random string cannot, so the gap between the
        # two is the part of the score that was reading rather than guessing.
        self.size = size
        self.plaque = plaque
        self.nonwords = nonwords

    @staticmethod
    def _nonword(rng, n=6):
        return "".join("BCDFGHJKLMNPRSTVWZ"[int(rng.integers(18))] if k % 2 == 0
                       else "AEIOU"[int(rng.integers(5))] for k in range(n))

    def build(self, bank, ids, rng):
        docs = []
        for i in ids:
            w = (self._nonword(rng) if self.nonwords
                 else WORDS[int(rng.integers(len(WORDS)))])
            docs.append({"image": int(i), "word": w, "target": w.lower(),
                         "seed": int(rng.integers(1 << 30)),
                         "question": "What is the word written in the image? "
                                     "Answer with the single word."})
        return docs

    def images(self, bank, docs):
        return np.stack([draw_ocr(bank.images[d["image"]], d["word"],
                                  np.random.default_rng(d["seed"]), self.size,
                                  self.plaque)
                         for d in docs])

    def grade(self, docs, raw):
        preds, ok = [], []
        for d, t in zip(docs, raw):
            words = _norm(t)
            preds.append(words[0] if words else "?")
            ok.append(d["target"] in words)
        return ({"n": len(docs), "accuracy": float(np.mean(ok)),
                 "unparsed": sum(p == "?" for p in preds)}, preds)


class Count(Task):
    kind = "freeform"
    name = "count-mini"

    def build(self, bank, ids, rng):
        docs = []
        for k, _ in enumerate(ids):
            n = int(rng.integers(1, 5))
            color = list(COLORS)[int(rng.integers(len(COLORS)))]
            docs.append({"image": -1, "count": n, "color": color,
                         "target": str(n), "seed": int(rng.integers(1 << 30)),
                         "question": f"How many {color} circles are in the "
                                     f"image? Answer with a number."})
        return docs

    def images(self, bank, docs):
        return np.stack([draw_count(d["count"], d["color"],
                                    np.random.default_rng(d["seed"]))
                         for d in docs])

    def grade(self, docs, raw):
        preds = [parse_number(t) for t in raw]
        ok = [p is not None and p == d["count"] for p, d in zip(preds, docs)]
        return ({"n": len(docs), "accuracy": float(np.mean(ok)),
                 "unparsed": sum(p is None for p in preds),
                 "mean_predicted": float(np.mean([p for p in preds
                                                  if p is not None] or [0]))},
                [str(p) for p in preds])


class Spatial(Task):
    kind = "freeform"
    name = "spatial-mini"

    def build(self, bank, ids, rng):
        docs = []
        ids = list(ids)
        for k, i in enumerate(ids):
            j = int(rng.choice([x for x in ids if x != i]))
            present_i, present_j = bank.objects(i), bank.objects(j)
            only_i = [g for g in present_i if g not in present_j]
            if not only_i:
                continue
            obj = only_i[int(rng.integers(len(only_i)))]
            side = "left" if rng.integers(2) == 0 else "right"
            left, right = (i, j) if side == "left" else (j, i)
            docs.append({"image": int(left), "other": int(right),
                         "object": obj, "target": side,
                         "question": f"Is the {obj} on the left or the right "
                                     f"side of the image? Answer left or right."})
        return docs

    def images(self, bank, docs):
        return np.stack([draw_pair(bank.images[d["image"]],
                                   bank.images[d["other"]]) for d in docs])

    def grade(self, docs, raw):
        preds = [parse_side(t) for t in raw]
        ok = [p == d["target"] for p, d in zip(preds, docs)]
        left_rate = float(np.mean([p == "left" for p in preds]))
        return ({"n": len(docs), "accuracy": float(np.mean(ok)),
                 "unparsed": sum(p == "?" for p in preds),
                 "left_rate": left_rate}, preds)


class CaptionGen(Task):
    kind = "caption"
    name = "caption-gen"
    primary = "cider"

    def build(self, bank, ids, rng):
        return [{"image": int(i), "target": bank.captions[i][0],
                 "refs": bank.captions[i],
                 "question": "Describe the image in one short sentence."}
                for i in ids]

    def grade(self, docs, raw):
        refs = [d["refs"] for d in docs]
        c, per = cider(raw, refs)
        return ({"n": len(docs), "bleu4": bleu(raw, refs), "cider": c,
                 "mean_words": float(np.mean([len(_norm(t)) for t in raw])),
                 "unparsed": sum(len(_norm(t)) == 0 for t in raw)}, raw)


def suite(circular=True, hard_captions=True):
    return [Pope("random"), Pope("adversarial"), ObjectMCQ(circular),
            CaptionMCQ(circular, hard_captions), Ocr(), Count(), Spatial(),
            CaptionGen()]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class Chance:
    """Answer without looking at anything -- the floor every task needs.

    Not a strawman: it is the only row that tells you what a score of 0.25
    means on a 4-way question, and the only way to notice that a "hard" task is
    one where the model is guessing.
    """

    name = "chance"
    can = staticmethod(lambda kind: kind != "caption")

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, task, docs, images):
        out = []
        for d in docs:
            if task.kind == "yesno":
                out.append("yes" if self.rng.integers(2) else "no")
            elif task.kind == "mcq":
                out.append(LETTERS[int(self.rng.integers(len(d["options"])))])
            elif task.name == "count-mini":
                out.append(str(int(self.rng.integers(1, 5))))
            elif task.name == "spatial-mini":
                out.append("left" if self.rng.integers(2) else "right")
            else:
                out.append(WORDS[int(self.rng.integers(len(WORDS)))].lower())
        return out


class ClipMatch:
    """Frozen CLIP used as an answerer: score the image against a phrase for
    each option and take the best.

    It is here to measure how much of a "vision-language understanding"
    benchmark is really an image-text *matching* problem. Whatever CLIP gets is
    the part that needed no language modelling at all; a VLM's margin above it
    is what its language half is buying. It cannot write, so it declines the
    free-form tasks rather than scoring a fake zero.
    """

    name = "clip-zeroshot"
    can = staticmethod(lambda kind: kind in ("yesno", "mcq"))

    def __init__(self, threshold=0.25):
        sys.path.insert(0, str(PROJECTS / "37-mini-laion-pipeline"))
        import pipeline_lib as P
        self.P = P
        self.model, self.tok = P.load_clip()
        self.threshold = threshold

    @torch.no_grad()
    def _score(self, images, texts):
        s, _ = self.P.clip_scores(images, texts, self.model, self.tok,
                                  verbose=False)
        return np.asarray(s)

    def calibrate(self, docs, images):
        s = self._score(images, [f"a photo of {D.art(d['object'])}" for d in docs])
        truth = np.array([d["target"] == "yes" for d in docs])
        best, cut = -1.0, 0.25
        for t in np.quantile(s, np.linspace(0.02, 0.98, 97)):
            acc = float(((s >= t) == truth).mean())
            if acc > best:
                best, cut = acc, float(t)
        self.threshold = cut
        return {"threshold": cut, "dev_accuracy": best}

    def predict(self, task, docs, images):
        if task.kind == "yesno":
            s = self._score(images, [f"a photo of {D.art(d['object'])}"
                                     for d in docs])
            return ["yes" if v >= self.threshold else "no" for v in s]
        # mcq: score all four options against the same picture, take the best.
        flat_img, flat_txt, spans = [], [], []
        for k, d in enumerate(docs):
            spans.append((len(flat_txt), len(d["options"])))
            for o in d["options"]:
                flat_img.append(images[k])
                flat_txt.append(o if " " in o else f"a photo of {D.art(o)}")
        s = self._score(np.stack(flat_img), flat_txt)
        return [LETTERS[int(np.argmax(s[a:a + n]))] for a, n in spans]


class BlindLLM:
    """The text half of a VLM, asked the question with **no image at all**.

    Why bother, when we already have `chance`: chance measures what luck gets
    you, while this measures what *language* gets you. A question like "which
    caption describes the image?" can often be answered without looking,
    because three of the four sentences are about a snowy mountain and one is
    about a kitchen, and only one of them fits the sort of picture the dataset
    contains. Anything this row scores above chance is benchmark leakage
    through the text, and it is the single most useful diagnostic a harness can
    print.
    """

    name = "blind-llm"
    can = staticmethod(lambda kind: True)

    def __init__(self, repo=V.LLM_NAME):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(6)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float32).eval()

    def _prompt(self, q):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True)

    @torch.no_grad()
    def predict(self, task, docs, images, bs=16):
        out = []
        for k in range(0, len(docs), bs):
            chunk = docs[k:k + bs]
            enc = self.tok([self._prompt(d["question"]) for d in chunk],
                           return_tensors="pt", padding=True)
            gen = self.llm.generate(**enc, max_new_tokens=24 if task.kind ==
                                    "caption" else 6, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        return out


class SmolVLMModel:
    """A real open VLM small enough for a CPU: HuggingFaceTB/SmolVLM.

    `do_image_splitting=False` gives it one view of the picture instead of four
    tiles plus a thumbnail: 88 prompt tokens instead of 1,148, which project 41
    measured at 14x faster. It is a real accuracy cost on small objects and it
    is applied identically to every model, so the comparison between rows is
    unaffected -- but the absolute numbers are lower than a paper's.
    """

    can = staticmethod(lambda kind: True)

    def __init__(self, repo="HuggingFaceTB/SmolVLM-256M-Instruct",
                 split_images=False, threads=6, name=None, mode="generate"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch.set_num_threads(threads)
        self.name = name or repo.split("/")[-1].replace("-Instruct", "").lower()
        self.proc = AutoProcessor.from_pretrained(repo)
        self.proc.image_processor.do_image_splitting = split_images
        self.proc.tokenizer.padding_side = "left"
        self.mode = mode
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo, dtype=torch.float32).eval()

    def _ids(self, words):
        out = []
        for w in words:
            ids = self.proc.tokenizer.encode(w, add_special_tokens=False)
            if ids:
                out.append(ids[0])
        return sorted(set(out))

    def _prompt(self, q):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": q}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True)

    @torch.no_grad()
    def predict(self, task, docs, images, bs=8, verbose=True):
        max_new = {"yesno": 4, "mcq": 4, "caption": 24}.get(task.kind, 8)
        out, t0 = [], time.time()
        for k in range(0, len(docs), bs):
            chunk = docs[k:k + bs]
            pil = [Image.fromarray(images[k + j]) for j in range(len(chunk))]
            inp = self.proc(text=[self._prompt(d["question"]) for d in chunk],
                            images=[[p] for p in pil], padding=True,
                            return_tensors="pt")
            if self.mode == "likelihood" and task.kind in ("yesno", "mcq"):
                # Never let the model speak: read off which of the allowed
                # answer tokens it finds most likely at the very next position.
                # A harness prefers this because it always parses -- there is no
                # "I'm not sure, but..." to argue with -- but it is a different
                # protocol from the one a user experiences, so the two can and
                # do disagree.
                logits = self.model(**inp).logits[:, -1]
                words = (("yes", "no") if task.kind == "yesno"
                         else tuple(LETTERS))
                cols = [self._ids([w.capitalize(), " " + w.capitalize(), w])
                        for w in words]
                best = [words[int(np.argmax([float(logits[r, c].max())
                                             for c in cols]))]
                        for r in range(len(chunk))]
                out += [w.upper() if task.kind == "mcq" else w for w in best]
            else:
                gen = self.model.generate(**inp, max_new_tokens=max_new,
                                          do_sample=False)
                out += self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
            if verbose and (k // bs) % 8 == 0:
                print(f"      {min(k + bs, len(docs))}/{len(docs)}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
        return out


class GenericCaption:
    """Always emits the same sentence, ignoring the picture entirely.

    Its only job is to give the captioning metrics something they *should*
    score near zero, and to find out whether they do.
    """

    name = "generic-caption"
    can = staticmethod(lambda kind: kind == "caption")

    def __init__(self, text="a man is standing next to a table in a room"):
        self.text = text

    def predict(self, task, docs, images):
        return [self.text] * len(docs)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
# How many *images* each task draws on. The two POPE tasks and the two
# multiple-choice tasks turn one image into several questions (two questions for
# a POPE pair, four for a circular-evaluation rotation), so the budgets below are
# chosen to land every task at roughly 100-120 forward passes.
ITEMS = {"pope-random": 50, "pope-adversarial": 50, "mmbench-mini": 30,
         "caption-match": 90, "ocr-mini": 50, "count-mini": 50,
         "spatial-mini": 55, "caption-gen": 50}
# `caption-match` is given a larger pool because it *skips* captions too plain
# to edit three different ways; it stops at the first 30 it can build.


def build_all(bank, items=None, seed=0, circular=True, offset=0,
              hard_captions=True):
    """Build every task's questions from one seed, so a rerun is identical.

    `offset` shifts which photos are used, which is how project 43 asks "how
    much of a reproduction gap is just a different sample of images?".
    """
    items = dict(ITEMS if items is None else items)
    tasks = {}
    for t in suite(circular, hard_captions):
        n = items.get(t.name, 50)
        ids = np.arange(offset, min(offset + n, bank.n_photos))
        # A per-task seed derived from the task's name, fixed across Python runs
        # (`hash` of a string is randomised per process unless PYTHONHASHSEED is
        # set, so it must not be used here).
        rng = np.random.default_rng(seed + sum(ord(c) for c in t.name))
        tasks[t.name] = (t, t.build(bank, ids, rng))
    return tasks


def run(model, tasks, bank, only=None, verbose=True):
    rows = {}
    for name, (task, docs) in tasks.items():
        if only and name not in only:
            continue
        if not model.can(task.kind):
            rows[name] = {"skipped": "model cannot produce this answer type"}
            continue
        t0 = time.time()
        images = task.images(bank, docs)
        raw = model.predict(task, docs, images)
        metrics, preds = task.grade(docs, raw)
        metrics["seconds"] = round(time.time() - t0, 1)
        rows[name] = metrics
        if verbose:
            head = metrics.get(task.primary, metrics.get("accuracy"))
            print(f"    {model.name:>18s} {name:<18s} "
                  f"{task.primary}={head:.3f}  ({metrics['seconds']:.0f}s)",
                  flush=True)
        rows[name]["_raw"] = raw[:400]
        rows[name]["_pred"] = preds[:400]
    return rows


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
