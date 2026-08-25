"""A POPE-style hallucination benchmark, plus adapters so different kinds of
model can sit the same exam.

POPE = *Polling-based Object Probing Evaluation* (Li et al., 2023). The idea is
deliberately boring: instead of asking a model to describe a picture and then
arguing about whether the description is right, you *poll* it with yes/no
questions about single objects, half of which are present and half of which are
not. Grading becomes exact, and the balance is what makes the score meaningful.

Why balance matters so much here
--------------------------------
Vision-language models are heavily biased towards "yes". A benchmark made only
of questions about objects that *are* in the picture gives a model that answers
"yes" to everything a perfect score. Pairing every yes-question with a
no-question about the same image caps that strategy at exactly 50%, so anything
above 50% is evidence the model looked.

The three ways to choose the absent object
------------------------------------------
    random        pick any object the picture does not contain
    popular       pick one of the objects that are most common in the dataset
                  overall -- a model that has memorised "COCO photos usually
                  contain people" will say yes
    adversarial   pick the object that most often appears *together with* the
                  objects that really are in this picture -- a model that has
                  learned "surfboards come with water" will say yes

The three splits share the same images and the same yes-questions. Only the
no-question changes, so the difference between the three scores is caused by
the negative-sampling rule and nothing else.
"""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
sys.path.insert(0, str(PROJECTS / "40-multimodal-dpo"))
import dpo_lib as D  # noqa: E402
import vlm_lib as V  # noqa: E402

SPLITS = ["random", "popular", "adversarial"]
PROMPTS = {
    "plain": "Is there {a} in the image?",
    "yesno": "Is there {a} in the image? Answer yes or no.",
    "strict": "Answer the question using a single word, yes or no. "
              "Is there {a} in the image?",
}
FULL_SIZE = 336


def data_dir():
    d = HERE / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# building the questions
# ---------------------------------------------------------------------------
def build(data, ids, seed=0, prompt="yesno", stats_ids=None):
    """-> dict split -> list of question records.

    Every record is {image, object, answer, question}. The `answer` is the
    ground truth read off the five human captions, which is the same rule
    project 40's hallucination meter uses -- one definition of "in the picture"
    for the whole phase.

    `stats_ids` is the pool the "popular" and "adversarial" statistics are
    counted over, and it defaults to the *whole* dataset rather than the
    hundred-odd images we ask about. That matters: "which objects are popular"
    and "which objects co-occur" are facts about the corpus a model was trained
    on, and estimating them from a hundred images would mostly measure sampling
    noise.
    """
    rng = np.random.default_rng(seed)
    pool = range(len(data.captions)) if stats_ids is None else stats_ids
    # How often each object shows up across the corpus -- the "popular" list.
    freq = {}
    for i in pool:
        for g in D.objects_in(data.captions[int(i)]):
            freq[g] = freq.get(g, 0) + 1
    popular = [g for g, _ in sorted(freq.items(), key=lambda kv: -kv[1])]
    # Co-occurrence counts, for the adversarial split.
    co = {}
    for i in pool:
        present = sorted(D.objects_in(data.captions[int(i)]))
        for a in present:
            for b in present:
                if a != b:
                    co.setdefault(a, {})[b] = co.setdefault(a, {}).get(b, 0) + 1

    out = {s: [] for s in SPLITS}
    for i in ids:
        present = sorted(D.objects_in(data.captions[int(i)]))
        absent = [g for g in D.GROUPS if g not in present]
        if not present or not absent:
            continue
        yes_obj = present[int(rng.integers(len(present)))]
        neg = {}
        neg["random"] = absent[int(rng.integers(len(absent)))]
        neg["popular"] = next((g for g in popular if g in absent), neg["random"])
        partners = {}
        for a in present:
            for b, c in co.get(a, {}).items():
                if b in absent:
                    partners[b] = partners.get(b, 0) + c
        neg["adversarial"] = (max(partners, key=partners.get) if partners
                              else neg["popular"])
        for s in SPLITS:
            for obj, ans in ((yes_obj, "yes"), (neg[s], "no")):
                out[s].append({
                    "image": int(i), "object": obj, "answer": ans,
                    "question": PROMPTS[prompt].format(a=D.art(obj)),
                })
    return out


def score(records, answers):
    """Grade one model's answers. "yes" is the positive class.

    precision = of the times it said yes, how often was the object really there
    recall    = of the objects really there, how often did it say yes
    F1        = the harmonic mean of the two, which punishes a model that wins
                one of them by giving up the other (always saying "yes" gets
                recall 1.0 and F1 only ~0.67)
    """
    tp = fp = fn = tn = unparsed = 0
    for r, a in zip(records, answers):
        if a not in ("yes", "no"):
            unparsed += 1
            a = "no" if r["answer"] == "yes" else "yes"   # count it as wrong
        if r["answer"] == "yes":
            tp += a == "yes"
            fn += a == "no"
        else:
            fp += a == "yes"
            tn += a == "no"
    n = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": n, "accuracy": (tp + tn) / max(n, 1),
        "precision": prec, "recall": rec,
        "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
        "yes_rate": (tp + fp) / max(n, 1),
        "unparsed": unparsed,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# full-resolution images (the cached CLIP features are not enough for a real VLM)
# ---------------------------------------------------------------------------
def fetch_images(ids, size=FULL_SIZE, workers=12):
    """Download the COCO photos for `ids` at `size` px, aligned to `ids`.

    `ids` normally repeats every image (one yes-question and one no-question per
    photo), so the download is done over the *unique* ids and the result is then
    expanded back to the requested order. The URL list is already on disk from
    project 20's cache, so this costs no listing calls.

    Why full-resolution photos at all, when project 20 already cached CLIP
    features for these images: a real open VLM brings its own vision tower and
    its own preprocessing. Feeding it someone else's frozen features would be
    testing our pipeline, not the model.
    """
    cache = data_dir() / f"images_{size}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache)
        have = {int(v): z["images"][k] for k, v in enumerate(z["ids"])}
    todo = sorted({int(i) for i in ids} - set(have))
    if todo:
        rows = json.loads((V.data_dir() / "rows.json").read_text())

        def grab(i):
            img = Image.open(BytesIO(_get(rows[int(i)]["src"]))).convert("RGB")
            return int(i), np.asarray(img.resize((size, size), Image.BICUBIC),
                                      dtype=np.uint8)

        with ThreadPoolExecutor(workers) as pool:
            for i, arr in pool.map(grab, todo):
                have[i] = arr
        keys = sorted(have)
        np.savez_compressed(cache, images=np.stack([have[k] for k in keys]),
                            ids=np.asarray(keys, dtype=np.int64))
    return np.stack([have[int(i)] for i in ids])


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


# ---------------------------------------------------------------------------
# the systems under test
# ---------------------------------------------------------------------------
def parse_yesno(text):
    t = text.strip().lower().lstrip("*: ").replace("**", "")
    for word in t.replace(".", " ").replace(",", " ").split():
        if word in ("yes", "yeah", "yep"):
            return "yes"
        if word in ("no", "nope"):
            return "no"
    return "?"


class AlwaysYes:
    """The degenerate baseline every balanced benchmark needs. If a model cannot
    beat this, its score contains no evidence that it looked at the image."""

    name = "always-yes"

    def answer(self, records, images):
        return ["yes"] * len(records)


class ClipZeroShot:
    """Frozen CLIP as a yes/no answerer: is the picture closer to "a photo of a
    dog" than a calibrated threshold? No text is generated at all -- this is a
    *matcher*, not a talker, and it is here to show how much of the benchmark
    can be solved without any language modelling."""

    name = "clip-zeroshot"

    def __init__(self, threshold=None):
        sys.path.insert(0, str(PROJECTS / "37-mini-laion-pipeline"))
        import pipeline_lib as P
        self.P = P
        self.model, self.tok = P.load_clip()
        self.threshold = threshold

    @torch.no_grad()
    def _scores(self, records, images):
        texts = [f"a photo of {D.art(r['object'])}" for r in records]
        out, _ = self.P.clip_scores(images, texts, self.model, self.tok,
                                    verbose=False)
        return out

    def calibrate(self, records, images):
        """Pick the threshold that maximises accuracy on a *development* set.
        Tuning it on the test set would be cheating -- the number it produced
        would be the best of many tries, not a prediction about new data."""
        s = self._scores(records, images)
        truth = np.array([r["answer"] == "yes" for r in records])
        best, cut = -1, 0.25
        for t in np.quantile(s, np.linspace(0.02, 0.98, 97)):
            acc = float((((s >= t) == truth)).mean())
            if acc > best:
                best, cut = acc, float(t)
        self.threshold = cut
        return {"threshold": cut, "dev_accuracy": best}

    def answer(self, records, images):
        s = self._scores(records, images)
        return ["yes" if v >= self.threshold else "no" for v in s]


class SmolVLM:
    """A real open VLM, small enough to run here: HuggingFaceTB/SmolVLM.

    `mode="generate"` asks it the question and reads its words -- what a user
    would see. `mode="likelihood"` never lets it speak: it compares the
    probability the model assigns to the single token " Yes" against " No" and
    takes the larger. Evaluation harnesses use the second because it always
    parses; the two do not have to agree, and measuring the gap is one of the
    points of this project.
    """

    def __init__(self, repo="HuggingFaceTB/SmolVLM-256M-Instruct",
                 mode="generate", split_images=False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.name = repo.split("/")[-1].replace("-Instruct", "").lower() + "-" + mode
        self.proc = AutoProcessor.from_pretrained(repo)
        # One 512x512 view instead of four tiles plus a thumbnail: 88 prompt
        # tokens instead of 1,148, which is 14x faster on a CPU. Project 22
        # measured what tiling buys and what it costs -- here we buy the speed.
        self.proc.image_processor.do_image_splitting = split_images
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo, dtype=torch.float32).eval()
        # Questions about "a cat" and "a motorcycle" tokenize to different
        # lengths, so a batch has to be padded -- on the LEFT. With right
        # padding the last position of a short row is a pad token, which breaks
        # both generation (the model continues from padding) and the likelihood
        # read-out (we look at the final position on purpose).
        self.proc.tokenizer.padding_side = "left"
        self.mode = mode
        # "Yes" and " Yes" are different tokens in a byte-level BPE vocabulary,
        # and which one the model reaches for depends on whether the chat
        # template already left a space. Score both spellings and keep the
        # better -- otherwise a likelihood test can measure tokenizer trivia.
        self.yes_ids = self._ids(["Yes", " Yes", "yes"])
        self.no_ids = self._ids(["No", " No", "no"])

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
    def answer(self, records, images, bs=8, verbose=True):
        out, t0 = [], time.time()
        for k in range(0, len(records), bs):
            chunk = records[k:k + bs]
            pil = [Image.fromarray(images[k + j]) for j in range(len(chunk))]
            inp = self.proc(text=[self._prompt(r["question"]) for r in chunk],
                            images=[[p] for p in pil], padding=True,
                            return_tensors="pt")
            if self.mode == "generate":
                gen = self.model.generate(**inp, max_new_tokens=4, do_sample=False)
                txt = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                out += [parse_yesno(t) for t in txt]
            else:
                logits = self.model(**inp).logits[:, -1]
                y = logits[:, self.yes_ids].max(-1).values
                n = logits[:, self.no_ids].max(-1).values
                out += ["yes" if a > b else "no" for a, b in zip(y, n)]
            if verbose and (k // bs) % 5 == 0:
                print(f"    {min(k + bs, len(records))}/{len(records)}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
        return out


class TinyVLM:
    """Project 40's model -- our own frozen-CLIP + SmolLM2 captioner, scored by
    likelihood because it was never taught to answer questions. Comparing the
    checkpoint before and after DPO on this benchmark is how we find out whether
    preference training moved anything a *different* test can see."""

    def __init__(self, checkpoint, name, unfreeze=4):
        self.vlm, self.tok = D.build_vlm(unfreeze_last=unfreeze)
        D.load_into(self.vlm, checkpoint)
        self.name = name
        self.yes_ids = sorted({self.tok.encode(w, add_special_tokens=False)[0]
                               for w in ("Yes", "yes")})
        self.no_ids = sorted({self.tok.encode(w, add_special_tokens=False)[0]
                              for w in ("No", "no")})

    @torch.no_grad()
    def answer(self, records, images, data=None, bs=8, verbose=True):
        out = [None] * len(records)
        qs = [r["question"] for r in records]
        ids = np.array([r["image"] for r in records])
        for batch, part in V.prompt_batches(self.tok, qs, n_img=V.CLIP_TOKENS,
                                            bs=bs):
            feats = data.image_tokens(ids[part])
            emb = self.vlm.embed(batch, feats)
            h = self.vlm.body(inputs_embeds=emb, attention_mask=batch.attn,
                              use_cache=False)[0][:, -1]
            logits = self.vlm.head(h).float()
            for j, k in enumerate(part):
                out[k] = ("yes" if logits[j, self.yes_ids].max()
                          > logits[j, self.no_ids].max() else "no")
        return out
