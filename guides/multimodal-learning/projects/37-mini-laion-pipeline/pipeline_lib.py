"""Shared Phase-8 data plumbing: a miniature web crawl and the filters that clean it.

Projects 37 (this pipeline) and 38 (the caption ablation) both need the *same*
pile of dirty image-text pairs, so the pile is built here once and project 38
imports this file.

Why we build a crawl instead of downloading LAION
-------------------------------------------------
LAION-2B is 2 billion (image URL, alt-text) rows; a meaningful sample is
hundreds of gigabytes and many of the URLs are dead. So we do the next best
thing: take real photographs with real human captions (MS-COCO), then *inject*
the exact failure modes that make a real crawl unusable -- duplicates, alt-text
that belongs to a different picture, boilerplate like ``IMG_2043.JPG``, images
too small to be worth training on, and banner-shaped strips.

The injection is not a shortcut, it is the point: because we broke the data
ourselves, we know the correct answer for every row. That turns "our filter
looks reasonable" into a precision and recall number per filter, which is
something you can never compute on the real thing.

What lives here
    fetch_base / load_base   real COCO images + their five human captions
    build_crawl              the dirty corpus, with a defect label per record
    dhash / hamming          a 64-bit perceptual hash and its distance
    dedup                    banded (LSH) near-duplicate removal
    size_filter              resolution and aspect-ratio rules
    text_filter              alt-text quality + a keyword blocklist
    clip_scores              real frozen CLIP ViT-B/32 image-text cosines

Cache layout, all under ``37-mini-laion-pipeline/data/`` (git-ignored):

    rows.json        the COCO listing (image URL + captions per row)
    base.npz         2,400 photos at 128x128 + captions + native pixel sizes
    crawl.npz        the dirty corpus: images, alt-text, defect labels
    recaptions.json  BLIP's rewritten captions (written by run.py)
"""

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
STORE = 128            # every crawled image is cached as a 128x128 square
N_BASE = 2400          # clean (image, 5 captions) rows downloaded
CLIP_NAME = "openai/clip-vit-base-patch32"

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=clip-benchmark%2Fwds_mscoco_captions"
    "&config=default&split=test&offset={offset}&length={length}"
)

# The eight kinds of record in our crawl. "ok" is a genuine pair; everything
# else is a defect we injected on purpose and therefore know the truth about.
DEFECTS = ["ok", "dup_exact", "dup_near", "mismatch", "boilerplate",
           "tiny", "banner", "blocked"]

# Real web alt-text is often the camera's filename or a template the site
# stamps on every image. These are copied from the shapes that actually show up
# in web crawls.
BOILERPLATE = [
    "IMG_{n:04d}.JPG", "DSC{n:05d}.jpg", "photo", "image", "untitled",
    "stock photo", "click here to enlarge", "download this image",
    "no description available", "thumbnail", "default.jpg", "{n} x {n} pixels",
    "see more photos", "picture {n}", "-", "image not available",
]

# Stand-ins for the words a real safety blocklist matches. We use invented
# tokens on purpose: the mechanism (a keyword match that runs before anything
# expensive) is what the project teaches, and a real pipeline does NOT rely on
# keywords anyway -- it runs a trained NSFW image classifier plus hash matching
# against known-illegal material, which is mandatory and not optional.
BLOCKLIST = {"nsfwplaceholder", "adultonlyplaceholder", "blockedtermplaceholder"}

_WORD = re.compile(r"[a-zA-Z]+")


def data_dir():
    d = Path(__file__).resolve().parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# downloading the clean base corpus
# ---------------------------------------------------------------------------
def _get(url, tries=10, base=3.0):
    """GET with exponential backoff. Both the listing API and the image CDN
    answer HTTP 429 ('slow down') if you ask too fast."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(min(base * 2 ** attempt, 60.0))
    raise RuntimeError


def _list_rows(n, verbose=True):
    """Page through the dataset listing, caching every page immediately so an
    interrupted download resumes instead of restarting."""
    cache = data_dir() / "rows.json"
    rows = json.loads(cache.read_text()) if cache.exists() else []
    while len(rows) < n:
        batch = min(100, n - len(rows))
        payload = _get(_ROWS_URL.format(offset=len(rows), length=batch))
        rows += [{"src": r["row"]["jpg"]["src"], "txt": r["row"]["txt"]}
                 for r in json.loads(payload)["rows"]]
        cache.write_text(json.dumps(rows))
        if verbose and len(rows) % 500 == 0:
            print(f"    listed {len(rows)}/{n} rows", flush=True)
        time.sleep(1.0)
    return rows[:n]


def _square(img, size=STORE):
    """Shortest side -> size, then centre-crop."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def fetch_base(n=N_BASE, workers=12, verbose=True):
    """Download n real (photo, five human captions) rows. Idempotent."""
    out = data_dir() / "base.npz"
    if out.exists():
        return out
    rows = _list_rows(n, verbose)

    def grab(i_row):
        i, row = i_row
        img = Image.open(BytesIO(_get(row["src"])))
        w, h = img.size                       # the *native* size, before squaring
        arr = np.asarray(_square(img), dtype=np.uint8)
        caps = [c.strip() for c in row["txt"].split("\n") if c.strip()]
        return i, arr, caps, w, h

    imgs = np.zeros((len(rows), STORE, STORE, 3), dtype=np.uint8)
    caps, wh = [None] * len(rows), np.zeros((len(rows), 2), dtype=np.int32)
    with ThreadPoolExecutor(workers) as pool:
        for done, (i, arr, c, w, h) in enumerate(pool.map(grab, enumerate(rows)), 1):
            imgs[i], caps[i], wh[i] = arr, c, (w, h)
            if verbose and done % 400 == 0:
                print(f"    fetched {done}/{len(rows)} images", flush=True)

    width = max(len(c) for c in caps)
    caps = [c + [""] * (width - len(c)) for c in caps]
    np.savez_compressed(out, images=imgs, captions=np.array(caps, dtype=object), wh=wh)
    return out


def load_base(n=N_BASE):
    """-> images uint8 (N,128,128,3), captions list[list[str]], native (N,2) sizes."""
    z = np.load(fetch_base(n), allow_pickle=True)
    caps = [[s for s in row if s] for row in z["captions"]]
    return z["images"][:n], caps[:n], z["wh"][:n]


# ---------------------------------------------------------------------------
# building the dirty crawl
# ---------------------------------------------------------------------------
def _jpeg_roundtrip(arr, quality=35):
    """Re-encode through JPEG. A near-duplicate on the web is usually the same
    picture re-saved by another site, which is exactly this."""
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _shift(arr, dx=3, dy=2):
    """Crop a few pixels off one side and resize back -- a re-crop, the other
    common way the same photo reappears with different bytes."""
    img = Image.fromarray(arr).crop((dx, dy, arr.shape[1], arr.shape[0]))
    return np.asarray(img.resize((arr.shape[1], arr.shape[0]), Image.BICUBIC),
                      dtype=np.uint8)


def _downscale(arr, small=24):
    """Throw the detail away, then blow it back up: a thumbnail passed off as a
    photo. The stored array stays 128x128 so every record has the same shape;
    the *declared* size in the record is what the resolution filter reads."""
    img = Image.fromarray(arr).resize((small, small), Image.BICUBIC)
    return np.asarray(img.resize((arr.shape[1], arr.shape[0]), Image.NEAREST),
                      dtype=np.uint8)


def _banner(arr):
    """Squash to a wide strip and letterbox it back into the square."""
    h = max(4, arr.shape[0] // 8)
    img = Image.fromarray(arr).resize((arr.shape[1], h), Image.BICUBIC)
    out = np.zeros_like(arr)
    top = (arr.shape[0] - h) // 2
    out[top:top + h] = np.asarray(img, dtype=np.uint8)
    return out


def build_crawl(n_base=N_BASE, seed=0, verbose=True):
    """Turn the clean base corpus into a dirty crawl with known defect labels.

    Returns a dict with parallel arrays:
        images (M,128,128,3) uint8, alt list[str], defect list[str],
        w (M,), h (M,), src (M,)  -- src is the base row the record came from.
    """
    cache = data_dir() / "crawl.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        # The npz stores the five human captions as a rectangular array padded
        # with empty strings; strip them, or a caller that samples "one of the
        # captions" will sometimes sample nothing.
        return {"images": z["images"], "alt": list(z["alt"]),
                "defect": list(z["defect"]), "w": z["w"], "h": z["h"],
                "src": z["src"],
                "human": [[s for s in r if s] for r in z["human"]]}

    imgs, caps, wh = load_base(n_base)
    rng = np.random.default_rng(seed)
    n = len(imgs)

    # Which defect each base row gets. The shares are chosen to sit in the same
    # ballpark as published audits of web alt-text.
    plan = np.array(["ok"] * n, dtype=object)
    order = rng.permutation(n)
    cuts = [("mismatch", 0.12), ("boilerplate", 0.08), ("tiny", 0.04),
            ("banner", 0.04), ("blocked", 0.02)]
    at = 0
    for name, frac in cuts:
        k = int(round(frac * n))
        plan[order[at:at + k]] = name
        at += k

    out_img, out_alt, out_def, out_w, out_h, out_src, out_hum = [], [], [], [], [], [], []
    for i in range(n):
        arr, alt, w, h = imgs[i], caps[i][0], int(wh[i, 0]), int(wh[i, 1])
        kind = plan[i]
        if kind == "mismatch":
            j = int(rng.integers(n))
            while j == i:
                j = int(rng.integers(n))
            alt = caps[j][0]                       # fluent text, wrong picture
        elif kind == "boilerplate":
            t = BOILERPLATE[int(rng.integers(len(BOILERPLATE)))]
            alt = t.format(n=int(rng.integers(10, 9999)))
        elif kind == "tiny":
            arr, w, h = _downscale(arr), 24, 24
        elif kind == "banner":
            arr, w, h = _banner(arr), 960, 120
        elif kind == "blocked":
            word = sorted(BLOCKLIST)[int(rng.integers(len(BLOCKLIST)))]
            alt = f"{alt} {word}"
        out_img.append(arr)
        out_alt.append(alt)
        out_def.append(str(kind))
        out_w.append(w)
        out_h.append(h)
        out_src.append(i)
        out_hum.append(caps[i])

    # Duplicates are appended as *extra* rows, because that is how they arrive:
    # the same photo crawled again from a second site.
    n_dup = int(round(0.09 * n))
    for k in range(n_dup):
        j = int(rng.integers(n))
        exact = k < n_dup // 2
        arr = out_img[j] if exact else _shift(_jpeg_roundtrip(out_img[j]))
        out_img.append(arr)
        out_alt.append(out_alt[j] if exact else out_alt[j].lower())
        out_def.append("dup_exact" if exact else "dup_near")
        out_w.append(out_w[j])
        out_h.append(out_h[j])
        out_src.append(out_src[j])
        out_hum.append(out_hum[j])

    perm = rng.permutation(len(out_img))           # shuffle: a crawl is unordered
    crawl = {
        "images": np.stack([out_img[i] for i in perm]),
        "alt": [out_alt[i] for i in perm],
        "defect": [out_def[i] for i in perm],
        "w": np.array([out_w[i] for i in perm], dtype=np.int32),
        "h": np.array([out_h[i] for i in perm], dtype=np.int32),
        "src": np.array([out_src[i] for i in perm], dtype=np.int32),
        "human": [out_hum[i] for i in perm],
    }
    width = max(len(c) for c in crawl["human"])
    np.savez_compressed(
        cache, images=crawl["images"], alt=np.array(crawl["alt"], dtype=object),
        defect=np.array(crawl["defect"], dtype=object), w=crawl["w"],
        h=crawl["h"], src=crawl["src"],
        human=np.array([c + [""] * (width - len(c)) for c in crawl["human"]],
                       dtype=object))
    if verbose:
        print(f"    crawl: {len(crawl['alt'])} records", flush=True)
    return crawl


# ---------------------------------------------------------------------------
# filter 1: near-duplicate removal with a perceptual hash
# ---------------------------------------------------------------------------
def dhash(arr, size=8):
    """A 64-bit *difference hash*.

    Shrink the picture to 9x8 grey pixels and record, for each row, whether
    each pixel is brighter than the one to its right. 8 rows x 8 comparisons =
    64 bits. It is called a *perceptual* hash because -- unlike a normal file
    hash such as SHA-256, where changing one byte changes everything -- two
    pictures that *look* the same land on almost the same bits, so "nearly
    equal" becomes a small integer distance instead of a yes/no answer.
    """
    g = np.asarray(Image.fromarray(arr).convert("L")
                   .resize((size + 1, size), Image.BICUBIC), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


_DCT_CACHE = {}


def _dct_matrix(n):
    """The DCT-II basis as a matrix, so a 2-D DCT is two matrix products.

    DCT = *discrete cosine transform*: it rewrites the picture as a sum of
    cosine waves, cheapest (smoothest) wave first. The top-left corner of the
    result is the low-frequency part -- the broad shapes -- which is exactly the
    part that survives re-compression and resizing.
    """
    if n not in _DCT_CACHE:
        k = np.arange(n)[:, None]
        x = np.arange(n)[None, :]
        _DCT_CACHE[n] = np.cos(np.pi * (x + 0.5) * k / n)
    return _DCT_CACHE[n]


def phash(arr, size=8, work=32):
    """A 64-bit *pHash*: the classic frequency-domain perceptual hash.

    dhash compares neighbouring pixels, so a small blur or re-compression can
    flip many of its bits. pHash instead keeps only the 8x8 lowest-frequency DCT
    coefficients and records which are above their own median. Rewriting the
    file, resizing it, or nudging the brightness barely touches those numbers.
    """
    g = np.asarray(Image.fromarray(arr).convert("L")
                   .resize((work, work), Image.BICUBIC), dtype=np.float64)
    d = _dct_matrix(work)
    low = (d @ g @ d.T)[:size, :size]
    med = np.median(low.reshape(-1)[1:])          # ignore the DC term (the mean)
    return int(np.packbits(low.reshape(-1) > med).view(">u8")[0])


def hamming(a, b):
    """How many of the 64 bits differ. Named after Richard Hamming, who defined
    this distance in 1950 for error-correcting codes."""
    return bin(a ^ b).count("1")


def dedup(hashes, max_dist=6, bands=8):
    """Keep the first copy of every group of near-identical images.

    Comparing all pairs is O(M^2). Instead we cut each 64-bit hash into
    `bands` chunks of 8 bits and only compare records that share a whole chunk
    -- that is locality-sensitive hashing (LSH): two hashes within 6 bits of
    each other almost always agree on at least one of 8 chunks, so the cheap
    bucket test loses very little and skips almost every comparison.

    -> (keep_mask, dropped_as_duplicate_of dict)
    """
    keep = np.ones(len(hashes), dtype=bool)
    buckets = [{} for _ in range(bands)]
    dup_of, comparisons = {}, 0
    for i, hv in enumerate(hashes):
        cand = set()
        for b in range(bands):
            chunk = (hv >> (8 * b)) & 0xFF
            cand.update(buckets[b].get(chunk, ()))
        hit = None
        for j in sorted(cand):
            comparisons += 1
            if hamming(hv, hashes[j]) <= max_dist:
                hit = j
                break
        if hit is None:
            for b in range(bands):
                buckets[b].setdefault((hv >> (8 * b)) & 0xFF, []).append(i)
        else:
            keep[i] = False
            dup_of[i] = hit
    return keep, dup_of, comparisons


# ---------------------------------------------------------------------------
# filter 2: resolution and aspect ratio
# ---------------------------------------------------------------------------
def size_filter(w, h, min_side=64, max_ratio=3.0):
    """Drop images that are too small to carry detail, or shaped like a banner.

    Both rules are about *training value*: a 24x24 thumbnail upscaled to the
    model's input size adds no information, and a 960x120 strip is almost
    always a website banner rather than a photograph.
    """
    w, h = np.asarray(w, dtype=np.float64), np.asarray(h, dtype=np.float64)
    ratio = np.maximum(w / np.maximum(h, 1), h / np.maximum(w, 1))
    return (np.minimum(w, h) >= min_side) & (ratio <= max_ratio)


# ---------------------------------------------------------------------------
# filter 3: alt-text quality and the safety blocklist
# ---------------------------------------------------------------------------
_FILENAME = re.compile(r"^(img|dsc|dscn|pic|photo|image|untitled|default)[\W_]*\d*"
                       r"(\.(jpe?g|png|gif|webp))?$", re.I)
_TEMPLATE = re.compile(
    r"click here|download this|no description|see more|thumbnail|stock photo|"
    r"picture \d+|\d+ x \d+ pixels|image not available", re.I)


def text_ok(alt, min_words=4):
    """-> (keep, reason). Cheap string rules, run before anything neural."""
    t = alt.strip()
    low = t.lower()
    if set(_WORD.findall(low)) & BLOCKLIST:
        return False, "blocked"
    if _FILENAME.match(low) or _TEMPLATE.search(low):
        return False, "boilerplate"
    words = _WORD.findall(low)
    if len(words) < min_words:
        return False, "too_short"
    if len(low) and sum(c.isalpha() or c.isspace() for c in low) / len(low) < 0.7:
        return False, "not_prose"
    return True, "ok"


def text_filter(alts):
    keep = np.ones(len(alts), dtype=bool)
    reasons = []
    for i, a in enumerate(alts):
        ok, why = text_ok(a)
        keep[i] = ok
        reasons.append(why)
    return keep, reasons


# ---------------------------------------------------------------------------
# filter 4: the CLIP score
# ---------------------------------------------------------------------------
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _pooled(out):
    """transformers >= 5 returns an output object here; older versions a tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


def load_clip(name=CLIP_NAME):
    from transformers import CLIPModel, CLIPTokenizerFast
    model = CLIPModel.from_pretrained(name, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, CLIPTokenizerFast.from_pretrained(name)


@torch.no_grad()
def clip_scores(images, texts, model=None, tok=None, batch=64, verbose=True):
    """Cosine similarity between each image and its own alt-text, from a real
    frozen CLIP ViT-B/32. This is the Phase-3 filter (project 14) at crawl scale.
    """
    if model is None:
        model, tok = load_clip()
    out = np.zeros(len(images), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(images), batch):
        sl = slice(i, i + batch)
        px = np.stack([np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC),
                                  dtype=np.float32) / 255.0 for a in images[sl]])
        px = torch.from_numpy(((px - _CLIP_MEAN) / _CLIP_STD).transpose(0, 3, 1, 2))
        v = _pooled(model.get_image_features(pixel_values=px))
        enc = tok(list(texts[sl]), padding=True, truncation=True, max_length=77,
                  return_tensors="pt")
        t = _pooled(model.get_text_features(**enc))
        v = v / v.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        out[sl] = (v * t).sum(-1).numpy()
        if verbose and (i // batch) % 10 == 0:
            print(f"    scored {min(i + batch, len(images))}/{len(images)}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return out, time.time() - t0


# ---------------------------------------------------------------------------
# small helpers shared with project 38
# ---------------------------------------------------------------------------
def survivors(crawl, keep):
    """Index array of the records that passed every filter, in crawl order."""
    return np.flatnonzero(np.asarray(keep))


def load_recaptions():
    path = data_dir() / "recaptions.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# the recaptioner
# ---------------------------------------------------------------------------
CAPTIONER = "Salesforce/blip-image-captioning-base"


def load_captioner(name=CAPTIONER):
    """A real pretrained captioning VLM (BLIP-base, 224M parameters).

    This is the "strong VLM" of the recaptioning recipe. It is ~150x smaller
    than the models DALL-E 3 or ShareGPT4V used, but it plays the same role:
    it is a *separate, already-trained* model whose only job is to look at a
    picture and write what it sees. Nothing about it is trained here.
    """
    from transformers import AutoProcessor, BlipForConditionalGeneration
    proc = AutoProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


@torch.no_grad()
def recaption(images, ids, model=None, proc=None, batch=16, max_new=20,
              verbose=True, cache_path=None):
    """Rewrite the alt-text of `ids` by looking at the pictures.

    Results are cached in ``data/recaptions.json`` keyed by record id, so
    projects 37 and 38 never caption the same image twice.
    """
    cache_path = cache_path or (data_dir() / "recaptions.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [int(i) for i in ids if str(int(i)) not in cache]
    if not todo:
        return cache, 0.0
    if model is None:
        model, proc = load_captioner()
    t0 = time.time()
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        pil = [Image.fromarray(images[i]) for i in chunk]
        inp = proc(images=pil, return_tensors="pt")
        out = model.generate(**inp, max_new_tokens=max_new, num_beams=1)
        for i, txt in zip(chunk, proc.batch_decode(out, skip_special_tokens=True)):
            cache[str(i)] = txt.strip()
        if verbose and (k // batch) % 10 == 0:
            done = min(k + batch, len(todo))
            print(f"    recaptioned {done}/{len(todo)}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        cache_path.write_text(json.dumps(cache))
    secs = time.time() - t0
    cache_path.write_text(json.dumps(cache))
    return cache, secs
