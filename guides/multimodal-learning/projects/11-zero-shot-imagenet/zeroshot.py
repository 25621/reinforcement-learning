"""Zero-shot classification with a frozen CLIP: data, prompts, and the classifier.

Two ImageNet subsets, chosen so the same experiment can be run on an easy task
and a hard one:

    imagenette  10 ImageNet classes picked to be *easy* to tell apart
    imagewoof   10 ImageNet classes that are all dog breeds -- *hard*

Both are official fast.ai subsets of real ImageNet, ~95 MB each at 160px.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "02-visualize-the-modality-gap"))
import clip_lib                                                   # noqa: E402

DATA = HERE / "data"

URLS = {
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "imagewoof": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz",
}

# WordNet ids are what the folders are actually called. The human-readable names
# come from the ImageNet class list -- CLIP has never seen a WordNet id, which is
# exactly what makes the "wnid" prompt below a useful control.
CLASSES = {
    "imagenette": [
        ("n01440764", "tench"),
        ("n02102040", "English springer"),
        ("n02979186", "cassette player"),
        ("n03000684", "chain saw"),
        ("n03028079", "church"),
        ("n03394916", "French horn"),
        ("n03417042", "garbage truck"),
        ("n03425413", "gas pump"),
        ("n03445777", "golf ball"),
        ("n03888257", "parachute"),
    ],
    "imagewoof": [
        ("n02086240", "Shih-Tzu"),
        ("n02087394", "Rhodesian ridgeback"),
        ("n02088364", "beagle"),
        ("n02089973", "English foxhound"),
        ("n02093754", "Border terrier"),
        ("n02096294", "Australian terrier"),
        ("n02099601", "golden retriever"),
        ("n02105641", "Old English sheepdog"),
        ("n02111889", "Samoyed"),
        ("n02115641", "dingo"),
    ],
}

# What kind of thing the classes are. Used by the "context" prompt, which tells
# the text encoder which sense of an ambiguous word to use.
CATEGORY = {"imagenette": "object", "imagewoof": "dog"}

# The subset of OpenAI's published 80 ImageNet templates that they report as a
# strong small ensemble.
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a photo of one {}.",
]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fetch(name, verbose=True):
    root = DATA / name
    if root.exists():
        return root
    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / f"{name}.tgz"
    if not tgz.exists():
        if verbose:
            print(f"  downloading {name} (~95 MB)...", flush=True)
        urllib.request.urlretrieve(URLS[name], tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(DATA, filter="data")
    extracted = next(p for p in DATA.iterdir() if p.is_dir() and p.name.startswith(name))
    if extracted != root:
        extracted.rename(root)
    tgz.unlink()
    return root


def load_split(name, per_class=100, split="val", seed=0):
    """-> (list of image paths, int64 labels). Class-balanced and deterministic."""
    root = fetch(name) / split
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for label, (wnid, _) in enumerate(CLASSES[name]):
        files = sorted((root / wnid).glob("*.JPEG"))
        pick = rng.permutation(len(files))[:per_class]
        paths += [files[i] for i in sorted(pick)]
        labels += [label] * len(pick)
    return paths, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def _pixels(paths):
    arr = np.stack([
        np.asarray(clip_lib._center_crop_resize(Image.open(p)), dtype=np.float32) / 255.0
        for p in paths])
    arr = (arr - clip_lib.CLIP_MEAN) / clip_lib.CLIP_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))


@torch.no_grad()
def encode_texts(texts, batch=256):
    """Faster than clip_lib's version: pad to the longest caption in the batch
    instead of CLIP's full 77-token context.

    That is safe here and it is not an approximation. CLIP's text transformer is
    causal -- every token may only look left -- and the sentence vector is read
    at the end-of-text token, so tokens *after* it cannot influence the result.
    Padding to 77 makes the model do 7x the work for a bit-identical answer.
    We encode ~27,000 class-name sentences below, so this matters.
    """
    model, tok = clip_lib.get_model()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=77, return_tensors="pt")
        out.append(clip_lib._pooled(model.get_text_features(**enc)).numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def encode_images(paths, batch=32, verbose=True):
    model, _ = clip_lib.get_model()
    out = []
    for i in range(0, len(paths), batch):
        out.append(clip_lib._pooled(
            model.get_image_features(pixel_values=_pixels(paths[i:i + batch]))).numpy())
        if verbose and (i // batch) % 10 == 0:
            print(f"    images {i}/{len(paths)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def cached_features(name, per_class=100):
    """Encode each dataset once; every prompt experiment reuses the same vectors."""
    cache = DATA / f"feat_{name}_{per_class}.npz"
    paths, labels = load_split(name, per_class)
    if cache.exists():
        return np.load(cache)["feat"], labels
    print(f"  encoding {len(paths)} {name} images with CLIP...")
    feat = encode_images(paths)
    np.savez(cache, feat=feat)
    return feat, labels


# ---------------------------------------------------------------------------
# the full 1,000-class ImageNet label space
# ---------------------------------------------------------------------------
# We only hold images for 20 classes, but zero-shot classification does not need
# images to *define* a class -- a class is just a sentence. So the candidate list
# can be all 1,000 ImageNet classes, which is the setting the project is named
# for, and the setting where a 10-class subset flatters CLIP badly.
_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
_OPENAI_NB = "https://raw.githubusercontent.com/openai/CLIP/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb"


def _cached_download(url, filename):
    path = DATA / filename
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as r:
            path.write_bytes(r.read())
    return path


def imagenet_index():
    """-> {wnid: class index 0..999}, the standard ImageNet-1k ordering."""
    import json
    d = json.loads(_cached_download(_INDEX_URL, "imagenet_class_index.json").read_text())
    return {v[0]: int(k) for k, v in d.items()}


def imagenet_names(source):
    """The 1,000 class names, in index order, from one of two naming conventions.

    lemma   the raw WordNet lemma shipped with ImageNet ("kite", "crane").
            Several are ambiguous or archaic, and nobody fixed them.
    openai  the list OpenAI hand-edited for CLIP: ambiguous words got a
            parenthetical ("kite (bird of prey)"), archaic words got modern
            synonyms. Same 1,000 classes, better sentences.
    """
    import json
    import re as _re
    if source == "lemma":
        d = json.loads(_cached_download(_INDEX_URL, "imagenet_class_index.json").read_text())
        return [d[str(i)][1].replace("_", " ") for i in range(1000)]
    if source == "openai":
        nb = json.loads(_cached_download(_OPENAI_NB, "clip_prompts.ipynb").read_text())
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if "imagenet_classes" in src and '"tench"' in src:
                names = _re.findall(r'"([^"]*)"', src)
                assert len(names) == 1000, len(names)
                return names
        raise RuntimeError("could not find the class list in the CLIP notebook")
    if source == "wnid":
        d = json.loads(_cached_download(_INDEX_URL, "imagenet_class_index.json").read_text())
        return [d[str(i)][0] for i in range(1000)]
    raise ValueError(source)


def dataset_indices(name):
    """Where this subset's 10 classes sit in the 1,000-class list."""
    idx = imagenet_index()
    return np.array([idx[w] for w, _ in CLASSES[name]], dtype=np.int64)


@torch.no_grad()
def full_classifier(source, style, cache=True):
    """A 1,000-way zero-shot classifier: one weight vector per ImageNet class."""
    path = DATA / f"w1000_{source}_{style}.npy"
    if cache and path.exists():
        return np.load(path)
    names = imagenet_names(source)
    templates = {"bare": ["{}"], "photo": ["a photo of a {}."],
                 "ensemble": TEMPLATES}[style]
    weights = []
    for start in range(0, 1000, 100):
        chunk = names[start:start + 100]
        flat = [t.format(n) for n in chunk for t in templates]
        emb = clip_lib.l2_normalize(encode_texts(flat))
        emb = emb.reshape(len(chunk), len(templates), -1).mean(1)
        weights.append(clip_lib.l2_normalize(emb))
    weights = np.concatenate(weights).astype(np.float32)
    if cache:
        DATA.mkdir(parents=True, exist_ok=True)
        np.save(path, weights)
    return weights


def full_space_accuracy(feat, labels, name, weights):
    """Score every image against all 1,000 classes; correct means the argmax is
    the image's true ImageNet index."""
    true_idx = dataset_indices(name)[labels]
    scores = clip_lib.l2_normalize(feat) @ weights.T
    pred = scores.argmax(1)
    return float((pred == true_idx).mean()), pred, true_idx


# ---------------------------------------------------------------------------
# prompts -> a zero-shot classifier (10-way)
# ---------------------------------------------------------------------------
def prompt_texts(name, style):
    """The list of sentences used for each class, under one prompting style."""
    names = [n for _, n in CLASSES[name]]
    wnids = [w for w, _ in CLASSES[name]]
    if style == "wnid":                    # control: the raw WordNet folder name
        return [[w] for w in wnids]
    if style == "bare":                    # just the label, no sentence
        return [[n] for n in names]
    if style == "photo":
        return [[f"a photo of a {n}."] for n in names]
    if style == "context":                 # tells CLIP which sense of the word
        return [[f"a photo of a {n}, a type of {CATEGORY[name]}."] for n in names]
    if style == "ensemble":
        return [[t.format(n) for t in TEMPLATES] for n in names]
    raise ValueError(style)


@torch.no_grad()
def classifier_weights(name, style, average="normalized"):
    """One weight vector per class -- literally a linear classifier CLIP wrote
    from words alone, with no training and no labelled images.

    `average` matters when a class has several templates:
      normalized  L2-normalize each sentence first, then average, then normalize
                  again (what OpenAI's code does)
      raw         average the un-normalized vectors, then normalize once
    """
    per_class = prompt_texts(name, style)
    flat = [s for group in per_class for s in group]
    emb = encode_texts(flat)
    weights, cursor = [], 0
    for group in per_class:
        chunk = emb[cursor:cursor + len(group)]
        cursor += len(group)
        if average == "normalized":
            chunk = clip_lib.l2_normalize(chunk)
        weights.append(clip_lib.l2_normalize(chunk.mean(0, keepdims=True))[0])
    return np.stack(weights)                                     # (n_classes, 512)


def zero_shot_accuracy(feat, labels, weights):
    scores = clip_lib.l2_normalize(feat) @ weights.T
    return float((scores.argmax(1) == labels).mean()), scores


# ---------------------------------------------------------------------------
# the supervised reference point
# ---------------------------------------------------------------------------
def linear_probe(train_x, train_y, test_x, test_y, steps=600, lr=1e-2, seed=0):
    """Plain logistic regression on the same frozen features -- the accuracy you
    could reach if you *did* have labels. Zero-shot's score only means something
    next to this number."""
    torch.manual_seed(seed)
    xs = torch.from_numpy(clip_lib.l2_normalize(train_x))
    ys = torch.from_numpy(train_y)
    head = torch.nn.Linear(xs.shape[1], int(train_y.max()) + 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(head(xs), ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(torch.from_numpy(clip_lib.l2_normalize(test_x))).argmax(1).numpy()
    return float((pred == test_y).mean())
