"""Shared Phase-1 backbone: a real, frozen CLIP + a small COCO image-caption set.

Projects 02 (modality gap) and 03 (toy retrieval) both need exactly the same two
things: a few hundred real (image, caption) pairs, and a pretrained CLIP that
turns each of them into one vector. That shared machinery lives here so the two
projects cannot drift apart -- they compare the *same* embeddings.

Everything is cached on disk:

    data/images/0000.jpg ...   224x224 center crops of COCO test images
    data/captions.json         the 5 human captions COCO ships per image
    data/emb_<tag>.npz         CLIP embeddings, so you encode once, not once per stage

Model: openai/clip-vit-base-patch32 (~600 MB, downloaded once by transformers).
On 12 CPU threads it encodes ~35 images/second, so 1,000 images take ~30 s.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "openai/clip-vit-base-patch32"

# The exact pixel statistics CLIP was trained with. Feeding it ImageNet's
# constants instead is a classic silent bug: no crash, just worse numbers.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMG_SIZE = 224

# COCO's 5,000-image test split, already paired with its 5 human captions each,
# served as plain rows by the Hugging Face datasets server.
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=clip-benchmark%2Fwds_mscoco_captions"
    "&config=default&split=test&offset={offset}&length={length}"
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _center_crop_resize(img, size=IMG_SIZE):
    """Shortest side -> size, then center-crop to size x size (CLIP's recipe)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def fetch_coco(data_dir, n=1000, workers=16):
    """Download n COCO (image, 5 captions) pairs into data_dir. Idempotent."""
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    cap_path = data_dir / "captions.json"
    if cap_path.exists() and len(list(img_dir.glob("*.jpg"))) >= n:
        return
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    while len(rows) < n:
        batch = min(100, n - len(rows))                      # server caps at 100
        url = _ROWS_URL.format(offset=len(rows), length=batch)
        with urllib.request.urlopen(url, timeout=60) as r:
            rows += json.load(r)["rows"]

    def grab(i_row):
        i, row = i_row
        dst = img_dir / f"{i:04d}.jpg"
        if not dst.exists():
            with urllib.request.urlopen(row["row"]["jpg"]["src"], timeout=60) as r:
                raw = r.read()
            from io import BytesIO
            _center_crop_resize(Image.open(BytesIO(raw))).save(dst, quality=92)
        # COCO ships ~5 captions per image, newline-separated in this mirror.
        return [c.strip() for c in row["row"]["txt"].split("\n") if c.strip()]

    with ThreadPoolExecutor(workers) as pool:
        captions = list(pool.map(grab, enumerate(rows)))
    cap_path.write_text(json.dumps(captions))


def load_pairs(data_dir, n=1000, caption_index=0):
    """Return (image paths, one caption each). caption_index picks which of the 5."""
    data_dir = Path(data_dir)
    captions = json.loads((data_dir / "captions.json").read_text())[:n]
    paths = [data_dir / "images" / f"{i:04d}.jpg" for i in range(len(captions))]
    picked = [c[caption_index % len(c)] for c in captions]
    return paths, picked, captions


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
_MODEL = None
_TOK = None


def get_model():
    """Load CLIP once per process and freeze it. Nothing here is ever trained."""
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import CLIPModel, CLIPTokenizerFast
        _MODEL = CLIPModel.from_pretrained(MODEL_ID).eval()
        for p in _MODEL.parameters():
            p.requires_grad_(False)
        _TOK = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    return _MODEL, _TOK


def _pooled(out):
    """transformers >=5 returns an output object; older versions a bare tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


def _preprocess(paths):
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                    for p in paths])                          # (B,224,224,3)
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))        # (B,3,224,224)


@torch.no_grad()
def encode_images(paths, batch=32, verbose=True):
    """-> (N, 512) raw (un-normalized) CLIP image embeddings."""
    model, _ = get_model()
    out = []
    for i in range(0, len(paths), batch):
        out.append(_pooled(model.get_image_features(
            pixel_values=_preprocess(paths[i:i + batch]))).numpy())
        if verbose and (i // batch) % 10 == 0:
            print(f"    images {i + len(out[-1])}/{len(paths)}", flush=True)
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def encode_texts(texts, batch=64, verbose=True):
    """-> (N, 512) raw (un-normalized) CLIP text embeddings."""
    model, tok = get_model()
    out = []
    for i in range(0, len(texts), batch):
        # 77 is CLIP's hard context length; longer captions are silently cut.
        enc = tok(texts[i:i + batch], padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt")
        out.append(_pooled(model.get_text_features(**enc)).numpy())
        if verbose and (i // batch) % 10 == 0:
            print(f"    texts {i + len(out[-1])}/{len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def cached_embeddings(data_dir, n=1000, caption_index=0):
    """Encode once, reuse forever. -> (img_emb, txt_emb, paths, captions, all_caps).

    Images and captions are cached in *separate* files on purpose: COCO ships 5
    captions per image, and sweeping over which one you use must not re-encode
    the (much more expensive) images five times.
    """
    data_dir = Path(data_dir)
    paths, captions, all_caps = load_pairs(data_dir, n, caption_index)

    img_cache = data_dir / f"emb_img_{n}.npz"
    if img_cache.exists():
        img = np.load(img_cache)["img"]
    else:
        print(f"  encoding {len(paths)} images with CLIP...")
        img = encode_images(paths)
        np.savez(img_cache, img=img)

    txt_cache = data_dir / f"emb_txt_{n}_{caption_index}.npz"
    if txt_cache.exists():
        txt = np.load(txt_cache)["txt"]
    else:
        print(f"  encoding {len(captions)} captions with CLIP...")
        txt = encode_texts(captions)
        np.savez(txt_cache, txt=txt)
    return img, txt, paths, captions, all_caps


# --------------------------------------------------------------------------
# small numeric helpers (no sklearn/scipy in this environment)
# --------------------------------------------------------------------------
def l2_normalize(x, eps=1e-8):
    """Scale every row to length 1. Then dot product == cosine similarity."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def pca(X, k=2):
    """Return (scores, components, explained_variance_ratio, mean) via SVD."""
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(len(X) - 1, 1)
    evr = var / var.sum()
    return Xc @ Vt[:k].T, Vt[:k], evr, mean


def recall_at_k(sim, ks=(1, 5, 10)):
    """sim[i, j] = score of query i against gallery j; correct answer is j == i."""
    order = np.argsort(-sim, axis=1)
    rank = np.argmax(order == np.arange(len(sim))[:, None], axis=1)   # 0-based
    return {k: float((rank < k).mean()) for k in ks}
