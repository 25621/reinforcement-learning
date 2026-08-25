"""Grounding a phrase to a region, with a real frozen CLIP.

[CLIP] was trained to say whether a picture and a caption go together.  It was
never trained to find things, to segment, or to grasp.  Everything in this
file is the standard trick for borrowing it anyway: propose regions with
something else, crop each one, and ask CLIP which crop best matches the
phrase.  That is all "open-vocabulary detection" is at its simplest -- and
experiment 2 measures precisely what it can and cannot do as a result.
"""

import os

import numpy as np
import torch

_MODEL = "openai/clip-vit-base-patch32"
_cache = {}


def load():
    """Load CLIP once.  ~600 MB the first time, then cached on disk."""
    if "model" not in _cache:
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(10)
        _cache["model"] = CLIPModel.from_pretrained(_MODEL).eval()
        _cache["proc"] = CLIPProcessor.from_pretrained(_MODEL)
    return _cache["model"], _cache["proc"]


@torch.no_grad()
def encode_images(crops, batch=32):
    """(N, H, W, 3) uint8 crops -> L2-normalized CLIP image embeddings."""
    model, proc = load()
    out = []
    for i in range(0, len(crops), batch):
        px = proc(images=list(crops[i:i + batch]), return_tensors="pt")["pixel_values"]
        # Go through the vision tower and the projection explicitly rather
        # than calling `get_image_features`: in transformers 5.x that helper
        # returns a model-output object, not a tensor, and the shape of the
        # convenience API has moved between versions.  These two lines are
        # what it does anyway, and they cannot drift.
        f = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).numpy()


_TEXT_CACHE = {}


@torch.no_grad()
def encode_texts(texts):
    """L2-normalized CLIP text embeddings, memoized.

    The same handful of phrases is scored against hundreds of regions, and a
    text embedding depends only on the string -- so computing it once turns
    the phrase side of this project from most of its runtime into nothing.
    """
    texts = list(texts)
    todo = [t for t in texts if t not in _TEXT_CACHE]
    if todo:
        _fill_text_cache(todo)
    return np.stack([_TEXT_CACHE[t] for t in texts])


@torch.no_grad()
def _fill_text_cache(texts):
    model, proc = load()
    tok = proc(text=list(texts), return_tensors="pt", padding=True)
    f = model.text_projection(model.text_model(
        input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output)
    f = (f / f.norm(dim=-1, keepdim=True)).numpy()
    for t, v in zip(texts, f):
        _TEXT_CACHE[t] = v


# --------------------------------------------------------------------------
# turning a mask into something CLIP can look at
# --------------------------------------------------------------------------

def crop_for(img, mask, mode="pad", pad=0.35, dim=112):
    """Cut out the region a mask covers, in one of several styles.

    mode:
      "tight"     the bounding box, nothing else
      "pad"       the bounding box grown by `pad` of its size -- keeps some
                  surroundings, which CLIP's training data always had
      "masked"    tight box with everything outside the mask blanked out
      "highlight" the WHOLE image with a box drawn round the region

    These are not stylistic choices.  CLIP was trained on whole photographs
    with captions, so a 30x30 pixel patch of unfamiliar framing is far outside
    what it ever saw.  Experiment 3 measures how much that matters.
    """
    import cv2
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((dim, dim, 3), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if mode == "highlight":
        out = img.copy()
        cv2.rectangle(out, (int(x0) - 2, int(y0) - 2), (int(x1) + 2, int(y1) + 2),
                      (255, 40, 40), 2)
        return cv2.resize(out, (dim, dim))
    if mode == "pad":
        w, h = x1 - x0, y1 - y0
        x0 = max(int(x0 - pad * w), 0); x1 = min(int(x1 + pad * w), W - 1)
        y0 = max(int(y0 - pad * h), 0); y1 = min(int(y1 + pad * h), H - 1)
    sub = img[int(y0):int(y1) + 1, int(x0):int(x1) + 1].copy()
    if mode == "masked":
        m = mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        sub[~m] = 255
    return cv2.resize(sub, (dim, dim))


_REGION_CACHE = {}


def region_features(img, masks, mode="pad"):
    """CLIP embeddings for a scene's regions, computed once per (scene, mode).

    Without this the same crops are re-encoded for every phrase, and since
    every experiment here asks several phrases about the same scenes, that is
    most of the project's runtime spent recomputing identical numbers.
    """
    key = (id(img), mode, len(masks))
    if key not in _REGION_CACHE:
        crops = [crop_for(img, m, mode=mode) for m in masks]
        _REGION_CACHE[key] = encode_images(np.array(crops))
        if len(_REGION_CACHE) > 400:
            _REGION_CACHE.pop(next(iter(_REGION_CACHE)))
    return _REGION_CACHE[key]


def ground(img, masks, phrase, mode="pad", template="a photo of {}"):
    """Which mask does the phrase refer to?  Returns (index, scores)."""
    fi = region_features(img, masks, mode=mode)
    ft = encode_texts([template.format(phrase)])
    s = (fi @ ft.T).reshape(-1)
    return int(np.argmax(s)), s
