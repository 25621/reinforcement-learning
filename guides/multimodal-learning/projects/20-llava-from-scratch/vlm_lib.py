"""Shared Phase-5 VLM stack: frozen CLIP + frozen real LLM + a trainable projector.

Every Phase-5 project needs the same three pieces, so they live here and
projects 21, 24 and 25 import this file (project 22 and 23 import the model
half and bring their own images).

  1. `build_cache()` -- real COCO images encoded ONCE by a frozen CLIP ViT-B/32
     into 49 patch tokens of 768 numbers each. Two taps are cached: the
     penultimate Transformer layer (what LLaVA actually uses) and the last one.
  2. `load_llm()` -- a real pretrained instruction-tuned LLM,
     HuggingFaceTB/SmolLM2-135M-Instruct, frozen, plus its tokenizer with one
     added `<image>` placeholder token.
  3. `TinyVLM` -- the LLaVA forward pass: project image tokens into the LLM's
     word-embedding space, splice them in where `<image>` sits, run the LLM.

Why cache CLIP's output instead of running CLIP in the training loop: CLIP is
frozen, so its answer for a given image never changes. One pass costs ~24 ms per
image; recomputing it every epoch would pay that again and again.

Why a 135M LLM when the guide says 1-3B: this runs on a CPU in minutes. The
recipe is identical -- SmolLM2-135M-Instruct is a real pretrained,
instruction-tuned model with a real chat template, just small. Every claim we
make is about the *glue*, which is what Phase 5 is about.

Cache layout, all under ``20-llava-from-scratch/data/``:

    rows.json           the COCO listing (image URL + 5 captions per row)
    clip_penult.npy     (N, 49, 768) float16   frozen CLIP, second-to-last layer
    clip_last.npy       (N, 49, 768) float16   frozen CLIP, last layer
    thumbs.npy          (N, 96, 96, 3) uint8   small copies, only for figures
    captions.json       the 5 captions per image
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

N_IMAGES = 3000
N_VAL = 400
CLIP_TOKENS = 49          # 7x7 patch grid at 32 px per patch; CLS is dropped
CLIP_DIM = 768
LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
IMAGE_TOKEN = "<image>"
THREADS = 12

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=clip-benchmark%2Fwds_mscoco_captions"
    "&config=default&split=test&offset={offset}&length={length}"
)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def data_dir():
    return Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# COCO download + frozen-CLIP encoding
# ---------------------------------------------------------------------------
def _get(url, tries=10, base=3.0):
    """GET with exponential backoff. The Hugging Face listing endpoint answers
    HTTP 429 ('slow down') after ~25 quick calls, far sooner than the image CDN
    does, so every retry waits twice as long as the last."""
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
    """Page through the listing 100 rows at a time, caching after every page so
    an interrupted run resumes instead of restarting."""
    cache = data_dir() / "rows.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
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


def square(img, size):
    """Shortest side -> size, then centre-crop. CLIP's own preprocessing."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def clip_vision(name="openai/clip-vit-base-patch32"):
    """The frozen vision tower on its own (we never need CLIP's text side here)."""
    from transformers import CLIPModel
    torch.set_num_threads(THREADS)
    tower = CLIPModel.from_pretrained(name, dtype=torch.float32).vision_model.eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower


def encode_views(tower, pixels, layers=(-2, -1), batch=64):
    """Run frozen CLIP over uint8 images (B, H, W, 3) and return the patch tokens
    of the requested Transformer layers, CLS dropped.

    Returning several layers costs nothing extra: one forward pass produces all
    of them, we just keep more of the output.
    """
    x = pixels.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
    out = {l: [] for l in layers}
    with torch.no_grad():
        for j in range(0, len(x), batch):
            hs = tower(pixel_values=x[j:j + batch], output_hidden_states=True).hidden_states
            for l in layers:
                out[l].append(hs[l][:, 1:].numpy().astype(np.float16))
    return {l: np.concatenate(v) for l, v in out.items()}


def build_cache(n=N_IMAGES, workers=12, chunk=250, verbose=True):
    """Download n COCO images, run frozen CLIP over them, keep only the features.

    Images are handled in chunks so we never hold 3,000 full-size JPEGs in
    memory: fetch 250, encode 250, throw the pixels away, repeat.
    """
    penult_path = data_dir() / "clip_penult.npy"
    if penult_path.exists():
        return
    rows = _list_rows(n, verbose)
    tower = clip_vision()

    feats = {-2: np.zeros((n, CLIP_TOKENS, CLIP_DIM), dtype=np.float16),
             -1: np.zeros((n, CLIP_TOKENS, CLIP_DIM), dtype=np.float16)}
    thumbs = np.zeros((n, 96, 96, 3), dtype=np.uint8)
    captions = [None] * n

    def grab(i):
        img = Image.open(BytesIO(_get(rows[i]["src"])))
        caps = [c.strip() for c in rows[i]["txt"].split("\n") if c.strip()]
        return (i, np.asarray(square(img, 224), dtype=np.uint8),
                np.asarray(square(img, 96), dtype=np.uint8), caps)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, n, chunk):
            idx = list(range(start, min(start + chunk, n)))
            buf = np.zeros((len(idx), 224, 224, 3), dtype=np.uint8)
            for i, big, small, caps in pool.map(grab, idx):
                buf[i - start] = big
                thumbs[i] = small
                captions[i] = caps
            got = encode_views(tower, buf)
            for l in feats:
                feats[l][start:start + len(idx)] = got[l]
            if verbose:
                print(f"    encoded {min(start + chunk, n)}/{n} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    np.save(data_dir() / "clip_last.npy", feats[-1])
    np.save(data_dir() / "thumbs.npy", thumbs)
    (data_dir() / "captions.json").write_text(json.dumps(captions))
    np.save(penult_path, feats[-2])       # written last: it is the "done" marker


class CocoVLMData:
    """Cached CLIP tokens + the 5 captions per image, split into train/val."""

    def __init__(self, n=N_IMAGES, n_val=N_VAL, layer="penult", seed=0):
        build_cache(n)
        name = {"penult": "clip_penult.npy", "last": "clip_last.npy"}[layer]
        self.feats = np.load(data_dir() / name, mmap_mode="r")[:n]
        self.thumbs = np.load(data_dir() / "thumbs.npy", mmap_mode="r")[:n]
        self.captions = json.loads((data_dir() / "captions.json").read_text())[:n]
        order = np.random.default_rng(seed).permutation(n)
        self.val_ids, self.train_ids = order[:n_val], order[n_val:]

    def image_tokens(self, ids):
        return torch.from_numpy(np.asarray(self.feats[np.asarray(ids)], dtype=np.float32))

    def caption(self, i, k=0):
        caps = self.captions[i]
        return caps[k % len(caps)]


# ---------------------------------------------------------------------------
# the frozen LLM
# ---------------------------------------------------------------------------
def load_llm(name=LLM_NAME, freeze=True):
    """Load the pretrained LLM and add one `<image>` placeholder token.

    Why add a token we immediately overwrite: it keeps the prompt an ordinary
    *string*, so the model's own chat template and tokenizer build the sequence
    exactly as they do for text. The token marks *where* the picture goes; its
    embedding row is never read, because we overwrite those positions with
    projected image features before the LLM sees them.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    llm.resize_token_embeddings(len(tok))
    llm.eval()
    if freeze:
        for p in llm.parameters():
            p.requires_grad_(False)
    return tok, llm


def embedding_rms(llm):
    """Root-mean-square size of one number in the LLM's word embeddings.

    The frozen LLM has only ever seen inputs of roughly this size. Handing it
    vectors ten times larger is like shouting: the layer norms and attention
    scales were tuned for the quiet version.
    """
    w = llm.get_input_embeddings().weight.detach()
    return float(w[:32000].pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# projectors: the only trained weights in stage 1
# ---------------------------------------------------------------------------
class Projector(nn.Module):
    """Rewrite CLIP patch features as things the LLM reads as words.

    kind:
      linear  one matrix                      (LLaVA 1.0)      49 tokens out
      mlp2    two matrices with a GELU        (LLaVA 1.5)      49 tokens out
      pool    average the 7x7 patch grid down to 4x4, then mlp2 (Qwen2-VL's
              patch merger)                                    16 tokens out
      qformer 16 learned queries reading the patches (BLIP-2)  16 tokens out
      prefix  49 learned vectors that ignore the image entirely -- the control
              that separates "learned to write captions" from "learned to look"
    """

    def __init__(self, kind, in_dim, out_dim, out_rms=0.02, n_queries=16,
                 n_layers=2, heads=8, n_prefix=CLIP_TOKENS):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(in_dim)      # CLIP features are big and off-centre
        self.out_rms = out_rms
        if kind == "prefix":
            self.prefix = nn.Parameter(torch.randn(n_prefix, out_dim) * 0.02)
        elif kind == "qformer":
            self.queries = nn.Parameter(torch.randn(n_queries, out_dim) * 0.02)
            self.kv = nn.Linear(in_dim, out_dim)
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "ln1": nn.LayerNorm(out_dim),
                    "attn": nn.MultiheadAttention(out_dim, heads, batch_first=True),
                    "ln2": nn.LayerNorm(out_dim),
                    "ff": nn.Sequential(nn.Linear(out_dim, 2 * out_dim), nn.GELU(),
                                        nn.Linear(2 * out_dim, out_dim)),
                }) for _ in range(n_layers)])
        elif kind == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:                                  # mlp2 and pool
            self.pool_grid = 4                 # 7x7 patches -> 4x4 = 16 tokens
            self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(),
                                      nn.Linear(out_dim, out_dim))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_tokens(self, n_patches=CLIP_TOKENS):
        if self.kind == "qformer":
            return self.queries.shape[0]
        if self.kind == "prefix":
            return self.prefix.shape[0]
        if self.kind == "pool":
            return self.pool_grid ** 2
        return n_patches

    def forward(self, feats):
        if self.kind == "prefix":
            out = self.prefix.unsqueeze(0).expand(feats.shape[0], -1, -1)
            return out * (self.out_rms / out.pow(2).mean().sqrt().clamp(min=1e-6))
        x = self.norm(feats)
        if self.kind == "qformer":
            kv = self.kv(x)
            q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            for b in self.blocks:
                h = b["ln1"](q)
                q = q + b["attn"](h, kv, kv, need_weights=False)[0]
                q = q + b["ff"](b["ln2"](q))
            out = q
        else:
            if self.kind == "pool":
                # average neighbouring patches down to a smaller grid. 7x7 does
                # not divide by 2, so use adaptive pooling (uneven bins) exactly
                # as LLaVA-1.6 does rather than throwing a row and column away.
                b, n, d = x.shape
                g = int(n ** 0.5)
                x = x.transpose(1, 2).reshape(b, d, g, g)
                x = F.adaptive_avg_pool2d(x, self.pool_grid)
                x = x.reshape(b, d, -1).transpose(1, 2)
            out = self.proj(x)
        # match the size the frozen LLM expects from its own embeddings
        return out * (self.out_rms / out.pow(2).mean().sqrt().clamp(min=1e-6))


# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------
def chat(instruction, answer=None, with_image=True, n_img=1):
    """The chat template SmolLM2 was trained with, minus its default system
    message (that block costs 20 tokens of every example and teaches us nothing).
    `<image>` is repeated once per image token so the id sequence and the
    embedding sequence line up position for position."""
    img = (IMAGE_TOKEN * n_img + "\n") if with_image else ""
    s = f"<|im_start|>user\n{img}{instruction}<|im_end|>\n<|im_start|>assistant\n"
    if answer is not None:
        s += f"{answer}<|im_end|>"
    return s


class Batch:
    """Token ids, the mask saying which positions are image slots, and the
    labels (only answer tokens are scored)."""

    def __init__(self, ids, image_mask, labels, attn):
        self.ids, self.image_mask, self.labels, self.attn = ids, image_mask, labels, attn

    def to_dict(self):
        return dict(ids=self.ids, image_mask=self.image_mask,
                    labels=self.labels, attn=self.attn)


def make_batch(tok, instructions, answers, n_img=1, with_image=True, max_len=None):
    """Tokenize prompt+answer pairs into one padded batch.

    Labels are -100 everywhere except the answer tokens: the prompt is *given*,
    not predicted, so scoring it would spend the model's capacity on text it
    always receives for free (project 21 measures what happens if you forget).
    """
    img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    seqs, labs = [], []
    for ins, ans in zip(instructions, answers):
        prompt = tok.encode(chat(ins, None, with_image, n_img), add_special_tokens=False)
        full = tok.encode(chat(ins, ans, with_image, n_img), add_special_tokens=False)
        if max_len:
            full = full[:max_len]
        seqs.append(full)
        labs.append([-100] * len(prompt) + full[len(prompt):])
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), pad_id, dtype=np.int64)
    labels = np.full((len(seqs), T), -100, dtype=np.int64)
    attn = np.zeros((len(seqs), T), dtype=np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = s
        labels[i, :len(l)] = l[:len(s)]
        attn[i, :len(s)] = 1
    ids = torch.from_numpy(ids)
    return Batch(ids, ids == img_id, torch.from_numpy(labels), torch.from_numpy(attn))


def prompt_batches(tok, instructions, n_img=1, with_image=True, bs=16):
    """Prompt-only batches for generation, grouped so every row in a batch has
    the *same* length.

    Why bother grouping: padding a prompt on the left shifts every token's
    position, and a model that reads positions (RoPE) answers a shifted prompt
    slightly differently. Grouping by length means zero padding and therefore
    zero doubt about what caused an answer.
    """
    img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    enc = [tok.encode(chat(ins, None, with_image, n_img), add_special_tokens=False)
           for ins in instructions]
    buckets = {}
    for i, e in enumerate(enc):
        buckets.setdefault(len(e), []).append(i)
    out = []
    for L, idxs in buckets.items():
        for j in range(0, len(idxs), bs):
            part = idxs[j:j + bs]
            ids = torch.tensor([enc[i] for i in part])
            out.append((Batch(ids, ids == img_id,
                              torch.full_like(ids, -100), torch.ones_like(ids)), part))
    return out


# ---------------------------------------------------------------------------
# the VLM itself
# ---------------------------------------------------------------------------
class TinyVLM(nn.Module):
    """LLaVA in one class: frozen LLM + trainable projector + a splice."""

    def __init__(self, llm, projector):
        super().__init__()
        self.llm = llm                     # frozen (not registered as trainable)
        self.projector = projector
        self.body = llm.model               # the Transformer stack without the head
        self.head = llm.lm_head             # vocabulary read-out

    def embed(self, batch, image_feats):
        """Word embeddings, with the `<image>` slots overwritten by image tokens."""
        emb = self.llm.get_input_embeddings()(batch.ids)
        if image_feats is not None:
            vis = self.projector(image_feats).reshape(-1, emb.shape[-1])
            emb = emb.masked_scatter(batch.image_mask.unsqueeze(-1), vis.to(emb.dtype))
        return emb

    def forward(self, batch, image_feats):
        """Return mean cross-entropy over the answer tokens only.

        Only the hidden states that actually predict something go through the
        49,153-row output head -- for a 70-token sequence with a 12-token answer
        that removes about a quarter of the arithmetic and changes no number.
        """
        emb = self.embed(batch, image_feats)
        h = self.body(inputs_embeds=emb, attention_mask=batch.attn, use_cache=False)[0]
        # position t predicts token t+1
        tgt = batch.labels[:, 1:]
        keep = tgt.reshape(-1) != -100
        h = h[:, :-1].reshape(-1, h.shape[-1])[keep]
        return F.cross_entropy(self.head(h).float(), tgt.reshape(-1)[keep])

    @torch.no_grad()
    def answer_nll(self, batch, image_feats):
        """Total (not mean) negative log-likelihood of each answer, per row.
        Used to score candidate captions against one image."""
        emb = self.embed(batch, image_feats)
        h = self.body(inputs_embeds=emb, attention_mask=batch.attn, use_cache=False)[0]
        tgt = batch.labels[:, 1:]
        logp = torch.log_softmax(self.head(h[:, :-1]).float(), -1)
        pick = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (tgt != -100).float()
        return -(pick * mask).sum(1), mask.sum(1)

    @torch.no_grad()
    def greedy_batch(self, tok, batch, image_feats, max_new=8, bad_ids=()):
        """Greedy decoding for a whole batch of prompts at once, with a KV cache.

        The prompts must be *left*-aligned and equal length (build them with
        `make_batch(..., prompt_only)` style padding on the right of the answer
        only), which is the case for every fixed-template question we ask.
        """
        emb = self.embed(batch, image_feats)
        stop = {tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>")}
        past, outs = None, [[] for _ in range(emb.shape[0])]
        live = [True] * emb.shape[0]
        for _ in range(max_new):
            res = self.body(inputs_embeds=emb, past_key_values=past, use_cache=True)
            past = res.past_key_values
            logits = self.head(res[0][:, -1]).float()
            for i in bad_ids:
                logits[:, i] = -1e9
            nxt = logits.argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if live[i] and t in stop:
                    live[i] = False
                elif live[i]:
                    outs[i].append(t)
            if not any(live):
                break
            emb = self.llm.get_input_embeddings()(nxt).unsqueeze(1)
        return [tok.decode(o) for o in outs]

    @torch.no_grad()
    def generate(self, tok, instruction, image_feats, max_new=24, n_img=1,
                 with_image=True, bad_ids=()):
        """Greedy decoding with a KV cache, one prompt at a time."""
        b = make_batch(tok, [instruction], [""], n_img=n_img, with_image=with_image)
        ids = b.ids[:, :-1]                     # drop the trailing <|im_end|>
        b = Batch(ids, b.image_mask[:, :-1], b.labels[:, :-1], b.attn[:, :-1])
        emb = self.embed(b, image_feats)
        past, out = None, []
        for _ in range(max_new):
            res = self.body(inputs_embeds=emb, past_key_values=past, use_cache=True)
            past = res.past_key_values
            logits = self.head(res[0][:, -1]).float()
            for i in bad_ids:
                logits[:, i] = -1e9
            nxt = int(logits.argmax(-1))
            if nxt in (tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<|endoftext|>")):
                break
            out.append(nxt)
            emb = self.llm.get_input_embeddings()(torch.tensor([[nxt]]))
        return tok.decode(out)


# ---------------------------------------------------------------------------
# stage-1 alignment: shared by project 20 (the recipe) and 24 (the comparison)
# ---------------------------------------------------------------------------
INSTRUCTION = "Describe the image."


def zero_feats(n):
    """Blank "image" features: the same shape a real image has, all zeros.

    Every no-image control keeps the image *slots* in the prompt and only empties
    their contents, so the prompt length -- and therefore every position the LLM
    reads -- stays identical. If the control also shortened the prompt we could
    not tell "the picture mattered" from "the sequence got shorter".
    """
    return torch.zeros(n, CLIP_TOKENS, CLIP_DIM)
GALLERY = 20
GROUPS = 5


def val_caption_loss(vlm, tok, data, ids, with_image=True, n_img=None, bs=16):
    """Mean nats per caption token on held-out images."""
    n_img = n_img or vlm.projector.n_tokens()
    tot, n = 0.0, 0
    for i in range(0, len(ids), bs):
        chunk = ids[i:i + bs]
        caps = [data.caption(j, 0) for j in chunk]
        b = make_batch(tok, [INSTRUCTION] * len(chunk), caps, n_img=n_img,
                       with_image=True)
        feats = (data.image_tokens(chunk) if with_image
                 else zero_feats(len(chunk)))
        nll, cnt = vlm.answer_nll(b, feats)
        tot += float(nll.sum())
        n += int(cnt.sum())
    return tot / n


def caption_choice(vlm, tok, data, ids, with_image=True, groups=GROUPS,
                   gallery=GALLERY):
    """Given one image, is its own caption the cheapest of `gallery` candidates?

    Two readings of the same scores:
      raw  -- pick the caption with the lowest cost per token.
      lift -- pick the caption the image *helped* most: subtract the cost of the
              same caption with no image behind it. That removes "this sentence
              is just common English", which is otherwise most of the signal.
    Chance is 1/gallery either way.
    """
    n_img = vlm.projector.n_tokens()
    raw_hit, lift_hit, total = 0, 0, 0
    for k in range(groups):
        g = ids[k * gallery:(k + 1) * gallery]
        if len(g) < gallery:
            break
        caps = [data.caption(j, 0) for j in g]
        b0 = make_batch(tok, [INSTRUCTION] * len(caps), caps, n_img=n_img,
                        with_image=True)
        nll0, n0 = vlm.answer_nll(b0, zero_feats(len(caps)))
        base = (nll0 / n0).numpy()
        scores = np.zeros((len(g), len(caps)))
        for i, img in enumerate(g):
            b = make_batch(tok, [INSTRUCTION] * len(caps), caps, n_img=n_img,
                           with_image=True)
            feats = (data.image_tokens([img] * len(caps)) if with_image
                     else zero_feats(len(caps)))
            nll, cnt = vlm.answer_nll(b, feats)
            scores[i] = (nll / cnt).numpy()
        raw_hit += int((scores.argmin(1) == np.arange(len(g))).sum())
        lift_hit += int(((scores - base[None, :]).argmin(1) == np.arange(len(g))).sum())
        total += len(g)
    return raw_hit / total, lift_hit / total, total


def align_train(kind, data, steps, bs=8, lr=3e-3, seed=0, log_every=25,
                tag=None, eval_groups=GROUPS, val_n=200):
    """Train one projector on COCO captions with everything else frozen.

    `kind="prefix"` is the control: identical machinery, but the tokens handed to
    the LLM do not depend on the image at all.
    """
    tag = tag or kind
    uses_image = kind != "prefix"
    tok, llm = load_llm()
    torch.manual_seed(seed)
    proj = Projector(kind, CLIP_DIM, llm.config.hidden_size,
                     out_rms=embedding_rms(llm))
    vlm = TinyVLM(llm, proj)
    opt = torch.optim.AdamW(trainable(proj), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    n_img = proj.n_tokens()

    val0 = val_caption_loss(vlm, tok, data, data.val_ids[:val_n], uses_image)
    print(f"  [{tag}] step 0 val {val0:.4f}", flush=True)
    curve, t0 = [], time.time()
    for step in range(steps):
        ids = rng.choice(data.train_ids, bs, replace=False)
        caps = [data.caption(i, int(rng.integers(0, 5))) for i in ids]
        b = make_batch(tok, [INSTRUCTION] * bs, caps, n_img=n_img, with_image=True)
        feats = data.image_tokens(ids) if uses_image else zero_feats(bs)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        loss = vlm(b, feats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(proj), 1.0)
        opt.step()
        curve.append((step, float(loss.detach())))
        if (step + 1) % log_every == 0:
            print(f"  [{tag}] {step + 1}/{steps} train {float(loss.detach()):.4f} "
                  f"({(time.time() - t0) / (step + 1) * 1000:.0f} ms/step)", flush=True)
    ms = (time.time() - t0) / steps * 1000
    vl = val_caption_loss(vlm, tok, data, data.val_ids[:val_n], uses_image)
    raw, lift, n = caption_choice(vlm, tok, data, data.val_ids, uses_image,
                                  groups=eval_groups)
    res = dict(arm=tag, kind=kind, steps=steps, bs=bs, lr=lr, image_tokens=n_img,
               uses_image=uses_image, proj_params=n_params(proj),
               val_loss_start=val0, val_loss=vl, choice_raw=raw, choice_lift=lift,
               choice_n=n, ms_per_step=ms, train_seconds=ms * steps / 1000)
    print(f"  [{tag}] done: val {vl:.4f}  choice raw {raw:.3f} lift {lift:.3f}  "
          f"{ms:.0f} ms/step", flush=True)
    return res, curve, (vlm, tok, proj)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def n_params(model):
    return sum(p.numel() for p in trainable(model))


def cosine_lr(step, total, peak, warmup=20, floor=0.05):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))
