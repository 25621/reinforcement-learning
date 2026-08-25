"""Real text encoders in front of a small video DiT — and the 77-token wall.

This file is the Phase-7 backbone: projects 31, 33 and 34 all import it and
reuse the trained model, exactly as Phase 6 reused project 25's `dit_lib`.

What it contains
----------------
* a prompt builder that writes English captions of three different shapes,
  so we can ask "where in the sentence did you put the important words?"
* `encode`, which runs the REAL CLIP-L and T5-base encoders once and caches
  the result (the encoders are frozen, so their output is a constant)
* `TextVideoDiT`, a cross-attention DiT that reads whatever context vectors
  it is handed, from one encoder or two

Why a separate text encoder at all?
-----------------------------------
The DiT is already a transformer, so a fair question is why it cannot just
read the words itself.  Two reasons.

1. *Knowledge.*  CLIP and T5 were trained on far more text than any video
   dataset contains.  They arrive already knowing that "drifting" is a kind of
   motion and that "teal" is a colour.  A text pathway trained from scratch on
   video captions would have to rediscover all of that from a few million
   captions.
2. *Protection.*  Because the encoder is frozen, video training cannot damage
   that knowledge.  The DiT learns to *use* the representation, not to change
   it.

The two encoders are not interchangeable, which is the point of this project:
CLIP was trained to match a whole image to a whole caption and physically
cannot read past 77 tokens; T5 was trained to read and rewrite text and
handles 512.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "28-mmdit-for-video"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import mmdit_lib as MM                                         # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

CLIP_ID = "openai/clip-vit-large-patch14"
T5_ID = "t5-base"
CLIP_LIMIT = 77            # hard-wired into CLIP's position table
T5_PAD = 128               # long enough for every prompt written here
CHUNK_PAD = 154            # two CLIP windows, for the chunking workaround

# ---------------------------------------------------------------------------
# the prompts
# ---------------------------------------------------------------------------
# The decisive clause names the digit and the direction; everything else is
# scenery that carries no information about either.  Two different fillers stop
# the model from memorising one particular blob of text.

FILLER = [
    ("the footage was captured on a vintage anamorphic lens with a shallow "
     "depth of field, and the grain of the film stock is clearly visible in "
     "the darker regions of every frame; the colour grade leans towards a cool "
     "teal in the shadows while the highlights keep a warm amber cast, like a "
     "late-night documentary filmed in a quiet industrial district, with soft "
     "reflections spilling across wet pavement and a faint haze hanging in the "
     "still night air"),
    ("shot handheld on a modern mirrorless camera with a fast prime lens, the "
     "image is clean and almost noiseless, the palette is dominated by muted "
     "greys and dusty pastels, the lighting is diffuse as if the sun were "
     "hidden behind a thin layer of cloud, and the background dissolves into a "
     "gentle blur that keeps a viewer's attention on the single subject in the "
     "middle of the frame, a composition favoured by contemporary advertising"),
]

STYLES = ["short", "long_early", "long_late"]
STYLE_HELP = {
    "short": "the clause alone",
    "long_early": "clause first, then 100 tokens of scenery",
    "long_late": "100 tokens of scenery, then the clause",
}


def clause(digit, direction):
    return f"a {int(digit)} drifting {L.DIRECTIONS[int(direction)]}"


def make_prompt(digit, direction, style, filler):
    c = clause(digit, direction)
    f = FILLER[filler]
    if style == "short":
        return c + "."
    if style == "long_early":
        return f"{c}, {f}."
    return f"{f}, and {c}."


COMBOS = [(d, k) for d in range(10) for k in range(4)]
N_FILLER = len(FILLER)


def prompt_index(digit, direction, style, filler):
    """Position of one prompt in the flat cached list."""
    s = STYLES.index(style)
    return ((int(digit) * 4 + int(direction)) * len(STYLES) + s) * N_FILLER \
        + int(filler)


def all_prompts():
    out = []
    for d, k in COMBOS:
        for s in STYLES:
            for f in range(N_FILLER):
                out.append((d, k, s, f, make_prompt(d, k, s, f)))
    return out


# ---------------------------------------------------------------------------
# running the frozen encoders, once
# ---------------------------------------------------------------------------

def _pad_to(seq, mask, length):
    """Pad (or cut) a batch of token features to a fixed length."""
    n = seq.shape[1]
    if n >= length:
        return seq[:, :length], mask[:, :length]
    pad = length - n
    seq = F.pad(seq, (0, 0, 0, pad))
    mask = F.pad(mask, (0, pad))
    return seq, mask


@torch.no_grad()
def encode_clip(texts, chunked=False, batch=16):
    """CLIP-L text encoder.  `chunked=True` is the community workaround.

    Plain CLIP: tokenise with truncation, so anything past token 77 simply
    never reaches the model.  There is no flag to switch this off — CLIP's
    learned position table has exactly 77 rows, so token 78 has no position to
    be given.

    Chunked CLIP: cut the token stream into 75-token pieces, wrap each piece in
    its own start/end markers, encode the pieces separately and lay the results
    side by side.  Nothing is thrown away — but each piece is read in ignorance
    of the others, so a phrase split across the seam is read as two fragments.
    """
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    enc = CLIPTextModel.from_pretrained(CLIP_ID).eval()
    bos, eos = tok.bos_token_id, tok.eos_token_id
    seqs, masks = [], []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        if not chunked:
            b = tok(part, padding="max_length", truncation=True,
                    max_length=CLIP_LIMIT, return_tensors="pt")
            out = enc(**b).last_hidden_state
            seqs.append(out)
            masks.append(b["attention_mask"])
        else:
            ids = tok(part, truncation=False)["input_ids"]
            for row in ids:
                body = row[1:-1]                       # drop CLIP's own markers
                chunks = [body[j:j + CLIP_LIMIT - 2]
                          for j in range(0, len(body), CLIP_LIMIT - 2)] or [[]]
                built, m = [], []
                for ch in chunks:
                    c = [bos] + list(ch) + [eos]
                    m += [1] * len(c) + [0] * (CLIP_LIMIT - len(c))
                    built.append(c + [eos] * (CLIP_LIMIT - len(c)))
                h = enc(input_ids=torch.tensor(built)) \
                    .last_hidden_state.reshape(1, -1, 768)
                h, mm = _pad_to(h, torch.tensor(m, dtype=torch.float)[None],
                                CHUNK_PAD)
                seqs.append(h)
                masks.append(mm)
    return torch.cat(seqs).half(), torch.cat(masks).float()


@torch.no_grad()
def encode_t5(texts, batch=16):
    """T5-base encoder.  Same width as CLIP-L (768), 512-token limit."""
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(T5_ID)
    enc = T5EncoderModel.from_pretrained(T5_ID).eval()
    seqs, masks = [], []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding="max_length", truncation=True,
                max_length=T5_PAD, return_tensors="pt")
        seqs.append(enc(**b).last_hidden_state)
        masks.append(b["attention_mask"])
    return torch.cat(seqs).half(), torch.cat(masks).float()


SOURCES = {"clip": 768, "clip_chunk": 768, "t5": 768}
ARMS = {"clip": ["clip"], "clip_chunk": ["clip_chunk"], "t5": ["t5"],
        "both": ["clip", "t5"]}


def cache_path(src):
    return CK / f"text_{src}.pt"


def load_text(src):
    p = cache_path(src)
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 train.py --stage encode`")
    return torch.load(p, map_location="cpu", weights_only=False)


class TextBank:
    """Holds the cached encoder outputs for one arm and hands out batches."""

    def __init__(self, arm):
        self.arm = arm
        self.srcs = ARMS[arm]
        self.data = {s: load_text(s) for s in self.srcs}

    def get(self, idx):
        return {s: (self.data[s]["seq"][idx].float(),
                    self.data[s]["mask"][idx]) for s in self.srcs}

    def null(self, n):
        return {s: (self.data[s]["null_seq"].float().expand(n, -1, -1),
                    self.data[s]["null_mask"].expand(n, -1))
                for s in self.srcs}


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class MaskedCrossAttention(nn.Module):
    """Video tokens ask questions; text tokens answer.

    The mask matters more than it looks.  Every prompt is padded to the same
    length so a batch can be one tensor, and without a mask the padding slots
    are ordinary keys the video is free to attend to.  Since padding carries no
    meaning, attending to it is attending to noise — and the amount of padding
    depends on prompt length, so an unmasked model would behave differently for
    long and short prompts *for reasons that have nothing to do with the
    words*.  That would ruin exactly the comparison this project is making.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, ctx, mask):
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        am = mask[:, None, None, :].bool().expand(B, self.h, N, M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class CrossBlock(nn.Module):
    """Self-attention over video, then a look at the prompt, then an MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = L.Attention(dim, heads)
        self.nc = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = MaskedCrossAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, ctx, mask, c, rope=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = \
            self.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
        x = x + g_a * self.attn(self.n1(x) * (1 + sc_a) + sh_a, rope)
        x = x + self.cross(self.nc(x), ctx, mask)
        return x + g_m * self.mlp(self.n2(x) * (1 + sc_m) + sh_m)


class TextVideoDiT(nn.Module):
    """A video DiT steered by one or two frozen text encoders."""

    def __init__(self, arm, in_ch=4, patch=(1, 2, 2), dim=128, depth=5,
                 heads=4):
        super().__init__()
        self.arm, self.patch, self.in_ch = arm, patch, in_ch
        self.dim, self.heads = dim, heads
        self.srcs = ARMS[arm]
        pdim = in_ch * patch[0] * patch[1] * patch[2]
        self.embed = nn.Linear(pdim, dim)
        self.tmlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        # One projection per encoder: CLIP's 768 numbers and T5's 768 numbers
        # mean completely different things, so they must not share a matrix.
        self.ctx_proj = nn.ModuleDict(
            {s: nn.Linear(SOURCES[s], dim) for s in self.srcs})
        self.ctx_norm = nn.ModuleDict(
            {s: nn.LayerNorm(SOURCES[s]) for s in self.srcs})
        # Pooled text is ADDED to the timestep vector driving every AdaLN.
        # Phase 4's project 17 showed what happens when a conditioning path
        # starts far weaker than the signal it is added to: it never catches
        # up.  Hence a plain init and a LayerNorm, not a zero init.
        self.pool_norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [CrossBlock(dim, heads) for _ in range(depth)])
        self.fnorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fada = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.fada.weight)
        nn.init.zeros_(self.fada.bias)
        self.head = nn.Linear(dim, pdim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._rope = {}

    def rope_for(self, grid):
        if grid not in self._rope:
            self._rope[grid] = L.rope_3d(grid, self.dim // self.heads)
        return self._rope[grid]

    def context(self, text):
        """Project every encoder's output into the model's width and stack."""
        seqs, masks = [], []
        for s in self.srcs:
            seq, mask = text[s]
            seqs.append(self.ctx_proj[s](self.ctx_norm[s](seq)))
            masks.append(mask)
        return torch.cat(seqs, 1), torch.cat(masks, 1)

    def forward(self, x, t, text, ctx=None):
        tok, grid = L.patchify(x, self.patch)
        v = self.embed(tok)
        if ctx is None:
            ctx = self.context(text)
        ctx_seq, mask = ctx
        pooled = (ctx_seq * mask[..., None]).sum(1) \
            / mask.sum(1, keepdim=True).clamp(min=1.0)
        c = self.tmlp(L.timestep_embedding(t, self.dim)) \
            + self.pool(self.pool_norm(pooled))
        rope = self.rope_for(grid)
        for blk in self.blocks:
            v = blk(v, ctx_seq, mask, c, rope)
        sh, sc = self.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = self.head(self.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, self.patch, grid, self.in_ch)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

LATENT_SHAPE = (4, 4, 8, 8)          # (C, T, H, W) from project 21's 3D VAE


@torch.no_grad()
def cfg_sample(model, text, null, shape, scale=3.0, steps=30, generator=None,
               x=None, control=None):
    """Rectified-flow sampling with classifier-free guidance.

    Each step asks the model twice — once with the prompt, once with the empty
    prompt — and pushes the answer away from the uninstructed one:

        v = v_null + scale * (v_prompt - v_null)
    """
    flow = FL.RectifiedFlow()
    if x is None:
        x = torch.randn(shape, generator=generator)
    ctx_c = model.context(text)
    ctx_n = model.context(null)
    kw = {} if control is None else {"control": control}
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(shape[0]) * flow.T_SCALE
        v_c = model(x, t, None, ctx=ctx_c, **kw)
        if scale != 1.0:
            v_n = model(x, t, None, ctx=ctx_n, **kw)
            v_c = v_n + scale * (v_c - v_n)
        x = x + (ts[i + 1] - ts[i]) * v_c
    return x


def build(arm, **kw):
    return TextVideoDiT(arm, **kw)


def load_arm(arm, where=None):
    path = (where or CK) / f"{arm}.pt"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run project 30 first:\n"
            f"  cd {HERE} && python3 train.py --stage train --arm {arm}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(arm)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def load_digit_judge():
    """Project 28's digit grader — an independent opinion on what was drawn."""
    import fid_lib
    p = HERE.parent / "28-mmdit-for-video" / "checkpoints" / "probe.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run project 28's "
                         f"`python3 train.py --stage probe` first")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = fid_lib.FeatureNet()
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["acc"]


@torch.no_grad()
def grade(clips, digits, dirs, judge):
    """Did the clip show the digit it was asked for, going the right way?"""
    probs = 0
    for f in (0, 5, 10, 15):
        probs = probs + F.softmax(judge(clips[:, :, f]), dim=1)
    dig_ok = (probs.argmax(1) == digits).float()
    dir_ok = (L.predicted_direction(clips) == dirs).float()
    return (float(dig_ok.mean()), float(dir_ok.mean()),
            float((dig_ok * dir_ok).mean()))


_VAE = None


def decode(z):
    """Latents back to pixels with project 21's frozen 3D VAE."""
    global _VAE
    if _VAE is None:
        _VAE = L.load_vae("3d")
    vae, sc = _VAE
    return vae.decoder(z / sc).clamp(-1, 1)


__all__ = ["ARMS", "STYLES", "TextBank", "TextVideoDiT", "build", "load_arm",
           "cfg_sample", "grade", "decode", "make_prompt", "prompt_index",
           "LATENT_SHAPE", "MM", "FL", "L"]
