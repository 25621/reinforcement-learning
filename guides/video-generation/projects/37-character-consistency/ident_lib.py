"""Keeping the SAME character across many shots.

The problem
-----------
A caption can say "a 3 drifting right".  It cannot say *which* 3 — whose
handwriting, which exact shape.  So a text-only model draws a fresh, unrelated
3 in every shot, and a viewer watching four shots in a row sees the character
change between cuts.  That is *drift*, and it is the reason a story made of
several generations falls apart even when every single shot looks fine.

"But the model already reads text — isn't the digit already specified?"
-----------------------------------------------------------------------
Only the *class* is.  Words are a coarse handle: "3" names a category with
thousands of members.  Identity lives below the resolution of language, which
is why every production system solves it with a **picture**, not a longer
caption.  This project compares the two standard ways of handing a model a
picture of the character:

    IP-Adapter    an extra image branch, trained ONCE across many characters.
                  At generation time you hand it a reference photo and it
                  works — including for a character it has never seen.

    character LoRA  a few hundred KB of weights, fine-tuned per character on a
                  handful of that character's clips.  Nothing new at
                  generation time; the character is baked into the weights.

Why the IP-Adapter needs its own attention instead of reusing the text one
-------------------------------------------------------------------------
A reasonable objection: the model already has a cross-attention layer that
reads a sequence of context vectors — why not just append the image tokens to
the text tokens and be done?  Because those two sequences are answers to
different questions.  The text keys were shaped, over the whole of training,
to answer "what should happen"; the image keys answer "what should it look
like".  Mixing them into one softmax makes the two compete for the same
attention budget, so a strong reference silently suppresses the prompt.
IP-Adapter therefore gives the image its own keys, values and output
projection, and *adds* the result:

    out = cross_attn_text(x) + s * cross_attn_image(x)

Both branches get their full share, and `s` becomes a dial for how strongly
the reference is imposed.  The name of the design is *decoupled*
cross-attention, and "decoupled" is exactly what it says: the two attentions
are pulled apart rather than merged.

The identity ruler
------------------
Everything is measured with `long_lib.glyph_crops` — a 28x28 box centred on
the digit, so position and motion are removed and only handwriting is left.
Two reference values make the numbers readable, and both are measured, not
assumed:

    ~0.051   the same character after a VAE round-trip  (the best achievable)
    ~0.131   a different person's 3                     (no identity at all)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
sys.path.insert(0, str(HERE.parent / "34-lora-for-video"))
sys.path.insert(0, str(HERE.parent / "35-sliding-window-t2v"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402
import lora_lib as LR                                          # noqa: E402
import long_lib as LL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

N_CHARS_TRAIN = 64          # how many different people's handwriting it sees
N_CHARS_TEST = 8            # characters kept back, never trained on
CLIPS_PER_CHAR = 16
SHOTS = [0, 1, 2, 3]        # a four-shot story: right, down, left, up
ARMS = ["text", "ip", "lora"]
CFG = 3.0
STEPS = 30


# ---------------------------------------------------------------------------
# characters and their clips
# ---------------------------------------------------------------------------

def character_sprites(n_train=N_CHARS_TRAIN, n_test=N_CHARS_TEST, seed=37):
    """Pick sprite indices to act as our cast.

    Training characters come from MNIST's train split (the split the base
    model and the VAE already saw); test characters come from the test split,
    so "works on an unseen character" means genuinely unseen.
    """
    rng = np.random.default_rng(seed)
    _, tr_lab = L._digit_sprites(True)
    _, te_lab = L._digit_sprites(False)
    train = [(int(rng.choice(np.nonzero(tr_lab == d)[0])), d)
             for d in [i % 10 for i in range(n_train)]]
    # Test digits repeat (0,1,2,3,0,1,2,3) on purpose: the identity ruler needs
    # a "different person writing the SAME digit" pair, which is impossible if
    # every test character has a digit to itself.
    test = [(int(rng.choice(np.nonzero(te_lab == d)[0])), d)
            for d in [i % 4 for i in range(n_test)]]
    return train, test


@torch.no_grad()
def build_cache(chars, per_char=CLIPS_PER_CHAR, train_split=True, seed=1,
                name="chars", batch=16):
    """Several clips of each character, plus their latents.

    Storing several clips per character is what makes the reference image
    honest: at training time the reference is taken from a DIFFERENT clip of
    the same character, so the adapter cannot succeed by copying the position
    or the motion — only the handwriting is shared between the reference and
    the target.
    """
    vae, scale = L.load_vae("3d")
    rng = np.random.default_rng(seed)
    lats, clips, digs, dirs, cid = [], [], [], [], []
    flat = [(i, s, d) for i, (s, d) in enumerate(chars)
            for _ in range(per_char)]
    for i in range(0, len(flat), batch):
        part = flat[i:i + batch]
        # `L.attr_batch` picks a RANDOM sprite of the requested class, which is
        # exactly what we must not have here — every clip of one character has
        # to show the same person's handwriting.  `_redraw` is `attr_batch`
        # with the sprite pinned instead of drawn.
        x, dg, dd = _redraw(rng, part, train_split)
        mean, _ = vae.encode(x)
        lats.append(mean * scale)
        clips.append(x)
        digs.append(dg)
        dirs.append(dd)
        cid += [p[0] for p in part]
    out = dict(latents=torch.cat(lats), clips=torch.cat(clips),
               digit=torch.cat(digs), direction=torch.cat(dirs),
               char=torch.tensor(cid), chars=chars, scale=scale)
    torch.save(out, CK / f"{name}.pt")
    return out


def _redraw(rng, part, train_split):
    """Render one clip per (character, direction) with the character's sprite."""
    sprites, _ = L._digit_sprites(train_split)
    B = len(part)
    x = torch.zeros(B, 1, L.T_FRAMES, L.CANVAS, L.CANVAS)
    dg = torch.zeros(B, dtype=torch.long)
    dd = torch.zeros(B, dtype=torch.long)
    D, H, W = L.DIGIT_PX, L.CANVAS, L.CANVAS
    for b, (_, sidx, digit) in enumerate(part):
        d = int(rng.integers(4))
        dy, dx = L.DIR_VEC[L.DIRECTIONS[d]]
        sp = float(rng.uniform(1.4, 2.2))
        travel = sp * (L.T_FRAMES - 1)
        lim_y, lim_x = H - D, W - D
        y0 = rng.uniform(0, max(lim_y - travel, 0)) if dy > 0 else \
            rng.uniform(min(travel, lim_y), lim_y) if dy < 0 else \
            rng.uniform(0, lim_y)
        x0 = rng.uniform(0, max(lim_x - travel, 0)) if dx > 0 else \
            rng.uniform(min(travel, lim_x), lim_x) if dx < 0 else \
            rng.uniform(0, lim_x)
        frame = np.zeros((L.T_FRAMES, H, W), dtype=np.float32)
        for t in range(L.T_FRAMES):
            y = int(round(np.clip(y0 + dy * sp * t, 0, lim_y)))
            xx = int(round(np.clip(x0 + dx * sp * t, 0, lim_x)))
            np.maximum(frame[t, y:y + D, xx:xx + D], sprites[sidx],
                       out=frame[t, y:y + D, xx:xx + D])
        x[b, 0] = torch.from_numpy(frame) * 2 - 1
        dg[b], dd[b] = digit, d
    return x, dg, dd


def load_cache(name="chars"):
    p = CK / f"{name}.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage cache`")
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# the IP-Adapter
# ---------------------------------------------------------------------------

class RefEncoder(nn.Module):
    """One reference frame (64x64) -> 16 context tokens.

    Deliberately small and trained from scratch.  A real IP-Adapter uses a
    frozen CLIP *image* encoder here for the same reason project 30 used a
    frozen CLIP text encoder: it arrives already knowing what things look
    like.  Our world contains ten handwritten digits, so there is nothing for
    a giant pretrained encoder to contribute that 200k parameters cannot learn
    from the data itself.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(),      # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),     # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),    # 8x8
            nn.Conv2d(128, dim, 4, 2, 1),              # 4x4
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):                            # (B,1,H,W)
        h = self.net(img)                              # (B,dim,4,4)
        return self.norm(h.flatten(2).transpose(1, 2))  # (B,16,dim)


class ImageCrossAttention(nn.Module):
    """The second, image-only cross-attention of a decoupled block.

    The output projection starts at zero, so the adapted model is bit-for-bit
    the original at step 0 — the same guarantee ControlNet's zero convolutions
    (project 31) and LoRA's zero `B` (project 34) give, for the same reason: a
    new side path must not damage a model that already works.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = 1.0                     # inference-time strength dial

    def forward(self, x, ref):
        if ref is None or self.scale == 0.0:
            return 0.0
        B, N, D = x.shape
        M = ref.shape[1]
        xn = self.norm(x)
        q = self.q(xn).view(B, N, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ref).view(B, M, 2, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.scale * self.proj(out.transpose(1, 2).reshape(B, N, D))


class IPAdapterDiT(nn.Module):
    """Project 30's frozen model plus a trainable image branch."""

    def __init__(self, arm="t5"):
        super().__init__()
        base, _ = LL.fresh_base(arm)
        base.requires_grad_(False)
        self.base = base
        self.dim, self.heads = base.dim, base.heads
        self.ref_enc = RefEncoder(self.dim)
        self.img_attn = nn.ModuleList(
            [ImageCrossAttention(self.dim, self.heads)
             for _ in base.blocks])
        self.patch, self.in_ch = base.patch, base.in_ch

    def trainable(self):
        return list(self.ref_enc.parameters()) + \
            list(self.img_attn.parameters())

    def context(self, text):
        return self.base.context(text)

    def set_scale(self, s):
        for m in self.img_attn:
            m.scale = s

    def forward(self, x, t, text, ctx=None, ref=None):
        b = self.base
        tok, grid = L.patchify(x, b.patch)
        v = b.embed(tok)
        if ctx is None:
            ctx = b.context(text)
        ctx_seq, mask = ctx
        pooled = (ctx_seq * mask[..., None]).sum(1) \
            / mask.sum(1, keepdim=True).clamp(min=1.0)
        c = b.tmlp(L.timestep_embedding(t, b.dim)) \
            + b.pool(b.pool_norm(pooled))
        rope = b.rope_for(grid)
        rtok = None if ref is None else self.ref_enc(ref)
        for blk, ia in zip(b.blocks, self.img_attn):
            sh_a, sc_a, g_a, sh_m, sc_m, g_m = \
                blk.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
            v = v + g_a * blk.attn(blk.n1(v) * (1 + sc_a) + sh_a, rope)
            v = v + blk.cross(blk.nc(v), ctx_seq, mask) + ia(v, rtok)
            v = v + g_m * blk.mlp(blk.n2(v) * (1 + sc_m) + sh_m)
        sh, sc = b.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = b.head(b.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, b.patch, grid, b.in_ch)


def build_lora_model(r=4, alpha=None, targets=None, arm="t5"):
    """Project 34's LoRA injection, but on THIS phase's longer-trained base.

    `lora_lib.build_lora_model` loads project 30's checkpoint.  Every other arm
    here uses project 35's continued-training checkpoint, and comparing arms
    that sit on two different base models would measure the bases, not the
    adapters.
    """
    model, _ = LL.fresh_base(arm)
    model.requires_grad_(False)
    names = LR.inject_lora(model, r=r, alpha=alpha,
                           targets=targets or LR.ATTN)
    return model, names


def load_ip(path=None):
    p = path or (CK / "ip.pt")
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage ip`")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = IPAdapterDiT()
    m.ref_enc.load_state_dict(ck["ref_enc"])
    m.img_attn.load_state_dict(ck["img_attn"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, text, null, shape, ref=None, steps=STEPS, cfg=CFG,
           generator=None, drop_ref_in_null=False, scale=None):
    """Rectified-flow sampling that also works for the plain base model.

    `drop_ref_in_null` decides what the unconditional branch of
    [classifier-free guidance](/shared/glossary/#cfg-classifier-free-guidance)
    is allowed to see, and it matters more than it looks.  Guidance computes

        v = v_null + cfg * (v_prompt - v_null)

    and only *amplifies whatever differs between the two branches*.  If the
    reference image is present in both, it cancels out of the difference and
    survives at strength 1 while the text is amplified `cfg` times — so a
    guidance scale of 3 quietly makes the prompt three times louder than the
    character.  Dropping the reference from the null branch as well puts the
    identity inside the amplified term.  The `dial` stage measures both.
    """
    flow = FL.RectifiedFlow()
    if scale is not None and hasattr(model, "set_scale"):
        model.set_scale(scale)
    ctx = model.context(text)
    ctx_n = model.context(null)
    kw = {} if ref is None else {"ref": ref}
    kw_n = {} if (ref is None or drop_ref_in_null) else {"ref": ref}
    if ref is not None and drop_ref_in_null:
        kw_n = {"ref": torch.full_like(ref, -1.0)}   # the "no reference" frame
    x = torch.randn(shape, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(shape[0]) * flow.T_SCALE
        v = model(x, t, None, ctx=ctx, **kw)
        if cfg != 1.0:
            v_n = model(x, t, None, ctx=ctx_n, **kw_n)
            v = v_n + cfg * (v - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
    if scale is not None and hasattr(model, "set_scale"):
        model.set_scale(1.0)
    return x


# ---------------------------------------------------------------------------
# measuring identity
# ---------------------------------------------------------------------------

def glyph_of(clips, frames=(0, 5, 10, 15)):
    """Average centred glyph of a clip — one 28x28 picture of the character."""
    cr = LL.glyph_crops(clips)
    return cr[:, list(frames)].mean(1)


def identity_distance(clips, ref_glyph, frames=(0, 5, 10, 15)):
    return LL.glyph_distance(glyph_of(clips, frames), ref_glyph)


__all__ = ["ARMS", "SHOTS", "character_sprites", "build_cache", "load_cache",
           "build_lora_model",
           "RefEncoder", "ImageCrossAttention", "IPAdapterDiT", "load_ip",
           "sample", "glyph_of", "identity_distance", "LL", "T", "L", "LR",
           "FL"]
