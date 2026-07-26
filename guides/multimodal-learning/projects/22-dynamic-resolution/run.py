"""Dynamic resolution (AnyRes): tile the image instead of squashing it.

We build a small "document" world -- a big coloured shape plus a 4-digit code
printed in small glyphs on a 448x448 page -- and ask two questions about it:

    "what colour is the shape?"   survives any amount of downscaling
    "what is the code?"           only survives if the small text stays sharp

Then we feed the *same* frozen CLIP ViT-B/32 four different views of the page
and compare what a read-out can recover:

    squash        the whole page resized to 224x224          49 image tokens
    crop          one native-resolution 224 quadrant         49
    anyres        all four native quadrants                 196
    anyres+thumb  four quadrants plus the squashed page     245

Stages
    data    render the pages and encode every view with frozen CLIP (~5 min)
    train   train the read-out once per condition            (~3 min)
    cost    what the extra tokens cost a real 135M LLM       (~1 min)
    plot    figures

Requires project 20's vlm_lib.py (frozen CLIP loader, chat helpers) via sys.path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20-llava-from-scratch"))
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
DATA = HERE / "data"
PAGE = 448                 # the page is 4x the area CLIP accepts at once
TILE = 224                 # CLIP's input size
N_IMAGES = 1600
N_VAL = 300
CODE_LEN = 4
GLYPH = 4                  # bitmap font is 5x7 dots; each dot is GLYPH px
COLORS = {"red": (200, 40, 40), "green": (40, 150, 60), "blue": (50, 80, 200),
          "yellow": (220, 190, 40), "purple": (140, 60, 170), "orange": (230, 130, 30)}
SHAPES = ["circle", "square", "triangle"]
CONDITIONS = ["squash", "crop", "anyres", "anyres+thumb"]

# 5x7 bitmap font, digits only. Hand-typed so the project needs no font file.
FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


# ---------------------------------------------------------------------------
# the AnyRes grid chooser (this is the actual LLaVA-1.6 / InternVL rule)
# ---------------------------------------------------------------------------
def select_grid(w, h, max_tiles=6, tile=TILE):
    """Pick the tile grid (cols, rows) for an image of size w x h.

    This is LLaVA-1.6's `select_best_resolution` rule. For every allowed grid
    (1x1, 1x2, 2x1, 2x2, 1x3 ... up to `max_tiles` tiles):

      * scale the image so it fits inside the grid,
      * count the *effective* pixels -- capped at the image's own pixel count,
        because blowing a small image up to fill a big grid adds no detail,
      * count the wasted grid area (padding).

    Then keep the most effective pixels, breaking ties by least waste. That is
    what makes a wide receipt get wide tiling instead of being squeezed into a
    square, and what stops a small photo from being upscaled into four tiles.
    """
    best, best_key = (1, 1), None
    for cols in range(1, max_tiles + 1):
        for rows in range(1, max_tiles + 1):
            if cols * rows > max_tiles:
                continue
            gw, gh = cols * tile, rows * tile
            s = min(gw / w, gh / h)
            effective = min((w * s) * (h * s), w * h)
            wasted = gw * gh - effective
            key = (-effective, wasted, cols * rows)
            if best_key is None or key < best_key:
                best, best_key = (cols, rows), key
    return best


def grid_table():
    rows = []
    for (w, h) in [(448, 448), (672, 224), (224, 672), (896, 224), (336, 448), (224, 224)]:
        cols, r = select_grid(w, h)
        rows.append(dict(size=f"{w}x{h}", grid=f"{cols}x{r}", tiles=cols * r,
                         tokens=(cols * r + 1) * V.CLIP_TOKENS))
    return rows


# ---------------------------------------------------------------------------
# rendering the pages
# ---------------------------------------------------------------------------
def draw_glyph(canvas, ch, x, y, scale=GLYPH, color=(20, 20, 20)):
    bits = np.array([[int(c) for c in row] for row in FONT[ch]], dtype=bool)
    big = np.kron(bits, np.ones((scale, scale), dtype=bool))
    h, w = big.shape
    patch = canvas[y:y + h, x:x + w]
    patch[big] = color
    return w


def render_page(rng):
    """One page: background + a big coloured shape + a small 4-digit code."""
    page = np.full((PAGE, PAGE, 3), 245, dtype=np.uint8)
    page += rng.integers(-6, 7, page.shape, dtype=np.int16).clip(-6, 6).astype(np.uint8)

    shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
    cname = list(COLORS)[int(rng.integers(0, len(COLORS)))]
    color = COLORS[cname]
    r = int(rng.integers(55, 85))
    cx, cy = (int(rng.integers(r + 5, PAGE - r - 5)), int(rng.integers(r + 5, PAGE - r - 5)))
    yy, xx = np.mgrid[0:PAGE, 0:PAGE]
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:
        mask = (np.abs(xx - cx) <= (yy - cy + r) / 2) & (yy >= cy - r) & (yy <= cy + r)
    page[mask] = color

    code = "".join(str(int(d)) for d in rng.integers(0, 10, CODE_LEN))
    gw, gh = 5 * GLYPH, 7 * GLYPH
    total_w = CODE_LEN * (gw + GLYPH)
    # keep the code inside one quadrant so tiling never cuts it in half
    q = int(rng.integers(0, 4))
    qx, qy = (q % 2) * TILE, (q // 2) * TILE
    x = qx + int(rng.integers(8, TILE - total_w - 8))
    y = qy + int(rng.integers(8, TILE - gh - 8))
    page[y - 3:y + gh + 3, x - 3:x + total_w + 3] = 252     # a white text box
    for ch in code:
        x += draw_glyph(page, ch, x, y) + GLYPH
    return page, cname, shape, code, q


def views_of(page):
    """The five CLIP inputs we cache per page: the squashed whole page, then the
    four native-resolution quadrants."""
    from PIL import Image
    small = np.asarray(Image.fromarray(page).resize((TILE, TILE), Image.BICUBIC))
    tiles = [page[(q // 2) * TILE:(q // 2 + 1) * TILE, (q % 2) * TILE:(q % 2 + 1) * TILE]
             for q in range(4)]
    return np.stack([small] + tiles)


def stage_data(args):
    DATA.mkdir(exist_ok=True)
    if (DATA / "feats.npy").exists():
        print("cache exists")
        return
    rng = np.random.default_rng(0)
    tower = V.clip_vision()
    n = args.n
    feats = np.zeros((n, 5, V.CLIP_TOKENS, V.CLIP_DIM), dtype=np.float16)
    meta, pages = [], []
    t0 = time.time()
    for start in range(0, n, 50):
        buf = []
        for i in range(start, min(start + 50, n)):
            page, cname, shape, code, q = render_page(rng)
            buf.append(views_of(page))
            meta.append(dict(color=cname, shape=shape, code=code, quadrant=q))
            if i < 4:
                pages.append(page)
        got = V.encode_views(tower, np.concatenate(buf), layers=(-2,))[-2]
        feats[start:start + len(buf)] = got.reshape(len(buf), 5, V.CLIP_TOKENS, V.CLIP_DIM)
        if (start // 50) % 4 == 0:
            print(f"    encoded {start + len(buf)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    np.save(DATA / "pages.npy", np.stack(pages))
    (DATA / "meta.json").write_text(json.dumps(meta))
    np.save(DATA / "feats.npy", feats)
    print(f"cache done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# the read-out: a handful of queries that cross-attend to the image tokens
# ---------------------------------------------------------------------------
class ReadOut(nn.Module):
    """Six output slots (colour, shape, code digit 1..4), each read by its own
    learned query. This is a Q-Former with classification heads instead of a
    language model: cheap enough to train four times in one run, and identical
    for every condition, so any difference comes from the image tokens alone.
    """

    def __init__(self, in_dim=V.CLIP_DIM, d=192, heads=4, layers=2):
        super().__init__()
        self.slots = 2 + CODE_LEN
        self.norm = nn.LayerNorm(in_dim)
        self.kv = nn.Linear(in_dim, d)
        self.queries = nn.Parameter(torch.randn(self.slots, d) * 0.02)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "ln1": nn.LayerNorm(d),
            "x": nn.MultiheadAttention(d, heads, batch_first=True),
            "ln2": nn.LayerNorm(d),
            "s": nn.MultiheadAttention(d, heads, batch_first=True),
            "ln3": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, 3 * d), nn.GELU(), nn.Linear(3 * d, d)),
        }) for _ in range(layers)])
        self.out_ln = nn.LayerNorm(d)
        self.head_color = nn.Linear(d, len(COLORS))
        self.head_shape = nn.Linear(d, len(SHAPES))
        self.head_digit = nn.Linear(d, 10)

    def forward(self, feats):
        kv = self.kv(self.norm(feats))
        q = self.queries.unsqueeze(0).expand(feats.shape[0], -1, -1)
        for b in self.blocks:
            q = q + b["x"](b["ln1"](q), kv, kv, need_weights=False)[0]
            h = b["ln2"](q)
            q = q + b["s"](h, h, h, need_weights=False)[0]
            q = q + b["ff"](b["ln3"](q))
        q = self.out_ln(q)
        return (self.head_color(q[:, 0]), self.head_shape(q[:, 1]),
                self.head_digit(q[:, 2:]))


def view_slice(cond):
    """Which cached views a condition feeds to the read-out."""
    return {"squash": [0], "crop": [1], "anyres": [1, 2, 3, 4],
            "anyres+thumb": [0, 1, 2, 3, 4]}[cond]


class DocData:
    def __init__(self, n_val=N_VAL):
        self.feats = np.load(DATA / "feats.npy", mmap_mode="r")
        self.meta = json.loads((DATA / "meta.json").read_text())
        n = len(self.meta)
        n_val = min(n_val, n // 5)              # so a tiny smoke-test cache still splits
        self.color = np.array([list(COLORS).index(m["color"]) for m in self.meta])
        self.shape = np.array([SHAPES.index(m["shape"]) for m in self.meta])
        self.code = np.array([[int(c) for c in m["code"]] for m in self.meta])
        self.quad = np.array([m["quadrant"] for m in self.meta])
        self.val_ids = np.arange(n_val)
        self.train_ids = np.arange(n_val, n)

    def batch(self, ids, cond):
        v = view_slice(cond)
        f = np.asarray(self.feats[np.asarray(ids)][:, v], dtype=np.float32)
        f = f.reshape(len(ids), -1, V.CLIP_DIM)
        return (torch.from_numpy(f), torch.from_numpy(self.color[ids]),
                torch.from_numpy(self.shape[ids]), torch.from_numpy(self.code[ids]))


def evaluate(model, data, cond, bs=64):
    model.eval()
    hits = dict(color=0, shape=0, digit=0, exact=0)
    by_quad = np.zeros((4, 2))
    with torch.no_grad():
        for i in range(0, len(data.val_ids), bs):
            ids = data.val_ids[i:i + bs]
            f, c, s, code = data.batch(ids, cond)
            pc, ps, pd = model(f)
            hits["color"] += int((pc.argmax(-1) == c).sum())
            hits["shape"] += int((ps.argmax(-1) == s).sum())
            dig = pd.argmax(-1) == code
            hits["digit"] += int(dig.sum())
            ex = dig.all(-1)
            hits["exact"] += int(ex.sum())
            for q in range(4):
                m = data.quad[ids] == q
                by_quad[q] += [int(ex.numpy()[m].sum()), int(m.sum())]
    model.train()
    n = len(data.val_ids)
    return dict(color=hits["color"] / n, shape=hits["shape"] / n,
                digit=hits["digit"] / (n * CODE_LEN), exact=hits["exact"] / n,
                by_quadrant=[float(a / max(b, 1)) for a, b in by_quad])


def train_cond(cond, data, steps, bs, lr, seed=0):
    torch.manual_seed(seed)
    model = ReadOut()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    n_tok = len(view_slice(cond)) * V.CLIP_TOKENS
    curve, t0 = [], time.time()
    for step in range(steps):
        ids = rng.choice(data.train_ids, bs, replace=False)
        f, c, s, code = data.batch(ids, cond)
        pc, ps, pd = model(f)
        loss = (F.cross_entropy(pc, c) + F.cross_entropy(ps, s)
                + F.cross_entropy(pd.reshape(-1, 10), code.reshape(-1)))
        for g in opt.param_groups:
            g["lr"] = V.cosine_lr(step, steps, lr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            curve.append((step + 1, float(loss.detach())))
            print(f"  [{cond}] {step + 1}/{steps} loss {float(loss.detach()):.3f} "
                  f"({(time.time() - t0) / (step + 1) * 1000:.0f} ms/step)", flush=True)
    res = evaluate(model, data, cond)
    res.update(cond=cond, tokens=n_tok, steps=steps,
               params=sum(p.numel() for p in model.parameters()),
               ms_per_step=(time.time() - t0) / steps * 1000)
    print(f"  [{cond}] tokens={n_tok} colour {res['color']:.3f} "
          f"code-exact {res['exact']:.3f} per-digit {res['digit']:.3f}", flush=True)
    return res, curve


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    data = DocData()
    rows, curves = [], {}
    for cond in args.conds:
        r, c = train_cond(cond, data, args.steps, args.bs, args.lr)
        rows.append(r)
        curves[cond] = c
    old = json.loads((OUT / "readout.json").read_text()) if (OUT / "readout.json").exists() else []
    keep = [r for r in old if r["cond"] not in args.conds]
    (OUT / "readout.json").write_text(json.dumps(keep + rows, indent=1))
    lines = ["cond,step,loss"] + [f"{c},{s},{l:.4f}" for c, v in curves.items() for s, l in v]
    (OUT / "curves.csv").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# what the extra tokens cost the real LLM
# ---------------------------------------------------------------------------
def stage_cost(args):
    OUT.mkdir(exist_ok=True)
    tok, llm = V.load_llm()
    cfg = llm.config
    kv_per_token = (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
                    * (cfg.hidden_size // cfg.num_attention_heads) * 4)
    rows = []
    proj = V.Projector("mlp2", V.CLIP_DIM, cfg.hidden_size, out_rms=V.embedding_rms(llm))
    vlm = V.TinyVLM(llm, proj)
    for cond in CONDITIONS:
        n_tok = len(view_slice(cond)) * V.CLIP_TOKENS
        feats = torch.randn(1, n_tok, V.CLIP_DIM)
        b = V.make_batch(tok, ["What is the code?"], ["1234"], n_img=n_tok)
        with torch.no_grad():
            vlm(b, feats)                                    # warm up
            runs = []
            for _ in range(9):        # median of 9: a single timing on a shared
                t0 = time.time()      # CPU can be 30% off
                vlm(b, feats)
                runs.append((time.time() - t0) * 1000)
            ms = float(np.median(runs))
        rows.append(dict(cond=cond, image_tokens=n_tok, seq_len=int(b.ids.shape[1]),
                         prefill_ms=ms, kv_kb=n_tok * kv_per_token / 1024,
                         clip_views=len(view_slice(cond))))
        print(f"  {cond:14s} {n_tok:4d} image tokens  prefill {ms:7.1f} ms  "
              f"KV {n_tok * kv_per_token / 1024:7.0f} KB", flush=True)
    (OUT / "cost.json").write_text(json.dumps(
        dict(kv_bytes_per_token=kv_per_token, rows=rows, grids=grid_table()), indent=1))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {r["cond"]: r for r in json.loads((OUT / "readout.json").read_text())}
    cost = json.loads((OUT / "cost.json").read_text())
    cost = {r["cond"]: r for r in cost["rows"]}
    conds = [c for c in CONDITIONS if c in res]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(conds))
    ax = axes[0]
    ax.bar(x - .2, [res[c]["color"] for c in conds], .4, label="colour of the big shape")
    ax.bar(x + .2, [res[c]["exact"] for c in conds], .4, label="4-digit code, exact")
    ax.axhline(1 / len(COLORS), ls="--", c="gray", lw=1)
    ax.axhline(1e-4, ls=":", c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{res[c]['tokens']} tok" for c in conds], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_title("Big object vs small text")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    ax.bar(x, [res[c]["digit"] for c in conds], .5, color="tab:purple")
    ax.axhline(0.1, ls="--", c="k", lw=1, label="chance (1 of 10)")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=8, rotation=15)
    ax.set_ylabel("per-digit accuracy")
    ax.set_title("Reading one digit")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    ax.plot([cost[c]["image_tokens"] for c in conds],
            [cost[c]["prefill_ms"] for c in conds], "o-")
    for c in conds:
        ax.annotate(c, (cost[c]["image_tokens"], cost[c]["prefill_ms"]),
                    fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("image tokens")
    ax.set_ylabel("prefill time (ms), SmolLM2-135M")
    ax.set_title("What the tokens cost the LLM")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "anyres.png", dpi=130)

    from PIL import Image
    pages = np.load(DATA / "pages.npy")
    meta = json.loads((DATA / "meta.json").read_text())
    page, q = pages[0], meta[0]["quadrant"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    axes[0].imshow(page)
    axes[0].set_title(f"the page, {PAGE}x{PAGE} (code: {meta[0]['code']})", fontsize=9)
    axes[0].axhline(TILE, c="w", lw=1)
    axes[0].axvline(TILE, c="w", lw=1)
    axes[1].imshow(np.asarray(Image.fromarray(page).resize((TILE, TILE), Image.BICUBIC)))
    axes[1].set_title(f"squash: {TILE}x{TILE}, {V.CLIP_TOKENS} tokens", fontsize=9)
    axes[2].imshow(page[(q // 2) * TILE:(q // 2 + 1) * TILE,
                        (q % 2) * TILE:(q % 2 + 1) * TILE])
    axes[2].set_title("the tile holding the code: native pixels", fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "page.png", dpi=85)
    print("wrote figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "train", "cost", "plot"])
    p.add_argument("--conds", nargs="+", default=CONDITIONS)
    p.add_argument("--n", type=int, default=N_IMAGES)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    {"data": stage_data, "train": stage_train, "cost": stage_cost,
     "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
