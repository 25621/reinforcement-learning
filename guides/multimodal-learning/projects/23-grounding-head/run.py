"""Grounding head: teach a VLM to say *where*, using coordinate tokens.

The scene is two coloured shapes on a 224x224 canvas. The question names one of
them; the answer is its bounding box. Six arms:

    bins          a real grounding head -- 1 `<box>` token + 2x32 coordinate
                  tokens that we add to the model ourselves (new input
                  embeddings AND new output-logit rows), so a box costs 5 tokens
    digits        the same boxes written as ordinary text ("0.12 0.30 0.44
                  0.61"), which needs no new vocabulary but costs ~4x the tokens
    blind         the `bins` arm with the image removed -- the control that says
                  how much of the score is "guess the average box"
    bins-tuned    `bins` plus the last 8 LLM blocks unfrozen at a 20x smaller
                  learning rate (project 21's stage-2 recipe)
    digits-tuned  the same for the text format
    bins-pos      `bins` plus one learned vector per patch slot, so an image
                  token can say *where in the picture* it came from

Stages
    data    render scenes, encode once with frozen CLIP        (~2 min)
    train   one arm at a time                                  (~5 min each)
    plot    figures

Requires project 20's vlm_lib.py via sys.path.
"""

import argparse
import json
import re
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
SIZE = 224
BINS = 32
N_IMAGES = 2600
N_VAL = 300
COLORS = {"red": (200, 40, 40), "green": (40, 150, 60), "blue": (50, 80, 200),
          "yellow": (220, 190, 40), "purple": (140, 60, 170), "orange": (230, 130, 30)}
SHAPES = ["circle", "square", "triangle"]
# arm name -> (answer format, LLM blocks to unfreeze, learned patch-position
# embeddings on the image tokens)
ARMS = {"bins": ("bins", 0, False), "digits": ("digits", 0, False),
        "blind": ("bins", 0, False), "bins-tuned": ("bins", 8, False),
        "digits-tuned": ("digits", 8, False), "bins-pos": ("bins", 0, True)}

# the extra vocabulary: <box>, <x00>..<x31>, <y00>..<y31>
COORD_TOKENS = ["<box>"] + [f"<x{i:02d}>" for i in range(BINS)] + [f"<y{i:02d}>" for i in range(BINS)]
BOX, X0, Y0 = 0, 1, 1 + BINS


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def render_scene(rng):
    img = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    picks = rng.permutation(len(COLORS))[:2]
    shape_ids = rng.permutation(len(SHAPES))[:2]
    objs = []
    for k in range(2):
        cname = list(COLORS)[picks[k]]
        shape = SHAPES[shape_ids[k]]
        r = int(rng.integers(18, 34))
        for _ in range(40):                       # keep the two shapes apart
            cx = int(rng.integers(r + 2, SIZE - r - 2))
            cy = int(rng.integers(r + 2, SIZE - r - 2))
            if all(abs(cx - o["cx"]) > r + o["r"] + 6 or abs(cy - o["cy"]) > r + o["r"] + 6
                   for o in objs):
                break
        if shape == "circle":
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        elif shape == "square":
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        else:
            mask = (np.abs(xx - cx) <= (yy - cy + r) / 2) & (yy >= cy - r) & (yy <= cy + r)
        img[mask] = COLORS[cname]
        ys, xs = np.where(mask)
        objs.append(dict(color=cname, shape=shape, cx=cx, cy=cy, r=r,
                         box=[float(xs.min()) / SIZE, float(ys.min()) / SIZE,
                              float(xs.max() + 1) / SIZE, float(ys.max() + 1) / SIZE]))
    return img, objs


def stage_data(args):
    DATA.mkdir(exist_ok=True)
    if (DATA / "feats.npy").exists():
        print("cache exists")
        return
    rng = np.random.default_rng(0)
    tower = V.clip_vision()
    n = args.n
    feats = np.zeros((n, V.CLIP_TOKENS, V.CLIP_DIM), dtype=np.float16)
    meta, thumbs = [], []
    t0 = time.time()
    for start in range(0, n, 100):
        buf = []
        for i in range(start, min(start + 100, n)):
            img, objs = render_scene(rng)
            buf.append(img)
            meta.append(objs)
            if i < 6:
                thumbs.append(img)
        feats[start:start + len(buf)] = V.encode_views(tower, np.stack(buf), layers=(-2,))[-2]
        if (start // 100) % 5 == 0:
            print(f"    encoded {start + len(buf)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    np.save(DATA / "thumbs.npy", np.stack(thumbs))
    (DATA / "meta.json").write_text(json.dumps(meta))
    np.save(DATA / "feats.npy", feats)
    print(f"cache done in {time.time() - t0:.0f}s")


class SceneData:
    def __init__(self, n_val=N_VAL):
        self.feats = np.load(DATA / "feats.npy", mmap_mode="r")
        self.meta = json.loads((DATA / "meta.json").read_text())
        n = len(self.meta)
        n_val = min(n_val, n // 5)              # so a tiny smoke-test cache still splits
        self.val_ids = np.arange(n_val)
        self.train_ids = np.arange(n_val, n)

    def question(self, i, which):
        o = self.meta[i][which]
        return f"Where is the {o['color']} {o['shape']}?", o["box"]

    def image_tokens(self, ids):
        return torch.from_numpy(np.asarray(self.feats[np.asarray(ids)], dtype=np.float32))


# ---------------------------------------------------------------------------
# coordinates <-> tokens
# ---------------------------------------------------------------------------
def quantize(v, bins=BINS):
    """Map a coordinate in [0, 1] to one of `bins` evenly spaced bin centres."""
    return int(np.clip(round(v * (bins - 1)), 0, bins - 1))


def dequantize(b, bins=BINS):
    return b / (bins - 1)


def box_to_ids(box):
    x1, y1, x2, y2 = box
    return [BOX, X0 + quantize(x1), Y0 + quantize(y1), X0 + quantize(x2), Y0 + quantize(y2)]


def ids_to_box(ids):
    """Parse a generated coordinate-token sequence; None if it is malformed."""
    if len(ids) < 5 or ids[0] != BOX:
        return None
    try:
        x1, y1, x2, y2 = ids[1], ids[2], ids[3], ids[4]
        if not (X0 <= x1 < X0 + BINS and Y0 <= y1 < Y0 + BINS
                and X0 <= x2 < X0 + BINS and Y0 <= y2 < Y0 + BINS):
            return None
        return [dequantize(x1 - X0), dequantize(y1 - Y0),
                dequantize(x2 - X0), dequantize(y2 - Y0)]
    except Exception:
        return None


def box_to_text(box):
    return " ".join(f"{v:.2f}" for v in box)


def text_to_box(s):
    nums = re.findall(r"\d\.\d+", s)
    if len(nums) < 4:
        return None
    return [float(v) for v in nums[:4]]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def quantization_ceiling(data, bins_list=(4, 8, 16, 32, 64)):
    """The best IoU any binned model could reach: score the ground-truth box
    against its own quantized version. This is the price of the vocabulary."""
    out = {}
    for bins in bins_list:
        s = []
        for i in data.val_ids:
            for w in (0, 1):
                b = data.meta[i][w]["box"]
                q = [dequantize(quantize(v, bins), bins) for v in b]
                s.append(iou(b, q))
        out[bins] = float(np.mean(s))
    return out


# ---------------------------------------------------------------------------
# the grounding head: new input embeddings + new output rows
# ---------------------------------------------------------------------------
class GroundingVLM(nn.Module):
    """TinyVLM plus a small vocabulary the base model never had.

    Two halves, both living in one (65, d) parameter:
      input side   the embedding row a coordinate token turns into
      output side  the logit row that scores that token
    Using the *same* matrix for both is weight tying, exactly what the base
    model does with its own 49,152 words.
    """

    def __init__(self, llm, projector, n_extra=len(COORD_TOKENS), pos_tokens=0):
        super().__init__()
        self.vlm = V.TinyVLM(llm, projector)
        d = llm.config.hidden_size
        self.rows = nn.Parameter(torch.randn(n_extra, d) * 0.02)
        # one learned vector per patch slot: "I am the cell at row r, column c".
        # The projector applies the *same* map to every patch, so without this
        # nothing in an image token says where in the picture it came from.
        self.pos = nn.Parameter(torch.randn(pos_tokens, d) * 0.02) if pos_tokens else None
        self.base_vocab = llm.get_input_embeddings().weight.shape[0]

    def embed(self, batch, feats):
        """Word embeddings, with image slots and coordinate slots filled in."""
        ids = batch.ids.clamp(max=self.base_vocab - 1)
        emb = self.vlm.llm.get_input_embeddings()(ids)
        if feats is not None:
            vis = self.vlm.projector(feats)
            if self.pos is not None:
                vis = vis + self.pos.unsqueeze(0)
                rms = vis.pow(2).mean().sqrt().clamp(min=1e-6)
                vis = vis * (self.vlm.projector.out_rms / rms)
            vis = vis.reshape(-1, emb.shape[-1])
            emb = emb.masked_scatter(batch.image_mask.unsqueeze(-1), vis.to(emb.dtype))
        extra = batch.ids >= self.base_vocab
        if extra.any():
            rows = self.rows * (self.vlm.projector.out_rms
                                / self.rows.pow(2).mean().sqrt().clamp(min=1e-6))
            emb = emb.masked_scatter(extra.unsqueeze(-1),
                                     rows[(batch.ids[extra] - self.base_vocab)])
        return emb

    def logits(self, h):
        return torch.cat([self.vlm.head(h), h @ self.rows.T], -1)

    def forward(self, batch, feats):
        emb = self.embed(batch, feats)
        h = self.vlm.body(inputs_embeds=emb, attention_mask=batch.attn,
                          use_cache=False)[0]
        tgt = batch.labels[:, 1:]
        keep = tgt.reshape(-1) != -100
        h = h[:, :-1].reshape(-1, h.shape[-1])[keep]
        return F.cross_entropy(self.logits(h).float(), tgt.reshape(-1)[keep])

    @torch.no_grad()
    def greedy(self, batch, feats, max_new=8):
        """Decode with a KV cache; returns the generated id list per row."""
        emb = self.embed(batch, feats)
        past, outs = None, [[] for _ in range(emb.shape[0])]
        done = [False] * emb.shape[0]
        for _ in range(max_new):
            res = self.vlm.body(inputs_embeds=emb, past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = self.logits(res[0][:, -1]).float().argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    outs[i].append(t)
            emb = self.embed_step(nxt)
        return outs

    def embed_step(self, ids):
        base = self.vlm.llm.get_input_embeddings()(ids.clamp(max=self.base_vocab - 1))
        extra = ids >= self.base_vocab
        if extra.any():
            rows = self.rows * (self.vlm.projector.out_rms
                                / self.rows.pow(2).mean().sqrt().clamp(min=1e-6))
            base = base.masked_scatter(extra.unsqueeze(-1),
                                       rows[(ids[extra] - self.base_vocab)])
        return base.unsqueeze(1)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
QUESTION_SUFFIX = ""


def make_ground_batch(tok, model, questions, boxes, arm, n_img, prompt_only=False):
    """Build ids where the answer is either coordinate tokens (ids past the base
    vocabulary) or ordinary text."""
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    img_id = tok.convert_tokens_to_ids(V.IMAGE_TOKEN)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    with_image = arm != "blind"
    fmt = ARMS[arm][0]
    seqs, labs = [], []
    for q, box in zip(questions, boxes):
        prompt = tok.encode(V.chat(q, None, with_image, n_img), add_special_tokens=False)
        if prompt_only:
            seqs.append(prompt)
            labs.append([-100] * len(prompt))
            continue
        if fmt == "digits":
            ans = tok.encode(box_to_text(box), add_special_tokens=False) + [end_id]
        else:
            ans = [model.base_vocab + t for t in box_to_ids(box)] + [end_id]
        seqs.append(prompt + ans)
        labs.append([-100] * len(prompt) + ans)
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), pad_id, dtype=np.int64)
    labels = np.full((len(seqs), T), -100, dtype=np.int64)
    attn = np.zeros((len(seqs), T), dtype=np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = s
        labels[i, :len(l)] = l
        attn[i, :len(s)] = 1
    ids = torch.from_numpy(ids)
    return V.Batch(ids, ids == img_id, torch.from_numpy(labels), torch.from_numpy(attn))


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tok, data, arm, n_img, n=120, bs=12):
    n = min(n, len(data.val_ids))
    fmt = ARMS[arm][0]
    ious, valid, preds = [], 0, []
    for i in range(0, n, bs):
        ids = data.val_ids[i:i + bs]
        qs, boxes = zip(*[data.question(j, j % 2) for j in ids])
        b = make_ground_batch(tok, model, qs, boxes, arm, n_img, prompt_only=True)
        feats = data.image_tokens(ids) if arm != "blind" else None
        outs = model.greedy(b, feats, max_new=6 if fmt != "digits" else 16)
        for k, o in enumerate(outs):
            if fmt == "digits":
                txt = tok.decode([t for t in o if t < model.base_vocab])
                pred = text_to_box(txt)
            else:
                pred = ids_to_box([t - model.base_vocab for t in o])
            valid += pred is not None
            ious.append(iou(pred, boxes[k]))
            if i == 0 and k < 4:
                preds.append(dict(q=qs[k], truth=[round(v, 3) for v in boxes[k]],
                                  pred=None if pred is None else [round(v, 3) for v in pred],
                                  iou=round(ious[-1], 3)))
    ious = np.array(ious)
    return dict(iou=float(ious.mean()), acc50=float((ious > .5).mean()),
                acc25=float((ious > .25).mean()), valid=valid / len(ious),
                n_eval=len(ious), examples=preds)


def train_arm(arm, data, steps, bs, lr, unfreeze=None, seed=0):
    unfreeze = ARMS[arm][1] if unfreeze is None else unfreeze
    tok, llm = V.load_llm()
    proj = V.Projector("mlp2", V.CLIP_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))
    torch.manual_seed(seed)
    model = GroundingVLM(llm, proj, pos_tokens=V.CLIP_TOKENS if ARMS[arm][2] else 0)
    params = list(proj.parameters()) + [model.rows]
    if model.pos is not None:
        params.append(model.pos)
    if unfreeze:
        for blk in llm.model.layers[-unfreeze:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                params.append(p)
    # a pretrained network's own weights need a far smaller step than a
    # from-scratch projector: same reasoning as project 21's stage-2 arm
    head = list(proj.parameters()) + [model.rows]
    if model.pos is not None:
        head.append(model.pos)
    groups = [dict(params=head, lr=lr)]
    if unfreeze:
        groups.append(dict(params=[p for blk in llm.model.layers[-unfreeze:]
                                   for p in blk.parameters()], lr=lr * 0.05))
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    n_img = proj.n_tokens()
    curve, t0 = [], time.time()
    for step in range(steps):
        ids = rng.choice(data.train_ids, bs, replace=False)
        qs, boxes = zip(*[data.question(i, int(rng.integers(0, 2))) for i in ids])
        b = make_ground_batch(tok, model, qs, boxes, arm, n_img)
        feats = data.image_tokens(ids) if arm != "blind" else None
        for g, base in zip(opt.param_groups, [lr, lr * 0.05]):
            g["lr"] = V.cosine_lr(step, steps, base)
        loss = model(b, feats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        curve.append((step, float(loss.detach())))
        if (step + 1) % 25 == 0:
            print(f"  [{arm}] {step + 1}/{steps} loss {float(loss.detach()):.4f} "
                  f"({(time.time() - t0) / (step + 1) * 1000:.0f} ms/step)", flush=True)
    res = evaluate(model, tok, data, arm, n_img)
    ans_tokens = 5 if ARMS[arm][0] != "digits" else len(
        tok.encode(box_to_text([0.12, 0.3, 0.44, 0.61]), add_special_tokens=False))
    res.update(arm=arm, steps=steps, bs=bs, lr=lr, unfreeze=unfreeze,
               patch_pos=ARMS[arm][2],
               answer_tokens=ans_tokens, extra_params=int(model.rows.numel()),
               trainable=sum(p.numel() for p in params),
               ms_per_step=(time.time() - t0) / steps * 1000)
    print(f"  [{arm}] IoU {res['iou']:.3f}  [email protected] {res['acc50']:.3f}  "
          f"valid {res['valid']:.3f}", flush=True)
    return res, curve


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    data = SceneData()
    rows, curves = [], {}
    for arm in args.arms:
        r, c = train_arm(arm, data, args.steps, args.bs, args.lr, args.unfreeze)
        rows.append(r)
        curves[arm] = c
    old = json.loads((OUT / "ground.json").read_text()) if (OUT / "ground.json").exists() else []
    keep = [r for r in old if r["arm"] not in args.arms]
    (OUT / "ground.json").write_text(json.dumps(keep + rows, indent=1))
    (OUT / "ceiling.json").write_text(json.dumps(quantization_ceiling(data), indent=1))
    prev = (OUT / "curves.csv").read_text().splitlines()[1:] if (OUT / "curves.csv").exists() else []
    prev = [p for p in prev if p.split(",")[0] not in args.arms]
    lines = [f"{a},{s},{l:.4f}" for a, v in curves.items() for s, l in v]
    (OUT / "curves.csv").write_text("\n".join(["arm,step,loss"] + prev + lines) + "\n")


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    res = {r["arm"]: r for r in json.loads((OUT / "ground.json").read_text())}
    ceil = json.loads((OUT / "ceiling.json").read_text())
    arms = [a for a in ARMS if a in res]
    curves = {}
    for line in (OUT / "curves.csv").read_text().splitlines()[1:]:
        a, s, l = line.split(",")
        curves.setdefault(a, []).append((int(s), float(l)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for a in arms:
        c = np.array(curves[a])
        k = 10
        ax.plot(c[k - 1:, 0], np.convolve(c[:, 1], np.ones(k) / k, "valid"), label=a)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title("Learning to emit boxes")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[1]
    x = np.arange(len(arms))
    ax.bar(x - .2, [res[a]["iou"] for a in arms], .4, label="mean IoU")
    ax.bar(x + .2, [res[a]["acc50"] for a in arms], .4, label="[email protected]")
    ax.axhline(ceil[str(BINS)], ls="--", c="k", lw=1,
               label=f"{BINS}-bin ceiling ({ceil[str(BINS)]:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_title("Localisation quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    bins = sorted(int(k) for k in ceil)
    ax.plot(bins, [ceil[str(b)] for b in bins], "o-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bins per axis (vocabulary size = 2 x bins + 1)")
    ax.set_ylabel("best possible IoU")
    ax.set_title("The price of quantizing coordinates")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "grounding.png", dpi=130)

    # qualitative panel: show the best arm's boxes
    thumbs = np.load(DATA / "thumbs.npy")
    best = max(arms, key=lambda a: res[a]["iou"])
    ex = res[best]["examples"]
    fig, axes = plt.subplots(1, min(4, len(ex)), figsize=(3.1 * min(4, len(ex)), 3.6))
    for ax, e, im in zip(np.atleast_1d(axes), ex, thumbs):
        ax.imshow(im)
        for box, col, lab in ((e["truth"], "lime", "truth"), (e["pred"], "red", "predicted")):
            if box:
                ax.add_patch(Rectangle((box[0] * SIZE, box[1] * SIZE),
                                       (box[2] - box[0]) * SIZE, (box[3] - box[1]) * SIZE,
                                       fill=False, ec=col, lw=2, label=lab))
        ax.set_title(f"{e['q']}\n{best}: IoU {e['iou']}", fontsize=8)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "boxes.png", dpi=130)
    print("wrote figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "train", "plot"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--n", type=int, default=N_IMAGES)
    p.add_argument("--steps", type=int, default=260)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--unfreeze", type=int, default=None)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    {"data": stage_data, "train": stage_train, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
