"""Project 18 -- Perceiver IO on the Phase-4 shapes world.

Perceiver IO is one architecture with three moving parts:

  encode   a small, FIXED set of learned latent vectors cross-attends to the
           input, however big and however shaped that input is
  process  self-attention runs among the latents only
  decode   output queries cross-attend to the processed latents, producing an
           output of whatever shape the task wants

Nothing in that pipeline assumes a grid, a sequence or a fixed length, which is
why the same code can take pixels, audio samples or tokens. The "IO" in the
name is the third part: earlier Perceiver could only emit one vector, and IO
added query-driven outputs.

What this project measures:

  1. cost vs input size N -- Perceiver is O(N x L), a plain Transformer O(N^2)
  2. the same model on two very different inputs:
       pixels   4,096 raw pixels, each with its colour and a Fourier code of
                where it is -- no patches, no grid, no convolution
       tokens   the 16 frozen patch tokens from project 15's encoder
     with one output query per question, so all five questions about a scene
     are answered in a single forward pass
  3. permutation invariance -- shuffle the input elements and the answers do
     not change, because position is a *feature* of each element, not its slot
  4. how accuracy moves with the size of the latent bottleneck

Usage:
    python3 run.py --stage scaling    # cost vs N (~1 min)
    python3 run.py --stage train      # both inputs + latent sweep (~9 min)
    python3 run.py --stage plot
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P15 = Path(__file__).resolve().parent.parent / "15-concat-vs-cross-attn"
sys.path.insert(0, str(P15))
import vqa_lib as V                                            # noqa: E402

torch.set_num_threads(12)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
D = 128
BANDS = 6
PIXEL_STEPS = 1800
TOKEN_STEPS = 2500
PIXEL_BATCH = 32
TOKEN_BATCH = 128
LATENT_SWEEP = [4, 16, 64]
TRAIN_SCENES = 8000
TEST_SCENES = 1000


# ---------------------------------------------------------------------------
# turning an image into an unordered set of elements
# ---------------------------------------------------------------------------
def fourier_positions(size=V.IMG, bands=BANDS):
    """Give every pixel a code of WHERE it is, so geometry survives having no
    order.

    Why sines and cosines at several frequencies rather than just (y, x): two
    raw coordinates change very slowly, and a network has to work hard to turn
    "0.51 versus 0.53" into "a different place". Doubling the frequency each
    band gives coarse-to-fine digits of the position -- the lowest band says
    which half of the image, the highest says which pixel. It is the same trick
    as a Transformer's sinusoidal positional embedding, and the same reason
    NeRF applies it to 3D coordinates.
    """
    ys, xs = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size),
                         indexing="ij")
    feats = [ys, xs]
    for b in range(bands):
        f = 2.0 ** b * np.pi
        for c in (ys, xs):
            feats += [np.sin(f * c), np.cos(f * c)]
    return torch.from_numpy(
        np.stack(feats, axis=-1).reshape(size * size, -1).astype(np.float32))


POS = fourier_positions()
PIXEL_DIM = 3 + POS.shape[1]


def as_set(px, pos):
    """(B,3,H,W) float image -> (B, H*W, 3 + pos_dim) unordered set."""
    B, C, H, W = px.shape
    rgb = px.reshape(B, C, H * W).transpose(1, 2)
    return torch.cat([rgb, pos[None].expand(B, -1, -1)], dim=-1)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class PerceiverIO(nn.Module):
    """The latents are a learned array of shape (n_latents, d): a fixed-size
    notebook the whole input gets written into. Everything expensive happens
    among the latents, so the depth of the model is decoupled from the size of
    the input -- that is the entire idea."""

    def __init__(self, n_latents=16, in_dim=PIXEL_DIM, d=D, heads=4, self_layers=6):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d) * 0.02)
        self.in_proj = nn.Linear(in_dim, d)
        self.encode = V.CrossAttention(d, heads)
        self.encode2 = V.CrossAttention(d, heads)
        self.enc_mlp = V.MLP(d)
        self.layers = nn.ModuleList()
        for _ in range(self_layers):
            self.layers += [V.SelfAttention(d, heads), V.MLP(d)]
        self.text = V.TextTower(d)
        self.decode = V.CrossAttention(d, heads)
        self.dec_mlp = V.MLP(d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(V.ANSWERS))

    def forward(self, inputs, tok):
        """inputs (B, N, in_dim) -- any N; tok (B, Q, Q_LEN) -- Q questions."""
        B, Q, _ = tok.shape
        x = self.in_proj(inputs)
        z = self.enc_mlp(self.encode(self.latents.expand(B, -1, -1), x))
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i == 1:                        # a second look at the raw input,
                z = self.encode2(z, x)        # as in the original Perceiver
        queries = self.text(tok.reshape(B * Q, -1))[1].reshape(B, Q, -1)
        return self.head(self.norm(self.dec_mlp(self.decode(queries, z))))


class DenseTransformer(nn.Module):
    """The O(N^2) control: plain self-attention over all N input elements."""

    def __init__(self, in_dim=PIXEL_DIM, d=D, heads=4, layers=6):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [V.SelfAttention(d, heads), V.MLP(d)]

    def forward(self, inputs):
        x = self.in_proj(inputs)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# scene-level data (all 5 questions of a scene travel together)
# ---------------------------------------------------------------------------
class SceneData:
    """`source='pixels'` builds the 4,096-element set on the fly;
    `source='tokens'` serves project 15's cached frozen patch tokens."""

    def __init__(self, n_scenes, seed, source, feats=None):
        self.images, shapes, colors, cells, yx = V.make_scenes(n_scenes, seed)
        self.tokens, self.answers = V.make_questions(shapes, colors, yx)
        self.source = source
        self.feats = feats

    def __len__(self):
        return len(self.images)

    def inputs(self, ids, pos=None):
        if self.source == "tokens":
            return torch.from_numpy(self.feats[ids].astype(np.float32))
        return as_set(V.pixels(self.images[ids]), POS if pos is None else pos)

    def batch(self, ids, pos=None):
        return (self.inputs(ids, pos), torch.from_numpy(self.tokens[ids]),
                torch.from_numpy(self.answers[ids]))


def train(model, data, steps, batch, lr=1e-3, seed=0, log_every=250):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05,
                            betas=(0.9, 0.98), eps=1e-6)
    hist = []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = V.cosine_lr(step, steps, lr)
        ids = rng.integers(0, len(data), size=batch)
        inp, tok, ans = data.batch(ids)
        logits = model(inp, tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ans.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {np.mean(hist[-50:]):.4f}", flush=True)
    return hist


@torch.no_grad()
def evaluate(model, data, batch=100, pos=None, perm=None):
    model.eval()
    hits = np.zeros((len(data), V.N_QUESTIONS), dtype=bool)
    for i in range(0, len(data), batch):
        ids = np.arange(i, min(i + batch, len(data)))
        inp, tok, ans = data.batch(ids, pos)
        if perm is not None:
            inp = inp[:, perm]
        hits[ids] = (model(inp, tok).argmax(-1) == ans).numpy()
    return float(hits.mean()), {q: float(hits[:, q].mean())
                                for q in range(V.N_QUESTIONS)}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_scaling(_args):
    """Forward-pass cost of Perceiver vs a dense Transformer as N grows."""
    OUT.mkdir(exist_ok=True)
    rows = []
    for size in (16, 32, 64, 96):
        n = size * size
        pos = fourier_positions(size)
        px = torch.randn(2, 3, size, size)
        inp = as_set(px, pos)
        tok = torch.ones(2, V.N_QUESTIONS, V.Q_LEN, dtype=torch.long)
        per = PerceiverIO(16, in_dim=inp.shape[-1])
        dense = DenseTransformer(in_dim=inp.shape[-1])
        with torch.no_grad():
            for name, fn in (("perceiver", lambda: per(inp, tok)),
                             ("dense", lambda: dense(inp))):
                fn()                                    # warm up
                t0 = time.time()
                for _ in range(2):
                    fn()
                ms = 1000 * (time.time() - t0) / 2 / 2
                rows.append(dict(model=name, side=size, n_inputs=n, ms_per_image=ms))
                print(f"  {name:10s} N={n:5d}  {ms:8.1f} ms/image", flush=True)
    with open(OUT / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT / "scaling.csv")


def _token_data():
    feats_tr = np.load(P15 / "data" / "train_feats.npy")
    feats_te = np.load(P15 / "data" / "test_feats.npy")
    return (SceneData(TRAIN_SCENES, 0, "tokens", feats_tr),
            SceneData(TEST_SCENES, 999, "tokens", feats_te))


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    rows, curves = [], {}
    jobs = args.only or ["pixels-16", "tokens-4", "tokens-16", "tokens-64"]

    pix_tr = pix_te = tok_tr = tok_te = None
    for job in jobs:
        source, n_lat = job.rsplit("-", 1)
        n_lat = int(n_lat)
        print(f"\n=== {job}", flush=True)
        if source == "pixels":
            if pix_tr is None:
                pix_tr = SceneData(TRAIN_SCENES, 0, "pixels")
                pix_te = SceneData(TEST_SCENES, 999, "pixels")
            tr, te = pix_tr, pix_te
            steps, batch, in_dim = PIXEL_STEPS, PIXEL_BATCH, PIXEL_DIM
        else:
            if tok_tr is None:
                tok_tr, tok_te = _token_data()
            tr, te = tok_tr, tok_te
            steps, batch, in_dim = TOKEN_STEPS, TOKEN_BATCH, D

        torch.manual_seed(0)
        model = PerceiverIO(n_lat, in_dim=in_dim)
        t0 = time.time()
        hist = train(model, tr, steps, batch)
        secs = time.time() - t0
        acc, per_q = evaluate(model, te)
        row = dict(job=job, source=source, latents=n_lat,
                   params=V.count_params(model), acc=acc,
                   ms_per_step=1000 * secs / steps, steps=steps, batch=batch,
                   **{f"kind{q}": per_q[q] for q in per_q})

        # permutation invariance: shuffle the input elements, keep each element's
        # own features intact. A grid model would break; this must not.
        n_in = te.inputs(np.arange(2)).shape[1]
        perm = torch.from_numpy(np.random.default_rng(0).permutation(n_in))
        inp, tok, _ = te.batch(np.arange(64))
        with torch.no_grad():
            a = model(inp, tok)
            b = model(inp[:, perm], tok)
        row["perm_max_abs_diff"] = float((a - b).abs().max())
        row["acc_shuffled"] = evaluate(model, te, perm=perm)[0]
        print(f"  acc {acc:.3f} (shuffled inputs {row['acc_shuffled']:.3f}, "
              f"max |delta logit| {row['perm_max_abs_diff']:.2e})", flush=True)
        print(f"  {1000*secs/steps:.0f} ms/step, {V.count_params(model)/1e6:.2f}M params",
              flush=True)
        rows.append(row)
        curves[job] = np.array(hist)

    path = OUT / "perceiver.csv"
    keys = list(rows[0].keys())
    old = {r["job"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    old.update({r["job"]: r for r in rows})
    order = ["pixels-16", "tokens-4", "tokens-16", "tokens-64"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([old[j] for j in order if j in old])
    old_curves = dict(np.load(OUT / "curves.npz")) if (OUT / "curves.npz").exists() else {}
    old_curves.update(curves)
    np.savez(OUT / "curves.npz", **old_curves)
    print("\nwrote", path)


def stage_plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scaling = list(csv.DictReader(open(OUT / "scaling.csv")))
    rows = list(csv.DictReader(open(OUT / "perceiver.csv")))
    curves = np.load(OUT / "curves.npz")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for name, color in (("perceiver", "#4c72b0"), ("dense", "#c44e52")):
        pts = [(int(r["n_inputs"]), float(r["ms_per_image"]))
               for r in scaling if r["model"] == name]
        ax.plot(*zip(*pts), "o-", color=color, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("input elements N")
    ax.set_ylabel("ms per image (forward)")
    ax.set_title("Cost vs input size")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for r in rows:
        h = curves[r["job"]]
        ax.plot(np.convolve(h, np.ones(50) / 50, mode="valid"), label=r["job"])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (nats)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(len(rows))
    ax.bar(x, [float(r["acc"]) for r in rows],
           color=["#b07aa1" if r["source"] == "pixels" else "#4c72b0" for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, float(r["acc"]) + 0.012, f"{float(r['acc']):.3f}", ha="center",
                fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["job"] for r in rows], rotation=18, fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy: input type and bottleneck size")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "perceiver.png", dpi=130)
    print("wrote", OUT / "perceiver.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="train", choices=["scaling", "train", "plot"])
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    {"scaling": stage_scaling, "train": stage_train, "plot": stage_plot}[a.stage](a)
