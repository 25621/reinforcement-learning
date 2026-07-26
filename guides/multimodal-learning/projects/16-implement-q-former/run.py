"""Project 16 -- build a small Q-Former and measure what its 16 tokens cost.

A frozen CLIP turns one image into 50 tokens of 768 numbers. A language model
would happily read all 50, but 50 tokens per image is expensive once you have
several images or a long conversation, and it grows if you raise the
resolution. BLIP-2's answer is a Q-Former: a small module holding a FIXED
number of learned query vectors that cross-attend to the image and come back
with a fixed-size summary.

Five ways to hand the same frozen image to the same caption model:

  no-image      1 token   -- a learned token that never looks at the image
  pooled-1      1 token   -- CLIP's own summary vector (the CLS token)
  gridpool-16   16 tokens -- average the 7x7 patch grid down to 4x4
  qformer-16    16 tokens -- 16 LEARNED queries that cross-attend to the image
  patches-50    50 tokens -- every patch token, LLaVA-style (no compression)

Two of those are controls, and they are what make the result readable.
`gridpool-16` spends the same token budget as the Q-Former but compresses by
averaging instead of learning, so any gap between them is what the learned
queries are worth. `no-image` is the floor: without it, a tie among the rest
could mean "extra tokens do not help" OR "this benchmark cannot see images".

Usage:
    python3 run.py --stage train   # 5 conditions (~9 min after the cache exists)
    python3 run.py --stage plot
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import caption_lib as C

torch.set_num_threads(12)

OUT = Path(__file__).resolve().parent / "outputs"
D = 256
STEPS = 1600
BATCH = 64
CONDITIONS = ["no-image", "pooled-1", "gridpool-16", "qformer-16", "patches-50"]
N_QUERIES = 16


# ---------------------------------------------------------------------------
# the Q-Former
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Queries in one width (d), keys/values in another (kv_dim).

    The kv projection is what lets a 256-wide module read 768-wide CLIP
    features without anyone resizing anything by hand.
    """

    def __init__(self, d, kv_dim, heads=4):
        super().__init__()
        self.h = heads
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(kv_dim)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(kv_dim, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx):
        B, T, Dm = x.shape
        N = ctx.shape[1]
        q = self.q(self.nq(x)).view(B, T, self.h, Dm // self.h).transpose(1, 2)
        k, v = self.kv(self.nkv(ctx)).split(Dm, dim=-1)
        k, v = (t.view(B, N, self.h, Dm // self.h).transpose(1, 2) for t in (k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(a.transpose(1, 2).reshape(B, T, Dm))


class SelfAttention(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, Dm = x.shape
        q, k, v = self.qkv(self.norm(x)).split(Dm, dim=-1)
        q, k, v = (t.view(B, T, self.h, Dm // self.h).transpose(1, 2)
                   for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(a.transpose(1, 2).reshape(B, T, Dm))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.net(self.norm(x))


class QFormer(nn.Module):
    """`n_query` learned vectors that read an image and come back summarised.

    The queries are ordinary nn.Parameters -- they do not depend on the image at
    all. Each one is a standing question ("is there a person?", "what is the
    background?") that gets asked of every image, and the answer is what the
    query carries away. That is why the output length is constant no matter how
    many patch tokens went in.
    """

    def __init__(self, n_query=N_QUERIES, d=D, kv_dim=C.CLIP_DIM, layers=2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [SelfAttention(d), CrossAttention(d, kv_dim), MLP(d)]
        self.norm = nn.LayerNorm(d)

    def forward(self, feats):
        z = self.query.expand(len(feats), -1, -1)
        for layer in self.layers:
            z = layer(z, feats) if isinstance(layer, CrossAttention) else layer(z)
        return self.norm(z)


class Captioner(nn.Module):
    """frozen CLIP features -> K prefix tokens -> caption language model."""

    def __init__(self, mode, vocab_size, d=D):
        super().__init__()
        self.mode = mode
        # CLIP's raw patch features are not normalised and carry a few very
        # large dimensions; a LayerNorm here puts every condition on the same
        # footing before the bridge sees them.
        self.in_norm = nn.LayerNorm(C.CLIP_DIM)
        if mode == "qformer-16":
            self.bridge = QFormer(N_QUERIES, d)
            self.proj = nn.Linear(d, d)
            n_prefix = N_QUERIES
        elif mode == "no-image":
            # the floor: one learned prefix token that never looks at the image
            self.bridge = None
            self.proj = nn.Linear(C.CLIP_DIM, d)
            self.blind = nn.Parameter(torch.zeros(1, 1, d))
            n_prefix = 1
        else:
            self.bridge = None
            self.proj = nn.Linear(C.CLIP_DIM, d)
            n_prefix = {"pooled-1": 1, "gridpool-16": 16, "patches-50": 50}[mode]
        self.n_prefix = n_prefix
        self.lm = C.CaptionLM(vocab_size, d=d, n_prefix=n_prefix)

    def prefix(self, feats):
        if self.mode == "no-image":
            return self.blind.expand(len(feats), -1, -1)
        feats = self.in_norm(feats)
        if self.mode == "pooled-1":
            return self.proj(feats[:, :1])
        if self.mode == "gridpool-16":
            grid = feats[:, 1:].transpose(1, 2).reshape(len(feats), -1, 7, 7)
            pooled = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)
            return self.proj(pooled)
        if self.mode == "patches-50":
            return self.proj(feats)
        return self.proj(self.bridge(feats))

    def forward(self, feats, tok):
        return self.lm(tok, prefix=self.prefix(feats))


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def train(model, pool, steps=STEPS, batch=BATCH, lr=6e-4, seed=0, log_every=200):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05,
                            betas=(0.9, 0.98), eps=1e-6)
    hist = []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = C.cosine_lr(step, steps, lr)
        ids = rng.choice(pool.train_ids, size=batch, replace=False)
        feats, tok = pool.batch(ids, rng)
        logits, n_prefix = model(feats, tok)
        loss = C.caption_loss(logits, tok, n_prefix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {np.mean(hist[-50:]):.4f}", flush=True)
    return hist


@torch.no_grad()
def val_loss(model, pool, batch=100):
    model.eval()
    total, n = 0.0, 0
    for cap in range(5):
        for i in range(0, len(pool.val_ids), batch):
            ids = pool.val_ids[i:i + batch]
            feats, tok = pool.batch(ids, caption=cap)
            logits, n_prefix = model(feats, tok)
            total += float(C.caption_loss(logits, tok, n_prefix)) * len(ids)
            n += len(ids)
    return total / n


@torch.no_grad()
def caption_ranking(model, pool, n_distractors=49, seed=0, batch=10):
    """Does the model score the RIGHT caption above 49 random ones?

    Caption loss alone is hard to read -- 3.1 vs 3.3 nats means little on its
    own. This turns the same model into a 50-way multiple-choice test, where
    chance is exactly 2%, so the numbers say how much the image is actually
    being used.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    ids = pool.val_ids
    hits = 0
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        feats, _ = pool.batch(chunk)
        cand = np.stack([np.concatenate(
            [[j], rng.choice(ids[ids != j], n_distractors, replace=False)])
            for j in chunk])                                  # (b, 50) image ids
        toks = torch.from_numpy(pool.tokens[cand, 0])          # (b, 50, CTX)
        b, k, ctx = toks.shape
        rep = feats[:, None].expand(-1, k, -1, -1).reshape(b * k, *feats.shape[1:])
        logits, n_prefix = model(rep, toks.reshape(b * k, ctx))
        lp = logits[:, n_prefix:][:, :-1]
        tgt = toks.reshape(b * k, ctx)[:, 1:]
        ll = -F.cross_entropy(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1),
                              reduction="none").reshape(b * k, -1)
        keep = (tgt != C.PAD).float()
        score = ((ll * keep).sum(1) / keep.sum(1)).reshape(b, k)
        hits += int((score.argmax(1) == 0).sum())
    return hits / len(ids)


@torch.no_grad()
def greedy(model, pool, ids, max_new=18):
    model.eval()
    feats, _ = pool.batch(ids)
    tok = torch.full((len(ids), 1), C.SOT, dtype=torch.long)
    for _ in range(max_new):
        logits, n_prefix = model(feats, tok)
        nxt = logits[:, n_prefix + tok.shape[1] - 1].argmax(-1, keepdim=True)
        tok = torch.cat([tok, nxt], dim=1)
    return [C.detokenize(t.tolist()[1:], pool.vocab) for t in tok]


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    pool = C.CocoFeats()
    print(f"vocab {len(pool.vocab)} words, {len(pool.train_ids)} train images",
          flush=True)
    rows, curves, samples = [], {}, {}
    for cond in (args.only or CONDITIONS):
        print(f"\n=== {cond}", flush=True)
        torch.manual_seed(0)
        model = Captioner(cond, len(pool.vocab))
        t0 = time.time()
        hist = train(model, pool, STEPS)
        secs = time.time() - t0
        vl = val_loss(model, pool)
        rank = caption_ranking(model, pool)
        bridge_params = (sum(p.numel() for p in model.bridge.parameters())
                         if model.bridge is not None else 0)
        rows.append(dict(condition=cond, prefix_tokens=model.n_prefix,
                         val_loss=vl, val_ppl=float(np.exp(vl)),
                         retrieval_top1=rank,
                         bridge_params=bridge_params + sum(p.numel()
                                                           for p in model.proj.parameters()),
                         ms_per_step=1000 * secs / STEPS))
        curves[cond] = np.array(hist)
        samples[cond] = greedy(model, pool, pool.val_ids[:6])
        print(f"  val loss {vl:.4f} (ppl {np.exp(vl):.1f})  "
              f"50-way caption choice {rank:.3f}  {1000*secs/STEPS:.0f} ms/step",
              flush=True)
        for s in samples[cond][:3]:
            print("   ", s, flush=True)

    path = OUT / "qformer.csv"
    old = ({r["condition"]: r for r in csv.DictReader(open(path))}
           if path.exists() else {})
    old.update({r["condition"]: r for r in rows})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([old[c] for c in CONDITIONS if c in old])
    old_curves = dict(np.load(OUT / "curves.npz")) if (OUT / "curves.npz").exists() else {}
    old_curves.update(curves)
    np.savez(OUT / "curves.npz", **old_curves)
    old_samples = (json.loads((OUT / "samples.json").read_text())
                   if (OUT / "samples.json").exists() else {})
    old_samples.update(samples)
    (OUT / "samples.json").write_text(json.dumps(old_samples, indent=1))
    print("\nwrote", OUT / "qformer.csv")


def stage_plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(OUT / "qformer.csv")))
    curves = np.load(OUT / "curves.npz")
    colors = dict(zip(CONDITIONS,
                      ["#444444", "#888888", "#dd8452", "#4c72b0", "#55a868"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for r in rows:
        h = curves[r["condition"]]
        ax.plot(np.convolve(h, np.ones(50) / 50, mode="valid"),
                label=r["condition"], color=colors[r["condition"]])
    ax.set_xlabel("step")
    ax.set_ylabel("caption loss (nats/word)")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = [int(r["prefix_tokens"]) for r in rows]
    ax.plot(x, [float(r["val_loss"]) for r in rows], "o--", color="#666666")
    for r in rows:
        ax.scatter(int(r["prefix_tokens"]), float(r["val_loss"]), s=110,
                   color=colors[r["condition"]], zorder=3)
        ax.annotate(r["condition"], (int(r["prefix_tokens"]), float(r["val_loss"])),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("image tokens fed to the language model")
    ax.set_ylabel("validation caption loss")
    ax.set_title("What each token buys")
    ax.grid(alpha=0.3)

    ax = axes[2]
    idx = np.arange(len(rows))
    ax.bar(idx, [float(r["retrieval_top1"]) for r in rows],
           color=[colors[r["condition"]] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, float(r["retrieval_top1"]) + 0.01,
                f"{float(r['retrieval_top1']):.3f}", ha="center", fontsize=9)
    ax.axhline(0.02, color="k", ls=":", lw=1)
    ax.text(len(rows) - 0.5, 0.03, "chance (1/50)", ha="right", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([r["condition"] for r in rows], rotation=18, fontsize=8)
    ax.set_ylabel("50-way caption choice accuracy")
    ax.set_title("Is the image actually being used?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "qformer.png", dpi=130)

    # sample captions next to their images
    pool_thumbs = np.load(C.data_dir() / "thumbs.npy", mmap_mode="r")
    order = np.random.default_rng(0).permutation(C.N_IMAGES)
    val_ids = order[:C.N_VAL][:6]
    samples = json.loads((OUT / "samples.json").read_text())
    fig2, axes2 = plt.subplots(1, 6, figsize=(16, 4.6))
    for j, ax in enumerate(axes2):
        ax.imshow(pool_thumbs[val_ids[j]])
        ax.axis("off")
        ax.set_title("\n".join(f"{c}:\n  {samples[c][j]}" for c in CONDITIONS
                               if c in samples), fontsize=5, loc="left")
    fig2.tight_layout()
    fig2.savefig(OUT / "captions.png", dpi=130)
    print("wrote", OUT / "qformer.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="train", choices=["train", "plot"])
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    {"train": stage_train, "plot": stage_plot}[a.stage](a)
