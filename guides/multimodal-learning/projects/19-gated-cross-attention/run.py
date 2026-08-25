"""Project 19 -- Flamingo's gated cross-attention, and the identity check.

Flamingo's problem: you have a language model that already writes good English,
and you want it to look at pictures. Fine-tuning it on image data risks
destroying the language ability you paid for. Flamingo's answer is to FREEZE
the language model completely and insert brand-new cross-attention layers
between its blocks -- each multiplied by tanh(g) with the parameter g starting
at exactly 0.

tanh(0) = 0, so at initialisation the new layers contribute literally nothing
and the whole thing is bit-for-bit the original language model. Training then
opens the gates gradually. This project builds that, verifies the identity
exactly (the difference must be 0.0, not "small"), and measures what the gate
is worth by running the same architecture without it.

Stages:
    python3 run.py --stage lm      # train the text-only LM we will freeze (~3 min)
    python3 run.py --stage gate    # identity check + gated vs ungated (~13 min)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "16-implement-q-former"))
import caption_lib as C                                        # noqa: E402

torch.set_num_threads(12)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
D = 256
LM_STEPS = 1400
XATTN_STEPS = 1400
BATCH = 64


# ---------------------------------------------------------------------------
# the new layer
# ---------------------------------------------------------------------------
class GatedCrossAttention(nn.Module):
    """One Flamingo layer: text queries the image, then a gate decides how much
    of the answer is allowed through.

    Two gates, exactly as in the paper: one on the attention branch and one on
    the feed-forward branch that follows it. Both are single scalars passed
    through tanh, and both start at 0.

    Why tanh(g) and not just g: a raw scalar starting at 0 would work for the
    identity check too, but it can grow without bound and can flip sign
    abruptly. tanh keeps the multiplier inside (-1, 1), so the new branch can
    never overpower the frozen stream it is being added to, and the derivative
    is largest near 0 -- the gate opens fastest exactly when it is still shut.
    """

    def __init__(self, d, kv_dim, heads=4, gated=True):
        super().__init__()
        self.h = heads
        self.gated = gated
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(kv_dim)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(kv_dim, 2 * d)
        self.proj = nn.Linear(d, d)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def gates(self):
        if not self.gated:
            return 1.0, 1.0
        return torch.tanh(self.attn_gate), torch.tanh(self.ff_gate)

    def forward(self, x, ctx):
        B, T, Dm = x.shape
        N = ctx.shape[1]
        g_attn, g_ff = self.gates()
        q = self.q(self.nq(x)).view(B, T, self.h, Dm // self.h).transpose(1, 2)
        k, v = self.kv(self.nkv(ctx)).split(Dm, dim=-1)
        k, v = (t.view(B, N, self.h, Dm // self.h).transpose(1, 2) for t in (k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + g_attn * self.proj(a.transpose(1, 2).reshape(B, T, Dm))
        return x + g_ff * self.ff(self.ff_norm(x))


class FlamingoLM(nn.Module):
    """A frozen caption LM with one gated cross-attention layer in front of each
    of its blocks. The frozen weights are never touched -- the new layers are
    attached through the LM's `hooks` argument."""

    def __init__(self, lm, kv_dim=C.CLIP_DIM, gated=True, d=D):
        super().__init__()
        self.lm = lm
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.xattn = nn.ModuleList(
            GatedCrossAttention(d, kv_dim, gated=gated) for _ in lm.blocks)

    def trainable(self):
        return [p for p in self.xattn.parameters()]

    def forward(self, tok, feats):
        hooks = [(lambda x, layer=layer: layer(x, feats)) for layer in self.xattn]
        return self.lm(tok, hooks=hooks)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def _pool():
    return C.CocoFeats()


def stage_lm(_args):
    """Train the text-only language model that everything else freezes."""
    CKPT.mkdir(exist_ok=True)
    pool = _pool()
    torch.manual_seed(0)
    lm = C.CaptionLM(len(pool.vocab), d=D)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(lm.parameters(), lr=6e-4, weight_decay=0.05,
                            betas=(0.9, 0.98), eps=1e-6)
    hist = []
    t0 = time.time()
    for step in range(LM_STEPS):
        for g in opt.param_groups:
            g["lr"] = C.cosine_lr(step, LM_STEPS, 6e-4)
        ids = rng.choice(pool.train_ids, size=BATCH, replace=False)
        _, tok = pool.batch(ids, rng)
        logits, n_prefix = lm(tok)
        loss = C.caption_loss(logits, tok, n_prefix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if step % 200 == 0 or step == LM_STEPS - 1:
            print(f"    step {step:5d}  loss {np.mean(hist[-50:]):.4f}", flush=True)
    torch.save(lm.state_dict(), CKPT / "text_lm.pt")
    vl = val_loss_text(lm, pool)
    print(f"  text-only LM val loss {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "text_lm.json").write_text(json.dumps(dict(val_loss=vl,
                                                      steps=LM_STEPS)))
    np.save(OUT / "lm_curve.npy", np.array(hist))


@torch.no_grad()
def val_loss_text(lm, pool, batch=100):
    lm.eval()
    total, n = 0.0, 0
    for cap in range(5):
        for i in range(0, len(pool.val_ids), batch):
            ids = pool.val_ids[i:i + batch]
            _, tok = pool.batch(ids, caption=cap)
            logits, n_prefix = lm(tok)
            total += float(C.caption_loss(logits, tok, n_prefix)) * len(ids)
            n += len(ids)
    lm.train()
    return total / n


@torch.no_grad()
def val_loss_flamingo(model, pool, batch=100):
    model.eval()
    total, n = 0.0, 0
    for cap in range(5):
        for i in range(0, len(pool.val_ids), batch):
            ids = pool.val_ids[i:i + batch]
            feats, tok = pool.batch(ids, caption=cap)
            logits, n_prefix = model(tok, feats)
            total += float(C.caption_loss(logits, tok, n_prefix)) * len(ids)
            n += len(ids)
    model.train()
    return total / n


def stage_gate(args):
    OUT.mkdir(exist_ok=True)
    pool = _pool()

    def fresh_lm():
        lm = C.CaptionLM(len(pool.vocab), d=D)
        lm.load_state_dict(torch.load(CKPT / "text_lm.pt"))
        return lm.eval()

    # ---- the identity check -------------------------------------------------
    torch.manual_seed(0)
    lm = fresh_lm()
    model = FlamingoLM(fresh_lm(), gated=True)
    feats, tok = pool.batch(pool.val_ids[:64])
    with torch.no_grad():
        base = lm(tok)[0]
        with_image = model(tok, feats)[0]
        diff = float((base - with_image).abs().max())
        # and the same model once a gate has been nudged off zero
        model.xattn[0].attn_gate.data.fill_(0.05)
        nudged = float((base - model(tok, feats)[0]).abs().max())
        model.xattn[0].attn_gate.data.zero_()
        ungated = FlamingoLM(fresh_lm(), gated=False)
        ungated_diff = float((base - ungated(tok, feats)[0]).abs().max())
    print(f"identity at init:      max |delta logit| = {diff:.3e}", flush=True)
    print(f"after nudging 1 gate:  max |delta logit| = {nudged:.3e}", flush=True)
    print(f"same model, no gate:   max |delta logit| = {ungated_diff:.3e}", flush=True)
    identity = dict(gated_diff=diff, nudged_diff=nudged, ungated_diff=ungated_diff,
                    exact=bool(diff == 0.0))
    (OUT / "identity.json").write_text(json.dumps(identity, indent=1))

    # ---- gated vs ungated ---------------------------------------------------
    base_val = json.loads((OUT / "text_lm.json").read_text())["val_loss"]
    rows, curves, gate_tracks = [], {}, {}
    for name in (args.variants or ["gated", "ungated"]):
        gated = name == "gated"
        print(f"\n=== {name}", flush=True)
        torch.manual_seed(0)
        model = FlamingoLM(fresh_lm(), gated=gated)
        params = model.trainable()
        opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=0.05,
                                betas=(0.9, 0.98), eps=1e-6)
        rng = np.random.default_rng(0)
        hist, track, evals = [], [], []
        t0 = time.time()
        start_val = val_loss_flamingo(model, pool)
        print(f"  val loss at step 0: {start_val:.4f} "
              f"(frozen text-only LM: {base_val:.4f})", flush=True)
        for step in range(XATTN_STEPS):
            for g in opt.param_groups:
                g["lr"] = C.cosine_lr(step, XATTN_STEPS, 6e-4)
            ids = rng.choice(pool.train_ids, size=BATCH, replace=False)
            feats, tok = pool.batch(ids, rng)
            logits, n_prefix = model(tok, feats)
            loss = C.caption_loss(logits, tok, n_prefix)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            hist.append(float(loss.detach()))
            if step % 20 == 0:
                track.append([float(torch.tanh(l.attn_gate)) for l in model.xattn])
            if step % 200 == 0 or step == XATTN_STEPS - 1:
                print(f"    step {step:5d}  loss {np.mean(hist[-50:]):.4f}", flush=True)
        secs = time.time() - t0
        vl = val_loss_flamingo(model, pool)
        rows.append(dict(variant=name, start_val_loss=start_val, val_loss=vl,
                         val_ppl=float(np.exp(vl)), frozen_lm_val_loss=base_val,
                         trainable=sum(p.numel() for p in params),
                         frozen=sum(p.numel() for p in model.lm.parameters()),
                         ms_per_step=1000 * secs / XATTN_STEPS))
        curves[name] = np.array(hist)
        gate_tracks[name] = np.array(track)
        evals.append(vl)
        print(f"  final val loss {vl:.4f} (ppl {np.exp(vl):.1f})", flush=True)

    path = OUT / "gating.csv"
    old = {r["variant"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    old.update({r["variant"]: r for r in rows})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([old[k] for k in ("gated", "ungated") if k in old])
    np.savez(OUT / "curves.npz", **curves)
    np.savez(OUT / "gates.npz", **gate_tracks)
    print("\nwrote", path)


def stage_plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(OUT / "gating.csv")))
    curves = np.load(OUT / "curves.npz")
    gates = np.load(OUT / "gates.npz")
    base = float(rows[0]["frozen_lm_val_loss"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    colors = {"gated": "#55a868", "ungated": "#c44e52"}
    for r in rows:
        h = curves[r["variant"]]
        ax.plot(np.convolve(h, np.ones(30) / 30, mode="valid"),
                label=r["variant"], color=colors[r["variant"]])
    ax.axhline(base, color="k", ls="--", lw=1)
    ax.text(len(curves["gated"]) * 0.55, base + 0.02,
            "frozen text-only LM (no image)", fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("caption loss (nats/word)")
    ax.set_title("Training with and without the gate")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["step 0", "after training", "frozen LM"]
    width = 0.35
    for i, r in enumerate(rows):
        vals = [float(r["start_val_loss"]), float(r["val_loss"]), base]
        ax.bar(np.arange(3) + (i - 0.5) * width, vals, width,
               label=r["variant"], color=colors[r["variant"]])
    ax.axhline(base, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("validation caption loss")
    ax.set_title("The cost of a bad start")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    g = gates["gated"]
    for layer in range(g.shape[1]):
        ax.plot(np.arange(len(g)) * 20, g[:, layer], label=f"block {layer}")
    ax.set_xlabel("step")
    ax.set_ylabel("tanh(gate) on the attention branch")
    ax.set_title("The gates opening")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gating.png", dpi=130)
    print("wrote", OUT / "gating.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="gate", choices=["lm", "gate", "plot"])
    p.add_argument("--variants", nargs="*", default=None,
                   choices=["gated", "ungated"])
    a = p.parse_args()
    {"lm": stage_lm, "gate": stage_gate, "plot": stage_plot}[a.stage](a)
