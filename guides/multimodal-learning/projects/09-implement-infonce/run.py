"""Project 09 -- Implement InfoNCE.

Stages:
    verify    hand-written loss vs torch, analytic gradient vs autograd,
              autograd vs finite differences
    anatomy   where the gradient actually goes: the softmax push weights
    floor     why an InfoNCE number is meaningless without its batch size
    symmetry  train a small aligner with rows-only / cols-only / both

    python3 run.py --stage all
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "02-visualize-the-modality-gap"))

import clip_lib                                                   # noqa: E402
import plot_style as ps                                           # noqa: E402
from infonce import (analytic_grad_wrt_logits, finite_difference,  # noqa: E402
                     in_batch_accuracy, infonce_manual, push_weights,
                     reference_infonce, similarity_matrix)

OUT = HERE / "outputs"
# Project 02 already caches 1,000 COCO pairs; reuse them instead of a second copy.
DATA = PROJECTS / "02-visualize-the-modality-gap" / "data"
N_PAIRS = 1000


def real_embeddings():
    """Frozen CLIP embeddings of 1,000 real COCO (image, caption) pairs."""
    clip_lib.fetch_coco(DATA, n=N_PAIRS)
    img, txt, _, _, _ = clip_lib.cached_embeddings(DATA, n=N_PAIRS)
    return torch.from_numpy(img), torch.from_numpy(txt)


# ---------------------------------------------------------------------------
# stage: verify
# ---------------------------------------------------------------------------
def stage_verify():
    torch.manual_seed(0)
    v, t = real_embeddings()
    v, t = v[:256].double(), t[:256].double()
    results = {}

    # 1. hand-written loss == torch's cross_entropy version
    for d in ("rows", "cols", "both"):
        mine = infonce_manual(v, t, 0.07, d).item()
        theirs = reference_infonce(v, t, 0.07, d).item()
        results[f"loss_{d}_manual"] = mine
        results[f"loss_{d}_reference"] = theirs
        results[f"loss_{d}_absdiff"] = abs(mine - theirs)

    # 2. the gradient we derived on paper == the one autograd computes
    s = similarity_matrix(v, t, 0.07).detach().requires_grad_(True)
    for d in ("rows", "cols", "both"):
        s.grad = None
        n = s.shape[0]
        idx = torch.arange(n)
        row = (-s[idx, idx] + torch.logsumexp(s, 1)).mean()
        col = (-s[idx, idx] + torch.logsumexp(s, 0)).mean()
        loss = {"rows": row, "cols": col, "both": 0.5 * (row + col)}[d]
        loss.backward()
        auto = s.grad.detach()
        paper = analytic_grad_wrt_logits(s.detach(), d)
        results[f"grad_{d}_maxdiff"] = float((auto - paper).abs().max())

    # 3. autograd == finite differences, on the embeddings themselves
    results["fd_maxdiff_images"] = finite_difference(
        lambda x: infonce_manual(x, t, 0.07), v)
    results["fd_maxdiff_texts"] = finite_difference(
        lambda x: infonce_manual(v, x, 0.07), t)

    # 4. the two halves of the symmetric loss really are different numbers
    results["rows_minus_cols"] = (results["loss_rows_manual"]
                                  - results["loss_cols_manual"])

    OUT.mkdir(exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(results, indent=2))
    for k, val in results.items():
        print(f"  {k:28s} {val:.3e}" if "diff" in k else f"  {k:28s} {val:.6f}")


# ---------------------------------------------------------------------------
# stage: anatomy
# ---------------------------------------------------------------------------
TAUS = [0.01, 0.07, 0.2, 1.0]


def stage_anatomy():
    v, t = real_embeddings()
    v, t = v[:256], t[:256]
    rows = []
    curves = {}
    for tau in TAUS:
        s = similarity_matrix(v, t, tau)
        w = push_weights(s)                      # (N, N) sorted, rows sum to 1
        mean_w = w.mean(0).numpy()
        curves[tau] = mean_w
        # How many negatives does it take to account for half the push?
        c = np.cumsum(mean_w)
        half = int(np.searchsorted(c, 0.5)) + 1
        rows.append({
            "tau": tau,
            "loss": round(float(infonce_manual(v, t, tau)), 4),
            "in_batch_acc": round(in_batch_accuracy(s), 4),
            "top1_share": round(float(mean_w[0]), 4),
            "top10_share": round(float(mean_w[:10].sum()), 4),
            "negatives_for_half_the_push": half,
            "effective_negatives": round(float(1.0 / np.sum(mean_w ** 2)), 1),
        })
        print(f"  tau={tau:<5} loss={rows[-1]['loss']:<8} "
              f"top1_share={rows[-1]['top1_share']:<8} "
              f"eff_negatives={rows[-1]['effective_negatives']}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "anatomy.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, tau in enumerate(TAUS):
        ax.plot(np.arange(1, 51), curves[tau][:50], color=ps.SERIES[i],
                linewidth=2, label=f"tau = {tau}")
    ax.axhline(1 / 255, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(30, 1 / 255 * 1.15, "equal share (1/255)", color=ps.INK_MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Who gets pushed? Share of the repulsive gradient, by rank",
              "negative rank (1 = hardest)", "share of the push",
              OUT / "push_weights.png")


# ---------------------------------------------------------------------------
# stage: floor
# ---------------------------------------------------------------------------
SIZES = [8, 16, 32, 64, 128, 256, 512, 1000]


def stage_floor():
    v, t = real_embeddings()
    rng = np.random.default_rng(0)
    rand_v = torch.from_numpy(rng.normal(size=v.shape).astype(np.float32))
    rand_t = torch.from_numpy(rng.normal(size=t.shape).astype(np.float32))

    rows = []
    for n in SIZES:
        real = similarity_matrix(v[:n], t[:n], 0.07)
        rnd = similarity_matrix(rand_v[:n], rand_t[:n], 0.07)
        loss_real = float(infonce_manual(v[:n], t[:n], 0.07))
        rows.append({
            "batch_size": n,
            "chance_loss_lnN": round(float(np.log(n)), 4),
            "loss_random_emb": round(float(infonce_manual(rand_v[:n], rand_t[:n], 0.07)), 4),
            "loss_real_clip": round(loss_real, 4),
            "acc_random_emb": round(in_batch_accuracy(rnd), 4),
            "acc_real_clip": round(in_batch_accuracy(real), 4),
            "chance_acc": round(1.0 / n, 5),
            # InfoNCE is a lower bound on the mutual information between the two
            # modalities: I(image; caption) >= log(N) - loss. The bound can never
            # exceed log(N), so a small batch cannot certify a large I no matter
            # how good the model is.
            "mi_lower_bound_nats": round(float(np.log(n)) - loss_real, 4),
            "mi_ceiling_nats": round(float(np.log(n)), 4),
        })
        print(f"  N={n:<5} lnN={rows[-1]['chance_loss_lnN']:<7} "
              f"real={rows[-1]['loss_real_clip']:<8} acc={rows[-1]['acc_real_clip']}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "batch_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    x = [r["batch_size"] for r in rows]
    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(x, [r["chance_loss_lnN"] for r in rows], color=ps.INK_MUTED,
            linestyle="--", linewidth=1.5, label="chance level  ln(N)")
    ax.plot(x, [r["loss_random_emb"] for r in rows], "o-", color=ps.SERIES[2],
            linewidth=2, label="random embeddings")
    ax.plot(x, [r["loss_real_clip"] for r in rows], "o-", color=ps.SERIES[0],
            linewidth=2, label="real frozen CLIP")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The same loss value means different things at different batch sizes",
              "batch size N", "InfoNCE loss (tau = 0.07)", OUT / "batch_size.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(x, [r["mi_ceiling_nats"] for r in rows], color=ps.INK_MUTED,
            linestyle="--", linewidth=1.5, label="ceiling of the bound  ln(N)")
    ax.plot(x, [r["mi_lower_bound_nats"] for r in rows], "o-", color=ps.SERIES[1],
            linewidth=2, label="certified I(image; caption)  =  ln(N) - loss")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Small batches cannot certify much shared information",
              "batch size N", "nats of mutual information", OUT / "mi_bound.png")


# ---------------------------------------------------------------------------
# stage: symmetry
# ---------------------------------------------------------------------------
class Aligner(nn.Module):
    """A zero-initialized residual on top of frozen CLIP features.

    Why the residual instead of a plain linear layer: a plain layer starts from
    random weights, so step 0 already scrambles CLIP's features and every
    condition begins from a broken model. With `out = x + W x` and `W = 0`, the
    model at step 0 *is* frozen CLIP, so any difference between the three
    conditions has to have been produced by the loss. (This zero-init trick is
    Flamingo's gating idea from Phase 4, borrowed for an experiment.)
    """

    def __init__(self, dim=512):
        super().__init__()
        self.v = nn.Linear(dim, dim, bias=False)
        self.t = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.t.weight)

    def forward(self, v, t):
        return v + self.v(v), t + self.t(t)


def hubness(sim):
    """How concentrated are the top-1 answers? 1.0 = every query picks a
    different gallery item; large = a few 'hub' items answer for everybody."""
    top = sim.argmax(axis=1)
    counts = np.bincount(top, minlength=sim.shape[1])
    return float(counts.max()), float((counts ** 2).sum() / counts.sum())


def stage_symmetry(steps=400, batch=256, seed=0):
    v, t = real_embeddings()
    n_train = 800
    tr_v, tr_t = v[:n_train], t[:n_train]
    te_v, te_t = v[n_train:], t[n_train:]

    # First, the mechanism: how much gradient does each half of the loss send to
    # each tower? If the row half already pushes on the text embeddings just as
    # hard as the column half does, the two halves cannot be very different.
    pressure = {}
    for direction in ("rows", "cols", "both"):
        a, b = v[:256].clone().requires_grad_(True), t[:256].clone().requires_grad_(True)
        infonce_manual(a, b, 0.07, direction).backward()
        pressure[direction] = {"grad_norm_images": round(float(a.grad.norm()), 5),
                               "grad_norm_texts": round(float(b.grad.norm()), 5)}
        print(f"  {direction:6s} gradient on images {pressure[direction]['grad_norm_images']:.5f}"
              f"   on texts {pressure[direction]['grad_norm_texts']:.5f}")
    (OUT).mkdir(exist_ok=True)
    (OUT / "gradient_pressure.json").write_text(json.dumps(pressure, indent=2))

    rows = []
    for direction in ("rows", "cols", "both"):
        torch.manual_seed(seed)
        model = Aligner(v.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        rng = np.random.default_rng(seed)
        for step in range(steps):
            idx = rng.choice(n_train, batch, replace=False)
            pv, pt = model(tr_v[idx], tr_t[idx])
            loss = infonce_manual(pv, pt, 0.07, direction)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pv, pt = model(te_v, te_t)
            pv = (pv / pv.norm(dim=-1, keepdim=True)).numpy()
            pt = (pt / pt.norm(dim=-1, keepdim=True)).numpy()
        sim = pv @ pt.T
        i2t = clip_lib.recall_at_k(sim)
        t2i = clip_lib.recall_at_k(sim.T)
        hub_max, hub_conc = hubness(sim)
        rows.append({
            "loss_direction": direction,
            "i2t_r1": round(i2t[1], 4), "i2t_r5": round(i2t[5], 4),
            "t2i_r1": round(t2i[1], 4), "t2i_r5": round(t2i[5], 4),
            "asymmetry_i2t_minus_t2i": round(i2t[1] - t2i[1], 4),
            "caption_hub_max": hub_max,
            "caption_hub_concentration": round(hub_conc, 3),
        })
        print(f"  {direction:6s} i2t R@1 {i2t[1]:.3f}  t2i R@1 {t2i[1]:.3f}  "
              f"hub_max {hub_max:.0f}")

    # the untrained baseline, for reference
    sim = (te_v / te_v.norm(dim=-1, keepdim=True)).numpy() @ \
          (te_t / te_t.norm(dim=-1, keepdim=True)).numpy().T
    i2t, t2i = clip_lib.recall_at_k(sim), clip_lib.recall_at_k(sim.T)
    hub_max, hub_conc = hubness(sim)
    rows.insert(0, {
        "loss_direction": "none (frozen CLIP)",
        "i2t_r1": round(i2t[1], 4), "i2t_r5": round(i2t[5], 4),
        "t2i_r1": round(t2i[1], 4), "t2i_r5": round(t2i[5], 4),
        "asymmetry_i2t_minus_t2i": round(i2t[1] - t2i[1], 4),
        "caption_hub_max": hub_max,
        "caption_hub_concentration": round(hub_conc, 3),
    })

    OUT.mkdir(exist_ok=True)
    with open(OUT / "symmetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = ps.new_axes(7.2, 4.2)
    labels = [r["loss_direction"] for r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["i2t_r1"] for r in rows], 0.38, color=ps.SERIES[0],
           label="image -> text  R@1")
    ax.bar(x + 0.2, [r["t2i_r1"] for r in rows], 0.38, color=ps.SERIES[1],
           label="text -> image  R@1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Train on one direction only and retrieval tilts that way",
              "which half of the symmetric loss was used", "R@1 (200-item gallery)",
              OUT / "symmetry.png")


STAGES = {"verify": stage_verify, "anatomy": stage_anatomy,
          "floor": stage_floor, "symmetry": stage_symmetry}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    torch.set_num_threads(12)
    for name in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {name}")
        STAGES[name]()
