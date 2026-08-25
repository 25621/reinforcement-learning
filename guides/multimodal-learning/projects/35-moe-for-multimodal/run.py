"""MoE for multimodal: do experts pick a modality, if you let them?

Project 34 trained one transformer on text, images and audio. This project
swaps every feed-forward layer for a Mixture-of-Experts layer and asks a
question nobody has to answer by hand: given eight identical experts and a
router that is free to choose, does the model spontaneously send image tokens
one way and audio tokens another?

Nothing here forces specialisation. The router only ever sees a vector; it is
never told which modality produced it. Whatever structure shows up in the
routing table was inferred.

Stages
    train  dense / dense-wide / MoE / MoE-without-balancing   (~7 min)
    route  the routing table, per layer and per modality      (~1 min)
    plot   figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "32-discrete-image-tokens"))
sys.path.insert(0, str(PROJECTS / "33-tiny-chameleon"))
sys.path.insert(0, str(PROJECTS / "34-modality-balancing"))
sys.path.insert(0, str(HERE))
import moe as M  # noqa: E402
import plot_style as ps  # noqa: E402
import tri_modal as T  # noqa: E402  (project 34's corpus)
import unified as U  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
CTX = T.CTX
D, LAYERS, HEADS = 192, 4, 4
STEPS, BATCH, LR = 800, 32, 3e-3
N_EXPERTS, TOP_K = 8, 2
AUX_W = 0.02
MODALITIES = ["special", "text", "image", "audio"]
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def corpus():
    """The same tri-modal corpus project 34 built -- oversampled 50/50 so no
    modality is starved. Specialisation is the question here, not balance."""
    d, vocab = T.build()
    ir = T.image_rows(vocab, d["cap_tr"], d["img_tr"])
    ar = T.audio_rows(vocab, d["acap_tr"], d["aud_tr"])
    val = T.mixed_val(vocab, d)
    rows = U.pad_batch(ir + ar, CTX)
    n_img, n_aud = len(ir), len(ar)

    def sampler(rng, batch):
        take = rng.random(batch) < 0.5
        return np.where(take, rng.integers(0, n_img, batch),
                        n_img + rng.integers(0, n_aud, batch))
    return rows, val, vocab, sampler


# ---------------------------------------------------------------------------
def stage_train(steps=STEPS):
    rows, val, vocab, sampler = corpus()
    CKPT.mkdir(exist_ok=True)
    arms = {
        # the plain block from projects 33/34
        "dense": dict(factory=M.dense_factory(4), aux=0.0),
        # a dense block with the SAME arithmetic per token as top-2-of-8:
        # two experts of width 4d run per token, so one MLP of width 8d matches
        "dense_wide": dict(factory=M.dense_factory(8), aux=0.0),
        "moe": dict(factory=M.moe_factory(N_EXPERTS, TOP_K), aux=AUX_W),
        "moe_no_balance": dict(factory=M.moe_factory(N_EXPERTS, TOP_K), aux=0.0),
    }
    out = {}
    for name, cfg in arms.items():
        torch.manual_seed(0)
        model = U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX,
                            mlp_factory=cfg["factory"])
        total = sum(p.numel() for p in model.parameters())
        # active parameters = what actually multiplies for one token
        if "moe" in name:
            per_expert = 2 * D * 4 * D
            active = total - LAYERS * N_EXPERTS * per_expert + LAYERS * TOP_K * per_expert
        else:
            active = total
        print(f"\n--- {name}: {total/1e6:.2f}M total, {active/1e6:.2f}M active per token")
        t0 = time.time()
        hist = U.train_lm(model, rows, vocab, val_seqs=val, steps=steps,
                          batch=BATCH, lr=LR, sampler=sampler,
                          aux_loss_w=cfg["aux"], log_every=max(steps // 4, 100))
        ev = U.evaluate_lm(model, val, vocab)
        secs = time.time() - t0
        out[name] = {"history": hist, "eval": ev, "total_params": total,
                     "active_params": active, "secs": secs,
                     "ms_per_step": 1000 * secs / steps, "aux_w": cfg["aux"]}
        print(f"    text {ev['text']:.3f}  image {ev['image']:.3f}  "
              f"audio {ev['audio']:.3f}   {1000*secs/steps:.0f} ms/step")
        torch.save(model.state_dict(), CKPT / f"{name}.pt")
    _save("train.json", {"steps": steps, "n_experts": N_EXPERTS, "top_k": TOP_K,
                         "arms": out})


# ---------------------------------------------------------------------------
def stage_route():
    rows, val, vocab, _ = corpus()
    kinds = vocab.kind(val)
    res = {}
    for name in ("moe", "moe_no_balance"):
        model = U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX,
                            mlp_factory=M.moe_factory(N_EXPERTS, TOP_K))
        model.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location="cpu"))
        table = M.routing_table(model, val, kinds)
        mi = M.specialisation(table)
        # per-layer expert load, ignoring which modality
        load = table.sum(1)
        load = load / load.sum(1, keepdims=True)
        res[name] = {
            "table": table.tolist(),
            "mutual_information_bits": mi,
            "max_bits": float(np.log2(3)),      # text / image / audio
            "expert_load": load.tolist(),
            "max_expert_share": [float(l.max()) for l in load],
            "dead_experts": [int((l < 0.01).sum()) for l in load],
        }
        print(f"{name}: modality->expert information {['%.3f' % m for m in mi]} bits "
              f"(max {np.log2(3):.3f}); busiest expert holds "
              f"{max(float(l.max()) for l in load):.1%} of a layer's tokens")
    _save("route.json", res)


# ---------------------------------------------------------------------------
def stage_plot():
    tr = json.loads((OUT / "train.json").read_text())
    rt = json.loads((OUT / "route.json").read_text())
    import matplotlib.pyplot as plt

    # 1) routing heatmaps
    for name in ("moe", "moe_no_balance"):
        table = np.array(rt[name]["table"])
        n_layers = table.shape[0]
        fig, axes = plt.subplots(1, n_layers, figsize=(1.75 * n_layers, 2.3), dpi=110)
        fig.patch.set_facecolor(ps.SURFACE)
        for li in range(n_layers):
            t = table[li, 1:4]                       # text / image / audio
            t = t / np.maximum(t.sum(1, keepdims=True), 1e-9)
            axes[li].imshow(t, cmap="magma", vmin=0, vmax=max(0.35, t.max()),
                            aspect="auto")
            axes[li].set_title(f"layer {li}", fontsize=9, color=ps.INK_SECONDARY)
            axes[li].set_xticks(range(table.shape[2]))
            axes[li].set_xticklabels(range(table.shape[2]), fontsize=6)
            axes[li].set_yticks([0, 1, 2])
            axes[li].set_yticklabels(["text", "image", "audio"] if li == 0 else [],
                                     fontsize=8, color=ps.INK_SECONDARY)
        fig.suptitle(f"Where each modality's tokens go  ({name.replace('_', ' ')})",
                     color=ps.INK, fontsize=11, x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(OUT / f"routing_{name}.png", facecolor=ps.SURFACE,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"wrote outputs/routing_{name}.png")

    # 2) specialisation per layer
    fig, ax = ps.new_axes(6.6, 4.0)
    for i, name in enumerate(("moe", "moe_no_balance")):
        mi = rt[name]["mutual_information_bits"]
        ax.plot(range(len(mi)), mi, "o-", color=ps.SERIES[i], lw=2,
                label=name.replace("_", " "))
    ax.axhline(np.log2(3), color=ps.BASELINE, ls="--", lw=1.5,
               label="one expert group per modality (max)")
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "How much the router's choice depends on the modality",
              "layer", "mutual information (bits)", OUT / "specialisation.png")

    # 3) loss vs compute
    fig, ax = ps.new_axes(7.0, 4.2)
    names = list(tr["arms"])
    x = np.arange(len(names))
    for i, m in enumerate(("text", "image", "audio")):
        ax.bar(x + (i - 1) * 0.27, [tr["arms"][n]["eval"][m] for n in names], 0.25,
               color=ps.SERIES[i], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{tr['arms'][n]['active_params']/1e6:.1f}M active"
                        for n in names], fontsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Validation loss by modality, four feed-forward designs",
              "", "loss (nats/token)", OUT / "arms.png")


STAGES = {"train": stage_train, "route": stage_route, "plot": stage_plot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for nm in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"\n=== {nm} " + "=" * (60 - len(nm)))
        if a.steps and nm == "train":
            STAGES[nm](steps=a.steps)
        else:
            STAGES[nm]()
