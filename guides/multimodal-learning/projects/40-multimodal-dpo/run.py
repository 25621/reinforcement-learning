"""Multimodal DPO: teach a VLM to stop naming things that are not in the picture.

The model is Phase 5's LLaVA-in-miniature (frozen CLIP -> projector -> a real
frozen-except-the-top SmolLM2-135M). We first train it to caption, then measure
how often it invents objects, then try four ways to fix that and measure again.

Arms
    base         the captioner, straight out of supervised fine-tuning
    dpo          preference training on (true caption, caption + a fake object)
    dpo-lennorm  the same, with each log-probability divided by its length
    sft-chosen   the control: more supervised training on the chosen captions
                 only, for exactly the same number of steps. If this matches
                 DPO then the rejected half of every pair bought nothing.

Stages
    data     build (or reuse) project 20's CLIP cache and the pairs   (~4 min once)
    sft      train the base captioner                                 (~7 min)
    train    one preference arm (--arm dpo)                           (~4 min each)
    eval     captions + CHAIR + caption loss for one arm (--arm base) (~3 min each)
    plot     figures and tables                                       (~10 s)
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
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
sys.path.insert(0, str(HERE))
import dpo_lib as D  # noqa: E402
import plot_style as ps  # noqa: E402
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
ARMS = ["base", "dpo", "dpo-lennorm", "sft-chosen"]
SFT_STEPS, SFT_BS, SFT_LR = 350, 8, 2e-3
DPO_STEPS, DPO_BS, DPO_LR, BETA = 200, 4, 3e-4, 0.1
N_EVAL = 150
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def _load(name):
    return json.loads((OUT / name).read_text())


def _pairs_path():
    return HERE / "data" / "pairs.json"


# ---------------------------------------------------------------------------
# stage: data
# ---------------------------------------------------------------------------
def stage_data(args):
    data = V.CocoVLMData()
    pairs = D.make_pairs(data, data.train_ids, seed=0)
    (HERE / "data").mkdir(exist_ok=True)
    _pairs_path().write_text(json.dumps(pairs))
    lens = [len(p["chosen"].split()) for p in pairs]
    rl = [len(p["rejected"].split()) for p in pairs]
    counts = {}
    for p in pairs:
        counts[p["object"]] = counts.get(p["object"], 0) + 1
    # How many objects the human captions name per image -- the number CHAIR
    # divides by, so it sets the resolution of the whole measurement.
    named = [len(D.objects_in(data.captions[int(i)])) for i in data.val_ids[:N_EVAL]]
    stats = {
        "train_images": int(len(data.train_ids)),
        "val_images": int(len(data.val_ids)),
        "pairs": len(pairs),
        "chosen_mean_words": float(np.mean(lens)),
        "rejected_mean_words": float(np.mean(rl)),
        "extra_words_in_rejected": float(np.mean(rl) - np.mean(lens)),
        "distinct_hallucinated_objects": len(counts),
        "most_common_inserts": sorted(counts.items(), key=lambda kv: -kv[1])[:8],
        "objects_per_image_in_human_captions": float(np.mean(named)),
    }
    _save("data.json", stats)
    _save("pair_examples.json", pairs[:8])
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------
# stage: sft
# ---------------------------------------------------------------------------
def stage_sft(args):
    data = V.CocoVLMData()
    vlm, tok = D.build_vlm(unfreeze_last=args.unfreeze)
    n_train = sum(p.numel() for p in D.trainable(vlm))
    print(f"trainable parameters: {n_train / 1e6:.2f}M "
          f"of {sum(p.numel() for p in vlm.parameters()) / 1e6:.1f}M")
    t0 = time.time()
    hist = D.sft(vlm, tok, data, data.train_ids, args.steps, bs=SFT_BS, lr=SFT_LR)
    CKPT.mkdir(exist_ok=True)
    D.save(vlm, CKPT / "base.pt")
    _save("sft.json", {"steps": args.steps, "seconds": time.time() - t0,
                       "trainable_params": n_train,
                       "loss_first50": float(np.mean(hist[:50])),
                       "loss_last50": float(np.mean(hist[-50:])),
                       "history": hist[::5]})


# ---------------------------------------------------------------------------
# stage: train (one preference arm)
# ---------------------------------------------------------------------------
def _batch_for(tok, data, rows, key_a="chosen", key_b="rejected"):
    ids = np.array([r["image"] for r in rows])
    feats = data.image_tokens(ids)
    ba = V.make_batch(tok, [D.CAPTION_PROMPT] * len(rows),
                      [r[key_a] for r in rows], n_img=V.CLIP_TOKENS)
    bb = V.make_batch(tok, [D.CAPTION_PROMPT] * len(rows),
                      [r[key_b] for r in rows], n_img=V.CLIP_TOKENS)
    return ba, bb, feats


def stage_train(args):
    arm = args.arm
    assert arm in ("dpo", "dpo-lennorm", "sft-chosen"), arm
    data = V.CocoVLMData()
    pairs = json.loads(_pairs_path().read_text())
    rng = np.random.default_rng(1)
    vlm, tok = D.build_vlm(unfreeze_last=args.unfreeze)
    D.load_into(vlm, CKPT / "base.pt")

    if arm == "sft-chosen":
        # Same optimiser, same steps, same images, same number of scored
        # captions -- the only thing missing is the rejected half.
        texts = {}
        for p in pairs:
            texts.setdefault(p["image"], p["chosen"])
        ids = np.array(sorted(texts))
        t0 = time.time()
        hist = D.sft(vlm, tok, data, ids, args.steps, bs=DPO_BS * 2, lr=DPO_LR,
                     seed=1, texts=texts, log_every=50)
        D.save(vlm, CKPT / f"{arm}.pt")
        _save(f"train_{arm}.json", {"arm": arm, "steps": args.steps,
                                    "seconds": time.time() - t0,
                                    "loss_first": float(np.mean(hist[:20])),
                                    "loss_last": float(np.mean(hist[-20:]))})
        return

    # --- DPO ---------------------------------------------------------------
    # The reference model is this same checkpoint, frozen. Because it never
    # changes, its log-probabilities are computed once, up front, and cached.
    order = rng.permutation(len(pairs))[:args.steps * DPO_BS]
    rows = [pairs[i] for i in order]
    print(f"precomputing reference log-probabilities for {len(rows)} pairs",
          flush=True)
    ref = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(rows), 16):
            chunk = rows[k:k + 16]
            ba, bb, feats = _batch_for(tok, data, chunk)
            lc, nc = D.sequence_logp(vlm, ba, feats)
            lr_, nr = D.sequence_logp(vlm, bb, feats)
            ref += [(float(a), float(b), float(c), float(d))
                    for a, b, c, d in zip(lc, lr_, nc, nr)]
    print(f"    {time.time() - t0:.0f}s", flush=True)

    params = D.trainable(vlm)
    opt = torch.optim.AdamW(D.param_groups(vlm, DPO_LR), weight_decay=0.0,
                            betas=(0.9, 0.95))
    hist, t0 = [], time.time()
    vlm.train()
    for step in range(args.steps):
        sl = slice(step * DPO_BS, (step + 1) * DPO_BS)
        chunk = rows[sl]
        if not chunk:
            break
        for g in opt.param_groups:
            g["lr"] = V.cosine_lr(step, args.steps, g["base_lr"])
        ba, bb, feats = _batch_for(tok, data, chunk)
        pc, nc = D.sequence_logp(vlm, ba, feats)
        pr, nr = D.sequence_logp(vlm, bb, feats)
        r = torch.tensor(ref[sl], dtype=torch.float32)
        loss, rew_c, rew_r = D.dpo_loss(
            pc, pr, r[:, 0], r[:, 1], beta=BETA, n_c=nc, n_r=nr,
            length_norm=(arm == "dpo-lennorm"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append({
            "step": step, "loss": float(loss.detach()),
            "logp_chosen": float(pc.mean().detach()),
            "logp_rejected": float(pr.mean().detach()),
            "ref_chosen": float(r[:, 0].mean()), "ref_rejected": float(r[:, 1].mean()),
            "reward_chosen": float(rew_c.mean()), "reward_rejected": float(rew_r.mean()),
            "margin": float((rew_c - rew_r).mean()),
            "accuracy": float((rew_c > rew_r).float().mean()),
        })
        if step % 25 == 0 or step == args.steps - 1:
            h = hist[-1]
            print(f"    step {step:4d}  loss {h['loss']:.4f}"
                  f"  margin {h['margin']:+.3f}  acc {h['accuracy']:.2f}"
                  f"  logp_chosen {h['logp_chosen']:.1f}"
                  f"  {time.time() - t0:5.0f}s", flush=True)
    vlm.eval()
    CKPT.mkdir(exist_ok=True)
    D.save(vlm, CKPT / f"{arm}.pt")
    _save(f"train_{arm}.json", {"arm": arm, "steps": args.steps, "beta": BETA,
                                "lr": DPO_LR, "seconds": time.time() - t0,
                                "history": hist})


# ---------------------------------------------------------------------------
# stage: eval
# ---------------------------------------------------------------------------
def stage_eval(args):
    data = V.CocoVLMData()
    arms = args.arm.split(",") if args.arm else ARMS
    results = _load("eval.json") if (OUT / "eval.json").exists() else {}
    ids = np.asarray(data.val_ids[:N_EVAL])
    truths = [D.objects_in(data.captions[int(i)]) for i in ids]
    for arm in arms:
        vlm, tok = D.build_vlm(unfreeze_last=args.unfreeze)
        D.load_into(vlm, CKPT / ("base.pt" if arm == "base" else f"{arm}.pt"))
        t0 = time.time()
        caps = D.generate_captions(vlm, tok, data, ids, max_new=22, bs=25)
        met = D.chair(caps, truths)
        met["seconds"] = time.time() - t0
        met["mean_words"] = float(np.mean([len(c.split()) for c in caps]))
        met["distinct_captions"] = len(set(caps)) / len(caps)
        met["caption_loss"] = float(V.val_caption_loss(vlm, tok, data, ids[:100]))
        met["examples"] = [{"caption": c, "hallucinated": h,
                            "human": data.caption(int(i), 0)}
                           for c, h, i in list(zip(caps, met["per_caption"], ids))[:8]]
        met.pop("per_caption")
        results[arm] = met
        _save("eval.json", results)
        print(f"{arm:12s} CHAIR_i {met['chair_i']:.3f}  CHAIR_s {met['chair_s']:.3f}"
              f"  objects/caption {met['objects_per_caption']:.2f}"
              f"  words {met['mean_words']:.1f}"
              f"  caption loss {met['caption_loss']:.3f}", flush=True)


# ---------------------------------------------------------------------------
# stage: plot
# ---------------------------------------------------------------------------
def stage_plot(args):
    res = _load("eval.json")
    order = [a for a in ARMS if a in res]
    label = {"base": "base\n(SFT only)", "dpo": "DPO",
             "dpo-lennorm": "DPO\n(length-normalised)",
             "sft-chosen": "control:\nmore SFT"}

    fig, ax = ps.new_axes(7.4, 4.2)
    x = np.arange(len(order))
    ax.bar(x - 0.19, [res[a]["chair_i"] for a in order], 0.36, color=ps.SERIES[2],
           label="CHAIR$_i$  (share of named objects that are absent)")
    ax.bar(x + 0.19, [res[a]["chair_s"] for a in order], 0.36, color=ps.SERIES[3],
           label="CHAIR$_s$  (share of captions with any absent object)")
    for i, a in enumerate(order):
        ax.text(i - 0.19, res[a]["chair_i"] + 0.006, f"{res[a]['chair_i']:.3f}",
                ha="center", fontsize=8, color=ps.INK_SECONDARY)
        ax.text(i + 0.19, res[a]["chair_s"] + 0.006, f"{res[a]['chair_s']:.3f}",
                ha="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([label[a] for a in order], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5)
    ps.finish(fig, ax, "Hallucination rate on 150 held-out images", "",
              "rate", OUT / "chair.png")

    # what it cost: fewer objects named, shorter captions, worse caption loss
    fig, ax = ps.new_axes(7.4, 4.2)
    ax2 = ax.twinx()
    ps.style_axes(ax)
    ax.plot(x, [res[a]["objects_per_caption"] for a in order], "-o",
            color=ps.SERIES[0], label="objects named per caption")
    ax.plot(x, [res[a]["mean_words"] / 10 for a in order], "-o",
            color=ps.SERIES[4], label="words per caption / 10")
    ax2.plot(x, [res[a]["caption_loss"] for a in order], "-s",
             color=ps.SERIES[2], label="caption loss (right axis)")
    ax2.set_ylabel("nats / token", color=ps.INK_SECONDARY, fontsize=10)
    ax2.tick_params(colors=ps.INK_MUTED, labelsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label[a] for a in order], fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5)
    ps.finish(fig, ax, "What the fix costs", "", "count", OUT / "cost.png")

    # the training curves DPO is famous for
    for arm in ("dpo", "dpo-lennorm"):
        p = OUT / f"train_{arm}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())["history"]
        fig, ax = ps.new_axes(7.4, 4.2)
        xs = [r["step"] for r in h]
        ax.plot(xs, [r["reward_chosen"] for r in h], color=ps.SERIES[0],
                linewidth=1.6, label="implicit reward, chosen")
        ax.plot(xs, [r["reward_rejected"] for r in h], color=ps.SERIES[2],
                linewidth=1.6, label="implicit reward, rejected")
        ax.plot(xs, [r["margin"] for r in h], color=ps.SERIES[1], linewidth=2.0,
                label="margin (what DPO maximises)")
        ax.axhline(0, color=ps.BASELINE, linewidth=1.0)
        ax.legend(frameon=False, fontsize=9)
        ps.finish(fig, ax, f"{arm}: both answers get less likely, the gap grows",
                  "step", "log-probability ratio to the frozen reference",
                  OUT / f"rewards_{arm}.png")

    table = [{"arm": a, **{k: res[a][k] for k in
                           ("chair_i", "chair_s", "objects_per_caption",
                            "mean_words", "caption_loss", "distinct_captions")}}
             for a in order]
    _save("table.json", table)
    for r in table:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "sft", "train", "eval", "plot"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--unfreeze", type=int, default=4)
    args = ap.parse_args()
    if not args.steps:
        args.steps = SFT_STEPS if args.stage == "sft" else DPO_STEPS
    OUT.mkdir(exist_ok=True)
    {"data": stage_data, "sft": stage_sft, "train": stage_train,
     "eval": stage_eval, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
