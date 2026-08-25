"""Video frame VLM: 8 frames in, treated as 8 pictures, questions out.

The cheapest way to make an image model watch a video is to refuse to treat it
as video: sample a handful of frames, encode each one with the frozen image
encoder you already have, and hand the whole pile to the language model. No new
architecture, no video pretraining. This project builds that, then measures the
one thing it is supposed to be bad at.

Stages
    data    render 1,200 clips and cache frozen CLIP features   (~5 min, once)
    train   one arm: video (real frames) or blind (the control)  (~6 min)
    eval    the trained model under four different inputs        (~3 min)
    plot    figures

The four eval conditions run the *same trained weights* on different inputs, so
nothing but the information in the frames changes:
    ordered    the frames as they happened
    shuffled   the same frames in a random order
    one_frame  the middle frame repeated 8 times (same token count, no motion)
    blind      the separately trained no-video control
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
import plot_style as ps  # noqa: E402
import video_lib as VD  # noqa: E402
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
DATA = HERE / "data"
N_CLIPS, N_VAL = 1000, 200
GRID = 2                       # 7x7 CLIP patches -> 2x2 = 4 tokens per frame
N_TOK = VD.FRAMES * GRID * GRID           # 32 image tokens per clip
STEPS, BS, LR = 400, 8, 3e-3


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


# ---------------------------------------------------------------------------
def build_cache(n=N_CLIPS):
    """Encode every frame once with frozen CLIP, pooled to 4 tokens.

    Why pool 49 patches down to 4: token count is the whole cost of this design.
    Eight frames at CLIP's native 49 tokens is 392 image tokens, and attention
    cost grows with the square of the sequence -- that alone would make this
    project a 40-minute job instead of a 6-minute one. Video-LLaVA and Qwen2-VL
    pool for the same reason. Our objects are 12 pixels wide on a 64-pixel
    canvas, so a 2x2 grid still says which quadrant everything is in.
    """
    path = DATA / "feats.npy"
    if path.exists():
        if len(np.load(path, mmap_mode="r")) >= n:
            return
        path.unlink()          # a cache built for fewer clips is not a cache
    clips, labels = VD.make_dataset(n, cache_dir=DATA)
    tower = V.clip_vision()
    feats = np.zeros((n, VD.FRAMES, GRID * GRID, V.CLIP_DIM), dtype=np.float16)
    t0 = time.time()
    for start in range(0, n, 32):
        block = clips[start:start + 32]
        big = np.stack([np.asarray(Image.fromarray(f).resize((224, 224),
                                                             Image.BICUBIC))
                        for clip in block for f in clip])
        got = V.encode_views(tower, big, layers=(-2,))[-2]        # (B*8, 49, 768)
        g = torch.from_numpy(got.astype(np.float32))
        b, p, d = g.shape
        s = int(p ** 0.5)
        g = torch.nn.functional.adaptive_avg_pool2d(
            g.transpose(1, 2).reshape(b, d, s, s), GRID)
        feats[start:start + len(block)] = (g.reshape(b, d, -1).transpose(1, 2)
                                           .numpy().astype(np.float16)
                                           .reshape(len(block), VD.FRAMES,
                                                    GRID * GRID, V.CLIP_DIM))
        if start % 320 == 0:
            print(f"    encoded {start}/{n} clips ({time.time() - t0:.0f}s)", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    print(f"    cache built in {time.time() - t0:.0f}s")


class VideoData:
    def __init__(self, n=N_CLIPS, n_val=N_VAL, seed=0):
        build_cache(n)
        self.clips, self.labels = VD.make_dataset(n, cache_dir=DATA)
        self.feats = np.load(DATA / "feats.npy", mmap_mode="r")[:n]
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        self.val_ids, self.train_ids = order[:n_val], order[n_val:]
        self.qa = [VD.questions(l, np.random.default_rng(1000 + i))
                   for i, l in enumerate(self.labels)]

    def tokens(self, ids, condition="ordered", seed=0):
        """(B, 32, 768) image tokens, optionally with the time axis attacked."""
        f = np.asarray(self.feats[np.asarray(ids)], dtype=np.float32)
        if condition == "shuffled":
            rng = np.random.default_rng(seed)
            for i in range(len(f)):
                f[i] = f[i][rng.permutation(VD.FRAMES)]
        elif condition == "one_frame":
            f = np.repeat(f[:, VD.FRAMES // 2:VD.FRAMES // 2 + 1], VD.FRAMES, axis=1)
        return torch.from_numpy(f.reshape(len(f), N_TOK, V.CLIP_DIM))


def zero_tokens(n):
    return torch.zeros(n, N_TOK, V.CLIP_DIM)


# ---------------------------------------------------------------------------
def stage_data():
    t0 = time.time()
    d = VideoData()
    print(f"  {len(d.train_ids)} train / {len(d.val_ids)} val clips, "
          f"{N_TOK} image tokens each ({time.time() - t0:.0f}s)")
    _save("data.json", {"clips": N_CLIPS, "frames": VD.FRAMES,
                        "tokens_per_frame": GRID * GRID, "tokens_per_clip": N_TOK,
                        "tokens_if_unpooled": VD.FRAMES * 49,
                        "train": len(d.train_ids), "val": len(d.val_ids),
                        "seconds": time.time() - t0})
    fig, axes = ps.plt.subplots(2, VD.FRAMES, figsize=(11.0, 3.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r in range(2):
        i = int(d.val_ids[r])
        for t in range(VD.FRAMES):
            axes[r, t].imshow(d.clips[i, t])
            axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
            for s in axes[r, t].spines.values():
                s.set_visible(False)
            if r == 0:
                axes[r, t].set_title(f"t={t}", fontsize=8, color=ps.INK_MUTED)
        axes[r, 0].set_ylabel(f"{d.labels[i]['mover']}\nmoves {d.labels[i]['direction']}\n"
                              f"{d.labels[i]['speed']}", fontsize=8,
                              color=ps.INK_SECONDARY, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "clips.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/clips.png")


def build_projector(arm, llm):
    if arm == "blind":
        return V.Projector("prefix", V.CLIP_DIM, llm.config.hidden_size,
                           out_rms=V.embedding_rms(llm), n_prefix=N_TOK)
    return V.Projector("mlp2", V.CLIP_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))


def stage_train(arm):
    """Train one arm. `tuned` is the only one that unfreezes the language model.

    Why an arm that unfreezes it at all: with the LLM frozen, the projector maps
    each token on its own, so every *comparison* -- between two frames, or
    between the picture and the words of the question -- has to be performed by
    a 135M model that has never seen these vectors before. That is stage 1 of
    the LLaVA recipe, and stage 1 alone is not supposed to answer questions;
    project 21 measured the same thing on images. `tuned` is stage 2.
    """
    data = VideoData()
    tok, llm = V.load_llm(freeze=(arm != "tuned"))
    torch.manual_seed(0)
    proj = build_projector(arm, llm)
    vlm = V.TinyVLM(llm, proj)
    groups = [{"params": list(proj.parameters()), "lr": LR}]
    if arm == "tuned":
        # a much smaller step for the pretrained weights: they are already good,
        # and the projector's output is still noise for the first few dozen steps
        groups.append({"params": list(llm.parameters()), "lr": LR * 0.05})
    opt = torch.optim.AdamW(groups, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(0)
    curve, t0 = [], time.time()
    for step in range(STEPS):
        ids = data.train_ids[rng.integers(0, len(data.train_ids), BS)]
        task = VD.TASKS[step % 3]
        qs, ans = zip(*[data.qa[i][task] for i in ids])
        batch = V.make_batch(tok, list(qs), list(ans), n_img=N_TOK)
        feats = zero_tokens(BS) if arm == "blind" else data.tokens(ids)
        loss = vlm(batch, feats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(float(loss.detach()))
        if step % 25 == 0:
            print(f"  step {step:3d}  loss {np.mean(curve[-25:]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    took = time.time() - t0
    CKPT.mkdir(exist_ok=True)
    torch.save(proj.state_dict(), CKPT / f"proj_{arm}.pt")
    n_train = sum(p.numel() for g in groups for p in g["params"])
    _save(f"train_{arm}.json", {"arm": arm, "steps": STEPS, "bs": BS, "lr": LR,
                                "trainable_params": int(n_train),
                                "seconds": took, "s_per_step": took / STEPS,
                                "curve": curve})
    print(f"  {arm}: {took:.0f}s ({took / STEPS:.2f} s/step)")
    if arm == "tuned":
        # the tuned arm changes the LLM too, and a 135M checkpoint is 500 MB, so
        # evaluate it here instead of writing it to disk and loading it back
        _eval_rows(V.TinyVLM(llm, proj).eval(), tok, data, arm,
                   ["ordered", "shuffled", "one_frame"])


@torch.no_grad()
def _answers(vlm, tok, data, ids, task, feats, bs=16):
    qs = [data.qa[i][task][0] for i in ids]
    out = [None] * len(ids)
    for batch, part in V.prompt_batches(tok, qs, n_img=N_TOK, bs=bs):
        got = vlm.greedy_batch(tok, batch, feats[part], max_new=3)
        for k, g in zip(part, got):
            out[k] = g.strip().lower()
    return out


def stage_eval(arm="video"):
    data = VideoData()
    tok, llm = V.load_llm()
    conditions = (["ordered", "shuffled", "one_frame"] if arm != "blind"
                  else ["blind"])
    proj = build_projector(arm, llm)
    proj.load_state_dict(torch.load(CKPT / f"proj_{arm}.pt"))
    _eval_rows(V.TinyVLM(llm, proj).eval(), tok, data, arm, conditions)


def _eval_rows(vlm, tok, data, arm, conditions):
    ids = data.val_ids
    rows = []
    for cond in conditions:
        feats = (zero_tokens(len(ids)) if cond == "blind"
                 else data.tokens(ids, cond))
        row = {"arm": arm, "condition": cond, "n": len(ids)}
        for task in VD.TASKS:
            got = _answers(vlm, tok, data, ids, task, feats)
            truth = [data.qa[i][task][1] for i in ids]
            row[task] = float(np.mean([g.startswith(t) for g, t in zip(got, truth)]))
            row[task + "_parseable"] = float(np.mean(
                [any(g.startswith(a) for a in VD.ANSWERS[task]) for g in got]))
        rows.append(row)
        print(f"  {cond}: " + "  ".join(f"{t} {row[t]:.3f}" for t in VD.TASKS),
              flush=True)
    old = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else []
    keep = [r for r in old if not (r["arm"] == arm)]
    _save("eval.json", keep + rows)


# ---------------------------------------------------------------------------
# the read-out probe: what is in the cached features, regardless of the LLM
# ---------------------------------------------------------------------------
def stage_probe():
    """Train a small MLP directly on the cached frame features.

    Why this exists: if the VLM cannot answer a question, there are two possible
    reasons -- the frames do not contain the answer, or the model could not get
    at it. A probe separates them. It reads the *same* cached CLIP features the
    VLM reads, so anything it can learn was available to the VLM too.

    It also lets us run the experiment this project is about with a read-out
    that is strong enough to show the effect: the same features are offered
    in ordered, shuffled, frame-averaged and single-frame form.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    data = VideoData()
    feats = torch.from_numpy(np.asarray(data.feats, dtype=np.float32))
    feats = (feats - feats.mean()) / feats.std()          # (N, 8, 4, 768)
    n_val = len(data.val_ids)
    rng = np.random.default_rng(0)

    views = {
        # every frame, in order -- what the model is given
        "ordered": lambda f: f.reshape(len(f), -1),
        # the same frames, order destroyed per clip
        "shuffled": lambda f: torch.stack(
            [f[i][torch.from_numpy(rng.permutation(VD.FRAMES))].reshape(-1)
             for i in range(len(f))]),
        # averaged over time: the honest form of "pool the frames"
        "time_averaged": lambda f: f.mean(1).reshape(len(f), -1),
        # one frame, repeated information only
        "one_frame": lambda f: f[:, VD.FRAMES // 2].reshape(len(f), -1),
    }
    # NOTE the first task is not the VLM's presence question. The probe never
    # sees the question text, so "is there a red ball?" is unanswerable for it;
    # we ask the query-free version of the same skill instead -- "is a triangle
    # anywhere in this clip?" -- which one frame also settles.
    labels = {
        "triangle": torch.tensor([int("triangle" in (l["mover"] + " " + l["other"]))
                                  for l in data.labels]),
        "direction": torch.tensor([list(VD.DIRECTIONS).index(l["direction"])
                                   for l in data.labels]),
        "speed": torch.tensor([list(VD.SPEEDS).index(l["speed"])
                               for l in data.labels]),
    }
    n_cls = {"triangle": 2, "direction": 4, "speed": 2}
    tr, te = data.train_ids, data.val_ids
    rows = []
    for view, fn in views.items():
        x = fn(feats)
        row = {"view": view, "inputs": int(x.shape[1]), "n": n_val}
        for task, y in labels.items():
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(),
                              nn.Linear(256, n_cls[task]))
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            r = np.random.default_rng(0)
            for _ in range(600):
                i = tr[r.integers(0, len(tr), 128)]
                F.cross_entropy(m(x[i]), y[i]).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                row[task] = float((m(x[te]).argmax(1) == y[te]).float().mean())
        rows.append(row)
        print(f"  {view:14s} " + "  ".join(f"{t} {row[t]:.3f}"
                                           for t in labels), flush=True)
    _save("probe.json", {"rows": rows,
                         "chance": {"triangle": 0.5, "direction": 0.25,
                                    "speed": 0.5},
                         "note": "2-layer MLP on the same cached CLIP features "
                                 "the VLM reads"})

    fig, ax = ps.new_axes(9.0, 4.0)
    order = list(views)
    x = np.arange(len(order))
    w = 0.26
    chance = {"triangle": 0.5, "direction": 0.25, "speed": 0.5}
    for k, task in enumerate(chance):
        vals = [r[task] for r in rows]
        ax.bar(x + (k - 1) * w, vals, w - 0.02, color=ps.SERIES[k],
               label=f"{task} (chance {chance[task]:.2f})")
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                    color=ps.INK_SECONDARY)
    for k, task in enumerate(chance):
        ax.plot([x[0] + (k - 1) * w - w / 2, x[-1] + (k - 1) * w + w / 2],
                [chance[task]] * 2, color=ps.BASELINE, ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["8 frames\nin order", "8 frames\nshuffled",
                        "8 frames\naveraged", "1 frame"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ps.finish(fig, ax, "what the sampled frames contain (read-out probe)", "",
              "accuracy on 200 held-out clips", OUT / "probe.png")


def stage_plot():
    rows = json.loads((OUT / "eval.json").read_text())
    want = [("tuned", "ordered"), ("tuned", "shuffled"), ("tuned", "one_frame"),
            ("video", "ordered"), ("blind", "blind")]
    rows = [r for key in want for r in rows
            if (r["arm"], r["condition"]) == key]
    label = {("tuned", "ordered"): "8 frames\nin order",
             ("tuned", "shuffled"): "8 frames\nshuffled",
             ("tuned", "one_frame"): "1 frame\nrepeated",
             ("video", "ordered"): "8 frames, but\nLLM kept frozen",
             ("blind", "blind"): "no video\n(trained control)"}
    fig, ax = ps.new_axes(10.0, 4.2)
    x = np.arange(len(rows))
    w = 0.26
    for k, task in enumerate(VD.TASKS):
        vals = [r[task] for r in rows]
        ax.bar(x + (k - 1) * w, vals, w - 0.02, color=ps.SERIES[k],
               label=f"{task} (chance {VD.CHANCE[task]:.2f})")
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                    color=ps.INK_SECONDARY)
    for k, task in enumerate(VD.TASKS):
        ax.plot([x[0] + (k - 1) * w - w / 2, x[-1] + (k - 1) * w + w / 2],
                [VD.CHANCE[task]] * 2, color=ps.BASELINE, ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label[(r["arm"], r["condition"])] for r in rows],
                       fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ps.finish(fig, ax, "every arm answers at chance, whatever the frames say "
              "(first three bars: one model, three inputs)", "",
              "accuracy on 200 held-out clips", OUT / "results.png")


STAGES = {"data": stage_data, "probe": stage_probe, "plot": stage_plot}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "probe", "train", "eval", "plot"])
    p.add_argument("--arm", default="tuned",
                   choices=["tuned", "video", "blind"])
    a = p.parse_args()
    if a.stage == "train":
        stage_train(a.arm)
    elif a.stage == "eval":
        stage_eval(a.arm)
    else:
        STAGES[a.stage]()
