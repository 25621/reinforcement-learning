"""Visual instruction tuning: stage 2 of the LLaVA recipe.

Stage 1 (project 20) taught the projector to feed images into a frozen LLM
while the only target was a plain caption. This project changes the *data*:
conversations with a question and a short answer, three kinds of them, built
from the same COCO captions. Then it asks which part of the recipe the gain
comes from, with four arms:

    stage1         project 20's projector, no instruction data at all (eval only)
    stage2         projector + LLM both trained on the instruction mix  (LLaVA-1.5)
    stage2-frozen  projector only, LLM still frozen
    blind          LLM trained on the same instructions with the image removed
                   (a learned prefix stands in for the picture)

Stages
    data    build the instruction set from COCO captions   (seconds)
    train   run the arms                                   (~4 min each)
    plot    figures

Requires project 20: run `python3 ../20-llava-from-scratch/run.py --stage data`
and `--stage align` first (the cached CLIP features and the stage-1 projector).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P20 = HERE.parent / "20-llava-from-scratch"
sys.path.insert(0, str(P20))
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
ARMS = ["stage1", "stage2", "stage2-frozen", "blind"]

# Nouns grouped by concept. A group counts as "in the image" if *any* of its
# words shows up in *any* of the five human captions -- otherwise a picture
# captioned "a man" would count as having no "person" in it.
GROUPS = {
    "person": "man woman person people child boy girl guy lady player men women kid",
    "dog": "dog puppy", "cat": "cat kitten", "horse": "horse", "cow": "cow cattle",
    "sheep": "sheep lamb", "bird": "bird duck goose", "bear": "bear",
    "elephant": "elephant", "zebra": "zebra", "giraffe": "giraffe",
    "car": "car taxi", "bus": "bus", "truck": "truck", "train": "train",
    "motorcycle": "motorcycle motorbike", "bicycle": "bicycle bike",
    "boat": "boat ship canoe", "airplane": "airplane plane jet",
    "skateboard": "skateboard", "surfboard": "surfboard",
    "table": "table desk", "chair": "chair stool", "couch": "couch sofa",
    "bed": "bed", "bench": "bench",
    "laptop": "laptop computer", "phone": "phone cellphone",
    "television": "television tv monitor screen", "clock": "clock",
    "bottle": "bottle", "cup": "cup mug", "plate": "plate", "bowl": "bowl",
    "pizza": "pizza", "cake": "cake", "sandwich": "sandwich burger",
    "banana": "banana", "apple": "apple", "orange": "orange",
    "broccoli": "broccoli", "carrot": "carrot", "donut": "donut doughnut",
    "umbrella": "umbrella", "vase": "vase", "book": "book",
    "flower": "flower flowers", "kite": "kite", "ball": "ball",
    "kitchen": "kitchen", "bathroom": "bathroom toilet", "bedroom": "bedroom",
    "street": "street road sidewalk", "beach": "beach sand shore",
    "water": "water ocean sea lake river", "snow": "snow",
    "grass": "grass field lawn", "mountain": "mountain hill",
    "tree": "tree trees forest", "building": "building buildings tower",
    "sink": "sink", "mirror": "mirror", "window": "window", "sign": "sign",
}
QUESTIONS = {
    "presence": "Is there {a} in the image? Answer yes or no.",
    "choice": "Which is in the image, {a} or {b}?",
    "describe": "Describe the image.",
}


def art(noun):
    """"a dog" but "an airplane" -- ungrammatical prompts are their own confound."""
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def build_examples(data, ids, rng, n_per_image=3):
    """Turn (image, five captions) into (question, answer) pairs.

    LLaVA's instructions were written by GPT-4 from human annotations; ours are
    written by a rule from the same kind of annotation. The point of the project
    is the *format* -- a question the model must answer -- not the author, so a
    rule is enough to test the recipe (and it is reproducible for free).
    """
    ex = []
    for i in ids:
        text = " ".join(data.captions[i]).lower()
        words = set(re.findall(r"[a-z]+", text))
        present = [g for g, syn in GROUPS.items() if words & set(syn.split())]
        absent = [g for g in GROUPS if g not in present]
        if not present or not absent:
            continue
        for _ in range(n_per_image):
            kind = ["presence", "presence", "choice", "describe"][int(rng.integers(0, 4))]
            if kind == "describe":
                ex.append((int(i), "describe", QUESTIONS["describe"],
                           data.caption(int(i), int(rng.integers(0, 5)))))
            elif kind == "presence":
                yes = bool(rng.integers(0, 2))
                q = present[int(rng.integers(0, len(present)))] if yes else \
                    absent[int(rng.integers(0, len(absent)))]
                ex.append((int(i), "presence", QUESTIONS["presence"].format(a=art(q)),
                           "Yes" if yes else "No"))
            else:
                a = present[int(rng.integers(0, len(present)))]
                b = absent[int(rng.integers(0, len(absent)))]
                first = bool(rng.integers(0, 2))
                ex.append((int(i), "choice",
                           QUESTIONS["choice"].format(a=art(a if first else b),
                                                      b=art(b if first else a)), a))
    return ex


def stage_data(args):
    OUT.mkdir(exist_ok=True)
    data = V.CocoVLMData()
    rng = np.random.default_rng(0)
    train = build_examples(data, data.train_ids, rng, args.per_image)
    val = build_examples(data, data.val_ids, np.random.default_rng(1), 3)
    counts = {k: sum(1 for e in val if e[1] == k) for k in QUESTIONS}
    yes = sum(1 for e in val if e[1] == "presence" and e[3] == "Yes")
    stats = dict(train=len(train), val=len(val), val_by_kind=counts,
                 val_presence_yes_rate=yes / max(1, counts["presence"]))
    (OUT / "dataset.json").write_text(json.dumps(stats, indent=1))
    (OUT / "examples.json").write_text(json.dumps(
        [dict(image=i, kind=k, q=q, a=a) for i, k, q, a in train[:12]], indent=1))
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def parse(kind, text, q):
    t = text.strip().lower()
    if kind == "presence":
        if t.startswith("yes"):
            return "Yes"
        if t.startswith("no"):
            return "No"
        return None
    if kind == "choice":
        opts = re.findall(r"\ban? ([a-z]+)", q.lower())
        hit = [o for o in opts if re.search(rf"\b{o}\b", t)]
        return hit[0] if len(hit) == 1 else None
    return t


def evaluate(vlm, tok, data, examples, uses_image, n_img, bs=16, max_new=6):
    """Greedy-decode every question, then score by kind.

    `uses_image` only controls whether the image *slots* hold real features or
    zeros: the prompt keeps its 49 slots either way, so the blind arm answers a
    sequence of exactly the same shape (its slots hold learned, image-independent
    vectors -- a soft prompt).
    """
    qa = [e for e in examples if e[1] != "describe"]
    hits = {k: [0, 0] for k in ("presence", "choice")}
    valid = [0, 0]
    shown = []
    for batch, idxs in V.prompt_batches(tok, [qa[i][2] for i in range(len(qa))],
                                       n_img=n_img, with_image=True, bs=bs):
        feats = (data.image_tokens([qa[i][0] for i in idxs]) if uses_image
                 else V.zero_feats(len(idxs)))
        texts = vlm.greedy_batch(tok, batch, feats, max_new=max_new)
        for i, txt in zip(idxs, texts):
            _, kind, q, gold = qa[i]
            got = parse(kind, txt, q)
            valid[0] += got is not None
            valid[1] += 1
            hits[kind][0] += int(got == gold)
            hits[kind][1] += 1
            if len(shown) < 8:
                shown.append(dict(kind=kind, q=q, gold=gold, said=txt.strip()[:60]))
    # caption loss on the describe subset
    caps = [e for e in examples if e[1] == "describe"][:160]
    tot = cnt = 0
    for i in range(0, len(caps), 16):
        part = caps[i:i + 16]
        b = V.make_batch(tok, [p[2] for p in part], [p[3] for p in part],
                         n_img=n_img, with_image=True)
        f = (data.image_tokens([p[0] for p in part]) if uses_image
             else V.zero_feats(len(part)))
        nll, c = vlm.answer_nll(b, f)
        tot += float(nll.sum())
        cnt += int(c.sum())
    return dict(presence=hits["presence"][0] / max(1, hits["presence"][1]),
                choice=hits["choice"][0] / max(1, hits["choice"][1]),
                n_presence=hits["presence"][1], n_choice=hits["choice"][1],
                format_valid=valid[0] / valid[1], caption_loss=tot / cnt,
                samples=shown)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def build_arm(arm):
    tok, llm = V.load_llm()
    kind = "prefix" if arm == "blind" else "mlp2"
    proj = V.Projector(kind, V.CLIP_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))
    if arm != "blind":
        ck = P20 / "checkpoints" / "proj_mlp.pt"
        if not ck.exists():
            raise SystemExit("run project 20's --stage align first (needs proj_mlp.pt)")
        proj.load_state_dict(torch.load(ck, weights_only=False)["state"])
    vlm = V.TinyVLM(llm, proj)
    params = list(proj.parameters())
    if arm in ("stage2", "blind"):           # unfreeze the language model too
        for p in llm.parameters():
            p.requires_grad_(True)
        params = [p for p in llm.parameters()] + params
    return tok, llm, proj, vlm, params


def train_arm(arm, data, examples, val_examples, steps, bs, lr, seed=0):
    torch.manual_seed(seed)
    tok, llm, proj, vlm, params = build_arm(arm)
    uses_image = arm != "blind"
    n_img = proj.n_tokens()
    before = evaluate(vlm, tok, data, val_examples, uses_image, n_img)
    print(f"  [{arm}] before: presence {before['presence']:.3f} "
          f"choice {before['choice']:.3f} valid {before['format_valid']:.3f} "
          f"caption {before['caption_loss']:.3f}", flush=True)
    if arm == "stage1" or steps == 0:
        return dict(arm=arm, steps=0, before=before, after=before,
                    trainable=0, ms_per_step=0.0), []

    if arm in ("stage2", "blind"):
        lr = lr * 0.05        # a pretrained LLM needs a much smaller step than a
                              # from-scratch projector: see the README callout
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    curve, t0 = [], time.time()
    for step in range(steps):
        pick = rng.integers(0, len(examples), bs)
        rows = [examples[i] for i in pick]
        b = V.make_batch(tok, [r[2] for r in rows], [r[3] for r in rows],
                         n_img=n_img, with_image=True)
        f = (data.image_tokens([r[0] for r in rows]) if uses_image
             else V.zero_feats(bs))
        for g in opt.param_groups:
            g["lr"] = V.cosine_lr(step, steps, lr)
        loss = vlm(b, f)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        curve.append((step, float(loss.detach())))
        if (step + 1) % 25 == 0:
            print(f"  [{arm}] {step + 1}/{steps} loss {float(loss.detach()):.4f} "
                  f"({(time.time() - t0) / (step + 1) * 1000:.0f} ms/step)", flush=True)
    ms = (time.time() - t0) / steps * 1000
    after = evaluate(vlm, tok, data, val_examples, uses_image, n_img)
    print(f"  [{arm}] after:  presence {after['presence']:.3f} "
          f"choice {after['choice']:.3f} valid {after['format_valid']:.3f} "
          f"caption {after['caption_loss']:.3f}  ({ms:.0f} ms/step)", flush=True)
    return dict(arm=arm, steps=steps, bs=bs, lr=lr, before=before, after=after,
                trainable=sum(p.numel() for p in params), ms_per_step=ms), curve


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    data = V.CocoVLMData()
    train = build_examples(data, data.train_ids, np.random.default_rng(0), args.per_image)
    val = build_examples(data, data.val_ids, np.random.default_rng(1), 3)
    rows, curves = [], {}
    for arm in args.arms:
        r, c = train_arm(arm, data, train, val, args.steps, args.bs, args.lr)
        rows.append(r)
        curves[arm] = c
    old = json.loads((OUT / "arms.json").read_text()) if (OUT / "arms.json").exists() else []
    keep = [r for r in old if r["arm"] not in args.arms]
    (OUT / "arms.json").write_text(json.dumps(keep + rows, indent=1))
    prev = (OUT / "curves.csv").read_text().splitlines()[1:] if (OUT / "curves.csv").exists() else []
    prev = [p for p in prev if p.split(",")[0] not in args.arms]
    lines = [f"{a},{s},{l:.4f}" for a, v in curves.items() for s, l in v]
    (OUT / "curves.csv").write_text("\n".join(["arm,step,loss"] + prev + lines) + "\n")


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {r["arm"]: r for r in json.loads((OUT / "arms.json").read_text())}
    arms = [a for a in ARMS if a in res]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    x = np.arange(len(arms))
    ax.bar(x - .2, [res[a]["after"]["presence"] for a in arms], .4,
           label="yes/no about an object")
    ax.bar(x + .2, [res[a]["after"]["choice"] for a in arms], .4,
           label="which of two objects")
    ax.axhline(.5, ls="--", c="k", lw=1, label="chance (both are balanced)")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8, rotation=12)
    ax.set_ylabel("accuracy")
    ax.set_title("Answering visual questions")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    ax.bar(x - .2, [res[a]["before"]["format_valid"] for a in arms], .4, label="before")
    ax.bar(x + .2, [res[a]["after"]["format_valid"] for a in arms], .4, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8, rotation=12)
    ax.set_ylabel("answers we can parse at all")
    ax.set_title("Instruction *following*")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    curves = {}
    for line in (OUT / "curves.csv").read_text().splitlines()[1:]:
        a, s, l = line.split(",")
        curves.setdefault(a, []).append((int(s), float(l)))
    for a, c in curves.items():
        c = np.array(c)
        if len(c) < 12:
            continue
        k = 10
        ax.plot(c[k - 1:, 0], np.convolve(c[:, 1], np.ones(k) / k, "valid"), label=a)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title("Instruction-tuning loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "instruction.png", dpi=130)
    print("wrote", OUT / "instruction.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "train", "plot"])
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--per-image", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    {"data": stage_data, "train": stage_train, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
