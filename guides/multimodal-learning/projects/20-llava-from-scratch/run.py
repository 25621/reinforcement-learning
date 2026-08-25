"""LLaVA from scratch: frozen CLIP + frozen SmolLM2-135M + a trainable projector.

Stages
    data    download 3,000 COCO images, encode once with frozen CLIP  (~3 min)
    align   stage-1 alignment: the MLP projector and the prefix control
    layer   which CLIP layer to tap: penultimate (LLaVA's choice) vs last
    samples greedy captions for a few held-out images (needs `align` first)
    plot    figures from the JSON/CSV

Everything reusable lives in vlm_lib.py, which projects 21-25 import.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import vlm_lib as V

OUT = Path(__file__).resolve().parent / "outputs"
CKPT = Path(__file__).resolve().parent / "checkpoints"
ARMS = {"mlp": "mlp2", "prefix": "prefix"}


def frozen_floor(data, val_n=200):
    """The frozen LLM's caption loss with no image tokens in the prompt at all.

    This is the "language prior" floor: how well SmolLM2 predicts a COCO caption
    from the instruction alone, before we add anything.
    """
    tok, llm = V.load_llm()
    vlm = V.TinyVLM(llm, V.Projector("prefix", V.CLIP_DIM, llm.config.hidden_size))
    tot = n = 0
    ids = data.val_ids[:val_n]
    for i in range(0, len(ids), 16):
        chunk = ids[i:i + 16]
        b = V.make_batch(tok, [V.INSTRUCTION] * len(chunk),
                         [data.caption(j, 0) for j in chunk], with_image=False)
        nll, cnt = vlm.answer_nll(b, None)
        tot += float(nll.sum())
        n += int(cnt.sum())
    return tot / n


def save(rows, curves, names, res_file, curve_file):
    old = json.loads((OUT / res_file).read_text()) if (OUT / res_file).exists() else []
    keep = [r for r in old if r["arm"] not in names]
    (OUT / res_file).write_text(json.dumps(keep + rows, indent=1))
    prev = ((OUT / curve_file).read_text().splitlines()[1:]
            if (OUT / curve_file).exists() else [])
    prev = [p for p in prev if p.split(",")[0] not in names]
    lines = [f"{a},{s},{l:.5f}" for a, c in curves.items() for s, l in c]
    (OUT / curve_file).write_text("\n".join(["arm,step,train_loss"] + prev + lines) + "\n")


def stage_align(args):
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    data = V.CocoVLMData(layer=args.layer)
    rows, curves, ctx = [], {}, None
    for arm in args.arms:
        res, curve, c = V.align_train(ARMS[arm], data, args.steps, args.bs, args.lr,
                                      tag=arm, eval_groups=args.eval_groups)
        rows.append(res)
        curves[arm] = curve
        torch.save({"kind": ARMS[arm], "layer": args.layer, "state": c[2].state_dict()},
                   CKPT / f"proj_{arm}.pt")
        if arm == "mlp":
            ctx = c
    floor = frozen_floor(data)
    print(f"  frozen LLM, no image tokens: val {floor:.4f}")
    (OUT / "floor.json").write_text(json.dumps(dict(frozen_no_image=floor), indent=1))
    save(rows, curves, list(args.arms), "align.json", "align_curves.csv")
    if ctx is not None:
        write_samples(ctx[0], ctx[1], data)


def stage_layer(args):
    """Same projector, same steps, two different CLIP layers as the source."""
    OUT.mkdir(exist_ok=True)
    rows, curves = [], {}
    for layer in ("penult", "last"):
        data = V.CocoVLMData(layer=layer)
        res, curve, _ = V.align_train("mlp2", data, args.layer_steps, args.bs, args.lr,
                                      tag=f"layer-{layer}", eval_groups=args.eval_groups)
        res["layer"] = layer
        rows.append(res)
        curves[f"layer-{layer}"] = curve
    save(rows, curves, [r["arm"] for r in rows], "layer.json", "layer_curves.csv")


def write_samples(vlm, tok, data, n=6):
    """Greedy captions for held-out images: the VLM's, and the same frozen LLM
    with no image (which shows what "fluent but blind" looks like)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bad = [tok.convert_tokens_to_ids(V.IMAGE_TOKEN)]
    rows = []
    for i in data.val_ids[:n]:
        f = data.image_tokens([i])
        said = vlm.generate(tok, V.INSTRUCTION, f, n_img=vlm.projector.n_tokens(),
                            max_new=24, bad_ids=bad)
        blind = vlm.generate(tok, V.INSTRUCTION, None, with_image=False, max_new=24)
        rows.append(dict(id=int(i), truth=data.caption(int(i), 0), vlm=said, blind=blind))
        print(f"  {int(i)}: {said!r}", flush=True)
    (OUT / "samples.json").write_text(json.dumps(rows, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
    for ax, r in zip(axes.ravel(), rows):
        ax.imshow(np.asarray(data.thumbs[r["id"]]))
        ax.set_axis_off()
        ax.set_title(f"caption: {r['truth'][:52]}\nVLM: {r['vlm'][:52]}\n"
                     f"blind LLM: {r['blind'][:52]}", fontsize=7, loc="left")
    fig.suptitle("Stage-1 projector, greedy captions on held-out COCO images",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=130)


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {r["arm"]: r for r in json.loads((OUT / "align.json").read_text())}
    floor = json.loads((OUT / "floor.json").read_text())["frozen_no_image"]
    curves = {}
    for line in (OUT / "align_curves.csv").read_text().splitlines()[1:]:
        a, s, l = line.split(",")
        curves.setdefault(a, []).append((int(s), float(l)))
    arms = [a for a in ("prefix", "mlp") if a in res]
    color = {"mlp": "tab:green", "prefix": "tab:red", "layer-penult": "tab:green",
             "layer-last": "tab:orange"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for a in arms:
        c = np.array(curves[a])
        k = 15
        ax.plot(c[k - 1:, 0], np.convolve(c[:, 1], np.ones(k) / k, "valid"),
                label=a, color=color.get(a))
    ax.axhline(floor, ls="--", c="k", lw=1, label="frozen LLM, no image tokens")
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_title("Stage-1 alignment")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1]
    x = np.arange(len(arms))
    ax.bar(x - .2, [res[a]["val_loss_start"] for a in arms], .4, label="at init")
    ax.bar(x + .2, [res[a]["val_loss"] for a in arms], .4, label="after training")
    ax.axhline(floor, ls="--", c="k", lw=1, label="no image tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("held-out caption loss (nats/token)")
    ax.set_title("Does the picture help?")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    w = .35
    ax.bar(x - w / 2, [res[a]["choice_raw"] for a in arms], w, label="raw score")
    ax.bar(x + w / 2, [res[a]["choice_lift"] for a in arms], w, label="image-lift score")
    ax.axhline(1 / V.GALLERY, ls="--", c="k", lw=1, label="chance (1/20)")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Pick the matching caption out of {V.GALLERY}")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "align.png", dpi=130)
    print("wrote", OUT / "align.png")

    if (OUT / "layer.json").exists():
        lay = {r["arm"]: r for r in json.loads((OUT / "layer.json").read_text())}
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
        lcur = {}
        for line in (OUT / "layer_curves.csv").read_text().splitlines()[1:]:
            a, s, l = line.split(",")
            lcur.setdefault(a, []).append((int(s), float(l)))
        for a, c in lcur.items():
            c = np.array(c)
            k = 15
            axes[0].plot(c[k - 1:, 0], np.convolve(c[:, 1], np.ones(k) / k, "valid"),
                         label=a, color=color.get(a))
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("training loss")
        axes[0].set_title("Which CLIP layer to tap")
        axes[0].legend()
        axes[0].grid(alpha=.3)
        names = list(lay)
        axes[1].bar(np.arange(len(names)), [lay[a]["val_loss"] for a in names],
                    .5, color=[color.get(a) for a in names])
        axes[1].set_xticks(np.arange(len(names)))
        axes[1].set_xticklabels(names)
        axes[1].set_ylabel("held-out caption loss")
        axes[1].set_title("Penultimate vs last layer")
        axes[1].grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT / "layer.png", dpi=130)
        print("wrote", OUT / "layer.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["data", "align", "layer", "plot", "samples"])
    p.add_argument("--arms", nargs="+", default=["mlp", "prefix"], choices=list(ARMS))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--layer-steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--layer", default="penult")
    p.add_argument("--eval-groups", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    if args.stage == "data":
        V.build_cache()
    elif args.stage == "align":
        stage_align(args)
    elif args.stage == "layer":
        stage_layer(args)
    elif args.stage == "samples":
        data = V.CocoVLMData(layer=args.layer)
        tok, llm = V.load_llm()
        ck = torch.load(CKPT / "proj_mlp.pt", weights_only=False)
        proj = V.Projector(ck["kind"], V.CLIP_DIM, llm.config.hidden_size,
                           out_rms=V.embedding_rms(llm))
        proj.load_state_dict(ck["state"])
        write_samples(V.TinyVLM(llm, proj), tok, data)
    else:
        stage_plot(args)


if __name__ == "__main__":
    main()
