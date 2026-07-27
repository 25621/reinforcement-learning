"""Hallucination eval: build a small POPE benchmark and run several models on it.

Systems under test
    always-yes            the degenerate baseline that exposes an unbalanced test
    clip-zeroshot         frozen CLIP + a calibrated threshold, no language at all
    smolvlm-256m          a real open VLM, asked the question and read literally
    smolvlm-256m (lik.)   the same model, scored by "Yes" vs "No" probability
    smolvlm-500m          a bigger sibling of the same family
    tinyvlm-sft / -dpo    project 40's captioner before and after preference
                          training, scored by likelihood

Stages
    data      build the three splits and download the photos       (~1 min)
    run       one system  (python3 run.py --stage run --model smolvlm-256m)
    prompts   prompt-wording sensitivity for one model             (~6 min)
    plot      figures and tables                                   (~10 s)
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
sys.path.insert(0, str(PROJECTS / "40-multimodal-dpo"))
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402
import pope  # noqa: E402
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
N_IMAGES = 110
N_DEV = 40
MODELS = ["always-yes", "clip-zeroshot", "smolvlm-256m", "smolvlm-256m-lik",
          "smolvlm-500m", "tinyvlm-sft", "tinyvlm-dpo"]
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def _load(name):
    return json.loads((OUT / name).read_text())


def bench(prompt="yesno"):
    data = V.CocoVLMData()
    ids = [int(i) for i in data.val_ids[:N_IMAGES]]
    qs = pope.build(data, ids, prompt=prompt)
    return data, qs


def dev_bench(prompt="yesno"):
    """A separate, disjoint set of images used only to pick CLIP's threshold.
    Tuning a threshold on the test questions and then reporting the accuracy it
    achieved there is not a prediction about new data -- it is the best of many
    tries, and it would flatter CLIP against every model that gets no such
    knob."""
    data = V.CocoVLMData()
    ids = [int(i) for i in data.val_ids[N_IMAGES:N_IMAGES + N_DEV]]
    return data, pope.build(data, ids, prompt=prompt, seed=7)


# ---------------------------------------------------------------------------
def stage_data(args):
    data, qs = bench()
    used = sorted({r["image"] for r in qs["random"]})
    print(f"downloading {len(used)} photos at {pope.FULL_SIZE}px", flush=True)
    pope.fetch_images(used)
    stats = {"images": len(used), "per_split": {s: len(qs[s]) for s in pope.SPLITS},
             "yes_rate_by_construction":
                 {s: float(np.mean([r["answer"] == "yes" for r in qs[s]]))
                  for s in pope.SPLITS}}
    # How often the three rules pick a *different* object -- if they agreed the
    # three splits would be the same exam three times.
    per_img = {}
    for s in pope.SPLITS:
        for r in qs[s]:
            if r["answer"] == "no":
                per_img.setdefault(r["image"], {})[s] = r["object"]
    distinct = [len(set(v.values())) for v in per_img.values()]
    stats["distinct_negatives_per_image"] = float(np.mean(distinct))
    counts = {}
    for s in pope.SPLITS:
        c = {}
        for r in qs[s]:
            if r["answer"] == "no":
                c[r["object"]] = c.get(r["object"], 0) + 1
        counts[s] = sorted(c.items(), key=lambda kv: -kv[1])[:6]
    stats["most_asked_negatives"] = counts
    _save("data.json", stats)
    _save("examples.json", [qs[s][i] for s in pope.SPLITS for i in (0, 1)])
    print(json.dumps(stats, indent=1))


def _make(name):
    if name == "always-yes":
        return pope.AlwaysYes()
    if name == "clip-zeroshot":
        return pope.ClipZeroShot()
    if name == "smolvlm-256m":
        return pope.SmolVLM("HuggingFaceTB/SmolVLM-256M-Instruct", "generate")
    if name == "smolvlm-256m-lik":
        return pope.SmolVLM("HuggingFaceTB/SmolVLM-256M-Instruct", "likelihood")
    if name == "smolvlm-500m":
        return pope.SmolVLM("HuggingFaceTB/SmolVLM-500M-Instruct", "generate")
    if name == "tinyvlm-sft":
        return pope.TinyVLM(PROJECTS / "40-multimodal-dpo/checkpoints/base.pt",
                            "tinyvlm-sft")
    if name == "tinyvlm-dpo":
        return pope.TinyVLM(PROJECTS / "40-multimodal-dpo/checkpoints/dpo.pt",
                            "tinyvlm-dpo")
    raise ValueError(name)


def stage_run(args):
    data, qs = bench()
    results = _load("results.json") if (OUT / "results.json").exists() else {}
    for name in (args.model.split(",") if args.model else MODELS):
        sys_ = _make(name)
        extra = {}
        for split in pope.SPLITS:
            recs = qs[split]
            imgs = None
            if name.startswith(("clip", "smolvlm")):
                imgs = pope.fetch_images([r["image"] for r in recs])
            if name == "clip-zeroshot" and split == pope.SPLITS[0]:
                _, dqs = dev_bench()
                dev = dqs["random"]
                extra["calibration"] = sys_.calibrate(
                    dev, pope.fetch_images([r["image"] for r in dev]))
                extra["calibration"]["dev_images"] = N_DEV
                print(f"    CLIP threshold {extra['calibration']}", flush=True)
            t0 = time.time()
            if name.startswith("tinyvlm"):
                ans = sys_.answer(recs, imgs, data=data)
            else:
                ans = sys_.answer(recs, imgs)
            s = pope.score(recs, ans)
            s["seconds"] = time.time() - t0
            results.setdefault(name, {})[split] = s
            results[name].update(extra)
            print(f"{name:20s} {split:12s} acc {s['accuracy']:.3f}"
                  f"  F1 {s['f1']:.3f}  yes-rate {s['yes_rate']:.3f}"
                  f"  ({s['seconds']:.0f}s)", flush=True)
            if split == "random":
                results[name]["answers_random"] = ans
        _save("results.json", results)


def stage_prompts(args):
    """Same model, same pictures, three ways of asking."""
    out = {}
    sys_ = _make(args.model or "smolvlm-256m")
    for style in pope.PROMPTS:
        data, qs = bench(prompt=style)
        recs = qs["random"]
        imgs = pope.fetch_images([r["image"] for r in recs])
        ans = sys_.answer(recs, imgs)
        out[style] = pope.score(recs, ans)
        out[style]["prompt"] = pope.PROMPTS[style]
        print(f"{style:8s} acc {out[style]['accuracy']:.3f}"
              f"  yes-rate {out[style]['yes_rate']:.3f}"
              f"  unparsed {out[style]['unparsed']}", flush=True)
    _save("prompts.json", {"model": sys_.name, "styles": out})


# ---------------------------------------------------------------------------
def stage_plot(args):
    res = _load("results.json")
    order = [m for m in MODELS if m in res]
    nice = {"always-yes": "always-yes", "clip-zeroshot": "CLIP\nzero-shot",
            "smolvlm-256m": "SmolVLM-256M\n(generate)",
            "smolvlm-256m-lik": "SmolVLM-256M\n(likelihood)",
            "smolvlm-500m": "SmolVLM-500M\n(generate)",
            "tinyvlm-sft": "project 40\nSFT", "tinyvlm-dpo": "project 40\nDPO"}

    fig, ax = ps.new_axes(9.0, 4.4)
    x = np.arange(len(order))
    for k, s in enumerate(pope.SPLITS):
        ax.bar(x + (k - 1) * 0.27, [res[m][s]["accuracy"] for m in order], 0.25,
               color=ps.SERIES[k], label=f"{s} negatives")
    ax.axhline(0.5, color=ps.INK_MUTED, linestyle="--", linewidth=1.1)
    ax.text(len(order) - 0.4, 0.515, "chance / always-yes", fontsize=8,
            color=ps.INK_MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in order], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Harder negatives, lower scores", "",
              "accuracy", OUT / "accuracy.png")

    fig, ax = ps.new_axes(9.0, 4.2)
    for k, s in enumerate(pope.SPLITS):
        ax.bar(x + (k - 1) * 0.27, [res[m][s]["yes_rate"] for m in order], 0.25,
               color=ps.SERIES[k], label=f"{s} negatives")
    ax.axhline(0.5, color=ps.INK_MUTED, linestyle="--", linewidth=1.1)
    ax.text(len(order) - 0.4, 0.52, "the truth: half the answers are yes",
            fontsize=8, color=ps.INK_MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in order], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "How often each model says yes", "",
              "yes-rate", OUT / "yes_rate.png")

    # precision / recall on the hardest split
    fig, ax = ps.new_axes(7.4, 4.4)
    for k, m in enumerate(order):
        s = res[m]["adversarial"]
        ax.scatter(s["recall"], s["precision"], s=70,
                   color=ps.SERIES[k % len(ps.SERIES)], zorder=3)
        ax.annotate(nice[m].replace("\n", " "), (s["recall"], s["precision"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8,
                    color=ps.INK_SECONDARY)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0.3, 1.02)
    ps.finish(fig, ax, "Adversarial split: saying yes a lot buys recall, not precision",
              "recall (present objects it confirmed)",
              "precision (of its yes answers)", OUT / "precision_recall.png")

    if (OUT / "prompts.json").exists():
        pr = _load("prompts.json")
        fig, ax = ps.new_axes(7.0, 4.0)
        styles = list(pr["styles"])
        xs = np.arange(len(styles))
        ax.bar(xs - 0.19, [pr["styles"][s]["accuracy"] for s in styles], 0.36,
               color=ps.SERIES[0], label="accuracy")
        ax.bar(xs + 0.19, [pr["styles"][s]["yes_rate"] for s in styles], 0.36,
               color=ps.SERIES[3], label="yes-rate")
        for i, s in enumerate(styles):
            ax.text(i - 0.19, pr["styles"][s]["accuracy"] + 0.01,
                    f"{pr['styles'][s]['accuracy']:.3f}", ha="center", fontsize=8,
                    color=ps.INK_SECONDARY)
        ax.axhline(0.5, color=ps.INK_MUTED, linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(styles, fontsize=9)
        ax.legend(frameon=False, fontsize=9)
        ps.finish(fig, ax, f"{pr['model']}: same questions, three wordings", "",
                  "", OUT / "prompt_sensitivity.png")

    table = []
    for m in order:
        row = {"model": m}
        for s in pope.SPLITS:
            row[f"{s}_acc"] = res[m][s]["accuracy"]
            row[f"{s}_f1"] = res[m][s]["f1"]
            row[f"{s}_yes"] = res[m][s]["yes_rate"]
        row["mean_acc"] = float(np.mean([res[m][s]["accuracy"] for s in pope.SPLITS]))
        row["drop_random_to_adversarial"] = (res[m]["random"]["accuracy"]
                                             - res[m]["adversarial"]["accuracy"])
        table.append(row)
    _save("table.json", table)
    for r in table:
        print(f"{r['model']:20s} random {r['random_acc']:.3f}"
              f"  popular {r['popular_acc']:.3f}"
              f"  adversarial {r['adversarial_acc']:.3f}"
              f"  drop {r['drop_random_to_adversarial']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "run", "prompts", "plot"])
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"data": stage_data, "run": stage_run, "prompts": stage_prompts,
     "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
