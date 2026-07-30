"""Run the mini evaluation harness.

    python3 run.py --stage data                      # photos + questions (~1 min)
    python3 run.py --stage run --model chance,generic-caption,clip-zeroshot
    python3 run.py --stage run --model blind-llm
    python3 run.py --stage run --model smolvlm-256m
    python3 run.py --stage run --model smolvlm-500m
    python3 run.py --stage plot
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
RESULTS = OUT / "results.json"
ORDER = ["pope-random", "pope-adversarial", "mmbench-mini", "caption-match",
         "ocr-mini", "count-mini", "spatial-mini", "caption-gen"]
SHORT = {"pope-random": "POPE-r", "pope-adversarial": "POPE-a",
         "mmbench-mini": "MCQ-obj", "caption-match": "MCQ-cap",
         "ocr-mini": "OCR", "count-mini": "Count", "spatial-mini": "Spatial",
         "caption-gen": "Caption"}
# What a system that cannot see gets by guessing. `ocr-mini` is 1/25 because a
# guesser that knows the word list has 25 words to choose from; a guesser that
# does not know the list scores 0. Printing this next to every column is the
# cheapest defence against reading 0.26 on a four-way task as a real result.
CHANCE = {"pope-random": 0.5, "pope-adversarial": 0.5, "mmbench-mini": 0.25,
          "caption-match": 0.25, "ocr-mini": 1 / len(H.WORDS),
          "count-mini": 0.25, "spatial-mini": 0.5, "caption-gen": 0.0}


def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def headline(task_name, row):
    if row.get("skipped"):
        return None
    return row["cider"] if task_name == "caption-gen" else row["accuracy"]


# ---------------------------------------------------------------------------
def stage_data(args):
    bank = H.Bank()
    tasks = H.build_all(bank)
    comp, examples = {}, {}
    for name, (t, docs) in tasks.items():
        photos = {d["image"] for d in docs} - {-1}
        comp[name] = {"kind": t.kind, "questions": len(docs),
                      "photos": len(photos) or "synthetic canvas",
                      "primary": t.primary, "chance": CHANCE[name]}
        examples[name] = [{"question": d["question"], "target": d["target"]}
                          for d in docs[:2]]
    H.save(OUT / "tasks.json", comp)
    H.save(OUT / "examples.json", examples)
    print(json.dumps(comp, indent=1))

    # one picture of each task's input, so the README can show what is asked
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.4))
    for ax, name in zip(axes.ravel(), ORDER):
        t, docs = tasks[name]
        img = t.images(bank, docs[:1])[0]
        ax.imshow(img)
        ax.set_axis_off()
        q = docs[0]["question"].split("\n")[0]
        ax.set_title(f"{name}\n{q[:46]}", fontsize=8.5)
    fig.suptitle("One question from each of the eight tasks", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "tasks.png", dpi=115)
    print(f"  wrote {OUT/'tasks.png'}")


def build_model(name):
    if name == "chance":
        return H.Chance()
    if name == "generic-caption":
        return H.GenericCaption()
    if name == "blind-llm":
        return H.BlindLLM()
    if name == "clip-zeroshot":
        return H.ClipMatch()
    if name == "smolvlm-256m":
        return H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct")
    if name == "smolvlm-500m":
        return H.SmolVLMModel("HuggingFaceTB/SmolVLM-500M-Instruct")
    raise SystemExit(f"unknown model {name}")


def stage_run(args):
    bank = H.Bank()
    tasks = H.build_all(bank)
    results = load_results()
    for name in args.model.split(","):
        model = build_model(name)
        if name == "clip-zeroshot":
            # Calibrate the yes/no threshold on photos 120-170, which no task
            # asks about. Tuning it on the test questions would report the best
            # of ninety-seven tries as if it were a prediction.
            dev = H.Pope("random").build(bank, np.arange(120, 170),
                                         np.random.default_rng(7))
            info = model.calibrate(dev, H.Pope("random").images(bank, dev))
            print(f"  clip threshold {info['threshold']:.4f} "
                  f"(dev acc {info['dev_accuracy']:.3f})")
            results.setdefault("_meta", {})["clip_calibration"] = info
        t0 = time.time()
        results[model.name] = H.run(model, tasks, bank)
        print(f"  {model.name}: {time.time() - t0:.0f}s total")
        H.save(RESULTS, results)


def stage_ocr_probe(args):
    """Both VLMs scored a flat 1.000 on `ocr-mini`, so the task's difficulty
    knobs were all sitting at their easiest settings. Turn them one at a time
    and find out which one the score actually depends on."""
    bank = H.Bank()
    model = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct")
    arms = [
        ("words 40px, plaque", dict(size=40)),
        ("words 20px, plaque", dict(size=20)),
        ("words 10px, plaque", dict(size=10)),
        ("words 34-46px, NO plaque", dict(plaque=False)),
        ("nonwords 34-46px, plaque", dict(nonwords=True)),
        ("nonwords 10px, plaque", dict(nonwords=True, size=10)),
    ]
    rows = []
    for label, cfg in arms:
        t = H.Ocr(**cfg)
        docs = t.build(bank, np.arange(30), np.random.default_rng(
            sum(ord(c) for c in t.name)))
        raw = model.predict(t, docs, t.images(bank, docs), verbose=False)
        m, preds = t.grade(docs, raw)
        rows.append({"arm": label, "accuracy": m["accuracy"], "n": m["n"],
                     "example": {"truth": docs[0]["target"], "said": preds[0]}})
        print(f"  {label:<26s} accuracy {m['accuracy']:.3f}"
              f"   (e.g. '{docs[0]['target']}' -> '{preds[0]}')")
    H.save(OUT / "ocr_probe.json", rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [r["arm"] for r in rows]
    ax.barh(range(len(rows)), [r["accuracy"] for r in rows],
            color=["#7fb3d5"] * 4 + ["#c44e52"] * 2)
    for i, r in enumerate(rows):
        ax.text(r["accuracy"] + 0.01, i, f"{r['accuracy']:.2f}", va="center",
                fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("exact-match accuracy (smolvlm-256m, 30 questions per arm)")
    ax.set_title("Which knob does `ocr-mini` actually depend on?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ocr_probe.png", dpi=125)


def stage_plot(args):
    results = load_results()
    models = [m for m in results if not m.startswith("_")]
    table = {}
    for m in models:
        table[m] = {t: headline(t, results[m].get(t, {})) for t in ORDER}
    H.save(OUT / "table.json", table)

    # ---- 1. the score table as a heat map ------------------------------------
    grid = np.full((len(models), len(ORDER)), np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(ORDER):
            v = table[m][t]
            if v is not None:
                grid[i, j] = v / 10.0 if t == "caption-gen" else v
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(models) + 2.4))
    ax.imshow(np.ma.masked_invalid(grid), cmap="YlGnBu", vmin=0, vmax=1,
              aspect="auto")
    for i, m in enumerate(models):
        for j, t in enumerate(ORDER):
            v = table[m][t]
            txt = "n/a" if v is None else (f"{v:.2f}" if t != "caption-gen"
                                           else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="0.25" if (np.isnan(grid[i, j]) or grid[i, j] < 0.55)
                    else "white")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Eight tasks, one harness  (accuracy; CIDEr for Caption)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "scoreboard.png", dpi=125)

    # ---- 2. how much each task is solvable without the image -----------------
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(ORDER))
    w = 0.26
    for k, (m, c) in enumerate([("chance", "0.7"), ("blind-llm", "#c26a3d"),
                                ("smolvlm-256m", "#2b6b8f")]):
        if m not in table:
            continue
        vals = [(table[m][t] or 0) / (10 if t == "caption-gen" else 1)
                for t in ORDER]
        ax.bar(x + (k - 1) * w, vals, w, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in ORDER])
    ax.set_ylabel("score (CIDEr/10 for Caption)")
    ax.set_title("What is left once you subtract luck and language priors")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "blind_baseline.png", dpi=125)

    # ---- 3. circular evaluation and the parser -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mcq = ["mmbench-mini", "caption-match"]
    vlms = [m for m in models if m.startswith("smolvlm") or m == "blind-llm"
            or m == "clip-zeroshot"]
    x = np.arange(len(vlms))
    for j, t in enumerate(mcq):
        ax = axes[j]
        plain = [results[m].get(t, {}).get("accuracy", np.nan) for m in vlms]
        circ = [results[m].get(t, {}).get("circular_accuracy", np.nan)
                for m in vlms]
        ax.bar(x - 0.2, plain, 0.4, label="per-question", color="#7fb3d5")
        ax.bar(x + 0.2, circ, 0.4, label="circular (all 4 rotations)",
               color="#1f5f8b")
        ax.axhline(0.25, ls="--", c="0.4", lw=1, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(vlms, rotation=20, ha="right", fontsize=8)
        ax.set_title(t)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rotating the options is a different exam", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "circular.png", dpi=125)

    # ---- 4. rank instability across tasks ------------------------------------
    ranked = [m for m in models if m not in ("generic-caption",)]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for m in ranked:
        ys = []
        for t in ORDER:
            vals = {mm: table[mm][t] for mm in ranked if table[mm][t] is not None}
            if table[m][t] is None:
                ys.append(np.nan)
            else:
                order = sorted(vals, key=lambda k: -vals[k])
                ys.append(order.index(m) + 1)
        ax.plot(range(len(ORDER)), ys, "o-", label=m)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[t] for t in ORDER])
    ax.invert_yaxis()
    ax.set_ylabel("rank on this task (1 = best)")
    ax.set_title("No single ranking: the winner changes with the task")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ranks.png", dpi=125)
    print("  wrote 4 figures")

    for m in models:
        row = " ".join(f"{SHORT[t]}={'  n/a' if table[m][t] is None else f'{table[m][t]:5.3f}'}"
                       for t in ORDER)
        print(f"{m:>16s}  {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "run", "ocr-probe", "plot"])
    ap.add_argument("--model", default="chance")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"data": stage_data, "run": stage_run, "ocr-probe": stage_ocr_probe,
     "plot": stage_plot}[args.stage](args)
