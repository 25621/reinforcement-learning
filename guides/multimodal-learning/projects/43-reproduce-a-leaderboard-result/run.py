"""Reproduce a published benchmark number, then find out what it takes to miss it.

The "leaderboard entry" we go after is project 42's headline for
`smolvlm-256m` on `mmbench-mini`. Using a neighbouring project instead of a real
paper is not a shortcut -- it is what makes the exercise sharp. We have the
original code, the original seed and the original machine, so any difference we
see afterwards is caused by exactly the knob we turned, and nothing else.

    python3 run.py --stage pinned    # everything specified  (~2 min)
    python3 run.py --stage sweep     # one unstated choice at a time (~7 min)
    python3 run.py --stage subsets   # a different sample of photos (~7 min)
    python3 run.py --stage tiling    # the preprocessing nobody writes down (~7 min)
    python3 run.py --stage report
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "42-run-a-vlm-evaluation-harness"))
import harness as H  # noqa: E402

OUT = HERE / "outputs"
ARMS = OUT / "arms.json"
TASK = "mmbench-mini"
PUBLISHED = (HERE.parent / "42-run-a-vlm-evaluation-harness" / "outputs"
             / "results.json")

SUFFIXES = {
    "pinned": H.MCQ_SUFFIX,
    "no-suffix": "",
    "verbose-suffix": ("\nPlease look at the image, choose the correct option "
                       "from the list above, and reply with its letter."),
    "cot-suffix": "\nThink step by step, then give the letter of the answer.",
}


def load_arms():
    return json.loads(ARMS.read_text()) if ARMS.exists() else {}


def save_arms(a):
    OUT.mkdir(exist_ok=True)
    ARMS.write_text(json.dumps(a, indent=1))


def build_docs(bank, offset=0, circular=True, suffix="pinned", items=30):
    """Rebuild project 42's `mmbench-mini` questions, optionally with one thing
    changed. Going through `H.build_all` rather than re-implementing the task is
    the point: the *only* way to guarantee the questions are identical is to run
    the same constructor with the same seed."""
    tasks = H.build_all(bank, items={**H.ITEMS, TASK: items}, offset=offset,
                        circular=circular)
    task, docs = tasks[TASK]
    if suffix != "pinned":
        new = SUFFIXES[suffix]
        docs = [{**d, "question": d["question"].replace(H.MCQ_SUFFIX, new)}
                for d in docs]
    return task, docs


def run_arm(name, bank, model=None, **cfg):
    task, docs = build_docs(bank, offset=cfg.get("offset", 0),
                            circular=cfg.get("circular", True),
                            suffix=cfg.get("suffix", "pinned"),
                            items=cfg.get("items", 30))
    if model is None:
        model = H.SmolVLMModel(cfg.get("repo",
                                       "HuggingFaceTB/SmolVLM-256M-Instruct"),
                               split_images=cfg.get("split_images", False),
                               mode=cfg.get("mode", "generate"))
    t0 = time.time()
    images = task.images(bank, docs)
    raw = model.predict(task, docs, images, verbose=False)
    metrics, _ = task.grade(docs, raw)
    metrics.update({"seconds": round(time.time() - t0, 1), "config": cfg,
                    "sample": raw[:6]})
    arms = load_arms()
    arms[name] = metrics
    save_arms(arms)
    print(f"  {name:>18s}  acc={metrics['accuracy']:.3f} "
          f"circular={metrics['circular_accuracy']:.3f} "
          f"strict={metrics['strict_accuracy']:.3f} "
          f"unparsed={metrics['unparsed']}  ({metrics['seconds']:.0f}s)")
    return metrics


# ---------------------------------------------------------------------------
def stage_pinned(args):
    bank = H.Bank()
    published = json.loads(PUBLISHED.read_text())["smolvlm-256m"][TASK]
    print(f"  published: accuracy {published['accuracy']:.4f}, "
          f"circular {published['circular_accuracy']:.4f}")
    got = run_arm("pinned", bank)
    same = all(abs(got[k] - published[k]) < 1e-12
               for k in ("accuracy", "circular_accuracy", "strict_accuracy"))
    same_answers = got["sample"] == published["_raw"][:6]
    print(f"  identical to the published run: {same} "
          f"(first six raw answers match: {same_answers})")
    arms = load_arms()
    arms["_published"] = {k: published[k] for k in
                          ("accuracy", "circular_accuracy", "strict_accuracy",
                           "unparsed", "n")}
    arms["_reproduction"] = {"exact_match": bool(same),
                             "raw_answers_match": bool(same_answers)}
    save_arms(arms)


def stage_sweep(args):
    bank = H.Bank()
    model = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct")
    for suffix in ("no-suffix", "verbose-suffix", "cot-suffix"):
        run_arm(suffix, bank, model=model, suffix=suffix)
    lik = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct",
                         mode="likelihood")
    run_arm("likelihood", bank, model=lik, mode="likelihood")


def stage_subsets(args):
    """How much of a "reproduction gap" is just a different sample of photos?"""
    bank = H.Bank()
    model = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct")
    for off in (30, 60, 90, 120):
        run_arm(f"photos-{off}", bank, model=model, offset=off)


def stage_tiling(args):
    """The preprocessing choice papers almost never state."""
    bank = H.Bank()
    small = {"items": 12}
    plain = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct",
                           split_images=False)
    run_arm("subset-1view", bank, model=plain, **small)
    tiled = H.SmolVLMModel("HuggingFaceTB/SmolVLM-256M-Instruct",
                           split_images=True)
    run_arm("subset-5views", bank, model=tiled, split_images=True, **small)


def stage_report(args):
    arms = load_arms()
    pub = arms["_published"]
    names = [k for k in arms if not k.startswith("_")]

    # ---- the envelope --------------------------------------------------------
    readings = {
        "as published": arms["pinned"]["accuracy"],
        "strict letter parser": arms["pinned"]["strict_accuracy"],
        "no answer-format line": arms["no-suffix"]["accuracy"],
        "wordier answer-format line": arms["verbose-suffix"]["accuracy"],
        "\"think step by step\"": arms["cot-suffix"]["accuracy"],
        "likelihood scoring": arms["likelihood"]["accuracy"],
        "circular scoring": arms["pinned"]["circular_accuracy"],
    }
    subsets = [arms[k]["accuracy"] for k in names if k.startswith("photos-")]
    subsets.append(arms["pinned"]["accuracy"])
    spread = {"subset_mean": float(np.mean(subsets)),
              "subset_std": float(np.std(subsets, ddof=1)),
              "subset_min": float(np.min(subsets)),
              "subset_max": float(np.max(subsets)),
              "standard_error_of_one_run": float(np.std(subsets, ddof=1))}
    report = {"published": pub, "reproduction": arms["_reproduction"],
              "readings": readings, "subset_spread": spread,
              "envelope": [min(readings.values()), max(readings.values())]}
    if "subset-1view" in arms:
        report["tiling"] = {
            "one view": arms["subset-1view"]["accuracy"],
            "four tiles + thumbnail": arms["subset-5views"]["accuracy"],
            "seconds one view": arms["subset-1view"]["seconds"],
            "seconds tiled": arms["subset-5views"]["seconds"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ks = list(readings)
    vs = [readings[k] for k in ks]
    colors = ["#2b6b8f" if k == "as published" else "#a3bcd0" for k in ks]
    ax.barh(range(len(ks)), vs, color=colors)
    ax.axvline(pub["accuracy"], color="#c44e52", lw=2,
               label=f"published {pub['accuracy']:.3f}")
    lo = pub["accuracy"] - 2 * spread["subset_std"]
    hi = pub["accuracy"] + 2 * spread["subset_std"]
    ax.axvspan(lo, hi, color="#c44e52", alpha=0.12,
               label="±2 s.d. of the photo sample")
    for i, v in enumerate(vs):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("accuracy on the same 120 questions, same model, same photos")
    ax.set_title("One published number, seven defensible ways to read the recipe")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "envelope.png", dpi=125)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["pinned"] + [k for k in names if k.startswith("photos-")]
    vals = [arms[k]["accuracy"] for k in labels]
    ax.bar(range(len(labels)), vals, color="#4c72b0")
    ax.axhline(float(np.mean(vals)), ls="--", c="0.35",
               label=f"mean {np.mean(vals):.3f}  (s.d. {np.std(vals, ddof=1):.3f})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["photos 0–29"] + [f"photos {k.split('-')[1]}–"
                                          f"{int(k.split('-')[1])+29}"
                                          for k in labels[1:]],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Same recipe, different 30 photos: the noise floor")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "subsets.png", dpi=125)
    print("  wrote 2 figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pinned", "sweep", "subsets", "tiling", "report"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"pinned": stage_pinned, "sweep": stage_sweep, "subsets": stage_subsets,
     "tiling": stage_tiling, "report": stage_report}[args.stage](args)
