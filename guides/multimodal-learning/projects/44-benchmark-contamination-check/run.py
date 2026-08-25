"""Search a pretraining shard for a benchmark's test questions.

    python3 run.py --stage detect     # six detectors x four leak disguises
    python3 run.py --stage order      # why n-gram scans use n = 13
    python3 run.py --stage template   # the templated-benchmark trap and its fix
    python3 run.py --stage dose       # how much does a leak inflate a score?
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

import contam as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
LEAK_RATE = 0.5


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))


def load(name):
    return json.loads((OUT / name).read_text())


def setup(leak_rate=LEAK_RATE, seed=0):
    bank = C.bank()
    bench = C.make_benchmark(bank)
    docs, leaked = C.build_shard(bank, bench, leak_rate=leak_rate, seed=seed)
    return bank, bench, docs, C.expand_image_leaks(bench, leaked)


# ---------------------------------------------------------------------------
def stage_detect(args):
    bank, bench, docs, leaked = setup()
    lens = [len(C.toks(b["question"])) for b in bench]
    print(f"  benchmark {len(bench)} items, shard {len(docs)} documents, "
          f"{len(leaked)} leaked ({len(leaked)/len(bench):.0%})")
    counts = {f: sum(v == f for v in leaked.values()) for f in C.FLAVORS}
    print("  leaks by disguise:", counts)
    print(f"  question length: median {int(np.median(lens))} words, "
          f"{sum(l < 13 for l in lens)}/{len(lens)} shorter than 13")

    detectors = {
        "exact": lambda: C.detect_exact(bench, docs),
        "ngram-13": lambda: C.detect_ngram(bench, docs, n=13),
        "ngram-8": lambda: C.detect_ngram(bench, docs, n=8),
        "minhash-0.4": lambda: C.detect_minhash(bench, docs, thresh=0.4),
        "clip-text-0.92": lambda: C.detect_embedding(bench, docs, thresh=0.92),
        "phash-image": lambda: C.detect_image(bench, docs, bank),
    }
    rows = {"_setup": {"items": len(bench), "docs": len(docs),
                       "leaked": len(leaked), "by_flavor": counts,
                       "median_question_words": int(np.median(lens)),
                       "questions_under_13_words": sum(l < 13 for l in lens)}}
    for name, fn in detectors.items():
        t0 = time.time()
        flagged = fn()
        r = C.score_detector(flagged, leaked, bench)
        r["seconds"] = round(time.time() - t0, 2)
        rows[name] = r
        print(f"  {name:>16s}  P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  by flavor "
              + " ".join(f"{k[:4]}={v:.2f}" for k, v in
                         r["recall_by_flavor"].items())
              + f"  ({r['seconds']}s)")
    save("detect.json", rows)

    # the union of every text detector, versus adding the image one
    text_union = set()
    for name, fn in detectors.items():
        if name != "phash-image":
            text_union |= fn()
    both = text_union | detectors["phash-image"]()
    save("union.json", {
        "text_only": C.score_detector(text_union, leaked, bench),
        "text_plus_image": C.score_detector(both, leaked, bench)})
    print("  union of text detectors: recall "
          f"{C.score_detector(text_union, leaked, bench)['recall']:.3f}; "
          f"with the image hash {C.score_detector(both, leaked, bench)['recall']:.3f}")


def stage_order(args):
    """How the n in "n-gram" changes what a scan reports."""
    bank, bench, docs, leaked = setup()
    clean_docs, _ = C.build_shard(bank, bench, leak_rate=0.0, seed=0)
    rows = []
    for n in range(3, 19):
        flagged = C.detect_ngram(bench, docs, n=n)
        r = C.score_detector(flagged, leaked, bench)
        false_alarm = len(C.detect_ngram(bench, clean_docs, n=n)) / len(bench)
        rows.append({"n": n, "precision": r["precision"], "recall": r["recall"],
                     "flagged": r["flagged"], "false_alarm_clean": false_alarm})
        print(f"  n={n:>2d}  P={r['precision']:.3f} R={r['recall']:.3f}  "
              f"flagged {r['flagged']:>3d}  "
              f"false alarms on an uncontaminated shard {false_alarm:.3f}")
    save("order.json", rows)


def stage_template(args):
    """The trap: a benchmark whose questions share boilerplate."""
    bank, bench, docs, leaked = setup()
    rows = {}
    for n in (5, 8, 13):
        mask = C.template_ngrams(bench, n)
        plain = C.detect_ngram(bench, docs, n=n)
        masked = C.detect_ngram(bench, docs, n=n, mask=mask)
        rows[f"n={n}"] = {
            "template_ngrams": len(mask),
            "plain": C.score_detector(plain, leaked, bench),
            "template_removed": C.score_detector(masked, leaked, bench),
            "example_template": [" ".join(g) for g in list(mask)[:5]],
        }
        p, m = rows[f"n={n}"]["plain"], rows[f"n={n}"]["template_removed"]
        print(f"  n={n}: {len(mask)} boilerplate n-grams; "
              f"plain P={p['precision']:.3f} R={p['recall']:.3f} -> "
              f"template removed P={m['precision']:.3f} R={m['recall']:.3f}")
    # what the trap looks like at its worst: only ONE question was leaked
    one = {bench[0]["id"]: "verbatim"}
    docs_one = [d for d in docs if d["leak"] is None]
    docs_one.append({"text": f"Q: {bench[0]['question']} A: {bench[0]['answer']}",
                     "image": None, "leak": "verbatim", "item": bench[0]["id"]})
    for n in (5, 8, 13):
        flagged = C.detect_ngram(bench, docs_one, n=n)
        rows[f"single-leak n={n}"] = C.score_detector(flagged, one, bench,
                                                      flavors=["verbatim"])
        print(f"  one leaked question, n={n}: the scan flags "
              f"{len(flagged)}/{len(bench)} items")
    save("template.json", rows)


def stage_dose(args):
    """Score inflation as a function of how much leaked."""
    bank = C.bank()
    bench = C.make_benchmark(bank)
    rows = []
    for rate in (0.0, 0.1, 0.25, 0.5, 1.0):
        docs, leaked = C.build_shard(bank, bench, leak_rate=rate, seed=0)
        leaked = C.expand_image_leaks(bench, leaked)
        model = C.LookupModel(docs)
        preds = [model.answer(b) for b in bench]
        acc = float(np.mean([p.lower() == b["answer"].lower()
                             for p, b in zip(preds, bench)]))
        det13 = C.score_detector(C.detect_ngram(bench, docs, n=13), leaked, bench)
        rows.append({"leak_rate": rate, "accuracy": acc,
                     "n_leaked": len(leaked),
                     "detector_recall": det13["recall"],
                     "reworded_recall": det13["recall_by_flavor"]["reworded"],
                     "image_only_recall": det13["recall_by_flavor"]["image-only"]})
        print(f"  leak {rate:>5.0%}: lookup-model accuracy {acc:.3f}   "
              f"13-gram recall {det13['recall']:.3f}  "
              f"(reworded {det13['recall_by_flavor']['reworded']:.2f}, "
              f"image-only {det13['recall_by_flavor']['image-only']:.2f})")
    save("dose.json", rows)


def stage_plot(args):
    det = load("detect.json")
    names = [k for k in det if not k.startswith("_")]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [det[n]["precision"] for n in names], 0.4, label="precision",
           color="#4c72b0")
    ax.bar(x + 0.2, [det[n]["recall"] for n in names], 0.4, label="recall",
           color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Six detectors on the same shard")

    ax = axes[1]
    grid = np.array([[det[n]["recall_by_flavor"][f] for f in C.FLAVORS]
                     for n in names])
    ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(C.FLAVORS)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="0.2" if grid[i, j] < 0.55 else "white")
    ax.set_xticks(range(len(C.FLAVORS)))
    ax.set_xticklabels(C.FLAVORS, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Recall by disguise")
    fig.tight_layout()
    fig.savefig(OUT / "detectors.png", dpi=125)

    order = load("order.json")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ns = [r["n"] for r in order]
    ax.plot(ns, [r["precision"] for r in order], "o-", label="precision")
    ax.plot(ns, [r["recall"] for r in order], "s-", label="recall")
    ax.plot(ns, [r["false_alarm_clean"] for r in order], "^--",
            label="flagged on a clean shard")
    ax.axvline(13, ls=":", c="0.4")
    ax.text(13.15, 0.55, "n = 13\n(GPT-3's rule)", fontsize=9, color="0.3")
    ax.set_xlabel("n  (consecutive words that must match)")
    ax.set_title("Short n-grams flag everything; long ones flag only copies")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ngram_order.png", dpi=125)

    dose = load("dose.json")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    r = [d["leak_rate"] for d in dose]
    ax.plot(r, [d["accuracy"] for d in dose], "o-", color="#c44e52",
            label="score of a model that only remembers")
    ax.plot(r, [d["detector_recall"] for d in dose], "s-", color="#4c72b0",
            label="13-gram detector recall (all disguises)")
    ax.plot(r, [d["reworded_recall"] for d in dose], "^--", color="#55a868",
            label="...on reworded leaks only")
    ax.axhline(dose[0]["accuracy"], ls=":", c="0.5", lw=1)
    ax.set_xlabel("fraction of the benchmark present in the pretraining shard")
    ax.set_title("A leak lifts the score long before a scan can see all of it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dose_response.png", dpi=125)
    print("  wrote 3 figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["detect", "order", "template", "dose", "plot"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"detect": stage_detect, "order": stage_order, "template": stage_template,
     "dose": stage_dose, "plot": stage_plot}[args.stage](args)
