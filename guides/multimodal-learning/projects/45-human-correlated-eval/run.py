"""Measure whether the automatic judges agree with people.

    python3 run.py --stage outputs   # 100 captions from 5 systems (~3 min)
    python3 run.py --stage rate      # 3 human-derived + 4 automatic judges (~5 min)
    python3 run.py --stage analyse   # agreement, bias probes, plots
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import judge as J  # noqa: E402
import harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
DATA = Path(__file__).resolve().parent / "data"
N_IMAGES = 20
IDS = np.arange(150, 150 + N_IMAGES)      # photos no other Phase-9 task asks about


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))


def load(name):
    return json.loads((OUT / name).read_text())


def stage_outputs(args):
    bank = H.Bank()
    DATA.mkdir(exist_ok=True)
    t0 = time.time()
    outs = J.make_outputs(bank, IDS, cache=DATA / "outputs.json")
    save("captions.json", outs)
    print(f"  {sum(len(v) for v in outs.values())} outputs from "
          f"{len(outs)} systems ({time.time() - t0:.0f}s)")
    for s, v in outs.items():
        print(f"    {s:>14s}  {v[0][:70]}")


def stage_rate(args):
    bank = H.Bank()
    outs = load("captions.json")
    ratings = {}

    panel = J.human_panel(bank, IDS, outs)
    ratings.update(panel)
    print("  human panel built from 3 independent annotators")

    for repo, blind in (("HuggingFaceTB/SmolVLM-256M-Instruct", False),
                        ("HuggingFaceTB/SmolVLM-500M-Instruct", False),
                        ("HuggingFaceTB/SmolVLM-256M-Instruct", True)):
        jm = J.VlmJudge(repo, blind=blind)
        t0 = time.time()
        ratings[jm.name] = {s: [float(x) for x in
                                jm.rate(bank, IDS, outs[s])] for s in outs}
        print(f"  {jm.name}: {time.time() - t0:.0f}s")
        del jm

    for jm in (J.ClipScoreJudge(), J.CiderJudge(bank)):
        ratings[jm.name] = {s: [float(x) for x in jm.rate(bank, IDS, outs[s])]
                            for s in outs}
        print(f"  {jm.name}: done")

    save("ratings.json", ratings)
    for name, r in ratings.items():
        print(f"    {name:>18s}  "
              + "  ".join(f"{s}={np.mean(r[s]):.2f}" for s in outs))


def flat(r, systems):
    return np.concatenate([np.asarray(r[s], dtype=float) for s in systems])


def stage_analyse(args):
    outs = load("captions.json")
    ratings = load("ratings.json")
    systems = list(outs)
    humans = [k for k in ratings if k.startswith("human-")]
    judges = [k for k in ratings if not k.startswith("human-")]

    H_mat = [flat(ratings[h], systems) for h in humans]
    human_mean = np.mean(H_mat, axis=0)
    levels = [1, 2, 3, 4, 5]

    report = {}

    # ---- 1. the ceiling: how well do the three annotators agree? -------------
    hh_spear = J.pairwise(H_mat)
    hh_kappa = float(np.mean([J.cohen_kappa(a, b, levels)
                              for a, b in combinations(
                                  [x.astype(int) for x in H_mat], 2)]))
    fleiss = J.fleiss_kappa([x.astype(int) for x in H_mat], levels)
    report["human_ceiling"] = {"mean_pairwise_spearman": hh_spear,
                               "mean_pairwise_cohen_kappa": hh_kappa,
                               "fleiss_kappa": fleiss}
    print(f"  human-human: Spearman {hh_spear:.3f}  Cohen kappa {hh_kappa:.3f}"
          f"  Fleiss kappa {fleiss:.3f}")

    # ---- 2. each judge against the panel -------------------------------------
    rows = {}
    for j in judges:
        v = flat(ratings[j], systems)
        per_h = [J.spearman(v, h) for h in H_mat]
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        binned = J.bin3(v, lo, hi)
        kap = [J.cohen_kappa(binned, J.bin3(h, 2.5, 3.5), [1, 2, 3])
               for h in H_mat]
        # system-level: average the ratings per system first, then correlate
        sysj = [float(np.mean(ratings[j][s])) for s in systems]
        sysh = [float(np.mean([np.mean(ratings[h][s]) for h in humans]))
                for s in systems]
        rows[j] = {"spearman_vs_panel_mean": J.spearman(v, human_mean),
                   "mean_spearman_vs_individual": float(np.mean(per_h)),
                   "mean_cohen_kappa_binned": float(np.mean(kap)),
                   "system_level_spearman": J.spearman(sysj, sysh),
                   "system_means": dict(zip(systems, sysj))}
        print(f"  {j:>18s}  item-level rho={rows[j]['spearman_vs_panel_mean']:.3f}"
              f"  kappa={rows[j]['mean_cohen_kappa_binned']:.3f}"
              f"  system-level rho={rows[j]['system_level_spearman']:.3f}")
    report["judges"] = rows

    # ---- 3. bias probes ------------------------------------------------------
    lens = np.concatenate([[len(outs[s][k].split()) for k in range(N_IMAGES)]
                           for s in systems]).astype(float)
    bias = {"length_correlation": {}}
    for j in judges + humans:
        bias["length_correlation"][j] = J.spearman(flat(ratings[j], systems), lens)
    self_pref = {}
    for j, own in (("judge-smolvlm-256m", "smolvlm-256m"),
                   ("judge-smolvlm-500m", "smolvlm-500m")):
        if j not in ratings:
            continue
        mine = float(np.mean(ratings[j][own]))
        others = float(np.mean([np.mean(ratings[j][s]) for s in systems
                                if s != own]))
        panel_mine = float(np.mean([np.mean(ratings[h][own]) for h in humans]))
        panel_others = float(np.mean([np.mean(ratings[h][s]) for h in humans
                                      for s in systems if s != own]))
        self_pref[j] = {"own_minus_others": mine - others,
                        "panel_own_minus_others": panel_mine - panel_others,
                        "excess": (mine - others) - (panel_mine - panel_others)}
    bias["self_preference"] = self_pref
    # How much of the 1-5 scale each rater actually uses. A judge that only ever
    # says 4 or 5 is carrying about one bit per item, and no amount of
    # averaging recovers information that was never collected -- the cheapest
    # warning sign there is, and it needs no human panel to read.
    bias["scale_usage"] = {j: sorted({float(v) for v in flat(ratings[j], systems)})
                           for j in judges + humans
                           if j not in ("clipscore", "cider")}
    report["bias"] = bias
    print("  scale usage:", bias["scale_usage"])
    print("  length correlation:",
          {k: round(v, 2) for k, v in bias["length_correlation"].items()})
    for k, v in self_pref.items():
        print(f"  {k} rates its own captions {v['own_minus_others']:+.2f} vs "
              f"others; the panel says {v['panel_own_minus_others']:+.2f} "
              f"(excess {v['excess']:+.2f})")

    save("agreement.json", report)

    # ---- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    names = judges
    item = [rows[j]["spearman_vs_panel_mean"] for j in names]
    syst = [rows[j]["system_level_spearman"] for j in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, item, 0.4, label="item level (per caption)", color="#4c72b0")
    ax.bar(x + 0.2, syst, 0.4, label="system level (per model)", color="#dd8452")
    ax.axhline(hh_spear, ls="--", c="#c44e52",
               label=f"human-human ceiling {hh_spear:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.axhline(0, c="0.3", lw=1)
    ax.set_ylabel("Spearman correlation with the human panel")
    ax.set_ylim(-0.85, 1.05)
    ax.set_title("Two 2015-era metrics reach the human ceiling; "
                 "three LLM judges point the wrong way")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "agreement.png", dpi=125)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    allr = humans + judges
    grid = np.zeros((len(allr), len(systems)))
    for i, r in enumerate(allr):
        for j, s in enumerate(systems):
            v = np.mean(ratings[r][s])
            span = max(np.max([np.mean(ratings[r][t]) for t in systems]), 1e-6)
            grid[i, j] = v / span
    ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(allr):
        for j, s in enumerate(systems):
            ax.text(j, i, f"{np.mean(ratings[r][s]):.2f}", ha="center",
                    va="center", fontsize=8.5,
                    color="0.2" if grid[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_yticks(range(len(allr)))
    ax.set_yticklabels(allr, fontsize=9)
    ax.set_title("Mean rating per system (each row on its own scale)")
    fig.tight_layout()
    fig.savefig(OUT / "system_means.png", dpi=125)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ks = list(bias["length_correlation"])
    vs = [bias["length_correlation"][k] for k in ks]
    ax.bar(range(len(ks)), vs,
           color=["#c44e52" if k.startswith("human") else "#4c72b0" for k in ks])
    ax.axhline(0, c="0.3", lw=1)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Spearman(rating, caption length in words)")
    ax.set_title("Does the judge reward saying more?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "length_bias.png", dpi=125)
    print("  wrote 3 figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["outputs", "rate", "analyse"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"outputs": stage_outputs, "rate": stage_rate,
     "analyse": stage_analyse}[args.stage](args)
