"""Project 01 — Modality survey: put five multimodal papers on one small grid.

Reading five papers and writing five summaries is the easy half. The half that
actually pays off is filling in the *same* two fields for each of them --

    fusion     where the modalities meet   (late / middle / early)
    objective  what the training loss is   (contrastive / masked / generative)

-- and then looking at what the filled grid says. This script draws that grid,
a capability matrix, and a "cost of the glue" chart, and writes the survey out
as markdown.

    python3 survey.py            # ~3 seconds, no model, no data, no network
"""

import argparse
from pathlib import Path

import numpy as np

import papers as P
import plot_style as ps

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


# ---------------------------------------------------------------------------
def figure_taxonomy(path):
    """The 3x3 grid: fusion on x, training objective on y."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax.set_facecolor(ps.SURFACE)

    # group the papers by cell so co-located ones stack instead of overlapping
    cells = {}
    for p in P.PAPERS:
        cells.setdefault((p["fusion"], p["objective"]), []).append(p)

    for xi, fusion in enumerate(P.FUSIONS):
        for yi, obj in enumerate(P.OBJECTIVES):
            occupants = cells.get((fusion, obj), [])
            filled = bool(occupants)
            ax.add_patch(FancyBboxPatch(
                (xi + 0.06, yi + 0.06), 0.88, 0.88,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                linewidth=1.2,
                edgecolor=ps.SERIES[0] if filled else ps.BASELINE,
                facecolor="#e8f1fc" if filled else ps.SURFACE))
            if filled:
                for k, p in enumerate(occupants):
                    ax.text(xi + 0.5, yi + 0.72 - 0.22 * k, p["key"],
                            ha="center", va="center", fontsize=12.5,
                            color=ps.INK, fontweight="bold")
                    ax.text(xi + 0.5, yi + 0.60 - 0.22 * k, str(p["year"]),
                            ha="center", va="center", fontsize=8.5,
                            color=ps.INK_MUTED)
            else:
                ax.text(xi + 0.5, yi + 0.5, P.EMPTY_CELL_HINTS[(fusion, obj)],
                        ha="center", va="center", fontsize=10,
                        color=ps.INK_MUTED, style="italic")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["LATE\n(compare only at the end)",
                        "MIDDLE\n(attend across, inside)",
                        "EARLY\n(one token sequence)"], fontsize=9.5)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["contrastive", "masked", "generative"], fontsize=10)
    ax.tick_params(colors=ps.INK_SECONDARY, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Five papers, two coordinates — and six of nine cells empty",
                 color=ps.INK, fontsize=12.5, loc="left", pad=14)
    ax.set_xlabel("fusion: WHERE the modalities meet",
                  color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("objective: WHAT the loss asks for",
                  color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_capabilities(path):
    """Which of the four canonical jobs each architecture can actually do."""
    import matplotlib.pyplot as plt

    keys = [p["key"] for p in P.PAPERS]
    cols = [c for c, _ in P.CAPABILITY_LABELS]
    M = np.array([[float(p["can"][c]) for c in cols] for p in P.PAPERS])

    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax.set_facecolor(ps.SURFACE)
    for i in range(len(keys)):
        for j in range(len(cols)):
            yes = M[i, j] > 0.5
            ax.add_patch(plt.Rectangle((j, len(keys) - 1 - i), 1, 1,
                                       facecolor="#dff0e6" if yes else "#f6e7e7",
                                       edgecolor=ps.SURFACE, linewidth=2.5))
            ax.text(j + 0.5, len(keys) - 0.5 - i, "yes" if yes else "no",
                    ha="center", va="center", fontsize=11,
                    color=ps.SERIES[1] if yes else ps.SERIES[2],
                    fontweight="bold")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(keys))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([lbl for _, lbl in P.CAPABILITY_LABELS], fontsize=9)
    ax.set_yticks(np.arange(len(keys)) + 0.5)
    ax.set_yticklabels(keys[::-1], fontsize=10.5)
    ax.tick_params(colors=ps.INK_SECONDARY, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("The grid position predicts the job list",
                 color=ps.INK, fontsize=12.5, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_glue_cost(path):
    """Trainable connector size over time — the simplification story, in numbers."""
    order = sorted(P.PAPERS, key=lambda p: (p["year"], -p["connector_params"]))
    labels = [f'{p["key"]}\n{p["year"]}' for p in order]
    vals = [p["connector_params"] for p in order]
    # a bar chart cannot draw 0 on a log axis; floor it and label it honestly
    floor = 0.3
    heights = [max(v, floor) for v in vals]

    fig, ax = ps.new_axes(7.6, 4.2)
    bars = ax.bar(labels, heights, color=[ps.SERIES[0]] * len(vals), width=0.6)
    bars[-1].set_color(ps.SERIES[1])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        txt = "0\n(no connector)" if v == 0 else (
            f"{v/1000:.0f}B" if v >= 1000 else f"{v:.0f}M")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.25, txt,
                ha="center", fontsize=9.5, color=ps.INK_SECONDARY)
    ax.set_ylim(floor * 0.6, 40000)
    ps.finish(fig, ax,
              "Glue you must train: it explodes at Flamingo, then shrinks to nothing",
              "", "trainable connector parameters (millions, log scale)", path)


# ---------------------------------------------------------------------------
def write_survey(path):
    lines = ["# Survey: five multimodal papers on two coordinates", "",
             "*Generated by `survey.py` from `papers.py` — edit the table, not this file.*",
             "",
             "| paper | year | fusion | objective | what is frozen | trainable glue |",
             "|---|---|---|---|---|---|"]
    for p in P.PAPERS:
        glue = "none" if p["connector_params"] == 0 else (
            f'{p["connector_params"]/1000:.0f}B' if p["connector_params"] >= 1000
            else f'{p["connector_params"]:.0f}M')
        lines.append(f'| [{p["key"]}]({p["url"]}) | {p["year"]} | {p["fusion"]} | '
                     f'{p["objective"]} | {p["frozen"]} | {glue} |')
    lines += ["", "---", ""]
    for p in P.PAPERS:
        lines += [f'## {p["key"]} — {p["title"]} ({p["org"]}, {p["year"]})', "",
                  f'**Coordinates:** {p["fusion"]} fusion, {p["objective"]} objective. '
                  f'**Training data:** {p["data"]}.', "",
                  p["summary"], ""]
    Path(path).write_text("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(exist_ok=True)
    figure_taxonomy(OUT / "taxonomy_grid.png")
    figure_capabilities(OUT / "capability_matrix.png")
    figure_glue_cost(OUT / "glue_cost.png")
    write_survey(OUT / "survey.md")


if __name__ == "__main__":
    main()
