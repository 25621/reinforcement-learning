"""Shared matplotlib styling for the multimodal-learning Phase-1 figures.

One place for colors and axis chrome so every chart in these projects reads the
same. Series colors follow a fixed categorical order; axis chrome is recessive.
Imported by projects 02 and 03 via sys.path.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order — assign series to slots in this order, never cycle.
SERIES = ["#2a78d6", "#1baf7a", "#e34948", "#e0873a", "#8a5cd0", "#d64a9c"]


def new_axes(width=7.2, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, ax, title, xlabel, ylabel, out_path):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
