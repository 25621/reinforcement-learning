"""Shared matplotlib styling for the robotics Phase 1 projects.

Every project imports this so the figures in the guide look like one set.
Keeping it in project 01 (the first project of the phase) means the later
projects can find it with a two-line sys.path insert.
"""

import matplotlib

matplotlib.use("Agg")  # no display on a headless machine; write PNGs directly
import matplotlib.pyplot as plt

# A colour-blind-safe qualitative palette (Okabe-Ito).
COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#8C8C8C",  # grey
]

FRAME_COLORS = ("#D55E00", "#009E73", "#0072B2")  # x, y, z axes of a triad


def use_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": matplotlib.cycler(color=COLORS),
        }
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
