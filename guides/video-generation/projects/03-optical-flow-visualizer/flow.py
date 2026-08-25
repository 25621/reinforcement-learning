"""Dense optical flow between adjacent frames — classic vs neural, in color.

For each test scene we compute flow with Farneback (OpenCV, 2003-era
classic) and RAFT (torchvision, neural), paint both with the standard
color wheel (hue = direction, brightness = speed), and then *score* them
with a warp check: use the flow to reconstruct one frame from the other
and measure how wrong the reconstruction is.

Run:  python3 flow.py        (~1 minute; downloads RAFT weights, ~4 MB)
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import vid_lib
import plot_style as ps

warnings.filterwarnings("ignore")
import torch
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

OUT = HERE / "outputs"
SIZE = (384, 288)  # multiple of 8, as RAFT requires


# ------------------------------------------------------------ flow methods --
def farneback_flow(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


_raft = None


def raft_flow(img1, img2):
    global _raft
    if _raft is None:
        _raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
    to_t = lambda im: (torch.from_numpy(im).permute(2, 0, 1).float()[None]
                       / 127.5 - 1.0)
    with torch.no_grad():
        flows = _raft(to_t(img1), to_t(img2))
    return flows[-1][0].permute(1, 2, 0).numpy()


# -------------------------------------------------------------- color wheel --
def flow_to_color(flow, max_mag=None):
    """hue = direction of motion, brightness = speed (white = still)."""
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)                       # -pi..pi
    if max_mag is None:
        max_mag = max(np.percentile(mag, 99), 1e-6)
    hsv = np.zeros((*mag.shape, 3), np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def color_wheel_legend(size=120):
    """The legend: every direction/speed combination painted once."""
    r = size // 2
    y, x = np.mgrid[-r:r, -r:r]
    flow = np.stack([x, y], axis=-1).astype(np.float32)
    img = flow_to_color(flow, max_mag=r * 0.9)
    mask = x ** 2 + y ** 2 > r * r
    img[mask] = 252
    return img


# --------------------------------------------------------------- warp check --
def warp_error(img1, img2, flow):
    """Reconstruct img1 by sampling img2 where the flow says each pixel went.

    Pixel at (x, y) in img1 moved to (x + flow_x, y + flow_y) in img2, so
    looking up img2 at that spot should give img1 back ("backward warping").
    Returns mean squared error of the reconstruction, in 0-255 pixel units.
    """
    h, w = img1.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    recon = cv2.remap(img2, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean((recon.astype(np.float32)
                          - img1.astype(np.float32)) ** 2)), recon


def main():
    OUT.mkdir(exist_ok=True)
    vid_lib.ensure_sources()
    vtest = vid_lib.read_frames(vid_lib.DATA / "vtest.avi", resize=SIZE)
    mega = vid_lib.read_frames(vid_lib.DATA / "Megamind.avi", resize=SIZE)
    bunny = vid_lib.read_frames(vid_lib.DATA / "bunny.mp4", resize=SIZE)

    scenes = {
        # name: (frame t, frame t+dt) — dt chosen so motion is visible
        "pedestrians": (vtest[120], vtest[123]),
        "trailer": (mega[95], mega[97]),
        "animation": (bunny[60], bunny[62]),
    }

    errors = {}
    fig, axes = ps.plt.subplots(len(scenes), 4, figsize=(10.4, 6.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for row, (name, (f1, f2)) in enumerate(scenes.items()):
        fb = farneback_flow(f1, f2)
        rf = raft_flow(f1, f2)
        shared_max = max(np.percentile(np.linalg.norm(f, axis=-1), 99)
                         for f in (fb, rf))
        e0 = float(np.mean((f1.astype(np.float32)
                            - f2.astype(np.float32)) ** 2))
        efb, _ = warp_error(f1, f2, fb)
        erf, _ = warp_error(f1, f2, rf)
        errors[name] = {"no flow": e0, "Farneback": efb, "RAFT": erf}
        print(f"{name:12s} warp MSE  no-flow {e0:7.1f}   "
              f"Farneback {efb:6.1f}   RAFT {erf:6.1f}   "
              f"mean |flow| fb {np.linalg.norm(fb, axis=-1).mean():.2f} "
              f"raft {np.linalg.norm(rf, axis=-1).mean():.2f} px")

        panels = [f1, f2, flow_to_color(fb, shared_max),
                  flow_to_color(rf, shared_max)]
        titles = ["frame t", "frame t+dt", "Farneback flow", "RAFT flow"]
        for col, (img, title) in enumerate(zip(panels, titles)):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(title, color=ps.INK, fontsize=10)
            if col == 0:
                ax.set_ylabel(name, color=ps.INK_SECONDARY, fontsize=10,
                              rotation=90)
    fig.tight_layout()
    fig.savefig(OUT / "flow_panels.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print(f"wrote {OUT / 'flow_panels.png'}")
    ps.plt.close(fig)

    cv2.imwrite(str(OUT / "color_wheel.png"), color_wheel_legend()[:, :, ::-1])
    print(f"wrote {OUT / 'color_wheel.png'}")

    # Warp-error chart: does the flow actually explain the next frame?
    fig, ax = ps.new_axes(7.0, 3.4)
    names = list(errors)
    width = 0.26
    for i, (method, color) in enumerate([("no flow", ps.INK_MUTED),
                                         ("Farneback", ps.SERIES[0]),
                                         ("RAFT", ps.SERIES[1])]):
        xs = np.arange(len(names)) + (i - 1) * width
        vals = [errors[n][method] for n in names]
        ax.bar(xs, vals, width=width * 0.92, color=color, label=method)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.0f} ", ha="center", va="bottom",
                    color=ps.INK_SECONDARY, fontsize=8)
    ax.set_xticks(np.arange(len(names)), names)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax,
              "Warp check - reconstruct frame t from t+dt using the flow",
              "", "reconstruction MSE",
              OUT / "warp_error.png")


if __name__ == "__main__":
    main()
