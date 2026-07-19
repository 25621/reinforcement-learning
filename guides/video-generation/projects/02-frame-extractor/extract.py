"""Sample N frames from clips two ways — evenly by index vs evenly by time.

Three demos:

  1. CFR clips (constant frame rate): index-sampling and time-sampling agree
     exactly — verified, not assumed.
  2. Same N from a short fast clip and a long slow clip: the real-time gap
     between picks differs by 16x, so "motion between adjacent sampled
     frames" means completely different things to a model.
  3. A VFR clip (variable frame rate) built frame-by-frame with PyAV:
     8 s of a near-still scene stored at 24 fps (192 frames) followed by a
     2 s action burst stored at only 4 fps (8 frames). Index-sampling and
     time-sampling now disagree hard.

Run:  python3 extract.py        (~30 seconds)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import vid_lib
import plot_style as ps

OUT = HERE / "outputs"
DATA = HERE / "data"
W, H = 320, 240


# ------------------------------------------------------- the synthetic VFR --
def draw_room(ball_xy):
    """A simple static room with one colored ball at ball_xy."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (34, 40, 49)                                   # dark wall
    img[180:, :] = (57, 62, 70)                             # floor
    cv2.rectangle(img, (250, 60), (305, 175), (120, 90, 60), -1)  # bookshelf
    cv2.rectangle(img, (30, 70), (95, 130), (150, 180, 200), -1)  # window
    x, y = int(ball_xy[0]), int(ball_xy[1])
    cv2.circle(img, (x, y), 16, (60, 120, 230), -1, cv2.LINE_AA)
    cv2.circle(img, (x - 5, y - 5), 5, (140, 190, 250), -1, cv2.LINE_AA)
    return img


def make_vfr_clip():
    """Still scene stored densely, action burst stored sparsely.

    Returns (path, duration). Timestamps are written explicitly per frame,
    which is what makes the file variable-frame-rate.
    """
    frames, pts_ms = [], []
    rng = np.random.default_rng(0)
    # Phase 1: 0-8 s, ball resting on the floor, stored at 24 fps.
    for i in range(192):
        t = i / 24.0
        jitter = 1.5 * np.sin(2 * np.pi * 0.3 * t)          # barely moves
        frames.append(draw_room((160 + jitter, 164)))
        pts_ms.append(int(t * 1000))
    # Phase 2: 8-10 s, ball bouncing wildly, stored at only 4 fps.
    for i in range(8):
        t = 8.0 + i / 4.0
        x = 40 + 240 * ((0.9 * (t - 8.0)) % 1.0)
        y = 60 + 100 * abs(np.sin(2 * np.pi * 1.3 * (t - 8.0)))
        frames.append(draw_room((x, y)))
        pts_ms.append(int(t * 1000))
    path = DATA / "vfr_room.mp4"
    vid_lib.write_video(np.stack(frames), path, fps=24, pts_ms=pts_ms)
    return path, 10.0


# --------------------------------------------------------------- samplers --
def sample_by_index(n_stored, n_pick):
    """Every Kth stored frame: evenly spaced positions in the FILE."""
    return np.linspace(0, n_stored - 1, n_pick).round().astype(int)


def sample_by_time(times, duration, n_pick):
    """Evenly spaced instants in SECONDS; for each, take the nearest stored
    frame. This is what 'sample at a uniform fps' means."""
    targets = np.linspace(0, duration, n_pick)
    return np.array([np.abs(times - t).argmin() for t in targets])


# ------------------------------------------------------------------ demos --
def demo_cfr_agreement():
    """On a constant-frame-rate clip the two samplers pick the same frames."""
    clip = sorted(vid_lib.make_benchmark_clips().glob("clip_*.mp4"))[0]
    times = vid_lib.frame_times(clip)
    by_idx = sample_by_index(len(times), 8)
    by_time = sample_by_time(times, times[-1], 8)
    print(f"CFR clip:  index picks {by_idx.tolist()}")
    print(f"           time picks  {by_time.tolist()}")
    assert np.array_equal(by_idx, by_time), "should agree on CFR!"
    print("           -> identical, as expected on constant frame rate\n")


def demo_fast_vs_slow():
    """Same N picks from a 5 s clip and an 80 s clip = very different motion."""
    vid_lib.ensure_sources()
    slow = vid_lib.read_frames(vid_lib.DATA / "vtest.avi", resize=(W, H))
    slow_t = vid_lib.frame_times(vid_lib.DATA / "vtest.avi")
    fast = vid_lib.read_frames(vid_lib.DATA / "Megamind.avi", resize=(W, H))
    fast_t = vid_lib.frame_times(vid_lib.DATA / "Megamind.avi")
    fast, fast_t = fast[:120], fast_t[:120]                 # one 5 s stretch

    n = 6
    rows, labels = [], []
    for name, frames, times in [("slow", slow, slow_t), ("fast", fast, fast_t)]:
        picks = sample_by_index(len(frames), n)
        gap = np.diff(times[picks]).mean()
        print(f"{name} clip: {len(frames)} frames / {times[-1]:.0f} s "
              f"-> {gap:.1f} s between picks")
        rows.extend(frames[picks])
        labels.extend([f"{times[p]:.1f}s" for p in picks])
    vid_lib.contact_sheet(rows, cols=n, out_path=OUT / "fast_vs_slow.png",
                          scale=0.6, labels=labels)


def demo_vfr():
    path, duration = make_vfr_clip()
    frames = vid_lib.read_frames(path)
    times = vid_lib.frame_times(path)
    print(f"\nVFR clip: {len(frames)} stored frames over {duration:.0f} s "
          f"(action burst = last 2 s, only 8 stored frames)")

    n = 12
    idx_picks = sample_by_index(len(frames), n)
    time_picks = sample_by_time(times, duration, n)
    n_action_idx = int((times[idx_picks] >= 8.0).sum())
    n_action_time = int((times[time_picks] >= 8.0).sum())
    print(f"index picks land in the action burst: {n_action_idx}/{n}")
    print(f"time  picks land in the action burst: {n_action_time}/{n}")

    # Timeline figure: where stored frames sit, where each sampler picks.
    fig, ax = ps.new_axes(8.6, 2.9)
    ax.vlines(times, 2.6, 2.9, color=ps.INK_MUTED, linewidth=0.6)
    ax.text(0.05, 3.05, "stored frames (one tick per frame)",
            color=ps.INK_SECONDARY, fontsize=9)
    for y, picks, color, label in [
        (1.8, idx_picks, ps.SERIES[2], "picks: even over frame index"),
        (1.0, time_picks, ps.SERIES[0], "picks: even over time (uniform fps)"),
    ]:
        ax.scatter(times[picks], [y] * n, s=48, color=color, zorder=3)
        ax.text(0.05, y + 0.22, label, color=ps.INK_SECONDARY, fontsize=9)
    ax.axvspan(8.0, 10.0, color=ps.SERIES[3], alpha=0.12, lw=0)
    ax.text(9.0, 0.35, "action burst\n(sparsely stored)", ha="center",
            color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(0.2, 3.4)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ps.finish(fig, ax, "Same clip, two samplers — where the 12 picks land",
              "clip time (seconds)", "", OUT / "timeline.png")

    strip = [frames[p] for p in idx_picks] + [frames[p] for p in time_picks]
    labels = ([f"idx {p} - {times[p]:.1f}s" for p in idx_picks]
              + [f"t={times[p]:.1f}s" for p in time_picks])
    vid_lib.contact_sheet(strip, cols=n, out_path=OUT / "vfr_strips.png",
                          scale=0.55, labels=labels)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    demo_cfr_agreement()
    demo_fast_vs_slow()
    demo_vfr()
