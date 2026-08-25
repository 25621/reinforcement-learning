"""Project 50 — Physical-plausibility probe.

A generated clip can look crisp frame by frame and still break the world's
rules.  Our toy has one simple, checkable law: the sprite is a ball in a closed
room — it must *stay in the room* (walls stop it), must *not vanish* (object
permanence), and must *bounce* off a wall rather than pass through it (world
consistency).  We build a probe of fast-moving clips that are forced to hit a
wall, and count the violations.

The point is the same one Sora's own report makes: the per-frame quality
metrics say nothing about whether physics held.  A weak model can score almost
as sharp as a strong one and still let the ball leak through the wall.

Stages
    weak     train a deliberately under-trained model (the physics-blind one)
    probe    run the probe on the strong (project 45) and weak models
    figures  draw the violation breakdown and show failure clips
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "45-run-vbench-end-to-end"))
import eval_lib as E                                            # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CK = Path(__file__).resolve().parent / "checkpoints"
CK.mkdir(exist_ok=True)


def stage_weak(args):
    """Same model, far fewer training steps: sharp enough, physics not learned."""
    torch.manual_seed(0)
    ds = E.make_dataset(3000, seed=1)
    net = E.VideoGen(base=32)
    t = time.time()
    E.train(net, ds, steps=args.steps, batch=128, lr=2e-3, seed=0,
            log_every=200)
    print(f"weak model trained ({args.steps} steps) in {time.time() - t:.0f}s")
    E.save_gen(net, "weak", base=32, where=CK)


# ---------------------------------------------------------------------------
# the physics checks, all read from the generated pixels
# ---------------------------------------------------------------------------
WALL_LO = E.RADIUS - 0.6
WALL_HI = E.H - 1 - E.RADIUS + 0.6
# the fastest legal move is 2.8 px/frame; anything much larger is a teleport,
# a jump no continuous motion could produce.
TELEPORT = 5.0


def physics_report(clip):
    """Classify one clip's physics.  Returns a dict of pass/fail flags."""
    r = E.read_clip(clip)
    centres = r["centres"]
    present = [c is not None for c in centres]
    # object permanence: the sprite is present in every frame
    permanence = all(present)
    # containment: it never leaves the room (centre past the wall line)
    contained = True
    for c in centres:
        if c is None:
            continue
        if (c[0] < WALL_LO or c[0] > WALL_HI
                or c[1] < WALL_LO or c[1] > WALL_HI):
            contained = False
    # continuity: no frame-to-frame teleport (a jump no real motion can make)
    teleport = False
    idxc = [(i, c) for i, c in enumerate(centres) if c is not None]
    for (i, a), (j, b) in zip(idxc, idxc[1:]):
        if j == i + 1:
            if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > TELEPORT:
                teleport = True
    continuity = not teleport
    return dict(permanence=permanence, contained=contained,
                continuity=continuity,
                violation=(not permanence) or (not contained)
                or (not continuity))


def _shuffle_time(clips, seed=0):
    """Scramble each clip's frame ORDER.  Per-frame content is untouched, so
    per-frame image quality is *identical* — but the motion is destroyed."""
    rng = np.random.default_rng(seed)
    out = np.array(clips)
    for i in range(len(out)):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out


def stage_probe(args):
    strong = E.load_gen("base")
    weak = E.load_gen("weak", where=CK)
    # the probe: 200 FAST clips (they must hit a wall within 8 frames)
    ds = E.make_dataset(4000, seed=321)
    fast = np.where(ds["speed"] == 1)[0][:200]
    caps = E.caption_tensor(ds, fast)

    strong_gen = E.sample(strong, caps, steps=25, scale=2.0,
                          generator=torch.Generator().manual_seed(0)).clamp(0, 1).numpy()
    weak_gen = E.sample(weak, caps, steps=25, scale=2.0,
                        generator=torch.Generator().manual_seed(0)).clamp(0, 1).numpy()
    # the decoupler: strong model's frames, scrambled in time
    shuffled = _shuffle_time(strong_gen, seed=1)

    rows = []
    saved = {}
    for name, gen in [("strong (project 45)", strong_gen),
                      ("weak (undertrained)", weak_gen),
                      ("strong, frames shuffled", shuffled)]:
        reps = [physics_report(c) for c in gen]
        viol = np.mean([r["violation"] for r in reps])
        vanish = np.mean([not r["permanence"] for r in reps])
        leak = np.mean([not r["contained"] for r in reps])
        teleport = np.mean([not r["continuity"] for r in reps])
        sharp = E.imaging_quality(gen)
        rows.append(dict(model=name, violation=viol, vanish=vanish, leak=leak,
                         teleport=teleport, sharpness=sharp, n=len(reps)))
        print(f"{name:26s}  violation {viol:.2f}  "
              f"(teleport {teleport:.2f}, leak {leak:.2f}, vanish {vanish:.2f})"
              f"   sharpness {sharp:.2f}")
        bad = [i for i, r in enumerate(reps) if r["violation"]][:4]
        saved[name] = gen[bad] if bad else gen[:2]
    np.save(OUT / "_samples.npy", saved, allow_pickle=True)
    with open(OUT / "probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "violation", "vanish", "leak", "teleport",
                    "sharpness", "n"])
        for r in rows:
            w.writerow([r["model"], f"{r['violation']:.4f}",
                        f"{r['vanish']:.4f}", f"{r['leak']:.4f}",
                        f"{r['teleport']:.4f}", f"{r['sharpness']:.4f}",
                        r["n"]])


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (OUT / "probe.csv").exists():
        rows = list(csv.DictReader(open(OUT / "probe.csv")))
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        cats = ["sharpness", "violation", "teleport", "leak"]
        labels = ["sharpness\n(per-frame)", "any physics\nviolation",
                  "teleport\n(continuity)", "left room\n(containment)"]
        x = np.arange(len(cats))
        colors = ["#c98a2b", "#8a8f98", "#c0392b"]
        for j, r in enumerate(rows):
            vals = [float(r[c]) for c in cats]
            ax.bar(x + (j - 1) * 0.28, vals, 0.28, label=r["model"],
                   color=colors[j])
        ax.axvspan(0.5, len(cats) - 0.5, color="#ffecec", alpha=0.5, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("The shuffled clips are exactly as sharp as the strong ones\n"
                     "(same frames!) yet every physics check (shaded) convicts them")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "physics.png", dpi=110)
        plt.close(fig)

    if (OUT / "_samples.npy").exists():
        s = np.load(OUT / "_samples.npy", allow_pickle=True).item()
        for name, clips in s.items():
            if len(clips):
                tag = name.split(",")[0].split()[0] \
                    + ("_shuf" if "shuffled" in name else "")
                rows = [list(clips[i]) for i in range(min(3, len(clips)))]
                E.strip(rows, OUT / f"{tag}_clips.png", scale=6)
                E.write_gif(clips[0], OUT / f"{tag}_clip.gif")
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["weak", "probe", "figures"])
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    {"weak": stage_weak, "probe": stage_probe,
     "figures": stage_figures}[a.stage](a)
