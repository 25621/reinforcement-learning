"""A tiny video world where "what" and "when" can be measured apart.

Real video benchmarks mix everything together: a question about a cooking video
needs objects, motion, order, and world knowledge at once, so when a model fails
you cannot say which part failed. This dataset is deliberately narrow instead.
Every clip is 8 frames of 64x64 pixels holding exactly two objects:

    * one that MOVES, in one of four directions, at one of two speeds
    * one that STAYS STILL, as a distractor

and every clip comes with three questions that need different amounts of video:

    presence   "Is there a red ball?"                 -> one frame is enough
    speed      "Does the red ball move slowly?"       -> needs 2+ frames, any order
    direction  "Which way does the red ball move?"    -> needs 2+ frames IN ORDER

That last column is the whole point. Shuffling the frames destroys direction and
leaves the other two untouched, which turns "frame sampling loses motion" from a
slogan into a number.

Two things are deliberately controlled so that a single frame gives away nothing:

  * the object's *middle* position is drawn from the same box for every
    direction and speed, so "it is near the left edge" does not imply "it moved
    left" (in a bounded frame, a rightward-moving object must start on the left
    -- that leak would let a one-frame model score above chance and quietly
    invalidate the experiment)
  * both speeds use the same box, so position does not hint at speed either

Project 31 imports this file and trains a video classifier on the same clips, so
the two projects are graded on identical data.
"""

from pathlib import Path

import numpy as np

SIZE = 64
FRAMES = 8
OBJ = 12                       # object bounding box in pixels
CENTRE_LO, CENTRE_HI = 18, 46  # where an object's mid-trajectory point may sit
SPEEDS = {"slowly": 1.5, "quickly": 3.5}      # pixels per frame
DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
SHAPES = ["ball", "square", "triangle"]
COLOURS = {"red": (222, 62, 60), "green": (60, 190, 110),
           "blue": (70, 120, 235), "yellow": (235, 195, 60)}
BG = (26, 26, 32)


def _masks():
    """Pre-render each shape once as a boolean stamp."""
    y, x = np.mgrid[0:OBJ, 0:OBJ]
    c = (OBJ - 1) / 2
    out = {"square": np.ones((OBJ, OBJ), bool),
           "ball": ((x - c) ** 2 + (y - c) ** 2) <= (c + 0.2) ** 2}
    tri = np.zeros((OBJ, OBJ), bool)
    for r in range(OBJ):
        half = (r + 1) / 2
        tri[r, int(round(c - half)):int(round(c + half)) + 1] = True
    out["triangle"] = tri
    return out


MASKS = _masks()


def _stamp(frame, mask, colour, cx, cy):
    """Draw one shape centred at (cx, cy), clipped at the frame edges."""
    x0, y0 = int(round(cx - OBJ / 2)), int(round(cy - OBJ / 2))
    xs, ys = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    xe, ye = min(SIZE, x0 + OBJ - xs), min(SIZE, y0 + OBJ - ys)
    if xe <= x0 or ye <= y0:
        return
    m = mask[ys:ys + ye - y0, xs:xs + xe - x0]
    frame[y0:ye, x0:xe][m] = colour


def render_clip(rng):
    """One clip plus its ground-truth labels."""
    d_name = list(DIRECTIONS)[rng.integers(4)]
    s_name = list(SPEEDS)[rng.integers(2)]
    dx, dy = DIRECTIONS[d_name]
    step = SPEEDS[s_name]
    travel = step * (FRAMES - 1)
    mid = rng.uniform(CENTRE_LO, CENTRE_HI, 2)
    cx = mid[0] - dx * travel / 2
    cy = mid[1] - dy * travel / 2

    mover = {"shape": SHAPES[rng.integers(3)], "colour": list(COLOURS)[rng.integers(4)]}
    while True:                                   # the distractor must differ
        other = {"shape": SHAPES[rng.integers(3)], "colour": list(COLOURS)[rng.integers(4)]}
        if (other["shape"], other["colour"]) != (mover["shape"], mover["colour"]):
            break
    while True:                                   # ... and keep out of the path
        ox, oy = rng.uniform(8, SIZE - 8, 2)
        far = (abs(ox - mid[0]) > OBJ + 4) if dx else (abs(oy - mid[1]) > OBJ + 4)
        if far:
            break

    clip = np.zeros((FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    clip[:] = np.array(BG, dtype=np.uint8)
    for t in range(FRAMES):
        f = clip[t]
        _stamp(f, MASKS[other["shape"]], COLOURS[other["colour"]], ox, oy)
        _stamp(f, MASKS[mover["shape"]], COLOURS[mover["colour"]],
               cx + dx * step * t, cy + dy * step * t)
    label = {"direction": d_name, "speed": s_name,
             "mover": f"{mover['colour']} {mover['shape']}",
             "other": f"{other['colour']} {other['shape']}"}
    return clip, label


def make_dataset(n, seed=0, cache_dir=None):
    """n clips plus labels, cached to disk so every project sees the same data."""
    path = Path(cache_dir) / f"clips_{n}_{seed}.npz" if cache_dir else None
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=True)
        return z["clips"], list(z["labels"])
    rng = np.random.default_rng(seed)
    clips = np.zeros((n, FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    labels = []
    for i in range(n):
        clips[i], lab = render_clip(rng)
        labels.append(lab)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, clips=clips, labels=np.array(labels, dtype=object))
    return clips, labels


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------
def questions(label, rng):
    """The three question/answer pairs for one clip.

    The presence question is balanced by construction: half the time we ask
    about an object that is really there, half the time about a colour/shape
    combination that is not. An unbalanced yes/no set is worse than useless --
    a model that always says "yes" would score 0.75 and look like it can see.
    """
    if rng.random() < 0.5:
        obj = label["mover"] if rng.random() < 0.5 else label["other"]
        presence = (obj, "yes")
    else:
        while True:
            obj = (f"{list(COLOURS)[rng.integers(4)]} {SHAPES[rng.integers(3)]}")
            if obj not in (label["mover"], label["other"]):
                break
        presence = (obj, "no")
    return {
        "presence": (f"Is there a {presence[0]} in the video?", presence[1]),
        "direction": (f"Which way does the {label['mover']} move?", label["direction"]),
        "speed": (f"Does the {label['mover']} move slowly or quickly?", label["speed"]),
    }


TASKS = ("presence", "direction", "speed")
CHANCE = {"presence": 0.5, "direction": 0.25, "speed": 0.5}
ANSWERS = {"presence": ["yes", "no"], "direction": list(DIRECTIONS),
           "speed": list(SPEEDS)}
