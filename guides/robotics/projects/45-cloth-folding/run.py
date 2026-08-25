"""Project 45 -- folding a square of cloth with pick-and-place actions.

Six experiments:

  1. the cloth, one fold sequence, and what the camera sees
  2. the metric, and the control that shows it is measuring something
  3. how accurately you have to grab
  4. grabbing the wrong layer
  5. a learned pick-and-place policy against the expert and two controls
  6. how many demonstrations it takes

Runs in about nine minutes.  NumPy, OpenCV, Matplotlib and torch.
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import cloth as C                                                             # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402

torch.set_num_threads(8)

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

EXTENT = 0.16
RES = 64
N_DEMO = 120
N_EVAL = 20


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<52s} {value}")


def fresh(rng, **kw):
    c = C.Cloth(yaw=float(rng.uniform(0, np.pi / 2)),
                centre=(float(rng.uniform(-0.02, 0.02)),
                        float(rng.uniform(-0.02, 0.02))), **kw)
    C.settle(c, 50)
    return c


def demo(c, n_folds=2):
    """Fold in place with the expert, recording (mask, action) at each step."""
    obs, acts = [], []
    for k in range(n_folds):
        obs.append(c.mask(RES, EXTENT))
        pk, pl = C.extreme_fold(c, axis=k % 2)
        acts.append(np.concatenate([pk, pl]) / EXTENT)
        C.pick_place(c, pk, pl)
    return obs, acts


# ---------------------------------------------------------------------------
# 1. the picture
# ---------------------------------------------------------------------------

def exp1_picture():
    print("\n[1] the cloth and one fold sequence")
    c = C.Cloth()
    C.settle(c, 50)
    stages = [(c.p.copy(), c.mask(RES, EXTENT), c.layers(RES, EXTENT))]
    areas = [stages[0][1].sum()]
    for k in range(2):
        pk, pl = C.extreme_fold(c, axis=k % 2)
        C.pick_place(c, pk, pl)
        stages.append((c.p.copy(), c.mask(RES, EXTENT), c.layers(RES, EXTENT)))
        areas.append(stages[-1][1].sum())
    fig = plt.figure(figsize=(10.5, 6.2))
    titles = ["flat", "after one fold", "after two folds"]
    for i, (P, m, L) in enumerate(stages):
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        G = P.reshape(C.GRID, C.GRID, 3)
        ax.plot_wireframe(G[:, :, 0], G[:, :, 1], G[:, :, 2], lw=0.5,
                          color=COLORS[0])
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(titles[i], fontsize=9)
        ax = fig.add_subplot(3, 3, i + 4)
        ax.imshow(m, cmap="Greys", origin="lower")
        ax.set_title(f"footprint: {m.sum()} px", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(3, 3, i + 7)
        im = ax.imshow(L, cmap="magma", origin="lower", vmin=0, vmax=6)
        ax.set_title(f"layers: up to {L.max()}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046)
    save(fig, os.path.join(OUT, "folds.png"))
    for i, a in enumerate(areas):
        record("1_picture", f"footprint after {i} folds (px)", int(a))
    record("1_picture", "area ratio, one fold", round(areas[1] / areas[0], 3))
    record("1_picture", "area ratio, two folds", round(areas[2] / areas[0], 3))


# ---------------------------------------------------------------------------
# 2. the metric
# ---------------------------------------------------------------------------

def exp2_metric():
    print("\n[2] the metric, and a control")
    rng = np.random.default_rng(1)
    rows = {}
    for tag, kw in (("with self-collision", dict(self_collide=True)),
                    ("without self-collision", dict(self_collide=False))):
        ious, ratios, layers = [], [], []
        for _ in range(10):
            c = fresh(np.random.default_rng(int(rng.integers(1 << 30))), **kw)
            a0 = c.mask(RES, EXTENT).sum()
            tgt, _ = C.expert_result(c, 2, RES, EXTENT)
            demo(c, 2)
            m = c.mask(RES, EXTENT)
            ious.append(C.iou(m, tgt))
            ratios.append(m.sum() / a0)
            layers.append(c.layers(RES, EXTENT).max())
        rows[tag] = (np.mean(ious), np.mean(ratios), np.mean(layers))
        record("2_metric", f"{tag}: IoU vs the expert's own result",
               round(float(rows[tag][0]), 3))
        record("2_metric", f"{tag}: area ratio after two folds",
               round(float(rows[tag][1]), 3))
        record("2_metric", f"{tag}: peak layer count",
               round(float(rows[tag][2]), 1))
    # the do-nothing control: how good does IoU look if you fold NOTHING?
    null_iou = []
    for _ in range(10):
        c = fresh(np.random.default_rng(int(rng.integers(1 << 30))))
        tgt, _ = C.expert_result(c, 2, RES, EXTENT)
        null_iou.append(C.iou(c.mask(RES, EXTENT), tgt))
    record("2_metric", "IoU of an UNFOLDED sheet against the target",
           round(float(np.mean(null_iou)), 3))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ks = ["unfolded sheet\n(control)"] + list(rows)
    vs = [np.mean(null_iou)] + [rows[k][0] for k in rows]
    ax.barh(ks, vs, color=[COLORS[6], COLORS[2], COLORS[4]])
    ax.set_xlabel("IoU against the expert's result on the same sheet")
    ax.set_xlim(0, 1)
    save(fig, os.path.join(OUT, "metric.png"))


# ---------------------------------------------------------------------------
# 3 + 4. how accurately you have to grab, and which layer
# ---------------------------------------------------------------------------

def exp34_precision():
    print("\n[3+4] pick accuracy, and the layer you grab")
    deltas = [0.0, 0.007, 0.015, 0.025, 0.040]
    rng = np.random.default_rng(21)
    curve = []
    for d in deltas:
        ious = []
        for _ in range(8):
            seed = int(rng.integers(1 << 30))
            c = fresh(np.random.default_rng(seed))
            tgt, _ = C.expert_result(c, 2, RES, EXTENT)
            r2 = np.random.default_rng(seed + 1)
            for k in range(2):
                pk, pl = C.extreme_fold(c, axis=k % 2)
                th = r2.uniform(0, 2 * np.pi)
                pk = pk + d * np.array([np.cos(th), np.sin(th)])
                C.pick_place(c, pk, pl)
            ious.append(C.iou(c.mask(RES, EXTENT), tgt))
        curve.append(float(np.mean(ious)))
        record("3_precision", f"pick error {1000 * d:.0f} mm: IoU",
               round(curve[-1], 3))

    # grabbing the BOTTOM layer instead of the top
    ious_top, ious_bot = [], []
    for _ in range(8):
        seed = int(rng.integers(1 << 30))
        for bottom, store in ((False, ious_top), (True, ious_bot)):
            c = fresh(np.random.default_rng(seed))
            tgt, _ = C.expert_result(c, 2, RES, EXTENT)
            for k in range(2):
                pk, pl = C.extreme_fold(c, axis=k % 2)
                if bottom:
                    _pick_bottom(c, pk, pl)
                else:
                    C.pick_place(c, pk, pl)
            store.append(C.iou(c.mask(RES, EXTENT), tgt))
    record("4_layer", "grab the TOP layer: IoU", round(float(np.mean(ious_top)), 3))
    record("4_layer", "grab the BOTTOM layer: IoU",
           round(float(np.mean(ious_bot)), 3))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].plot([1000 * d for d in deltas], curve, "o-", ms=4)
    axes[0].axvline(1000 * C.SPACING, color="#8C8C8C", ls=":", lw=1)
    axes[0].text(1000 * C.SPACING + 0.5, min(curve) + 0.02,
                 "one particle spacing", fontsize=7.5)
    axes[0].set_xlabel("random error added to the pick point (mm)")
    axes[0].set_ylabel("IoU after two folds")
    axes[1].bar(["top layer", "bottom layer"],
                [np.mean(ious_top), np.mean(ious_bot)],
                color=[COLORS[2], COLORS[1]])
    axes[1].set_ylabel("IoU after two folds")
    axes[1].set_title("which layer the gripper pinches", fontsize=9)
    save(fig, os.path.join(OUT, "precision.png"))


def _pick_bottom(c, pick_xy, place_xy):
    """Same primitive, but pinching the LOWEST particle at that spot."""
    d = np.linalg.norm(c.p[:, :2] - np.asarray(pick_xy)[None, :], axis=1)
    close = np.flatnonzero(d < 1.5 * C.SPACING)
    k = int(close[np.argmin(c.p[close, 2])]) if len(close) else int(np.argmin(d))
    saved = C.nearest_particle
    C.nearest_particle = lambda cl, xy: k
    try:
        C.pick_place(c, pick_xy, place_xy)
    finally:
        C.nearest_particle = saved


# ---------------------------------------------------------------------------
# 5 + 6. a learned policy
# ---------------------------------------------------------------------------

class FoldNet(nn.Module):
    """Mask in, four numbers out: where to pick and where to place."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.f1 = nn.Linear(48 * 8 * 8 + 1, 128)
        self.f2 = nn.Linear(128, 4)

    def forward(self, x, step):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.f1(torch.cat([h.flatten(1), step], 1)))
        return torch.tanh(self.f2(h))


def collect(n, seed):
    rng = np.random.default_rng(seed)
    X, S, Y = [], [], []
    for _ in range(n):
        c = fresh(rng)
        obs, acts = demo(c, 2)
        for k, (o, a) in enumerate(zip(obs, acts)):
            X.append(o.astype(np.float32)[None])
            S.append(np.float32([k]))
            Y.append(a.astype(np.float32))
    return (np.stack(X), np.stack(S), np.stack(Y))


def train_net(data, epochs=90, bs=32, lr=1e-3, seed=0, log=None):
    X, S, Y = data
    torch.manual_seed(seed)
    net = FoldNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, St, Yt = (torch.from_numpy(X), torch.from_numpy(S), torch.from_numpy(Y))
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for i in range(0, len(Yt), bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            loss = F.mse_loss(net(Xt[k], St[k]), Yt[k])
            loss.backward()
            opt.step()
            tot += loss.item() * len(k)
        curve.append(tot / len(Yt))
        if log and (ep + 1) % 30 == 0:
            log(f"    epoch {ep + 1}/{epochs}  loss {curve[-1]:.5f}")
    return net, curve


@torch.no_grad()
def net_action(net, c, step):
    net.eval()
    m = torch.from_numpy(c.mask(RES, EXTENT).astype(np.float32)[None, None])
    s = torch.tensor([[float(step)]])
    a = net(m, s)[0].numpy() * EXTENT
    return a[:2], a[2:]


def run_policy(kind, seed, net=None):
    rng = np.random.default_rng(seed)
    c = fresh(rng)
    tgt, _ = C.expert_result(c, 2, RES, EXTENT)
    for k in range(2):
        if kind == "expert":
            pk, pl = C.extreme_fold(c, axis=k % 2)
        elif kind == "learned":
            pk, pl = net_action(net, c, k)
        elif kind == "random":
            pk = rng.uniform(-0.10, 0.10, 2)
            pl = rng.uniform(-0.10, 0.10, 2)
        else:      # "centre": drag the furthest point to the middle
            key = c.p[:, 0] + c.p[:, 1] if k == 0 else c.p[:, 0] - c.p[:, 1]
            pk = c.p[int(np.argmin(key)), :2].copy()
            pl = c.p[:, :2].mean(0)
        C.pick_place(c, pk, pl)
    return C.iou(c.mask(RES, EXTENT), tgt), c


def exp56_learned():
    print("\n[5] a learned pick-and-place policy")
    t0 = time.time()
    data = collect(N_DEMO, seed=100)
    record("5_learned", "demonstration episodes", N_DEMO)
    record("5_learned", "training examples (2 folds each)", len(data[2]))
    record("5_learned", "collection time (s)", round(time.time() - t0))
    net, curve = train_net(data, log=lambda s: print(s))

    rows = {}
    for kind in ("expert", "learned", "fold to the centre", "random"):
        key = {"fold to the centre": "centre"}.get(kind, kind)
        ious = [run_policy(key, 900 + i, net)[0] for i in range(N_EVAL)]
        rows[kind] = float(np.mean(ious))
        record("5_learned", f"{kind}: mean IoU", round(rows[kind], 3))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot(curve, lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("regression loss")
    ks = list(rows)
    axes[1].barh(ks, [rows[k] for k in ks],
                 color=["#42505e", COLORS[2], COLORS[4], COLORS[6]])
    axes[1].set_xlabel("IoU against the expert on the same sheet")
    axes[1].set_xlim(0, 1)
    save(fig, os.path.join(OUT, "learned.png"))

    print("\n[6] how many demonstrations")
    sizes = [15, 40, 80, N_DEMO]
    n_probe = 12
    vals = []
    for n in sizes:
        sub = tuple(d[:2 * n] for d in data)
        net_n, _ = train_net(sub, seed=1)
        vals.append(float(np.mean([run_policy("learned", 950 + i, net_n)[0]
                                   for i in range(n_probe)])))
        record("6_data", f"{n} demonstrations: mean IoU", round(vals[-1], 3))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(sizes, vals, "o-", ms=4)
    ax.axhline(rows["expert"], color="#42505e", ls="--", lw=1)
    ax.text(sizes[0], rows["expert"] - 0.05, "expert", fontsize=8)
    ax.set_xlabel("demonstration episodes")
    ax.set_ylabel("IoU")
    save(fig, os.path.join(OUT, "data.png"))


def main():
    use_style()
    t0 = time.time()
    exp1_picture()
    exp2_metric()
    exp34_precision()
    exp56_learned()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
