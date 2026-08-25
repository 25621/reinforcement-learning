"""Project 22 -- Predicting an object's full 6-DoF pose, and scoring it honestly.

Six experiments:

  1. train a pose network on an asymmetric object, against a do-nothing control
  2. how you REPRESENT the rotation decides whether it can be learned at all
  3. the symmetry ladder: four objects, from "no symmetry" to "a full circle
     of indistinguishable poses", scored by ADD and by ADD-S
  4. training with a symmetry-aware loss
  5. occlusion
  6. where the depth comes from (apparent size), and how well that works

Runs in about six minutes on a CPU.
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
for _p in ("16-camera-calibration", "01-transform-calculator"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

from camera import Camera, rodrigues, rot_to_rvec, rot_angle_deg        # noqa: E402
from mesh import (OBJECTS, SYMMETRIC, model_points, render_mesh,        # noqa: E402
                  add, add_s, diameter)
from plot_style import COLORS, use_style                                # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

torch.set_num_threads(10)
OUT = os.path.join(_HERE, "outputs")
DATA = os.path.join(_HERE, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
RESULTS = []

# The camera is chosen so the object FILLS the frame.  A first version put it
# 30-55 cm away, where a 7 cm object covers about 40 of the 64 pixels and its
# thinner parts are 5 px across -- and the rotation head then never learned
# anything at all.  Real pose networks never see a whole scene either: they
# run on a detection crop, for exactly this reason.
CAM = Camera(fx=300.0, fy=300.0, cx=31.5, cy=31.5, dist=(0, 0, 0, 0, 0),
             width=64, height=64)
N_TRAIN, N_TEST = 4000, 800
Z_RANGE = (0.33, 0.40)
JITTER = 0.012


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def random_rotation(rng):
    """A rotation drawn uniformly from all of SO(3).

    Sampling three Euler angles uniformly does NOT do this -- it piles up
    near the poles, and a network trained on that data is being shown a
    biased view of the problem.  Sampling a unit quaternion uniformly does.
    """
    u = rng.random(3)
    return quat_to_R(np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])[[3, 0, 1, 2]])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def make_dataset(name, n, seed, occlude=0.0):
    v, f = OBJECTS[name]()
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 64, 64), np.uint8)
    Rs = np.zeros((n, 3, 3))
    ts = np.zeros((n, 3))
    for i in range(n):
        R = random_rotation(rng)
        z = rng.uniform(*Z_RANGE)
        t = np.array([rng.uniform(-JITTER, JITTER), rng.uniform(-JITTER, JITTER), z])
        img, mask, _ = render_mesh(v, f, R, t, CAM, rng=rng)
        g = img.mean(axis=2)
        if occlude > 0 and rng.random() < 0.9:
            h = int(64 * np.sqrt(occlude) * rng.uniform(0.7, 1.3))
            x0, y0 = rng.integers(0, 64 - h + 1, 2)
            g[y0:y0 + h, x0:x0 + h] = rng.uniform(0, 60)
        imgs[i] = np.clip(g, 0, 255)
        Rs[i] = R
        ts[i] = t
    return imgs, Rs, ts


def cached_dataset(name, occlude=0.0):
    tag = f"{name}{'_occ' if occlude else ''}"
    path = os.path.join(DATA, f"{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return (d["tr_i"], d["tr_R"], d["tr_t"]), (d["te_i"], d["te_R"], d["te_t"])
    t0 = time.time()
    tr = make_dataset(name, N_TRAIN, seed=hash(tag) % 10000, occlude=occlude)
    te = make_dataset(name, N_TEST, seed=hash(tag) % 10000 + 7777, occlude=occlude)
    np.savez_compressed(path, tr_i=tr[0], tr_R=tr[1], tr_t=tr[2],
                        te_i=te[0], te_R=te[1], te_t=te[2])
    print(f"    rendered {tag}: {N_TRAIN + N_TEST} images in {time.time() - t0:.0f} s",
          flush=True)
    return tr, te


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

ROT_DIM = {"6d": 6, "quat": 4, "euler": 3}


class PoseNet(nn.Module):
    def __init__(self, rot="6d"):
        super().__init__()
        self.rot = rot
        c = [1, 32, 64, 96, 128]
        self.conv = nn.Sequential(*[
            layer for i in range(4) for layer in
            (nn.Conv2d(c[i], c[i + 1], 3, 2, 1), nn.BatchNorm2d(c[i + 1]), nn.ReLU())])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU())
        self.rot_out = nn.Linear(256, ROT_DIM[rot])
        self.trans_out = nn.Linear(256, 3)          # (x/z, y/z, log z)

    def forward(self, x):
        h = self.head(self.conv(x))
        return self.rot_out(h), self.trans_out(h)


def gram_schmidt(x):
    """Turn 6 free numbers into a rotation matrix.

    Take the 6 numbers as two 3-vectors; normalize the first to get column 1;
    remove column 1's component from the second and normalize to get column
    2; cross them for column 3.  The point is that EVERY rotation has a
    nearby set of 6 numbers, with no jumps -- unlike Euler angles, where a
    rotation just past 180 degrees is written as a number just past -180.
    """
    a, b = x[:, :3], x[:, 3:]
    e1 = F.normalize(a, dim=1)
    e2 = F.normalize(b - (e1 * b).sum(1, keepdim=True) * e1, dim=1)
    e3 = torch.cross(e1, e2, dim=1)
    return torch.stack([e1, e2, e3], dim=2)


def quat_to_R_t(q):
    q = F.normalize(q, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)
    ], dim=1)


def euler_to_R_t(e):
    cx, sx = torch.cos(e[:, 0]), torch.sin(e[:, 0])
    cy, sy = torch.cos(e[:, 1]), torch.sin(e[:, 1])
    cz, sz = torch.cos(e[:, 2]), torch.sin(e[:, 2])
    z = torch.zeros_like(cx)
    o = torch.ones_like(cx)
    Rz = torch.stack([torch.stack([cz, -sz, z], 1), torch.stack([sz, cz, z], 1),
                      torch.stack([z, z, o], 1)], 1)
    Ry = torch.stack([torch.stack([cy, z, sy], 1), torch.stack([z, o, z], 1),
                      torch.stack([-sy, z, cy], 1)], 1)
    Rx = torch.stack([torch.stack([o, z, z], 1), torch.stack([z, cx, -sx], 1),
                      torch.stack([z, sx, cx], 1)], 1)
    return Rz @ Ry @ Rx


def to_R(pred, rot):
    if rot == "6d":
        return gram_schmidt(pred)
    if rot == "quat":
        return quat_to_R_t(pred)
    return euler_to_R_t(pred)


def prep(imgs, Rs, ts):
    x = torch.from_numpy(imgs).float().unsqueeze(1) / 128.0 - 1.0
    R = torch.from_numpy(Rs).float()
    tt = torch.from_numpy(ts).float()
    tgt = torch.stack([tt[:, 0] / tt[:, 2], tt[:, 1] / tt[:, 2], torch.log(tt[:, 2])], 1)
    return x, R, tt, tgt


def decode_t(pred_t):
    z = torch.exp(pred_t[:, 2])
    return torch.stack([pred_t[:, 0] * z, pred_t[:, 1] * z, z], 1)


def train(name, rot="6d", loss_kind="pose", epochs=16, occlude=0.0, seed=0, quiet=False):
    torch.manual_seed(seed)
    (tri, trR, trt), (tei, teR, tet) = cached_dataset(name, occlude)
    x, R, t, tgt = prep(tri, trR, trt)
    xv, Rv, tv, tgtv = prep(tei, teR, tet)
    pts = torch.from_numpy(model_points(OBJECTS[name]()[0])).float()

    net = PoseNet(rot)
    opt = torch.optim.AdamW(net.parameters(), 1.5e-3, weight_decay=1e-4)
    n = len(x)
    steps = epochs * (n // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - 63, 64):
            b = perm[i:i + 64]
            pr, pt = net(x[b])
            Rp = to_R(pr, rot)
            if loss_kind == "pose":
                lr_ = ((Rp - R[b]) ** 2).mean()
            else:
                # a point-matching loss: move the model points by both poses
                # and penalize the distance, with each predicted point matched
                # to its NEAREST true point.  Same idea as ADD-S, used as the
                # training objective instead of only as the metric.
                a = pts @ Rp.transpose(1, 2)
                bb = pts @ R[b].transpose(1, 2)
                d = torch.cdist(a, bb)
                lr_ = 200.0 * d.min(dim=2).values.mean()
            lt = ((pt - tgt[b]) ** 2).mean()
            loss = lr_ + lt
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1

    net.eval()
    with torch.no_grad():
        pr, pt = net(xv)
        Rp = to_R(pr, rot).numpy()
        tp = decode_t(pt).numpy()
    res = evaluate(name, Rp, tp, teR, tet)
    res.update(seconds=round(time.time() - t0, 1), rot_rep=rot, loss=loss_kind)
    if not quiet:
        log(dict(stage="train", object=name, occlude=occlude, **res))
    return net, res, (Rp, tp, teR, tet, tei)


def evaluate(name, Rp, tp, Rt, tt):
    v = OBJECTS[name]()[0]
    pts = model_points(v)
    d = diameter(pts)
    adds = np.array([add(pts, Rp[i], tp[i], Rt[i], tt[i]) for i in range(len(Rp))])
    addss = np.array([add_s(pts, Rp[i], tp[i], Rt[i], tt[i]) for i in range(len(Rp))])
    rot = np.array([rot_angle_deg(Rt[i], Rp[i]) for i in range(len(Rp))])
    tr = np.linalg.norm(tp - tt, axis=1) * 1000
    return dict(add_mm=round(float(np.median(adds)) * 1000, 2),
                adds_mm=round(float(np.median(addss)) * 1000, 2),
                add_success_pct=round(100 * float((adds < 0.1 * d).mean()), 1),
                adds_success_pct=round(100 * float((addss < 0.1 * d).mean()), 1),
                rot_err_deg=round(float(np.median(rot)), 2),
                trans_err_mm=round(float(np.median(tr)), 2))


def mean_pose_control(name):
    """The control every pose paper should report and almost none do:
    ignore the image and always predict the average pose of the training set."""
    (tri, trR, trt), (tei, teR, tet) = cached_dataset(name)
    U, _, Vt = np.linalg.svd(trR.mean(axis=0))
    Rm = U @ Vt
    tm = trt.mean(axis=0)
    Rp = np.tile(Rm, (len(teR), 1, 1))
    tp = np.tile(tm, (len(teR), 1))
    return evaluate(name, Rp, tp, teR, tet)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def stage_baseline():
    print("\n[1] a pose network on the L, against the do-nothing control")
    c = mean_pose_control("ell")
    log(dict(stage="control", object="ell", method="always predict the mean pose", **c))
    net, res, out = train("ell", rot="6d")

    Rp, tp, Rt, tt, imgs = out
    v, f = OBJECTS["ell"]()
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
    order = np.argsort([rot_angle_deg(Rt[i], Rp[i]) for i in range(len(Rp))])
    picks = list(order[:3]) + list(order[len(order) // 2:len(order) // 2 + 1]) + list(order[-2:])
    for k, i in enumerate(picks):
        axes[0, k].imshow(imgs[i], cmap="gray")
        axes[0, k].set_title(f"error {rot_angle_deg(Rt[i], Rp[i]):.1f} deg", fontsize=8)
        pred, _, _ = render_mesh(v, f, Rp[i], tp[i], CAM)
        axes[1, k].imshow(pred.mean(axis=2), cmap="gray")
    for a in axes.reshape(-1):
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("predicted pose,\nrendered back", fontsize=9)
    fig.suptitle("Best three, median, and worst two predictions", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "predictions.png"))
    plt.close(fig)
    return res


def stage_representations():
    print("\n[2] rotation representations")
    rows = []
    for rot in ("6d", "quat", "euler"):
        _, res, _ = train("ell", rot=rot)
        rows.append((rot, res["rot_err_deg"], res["add_mm"], res["add_success_pct"]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(rows))
    ax[0].bar(x, [r[1] for r in rows], color=COLORS[0])
    ax[0].set_xticks(x); ax[0].set_xticklabels([r[0] for r in rows])
    ax[0].set_ylabel("median rotation error (deg)")
    ax[0].set_title("how the rotation is written down")
    ax[1].bar(x, [r[3] for r in rows], color=COLORS[2])
    ax[1].set_xticks(x); ax[1].set_xticklabels([r[0] for r in rows])
    ax[1].set_ylabel("ADD success (%)"); ax[1].set_title("and what it costs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "representations.png"))
    plt.close(fig)


def stage_symmetry():
    print("\n[3] the symmetry ladder")
    order = [("ell", "L-shape: no symmetry at all"),
             ("mug", "mug: a cylinder whose handle breaks the symmetry"),
             ("block", "box: FOUR indistinguishable poses (180 deg about each axis)"),
             ("cylinder", "cylinder: a whole circle of indistinguishable poses")]
    res = {}
    for name, label in order:
        _, r, out = train(name, rot="6d", quiet=True)
        res[name] = r
        log(dict(stage="symmetry", object=label, **r))
        if name == "cylinder":
            # where exactly does the cylinder's rotation error live?
            Rp, tp, Rt, tt, imgs = out
            axis_err = [float(np.degrees(np.arccos(np.clip(
                abs(float(Rp[i][:, 2] @ Rt[i][:, 2])), -1, 1)))) for i in range(len(Rp))]
            log(dict(stage="symmetry_split", object="cylinder",
                     axis_direction_err_deg=round(float(np.median(axis_err)), 2),
                     full_rotation_err_deg=res["cylinder"]["rot_err_deg"]))

    print("\n[4] training with a point-matching (ADD-S) loss instead")
    for obj in ("ell", "cylinder"):
        _, r_pm, _ = train(obj, rot="6d", loss_kind="adds", quiet=True)
        log(dict(stage="sym_loss", object=obj + " + point-matching loss", **r_pm))
        if obj == "cylinder":
            res_sym = r_pm

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    names = [n for n, _ in order] + ["cylinder\n+ADD-S loss"]
    rs = [res[n] for n, _ in order] + [res_sym]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [r["add_success_pct"] for r in rs], 0.4, color=COLORS[1], label="ADD")
    ax[0].bar(x + 0.2, [r["adds_success_pct"] for r in rs], 0.4, color=COLORS[2], label="ADD-S")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7.5)
    ax[0].set_ylabel("success (%), threshold 10% of diameter")
    ax[0].legend(fontsize=8); ax[0].set_title("the same predictions, two metrics")
    ax[1].bar(x, [r["rot_err_deg"] for r in rs], color=COLORS[0])
    ax[1].axhline(90, color=COLORS[6], ls="--", lw=1)
    ax[1].text(0.02, 92, "roughly chance for a random rotation", color=COLORS[6], fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7.5)
    ax[1].set_ylabel("median rotation error (deg)")
    ax[1].set_title("what the network could learn at all")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "symmetry.png"))
    plt.close(fig)


def stage_occlusion():
    print("\n[5] occlusion")
    for occ in (0.0, 0.25):
        _, res, _ = train("ell", rot="6d", occlude=occ, quiet=True)
        log(dict(stage="occlusion", trained_and_tested_with_occlusion=occ, **res))
    # trained clean, tested occluded -- the realistic mismatch
    net, _, _ = train("ell", rot="6d", quiet=True)
    (_, _, _), (tei, teR, tet) = cached_dataset("ell", occlude=0.25)
    xv, _, _, _ = prep(tei, teR, tet)
    with torch.no_grad():
        pr, pt = net(xv)
        Rp = to_R(pr, "6d").numpy()
        tp = decode_t(pt).numpy()
    log(dict(stage="occlusion", trained_and_tested_with_occlusion="trained clean, tested occluded",
             **evaluate("ell", Rp, tp, teR, tet)))


def stage_depth():
    print("\n[6] where the depth comes from")
    _, _, out = train("ell", rot="6d", quiet=True)
    Rp, tp, Rt, tt, _ = out
    err = np.abs(tp[:, 2] - tt[:, 2]) * 1000
    lat = np.linalg.norm(tp[:, :2] - tt[:, :2], axis=1) * 1000
    log(dict(stage="depth", median_depth_err_mm=round(float(np.median(err)), 2),
             median_lateral_err_mm=round(float(np.median(lat)), 2),
             depth_err_pct_of_range=round(float(np.median(err / (tt[:, 2] * 1000))) * 100, 2)))
    bins = np.linspace(Z_RANGE[0], Z_RANGE[1], 6)
    xs, ys = [], []
    for i in range(5):
        m = (tt[:, 2] >= bins[i]) & (tt[:, 2] < bins[i + 1])
        if m.sum() > 20:
            xs.append(float(tt[m, 2].mean()))
            ys.append(float(np.median(err[m])))
            log(dict(stage="depth_bin", z_m=round(xs[-1], 3),
                     depth_err_mm=round(ys[-1], 2)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(xs, ys, "o-", color=COLORS[0])
    ax[0].set_xlabel("true distance (m)"); ax[0].set_ylabel("median depth error (mm)")
    ax[0].set_title("depth is inferred from apparent size")
    ax[1].hist(err, bins=40, color=COLORS[1], alpha=0.8, label="along the camera axis")
    ax[1].hist(lat, bins=40, color=COLORS[2], alpha=0.8, label="sideways")
    ax[1].set_xlabel("translation error (mm)"); ax[1].legend(fontsize=8)
    ax[1].set_title("depth is much harder than sideways position")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth.png"))
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    stage_baseline()
    stage_representations()
    stage_symmetry()
    stage_occlusion()
    stage_depth()

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
