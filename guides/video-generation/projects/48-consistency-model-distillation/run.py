"""Project 48 — Consistency-model distillation.

Project 45's generator is a 30-step rectified-flow teacher: to make one clip it
walks noise -> clean in ~30 small steps.  Here we distil a *student* that jumps
to the finished clip in 1 or 4 steps, and we chart the bargain: steps saved
(speed) against quality lost.

The student is a consistency model.  A rectified-flow teacher outputs a
*direction to move*; the student outputs the *destination* — the clean clip it
thinks this noisy input will become.  It is trained so that from *any* point on
the teacher's noise->clean path it names the same destination.  That is the
"consistency" the name refers to: every point on one path must agree on where
the path ends.  If they agree, one look is already an answer, so you can skip
the walk.

Where project 44 sits next to this
----------------------------------
Project 44 also distilled a consistency student, but asked a different question:
raw milliseconds per frame, for real-time interactivity on a world model.  This
project fixes the eye on the *quality-versus-steps curve* of a text-to-video
model — how much does each halving of the step count actually cost? — and finds
the same uncomfortable answer from the other side.

Stages
    cache    generate teacher clips (the distillation targets)
    distill  train the student to reproduce them in one jump
    bench    score teacher @ many step counts vs student @ 1 and 4
    figures  draw the speed-quality curve
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "45-run-vbench-end-to-end"))
import eval_lib as E                                            # noqa: E402
import flow_lib as FL                                           # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CK = Path(__file__).resolve().parent / "checkpoints"
CK.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
def stage_cache(args):
    """Generate a fixed pile of teacher clips to distil against."""
    teacher = E.load_gen("base")
    ds = E.make_dataset(6000, seed=11)
    n = args.n
    clips, caps = [], []
    g = torch.Generator().manual_seed(0)
    t = time.time()
    for i in range(0, n, 256):
        idx = np.arange(i, min(i + 256, n))
        cap = E.caption_tensor(ds, idx)
        c = E.sample(teacher, cap, steps=30, scale=2.0, generator=g)
        clips.append(c.clamp(0, 1))
        caps.append(cap)
    clips = torch.cat(clips).float()
    caps = torch.cat(caps)
    torch.save({"clips": clips, "caps": caps}, CK / "teacher.pt")
    print(f"cached {clips.shape[0]} teacher clips in {time.time() - t:.0f}s")


# ---------------------------------------------------------------------------
@torch.no_grad()
def student_sample(student, cap, steps=1, generator=None):
    """Multi-step consistency sampling.

    1 step: from pure noise, the student names the clean clip directly.
    k steps: name a clean clip, add a *smaller* amount of fresh noise, name
    again — each pass cleans up the last one's mistakes.  This is the standard
    consistency-model multistep sampler.
    """
    flow = FL.RectifiedFlow()
    n = cap.shape[0]
    shape = (n, E.T, E.H, E.W)
    x = torch.randn(shape, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(n) * flow.T_SCALE
        x0 = student(x, t, cap)
        if i + 1 < steps:
            t_next = ts[i + 1]
            noise = torch.randn(shape, generator=generator)
            x = (1 - t_next) * x0 + t_next * noise
        else:
            x = x0
    return x


def stage_distill(args):
    data = torch.load(CK / "teacher.pt", weights_only=False)
    clips, caps = data["clips"], data["caps"]
    student = E.VideoGen(base=32)             # same body; now predicts x0
    flow = FL.RectifiedFlow()
    opt = torch.optim.Adam(student.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    n = clips.shape[0]
    losses = []
    t0 = time.time()
    student.train()
    for step in range(args.steps):
        idx = rng.integers(0, n, size=128)
        x0 = clips[idx]
        cap = caps[idx].clone()
        mask = torch.rand(128, generator=g) < 0.1
        if mask.any():
            cap[mask] = E.null_caption(int(mask.sum()))
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(128, generator=g)
        xt = flow.interpolate(x0, t, noise)
        pred = student(xt, t * flow.T_SCALE, cap)          # predict the endpoint
        loss = F.mse_loss(pred, x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 400 == 0:
            print(f"  distill step {step:4d} loss {np.mean(losses[-100:]):.4f}")
    print(f"distilled in {time.time() - t0:.0f}s")
    E.save_gen(student, "student", base=32, where=CK)
    with open(OUT / "distill_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for i, l in enumerate(losses):
            if i % 20 == 0:
                w.writerow([i, f"{l:.5f}"])


# ---------------------------------------------------------------------------
def _time_per_clip(fn, cap):
    g = torch.Generator().manual_seed(1)
    fn(cap[:64], g)                                        # warm up
    t = time.time()
    fn(cap[:256], torch.Generator().manual_seed(2))
    return (time.time() - t) / 256 * 1000                 # ms per clip


def stage_bench(args):
    teacher = E.load_gen("base")
    student = E.load_gen("student", where=CK)
    ds = E.make_dataset(600, seed=77)
    idx = np.arange(300)
    caps = E.caption_tensor(ds, idx)
    caps_np = caps.numpy()
    rows = []

    def record(name, steps, sampler):
        gen = sampler(caps, torch.Generator().manual_seed(3)).clamp(0, 1).numpy()
        s = E.vbench_score(gen, caps_np)
        ms = _time_per_clip(lambda c, g: sampler(c, g), caps)
        rows.append(dict(model=name, steps=steps, ms=ms, **s))
        print(f"{name:12s} @{steps:2d}  {ms:6.1f} ms  align "
              f"{s['text_alignment']:.2f}  sharp {s['imaging_quality']:.2f}  "
              f"overall {s['overall']:.2f}")
        return gen[:6]

    saved = {}
    for st in [30, 8, 4, 2, 1]:
        saved[f"teacher@{st}"] = record("teacher", st,
            lambda c, g, st=st: E.sample(teacher, c, steps=st, scale=2.0,
                                         generator=g))
    for st in [4, 1]:
        saved[f"student@{st}"] = record("student", st,
            lambda c, g, st=st: student_sample(student, c, steps=st,
                                               generator=g))
    np.save(OUT / "_samples.npy", saved, allow_pickle=True)
    axes = list(E.AXES) + ["overall"]
    with open(OUT / "bench.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "steps", "ms_per_clip"] + axes)
        for r in rows:
            w.writerow([r["model"], r["steps"], f"{r['ms']:.2f}"]
                       + [f"{r[a]:.4f}" for a in axes])


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (OUT / "bench.csv").exists():
        rows = list(csv.DictReader(open(OUT / "bench.csv")))
        fig, ax = plt.subplots(figsize=(7.6, 5))
        for model, color, mark in [("teacher", "#2b6fc9", "o"),
                                   ("student", "#c98a2b", "s")]:
            sub = [r for r in rows if r["model"] == model]
            xs = [float(r["ms_per_clip"]) for r in sub]
            ys = [float(r["overall"]) for r in sub]
            ax.plot(xs, ys, mark + "-", color=color, label=model, ms=8)
            for r, x, y in zip(sub, xs, ys):
                ax.annotate(f"{r['steps']} step", (x, y),
                            textcoords="offset points", xytext=(6, 5),
                            fontsize=8, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("milliseconds per clip (log scale — left is faster)")
        ax.set_ylabel("overall VBench-style score")
        ax.set_title("Few-stepping is nearly free down to 4 steps; the distilled\n"
                     "student's payoff appears at 1 step, where the teacher collapses")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "speed_quality.png", dpi=110)
        plt.close(fig)

    if (OUT / "_samples.npy").exists():
        s = np.load(OUT / "_samples.npy", allow_pickle=True).item()
        order = ["teacher@30", "teacher@4", "teacher@1", "student@4",
                 "student@1"]
        rows = [list(s[k][0]) for k in order if k in s]
        E.strip(rows, OUT / "teacher_vs_student.png", scale=6)
        if "student@1" in s:
            E.write_gif(s["student@1"][0], OUT / "student_1step.gif")
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "distill", "bench", "figures"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=1600)
    a = ap.parse_args()
    {"cache": stage_cache, "distill": stage_distill,
     "bench": stage_bench, "figures": stage_figures}[a.stage](a)
