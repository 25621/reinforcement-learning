"""Project 44 — how few network passes can one frame cost?

    python3 run.py --stage distill    # ~6 min
    python3 run.py --stage bench      # ~4 min
    python3 run.py --stage figures    # ~1 min

The teacher is project 41's `aug` world model: 30 denoising steps per frame.
Everything here is measured on the same 150-frame rollouts project 41 used, so
the quality numbers are directly comparable with its drift curves.

Three families are compared:

    teacher @ N steps   the same weights, sampled with fewer steps
    student  @ N steps  a distilled model trained to jump to the answer
    regressor           project 43's one-pass world model -- no diffusion at
                        all, the hard floor on how fast this can possibly get
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
sys.path.insert(0, str(HERE.parent / "41-gamengen-reproduction-mini"))
sys.path.insert(0, str(HERE.parent / "43-world-model-for-rl"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import world_lib as W                                          # noqa: E402
import distill_lib as DL                                       # noqa: E402

P41 = HERE.parent / "41-gamengen-reproduction-mini"
sys.path.insert(0, str(P41))

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.set_num_threads(12)

CTX, HORIZON = 2, 1
TEACHER_STEPS = 30
INFER_SIGMA = 0.05
DISTILL_STEPS, BATCH, LR = 2500, 64, 3e-4
ROLL_LEN, N_ROLL = 100, 24


def teacher():
    import run as R41
    net = W.ActionUNet(ctx=CTX, horizon=HORIZON, ctx_noise=True)
    net.load_state_dict(torch.load(P41 / "checkpoints" / "aug.pt"))
    net.eval()
    return net


# --------------------------------------------------------------------------
# distil
# --------------------------------------------------------------------------

CACHE_BATCHES = 150           # teacher paths to generate, in batches of BATCH
POINTS_PER_PATH = 6           # student training points harvested per path


def _build_cache(tea, ep, rng):
    """Run the teacher once, keep many training points per run.

    The naive loop -- generate a fresh teacher path for every student update --
    costs 30 teacher passes per update and turns a five-minute job into a
    twenty-minute one.  But one path already contains everything the student
    needs: 30 different starting points that all end at the *same* answer.  So
    we generate the paths once, harvest several points from each, and then
    train the student on that fixed set for as long as we like.
    """
    flow = W.FL.RectifiedFlow()
    Ctx, Act, Sig, X, T, X0 = [], [], [], [], [], []
    t0 = time.time()
    for b in range(CACHE_BATCHES):
        ctx, _, act, _ = W.sample_batch(ep, BATCH, CTX, HORIZON, rng)
        sigma = torch.full((BATCH,), INFER_SIGMA)
        ctx = ctx + sigma[:, None, None, None] * torch.randn_like(ctx)
        xs, ts = DL.teacher_path(tea, ctx, act, TEACHER_STEPS, sigma=sigma)
        i = torch.randint(0, TEACHER_STEPS, (POINTS_PER_PATH, BATCH))
        cols = torch.arange(BATCH).expand(POINTS_PER_PATH, BATCH)
        X.append(xs[i, cols].reshape(-1, HORIZON, W.GRID, W.GRID))
        T.append(ts[i].reshape(-1))
        X0.append(xs[-1].repeat(POINTS_PER_PATH, 1, 1, 1))
        Ctx.append(ctx.repeat(POINTS_PER_PATH, 1, 1, 1))
        Act.append(act.repeat(POINTS_PER_PATH, 1))
        Sig.append(sigma.repeat(POINTS_PER_PATH))
        if (b + 1) % 50 == 0:
            print(f"  cache {b+1}/{CACHE_BATCHES} teacher paths "
                  f"({time.time()-t0:.0f}s)")
    return (torch.cat(Ctx), torch.cat(Act), torch.cat(Sig), torch.cat(X),
            torch.cat(T) * flow.T_SCALE, torch.cat(X0))


def stage_distill():
    ep = W.load_episodes("train", where=P41 / "checkpoints")
    tea = teacher()
    for p in tea.parameters():
        p.requires_grad_(False)
    torch.manual_seed(44)
    rng = np.random.default_rng(44)
    ctxs, acts, sigs, xs, ts, x0s = _build_cache(tea, ep, rng)
    print(f"  {len(xs)} student training points")

    student = DL.Student(ctx=CTX, horizon=HORIZON)
    student.net.load_state_dict(tea.state_dict())     # warm start
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DISTILL_STEPS)
    g = torch.Generator().manual_seed(44)
    t0, log = time.time(), []
    for step in range(1, DISTILL_STEPS + 1):
        i = torch.randint(0, len(xs), (BATCH,), generator=g)
        pred = student(xs[i], ts[i], ctxs[i], acts[i], sigma=sigs[i])
        loss = F.mse_loss(pred, x0s[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            log.append([step, f"{loss.item():.6f}"])
            print(f"  distill {step:5d}/{DISTILL_STEPS} loss "
                  f"{loss.item():.5f} ({time.time()-t0:.0f}s)")
    torch.save(student.state_dict(), CK / "student.pt")
    print(f"  student: {W.count_params(student)} params, "
          f"{time.time()-t0:.0f}s")
    with open(OUT / "distill_loss.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "loss"])
        wr.writerows(log)


def load_student():
    s = DL.Student(ctx=CTX, horizon=HORIZON)
    s.load_state_dict(torch.load(CK / "student.pt"))
    s.eval()
    return s


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------

def _eval_set():
    long = W.load_episodes("long", where=P41 / "checkpoints")
    walls = long["walls"][:N_ROLL]
    start = [(long["agent"][e, CTX - 1] // W.GRID,
              long["agent"][e, CTX - 1] % W.GRID) for e in range(N_ROLL)]
    ctx0 = torch.from_numpy(
        W.frames_of(long, np.repeat(np.arange(N_ROLL)[:, None], CTX, 1),
                    np.repeat(np.arange(CTX)[None], N_ROLL, 0)))
    acts = torch.from_numpy(long["act"][:N_ROLL, CTX - 1:CTX - 1 + ROLL_LEN])
    return ctx0, acts, walls, start


def _score(frames, walls, start, acts):
    """Drift-robust quality: 'button obeyed' (given where the MODEL put the
    player last frame, did this frame move it the way the button says?), the
    fraction of legal screens (snap), and whether a coin exists.

    We deliberately do NOT score cell-match against the real trajectory: over a
    100-frame self-fed rollout that is a ratchet that collapses to chance for
    teacher and student alike, and it would hide exactly the quality difference
    distillation is supposed to preserve.
    """
    b, n = frames.shape[:2]
    obeyed = snap = has = 0.0
    total = 0
    for e in range(b):
        prev = start[e]
        for k in range(n):
            sym, ai, _, sn = W.read_frame(frames[e, k])
            cur = (ai // W.GRID, ai % W.GRID)
            dr, dc = W.DELTA[int(acts[e, k])]
            nr, nc = prev[0] + dr, prev[1] + dc
            legal = prev if (not (0 <= nr < W.GRID and 0 <= nc < W.GRID)
                             or walls[e][nr, nc]) else (nr, nc)
            obeyed += (cur == legal)
            snap += sn
            has += W.coin_report(frames[e, k], walls[e])[0]
            prev = cur
            total += 1
    return obeyed / total, snap / total, has / total


@torch.no_grad()
def _rollout_student(student, ctx0, acts, steps, sigma):
    ctx = ctx0.clone()
    out = []
    g = torch.Generator().manual_seed(44)
    for i in range(acts.shape[1]):
        f = student.sample(ctx, acts[:, i:i + 1], steps=steps, generator=g,
                           sigma=sigma)
        out.append(f[:, -1])
        ctx = torch.cat([ctx[:, 1:], f[:, -1:]], dim=1)
    return torch.stack(out, dim=1)


def _time_one_frame(fn, n=25):
    fn()                                   # warm up
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


def stage_bench():
    ctx0, acts, walls, start = _eval_set()
    an = acts.numpy()
    tea = teacher()
    student = load_student()
    sig = torch.full((N_ROLL,), INFER_SIGMA)
    sig1 = torch.full((1,), INFER_SIGMA)
    one_ctx, one_act = ctx0[:1], acts[:1, :1]
    rows = []

    for steps in [30, 8, 4, 2, 1]:
        g = torch.Generator().manual_seed(44)
        fr = W.rollout(tea, ctx0, acts, steps=steps, sigma=sig,
                       generator=g).numpy()
        obeyed, snap, has = _score(fr, walls, start, an)
        ms = _time_one_frame(lambda: W.sample_frames(
            tea, one_ctx, one_act, steps=steps, sigma=sig1))
        rows.append(dict(method="teacher", steps=steps, ms_per_frame=ms,
                         fps=1000 / ms, button_obeyed=obeyed, snap=snap,
                         has_coin=has))
        print(f"teacher @{steps:2d}  {ms:7.1f} ms  obeyed {obeyed:.3f}  "
              f"snap {snap:.4f}  coin {has:.3f}")

    for steps in [4, 2, 1]:
        fr = _rollout_student(student, ctx0, acts, steps, sig).numpy()
        obeyed, snap, has = _score(fr, walls, start, an)
        ms = _time_one_frame(lambda: student.sample(
            one_ctx, one_act, steps=steps, sigma=sig1))
        rows.append(dict(method="student", steps=steps, ms_per_frame=ms,
                         fps=1000 / ms, button_obeyed=obeyed, snap=snap,
                         has_coin=has))
        print(f"student @{steps:2d}  {ms:7.1f} ms  obeyed {obeyed:.3f}  "
              f"snap {snap:.4f}  coin {has:.3f}")
        if steps == 1:
            W.write_gif(fr[0, :80], OUT / "student_1step.gif")

    # the no-diffusion floor
    reg = _load_regressor()
    if reg is not None:
        fr = _rollout_regressor(reg, ctx0, acts).numpy()
        obeyed, snap, has = _score(fr, walls, start, an)
        ms = _time_one_frame(lambda: reg(one_ctx[:, -1], one_act[:, 0]))
        rows.append(dict(method="regressor (project 43)", steps=1,
                         ms_per_frame=ms, fps=1000 / ms, button_obeyed=obeyed,
                         snap=snap, has_coin=has))
        print(f"regressor    {ms:7.1f} ms  obeyed {obeyed:.3f}  snap {snap:.4f}  "
              f"coin {has:.3f}")

    with open(OUT / "bench.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items()})
    print(f"wrote {OUT / 'bench.csv'}")


def _load_regressor():
    p = HERE.parent / "43-world-model-for-rl" / "checkpoints" / "world.pt"
    if not p.exists():
        print("(skipping the regressor floor: run project 43's `world` stage "
              "first)")
        return None
    import dream_lib as D
    m = D.DreamWorld()
    m.load_state_dict(torch.load(p))
    m.eval()

    @torch.no_grad()
    def call(f, a):
        return m(f, a)[0]
    return call


@torch.no_grad()
def _rollout_regressor(reg, ctx0, acts):
    f = ctx0[:, -1]
    out = []
    for i in range(acts.shape[1]):
        f = reg(f, acts[:, i]).clamp(0, 1)
        out.append(f)
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def stage_figures():
    rows = list(csv.DictReader(open(OUT / "bench.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    ax = axes[0]
    ps.style_axes(ax)
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    for i, (name, rs) in enumerate(groups.items()):
        rs = sorted(rs, key=lambda r: float(r["ms_per_frame"]))
        ax.plot([float(r["ms_per_frame"]) for r in rs],
                [float(r["button_obeyed"]) for r in rs], marker="o",
                color=ps.SERIES[i], label=name)
        for r in rs:
            ax.annotate(f"{r['steps']}", (float(r["ms_per_frame"]),
                                          float(r["button_obeyed"])),
                        textcoords="offset points", xytext=(4, 5), fontsize=7,
                        color=ps.INK_MUTED)
    ax.axvline(33.3, color=ps.SERIES[2], ls="--", lw=1.3)
    ax.text(33.3 * 1.05, 0.05, "33 ms = 30 fps", color=ps.SERIES[2],
            fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Quality against the clock", color=ps.INK, fontsize=11,
                 loc="left")
    ax.set_xlabel("ms per frame (lower is better)", color=ps.INK_SECONDARY,
                  fontsize=9)
    ax.set_ylabel("button obeyed (drift-robust quality)",
                  color=ps.INK_SECONDARY, fontsize=9)

    ax = axes[1]
    ps.style_axes(ax)
    labels = [f"{r['method'].split()[0]} @{r['steps']}" for r in rows]
    ms = [float(r["ms_per_frame"]) for r in rows]
    ax.barh(np.arange(len(rows)), ms,
            color=[ps.SERIES[0] if r["method"] == "teacher"
                   else ps.SERIES[1] if r["method"] == "student"
                   else ps.BASELINE for r in rows])
    ax.axvline(33.3, color=ps.SERIES[2], ls="--", lw=1.3)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, v in enumerate(ms):
        ax.text(v * 1.08, i, f"{v:.1f}", va="center", fontsize=7,
                color=ps.INK_SECONDARY)
    ax.set_title("Milliseconds per generated frame", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("ms (log scale)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'latency.png'}")

    # side by side pictures
    ctx0, acts, walls, start = _eval_set()
    tea, student = teacher(), load_student()
    sig = torch.full((N_ROLL,), INFER_SIGMA)
    picks = list(range(0, 60, 8))
    real = W.load_episodes("long", where=P41 / "checkpoints")
    strips = [[W.frames_of(real, np.array([0]), np.array([CTX + k]))[0]
               for k in picks]]
    g = torch.Generator().manual_seed(44)
    fr = W.rollout(tea, ctx0, acts[:, :60], steps=30, sigma=sig,
                   generator=g).numpy()
    strips.append([fr[0, k] for k in picks])
    fr = _rollout_student(student, ctx0, acts[:, :60], 1, sig).numpy()
    strips.append([fr[0, k] for k in picks])
    W.strip_image(strips, OUT / "teacher_vs_student.png")
    print("teacher_vs_student.png rows: real game, teacher @30 steps, "
          "student @1 step")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["distill", "bench", "figures"])
    a = p.parse_args()
    {"distill": stage_distill, "bench": stage_bench,
     "figures": stage_figures}[a.stage]()


if __name__ == "__main__":
    main()
