"""Project 38 — a 0.5B language model writes the shot list, the video model shoots it.

    python3 run.py --stage plan       # ~6 min  the LLM writes 8 stories x 3 ways
    python3 run.py --stage render     # ~4 min  shoot every plan, grade the result
    python3 run.py --stage figures    # ~1 min

The language model is real (Qwen2.5-0.5B-Instruct, running on the CPU) and so
is the video model (project 35's).  Nothing about the planning half is
simulated.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from PIL import Image                                          # noqa: E402

import plan_lib as PL                                          # noqa: E402
LL = PL.LL
T, L = LL.T, LL.L

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

REPEATS = 4
SEED = 38


# --------------------------------------------------------------------------
# stage: plan
# --------------------------------------------------------------------------

def plan():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {}
    for arm in PL.ARMS:
        rows = []
        for story in PL.STORIES:
            if arm == "random":
                # The control: a plan with no thought behind it at all.  It
                # still renders, which is the point — "it produced a video" is
                # not evidence that the planning worked.
                p = [{"shot": i + 1, "subject": int(rng.integers(10)),
                      "motion": PL.MOTIONS[int(rng.integers(4))],
                      "caption": ""} for i in range(PL.N_SHOTS)]
                raw, parsed = "", p
            elif arm == "constrained":
                raw, parsed = "", PL.constrained_plan(story)
            else:
                raw = PL.generate_plan(story, few_shot=(arm == "few_shot"))
                parsed = PL.extract_json(raw)
            ok, why = (False, ["no JSON"]) if parsed is None \
                else PL.validate(parsed)
            fixed, changed = PL.repair(parsed if parsed is not None else [])
            rows.append(dict(story=story, arm=arm, raw=raw, parsed=parsed,
                             valid=ok, why=why, plan=fixed, repairs=changed,
                             **PL.plan_stats(fixed)))
            print(f"[plan] {arm:<12} valid={ok} repairs={changed} "
                  f"subj={rows[-1]['subject']} "
                  f"{[s['motion'] for s in fixed]}", flush=True)
        out[arm] = rows
    torch.save(out, CK / "plans.pt")
    with open(OUT / "plans.json", "w") as f:
        json.dump({a: [{k: v for k, v in r.items() if k != "parsed"}
                       for r in rs] for a, rs in out.items()}, f, indent=1)
    summary = []
    for arm, rows in out.items():
        summary.append(dict(
            arm=arm,
            json_valid=float(np.mean([r["valid"] for r in rows])),
            repairs_per_plan=float(np.mean([r["repairs"] for r in rows])),
            subject_consistency=float(np.mean([r["subject_consistency"]
                                               for r in rows])),
            motion_variety=float(np.mean([r["motion_variety"]
                                          for r in rows]))))
        print(f"[plan] {summary[-1]}", flush=True)
    with open(OUT / "planning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[plan] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: render
# --------------------------------------------------------------------------

@torch.no_grad()
def render():
    t0 = time.time()
    plans = torch.load(CK / "plans.pt", weights_only=False)
    model, bank, _ = LL.load_base()
    judge, _ = T.load_digit_judge()
    rows, keep = [], {}
    for arm, prows in plans.items():
        for si, r in enumerate(prows):
            schedule, subjects = PL.to_render(r["plan"])
            digits = torch.full((REPEATS,), subjects[0])
            dsched = [torch.full((REPEATS,), s) for s in subjects]
            _, pix = LL.generate_long(model, bank, digits, "anchored",
                                      schedule=schedule, seed=SEED + si,
                                      digit_schedule=dsched)
            pos, votes = LL.digit_votes(pix, judge)
            # grade each window against the subject the PLAN asked for there
            track = LL.latent_direction_track(schedule)
            per_frame_subject = []
            for k, (s, e) in enumerate(LL.window_slices(schedule)):
                st = (s + (LL.OVERLAP if k else 0))
                for lat in range(st, e):
                    per_frame_subject.append(subjects[k])
            want = torch.tensor([per_frame_subject[p // LL.PIX_PER_LAT]
                                 for p in pos])
            _, drift = LL.identity_drift(pix)
            row = dict(arm=arm, story=r["story"],
                       shots=len(schedule),
                       subject_acc=float((votes == want[None]).float().mean()),
                       direction_follow=float(
                           LL.direction_follow(pix, schedule).mean()),
                       identity_drift_end=float(drift[:, -1].mean()),
                       digit_stable=float((votes == votes[:, :1]).float()
                                          .mean()),
                       path_jerk=float(LL.path_jerk(pix).mean()))
            rows.append(row)
            print(f"[render] {row}", flush=True)
            if si == 0:
                keep[arm] = pix[:1]
    torch.save({"rows": rows, "keep": keep}, CK / "render.pt")
    with open(OUT / "render.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[render] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def save_gif(clip, path, scale=2, ms=90):
    x = ((clip.clamp(-1, 1) + 1) / 2)[0, 0].numpy()
    fr = [Image.fromarray((f * 255).astype(np.uint8)).resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in x]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=ms, loop=0)


def figures():
    plans = torch.load(CK / "plans.pt", weights_only=False)
    rend = torch.load(CK / "render.pt", weights_only=False)
    rows = rend["rows"]
    arms = PL.ARMS

    summary = []
    for a in arms:
        pr = plans[a]
        rr = [r for r in rows if r["arm"] == a]
        summary.append(dict(
            arm=a,
            json_valid=float(np.mean([r["valid"] for r in pr])),
            repairs=float(np.mean([r["repairs"] for r in pr])),
            subject_consistency=float(np.mean([r["subject_consistency"]
                                               for r in pr])),
            motion_variety=float(np.mean([r["motion_variety"] for r in pr])),
            subject_acc=float(np.mean([r["subject_acc"] for r in rr])),
            direction_follow=float(np.mean([r["direction_follow"]
                                            for r in rr])),
            digit_stable=float(np.mean([r["digit_stable"] for r in rr])),
            identity_drift_end=float(np.mean([r["identity_drift_end"]
                                              for r in rr]))))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(summary, flush=True)

    # ---- 1. does the plan survive contact with a parser? ------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ps.style_axes(axes[0])
    xs = np.arange(len(arms))
    axes[0].bar(xs - 0.2, [s["json_valid"] for s in summary], 0.38,
                color=ps.SERIES[0], label="valid JSON, unrepaired")
    axes[0].bar(xs + 0.2, [s["subject_consistency"] for s in summary], 0.38,
                color=ps.SERIES[1], label="same character in every shot")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(arms, rotation=14, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("fraction of the 8 stories")
    ps.style_axes(axes[1])
    axes[1].bar(xs - 0.2, [s["direction_follow"] for s in summary], 0.38,
                color=ps.SERIES[2], label="video follows the planned motion")
    axes[1].bar(xs + 0.2, [s["digit_stable"] for s in summary], 0.38,
                color=ps.SERIES[3], label="character stays the same on screen")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(arms, rotation=14, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.08)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Left: the plan.   Right: the video that plan produced.",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "planning.png", dpi=150)
    plt.close(fig)

    # ---- 2. one rendered story per arm ------------------------------------
    fig, axes = plt.subplots(len(arms), 1, figsize=(9.8, 1.3 * len(arms)))
    for ax, a in zip(axes, arms):
        ax.imshow(LL.contact_sheet(rend["keep"][a], every=3), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_ylabel(a, rotation=0, ha="right", va="center", fontsize=9,
                      color=ps.INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"'{PL.STORIES[0]}' — every 3rd frame of the rendered plan",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "stories.png", dpi=150)
    plt.close(fig)

    for a in arms:
        save_gif(rend["keep"][a], OUT / f"story_{a}.gif")

    # ---- 3. an actual shot list, as text ---------------------------------
    lines = []
    for a in arms:
        r = plans[a][0]
        lines.append(f"### {a}")
        lines.append(f"story: {r['story']}")
        if r["raw"]:
            lines.append("raw reply:")
            lines.append(r["raw"].strip()[:600])
        lines.append("plan used:")
        for sh in r["plan"]:
            lines.append(f'  shot {sh["shot"]}: subject {sh["subject"]}, '
                         f'{sh["motion"]:<6} {sh["caption"]}')
        lines.append("")
    (OUT / "example_plans.txt").write_text("\n".join(lines))
    print("[figures] wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["plan", "render", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    globals()[args.stage]()
