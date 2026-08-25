"""Project 46 — Recaption a dataset.

Same clips, two sets of captions.  We train one generator on realistic *web*
captions (they name the object but rarely its motion) and one on *recaptioned*
data (a VLM watched each clip and wrote the motion down too).  Then we measure
how well each follows a held-out prompt.  The gap is the recaptioning win, and
it lands exactly on the attributes the web captions were missing.

Stages
    train    train the web-caption model and the recaptioned model
    eval     measure per-attribute prompt following for both
    figures  draw it

Why this is the same clips, only different words
------------------------------------------------
Recaptioning does not touch a single pixel of training video.  It rewrites the
*labels*.  So the only thing that can change between the two models is what the
captions taught — which is exactly what makes the comparison clean.
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


# ---------------------------------------------------------------------------
# the two captioners
# ---------------------------------------------------------------------------
def web_caption(ds, idx, rng):
    """Realistic bad web captions.

    A YouTube description or alt-text almost always names the *thing* (our
    shape) but hardly ever the *motion*.  So: keep the shape, but drop the
    direction to "unknown" most of the time, and when it is present make it
    wrong as often as a careless human tag would be.  Speed is almost never
    mentioned at all.
    """
    n = len(idx)
    shape = torch.from_numpy(ds["shape"][idx]).long()
    # direction: 70% unknown, 15% correct, 15% a random (often wrong) guess
    roll = rng.random(n)
    direction = np.full(n, E.UNK["dir"], dtype=np.int64)
    keep = roll > 0.85
    direction[keep] = ds["dir"][idx][keep]
    guess = (roll > 0.70) & (roll <= 0.85)
    direction[guess] = rng.integers(0, E.N_DIR, size=guess.sum())
    # speed: named only 10% of the time
    speed = np.where(rng.random(n) < 0.10, ds["speed"][idx],
                     E.UNK["speed"]).astype(np.int64)
    return torch.stack([shape, torch.from_numpy(direction),
                        torch.from_numpy(speed)], dim=1)


def recaptioned(ds, idx, rng):
    """A strong VLM watched the clip: every attribute correct."""
    return E.caption_tensor(ds, idx)


# ---------------------------------------------------------------------------
def stage_train(args):
    torch.manual_seed(0)
    ds = E.make_dataset(3000, seed=1)
    for name, cap_fn in [("web", web_caption), ("recap", recaptioned)]:
        print(f"=== training on {name} captions ===")
        net = E.VideoGen(base=32)
        t = time.time()
        E.train(net, ds, steps=args.steps, batch=128, lr=2e-3, seed=0,
                corrupt=cap_fn)
        print(f"  {name} trained in {time.time() - t:.0f}s")
        E.save_gen(net, f"gen_{name}", base=32, where=CK)


# ---------------------------------------------------------------------------
def stage_eval(args):
    # held-out prompts: every one of the 16 captions, many samples each
    caps = []
    for (s, d, v) in E.COMBOS:
        caps += [[s, d, v]] * 24
    caps = np.array(caps)
    cap_t = torch.from_numpy(caps).long()
    rows = {}
    for name in ["web", "recap"]:
        net = E.load_gen(f"gen_{name}", where=CK)
        gen = E.sample(net, cap_t, steps=25, scale=2.0,
                       generator=torch.Generator().manual_seed(0)).numpy()
        gen = np.clip(gen, 0, 1)
        al = E.text_alignment(gen, caps)
        rows[name] = al
        print(f"{name:6s}  dir {al['direction']:.2f}  speed {al['speed']:.2f}  "
              f"shape {al['shape']:.2f}  mean {al['mean']:.2f}")
        np.save(OUT / f"_gen_{name}.npy", gen[:48])
        np.save(OUT / "_caps.npy", caps[:48])
    with open(OUT / "eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["captions", "direction", "speed", "shape", "mean"])
        for name in ["web", "recap"]:
            a = rows[name]
            w.writerow([name, f"{a['direction']:.4f}", f"{a['speed']:.4f}",
                        f"{a['shape']:.4f}", f"{a['mean']:.4f}"])


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (OUT / "eval.csv").exists():
        rows = {r["captions"]: r for r in csv.DictReader(open(OUT / "eval.csv"))}
        attrs = ["direction", "speed", "shape"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(len(attrs))
        for j, name in enumerate(["web", "recap"]):
            vals = [float(rows[name][a]) for a in attrs]
            ax.bar(x + (j - 0.5) * 0.36, vals, 0.36,
                   label={"web": "web captions (motion missing)",
                          "recap": "recaptioned (VLM watched it)"}[name],
                   color=["#8a8f98", "#c98a2b"][j])
        ax.set_xticks(x)
        ax.set_xticklabels(attrs)
        ax.set_ylabel("prompt-following accuracy")
        ax.axhline(0.25, ls=":", color="gray", lw=1)
        ax.text(2.35, 0.27, "chance (direction)", fontsize=7, color="gray")
        ax.set_ylim(0, 1.05)
        ax.set_title("Recaptioning's win lands on the attributes the\n"
                     "web captions never mentioned — not on shape")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "recaption_gain.png", dpi=110)
        plt.close(fig)

    # side-by-side clips for the same prompt
    if (OUT / "_gen_web.npy").exists():
        web = np.load(OUT / "_gen_web.npy")
        rec = np.load(OUT / "_gen_recap.npy")
        caps = np.load(OUT / "_caps.npy")
        # pick a few prompts that ask for a clear direction
        picks = [i for i in range(len(caps)) if caps[i][1] in (0, 3)][:4]
        rows = []
        for i in picks:
            rows.append(list(web[i]))
            rows.append(list(rec[i]))
        E.strip(rows, OUT / "web_vs_recap.png", scale=5)
        for i in picks[:1]:
            E.write_gif(web[i], OUT / "web_clip.gif")
            E.write_gif(rec[i], OUT / "recap_clip.gif")
    print("figures written")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "eval", "figures"])
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    {"train": stage_train, "eval": stage_eval,
     "figures": stage_figures}[a.stage](a)
