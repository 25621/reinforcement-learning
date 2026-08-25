"""Project 30 — what a 77-token text encoder throws away, and what to do.

    python3 train.py --stage prompts               # ~1 min  the wall, measured
    python3 train.py --stage encode                # ~3 min  freeze the encoders
    python3 train.py --stage train --arm clip      # ~8 min
    python3 train.py --stage train --arm clip_chunk
    python3 train.py --stage train --arm t5
    python3 train.py --stage train --arm both
    python3 train.py --stage figures               # ~9 min

Four arms, one video DiT, the same captions.  Only the text encoder in front
of the model changes.
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
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import fid_lib                                                 # noqa: E402
import text_lib as T                                           # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"

STEPS, BATCH, LR = 2800, 16, 6e-4
DROP_PROMPT = 0.1
SAMPLE_STEPS, CFG = 30, 3.0
REPEATS = 2                    # 40 prompts x 2 = 80 clips per measurement
ARMS = ["clip", "clip_chunk", "t5", "both"]


# --------------------------------------------------------------------------
# stage: prompts — measure the wall before building anything on it
# --------------------------------------------------------------------------

def prompts():
    from transformers import AutoTokenizer, CLIPTokenizer, CLIPTextModel
    ctok = CLIPTokenizer.from_pretrained(T.CLIP_ID)
    ttok = AutoTokenizer.from_pretrained(T.T5_ID)

    rows = []
    for style in T.STYLES:
        p = T.make_prompt(7, 2, style, 0)
        n_clip = len(ctok(p, truncation=False)["input_ids"])
        n_t5 = len(ttok(p)["input_ids"])
        kept = min(n_clip, T.CLIP_LIMIT)
        # where in the token stream does the decisive clause sit?
        head = T.make_prompt(7, 2, style, 0).split("a 7 drifting")[0]
        clause_at = len(ctok(head, truncation=False)["input_ids"]) - 1
        rows.append(dict(style=style, what=T.STYLE_HELP[style],
                         clip_tokens=n_clip, clip_kept=kept,
                         clip_dropped=max(n_clip - T.CLIP_LIMIT, 0),
                         t5_tokens=n_t5, clause_starts_at_clip_token=clause_at,
                         clause_survives_clip=bool(clause_at < T.CLIP_LIMIT)))
        print(rows[-1], flush=True)
    with open(OUT / "token_budget.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- the proof: two prompts that differ only past token 77 -----------
    enc = CLIPTextModel.from_pretrained(T.CLIP_ID).eval()
    pairs = [("short", (7, 2), (1, 0)), ("long_early", (7, 2), (1, 0)),
             ("long_late", (7, 2), (1, 0))]
    lines = []
    with torch.no_grad():
        for style, a, b in pairs:
            pa = T.make_prompt(a[0], a[1], style, 0)
            pb = T.make_prompt(b[0], b[1], style, 0)
            bt = ctok([pa, pb], padding="max_length", truncation=True,
                      max_length=T.CLIP_LIMIT, return_tensors="pt")
            h = enc(**bt).last_hidden_state
            same_ids = bool((bt["input_ids"][0] == bt["input_ids"][1]).all())
            cos = float(F.cosine_similarity(h[0].flatten(), h[1].flatten(), 0))
            d = float((h[0] - h[1]).abs().max())
            lines.append(f"{style:<11} identical token ids after truncation: "
                         f"{str(same_ids):<5} cosine {cos:.4f}  "
                         f"max|difference| {d:.5f}")
            print(lines[-1], flush=True)
    text = ("Two prompts, 'a 7 drifting left' vs 'a 1 drifting right',\n"
            "compared AFTER CLIP's 77-token truncation:\n\n" +
            "\n".join(lines) + "\n\n"
            "For long_late the two captions reach CLIP as the same tokens, so\n"
            "the model receives the same numbers for both.  No amount of\n"
            "training can recover a difference that was deleted upstream.\n")
    (OUT / "truncation_proof.txt").write_text(text)
    fig_budget(rows)


def fig_budget(rows):
    fig, ax = ps.new_axes(7.6, 3.6)
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r["clip_tokens"] for r in rows], 0.34,
            color=ps.SERIES[0], label="CLIP-L tokens needed")
    ax.barh(y - 0.18, [r["t5_tokens"] for r in rows], 0.34,
            color=ps.SERIES[1], label="T5-base tokens needed")
    ax.axvline(T.CLIP_LIMIT, color=ps.SERIES[2], ls="--", lw=1.6)
    ax.text(T.CLIP_LIMIT + 2, len(rows) - 0.4, "CLIP's hard limit: 77",
            color=ps.SERIES[2], fontsize=9)
    for i, r in enumerate(rows):
        if r["clip_dropped"]:
            ax.text(r["clip_tokens"] + 2, i + 0.18,
                    f"-{r['clip_dropped']} dropped", fontsize=8,
                    color=ps.INK_MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r["style"] for r in rows])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "How long is the prompt, and how much fits?",
              "tokens", "", OUT / "token_budget.png")


# --------------------------------------------------------------------------
# stage: encode — run the frozen encoders once
# --------------------------------------------------------------------------

def encode():
    items = T.all_prompts()
    texts = [it[4] for it in items]
    print(f"{len(texts)} prompts", flush=True)
    for src in ("clip", "clip_chunk", "t5"):
        t0 = time.time()
        if src == "t5":
            seq, mask = T.encode_t5(texts)
            nseq, nmask = T.encode_t5([""])
        else:
            ch = src == "clip_chunk"
            seq, mask = T.encode_clip(texts, chunked=ch)
            nseq, nmask = T.encode_clip([""], chunked=ch)
        torch.save({"seq": seq, "mask": mask, "null_seq": nseq,
                    "null_mask": nmask,
                    "meta": [it[:4] for it in items]}, T.cache_path(src))
        print(f"[{src}] {tuple(seq.shape)} mask mean {mask.mean():.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(arm):
    torch.manual_seed(0)
    cache = L.load_latent_cache("latents", where=P25)
    data, digit, direction = (cache["latents"], cache["digit"],
                              cache["direction"])
    bank = T.TextBank(arm)
    model = T.build(arm)
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    print(f"[{arm}] {L.count_params(model):,} params", flush=True)

    # every clip can be described by any of the three prompt shapes
    style_ids = torch.arange(len(T.STYLES))
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        s = style_ids[torch.randint(0, len(T.STYLES), (BATCH,), generator=g)]
        f = torch.randint(0, T.N_FILLER, (BATCH,), generator=g)
        pidx = torch.tensor([
            T.prompt_index(int(digit[i]), int(direction[i]),
                           T.STYLES[int(si)], int(fi))
            for i, si, fi in zip(idx, s, f)])
        text = bank.get(pidx)
        drop = torch.rand(BATCH, generator=g) < DROP_PROMPT
        if drop.any():
            null = bank.null(BATCH)
            for k in text:
                seq, mask = text[k]
                nseq, nmask = null[k]
                seq, mask = seq.clone(), mask.clone()
                seq[drop] = nseq[drop][:, :seq.shape[1]]
                mask[drop] = nmask[drop][:, :mask.shape[1]]
                text[k] = (seq, mask)
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE, text),
                          flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm,
                "elapsed": time.time() - t0,
                "params": L.count_params(model)}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def run_style(model, bank, style, judge, seed=11):
    digits = torch.tensor([c[0] for c in T.COMBOS] * REPEATS)
    dirs = torch.tensor([c[1] for c in T.COMBOS] * REPEATS)
    fill = torch.arange(len(digits)) % T.N_FILLER
    pidx = torch.tensor([T.prompt_index(int(d), int(k), style, int(f))
                         for d, k, f in zip(digits, dirs, fill)])
    text = bank.get(pidx)
    null = bank.null(len(pidx))
    g = torch.Generator().manual_seed(seed)
    z = T.cfg_sample(model, text, null, (len(pidx),) + T.LATENT_SHAPE,
                     scale=CFG, steps=SAMPLE_STEPS, generator=g)
    clips = T.decode(z)
    dig, dr, both = T.grade(clips, digits, dirs, judge)
    return dict(digit_acc=dig, direction_acc=dr, both_acc=both), clips, \
        digits, dirs


@torch.no_grad()
def figures():
    judge, judge_acc = T.load_digit_judge()
    ev = L.load_latent_cache("latents_eval", where=P25)
    reals = ev["clips"][:80]
    fnet = fid_lib.load_features()

    rows, showcase = [], {}
    for arm in ARMS:
        model, ck = T.load_arm(arm)
        bank = T.TextBank(arm)
        for style in T.STYLES:
            t0 = time.time()
            acc, clips, digits, dirs = run_style(model, bank, style, judge)
            row = dict(arm=arm, style=style, params=ck["params"],
                       digit_acc=round(acc["digit_acc"], 3),
                       direction_acc=round(acc["direction_acc"], 3),
                       both_acc=round(acc["both_acc"], 3),
                       fid_proxy=round(fid_lib.frechet(reals, clips, fnet), 1),
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(row, flush=True)
            showcase[(arm, style)] = (clips, digits, dirs)
    rows.append(dict(arm="chance", style="", params="", digit_acc=0.1,
                     direction_acc=0.25, both_acc=0.025, fid_proxy="",
                     seconds=""))
    rows.append(dict(arm="real clips (judge ceiling)", style="", params="",
                     digit_acc=round(judge_acc, 3), direction_acc=1.0,
                     both_acc="",
                     fid_proxy=round(fid_lib.frechet(reals, ev["clips"][80:160],
                                                     fnet), 1), seconds=""))
    with open(OUT / "adherence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_adherence(rows)
    fig_grid(showcase)
    fig_loss()
    print("wrote", OUT)


def fig_adherence(rows):
    data = [r for r in rows if r["arm"] in ARMS]
    fig, ax = ps.new_axes(8.4, 4.4)
    x = np.arange(len(T.STYLES))
    width = 0.8 / len(ARMS)
    for i, arm in enumerate(ARMS):
        vals = [next(r["both_acc"] for r in data
                     if r["arm"] == arm and r["style"] == s) for s in T.STYLES]
        ax.bar(x + (i - (len(ARMS) - 1) / 2) * width, vals, width,
               color=ps.SERIES[i], label=arm)
    ax.axhline(0.025, color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.035, "chance (1 of 40)", color=ps.INK_MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({T.STYLE_HELP[s]})" for s in T.STYLES],
                       fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax,
              "Right digit AND right direction, by where the clause sits",
              "", "fraction correct", OUT / "adherence.png")


def fig_grid(showcase):
    picks = [(7, 2), (3, 0)]
    fig, axes = plt.subplots(len(picks) * len(T.STYLES), len(ARMS),
                             figsize=(10.5, 1.15 * len(picks) * len(T.STYLES)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    r = 0
    for d, k in picks:
        for style in T.STYLES:
            for c, arm in enumerate(ARMS):
                clips, digits, dirs = showcase[(arm, style)]
                hit = ((digits == d) & (dirs == k)).nonzero()[0, 0]
                ax = axes[r, c]
                ax.imshow(L.strip(clips[hit:hit + 1], n=6), cmap="gray",
                          vmin=0, vmax=1)
                ax.set_xticks([]), ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"'{d} {L.DIRECTIONS[k]}'\n{style}",
                                  color=ps.INK_SECONDARY, fontsize=7)
                if r == 0:
                    ax.set_title(arm, color=ps.INK, fontsize=10)
            r += 1
    fig.suptitle("Same request, three sentence shapes, four text front ends",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "prompt_grid.png", facecolor=ps.SURFACE)
    plt.close(fig)


def fig_loss():
    fig, ax = ps.new_axes(7.4, 4.0)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"log_{arm}.npy")
        k = 12
        sm = np.convolve(a[:, 2], np.ones(k) / k, mode="valid")
        ax.plot(a[k - 1:, 0], sm, color=ps.SERIES[i], lw=1.5, label=arm)
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Training loss (same objective for all four arms)",
              "training step", "flow-matching MSE", OUT / "loss_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prompts", "encode", "train", "figures"])
    ap.add_argument("--arm", default="t5", choices=ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "prompts":
        prompts()
    elif args.stage == "encode":
        encode()
    elif args.stage == "train":
        train(args.arm)
    else:
        figures()
