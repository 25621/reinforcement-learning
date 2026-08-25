"""Speech LLM: frozen Whisper encoder + trainable projector + frozen SmolLM2.

This is project 20's LLaVA recipe with the vision tower swapped for ears. The
LLM never changes, the encoder never changes; the only weights that learn are
the projector that rewrites audio vectors as things the LLM reads as words.

Two questions are asked of every clip, and they behave completely differently:

    "What digits are spoken?"  -> content the encoder was built to keep
    "Who is speaking?"         -> identity the encoder was built to discard

Stages
    data    build 2,600 two-digit utterances and cache frozen Whisper features
    train   one arm: mlp (real audio) / prefix (blind control) / early (layer 1)
    eval    exact-sequence accuracy, per-digit accuracy, speaker accuracy
    plot    figures

Arms
    mlp     26 pooled audio tokens from Whisper's LAST encoder layer
    early   the same, from Whisper's FIRST encoder layer
    prefix  26 learned vectors that never see the audio -- the control
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
import plot_style as ps  # noqa: E402
import speech_lib as SL  # noqa: E402
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS, BS, LR = 400, 8, 3e-3
TASKS = ("digits", "speaker")


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def build(arm, data, llm):
    """One projector per arm. `prefix` gets the identical machinery minus the
    audio: same token count, same training, same everything else."""
    n_tok = data.n_tokens()
    if arm == "prefix":
        return V.Projector("prefix", SL.AUDIO_DIM, llm.config.hidden_size,
                           out_rms=V.embedding_rms(llm), n_prefix=n_tok)
    return V.Projector("mlp2", SL.AUDIO_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))


def feats_for(arm, data, ids):
    if arm == "prefix":
        return torch.zeros(len(ids), data.n_tokens(), SL.AUDIO_DIM)
    return data.audio_tokens(ids)


# ---------------------------------------------------------------------------
def stage_data():
    t0 = time.time()
    SL.build_cache()
    d = SL.AudioLLMData()
    print(f"  {len(d.train_ids)} train / {len(d.seen_voice)} held-out takes / "
          f"{len(d.new_voice)} unheard voice  ({time.time() - t0:.0f}s)")
    print("  audio tokens per clip:", d.n_tokens())
    q, a = d.qa(0, "digits")
    print(f"  example: {q!r} -> {a!r}")
    _save("data.json", {"utterances": len(d.wav), "train": len(d.train_ids),
                        "seen_voice": len(d.seen_voice),
                        "new_voice": len(d.new_voice),
                        "audio_tokens": d.n_tokens(),
                        "encoder_frames_kept": SL.VALID_FRAMES,
                        "encoder_frames_total": 1500,
                        "seconds": time.time() - t0})


def stage_train(arm, layer=None):
    layer = layer or ("early" if arm == "early" else "last")
    data = SL.AudioLLMData(layer=layer)
    tok, llm = V.load_llm()
    torch.manual_seed(0)
    proj = build(arm, data, llm)
    vlm = V.TinyVLM(llm, proj)
    opt = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(0)
    n_tok = data.n_tokens()
    curve, t0 = [], time.time()
    for step in range(STEPS):
        ids = data.train_ids[rng.integers(0, len(data.train_ids), BS)]
        task = TASKS[step % 2]                    # alternate, so both stay fresh
        qs, ans = zip(*[data.qa(i, task) for i in ids])
        batch = V.make_batch(tok, list(qs), list(ans), n_img=n_tok)
        loss = vlm(batch, feats_for(arm, data, ids))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(float(loss.detach()))
        if step % 25 == 0:
            print(f"  step {step:3d}  loss {np.mean(curve[-25:]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    took = time.time() - t0
    CKPT.mkdir(exist_ok=True)
    torch.save(proj.state_dict(), CKPT / f"proj_{arm}.pt")
    _save(f"train_{arm}.json",
          {"arm": arm, "layer": layer, "steps": STEPS, "bs": BS, "lr": LR,
           "trainable_params": sum(p.numel() for p in proj.parameters()),
           "seconds": took, "s_per_step": took / STEPS, "curve": curve})
    print(f"  {arm}: {took:.0f}s ({took / STEPS:.2f} s/step)")


@torch.no_grad()
def _decode(vlm, tok, data, arm, ids, task, bs=16, max_new=6):
    qs = [data.qa(i, task)[0] for i in ids]
    outs = [None] * len(ids)
    for batch, part in V.prompt_batches(tok, qs, n_img=data.n_tokens(), bs=bs):
        got = vlm.greedy_batch(tok, batch, feats_for(arm, data, ids[part]),
                               max_new=max_new)
        for k, g in zip(part, got):
            outs[k] = g
    return outs


def stage_eval(arm, layer=None):
    layer = layer or ("early" if arm == "early" else "last")
    data = SL.AudioLLMData(layer=layer)
    tok, llm = V.load_llm()
    proj = build(arm, data, llm)
    proj.load_state_dict(torch.load(CKPT / f"proj_{arm}.pt"))
    vlm = V.TinyVLM(llm, proj).eval()
    res = {"arm": arm, "layer": layer}
    for split in ("new_voice", "seen_voice"):
        ids = getattr(data, split)[:250]
        got = _decode(vlm, tok, data, arm, ids, "digits")
        truth = [[SL.WORDS[d] for d in data.digits[i]] for i in ids]
        pred = [data.answer_digits(g) for g in got]
        exact = np.mean([p == t for p, t in zip(pred, truth)])
        per_digit = np.mean([[(p[k] if k < len(p) else "") == t[k]
                              for k in range(SL.N_DIGITS)]
                             for p, t in zip(pred, truth)])
        first = np.mean([(p[0] if p else "") == t[0] for p, t in zip(pred, truth)])
        second = np.mean([(p[1] if len(p) > 1 else "") == t[1]
                          for p, t in zip(pred, truth)])
        res[split] = {"digits_exact": float(exact),
                      "digits_per_slot": float(per_digit),
                      "first_digit": float(first), "second_digit": float(second),
                      "n": len(ids),
                      "examples": [{"truth": " ".join(t), "said": g}
                                   for t, g in zip(truth[:8], got[:8])]}
        print(f"  {arm} {split} digits: exact {exact:.3f} per-slot {per_digit:.3f}")
    ids = data.seen_voice[:250]
    got = _decode(vlm, tok, data, arm, ids, "speaker")
    truth = [SL.A.SPEAKERS[data.spk[i]] for i in ids]
    acc = np.mean([g.strip().lower().startswith(t) for g, t in zip(got, truth)])
    res["speaker"] = {"accuracy": float(acc), "n": len(ids), "classes": 5,
                      "examples": [{"truth": t, "said": g}
                                   for t, g in zip(truth[:8], got[:8])]}
    print(f"  {arm} speaker: {acc:.3f}")
    old = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else []
    _save("eval.json", [r for r in old if r["arm"] != arm] + [res])


# ---------------------------------------------------------------------------
def stage_plot():
    rows = {r["arm"]: r for r in json.loads((OUT / "eval.json").read_text())}
    order = [a for a in ("prefix", "early", "mlp") if a in rows]
    label = {"prefix": "blind control\n(no audio)", "early": "audio, Whisper\nlayer 1",
             "mlp": "audio, Whisper\nlast layer"}
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    x = np.arange(len(order))
    ps.style_axes(axes[0])
    for k, (split, colour, name) in enumerate([
            ("seen_voice", ps.SERIES[0], "held-out takes"),
            ("new_voice", ps.SERIES[1], "unheard voice")]):
        vals = [rows[a][split]["digits_exact"] for a in order]
        axes[0].bar(x + (k - 0.5) * 0.36, vals, 0.34, color=colour, label=name)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            axes[0].text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                         color=ps.INK_SECONDARY)
    axes[0].axhline(0.01, color=ps.BASELINE, ls="--", lw=1.0)
    axes[0].text(len(order) - 0.7, 0.03, "chance = 0.01", fontsize=8,
                 color=ps.INK_MUTED)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label[a] for a in order], fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_title('"What digits are spoken?" (both, in order)', color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[0].set_ylabel("exact match", color=ps.INK_SECONDARY, fontsize=10)

    ps.style_axes(axes[1])
    vals = [rows[a]["speaker"]["accuracy"] for a in order]
    axes[1].bar(x, vals, 0.5, color=ps.SERIES[3])
    for xi, v in zip(x, vals):
        axes[1].text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                     color=ps.INK_SECONDARY)
    axes[1].axhline(0.2, color=ps.BASELINE, ls="--", lw=1.0)
    axes[1].text(len(order) - 0.7, 0.22, "chance = 0.20", fontsize=8,
                 color=ps.INK_MUTED)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label[a] for a in order], fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('"Who is speaking?" (5 known voices)', color=ps.INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].set_ylabel("accuracy", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/results.png")

    fig, ax = ps.new_axes(7.0, 3.6)
    for i, arm in enumerate(order):
        c = json.loads((OUT / f"train_{arm}.json").read_text())["curve"]
        k = 20
        sm = np.convolve(c, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k, sm, color=ps.SERIES[i], lw=1.6,
                label=label[arm].replace("\n", " "))
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "training loss (20-step moving average)", "step",
              "nats per answer token", OUT / "curves.png")


STAGES = {"data": stage_data, "plot": stage_plot}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "train", "eval", "plot"])
    p.add_argument("--arm", default="mlp", choices=["mlp", "prefix", "early"])
    a = p.parse_args()
    if a.stage == "train":
        stage_train(a.arm)
    elif a.stage == "eval":
        stage_eval(a.arm)
    else:
        STAGES[a.stage]()
