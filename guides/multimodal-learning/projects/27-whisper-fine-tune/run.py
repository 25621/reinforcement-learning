"""Fine-tune Whisper on a domain it was not built for, and measure honestly.

The domain: the Free Spoken Digit Dataset -- 3,000 recordings of six people
saying a single digit, recorded at 8 kHz. Whisper has never been trained on
isolated words with no sentence around them, and 8 kHz throws away everything
above 4 kHz, so this is a small, real domain shift you can run on a CPU.

Stages
    probe    zero-shot Whisper on this data, scored three different ways
    train    fine-tune one arm (full / decoder / small)  -- ~5 min each
    eval     accuracy on a held-out *speaker* and on held-out *takes*
    plot     figures from the saved JSON

Arms
    zeroshot  the pretrained checkpoint, untouched
    full      every weight trained (37.8M)
    decoder   the encoder is frozen; only the decoder learns (29.5M)
    small     same as `full` but only 100 training clips (~1 minute of audio)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "06-mel-spectrogram-pipeline"))
import audio_lib as A  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
MODEL = "openai/whisper-tiny"
WORDS = ["zero", "one", "two", "three", "four", "five",
         "six", "seven", "eight", "nine"]
HELD_OUT_SPEAKER = "yweweler"        # the voice no arm ever trains on
N_EVAL = 250
STEPS, BS, LR = 150, 4, 3e-5


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_split(seed=0):
    """FSDD at 16 kHz (Whisper's rate), split three ways.

    * train        five speakers, 45 of their 50 takes per digit
    * seen_voice   the same five speakers, their 5 held-out takes
    * new_voice    every recording of a sixth speaker

    Two test sets, because they answer different questions. `seen_voice` asks
    "did the model learn this task?"; `new_voice` asks "did it learn the task, or
    did it learn these five voices?" -- a distinction a random split hides,
    since a random split puts other recordings of the same person saying the
    same digit on both sides.
    """
    xs, digits, spk = A.load_fsdd(PROJECTS / "06-mel-spectrogram-pipeline" / "data",
                                  sr_out=16000)
    held = A.SPEAKERS.index(HELD_OUT_SPEAKER)
    rng = np.random.default_rng(seed)
    new_voice = np.where(spk == held)[0]
    rest = np.where(spk != held)[0]
    rng.shuffle(rest)
    n_seen = 250
    return {"train": rest[n_seen:], "seen_voice": rest[:n_seen],
            "new_voice": rng.permutation(new_voice)}, xs, digits, spk


def load_model(name=MODEL):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(12)
    pr = WhisperProcessor.from_pretrained(name)
    pr.tokenizer.set_prefix_tokens(language="english", task="transcribe",
                                   predict_timestamps=False)
    m = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float32)
    return pr, m


def features(pr, xs, idx):
    """Waveforms -> the 80 x 3000 log-mel Whisper expects.

    Whisper always pads to 30 seconds, so a 0.5-second digit fills 25 of 1,500
    encoder positions and the other 1,475 are padded silence. That is wasteful
    and it is also the model's fixed input contract -- changing it would mean
    changing the pretrained positional embeddings, which is a different
    experiment. It is the single biggest reason this fine-tune is slow.
    """
    return pr.feature_extractor([xs[i] for i in idx], sampling_rate=16000,
                                return_tensors="pt").input_features


def labels_for(pr, digits, idx, pad=-100):
    seqs = [pr.tokenizer(" " + WORDS[digits[i]]).input_ids for i in idx]
    T = max(len(s) for s in seqs)
    out = np.full((len(seqs), T), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
_DIGIT_RE = re.compile(r"[0-9]")


def normalise(text):
    """Whisper's own recipe, in miniature: lower-case, drop punctuation, and
    write numbers as words.

    Why this matters more than it looks: the pretrained model answers "Seven."
    and the reference says "seven". Character for character that is wrong, so a
    raw score reports a failure that no listener would call one. Whisper's paper
    normalises text before scoring for exactly this reason, and the gap between
    the raw and normalised numbers is the part of "fine-tuning helped" that is
    really "fine-tuning taught it our formatting".
    """
    t = text.strip().lower()
    t = _DIGIT_RE.sub(lambda m: WORDS[int(m.group())], t)
    t = re.sub(r"[^a-z ]", "", t).strip()
    return re.sub(r"\s+", " ", t)


@torch.no_grad()
def transcribe(pr, model, xs, idx, bs=16, max_new=6):
    model.eval()
    outs = []
    for j in range(0, len(idx), bs):
        f = features(pr, xs, idx[j:j + bs])
        g = model.generate(f, language="english", task="transcribe",
                           max_new_tokens=max_new)
        outs += pr.batch_decode(g, skip_special_tokens=True)
    return outs


def score(texts, digits, idx):
    truth = [WORDS[digits[i]] for i in idx]
    raw = np.mean([t.strip() == r for t, r in zip(texts, truth)])
    norm = np.mean([normalise(t) == r for t, r in zip(texts, truth)])
    # "did the right word appear anywhere in the answer" -- the most forgiving
    # reading, which separates "misheard" from "said more than we asked"
    loose = np.mean([r in normalise(t).split() for t, r in zip(texts, truth)])
    return {"exact_raw": float(raw), "exact_normalised": float(norm),
            "word_present": float(loose), "n": len(idx)}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_probe():
    splits, xs, digits, _ = load_split()
    pr, m = load_model()
    rows = {}
    for name in ("new_voice", "seen_voice"):
        idx = splits[name][:N_EVAL]
        t0 = time.time()
        texts = transcribe(pr, m, xs, idx)
        rows[name] = score(texts, digits, idx)
        rows[name]["seconds"] = time.time() - t0
        print(f"  {name}: {rows[name]}")
    ex = splits["new_voice"][:12]
    rows["examples"] = [{"truth": WORDS[digits[i]], "said": t,
                         "normalised": normalise(t)}
                        for i, t in zip(ex, transcribe(pr, m, xs, ex))]
    _save("zeroshot.json", rows)


def stage_train(arm):
    splits, xs, digits, _ = load_split()
    pr, m = load_model()
    pool = splits["train"]
    if arm == "small":
        pool = pool[:100]
    if arm == "decoder":
        for p in m.model.encoder.parameters():
            p.requires_grad_(False)
    trainable = [p for p in m.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(0)
    m.train()
    curve, t0 = [], time.time()
    for step in range(STEPS):
        idx = pool[rng.integers(0, len(pool), BS)]
        f = features(pr, xs, idx)
        lab = labels_for(pr, digits, idx)
        loss = m(input_features=f, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(float(loss))
        if step % 10 == 0:
            print(f"  step {step:3d}  loss {np.mean(curve[-10:]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    took = time.time() - t0
    CKPT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), CKPT / f"whisper_{arm}.pt")
    _save(f"train_{arm}.json",
          {"arm": arm, "steps": STEPS, "bs": BS, "lr": LR,
           "pool": int(len(pool)), "clips_seen": STEPS * BS,
           "trainable_params": int(n_train), "seconds": took,
           "s_per_step": took / STEPS, "curve": curve})
    print(f"  {arm}: {took:.0f}s, {n_train/1e6:.1f}M trainable")


def stage_eval(arm):
    splits, xs, digits, _ = load_split()
    pr, m = load_model()
    if arm != "zeroshot":
        m.load_state_dict(torch.load(CKPT / f"whisper_{arm}.pt"))
    res = {"arm": arm}
    for name in ("new_voice", "seen_voice"):
        idx = splits[name][:N_EVAL]
        texts = transcribe(pr, m, xs, idx)
        res[name] = score(texts, digits, idx)
        print(f"  {arm} {name}: {res[name]}")
    ex = splits["new_voice"][:12]
    res["examples"] = [{"truth": WORDS[digits[i]], "said": t}
                       for i, t in zip(ex, transcribe(pr, m, xs, ex))]
    old = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else []
    _save("eval.json", [r for r in old if r["arm"] != arm] + [res])


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def stage_plot():
    rows = {r["arm"]: r for r in json.loads((OUT / "eval.json").read_text())}
    order = [a for a in ("zeroshot", "small", "decoder", "full") if a in rows]
    labels = {"zeroshot": "zero-shot", "small": "fine-tune\n100 clips",
              "decoder": "fine-tune\ndecoder only", "full": "fine-tune\nall weights"}
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    x = np.arange(len(order))
    for ax, split, ttl in zip(axes, ("new_voice", "seen_voice"),
                              ("held-out speaker (new voice)",
                               "held-out takes (voices seen in training)")):
        ps.style_axes(ax)
        for k, (key, colour) in enumerate([("exact_raw", ps.SERIES[2]),
                                           ("exact_normalised", ps.SERIES[0])]):
            vals = [rows[a][split][key] for a in order]
            ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=colour,
                   label={"exact_raw": "raw string match",
                          "exact_normalised": "after text normalisation"}[key])
            for xi, v in zip(x + (k - 0.5) * 0.36, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                        color=ps.INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[a] for a in order], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(ttl, color=ps.INK, fontsize=11, loc="left", pad=10)
        ax.axhline(0.1, color=ps.BASELINE, ls="--", lw=1.0)
        ax.text(len(order) - 0.6, 0.12, "chance", fontsize=8, color=ps.INK_MUTED)
    axes[0].set_ylabel("digit correct", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote outputs/accuracy.png")

    fig, ax = ps.new_axes(7.0, 3.6)
    for i, arm in enumerate([a for a in order if a != "zeroshot"]):
        c = json.loads((OUT / f"train_{arm}.json").read_text())["curve"]
        k = 10
        sm = np.convolve(c, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k, sm, color=ps.SERIES[i], lw=1.6,
                label=labels[arm].replace("\n", " "))
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "training loss (10-step moving average)", "step",
              "nats per token", OUT / "curves.png")


STAGES = {"probe": stage_probe, "plot": stage_plot}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["probe", "train", "eval", "plot"])
    p.add_argument("--arm", default="full",
                   choices=["zeroshot", "full", "decoder", "small"])
    a = p.parse_args()
    if a.stage == "train":
        stage_train(a.arm)
    elif a.stage == "eval":
        stage_eval(a.arm)
    else:
        STAGES[a.stage]()
