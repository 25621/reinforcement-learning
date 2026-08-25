"""Fine-tune a pretrained VLA on a new task, and find out what carried over.

Six experiments:

1. the control -- does the instruction matter at all on this task?
2. pretraining -- one VLA on red/blue tasks, then a green task it has not seen
3. the sample-efficiency curve: fine-tuned vs from scratch, 5 to 150 demos
4. which half transferred -- vision, language, or the action head
5. language encoders: a learned lookup, a bag of words, a real frozen LLM
6. what fine-tuning costs you elsewhere (forgetting)

Run:  python3 run.py     (about 8 minutes; needs numpy, torch, transformers,
                          matplotlib, and SmolLM2-135M in the HF cache)
"""

import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vla as V            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
N_EVAL = 40
PRE_PALETTE = ("red", "blue")
NEW_PALETTE = ("green", "red")


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<50s} {value:>9}   {note}")


class LookupLang:
    """One trained vector per sentence.  The smallest thing that can work."""

    def __init__(self, sentences, dim=32, seed=0):
        self.idx = {s: i for i, s in enumerate(sentences)}
        torch.manual_seed(seed)
        self.table = nn.Embedding(len(sentences), dim)
        with torch.no_grad():
            self.table.weight.mul_(0.1)

    def __call__(self, sents):
        return self.table(torch.tensor([self.idx[s] for s in sents]))

    def parameters(self):
        return self.table.parameters()


class FrozenLang:
    """Cached sentence vectors from a real language model, never updated."""

    def __init__(self, table, dim=32, seed=0):
        self.table = table
        torch.manual_seed(seed)
        d = next(iter(table.values())).shape[0]
        self.proj = nn.Linear(d, dim)

    def __call__(self, sents):
        x = torch.stack([self.table[s] for s in sents])
        return self.proj(x)

    def parameters(self):
        return self.proj.parameters()


def main():
    t0 = time.time()
    torch.set_num_threads(6)

    print("\n[0] data")
    pre = V.collect(250, palette=PRE_PALETTE, seed=0)
    record("data", "pretraining demonstrations", len(pre[4]),
           f"{len(pre[0])} frames")
    new_full = V.collect(200, palette=NEW_PALETTE, seed=7)
    record("data", "new-task demonstrations available", len(new_full[4]),
           f"{len(new_full[0])} frames")
    subset = V.take

    lookup = LookupLang(V.ALL_SENTENCES)

    def lang(s):
        return lookup(s)

    # -- 1/2. pretrain ------------------------------------------------------
    print("\n[1] pretraining on red/blue, testing on green")
    base = V.VLA()
    V.train(base, subset(pre, 10 ** 9), lang, epochs=60,
            params=list(base.parameters()) + list(lookup.parameters()),
            log="pretrain")
    pre_home = V.evaluate(base, lang, PRE_PALETTE, n=N_EVAL)
    pre_zero = V.evaluate(base, lang, NEW_PALETTE, n=N_EVAL)
    record("pretrain", "pretrained VLA on its own tasks",
           round(pre_home["success"], 3))
    record("pretrain", "pretrained VLA zero-shot on the green task",
           round(pre_zero["success"], 3),
           f"went to the decoy {pre_zero['decoy']:.3f}")

    # -- 3. sample efficiency ------------------------------------------------
    print("\n[2] the sample-efficiency curve")
    curve = {}
    big = None
    for nd in (5, 15, 50, 150):
        d = subset(new_full, nd)
        scratch = V.VLA()
        lk_s = LookupLang(V.ALL_SENTENCES, seed=1)
        V.train(scratch, d, lambda s: lk_s(s), epochs=70,
                params=list(scratch.parameters()) + list(lk_s.parameters()))
        s_scratch = V.evaluate(scratch, lambda s: lk_s(s), NEW_PALETTE,
                               n=N_EVAL)
        ft = copy.deepcopy(base)
        lk_f = copy.deepcopy(lookup)
        V.train(ft, d, lambda s: lk_f(s), epochs=70, lr=3e-4,
                params=list(ft.parameters()) + list(lk_f.parameters()))
        s_ft = V.evaluate(ft, lambda s: lk_f(s), NEW_PALETTE, n=N_EVAL)
        curve[nd] = (s_scratch["success"], s_ft["success"])
        if nd == 150:
            big = (scratch, lambda s, e=lk_s: e(s), s_scratch)
        record("curve", f"{nd} demos: from scratch",
               round(s_scratch["success"], 3),
               f"fine-tuned {s_ft['success']:.3f}  "
               f"(advantage {s_ft['success'] - s_scratch['success']:+.3f})")

    # the control: the best policy we have, with the sentence blanked
    print("\n[2b] the control: the same policy, unable to read")
    seen = big[2]
    blind = V.evaluate(big[0], big[1], NEW_PALETTE, n=N_EVAL, blind=True)
    record("control", "150-demo policy, reads the instruction",
           round(seen["success"], 3), f"went to the decoy {seen['decoy']:.3f}")
    record("control", "same policy with the instruction blanked",
           round(blind["success"], 3), f"went to the decoy {blind['decoy']:.3f}")

    # -- 4. which half transferred ------------------------------------------
    print("\n[3] which half of the VLA carried the transfer")
    d15 = subset(new_full, 15)
    variants = {}
    for name in ("full fine-tune", "vision frozen", "vision reinitialised",
                 "head reinitialised", "language reinitialised"):
        m = copy.deepcopy(base)
        lk = copy.deepcopy(lookup)
        ps = list(m.parameters()) + list(lk.parameters())
        if name == "vision frozen":
            for p in m.vision.parameters():
                p.requires_grad_(False)
            ps = [p for p in ps if p.requires_grad]
        elif name == "vision reinitialised":
            fresh = V.VLA()
            m.vision.load_state_dict(fresh.vision.state_dict())
        elif name == "head reinitialised":
            fresh = V.VLA()
            m.head.load_state_dict(fresh.head.state_dict())
        elif name == "language reinitialised":
            lk = LookupLang(V.ALL_SENTENCES, seed=4)
            ps = list(m.parameters()) + list(lk.parameters())
        V.train(m, d15, lambda s, lk=lk: lk(s), epochs=70, lr=3e-4, params=ps)
        sc = V.evaluate(m, lambda s, lk=lk: lk(s), NEW_PALETTE,
                        n=N_EVAL)["success"]
        variants[name] = sc
        record("ablate", f"15 demos, {name}", round(sc, 3))

    # -- 5. language encoders ------------------------------------------------
    print("\n[4] three ways to turn a sentence into a vector")
    tab = V.frozen_llm_embeddings(V.ALL_SENTENCES)
    record("language", "sentences embedded by the frozen LLM", len(tab),
           f"{next(iter(tab.values())).shape[0]}-d, computed once")
    encs = {}
    for name, mk in [("learned lookup", lambda: LookupLang(V.ALL_SENTENCES, seed=2)),
                     ("bag of words", lambda: V.BagOfWords(V.ALL_SENTENCES, seed=2)),
                     ("frozen SmolLM2-135M", lambda: FrozenLang(tab, seed=2))]:
        enc = mk()
        m = V.VLA()
        ps = list(m.parameters()) + list(enc.parameters()
                                         if hasattr(enc, "parameters")
                                         else enc.table.parameters())
        V.train(m, subset(pre, 10 ** 9), lambda s, e=enc: e(s), epochs=60,
                params=ps)
        home = V.evaluate(m, lambda s, e=enc: e(s), PRE_PALETTE,
                          n=N_EVAL)["success"]
        # the point of a real encoder: an unseen word
        zero = V.evaluate(m, lambda s, e=enc: e(s), NEW_PALETTE,
                          n=N_EVAL)["success"]
        encs[name] = (home, zero)
        record("language", f"{name}: on red/blue", round(home, 3),
               f"zero-shot on the unseen word 'green' {zero:.3f}")

    # -- 6. forgetting -------------------------------------------------------
    print("\n[5] what fine-tuning costs on the old tasks")
    ft50 = copy.deepcopy(base)
    lk50 = copy.deepcopy(lookup)
    V.train(ft50, subset(new_full, 50), lambda s: lk50(s), epochs=70, lr=3e-4,
            params=list(ft50.parameters()) + list(lk50.parameters()))
    after = V.evaluate(ft50, lambda s: lk50(s), PRE_PALETTE, n=N_EVAL)["success"]
    record("forget", "old red/blue tasks before fine-tuning",
           round(pre_home["success"], 3))
    record("forget", "old red/blue tasks after 50 green demos", round(after, 3),
           f"change {after - pre_home['success']:+.3f}")

    record("cost", "VLA parameters", sum(p.numel() for p in base.parameters()))
    record("cost", "total runtime (s)", round(time.time() - t0, 1))

    # ---------------- figures ----------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    nds = sorted(curve)
    ax[0].plot(nds, [curve[n][0] for n in nds], "o-", c="#d1495b",
               label="from scratch")
    ax[0].plot(nds, [curve[n][1] for n in nds], "s-", c="#2a9d8f",
               label="fine-tuned from the VLA")
    ax[0].axhline(pre_zero["success"], ls="--", c="#adb5bd",
                  label="zero-shot (0 demos)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("demonstrations of the new task")
    ax[0].set_ylabel(f"success over {N_EVAL} episodes")
    ax[0].set_title("does pretraining pay for itself?")
    ax[0].legend(fontsize=8)
    names = list(variants)
    ax[1].barh(range(len(names)), [variants[n] for n in names], color="#457b9d")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].axvline(curve[15][0], ls="--", c="#d1495b", label="from scratch")
    ax[1].set_xlabel("success after 15 demos")
    ax[1].set_title("which part was worth keeping")
    ax[1].legend(fontsize=8)
    en = list(encs)
    x = np.arange(len(en))
    ax[2].bar(x - 0.2, [encs[n][0] for n in en], 0.38, label="seen words",
              color="#264653")
    ax[2].bar(x + 0.2, [encs[n][1] for n in en], 0.38,
              label="unseen word 'green'", color="#e9c46a")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([n.replace(" ", "\n") for n in en], fontsize=7)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("language encoders")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vla.png"), dpi=120)
    plt.close(fig)

    # a picture of what the model sees
    rng = np.random.default_rng(2)
    env = V.TwoTargetEnv(rng, palette=NEW_PALETTE)
    fig, ax = plt.subplots(3, 4, figsize=(11, 8))
    for k in range(4):
        env.reset()
        im = env.image()
        rgb = np.stack([im[0], im[1], np.zeros_like(im[0])], -1)
        ax[0, k].imshow(np.clip(rgb, 0, 1))
        ax[0, k].set_title(env.instr, fontsize=7)
        cr = env.crops()
        for j in range(2):
            ax[j + 1, k].imshow(cr[j], vmin=0, vmax=1, cmap="viridis")
            ax[j + 1, k].set_title(f"detection {j}", fontsize=7)
        for j in range(3):
            ax[j, k].axis("off")
    fig.suptitle("top: the scene (red = arm, green = the two discs).  below: "
                 "the two crops the network scores, in random order", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scenes.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
