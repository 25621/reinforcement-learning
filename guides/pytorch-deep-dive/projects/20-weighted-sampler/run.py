"""Project 20 — Weighted sampler.

Class-balanced sampling on an imbalanced dataset, measured against the two
alternatives people actually reach for (do nothing / weight the loss), plus the
sampler bugs that do not raise:

  1. the imbalance, and what "97% accuracy" hides
  2. WeightedRandomSampler vs loss weighting vs nothing — 3 seeds each
  3. what balancing costs: precision, and the model's implied prior
  4. the silent bug: per-CLASS weights instead of per-SAMPLE weights
  5. replacement=True vs False, and what num_samples really means
  6. sampler + shuffle=True (raises), and DistributedSampler's set_epoch trap

Runtime ~2 min. Needs torch, numpy, matplotlib.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 0
N_CLASSES = 4
DIM = 20
SHARE = np.array([0.88, 0.08, 0.03, 0.01])   # the imbalance
N_TRAIN = 8000
N_TEST_PER_CLASS = 500
BATCH = 64
EPOCHS = 8
SEEDS = [0, 1, 2]

torch.set_num_threads(4)


# ----------------------------------------------------------------------------
# 0. an imbalanced dataset
# ----------------------------------------------------------------------------
def make_data(n_per_class, seed):
    rng = np.random.default_rng(seed)
    centres = np.random.default_rng(123).normal(0, 1.0, size=(N_CLASSES, DIM))
    xs, ys = [], []
    for c, n in enumerate(n_per_class):
        xs.append(centres[c] + rng.normal(0, 1.35, size=(n, DIM)))
        ys.append(np.full(n, c))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(len(y))
    return TensorDataset(torch.from_numpy(x[perm]), torch.from_numpy(y[perm]))


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, N_CLASSES))


def evaluate(model, test):
    x, y = test.tensors
    with torch.no_grad():
        pred = model(x).argmax(1)
    acc = (pred == y).float().mean().item()
    recall, precision = [], []
    for c in range(N_CLASSES):
        istrue, ispred = y == c, pred == c
        recall.append(((pred == c) & istrue).sum().item() / max(1, int(istrue.sum())))
        precision.append(((y == c) & ispred).sum().item() / max(1, int(ispred.sum())))
    f1 = [0.0 if r + p == 0 else 2 * r * p / (r + p) for r, p in zip(recall, precision)]
    return dict(acc=acc, recall=recall, precision=precision,
                balanced=float(np.mean(recall)), macro_f1=float(np.mean(f1)),
                pred_share=[float((pred == c).float().mean()) for c in range(N_CLASSES)])


def train(loader, tests, seed, class_weight=None, epochs=EPOCHS):
    """`tests` is a dict of name -> dataset; returns one metric dict per test."""
    model = make_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(weight=class_weight)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            lossf(model(x), y).backward()
            opt.step()
    model.eval()
    out = {k: evaluate(model, d) for k, d in tests.items()}
    return out


def loaders_for(train_ds, seed, mode):
    """One DataLoader per strategy, all consuming the same dataset."""
    y = train_ds.tensors[1].numpy()
    counts = np.bincount(y, minlength=N_CLASSES)
    g = torch.Generator().manual_seed(seed)
    if mode == "shuffle":
        return DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=g,
                          drop_last=True), None
    if mode == "weighted":
        # ONE WEIGHT PER SAMPLE — 8000 of them, not 4
        w = torch.as_tensor((1.0 / counts)[y], dtype=torch.double)
        s = WeightedRandomSampler(w, num_samples=len(y), replacement=True, generator=g)
        return DataLoader(train_ds, batch_size=BATCH, sampler=s, drop_last=True), None
    if mode == "loss_weight":
        cw = torch.tensor(len(y) / (N_CLASSES * counts), dtype=torch.float32)
        return DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=g,
                          drop_last=True), cw
    if mode == "per_class_weights_bug":
        # THE BUG: 4 weights instead of 8000. No error, no warning.
        w = torch.as_tensor(1.0 / counts, dtype=torch.double)
        s = WeightedRandomSampler(w, num_samples=len(y), replacement=True, generator=g)
        return DataLoader(train_ds, batch_size=BATCH, sampler=s, drop_last=True), None
    raise ValueError(mode)


def main():
    n_per_class = (SHARE * N_TRAIN).astype(int)
    train_ds = make_data(n_per_class, SEED)
    test_ds = make_data([N_TEST_PER_CLASS] * N_CLASSES, SEED + 99)
    # a second test set drawn with the DEPLOYMENT prior — the same imbalance as
    # training. Balanced sampling is supposed to help on one of these and hurt
    # on the other; without both you only see half the trade.
    skew_ds = make_data((SHARE * 2000).astype(int), SEED + 77)
    tests = {"balanced": test_ds, "skewed": skew_ds}
    y = train_ds.tensors[1].numpy()
    counts = np.bincount(y, minlength=N_CLASSES)
    rows = []

    print("== 1. the dataset ==")
    for c in range(N_CLASSES):
        print(f"  class {c}: {counts[c]:5d} samples  ({100*counts[c]/len(y):5.2f}%)")
    print(f"  a model that always predicts class 0 scores "
          f"{counts[0]/len(y):.3f} on this training distribution")
    print(f"  the test set is BALANCED ({N_TEST_PER_CLASS} per class), so the same "
          f"model would score {1/N_CLASSES:.3f} there")
    rows.append(dict(section="dataset", config="majority_share",
                     value=round(float(counts[0] / len(y)), 4), note=str([int(v) for v in counts])))

    # --- 2. the three strategies -----------------------------------------
    print("\n== 2. three strategies, 3 seeds each (balanced test set) ==")
    print(f"  {'strategy':<24}{'accuracy':>10}{'balanced acc':>14}{'macro F1':>10}"
          f"   per-class recall")
    results, skewed = {}, {}
    for mode in ("shuffle", "weighted", "loss_weight"):
        per_seed = []
        for s in SEEDS:
            loader, cw = loaders_for(train_ds, s, mode)
            per_seed.append(train(loader, tests, s, class_weight=cw))
        results[mode] = [r["balanced"] for r in per_seed]
        skewed[mode] = [r["skewed"] for r in per_seed]
        per_seed = results[mode]
        acc = np.mean([r["acc"] for r in per_seed])
        bal = np.mean([r["balanced"] for r in per_seed])
        f1 = np.mean([r["macro_f1"] for r in per_seed])
        sd = np.std([r["balanced"] for r in per_seed])
        rec = np.mean([r["recall"] for r in per_seed], axis=0)
        print(f"  {mode:<24}{acc:10.3f}{bal:14.3f}{f1:10.3f}   "
              + " ".join(f"{v:.2f}" for v in rec))
        rows.append(dict(section="strategies", config=mode, value=round(float(bal), 4),
                         note=f"acc {acc:.4f}, macroF1 {f1:.4f}, sd {sd:.4f}, "
                              f"recall {[round(float(v),3) for v in rec]}"))
    print(f"  seed-to-seed sd of balanced accuracy: "
          + ", ".join(f"{m} {np.std([r['balanced'] for r in results[m]]):.4f}"
                      for m in results))

    # --- 3. what balancing costs -----------------------------------------
    print("\n== 3. the price of balancing: the same models on a SKEWED test set ==")
    print("     (skewed test set = the deployment prior, 88/8/3/1)")
    print(f"  {'strategy':<14}{'plain acc':>10}{'balanced acc':>14}"
          f"{'recall c3':>11}{'precision c3':>14}{'predicted share c3':>20}")
    for mode in ("shuffle", "weighted", "loss_weight"):
        a = np.mean([x["acc"] for x in skewed[mode]])
        b = np.mean([x["balanced"] for x in skewed[mode]])
        r = np.mean([x["recall"][3] for x in skewed[mode]])
        p = np.mean([x["precision"][3] for x in skewed[mode]])
        share = np.mean([x["pred_share"][3] for x in skewed[mode]])
        print(f"  {mode:<14}{a:10.3f}{b:14.3f}{r:11.3f}{p:14.3f}{share:20.3f}")
        rows.append(dict(section="skewed_test", config=mode, value=round(float(a), 4),
                         note=f"balanced {b:.4f}, recall c3 {r:.4f}, "
                              f"precision c3 {p:.4f}, predicted share c3 {share:.4f} "
                              f"(true share 0.010)"))

    # --- 4. the silent per-class-weights bug ------------------------------
    print("\n== 4. the bug: 4 weights instead of 8000 ==")
    loader, _ = loaders_for(train_ds, SEED, "per_class_weights_bug")
    seen = Counter()
    idx_seen = set()
    s = loader.sampler
    for i in s:
        idx_seen.add(int(i))
        seen[int(y[i])] += 1
    print(f"  weights tensor length : {len(s.weights)}   dataset length: {len(y)}")
    print(f"  distinct indices drawn: {len(idx_seen)}  -> {sorted(idx_seen)}")
    print(f"  classes actually seen : {dict(sorted(seen.items()))}")
    print("  no exception, no warning: the sampler samples from range(len(weights))")
    bug = train(loader, tests, SEED)["balanced"]
    print(f"  training on it: accuracy {bug['acc']:.3f}, balanced accuracy "
          f"{bug['balanced']:.3f}, recall {[round(v,2) for v in bug['recall']]}")
    rows.append(dict(section="weight_length_bug", config="4 weights",
                     value=round(float(bug["balanced"]), 4),
                     note=f"{len(idx_seen)} distinct indices of {len(y)}"))

    # --- 5. replacement and num_samples -----------------------------------
    print("\n== 5. replacement and num_samples ==")
    w = torch.as_tensor((1.0 / counts)[y], dtype=torch.double)
    for repl in (True, False):
        g = torch.Generator().manual_seed(SEED)
        s = WeightedRandomSampler(w, num_samples=len(y), replacement=repl, generator=g)
        drawn = list(s)
        cls = np.bincount(y[drawn], minlength=N_CLASSES)
        print(f"  replacement={str(repl):<5} distinct indices {len(set(drawn)):5d}/"
              f"{len(y)}   class counts {[int(v) for v in cls]}")
        rows.append(dict(section="replacement", config=f"replacement={repl}",
                         value=len(set(drawn)), note=f"class counts {[int(v) for v in cls]}"))
    g = torch.Generator().manual_seed(SEED)
    s = WeightedRandomSampler(w, num_samples=4 * int(counts.min()), replacement=True,
                              generator=g)
    drawn = list(s)
    print(f"  num_samples=4*min_class={4*counts.min()}: epoch is "
          f"{len(drawn)} samples, class counts {[int(v) for v in np.bincount(y[drawn], minlength=4)]}")
    rows.append(dict(section="replacement", config="num_samples=4*min_class",
                     value=len(drawn),
                     note=f"class counts {[int(v) for v in np.bincount(y[drawn], minlength=4)]}"))
    dup = Counter(list(WeightedRandomSampler(w, num_samples=len(y), replacement=True,
                                             generator=torch.Generator().manual_seed(SEED))))
    rare = [i for i in dup if y[i] == 3]
    print(f"  the rarest class has {counts[3]} distinct samples; a balanced epoch draws "
          f"{sum(dup[i] for i in rare)} of them, so each appears "
          f"{sum(dup[i] for i in rare)/max(1,len(rare)):.1f}x on average")
    rows.append(dict(section="replacement", config="rare_class_duplication",
                     value=round(sum(dup[i] for i in rare) / max(1, len(rare)), 2),
                     note=f"{counts[3]} distinct samples in class 3"))

    # --- 6. the two API traps ----------------------------------------------
    print("\n== 6. two API traps ==")
    try:
        DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                   sampler=WeightedRandomSampler(w, num_samples=10, replacement=True))
        print("  sampler + shuffle=True: no error (unexpected)")
    except Exception as e:
        print(f"  sampler + shuffle=True -> {type(e).__name__}: {str(e).splitlines()[0]}")
        rows.append(dict(section="api_traps", config="sampler+shuffle",
                         value=type(e).__name__, note=str(e).splitlines()[0]))

    ds4 = [DistributedSampler(train_ds, num_replicas=4, rank=r, shuffle=True)
           for r in range(4)]
    parts = [list(s) for s in ds4]
    union = set().union(*[set(p) for p in parts])
    overlap = sum(len(set(a) & set(b)) for i, a in enumerate(parts)
                  for b in parts[i + 1:])
    print(f"  DistributedSampler(4 ranks): each rank gets {len(parts[0])} indices, "
          f"union covers {len(union)}/{len(y)}, pairwise overlap {overlap}")
    for s in ds4:
        s.set_epoch(0)
    e0 = list(ds4[0])
    e1_no_set = list(ds4[0])                 # forgot set_epoch
    ds4[0].set_epoch(1)
    e1_set = list(ds4[0])
    print(f"  epoch 1 without set_epoch: identical order to epoch 0 -> "
          f"{e0 == e1_no_set}")
    print(f"  epoch 1 with    set_epoch: identical order to epoch 0 -> "
          f"{e0 == e1_set}")
    rows.append(dict(section="api_traps", config="distributed_no_set_epoch",
                     value=str(e0 == e1_no_set),
                     note=f"with set_epoch: {e0 == e1_set}; "
                          f"{len(parts[0])} per rank, overlap {overlap}"))

    with open(OUT / "findings.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["section", "config", "value", "note"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'}")

    # --- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    ax.bar(np.arange(N_CLASSES), counts, 0.6, color=ps.SERIES[0])
    for c in range(N_CLASSES):
        ax.text(c, counts[c] + 60, f"{counts[c]}", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_title("Training set: 88 / 8 / 3 / 1", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("class", color=ps.INK_SECONDARY)
    ax.set_ylabel("samples", color=ps.INK_SECONDARY)
    ax.set_xticks(np.arange(N_CLASSES))

    ax = axes[1]
    width = 0.26
    for i, mode in enumerate(("shuffle", "weighted", "loss_weight")):
        rec = np.mean([r["recall"] for r in results[mode]], axis=0)
        ax.bar(np.arange(N_CLASSES) + (i - 1) * width, rec, width,
               color=ps.SERIES[i], label=mode)
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class recall on a balanced test set", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("class", color=ps.INK_SECONDARY)
    ax.set_ylabel("recall", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    modes = ["shuffle", "weighted", "loss_weight"]
    bal = [np.mean([r["balanced"] for r in results[m]]) for m in modes]
    err = [np.std([r["balanced"] for r in results[m]]) for m in modes]
    acc = [np.mean([r["acc"] for r in results[m]]) for m in modes]
    xs = np.arange(len(modes))
    ax.bar(xs - 0.18, acc, 0.34, color=ps.SERIES[4], label="plain accuracy")
    ax.bar(xs + 0.18, bal, 0.34, yerr=err, capsize=4, color=ps.SERIES[1],
           label="balanced accuracy (± sd over 3 seeds)")
    ax.set_xticks(xs)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_title("Balancing moves the metric that matters", color=ps.INK,
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ps.save(fig, OUT / "weighted_sampler.png")


if __name__ == "__main__":
    main()
