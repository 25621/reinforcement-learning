"""Project 19 — Custom collate.

A collate function turns a *list of samples* into *one batch*. The default one
only knows how to stack equal-shaped tensors, so ragged sequence data needs your
own. This script measures:

  1. what `default_collate` actually does on ragged input (it raises — verbatim)
  2. padding waste: pad-to-global-max vs pad-to-batch-max vs length bucketing
  3. the attention mask — train with and without one, on the same batches
  4. the right-padding + "last hidden state" bug, which is silent
  5. where collate runs (worker vs main process) and what that is worth
  6. pinning a `pad_id` that collides with a real token

Runtime ~3 min. Needs torch, numpy, matplotlib.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 0
VOCAB = 32          # 0 is reserved for padding
PAD = 0
N_TRAIN, N_TEST = 6000, 1500
BATCH = 32
N_CLASSES = 4
EPOCHS = 3

torch.set_num_threads(4)


# ----------------------------------------------------------------------------
# 0. a ragged dataset
# ----------------------------------------------------------------------------
class RaggedSequences(Dataset):
    """Variable-length token sequences with two kinds of signal in them.

    * a *marginal* signal — group `y` owns eight token ids and they are
      over-drawn, so the sequence label can be read off the token proportions
    * a *local* signal — 65% of the time the next token is the previous one
      advanced by `y+1`, so next-token prediction is genuinely learnable

    The first makes the classification task in sections 4 work; the second makes
    the language-model task in section 3 work.
    """

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.seqs, self.labels = [], []
        ids = np.arange(1, VOCAB)
        for _ in range(n):
            # log-normal lengths: many short, a long tail — like real text
            L = int(np.clip(rng.lognormal(3.4, 0.75), 8, 256))
            y = int(rng.integers(0, N_CLASSES))
            # group y owns token ids 1+8y .. 8+8y; the true class is over-drawn
            p = np.full(VOCAB - 1, 1.0)
            p[y * 8: y * 8 + 8] *= 2.6
            p /= p.sum()
            draws = rng.choice(ids, size=L, p=p)
            follow = rng.random(L) < 0.65
            toks = np.empty(L, dtype=np.int64)
            toks[0] = draws[0]
            for t in range(1, L):
                toks[t] = (1 + (toks[t - 1] - 1 + y + 1) % (VOCAB - 1)) if follow[t] else draws[t]
            self.seqs.append(torch.from_numpy(toks))
            self.labels.append(y)
        self.labels = torch.tensor(self.labels)
        self.lengths = np.array([len(s) for s in self.seqs])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


# ----------------------------------------------------------------------------
# 1. collate functions
# ----------------------------------------------------------------------------
def collate_batch_max(samples):
    """Pad every sequence up to the longest one *in this batch*."""
    seqs, labels = zip(*samples)
    lengths = torch.tensor([len(s) for s in seqs])
    T = int(lengths.max())
    x = torch.full((len(seqs), T), PAD, dtype=torch.int64)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = s
    mask = (torch.arange(T)[None, :] < lengths[:, None])
    return x, mask, lengths, torch.stack(labels)


def make_collate_global_max(T):
    """Pad every sequence up to one fixed length, for every batch."""
    def collate(samples):
        seqs, labels = zip(*samples)
        lengths = torch.tensor([len(s) for s in seqs])
        x = torch.full((len(seqs), T), PAD, dtype=torch.int64)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = s[:T]
        mask = (torch.arange(T)[None, :] < lengths[:, None])
        return x, mask, lengths, torch.stack(labels)
    return collate


def collate_left_pad(samples):
    """Same, but the padding goes on the left."""
    seqs, labels = zip(*samples)
    lengths = torch.tensor([len(s) for s in seqs])
    T = int(lengths.max())
    x = torch.full((len(seqs), T), PAD, dtype=torch.int64)
    for i, s in enumerate(seqs):
        x[i, T - len(s):] = s
    mask = (torch.arange(T)[None, :] >= (T - lengths)[:, None])
    return x, mask, lengths, torch.stack(labels)


def collate_expensive(samples):
    """Batch-max padding plus per-batch work heavy enough to be worth moving
    off the main process — here, a small n-gram feature count per sample."""
    x, mask, lengths, y = collate_batch_max(samples)
    a = x.numpy()
    counts = np.zeros((a.shape[0], VOCAB * 4), dtype=np.float32)
    for _ in range(12):
        for i in range(a.shape[0]):
            for t in range(int(lengths[i]) - 1):
                counts[i, a[i, t] * 4 + (a[i, t + 1] % 4)] += 1.0
    return x, mask, lengths, y


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffle, then sort inside a window of `pool` batches, then shuffle the
    batch order. Similar lengths land together, so batches are nearly square."""

    def __init__(self, lengths, batch_size, pool=50, seed=SEED):
        self.lengths = lengths
        self.bs = batch_size
        self.pool = pool
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        chunk = self.bs * self.pool
        batches = []
        for s in range(0, len(idx), chunk):
            window = idx[s: s + chunk]
            window = window[np.argsort(self.lengths[window], kind="stable")]
            for b in range(0, len(window), self.bs):
                if len(window[b: b + self.bs]) == self.bs:
                    batches.append(window[b: b + self.bs].tolist())
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.lengths) // self.bs


# ----------------------------------------------------------------------------
# 2. the model — two pooling strategies, one of them buggy
# ----------------------------------------------------------------------------
class Classifier(nn.Module):
    """Two families of pooling, deliberately kept separate.

    The mean-pooling pair reads the *embeddings* directly: a padded position
    then contributes exactly zero (that is what `padding_idx` guarantees), so
    forgetting the mask corrupts only the divisor and nothing else. Running them
    through the GRU first would hide the bug, because a GRU keeps emitting its
    last real hidden state while it chews through the pads — the pads would
    quietly carry the answer. The last-timestep pair needs the GRU, because
    "the last state" only means something for a recurrent model.
    """

    def __init__(self, pooling="masked_mean", force_gru=False):
        super().__init__()
        torch.manual_seed(SEED)
        self.emb = nn.Embedding(VOCAB, 48, padding_idx=PAD)
        self.pooling = pooling
        self.recurrent = force_gru or pooling in ("last_timestep", "gathered_last")
        self.gru = nn.GRU(48, 48, batch_first=True) if self.recurrent else None
        self.head = nn.Sequential(nn.Linear(48, 48), nn.ReLU(), nn.Linear(48, N_CLASSES))

    def forward(self, x, mask, lengths):
        h = self.emb(x)
        if self.recurrent:
            h, _ = self.gru(h)
        if self.pooling == "masked_mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "unmasked_mean":
            pooled = h.mean(1)                       # divides by the PADDED length
        elif self.pooling == "last_timestep":
            pooled = h[:, -1]                        # the state *after* the pads
        elif self.pooling == "gathered_last":
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
            pooled = h.gather(1, idx).squeeze(1)
        return self.head(pooled)


class TinyLM(nn.Module):
    """Predict the next token. Every padded position is a target too — unless
    you tell the loss to skip it."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.emb = nn.Embedding(VOCAB, 48, padding_idx=PAD)
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.head = nn.Linear(64, VOCAB)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


def train_eval_lm(train_loader, test_loader, ignore_pad, epochs=2):
    """Returns (accuracy over ALL positions, accuracy over REAL tokens only)."""
    torch.manual_seed(SEED)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # ignore_index=PAD tells cross-entropy to drop those positions entirely;
    # -100 is torch's default sentinel, i.e. "ignore nothing here"
    lossf = nn.CrossEntropyLoss(ignore_index=PAD if ignore_pad else -100)
    for _ in range(epochs):
        model.train()
        for x, mask, lengths, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x[:, :-1])
            lossf(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1)).backward()
            opt.step()
    model.eval()
    hit_all = n_all = hit_real = n_real = 0
    with torch.no_grad():
        for x, mask, lengths, y in test_loader:
            pred = model(x[:, :-1]).argmax(-1)
            tgt = x[:, 1:]
            ok = pred == tgt
            real = tgt != PAD
            hit_all += int(ok.sum()); n_all += ok.numel()
            hit_real += int((ok & real).sum()); n_real += int(real.sum())
    return hit_all / n_all, hit_real / n_real


def train_eval(pooling, train_loader, test_loader, epochs=EPOCHS):
    torch.manual_seed(SEED)
    model = Classifier(pooling)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, mask, lengths, y in train_loader:
            opt.zero_grad(set_to_none=True)
            lossf(model(x, mask, lengths), y).backward()
            opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, mask, lengths, y in test_loader:
            correct += (model(x, mask, lengths).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


# ----------------------------------------------------------------------------
def main():
    train = RaggedSequences(N_TRAIN, SEED)
    test = RaggedSequences(N_TEST, SEED + 1)
    Tmax = int(max(train.lengths.max(), test.lengths.max()))
    print(f"train {len(train)} seqs, lengths {train.lengths.min()}-{train.lengths.max()}, "
          f"mean {train.lengths.mean():.1f}, global max {Tmax}")
    rows = []

    # --- 1. what default_collate does -------------------------------------
    print("\n== 1. the default collate on ragged samples ==")
    try:
        default_collate([train[0], train[1]])
        print("  no error (unexpected)")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        print(f"  {msg}")
        rows.append(dict(section="default_collate", config="ragged input",
                         value=msg, note="raises"))
    same = default_collate([(train.seqs[0][:16], train.labels[0]),
                            (train.seqs[1][:16], train.labels[1])])
    print(f"  equal-length samples collate fine -> {tuple(same[0].shape)}")

    # --- 2. padding waste --------------------------------------------------
    print("\n== 2. how much of each batch is padding ==")
    g = torch.Generator().manual_seed(SEED)
    strategies = {
        "global max": DataLoader(train, batch_size=BATCH, shuffle=True, generator=g,
                                 collate_fn=make_collate_global_max(Tmax), drop_last=True),
        "batch max": DataLoader(train, batch_size=BATCH, shuffle=True, generator=g,
                                collate_fn=collate_batch_max, drop_last=True),
        "bucketed": DataLoader(train, batch_sampler=BucketBatchSampler(train.lengths, BATCH),
                               collate_fn=collate_batch_max),
    }
    waste = {}
    for name, loader in strategies.items():
        real = pad = 0
        t0 = time.perf_counter()
        for x, mask, lengths, y in loader:
            real += int(mask.sum())
            pad += int((~mask).sum())
        dt = time.perf_counter() - t0
        waste[name] = 100 * pad / (real + pad)
        print(f"  {name:<11} {real+pad:9d} token slots, {pad:8d} padding "
              f"({waste[name]:5.1f}%)   epoch {dt:5.2f}s")
        rows.append(dict(section="padding_waste", config=name,
                         value=round(waste[name], 2), note=f"{real+pad} slots, {dt:.2f}s"))
    print(f"  -> batch-max moves {waste['global max']:.1f}% waste to "
          f"{waste['batch max']:.1f}%, bucketing to {waste['bucketed']:.1f}%")

    print("\n== 2b. what that waste costs: one real training epoch each ==")
    cost = {}
    for name, loader in strategies.items():
        torch.manual_seed(SEED)
        model = Classifier("masked_mean", force_gru=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        lossf = nn.CrossEntropyLoss()
        t0 = time.perf_counter()
        for x, mask, lengths, y in loader:
            opt.zero_grad(set_to_none=True)
            lossf(model(x, mask, lengths), y).backward()
            opt.step()
        dt = time.perf_counter() - t0
        cost[name] = dt
        print(f"  {name:<11} {dt:5.2f}s for one epoch of forward+backward")
        rows.append(dict(section="padding_cost", config=name, value=round(dt, 3),
                         note="seconds for one training epoch"))
    print(f"  -> bucketing is {cost['global max']/cost['bucketed']:.1f}x faster than "
          f"global-max padding, on identical data")

    # --- 3. the mask, where it really bites: the loss ----------------------
    print("\n== 3. next-token training with and without ignore_index ==")
    print("     'all positions' is the number a naive script prints;"
          " 'real tokens' is the truth")
    accs = {}
    lm_setups = [
        ("batch max", DataLoader(train, batch_size=BATCH, shuffle=True,
                                 generator=torch.Generator().manual_seed(SEED),
                                 collate_fn=collate_batch_max, drop_last=True),
         DataLoader(test, batch_size=128, collate_fn=collate_batch_max)),
        ("bucketed", DataLoader(train, batch_sampler=BucketBatchSampler(train.lengths, BATCH),
                                collate_fn=collate_batch_max),
         DataLoader(test, batch_size=128, collate_fn=collate_batch_max)),
    ]
    for pad_name, tr_lm, te_lm in lm_setups:
        for ignore in (True, False):
            all_acc, real_acc = train_eval_lm(tr_lm, te_lm, ignore_pad=ignore)
            key = f"{'ignore_index' if ignore else 'no ignore_index'} @ {pad_name}"
            accs[key] = real_acc
            print(f"  {key:<32} all positions {all_acc:.3f}   real tokens {real_acc:.3f}")
            rows.append(dict(section="mask", config=key, value=round(real_acc, 4),
                             note=f"all-position accuracy {all_acc:.4f}"))

    # --- 4. the silent right-padding bug ----------------------------------
    print("\n== 4. 'take the last hidden state' with right padding ==")
    tr = DataLoader(train, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    collate_fn=collate_batch_max, drop_last=True)
    te = DataLoader(test, batch_size=128, collate_fn=collate_batch_max)
    tr_glob = DataLoader(train, batch_size=BATCH, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED),
                         collate_fn=make_collate_global_max(Tmax), drop_last=True)
    te_glob = DataLoader(test, batch_size=128, collate_fn=make_collate_global_max(Tmax))
    last = {}
    for pad_name, (a, b) in (("batch max", (tr, te)), ("global max", (tr_glob, te_glob))):
        for pooling in ("gathered_last", "last_timestep"):
            key = f"{pooling} @ {pad_name}"
            last[key] = train_eval(pooling, a, b)
            print(f"  right-pad, {key:<28} {last[key]:.3f}")
            rows.append(dict(section="last_state", config=f"right_pad_{key}",
                             value=round(last[key], 4), note=""))
    tr_l = DataLoader(train, batch_size=BATCH, shuffle=True,
                      generator=torch.Generator().manual_seed(SEED),
                      collate_fn=collate_left_pad, drop_last=True)
    te_l = DataLoader(test, batch_size=128, collate_fn=collate_left_pad)
    last["left_pad last_timestep"] = train_eval("last_timestep", tr_l, te_l)
    print(f"  left-pad,  last_timestep @ batch max    "
          f"{last['left_pad last_timestep']:.3f}   <- same buggy line, correct answer")
    rows.append(dict(section="last_state", config="left_pad_last_timestep",
                     value=round(last["left_pad last_timestep"], 4), note=""))

    # --- 5. where collate runs --------------------------------------------
    print("\n== 5. collate runs in the worker, so its cost parallelizes ==")
    for label, fn in (("cheap collate", collate_batch_max),
                      ("expensive collate", collate_expensive)):
        for nw in (0, 4):
            loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=nw,
                                generator=torch.Generator().manual_seed(SEED),
                                collate_fn=fn, drop_last=True)
            t0 = time.perf_counter()
            for _ in loader:
                pass
            dt = time.perf_counter() - t0
            print(f"  {label:<18} num_workers={nw}  {dt:5.2f}s per epoch")
            rows.append(dict(section="collate_where", config=f"{label}, num_workers={nw}",
                             value=round(dt, 3), note="seconds per epoch"))

    # --- 6. a colliding pad id --------------------------------------------
    print("\n== 6. what if pad_id is a real token? ==")
    emb = nn.Embedding(VOCAB, 8, padding_idx=PAD)
    print(f"  padding_idx={PAD}: row {PAD} is all-zero -> "
          f"{bool((emb.weight[PAD] == 0).all())}, and its gradient stays zero")
    x = torch.tensor([[PAD, 3, 4]])
    emb(x).sum().backward()
    print(f"  after a backward pass, grad of row {PAD} is all zero -> "
          f"{bool((emb.weight.grad[PAD] == 0).all())}, "
          f"row 3 is not -> {bool((emb.weight.grad[3] != 0).any())}")
    rows.append(dict(section="pad_id", config="padding_idx grad",
                     value=float(emb.weight.grad[PAD].abs().sum()),
                     note="abs grad sum of the pad row"))

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "config", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'}")

    # --- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    ax.hist(train.lengths, bins=40, color=ps.SERIES[0])
    ax.axvline(train.lengths.mean(), color=ps.SERIES[2], ls="--", lw=1.5,
               label=f"mean {train.lengths.mean():.0f}")
    ax.axvline(Tmax, color=ps.SERIES[3], ls="--", lw=1.5, label=f"global max {Tmax}")
    ax.set_title("Sequence lengths", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("tokens", color=ps.INK_SECONDARY)
    ax.set_ylabel("sequences", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    names = list(waste)
    ax.bar(np.arange(len(names)), [waste[n] for n in names], 0.6,
           color=[ps.SERIES[i] for i in range(len(names))])
    for i, n in enumerate(names):
        ax.text(i, waste[n] + 1, f"{waste[n]:.1f}%", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(waste.values()) * 1.2)
    ax.set_title("Share of each batch that is padding", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("% padding", color=ps.INK_SECONDARY)

    ax = axes[2]
    keys = ["ignore_index @ batch max", "no ignore_index @ batch max",
            "ignore_index @ bucketed", "no ignore_index @ bucketed"]
    short = ["ignore_index\nbatch max", "no ignore\nbatch max",
             "ignore_index\nbucketed", "no ignore\nbucketed"]
    cols = [ps.SERIES[1], ps.SERIES[2], ps.SERIES[1], ps.SERIES[2]]
    ax.bar(np.arange(len(keys)), [accs[k] for k in keys], 0.6, color=cols)
    ax.axhline(1 / (VOCAB - 1), color=ps.INK_MUTED, ls="--", lw=1.2,
               label="chance on real tokens")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0, max(accs.values()) * 1.3)
    ax.set_title("Next-token accuracy on REAL tokens", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("accuracy", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ps.save(fig, OUT / "collate.png")


if __name__ == "__main__":
    main()
