"""Project 36 — Two-process DDP: does the batch really split, and do you really go faster?

Run:  python3 run.py           (~4 minutes)

There is no usable GPU on this machine (sm_61), so "one GPU" is played by "one OS
process with its own slice of the CPU threads", and the backend is gloo instead
of NCCL. Every line of DDP code is unchanged.

Sections
  1. DDP == one big batch            (numerical proof)
  2. replicas stay bit-identical     (that is what the all-reduce buys)
  3. scaling under a FIXED cpu budget
  4. the oversubscription trap       (why torchrun sets OMP_NUM_THREADS=1)
  5. DistributedSampler: what breaks silently without it
  6. what the communication actually costs
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dist_lib as D  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<44} {value}")


# ---------------------------------------------------------------------------
# the task: 8 classes of 16x16 images, each class a coloured patch somewhere
# ---------------------------------------------------------------------------

N_CLASS, IMG = 8, 16


def make_images(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(N_CLASS, 3, 6, 6, generator=g) * 2.0
    y = torch.randint(0, N_CLASS, (n,), generator=g)
    x = torch.randn(n, 3, IMG, IMG, generator=g) * 0.6
    pos = torch.randint(0, IMG - 6, (n, 2), generator=g)
    for i in range(n):
        r, c = int(pos[i, 0]), int(pos[i, 1])
        x[i, :, r:r + 6, c:c + 6] += proto[y[i]]
    return x, y


class SmallCNN(nn.Module):
    """~0.4M parameters — small enough to run fast, big enough that the
    convolutions, not Python, dominate the step time."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, N_CLASS)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x))
        return self.fc(x.flatten(1))


def build(seed=1234):
    torch.manual_seed(seed)
    return SmallCNN()


def flat_weights(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


# ---------------------------------------------------------------------------
# 1 + 2. correctness: DDP with N ranks == one process with the N-times batch
# ---------------------------------------------------------------------------

GLOBAL_BATCH, STEPS = 64, 20


def w_ddp_equivalence(rank, world):
    per_rank = GLOBAL_BATCH // world
    x, y = make_images(GLOBAL_BATCH * STEPS, seed=7)
    model = build()
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05, momentum=0.9)

    losses = []
    for s in range(STEPS):
        lo = s * GLOBAL_BATCH + rank * per_rank
        xb, yb = x[lo:lo + per_rank], y[lo:lo + per_rank]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    return {"w": flat_weights(model), "losses": losses}


def w_single(rank, world):
    x, y = make_images(GLOBAL_BATCH * STEPS, seed=7)
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for s in range(STEPS):
        xb = x[s * GLOBAL_BATCH:(s + 1) * GLOBAL_BATCH]
        yb = y[s * GLOBAL_BATCH:(s + 1) * GLOBAL_BATCH]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    return {"w": flat_weights(model), "losses": losses}


def section_1_2():
    print("\n[1] DDP with 2 ranks vs one process with the whole batch")
    ref = D.launch(w_single, 1, threads=6)[0]
    two = D.launch(w_ddp_equivalence, 2, threads=3)
    four = D.launch(w_ddp_equivalence, 4, threads=2)

    d2 = float((two[0]["w"] - ref["w"]).abs().max())
    d4 = float((four[0]["w"] - ref["w"]).abs().max())
    record("equivalence", "max |w(ddp,2 ranks) - w(single, batch 64)|", f"{d2:.3e}")
    record("equivalence", "max |w(ddp,4 ranks) - w(single, batch 64)|", f"{d4:.3e}")
    record("equivalence", "final loss single / ddp2 / ddp4",
           f"{ref['losses'][-1]:.4f} / {two[0]['losses'][-1]:.4f} / {four[0]['losses'][-1]:.4f}")

    print("\n[2] do the replicas stay identical?")
    spread2 = float((two[0]["w"] - two[1]["w"]).abs().max())
    spread4 = max(float((four[0]["w"] - four[r]["w"]).abs().max()) for r in range(1, 4))
    record("sync", "max |w(rank 0) - w(rank 1)| after 20 steps", f"{spread2:.3e}")
    record("sync", "max spread across 4 ranks after 20 steps", f"{spread4:.3e}")
    # per-rank losses differ because each rank sees different data
    lossdiff = abs(two[0]["losses"][0] - two[1]["losses"][0])
    record("sync", "|loss(rank 0) - loss(rank 1)| at step 0", f"{lossdiff:.4f}",
           "different data -> different loss, same gradient after all-reduce")
    return ref["losses"], two[0]["losses"]


# ---------------------------------------------------------------------------
# 3 + 4. throughput
# ---------------------------------------------------------------------------

BENCH_STEPS = 30
PER_RANK_BATCH = 32


def w_throughput(rank, world, use_ddp=True):
    x, y = make_images(PER_RANK_BATCH * (BENCH_STEPS + 3), seed=11 + rank)
    model = build()
    net = DDP(model) if use_ddp else model
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

    def one_epoch(nsteps):
        for s in range(nsteps):
            xb = x[s * PER_RANK_BATCH:(s + 1) * PER_RANK_BATCH]
            yb = y[s * PER_RANK_BATCH:(s + 1) * PER_RANK_BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(xb), yb).backward()
            opt.step()

    one_epoch(3)                       # warm up caches / allocator
    torch.distributed.barrier()        # start the clock together
    t0 = time.perf_counter()
    one_epoch(BENCH_STEPS)
    torch.distributed.barrier()        # stop when the slowest rank is done
    dt = time.perf_counter() - t0
    return {"secs": dt, "imgs": BENCH_STEPS * PER_RANK_BATCH * world,
            "threads": torch.get_num_threads()}


def bench(world, threads, use_ddp=True):
    """One timed launch. Interleaving happens in the caller."""
    res = D.launch(w_throughput, world, threads=threads, args=(use_ddp,))
    secs = max(r["secs"] for r in res)            # the job ends with the slowest rank
    return res[0]["imgs"] / secs


def paired(configs, rounds=4):
    """Measure every config once per round, then compare *within* a round.

    This machine is shared: something else was using 6 of the 12 cores while
    these numbers were taken, and its load changes minute to minute. Comparing
    the best of A against the best of B measures who got lucky. Comparing A and
    B inside the same round, and then taking the median of those per-round
    ratios, cancels most of the drift because both candidates saw the same
    machine seconds apart.
    """
    per_cfg = {c: [] for c in configs}
    for _ in range(rounds):
        for cfg in configs:
            per_cfg[cfg].append(bench(*cfg))
    return {c: {"median": statistics.median(v), "best": max(v), "all": v}
            for c, v in per_cfg.items()}


def section_3_4():
    print("\n[3] scaling with a FIXED budget of 4 threads in total")
    configs = [(1, 4), (2, 2), (4, 1)]
    res = paired(configs, rounds=6)
    base = res[(1, 4)]
    scaling = {}
    for (world, threads) in configs:
        r = res[(world, threads)]
        # speedup computed round by round, then median -> load-drift resistant
        ratios = sorted(a / b for a, b in zip(r["all"], base["all"]))
        sp = statistics.median(ratios)
        scaling[world] = {"ips": r["median"], "speedup": sp,
                          "lo": ratios[0], "hi": ratios[-1]}
        record("scaling", f"{world} rank(s) x {threads} thread(s)",
               f"{r['median']:8.0f} img/s   speedup {sp:.2f}x "
               f"[{ratios[0]:.2f} - {ratios[-1]:.2f} across 6 rounds]")

    print("\n[4] the oversubscription trap: every rank grabs every core")
    over = paired([(4, 1), (4, 6)], rounds=3)
    good_ips = over[(4, 1)]["median"]
    bad_ips = over[(4, 6)]["median"]
    ratio = statistics.median([a / b for a, b in
                               zip(over[(4, 1)]["all"], over[(4, 6)]["all"])])
    record("oversubscribe", "4 ranks x 1 thread (4 threads on 12 cores)",
           f"{good_ips:8.0f} img/s")
    record("oversubscribe", "4 ranks x 6 threads (24 threads on 12 cores)",
           f"{bad_ips:8.0f} img/s")
    record("oversubscribe", "cost of oversubscription", f"{ratio:.2f}x slower")
    # honesty check: the SAME config, measured in two different sections
    record("noise", "4 ranks x 1 thread, measured in [3] and again in [4]",
           f"{scaling[4]['ips']:.0f} vs {good_ips:.0f} img/s",
           "the size of the error bar on every absolute number here")
    return scaling, good_ips, bad_ips


# ---------------------------------------------------------------------------
# 5. DistributedSampler
# ---------------------------------------------------------------------------

def w_sampler(rank, world, use_sampler, epochs=3, set_epoch=True):
    x, y = make_images(512, seed=23)
    ds = TensorDataset(x, y)
    if use_sampler:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=True, seed=99)
        loader = DataLoader(ds, batch_size=32, sampler=sampler)
    else:
        sampler = None
        g = torch.Generator().manual_seed(99)      # same seed on every rank!
        loader = DataLoader(ds, batch_size=32, shuffle=True, generator=g)

    model = build()
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05, momentum=0.9)

    seen, order_per_epoch, losses = [], [], []
    for ep in range(epochs):
        if sampler is not None and set_epoch:
            sampler.set_epoch(ep)
        idx_this_epoch = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(ddp(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        # record which dataset indices this rank walked
        if sampler is not None:
            if set_epoch:
                sampler.set_epoch(ep)
            idx_this_epoch = list(iter(sampler))
        else:
            g2 = torch.Generator().manual_seed(99)
            idx_this_epoch = torch.randperm(len(ds), generator=g2).tolist()
        order_per_epoch.append(idx_this_epoch)
        seen += idx_this_epoch
    return {"seen": sorted(set(seen)), "n_batches": len(losses) // epochs,
            "orders": order_per_epoch, "final_loss": losses[-1]}


def section_5():
    print("\n[5] DistributedSampler: the silent duplicate-work bug")
    with_s = D.launch(w_sampler, 2, threads=3, args=(True,))
    without = D.launch(w_sampler, 2, threads=3, args=(False,))

    overlap_with = len(set(with_s[0]["seen"]) & set(with_s[1]["seen"]))
    overlap_without = len(set(without[0]["seen"]) & set(without[1]["seen"]))
    record("sampler", "samples seen by BOTH ranks, with sampler", overlap_with)
    record("sampler", "samples seen by BOTH ranks, without sampler", overlap_without,
           "every rank replays the identical epoch")
    record("sampler", "batches per epoch per rank, with / without",
           f"{with_s[0]['n_batches']} / {without[0]['n_batches']}")
    record("sampler", "final loss with / without sampler",
           f"{with_s[0]['final_loss']:.4f} / {without[0]['final_loss']:.4f}",
           "no crash, no warning - just half the data per unit of work")

    no_setepoch = D.launch(w_sampler, 2, threads=3, args=(True, 3, False))
    o = no_setepoch[0]["orders"]
    same = sum(1 for e in range(1, len(o)) if o[e] == o[0])
    o2 = with_s[0]["orders"]
    same2 = sum(1 for e in range(1, len(o2)) if o2[e] == o2[0])
    record("sampler", "epochs with an order identical to epoch 0 (no set_epoch)",
           f"{same} of {len(o) - 1}")
    record("sampler", "epochs with an order identical to epoch 0 (set_epoch)",
           f"{same2} of {len(o2) - 1}")
    return with_s, without


# ---------------------------------------------------------------------------
# 6. what does the communication cost?
# ---------------------------------------------------------------------------

def section_6(scaling):
    print("\n[6] the price of the all-reduce")
    res = paired([(2, 2, True), (2, 2, False)], rounds=6)
    ddp_ips = res[(2, 2, True)]["median"]
    solo_ips = res[(2, 2, False)]["median"]
    ratios = sorted(b / a for a, b in
                    zip(res[(2, 2, True)]["all"], res[(2, 2, False)]["all"]))
    ratio = statistics.median(ratios)
    n_params = sum(p.numel() for p in build().parameters())
    bytes_per_step = n_params * 4
    record("comm", "parameters in the model", f"{n_params:,}")
    record("comm", "gradient bytes all-reduced per step", D.fmt_bytes(bytes_per_step))
    record("comm", "2 ranks WITH all-reduce (real DDP)", f"{ddp_ips:8.0f} img/s")
    record("comm", "2 ranks WITHOUT all-reduce (wrong, but fast)", f"{solo_ips:8.0f} img/s")
    record("comm", "communication overhead (median of 6 paired rounds)",
           f"{(ratio - 1) * 100:.0f}%   "
           f"[{(ratios[0] - 1) * 100:.0f}% - {(ratios[-1] - 1) * 100:.0f}% across rounds]")
    return ddp_ips, solo_ips


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def figure(scaling, single_losses, ddp_losses, good, over, ddp_ips, solo_ips):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ranks = sorted(scaling)
    ax[0].plot(ranks, [r for r in ranks], "k--", label="ideal (linear)")
    ax[0].errorbar(ranks, [scaling[r]["speedup"] for r in ranks],
                   yerr=[[scaling[r]["speedup"] - scaling[r]["lo"] for r in ranks],
                         [scaling[r]["hi"] - scaling[r]["speedup"] for r in ranks]],
                   fmt="o-", color="#c0392b", capsize=4,
                   label="measured (4 threads shared out)")
    ax[0].set_xlabel("ranks (processes)")
    ax[0].set_ylabel("speedup vs 1 rank")
    ax[0].set_title("Fixed CPU budget: splitting, not adding")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(single_losses, label="1 process, batch 64", lw=3, alpha=0.6)
    ax[1].plot(ddp_losses, "--", label="DDP 2 ranks, batch 32 each")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("training loss")
    ax[1].set_title("Same optimisation, different hardware")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    names = ["4x1 thread", "4x6 threads\n(oversubscribed)", "2 ranks\n+all-reduce",
             "2 ranks\nno all-reduce"]
    vals = [good, over, ddp_ips, solo_ips]
    ax[2].bar(names, vals, color=["#27ae60", "#c0392b", "#2980b9", "#7f8c8d"])
    ax[2].set_ylabel("img/s")
    ax[2].set_title("Two ways to lose throughput")
    ax[2].tick_params(axis="x", labelsize=8)
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT, "ddp.png")
    fig.savefig(path, dpi=120)
    print(f"\n  figure -> {path}")


def main():
    t0 = time.time()
    single_losses, ddp_losses = section_1_2()
    scaling, good, over = section_3_4()
    section_5()
    ddp_ips, solo_ips = section_6(scaling)
    figure(scaling, single_losses, ddp_losses, good, over, ddp_ips, solo_ips)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(FINDINGS)
    with open(os.path.join(OUT, "scaling.json"), "w") as f:
        json.dump({"scaling_img_per_s": scaling}, f, indent=2)
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
