"""Project 23 — Profile and fix.

Takes a deliberately slow training script, profiles it with `torch.profiler`,
and fixes it in the order the profile dictates — measuring what each fix was
worth instead of guessing.

  1. the slow script, and the profiler's verdict on it
  2. five fixes applied one at a time, each measured
  3. what the profiler can and cannot see once workers are involved
  4. a control: an "optimization" the profile says is worthless

Runtime ~7 min. Needs torch, numpy, matplotlib, Pillow.
"""

import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

SEED = 0
N_IMAGES = 1600
IMG = 64
BATCH = 32
N_CLASSES = 4

torch.set_num_threads(4)


def build():
    if DATA.exists() and len(list(DATA.glob("*.jpg"))) == N_IMAGES:
        return
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    for i in range(N_IMAGES):
        cls = i % N_CLASSES
        a = rng.integers(60, 200, size=(IMG, IMG, 3), dtype=np.uint8)
        cy, cx = rng.integers(12, IMG - 12, size=2)
        yy, xx = np.ogrid[:IMG, :IMG]
        a[(yy - cy) ** 2 + (xx - cx) ** 2 < 9**2] = np.array(
            [[35, 230][cls & 1], [35, 230][(cls >> 1) & 1], 120], dtype=np.uint8)
        Image.fromarray(a).save(DATA / f"{i:05d}_{cls}.jpg", quality=88)


# ----------------------------------------------------------------------------
# the dataset, with four planted problems that can be switched off one by one
# ----------------------------------------------------------------------------
class ImageSet(Dataset):
    def __init__(self, *, scan_per_item=True, python_tensor=True, python_norm=True):
        self.scan_per_item = scan_per_item
        self.python_tensor = python_tensor
        self.python_norm = python_norm
        self.files = sorted(DATA.glob("*.jpg"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # PROBLEM 1: re-scan the directory on every single sample
        files = sorted(DATA.glob("*.jpg")) if self.scan_per_item else self.files
        path = files[i]
        img = Image.open(path).convert("RGB")
        a = np.asarray(img)

        if self.python_tensor:
            # PROBLEM 2: build the tensor from a nested Python list
            x = torch.tensor(a.tolist(), dtype=torch.float32).permute(2, 0, 1)
        else:
            x = torch.from_numpy(np.ascontiguousarray(
                a.transpose(2, 0, 1).astype(np.float32)))

        if self.python_norm:
            # PROBLEM 3: normalize with a Python loop over channels and rows
            for c in range(3):
                for r in range(x.shape[1]):
                    x[c, r] = (x[c, r] / 255.0 - 0.45) / 0.25
        else:
            x = (x / 255.0 - 0.45) / 0.25

        return x, int(path.stem.split("_")[1])


def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, N_CLASSES))


def make_loader(cfg):
    ds = ImageSet(scan_per_item=cfg["scan_per_item"],
                  python_tensor=cfg["python_tensor"],
                  python_norm=cfg["python_norm"])
    kw = dict(batch_size=BATCH, shuffle=True, drop_last=True,
              num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
              generator=torch.Generator().manual_seed(SEED))
    if cfg["num_workers"] > 0:
        kw["persistent_workers"] = cfg["persistent_workers"]
    return DataLoader(ds, **kw)


def run(cfg, epochs=1, prof_steps=0):
    """One (or more) epochs. Returns wall seconds and a per-stage breakdown."""
    loader = make_loader(cfg)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    stage = dict(data=0.0, forward=0.0, backward=0.0, optimizer=0.0, logging=0.0)
    log = []
    t_start = time.perf_counter()
    for _ in range(epochs):
        mark = time.perf_counter()
        for x, y in loader:
            stage["data"] += time.perf_counter() - mark
            mark = time.perf_counter()
            out = model(x)
            loss = lossf(out, y)
            stage["forward"] += time.perf_counter() - mark
            mark = time.perf_counter()
            opt.zero_grad(set_to_none=cfg["set_to_none"])
            loss.backward()
            stage["backward"] += time.perf_counter() - mark
            mark = time.perf_counter()
            opt.step()
            stage["optimizer"] += time.perf_counter() - mark
            mark = time.perf_counter()
            if cfg["chatty_logging"]:
                # PROBLEM 5: a full copy of the batch and a print, every step
                log.append((loss.item(), float(out.detach().numpy().std()),
                            x.numpy().mean()))
            else:
                log.append(loss.detach())
            stage["logging"] += time.perf_counter() - mark
            mark = time.perf_counter()
    wall = time.perf_counter() - t_start
    del loader
    return wall, stage


def profile_run(cfg, steps=12, trace_name=None):
    """The same loop under torch.profiler, with named regions."""
    loader = make_loader(cfg)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(steps):
            with record_function("## data ##"):
                x, y = next(it)
            with record_function("## forward ##"):
                loss = lossf(model(x), y)
            with record_function("## backward ##"):
                opt.zero_grad(set_to_none=True)
                loss.backward()
            with record_function("## optimizer ##"):
                opt.step()
    if trace_name:
        prof.export_chrome_trace(str(OUT / trace_name))
    del loader
    return prof


SLOW = dict(scan_per_item=True, python_tensor=True, python_norm=True,
            num_workers=0, persistent_workers=False, pin_memory=False,
            set_to_none=False, chatty_logging=True)


def main():
    build()
    print(f"corpus: {N_IMAGES} jpegs, batch {BATCH}, "
          f"{N_IMAGES//BATCH} steps per epoch\n")
    rows = []

    # --- 1. profile the slow script ---------------------------------------
    print("== 1. the slow script, under torch.profiler ==")
    # NOTE: no chrome trace for this one. The slow loop executes ~200k tiny
    # tensor ops per step, and the exported trace comes out at 288 MB.
    prof = profile_run(SLOW, steps=8)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(table)
    (OUT / "profile_slow.txt").write_text(table)
    regions = {e.key: e.cpu_time_total / 1000 for e in prof.key_averages()
               if e.key.startswith("## ")}
    total = sum(regions.values())
    print("  named regions (8 steps):")
    for k, v in sorted(regions.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {v:9.1f} ms   {100*v/total:5.1f}%")
        rows.append(dict(section="profile_slow", config=k.strip('# '),
                         value=round(v, 1), note=f"{100*v/total:.1f}% of 8 steps"))

    # --- 2. fix in the order the profile dictates -------------------------
    print("\n== 2. one fix at a time, cumulative ==")
    stages = [
        ("0. as found", {}),
        ("1. hoist the directory scan", dict(scan_per_item=False)),
        ("2. from_numpy, not torch.tensor(list)", dict(python_tensor=False)),
        ("3. vectorize the normalization", dict(python_norm=False)),
        ("4. num_workers=4 + persistent", dict(num_workers=4, persistent_workers=True)),
        ("5. drop the chatty logging", dict(chatty_logging=False)),
    ]
    cfg = dict(SLOW)
    walls, breakdown = [], []
    prev = None
    for label, patch in stages:
        cfg.update(patch)
        results = [run(dict(cfg))]
        # cheap stages get repeated; the slow ones are unambiguous anyway
        while results[0][0] < 6.0 and len(results) < 3:
            results.append(run(dict(cfg)))
        wall, stage = min(results, key=lambda r: r[0])
        spread = max(r[0] for r in results) - wall
        walls.append(wall)
        breakdown.append(stage)
        speedup = "" if prev is None else f"  x{prev/wall:5.2f}"
        print(f"  {label:<38} {wall:7.2f}s ({N_IMAGES/wall:7.1f}/s)"
              f"{speedup}   spread {spread:5.2f}s")
        rows.append(dict(section="fixes", config=label, value=round(wall, 3),
                         note=f"{N_IMAGES/wall:.1f} samples/s, spread {spread:.2f}s, "
                              f"data {stage['data']:.2f}s model "
                              f"{stage['forward']+stage['backward']:.2f}s"))
        prev = wall
    print(f"  -> end to end: {walls[0]:.2f}s to {walls[-1]:.2f}s = "
          f"{walls[0]/walls[-1]:.1f}x faster")

    # --- 3. the control: an optimization the profile does not endorse ------
    print("\n== 3. the control: two 'obvious' knobs the profile never flagged ==")
    fast = dict(cfg)
    reps = [run(dict(fast))[0] for _ in range(5)]
    base, spread = min(reps), max(reps) - min(reps)
    print(f"  {'baseline, 5 repeats':<18} best {base:6.2f}s   spread {spread:5.2f}s"
          f"  <- anything smaller than this spread is not a result")
    rows.append(dict(section="control", config="baseline", value=round(base, 3),
                     note=f"spread over 5 repeats {spread:.3f}s"))
    for label, patch in (("set_to_none=True", dict(set_to_none=True)),
                         ("pin_memory=True", dict(pin_memory=True)),
                         ("num_workers=12", dict(num_workers=12))):
        t = min(run({**fast, **patch})[0] for _ in range(3))
        verdict = "inside the noise" if abs(t - base) < spread else "REAL"
        print(f"  {label:<18} best {t:6.2f}s   x{base/t:.3f}   {verdict}")
        rows.append(dict(section="control", config=label, value=round(t, 3),
                         note=f"baseline {base:.3f}s, speedup {base/t:.3f}, {verdict}"))

    # --- 4. what the profiler cannot see -----------------------------------
    print("\n== 4. the profiler and worker processes ==")
    for nw in (0, 4):
        p = profile_run({**cfg, "num_workers": nw, "python_norm": True,
                         "scan_per_item": False, "python_tensor": False},
                        steps=8,
                        trace_name=f"trace_workers{nw}.json.gz" if nw == 4 else None)
        reg = {e.key: e.cpu_time_total / 1000 for e in p.key_averages()
               if e.key.startswith("## ")}
        tot = sum(reg.values())
        print(f"  num_workers={nw}: data region {reg.get('## data ##', 0):8.1f} ms "
              f"({100*reg.get('## data ##', 0)/tot:5.1f}% of the loop), total "
              f"self CPU in profile {sum(e.self_cpu_time_total for e in p.key_averages())/1000:8.1f} ms")
        rows.append(dict(section="worker_visibility", config=f"num_workers={nw}",
                         value=round(reg.get("## data ##", 0), 1),
                         note=f"{100*reg.get('## data ##',0)/tot:.1f}% of the named regions"))
    print("  the wall-clock wait still shows up; the WORK does not — it happens")
    print("  in another process, which this profiler never entered.")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "config", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'}")

    # --- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    keys = sorted(regions, key=lambda k: -regions[k])
    ax.barh(np.arange(len(keys)), [regions[k] for k in keys], 0.6,
            color=[ps.SERIES[i % len(ps.SERIES)] for i in range(len(keys))])
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([k.strip("# ") for k in keys])
    ax.invert_yaxis()
    ax.set_title("Where 8 slow steps go", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("ms", color=ps.INK_SECONDARY)

    ax = axes[1]
    xs = np.arange(len(walls))
    ax.plot(xs, walls, "o-", lw=2, color=ps.SERIES[0])
    for i, w in enumerate(walls):
        ax.text(i, w * 1.06, f"{w:.1f}s", ha="center", color=ps.INK_SECONDARY,
                fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0][0] for s in stages])
    ax.set_title("Epoch time after each fix (log scale)", color=ps.INK,
                 fontsize=12, loc="left")
    ax.set_xlabel("fixes applied, cumulative", color=ps.INK_SECONDARY)
    ax.set_ylabel("seconds", color=ps.INK_SECONDARY)

    ax = axes[2]
    names = ["data", "forward", "backward", "optimizer", "logging"]
    bottom = np.zeros(len(breakdown))
    for i, n in enumerate(names):
        vals = np.array([b[n] for b in breakdown])
        ax.bar(np.arange(len(breakdown)), vals, 0.6, bottom=bottom,
               color=ps.SERIES[i], label=n)
        bottom += vals
    ax.set_xticks(np.arange(len(breakdown)))
    ax.set_xticklabels([s[0][0] for s in stages])
    ax.set_yscale("log")
    ax.set_title("What the epoch is made of", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("fixes applied, cumulative", color=ps.INK_SECONDARY)
    ax.set_ylabel("seconds (log)", color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9)

    ps.save(fig, OUT / "profile_and_fix.png")


if __name__ == "__main__":
    main()
