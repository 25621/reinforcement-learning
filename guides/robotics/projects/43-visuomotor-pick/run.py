"""Project 43 -- a policy that goes straight from camera pixels to hand motion.

Seven experiments:

  1. the task, and what one expert episode looks like
  2. cloning the expert: three policies, one architecture
  3. success rates, against a ceiling and against a control that cannot see
  4. how many demonstrations it takes
  5. compounding error: the gap between watching and doing
  6. a second object on the table that has nothing to do with the task
  7. move the camera two centimetres

Runs in about eight minutes on CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import env as E                                                               # noqa: E402
import policy as PL                                                           # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

N_DEMOS = 300
N_EVAL = 60
MODES = ["state", "pixel", "image"]
LABEL = {"state": "state policy (given the object's pose)",
         "pixel": "pixel policy (camera + hand position)",
         "image": "pixel policy (camera only)"}


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<54s} {value}")


def collect(n, seed, **kw):
    rng = np.random.default_rng(seed)
    O, A, ok = [], [], 0
    for _ in range(n):
        e = E.PickEnv(rng, **kw)
        r = E.rollout(e)
        e.close()
        if not r["success"]:
            continue          # only clone the demonstrations that worked
        ok += 1
        O.extend(r["obs"])
        A.append(r["act"])
    img, prop, priv = PL.pack(O)
    return (img, prop, priv, np.concatenate(A)), ok / n


def evaluate(pol, n, seed, **kw):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        e = E.PickEnv(rng, **kw)
        r = E.rollout(e, policy=pol)
        e.close()
        hits += r["success"]
    return hits / n


# ---------------------------------------------------------------------------
# 1. the task
# ---------------------------------------------------------------------------

def exp1_task():
    print("\n[1] the task")
    rng = np.random.default_rng(11)
    e = E.PickEnv(rng)
    r = E.rollout(e, render=True)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.2))
    idx = np.linspace(0, len(r["frames"]) - 1, 8).astype(int)
    for ax, j in zip(axes.ravel(), idx):
        ax.imshow(r["frames"][j])
        ax.set_title(f"step {j}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"one expert episode, as the policy sees it "
                 f"({E.IMG_W}x{E.IMG_H} pixels) -- lifted: {r['success']}")
    save(fig, os.path.join(OUT, "task.png"))
    e.close()
    record("1_task", "image size fed to the network", f"{E.IMG_W}x{E.IMG_H}")
    record("1_task", "decisions per episode", E.EP_LEN)
    record("1_task", "action", "dx, dy, dz, d(yaw), open/close")


# ---------------------------------------------------------------------------
# 2 + 3. clone and compare
# ---------------------------------------------------------------------------

def exp23(state):
    print("\n[2] collecting demonstrations")
    t0 = time.time()
    data, rate = collect(N_DEMOS, seed=0)
    record("2_clone", "expert episodes attempted", N_DEMOS)
    record("2_clone", "expert success rate", round(rate, 3))
    record("2_clone", "frames kept", len(data[3]))
    record("2_clone", "collection time (s)", round(time.time() - t0))
    val, _ = collect(40, seed=999)
    state["data"] = data
    state["val"] = val

    curves, nets, errs = {}, {}, {}
    for mode in MODES:
        t0 = time.time()
        net, c = PL.train(data, mode, log=lambda s: print(s))
        nets[mode] = net
        curves[mode] = c
        import torch
        with torch.no_grad():
            net.eval()
            out = net(torch.from_numpy(val[0]), torch.from_numpy(val[1]),
                      torch.from_numpy(val[2])).numpy()
        errs[mode] = float(np.abs(out[:, :4] - val[3][:, :4]).mean())
        record("2_clone", f"{LABEL[mode]}: held-out action error", round(errs[mode], 4))
        record("2_clone", f"{LABEL[mode]}: training time (s)", round(time.time() - t0))
    state["nets"] = nets

    print("\n[3] success rates")
    rates = {}
    for mode in MODES:
        rates[LABEL[mode]] = evaluate(PL.make_policy(nets[mode]), N_EVAL, seed=77)
    # the ceiling
    rates["scripted expert (the ceiling)"] = evaluate(None, N_EVAL, seed=77)
    # the control: replay the average expert action sequence with eyes shut
    mean_seq = np.zeros((E.EP_LEN, 5))
    A = data[3].reshape(-1, E.EP_LEN, 5) if len(data[3]) % E.EP_LEN == 0 else None
    if A is not None:
        mean_seq = A.mean(0)
        mean_seq[:, 4] = (mean_seq[:, 4] > 0.5).astype(float)
    box = dict(t=0)

    def blind(obs):
        a = mean_seq[min(box["t"], E.EP_LEN - 1)]
        box["t"] += 1
        return a

    hits = 0
    rng = np.random.default_rng(77)
    for _ in range(N_EVAL):
        box["t"] = 0
        e = E.PickEnv(rng)
        r = E.rollout(e, policy=blind)
        e.close()
        hits += r["success"]
    rates["open loop, eyes shut (control)"] = hits / N_EVAL

    for k, v in rates.items():
        record("3_success", f"{k}", round(v, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for mode in MODES:
        axes[0].plot(range(1, len(curves[mode]) + 1), curves[mode], "o-", ms=3,
                     label=LABEL[mode])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cloning loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    ks = ["scripted expert (the ceiling)", LABEL["state"], LABEL["pixel"],
          LABEL["image"], "open loop, eyes shut (control)"]
    axes[1].barh([k.replace(" (", "\n(") for k in ks],
                 [100 * rates[k] for k in ks],
                 color=["#42505e", COLORS[0], COLORS[2], COLORS[4], COLORS[6]])
    axes[1].set_xlabel("objects lifted (% of 60 episodes)")
    axes[1].tick_params(labelsize=7)
    save(fig, os.path.join(OUT, "compare.png"))
    state["rates"] = rates


# ---------------------------------------------------------------------------
# 4. how many demonstrations
# ---------------------------------------------------------------------------

def exp4_data(state):
    print("\n[4] how many demonstrations")
    data = state["data"]
    sizes = [30, 80, 160, N_DEMOS]
    out = {m: [] for m in ("state", "pixel")}
    for n in sizes:
        frames = min(len(data[3]), n * E.EP_LEN)
        sub = tuple(d[:frames] for d in data)
        for m in out:
            net, _ = PL.train(sub, m, seed=1)
            v = evaluate(PL.make_policy(net), 40, seed=310)
            out[m].append(v)
            record("4_data", f"{LABEL[m]}, ~{n} demos", round(v, 4))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m, v in out.items():
        ax.plot(sizes, [100 * x for x in v], "o-", ms=4, label=LABEL[m])
    ax.axhline(100 * state["rates"]["scripted expert (the ceiling)"],
               color="#42505e", ls="--", lw=1)
    ax.text(sizes[0], 100 * state["rates"]["scripted expert (the ceiling)"] - 6,
            "expert", fontsize=8)
    ax.set_xlabel("demonstrations")
    ax.set_ylabel("objects lifted (%)")
    ax.legend(fontsize=7.5)
    save(fig, os.path.join(OUT, "data.png"))


# ---------------------------------------------------------------------------
# 5. compounding error
# ---------------------------------------------------------------------------

def exp5_drift(state):
    print("\n[5] compounding error")
    nets = state["nets"]
    curves = {}
    for mode in ("state", "pixel"):
        pol = PL.make_policy(nets[mode])
        rng = np.random.default_rng(505)
        per_t = [[] for _ in range(E.EP_LEN)]
        for _ in range(30):
            e = E.PickEnv(rng)
            obs = e.reset()
            for t in range(E.EP_LEN):
                a_pi = pol(obs)
                a_ex = np.clip(E.expert_action(e), -1, 1)
                per_t[t].append(float(np.abs(a_pi[:4] - a_ex[:4]).mean()))
                obs, _, done = e.step(a_pi)
                if done:
                    break
            e.close()
        curves[mode] = [np.mean(x) if x else np.nan for x in per_t]
        record("5_drift", f"{LABEL[mode]}: action error at step 1",
               round(float(curves[mode][0]), 4))
        record("5_drift", f"{LABEL[mode]}: action error at step {E.EP_LEN}",
               round(float(curves[mode][-1]), 4))
        record("5_drift", f"{LABEL[mode]}: error on the expert's own states "
                          f"(held out)",
               round(float(_val_err(nets[mode], state["val"])), 4))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for m, v in curves.items():
        ax.plot(range(1, E.EP_LEN + 1), v, "o-", ms=3, label=LABEL[m])
        ax.axhline(_val_err(nets[m], state["val"]), ls="--", lw=1,
                   color=COLORS[0] if m == "state" else COLORS[2])
    ax.set_xlabel("step within the episode")
    ax.set_ylabel("|policy action - expert action|")
    ax.legend(fontsize=7.5)
    ax.set_title("solid: states the POLICY visits.  dashed: states the EXPERT visits",
                 fontsize=8.5)
    save(fig, os.path.join(OUT, "drift.png"))


def _val_err(net, val):
    import torch
    with torch.no_grad():
        net.eval()
        out = net(torch.from_numpy(val[0]), torch.from_numpy(val[1]),
                  torch.from_numpy(val[2])).numpy()
    return float(np.abs(out[:, :4] - val[3][:, :4]).mean())


# ---------------------------------------------------------------------------
# 6 + 7. two ways to break it
# ---------------------------------------------------------------------------

def exp67_break(state):
    print("\n[6] a distractor object")
    nets = state["nets"]
    rows = {}
    for mode in ("state", "pixel"):
        pol = PL.make_policy(nets[mode])
        rows.setdefault("clean", {})[mode] = state["rates"][LABEL[mode]]
        rows.setdefault("+ 1 distractor", {})[mode] = evaluate(
            pol, 45, seed=606, n_distract=1)
        rows.setdefault("+ 2 distractors", {})[mode] = evaluate(
            pol, 45, seed=607, n_distract=2)
        print("\n[7] camera moved (mode %s)" % mode)
        for shift in (0.01, 0.02, 0.04):
            rows.setdefault(f"camera +{1000 * shift:.0f} mm", {})[mode] = evaluate(
                pol, 40, seed=708, cam_shift=shift)
    for k, v in rows.items():
        for m, x in v.items():
            record("67_break", f"{k}: {LABEL[m]}", round(x, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    keys = ["clean", "+ 1 distractor", "+ 2 distractors"]
    x = np.arange(len(keys))
    for i, m in enumerate(("state", "pixel")):
        axes[0].bar(x + (i - 0.5) * 0.36, [100 * rows[k][m] for k in keys],
                    0.34, label=LABEL[m], color=COLORS[i * 2])
    axes[0].set_xticks(x); axes[0].set_xticklabels(keys, fontsize=8)
    axes[0].set_ylabel("objects lifted (%)")
    axes[0].legend(fontsize=7)
    shifts = [0, 10, 20, 40]
    for i, m in enumerate(("state", "pixel")):
        vals = [100 * rows["clean"][m]] + \
               [100 * rows[f"camera +{s} mm"][m] for s in (10, 20, 40)]
        axes[1].plot(shifts, vals, "o-", ms=4, label=LABEL[m], color=COLORS[i * 2])
    axes[1].set_xlabel("camera moved sideways (mm)")
    axes[1].set_ylabel("objects lifted (%)")
    axes[1].legend(fontsize=7)
    save(fig, os.path.join(OUT, "break.png"))


def main():
    use_style()
    t0 = time.time()
    state = {}
    exp1_task()
    exp23(state)
    exp4_data(state)
    exp5_drift(state)
    exp67_break(state)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
