"""World-model planning: six experiments.

  1. what play data contains, and what the model learns from it
  2. planning with the learned model, against the true simulator
  3. the score the planner optimises
  4. how far ahead to plan
  5. how much play data is enough
  6. latent dynamics vs plain state dynamics
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import world as W          # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

N_PLAY = 200               # episodes per play-data kind (mixed uses both)
SHAPE_W = 1.0              # weight on the "get behind the puck" term
PLAN = dict(horizon=15, pop=100, iters=4, elites=10, replan_every=3)
N_EPS = 12
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:46s} {value}  {unit}", flush=True)


def get_play(kind, n_eps=N_PLAY, seed=0):
    if kind == "mixed":
        a = W.collect_play(n_eps // 2, seed=seed, kind="scripted")
        b = W.collect_play(n_eps // 2, seed=seed + 1, kind="random")
        S = np.concatenate([a[0], b[0]])
        Aa = np.concatenate([a[1], b[1]])
        S2 = np.concatenate([a[2], b[2]])
        G = np.concatenate([a[3], b[3]])
        meta = dict(n=len(S), touch_frac=(a[4]["touch_frac"] + b[4]["touch_frac"]) / 2)
        return S, Aa, S2, G, meta
    return W.collect_play(n_eps, seed=seed, kind=kind)


def model_errors(model, S, Aa, S2):
    """One-step puck error, overall and on the transitions with contact."""
    moved = np.linalg.norm(S2[:, 4:6] - S[:, 4:6], axis=1) > 1e-6
    with torch.no_grad():
        pred = model(torch.tensor(S), torch.tensor(Aa)).numpy()
    e = np.linalg.norm(pred[:, 4:6] - S2[:, 4:6], axis=1)
    return float(e.mean()), float(e[moved].mean()) if moved.any() else np.nan, \
        float(moved.mean())


def rollout_error(model, horizon=15, n_eps=20, seed=5):
    """Prediction error after k steps of feeding the model its own output."""
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    errs = np.zeros((n_eps, horizon))
    for i in range(n_eps):
        env.reset()
        s = torch.tensor(W.env_state(env), dtype=torch.float32)[None]
        a_prev = np.zeros(2)
        for k in range(horizon):
            a = np.clip(0.7 * a_prev + 0.7 * rng.uniform(-1, 1, 2), -1, 1)
            a_prev = a
            with torch.no_grad():
                s = model(s, torch.tensor(a, dtype=torch.float32)[None])
            env.step(a)
            errs[i, k] = np.linalg.norm(s[0, 4:6].numpy() - env.puck)
    return errs.mean(0)


# ---------------------------------------------------------------------------
def job(cfg):
    torch.set_num_threads(2)
    kind = cfg.get("kind", "mixed")
    S, Aa, S2, G, meta = get_play(kind, cfg.get("n_eps", N_PLAY))
    out = dict(cfg=cfg, meta=meta)

    if cfg["what"] == "latent":
        # the latent model gets 2.5x the epochs: it has a harder objective
        # (latent consistency AND reward prediction) and losing because it was
        # undertrained would not be a result about latent dynamics
        model = W.train_latent_model(S, Aa, S2, G, epochs=150)
        fn = lambda env, s0, g: W.model_rollout_fn(model, s0, g, kind="latent",
                                                   shape_w=cfg.get("shape_w", SHAPE_W))
    else:
        model = W.train_state_model(S, Aa, S2, epochs=60)
        out["err_all"], out["err_contact"], out["contact_frac"] = \
            model_errors(model, S, Aa, S2)
        out["rollout_err"] = rollout_error(model)
        fn = lambda env, s0, g: W.model_rollout_fn(model, s0, g,
                                                   shape_w=cfg.get("shape_w", SHAPE_W))

    p = dict(PLAN)
    p["horizon"] = cfg.get("horizon", PLAN["horizon"])
    out["plan"] = W.run_planner(fn, n_eps=N_EPS, **p)
    return out


def job_true(cfg):
    torch.set_num_threads(2)
    p = dict(PLAN)
    p.update(pop=60, iters=3, elites=6)
    fn = lambda env, s0, g: W.true_rollout_fn(env, s0, g, shape_w=SHAPE_W)
    return dict(cfg=cfg, plan=W.run_planner(fn, n_eps=6, **p))


def main():
    torch.set_num_threads(2)
    jobs = [dict(what="model", kind=k, tag=f"data-{k}")
            for k in ("random", "scripted", "mixed")]
    jobs += [dict(what="model", shape_w=w, tag=f"shape-{w}") for w in (0.0, 0.3, 3.0)]
    jobs += [dict(what="model", horizon=h, tag=f"H{h}") for h in (5, 10, 25)]
    jobs += [dict(what="model", n_eps=n, tag=f"n{n}") for n in (25, 50, 100)]
    jobs += [dict(what="latent", tag="latent")]

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        fut_true = ex.submit(job_true, dict(tag="true"))
        res = list(ex.map(job, jobs))
        true_res = fut_true.result()
    R = {r["cfg"]["tag"]: r for r in res}
    print(f"[{time.time() - T0:6.1f}s] all runs finished", flush=True)

    base = R["data-mixed"]

    # -- 1. the play data and the model -------------------------------------
    for k in ("random", "scripted", "mixed"):
        r = R[f"data-{k}"]
        record("1-play-data", f"{k}: transitions / fraction with contact",
               f"{r['meta']['n']} / {r['contact_frac']:.3f}")
        record("1-play-data", f"{k}: 1-step puck error, ALL transitions",
               f"{r['err_all'] * 1000:.3f} mm")
        record("1-play-data", f"{k}: 1-step puck error, CONTACT transitions",
               f"{r['err_contact'] * 1000:.3f} mm")
        record("1-play-data", f"{k}: contact error / overall error",
               round(r["err_contact"] / r["err_all"], 1), "x")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ks = ["random", "scripted", "mixed"]
    x = np.arange(len(ks))
    axes[0].bar(x - 0.18, [R[f"data-{k}"]["err_all"] * 1000 for k in ks], 0.36,
                label="all transitions")
    axes[0].bar(x + 0.18, [R[f"data-{k}"]["err_contact"] * 1000 for k in ks], 0.36,
                label="transitions where the puck moved")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ks)
    axes[0].set_ylabel("one-step puck error (mm)")
    axes[0].set_title("The average error hides the only part that matters")
    axes[0].legend(fontsize=8)
    for k in ks:
        axes[1].plot(np.arange(1, 16), R[f"data-{k}"]["rollout_err"] * 1000,
                     label=k)
    axes[1].set_xlabel("steps predicted ahead")
    axes[1].set_ylabel("puck error (mm)")
    axes[1].set_title("Errors compound when the model eats its own output")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_error.png"), dpi=110)
    plt.close(fig)

    # -- 2. planning --------------------------------------------------------
    for k in ks:
        p = R[f"data-{k}"]["plan"]
        record("2-planning", f"model from {k} play: success / ms per decision",
               f"{p['success']:.3f} / {p['ms_per_decision']:.1f}")
    tp = true_res["plan"]
    record("2-planning", "planning in the TRUE simulator: success / ms",
           f"{tp['success']:.3f} / {tp['ms_per_decision']:.1f}")
    record("2-planning", "learned model speed-up over simulating",
           round(tp["ms_per_decision"] / base["plan"]["ms_per_decision"], 1), "x")
    record("2-planning", "gap to the true model (success)",
           round(tp["success"] - base["plan"]["success"], 3))

    # -- 3. the score -------------------------------------------------------
    ws = [0.0, 0.3, 1.0, 3.0]
    sc = [R["shape-0.0"]["plan"]["success"], R["shape-0.3"]["plan"]["success"],
          base["plan"]["success"], R["shape-3.0"]["plan"]["success"]]
    for w, s in zip(ws, sc):
        record("3-planning-score", f"approach weight {w}: success", round(s, 3))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    axes[0].plot(ws, sc, "o-")
    axes[0].set_xlabel("weight on 'get behind the puck'")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("A planner needs a score it can climb")
    axes[0].grid(alpha=0.3)

    # -- 4. horizon ---------------------------------------------------------
    hs = [5, 10, 15, 25]
    hsc = [R["H5"]["plan"]["success"], R["H10"]["plan"]["success"],
           base["plan"]["success"], R["H25"]["plan"]["success"]]
    hms = [R["H5"]["plan"]["ms_per_decision"], R["H10"]["plan"]["ms_per_decision"],
           base["plan"]["ms_per_decision"], R["H25"]["plan"]["ms_per_decision"]]
    for h, s, m in zip(hs, hsc, hms):
        record("4-horizon", f"horizon {h}: success / ms per decision",
               f"{s:.3f} / {m:.1f}")
    axes[1].plot(hs, hsc, "o-")
    axes[1].set_xlabel("planning horizon (steps)")
    axes[1].set_ylabel("success rate")
    axes[1].set_title("Too short to reach, too long to trust")
    axes[1].grid(alpha=0.3)

    # -- 5. data scaling ----------------------------------------------------
    ns = [25, 50, 100, 200]
    nsc = [R["n25"]["plan"]["success"], R["n50"]["plan"]["success"],
           R["n100"]["plan"]["success"], base["plan"]["success"]]
    for n, s in zip(ns, nsc):
        tag = "data-mixed" if n == 200 else f"n{n}"
        record("5-data-scaling", f"{n} play episodes ({R[tag]['meta']['n']} "
               f"transitions): success", round(s, 3))
        record("5-data-scaling", f"{n} play episodes: contact error",
               f"{R[tag]['err_contact'] * 1000:.2f} mm")
    axes[2].plot([R["n25"]["meta"]["n"], R["n50"]["meta"]["n"],
                  R["n100"]["meta"]["n"], base["meta"]["n"]], nsc, "o-")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("play transitions")
    axes[2].set_ylabel("success rate")
    axes[2].set_title("More play, better plans")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "planning.png"), dpi=110)
    plt.close(fig)

    # -- 6. latent vs state -------------------------------------------------
    record("6-latent-vs-state", "latent-dynamics model: success",
           round(R["latent"]["plan"]["success"], 3))
    record("6-latent-vs-state", "state-space model: success",
           round(base["plan"]["success"], 3))
    record("6-latent-vs-state", "latent ms per decision",
           round(R["latent"]["plan"]["ms_per_decision"], 1))

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
