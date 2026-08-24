#!/usr/bin/env python3
"""Project 70 — MoE serving: measuring expert imbalance under a real workload.

  A. Where do tokens actually go?  Per-layer, per-expert load for a real MoE
     (granite-3.0-1b-a400m: 32 experts, top-8, 24 layers).
  B. The knob nobody mentions: how many tokens are in the step.  Part of the
     imbalance is a small-sample effect that batching cures; the rest is the
     router's own preference, and no batch size touches it.
  C. Does the workload change the routing?  Four corpora, one router.
  D. Expert parallelism: placement and capacity factor.  What fraction of the
     hardware the imbalance actually wastes, and whether a placement tuned on
     one workload survives another.
  E. The all-to-all bill, in bytes, from the measured shapes.

  python3 run.py           # ~5 minutes
  python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import moelib as M                                             # noqa: E402

OUT = os.path.join(HERE, "outputs")
FINDINGS = os.path.join(OUT, "findings.json")
ASSIGN = os.path.join(OUT, "assign.npz")

CORPORA = ("wiki", "code", "exam", "chat")
CHUNK, N_CHUNKS = 512, 6           # per corpus: 6 x 512 = 3072 tokens
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
REPEATS = 300


# ---------------------------------------------------------------------------

def collect():
    M.add_quantlib_to_path()
    import quantlib

    print("loading granite-3.0-1b-a400m-instruct (a real MoE) ...", flush=True)
    t0 = time.time()
    tok, model = M.load()
    cfg = M.moe_config(model)
    print(f"  loaded in {time.time() - t0:.1f}s  {cfg}", flush=True)

    texts = quantlib.corpora(tok, names=CORPORA)
    per = {}
    with M.RouterTap(model) as tap:
        for name in CORPORA:
            chunks = quantlib.token_chunks(tok, texts[name], chunk=CHUNK,
                                           n=N_CHUNKS)
            t0 = time.time()
            per[name] = M.route_corpus(model, tap, chunks)
            print(f"  {name:<5} {per[name].shape} routed in "
                  f"{time.time() - t0:.1f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(ASSIGN, **per,
                        cfg=np.array(json.dumps(cfg), dtype=object))
    return per, cfg


def load_assign():
    z = np.load(ASSIGN, allow_pickle=True)
    cfg = json.loads(str(z["cfg"]))
    return {k: z[k] for k in CORPORA}, cfg


# ---------------------------------------------------------------------------

def section_A(per, cfg):
    E, K = cfg["n_experts"], cfg["top_k"]
    allA = np.concatenate([per[c] for c in CORPORA], axis=1)   # [L, T, K]
    L, T, _ = allA.shape
    per_layer = []
    for li in range(L):
        c = M.counts(allA[li], E)
        per_layer.append(dict(layer=li, imbalance=round(M.imbalance(c), 3),
                              entropy=round(M.norm_entropy(c), 4),
                              hottest=int(c.argmax()),
                              hot_share=round(float(c.max() / c.sum()), 4),
                              coldest=int(c.argmin()),
                              cold_share=round(float(c.min() / c.sum()), 4)))
    imb = [r["imbalance"] for r in per_layer]
    # "always-on" experts: how many tokens does each expert see, as a share of
    # all tokens (not of all dispatches)?  An expert on the top-8 list of every
    # token has share 1.0 and is really part of the dense model.
    always_on, nearly_dead = 0, 0
    for li in range(L):
        c = M.counts(allA[li], E)
        share = c / float(T)
        always_on += int((share > 0.95).sum())
        nearly_dead += int((share < 0.05).sum())
    return dict(n_tokens=int(T), n_layers=int(L), n_experts=E, top_k=K,
                ceiling=round(E / K, 2),
                always_on=always_on, nearly_dead=nearly_dead,
                slots=L * E,
                per_layer=per_layer,
                mean_imbalance=round(float(np.mean(imb)), 3),
                worst_layer=int(np.argmax(imb)), worst=round(max(imb), 3),
                best_layer=int(np.argmin(imb)), best=round(min(imb), 3),
                mean_entropy=round(float(np.mean(
                    [r["entropy"] for r in per_layer])), 4),
                heat=np.stack([M.counts(allA[li], E) for li in range(L)]
                              ).tolist())


def section_B(per, cfg):
    """Imbalance as a function of how many tokens are in one step."""
    E, K = cfg["n_experts"], cfg["top_k"]
    allA = np.concatenate([per[c] for c in CORPORA], axis=1)
    L, T, _ = allA.shape
    rng = np.random.default_rng(0)
    off = (np.arange(L) * E)[:, None, None]
    rows = []
    for B in BATCH_SIZES:
        if B > T:
            continue
        imbs, drops = [], []
        for _ in range(REPEATS):
            idx = rng.choice(T, size=B, replace=False)
            codes = (allA[:, idx, :].astype(np.int64) + off).reshape(-1)
            cnt = np.bincount(codes, minlength=L * E).reshape(L, E)
            imbs.append(float((cnt.max(axis=1) / cnt.mean(axis=1)).mean()))
            cap = 1.25 * B * K / E
            drops.append(float(np.maximum(cnt - cap, 0).sum()
                               / (L * B * K)))
        rows.append(dict(batch=B, imbalance=round(float(np.mean(imbs)), 3),
                         imbalance_p95=round(M.pct(imbs, 95), 3),
                         drop_at_cf125=round(float(np.mean(drops)), 4),
                         idle=round(1 - 1 / float(np.mean(imbs)), 3)))
    return dict(rows=rows, repeats=REPEATS,
                sqrt_law=[dict(batch=r["batch"],
                               predicted=round(1 + (E / (r["batch"] * K)) ** 0.5
                                               * 2.0, 3))
                          for r in rows])


def section_C(per, cfg):
    E = cfg["n_experts"]
    cnt = {c: M.counts(per[c], E) for c in CORPORA}
    js = {a: {b: round(M.js_divergence(cnt[a], cnt[b]), 4) for b in CORPORA}
          for a in CORPORA}
    top = {c: [int(i) for i in np.argsort(-cnt[c])[:5]] for c in CORPORA}
    # a per-layer view: is the disagreement concentrated in some layers?
    L = per[CORPORA[0]].shape[0]
    per_layer_js = []
    for li in range(L):
        vals = []
        for i, a in enumerate(CORPORA):
            for b in CORPORA[i + 1:]:
                vals.append(M.js_divergence(M.counts(per[a][li], E),
                                            M.counts(per[b][li], E)))
        per_layer_js.append(round(float(np.mean(vals)), 4))
    return dict(js=js, top5=top, per_layer_js=per_layer_js,
                mean_js=round(float(np.mean(
                    [js[a][b] for i, a in enumerate(CORPORA)
                     for b in CORPORA[i + 1:]])), 4),
                worst_layer=int(np.argmax(per_layer_js)),
                worst_layer_js=round(max(per_layer_js), 4),
                counts={c: cnt[c].tolist() for c in CORPORA})


def section_D(per, cfg):
    """Expert parallelism: placement, and the capacity factor."""
    E, K = cfg["n_experts"], cfg["top_k"]
    L = per[CORPORA[0]].shape[0]
    out = {"placement": {}, "capacity": {}}

    # placement is fitted on `wiki` and evaluated on the other three
    fit = np.concatenate([M.counts(per["wiki"][li], E)[None] for li in range(L)])
    for n_dev in (4, 8):
        res = {}
        for name in ("contiguous", "round_robin", "balanced"):
            effs = {}
            for corpus in CORPORA:
                per_layer_eff = []
                for li in range(L):
                    order = np.argsort(-fit[li])          # fitted on wiki only
                    pl = M.placements(E, n_dev, order=order)[name]
                    c = M.counts(per[corpus][li], E)
                    loads = M.device_loads(c, pl, n_dev)
                    per_layer_eff.append(loads.mean() / loads.max())
                effs[corpus] = round(float(np.mean(per_layer_eff)), 4)
            res[name] = effs
        out["placement"][str(n_dev)] = res

    allA = np.concatenate([per[c] for c in CORPORA], axis=1)
    T = allA.shape[1]
    rng = np.random.default_rng(1)
    for B in (8, 64, 512):
        rows = []
        for cf in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
            drops = []
            for _ in range(120):
                idx = rng.choice(T, size=B, replace=False)
                sub = allA[:, idx, :]
                d = []
                for li in range(L):
                    c = M.counts(sub[li], E)
                    d.append(M.dropped_fraction(c, B, K, cf))
                drops.append(float(np.mean(d)))
            rows.append(dict(cf=cf, dropped=round(float(np.mean(drops)), 4),
                             buffer_x=cf))
        out["capacity"][str(B)] = rows
    return out


def section_E(cfg, B_stats):
    """The all-to-all bill, from the model's own shapes."""
    h, ffn, K, E = cfg["hidden"], cfg["ffn"], cfg["top_k"], cfg["n_experts"]
    bytes_per_el = 2                                    # bf16 on the wire
    dispatch = K * h * bytes_per_el                     # token copies out
    combine = K * h * bytes_per_el                      # results back
    wire = dispatch + combine
    flops = 2 * K * (3 * h * ffn)                       # gate+up+down per token
    return dict(hidden=h, ffn=ffn, top_k=K, n_experts=E,
                wire_bytes_per_token=wire,
                flops_per_token=flops,
                flops_per_wire_byte=round(flops / wire, 1),
                note="bf16 payload; two hops (dispatch and combine)")


# ---------------------------------------------------------------------------

def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C, D = F["A"], F["B"], F["C"], F["D"]
    fig = plt.figure(figsize=(14, 9.4))
    fig.suptitle("MoE serving: 32 experts, top-8, 24 layers — where the tokens "
                 "go and what it costs", fontsize=13)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    p = fig.add_subplot(gs[0, 0])
    heat = np.array(A["heat"], dtype=float)
    heat = heat / heat.mean(axis=1, keepdims=True)
    im = p.imshow(heat, aspect="auto", cmap="magma", vmin=0, vmax=2.5)
    p.set_xlabel("expert id")
    p.set_ylabel("layer")
    p.set_title(f"A. Load per expert / average  ({A['n_tokens']} tokens)\n"
                f"mean imbalance {A['mean_imbalance']}x, "
                f"worst layer {A['worst_layer']} at {A['worst']}x")
    fig.colorbar(im, ax=p, shrink=0.85, label="x average load")

    p = fig.add_subplot(gs[0, 1])
    xs = [r["batch"] for r in B["rows"]]
    p.plot(xs, [r["imbalance"] for r in B["rows"]], "o-", color="#c0504d",
           label="mean imbalance (max/avg)")
    p.plot(xs, [r["imbalance_p95"] for r in B["rows"]], "s--", color="#8d9db6",
           label="p95 over steps")
    p.axhline(1.0, color="k", lw=1)
    floor = B["rows"][-1]["imbalance"]
    p.axhline(floor, color="#4c9f70", ls=":", lw=1.5,
              label=f"floor {floor}x — the router's own bias")
    p.set_xscale("log", base=2)
    p.set_xlabel("tokens in the step  (decode batch  →  prefill)")
    p.set_ylabel("max expert load / average")
    p.set_title("B. Batching helps, then stops helping")
    p.grid(alpha=0.3)
    p.legend(fontsize=8)

    p = fig.add_subplot(gs[1, 0])
    names = list(CORPORA)
    mat = np.array([[C["js"][a][b] for b in names] for a in names])
    im = p.imshow(mat, cmap="viridis")
    p.set_xticks(range(len(names)))
    p.set_xticklabels(names)
    p.set_yticks(range(len(names)))
    p.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            p.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                   color="w", fontsize=8)
    p.set_title("C. Do different workloads use different experts?\n"
                "Jensen-Shannon divergence (bits, 0 = identical)")
    fig.colorbar(im, ax=p, shrink=0.85)

    p = fig.add_subplot(gs[1, 1])
    w = 0.25
    devs = ["4", "8"]
    schemes = ["contiguous", "round_robin", "balanced"]
    colors = {"contiguous": "#8d9db6", "round_robin": "#4a6fa5",
              "balanced": "#4c9f70"}
    for si, s in enumerate(schemes):
        vals = []
        for nd in devs:
            v = D["placement"][nd][s]
            vals.append(float(np.mean([v[c] for c in CORPORA if c != "wiki"])))
        p.bar([i + (si - 1) * w for i in range(len(devs))], vals, w,
              label=s, color=colors[s])
    p.set_xticks(range(len(devs)))
    p.set_xticklabels([f"EP={d} devices" for d in devs])
    p.set_ylim(0.5, 1.02)
    p.axhline(1.0, color="k", lw=1)
    p.set_ylabel("hardware efficiency (avg load / slowest device)")
    p.set_title("D. Expert placement, fitted on wiki, scored on the rest")
    p.legend(fontsize=8)

    path = os.path.join(OUT, "moe.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", path)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse outputs/assign.npz instead of re-running")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    if args.reuse and os.path.exists(ASSIGN):
        per, cfg = load_assign()
        print("reusing", ASSIGN)
    else:
        per, cfg = collect()

    A = section_A(per, cfg)
    B = section_B(per, cfg)
    C = section_C(per, cfg)
    D = section_D(per, cfg)
    E = section_E(cfg, B)
    F = dict(A=A, B=B, C=C, D=D, E=E, cfg=cfg)
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- A. routing over all four corpora -----------------------------")
    print(f"{A['n_tokens']} tokens x {A['n_layers']} layers x {A['top_k']}-of-"
          f"{A['n_experts']} experts")
    print(f"mean imbalance {A['mean_imbalance']}x   worst layer "
          f"{A['worst_layer']} {A['worst']}x   best layer {A['best_layer']} "
          f"{A['best']}x   mean entropy {A['mean_entropy']}")
    print("\n--- B. tokens per step -------------------------------------------")
    for r in B["rows"]:
        print(f"batch {r['batch']:>5}: imbalance {r['imbalance']:5.2f}x  "
              f"p95 {r['imbalance_p95']:5.2f}x  idle {r['idle'] * 100:4.1f}%  "
              f"dropped at cf=1.25 {r['drop_at_cf125'] * 100:5.2f}%")
    print("\n--- C. workload dependence ---------------------------------------")
    for a in CORPORA:
        print(f"  {a:<5} top experts {C['top5'][a]}   "
              + "  ".join(f"{b}:{C['js'][a][b]:.3f}" for b in CORPORA))
    print(f"mean pairwise JS {C['mean_js']} bits; worst layer "
          f"{C['worst_layer']} at {C['worst_layer_js']}")
    print("\n--- D. expert parallelism ----------------------------------------")
    for nd, res in D["placement"].items():
        for s, effs in res.items():
            held = np.mean([effs[c] for c in CORPORA if c != "wiki"])
            print(f"EP={nd} {s:<12} wiki(fit) {effs['wiki']:.3f}  "
                  f"held-out {held:.3f}   "
                  + " ".join(f"{c}:{effs[c]:.3f}" for c in CORPORA))
    for B_, rows in D["capacity"].items():
        print(f"  batch {B_:>4}: " + "  ".join(
            f"cf={r['cf']}:{r['dropped'] * 100:.2f}%" for r in rows))
    print("\n--- E. the wire --------------------------------------------------")
    print(f"{E['wire_bytes_per_token']} bytes across the all-to-all per token "
          f"per layer, {E['flops_per_token']:,} FLOPs of expert work: "
          f"{E['flops_per_wire_byte']} FLOPs per wire byte")
    plot(F)


if __name__ == "__main__":
    main()
