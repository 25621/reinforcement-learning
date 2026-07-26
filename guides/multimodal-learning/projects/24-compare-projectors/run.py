"""Compare projectors: linear vs 2-layer MLP vs average-pooling vs Q-Former.

Same frozen CLIP, same frozen SmolLM2-135M, same COCO captions, same steps,
same seed -- only the bridge between them changes:

    linear   one matrix, 49 image tokens                     (LLaVA 1.0)
    mlp2     two matrices with a GELU, 49 tokens             (LLaVA 1.5)
    pool     average 7x7 patches down to 4x4, then mlp2      (Qwen2-VL style)
    qformer  16 learned queries that read the patches        (BLIP-2)

Stages
    train   run the arms (each ~3-5 min)
    cost    prefill time and KV-cache size per arm at inference
    plot    figures

Requires project 20's cached CLIP features: run
`python3 ../20-llava-from-scratch/run.py --stage data` first.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20-llava-from-scratch"))
import vlm_lib as V  # noqa: E402

OUT = HERE / "outputs"
KINDS = ["linear", "mlp2", "pool", "qformer"]


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    data = V.CocoVLMData()
    rows, curves = [], {}
    for kind in args.kinds:
        res, curve, _ = V.align_train(kind, data, args.steps, args.bs, args.lr,
                                      tag=kind, eval_groups=args.eval_groups)
        rows.append(res)
        curves[kind] = curve
    old = json.loads((OUT / "projectors.json").read_text()) if (OUT / "projectors.json").exists() else []
    keep = [r for r in old if r["arm"] not in args.kinds]
    (OUT / "projectors.json").write_text(json.dumps(keep + rows, indent=1))
    prev = ((OUT / "curves.csv").read_text().splitlines()[1:]
            if (OUT / "curves.csv").exists() else [])
    prev = [p for p in prev if p.split(",")[0] not in args.kinds]
    lines = [f"{k},{s},{l:.5f}" for k, c in curves.items() for s, l in c]
    (OUT / "curves.csv").write_text("\n".join(["arm,step,train_loss"] + prev + lines) + "\n")


def stage_cost(args):
    """What each choice costs at *inference* time, which is where token count
    stops being an abstraction: prefill work and KV-cache bytes per image."""
    OUT.mkdir(exist_ok=True)
    tok, llm = V.load_llm()
    cfg = llm.config
    kv_per_token = (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
                    * (cfg.hidden_size // cfg.num_attention_heads) * 4)
    rows = []
    for kind in KINDS:
        proj = V.Projector(kind, V.CLIP_DIM, cfg.hidden_size,
                           out_rms=V.embedding_rms(llm))
        vlm = V.TinyVLM(llm, proj)
        n_img = proj.n_tokens()
        feats = torch.randn(1, V.CLIP_TOKENS, V.CLIP_DIM)
        b = V.make_batch(tok, [V.INSTRUCTION], ["a dog on a beach"], n_img=n_img)
        with torch.no_grad():
            vlm(b, feats)
            t0 = time.time()
            for _ in range(5):
                vlm(b, feats)
            prefill = (time.time() - t0) / 5 * 1000
            t0 = time.time()
            for _ in range(20):
                proj(feats)
            pms = (time.time() - t0) / 20 * 1000
        rows.append(dict(arm=kind, image_tokens=n_img, seq_len=int(b.ids.shape[1]),
                         params=V.n_params(proj), projector_ms=pms,
                         prefill_ms=prefill, kv_kb=n_img * kv_per_token / 1024))
        print(f"  {kind:8s} {n_img:3d} tokens  {V.n_params(proj) / 1e3:7.1f}k params  "
              f"projector {pms:5.2f} ms  prefill {prefill:6.1f} ms  "
              f"KV {n_img * kv_per_token / 1024:6.0f} KB", flush=True)
    (OUT / "cost.json").write_text(json.dumps(
        dict(kv_bytes_per_token=kv_per_token, rows=rows), indent=1))


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {r["arm"]: r for r in json.loads((OUT / "projectors.json").read_text())}
    cost = {r["arm"]: r for r in json.loads((OUT / "cost.json").read_text())["rows"]}
    kinds = [k for k in KINDS if k in res]
    curves = {}
    for line in (OUT / "curves.csv").read_text().splitlines()[1:]:
        a, s, l = line.split(",")
        curves.setdefault(a, []).append((int(s), float(l)))

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))
    ax = axes[0]
    for k in kinds:
        c = np.array(curves[k])
        w = 15
        ax.plot(c[w - 1:, 0], np.convolve(c[:, 1], np.ones(w) / w, "valid"), label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_title("Same data, same steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    x = np.arange(len(kinds))
    ax = axes[1]
    ax.bar(x, [res[k]["val_loss"] for k in kinds], .55, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=8, rotation=12)
    ax.set_ylabel("held-out caption loss")
    lo = min(res[k]["val_loss"] for k in kinds)
    hi = max(res[k]["val_loss"] for k in kinds)
    ax.set_ylim(lo - .12, hi + .06)
    ax.set_title("Quality")
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    ax.bar(x - .2, [res[k]["choice_raw"] for k in kinds], .4, label="raw score")
    ax.bar(x + .2, [res[k]["choice_lift"] for k in kinds], .4, label="image-lift score")
    ax.axhline(1 / V.GALLERY, ls="--", c="k", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=8, rotation=12)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Caption retrieval out of {V.GALLERY}")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[3]
    ax.scatter([cost[k]["prefill_ms"] for k in kinds],
               [res[k]["val_loss"] for k in kinds], s=70)
    for k in kinds:
        ax.annotate(f"{k} ({cost[k]['image_tokens']} tok)",
                    (cost[k]["prefill_ms"], res[k]["val_loss"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("prefill time for one image + prompt (ms)")
    ax.set_ylabel("held-out caption loss")
    ax.set_title("Quality vs cost")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "projectors.png", dpi=130)
    print("wrote", OUT / "projectors.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["train", "cost", "plot"])
    p.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-groups", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    {"train": stage_train, "cost": stage_cost, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
