"""Inference optimization for a VLM: where the time and the memory actually go.

vLLM and SGLang are not installed here (and would not fit a CPU-only box), so
instead of driving a serving engine we measure the *quantities those engines
optimise*, on the same real 135M-parameter VLM the rest of Phase 5 uses:

    images    how prefill, decode and tokens/sec move as images per request grow
    cache     KV cache on vs off (the difference between linear and quadratic)
    prefix    re-prefilling an image for every question vs reusing its KV once
    batch     throughput against per-request latency as the batch grows
    budget    the multimodal lever: fewer image tokens per image
    plot      figures

Timings do not depend on the projector's weights, so this project needs no
training -- but it does need project 20's vlm_lib.py and its cached features
for the sanity-check generation at the end.
"""

import argparse
import copy
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
NEW_TOKENS = 32
QUESTION = "Describe the image."


def build(kind="mlp2", n_pool=None):
    tok, llm = V.load_llm()
    proj = V.Projector(kind, V.CLIP_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))
    if n_pool:
        proj.pool_grid = n_pool
    return tok, llm, V.TinyVLM(llm, proj)


def kv_bytes_per_token(cfg, dtype_bytes=4):
    """Key and value, per layer, per token: 2 x layers x kv-heads x head-dim."""
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes


def prefill_once(vlm, batch, feats):
    emb = vlm.embed(batch, feats)
    res = vlm.body(inputs_embeds=emb, use_cache=True)
    return res.past_key_values, vlm.head(res[0][:, -1]).float().argmax(-1)


def decode_steps(vlm, past, first, n):
    """`n` greedy steps reusing the KV cache (what every engine does)."""
    nxt = first
    for _ in range(n):
        emb = vlm.llm.get_input_embeddings()(nxt).unsqueeze(1)
        res = vlm.body(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = res.past_key_values
        nxt = vlm.head(res[0][:, -1]).float().argmax(-1)
    return past


def decode_nocache(vlm, batch, feats, n):
    """`n` greedy steps *without* a cache: every step re-reads the whole prefix.

    This is the thing a KV cache removes. Nobody serves this way -- it is here so
    the cache's value is a measured number instead of a claim.
    """
    emb = vlm.embed(batch, feats)
    for _ in range(n):
        res = vlm.body(inputs_embeds=emb, use_cache=False)
        nxt = vlm.head(res[0][:, -1]).float().argmax(-1)
        emb = torch.cat([emb, vlm.llm.get_input_embeddings()(nxt).unsqueeze(1)], 1)
    return emb.shape[1]


# ---------------------------------------------------------------------------
# 1. images per request
# ---------------------------------------------------------------------------
def stage_images(args):
    OUT.mkdir(exist_ok=True)
    tok, llm, vlm = build()
    per_img = vlm.projector.n_tokens()
    rows = []
    for n_imgs in args.images:
        n_img_tok = per_img * n_imgs
        b = V.make_batch(tok, [QUESTION], ["x"], n_img=max(n_img_tok, 1),
                         with_image=n_imgs > 0)
        b = V.Batch(b.ids[:, :-2], b.image_mask[:, :-2], b.labels[:, :-2], b.attn[:, :-2])
        feats = torch.randn(1, n_img_tok, V.CLIP_DIM) if n_imgs else None
        with torch.no_grad():
            prefill_once(vlm, b, feats)                       # warm up
            t0 = time.time()
            past, first = prefill_once(vlm, b, feats)
            ttft = (time.time() - t0) * 1000
            t0 = time.time()
            decode_steps(vlm, past, first, NEW_TOKENS)
            dec = (time.time() - t0) * 1000
        rows.append(dict(images=n_imgs, image_tokens=n_img_tok,
                         prompt_tokens=int(b.ids.shape[1]), ttft_ms=ttft,
                         decode_ms=dec, ms_per_token=dec / NEW_TOKENS,
                         tokens_per_s=NEW_TOKENS / (dec / 1000),
                         end_to_end_tokens_per_s=NEW_TOKENS / ((ttft + dec) / 1000),
                         kv_mb=n_img_tok * kv_bytes_per_token(llm.config) / 1e6))
        print(f"  {n_imgs} image(s): {n_img_tok:4d} image tokens  "
              f"TTFT {ttft:7.1f} ms  decode {dec / NEW_TOKENS:5.1f} ms/token  "
              f"{rows[-1]['end_to_end_tokens_per_s']:5.1f} tok/s end-to-end", flush=True)
    (OUT / "images.json").write_text(json.dumps(
        dict(kv_bytes_per_token=kv_bytes_per_token(llm.config),
             new_tokens=NEW_TOKENS, rows=rows), indent=1))


# ---------------------------------------------------------------------------
# 2. KV cache on/off
# ---------------------------------------------------------------------------
def stage_cache(args):
    OUT.mkdir(exist_ok=True)
    tok, llm, vlm = build()
    n_img = vlm.projector.n_tokens()
    b = V.make_batch(tok, [QUESTION], ["x"], n_img=n_img)
    b = V.Batch(b.ids[:, :-2], b.image_mask[:, :-2], b.labels[:, :-2], b.attn[:, :-2])
    feats = torch.randn(1, n_img, V.CLIP_DIM)
    rows = []
    for n in args.decode_lengths:
        with torch.no_grad():
            t0 = time.time()
            past, first = prefill_once(vlm, b, feats)
            decode_steps(vlm, past, first, n)
            with_cache = (time.time() - t0) * 1000
            t0 = time.time()
            decode_nocache(vlm, b, feats, n)
            without = (time.time() - t0) * 1000
        rows.append(dict(new_tokens=n, with_cache_ms=with_cache,
                         without_cache_ms=without, speedup=without / with_cache))
        print(f"  {n:3d} new tokens: cache {with_cache:8.1f} ms   "
              f"no cache {without:8.1f} ms   {without / with_cache:5.2f}x", flush=True)
    (OUT / "cache.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# 3. reusing one image's KV across several questions
# ---------------------------------------------------------------------------
def stage_prefix(args):
    """Five questions about one picture.

    naive   send image + question again for every turn
    reuse   prefill the image once, then branch a copy of that cache per question
    """
    OUT.mkdir(exist_ok=True)
    tok, llm, vlm = build()
    n_img = vlm.projector.n_tokens()
    feats = torch.randn(1, n_img, V.CLIP_DIM)
    questions = ["Describe the image.", "What colour is it?", "How many are there?",
                 "Is it indoors?", "What is in the background?"]
    img_prefix = f"<|im_start|>user\n{V.IMAGE_TOKEN * n_img}"
    pre_ids = torch.tensor([tok.encode(img_prefix, add_special_tokens=False)])
    pre_batch = V.Batch(pre_ids, pre_ids == tok.convert_tokens_to_ids(V.IMAGE_TOKEN),
                        torch.full_like(pre_ids, -100), torch.ones_like(pre_ids))

    def tail_ids(q):
        return torch.tensor([tok.encode(
            f"\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)])

    naive_ttft, reuse_ttft = [], []
    with torch.no_grad():
        # naive: full prefill per question
        t0 = time.time()
        for q in questions:
            b = V.make_batch(tok, [q], ["x"], n_img=n_img)
            b = V.Batch(b.ids[:, :-2], b.image_mask[:, :-2], b.labels[:, :-2], b.attn[:, :-2])
            t1 = time.time()
            past, first = prefill_once(vlm, b, feats)
            naive_ttft.append((time.time() - t1) * 1000)
            decode_steps(vlm, past, first, args.turn_tokens)
        naive = (time.time() - t0) * 1000

        # reuse: one image prefill, then a cheap branch per question
        t0 = time.time()
        shared, _ = prefill_once(vlm, pre_batch, feats)
        shared_ms = (time.time() - t0) * 1000
        copy_ms = []
        for q in questions:
            t1 = time.time()
            past = copy.deepcopy(shared)     # a real engine shares pages instead
            copy_ms.append((time.time() - t1) * 1000)
            emb = vlm.llm.get_input_embeddings()(tail_ids(q))
            res = vlm.body(inputs_embeds=emb, past_key_values=past, use_cache=True)
            nxt = vlm.head(res[0][:, -1]).float().argmax(-1)
            reuse_ttft.append((time.time() - t1) * 1000)
            decode_steps(vlm, res.past_key_values, nxt, args.turn_tokens)
        reuse = (time.time() - t0) * 1000

    res = dict(turns=len(questions), tokens_per_turn=args.turn_tokens,
               image_tokens=n_img, naive_ms=naive, reuse_ms=reuse,
               shared_prefill_ms=shared_ms, speedup=naive / reuse,
               naive_ttft_ms=naive_ttft, reuse_ttft_ms=reuse_ttft,
               cache_copy_ms=copy_ms,
               ttft_speedup=float(np.mean(naive_ttft) / np.mean(reuse_ttft)))
    print(f"  naive {naive:.0f} ms   reuse {reuse:.0f} ms   {naive / reuse:.2f}x total; "
          f"TTFT {np.mean(naive_ttft):.0f} -> {np.mean(reuse_ttft):.0f} ms "
          f"({res['ttft_speedup']:.2f}x, cache copy {np.mean(copy_ms):.1f} ms)",
          flush=True)
    (OUT / "prefix.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------
# 4. batching
# ---------------------------------------------------------------------------
def stage_batch(args):
    OUT.mkdir(exist_ok=True)
    tok, llm, vlm = build()
    n_img = vlm.projector.n_tokens()
    rows = []
    for bs in args.batches:
        b = V.make_batch(tok, [QUESTION] * bs, ["x"] * bs, n_img=n_img)
        b = V.Batch(b.ids[:, :-2], b.image_mask[:, :-2], b.labels[:, :-2], b.attn[:, :-2])
        feats = torch.randn(bs, n_img, V.CLIP_DIM)
        with torch.no_grad():
            prefill_once(vlm, b, feats)
            t0 = time.time()
            past, first = prefill_once(vlm, b, feats)
            decode_steps(vlm, past, first, NEW_TOKENS)
            total = (time.time() - t0) * 1000
        rows.append(dict(batch=bs, latency_ms=total,
                         tokens_per_s=bs * NEW_TOKENS / (total / 1000),
                         kv_mb=bs * n_img * kv_bytes_per_token(llm.config) / 1e6))
        print(f"  batch {bs:2d}: latency {total:8.1f} ms   "
              f"{rows[-1]['tokens_per_s']:6.1f} tok/s aggregate", flush=True)
    (OUT / "batch.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# 5. the image-token budget
# ---------------------------------------------------------------------------
def stage_budget(args):
    OUT.mkdir(exist_ok=True)
    tok, llm, _ = build()
    rows = []
    for kind, grid in [("mlp2", None), ("pool", 4), ("pool", 3), ("pool", 2)]:
        proj = V.Projector(kind, V.CLIP_DIM, llm.config.hidden_size,
                           out_rms=V.embedding_rms(llm))
        if grid:
            proj.pool_grid = grid
        vlm = V.TinyVLM(llm, proj)
        n_img = proj.n_tokens()
        b = V.make_batch(tok, [QUESTION], ["x"], n_img=n_img)
        b = V.Batch(b.ids[:, :-2], b.image_mask[:, :-2], b.labels[:, :-2], b.attn[:, :-2])
        feats = torch.randn(1, V.CLIP_TOKENS, V.CLIP_DIM)
        with torch.no_grad():
            prefill_once(vlm, b, feats)
            t0 = time.time()
            past, first = prefill_once(vlm, b, feats)
            ttft = (time.time() - t0) * 1000
            t0 = time.time()
            decode_steps(vlm, past, first, NEW_TOKENS)
            dec = (time.time() - t0) * 1000
        name = f"{kind}" + (f"-{grid}x{grid}" if grid else "")
        rows.append(dict(arm=name, image_tokens=n_img, ttft_ms=ttft,
                         ms_per_token=dec / NEW_TOKENS,
                         kv_kb=n_img * kv_bytes_per_token(llm.config) / 1024))
        print(f"  {name:12s} {n_img:3d} tokens  TTFT {ttft:7.1f} ms  "
              f"decode {dec / NEW_TOKENS:5.1f} ms/tok", flush=True)
    (OUT / "budget.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# what this would look like on a real 7B VLM
# ---------------------------------------------------------------------------
def stage_scale(args):
    OUT.mkdir(exist_ok=True)
    models = [
        dict(name="SmolLM2-135M (this project)", layers=30, kv_heads=3, head_dim=64,
             bytes=4, tokens_per_image=49),
        dict(name="LLaVA-1.5-7B", layers=32, kv_heads=32, head_dim=128, bytes=2,
             tokens_per_image=576),
        dict(name="Qwen2-VL-7B (1 megapixel)", layers=28, kv_heads=4, head_dim=128,
             bytes=2, tokens_per_image=1280),
    ]
    rows = []
    for m in models:
        per_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * m["bytes"]
        rows.append(dict(model=m["name"], kv_bytes_per_token=per_tok,
                         tokens_per_image=m["tokens_per_image"],
                         mb_per_image=per_tok * m["tokens_per_image"] / 1e6,
                         images_in_10gb=10e9 / (per_tok * m["tokens_per_image"])))
        print(f"  {m['name']:32s} {per_tok / 1024:6.1f} KB/token  "
              f"{rows[-1]['mb_per_image']:7.2f} MB/image  "
              f"{rows[-1]['images_in_10gb']:8.0f} images in 10 GB", flush=True)
    (OUT / "scale.json").write_text(json.dumps(rows, indent=1))


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = json.loads((OUT / "images.json").read_text())
    cache = json.loads((OUT / "cache.json").read_text())
    batch = json.loads((OUT / "batch.json").read_text())
    budget = json.loads((OUT / "budget.json").read_text())
    rows = img["rows"]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))
    ax = axes[0]
    x = [r["images"] for r in rows]
    ax.plot(x, [r["ttft_ms"] for r in rows], "o-", label="prefill (time to first token)")
    ax.plot(x, [r["decode_ms"] for r in rows], "s-", label=f"decode {img['new_tokens']} tokens")
    ax.set_xlabel("images in the request")
    ax.set_ylabel("ms")
    ax.set_title("Prefill grows, decode barely moves")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot([c["new_tokens"] for c in cache], [c["with_cache_ms"] for c in cache],
            "o-", label="with KV cache")
    ax.plot([c["new_tokens"] for c in cache], [c["without_cache_ms"] for c in cache],
            "s-", label="no cache (re-read prefix)")
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("total ms")
    ax.set_title("What the KV cache buys")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[2]
    ax.plot([b["batch"] for b in batch], [b["tokens_per_s"] for b in batch], "o-",
            color="tab:green")
    ax.set_xlabel("requests in the batch")
    ax.set_ylabel("aggregate tokens/s")
    ax.set_title("Throughput from batching")
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot([b["batch"] for b in batch], [b["latency_ms"] for b in batch], "s--",
             color="tab:red", alpha=.7)
    ax2.set_ylabel("latency per request (ms)", color="tab:red")

    ax = axes[3]
    names = [b["arm"] for b in budget]
    xs = np.arange(len(names))
    ax.bar(xs, [b["ttft_ms"] for b in budget], .55, color="tab:purple")
    for i, b in enumerate(budget):
        ax.annotate(f"{b['image_tokens']} tok\n{b['kv_kb']:.0f} KB",
                    (i, b["ttft_ms"]), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("prefill ms")
    ax.set_title("Cutting the image-token budget")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "serving.png", dpi=130)
    print("wrote", OUT / "serving.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["images", "cache", "prefix", "batch", "budget", "scale",
                            "plot", "all"])
    p.add_argument("--images", nargs="+", type=int, default=[0, 1, 2, 4])
    p.add_argument("--decode-lengths", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--turn-tokens", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(V.THREADS)
    stages = dict(images=stage_images, cache=stage_cache, prefix=stage_prefix,
                  batch=stage_batch, budget=stage_budget, scale=stage_scale,
                  plot=stage_plot)
    if args.stage == "all":
        for name in ("images", "cache", "prefix", "batch", "budget", "scale", "plot"):
            print(f"[{name}]", flush=True)
            stages[name](args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
