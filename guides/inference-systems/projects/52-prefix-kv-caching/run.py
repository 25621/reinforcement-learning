"""Project 52 -- Prefix KV caching for retrieval-augmented serving.

Pre-compute one KV cache per retrieved document, then answer queries against
them. Four sections:

  A. Build the cache. 256 documents, measured: how long the pre-computation
     takes, and how many bytes it costs per byte of document text.
  B. Cold vs. warm TTFT. Same question, same document, cache absent vs.
     present -- plus a bit-exactness check, because a prefix cache is
     supposed to be a pure speed-up and nothing else.
  C. The trap that gives "prefix cache" its name: a cached document is only
     valid at the *position* it was computed at. Move it and the answer
     falls apart. Measured two ways -- what the model outputs, and how far
     the logits drift.
  D. What it is worth in a real system: Zipf-distributed document
     popularity, an LRU cache of varying capacity, and the measured cold and
     warm costs from B turned into a mean-TTFT curve.

Usage:
    python3 run.py            # ~6 minutes
    python3 run.py --plot     # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "51-needle-in-a-haystack"))
import ctxlib  # noqa: E402

from transformers.cache_utils import DynamicCache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

N_DOCS = 256          # documents whose KV we really build and hold
DOC_TOKENS = 128      # tokens per document
N_QUERIES = 48        # queries timed in section B
CATALOGUE = 1000      # the catalogue size the guide asks about (arithmetic)


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------


def cache_to_cpu(past, dtype=torch.float16):
    """Freeze a live cache into plain tensors we can keep in a dict.

    A `DynamicCache` is a mutable object that the next forward pass will
    append to. Storing it directly would mean the first query that used a
    document also *corrupted* it for every later query, so the tensors are
    copied out. float16 halves the store; section B's exactness check uses a
    float32 copy so that "warm equals cold" is a statement about caching and
    not about rounding.
    """
    return [(l.keys.detach().to(dtype).clone(),
             l.values.detach().to(dtype).clone()) for l in past.layers]


def store_bytes(store) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in store)


# ---------------------------------------------------------------------------
# Prompt layout
# ---------------------------------------------------------------------------
#
# Everything hangs on the layout being SPLITTABLE at a fixed token boundary:
#
#     [ chat header ][ document ][ question + chat footer ]
#     \______________________/   \_________________________/
#        cached once, reused        prefilled on every query
#
# The chat header has to be inside the cached part. If it were not, the
# cached document's tokens would start at position 0 while the real prompt
# puts them at position len(header) -- and section C is exactly a
# demonstration of how badly that goes.


def layout(tok, doc_text: str, question: str):
    head = tok("<|im_start|>user\n", add_special_tokens=False).input_ids
    doc = tok(doc_text, add_special_tokens=False).input_ids[:DOC_TOKENS]
    tail = tok("\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n",
               add_special_tokens=False).input_ids
    return head, doc, tail


QUESTION = "In one short sentence, what is the passage above about?"


def make_docs(tok, filler, n: int):
    """Cut the wikitext stream into n non-overlapping documents."""
    docs = []
    for i in range(n):
        chunk = filler[i * DOC_TOKENS:(i + 1) * DOC_TOKENS]
        docs.append(tok.decode(chunk))
    return docs


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@torch.inference_mode()
def build_store(model, tok, docs):
    """Section A: prefill each document once and keep its KV."""
    store, total_tok, total_txt = {}, 0, 0
    t0 = time.perf_counter()
    for i, d in enumerate(docs):
        head, doc, _ = layout(tok, d, QUESTION)
        ids = torch.tensor([head + doc])
        out = model(ids, use_cache=True, logits_to_keep=1)
        store[i] = cache_to_cpu(out.past_key_values)
        total_tok += ids.shape[1]
        total_txt += len(("<|im_start|>user\n" + d).encode())
        del out
        if (i + 1) % 64 == 0:
            print(f"  cached {i+1}/{len(docs)}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    return store, time.perf_counter() - t0, total_tok, total_txt


@torch.inference_mode()
def cold_vs_warm(model, tok, docs, n_queries):
    """Section B: the same answer, computed twice, timed twice."""
    rows = []
    for i in range(n_queries):
        head, doc, tail = layout(tok, docs[i], QUESTION)

        # cold: nothing cached, prefill head+doc+tail
        ids_full = torch.tensor([head + doc + tail])
        t0 = time.perf_counter()
        out_c = model(ids_full, use_cache=True, logits_to_keep=1)
        cold_s = time.perf_counter() - t0
        cold_logits = out_c.logits[:, -1, :].clone()

        # warm: head+doc already in cache, prefill only tail
        t0 = time.perf_counter()
        pre = model(torch.tensor([head + doc]), use_cache=True,
                    logits_to_keep=1)
        build_s = time.perf_counter() - t0
        cache = DynamicCache(ddp_cache_data=[
            (l.keys.clone(), l.values.clone()) for l in pre.past_key_values.layers])
        del pre
        t0 = time.perf_counter()
        out_w = model(torch.tensor([tail]), past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
        warm_s = time.perf_counter() - t0
        warm_logits = out_w.logits[:, -1, :].clone()

        rows.append({
            "doc": i,
            "prompt_tokens": ids_full.shape[1],
            "cached_tokens": len(head) + len(doc),
            "fresh_tokens": len(tail),
            "cold_s": cold_s, "warm_s": warm_s, "build_s": build_s,
            "max_logit_diff": float((cold_logits - warm_logits).abs().max()),
            "same_argmax": bool(cold_logits.argmax() == warm_logits.argmax()),
        })
        del out_c, out_w, cache
    return rows


@torch.inference_mode()
def position_trap(model, tok, docs, n=8):
    """Section C: reuse a cache at the wrong position on purpose.

    Doc B's cache is built with B sitting at positions [len(head), ...].
    Then we serve a prompt whose real layout is head + docA + docB + tail, so
    docB genuinely belongs at a much later position. Splicing the stale cache
    in is the mistake a naive "cache every document" design makes.
    """
    rows = []
    for i in range(n):
        a_text, b_text = docs[2 * i], docs[2 * i + 1]
        head, doc_a, _ = layout(tok, a_text, QUESTION)
        _, doc_b, tail = layout(tok, b_text, QUESTION)
        q = ("\n\nQuote the first five words of the SECOND passage."
             "<|im_end|>\n<|im_start|>assistant\n")
        tail = tok(q, add_special_tokens=False).input_ids

        # truth: one honest prefill over the whole thing
        ids = torch.tensor([head + doc_a + doc_b + tail])
        ref = model(ids, use_cache=True, logits_to_keep=1)
        ref_logits = ref.logits[:, -1, :].clone()
        ref_txt = greedy_from(model, tok, ref.past_key_values,
                              ref_logits, 12)
        del ref

        # correct reuse: docA is the PREFIX, so its cache is valid
        pre_a = model(torch.tensor([head + doc_a]), use_cache=True,
                      logits_to_keep=1)
        ca = DynamicCache(ddp_cache_data=[(l.keys.clone(), l.values.clone())
                                          for l in pre_a.past_key_values.layers])
        del pre_a
        ok = model(torch.tensor([doc_b + tail]), past_key_values=ca,
                   use_cache=True, logits_to_keep=1)
        ok_logits = ok.logits[:, -1, :].clone()
        ok_txt = greedy_from(model, tok, ok.past_key_values, ok_logits, 12)
        del ok, ca

        # wrong reuse: docB's cache was computed at position len(head), but
        # here docB really starts at len(head)+len(doc_a)
        pre_b = model(torch.tensor([head + doc_b]), use_cache=True,
                      logits_to_keep=1)
        stale = cache_to_cpu(pre_b.past_key_values, dtype=torch.float32)
        del pre_b
        # splice: keep head from the honest run, then bolt on docB's stale
        # rows. Shapes line up; positions do not.
        pre_h = model(torch.tensor([head + doc_a]), use_cache=True,
                      logits_to_keep=1)
        spliced = []
        for li, l in enumerate(pre_h.past_key_values.layers):
            k = torch.cat([l.keys.clone(), stale[li][0][:, :, len(head):]], 2)
            v = torch.cat([l.values.clone(), stale[li][1][:, :, len(head):]], 2)
            spliced.append((k, v))
        del pre_h
        cb = DynamicCache(ddp_cache_data=spliced)
        bad = model(torch.tensor([tail]), past_key_values=cb, use_cache=True,
                    logits_to_keep=1)
        bad_logits = bad.logits[:, -1, :].clone()
        bad_txt = greedy_from(model, tok, bad.past_key_values, bad_logits, 12)
        del bad, cb

        rows.append({
            "ok_max_diff": float((ref_logits - ok_logits).abs().max()),
            "bad_max_diff": float((ref_logits - bad_logits).abs().max()),
            "ok_same_argmax": bool(ref_logits.argmax() == ok_logits.argmax()),
            "bad_same_argmax": bool(ref_logits.argmax() == bad_logits.argmax()),
            "ref_text": ref_txt, "ok_text": ok_txt, "bad_text": bad_txt,
            "ref_abs_mean": float(ref_logits.abs().mean()),
        })
        print(f"  pair {i}: ok_diff={rows[-1]['ok_max_diff']:.2e} "
              f"bad_diff={rows[-1]['bad_max_diff']:.2f}", flush=True)
    return rows


@torch.inference_mode()
def greedy_from(model, tok, past, logits, n):
    nxt = logits.argmax(-1, keepdim=True)
    ids = [int(nxt)]
    for _ in range(n - 1):
        out = model(nxt, past_key_values=past, use_cache=True,
                    logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids.append(int(nxt))
    return tok.decode(ids, skip_special_tokens=True).strip()


def lru_economics(cold_s, warm_s, kv_mb_per_doc, n_docs=CATALOGUE,
                  n_requests=20000, zipf_s=1.0, seed=3):
    """Section D: measured costs, simulated traffic.

    Document popularity in a retrieval system is not uniform -- a handful of
    pages answer most questions. Zipf is the standard stand-in for that: the
    r-th most popular document is requested about 1/r^s as often as the most
    popular one. (Named after George Zipf, who noticed the same 1/rank shape
    in how often English words are used.)
    """
    import random
    rnd = random.Random(seed)
    w = [1.0 / (r ** zipf_s) for r in range(1, n_docs + 1)]
    tot = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)

    def draw():
        u = rnd.random()
        lo, hi = 0, n_docs - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    stream = [draw() for _ in range(n_requests)]
    rows = []
    for cap in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]:
        lru, hits = OrderedDict(), 0
        for d in stream:
            if d in lru:
                hits += 1
                lru.move_to_end(d)
            else:
                lru[d] = 1
                if len(lru) > cap:
                    lru.popitem(last=False)
        hr = hits / len(stream)
        rows.append({"capacity": cap, "hit_rate": hr,
                     "mean_ttft_s": hr * warm_s + (1 - hr) * cold_s,
                     "store_gb": cap * kv_mb_per_doc / 1000})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def measure():
    tok, model = ctxlib.load()
    filler = ctxlib.filler_tokens(tok, 400_000)
    docs = make_docs(tok, filler, N_DOCS + 2 * 8)
    kv_per_tok = ctxlib.kv_bytes_per_token(model.config, 4)

    print("== A. building the document cache ==")
    store, build_s, n_tok, n_txt = build_store(model, tok, docs[:N_DOCS])
    nbytes = sum(store_bytes(s) for s in store.values())
    per_doc_mb = nbytes / len(store) / 1e6
    print(f"  {len(store)} docs, {n_tok} tokens, {build_s:.1f}s, "
          f"{nbytes/1e6:.0f} MB fp16 ({per_doc_mb:.2f} MB/doc)")
    print(f"  document text: {n_txt/1e3:.0f} kB -> KV {nbytes/1e6:.0f} MB "
          f"= {nbytes/n_txt:.0f}x")
    del store

    print("\n== B. cold vs warm ==")
    rows = cold_vs_warm(model, tok, docs, N_QUERIES)
    cold = sorted(r["cold_s"] for r in rows)
    warm = sorted(r["warm_s"] for r in rows)
    mid = len(cold) // 2
    print(f"  cold p50 {cold[mid]*1000:.0f} ms   warm p50 {warm[mid]*1000:.0f} ms"
          f"   speedup {cold[mid]/warm[mid]:.2f}x")
    print(f"  identical argmax on {sum(r['same_argmax'] for r in rows)}"
          f"/{len(rows)}; max logit drift "
          f"{max(r['max_logit_diff'] for r in rows):.2e}")

    print("\n== C. the position trap ==")
    trap = position_trap(model, tok, docs[N_DOCS:], n=8)

    print("\n== D. economics ==")
    econ = lru_economics(cold[mid], warm[mid], per_doc_mb * 2)  # fp32 store
    for r in econ:
        print(f"  cap {r['capacity']:>4}  hit {r['hit_rate']*100:5.1f}%  "
              f"mean TTFT {r['mean_ttft_s']*1000:6.0f} ms  "
              f"store {r['store_gb']:.2f} GB")

    res = {
        "model": ctxlib.MODEL_ID,
        "n_docs": N_DOCS, "doc_tokens": DOC_TOKENS,
        "kv_bytes_per_token_fp32": kv_per_tok,
        "build_s": build_s, "build_tokens": n_tok,
        "store_bytes_fp16": nbytes, "text_bytes": n_txt,
        "bytes_ratio": nbytes / n_txt,
        "per_doc_mb_fp16": per_doc_mb,
        "catalogue": CATALOGUE,
        "catalogue_gb_fp32": CATALOGUE * per_doc_mb * 2 / 1000,
        "queries": rows, "trap": trap, "econ": econ,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    # A: TTFT distributions
    a = ax[0][0]
    cold = np.array([r["cold_s"] for r in res["queries"]]) * 1000
    warm = np.array([r["warm_s"] for r in res["queries"]]) * 1000
    a.bar([0, 1], [cold.mean(), warm.mean()],
          yerr=[cold.std(), warm.std()], capsize=6,
          color=["#c0392b", "#27ae60"], width=.55)
    a.set_xticks([0, 1], [f"cold\n{res['queries'][0]['prompt_tokens']} tok "
                          f"prefilled",
                          f"warm\n{res['queries'][0]['fresh_tokens']} tok "
                          f"prefilled"])
    a.set_ylabel("time to first token (ms)")
    a.set_title(f"A. Cold vs warm TTFT — "
                f"{cold.mean()/warm.mean():.1f}x on {len(cold)} queries")
    for i, v in enumerate([cold.mean(), warm.mean()]):
        a.text(i, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=10)

    # B: bytes
    a = ax[0][1]
    labels = ["document text", "KV cache (fp16)", "KV cache (fp32)"]
    vals = [res["text_bytes"] / 1e6, res["store_bytes_fp16"] / 1e6,
            res["store_bytes_fp16"] * 2 / 1e6]
    a.bar(labels, vals, color=["#7f8c8d", "#2980b9", "#8e44ad"])
    a.set_yscale("log")
    a.set_ylabel(f"MB for {res['n_docs']} documents")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f} MB", ha="center", va="bottom", fontsize=9)
    a.set_title(f"B. What you store — KV is {res['bytes_ratio']:.0f}x "
                f"the text")

    # C: the position trap
    a = ax[1][0]
    ok = [t["ok_max_diff"] for t in res["trap"]]
    bad = [t["bad_max_diff"] for t in res["trap"]]
    ref = np.mean([t["ref_abs_mean"] for t in res["trap"]])
    x = np.arange(len(ok))
    a.bar(x - .2, ok, .4, label="reused as a PREFIX (correct)",
          color="#27ae60")
    a.bar(x + .2, bad, .4, label="reused mid-prompt (wrong position)",
          color="#c0392b")
    a.axhline(ref, ls="--", color="#34495e",
              label=f"mean |logit| = {ref:.2f}")
    a.set_yscale("log")
    a.set_xlabel("document pair")
    a.set_ylabel("max |logit − honest prefill|")
    a.legend(fontsize=8)
    n_ok = sum(t["ok_same_argmax"] for t in res["trap"])
    n_bad = sum(t["bad_same_argmax"] for t in res["trap"])
    a.set_title(f"C. Position matters — same argmax {n_ok}/{len(ok)} "
                f"correct vs {n_bad}/{len(bad)} spliced")

    # D: economics
    a = ax[1][1]
    caps = [r["capacity"] for r in res["econ"]]
    a.plot(caps, [r["mean_ttft_s"] * 1000 for r in res["econ"]], "o-",
           color="#c0392b", label="mean TTFT (ms)")
    a.set_xscale("log")
    a.set_xlabel(f"documents kept warm (of {res['catalogue']})")
    a.set_ylabel("mean TTFT (ms)")
    a2 = a.twinx()
    a2.plot(caps, [r["hit_rate"] * 100 for r in res["econ"]], "s--",
            color="#2980b9")
    a2.set_ylabel("cache hit rate (%)", color="#2980b9")
    a.set_title("D. Zipf traffic + LRU: most of the win is in the first "
                "few dozen docs")
    a.grid(alpha=.3)

    fig.suptitle("Prefix KV caching: a big win, an exact one, and only ever "
                 "for a prefix", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "prefix_cache.png"), dpi=120)
    print("wrote", os.path.join(OUT, "prefix_cache.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure())
        print(f"total {time.time()-t0:.0f}s")
