#!/usr/bin/env python3
"""Project 72 — compile a 3B model for a device and measure what it does there.

The "device" here is this machine's CPU with no usable GPU, which is exactly the
situation a laptop or a phone is in: a few cores, memory shared with everything
else, and no data-centre accelerator.  The runtime is real `llama.cpp`, the
model is a real Qwen2.5-3B-Instruct converted to real [GGUF](/shared/glossary/#gguf)
files, and every number below comes from `llama-bench` or `llama-perplexity`.

  A. The build: convert to GGUF, quantise, and look at what came out.
  B. Does it fit?  Memory against a device budget — the question that decides
     everything else on an edge device.
  C. Speed: prompt processing and token generation for each quantisation, and
     the same model under PyTorch for comparison.
  D. Threads: the one knob an on-device runtime really has.
  E. Quality: perplexity for each quantisation, so the speed has a price tag.
  F. Context: how much KV cache is left after the weights.

  python3 run.py             # ~8 minutes once the build cache is warm
  python3 run.py --prepare   # only build llama.cpp + convert + quantise
  python3 run.py --plot      # redraw from outputs/findings.json

The first run also clones and builds llama.cpp (~4 minutes) and converts the
model (~3 minutes); both are cached in `vendor/` and skipped afterwards.
Set LLAMA_WORKDIR to keep those artefacts somewhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("LLAMA_WORKDIR", os.path.join(HERE, "vendor"))
OUT = os.path.join(HERE, "outputs")
FINDINGS = os.path.join(OUT, "findings.json")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QUANTS = ["Q8_0", "Q4_K_M", "Q4_0"]
BENCH_THREADS = 6
THREAD_SWEEP = [1, 2, 4, 6, 8, 12]
DEVICE_BUDGET_GB = 4.0          # what a mid-range phone will let one app have
PPL_CHUNKS = 4


def sh(cmd, **kw):
    print("   $", " ".join(cmd[:3]), "...", flush=True)
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------
# A: build the artefacts
# ---------------------------------------------------------------------------

def ensure_llama_cpp():
    repo = os.path.join(WORK, "llama.cpp")
    bins = os.path.join(repo, "build", "bin")
    if os.path.exists(os.path.join(bins, "llama-bench")):
        return repo, bins, 0.0
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    if not os.path.exists(repo):
        sh(["git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git", repo])
    sh(["cmake", "-S", repo, "-B", os.path.join(repo, "build"),
        "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=ON"])
    sh(["cmake", "--build", os.path.join(repo, "build"), "-j", "10",
        "--target", "llama-cli", "llama-bench", "llama-quantize",
        "llama-perplexity"])
    return repo, bins, time.time() - t0


def ensure_gguf(repo, bins):
    from huggingface_hub import snapshot_download
    gdir = os.path.join(WORK, "gguf")
    os.makedirs(gdir, exist_ok=True)
    f16 = os.path.join(gdir, "qwen2.5-3b-f16.gguf")
    times = {}
    if not os.path.exists(f16):
        src = snapshot_download(MODEL_ID,
                                allow_patterns=["*.json", "*.txt",
                                                "*.safetensors"])
        t0 = time.time()
        sh([sys.executable, os.path.join(repo, "convert_hf_to_gguf.py"), src,
            "--outtype", "f16", "--outfile", f16])
        times["convert_s"] = round(time.time() - t0, 1)
    paths = {"f16": f16}
    for q in QUANTS:
        p = os.path.join(gdir, f"qwen2.5-3b-{q}.gguf")
        if not os.path.exists(p):
            t0 = time.time()
            sh([os.path.join(bins, "llama-quantize"), f16, p, q, "8"])
            times[f"quantize_{q}_s"] = round(time.time() - t0, 1)
        paths[q] = p
    return paths, times


# ---------------------------------------------------------------------------
# C/D: llama-bench
# ---------------------------------------------------------------------------

def bench(bins, path, threads=BENCH_THREADS, pp=512, tg=64, reps=2):
    out = subprocess.run(
        [os.path.join(bins, "llama-bench"), "-m", path, "-p", str(pp),
         "-n", str(tg), "-t", str(threads), "-r", str(reps), "-o", "json"],
        check=True, capture_output=True, text=True).stdout
    rows = json.loads(out)
    res = {}
    for r in rows:
        key = "pp" if r["n_prompt"] else "tg"
        res[key] = dict(tok_s=round(r["avg_ts"], 2),
                        stddev=round(r.get("stddev_ts", 0.0), 2))
    return res


def model_bytes(path):
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# C: the PyTorch comparison, same weights, same machine
# ---------------------------------------------------------------------------

def torch_baseline(n_new=32):
    import resource
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(BENCH_THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    # float32 would be 12.3 GB and does not fit next to everything else here,
    # which is itself the point of the section: bfloat16 is the smallest thing
    # stock PyTorch will load on a CPU.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 dtype=torch.bfloat16).eval()
    load_s = time.time() - t0
    ids = tok("Explain in one paragraph why edge inference is useful.",
              return_tensors="pt").input_ids
    with torch.inference_mode():
        # One warm-up pass first.  The very first forward after loading has to
        # touch all 6 GB of weights, so timing it measures the page cache, not
        # the model.
        model(ids, use_cache=True, logits_to_keep=1)
        t0 = time.time()
        out = model(ids, use_cache=True, logits_to_keep=1)
        pp_s = time.time() - t0
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        t0 = time.time()
        for _ in range(n_new):
            o = model(nxt, past_key_values=past, use_cache=True,
                      logits_to_keep=1)
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        tg_s = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    del model
    return dict(load_s=round(load_s, 1),
                prefill_tok_s=round(ids.shape[1] / pp_s, 2),
                decode_tok_s=round(n_new / tg_s, 2),
                peak_rss_bytes=rss, dtype="bfloat16")


# ---------------------------------------------------------------------------
# E: perplexity
# ---------------------------------------------------------------------------

def ppl_file():
    """A small held-out text file for llama-perplexity."""
    path = os.path.join(WORK, "wiki.txt")
    if not os.path.exists(path):
        sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                        "51-needle-in-a-haystack"))
        import ctxlib
        text = ctxlib.wikitext(n_chars=120_000)
        with open(path, "w") as f:
            f.write(text)
    return path


def perplexity(bins, path, txt, chunks=PPL_CHUNKS, ctx=512, threads=12):
    r = subprocess.run(
        [os.path.join(bins, "llama-perplexity"), "-m", path, "-f", txt,
         "-c", str(ctx), "--chunks", str(chunks), "-t", str(threads)],
        capture_output=True, text=True)
    m = re.findall(r"Final estimate: PPL = ([0-9.]+)", r.stdout + r.stderr)
    if not m:
        m = re.findall(r"\[\d+\]([0-9.]+),", r.stdout + r.stderr)
    return float(m[-1]) if m else float("nan")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()
    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    print("A. build ...", flush=True)
    repo, bins, build_s = ensure_llama_cpp()
    paths, times = ensure_gguf(repo, bins)
    sizes = {k: model_bytes(p) for k, p in paths.items()}
    A = dict(build_s=round(build_s, 1), times=times,
             sizes={k: v for k, v in sizes.items()},
             bits_per_weight={k: round(v * 8 / 3.09e9, 2)
                              for k, v in sizes.items()},
             shrink={k: round(sizes["f16"] / v, 2) for k, v in sizes.items()})
    print("   " + "  ".join(f"{k}:{v / 1e9:.2f}GB" for k, v in sizes.items()),
          flush=True)
    if args.prepare:
        return

    print("C. llama-bench ...", flush=True)
    C = {}
    for k, p in paths.items():
        C[k] = bench(bins, p)
        C[k]["bytes"] = sizes[k]
        print(f"   {k:<7} prefill {C[k]['pp']['tok_s']:8.1f} tok/s   decode "
              f"{C[k]['tg']['tok_s']:6.2f} tok/s", flush=True)

    print("D. thread sweep on Q4_K_M ...", flush=True)
    D = {}
    for t in THREAD_SWEEP:
        r = bench(bins, paths["Q4_K_M"], threads=t, pp=256, tg=32, reps=1)
        D[str(t)] = r
        print(f"   t={t:<3} prefill {r['pp']['tok_s']:8.1f}  decode "
              f"{r['tg']['tok_s']:6.2f}", flush=True)

    print("E. perplexity ...", flush=True)
    txt = ppl_file()
    E = {}
    for k in ("f16", "Q8_0", "Q4_K_M", "Q4_0"):
        t0 = time.time()
        E[k] = dict(ppl=round(perplexity(bins, paths[k], txt), 4),
                    seconds=round(time.time() - t0, 1))
        print(f"   {k:<7} ppl {E[k]['ppl']:8.4f}  ({E[k]['seconds']}s)",
              flush=True)
    for k in E:
        E[k]["x"] = round(E[k]["ppl"] / E["f16"]["ppl"], 4)

    Bt = None
    if not args.skip_torch:
        print("C2. the same model under PyTorch ...", flush=True)
        Bt = torch_baseline()
        print(f"   bf16 decode {Bt['decode_tok_s']} tok/s, peak RSS "
              f"{Bt['peak_rss_bytes'] / 1e9:.2f} GB", flush=True)

    # --- B: does it fit? ---------------------------------------------------
    budget = DEVICE_BUDGET_GB * 1e9
    B = dict(budget_gb=DEVICE_BUDGET_GB, rows={})
    fp32_bytes = 3.09e9 * 4
    B["rows"]["PyTorch fp32"] = dict(bytes=fp32_bytes,
                                     fits=bool(fp32_bytes < budget))
    B["rows"]["PyTorch bf16"] = dict(bytes=3.09e9 * 2,
                                     fits=bool(3.09e9 * 2 < budget))
    for k in paths:
        B["rows"][f"GGUF {k}"] = dict(bytes=sizes[k], fits=bool(sizes[k] < budget))

    # --- F: what context is left ------------------------------------------
    # Qwen2.5-3B: 36 layers, 2 KV heads (GQA), head_dim 128 -> per token per
    # layer: 2 (K and V) * 2 heads * 128 * 2 bytes (f16)
    kv_per_token = 36 * 2 * 2 * 128 * 2
    F_ = dict(kv_bytes_per_token=kv_per_token,
              rows={k: dict(
                  weights=sizes[k],
                  ctx_tokens=int(max(0, budget - sizes[k]) / kv_per_token))
                  for k in paths})

    F = dict(A=A, B=B, C=C, D=D, E=E, torch=Bt, F=F_,
             model=MODEL_ID, threads=BENCH_THREADS)
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- summary --------------------------------------------------")
    for k in paths:
        print(f"{k:<7} {sizes[k] / 1e9:5.2f} GB  {A['bits_per_weight'][k]:5.2f} "
              f"bits/weight  decode {C[k]['tg']['tok_s']:6.2f} tok/s  "
              f"ppl x{E[k]['x']:.4f}  fits in {DEVICE_BUDGET_GB} GB: "
              f"{B['rows'][f'GGUF {k}']['fits']}  context left "
              f"{F_['rows'][k]['ctx_tokens']:,} tokens")
    if Bt:
        print(f"PyTorch bf16 decode {Bt['decode_tok_s']} tok/s  "
              f"({C['Q4_K_M']['tg']['tok_s'] / Bt['decode_tok_s']:.1f}x slower "
              f"than Q4_K_M)")
    plot(F)


def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C, D, E = F["A"], F["B"], F["C"], F["D"], F["E"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("On-device build: Qwen2.5-3B as GGUF, on a CPU with no GPU",
                 fontsize=13)

    p = ax[0]
    rows = list(B["rows"].items())
    vals = [v["bytes"] / 1e9 for _, v in rows]
    cols = ["#4c9f70" if v["fits"] else "#c0504d" for _, v in rows]
    p.barh(range(len(rows)), vals, color=cols)
    p.axvline(B["budget_gb"], color="k", ls="--", lw=1,
              label=f"device budget {B['budget_gb']} GB")
    p.set_yticks(range(len(rows)))
    p.set_yticklabels([k for k, _ in rows], fontsize=8)
    p.set_xlabel("weights (GB)")
    p.set_title("B. Does it fit on the device?")
    p.legend(fontsize=8)

    p = ax[1]
    ks = list(C.keys())
    tg = [C[k]["tg"]["tok_s"] for k in ks]
    x = [E[k]["x"] for k in ks]
    p.scatter(x, tg, s=80, color="#4a6fa5")
    for k, xx, yy in zip(ks, x, tg):
        p.annotate(k, (xx, yy), textcoords="offset points", xytext=(6, 4),
                   fontsize=9)
    if F.get("torch"):
        p.axhline(F["torch"]["decode_tok_s"], color="#c0504d", ls="--", lw=1,
                  label=f"PyTorch bf16: {F['torch']['decode_tok_s']} tok/s")
        p.legend(fontsize=8)
    p.set_xlabel("perplexity relative to f16")
    p.set_ylabel("decode tokens/second")
    p.set_title("C/E. Speed against quality")
    p.grid(alpha=0.3)

    p = ax[2]
    ts = [int(t) for t in D]
    p.plot(ts, [D[str(t)]["tg"]["tok_s"] for t in ts], "o-", color="#4a6fa5",
           label="decode")
    p2 = p.twinx()
    p2.plot(ts, [D[str(t)]["pp"]["tok_s"] for t in ts], "s--", color="#e0a458",
            label="prefill")
    p.set_xlabel("threads")
    p.set_ylabel("decode tok/s")
    p2.set_ylabel("prefill tok/s")
    p.set_title("D. The one knob on a device")
    p.legend(fontsize=8, loc="upper left")
    p2.legend(fontsize=8, loc="lower right")
    p.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "on_device.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    main()
