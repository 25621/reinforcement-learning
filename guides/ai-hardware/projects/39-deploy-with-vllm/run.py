"""Project 39 -- Deploy with vLLM (and what to do when vLLM refuses to start).

Four sections:
  A. Try the real vLLM.  It installs, then dies on this GPU -- recorded verbatim.
  B. Verify the from-scratch engine against Hugging Face, token for token.
  C. Serve it: a real HTTP endpoint, and tokens/sec across batch sizes 1..32.
  D. Paged vs contiguous KV allocation -- the memory that PagedAttention saves.

Runs in about 1 minute on 12 CPU threads.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import servelib as S

OUT = S.outdir(__file__)
BATCHES = [1, 2, 4, 8, 16, 32]
PROMPT_LEN, NEW_TOKENS = 64, 24
BLOCK_SIZES = [1, 8, 16, 32, 64, 128]

results = {}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- A. real vLLM
def section_a():
    """What the real vLLM does on this machine (transcript in outputs/)."""
    facts = dict(
        wheel="vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl",
        wheel_MB=312.9,
        installed_ok=True,
        bundled_torch="2.13.0+cu130",
        gpu="NVIDIA GeForce GTX 1070 Ti",
        compute_capability="6.1",
        torch_supported_arch=["sm_70", "sm_75", "sm_80", "sm_86", "sm_90",
                              "sm_100", "sm_120"],
        gpu_error="torch.AcceleratorError: CUDA error: no kernel image is "
                  "available for execution on the device",
        gpu_error_site="vllm/v1/worker/gpu/buffer_utils.py:137 -- the first "
                       "torch.zeros(..., device='cuda') of engine startup",
        cpu_fallback_error="AssertionError: DP adjusted local rank 0 is out of "
                           "bounds for 0 devices (the CUDA wheel has no CPU path)",
    )
    results["vllm_attempt"] = facts
    log("A. vLLM 0.27.1 installs (312.9 MB wheel) and fails at engine start:")
    log(f"   {facts['gpu_error']}")
    return facts


# ------------------------------------------------------------- B. verification
def section_b():
    log("\nB. verifying the from-scratch engine against Hugging Face ...")
    t0 = time.time()
    v = S.verify_against_hf(n_new=8)
    v["seconds"] = time.time() - t0
    results["verification"] = v
    log(f"   max |logit diff| = {v['max_abs_logit_diff']:.2e}   "
        f"relative = {v['rel_logit_diff']:.2e}")
    log(f"   greedy tokens identical: {v['tokens_match']}   -> {v['my_text']!r}")
    return v


# --------------------------------------------------------------- C. serving it
def http_demo(eng, w):
    """A real (tiny) HTTP endpoint, so 'deploy' means something."""
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, HTTPServer

    def generate(prompt, max_new):
        ids = w.tok(prompt, return_tensors=None)["input_ids"]
        seq = S.Sequence(0, ids, max_new=max_new)
        lg = eng.prefill(seq)
        for _ in range(max_new):
            t = S.greedy(lg)
            if t == w.tok.eos_token_id:
                break
            seq.out_ids.append(t)
            lg = eng.decode_step([seq])[0]
        eng.free(seq)
        return w.tok.decode(seq.out_ids)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.perf_counter()
            text = generate(body["prompt"], body.get("max_tokens", 16))
            payload = json.dumps({"text": text,
                                  "latency_s": time.perf_counter() - t0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps({"prompt": "The capital of France is",
                         "max_tokens": 12}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    srv.shutdown()
    out["port"] = port
    return out


def section_c(w):
    log("\nC. serving")
    eng = S.Engine(w, num_blocks=64)
    http = http_demo(eng, w)
    results["http"] = http
    log(f"   POST /generate -> {http['text']!r}  ({http['latency_s']:.2f} s)")

    peaks = S.measure_peaks()
    results["peaks"] = peaks
    log(f"   this machine: {peaks['read_GB_s']:.1f} GB/s streaming reads, "
        f"{peaks['matmul_GFLOP_s']:.0f} GFLOP/s fp32 matmul")

    weight_bytes = w.bytes_of_weights()
    prompt = S.prompt_ids(w.tok, "The history of computing hardware is a story "
                                 "of moving data. ", PROMPT_LEN)
    rows = []
    for B in BATCHES:
        blocks_needed = B * ((PROMPT_LEN + NEW_TOKENS) // 16 + 2)
        eng = S.Engine(w, num_blocks=blocks_needed, gather_stats=True)
        seqs = [S.Sequence(i, prompt, max_new=NEW_TOKENS) for i in range(B)]
        t0 = time.perf_counter()
        lg = eng.forward(seqs, [s.prompt_ids for s in seqs])
        prefill_s = time.perf_counter() - t0
        for s, g in zip(seqs, lg):
            s.out_ids.append(S.greedy(g))
        step_times = []
        for _ in range(NEW_TOKENS - 1):
            t1 = time.perf_counter()
            lg = eng.decode_step(seqs)
            step_times.append(time.perf_counter() - t1)
            for s, g in zip(seqs, lg):
                s.out_ids.append(S.greedy(g))
        step = sorted(step_times)[len(step_times) // 2]      # median step
        row = dict(
            batch=B,
            prefill_s=prefill_s,
            prefill_tok_s=B * PROMPT_LEN / prefill_s,
            prefill_GFLOP_s=2 * weight_bytes / 4 * B * PROMPT_LEN / prefill_s / 1e9,
            decode_GFLOP_s=2 * weight_bytes / 4 * B / step / 1e9,
            step_ms=step * 1e3,
            decode_tok_s=B / step,
            per_request_tok_s=1.0 / step,
            weight_GB_s=weight_bytes / step / 1e9,
            gather_share=eng.gather_time / (prefill_s + sum(step_times)),
            kv_MB=eng.pool.used * eng.pool.bytes_total / eng.pool.num_blocks / 1e6,
        )
        rows.append(row)
        log(f"   batch {B:2d}: prefill {row['prefill_tok_s']:7.1f} tok/s | "
            f"decode {row['decode_tok_s']:6.2f} tok/s | step {row['step_ms']:6.1f} ms "
            f"| weights {row['weight_GB_s']:5.2f} GB/s | gather {row['gather_share']:.1%}")
    results["throughput"] = rows
    results["weight_bytes"] = weight_bytes
    results["bytes_per_token_kv"] = S.KVPool(w, 1, 16).bytes_per_token()
    speedup = rows[-1]["decode_tok_s"] / rows[0]["decode_tok_s"]
    log(f"   batch {BATCHES[-1]} moves {speedup:.1f}x more tokens/s than batch 1, "
        f"and each request is only {rows[0]['step_ms'] / rows[-1]['step_ms']:.2f}x "
        f"as fast per token")
    log(f"   prefill peaks at {max(r['prefill_GFLOP_s'] for r in rows):.0f} GFLOP/s "
        f"({100 * max(r['prefill_GFLOP_s'] for r in rows) / peaks['matmul_GFLOP_s']:.0f}% "
        f"of this machine's matmul roof); decode at batch 1 reaches "
        f"{rows[0]['decode_GFLOP_s']:.1f} GFLOP/s while re-reading weights at "
        f"{rows[0]['weight_GB_s']:.1f} GB/s "
        f"({100 * rows[0]['weight_GB_s'] / peaks['read_GB_s']:.0f}% of the memory roof)")
    return rows


# ------------------------------------------------------------------ D. paging
def section_d(w):
    log("\nD. paged vs contiguous KV allocation")
    bpt = S.KVPool(w, 1, 16).bytes_per_token()
    max_len = 2048                       # what a contiguous server must reserve
    torch.manual_seed(0)
    # A realistic serving mix: most requests are short, a few are long.
    lengths = (torch.distributions.LogNormal(4.6, 0.9)
               .sample((256,)).clamp(8, max_len).int().tolist())
    n = len(lengths)
    used = sum(lengths)

    variants = {}
    for bs in BLOCK_SIZES:
        alloc = sum(-(-L // bs) * bs for L in lengths)
        variants[bs] = dict(block_size=bs, tokens_allocated=alloc,
                            waste_pct=100 * (alloc - used) / used,
                            MB=alloc * bpt / 1e6)
    contiguous = dict(tokens_allocated=n * max_len,
                      waste_pct=100 * (n * max_len - used) / used,
                      MB=n * max_len * bpt / 1e6)
    results["fragmentation"] = dict(
        n_requests=n, tokens_used=used, max_len=max_len,
        bytes_per_token=bpt, contiguous=contiguous, paged=variants,
        mean_len=used / n)
    log(f"   {n} requests, {used} tokens of real KV ({used * bpt / 1e6:.0f} MB)")
    log(f"   contiguous (reserve {max_len} per slot): {contiguous['MB']:.0f} MB "
        f"= {contiguous['waste_pct']:.0f}% waste")
    for bs, d in variants.items():
        log(f"   paged, block {bs:3d}: {d['MB']:.0f} MB = {d['waste_pct']:.1f}% waste")
    ratio = contiguous["MB"] / variants[16]["MB"]
    log(f"   -> block 16 fits {ratio:.1f}x more requests in the same memory")

    # How many concurrent requests fit in a fixed KV budget?
    for budget_gb in (1.0, 8.0):
        seats_c = int(budget_gb * 1e9 / (max_len * bpt))
        seats_p = int(budget_gb * 1e9 / (results["fragmentation"]["mean_len"] * bpt))
        results.setdefault("seats", {})[str(budget_gb)] = dict(
            contiguous=seats_c, paged=seats_p)
        log(f"   {budget_gb:.0f} GB of KV: {seats_c} contiguous slots vs "
            f"~{seats_p} paged requests")

    # Measured cost of the block-table gather at several block sizes.
    log("   measuring gather cost vs block size (decode, batch 16, 512 tokens) ...")
    gather = {}
    for bs in [8, 16, 64, 128]:
        eng = S.Engine(w, num_blocks=16 * (512 // bs + 2), block_size=bs,
                       gather_stats=True)
        seqs = [S.Sequence(i, list(range(512)), max_new=1) for i in range(16)]
        for s in seqs:                    # fake a filled cache without prefilling
            eng.ensure_blocks(s, 512)
            s.length = 512
            s.out_ids = [1]
        eng.decode_step(seqs)             # warm-up
        eng.gather_time, eng.gather_bytes = 0.0, 0
        t0 = time.perf_counter()
        for _ in range(3):
            eng.decode_step(seqs)
        total = time.perf_counter() - t0
        gather[bs] = dict(gather_ms=eng.gather_time / 3 * 1e3,
                          step_ms=total / 3 * 1e3,
                          gathered_MB=eng.gather_bytes / 3 / 1e6,
                          share=eng.gather_time / total)
        log(f"     block {bs:3d}: {gather[bs]['gather_ms']:6.1f} ms of "
            f"{gather[bs]['step_ms']:6.1f} ms ({gather[bs]['share']:.0%}), "
            f"{gather[bs]['gathered_MB']:.0f} MB/step")
    results["gather_cost"] = gather


# -------------------------------------------------------------------- figures
def make_plots(res):
    rows = res["throughput"]
    B = [r["batch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    ax[0].plot(B, [r["decode_tok_s"] for r in rows], "o-", color="#1f77b4",
               label="server throughput")
    ax[0].plot(B, [r["per_request_tok_s"] for r in rows], "s-", color="#d62728",
               label="per request")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("decode tokens / s")
    ax[0].set_title("One model, two throughputs")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    ax[1].plot(B, [r["weight_GB_s"] for r in rows], "o-", color="#2ca02c")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("batch size")
    ax[1].set_ylabel("GB/s of weights re-read")
    ax[1].set_title("Decode reads every weight, every step")
    ax[1].grid(alpha=.3)

    frag = res["fragmentation"]
    names = ["contiguous\n(reserve %d)" % frag["max_len"]] + \
            ["paged\nblock %s" % b for b in frag["paged"]]
    mb = [frag["contiguous"]["MB"]] + [d["MB"] for d in frag["paged"].values()]
    colors = ["#d62728"] + ["#1f77b4"] * len(frag["paged"])
    ax[2].bar(names, mb, color=colors)
    ax[2].axhline(frag["tokens_used"] * frag["bytes_per_token"] / 1e6, ls="--",
                  color="k", lw=1, label="KV actually used")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("KV memory (MB)")
    ax[2].set_title("Where the KV cache goes")
    ax[2].tick_params(axis="x", labelsize=7)
    ax[2].legend()
    fig.savefig(f"{OUT}/serving.png", dpi=130)
    log(f"   wrote {OUT}/serving.png")


def main():
    S.setup()
    t0 = time.time()
    section_a()
    section_b()
    w = S.Weights(S.SMALL)
    results["model"] = dict(name=w.name, layers=w.n_layer, d_model=w.d_model,
                            heads=w.n_head, kv_heads=w.n_kv,
                            params_GB=w.bytes_of_weights() / 1e9)
    section_c(w)
    section_d(w)
    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("batch,prefill_tok_s,decode_tok_s,per_request_tok_s,step_ms,weight_GB_s,gather_share\n")
        for r in results["throughput"]:
            f.write(f"{r['batch']},{r['prefill_tok_s']:.1f},{r['decode_tok_s']:.3f},"
                    f"{r['per_request_tok_s']:.3f},{r['step_ms']:.1f},"
                    f"{r['weight_GB_s']:.2f},{r['gather_share']:.4f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
