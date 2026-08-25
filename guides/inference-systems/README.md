# Inference Systems: From Beginner to Advanced

A comprehensive guide to **serving** trained models in production — the layer between a checkpoint sitting on disk and a user typing into a chat box and seeing tokens stream back. The goal is to take you from "I have called an API" to "I can stand up a production inference cluster, explain every hop a token takes from prompt to client, debug a P99 latency regression, and make a defensible cost-per-million-tokens commitment to a CFO."

> **An honest framing.** Training is a fixed cost; serving is a forever cost. A modest production system serves more tokens in its first month than the model ever saw during fine-tuning. The math that makes a serving system economically viable is *almost entirely* about how cleverly you handle memory bandwidth, batch heterogeneity, and the KV cache. The model architecture you can copy from the [LLM guide](../llm/). The silicon underneath you can read about in the [AI Hardware guide](../ai-hardware/). The serving system you cannot copy from anywhere — every workload is different, and the right stack for a chatbot is the wrong stack for a code-completion sidecar.

---

## What This Guide Covers (and What It Doesn't)

This guide is the production-serving track of the [AI Learning Guides](../../) collection. It is deliberately scoped so its content does not overlap with the other guides. Knowing what belongs here vs. elsewhere is part of using it well.

**In scope — this guide owns these topics:**
- The lifecycle of an inference request, end to end (prefill, decode, streaming, termination)
- KV-cache engineering: sizing, paging, sharing, quantizing, evicting, offloading
- Batching and scheduling algorithms for heterogeneous serving traffic
- Speculative decoding, prompt-lookup, Medusa, EAGLE, tree speculation
- *Serving-time* quantization decisions (which formats, where, with what calibration gates)
- *Inference-relevant* kernels (FlashAttention, FlashDecoding, fused decode, CUDA Graphs)
- Distributed and disaggregated serving (TP/EP for inference, prefill/decode split, KV transfer)
- Long-context serving, structured generation, multi-LoRA, prefix caching, agentic sessions
- Production concerns: observability, SLOs, capacity planning, $/M-token economics
- Non-LLM inference patterns at a high level (diffusion, embeddings, VLMs) — with pointers

**Out of scope — deferred to other guides:**
- *Model architecture and training* → [LLM](../llm/), [Image Generation](../image-generation/), [Video Generation](../video-generation/), [Multimodal Learning](../multimodal-learning/), [Robotics](../robotics/)
- *Post-training* (SFT, RLHF, DPO, GRPO, RLVR) → [LLM Phase 5](../llm/#phase-5-post-training--sft-rlhf-dpo-grpo-rlvr) and [RL Phase 9](../reinforcement-learning/#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr)
- *Evaluation methodology* → [LLM Phase 8](../llm/#phase-8-evaluation)
- *GPU architecture, memory-hierarchy theory, numeric-format design* → [AI Hardware Phases 2–7](../ai-hardware/)
- *Kernel writing as a discipline* (CUDA/Triton/CUTLASS fundamentals) → [AI Hardware Phase 4](../ai-hardware/#phase-4-cuda-triton-and-writing-real-kernels) and [PyTorch Deep Dive Phase 6](../pytorch-deep-dive/#phase-6-custom-kernels--c-cuda-and-triton-extensions)
- *Distributed **training**, FSDP/DDP/ZeRO internals* → [PyTorch Deep Dive Phase 7](../pytorch-deep-dive/#phase-7-distributed-training--ddp-fsdp-and-beyond) and [LLM Phase 4](../llm/#phase-4-scaling-laws-and-training-infrastructure)
- *Robot/agent control loops and embodied inference* → [Robotics Phase 8](../robotics/#phase-8-learning-for-robotics)

When this guide touches an out-of-scope topic, it does so only to the depth needed for serving decisions, and it links forward to the owning guide.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: The Anatomy of an Inference Request](#phase-1-the-anatomy-of-an-inference-request)
3. [Phase 2: The KV Cache](#phase-2-the-kv-cache)
4. [Phase 3: Batching and Scheduling](#phase-3-batching-and-scheduling)
5. [Phase 4: Speculative Decoding](#phase-4-speculative-decoding)
6. [Phase 5: Serving-Time Quantization Decisions](#phase-5-serving-time-quantization-decisions)
7. [Phase 6: Inference-Relevant Kernels and Hardware Choices](#phase-6-inference-relevant-kernels-and-hardware-choices)
8. [Phase 7: Distributed and Disaggregated Serving](#phase-7-distributed-and-disaggregated-serving)
9. [Phase 8: Long Context, Structured Output, and Multi-Tenant Tricks](#phase-8-long-context-structured-output-and-multi-tenant-tricks)
10. [Phase 9: Observability, SLOs, and Cost Economics](#phase-9-observability-slos-and-cost-economics)
11. [Phase 10: Frontier Topics in Serving](#phase-10-frontier-topics-in-serving)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls](#common-pitfalls)
15. [Additional Resources](#additional-resources)
16. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

Inference systems sit at the intersection of deep learning, GPU programming, distributed systems, and queueing theory. You don't need to be a specialist in all four to start — but if more than one of the topics below is completely new, slow down and learn just enough to read along.

### Concepts to Know

- **Transformer basics**: decoder-only stack, multi-head attention, MLP block, the fact that decoding produces one token at a time conditioning on the prefix. If shaky, do [LLM Phases 1–2](../llm/) first
- **GPU memory hierarchy** at a sketch level: registers → SRAM → L2 → HBM → NVLink → network, each step ~10× slower than the last. The deep treatment lives in [AI Hardware Phase 3](../ai-hardware/#phase-3-the-memory-hierarchy--where-your-time-actually-goes); this guide assumes you can already think "which level of memory am I hitting?"
- **Roofline thinking**: a kernel is either *compute-bound* (limited by FLOPs/s) or *memory-bound* (limited by bytes/s). The single most important diagnostic question in inference is "which side of the roofline am I on?" See [AI Hardware Phase 1](../ai-hardware/#phase-1-how-a-modern-computer-computes) for the underlying model
- **Async, batching, and queues**: requests arrive at random times with random lengths; you serve a fixed-throughput accelerator. Everything in the middle is queueing theory dressed up as Python
- **HTTP / gRPC / streaming**: tokens arrive at a client over an open connection (SSE or chunked transfer or WebSockets). Closing the connection wrong wastes a generation
- **Programming maturity**: you will read other people's C++/CUDA at 3 a.m. when a request gets stuck in `cudaEventSynchronize`

### The Two Numbers That Govern Everything

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │   Tokens / second / replica   ←   throughput  (cost economics)         │
   │   Time to first token + inter-token latency   ←   user-visible SLO     │
   └────────────────────────────────────────────────────────────────────────┘

Every design decision in this guide is, ultimately, a way to push one of those
two numbers in the right direction without wrecking the other. Batching is a
throughput lever that costs you latency. Speculative decoding is a latency
lever that costs you a little FLOPs. Quantization is both, when it works.
You should know, for every change you ship, which axis it moves on.
```

If your team can't quote those two numbers for your system off the top of its head, that is the first thing to fix — long before you optimize anything.

### What You Need Installed

- **Python 3.10+**, PyTorch, NumPy
- **A real inference engine** to read and run — at least one of `vllm`, `sglang`, `text-generation-inference`, `tensorrt-llm`. `vllm` is the easiest to read; `tensorrt-llm` is the fastest in many production benchmarks; `sglang` is the strongest for structured generation
- **CUDA + Nsight Systems / Nsight Compute** if you have NVIDIA hardware. You will not understand a memory-bound regression without a profiler
- **`huggingface_hub`, `transformers`, `accelerate`** — to fetch weights and run reference forward passes
- **A load generator** — `oha`, `wrk`, `vegeta`, or `vllm`'s own benchmark harness. Single-stream timing is a lie; you have to send concurrent load
- **A GPU** — 16 GB is enough for 7B inference with quantization. For production-grade work you need at least one H100/H200/B200 or equivalent

### Resources

- [Karpathy — *Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU) — the cleanest "build it yourself" tour of the forward pass
- [Horace He — *Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) — the roofline mental model, applied
- [vLLM docs](https://docs.vllm.ai/) — the inference engine you should know best by the end of this guide
- [PyTorch profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [AI Hardware Phases 1–4](../ai-hardware/) — for the hardware substrate this guide sits on top of

---

## Phase 1: The Anatomy of an Inference Request

Before optimizing anything, you need a precise mental model of what *one* request actually does inside the server. Most production bugs and most performance regressions come down to a misunderstanding of which step in this lifecycle is the bottleneck.

### Concepts to Learn

- **The request lifecycle** — from the user's POV one chat turn is `tokens go in, tokens come out`. From the server's POV it's a small state machine with at least these states:
  1. **Admission** — parse the request, validate, tokenize the prompt, attach a request ID
  2. **Queue** — wait until the scheduler picks the request for the next forward pass
  3. **Prefill** — run the entire prompt through the model in one (or a few) big forward passes; populate the KV cache
  4. **Decode** — one forward pass per generated token, reading and growing the KV cache each step
  5. **Detokenize / stream** — convert generated token IDs back to UTF-8, deal with surrogate-pair boundaries, stream chunks to the client
  6. **Termination** — stop on EOS, max_tokens, stop string, client disconnect, or timeout
- **Prefill is compute-bound; decode is memory-bandwidth-bound.** This is the single most important fact in LLM inference and it gets repeated in this guide because every design decision flows from it
  - Prefill: thousands of tokens, one big GEMM (or a few) per layer, FLOPs dominate
  - Decode: one token, one tiny GEMM per layer, but reads the *entire* weight matrix and KV cache from HBM. The arithmetic intensity collapses; the GPU sits idle waiting for memory
- **Time to first token (TTFT)** is dominated by prefill (and queue time). **Inter-token latency (ITL)** is dominated by decode. **Time per output token (TPOT)** is the productized version of ITL. **End-to-end latency** is `TTFT + N · TPOT`
- **Streaming**: the server cannot wait for generation to complete before sending bytes. Tokens flush as they're produced (Server-Sent Events, chunked HTTP, gRPC streaming). The detokenizer must handle partial UTF-8 sequences, BPE that splits a word across tokens, and the client's flush cadence
- **Stop conditions**: EOS token, max_new_tokens, stop strings (matched on decoded text — not on tokens, because tokens don't align with arbitrary string boundaries), client cancel
- **Sampling**: greedy / temperature / top-k / top-p / min-p / repetition penalty. All implemented as logit transforms before the random draw. Cheap individually, but expensive when stacked on a tight decode loop — many engines fuse the whole sampling path into one CUDA kernel
- **The "logprobs" interface**: returning the top-k token log-probabilities at each step. Cheap (you already have logits) but a memory-bandwidth tax if you stream them
- **Determinism is hard**: even greedy decoding is non-deterministic across batch sizes on a GPU because reductions reorder. You can get bitwise determinism, but it costs throughput. Decide up front whether you need it (legal/eval contexts: yes; user chat: usually no)

### One Request, Annotated

```
   Client                       Gateway              Inference engine          GPU
     │                             │                       │                    │
     │   POST /v1/chat/...   ──►   │                       │                    │
     │                             │  ── tokenize ────►    │                    │
     │                             │  (admission, queue)   │                    │
     │                             │                       │── batch w/others ─►│
     │                             │                       │    PREFILL         │
     │                             │                       │    (compute-bound) │
     │                             │  ◄─── KV cache built ─│◄──────────────────│
     │   ◄── first token (TTFT) ──── stream chunk 1 ───────│                    │
     │                             │                       │── DECODE step ────►│
     │   ◄── token 2 ───────────── stream chunk 2 ─────────│   (mem-bw bound)   │
     │   ◄── token 3 ───────────── stream chunk 3 ─────────│── DECODE step ────►│
     │              ...                                                         │
     │   ◄── token N + [DONE] ─── stream end ──────────────│── EOS reached ────►│
                                                            │── free KV blocks ─►│

The two-phase shape (one big prefill, many small decodes) is what every
optimization in this guide tries to exploit, work around, or hide.
```

### TTFT, TPOT, and Why You Quote Both

```
   ── Single-stream view (one user, no contention) ─────────────────────────────

     prompt: 2000 tokens, output: 200 tokens, 70B model on H100x2

       TTFT  ≈ 600 ms       (prefill: 2000 tokens × FLOPs/token)
       TPOT  ≈  40 ms       (decode: 200 GB weights / 3 TB/s ≈ 23 ms + overhead)
       E2E   ≈ TTFT + 200 · TPOT  ≈ 8.6 s

   ── Production view (50 concurrent users) ────────────────────────────────────

     Queue time creeps into TTFT; batch helps TPOT scale sub-linearly.
     A well-tuned system: TTFT P50 ≈ 800 ms, TPOT P50 ≈ 50 ms, throughput
     ≈ 1500 tok/s aggregate across the batch.

   If you only optimize TPOT you ship a system with great steady-state speed
   and a horrible first-impression delay. If you only optimize TTFT you ship
   one where long answers feel like they're crawling. You need both.
```

### Other Inference Workloads — Where the LLM Pattern Doesn't Apply

Most of this guide is written for autoregressive LLMs, because that's where serving complexity is densest. But "inference" is a broader umbrella, and the techniques shift in important ways for each non-LLM workload. The table below is the map; deeper treatment lives in the topic-specific guides.

```
   Workload                Dominant cost       Batching shape         Owning guide
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   Decoder-only LLM        Decode: HBM B/W     Heterogeneous,         this guide
                           Prefill: FLOPs       continuous
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   Diffusion (image/video) Many denoising      Big static batches,    Image Gen Phase 7
                           steps, all prefill-  same step-count       Video Gen Phase 4
                           shaped (FLOPs)       per request
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   Embeddings / rerankers  One forward pass,   Huge homogeneous       Multimodal Phase 3
                           pure encoder        batches, no decode
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   VLMs (image+text in,    Vision encoder      LLM-shaped after the   Multimodal Phase 5
   text out)               (prefill-like) +     vision tower; KV
                           LLM decode           cache lives in text
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   ASR / TTS               Mostly encoder      Big batches, low       Multimodal Phase 6
                           passes; some        per-request memory
                           autoregressive
   ─────────────────────   ─────────────────   ────────────────────   ─────────────────
   Robot policies / VLAs   Real-time control   Tiny batches (often    Robotics Phase 8
                           loop (10–100 Hz)    1), latency-critical
```

The big takeaways for non-LLM workloads:

- **Diffusion serving** is essentially "do prefill N times" — there's no KV cache, no autoregression. The optimizations that matter are step-distillation (turning 50-step sampling into 1–4 steps), batched cross-attention, and CFG fusion. See [Image Generation Phase 10](../image-generation/#phase-10-training-at-scale-distillation-evaluation-and-frontier-topics) and [Video Generation Phase 10](../video-generation/#phase-10-training-at-scale-evaluation-and-frontier-topics).
- **Embedding/rerank serving** is the easy case: pure encoder, no decode loop, batch as wide as memory allows. Don't try to serve them in the same engine as decoder LLMs — the optimal batch sizes differ by an order of magnitude.
- **VLM serving** is "an LLM with an extra prefill stage." Everything in this guide applies after the vision encoder runs; the new wrinkle is caching the vision encoder output across multi-turn conversations.
- **Robot policy inference** lives at the opposite end of the latency/throughput trade-off — batch size 1, hard deadlines, often on edge silicon. The KV-cache, continuous-batching, speculation playbook is not the right toolbox.

The rest of this guide focuses on the decoder-only LLM case unless otherwise noted, because that's where the system-engineering depth is.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Manual inference loop](projects/01-manual-inference-loop/README.md) | Load a 1B HF model, write a Python `while` loop that prefills the prompt, then decodes one token at a time using the KV cache | ⭐⭐ |
| [Streaming server](projects/02-streaming-server/README.md) | Wrap the loop in FastAPI with `StreamingResponse`; benchmark TTFT vs. ITL with `oha` | ⭐⭐⭐ |
| [Stop-string matcher](projects/03-stop-string-matcher/README.md) | Implement a streaming stop-string matcher that handles BPE token boundaries correctly; verify against pathological cases (stop string split across two tokens) | ⭐⭐⭐ |
| [Sampling kernel](projects/04-sampling-kernel/README.md) | Implement top-k + top-p + temperature sampling on logits in pure PyTorch; profile against `torch.multinomial` + `topk` | ⭐⭐⭐ |
| [Detokenizer fuzzer](projects/05-detokenizer-fuzzer/README.md) | Generate random token sequences, decode incrementally vs. all-at-once; find a case where they disagree | ⭐⭐⭐⭐ |
| [Determinism audit](projects/06-determinism-audit/README.md) | Run the same prompt 100× with greedy decoding at different batch sizes; report bitwise divergence rate; fix one source | ⭐⭐⭐⭐ |
| [Request-lifecycle tracer](projects/07-request-lifecycle-tracer/README.md) | Add per-stage timestamps to a request; produce a flamegraph for one slow request from a load test | ⭐⭐⭐ |
| [Diffusion vs LLM serving](projects/08-diffusion-vs-llm-serving/README.md) | Serve a Stable Diffusion model and a 7B LLM behind the same load generator; explain why batch sizes and TTFT shapes look so different | ⭐⭐⭐ |

### Sample Code: A Minimal Decode Loop With a KV Cache

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

@torch.inference_mode()
def generate(prompt, max_new_tokens=128, temperature=0.0):
    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    # ── Prefill: one forward pass over the whole prompt ──
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)

    yield tok.decode(next_id[0])

    # ── Decode: one token at a time, reusing the KV cache ──
    for _ in range(max_new_tokens - 1):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if temperature == 0.0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        if next_id.item() == tok.eos_token_id:
            break
        yield tok.decode(next_id[0])

for chunk in generate("Once upon a time"):
    print(chunk, end="", flush=True)
```

That is, almost literally, the inner loop every inference engine implements — minus everything that makes it fast (paged KV cache, batched decode, fused kernels, scheduler, etc.).

### Key Insight

LLM inference has a *bimodal* performance profile: one big compute-heavy prefill, followed by a long tail of small memory-bound decode steps. Almost every optimization in this guide picks one of these two regimes and attacks it. The engineers who get confused are the ones who optimize a kernel for *the wrong phase*: a fused-multiply-add improvement that helps prefill by 30% and decode by 0.5% is a great paper and a tiny throughput win in a real serving mix where most of the work is decode.

### Resources

- [vLLM blog — *How continuous batching enables 23× throughput*](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Pope et al. — *Efficiently Scaling Transformer Inference* (2022)](https://arxiv.org/abs/2211.05102) — the field-defining survey
- [Reiner Pope's talks on inference scaling](https://www.youtube.com/results?search_query=reiner+pope+inference)
- [FastAPI streaming docs](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

---

## Phase 2: The KV Cache

The KV cache is the single most important data structure in LLM serving. It is the reason decode is fast (you don't recompute attention over the prefix every step) and it is the reason serving is hard (its memory footprint dominates GPU RAM and grows with every active request). Master this and you understand 80% of why modern inference engines look the way they do.

> The [LLM guide introduces](../llm/#phase-9-efficient-inference-and-deployment) the KV cache as a one-paragraph idea. This phase is the engineering deep dive: sizing it, paging it, sharing it, quantizing it, evicting it, and tiering it. If any of these data-structure choices later show up in a production incident, the cause almost always traces here.

### Concepts to Learn

- **What's actually cached**: at each layer, for each previously-seen token, the projected `K` and `V` vectors. Once computed, they don't change — they only get attended to by future queries
- **Size formula** (memorize this):
  ```
  KV-cache bytes per request
     = 2 (K and V)
     × n_layers
     × n_kv_heads          ← post-GQA, NOT n_heads
     × d_head
     × seq_len
     × bytes_per_element   ← 2 for BF16, 1 for FP8, 0.5 for INT4
  ```
  For a 70B Llama-style model (80 layers, 8 KV heads, 128 d_head) at 8k context in BF16: ≈ 2.6 GB *per request*. Times 32 concurrent users: ≈ 84 GB. That's most of an H100.
- **Why GQA / MQA matter so much for serving**: by sharing K and V across query heads, GQA shrinks the KV cache by `n_heads / n_kv_heads`. This is the *biggest* inference-time win of GQA — not the FLOPs savings, which are minor at decode
- **Allocation strategy**:
  - **Contiguous** (early implementations) — allocate `max_seq_len × max_batch` up front; massive fragmentation and OOM
  - **Paged** (PagedAttention, vLLM, 2023) — chop the cache into fixed-size blocks (e.g., 16 tokens each), maintain a per-request block table, allocate blocks on demand
  - **TokenAttention / radix-tree** (sglang) — model the cache as a radix tree keyed on prompt prefixes; share blocks across requests with a common prefix automatically
- **Prefix sharing / prefix caching**: if many requests share a long system prompt or a retrieved document, the prefill compute and the KV cache memory can be reused. Production systems with a static system prompt get a 5–20× TTFT win from this alone
- **KV-cache quantization**: drop the cache to FP8, INT8, or even INT4. The cache is huge and bandwidth-dominant; halving it nearly halves decode latency. Quality loss is small for K/V (smaller than for weights) — they're activations, not parameters. (Format-design and outlier-handling theory live in [AI Hardware Phase 7](../ai-hardware/#phase-7-numeric-formats-and-quantization); this phase is about when and where to apply them in the cache.)
- **KV cache offload**: when GPU memory is tight, evict cold blocks to CPU RAM or NVMe. Used for very long sessions or huge agentic histories. The trade-off is reload latency on the next decode step that touches the offloaded blocks
- **Cache eviction policies**:
  - LRU at the *block* level for inactive sessions
  - **H2O / Scissorhands / attention sink** — drop tokens whose attention weights have been consistently small. Quality loss is small for long contexts where most past tokens don't matter
  - **Sliding-window** — Mistral-style: keep only the last `W` tokens of K/V. Matches the model's training-time attention pattern

### The Cache, Visualized

```
   Layer 0      Layer 1     ...     Layer L-1
   ┌────────┐  ┌────────┐          ┌────────┐
   │ K₀  V₀ │  │ K₀  V₀ │   ...    │ K₀  V₀ │
   │ K₁  V₁ │  │ K₁  V₁ │          │ K₁  V₁ │   ← one row per past token,
   │ K₂  V₂ │  │ K₂  V₂ │          │ K₂  V₂ │     per layer, per kv-head
   │  ⋮    │  │  ⋮    │          │  ⋮    │
   │ Kₜ Vₜ  │  │ Kₜ Vₜ  │          │ Kₜ Vₜ  │   ← growing each decode step
   └────────┘  └────────┘          └────────┘
       │            │                    │
       └────────────┴────────────────────┘
            attention reads ALL of this on EVERY decode step

The decode-step read traffic is what bounds tokens/sec on a memory-bound GPU.
This is why halving the KV cache (GQA, quantization, eviction) approximately
halves single-user decode latency.
```

### PagedAttention, In One Picture

```
    request A             request B           request C
    block table:          block table:        block table:
    [P3, P7, P12, …]      [P0, P4, P9, …]     [P1, P2, P15, …]
            │                     │                     │
            └─────────┬───────────┴──────────┬──────────┘
                      ▼                      ▼
              ┌─────────────────────────────────────┐
              │  Physical block pool (HBM)          │
              │  P0  P1  P2  P3  P4  P5 ... Pₙ      │   ← each block stores
              │  ▒▒  ▒▒  ▒▒  ▒▒  ▒▒  ▒▒     ▒▒      │     16 (or so) tokens'
              └─────────────────────────────────────┘     worth of K and V

   Benefits:
     • Zero allocation-time fragmentation
     • Common prefixes (e.g., system prompt) share physical blocks → near-free
     • Cache lives across many requests of differing length without OOM
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [KV cache from scratch](projects/09-kv-cache-from-scratch/README.md) | Add a contiguous KV cache to a toy transformer; verify outputs match no-cache decoding bit-for-bit | ⭐⭐⭐ |
| [KV size calculator](projects/10-kv-size-calculator/README.md) | Implement the formula above; sweep `n_kv_heads`, `seq_len`, `dtype`; plot memory vs. concurrency | ⭐⭐ |
| [Tiny paged cache](projects/11-tiny-paged-cache/README.md) | Implement a block-paginated KV cache (block size 16); allocate / free blocks per request; reproduce the vLLM block-table data structure | ⭐⭐⭐⭐⭐ |
| [Prefix-share benchmark](projects/12-prefix-share-benchmark/README.md) | In vLLM, send 100 requests sharing a 2000-token system prompt; measure TTFT with prefix cache on vs. off | ⭐⭐⭐ |
| [KV-quantization study](projects/13-kv-quantization-study/README.md) | Drop the KV cache to FP8 (or INT8 with a custom kernel); measure quality on a held-out eval and throughput delta | ⭐⭐⭐⭐ |
| [Attention-sink eviction](projects/14-attention-sink-eviction/README.md) | Implement an H2O-style "keep top-attention tokens + first 4 tokens" eviction policy; measure quality at long context | ⭐⭐⭐⭐⭐ |
| [CPU/NVMe offload](projects/15-cpu-nvme-offload/README.md) | Add a tier-2 cache (CPU RAM); evict cold blocks on GPU pressure; measure reload-cost vs. throughput on long-running sessions | ⭐⭐⭐⭐⭐ |

### Sample Code: The Memory Math, Made Concrete

```python
def kv_cache_bytes(seq_len, batch, *, n_layers, n_kv_heads, d_head, dtype_bytes=2):
    """KV cache size for a given dense decoder. Returns bytes."""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes

# Llama 3 70B-ish
b = kv_cache_bytes(seq_len=8192, batch=32,
                   n_layers=80, n_kv_heads=8, d_head=128)
print(f"{b / 1e9:.1f} GB")    # ≈ 84 GB — bigger than the weights at BF16!

# Same model, FP8 KV cache
b = kv_cache_bytes(seq_len=8192, batch=32,
                   n_layers=80, n_kv_heads=8, d_head=128, dtype_bytes=1)
print(f"{b / 1e9:.1f} GB")    # ≈ 42 GB — twice the concurrency in the same RAM

# Compare to the model weights at BF16 (≈ 140 GB for a 70B dense model)
# The cache and the weights are comparable in size — neither is "small."
```

### Key Insight

If you remember one thing from this guide: **the KV cache is the inference engine's working set, and almost every serving optimization is, at some level, a strategy for managing it.** Continuous batching exists to keep cache slots full. PagedAttention exists to eliminate cache fragmentation. KV quantization exists to shrink cache bandwidth. Speculative decoding works because it amortizes a fixed cache-read cost across multiple accepted tokens. Prefix caching exists so two requests can share the same physical cache pages. Get the KV cache right and the rest of the system falls into place; get it wrong and no amount of clever scheduling will save you.

### Resources

- [Kwon et al. — *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — the vLLM paper
- [Zheng et al. — *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — RadixAttention
- [Zhang et al. — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* (2023)](https://arxiv.org/abs/2306.14048)
- [Xiao et al. — *Efficient Streaming Language Models with Attention Sinks* (2023)](https://arxiv.org/abs/2309.17453)
- [vLLM source: `vllm/core/block_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py) — read it

---

## Phase 3: Batching and Scheduling

A GPU is happiest when it has a lot of work to do in parallel. A single user's decode step uses about 1% of an H100's FLOPs. The path from "wasted accelerator" to "production throughput" is **batching** — and because requests arrive at different times, are different lengths, and finish at different times, batching is not a one-line `torch.stack`. It is a scheduler.

### Concepts to Learn

- **Static batching (the naive approach)**: collect `N` requests, pad them to the max length, run one forward pass, return results. Two fatal problems: (1) padding wastes compute, (2) the whole batch waits for the slowest request to finish
- **Continuous batching / inflight batching / iteration-level scheduling** — the modern default:
  - At every decode iteration, the scheduler can add new requests to the batch (if they just finished prefill) and remove ones that hit EOS
  - The batch is *heterogeneous*: different requests are at different positions in their generation
  - Implemented in vLLM, TGI, TensorRT-LLM, sglang. The single biggest serving throughput win of the post-2022 era — up to ~20× throughput at iso-latency
- **Chunked prefill** — long prompts (say 32k tokens) would otherwise hog the GPU for hundreds of milliseconds, stalling all decode steps. Instead, split prefill into chunks (e.g., 2048 tokens at a time) and *interleave* prefill chunks with decode steps in the same forward pass. Trades a bit of prefill efficiency for far better tail latency
- **Disaggregated prefill / decode** — push the idea further: run prefill on one set of GPUs and decode on another, connected by a fast interconnect. Each pool is sized for its own workload (compute-heavy vs. bandwidth-heavy). Mooncake, DistServe, and increasingly the production systems behind frontier APIs use this. (Phase 7 covers the cross-node engineering.)
- **The scheduler's job**, formalized — at every iteration, given a queue of pending requests and a running batch, decide:
  - Which pending requests to admit (prefill them in this iteration?)
  - Which running requests to decode this iteration (and which to pause if cache is full)
  - Which requests to preempt (cache eviction or queue back) when memory pressure rises
  - All under SLO constraints (TTFT P99, TPOT P99)
- **Admission control and backpressure**: when the queue is too long, reject early rather than promising a response you can't meet. Better to return `429` immediately than silently exceed every SLO
- **Priority and multi-tenancy**: paid tier vs. free tier, latency-sensitive vs. batch jobs. Real schedulers carry priority classes and per-tenant quotas
- **The throughput / latency / fairness triangle**: pick two. A scheduler that maximizes throughput will starve short requests; one that minimizes P99 latency will under-utilize the GPU; one that's fair across tenants will leave throughput on the table. There is no free lunch
- **Speculative + batching interactions**: speculative decoding turns "1 decode step → 1 token" into "1 decode step → 1.5–4 tokens *on average*". Batches of speculations are jagged (some requests accept more tokens than others). Handling this efficiently is a research problem and an engineering one

### Continuous Batching, Step by Step

```
    Iteration 0:  Batch = [ ]                Queue = [A, B, C, D]
                  Scheduler admits A, B for prefill.
                  GPU runs PREFILL(A) + PREFILL(B) (batched).

    Iteration 1:  Batch = [A(d=0), B(d=0)]   Queue = [C, D]
                  Scheduler admits C for prefill, decodes A and B.
                  GPU runs PREFILL(C) + DECODE(A) + DECODE(B) in one pass.

    Iteration 2:  Batch = [A(d=1), B(d=1), C(d=0)]   Queue = [D]
                  Scheduler decodes all three; admits D for prefill.
                  GPU runs PREFILL(D) + 3× DECODE.

    Iteration k:  B finishes (EOS); its KV blocks are freed.
                  Scheduler can now admit another request.

   The forward pass is heterogeneous: some sequences are at position 0
   (prefill), some at position 47, some at 312. This is what production
   inference engines mean by "continuous" or "in-flight" batching.
```

### Chunked Prefill vs. The Tail

```
   Without chunked prefill:
       request X (prompt = 32k tokens) arrives → 500 ms of pure prefill
       → all other requests' TPOT spikes from 40 ms to 540 ms in that window
       → P99 ITL goes through the roof

   With chunked prefill (chunk size = 2048):
       Each iteration: 2048 prefill tokens for X + 1 decode token each for
       the other 30 requests, fused into one forward pass.
       X's TTFT goes up slightly (its prefill is spread over 16 iterations)
       Everyone else's ITL stays smooth.

   Net effect: a giant prompt no longer hijacks the system.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Static vs. continuous](projects/16-static-vs-continuous/README.md) | Build both schedulers around your toy decode loop; load-test under varying lengths and arrival rates | ⭐⭐⭐⭐ |
| [Padding waste audit](projects/17-padding-waste-audit/README.md) | Instrument a static-batching server: what fraction of decode FLOPs go to padding tokens? | ⭐⭐⭐ |
| [Chunked prefill simulator](projects/18-chunked-prefill-simulator/README.md) | Discrete-event sim: requests arrive Poisson, prefill consumes `P` time, decode `D` per step; sweep chunk size; plot TTFT P99 vs. throughput | ⭐⭐⭐⭐ |
| [Disaggregated PoC](projects/19-disaggregated-poc/README.md) | Two processes (prefill / decode), exchange KV cache over IPC or RDMA; measure overhead vs. unified | ⭐⭐⭐⭐⭐ |
| [Priority queue](projects/20-priority-queue/README.md) | Add a priority class to vLLM (or a fork); verify high-prio TTFT under load | ⭐⭐⭐⭐ |
| [Cache-aware admission](projects/21-cache-aware-admission/README.md) | Refuse admission when projected KV cache exceeds GPU; verify the system never OOMs | ⭐⭐⭐⭐ |
| [SLO-aware scheduler](projects/22-slo-aware-scheduler/README.md) | Given per-request deadlines, schedule to maximize on-time completions; compare to FCFS | ⭐⭐⭐⭐⭐ |

### Sample Code: A Toy Continuous-Batching Scheduler

```python
class Scheduler:
    def __init__(self, max_batch_tokens=8192):
        self.running = []        # list of Request objects, each with KV state
        self.queue   = []
        self.max_batch_tokens = max_batch_tokens

    def step(self, model):
        # 1. Admit new prefills until we'd exceed the per-iteration token budget
        budget = self.max_batch_tokens - sum(1 for _ in self.running)  # 1 tok/decode
        while self.queue and self.queue[0].prompt_len <= budget:
            req = self.queue.pop(0)
            budget -= req.prompt_len
            req.state = "prefill"
            self.running.append(req)

        # 2. Build a heterogeneous batch: prefill chunks + 1-token decodes
        inputs, kvs = pack_batch(self.running)
        logits, new_kvs = model.forward(inputs, kvs)
        update_kvs(self.running, new_kvs)
        sampled = sample(logits)

        # 3. For each running request, advance state and check termination
        finished = []
        for req, tok in zip(self.running, sampled):
            if req.state == "prefill":
                req.state = "decode"
            req.tokens.append(tok)
            if tok == EOS or len(req.tokens) >= req.max_new:
                finished.append(req)

        for req in finished:
            free_kv(req)
            self.running.remove(req)
            yield req
```

That sketch leaves out the hard parts — paged KV cache management, chunked prefill, GPU streams, etc. — but it captures the idea: every iteration, decide what to do next; never let the GPU run a homogeneous static batch when a heterogeneous one is available.

### Key Insight

The scheduler is the *most underappreciated* part of an inference stack. It rarely has a paper attached to it; it gets less attention than fancy kernels and clever attention variants; and it is responsible for more of the actual throughput-per-dollar number than any other component. If you are evaluating two inference engines on the same hardware, the one with the better scheduler usually wins by 2× — and the wider the latency distribution of your real workload, the bigger that gap gets.

### Resources

- [Yu et al. — *Orca: A Distributed Serving System for Transformer-Based Generative Models* (2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — the iteration-level batching paper
- [Anyscale blog — *Continuous Batching*](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Agrawal et al. — *Sarathi-Serve / Chunked Prefill* (2024)](https://arxiv.org/abs/2403.02310)
- [Zhong et al. — *DistServe* (2024)](https://arxiv.org/abs/2401.09670)
- [Qin et al. — *Mooncake* (2024)](https://arxiv.org/abs/2407.00079)
- vLLM source: `vllm/core/scheduler.py` and `vllm/engine/llm_engine.py`

---

## Phase 4: Speculative Decoding

If decode is memory-bandwidth-bound, then the GPU is mostly *waiting* during decode. Speculative decoding exploits this: spend the unused compute to verify several draft tokens at once. When it works (and modern variants almost always do, at acceptance rates of 60–90%), it gives you 2–4× faster decode for free — same quality, same model.

### Concepts to Learn

- **The core idea** (Leviathan et al. 2023):
  - A small **draft** model proposes `k` next tokens autoregressively (cheap)
  - The big **target** model verifies all `k+1` positions in *one* parallel forward pass (the cost is dominated by reading weights once, not k times)
  - Tokens that match the target's argmax (or that pass a probabilistic rejection step) are accepted; on the first mismatch, accept the target's choice and discard the rest
  - Output distribution is **identical** to the target model's — this is provable, not just empirical
- **Self-speculation / Medusa** — instead of a separate draft model, train extra prediction heads on the target model that predict tokens at positions `t+1, t+2, t+3, …`. No separate model to manage, no separate KV cache, much higher acceptance rates
- **EAGLE / EAGLE-2 / EAGLE-3** — predict the next-token *feature vector* (not token ID), then decode from that. Higher acceptance than Medusa because predicting a vector is easier than predicting a discrete token
- **N-gram / prompt lookup decoding** — when the model is paraphrasing or copying from the prompt (common in summarization, RAG, code edits), the draft is just *the prompt itself*. Free, no training, surprisingly large speedups on copy-heavy workloads
- **Tree-style speculation** — instead of a single chain of `k` drafts, propose a *tree* of alternatives at each position and verify the whole tree in one forward pass. Higher token-acceptance per iteration at the cost of more verification FLOPs
- **Acceptance rate** is the headline metric. Define it as `accepted_tokens / proposed_tokens`. For 1B-draft + 70B-target on chat, acceptance is typically 0.65–0.80. Medusa pushes 0.7–0.85. EAGLE pushes 0.85+. Higher acceptance → higher speedup, with diminishing returns
- **The math of speedup**: if you propose `k` tokens and the expected accepted prefix length is `α(k)`, the wall-clock speedup is roughly `α(k) / (1 + draft_cost / target_cost)`. For small drafts, `draft_cost / target_cost` is ~3–10% and you get most of `α(k)`. Production systems pick `k = 3–5` because the acceptance curve flattens past that
- **What can go wrong**:
  - Speculation interacts poorly with naive batching — a batch is only as fast as its slowest sequence, and acceptance length varies. Modern engines pack speculation into continuous batching with dynamic re-packing
  - Sampling-mode speculation needs a careful **rejection-resampling** step to preserve the target distribution; greedy speculation is much simpler
  - For very small models (<3B target), draft overhead dominates and speculation can be a *slowdown*

### A Speculative-Decode Iteration, Visualized

```
   Target prefix:   [t₀, t₁, ..., t_{n-1}]

   Step 1 (draft, autoregressive, cheap):
       d_n, d_{n+1}, d_{n+2}, d_{n+3} ← Draft.generate(prefix, k=4)

   Step 2 (target, ONE parallel forward over n + k positions):
       target_logits[n-1 : n+k-1] ← Target.forward(prefix + [d_n, …, d_{n+3}])

       This forward pass costs ~ the same as a single decode step,
       because it reads the target's weights from HBM exactly once.

   Step 3 (verify):
       For i in 0..k-1:
           if argmax(target_logits[n + i - 1]) == d_{n+i}:
               accept
           else:
               break
       Append accepted tokens + ONE bonus token (target's argmax at first mismatch)

   Best case (k=4, all accepted): 5 new tokens in one target forward pass.
   Average case (k=4, α≈0.75):    ~3 new tokens in one target forward pass.
   Worst case (k=4, α=0):         1 new token (still correct, just no speedup).
```

### Why It Costs Almost Nothing Extra

```
   Decode-step cost on a 70B target on H100:
       Read 140 GB weights once + small GEMV  ≈  45 ms
                                              ^^^^^^^^
                                              dominated by HBM read

   Verification cost (k=4):
       Same weight read + slightly bigger GEMV  ≈  47 ms
       (the GEMV grows from M=1 to M=5, but it's still tiny vs. the read)

   The "speculation tax" is ~5%; the speedup ceiling is ~k×.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Greedy speculative decoding](projects/23-greedy-speculative-decoding/README.md) | Pair a 1B draft with a 7B target; implement the verify loop; measure acceptance and wall-clock speedup | ⭐⭐⭐⭐ |
| [Sampling-mode rejection](projects/24-sampling-mode-rejection/README.md) | Add the probabilistic accept/reject step for non-greedy sampling; verify output distribution matches target | ⭐⭐⭐⭐⭐ |
| [N-gram lookup](projects/25-n-gram-lookup/README.md) | Implement prompt-lookup decoding: scan the prompt for matching n-grams to use as drafts. Measure on a summarization workload | ⭐⭐⭐ |
| [Tune `k`](projects/26-tune-k/README.md) | Sweep `k ∈ {1, 2, 3, 4, 5, 7, 10}`; plot acceptance, speedup, and tail latency. Find the knee | ⭐⭐⭐ |
| [Medusa heads](projects/27-medusa-heads/README.md) | Train 3 Medusa heads on a small base model; report acceptance vs. external draft | ⭐⭐⭐⭐⭐ |
| [Speculation + batching](projects/28-speculation-batching/README.md) | Add speculative decoding to a continuous-batching engine; handle ragged acceptance across the batch correctly | ⭐⭐⭐⭐⭐ |
| [Workload sensitivity](projects/29-workload-sensitivity/README.md) | Measure speedup on chat, code completion, summarization, JSON-mode. Explain the variance | ⭐⭐⭐ |

### Sample Code: Greedy Speculative Decoding

```python
@torch.inference_mode()
def speculative_decode(target, draft, prompt_ids, k=4, max_new=128):
    out = prompt_ids.clone()
    while out.size(1) - prompt_ids.size(1) < max_new:
        # 1. Draft proposes k tokens autoregressively
        draft_tokens = draft.greedy_generate(out, k)     # (1, k)

        # 2. Target verifies all k positions in one forward pass
        candidate = torch.cat([out, draft_tokens], dim=1)
        target_logits = target(candidate).logits          # (1, len, V)
        target_ids = target_logits[:, -k-1:, :].argmax(-1)  # (1, k+1)

        # 3. Walk forward, accepting matches; on first mismatch, take target's
        accepted = 0
        for i in range(k):
            if draft_tokens[0, i] == target_ids[0, i]:
                accepted += 1
            else:
                break

        new = draft_tokens[:, :accepted]
        bonus = target_ids[:, accepted:accepted+1]        # the "free" extra token
        out = torch.cat([out, new, bonus], dim=1)

        if bonus.item() == EOS:
            break
    return out
```

### Key Insight

Speculative decoding is one of the rare inference optimizations that is genuinely *free*: same output distribution, no quality loss, no retraining of the target, and a 2–4× speedup that compounds with every other optimization in this guide. The reason it works is the memory-bandwidth bottleneck of decode — the GPU has more compute than the HBM can feed; speculation just gives it more compute to do per round of HBM traffic. The lesson generalizes: anywhere you see an underutilized resource, look for a way to *front-load* it speculatively against the bottleneck.

### Resources

- [Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding* (2022)](https://arxiv.org/abs/2211.17192)
- [Chen et al. — *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind, 2023)](https://arxiv.org/abs/2302.01318)
- [Cai et al. — *Medusa: Simple LLM Inference Acceleration Framework* (2024)](https://arxiv.org/abs/2401.10774)
- [Li et al. — *EAGLE / EAGLE-2 / EAGLE-3* (2024–2025)](https://arxiv.org/abs/2401.15077)
- [Saxena — *Prompt Lookup Decoding*](https://github.com/apoorvumang/prompt-lookup-decoding)
- vLLM speculative-decoding source: `vllm/spec_decode/`

---

## Phase 5: Serving-Time Quantization Decisions

A 70B model in BF16 is 140 GB of weights. In INT4, it's 35 GB. That difference is the line between "needs eight H100s" and "runs on one." Quantization is the single most leveraged knob in inference cost.

> The theory of numeric formats (FP8 vs INT8 vs NF4, outlier-handling math, calibration algorithms) lives in [AI Hardware Phase 7](../ai-hardware/#phase-7-numeric-formats-and-quantization). This phase is the *serving-side* view: given that the formats exist, which knob do you turn for which workload, where do you put the calibration gate, and how do you keep quality from quietly regressing across deploys.

### Concepts to Learn

- **The serving-time decision tree** — for each of these, you make a separate call:
  - **Weights** — FP8 is the new default for server-grade; INT4 weight-only (AWQ / GPTQ) for memory-constrained / consumer deployments; sub-4-bit (GGUF I-quants, AWQ-2) for edge
  - **Activations** — FP8 with weights is "W8A8" and works well on Hopper+; INT8 activations need SmoothQuant-style outlier migration to be safe; INT4 activations are rare and risky
  - **KV cache** — separate decision from weights, often *better leverage*. FP8 KV is essentially lossless; INT8 KV with per-token scaling works well; INT4 KV is the frontier. See Phase 2 for the cache-engineering side
  - **Embeddings / lm_head** — usually quantized like weights, but sometimes left in higher precision because they're a small fraction of the model and a quality cliff if they go wrong
- **The "right" recipe depends on the workload regime**:
  - **Bandwidth-bound decode** (most serving) → quantize weights *and* KV. Reducing bytes/token directly speeds up decode
  - **Compute-bound prefill** (long-prompt RAG, batch jobs) → quantization helps less per request; pick formats with hardware Tensor Core support (FP8 on Hopper, FP4 on Blackwell, INT4-tensor-core on Hopper+)
  - **Memory-fit** (you just need it to load) → INT4 weight-only is the lever; quality gates as below
- **Calibration as a deploy-time step** — not a research artifact. Production teams keep a small (~256-prompt) calibration set drawn from production traffic. Re-calibrate when the model changes, when traffic shifts materially, or when a regression shows up
- **The AWQ vs. GPTQ vs. SmoothQuant choice, practically**:
  - **AWQ** — fast to produce, near-best 4-bit weight quality, works without per-layer Hessians. The default for new deployments
  - **GPTQ** — slower to produce but well-trodden; almost always close to AWQ on quality
  - **SmoothQuant** — when you need W8A8 (activations quantized too); migrates the outlier magnitude into weights so neither side blows up
  - Most teams ship AWQ for weight-only and SmoothQuant for activation quantization
- **The quality gate** — every quantized deployment needs a gate that runs *before* it sees traffic:
  - Perplexity on a small held-out set (sensitive but not human-aligned)
  - Capability evals matched to the workload (MMLU-Pro, GSM8K, HumanEval, an internal eval); within 1–2 points of baseline is the usual bar
  - A production-traffic shadow eval (mirror a small share of real traffic; compare quantized vs. baseline output side-by-side)
  - If you skip the gate, you discover the regression three weeks later from a customer ticket
- **Don't trust a vendor's "lossless 4-bit" claim**. Re-evaluate after every quantization, especially after quantization + fine-tuning combinations, where regressions compound silently
- **Quantization-aware training (QAT)** is *training* work, not serving work; if quality at 4-bit isn't acceptable post-training, that decision goes back to the training team. See [AI Hardware Phase 7](../ai-hardware/#phase-7-numeric-formats-and-quantization) for the format/QAT theory

### The Bit-Width Trade-Off, From The Serving Side

```
   Bits/weight     Memory     Quality (vs BF16)     When you ship it
   ───────────────────────────────────────────────────────────────────
   BF16 (16-bit)    1.0×       baseline              fine but expensive
   FP8  (8-bit)     0.5×       ~0% loss              server default in 2026
   INT8 (8-bit)     0.5×       ~0% loss (weights)    fine for weights
   INT4 (4-bit)     0.25×      ~0.5–2% loss          standard at the edge / mid
   INT3             0.19×      noticeable            niche
   INT2             0.125×     significant           research-only

   Server-grade sweet spot today:  FP8 weights + FP8 activations + FP8 KV cache.
   Memory-constrained sweet spot:  INT4 weights (AWQ) + INT8 KV.
   Edge / consumer:                INT4 weights + sliding-window KV.

   Blackwell adds FP4 to the menu; production deployment in 2026 is rolling out.
```

### Where Each Format Pays You Back

```
   Lever            Decode TPOT      Memory headroom    Quality risk
   ─────────────    ─────────────    ───────────────    ─────────────
   FP8 weights      ~1.8× faster     2× more concurrency  ~0% (safe default)
   INT4 weights     ~3× faster       4× more concurrency  small (gate it)
   FP8 KV cache     ~1.4× faster     2× longer context    ~0% (safe default)
   INT8 KV cache    ~1.6× faster     2× longer context    small (gate it)
   INT4 KV cache    ~1.8× faster     4× longer context    real (gate hard)
   FP8 activations  marginal         marginal             small (gate it)

   The two highest-leverage moves for most serving deployments are
   FP8 weights and FP8 KV cache. Get those right before reaching for
   anything more aggressive.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Quantize a 7B model end-to-end](projects/30-quantize-a-7b-model-end-to-end/README.md) | Pick AWQ; calibrate on 128 production-style prompts; serve with vLLM; pass a quality gate | ⭐⭐⭐ |
| [FP8 KV cache](projects/31-fp8-kv-cache/README.md) | Switch a serving deployment from BF16 KV to FP8 KV; measure decode speedup and verify a quality gate | ⭐⭐⭐⭐ |
| [W4A8 ablation](projects/32-w4a8-ablation/README.md) | Compare W4-only vs. W4A8 (4-bit weights, 8-bit activations); measure quality and tokens/sec | ⭐⭐⭐⭐ |
| [Mixed-precision deployment](projects/33-mixed-precision-deployment/README.md) | Keep attention output projections + lm_head in BF16 while quantizing the rest; measure the quality bump and the cost | ⭐⭐⭐⭐ |
| [Calibration drift study](projects/34-calibration-drift-study/README.md) | Calibrate on traffic from week 0; re-evaluate quality after 12 weeks of distribution drift; quantify the gap | ⭐⭐⭐⭐ |
| [Eval-suite for quantized models](projects/35-eval-suite-for-quantized-models/README.md) | Build an automated gate that blocks a quantized model from deploying if any of N evals regresses by more than X | ⭐⭐⭐ |
| [FP4 (Blackwell) deployment](projects/36-fp4-blackwell-deployment/README.md) | If hardware available, benchmark FP4 weights against FP8; report quality, throughput, and operational gotchas | ⭐⭐⭐⭐⭐ |

### Sample Code: A Production Quality-Gate Sketch

```python
def deploy_gate(quantized_model, baseline_model, evals, prod_shadow_set,
                max_per_eval_drop=2.0, max_shadow_disagreement=0.05):
    """Returns True if `quantized_model` may be promoted to traffic."""
    for eval_name, run_eval in evals.items():
        baseline_score   = run_eval(baseline_model)
        quantized_score  = run_eval(quantized_model)
        drop = baseline_score - quantized_score
        if drop > max_per_eval_drop:
            print(f"[BLOCK] {eval_name}: {baseline_score:.2f} → {quantized_score:.2f}")
            return False

    disagree = shadow_eval_disagreement(quantized_model, baseline_model, prod_shadow_set)
    if disagree > max_shadow_disagreement:
        print(f"[BLOCK] shadow disagreement {disagree:.1%} > {max_shadow_disagreement:.1%}")
        return False

    print("[OK] quantized model passed the deploy gate")
    return True
```

In production this gate runs in CI before every quantized-model promotion. The exact thresholds are tuned per workload, but the *shape* — a fixed eval suite plus a shadow comparison against the current production model — is non-negotiable.

### Key Insight

Quantization is a memory-and-bandwidth optimization first and a FLOPs optimization second. The reason FP8 weights make decode faster is not "the math is half as expensive" — at batch size 1 the math is irrelevant. It's that *the weights stream from HBM in half the time*. Once you internalize this, the serving design space becomes obvious: anything bandwidth-bound (decode, attention, KV cache reads) wants to be in the smallest precision that doesn't tank quality; anything compute-bound (prefill, large-batch GEMM) gets less benefit from low precision. Allocate your bits accordingly, and **gate every deploy with a real eval** — quantization regressions are silent and they compound.

### Resources

- [Lin et al. — *AWQ: Activation-aware Weight Quantization* (2023)](https://arxiv.org/abs/2306.00978)
- [Frantar et al. — *GPTQ* (2022)](https://arxiv.org/abs/2210.17323)
- [Xiao et al. — *SmoothQuant* (2022)](https://arxiv.org/abs/2211.10438)
- [NVIDIA — *FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433)
- [Marlin kernels](https://github.com/IST-DASLab/marlin) — fast INT4×FP16 GEMM for inference
- [TensorRT-LLM quantization docs](https://nvidia.github.io/TensorRT-LLM/) — the production reference on NVIDIA
- [AI Hardware Phase 7](../ai-hardware/#phase-7-numeric-formats-and-quantization) — the underlying format theory

---

## Phase 6: Inference-Relevant Kernels and Hardware Choices

Underneath the inference engine, every forward pass eventually becomes a sequence of CUDA / ROCm / Triton kernels running on a real piece of silicon. You can ship a system without writing a kernel yourself, but if you want to *understand* a profiler trace — and you will, the day a latency regression lands on you — you need a working mental model of which kernels matter for serving and why.

> The general theory of GPU architecture, the memory hierarchy, and how to write kernels from scratch is the subject of [AI Hardware Phases 2–4](../ai-hardware/#phase-2-gpu-architecture-inside-out) and [PyTorch Deep Dive Phase 6](../pytorch-deep-dive/#phase-6-custom-kernels--c-cuda-and-triton-extensions). This phase assumes that foundation and focuses on the *inference-relevant subset*: the kernels and hardware choices that move the two numbers in Phase 0.

### Concepts to Learn

- **The inference roofline picture** — apply the AI Hardware roofline model to the *serving* mix:
  - Single-user decode lives deep in the memory-bound regime — `tok/s ≈ HBM_bandwidth / (2 · N_params)`; for a 70B model on H100 (~3 TB/s) that's ~21 tok/s single-stream
  - Large-batch decode and prefill move you right, toward the compute ceiling
  - Buying decisions follow: if your workload is decode-heavy single-user, buy HBM bandwidth; if it's prefill/batch-heavy, buy FLOPs
- **FlashAttention (1/2/3)** — the canonical attention kernel for both training and inference. Standard attention materializes a `T × T` score matrix in HBM; FlashAttention tiles the computation, keeps the scores in SRAM, and never writes them out. **Every modern inference engine uses it** — you should be able to read it, even if you never write it
- **FlashDecoding / FlashAttention-decode** — the decode-specific variant: one query token per request, many KV positions to read. Reorganizes the KV-read pattern to keep HBM bandwidth saturated. The reason vLLM and friends hit ~80% of peak HBM bandwidth on decode-heavy mixes
- **GEMM shapes that show up in serving** — the matmul library doesn't know about "inference," but the shapes it sees do:
  - **Skinny decode GEMM** — `M = batch (often 1–32)`, `K = d_model`, `N = d_model`. Tiny M means Tensor Cores under-utilized; the right kernel is a *GEMV-like* path, not a generic GEMM
  - **Prefill GEMM** — `M = batch × seq_len (thousands)`, big enough that standard cuBLAS / CUTLASS GEMM kernels are happy
  - **Mixed-precision GEMM** — INT4 weights × BF16 activations is the common quantized path; Marlin and CUTLASS have specialized kernels
  - **Grouped GEMM** — per-LoRA-adapter and per-MoE-expert work; one kernel launch handles many small matmuls
- **CUDA Graphs for the decode loop** — decode launches dozens of tiny kernels per token; launch overhead becomes a noticeable share of wall-clock. Capturing the per-token sequence as a CUDA Graph and replaying it saves 5–20% on small models. Production engines do this by default
- **Kernel fusion for inference-specific patterns**:
  - RMSNorm + linear + (optional) SwiGLU fused into one kernel
  - QKV projection + RoPE + KV-cache write fused into one
  - Sampling (top-k + top-p + temperature + multinomial) fused into one kernel called once per decode step
  - These are the fusions that ship in vLLM / TRT-LLM / sglang and that you can't easily get from `torch.compile` because they cross the attention boundary
- **Streams, sync, and overlap for serving** — overlap detokenization on CPU with the next GPU forward pass; overlap KV transfer in disaggregated setups with the decode that follows it; overlap a draft model's GPU work with the target's HBM read in speculative decoding. The difference between a well-pipelined serving stack and a serial one is often 20–40% throughput
- **Picking the right silicon for the workload** (the serving-side cut of [AI Hardware Phase 5](../ai-hardware/#phase-5-tpus-npus-and-alternative-accelerators)):
  - **Decode-heavy, latency-sensitive (chat)** — HBM bandwidth dominates. H200 (4.8 TB/s) and B200 (8 TB/s) beat H100 (3.35 TB/s) by their bandwidth ratio for single-stream decode
  - **Prefill-heavy / long-context (RAG, batch)** — FLOPs and HBM *capacity* dominate. MI300X (192 GB) and B200 (192 GB) shine for memory-fit; B200 FP4 for compute
  - **Throughput-per-dollar batch (offline workloads)** — anything you can keep saturated. Older A100/H100 nodes on spot pricing often win on $/M-token
  - **Single-user latency king** — Groq LPU and Cerebras have a real architectural edge at batch-1; less competitive at high concurrency
  - **Edge / on-device** — Apple Silicon (MLX), Jetson, mobile NPUs; very different optimization profile (unified memory, thermal throttling)

### The Inference Roofline, Re-Drawn for Serving

```
      Throughput (tokens/s/replica)
         │
         │
         │       ┌────────── compute-bound regime ──────────
   peak  │      ╱           (BIG prefill, big batches)
         │     ╱
         │    ╱
         │   ╱
         │  ╱
         │ ╱  ← arithmetic intensity = batch × seq for prefill,
         │╱       just batch for decode
         │   ────── memory-bound regime ───────
         │   (single-stream decode lives here)
         └─────────────────────────────────────────────►
                       Arithmetic intensity (FLOPs/byte)

   Two takeaways:
     1. Single-user decode lives on the bandwidth ceiling, full stop.
        Buy HBM bandwidth, not FLOPs, if that's your workload.
     2. Batching (continuous batching, big-batch prefill) pushes you right,
        toward the compute ceiling, where bigger / better silicon helps.
```

### Why Decode and Prefill Want Different Kernels

```
   Prefill kernel (one big GEMM per layer):
       M = batch × seq_len ≈ thousands
       Tensor Cores happy; standard GEMM tile shapes (128×128, 256×128) are great
       cuBLAS / CUTLASS pick these by default

   Decode kernel (one tiny GEMM per layer):
       M = batch ≈ 1–32; K, N still big
       Tensor Cores ½ idle even at peak; this is a memory-bandwidth problem
       The kernel you want is a GEMV-like skinny-M path with weights streamed
       from HBM; CUTLASS has these and cuBLAS routes to them when M is small

   Picking the wrong tile shape for the wrong phase costs 20–50% of throughput.
   Production engines specialize; toy implementations almost never do.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Roofline plot for *your* engine](projects/37-roofline-plot-for-your-engine/README.md) | Sweep batch 1–128 and prompt length 128–32k; plot throughput vs. arithmetic intensity; identify the regime each operating point lives in | ⭐⭐⭐ |
| [Profile a single decode step](projects/38-profile-a-single-decode-step/README.md) | Use Nsight Systems; identify the longest kernel, the HBM-read pattern, the launch overhead | ⭐⭐⭐⭐ |
| [FlashDecoding ablation](projects/39-flashdecoding-ablation/README.md) | Run the same model with and without the FlashDecoding code path (vLLM exposes a flag); measure decode-throughput delta | ⭐⭐⭐ |
| [Skinny-M kernel study](projects/40-skinny-m-kernel-study/README.md) | Pick a decode-shape GEMM (`M=8, K=8192, N=8192`); compare cuBLAS, a Triton implementation, and Marlin; report TFLOPs and bandwidth | ⭐⭐⭐⭐ |
| [CUDA Graphs for decode](projects/41-cuda-graphs-for-decode/README.md) | Capture the per-token decode kernel sequence as a graph; measure launch-overhead savings on a small model | ⭐⭐⭐⭐ |
| [Stream-overlap audit](projects/42-stream-overlap-audit/README.md) | Identify a serial gap (e.g., detokenization on CPU stalling the GPU); pipeline it; measure | ⭐⭐⭐⭐ |
| [Hardware comparison](projects/43-hardware-comparison/README.md) | Run the same model and benchmark on two different GPUs (or a CPU+iGPU); explain the gap from the spec sheets | ⭐⭐⭐ |

### Sample Code: Reading a Fused Decode Kernel

```python
# This is the kind of pattern you'll see in production engines.
# The fusion crosses the norm boundary so torch.compile usually can't reach it.
import triton
import triton.language as tl

@triton.jit
def rmsnorm_qkv_proj_kernel(
    x_ptr, w_norm_ptr, w_qkv_ptr,        # input, norm weights, fused QKV weight
    q_out_ptr, k_out_ptr, v_out_ptr,
    n_cols, d_head, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / n_cols + eps)
    w_norm = tl.load(w_norm_ptr + cols, mask=mask, other=0.).to(tl.float32)
    normed = (x / rms) * w_norm

    # ... project normed → Q, K, V via fused matmul (omitted for brevity)
    # then write Q, K, V directly into the KV cache slot for this position
```

The point isn't that you'll write this — it's that when you read a vLLM or sglang kernel and see `rmsnorm_qkv_rope_proj_kernel`, you'll know what each fused step does and why it's fused.

### Key Insight

Hardware does not bend to your model; your serving stack bends to the hardware. The fastest open-source inference engines are not "well-engineered Python" — they are bundles of hand-tuned kernels carefully matched to specific silicon (CUTLASS shapes for H100, TMA-aware kernels for Hopper, hand-written ROCm kernels for AMD MI300X, FP4 kernels for Blackwell). When you compare two engines on the same model, the gap is almost always at the kernel layer. Long-term, the cost-of-inference race will be won by whoever closes the gap between a model architect's intention and the silicon's actual capabilities — and the easiest way to participate is to learn to read a profiler.

### Resources

- [Dao et al. — *FlashAttention 2* (2023)](https://arxiv.org/abs/2307.08691) and [*FlashAttention 3* (2024)](https://arxiv.org/abs/2407.08608)
- [FlashDecoding blog (Dao et al., 2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [NVIDIA *CUTLASS* docs and examples](https://github.com/NVIDIA/cutlass)
- [OpenAI *Triton* docs](https://triton-lang.org/)
- [NVIDIA *Nsight Systems* user guide](https://docs.nvidia.com/nsight-systems/)
- [`gpu-mode` Discord and lecture series](https://github.com/gpu-mode/lectures) — the inference-kernel community
- [AI Hardware Phases 2–4](../ai-hardware/) — for the architecture and kernel-writing foundation this phase builds on

---

## Phase 7: Distributed and Disaggregated Serving

A 70B model fits on a single high-end GPU; a 400B+ MoE doesn't. Beyond a single device, inference becomes a distributed-systems problem: how to split the model across GPUs, how to route requests, how to keep the KV cache coherent. This is also where the operational complexity of a real production stack lives.

> The distributed-*training* playbook (DDP, FSDP, ZeRO, training-time TP/PP) belongs to [PyTorch Deep Dive Phase 7](../pytorch-deep-dive/#phase-7-distributed-training--ddp-fsdp-and-beyond) and [LLM Phase 4](../llm/#phase-4-scaling-laws-and-training-infrastructure). The trade-offs are different at serving time: batch sizes are small, requests are heterogeneous, KV cache is shared state, and SLOs replace MFU as the objective. This phase is the inference-specific cut.

### Concepts to Learn

- **Tensor parallelism (TP) for inference** — sharded layer weights, all-reduce after attention and after the MLP. The serving twist: batch sizes are small, so per-iteration communication is a larger share of wall-clock than in training. TP up to 8 within an NVLink node is fine; across nodes the communication dominates
- **Pipeline parallelism (PP) for inference** — less common than for training because it adds latency (each request waits for all pipeline stages), but used for very large models that don't fit in one node. The "bubble" overhead is a tail-latency problem, not a throughput one
- **Expert parallelism (EP) for MoE** — experts distributed across GPUs; tokens routed to the GPU holding the chosen expert via all-to-all. Heavy network use, mitigated by capacity-factor tuning and expert co-location. DeepSeek-V3 and the current generation of frontier MoE checkpoints all use EP
- **Replication (data parallelism)** — many *complete copies* of the model, requests load-balanced across them. The default scaling axis for small-to-mid models. Independent fault domains, easy to reason about
- **Sharding across model parallelism + replication** — e.g., a 70B model with TP=2 (fits in 2 GPUs) replicated 4× for throughput. The most common production layout for mid-size models
- **Disaggregated prefill / decode** — separate GPU pools for prefill and decode:
  - Prefill nodes are compute-rich, sized for big GEMMs
  - Decode nodes are bandwidth-rich, sized for KV-cache reads
  - When a request finishes prefill, the **KV cache is transferred** to a decode node (over RDMA / NVLink)
  - Different aggregate ratios for different workloads — chat is more decode-heavy; RAG is more prefill-heavy
- **KV cache transfer** is the new wire protocol — for disaggregated systems, moving 2–10 GB of KV across NICs efficiently is itself a hard engineering problem. Mooncake, NIXL, and similar systems are emerging as the data plane
- **Load balancing strategies** (the routing layer that sits above the inference engines):
  - Round-robin (fine for homogeneous requests)
  - Least-outstanding-requests (better under heterogeneous lengths)
  - **Prefix-aware routing** — route to the replica whose prefix cache already has the request's system prompt. Huge TTFT win for multi-tenant systems with shared prompts
  - **Hash on session_id** for sticky sessions (preserves KV cache across turns)
- **Multi-region serving** — geo-distributed deployments add another routing axis. Use CDN-style nearest-region routing for TTFT; route to the model's home region for cold sessions
- **Failure modes**:
  - GPU dies mid-decode — the user's KV cache is gone; re-route or fail-fast
  - Disaggregated KV transfer fails — fall back to a unified node or re-prefill
  - One replica is silently slow ("gray failures") — health checks must include latency, not just liveness
  - Cache stampedes when a popular prompt evicts on all replicas at once

### Where The Pieces Sit

```
                              ┌────────────────────────────┐
   Internet ──────────────►   │       L7 Load Balancer     │
                              │  (TLS, auth, routing)      │
                              └──────────────┬─────────────┘
                                             │
                              ┌──────────────▼─────────────┐
                              │     Inference Gateway      │
                              │  (admission, prefix-route, │
                              │  rate limit, observability)│
                              └──────────────┬─────────────┘
                                             │
              ┌──────────────────────────────┼────────────────────────────┐
              ▼                              ▼                            ▼
        Prefill Pool                    Decode Pool                Spec-Draft Pool
        (compute-rich)                  (bandwidth-rich)            (optional)
        H100×4 nodes                    H200×8 nodes                small GPUs
              │   KV cache transfer (RDMA / NVLink) over inference fabric
              └──────────────────────────►◄──────────────────────────────

                              ┌─────────────────────────────┐
                              │ Weight + adapter store      │
                              │ (S3 / cluster filesystem)   │
                              └─────────────────────────────┘
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [TP=2 from scratch](projects/44-tp-2-from-scratch/README.md) | Two GPUs, manually shard a model's attention layers; verify outputs match single-GPU | ⭐⭐⭐⭐⭐ |
| [vLLM multi-replica](projects/45-vllm-multi-replica/README.md) | Stand up 4 vLLM replicas behind a simple round-robin LB; load-test with `oha`; report aggregate throughput | ⭐⭐⭐ |
| [Prefix-aware routing](projects/46-prefix-aware-routing/README.md) | Build a tiny routing layer that hashes on prompt prefix; verify cache hit rate improves | ⭐⭐⭐⭐ |
| [Disaggregated prototype](projects/47-disaggregated-prototype/README.md) | Two processes: prefill emits KV blocks, decode consumes them. Use shared memory or RDMA; measure overhead vs. unified | ⭐⭐⭐⭐⭐ |
| [Failure-mode drill](projects/48-failure-mode-drill/README.md) | Kill one replica mid-load-test; verify graceful failover; measure user-visible impact | ⭐⭐⭐⭐ |
| [Session-affinity routing](projects/49-session-affinity-routing/README.md) | Implement sticky routing on `session_id`; verify multi-turn KV cache hit rate | ⭐⭐⭐ |
| [Cross-region latency](projects/50-cross-region-latency/README.md) | Deploy in two regions, measure TTFT delta with regional routing on / off | ⭐⭐⭐ |

### Sample Code: A Prefix-Aware Router (Sketch)

```python
import hashlib
from collections import defaultdict

class PrefixRouter:
    def __init__(self, replicas, prefix_len=512):
        self.replicas = replicas
        self.prefix_len = prefix_len
        self.affinity = {}                    # prefix_hash → replica
        self.load = defaultdict(int)          # replica → outstanding requests

    def _hash(self, prompt_tokens):
        return hashlib.blake2b(
            bytes(prompt_tokens[: self.prefix_len]), digest_size=16
        ).digest()

    def pick(self, prompt_tokens):
        h = self._hash(prompt_tokens)
        if h in self.affinity:
            r = self.affinity[h]
            # Only honor affinity if the replica isn't overloaded
            min_load = min(self.load.values()) if self.load else 0
            if self.load[r] < min_load + 2:
                self.load[r] += 1
                return r
        # Fall back: least-loaded replica, remember its affinity
        r = min(self.replicas, key=lambda r: self.load[r])
        self.affinity[h] = r
        self.load[r] += 1
        return r

    def release(self, replica):
        self.load[replica] -= 1
```

### Key Insight

Distribution adds *coordination cost*. Every all-reduce, every KV-cache transfer, every routing decision is bandwidth and latency you weren't spending on tokens. The right answer is almost never "more parallelism" — it's "the smallest amount of parallelism that fits the model and lets each replica run hot." The exception is the largest frontier models, where you have no choice; even there, the engineering target is to minimize the share of wall-clock time spent on collectives. Always profile the *communication*, not just the compute.

### Resources

- [Pope et al. — *Efficiently Scaling Transformer Inference* (2022)](https://arxiv.org/abs/2211.05102) — the canonical TP/PP-for-inference paper
- [Zhong et al. — *DistServe* (2024)](https://arxiv.org/abs/2401.09670) — disaggregated prefill / decode
- [Qin et al. — *Mooncake* (2024)](https://arxiv.org/abs/2407.00079) — KV-centric serving
- [NVIDIA *NIXL*](https://developer.nvidia.com/blog/) — KV-cache transfer over the inference fabric
- [TensorRT-LLM examples](https://github.com/NVIDIA/TensorRT-LLM) — production reference for multi-GPU inference

---

## Phase 8: Long Context, Structured Output, and Multi-Tenant Tricks

Once the basics are solid, real workloads pile on requirements that don't fit cleanly into the canonical "prompt in, tokens out" picture: 1M-token contexts, JSON-schema-conformant output, many fine-tuned adapters served from one base model, retrieval prefixes shared across users. This phase is the grab bag of techniques that make a serving system *useful*, not just fast.

### Concepts to Learn

- **Long-context serving** — when the prompt is 100k–2M tokens:
  - Prefill cost grows quadratically with sequence length (attention is `O(T²)`)
  - KV cache grows linearly in `T` and is the memory dominator
  - **Context parallelism** — shard the sequence dimension across GPUs for prefill; communication during attention
  - **Ring attention / sequence parallelism** — pass KV chunks around a ring so each GPU sees the whole sequence over multiple rounds
  - **Chunked prefill** (Phase 3) is essential to keep TTFT smooth at long lengths
  - **Streaming attention with windowed eviction** (sliding window + attention sinks) — bound the cache while keeping useful long-range information
- **Retrieval-augmented serving** — most production "long context" is actually "shorter context plus retrieval":
  - Pre-encode and cache retrieved documents' KV at indexing time (KV-prefill caching)
  - Reuse the same retrieved document's cache across many queries (shared blocks via PagedAttention)
  - Trade-off: storing pre-computed KV is much bigger than storing the document text, but eliminates per-query prefill cost
  - The retrieval pipeline (chunking, embedding, reranking) itself belongs to the LLM guide; see [LLM Phase 7](../llm/#phase-7-retrieval-tools-and-agents). What this guide owns is the *serving side*: where the cache lives, how it's shared, and how a retrieved-document prefix becomes a TTFT win
- **Structured / constrained generation** — force outputs to conform to a regex, JSON schema, or formal grammar:
  - At each decode step, compute the set of token IDs that can continue a valid output, mask the rest in the logits, then sample
  - **Outlines, lm-format-enforcer, sglang regex/json modes, xgrammar** — production implementations
  - Cost: the mask computation per step. For JSON schemas, this is cheap; for arbitrary regex, it can require precomputed automata
  - Reliability: with constraints, JSON-mode failure rate drops from ~5% to ~0% on small models. The biggest reliability win in tool-calling
- **Tool-call / function-call serving** — a constrained-generation special case where the output format is a JSON schema describing a tool invocation. Most production agent stacks live here
- **Multi-LoRA serving** — one base model in GPU memory, many small LoRA adapters dynamically applied:
  - **S-LoRA / Punica / dLoRA / Lorax** — runtime engines that batch requests using different adapters in a single forward pass
  - Each adapter is small (1–100 MB); hundreds can sit in GPU memory at once
  - Per-request adapter application costs ~5–15% overhead vs. plain serving
  - The economic killer feature for SaaS deployments — one base model, thousands of fine-tunes
- **Speculative decoding for tool-using agents** — when most of the output is a fixed schema (e.g., `{"tool":"web_search","args":{...`), prompt-lookup or schema-aware speculation accepts huge spans for free
- **Quantization + LoRA + speculation interactions** — they mostly compose, but there are corner cases (e.g., a LoRA adapter trained against BF16 weights applied on top of an INT4 quantized base requires a different application path). Test every cross product
- **Stateful agent sessions** — long-running tool-using sessions that span minutes to hours. The KV cache becomes a *session resource*, not a per-request one. It needs lifecycle management (creation, renewal, eviction), backpressure (refuse new sessions when GPU is full), and graceful degradation (re-prefill on cache loss)
- **Embedding and reranker serving** — adjacent but distinct workloads: encoder models, fixed-size output, no decode loop. Different optimization profile: pure prefill, very high batch sizes, ideal for big-batch GEMM. Don't try to serve them in the same engine as decoder models unless the engine explicitly supports it

### Multi-LoRA Serving, Schematically

```
                    Request A (uses adapter-cust-42)
                    Request B (uses adapter-cust-19)
                    Request C (uses adapter-cust-42)
                    Request D (no adapter / base)
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Multi-LoRA engine    │
                  │                        │
                  │   Base weights         │  ← one copy in HBM
                  │   ◇ adapter-cust-42    │  ← small low-rank deltas
                  │   ◇ adapter-cust-19    │     (cached in HBM)
                  │   ◇ adapter-cust-77    │
                  │   ◇ ... (hundreds)     │
                  │                        │
                  │   One batched forward  │
                  │   per decode step,     │  ← uses adapter-aware GEMM
                  │   adapter selected     │     (BGMV / SGMV / Punica)
                  │   per request          │
                  └────────────────────────┘

Per-tenant fine-tunes, one base model, no replica explosion.
The unit economics of LLM SaaS were rewritten by this technique.
```

### Constrained JSON Generation, Implemented

```python
import json
from outlines import generate, models

model = models.transformers("meta-llama/Llama-3.2-1B")

schema = {
    "type": "object",
    "properties": {
        "name":   {"type": "string"},
        "age":    {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "skills"],
}

generator = generate.json(model, schema)
out = generator("Describe a fictional engineer named Ada.")
print(json.dumps(out, indent=2))
# Guaranteed valid JSON conforming to the schema, or generation
# is cut short — the model literally cannot emit anything else.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Needle-in-a-haystack](projects/51-needle-in-a-haystack/README.md) | Measure recall at increasing context lengths up to your engine's limit; identify the cliff | ⭐⭐⭐ |
| [Prefix KV caching](projects/52-prefix-kv-caching/README.md) | Pre-compute KV for 1000 retrieved documents; measure cold vs. warm TTFT delta | ⭐⭐⭐⭐ |
| [JSON-mode reliability](projects/53-json-mode-reliability/README.md) | Same prompt, same model, with and without constrained decoding; measure schema-validity rate on 1000 generations | ⭐⭐⭐ |
| [Custom grammar](projects/54-custom-grammar/README.md) | Build a regex-based grammar for a domain-specific output (e.g., SQL); enforce it at decode time | ⭐⭐⭐⭐ |
| [Multi-LoRA serving](projects/55-multi-lora-serving/README.md) | Stand up Lorax or S-LoRA; train 5 small adapters; serve them all from one base; measure throughput vs. 5 replicas | ⭐⭐⭐⭐⭐ |
| [Speculation + JSON-mode](projects/56-speculation-json-mode/README.md) | Add prompt-lookup speculation to a JSON-mode workload; measure speedup (often dramatic — schemas are highly predictable) | ⭐⭐⭐⭐ |
| [Stateful session API](projects/57-stateful-session-api/README.md) | Build a session API that keeps a multi-turn KV cache alive across calls and evicts under pressure; verify cache-hit rate | ⭐⭐⭐⭐⭐ |
| [Ring attention from scratch](projects/58-ring-attention-from-scratch/README.md) | 4-GPU ring attention; measure scaling efficiency at 64k context | ⭐⭐⭐⭐⭐ |

### Key Insight

The "table-stakes" serving features that turn a prototype into a product — long context, structured output, multi-tenant adapters, retrieval prefixes — are each a small research problem with a real production answer. The teams that ship reliable LLM systems are not the ones with the fastest decode kernel; they're the ones that got JSON-mode right, multi-LoRA right, prefix-cache right, and the agent's tool-call format right. Speed is a precondition; **reliability and breadth of supported workloads** are the actual product.

### Resources

- [Sheng et al. — *S-LoRA* (2023)](https://arxiv.org/abs/2311.03285)
- [Chen et al. — *Punica: Multi-Tenant LoRA Serving* (2024)](https://arxiv.org/abs/2310.18547)
- [Liu et al. — *Ring Attention with Blockwise Transformers* (2023)](https://arxiv.org/abs/2310.01889)
- [Willard & Louf — *Efficient Guided Generation for LLMs* (Outlines, 2023)](https://arxiv.org/abs/2307.09702)
- [Dong et al. — *xgrammar* (2024)](https://arxiv.org/abs/2411.15100) — fast structured-output engine
- [sglang docs — *RadixAttention and structured outputs*](https://github.com/sgl-project/sglang)
- [Predibase/Lorax](https://github.com/predibase/lorax) — multi-LoRA serving engine
- [Anthropic — *Prompt caching*](https://www.anthropic.com/news/prompt-caching) — productized prefix-cache concept

---

## Phase 9: Observability, SLOs, and Cost Economics

A working inference system needs to *prove* it works under load, in production, to engineers who weren't there when it was built. This means metrics, alerts, SLOs, capacity planning, and unit economics. Most of the failures of production LLM systems are not "the model gave a wrong answer" — they are "we couldn't see that P99 had quietly tripled, and we ran out of capacity at 9 a.m. on a Monday."

### Concepts to Learn

- **The metrics every LLM service should emit** (per request, aggregated):
  - **TTFT** (time to first token) — P50, P95, P99
  - **TPOT / ITL** — time per output token, P50/P95/P99
  - **End-to-end latency** — for non-streaming clients
  - **Tokens generated**, **tokens prompted** — per request, aggregated per minute
  - **Throughput** — input tokens/s, output tokens/s, requests/s
  - **Concurrency** — active requests at each instant
  - **Queue depth**, **queue wait** — request was admitted but not yet running
  - **Cache hit rate** — prefix cache hits, KV reuse, document-cache hits
  - **Speculation acceptance rate** — if applicable
  - **GPU utilization** — both compute and memory bandwidth (the latter is the one that matters, and the one most dashboards skip)
  - **KV cache occupancy** — bytes used / bytes available
  - **Per-replica health**: liveness, latency, error rate, KV-OOM events
- **SLOs (Service Level Objectives)** — quantified commitments your system makes, e.g.:
  - "P95 TTFT < 500 ms"
  - "P99 TPOT < 80 ms"
  - "Error rate < 0.1% over a rolling 30 days"
  - SLOs are *promises*, not aspirations; if you have a 99.9% SLO and you spend 0.5% of the month with elevated latency, you owe an explanation
- **Error budgets** — the inverse of an SLO. A 99.9% SLO permits 43.2 minutes of failure per month. You spend that budget on rollouts, experimentation, and risky deploys. If you've blown the budget, you stop deploying risky changes until next month
- **The two latency tails** — you have one for TTFT and one for TPOT, and they fail for different reasons. Big-prompt cold cache → TTFT spike. Cache-OOM eviction → TPOT spike. Confusing them in a postmortem is a classic mistake
- **Load shedding and admission control** — under overload, *refuse* low-priority requests early. Returning a fast `429` is much better than degrading every user
- **Capacity planning** — given a traffic forecast, decide how many replicas you need. The hard part is the long tail: most production traffic is bursty, and you need to size for the burst, not the average. A useful rule: provision for `P95 traffic × safety_factor (1.5–2×)`
- **Cost economics**:
  - **Cost per million tokens** is the universal unit
  - It breaks down into: (GPU $/hr × replicas) / (output tokens/hr aggregate)
  - Output is what users pay for; input is mostly a sunk prefill cost (and counted separately on most pricing schemes)
  - Move the cost lever by: bigger batches (cheaper per token, slower per user), smaller models (cheaper, lower quality), quantization (cheaper, slight quality risk), spot/preemptible GPUs (cheaper, harder ops), better silicon (capex)
- **Right-sizing the model**: it is almost always cheaper to serve a fine-tuned 8B model than to serve a 70B base for the same task — assuming the 8B has been trained well. Most production teams over-serve

### Cost-per-Million-Tokens, Practical Math

```
   Hardware:  H100 SXM, 8-GPU node, $32/hr (illustrative)
   Model:     Llama 3 70B FP8, TP=2, so 4 replicas per node
   Workload:  steady-state continuous batching, ~1800 output tok/s/replica

   Output tokens per hour per node:
       = 4 replicas × 1800 tok/s × 3600 s/hr
       = ~26 M output tokens/hr/node

   Cost per million output tokens:
       = $32 / 26 = $1.23 per M output tokens

   At 50% sustained utilization (which is generous for non-batch workloads):
       = $2.46 per M output tokens

   Throw in 25% overhead (gateways, observability, idle replicas, on-call buffer):
       ≈ $3.10 per M output tokens, all-in

   Quote that number with confidence intervals. Re-derive it monthly.
   Track the trajectory; production cost-per-token should be falling steadily
   (newer hardware, better engines, quantization gains, smarter batching).
```

### What A Good Dashboard Has

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  TTFT  P50 / P95 / P99      [══════════════════════════]         │
   │  TPOT  P50 / P95 / P99      [════════════              ]         │
   │  Requests/sec               [══════════════════        ]         │
   │  Tokens/sec  (in / out)     [════════════════════════  ]         │
   │  Concurrency / queue        [══════                    ]         │
   │  Cache hit rate (prefix)    [████████████████░░░░░░░░  ]  74%    │
   │  KV cache occupancy         [████████████░░░░░░░░░░░░  ]  52%    │
   │  GPU mem-bw utilization     [██████████████████░░░░░░  ]  78%    │
   │  GPU compute utilization    [██████░░░░░░░░░░░░░░░░░░  ]  29%    │
   │  Speculation acceptance     [████████████████░░░░░░░░  ]  68%    │
   │  Error rate (5xx, timeouts) [░░░░░░░░░░░░░░░░░░░░░░░░  ] 0.03%   │
   └──────────────────────────────────────────────────────────────────┘

   If your dashboard is missing the bottom four, you cannot debug a
   real production incident. Add them this week.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Metric instrumentation](projects/59-metric-instrumentation/README.md) | Wire a vLLM server to Prometheus + Grafana; ship dashboards for all metrics in the figure above | ⭐⭐⭐ |
| [Synthetic load tests](projects/60-synthetic-load-tests/README.md) | Generate workloads with realistic prompt/output length distributions; benchmark at 1×, 2×, 5× concurrency | ⭐⭐⭐ |
| [SLO simulation](projects/61-slo-simulation/README.md) | Define a P95 TTFT < 500 ms SLO; sweep arrival rate until it breaks; identify the bottleneck | ⭐⭐⭐⭐ |
| [Error budget tracker](projects/62-error-budget-tracker/README.md) | Implement a daily SLI/SLO computation that exhausts an error budget under a chosen failure mode | ⭐⭐⭐⭐ |
| [Cost report](projects/63-cost-report/README.md) | For your serving stack, produce a defensible $/M-output-tokens number; identify the top three line items | ⭐⭐⭐ |
| [Right-sizing experiment](projects/64-right-sizing-experiment/README.md) | Same workload, 7B vs. 13B vs. 70B; report quality (a real eval) and cost; recommend a tier | ⭐⭐⭐⭐ |
| [Load-shedding policy](projects/65-load-shedding-policy/README.md) | Implement priority-aware admission control; verify SLOs hold for high-prio under 2× overload | ⭐⭐⭐⭐ |
| [Postmortem drill](projects/66-postmortem-drill/README.md) | Inject a real failure (replica crash, cache thrash); run the incident; write the postmortem | ⭐⭐⭐⭐ |

### Key Insight

The cost-per-token graph and the latency-distribution graph are the two charts that decide whether your inference system has a business. Architectures, scaling laws, fancy kernels, and clever scheduling are all instruments for moving those two graphs. If you can't draw both of them from memory for your own service — with numbers, not adjectives — you have not yet finished the work, no matter how clever the rest of the stack is. **Numbers, with confidence intervals, on a dashboard you trust.** That is the deliverable.

### Resources

- [Google SRE Book — *Service Level Objectives*](https://sre.google/sre-book/service-level-objectives/) — the canonical reference, model-agnostic
- [Prometheus + Grafana docs](https://prometheus.io/docs/) — the boring infrastructure that runs the boring infrastructure
- [vLLM Prometheus integration](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [`oha`](https://github.com/hatoo/oha), [`vegeta`](https://github.com/tsenart/vegeta) — load generators
- [LMSys *Chatbot Arena* — public benchmarks on real workloads](https://lmarena.ai/)

---

## Phase 10: Frontier Topics in Serving

The state of the art in inference is moving fast. Some of what's described in this guide will look quaint in two years; some of it has been settled engineering since 2023 and is unlikely to change. This phase is the open thread of what's currently being researched, what is being deployed at the bleeding edge, and what people in the field are watching for in the next 24 months.

> The frontier of *silicon* itself (co-packaged optics, chiplets, compute-in-memory, neuromorphic, custom hyperscaler ASICs) belongs to [AI Hardware Phase 10](../ai-hardware/#phase-10-frontier-topics). This phase is the frontier of the *serving stack* that sits on top.

### Concepts and Trends to Watch

- **Reasoning-model inference economics** — [long-CoT](/shared/glossary/#long-cot) models (R1-style) generate 10× the tokens of a chat model for hard problems: "long chain-of-thought" means the model writes out its working, at length, before answering, and you pay for every one of those tokens. Inference cost balloons; serving stacks need to manage variance in output length more carefully — with a chat model you can guess how long a reply will be, with a reasoning model you cannot. Expect **adaptive [thinking budgets](/shared/glossary/#thinking-budget)**, where the model (or the server) decides at runtime how much thinking a request is allowed before it must answer
- **Inference-time scaling laws** — for a fixed model, plotting accuracy against how much compute you spend *at inference* is increasingly clean: let it think longer, or sample several answers and pick the best, and accuracy rises predictably. The frontier is no longer "bigger model" exclusively; it's "smarter use of inference tokens." Serving stacks need to expose, account, and bill for *thinking tokens* separately — a customer who is charged for 5,000 invisible tokens they never see wants to know why. The capability story lives in [LLM Phase 6](../llm/#phase-6-reasoning-and-inference-time-compute); this guide owns the *serving* implications
- **Speculative-everything** — [speculative decoding](/shared/glossary/#speculative-decoding) is the warm-up. Speculative tool calls, speculative retrieval, [speculative agent steps](/shared/glossary/#speculative-execution) are all under active research. Anything you can *guess and verify* is a candidate — with two conditions that decide whether it works: verifying must be much cheaper than doing, and a wrong guess must be **undoable**. Guessing a token costs a discarded tensor; guessing a `git commit` costs a commit
- **[Agentic inference](/shared/glossary/#agentic-inference)** — long-running, multi-turn, tool-using sessions. Serving systems must support [**stateful KV sessions**](/shared/glossary/#stateful-session) that live for minutes to hours, survive cache evictions, span retrieval stalls (the minutes an agent spends waiting for a tool, holding memory and doing nothing), and gracefully retry on tool failures. The single-prompt-single-response abstraction breaks
- **KV-cache sharing as a first-class API** — explicit prompt [prefix caching](/shared/glossary/#prefix-cache) (Anthropic's, OpenAI's, etc.) is the appetizer: the server notices a repeated prefix and reuses it. Coming: explicit user-controlled cache handles (*you* name a cache and reuse it deliberately), cross-request cache pinning, cache-as-a-service
- **Disaggregated everything** — beyond [prefill/decode separation](/shared/glossary/#disaggregated-serving), expect dedicated pools for embedding, reranking, tool execution, vision encoding, and draft models. The trend is fine-grained specialization: each job type gets machines tuned for it, and the *intermediate results* — a [KV cache](/shared/glossary/#kv-cache), an embedding — are shipped between pools over fast links instead of being recomputed
- **Serving new silicon** — [Blackwell](/shared/glossary/#blackwell) (B200/GB200) brings [FP4](/shared/glossary/#fp4) native, 8 TB/s [HBM](/shared/glossary/#hbm), and bigger fabrics; AMD MI355X and Trillium TPUs target similar; non-GPU accelerators (Groq, Cerebras, SambaNova, Etched) target *latency-per-token* as a differentiator — they are slower per dollar in bulk but answer one user faster. The cost-per-token curve will move sharply through 2026–2027 — but only for stacks that retune kernels and schedulers for each new fabric
- **Edge and on-device inference** — 1–8B models on phones, laptops, and embedded silicon. Apple's [MLX](/shared/glossary/#mlx), NVIDIA's TensorRT-LLM-on-Jetson, Qualcomm's NPUs, Intel's NPU stack. The same [KV-cache](/shared/glossary/#kv-cache) and [quantization](/shared/glossary/#quantization) principles apply, with extra constraints around battery and [unified memory](/shared/glossary/#unified-memory) — on a phone the model shares one pool of RAM with the operating system and every other app
- **Sub-1B "tiny" models as routers** — most queries don't need a frontier model. Cheap [routers](/shared/glossary/#router-model) that decide *which* model serves each query are increasingly the front of the stack. The hard part is not the routing, it is knowing which requests the big model would actually answer *better* — see [project 69](projects/69-router-model/README.md)
- **Mixture-of-Experts serving at scale** — [MoE](/shared/glossary/#moe) models (DeepSeek-V3, Qwen3-MoE, frontier MoE checkpoints) require [expert-parallelism](/shared/glossary/#expert-parallelism-ep), [all-to-all communication](/shared/glossary/#all-to-all-token-routing) on every token, and very careful [capacity-factor](/shared/glossary/#capacity-factor) tuning — the buffer size each expert gets, which decides whether tokens get silently dropped. Best practices are still solidifying
- **Trust and security for inference** — confidential computing (NVIDIA H100/H200 confidential, AMD SEV-SNP) encrypts a running job's memory so that even the machine's owner cannot read the weights, which is what lets a customer serve a model on infrastructure they do not trust; cryptographic attestations of which model served a request; sandboxing tool execution
- **The unbundling of "serving"** — what is one platform today is splintering into separately bought pieces: pure inference engines, schedulers, gateways, observability, eval-in-the-loop, fine-tune services. Buy-vs-build decisions get more numerous

### A Map of What's Open and What's Settled

```
   Settled (don't reinvent):                    Open (worth research):
   ──────────────────────────                   ─────────────────────────
   continuous batching                          stateful agent KV sessions
   PagedAttention                               cross-replica KV sharing
   FlashAttention 2/3                           speculative-decoding ceilings
   BF16/FP8 weight serving                      MoE expert-parallelism best practices
   speculative decoding                         disaggregated routing topologies
   prefix caching                               adaptive thinking budgets
   AWQ/GPTQ INT4                                FP4 production deployment
   multi-LoRA                                   cache-as-a-service APIs
   structured generation                        confidential-compute inference
   continuous-batching schedulers               on-device frontier-class models
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Reasoning-model serving](projects/67-reasoning-model-serving/README.md) | Serve a long-CoT model; measure output-token variance, design a thinking-budget knob | ⭐⭐⭐⭐ |
| [Stateful sessions](projects/68-stateful-sessions/README.md) | Build a session API that preserves KV cache across turns, evicts cleanly under pressure | ⭐⭐⭐⭐⭐ |
| [Router model](projects/69-router-model/README.md) | Train (or prompt) a tiny model to route between a 1B "fast path" and a 70B "slow path"; measure quality and cost | ⭐⭐⭐⭐ |
| [MoE serving](projects/70-moe-serving/README.md) | Stand up a Mixtral or DeepSeek MoE; measure expert-imbalance under your workload | ⭐⭐⭐⭐⭐ |
| [FP4 (Blackwell) inference](projects/71-fp4-blackwell-inference/README.md) | If hardware available, benchmark FP4 weights + activations against FP8; measure quality | ⭐⭐⭐⭐⭐ |
| [On-device build](projects/72-on-device-build/README.md) | Compile a 3B model to MLX / TensorRT-LLM-Jetson / GGUF; measure tokens/sec on a real device | ⭐⭐⭐⭐ |
| [Speculative agent steps](projects/73-speculative-agent-steps/README.md) | In an agent loop, speculatively execute the most likely next tool call; verify and roll back if wrong | ⭐⭐⭐⭐⭐ |

### Key Insight

The frontier of inference systems is not bigger models — it is **smarter use of the serving substrate**: more sharing across requests, more speculation across phases, more specialization across silicon, more state across turns. The next decade of "what wins" will come from the engineers who treat inference not as a downstream consequence of a trained model, but as a *first-class system* in its own right — with its own architectures, its own scaling laws, and its own research agenda.

### Resources

- [DeepSeek-V3 technical report (2024)](https://arxiv.org/abs/2412.19437) — MoE inference at scale, with serving-side details
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — reasoning-model inference patterns
- [NVIDIA Blackwell architecture announcements](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [Apple MLX](https://github.com/ml-explore/mlx) — on-device inference framework
- [Anthropic — *Prompt caching*](https://www.anthropic.com/news/prompt-caching) and [OpenAI prompt-caching docs](https://platform.openai.com/docs/guides/prompt-caching) — production cache-as-a-service designs
- [Confidential Computing Consortium](https://confidentialcomputing.io/)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–1 week | Tools installed; profiler runs; roofline thinking internalized |
| 1. Request anatomy | 3–5 days | Manual decode loop + streaming server in production-ish shape; non-LLM workload map drawn |
| 2. KV cache | 1 week | Cache size in head; paged-cache prototype; prefix-share measured |
| 3. Batching | 1–2 weeks | Continuous-batching simulator; one chunked-prefill experiment |
| 4. Speculative decoding | 1 week | Working spec-decode loop; acceptance and speedup measured |
| 5. Serving-time quantization | 1–2 weeks | INT4 + FP8 deployments served; quality regression gate gating them |
| 6. Inference kernels & hardware | 1–2 weeks | Profiler trace explained line by line; roofline plot for your engine |
| 7. Distributed serving | 2 weeks | Multi-replica deploy; prefix-aware routing; disaggregated PoC |
| 8. Long context + multi-LoRA | 1–2 weeks | Structured JSON-mode shipped; multi-LoRA serving stood up |
| 9. Observability + cost | 1 week | Dashboards live; $/M-tokens defensible; SLOs published |
| 10. Frontier | Ongoing | Picked one thread (MoE serving, reasoning models, edge) and going deep |

**Total to "comfortable inference engineer":** ~2–3 months of focused study + one real production deploy. To "leading an inference platform team": 9–18 months and at least one full SLO-driven incident cycle.

---

## Key Advice

1. **Measure before you optimize, and measure again after.** The number of "obvious wins" that don't move the needle (or that move the wrong number) is humbling. Tokens/sec and P99 latency are not the same axis; check both on every change.
2. **The KV cache is the working set.** Half of the entire serving stack is about managing it. If a change doesn't either shrink the cache, reuse the cache, or schedule around the cache, it's probably less important than you think.
3. **Match training and inference exactly.** Chat templates, BOS/EOS, special tokens. The "model is dumber in production" stories almost always trace here. Inference-time prompts must be byte-identical to training-time format.
4. **Profile in production-like conditions, not benchmark-perfect ones.** Single-stream benchmarks lie. Real workloads have heterogeneous lengths, concurrent users, jagged arrival distributions, and clients that disconnect. Build your load generator to look like reality.
5. **Speculative decoding is the rare "free" win.** Add it early, tune `k`, measure acceptance per workload. Then add prompt-lookup for copy-heavy workloads.
6. **Don't over-quantize, and gate every deploy.** FP8 is safe; INT4 needs a quality gate; sub-4-bit needs a real ablation. Re-evaluate every time you change the recipe.
7. **Run a smaller model, well-tuned, before reaching for a bigger one.** Most production teams over-serve. A fine-tuned 8B that fits in one GPU beats a 70B on three GPUs for many real tasks.
8. **One base + many adapters, not many replicas.** Multi-LoRA is the default architecture for any SaaS LLM platform with per-tenant fine-tunes. Adopt it before the replica count gets embarrassing.
9. **Build the observability before you need it.** Dashboards added during an incident are dashboards built badly. Day-1 metrics list: TTFT, TPOT, throughput, cache occupancy, GPU memory-bandwidth utilization, error rate.
10. **Budget for the long tail.** P50 is for marketing; P99 is for engineering. A great P50 with a terrible P99 means most of your users have one wonderful experience and the rest churn.
11. **Cost-per-million-tokens is the unit, denominate in it.** A 5% TPOT win that costs 30% more dollars is a loss. A 30% cost win that costs 5% on quality is usually a win.
12. **Read the engines.** vLLM, sglang, TensorRT-LLM, TGI. Reading their source — especially their schedulers and KV-cache managers — teaches more than any paper.

---

## Common Pitfalls

- ❌ Benchmarking with batch size 1 and shipping for batch size 64 → wildly wrong expectations
- ❌ Optimizing prefill kernels when the workload is decode-heavy (or vice versa) → no production win
- ❌ Quantizing without re-evaluating → silent quality regression discovered weeks later
- ❌ Forgetting GQA's KV-cache savings when sizing memory → 4–8× over-provisioning
- ❌ Static batching in production → padding waste + head-of-line blocking + bad tail latency
- ❌ No chunked prefill → one 30k-token prompt freezes ITL for every other user
- ❌ No admission control → request floods turn into queue floods which turn into timeout floods
- ❌ Treating GPU compute utilization as the throughput KPI → mostly meaningless for decode; memory-bandwidth utilization is the right one
- ❌ Speculative decoding with too small a target model → draft overhead dominates, system gets slower
- ❌ Speculation acceptance rate < 50% and not investigating why → workload mismatch, fix the draft
- ❌ Mismatched chat templates between training and serving → quality cliff that looks like "the model is dumb"
- ❌ Detokenizer that breaks on multi-byte UTF-8 → garbled output on non-Latin scripts
- ❌ Streaming tokens but not flushing the SSE buffer → user sees a one-shot dump instead of a stream
- ❌ Forgetting client cancellation → server keeps generating after the user is gone, wasting GPU time
- ❌ No replica health checks beyond liveness → "gray failures" cost SLOs silently
- ❌ KV-cache OOM under burst load → cascading failures because evictions cause re-prefill which causes more contention
- ❌ Mixing embedding/rerank/decode in one engine → throughput cliff because the optimal batch sizes are wildly different
- ❌ Believing a paper's "10× speedup" figure without checking which baseline it's against
- ❌ Reporting $/M-input-tokens to optics-conscious leadership but billing $/M-output-tokens to customers → the units don't agree and somebody is paying for it

---

## Additional Resources

### Papers Everyone Cites
- [Pope et al. — *Efficiently Scaling Transformer Inference* (2022)](https://arxiv.org/abs/2211.05102)
- [Dao et al. — *FlashAttention* (2022)](https://arxiv.org/abs/2205.14135), [*FlashAttention-2* (2023)](https://arxiv.org/abs/2307.08691), [*FlashAttention-3* (2024)](https://arxiv.org/abs/2407.08608)
- [Kwon et al. — *PagedAttention / vLLM* (2023)](https://arxiv.org/abs/2309.06180)
- [Yu et al. — *Orca* (2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Leviathan et al. — *Speculative Decoding* (2023)](https://arxiv.org/abs/2211.17192)
- [Cai et al. — *Medusa* (2024)](https://arxiv.org/abs/2401.10774)
- [Li et al. — *EAGLE* (2024)](https://arxiv.org/abs/2401.15077)
- [Frantar et al. — *GPTQ* (2022)](https://arxiv.org/abs/2210.17323)
- [Lin et al. — *AWQ* (2023)](https://arxiv.org/abs/2306.00978)
- [Xiao et al. — *SmoothQuant* (2022)](https://arxiv.org/abs/2211.10438)
- [Agrawal et al. — *Sarathi-Serve / Chunked Prefill* (2024)](https://arxiv.org/abs/2403.02310)
- [Zhong et al. — *DistServe* (2024)](https://arxiv.org/abs/2401.09670)
- [Qin et al. — *Mooncake* (2024)](https://arxiv.org/abs/2407.00079)
- [Sheng et al. — *S-LoRA* (2023)](https://arxiv.org/abs/2311.03285)
- [Zhang et al. — *H2O* (2023)](https://arxiv.org/abs/2306.14048)
- [Xiao et al. — *StreamingLLM / Attention Sinks* (2023)](https://arxiv.org/abs/2309.17453)
- [Dong et al. — *xgrammar* (2024)](https://arxiv.org/abs/2411.15100)

### Engines You Should Read
- [`vllm`](https://github.com/vllm-project/vllm) — the reference open-source inference engine; the cleanest scheduler/cache code
- [`sglang`](https://github.com/sgl-project/sglang) — RadixAttention + structured-generation excellence
- [`TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's production engine; fast on NVIDIA silicon, harder to read
- [`text-generation-inference` (TGI)](https://github.com/huggingface/text-generation-inference) — HuggingFace's production engine; great middle ground
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) — single-machine, edge-friendly, very fast quantized inference
- [`MLC-LLM`](https://github.com/mlc-ai/mlc-llm) — cross-platform compiled inference (mobile, web, edge)
- [`MLX`](https://github.com/ml-explore/mlx) — Apple Silicon-native; the on-device reference

### Tools and Libraries
- [Triton](https://triton-lang.org/) — GPU kernel DSL
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's GEMM template library
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — quick PTQ + QLoRA
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — PTQ toolkits
- [Outlines](https://github.com/dottxt-ai/outlines), [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer), [xgrammar](https://github.com/mlc-ai/xgrammar) — structured generation
- [Lorax](https://github.com/predibase/lorax), [Punica](https://github.com/punica-ai/punica) — multi-LoRA serving
- [oha](https://github.com/hatoo/oha), [vegeta](https://github.com/tsenart/vegeta) — load generators
- [Nsight Systems](https://developer.nvidia.com/nsight-systems), [Nsight Compute](https://developer.nvidia.com/nsight-compute) — NVIDIA profilers

### Companion Guides in This Collection
- [LLM](../llm/) — model architecture, training, post-training, evaluation; the upstream of every model you serve
- [AI Hardware](../ai-hardware/) — GPU/TPU architecture, kernel writing, numeric formats, the silicon substrate
- [PyTorch Deep Dive](../pytorch-deep-dive/) — autograd, `torch.compile`, profiling, custom kernels, distributed training
- [Multimodal Learning](../multimodal-learning/) — VLM, embedding, and any-to-any serving patterns
- [Image Generation](../image-generation/) and [Video Generation](../video-generation/) — diffusion serving (no KV cache, different optimization profile)
- [Robotics](../robotics/) — real-time policy inference at the latency end of the spectrum

### Talks Worth Watching
- [Karpathy — *State of GPT*](https://www.youtube.com/watch?v=bZQun8Y4L2A) — includes inference-side intuition
- [Woosuk Kwon — *vLLM*](https://www.youtube.com/results?search_query=vllm+talk) — the PagedAttention story
- [Tri Dao — *FlashAttention*](https://www.youtube.com/results?search_query=tri+dao+flashattention) — the canonical kernel
- [GPU Mode lectures](https://github.com/gpu-mode/lectures) — kernel-deep talks on attention, GEMMs, quantization

### Communities
- [GPU Mode Discord](https://github.com/gpu-mode) — the inference / kernel community
- [vLLM Slack](https://docs.vllm.ai/en/latest/community/contact.html)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — practitioner-heavy, especially for edge / quantization
- [EleutherAI Discord](https://www.eleuther.ai/) — open-source research, including inference

---

## Quick Start Checklist

- [ ] Can explain prefill vs. decode, including which side of the roofline each lives on
- [ ] Can compute KV-cache size for any model from its config in 30 seconds
- [ ] Can name the right serving pattern (autoregressive LLM, diffusion, embedding, VLM, robot policy) for a given workload
- [ ] Have implemented a manual decode loop with a KV cache, end to end
- [ ] Have stood up a streaming server with TTFT and ITL metrics
- [ ] Understand why continuous batching beats static batching by >10×
- [ ] Have run a chunked-prefill experiment and seen the tail-latency win
- [ ] Have implemented (or carefully read) a paged KV cache
- [ ] Have measured prefix-cache hit rate and TTFT delta on a real workload
- [ ] Have implemented greedy speculative decoding and measured acceptance rate
- [ ] Have quantized a model to INT4 (or FP8) and re-run a real quality eval through a deploy gate
- [ ] Have profiled a single decode step in Nsight and explained the longest kernel
- [ ] Have read a non-trivial scheduler implementation (vLLM, sglang, TGI)
- [ ] Have stood up a multi-replica deployment with a prefix-aware router
- [ ] Have run a multi-LoRA serving setup with at least 3 adapters
- [ ] Have shipped a constrained-JSON generation flow and measured reliability
- [ ] Have a dashboard with TTFT, TPOT, throughput, KV-cache occupancy, memory-bandwidth utilization
- [ ] Have written down (with confidence intervals) the $/M-output-tokens for your stack
- [ ] Have run an SLO-breaking load test and identified the bottleneck before it shipped
- [ ] Can read a contemporary inference paper and explain which production lever it pulls

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
