# Multimodal Learning: From Beginner to Advanced

A comprehensive guide to understanding and building systems that learn from and reason across **multiple modalities** — text, images, audio, video, and beyond — from contrastive pretraining to modern vision-language models and unified any-to-any architectures.

> **"Multimodal learning"** is the slice of machine learning that operates on more than one input/output type. A model that reads an image and writes a caption is multimodal. A model that listens to audio and produces text is multimodal. A model that takes a text prompt and a reference image and produces a video is *very* multimodal. This guide is about how those systems learn a **shared representation** across modalities, how to train them, and where the field is going.

### Scope and boundaries

This guide owns the problem of **getting two or more modalities to share one representation** — aligning them, fusing them, and reasoning jointly over them. To keep the AI Learning Guides mutually exclusive and collectively exhaustive (MECE), it deliberately stops at a few borders and links forward to the guide that owns each one.

**In scope — this guide owns these topics:**
- **Contrastive / aligned pretraining** (CLIP, SigLIP, ImageBind) — making separate modality spaces comparable
- **Fusion architectures** — cross-attention, Q-Former, projectors, gated attention, interleaved/early fusion
- **Vision-language models (VLMs)** — image(+text)→text understanding, VQA, grounding, OCR-heavy models
- **Audio and speech as modalities** — encoders, ASR/TTS, audio LMs, neural audio codecs (no dedicated audio guide exists, so multimodal is the home for them)
- **Video, audio, and image *understanding*** — encoding non-text modalities for joint reasoning
- **Unified / any-to-any models** — the "one transformer for every modality" framing (Chameleon, GPT-4o, Gemini, Janus, Transfusion)
- **Multimodal-specific training, data, alignment, and evaluation** — the parts that differ from the single-modality recipes

**Out of scope — deferred to the owning guide:**
- *Transformer/LLM architecture, tokenization, pretraining, post-training, and text-only agents* → [LLM](../llm/). This guide *uses* a pretrained LLM as a frozen (or fine-tuned) backbone but does not re-teach how to build one.
- *Generative modeling of images* (VAEs, GANs, diffusion, latent diffusion, DiT, flow matching) and *image tokenizers* (VQ-VAE, VQ-GAN, FSQ) → [Image Generation](../image-generation/). When a multimodal model needs to *produce* pixels, the generation machinery lives there; here we care about the cross-modal modeling.
- *Generative modeling of video* (video diffusion, 3D VAEs, world models) → [Video Generation](../video-generation/). This guide owns video *understanding*; video *synthesis* is theirs.
- *Vision-Language-Action models as a robot policy class, imitation learning, sim-to-real* → [Robotics Phase 8](../robotics/#phase-8-learning-for-robotics). We cover the multimodal-modeling side of VLAs and link out for the control side.
- *RLHF / DPO / GRPO algorithm internals* → [RL Phase 9](../reinforcement-learning/#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr) and [LLM Phase 5](../llm/#phase-5-post-training--sft-rlhf-dpo-grpo-rlvr). We cover what is *different* about preference data with image/audio inputs.
- *Serving, batching, KV-cache, and inference latency* for deployed multimodal models → [Inference Systems](../inference-systems/). *Kernel-level performance and quantization* → [AI Hardware](../ai-hardware/).
- *Tensor, autograd, mixed-precision, distributed-training, and training-loop fundamentals* → [PyTorch Deep Dive](../pytorch-deep-dive/).

When this guide touches an out-of-scope topic, it does so only to the depth needed to make a multimodal modeling decision, and it links to the owning guide.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Foundations — What "Multimodal" Actually Means](#phase-1-foundations--what-multimodal-actually-means)
3. [Phase 2: Encoders for Each Modality](#phase-2-encoders-for-each-modality)
4. [Phase 3: Contrastive Learning — CLIP and Friends](#phase-3-contrastive-learning--clip-and-friends)
5. [Phase 4: Fusion Architectures — How Modalities Talk to Each Other](#phase-4-fusion-architectures--how-modalities-talk-to-each-other)
6. [Phase 5: Vision-Language Models (VLMs)](#phase-5-vision-language-models-vlms)
7. [Phase 6: Audio, Speech, and Video](#phase-6-audio-speech-and-video)
8. [Phase 7: Unified and Any-to-Any Models](#phase-7-unified-and-any-to-any-models)
9. [Phase 8: Training at Scale — Data, Compute, and Alignment](#phase-8-training-at-scale--data-compute-and-alignment)
10. [Phase 9: Evaluation and Benchmarks](#phase-9-evaluation-and-benchmarks)
11. [Phase 10: Frontier Topics](#phase-10-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
15. [Additional Resources](#additional-resources)
16. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

Multimodal learning sits on top of two stacks (NLP and vision) and borrows from a third (audio). You do not need to be an expert in all of them, but the foundations cannot be skipped.

### Concepts to Know

- **Transformers**: self-attention, cross-attention, positional embeddings, layer norm, residual connections — from the [LLM guide Phase 2](../llm/#phase-2-the-transformer-architecture)
- **Vision basics**: convolution, ViT (Vision Transformer), how an image becomes a sequence of tokens
- **Text basics**: tokenization (BPE, SentencePiece), language modeling, the next-token-prediction objective — see [LLM Phase 1](../llm/#phase-1-tokenization-and-embeddings)
- **PyTorch fluency**: `nn.Module`, autograd, mixed precision, basic training loops — see [PyTorch Deep Dive](../pytorch-deep-dive/)
- **Embedding spaces**: what an L2-normalized vector looks like, cosine similarity, the geometry of high-dimensional spaces
- **Contrastive intuition** (helpful but not required yet): pulling similar things together, pushing different things apart

> **What you do *not* need yet.** You don't need diffusion or GAN internals to start — those are the [Image Generation guide](../image-generation/)'s territory, and you only need them once you want a model that *outputs* pixels (Phases 7 and 10 here). You also don't need RLHF internals; Phase 8 covers only what's *different* about multimodal preference data.

### The One Equation Everything Comes Back To

```
Multimodal learning = map every modality into a SHARED representation,
                      then either:
                        (a) compare them (retrieval, classification), or
                        (b) generate one from the other (caption, image-from-text), or
                        (c) reason jointly over all of them (VQA, agents).

The shared representation can be:
  - a single vector per item     (CLIP-style)
  - a sequence of tokens         (LLaVA / VLM-style)
  - a discrete token alphabet    (any-to-any, e.g. Chameleon, Gemini-style)
```

### Resources

- [Transformers from Scratch](https://e2eml.school/transformers.html) — Brandon Rohrer
- [Andrej Karpathy — Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [ViT paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — required reading
- [CLIP paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) — read this before Phase 3

---

## Phase 1: Foundations — What "Multimodal" Actually Means

Before architectures, get the conceptual map right. "Multimodal" is a fuzzy umbrella term that hides at least four distinct problems.

### Concepts to Learn

- **The four canonical tasks**:
  - **[Cross-modal retrieval](/shared/glossary/#cross-modal-retrieval)** — given an image, find the matching caption (or vice versa)
  - **Cross-modal generation** — given text, produce an image (or audio, or video). *The generative half lives in [Image Generation](../image-generation/) / [Video Generation](../video-generation/); here we care about the conditioning and the cross-modal interface.*
  - **Multimodal understanding** — given image + text, answer a question ([VQA](/shared/glossary/#vqa-visual-question-answering), captioning)
  - **Joint/any-to-any** — flexibly map any subset of [modalities](/shared/glossary/#modality) to any other
- **[Modality gap](/shared/glossary/#modality-gap)** — even well-trained models keep text and image [embeddings](/shared/glossary/#embedding) in noticeably different regions of the shared space. Concretely: a photo scores about 0.30 [cosine similarity](/shared/glossary/#cosine-similarity) against *its own* caption but about 0.49 against a completely unrelated photo, so raw [CLIP](/shared/glossary/#clip) scores tell you almost nothing on their own — only rankings do. Project [02](projects/02-visualize-the-modality-gap/README.md) measures this and traces it back to the [cone effect](/shared/glossary/#cone-effect), which is present *before* any training.
- **Alignment vs fusion** — alignment is *making spaces comparable* (so a photo and its caption can be scored against each other); fusion is *combining information* (so the model can reason over both at once). A [dual encoder](/shared/glossary/#dual-encoder) only aligns; a [VLM](/shared/glossary/#vlm) fuses.
- **Early vs late [fusion](/shared/glossary/#fusion-earlymiddlelate)** — "early/middle/late" refers to *how far into the network* the two modalities are first allowed to meet:
  - *Late fusion*: encode each modality separately, combine at the end ([CLIP](/shared/glossary/#clip)-style)
  - *Early fusion*: interleave modalities into one sequence from the start ([Chameleon](/shared/glossary/#chameleon)-style)
  - *Middle fusion*: encode separately, then attend across ([Flamingo](/shared/glossary/#flamingo), [LLaVA](/shared/glossary/#llava))
- **Pretraining objectives**: [contrastive](/shared/glossary/#infonce) (rank the true pair above the wrong ones), masked (hide part of the input, predict it), generative (predict the next [token](/shared/glossary/#token-visualaudio)), and combinations

### A Taxonomy Diagram

```
                          MULTIMODAL MODELS
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   DUAL-ENCODER           ENCODER-DECODER             UNIFIED
   (alignment)            (understanding)         (any-to-any)
        │                        │                        │
   CLIP, SigLIP,           Flamingo, BLIP-2,         Chameleon,
   ImageBind               LLaVA, Qwen2.5-VL,         Gemini, GPT-4o,
                           PaliGemma                  Janus, Transfusion
        │                        │                        │
   Best at:                Best at:                  Best at:
   - retrieval             - VQA                     - everything
   - zero-shot             - captioning              - but expensive
     classification        - dialogue                  to train
   - data filtering        - reasoning
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Modality survey](projects/01-modality-survey/README.md) | Pick 5 multimodal papers, classify each by fusion type and objective; write a one-paragraph summary of each | ⭐ |
| [Visualize the modality gap](projects/02-visualize-the-modality-gap/README.md) | Encode 1k images and 1k captions with a pretrained CLIP; PCA them; observe the separation | ⭐⭐ |
| [Toy retrieval](projects/03-toy-retrieval/README.md) | Build a tiny retrieval system: encode a few hundred images and captions with CLIP, retrieve top-5 for each query | ⭐⭐ |

### Key Insight

The choice of fusion strategy determines what your model can do. [Dual encoders](/shared/glossary/#dual-encoder) (CLIP) are fast and great at retrieval but can't reason or generate. Encoder-decoder [VLMs](/shared/glossary/#vlm) (LLaVA) reason and generate text but not images. Unified models (Chameleon, GPT-4o) do everything but need vastly more data and compute. There is no free lunch; pick the architecture that matches your task.

Two consequences of that "fast" claim, since it is easy to skim past. A dual encoder is fast *at query time* because each item is encoded once, in advance, and answering a query is then one [matrix multiplication](/shared/glossary/#matmul) over stored vectors — a million candidates cost one matmul, not a million forward passes. A VLM has no such precomputable per-item vector, so scoring a million candidates means running the whole model a million times *per query*. And "can't generate" is structural, not a missing feature: a dual encoder's output is a single vector, and a single vector is not a sequence, so there is nothing for a decoder to read off. Project [01](projects/01-modality-survey/README.md) lays this out as a grid where each architecture's position predicts exactly which jobs it can do.

### Resources

- [On the Opportunities and Risks of Foundation Models (Stanford CRFM)](https://arxiv.org/abs/2108.07258) — long but the multimodal sections are excellent context
- [A Survey on Multimodal Large Language Models (Yin et al., 2024)](https://arxiv.org/abs/2306.13549)
- [A Survey on Vision-Language Models (Zhang et al., 2024)](https://arxiv.org/abs/2304.00685)
- [Hugging Face — Vision Language Models Explained](https://huggingface.co/blog/vlms)

---

## Phase 2: Encoders for Each Modality

Before you can fuse modalities, you have to encode each one into vectors. This phase is about the **representation** building blocks — the encoders that turn pixels, waveforms, and frames into sequences a transformer can align.

> **MECE note.** This phase teaches encoders *as feature extractors for alignment and understanding*. The *generative* backbones that turn latents back into pixels (U-Nets, DiTs) belong to [Image Generation](../image-generation/) and [Video Generation](../video-generation/). The *discrete tokenizers* (VQ-VAE, VQ-GAN, FSQ, MagViT-v2) are taught in [Image Generation Phase 3](../image-generation/#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers); here we *use* them (Phase 7) and only summarize.

### Concepts to Learn

- **Image encoders**:
  - [CNNs](/shared/glossary/#cnn) ([ResNet](/shared/glossary/#resnet), EfficientNet) — still useful, especially for small models
  - [Vision Transformers](/shared/glossary/#vit) (ViT) — the modern default; how [patchification](/shared/glossary/#patchification) works
  - **Patch size and resolution tradeoffs** — smaller patches = more tokens = better detail = quadratically more compute. Project [08](projects/08-patch-size-study/README.md) measures each link in that chain: 17 → 257 tokens costs 19× the [FLOPs](/shared/glossary/#flops), and [attention](/shared/glossary/#attention)'s share of them climbs from 2% to 25% because attention compares every token with every other while all the other layers see each token alone. It also finds the trap in "better detail": on a task with no fine detail to find, the *largest* patch wins on accuracy as well as on cost.
  - **[SigLIP](/shared/glossary/#siglip) / SigLIP 2 / DFN / EVA-CLIP / [DINOv2](/shared/glossary/#dinov2)** — modern improvements over the original CLIP vision tower; DINOv2 is the dominant *[self-supervised](/shared/glossary/#self-supervised)* (non-contrastive) choice
- **Text encoders**:
  - BERT-style bidirectional encoders (for dual-encoder models)
  - Decoder-only LLMs as encoders (just take hidden states)
  - The tradeoff: bidirectional sees both sides but is non-causal; decoder-only is causal but composes naturally with generation
- **Audio encoders**:
  - [Mel spectrograms](/shared/glossary/#mel-spectrogram) — the standard input representation. Project [06](projects/06-mel-spectrogram-pipeline/README.md) builds one from scratch and removes each step in turn to show none of them is convention: dropping the log alone takes a digit classifier from 72% to 21%.
  - [Whisper](/shared/glossary/#whisper)-style encoders for speech — project [07](projects/07-whisper-encoder-reuse/README.md) freezes one and probes every layer, watching it *discard* speaker identity (0.82 → 0.44) while it builds up phonetic content (0.19 → 0.96), because identity is a nuisance variable for transcription
  - HuBERT, wav2vec 2.0 for general audio
  - Neural audio codecs (EnCodec, SoundStream, DAC, Mimi) for discrete audio tokens
- **Video encoders**:
  - Frame-by-frame ViT (cheap but loses motion)
  - 3D convolutions or 3D-ViT for spatiotemporal patches
  - Temporal pooling, hierarchical encoding

### Image Patchification, Visualized

```
Input image: 224×224×3

Split into 16×16 patches:    14 × 14 = 196 patches

Each patch: 16×16×3 = 768 numbers

Linear projection → embedding of dim D (e.g., 768)

Add positional embedding (learned or sinusoidal)

+ [CLS] token at position 0

→ sequence of 197 tokens, each D-dim
→ feed to a stack of transformer blocks
→ output is 197 contextualized vectors
→ pool (CLS token, mean, attention pool) → single image vec
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Implement ViT from scratch](projects/04-implement-vit-from-scratch/README.md) | Patchify, linear-project, transformer blocks, CLS pooling — train on CIFAR-10 | ⭐⭐⭐ |
| [Compare encoders](projects/05-compare-encoders/README.md) | Take ResNet-50, ViT-B/16, SigLIP, and DINOv2 — extract features for 1k ImageNet images, compare via linear probe | ⭐⭐ |
| [Mel spectrogram pipeline](projects/06-mel-spectrogram-pipeline/README.md) | Take a 10-second `.wav` file, produce a mel spectrogram, feed through a small CNN | ⭐⭐ |
| [Whisper encoder reuse](projects/07-whisper-encoder-reuse/README.md) | Use just the encoder of Whisper to get audio embeddings; build a simple audio classifier on top | ⭐⭐⭐ |
| [Patch-size study](projects/08-patch-size-study/README.md) | Train ViT with patches of 8, 16, 32 — measure accuracy and FLOPs | ⭐⭐⭐ |

### Sample Code: A Minimal ViT Patch Embedding

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # A conv with stride=patch_size is the standard trick:
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)                       # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)         # (B, N+1, D)
        return x + self.pos_embed
```

### Key Insight

The [patchification](/shared/glossary/#patchification) trick — using a single strided convolution to both split the image into patches *and* project them to the embedding dimension — is one of those "obvious in hindsight" moves that made ViT practical. It's mathematically identical to the unfold-then-linear approach but is dramatically faster. (It works because a convolution whose kernel size *equals* its stride never lets two windows overlap, so it visits exactly the squares you would have cut by hand — and applying the kernel to a window *is* the projection you wanted. Project [04](projects/04-implement-vit-from-scratch/README.md) checks the two routes agree to 8 × 10⁻⁷.) The deeper lesson: every modality reduces to "turn it into a sequence of D-dimensional vectors," after which a transformer doesn't care whether those vectors came from pixels, waveforms, or words.

Two threads run through all five projects here and are worth naming before you start, because each one cost a beginner-shaped mistake to find.

**Freeze first, and probe before you fine-tune.** Fine-tuning *changes* the features, so it measures how well an encoder adapts, not what it already knows — which is the wrong question when you are choosing what to build on. Project [05](projects/05-compare-encoders/README.md) freezes four towers and finds that [SigLIP](/shared/glossary/#siglip), trained on free noisy web captions, beats a ViT of identical size and shape trained on 14M hand-curated ImageNet-21k labels. Supervision beat architecture. Project [07](projects/07-whisper-encoder-reuse/README.md) shows the payoff in data: a frozen [Whisper](/shared/glossary/#whisper) encoder plus one linear layer reaches 68% on spoken digits from five examples each, where an equivalent network trained from scratch reaches 14% — barely above guessing.

**Your benchmark has to be able to see the thing you are measuring.** This bites in every one of these projects. Project [05](projects/05-compare-encoders/README.md)'s four encoders all score 0.94–0.99 with plenty of labels and look interchangeable; cut to one label per class and a 65-point gap opens. Project [07](projects/07-whisper-encoder-reuse/README.md)'s layer chart is two flat lines at full labels and reveals the encoder's entire design philosophy at two labels per class. And three separate projects found *extra resolution that was genuinely useless*, because the task had no structure at that scale — [04](projects/04-implement-vit-from-scratch/README.md)'s positional embeddings (used, but worth only 2 points on CIFAR-10), [06](projects/06-mel-spectrogram-pipeline/README.md)'s mel bins (8 beat 80 by 38 points), and [08](projects/08-patch-size-study/README.md)'s patch size (the *largest* patch won). Before concluding a component does not matter, check whether your measurement could have noticed if it did.

### Resources

- [ViT paper](https://arxiv.org/abs/2010.11929)
- [SigLIP paper](https://arxiv.org/abs/2303.15343) — sigmoid loss replaces softmax, scales better
- [DINOv2 paper](https://arxiv.org/abs/2304.07193) — strong self-supervised vision features
- [Whisper paper](https://arxiv.org/abs/2212.04356) — best read on a modern speech encoder
- [EnCodec paper](https://arxiv.org/abs/2210.13438) — for discrete audio tokens
- [VideoMAE paper](https://arxiv.org/abs/2203.12602) — masked autoencoding for video

---

## Phase 3: Contrastive Learning — CLIP and Friends

CLIP is the model that made modern multimodal learning take off, and **contrastive alignment is this guide's home turf** — the [Image Generation guide explicitly defers CLIP/contrastive pretraining here](../image-generation/). Understanding it cold pays compound interest.

### Concepts to Learn

- **The [contrastive](/shared/glossary/#contrastive-learning) objective**: pull matched (image, caption) pairs together, push unmatched pairs apart
- **[InfoNCE](/shared/glossary/#infonce) loss** — the workhorse contrastive loss; relationship to [mutual information](/shared/glossary/#mutual-information). The name is two abbreviations: *[NCE](/shared/glossary/#noise-contrastive-estimation)* (noise-contrastive estimation — tell the true pair apart from noise) and *Info* (the loss is a lower bound on how much the image and caption share). Project [09](projects/09-implement-infonce/README.md) writes it by hand, checks it against autograd and finite differences to 10⁻¹², and measures that bound: it is capped at `ln(N)`, so **a batch of 32 cannot certify more than 3.47 nats of shared information no matter how good the model is.** That is the strongest argument for large batches, and it is arithmetic rather than folklore.
- **The [temperature](/shared/glossary/#temperature) parameter τ** — what it controls, why it's learnable, and why it matters more than you'd think. Concretely: τ cannot change which caption ranks highest (it is a monotone rescaling — in-batch accuracy is identical at every τ) but it decides *which negatives receive gradient*. At τ = 0.01 a batch of 256 has only **2.7 effective negatives**; at τ = 0.07 it has 178. Project [13](projects/13-temperature-ablation/README.md) trains at each setting and finds a wide usable range on the low side with a hard wall at τ = 1.0, where the loss ends barely below its own chance level.
- **[Batch size](/shared/glossary/#batch) in contrastive learning** — why bigger is dramatically better, and the tricks (memory bank, MoCo, distributed gathering) to fake it cheaply. Two cautions project [10](projects/10-tiny-clip/README.md) measures: an InfoNCE *number* is not comparable across batch sizes (chance is `ln(N)`, so the same frozen model "scores" 0.674 at N = 8 and 4.904 at N = 1000), and at a fixed data budget bigger batches buy negatives by spending optimizer steps — 32 → 128 gained 70% relative on R@1, and 128 → 512 gave it back.
- **[Hard negatives](/shared/glossary/#hard-negatives)** — easy negatives don't teach the model anything; mining hard ones helps. Project [12](projects/12-hard-negative-mining/README.md) is the honest counterweight: mining only helps if the model doing the mining already knows what "similar" means, and mined batches concentrate on a topic, which quietly shrinks the *variety* of negatives even as it raises their difficulty. It also names the built-in cost — a [false negative](/shared/glossary/#false-negative) is a mined "wrong" caption that actually describes the image, and the harder you mine the more of those you get.
- **CLIP variants**:
  - **SigLIP / SigLIP 2** — sigmoid (per-pair) loss instead of softmax (over-batch) loss; works at smaller batch sizes
  - **ALIGN** — Google's CLIP-equivalent, trained on noisier web data
  - **OpenCLIP, EVA-CLIP, DFN, MetaCLIP** — community and Meta scaling efforts
  - **ImageBind** — extends contrastive learning to 6 modalities (text, image, audio, depth, thermal, IMU)
- **[Zero-shot](/shared/glossary/#zero-shot) classification** — how CLIP does classification without ever seeing labels. The classifier is a matrix you *write in English*: one sentence per class, encoded and [L2-normalized](/shared/glossary/#l2-normalization). Project [11](projects/11-zero-shot-imagenet/README.md) runs it against all 1,000 ImageNet class names and shows that the candidate list *is* the benchmark — the same model scores 0.988 against 10 candidates and 0.660 against 1,000 — and that the wording is worth ~0 points at 10 classes and up to 12 at 1,000.
- **[Prompt ensembling](/shared/glossary/#prompt-ensembling)** — encode each class in several sentences and average the vectors. Project [11](projects/11-zero-shot-imagenet/README.md) measures the part that is not obvious: the ensemble beats **every individual template**, not merely their average, because the templates' errors are partly independent (two templates for the same class agree at cosine 0.919, close but not identical).
- **CLIP as a filter** — using CLIP scores to clean web-scale training data (e.g., [LAION](/shared/glossary/#laion)); the same trick reappears for data curation in [Image Generation Phase 10](../image-generation/) and [Video Generation Phase 10](../video-generation/). Project [14](projects/14-data-filtering-with-clip/README.md) breaks 60% of a COCO subset on purpose and finds the filter separates genuine from broken pairs at [AUC](/shared/glossary/#auc) 0.994 — then shows where the trick stops working, because filtering harder eventually deletes good data faster than it deletes noise.

### The CLIP Training Step

```
Batch of N (image, text) pairs:

    images:    [I₁, I₂, ..., I_N]   → image encoder  → [v₁, v₂, ..., v_N]
    captions:  [T₁, T₂, ..., T_N]   → text encoder   → [u₁, u₂, ..., u_N]

L2-normalize both, then compute similarity matrix:

    S = (V · Uᵀ) / τ      # shape (N, N)

Diagonal entries S[i, i] are matched pairs (should be high).
Off-diagonal entries S[i, j], j≠i are unmatched (should be low).

Loss = (cross_entropy(S, labels=identity, axis=rows)
      + cross_entropy(S, labels=identity, axis=cols)) / 2

i.e. for each image, the matched caption should be the most similar
out of all N captions in the batch — and vice versa.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Implement InfoNCE](projects/09-implement-infonce/README.md) | Write the symmetric contrastive loss from scratch; verify gradients | ⭐⭐ |
| [Tiny CLIP](projects/10-tiny-clip/README.md) | Train a small CLIP on Flickr30k or COCO captions; image encoder = small ViT, text encoder = small transformer | ⭐⭐⭐⭐ |
| [Zero-shot ImageNet](projects/11-zero-shot-imagenet/README.md) | Use a pretrained CLIP to classify ImageNet without ever training on its labels; tune the prompt template | ⭐⭐ |
| [Hard-negative mining](projects/12-hard-negative-mining/README.md) | Train CLIP with mined hard negatives vs random; measure retrieval improvement | ⭐⭐⭐⭐ |
| [Temperature ablation](projects/13-temperature-ablation/README.md) | Vary τ from 0.01 to 1.0; observe accuracy and the geometry of the embedding space | ⭐⭐⭐ |
| [Data filtering with CLIP](projects/14-data-filtering-with-clip/README.md) | Filter a noisy image-text dataset by CLIP similarity score; train a downstream model on filtered vs unfiltered | ⭐⭐⭐ |

### Sample Code: CLIP-Style Contrastive Loss

```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, logit_scale):
    # L2-normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Cosine similarity matrix, scaled by learned temperature
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text  = logits_per_image.T

    n = image_features.size(0)
    labels = torch.arange(n, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2

# logit_scale is usually parameterized as exp(theta) where theta is learned,
# initialized to ln(1/0.07) ≈ 2.66, clamped to ≤ ln(100) for stability.
```

### Key Insight

CLIP's most important contribution was not the architecture; it was the realization that the *internet* is a labeled dataset. Every image with an [alt-text](/shared/glossary/#alt-text) or surrounding caption is a free supervised example. The contrastive objective turned this firehose of noisy data into a useful signal. The architecture (two transformers + cosine similarity) is almost an afterthought.

Project [10](projects/10-tiny-clip/README.md) is that claim taken literally: a 1.56M-parameter CLIP — 1% of CLIP B/32 — trained for two minutes on 4,500 photographs, reaching held-out retrieval at 17× chance. The architecture really is the easy part. Everything hard in this phase is about the *contest*: how many candidates it has, which of them receive gradient, and whether the pairs were true to begin with.

Three threads run through all six projects, and each cost a beginner-shaped mistake to find.

**A contrastive number is unreadable without the contest that produced it.** Chance for [InfoNCE](/shared/glossary/#infonce) is `ln(N)`, so the *same frozen CLIP* "scores" 0.674 at batch 8 and 4.904 at batch 1,000 (project [09](projects/09-implement-infonce/README.md)). The same trap eats [zero-shot](/shared/glossary/#zero-shot) accuracy — 0.988 against 10 candidate labels, 0.660 against 1,000, one unchanged model (project [11](projects/11-zero-shot-imagenet/README.md)) — and it eats loss comparisons across batch *compositions* too: project [12](projects/12-hard-negative-mining/README.md)'s mined runs end near chance because they are answering a harder question, not because they failed. Always report the [gallery](/shared/glossary/#gallery) or the candidate list next to the number.

**Damage shows up in the geometry before it shows up in the score.** Three separate mistakes produced the same fingerprint — embeddings bunching together instead of spreading over the sphere ([representation collapse](/shared/glossary/#representation-collapse)): dropping half the symmetric loss (project [10](projects/10-tiny-clip/README.md), text-side [uniformity](/shared/glossary/#alignment-and-uniformity) −1.39 → −1.07), setting τ = 0.01 (project [13](projects/13-temperature-ablation/README.md), matched and mismatched cosines 0.015 apart), and filling batches with mined negatives (project [12](projects/12-hard-negative-mining/README.md), all four conditions). In each case retrieval was still roughly fine when the space was already sick. And the reverse is just as true: a *bigger* average margin between matched and mismatched pairs went with *worse* retrieval in project 13 (τ = 1.0 separates them 29× better than τ = 0.01 and retrieves half as well), because retrieval is decided by rank within a row, not by an average. The [modality gap](/shared/glossary/#modality-gap) behaves the same way — it shrank monotonically from 1.150 to 0.060 as τ rose, and the smallest gap belonged to the worst model, reproducing project [02](projects/02-visualize-the-modality-gap/README.md)'s finding from the training side rather than by editing a frozen model.

**Two of the phase's headline recipes did not survive contact with a small budget, and the reasons are the lesson.** Bigger batches won only up to an interior optimum (project [10](projects/10-tiny-clip/README.md): 32 → 128 gained 70% relative on R@1 with 4× fewer updates; 128 → 512 gave it back), because at a fixed data budget a bigger batch buys negatives by spending optimizer steps. And [hard-negative mining](/shared/glossary/#hard-negatives) lost in all four conditions, *monotonically in mining quality* — random 0.170, self-mined 0.128, real-CLIP-mined 0.102 — because a mined batch trades **variety** for **difficulty** and only difficulty is what the name mentions. Every image in a mined batch is a desk, so that update teaches desk-vs-desk and nothing about desk-vs-beach. What *did* work exactly as advertised is the cheapest technique of the six: filtering by CLIP score deleted 60% of a deliberately-broken dataset, made the downstream model 54% better, and matched an oracle that knew the answer key (project [14](projects/14-data-filtering-with-clip/README.md)) — until it was pushed too far, at which point 99.9%-clean data starved the model and it memorized. Curate first; tune the contest later.

### Resources

- [CLIP paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP paper (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343) and [SigLIP 2 (2025)](https://arxiv.org/abs/2502.14786)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — reproducible CLIP training in the open
- [ImageBind](https://arxiv.org/abs/2305.05665) — contrastive across 6 modalities
- [Lucas Beyer — Lecture on CLIP and SigLIP](https://www.youtube.com/watch?v=eb6L_ikm5p4)

---

## Phase 4: Fusion Architectures — How Modalities Talk to Each Other

Once each modality is encoded, you have to combine them. There are more options than people realize.

### Concepts to Learn

- **[Concatenation](/shared/glossary/#concatenation)** — the trivial baseline; works surprisingly well sometimes
- **[Cross-attention](/shared/glossary/#cross-attention)** — one modality's tokens attend to another's; the most common fusion in modern VLMs
- **[Q-Former](/shared/glossary/#q-former) (BLIP-2)** — learnable queries that distill an image into a fixed number of tokens for an LLM
- **[Perceiver IO](/shared/glossary/#perceiver-io)** — cross-attention with a small latent set, modality-agnostic
- **[Adapter](/shared/glossary/#adapter) modules** — small layers inserted into a frozen backbone, trained to adapt to a new modality
- **[Gated](/shared/glossary/#gated) cross-attention ([Flamingo](/shared/glossary/#flamingo))** — adds new cross-attention layers between LLM layers, gated so the pretrained behavior isn't broken at init
- **[Projector](/shared/glossary/#projector)-only fusion ([LLaVA](/shared/glossary/#llava))** — just a linear or MLP projection from image features to LLM token space; surprisingly effective
- **Interleaved sequences** — treat image tokens and text tokens as one sequence ([early fusion](/shared/glossary/#fusion-earlymiddlelate))

### Five Fusion Patterns Side by Side

```
1. Concatenation (simplest):
     [text emb] ─┐
                 ├──► classifier
     [image emb]─┘

2. Cross-attention (Flamingo-style):
     LLM block ──► cross-attn(text tokens, image tokens) ──► next LLM block
                                       ▲
                                       │
                              image encoder features

3. Q-Former (BLIP-2):
     image encoder ──► 32 learned queries ◄── cross-attn ◄── frozen
                              │                              image features
                              ▼
                          → projected → fed into LLM as 32 "tokens"

4. Projector + interleaved tokens (LLaVA):
     image encoder ──► linear/MLP ──► N "image tokens"
                                          │
                                          ▼
     LLM sees: [<system>...<image_tokens>...<user_text>...<assistant>]

5. Early fusion (Chameleon, native multimodal):
     image ──► VQ tokenizer ──► discrete image tokens
                                   │
                                   ▼
     unified sequence: [text tokens][image tokens][text tokens]...
     trained with one next-token prediction loss over the whole alphabet
```

> **Two things a beginner reasonably asks at this point.**
>
> **"The image encoder already outputs vectors. Why does every diagram add a [projector](/shared/glossary/#projector) on top?"** Because two networks trained separately do not share a coordinate system — "vision dimension 7" and "language dimension 7" mean unrelated things — and they usually do not even share a *width* (1024 for a ViT, 4096 for a large LLM). The projector learns the change of basis and the resize. Crucially it stays **trainable while both encoders stay [frozen](/shared/glossary/#frozen)**, so it is the one piece that can adapt a frozen encoder to a model it was never trained with. Project [15](projects/15-concat-vs-cross-attn/README.md) measures what it buys.
>
> **"CLIP already produces one summary vector per image. Why does BLIP-2 build a [Q-Former](/shared/glossary/#q-former) to produce another?"** Because CLIP's summary was optimised for a different job — telling this photo apart from a million others in [contrastive](/shared/glossary/#contrastive-learning) training. Being *discriminative* is not the same as being *informative enough to write a sentence from*. The Q-Former's queries are trained on the downstream generation objective instead, so they can keep what the generator needs and drop what only mattered for matching. Project [16](projects/16-implement-q-former/README.md) builds one and tests whether that gap is real at small scale.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Concat vs cross-attn](projects/15-concat-vs-cross-attn/README.md) | On a small VQA task, compare concatenation, cross-attention, and projector fusion; report accuracy and parameter counts | ⭐⭐⭐ |
| [Implement Q-Former](projects/16-implement-q-former/README.md) | A small Q-Former with 16 learned queries; train on COCO captions | ⭐⭐⭐⭐ |
| [Adapter for a new modality](projects/17-adapter-for-a-new-modality/README.md) | Add a depth-image input to a frozen CLIP image encoder via an adapter layer | ⭐⭐⭐ |
| [Perceiver IO](projects/18-perceiver-io/README.md) | Implement the Perceiver IO architecture on a small toy task | ⭐⭐⭐⭐ |
| [Gated cross-attention](projects/19-gated-cross-attention/README.md) | Implement Flamingo's gated mechanism; verify that at init, output equals the unimodal LLM | ⭐⭐⭐⭐ |

### Sample Code: A Cross-Attention Block

```python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """LLM hidden states attend to image features."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))   # Flamingo trick

    def forward(self, x_text, x_image):
        # x_text: (B, T, D)   — query (LLM hidden states)
        # x_image: (B, N, D)  — key/value (image features, projected to dim D)
        q = self.norm_q(x_text)
        kv = self.norm_kv(x_image)
        out, _ = self.attn(q, kv, kv)
        return x_text + self.gate.tanh() * out     # tanh(0) = 0 at init → identity
```

### Key Insight

The progression from BLIP-2 → LLaVA → Chameleon is a story of *simplification*. BLIP-2 used a complex Q-Former with a multi-stage training recipe. LLaVA showed that a single linear projection works almost as well if you have good instruction data. Chameleon showed that you don't even need separate encoders — just tokenize everything. The lesson: in deep learning, "simpler + more data" usually wins.

### Resources

- [Flamingo paper (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)
- [BLIP-2 paper (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [LLaVA paper (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [Perceiver IO paper](https://arxiv.org/abs/2107.14795)
- [Chameleon paper (Meta, 2024)](https://arxiv.org/abs/2405.09818)

---

## Phase 5: Vision-Language Models (VLMs)

The current workhorse class. A VLM takes images (+ text) in and produces text. Most "multimodal AI" products you can name are VLMs.

### Concepts to Learn

- **The standard recipe**: pretrained vision encoder + projector + pretrained LLM → train projector first, then jointly fine-tune. (The vision encoder comes from Phase 2/3; the LLM comes from the [LLM guide](../llm/) — a VLM is mostly *glue and data*.) Project [20](projects/20-llava-from-scratch/README.md) builds exactly this on a real frozen CLIP and a real frozen 135M LLM, and its measurement is a warning: an image-blind [soft prompt](/shared/glossary/#prompt-tuning) *beat* the real thing on caption loss (3.071 vs 3.150) while scoring exactly [chance](/shared/glossary/#chance-level) at picking the matching caption out of 20 (0.050 vs 0.140). Caption loss mostly measures whether you write like the dataset.
- **Image preprocessing for VLMs**:
  - Fixed resolution vs **dynamic resolution / AnyRes** (Qwen2-VL, InternVL2): tile the image to handle any aspect ratio. Project [22](projects/22-dynamic-resolution/README.md) finds the split you should expect — identical accuracy on a big coloured shape, 0.303 → 0.397 on reading small print — and one surprise: native tiles *without* the global thumbnail scored 0.073, far worse than plain squashing, because detail you cannot locate is not usable detail.
  - **Native-resolution ViT** (Qwen2-VL's NaViT-style approach) — process the image at its true resolution instead of a fixed grid
  - Token budget per image — typically 256 to a few thousand image tokens
- **Instruction tuning for VLMs** — the "LLaVA-Instruct" recipe: GPT-4-generated multimodal instructions. Project [21](projects/21-visual-instruction-tuning/README.md) reproduces the format flip (yes/no accuracy 0.382 → 0.679, parseable answers 0.885 → 0.994) and then measures how much of it is vision: a blind model trained on the same questions reaches 0.649, so *looking* is worth about 3 points — inside the noise. Always run the blind baseline.
- **Visual question answering (VQA)** — classic benchmark task
- **OCR-heavy VLMs** — Donut, Nougat, GOT — for documents
- **[Grounding](/shared/glossary/#grounding)** — output [bounding boxes](/shared/glossary/#bounding-box) or pixel coordinates; teaching the LLM to "point" (Molmo's *pointing* supervision is a clean recent example). Project [23](projects/23-grounding-head/README.md) adds the [coordinate-token](/shared/glossary/#coordinate-tokens) vocabulary and separates two things that are easy to confuse: the *format* is learned immediately and perfectly (100% well-formed boxes, against 0% for the same boxes written as plain digits), while the *mapping* from image to position is not (0.097 IoU against a 0.866 ceiling — plausible boxes in the wrong places). Grounding is a data problem wearing an architecture problem's clothes.
- **Modern frontier VLMs** (2025–2026):
  - **Qwen2.5-VL / Qwen3-VL** — Alibaba, strong open VLM family
  - **InternVL2.5 / InternVL3** — Shanghai AI Lab
  - **PaliGemma 2** — Google's small, strong VLM family
  - **Molmo** — Allen AI, open weights *and* open data, strong grounding/pointing
  - **Pixtral** — Mistral's VLM
  - **Llama 3.2 Vision / Llama 4** — Meta's official VLM line (Llama 4 is natively multimodal, MoE, early-fusion)
  - **DeepSeek-VL2** — MoE VLM
  - **Closed**: GPT-4o, Claude (with vision), Gemini 2.0/2.5

### The Standard VLM Training Pipeline

```
Stage 1 (alignment):       Stage 2 (visual instruction tuning):
─────────────────────      ─────────────────────────────────────
Freeze: vision encoder     Unfreeze: projector + LLM
Freeze: LLM                Freeze:   vision encoder (often)
Train:  projector only     Train on: ~500k–5M instruction pairs
Data:   ~500k–10M           Format:   conversational (image, Q, A)
        image-caption       Sources:  LLaVA-Instruct, ShareGPT4V,
        pairs                         Cauldron, custom

Result: LLM "speaks image"  Result: VLM that follows visual instructions
```

> **Three things a beginner reasonably asks at this point.**
>
> **"Both big networks are pretrained and frozen in stage 1. So what is actually learning?"** Only the [projector](/shared/glossary/#projector) — in project [20](projects/20-llava-from-scratch/README.md) that is 0.78M weights against a 135M-parameter LLM, well under 1%. What it learns is a *change of coordinates*: CLIP's patch vectors and the LLM's word embeddings are both "vectors", but they were trained separately, so they share no axes and usually not even a width. Stage 1 is a translation exercise, which is why it is stable and cheap. It is also why the honest way to report it is against a control that gets the same training and no image — project 20 uses a learned [soft prompt](/shared/glossary/#prompt-tuning), and it closes most of the raw loss gap on its own.
>
> **"Why two stages? Why not train on instruction data from the beginning?"** Because at step 0 the projector's output is noise, and noise spliced into a pretrained LLM's input is actively harmful. Stage 1 gets the image into the LLM's vocabulary space using the easiest possible target (a caption describes *whatever is in* the picture, so almost any visual signal helps). Only then does stage 2 unfreeze the LLM and ask for specific answers to specific questions — a target that punishes a model for missing details it would otherwise be free to skip. Skipping stage 1 means asking the LLM to adapt to garbage inputs while also learning the task; the recipe exists to avoid that.
>
> **"The image encoder already sees the whole picture. Why does [AnyRes](/shared/glossary/#anyres) chop it into tiles as well?"** Because "sees the whole picture" hides a resize. A [ViT](/shared/glossary/#vit) accepts exactly the grid it was trained on, so a big page is *downscaled* first, and downscaling destroys small structures while leaving big ones recognisable. A tile is not extra information about the same pixels — it is the *original* pixels, at a resolution the encoder can resolve. Project [22](projects/22-dynamic-resolution/README.md) shows the split that follows: tiling leaves the big-object question untouched and transforms the small-text one.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [LLaVA from scratch](projects/20-llava-from-scratch/README.md) | Connect a CLIP-ViT-L/14 to a 1–3B LLM with an MLP projector; do stage-1 alignment on COCO captions | ⭐⭐⭐⭐ |
| [Visual instruction tuning](projects/21-visual-instruction-tuning/README.md) | Fine-tune the above on the LLaVA-Instruct dataset; evaluate on a few VQA benchmarks | ⭐⭐⭐⭐⭐ |
| [Dynamic resolution](projects/22-dynamic-resolution/README.md) | Implement AnyRes tiling; verify it improves OCR-heavy benchmarks | ⭐⭐⭐⭐ |
| [Grounding head](projects/23-grounding-head/README.md) | Add bounding-box outputs to a VLM via a special `<box>` token vocabulary | ⭐⭐⭐⭐ |
| [Compare projectors](projects/24-compare-projectors/README.md) | Linear vs 2-layer MLP vs Q-Former on the same downstream task; report quality and speed | ⭐⭐⭐ |
| [Inference optimization](projects/25-inference-optimization/README.md) | Take an open VLM, serve it with vLLM or SGLang; measure tokens/sec at different image counts (deep dive: [Inference Systems](../inference-systems/)) | ⭐⭐⭐ |

### Sample Code: A LLaVA-Style Forward Pass

```python
import torch
import torch.nn as nn

class TinyVLM(nn.Module):
    def __init__(self, vision_encoder, llm, vision_dim, llm_dim):
        super().__init__()
        self.vision_encoder = vision_encoder       # frozen in stage 1
        self.llm = llm                              # frozen in stage 1
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, image, input_ids, image_token_idx):
        # 1. Encode image → sequence of visual tokens
        v = self.vision_encoder(image)              # (B, N_img, vision_dim)
        v = self.projector(v)                       # (B, N_img, llm_dim)

        # 2. Get text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)  # (B, T, llm_dim)

        # 3. Replace the placeholder <image> token with the visual tokens
        # (in practice you splice them in at image_token_idx)
        # ... splicing logic ...
        merged = splice_in_visual_tokens(text_emb, v, image_token_idx)

        # 4. Run the LLM over the merged sequence
        return self.llm(inputs_embeds=merged).logits
```

### Key Insight

The biggest difference between a "good" and "great" VLM is rarely the architecture — it's the data. The projector is trivial. The vision encoder and LLM are both pretrained. What separates Qwen2.5-VL from LLaVA-1.5 is *millions of carefully curated visual instructions* and high-quality OCR data. If you want to build a competitive VLM, budget 70% of your effort for data, not modeling.

### Resources

- [LLaVA paper](https://arxiv.org/abs/2304.08485) and [LLaVA-1.5](https://arxiv.org/abs/2310.03744)
- [Qwen2-VL paper](https://arxiv.org/abs/2409.12191) and [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [PaliGemma paper](https://arxiv.org/abs/2407.07726)
- [Molmo paper (Allen AI, 2024)](https://arxiv.org/abs/2409.17146) — open weights and open data
- [InternVL2.5 / InternVL3](https://internvl.github.io/blog/)
- [The Cauldron dataset (HF)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) — 50 VLM instruction datasets unified

---

## Phase 6: Audio, Speech, and Video

Vision is the most popular non-text modality, but audio and video are catching up fast. **There is no dedicated audio guide in this collection, so this phase is the home for audio and speech.** For *video*, this phase owns *understanding* (encoding video for reasoning); video *synthesis* belongs to [Video Generation](../video-generation/).

### Concepts to Learn

- **Audio representations**:
  - Raw waveform — high resolution but very long sequences
  - Mel spectrogram — compact, perceptually grounded, the usual choice
  - **Discrete audio tokens** (EnCodec, SoundStream, DAC, Mimi) — the audio analog of BPE; enables LM-style modeling of audio
- **Speech recognition (ASR)** — Whisper, Conformer; encoder-decoder transformers on mel spectrograms
- **Text-to-speech (TTS)** — non-autoregressive (FastSpeech), autoregressive (Tortoise, VALL-E), and modern hybrid neural codec approaches
- **Music and general audio generation** — MusicGen, AudioGen, AudioLDM, Stable Audio (the audio analog of diffusion image models)
- **Speech LLMs and full-duplex voice** — taking the LLM-with-projector recipe and replacing the vision encoder with an audio encoder (Qwen2-Audio); real-time, full-duplex speech (Moshi, GPT-4o voice mode, Gemini Live)
- **Video as a modality**:
  - Frame-by-frame ViT encoding — cheap but discards motion
  - Spatiotemporal transformers — 3D attention over (frame, height, width)
  - **Video tokenization** (MagViT-v2, OmniTokenizer) — compress video into discrete tokens (the same tokenizers used for [video *generation*](../video-generation/#phase-5-latent-video-diffusion-and-video-tokenizers))
- **Video-language models**: Video-LLaVA, VideoChat, VideoLLaMA, LLaVA-Video, Qwen2.5-VL (handles video natively)

### The Audio Spectrogram Pipeline

```
Raw audio waveform (16 kHz, 1 channel):
    [-0.02, 0.01, 0.05, -0.03, ...]   ← millions of samples for a minute

Short-Time Fourier Transform (STFT):
    Sliding window (e.g., 25 ms hop 10 ms) → magnitude spectrogram (freq × time)

Mel-scale filterbank:
    Project linear-frequency bins onto a perceptually-spaced mel scale (e.g., 80 mel bins)

Log:
    log(mel_spectrogram + eps)   → range ~ [-10, 10]

Result shape: (T, 80)   ← T is the number of time frames
              ↓
              Treat as a "1D image" with 80 channels, feed to a convolutional or transformer encoder.
```

> **Three things a beginner reasonably asks at this point.**
>
> **"A [mel spectrogram](/shared/glossary/#mel-spectrogram) already turns sound into a compact array. Why do neural codecs like EnCodec exist — isn't that the same compression job?"** They solve different problems, and the giveaway is what comes out. A mel spectrogram is *continuous numbers for a network to read*: it throws away [phase](/shared/glossary/#phase-signal), so you cannot get the audio back from it without a [vocoder](/shared/glossary/#vocoder) guessing the missing part, and its values are floats, not symbols. A [neural codec](/shared/glossary/#neural-codec) produces *integers from a fixed alphabet* and can decode them back to sound. That difference is everything: integers from a fixed alphabet are exactly what a language model predicts, so a codec turns "generate audio" into "predict the next token". Project [26](projects/26-mel-spectrogram-from-scratch/README.md) measures what the mel step destroys (collapsing 257 frequency rows into 80 costs *more* than discarding all phase), and project [28](projects/28-encodec-tour/README.md) shows the codec's alphabet in action — and its bill: even the coarsest setting emits 150 tokens per second against roughly 3 text tokens per second for the same speech.
>
> **"[Whisper](/shared/glossary/#whisper) already has an encoder *and* a decoder. Why do [speech LLMs](/shared/glossary/#speech-llm) throw the decoder away and bolt on a separate language model?"** Because Whisper's decoder can do exactly one thing: write down the words. It cannot answer a question about the audio, follow an instruction, or compare two clips. The encoder is the expensive, hard-to-train part and it is worth keeping; the decoder is replaceable, so a speech LLM keeps the ears and borrows a general-purpose LLM's mouth — the identical [projector](/shared/glossary/#projector) recipe Phase 5 used for images. Project [29](projects/29-speech-llm/README.md) builds it by reusing project 20's model file with the vision tower swapped out, and finds a double dissociation worth remembering: Whisper's *first* encoder layer identifies the speaker (0.956) but barely reads the digits (0.168 per slot), while its *last* layer reads the digits three times better (0.554) and gives up only 4 points of speaker accuracy.
>
> **"If a [VLM](/shared/glossary/#vlm) can look at 8 frames at once, isn't that already a video model?"** Only for questions about *what is there*. Sampling frames and encoding each as a still image ([frame sampling](/shared/glossary/#frame-sampling)) keeps everything that a single snapshot could answer and silently discards *order* — and order is what motion is. Projects [30](projects/30-video-frame-vlm/README.md) and [31](projects/31-native-video-model/README.md) make that concrete on the same clips: a model whose only cross-frame operation is an average scores exactly the same on a motion question with the frames shuffled as in order (it cannot do otherwise), while a [tubelet](/shared/glossary/#tubelet) model that fuses two frames inside every token reads motion at 0.988 — and collapses when the frames are shuffled, which is how you know it was really using time.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Mel spectrogram from scratch](projects/26-mel-spectrogram-from-scratch/README.md) | Implement STFT and a mel filterbank; visualize a 10-second clip | ⭐⭐ |
| [Whisper fine-tune](projects/27-whisper-fine-tune/README.md) | Fine-tune Whisper-small on a low-resource language or custom domain | ⭐⭐⭐ |
| [EnCodec tour](projects/28-encodec-tour/README.md) | Use EnCodec to encode/decode audio at multiple bandwidths; listen to the reconstructions | ⭐⭐ |
| [Speech LLM](projects/29-speech-llm/README.md) | Glue an audio encoder to a small LLM with a projector; train on AudioSet captions | ⭐⭐⭐⭐⭐ |
| [Video frame VLM](projects/30-video-frame-vlm/README.md) | Sample 8 frames from a video, treat them as 8 images for a VLM, do video QA | ⭐⭐⭐ |
| [Native video model](projects/31-native-video-model/README.md) | Use spatiotemporal patches (TubeViT-style) and train a small video classifier | ⭐⭐⭐⭐ |

### Sample Code: Mel Spectrogram with `torchaudio`

```python
import torch
import torchaudio
import torchaudio.transforms as T

waveform, sr = torchaudio.load("audio.wav")        # (channels, samples)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

mel = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,        # 10 ms hop
    n_mels=80,
)(waveform)                # (channels, n_mels, time)

log_mel = torch.log(mel + 1e-6)
# log_mel is now ready to feed to an encoder — same shape semantics as an image with 80 channels.
```

### Key Insight

Once you tokenize a modality — turn it into a discrete sequence with a fixed vocabulary — it becomes "just another language" for a transformer. This is why neural audio codecs are such a big deal: they let you do language-model-style generation on audio. The same applies to images (VQ-VAE → discrete image tokens, [Image Generation Phase 3](../image-generation/#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers)), video (MagViT-v2, [Video Generation Phase 5](../video-generation/#phase-5-latent-video-diffusion-and-video-tokenizers)), and even actions (in [Robotics](../robotics/), action tokenizers). The unified-token view is the path to true any-to-any models — the subject of Phase 7.

### Resources

- [Whisper paper](https://arxiv.org/abs/2212.04356)
- [EnCodec paper](https://arxiv.org/abs/2210.13438)
- [MusicGen paper](https://arxiv.org/abs/2306.05284)
- [VALL-E paper](https://arxiv.org/abs/2301.02111) — neural codec language model for TTS
- [Moshi paper (Kyutai, 2024)](https://arxiv.org/abs/2410.00037) — full-duplex speech LM with the Mimi codec
- [MagViT-v2 paper](https://arxiv.org/abs/2310.05737) — best video tokenizer
- [Video-LLaVA paper](https://arxiv.org/abs/2311.10122)

---

## Phase 7: Unified and Any-to-Any Models

The frontier, and **this guide's signature territory** — both [Image Generation](../image-generation/#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers) and [Video Generation](../video-generation/) defer the "one transformer for every modality" framing to here. A single model that takes any combination of modalities in and produces any combination out.

### Concepts to Learn

- **The [unified-token hypothesis](/shared/glossary/#unified-token-hypothesis)** — if you can tokenize every modality, you can train one model on the union. (The tokenizers themselves are built in [Image Generation Phase 3](../image-generation/#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers); here we assemble them into a joint model.) Project [32](projects/32-discrete-image-tokens/README.md) builds the image half of that alphabet and prices it: 64 codes hold a blurry but unmistakable face at 22.0 dB, and 16× more codes buy 5.92 dB more — for 16× the sequence every downstream model has to read.
- **[Native multimodal](/shared/glossary/#native-multimodal) models** ([Chameleon](/shared/glossary/#chameleon), GPT-4o, Gemini, Llama 4): trained from scratch on all modalities, no separate "vision tower". Project [33](projects/33-tiny-chameleon/README.md) builds one in miniature — 512 image codes appended to a 22-word vocabulary, one loss, both directions — and project [34](projects/34-modality-balancing/README.md) adds audio and finds that the fix everyone reaches for ([oversampling](/shared/glossary/#oversampling) the rare modality) *cost* the majority modality four times as much as the imbalance did.
- **[Mixture-of-Experts (MoE)](/shared/glossary/#moe)** — a common scaling tool for unified models; [experts](/shared/glossary/#expert) can [specialize by modality](/shared/glossary/#expert-specialization) (Llama 4, DeepSeek-VL2). Project [35](projects/35-moe-for-multimodal/README.md) checks whether they actually do: they partially do (about a third of the maximum possible, unprompted, strongest at the network's ends), but the same run is a caution — 5× the parameters bought nothing at this scale, MoE was 1.5× *slower* than a dense model with identical [active parameters](/shared/glossary/#active-parameters), and without a [load-balancing loss](/shared/glossary/#load-balancing-loss) the specialisation score went *up* while routing collapsed onto one expert.
- **Generation across modalities**: a unified model can in principle output `<image_token>` sequences as easily as text tokens, then a decoder turns them into pixels. "In principle" is load-bearing — project [36](projects/36-reverse-direction/README.md) grafts exactly that head onto a finished [VLM](/shared/glossary/#vlm) and finds the pretraining transfers nothing.
- **AR + diffusion hybrids** — **[Transfusion](/shared/glossary/#transfusion)** (one transformer, next-token loss on text + diffusion loss on image patches) and **[Janus / Janus-Pro](/shared/glossary/#janus)** (decoupled visual encoders for understanding vs generation) are the 2024–2025 designs that close the gap with diffusion image quality while keeping a single backbone
- **[Omni models](/shared/glossary/#omni-model)** — single model, single inference path, all modalities in *and* out (GPT-4o for text/audio/vision; **Qwen2.5-Omni**, **MiniCPM-o** on the open side)
- **Late-stage vs early-stage fusion at scale** — the empirical evidence is increasingly that *earlier* fusion wins when you have enough compute
- **Trade-offs**: unified models lose some specialist quality; the question is whether the joint flexibility makes up for it

### Two Architectural Stances

```
Stance A: "Bolt-on" multimodality (LLaVA, Qwen2.5-VL)
─────────────────────────────────────────────────────
[image] → [vision encoder] → [projector] → fed as tokens to a pretrained LLM
[audio] → [audio encoder]  → [projector] → fed as tokens to the same LLM
Output: text only (or text + tool calls)

Pros: efficient; reuses huge pretrained LLMs; modular
Cons: can't generate non-text modalities; bottleneck at the projector


Stance B: "Native" multimodality (Chameleon, GPT-4o, Llama 4)
─────────────────────────────────────────────────────
Tokenize every modality into one shared discrete vocabulary:
   text tokens     [50000 entries]
   image tokens    [8192 entries from a VQ-VAE]
   audio tokens    [4096 entries from a neural codec]

One sequence: [text][image][text][audio][image]...
One transformer with one next-token prediction loss over the union.
(Variant — Transfusion: keep image patches continuous and apply a
 diffusion loss on them inside the same transformer.)

Pros: any-to-any natively; one model, one loss
Cons: enormous compute; harder to leverage existing LLMs;
      discrete-image-token decoder quality historically lagged diffusion
      (the gap Transfusion/Janus are closing)
```

> **Four things a beginner reasonably asks at this point.**
>
> **"If every modality just becomes tokens, why does anyone still argue about architecture? Isn't tokenizing everything the whole answer?"** Tokenizing is the easy half; the *price* of the tokens is where the arguments live. Project [32](projects/32-discrete-image-tokens/README.md) puts numbers on that price. Sixteen times more tokens per image bought 5.92 dB of reconstruction quality — and 16× the sequence length, which costs a [transformer](/shared/glossary/#transformer) roughly 256× the [attention](/shared/glossary/#attention) work, because attention compares every token with every other. It also found the two traps: [codebook collapse](/shared/glossary/#codebook-collapse) silently turned 512 entries into an effective 11–17 (so "9 bits per token" was really 3.5), and the tokenizer *loses to JPEG* above ~900 bytes while being the only option below ~730. What you buy with a tokenizer is not compression — it is a discrete alphabet a language model can predict.
>
> **"A single loss over one mixed vocabulary sounds too simple. Does the model really not need a fusion module?"** It really does not, and project [33](projects/33-tiny-chameleon/README.md) builds it in about 200 lines: 512 image codes appended to a 22-word vocabulary, a `<boi>` marker the model must learn like any other word, and ordinary [next-token prediction](/shared/glossary/#next-token-prediction). The result generates faces from captions *and* captions from faces with one set of weights. But it also shows why "does it work?" is the wrong question. Against single-modality controls, sharing **helped text (0.477 vs 0.492) and hurt images (4.747 vs 4.623)** — the minority modality gained, the majority one paid. And its captioning score of 0.879 collapses to a 3-point improvement once you run the model that answers the same questions *without seeing the image* (0.849). Unification working and unification being free are separate claims, and only controls separate them.
>
> **"[Modality balancing](/shared/glossary/#modality-balancing) sounds like standard hygiene. Why not always oversample the rare modality?"** Because the fix has a price and the diagnosis is harder than it looks. Project [34](projects/34-modality-balancing/README.md) trains text + images + audio together and starts with the measurement problem: losses of 0.61, 4.80 and 1.37 say nothing about which modality is neglected, since each is drawn from a different-sized alphabet. Only a *gap to that modality's own solo model* is comparable. With gaps in hand, starving audio to 1% of examples reproduced the classic failure exactly (+0.518) — and, unexpectedly, hurt **text** even more relative to normal (+0.210), because the digit captions rode along inside the audio rows. Then the honest inversion: applying either standard fix to the healthy mixture **quadrupled the image gap (+0.032 → +0.130) and bought audio nothing**, because nothing was broken. Balance is a treatment, not a hygiene step.
>
> **"A [VLM](/shared/glossary/#vlm) already has image tokens in its vocabulary. Isn't adding generation just a matter of training the output side?"** This is the most reasonable-sounding wrong idea in the phase, and project [36](projects/36-reverse-direction/README.md) measures how wrong. Because of [weight tying](/shared/glossary/#weight-tying) the codes *are* already there — and after 1,000 steps of never being a prediction target, the model scores **12.76 nats on image tokens against a 6.24 chance level**: it has been actively trained to suppress them, so it is twice as bad as guessing. Worse, grafting a head on and training gave **no advantage over a random-init model of the same size** (4.979 for from-scratch vs 5.026 for full fine-tuning). Two more surprises fall out: freezing the backbone did *not* prevent forgetting, because both heads share one [softmax](/shared/glossary/#softmax) denominator; and reusing the tied head drew marginally better while destroying captioning outright (+11.4 nats), because that same matrix embeds the words. This is exactly the wall [Janus](/shared/glossary/#janus) answers with separate encoders and [Transfusion](/shared/glossary/#transfusion) answers with a continuous generation path.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Discrete image tokens](projects/32-discrete-image-tokens/README.md) | Train a small VQ-VAE on a face dataset; verify reconstruction at 1024 tokens/image (the tokenizer recipe is [Image Gen Phase 3](../image-generation/#phase-3-discrete-latents--vq-vae-vq-gan-and-modern-tokenizers)) | ⭐⭐⭐⭐ |
| [Tiny Chameleon](projects/33-tiny-chameleon/README.md) | Tokenize images with the above VQ-VAE, interleave with text from COCO captions, train one transformer over the unified sequence | ⭐⭐⭐⭐⭐ |
| [Modality balancing](projects/34-modality-balancing/README.md) | Train a unified model on text+image+audio; observe and fix one modality dominating loss | ⭐⭐⭐⭐ |
| [MoE for multimodal](projects/35-moe-for-multimodal/README.md) | Add a small MoE layer to a multimodal model; observe whether experts naturally specialize | ⭐⭐⭐⭐⭐ |
| [Reverse direction](projects/36-reverse-direction/README.md) | Take a VLM (image-in → text-out) and add image generation by training an image-token output head | ⭐⭐⭐⭐⭐ |

### Key Insight

The bet behind native multimodal models is that *the same scaling laws that gave us GPT-4 from text will give us GPT-4o from text+vision+audio together* — if you have enough data and compute, the model will figure out the cross-modal structure on its own. The bet behind bolt-on multimodal models is that you can get 90% of the benefit at 10% of the cost. Both bets are still being played out; as of 2026 the bolt-on architecture still wins on cost/quality for most *understanding* tasks, while native + AR/diffusion-hybrid architectures (GPT-4o image generation, Janus-Pro, Transfusion) are pulling ahead on the *generation* side and on the capability ceiling.

**What building all five projects adds to that summary is a single recurring shape: every headline number in this phase came from a control, and three of the five controls contradicted the received wisdom.** Unification "worked" in project [33](projects/33-tiny-chameleon/README.md) — and the solo-model controls showed it helped text while hurting images, and a blind model captured 97% of its captioning score. Modality balancing is standard practice — and in project [34](projects/34-modality-balancing/README.md) applying it to a healthy mixture quadrupled the image gap for no gain. Grafting generation onto an understanding model is "the cheap path" — and in project [36](projects/36-reverse-direction/README.md) a random-init model of the same size did it better. None of these are arguments against unified models; they are arguments that *the specific claim you are making needs the specific control that could falsify it*, because in this phase the plausible story and the measured one keep coming apart.

### Resources

- [Chameleon paper (Meta, 2024)](https://arxiv.org/abs/2405.09818)
- [Transfusion paper (Meta, 2024)](https://arxiv.org/abs/2408.11039) — one transformer, next-token + diffusion losses
- [Janus-Pro paper (DeepSeek, 2025)](https://arxiv.org/abs/2501.17811) — decoupled understanding/generation encoders
- [Unified-IO 2 paper](https://arxiv.org/abs/2312.17172)
- [Emu3 paper (BAAI, 2024)](https://arxiv.org/abs/2409.18869) — next-token prediction, all modalities
- [Qwen2.5-Omni paper (2025)](https://arxiv.org/abs/2503.20215) — open any-to-any omni model
- [Gemini technical report](https://arxiv.org/abs/2312.11805)
- [GPT-4o announcement and analyses](https://openai.com/index/hello-gpt-4o/)

---

## Phase 8: Training at Scale — Data, Compute, and Alignment

Multimodal models are dataset-bound long before they are compute-bound. This phase is about everything that surrounds the model itself — the parts that are *specific to multimodality*. (The generic distributed-training machinery is [PyTorch Deep Dive Phase 7](../pytorch-deep-dive/#phase-7-distributed-training--ddp-fsdp-and-beyond); the RLHF/DPO *algorithms* are [RL Phase 9](../reinforcement-learning/#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr).)

### Concepts to Learn

- **Web-scale multimodal datasets**:
  - **LAION-5B, LAION-2B-en** — the canonical open image-text corpus (with all its problems)
  - **DataComp, COYO-700M** — alternatives and successors
  - **OBELICS** — interleaved image-text web documents
  - **WebLI** — Google's large internal alternative
- **Data filtering** — most of LAION is unusable; CLIP-score filtering, NSFW filtering, dedup, OCR filtering, aesthetic filtering (the CLIP-score filter is the Phase 3 trick reused at scale)
- **Synthetic captions** — recaptioning web images with a strong VLM dramatically improves downstream training (the trick behind DALL-E 3, ShareGPT4V); this is shared lore with [Image Gen Phase 10](../image-generation/) and [Video Gen Phase 10](../video-generation/)
- **Curriculum and staged training** — start with clean alignment data, then noisier scale data, then instruction data
- **[Modality balancing](/shared/glossary/#modality-balancing)** — in a unified model, if 99% of your tokens are text, the image loss will be ignored; need to upsample or reweight
- **Multimodal alignment / RLHF** — preference data with image inputs; sycophancy and hallucination are harder to fix when the model has multiple modalities to "hallucinate from." *The algorithms (PPO, DPO, GRPO) are owned by [RL Phase 9](../reinforcement-learning/#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr); what's multimodal-specific is the preference-data collection and the visual grounding of the reward.*
- **Safety**: NSFW filtering, CSAM detection (mandatory), bias evaluation across demographics, hallucination benchmarks
- **Compute budgets** — typical pretraining for an open VLM is 10⁸–10⁹ image-text pairs; native multimodal is 10× more

### A Pragmatic Data Pipeline

```
Raw web crawl (e.g., Common Crawl + image URLs)
        │
        ▼
Deduplicate (URL, perceptual hash)
        │
        ▼
NSFW + CSAM filtering (must, not optional)
        │
        ▼
CLIP-score filtering (keep top ~30%)
        │
        ▼
Aspect-ratio and resolution filtering (drop tiny / extreme ratios)
        │
        ▼
Synthetic recaption with a strong VLM (recommended)
        │
        ▼
Aesthetic + OCR-quality scoring (task-dependent)
        │
        ▼
Tokenize text, store as shards (WebDataset / Parquet / Arrow)
        │
        ▼
Training-ready: ~10–20% of the original crawl, dramatically higher quality
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Mini LAION pipeline](projects/37-mini-laion-pipeline/README.md) | Take 1M LAION URLs, download, filter with CLIP, dedup, recaption with a small VLM — produce a clean shard | ⭐⭐⭐⭐ |
| [Caption ablation](projects/38-caption-ablation/README.md) | Train two small VLMs: one on original alt-text, one on recaptioned text; compare downstream | ⭐⭐⭐⭐ |
| [Modality ratio sweep](projects/39-modality-ratio-sweep/README.md) | In a unified model run, deliberately under/oversample one modality; sweep the sampling ratio and measure each modality's loss curve as a diagnostic | ⭐⭐⭐⭐ |
| [Multimodal DPO](projects/40-multimodal-dpo/README.md) | Collect a small set of preference pairs over VLM outputs; fine-tune with DPO ([algorithm reference](../reinforcement-learning/#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr)) | ⭐⭐⭐⭐ |
| [Hallucination eval](projects/41-hallucination-eval/README.md) | Build a small benchmark of trick questions ("is there a dog in this image?" when there is none); evaluate several open VLMs | ⭐⭐⭐ |

### Key Insight

Two facts that dominate multimodal training at scale: (1) web [alt-text](/shared/glossary/#alt-text) is *terrible* — short, generic, often wrong; and (2) synthetic captions from a strong VLM are *much* better than human-written captions on average. The implication: data quality is itself a model-output problem. Better models make better captions, which make better models. This recursive improvement is one of the unexplained engines of recent progress.

### Resources

- [LAION-5B paper](https://arxiv.org/abs/2210.08402)
- [DataComp paper](https://arxiv.org/abs/2304.14108) — careful study of data curation
- [DALL-E 3 technical report](https://cdn.openai.com/papers/dall-e-3.pdf) — recaptioning recipe
- [ShareGPT4V](https://arxiv.org/abs/2311.12793) — recaptioning for VLMs

---

## Phase 9: Evaluation and Benchmarks

Multimodal evaluation is notoriously broken. Knowing which benchmarks to trust (and how they fail) is its own skill. (This is the *multimodal* evaluation playbook; text-only LM evaluation is [LLM Phase 8](../llm/#phase-8-evaluation).)

### Concepts to Learn

- **The benchmark landscape**:
  - **MMMU / MMMU-Pro** — multidiscipline multimodal understanding, hardest open VQA benchmark
  - **MMBench, MME** — general multimodal capability
  - **DocVQA, OCRBench** — OCR-heavy document understanding
  - **MathVista, MathVision** — math + diagrams
  - **ChartQA, AI2D** — charts and diagrams
  - **POPE, HallusionBench** — object/visual hallucination
  - **RefCOCO** — referring expressions / grounding
  - **VideoMME, MLVU, LongVideoBench** — video understanding
- **The captioning benchmarks** (CIDEr, BLEU, METEOR on COCO, NoCaps): largely solved and increasingly meaningless
- **LLM-as-judge** — using a strong model (often GPT-4 or Claude) to grade open-ended outputs; introduces its own biases
- **Hallucination measurement** — counting "things in the caption that aren't in the image"
- **Robustness probes** — adversarial images, distribution shifts, demographic balance
- **Reasoning benchmarks** — multimodal CoT, M³CoT, ScienceQA; the rise of multimodal *reasoning* models (QVQ, Gemini/o-series thinking with vision)
- **The leakage problem** — many benchmarks are now in pretraining corpora; suspect any too-good result

### A Sane Evaluation Suite for a New VLM

```
Capability                    Benchmark          What it measures
──────────────────────────    ───────────────    ─────────────────────
General multimodal QA         MMBench, MMMU      breadth, hard problems
OCR / docs                    DocVQA, OCRBench   text in images
Grounding                     RefCOCO            spatial precision
Hallucination                 POPE               object existence
Math + diagrams               MathVista          structured reasoning
Charts                        ChartQA            quantitative reading
Video (if applicable)         VideoMME           temporal understanding
Open-ended (LLM-judge)        MM-Vet, LLaVA-Wild user-style queries
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Run a VLM evaluation harness](projects/42-run-a-vlm-evaluation-harness/README.md) | Use `lmms-eval` or `VLMEvalKit` to score an open VLM across 6+ benchmarks | ⭐⭐ |
| [Build a hallucination probe](projects/41-hallucination-eval/README.md) | Construct 200 "is X in this image" questions; half true, half false; measure precision/recall | ⭐⭐⭐ |
| [Reproduce a leaderboard result](projects/43-reproduce-a-leaderboard-result/README.md) | Pick a paper's MMBench number; reproduce; document the gap | ⭐⭐⭐ |
| [Benchmark contamination check](projects/44-benchmark-contamination-check/README.md) | Search for a benchmark's test questions in a pretraining corpus shard | ⭐⭐⭐ |
| [Human-correlated eval](projects/45-human-correlated-eval/README.md) | For 100 outputs, get 3 human ratings and 3 LLM-as-judge ratings; measure agreement | ⭐⭐⭐ |

### Key Insight

There is no single number that captures "VLM quality." MMMU measures different things from POPE which measures different things from MathVista. The right move when launching a new VLM is to publish a *suite* — and to explicitly report the benchmarks where your model is *worse* than the prior state of the art. The field rewards honesty; reviewers see through cherry-picking.

### Resources

- [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) — modern multimodal eval harness
- [`VLMEvalKit`](https://github.com/open-compass/VLMEvalKit) — alternative, broad coverage
- [MMMU paper](https://arxiv.org/abs/2311.16502)
- [POPE paper](https://arxiv.org/abs/2305.10355) — hallucination eval
- [Are Emergent Abilities a Mirage? (Schaeffer et al., 2023)](https://arxiv.org/abs/2304.15004) — required reading on eval skepticism

---

## Phase 10: Frontier Topics

Where the field is going. Pick one or two threads and go deep. Several of these threads live on a border with another guide — the cross-links below tell you who owns the rest of the story.

### Vision-Language-Action (VLA) Models for Robotics
A VLA takes (image, instruction) → action. From the multimodal angle, a VLA is "a VLM whose output head emits action tokens instead of text" — the foundational works are RT-2, OpenVLA, π0, Gemini Robotics. **The robot policy, imitation-learning, and sim-to-real side is owned by [Robotics Phase 8](../robotics/#phase-8-learning-for-robotics)** (which reads better after this guide's Phase 5); here we care only about the multimodal-modeling interface.

### Long-Context Multimodal
A 1-hour video has tens of thousands of frames. How do you fit that into a context window? Streaming attention, hierarchical encoding, learned memory, image-token compression. (The *serving* of long multimodal contexts — KV-cache sizing, prefix caching — is [Inference Systems](../inference-systems/).)

### Generative Unified Models
Models that produce images, audio, and video natively from one transformer: Chameleon's image-generation half, Show-o, Janus-Pro, Emu3, Transfusion. The cross-modal modeling is this guide's Phase 7; the underlying *image/video generation* machinery they decode through is [Image Generation](../image-generation/) / [Video Generation](../video-generation/).

### Multimodal Reasoning
Multimodal chain-of-thought, visual program synthesis (ViperGPT, VisProg), self-consistency over visual problems, and the new wave of multimodal *thinking* models (QVQ, Gemini/o-series with vision). The text-reasoning core is [LLM Phase 6](../llm/#phase-6-reasoning-and-inference-time-compute); what's new here is reasoning *grounded in pixels*.

### Embodied and Agentic Multimodal
Web/GUI agents that see screenshots (SeeClick, ShowUI, OS-Atlas, UI-TARS), computer use (Claude Computer Use, Gemini's GUI mode), mobile agents. The screen is just another modality. (Tool-use and agent *orchestration* is [LLM Phase 7](../llm/#phase-7-retrieval-tools-and-agents); the embodied-control side is [Robotics](../robotics/).)

### Multimodal Interpretability
What does the projector actually do? Where do image features live inside the LLM? Mechanistic interpretability for VLMs is largely virgin territory (text-side interpretability is [LLM Phase 10](../llm/#phase-10-safety-interpretability-and-frontier-topics)).

### Efficient Multimodal Inference
Token merging, pruning, image-token compression, KV-cache sharing across modalities — the *model-side* techniques. The *serving-stack* side (batching, paged KV cache, multi-LoRA, throughput) is owned by [Inference Systems](../inference-systems/); kernel-level work and quantization are [AI Hardware](../ai-hardware/).

### Safety in Multimodal Models
Visual jailbreaks (typographic attacks, adversarial images), CSAM detection, deepfake detection, watermarking for generative outputs (SynthID). New attack surfaces and new defenses.

### Resources for the Frontier

- [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) — actively maintained survey
- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [π0 paper (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)
- [Show-o paper](https://arxiv.org/abs/2408.12528)
- [UI-TARS paper (ByteDance, 2025)](https://arxiv.org/abs/2501.12326) — native GUI agent
- [Anthropic — Computer Use blog](https://www.anthropic.com/news/3-5-models-and-computer-use)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | Transformers + PyTorch fluent; ViT and BPE understood |
| 1. Foundations | 1 week | Can map any multimodal paper to a taxonomy |
| 2. Encoders | 1–2 weeks | Implemented ViT; comfortable with mel spectrograms |
| 3. Contrastive | 2 weeks | Trained a tiny CLIP; zero-shot ImageNet working |
| 4. Fusion | 1–2 weeks | Implemented at least three fusion patterns |
| 5. VLMs | 3–4 weeks | Built and instruction-tuned a small VLM end to end |
| 6. Audio + video | 2 weeks | Trained a speech LM or video classifier |
| 7. Unified models | 2–4 weeks | Built a tiny Chameleon over interleaved tokens |
| 8. Scale + data | 2 weeks | Ran a real data filtering pipeline; understand recaptioning |
| 9. Evaluation | 1 week | Ran a real eval harness; built one hallucination probe |
| 10. Frontier | Ongoing | Picked one thread (VLA, unified, agents, ...) and going deep |

**Total to research-comfortable:** ~4 months of focused study. Longer if combined with serious projects (recommended).

---

## Key Advice

1. **Start with CLIP, end with CLIP.** Every multimodal model traces back to or generalizes CLIP. If you understand contrastive learning deeply, the rest comes faster.
2. **Build the smallest end-to-end VLM you can, early.** A 100M-parameter VLM on COCO is one weekend's work. The lessons from doing it transfer directly to everything bigger.
3. **Data is the model.** Spend at least as much time on data filtering and recaptioning as on architecture. The papers don't emphasize this enough.
4. **Beware the modality gap.** Even well-aligned dual encoders keep text and image embeddings in separable regions. This affects retrieval, generation, and downstream fine-tuning.
5. **Don't trust a single benchmark.** Especially captioning metrics. Always evaluate on a suite, including LLM-as-judge for open-ended outputs.
6. **Image tokens are expensive.** A single image is often 256–2000 tokens. Multi-image and video contexts blow up fast. Know your token budget — and see [Inference Systems](../inference-systems/) when you deploy.
7. **Reuse pretrained components.** Almost no one trains a vision encoder from scratch in 2026; you start from SigLIP, DINOv2, or similar, and from a pretrained LLM from the [LLM guide](../llm/).
8. **Synthetic captions are a superpower.** Recaptioning with a strong VLM is the highest-leverage data trick in the field.
9. **`bf16` everywhere.** The same advice as for LLMs and PyTorch; multimodal training is no different.
10. **Visualize your model's attention.** Especially the cross-attention in VLMs. It tells you whether the model is actually looking at the right region.

---

## Common Pitfalls to Avoid

- ❌ Comparing CLIP variants on retrieval and concluding one "is better" — depends entirely on the dataset
- ❌ Using captioning metrics (BLEU, CIDEr) as your primary VLM evaluation
- ❌ Training a VLM on COCO and being surprised it can't OCR a screenshot
- ❌ Ignoring the temperature parameter in contrastive learning
- ❌ Loading a VLM at fp32 and wondering why inference is slow
- ❌ Forgetting that web data contains CSAM; not filtering for it
- ❌ Trusting a single MMMU score without checking the rest of the suite
- ❌ Building a unified model with 1M tokens of text and 10k tokens of image and wondering why image quality is bad
- ❌ Skipping the alignment stage and going straight to instruction tuning
- ❌ Evaluating on a benchmark that's already in your pretraining corpus
- ❌ Re-deriving diffusion/VQ-VAE internals here instead of reading [Image Generation](../image-generation/) — this guide *uses* them, it doesn't re-teach them

---

## Additional Resources

### Books and Long-Form Reading
- [Dive into Deep Learning — Multimodal chapter](https://d2l.ai/chapter_natural-language-processing-applications/index.html)
- [Foundations of Multimodal Machine Learning (CMU 11-777 course notes)](https://cmu-multicomp-lab.github.io/mmml-course/)

### Talks and Lectures
- [Lucas Beyer — Vision and Multimodal Models](https://www.youtube.com/results?search_query=lucas+beyer+multimodal)
- [CMU 11-777 Multimodal Machine Learning](https://www.youtube.com/playlist?list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA)
- [Stanford CS231n](http://cs231n.stanford.edu/) — for the vision foundations

### Key Papers, Chronologically
| Year | Paper | Contribution |
|------|-------|-------------|
| 2020 | [ViT](https://arxiv.org/abs/2010.11929) | Transformers for vision |
| 2021 | [CLIP](https://arxiv.org/abs/2103.00020) | Web-scale contrastive |
| 2022 | [Flamingo](https://arxiv.org/abs/2204.14198) | Gated cross-attention into LLMs |
| 2023 | [BLIP-2](https://arxiv.org/abs/2301.12597) | Q-Former, modular fusion |
| 2023 | [LLaVA](https://arxiv.org/abs/2304.08485) | Linear projector + instruction tuning |
| 2023 | [SigLIP](https://arxiv.org/abs/2303.15343) | Sigmoid contrastive, scales better |
| 2024 | [Qwen2-VL](https://arxiv.org/abs/2409.12191) | Dynamic / native resolution, strong open VLM |
| 2024 | [Chameleon](https://arxiv.org/abs/2405.09818) | Native any-to-any with discrete tokens |
| 2024 | [OpenVLA](https://arxiv.org/abs/2406.09246) | VLA for robotics, open-source |
| 2024 | [Emu3](https://arxiv.org/abs/2409.18869) | Next-token prediction, all modalities |
| 2024 | [Transfusion](https://arxiv.org/abs/2408.11039) | AR text + diffusion image in one transformer |
| 2025 | [Janus-Pro](https://arxiv.org/abs/2501.17811) | Decoupled understanding/generation |
| 2025 | [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) | Open any-to-any omni model |

### Tools You Should Know
- **`transformers`** (Hugging Face) — VLMs, multimodal pipelines
- **`open_clip`** — reproducible CLIP training
- **`lmms-eval` / `VLMEvalKit`** — evaluation harnesses
- **`vLLM` / `SGLang`** — multimodal inference serving (see [Inference Systems](../inference-systems/))
- **`torchaudio`** — audio loading and transforms
- **`decord` / `pyav`** — fast video frame loading
- **`webdataset`** — streaming multimodal data

### Communities
- [Hugging Face forums](https://discuss.huggingface.co/) — most VLM Q&A
- [EleutherAI Discord](https://www.eleuther.ai/) — open research
- r/MachineLearning, X/Twitter ML community

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
