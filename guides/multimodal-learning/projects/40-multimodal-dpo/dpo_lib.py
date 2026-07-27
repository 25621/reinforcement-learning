"""Preference data, a DPO loss, and a hallucination meter for a real VLM.

Project 41 imports the object vocabulary and the hallucination meter from here,
so the benchmark it builds and the fix this project trains are graded by exactly
the same rule.

The model under test is Phase 5's stack (project 20): frozen CLIP ViT-B/32 ->
trainable projector -> a real pretrained SmolLM2-135M. What changes here is only
what we train it *on*.

Why DPO needs a second, frozen copy of the model
------------------------------------------------
"Prefer the good answer" is not a well-posed instruction on its own -- a model
can satisfy it by making *everything* less likely, as long as the bad answer
falls faster. DPO therefore scores every answer *relative to a frozen reference
copy* of the same model: the quantity it maximises is how much more the trained
model likes the chosen answer than the reference did. The reference never
trains, so it is a fixed yardstick, and drifting away from it is penalised. That
is where the "implicit reward" comes from -- there is no reward model, the log
ratio between the two copies plays that role.

Because the reference is frozen, its numbers never change, so we compute them
once before training and reuse them. That turns four forward passes per step
into two.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECTS / "20-llava-from-scratch"))
import vlm_lib as V  # noqa: E402

THREADS = 6

# ---------------------------------------------------------------------------
# the object vocabulary the hallucination meter counts
# ---------------------------------------------------------------------------
# Same grouping idea as project 21: a concept counts as present if *any* of its
# words appears in *any* of the five human captions, so a photo captioned "a
# man" is not scored as having no person in it.
GROUPS = {
    "person": "man woman person people child boy girl guy lady player men women kid",
    "dog": "dog puppy", "cat": "cat kitten", "horse": "horse", "cow": "cow cattle",
    "sheep": "sheep lamb", "bird": "bird duck goose", "bear": "bear",
    "elephant": "elephant", "zebra": "zebra", "giraffe": "giraffe",
    "car": "car taxi", "bus": "bus", "truck": "truck", "train": "train",
    "motorcycle": "motorcycle motorbike", "bicycle": "bicycle bike",
    "boat": "boat ship canoe", "airplane": "airplane plane jet",
    "skateboard": "skateboard", "surfboard": "surfboard", "kite": "kite",
    "table": "table desk", "chair": "chair stool", "couch": "couch sofa",
    "bed": "bed", "bench": "bench", "umbrella": "umbrella", "vase": "vase",
    "laptop": "laptop computer", "phone": "phone cellphone",
    "television": "television tv monitor", "clock": "clock", "book": "book",
    "bottle": "bottle", "cup": "cup mug", "plate": "plate", "bowl": "bowl",
    "pizza": "pizza", "cake": "cake", "sandwich": "sandwich burger",
    "banana": "banana", "apple": "apple", "orange": "orange",
    "broccoli": "broccoli", "carrot": "carrot", "donut": "donut doughnut",
    "flower": "flower flowers", "ball": "ball", "sign": "sign",
    "kitchen": "kitchen", "bathroom": "bathroom toilet", "bedroom": "bedroom",
    "street": "street road sidewalk", "beach": "beach sand shore",
    "water": "water ocean sea lake river", "snow": "snow",
    "grass": "grass field lawn", "mountain": "mountain hill",
    "tree": "tree trees forest", "building": "building buildings tower",
    "sink": "sink", "mirror": "mirror", "window": "window",
}
_WORD = re.compile(r"[a-z]+")
_SYN = {g: set(s.split()) for g, s in GROUPS.items()}


def art(noun):
    """"a dog" but "an airplane" -- an ungrammatical prompt is its own confound."""
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def objects_in(texts):
    """Which concepts the given text(s) mention."""
    if isinstance(texts, str):
        texts = [texts]
    words = set()
    for t in texts:
        words |= set(_WORD.findall(t.lower()))
    return {g for g, syn in _SYN.items() if words & syn}


def present_absent(data, i):
    """Ground truth for image i: what the five human captions agree is there."""
    present = objects_in(data.captions[i])
    return present, [g for g in GROUPS if g not in present]


# ---------------------------------------------------------------------------
# the hallucination meter (CHAIR)
# ---------------------------------------------------------------------------
def chair(captions, truths):
    """CHAIR = Caption Hallucination Assessment with Image Relevance.

    Two numbers, both from Rohrbach et al. (2018):

        CHAIR_i  -- of all the objects the model *named*, what share were not
                    in the picture? (an "instance" rate)
        CHAIR_s  -- what share of *sentences* contained at least one made-up
                    object? (a "sentence" rate)

    CHAIR_i is the honest headline: a model that names one object per caption
    and gets it wrong half the time, and a model that names ten and gets one
    wrong, look identical on CHAIR_s but very different on CHAIR_i.
    """
    named = 0
    wrong = 0
    bad_sentences = 0
    per_caption = []
    for cap, truth in zip(captions, truths):
        said = objects_in(cap)
        halluc = said - set(truth)
        named += len(said)
        wrong += len(halluc)
        bad_sentences += 1 if halluc else 0
        per_caption.append(sorted(halluc))
    return {
        "chair_i": wrong / max(named, 1),
        "chair_s": bad_sentences / max(len(captions), 1),
        "objects_named": named,
        "objects_hallucinated": wrong,
        "objects_per_caption": named / max(len(captions), 1),
        "captions": len(captions),
        "per_caption": per_caption,
    }


# ---------------------------------------------------------------------------
# preference pairs
# ---------------------------------------------------------------------------
INSERTS = ["with {a} nearby", "next to {a}", "and {a}", "beside {a}",
           "near {a}", "along with {a}"]


def make_pairs(data, ids, seed=0, per_image=1):
    """One (chosen, rejected) pair per image.

    chosen    a real human caption
    rejected  the SAME caption with one object that is NOT in the picture
              glued on

    The two differ by a handful of tokens and nothing else -- no change of
    style, length beyond the insert, or subject. That is deliberate: DPO learns
    from the *difference* between the pair, so any other difference is something
    else it could learn instead. Real pipelines (RLHF-V, HA-DPO) get the rejected
    half from the model's own mistakes, which is better but needs a human or a
    stronger model in the loop; a scripted insert costs nothing and isolates
    exactly one behaviour: naming things that are not there.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in ids:
        present, absent = present_absent(data, int(i))
        if not absent:
            continue
        for _ in range(per_image):
            k = int(rng.integers(len(data.captions[int(i)])))
            chosen = data.captions[int(i)][k].strip().rstrip(".")
            obj = absent[int(rng.integers(len(absent)))]
            tmpl = INSERTS[int(rng.integers(len(INSERTS)))]
            rejected = f"{chosen} {tmpl.format(a=art(obj))}"
            out.append({"image": int(i), "chosen": chosen + ".",
                        "rejected": rejected + ".", "object": obj})
    return out


# ---------------------------------------------------------------------------
# scoring a completion, with gradients
# ---------------------------------------------------------------------------
def sequence_logp(vlm, batch, feats):
    """-> (summed log-probability per row, number of scored tokens per row).

    `answer_nll` in vlm_lib does the same arithmetic but is wrapped in
    `torch.no_grad`; DPO needs the gradient, so the loop is repeated here.

    Only the positions that actually predict a caption token go through the
    49,153-row output head. A padded batch is mostly prompt and padding, so this
    is roughly a 5x saving on the most expensive matrix in the model -- and it
    changes no number, because the skipped rows are multiplied by zero anyway.
    """
    emb = vlm.embed(batch, feats)
    h = vlm.body(inputs_embeds=emb, attention_mask=batch.attn, use_cache=False)[0]
    tgt = batch.labels[:, 1:]
    keep = tgt != -100
    flat = vlm.head(h[:, :-1][keep]).float()
    logp = torch.log_softmax(flat, -1)
    pick = logp.gather(-1, tgt[keep].unsqueeze(-1)).squeeze(-1)
    rows = torch.zeros(len(tgt), dtype=pick.dtype).index_add(
        0, torch.nonzero(keep, as_tuple=True)[0], pick)
    return rows, keep.sum(1).float()


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, n_c=None, n_r=None,
             length_norm=False):
    """The DPO objective.

        logits = beta * [ (log pi(chosen)  - log ref(chosen))
                        - (log pi(rejected) - log ref(rejected)) ]
        loss   = -log sigmoid(logits)

    Read it as: "how much more does the trained model prefer the good answer
    than the frozen reference did?" The two bracketed terms are the *implicit
    rewards* -- DPO's trick is that a preference model built out of these log
    ratios has the same optimum as training a separate reward model and then
    doing RL, which is why no reward model appears anywhere in this file.

    `beta` controls how far the model may drift from the reference: small beta
    = loose leash, large beta = the reference wins ties.

    `length_norm=True` divides each log-probability by its token count before
    comparing. Summed log-probabilities always favour shorter text (every extra
    token can only subtract), and our rejected answers are longer than the
    chosen ones by construction, so plain DPO can score a win simply by
    preferring brevity. Normalising removes that shortcut -- this is the
    ingredient SimPO isolates.
    """
    if length_norm:
        pi_c, pi_r = pi_c / n_c, pi_r / n_r
        ref_c, ref_r = ref_c / n_c, ref_r / n_r
    r_c = pi_c - ref_c
    r_r = pi_r - ref_r
    logits = beta * (r_c - r_r)
    return -F.logsigmoid(logits).mean(), r_c.detach(), r_r.detach()


# ---------------------------------------------------------------------------
# building / training the base model
# ---------------------------------------------------------------------------
# Deliberately the *same* string vlm_lib uses, so `val_caption_loss` scores the
# model on the prompt it was trained with. A held-out loss measured under a
# different instruction is measuring prompt transfer, not captioning.
CAPTION_PROMPT = V.INSTRUCTION


def build_vlm(kind="mlp2", unfreeze_last=4, seed=0):
    """Frozen CLIP -> projector -> SmolLM2. The last few LLM blocks are
    unfrozen so preference training has somewhere to act: with the projector
    alone, the only thing DPO can change is the 49 image vectors, and project
    21 already measured that installing a *behaviour* (rather than a subject)
    needs the language side to move."""
    torch.manual_seed(seed)
    tok, llm = V.load_llm()
    # load_llm asks for 12 threads; on this box that only pays off if nothing
    # else is running, and two 12-thread jobs are far slower than two 6-thread
    # ones (they fight over the same memory bandwidth).
    torch.set_num_threads(THREADS)
    proj = V.Projector(kind, V.CLIP_DIM, llm.config.hidden_size,
                       out_rms=V.embedding_rms(llm))
    vlm = V.TinyVLM(llm, proj)
    blocks = llm.model.layers
    for blk in blocks[len(blocks) - unfreeze_last:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    return vlm, tok


def trainable(vlm):
    return [p for p in vlm.parameters() if p.requires_grad]


def param_groups(vlm, lr, llm_scale=0.03):
    """The projector is new and needs a big learning rate; the unfrozen LLM
    blocks are pretrained and need a small one. One shared rate would either
    leave the projector untrained or wreck the language model -- Phase 5
    measured the same 20-30x split."""
    proj = [p for p in vlm.projector.parameters() if p.requires_grad]
    proj_ids = {id(p) for p in proj}
    llm = [p for p in vlm.parameters() if p.requires_grad and id(p) not in proj_ids]
    return [{"params": proj, "base_lr": lr},
            {"params": llm, "base_lr": lr * llm_scale}]


def caption_batches(tok, data, ids, texts, bs=8):
    for k in range(0, len(ids), bs):
        chunk = slice(k, k + bs)
        yield (V.make_batch(tok, [CAPTION_PROMPT] * len(ids[chunk]),
                            list(texts[chunk]), n_img=V.CLIP_TOKENS),
               data.image_tokens(ids[chunk]))


def sft(vlm, tok, data, ids, steps, bs=8, lr=1e-4, seed=0, log_every=50,
        texts=None, verbose=True):
    """Plain supervised fine-tuning on captions -- the base model, and also the
    control arm that DPO has to beat."""
    rng = np.random.default_rng(seed)
    params = trainable(vlm)
    opt = torch.optim.AdamW(param_groups(vlm, lr), weight_decay=0.0,
                            betas=(0.9, 0.95))
    hist, t0 = [], time.time()
    vlm.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = V.cosine_lr(step, steps, g["base_lr"])
        pick = rng.choice(ids, size=bs, replace=False)
        answers = [texts[int(i)] if texts is not None
                   else data.caption(int(i), int(rng.integers(5))) for i in pick]
        batch = V.make_batch(tok, [CAPTION_PROMPT] * bs, answers, n_img=V.CLIP_TOKENS)
        loss = vlm(batch, data.image_tokens(pick))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:4d}  loss {hist[-1]:.4f}"
                  f"  {time.time() - t0:5.0f}s", flush=True)
    vlm.eval()
    return hist


@torch.no_grad()
def generate_captions(vlm, tok, data, ids, max_new=24, bs=16):
    """Greedy captions for a list of images."""
    out = [None] * len(ids)
    for batch, part in V.prompt_batches(tok, [CAPTION_PROMPT] * len(ids),
                                        n_img=V.CLIP_TOKENS, bs=bs):
        feats = data.image_tokens(np.asarray(ids)[part])
        texts = vlm.greedy_batch(tok, batch, feats, max_new=max_new)
        for j, t in zip(part, texts):
            out[j] = t.strip()
    return out


def save(vlm, path):
    """Only the weights that actually moved. Saving the whole state dict would
    write 100M frozen parameters we already have on disk from the HF cache --
    and `TinyVLM` registers the language model twice (once as `llm`, once as
    the `body`/`head` aliases), so the file would be larger still."""
    keep = {n for n, p in vlm.named_parameters() if p.requires_grad}
    torch.save({k: v for k, v in vlm.state_dict().items() if k in keep}, path)


def load_into(vlm, path):
    vlm.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return vlm
