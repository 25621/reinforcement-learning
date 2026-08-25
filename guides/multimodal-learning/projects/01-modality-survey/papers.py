"""The five surveyed papers, as data instead of prose.

Writing the survey as a Python table (rather than five paragraphs in a document)
is the whole point of the exercise: it forces every paper into the *same* small
set of fields, so "compare them" becomes a lookup instead of an argument. The
two fields that matter most are:

    fusion     WHERE the modalities meet   (late / middle / early)
    objective  WHAT the training loss is   (contrastive / masked / generative)

Everything else -- what is frozen, how big the glue is, what the model can do --
follows from those two coordinates surprisingly often, which is exactly the
claim this project checks.
"""

# fusion:    "late"   encode each modality alone, compare only at the very end
#            "middle" encode alone, then let one attend to the other inside the model
#            "early"  turn every modality into tokens of ONE sequence from step 1
# objective: "contrastive" | "masked" | "generative"  (dominant training loss)
#
# connector_params: parameters of the *glue* you must train to join the two
#   towers, in millions. This is the "cost of being multimodal" on top of parts
#   you could have downloaded. 0.0 means the design needs no glue at all.

PAPERS = [
    dict(
        key="CLIP",
        title="Learning Transferable Visual Models From Natural Language Supervision",
        year=2021,
        org="OpenAI",
        url="https://arxiv.org/abs/2103.00020",
        fusion="late",
        objective="contrastive",
        connector_params=1.0,        # two linear projections into the shared space
        total_params=428.0,          # ViT-L/14 image tower + text tower
        frozen="nothing (both towers trained from scratch)",
        data="400M web (image, alt-text) pairs",
        can=dict(retrieval=True, zeroshot_cls=True, vqa=False, gen_image=False),
        summary=(
            "CLIP trains two encoders — one for pixels, one for words — completely "
            "separately, and only ever compares their outputs at the very last step "
            "with a dot product. That is what 'late fusion' means: the two halves "
            "never see each other's internals. The training signal is contrastive: "
            "inside a batch of N (image, caption) pairs, each image must rank its own "
            "caption above the other N-1. Because the loss only needs *pairs* and the "
            "web is full of images sitting next to text, CLIP could be trained on 400 "
            "million pairs that nobody paid to label. The reward for the restrictive "
            "architecture is speed: every image and every caption is encoded once, "
            "ahead of time, so search over millions of items is one matrix "
            "multiplication. The price is that it can only ever say how well two "
            "things match — it cannot write a sentence or answer a question."
        ),
    ),
    dict(
        key="Flamingo",
        title="Flamingo: a Visual Language Model for Few-Shot Learning",
        year=2022,
        org="DeepMind",
        url="https://arxiv.org/abs/2204.14198",
        fusion="middle",
        objective="generative",
        connector_params=10000.0,    # ~10B trainable glue on a frozen 70B LM
        total_params=80000.0,        # Flamingo-80B
        frozen="vision encoder AND language model (both fully frozen)",
        data="~2B web image-text pairs + interleaved web pages (M3W)",
        can=dict(retrieval=False, zeroshot_cls=True, vqa=True, gen_image=False),
        summary=(
            "Flamingo keeps a big pretrained language model completely frozen and "
            "teaches it to see by *inserting new layers between its existing ones*. "
            "Those new layers are cross-attention: the text positions look up "
            "information in the image features. Each one starts behind a gate "
            "initialised to zero, so on the very first training step the whole model "
            "still behaves exactly like the original text-only LM — nothing is broken, "
            "and the gate opens gradually as training finds the visual signal useful. "
            "This is 'middle fusion': separate encoders, but they talk to each other "
            "*inside* the network rather than only at the output. Because it is trained "
            "to generate the next word, Flamingo can answer questions and describe "
            "images, which CLIP cannot — but the glue it needs is enormous (about 10 "
            "billion trainable parameters), which is exactly the cost the next two "
            "papers attack."
        ),
    ),
    dict(
        key="BLIP-2",
        title="BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Encoders and LLMs",
        year=2023,
        org="Salesforce",
        url="https://arxiv.org/abs/2301.12597",
        fusion="middle",
        objective="generative",      # stage 2; stage 1 also uses a contrastive term
        connector_params=188.0,      # the Q-Former
        total_params=3700.0,         # + frozen ViT-g and frozen OPT-2.7B
        frozen="vision encoder AND language model",
        data="129M image-text pairs (COCO, VG, CC, SBU, LAION subset)",
        can=dict(retrieval=True, zeroshot_cls=False, vqa=True, gen_image=False),
        summary=(
            "BLIP-2 asks how small the glue between a frozen image encoder and a frozen "
            "LLM can get. Its answer is the Q-Former: 32 learnable query vectors that "
            "cross-attend to the image and come out as exactly 32 'tokens' the LLM can "
            "read. The queries act as a fixed-width funnel — however many patches the "
            "image produced, the LLM always receives 32 slots — which is what keeps the "
            "cost from exploding. It is trained in two stages: first the Q-Former alone "
            "learns to pull out text-relevant visual detail (using a contrastive term "
            "among others), then it is attached to the LLM and trained to generate "
            "captions. At 188M trainable parameters it is roughly 50x cheaper glue than "
            "Flamingo's while reaching comparable captioning and VQA quality."
        ),
    ),
    dict(
        key="LLaVA",
        title="Visual Instruction Tuning (LLaVA)",
        year=2023,
        org="Wisconsin / Microsoft",
        url="https://arxiv.org/abs/2304.08485",
        fusion="middle",
        objective="generative",
        connector_params=21.0,       # LLaVA-1.5's two-layer MLP projector
        total_params=7000.0,         # + CLIP ViT-L/14 and Vicuna-7B
        frozen="vision encoder (LLM is unfrozen in stage 2)",
        data="558k caption pairs (stage 1) + 665k GPT-4-written instructions (stage 2)",
        can=dict(retrieval=False, zeroshot_cls=False, vqa=True, gen_image=False),
        summary=(
            "LLaVA throws out the Q-Former and replaces it with a two-layer MLP that "
            "maps each CLIP patch feature straight into the LLM's word-embedding space. "
            "The image simply becomes a run of extra 'words' spliced into the prompt. "
            "That is roughly 21M trainable parameters — 10x smaller glue than BLIP-2's, "
            "1/500th of Flamingo's — and it works about as well. What LLaVA spends its "
            "effort on instead is *data*: it asked GPT-4 to write realistic multi-turn "
            "questions and answers about images, then trained on those. The lesson the "
            "whole field took from it is that once the connector is good enough, extra "
            "connector cleverness buys much less than extra instruction data."
        ),
    ),
    dict(
        key="Chameleon",
        title="Chameleon: Mixed-Modal Early-Fusion Foundation Models",
        year=2024,
        org="Meta",
        url="https://arxiv.org/abs/2405.09818",
        fusion="early",
        objective="generative",
        connector_params=0.0,        # there is no connector: no separate towers to join
        total_params=34000.0,        # Chameleon-34B
        frozen="nothing (trained from scratch on all modalities together)",
        data="~9.2T mixed text and image tokens",
        can=dict(retrieval=False, zeroshot_cls=False, vqa=True, gen_image=True),
        summary=(
            "Chameleon removes the seam entirely. An image is first turned into 1,024 "
            "discrete codes by an image tokenizer, exactly the way a sentence is turned "
            "into word pieces, and then text codes and image codes are poured into one "
            "sequence with one shared vocabulary. A single transformer is trained on it "
            "with the ordinary next-token loss. There is no vision tower, no projector, "
            "no cross-attention layer — hence a connector cost of zero. Because image "
            "codes are just tokens like any other, the model can also *emit* them, which "
            "makes it the only model of the five that can produce an image as well as "
            "read one. The bill arrives as training cost and instability: everything is "
            "learned from scratch on trillions of tokens, and the paper spends much of "
            "its length on the normalization tricks needed to stop mixed-modal training "
            "from diverging."
        ),
    ),
]

FUSIONS = ["late", "middle", "early"]
OBJECTIVES = ["contrastive", "masked", "generative"]

# Cells our five papers do NOT occupy, with a real inhabitant named so the empty
# space reads as "the field went elsewhere" rather than "impossible".
EMPTY_CELL_HINTS = {
    ("late", "masked"): "FLAVA",
    ("late", "generative"): "ClipCap",
    ("middle", "contrastive"): "ALBEF",
    ("middle", "masked"): "ViLBERT",
    ("early", "contrastive"): "(rare)",
    ("early", "masked"): "BEiT-3",
}

CAPABILITY_LABELS = [
    ("retrieval", "image↔text\nretrieval"),
    ("zeroshot_cls", "zero-shot\nclassification"),
    ("vqa", "answer a\nquestion"),
    ("gen_image", "generate\nan image"),
]
