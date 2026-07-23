"""A language model as the director: one sentence in, a shot list out.

Where this fits
---------------
Projects 35 and 36 asked *how* to stitch shots together.  This one asks a
question that comes earlier: **what should the shots be?**  A minute of story
is not one prompt.  It is a list of moments in a deliberate order, and
somebody has to write that list.

"The video model already reads text — why add a second model?"
--------------------------------------------------------------
This is the objection to answer first, because the video model does contain a
text encoder (a frozen T5, from project 30) and it does read English.  But
look at what that encoder is for: it turns ONE caption into a set of vectors
describing ONE 16-frame clip.  It has no notion of before and after, no memory
between calls, and nothing in its training ever asked it to decide what should
happen next.  Hand it "a knight rescues a princess" and it will try to draw
that entire story into two seconds.

The language model fills a different gap: **decomposition and ordering**.  It
turns one sentence into four, in a sequence that makes narrative sense, and it
keeps the protagonist the same across them.  It never touches a pixel.  Then
the video model does what it is good at — rendering one short moment at a time.
That division of labour is what "hierarchical generation" means in practice,
and it is how VideoTetris and MovieDreamer are organised.

The renderer's vocabulary
-------------------------
Our video model can draw exactly one thing: a handwritten digit sliding in one
of four directions.  So the "studio" the director is writing for has a tiny
vocabulary, and the prompt says so.  This is not a toy simplification of the
real problem — it *is* the real problem.  Every production shot planner has to
be told what its renderer can and cannot do, and most of the failures are the
planner asking for something the renderer has no way to make.

Three ways to get a shot list, and what each one is really for
--------------------------------------------------------------
    zero_shot     just ask.  Measures whether a 0.5B model can follow a
                  format instruction unaided.
    few_shot      ask, after showing two finished examples.  Examples are a
                  cheaper, faster lever than fine-tuning.
    constrained   never let it write free text at all: we write the JSON
                  skeleton and the model only *chooses* between the legal
                  values at each slot, by scoring them.  Invalid output
                  becomes impossible rather than unlikely.
"""

import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "35-sliding-window-t2v"))
import long_lib as LL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MOTIONS = LL.L.DIRECTIONS                     # right, down, left, up
N_SHOTS = 4
ARMS = ["zero_shot", "few_shot", "constrained", "random"]

STORIES = [
    "a knight rescues a princess",
    "a robot learns to dance",
    "a cat chases a red laser dot across the kitchen",
    "a paper plane flies across the city and lands on a roof",
    "a diver swims down to a shipwreck and finds a lantern",
    "a farmer walks the field at dawn and opens the gate",
    "a detective follows a suspect through a rainy market",
    "a comet passes the earth and disappears behind the moon",
]

RULES = f"""You are a shot planner for a very small animation studio.

The studio can only render ONE kind of shot: a single handwritten digit
(0-9) sliding steadily in one of four directions ({', '.join(MOTIONS)}).
The digit stands in for the main character of the story.

Write a shot list of exactly {N_SHOTS} shots as a JSON array.  Every element
must be an object with exactly these keys:
  "shot"    : the shot number, 1 to {N_SHOTS}
  "subject" : one integer 0-9, the digit that plays the main character
  "motion"  : one of {MOTIONS}
  "caption" : a short sentence describing the moment

Keep "subject" the SAME in every shot: it is the same character throughout.
Reply with the JSON array and nothing else."""

EXAMPLES = [
    ("a mouse steals a piece of cheese",
     [{"shot": 1, "subject": 2, "motion": "right",
       "caption": "the mouse creeps out towards the kitchen"},
      {"shot": 2, "subject": 2, "motion": "up",
       "caption": "it climbs the table leg"},
      {"shot": 3, "subject": 2, "motion": "left",
       "caption": "it drags the cheese back across the table"},
      {"shot": 4, "subject": 2, "motion": "down",
       "caption": "it drops down and escapes into its hole"}]),
    ("a balloon escapes a child at the fair",
     [{"shot": 1, "subject": 7, "motion": "up",
       "caption": "the balloon slips out of the child's hand"},
      {"shot": 2, "subject": 7, "motion": "right",
       "caption": "the wind carries it over the stalls"},
      {"shot": 3, "subject": 7, "motion": "up",
       "caption": "it rises above the ferris wheel"},
      {"shot": 4, "subject": 7, "motion": "left",
       "caption": "it drifts away over the rooftops"}]),
]


# ---------------------------------------------------------------------------
# the language model
# ---------------------------------------------------------------------------

_LLM = None


def load_llm():
    global _LLM
    if _LLM is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(LLM_ID)
        model = AutoModelForCausalLM.from_pretrained(LLM_ID).eval()
        _LLM = (model, tok)
    return _LLM


def chat_prefix(tok, story, few_shot=False):
    msgs = [{"role": "system", "content": RULES}]
    if few_shot:
        for s, plan in EXAMPLES:
            msgs.append({"role": "user", "content": f"Story: {s}"})
            msgs.append({"role": "assistant",
                         "content": json.dumps(plan, indent=None)})
    msgs.append({"role": "user", "content": f"Story: {story}"})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate_plan(story, few_shot=False, max_new_tokens=200):
    """Let the model write freely, then try to find JSON in what it wrote."""
    model, tok = load_llm()
    prefix = chat_prefix(tok, story, few_shot)
    ids = tok(prefix, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


@torch.no_grad()
def score_options(model, tok, prefix_ids, options):
    """Total log-probability the model gives each candidate continuation.

    This is how "constrained decoding" is done in its simplest honest form:
    instead of sampling tokens and hoping the result is legal, we enumerate
    the legal answers and ask the model which it prefers.  The output cannot
    be malformed, because we never let the model choose the punctuation.

    Two things make this fast enough to be practical:

    * **One batched forward for all candidates.**  Every row is the same
      prefix followed by a different candidate, padded on the right.  Padding
      on the right is safe here because attention is causal — a token can only
      look backwards, so pad tokens cannot influence anything before them.
    * **`logits_to_keep`.**  Without it the model materialises a full
      (candidates x prefix_length x 150k-word) tensor of scores, most of which
      we throw away; it is gigabytes, and it is the difference between this
      running in seconds and in minutes.
    """
    ids = [tok(o, add_special_tokens=False,
               return_tensors="pt")["input_ids"][0] for o in options]
    lens = [len(i) for i in ids]
    lmax = max(lens)
    pad = tok.pad_token_id or tok.eos_token_id
    rows = []
    for i in ids:
        tail = torch.full((lmax - len(i),), pad, dtype=torch.long)
        rows.append(torch.cat([prefix_ids[0], i, tail]))
    batch = torch.stack(rows)
    logits = model(batch, logits_to_keep=lmax + 1).logits.log_softmax(-1)
    scores = []
    for r, (i, n) in enumerate(zip(ids, lens)):
        lp = logits[r, :n].gather(-1, i[:, None])[:, 0]
        scores.append(float(lp.sum()))
    return scores


@torch.no_grad()
def constrained_plan(story):
    """Fill a JSON skeleton slot by slot; only legal values are ever offered.

    The subject is chosen ONCE and reused, because the rules say the character
    is the same in every shot — so scoring ten digits at every shot would both
    waste compute and invite the model to (wrongly) change the character
    mid-story.  That is a small example of a general principle: a constraint
    the plan must satisfy is better built into the decoder than hoped for from
    the model.  The prefix here is the zero-shot one (no worked examples),
    which keeps each scoring forward short enough to run on a CPU in seconds.
    """
    model, tok = load_llm()
    prefix = chat_prefix(tok, story, few_shot=False)
    text = prefix + '[{"shot": 1, "subject": '
    ids = tok(text, return_tensors="pt")["input_ids"]
    sc = score_options(model, tok, ids, [str(d) for d in range(10)])
    subject = int(max(range(10), key=lambda d: sc[d]))
    plan = []
    for i in range(N_SHOTS):
        text += (f'{subject}, "motion": "' if i == 0
                 else f'{{"shot": {i+1}, "subject": {subject}, "motion": "')
        ids = tok(text, return_tensors="pt")["input_ids"]
        sc = score_options(model, tok, ids, MOTIONS)
        motion = MOTIONS[int(max(range(len(MOTIONS)), key=lambda m: sc[m]))]
        text += f'{motion}"}}' + (", " if i < N_SHOTS - 1 else "]")
        plan.append({"shot": i + 1, "subject": subject, "motion": motion,
                     "caption": ""})
    return plan


# ---------------------------------------------------------------------------
# reading whatever came back
# ---------------------------------------------------------------------------

def extract_json(text):
    """Pull the first JSON array out of a reply.  Returns None if there is none.

    Real systems need this because models wrap JSON in prose, in Markdown
    fences, or in both.  Counting how often the fallback is needed is part of
    the measurement.
    """
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate(plan):
    """Is this a shot list we could actually render?  Returns (ok, reasons)."""
    why = []
    if not isinstance(plan, list):
        return False, ["not a list"]
    if len(plan) != N_SHOTS:
        why.append(f"{len(plan)} shots, wanted {N_SHOTS}")
    for i, sh in enumerate(plan[:N_SHOTS]):
        if not isinstance(sh, dict):
            why.append(f"shot {i+1} is not an object")
            continue
        s = sh.get("subject")
        if not isinstance(s, int) or not 0 <= s <= 9:
            why.append(f"shot {i+1} subject {s!r}")
        if sh.get("motion") not in MOTIONS:
            why.append(f"shot {i+1} motion {sh.get('motion')!r}")
    return (len(why) == 0), why


def repair(plan):
    """Patch a nearly-good plan into a renderable one.

    A planner is not much use if one bad field throws the whole story away, so
    production pipelines always have a repair step.  Reporting the repair RATE
    alongside the raw validity rate is the honest way to present it: repair
    hides failures, it does not remove them.
    """
    fixed, changed = [], 0
    plan = plan if isinstance(plan, list) else []
    for i in range(N_SHOTS):
        sh = plan[i] if i < len(plan) and isinstance(plan[i], dict) else {}
        s = sh.get("subject")
        if not isinstance(s, int) or not 0 <= s <= 9:
            s, changed = (fixed[0]["subject"] if fixed else 0), changed + 1
        m = sh.get("motion")
        if m not in MOTIONS:
            m, changed = MOTIONS[i % len(MOTIONS)], changed + 1
        fixed.append({"shot": i + 1, "subject": s, "motion": m,
                      "caption": sh.get("caption", "")})
    return fixed, changed


def plan_stats(plan):
    subj = [sh["subject"] for sh in plan]
    mot = [sh["motion"] for sh in plan]
    return dict(subject_consistency=sum(s == subj[0] for s in subj) / len(subj),
                motion_variety=len(set(mot)) / len(mot),
                subject=subj[0])


def to_render(plan):
    """A validated plan -> the two things `generate_long` needs."""
    schedule = [MOTIONS.index(sh["motion"]) for sh in plan]
    subjects = [int(sh["subject"]) for sh in plan]
    return schedule, subjects


__all__ = ["LLM_ID", "MOTIONS", "N_SHOTS", "ARMS", "STORIES", "RULES",
           "EXAMPLES", "load_llm", "generate_plan", "constrained_plan",
           "extract_json", "validate", "repair", "plan_stats", "to_render",
           "LL"]
