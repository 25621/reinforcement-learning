"""A miniature vision-language-action model, and the task it has to be told.

A **VLA** takes a picture and a sentence and emits an action.  The three
letters are the whole architecture: Vision in, Language in, Action out.  The
interesting claim about them is not that they work -- a small network can be
trained to push a puck -- but that a VLA *pretrained on lots of other tasks*
learns a new one from far fewer demonstrations than the same network trained
from scratch.  This file builds the smallest honest test of that claim.

The task has to have language in it
-----------------------------------
Project 54's push task cannot test a VLA, because there is nothing to say: one
puck, one goal, no ambiguity, and a policy that ignores the instruction scores
exactly as well as one that reads it.  So the scene here has **two coloured
discs** on the table and the instruction names one of them::

    "touch the blue marker"

Now the same picture has two correct answers and the sentence picks between
them.  A policy that ignores the words can do no better than a coin flip -- and
that coin flip is the control every experiment below is measured against.

The task itself is deliberately a **reach**, not a push.  Pushing was tried
first and it is the wrong instrument here: a 32x32 camera cannot resolve a
3 cm contact well enough for contact-rich control, so the score ended up
measuring image resolution rather than grounding, which is the thing this
project is about.  Reaching keeps the control trivial and leaves the whole
difficulty where it belongs -- in reading the picture and the sentence.
Project 73 and 74 use the push; this one does not need it.

The held-out concept
--------------------
Pretraining scenes contain a **red** and a **blue** target.  The fine-tuning
task introduces a **green** one, which the model has never seen, named by a
word it has never read.  That is the realistic version of "fine-tune an open
VLA on your task": your objects are not in its training set.

Colours are rendered as three brightness levels in one channel, so telling them
apart is a real visual discrimination and not a lookup of which channel is
non-zero.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

# 40 x 40 pixels over a 0.34 m half-width -> 17 mm per pixel.
# The first version used 32 pixels over 0.42 m, which is 26 mm per
# pixel, and asked the policy to stop within 35 mm of a disc: one and a
# third pixels.  No amount of training fixes a target the camera cannot
# resolve, and the score was measuring resolution rather than grounding.
IMG = 40
CROP = 12
EXTENT = 0.34
CX = CY = (IMG - 1) / 2.0
R_TARGET = 0.035

_gx, _gy = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="xy")
_PIX = np.stack([(_gx - CX) * (2 * EXTENT / IMG),
                 (CY - _gy) * (2 * EXTENT / IMG)], -1)

COLOURS = {"red": 1.00, "blue": 0.60, "green": 0.30}


def _disc(centre, radius, soft=0.010):
    d = np.linalg.norm(_PIX - np.asarray(centre, float), axis=-1)
    return np.clip((radius + soft - d) / soft, 0.0, 1.0)


def _segment(p0, p1, radius, soft=0.008):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    t = np.clip(((_PIX - p0) @ v) / (float(v @ v) + 1e-12), 0.0, 1.0)
    d = np.linalg.norm(_PIX - (p0 + t[..., None] * v), axis=-1)
    return np.clip((radius + soft - d) / soft, 0.0, 1.0)


# ---------------------------------------------------------------------------
# the language
# ---------------------------------------------------------------------------
VERBS = ["touch", "reach", "go to", "point at"]
NOUNS = ["target", "circle", "spot", "marker"]


def instruction(rng, colour):
    return f"{rng.choice(VERBS)} the {colour} {rng.choice(NOUNS)}"


ALL_SENTENCES = [f"{v} the {c} {n}"
                 for v in VERBS for c in COLOURS for n in NOUNS]


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------
TOL = 0.050                # how close the tip must get (about 3 pixels)
EP_LEN = 26


class TwoTargetEnv:
    """Two coloured discs on a table, a two-link arm, and one sentence.

    Built on project 54's arm rather than its push task: the robot and its
    servo are identical, only the goal is different.
    """

    def __init__(self, rng, palette=("red", "blue")):
        self.rng = rng
        self.palette = list(palette)
        self.arm = A.PlanarArm()

    def reset(self):
        a = self.arm
        lo, hi = 0.38 * a.reach, 0.68 * a.reach
        pts = []
        while len(pts) < 2:
            th = self.rng.uniform(-0.5, 1.5)
            rr = self.rng.uniform(lo, hi)
            p = np.array([rr * np.cos(th), rr * np.sin(th)])
            if all(np.linalg.norm(p - q) > 0.11 for q in pts):
                pts.append(p)
        # Which disc is scored is decided independently of where the discs are,
        # so "always go to the brighter one" and "always go to the left one"
        # are both worth exactly a coin flip.
        i = 0 if self.rng.random() < 0.5 else 1
        self.named, self.other_colour = self.palette[i], self.palette[1 - i]
        self.pos_named, self.pos_other = pts[0], pts[1]
        if self.rng.random() < 0.5:
            self.pos_named, self.pos_other = self.pos_other, self.pos_named
        # The detector reports the two discs in a random order and says nothing
        # about their colour.  Shuffling matters: with a fixed order the policy
        # could score 1.0 by always driving to the first slot and never look at
        # the picture at all.
        self.slots = ([self.pos_named, self.pos_other]
                      if self.rng.random() < 0.5
                      else [self.pos_other, self.pos_named])
        self.instr = instruction(self.rng, self.named)
        self.q = np.array([self.rng.uniform(-0.2, 0.9),
                           self.rng.uniform(0.8, 2.0)])
        self.qd = np.zeros(2)
        self.t = 0
        return self.image()

    # -- physics ------------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        q_cmd = self.q + A.DQ_MAX * a
        for _ in range(A.SUBSTEPS):
            tau = self.arm.servo_torque(self.q, self.qd, q_cmd)
            self.q, self.qd = self.arm.step(self.q, self.qd, tau)
        self.t += 1
        err = self.err
        done = self.t >= EP_LEN or err < TOL
        return self.image(), -err, done, {"err": err, "success": err < TOL}

    @property
    def err(self):
        return float(np.linalg.norm(self.arm.tip(self.q) - self.pos_named))

    def went_to_decoy(self):
        return (np.linalg.norm(self.arm.tip(self.q) - self.pos_other) < TOL
                and self.err >= TOL)

    # -- what the network sees ----------------------------------------------
    def image(self):
        pts = self.arm.points(self.q)
        arm = np.zeros((IMG, IMG), np.float32)
        for k in range(self.arm.n):
            arm = np.maximum(arm, _segment(pts[k], pts[k + 1], 0.012))
        arm = np.maximum(arm, _disc(pts[-1], A.R_TIP))
        tgt = (_disc(self.pos_named, R_TARGET) * COLOURS[self.named]
               + _disc(self.pos_other, R_TARGET) * COLOURS[self.other_colour])
        return np.stack([arm, np.clip(tgt, 0, 1)]).astype(np.float32)

    def crops(self):
        """A CROP square of the picture around each detected disc.

        This is the standard shape of a real perception stack: a detector says
        *where* things are, and a recogniser looks at each place in turn to say
        *what* is there.  The network gets two little pictures, in the same
        random order as the two positions in ``proprio``, and has to decide
        which one the sentence is talking about.

        Cropping is not the project doing the perception on the model's behalf.
        The crops still contain only brightness -- nothing labels them "red" or
        "blue" -- so the mapping from the *word* to the *appearance* is still
        entirely learned, and that mapping is what grounding means.
        """
        img = self.image()[1]
        out = np.zeros((2, CROP, CROP), np.float32)
        for i, p in enumerate(self.slots):
            cx = int(round(p[0] / (2 * EXTENT / IMG) + CX))
            cy = int(round(CY - p[1] / (2 * EXTENT / IMG)))
            x0 = int(np.clip(cx - CROP // 2, 0, IMG - CROP))
            y0 = int(np.clip(cy - CROP // 2, 0, IMG - CROP))
            out[i] = img[y0:y0 + CROP, x0:x0 + CROP]
        return out

    def proprio(self):
        """What a real robot reads off its own joints.

        The two disc positions are in here, in a random order, the way an
        object detector would report them -- but **nothing here says which disc
        is which colour**.  That is only in the picture, and which colour is
        wanted is only in the sentence.  So the policy still cannot do the task
        without reading both, and what it has to read from the image is a
        coarse judgement ("which of these two blobs is the bright one") rather
        than a sub-pixel position.

        This split is deliberate.  An earlier version withheld the positions
        and asked the CNN to regress them, and the score then measured how well
        a 40 x 40 image localises a 3.5 cm disc -- image resolution, not
        grounding.  Handing over the detections isolates the thing this project
        is about.  Real VLA stacks are built the same way, on detected object
        poses plus an image plus a sentence.
        """
        tip = self.arm.tip(self.q)
        # Positions are divided by EXTENT so that every entry is order 1.  The
        # sines and cosines already are, and a raw tip position in metres is
        # five times smaller than they are -- which is exactly the entry the
        # task depends on most finely.  Project 54 met the same trap and fixed
        # it with a whitening layer; one constant does the job here.
        return np.concatenate([np.cos(self.q), np.sin(self.q), self.qd / 10.0,
                               tip / EXTENT, self.slots[0] / EXTENT,
                               self.slots[1] / EXTENT]).astype(np.float32)


def expert_action(env, noise=0.0, rng=None):
    """Move the tip at the named disc, through damped least squares."""
    tip = env.arm.tip(env.q)
    v = (env.pos_named - tip) * 9.0
    sp = np.linalg.norm(v)
    if sp > 0.5:
        v = v * (0.5 / sp)
    J = env.arm.jacobian(env.q)
    dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(2), v * A.CTRL_DT)
    a = dq / A.DQ_MAX
    if noise and rng is not None:
        a = a + rng.normal(0, noise, a.shape)
    return np.clip(a, -1.0, 1.0)


# ---------------------------------------------------------------------------
# the language encoders
# ---------------------------------------------------------------------------
class BagOfWords:
    """One learned vector per word, averaged.  The cheap baseline."""

    def __init__(self, sentences, dim=32, seed=0):
        vocab = sorted({w for s in sentences for w in s.split()})
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.dim = dim
        g = torch.Generator().manual_seed(seed)
        self.table = nn.Embedding(len(vocab), dim)
        with torch.no_grad():
            self.table.weight.copy_(torch.randn(len(vocab), dim,
                                                generator=g) * 0.1)

    def __call__(self, sentences):
        out = []
        for s in sentences:
            ids = torch.tensor([self.idx[w] for w in s.split()
                                if w in self.idx])
            out.append(self.table(ids).mean(0))
        return torch.stack(out)

    def parameters(self):
        return self.table.parameters()


def frozen_llm_embeddings(sentences, model_id="HuggingFaceTB/SmolLM2-135M-Instruct"):
    """Mean-pooled hidden states from a real, frozen language model.

    Why bother, when a 32-number lookup table can already tell four colours
    apart?  Because the lookup table only knows words it was trained on, and a
    real robot gets sentences nobody enumerated.  The frozen model is the part
    that is supposed to generalise -- it has read "green" a million times.  The
    experiment in run.py measures whether that actually buys anything *here*,
    where the vocabulary is closed and tiny.

    All sentences are embedded ONCE and cached: there are 48 of them, so a
    frozen encoder in the training loop would be recomputing the same 48
    vectors thousands of times.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(sentences), 16):
            batch = sentences[i:i + 16]
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            enc = tok(batch, return_tensors="pt", padding=True)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].float()
            v = (h * m).sum(1) / m.sum(1)
            for s, vv in zip(batch, v):
                out[s] = vv.float()
    return out


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class VLA(nn.Module):
    """Vision-Language-Action, in the smallest form that still has all three.

    The sentence does not steer the arm directly.  It steers *attention*: a
    small shared CNN looks at each detected disc, the sentence modulates those
    features through FiLM, a score per disc comes out, a softmax turns the two
    scores into weights, and the weighted average of the two disc positions is
    the goal handed to the action head.

    Why a softmax over detections rather than concatenating everything into one
    MLP?  Because the question the language has to answer is a *choice between
    two things*, and a flat network has to discover the comparison from data.
    Making the comparison explicit costs three lines and turns an ungrounded
    policy (measured: the same score with and without the sentence) into a
    grounded one.  FiLM -- Feature-wise Linear Modulation -- multiplies rather
    than concatenates for the same reason: a multiplier can switch a visual
    feature off, which is exactly what "not that one, the blue one" has to do.
    """

    def __init__(self, lang_dim=32, ch=16, hidden=128, prop_dim=12, feat=64):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(1, ch, 3, 2, 1), nn.GELU(),            # 12 -> 6
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.GELU(),       # 6 -> 3
            nn.Flatten(), nn.Linear(ch * 2 * 9, feat), nn.GELU(),
        )
        self.film = nn.Linear(lang_dim, feat * 2)
        self.score = nn.Sequential(nn.Linear(feat, feat), nn.GELU(),
                                   nn.Linear(feat, 1))
        self.head = nn.Sequential(
            nn.Linear(prop_dim + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def attend(self, crops, lang):
        b = crops.shape[0]
        f = self.vision(crops.reshape(b * 2, 1, crops.shape[-2],
                                      crops.shape[-1])).reshape(b, 2, -1)
        g, bias = self.film(lang).chunk(2, -1)
        f = f * (1.0 + g[:, None]) + bias[:, None]
        return torch.softmax(self.score(f).squeeze(-1), -1)

    def forward(self, crops, lang, prop):
        w = self.attend(crops, lang)
        slots = prop[:, -4:].reshape(-1, 2, 2)
        goal = (w[..., None] * slots).sum(1)
        return self.head(torch.cat([prop, goal], -1))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def collect(n_demos, palette=("red", "blue"), seed=0, noise=0.15):
    """Language-conditioned demonstrations."""
    rng = np.random.default_rng(seed)
    env = TwoTargetEnv(rng, palette=palette)
    IM, LA, PR, AC, ends = [], [], [], [], []
    got = tries = 0
    while got < n_demos and tries < n_demos * 6:
        tries += 1
        env.reset()
        im, pr, ac = [], [], []
        done = False
        while not done:
            a = expert_action(env, noise=noise, rng=rng)
            im.append(env.crops())
            pr.append(env.proprio())
            ac.append(a)
            _, _, done, info = env.step(a)
        if not info["success"]:
            continue
        IM += im
        PR += pr
        AC += ac
        LA += [env.instr] * len(im)
        got += 1
        ends.append(len(IM))
    return (np.array(IM, np.float32), LA, np.array(PR, np.float32),
            np.array(AC, np.float32), ends)


def take(data, n_demos):
    """The first ``n_demos`` complete episodes of a collected set."""
    IM, LA, PR, AC, ends = data
    end = ends[min(n_demos, len(ends)) - 1]
    return IM[:end], LA[:end], PR[:end], AC[:end]


def train(model, data, lang_fn, epochs=60, bs=128, lr=1e-3, seed=0,
          params=None, log=None):
    torch.manual_seed(seed)
    IM, LA, PR, AC = data
    Xi = torch.tensor(IM)
    Xp = torch.tensor(PR)
    Y = torch.tensor(AC)
    ps = list(model.parameters()) if params is None else list(params)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xi)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            # The sentence vectors are recomputed per batch rather than once
            # up front.  Computing them once builds a single autograd graph
            # that the second .backward() of the epoch would try to walk
            # again -- and, more importantly, the encoder is being TRAINED
            # here, so its output has to be part of this batch's graph.
            lb = lang_fn([LA[j] for j in b.tolist()])
            loss = ((model(Xi[b], lb, Xp[b]) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        if log and (ep + 1) % 20 == 0:
            print(f"  {log} epoch {ep + 1:3d}  mse {tot / n:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, lang_fn, palette, n=60, seed=1000, blind=False):
    rng = np.random.default_rng(seed)
    env = TwoTargetEnv(rng, palette=palette)
    ok = decoy = 0
    for _ in range(n):
        env.reset()
        lang = lang_fn([env.instr])
        if blind:
            lang = torch.zeros_like(lang)
        done = False
        while not done:
            img = torch.tensor(env.crops())[None]
            pr = torch.tensor(env.proprio())[None]
            a = model(img, lang, pr)[0].numpy()
            _, _, done, info = env.step(a)
        ok += info["success"]
        decoy += env.went_to_decoy()
    return {"success": ok / n, "decoy": decoy / n}
