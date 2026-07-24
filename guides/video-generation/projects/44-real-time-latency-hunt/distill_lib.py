"""Turning thirty network passes per frame into one.

The deadline
------------
Interactive means the picture answers your finger.  At 30 frames per second
that is one frame every 33 milliseconds -- 1000 / 30 = 33.3 -- and the budget
covers *everything*: reading the button, running the model, drawing the result.
Project 41's world model spends 30 network passes on one frame.  Thirty passes
inside 33 ms means each pass gets about one millisecond, which no honest
implementation of anything is going to manage.

Two different ways to spend fewer passes
----------------------------------------
1. **Just ask for fewer.**  A [rectified flow](/shared/glossary/#rectified-flow)
   sampler is an ODE solver; running it with 4 steps instead of 30 is legal, it
   is simply a coarser integration.  Free to implement, and project 26 already
   showed it degrades gently.  It has a floor, though: at 1 step you are
   assuming the velocity field is constant along the entire path, and it is not.

2. **Train a student that does not need them.**  This is
   [distillation](/shared/glossary/#distillation), and the name is literal --
   you boil a slow teacher down to a fast student that keeps the part you
   wanted.  The recipe here is *consistency distillation*: the teacher's
   30-step [ODE](/shared/glossary/#ode) path from noise to a clean frame
   is computed once, and the student is trained so that from ANY point on that
   path it jumps straight to the same endpoint.  "Consistency" is the property
   being trained in: every point on one path must agree about where the path
   ends.

Why the student is trained to output the endpoint, not a velocity
-----------------------------------------------------------------
The teacher predicts a *direction to move*, which is only useful if you then
take many small steps.  The student predicts *the destination*, so one call is
already an answer.  That is the whole speed-up, and it is also why a student
cannot simply be the teacher with the step count turned down: they are
answering different questions.

What is deliberately NOT here
-----------------------------
The serving side -- batching, kernel fusion, quantisation, compilation -- is
[Inference Systems](../../../inference-systems/)' and
[AI Hardware](../../../ai-hardware/)' subject.  This project changes how many
times the model runs, not how fast one run is.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import world_lib as W                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)


@torch.no_grad()
def teacher_path(teacher, context, actions, steps, generator=None,
                 sigma=None):
    """Run the teacher's ODE and keep every point on the way.

    Returns (xs, ts) with xs[i] the state at time ts[i], xs[0] pure noise and
    xs[-1] the finished frame.  These pairs are the training data for the
    student: "from here, the answer is xs[-1]".
    """
    flow = W.FL.RectifiedFlow()
    shape = (context.shape[0], teacher.horizon, W.GRID, W.GRID)
    x = torch.randn(shape, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    xs = [x.clone()]
    for i in range(steps):
        t = ts[i].expand(shape[0])
        v = teacher(x, t * flow.T_SCALE, context, actions, sigma=sigma)
        x = x + (ts[i + 1] - ts[i]) * v
        xs.append(x.clone())
    return torch.stack(xs), ts


class Student(nn.Module):
    """Same U-Net as the teacher, read as 'jump to the answer'.

    Architecturally identical on purpose: if the student were also smaller, a
    quality drop could be blamed on capacity rather than on the number of
    steps, and the experiment would say nothing.  Only the meaning of the
    output changes -- teacher outputs a velocity, student outputs the clean
    frame.
    """

    def __init__(self, ctx=2, horizon=1, ctx_noise=True):
        super().__init__()
        self.net = W.ActionUNet(ctx=ctx, horizon=horizon, ctx_noise=ctx_noise)
        self.horizon = horizon

    def forward(self, x, t, context, actions, sigma=None):
        # The network's last layer is zero-initialised, so at the start of
        # training it predicts x itself -- a sane identity, not a black screen.
        return x + self.net(x, t, context, actions, sigma=sigma)

    @torch.no_grad()
    def sample(self, context, actions, steps=1, generator=None, sigma=None):
        """One call, or a few alternating jump/re-noise rounds.

        Multi-step consistency sampling works because the student can be
        re-entered: jump to a guess, push the guess part-way back towards noise,
        jump again.  Each round starts from a better place, so the answer
        sharpens -- at the cost of one more network pass.
        """
        flow = W.FL.RectifiedFlow()
        shape = (context.shape[0], self.horizon, W.GRID, W.GRID)
        x = torch.randn(shape, generator=generator)
        ts = torch.linspace(1.0, 0.0, steps + 1)
        for i in range(steps):
            t = ts[i].expand(shape[0])
            x0 = self(x, t * flow.T_SCALE, context, actions, sigma=sigma)
            if i < steps - 1:
                nt = ts[i + 1]
                noise = torch.randn(shape, generator=generator)
                x = (1 - nt) * x0 + nt * noise
            else:
                x = x0
        return x
