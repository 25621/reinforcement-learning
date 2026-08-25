"""Project 15 — Implement AdamW from scratch.

Two moments, bias correction, and the one line that separates AdamW from Adam.

  1. the update rule, written out
  2. bit-exact against torch.optim.Adam and torch.optim.AdamW
  3. bias correction: what the zero-initialized moments actually do
  4. Adam's scale invariance -- the property that makes it hard to break
  5. decoupled weight decay: measuring what "decoupled" means
  6. epsilon: where it sits and what it costs
  7. memory: three copies of the model

Runs in about 15 seconds on CPU. No downloads.
"""

import csv
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(4)
FINDINGS = OrderedDict()


def rec(k, v):
    FINDINGS[k] = v
    return v


# =========================================================================
# the optimizer
# =========================================================================
class MyAdamW(Optimizer):
    """torch.optim.AdamW and torch.optim.Adam, in one class.

    decoupled=True  -> AdamW: the decay multiplies the parameter directly
    decoupled=False -> Adam : the decay is added to the gradient (L2)
    bias_correction=False is there so section 3 can turn it off.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=True, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      decoupled=decoupled,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            wd, decoupled, bc = group["weight_decay"], group["decoupled"], group["bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)       # first moment
                    state["v"] = torch.zeros_like(p)       # second moment
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]

                if wd != 0:
                    if decoupled:
                        p.mul_(1 - lr * wd)                # AdamW: shrink the weight
                    else:
                        g = g.add(p, alpha=wd)             # Adam + L2: shrink the gradient

                m.mul_(b1).add_(g, alpha=1 - b1)           # m = b1*m + (1-b1)*g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)    # v = b2*v + (1-b2)*g^2

                if bc:
                    # torch's formulation: fold both corrections into lr and denom
                    step_size = lr / (1 - b1 ** t)
                    denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
                else:
                    step_size = lr
                    denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-step_size)


# =========================================================================
# the toy problem
# =========================================================================
def data(n=512, d=20, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    W = torch.randn(d, c, generator=g)
    y = (X @ W + 0.3 * torch.randn(n, c, generator=g)).argmax(1)
    return X, y


X, Y = data()


def model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 3))


def train(make_opt, steps=200, seed=0, scale=1.0):
    m = model(seed)
    opt = make_opt(m.parameters())
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(X), Y) * scale
        loss.backward()
        opt.step()
        losses.append(loss.item() / scale)
    return m, np.array(losses)


def max_param_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


# =========================================================================
# 1 + 2
# =========================================================================
def rule_and_match():
    print("=" * 78)
    print("1. THE UPDATE RULE")
    print("=" * 78)
    print("""
      m = b1*m + (1-b1)*g            first moment  : the average gradient
      v = b2*v + (1-b2)*g*g          second moment : the average SQUARED gradient

      m_hat = m / (1 - b1^t)         bias correction
      v_hat = v / (1 - b2^t)

      p -= lr * m_hat / (sqrt(v_hat) + eps)
    """)
    print("  Read the last line as: step in the averaged gradient direction, but")
    print("  divide by how big this parameter's gradients usually are. A parameter")
    print("  with consistently large gradients gets a proportionally smaller step.")
    print("  So Adam does not choose one learning rate for the model -- it chooses")
    print("  a different effective one for every single weight, from that weight's")
    print("  own history. That is what 'adaptive' means in 'Adaptive Moment")
    print("  estimation', which is where the name Adam comes from (not a person).")
    print()

    print("=" * 78)
    print("2. BIT-EXACT AGAINST TORCH")
    print("=" * 78)
    worst = 0.0
    checks = [
        ("Adam,  wd=0", dict(weight_decay=0.0, decoupled=False),
         lambda p: torch.optim.Adam(p, lr=1e-2)),
        ("Adam,  wd=1e-2 (L2)", dict(weight_decay=1e-2, decoupled=False),
         lambda p: torch.optim.Adam(p, lr=1e-2, weight_decay=1e-2)),
        ("AdamW, wd=1e-2 (decoupled)", dict(weight_decay=1e-2, decoupled=True),
         lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2)),
        ("AdamW, wd=0.1", dict(weight_decay=0.1, decoupled=True),
         lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.1)),
    ]
    print(f"  {'configuration':<30}{'after 1 step':>15}{'after 200 steps':>18}")
    for label, kw, ref in checks:
        a1, _ = train(lambda p, kw=kw: MyAdamW(p, lr=1e-2, **kw), steps=1)
        b1, _ = train(ref, steps=1)
        a, _ = train(lambda p, kw=kw: MyAdamW(p, lr=1e-2, **kw))
        b, _ = train(ref)
        d1, d = max_param_diff(a1, b1), max_param_diff(a, b)
        worst = max(worst, d)
        print(f"  {label:<30}{d1:>15.3e}{d:>18.3e}")
    rec("bitexact_worst", worst)
    print()
    print("  Every configuration is BIT-IDENTICAL for one step, and about 1e-6 apart")
    print("  after two hundred. Nothing is wrong: the formulas match exactly, and the")
    print("  two implementations evaluate them in slightly different orders (torch")
    print("  computes sqrt(1 - b2^t) with its own dispatcher, we use math.sqrt). In")
    print("  float32 those last-bit differences feed back through the next gradient")
    print("  and grow -- project 7's chaos result, arriving in a place where you")
    print("  would rather it did not.")
    print()
    print("  So: check step 1 for correctness. Never check step 200.")
    print()

    a, _ = train(lambda p: MyAdamW(p, lr=1e-2, weight_decay=0.0, decoupled=True))
    b, _ = train(lambda p: MyAdamW(p, lr=1e-2, weight_decay=0.0, decoupled=False))
    print(f"  AdamW vs Adam with weight_decay=0: max |diff| {max_param_diff(a, b):.3e}")
    print("  With no weight decay there is nothing to decouple, and the two are the")
    print("  same optimizer. Every difference discussed below lives in that one term.")
    print()
    rec("adamw_vs_adam_no_wd", max_param_diff(a, b))


# =========================================================================
# 3. bias correction
# =========================================================================
def bias_correction():
    print("=" * 78)
    print("3. BIAS CORRECTION: WHAT THE ZERO-INITIALIZED MOMENTS DO")
    print("=" * 78)

    b1, b2 = 0.9, 0.999
    print("  Feed a constant gradient of 1.0 to a single weight, lr = 1.0, and")
    print("  measure the actual step taken with and without the correction:")
    print()
    print(f"  {'step':>6}{'corrected':>12}{'uncorrected':>14}{'ratio':>9}"
          f"{'predicted':>12}")
    steps_taken = {True: [], False: []}
    for bc in (True, False):
        w = nn.Parameter(torch.zeros(1))
        opt = MyAdamW([w], lr=1.0, betas=(b1, b2), bias_correction=bc)
        for _ in range(400):
            w.grad = torch.ones(1)
            before = w.item()
            opt.step()
            steps_taken[bc].append(abs(w.item() - before))
    for t in (1, 2, 5, 20, 100, 400):
        c, u = steps_taken[True][t - 1], steps_taken[False][t - 1]
        pred = math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        print(f"  {t:>6}{c:>12.4f}{u:>14.4f}{u / c:>9.2f}{1 / pred:>12.2f}")
    print()
    ratio1 = steps_taken[False][0] / steps_taken[True][0]
    print(f"  The first uncorrected step is {ratio1:.2f}x TOO BIG, not too small.")
    print()
    print("  Both moments start at zero, so both are biased toward zero early on.")
    print("  The trap is that they are biased by DIFFERENT amounts, and the update")
    print("  is their ratio:")
    print()
    print("    m_1 = (1-b1)*g       = 0.100*g        10x too small")
    print("    v_1 = (1-b2)*g^2     = 0.001*g^2    1000x too small")
    print("    sqrt(v_1)            = 0.0316*|g|     32x too small")
    print("    m_1 / sqrt(v_1)      = 3.16*sign(g)   3.16x too BIG")
    print()
    print("  The square root halves the second moment's error in log terms, so the")
    print("  denominator shrinks less than the numerator and the quotient comes out")
    print("  too large. Dividing each moment by (1 - beta^t) -- exactly the sum of")
    print("  the weights that were actually used -- turns both into honest averages,")
    print("  and the first step becomes exactly lr.")
    print()
    print("  Look at the middle of the ratio column: it does not fade smoothly, it")
    print(f"  climbs to about {max(u / c for u, c in zip(steps_taken[False][:60], steps_taken[True][:60])):.1f}x around step 20 and only then comes down")
    print(f"  ({steps_taken[False][399] / steps_taken[True][399]:.2f}x at step 400). The two moments have different memories:")
    print("  b1 = 0.9 forgets its zero start in ~10 steps, while b2 = 0.999 needs")
    print("  ~1000. So the numerator recovers first and the ratio gets WORSE before")
    print("  it gets better.")
    print()
    print("  Bias correction is therefore a warmup schedule built into the optimizer,")
    print("  and one that would otherwise run backwards.")
    print()
    rec("bias_first_step_ratio", ratio1)
    rec("bias_step20_ratio", steps_taken[False][19] / steps_taken[True][19])

    a, la = train(lambda p: MyAdamW(p, lr=1e-2, bias_correction=True))
    b, lb = train(lambda p: MyAdamW(p, lr=1e-2, bias_correction=False))
    print(f"  On the real problem: corrected loss {la[-1]:.4f}, uncorrected {lb[-1]:.4f},")
    print(f"  max weight difference {max_param_diff(a, b):.3e}")
    print()
    print("  Read that honestly: on this easy problem the UNCORRECTED version wins,")
    print("  because a 3-6x bigger step early is simply a bigger learning rate and")
    print("  this loss surface forgives it. The weights end up 1.3 apart, so the two")
    print("  are genuinely different runs -- one of them just got lucky.")
    print()
    print("  The reason correction is not optional is that the overshoot is UNASKED")
    print("  FOR and its size depends on your betas, not on your problem. Raise b2 to")
    print("  0.999 and the first step is 3.2x lr; raise it to 0.9999 and it is 10x lr.")
    print("  A hyperparameter that silently rescales your learning rate by a factor")
    print("  you did not compute is a bug, even on the runs where it helps.")
    print()
    rec("bc_final_loss", la[-1])
    rec("nobc_final_loss", lb[-1])
    return steps_taken


# =========================================================================
# 4. scale invariance
# =========================================================================
def scale_invariance():
    print("=" * 78)
    print("4. ADAM DOES NOT CARE HOW BIG YOUR LOSS IS")
    print("=" * 78)

    print(f"  {'loss multiplied by':>20}{'SGD final loss':>18}{'Adam final loss':>18}")
    ref = {}
    for scale in (0.01, 1.0, 100.0):
        _, ls = train(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9), scale=scale)
        _, la = train(lambda p: MyAdamW(p, lr=1e-2), scale=scale)
        ref[scale] = (ls[-1], la[-1])
        print(f"  {scale:>20g}{ls[-1]:>18.4f}{la[-1]:>18.4f}")
    print()
    print("  Multiplying the loss by 100 multiplies every gradient by 100, so SGD")
    print("  takes 100x larger steps; divide by 100 and it barely moves. Its final")
    print("  loss spans the whole column. Adam divides m by sqrt(v) and BOTH scale")
    print("  with the gradient, so the factor cancels exactly: the same number three")
    print("  times over.")
    print()
    print("  (Note the scaled-up SGD run does BETTER here, not worse. On this easy")
    print("  surface a 100x learning rate still converges. The point is not which")
    print("  wins -- it is that SGD's answer depends on a constant that has nothing")
    print("  to do with the model, and Adam's does not.)")
    print()
    print("  This is why Adam 'just works' on models where you have no idea what")
    print("  learning rate to pick, and why the same lr=3e-4 shows up in papers")
    print("  about wildly different architectures. The price is that Adam ignores")
    print("  information: a genuinely tiny gradient and a genuinely huge one both")
    print("  produce a step of about lr.")
    print()
    rec("sgd_scale1", ref[1.0][0])
    rec("sgd_scale100", ref[100.0][0])
    rec("adam_scale1", ref[1.0][1])
    rec("adam_scale100", ref[100.0][1])


# =========================================================================
# 5. what "decoupled" means, measured
# =========================================================================
def decoupled_meaning():
    print("=" * 78)
    print("5. WHAT 'DECOUPLED' MEANS, MEASURED")
    print("=" * 78)
    print("  Adam + L2 :  g <- g + wd*p         then the adaptive step divides by sqrt(v)")
    print("  AdamW     :  p <- p * (1 - lr*wd)  outside the adaptive step entirely")
    print()
    print("  So in Adam+L2 the decay term is divided by sqrt(v_hat) along with")
    print("  everything else. Each parameter's decay is scaled by its own gradient")
    print("  history. Measure that scaling directly:")
    print()

    for decoupled, label in ((False, "Adam + L2"), (True, "AdamW")):
        m = model()
        opt = MyAdamW(m.parameters(), lr=1e-2, weight_decay=0.1, decoupled=decoupled)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(X), Y).backward()
            opt.step()
        # the shrinkage each weight actually receives, per step, at this state
        p = m[0].weight
        st = opt.state[p]
        t = st["step"]
        denom = (st["v"].sqrt() / math.sqrt(1 - 0.999 ** t)) + 1e-8
        if decoupled:
            eff = torch.full_like(p, 1e-2 * 0.1)                # lr*wd, identical everywhere
        else:
            eff = (1e-2 / (1 - 0.9 ** t)) * 0.1 / denom          # lr*wd/denominator
        r = (eff.max() / eff.min()).item()
        print(f"  {label:<12} effective decay per step: min {eff.min():.3e}  "
              f"max {eff.max():.3e}  spread {r:,.0f}x")
        rec(f"decay_spread_{label.replace(' ', '_').replace('+', '')}", r)
    print()
    print("  In AdamW every weight is multiplied by the same (1 - lr*wd) -- spread")
    print("  exactly 1x, by construction. In Adam + L2 the spread across the weights")
    print("  of ONE layer is enormous: the weights with the largest gradient history")
    print("  are decayed least, and those are usually the important ones.")
    print()
    print("  That inversion is the whole argument of the AdamW paper. L2 penalty and")
    print("  weight decay are the same thing for plain SGD (differentiate 0.5*wd*p^2")
    print("  and you get wd*p), and they stop being the same thing the moment the")
    print("  optimizer rescales the gradient per parameter. 'Decoupled' means")
    print("  literally 'taken out of the part that gets rescaled'.")
    print()

    print("  And the practical consequence -- the decay strength you actually get:")
    print(f"  {'lr':>8}{'wd':>8}{'Adam+L2 ||w||':>16}{'AdamW ||w||':>14}")
    grid = {}
    for lr in (1e-3, 1e-2):
        for wd in (0.01, 0.1):
            norms = []
            for decoupled in (False, True):
                m = model()
                opt = MyAdamW(m.parameters(), lr=lr, weight_decay=wd, decoupled=decoupled)
                for _ in range(300):
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(m(X), Y).backward()
                    opt.step()
                norms.append(torch.cat([p.flatten() for p in m.parameters()]).norm().item())
            grid[(lr, wd)] = norms
            print(f"  {lr:>8.0e}{wd:>8.2f}{norms[0]:>16.3f}{norms[1]:>14.3f}")
    print()
    print("  The headline is the gap between the columns, not within them. At")
    print("  weight_decay=0.1 and lr=1e-2, Adam+L2 ends at ||w|| = 2.45 while AdamW")
    print("  ends at 13.15 -- FIVE TIMES more weight left, from the same number, because")
    print("  AdamW's decay is a flat (1 - lr*wd) = 0.999 per step while Adam+L2's got")
    print("  multiplied by 1/sqrt(v_hat), which is in the hundreds here.")
    print()
    print("  So `weight_decay=0.01` does not port between the two optimizers. Switch")
    print("  Adam -> AdamW and keep the number and you have quietly turned the")
    print("  regularization off; AdamW's published recipes use decays 10-100x larger")
    print("  for exactly this reason.")
    print()
    print("  And AdamW's is the one you can reason about: after N steps every weight")
    print("  has been multiplied by (1 - lr*wd)^N, full stop. Adam+L2's shrinkage is")
    print("  a different number for every weight in every layer, and it changes as")
    print("  training changes the gradients.")
    print()
    return grid


# =========================================================================
# 6. epsilon
# =========================================================================
def epsilon_section():
    print("=" * 78)
    print("6. EPSILON: WHERE IT SITS AND WHAT IT COSTS")
    print("=" * 78)

    print(f"  {'eps':>10}{'final loss':>14}")
    for eps in (1e-16, 1e-8, 1e-4, 1e-2, 1.0):
        _, ls = train(lambda p, e=eps: MyAdamW(p, lr=1e-2, eps=e))
        print(f"  {eps:>10.0e}{ls[-1]:>14.4f}")
        rec(f"eps_{eps:g}_final_loss", ls[-1])
    print()
    print("  eps is not there for accuracy. It is there so that a parameter whose")
    print("  gradient has been exactly zero for a while does not divide by zero -- a")
    print("  dead ReLU, a padding embedding, a frozen-in-practice weight.")
    print()
    print("  But it also puts a ceiling on the step: the update is at most")
    print("  lr*m_hat/eps. Raise eps far enough and Adam degrades toward SGD with")
    print("  momentum, which is exactly what the last row shows.")
    print()
    print("  Note where it sits: sqrt(v_hat) + eps, NOT sqrt(v_hat + eps). The two")
    print("  differ by a square root and people port the wrong one all the time;")
    print("  with eps=1e-8 the second form is a 1e-4 floor on the denominator, which")
    print("  is 10,000x stronger than intended.")
    print()


# =========================================================================
# 7. memory
# =========================================================================
def memory_section():
    print("=" * 78)
    print("7. WHAT TWO MOMENTS COST")
    print("=" * 78)

    m = model()
    n = sum(p.numel() for p in m.parameters())
    opt = MyAdamW(m.parameters(), lr=1e-2)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(m(X), Y).backward()
    opt.step()
    state_elems = sum(v["m"].numel() + v["v"].numel() for v in opt.state.values())
    print(f"  parameters        {n:>10}")
    print(f"  optimizer state   {state_elems:>10}   = {state_elems / n:.0f}x the parameters")
    print()
    print(f"  {'optimizer':<22}{'state / param':>15}{'training memory for a 7B model':>34}")
    for name, mult in (("SGD", 0), ("SGD + momentum", 1), ("Adam / AdamW", 2)):
        total = 7e9 * 4 * (2 + mult)     # weights + grads + state, fp32
        print(f"  {name:<22}{mult:>15}{total / 1e9:>30.0f} GB")
    print()
    print("  This one line of arithmetic explains most of the distributed-training")
    print("  chapter. A 7B model is 28 GB of fp32 weights and 112 GB before a single")
    print("  activation -- more than any single GPU has. It is why ZeRO and FSDP")
    print("  shard the OPTIMIZER STATE first (project 38): it is the biggest of the")
    print("  three pieces and the easiest to split, because Adam's update is")
    print("  elementwise and no weight's moments depend on any other weight's.")
    print()
    rec("state_multiplier", state_elems / n)


# =========================================================================
# figures
# =========================================================================
def figures(steps_taken, grid):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    t = np.arange(1, 61)
    ax.plot(t, steps_taken[True][:60], color=ps.SERIES[0], linewidth=2.0,
            label="with bias correction")
    ax.plot(t, steps_taken[False][:60], color=ps.SERIES[2], linewidth=2.0,
            linestyle="--", label="without")
    ax.axhline(1.0, color=ps.INK_MUTED, linewidth=0.9, linestyle=":")
    ax.text(40, 1.03, "the step you asked for", color=ps.INK_SECONDARY, fontsize=8)
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Step size on a constant gradient (lr = 1.0)", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("|actual update|", color=ps.INK_SECONDARY, fontsize=10)

    ax = axes[1]
    labels = [f"lr={lr:g}\nwd={wd:g}" for (lr, wd) in grid]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.19, [v[0] for v in grid.values()], width=0.36, color=ps.SERIES[2],
           label="Adam + L2")
    ax.bar(xs + 0.19, [v[1] for v in grid.values()], width=0.36, color=ps.SERIES[1],
           label="AdamW")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", color=ps.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Final weight norm after 300 steps", color=ps.INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_ylabel("||all parameters||", color=ps.INK_SECONDARY, fontsize=10)

    ps.save(fig, os.path.join(OUT, "bias_and_decay.png"))


def main():
    rule_and_match()
    steps_taken = bias_correction()
    scale_invariance()
    grid = decoupled_meaning()
    epsilon_section()
    memory_section()
    figures(steps_taken, grid)

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
