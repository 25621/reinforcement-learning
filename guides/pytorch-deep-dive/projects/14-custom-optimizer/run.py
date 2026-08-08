"""Project 14 — Custom optimizer.

Write SGD-with-momentum as an `optim.Optimizer` subclass, match PyTorch
bit-for-bit, and then use the fact that we own the code to answer the questions
the docs do not.

  1. the Optimizer contract: param_groups, state, step()
  2. bit-exact against torch.optim.SGD on five configurations
  3. what @torch.no_grad() on step() is actually for
  4. momentum: what the buffer does, and 1/(1-mu)
  5. two momentum conventions that agree until you add a scheduler
  6. param_groups: one optimizer, different rules per layer
  7. set_to_none=True is not just faster -- it changes the answer
  8. the optimizer's own state_dict, and what resuming without it costs

Runs in about 10 seconds on CPU. No downloads.
"""

import copy
import csv
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
class MySGD(Optimizer):
    """torch.optim.SGD, written out. Every line matches the reference."""

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            damp, wd, nesterov = group["dampening"], group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:            # this parameter got no gradient
                    continue
                d = p.grad
                if wd != 0:
                    d = d.add(p, alpha=wd)    # L2: fold the decay into the gradient
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = d.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(mu).add_(d, alpha=1 - damp)
                    d = d.add(buf, alpha=mu) if nesterov else buf
                p.add_(d, alpha=-lr)          # the actual update


class ClassicMomentum(Optimizer):
    """Heavy ball as the textbooks write it:  v = mu*v - lr*g ;  p = p + v."""

    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                v = self.state[p].setdefault("v", torch.zeros_like(p))
                v.mul_(group["momentum"]).add_(p.grad, alpha=-group["lr"])
                p.add_(v)


# =========================================================================
# a small, fast, completely deterministic training problem
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


def train(make_opt, steps=200, make_sched=None, seed=0, track=None):
    m = model(seed)
    opt = make_opt(m.parameters())
    sched = make_sched(opt) if make_sched else None
    losses, extra = [], []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(X), Y)
        loss.backward()
        if track is not None:
            extra.append(track(m, opt))
        opt.step()
        if sched is not None:
            sched.step()
        losses.append(loss.item())
    return m, np.array(losses), extra


def max_param_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


# =========================================================================
# 1 + 2. the contract, and bit-exactness
# =========================================================================
def contract_and_match():
    print("=" * 78)
    print("1. THE CONTRACT")
    print("=" * 78)

    m = model()
    opt = MySGD(m.parameters(), lr=0.1, momentum=0.9)
    print(f"  param_groups           : {len(opt.param_groups)} group(s)")
    print(f"  group keys             : {sorted(opt.param_groups[0])}")
    print(f"  params in group 0      : {len(opt.param_groups[0]['params'])} tensors")
    print(f"  state before any step  : {len(opt.state)} entries")
    F.cross_entropy(m(X), Y).backward()
    opt.step()
    p0 = opt.param_groups[0]["params"][0]
    print(f"  state after one step   : {len(opt.state)} entries, "
          f"each holding {list(opt.state[p0])}")
    print()
    print("  An Optimizer is two containers and one method:")
    print("    param_groups  a list of dicts: {'params': [...], 'lr': ..., ...}")
    print("    state         a dict keyed BY PARAMETER TENSOR, holding whatever")
    print("                  that parameter's update rule needs to remember")
    print("    step()        read .grad, update state, write the parameter in place")
    print()
    print("  Keying state by the tensor object (not by name, not by index) is what")
    print("  lets an optimizer accept any list of tensors from anywhere -- it never")
    print("  needs to know there was a model.")
    print()

    print("=" * 78)
    print("2. BIT-EXACT AGAINST torch.optim.SGD")
    print("=" * 78)
    configs = [
        dict(lr=0.1),
        dict(lr=0.1, momentum=0.9),
        dict(lr=0.1, momentum=0.9, weight_decay=1e-2),
        dict(lr=0.1, momentum=0.9, nesterov=True),
        dict(lr=0.1, momentum=0.9, dampening=0.3),
    ]
    worst = 0.0
    for cfg in configs:
        a, _, _ = train(lambda p, c=cfg: MySGD(p, **c))
        b, _, _ = train(lambda p, c=cfg: torch.optim.SGD(p, **c))
        d = max_param_diff(a, b)
        worst = max(worst, d)
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {label:<46} max |diff| after 200 steps: {d:.3e}")
    print()
    print("  Zero. Not 'close' -- the same floating-point operations in the same")
    print("  order produce the same bits.")
    print()
    print("  Three details are worth naming, because getting any of them wrong")
    print("  gives you an optimizer that trains and is not SGD:")
    print("   - weight_decay is folded into the GRADIENT (d = g + wd*p) before")
    print("     momentum sees it. Project 15 is about what happens if you do not.")
    print("   - the FIRST step sets buf = d, it does not do buf = mu*0 + d. Same")
    print("     number here, but not once dampening != 0.")
    print("   - dampening scales the incoming gradient (1-damp), not the buffer.")
    print()
    rec("bitexact_worst_diff", worst)


# =========================================================================
# 3. what @torch.no_grad() is for
# =========================================================================
def why_no_grad():
    print("=" * 78)
    print("3. WHAT @torch.no_grad() ON step() IS ACTUALLY FOR")
    print("=" * 78)

    p = nn.Parameter(torch.randn(3))
    p.grad = torch.randn(3)
    try:
        p.add_(p.grad, alpha=-0.1)
        msg = "(no error)"
    except RuntimeError as e:
        msg = str(e)
    print(f"  p.add_(p.grad, alpha=-0.1)  outside no_grad ->")
    print(f"    RuntimeError: {msg}")
    print()
    with torch.no_grad():
        p.add_(p.grad, alpha=-0.1)
    print("  inside torch.no_grad() -> fine")
    print()
    print("  A parameter is a LEAF tensor with requires_grad=True. Autograd refuses")
    print("  in-place writes to a leaf, because the leaf's value is what the next")
    print("  backward pass will differentiate around -- changing it under autograd's")
    print("  feet would make the recorded graph describe a model that no longer")
    print("  exists.")
    print()
    print("  `@torch.no_grad()` says: nothing in here is part of any graph. It is")
    print("  not an optimization you can skip. Leave it off and step() raises on the")
    print("  first parameter it touches.")
    print()
    print("  This is also why the same decorator appears on schedulers, EMA updates,")
    print("  and any weight-averaging code you write.")
    rec("no_grad_error", msg.split(".")[0])
    print()


# =========================================================================
# 4. momentum
# =========================================================================
def momentum_meaning():
    print("=" * 78)
    print("4. WHAT THE MOMENTUM BUFFER DOES")
    print("=" * 78)

    print("  A constant gradient g fed into  buf = mu*buf + g  converges to")
    print("  buf = g/(1-mu). The step size is amplified by 1/(1-mu):")
    print()
    print(f"  {'momentum':>9}{'1/(1-mu)':>11}{'step 10':>10}{'step 100':>10}"
          f"{'step 2000':>11}{'final loss':>13}")
    curves = {}
    for mu in (0.0, 0.5, 0.9, 0.99):
        m, losses, _ = train(lambda p, mu=mu: MySGD(p, lr=0.05, momentum=mu), steps=200)
        # feed a constant gradient of 1.0 and watch the single-step update grow
        w = nn.Parameter(torch.zeros(1))
        opt = MySGD([w], lr=1.0, momentum=mu)
        snap = {}
        for i in range(1, 2001):
            w.grad = torch.ones(1)
            before = w.item()
            opt.step()
            if i in (10, 100, 2000):
                snap[i] = abs(w.item() - before)
        theory = 1 / (1 - mu)
        print(f"  {mu:>9.2f}{theory:>11.1f}{snap[10]:>10.2f}{snap[100]:>10.2f}"
              f"{snap[2000]:>11.2f}{losses[-1]:>13.4f}")
        curves[mu] = losses
        rec(f"momentum_{mu}_final_loss", losses[-1])
        rec(f"momentum_{mu}_amplification", snap[2000])
        rec(f"momentum_{mu}_amplification_step10", snap[10])
    print()
    print("  The three middle columns are one step's update, with the gradient held")
    print("  at exactly 1.0 and lr = 1.0. They converge on 1/(1-mu) -- and the time")
    print("  they take to get there is itself about 1/(1-mu) steps. At mu=0.99 the")
    print("  buffer is still only two thirds full after 100 steps.")
    print()
    print("  So momentum is not only 'smoothing'. At mu=0.9 the effective step is")
    print("  ten times the learning rate once the gradient stops changing direction,")
    print("  and at 0.99 it is a hundred times. That is why raising momentum without")
    print("  lowering the learning rate blows a run up -- they are the same knob.")
    print()
    print("  The name is honest: a heavy object keeps moving in the direction it was")
    print("  already going, so consistent gradients accumulate into a big step, while")
    print("  gradients that flip sign every batch cancel out. It averages the noise")
    print("  away and keeps the signal.")
    print()
    return curves


# =========================================================================
# 5. two conventions
# =========================================================================
def two_conventions():
    print("=" * 78)
    print("5. TWO MOMENTUM CONVENTIONS THAT AGREE UNTIL YOU ADD A SCHEDULER")
    print("=" * 78)
    print("  PyTorch :  buf = mu*buf + g          then  p -= lr*buf")
    print("  Textbook:  v   = mu*v  - lr*g        then  p += v")
    print()
    print("  Substituting v = -lr*buf turns one into the other -- as long as lr is a")
    print("  constant. It is not, in any modern recipe.")
    print()

    out = {}
    for label, make_sched in (
            ("constant lr", None),
            ("StepLR, x0.1 at step 100", lambda o: torch.optim.lr_scheduler.StepLR(o, 100, 0.1)),
            ("CosineAnnealingLR", lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200))):
        a, la, _ = train(lambda p: MySGD(p, lr=0.2, momentum=0.9), make_sched=make_sched)
        b, lb, _ = train(lambda p: ClassicMomentum(p, lr=0.2, momentum=0.9), make_sched=make_sched)
        d = max_param_diff(a, b)
        print(f"  {label:<26} max |param diff| {d:.3e}   final loss "
              f"{la[-1]:.4f} vs {lb[-1]:.4f}")
        out[label] = (la, lb, d)
        rec(f"convention_diff_{label.split(',')[0].replace(' ', '_')}", d)
    print()
    ratio = out["StepLR, x0.1 at step 100"][2] / out["constant lr"][2]
    print("  Constant learning rate: the two agree to float32 rounding.")
    print(f"  Add a step schedule and the gap is {ratio:,.0f}x bigger -- a real")
    print("  difference in what got trained, not a rounding artefact.")
    rec("convention_ratio_sched_vs_constant", ratio)
    print()
    print("  The reason is where lr sits. In the textbook form the learning rate is")
    print("  baked into the buffer at the moment each gradient arrives, so an old")
    print("  gradient keeps the old (large) learning rate forever. In PyTorch's form")
    print("  the buffer holds raw gradients and the CURRENT lr multiplies the whole")
    print("  history at once, so dropping lr by 10x instantly shrinks the momentum")
    print("  that had already built up.")
    print()
    print("  PyTorch's is the one you want: 'lower the learning rate' should take")
    print("  effect now, not fade in over the next 1/(1-mu) steps. But if you port a")
    print("  paper's pseudocode literally and it uses a schedule, you have quietly")
    print("  implemented a different optimizer.")
    print()
    return out


# =========================================================================
# 6. param_groups
# =========================================================================
def param_groups_demo():
    print("=" * 78)
    print("6. ONE OPTIMIZER, DIFFERENT RULES PER LAYER")
    print("=" * 78)

    def build(split):
        m = model()
        decay, no_decay = [], []
        for name, p in m.named_parameters():
            (no_decay if (split and name.endswith("bias")) else decay).append(p)
        groups = [dict(params=decay, weight_decay=1e-1)]
        if no_decay:
            groups.append(dict(params=no_decay, weight_decay=0.0))
        return m, MySGD(groups, lr=0.1, momentum=0.9)

    for split, label in ((False, "decay everything"), (True, "biases exempt")):
        m, opt = build(split)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(X), Y).backward()
            opt.step()
        wn = torch.cat([p.flatten() for n, p in m.named_parameters() if not n.endswith("bias")]).norm()
        bn = torch.cat([p.flatten() for n, p in m.named_parameters() if n.endswith("bias")]).norm()
        loss = F.cross_entropy(m(X), Y).item()
        print(f"  {label:<18} ||weights|| {wn:6.3f}   ||biases|| {bn:6.3f}   loss {loss:.4f}")
        rec(f"group_{label.replace(' ', '_')}_bias_norm", bn.item())
    print()
    print("  `params` is the only required key in a group; anything else you put")
    print("  there overrides the optimizer's default for those tensors. Two groups,")
    print("  two weight decays, one optimizer, one step().")
    print()
    print("  Exempting biases and normalization parameters from weight decay is the")
    print("  standard recipe in every transformer codebase, and the reason is that")
    print("  weight decay is a capacity control on the FUNCTION the weights compute.")
    print("  A bias just shifts the output; shrinking it toward zero does not")
    print("  simplify anything, it only stops the layer from centring its output")
    print("  where it needs to. The same argument covers LayerNorm's gain: pulling")
    print("  it toward 0 fights the very normalization it was added to provide.")
    print()


# =========================================================================
# 7. set_to_none
# =========================================================================
class TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 1)
        self.b = nn.Linear(4, 1)

    def forward(self, x, which):
        return (self.a(x) if which == 0 else self.b(x)).sum()


def set_to_none_demo():
    print("=" * 78)
    print("7. set_to_none=True IS NOT JUST FASTER")
    print("=" * 78)

    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(7))
    finals = {}
    for stn in (True, False):
        torch.manual_seed(0)
        m = TwoBranch()
        opt = MySGD(m.parameters(), lr=0.1, momentum=0.9)
        for step in range(6):
            opt.zero_grad(set_to_none=stn)
            m(x, 0 if step == 0 else 1).backward()   # branch a used only once
            opt.step()
        finals[stn] = m.a.weight.detach().clone()
        print(f"  set_to_none={str(stn):<5}  branch-a weight after 6 steps: "
              f"{finals[stn].numpy().round(4)}")
    d = (finals[True] - finals[False]).abs().max().item()
    print(f"  max |difference| : {d:.4f}")
    print()
    print("  Branch a gets a gradient on step 0 and never again. Then:")
    print("   - set_to_none=True  -> p.grad is None -> step() SKIPS it. Its momentum")
    print("     buffer is frozen and the weight stops moving.")
    print("   - set_to_none=False -> p.grad is a tensor of zeros -> step() runs.")
    print("     buf = mu*buf + 0 keeps decaying and keeps pushing the weight.")
    print()
    print("  Both are defensible; they are not the same training run. This bites")
    print("  conditional architectures -- mixtures of experts, multi-task heads,")
    print("  anything with a branch that only some batches reach.")
    print()
    print("  set_to_none=True is the modern default because it is faster (no kernel")
    print("  to zero the tensor) and saves the memory of every .grad between steps.")
    print("  The behaviour change came along for the ride, and it is the reason the")
    print("  flag existed as an option for years before it became the default.")
    print()
    rec("set_to_none_max_diff", d)


# =========================================================================
# 8. the optimizer's own state_dict
# =========================================================================
def optimizer_state_dict():
    print("=" * 78)
    print("8. THE OPTIMIZER HAS A state_dict TOO")
    print("=" * 78)

    m = model()
    opt = MySGD(m.parameters(), lr=0.1, momentum=0.9)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(X), Y).backward()
        opt.step()
    sd_model = copy.deepcopy(m.state_dict())
    sd_opt = copy.deepcopy(opt.state_dict())
    ref_loss = F.cross_entropy(m(X), Y).item()

    print(f"  after 100 steps, loss {ref_loss:.4f}")
    print(f"  optimizer state_dict keys: {list(sd_opt)}")
    print(f"  'state' holds {len(sd_opt['state'])} entries, keyed by parameter INDEX "
          f"(not name)")
    print(f"  momentum buffer bytes: "
          f"{sum(v['momentum_buffer'].numel() * 4 for v in sd_opt['state'].values())} "
          f"= one extra copy of the model")
    print()

    curves = {}
    for label, load_opt in (("resume with optimizer state", True),
                            ("resume without it", False)):
        m2 = model()
        m2.load_state_dict(sd_model)
        opt2 = MySGD(m2.parameters(), lr=0.1, momentum=0.9)
        if load_opt:
            opt2.load_state_dict(sd_opt)
        ls, first_update = [], None
        for i in range(40):
            opt2.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m2(X), Y)
            loss.backward()
            snapshot = [p.detach().clone() for p in m2.parameters()]
            opt2.step()
            if i == 0:
                first_update = torch.cat([(p - q).flatten()
                                          for p, q in zip(m2.parameters(), snapshot)]).norm().item()
            ls.append(loss.item())
        curves[label] = np.array(ls)
        print(f"  {label:<30} first update ||dp|| {first_update:.5f}   "
              f"step 40 loss {ls[-1]:.4f}")
        rec(f"resume_{load_opt}_final", ls[-1])
        rec(f"resume_{load_opt}_first_update", first_update)
    print()
    ratio = (FINDINGS["resume_True_first_update"] / FINDINGS["resume_False_first_update"])
    print(f"  the very first step after resuming is {ratio:.1f}x larger when the")
    print("  momentum buffers came back with the weights.")
    gap = curves["resume without it"][:10].mean() - curves["resume with optimizer state"][:10].mean()
    print(f"  mean loss over the first 10 steps after resuming, gap: {gap:+.4f}")
    print()
    print("  Dropping the optimizer state does not break anything and does not raise.")
    print("  The model is identical; only the momentum buffers are gone, so the run")
    print("  restarts from a standstill and has to build them up again -- 1/(1-mu)")
    print("  ~ 10 steps of a smaller effective step size, at exactly the point in")
    print("  training where you were least expecting a change.")
    print()
    print("  Note the state is keyed by parameter INDEX in the saved dict, while it")
    print("  is keyed by the tensor OBJECT in memory. That is why load_state_dict")
    print("  requires the same parameters in the same order -- and why reordering a")
    print("  model's __init__ silently mismatches optimizer state that loads fine.")
    print()
    print("  A complete checkpoint is: model.state_dict(), optimizer.state_dict(),")
    print("  scheduler.state_dict(), the step number, and the RNG state (project 17).")
    print()
    rec("resume_gap", gap)
    return curves


# =========================================================================
# figures
# =========================================================================
def figures(mom_curves, conv, resume):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    for i, (mu, ls) in enumerate(mom_curves.items()):
        ax.plot(ls, color=ps.SERIES[i], linewidth=1.8, label=f"momentum {mu}")
    ax.set_yscale("log")
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Momentum at a fixed learning rate (0.05)", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("training loss (log)", color=ps.INK_SECONDARY, fontsize=10)

    ax = axes[1]
    la, lb, d = conv["StepLR, x0.1 at step 100"]
    ax.plot(la, color=ps.SERIES[0], linewidth=1.8, label="PyTorch form")
    ax.plot(lb, color=ps.SERIES[3], linewidth=1.8, linestyle="--", label="textbook form")
    ax.axvline(100, color=ps.INK_MUTED, linewidth=0.9, linestyle=":")
    ax.text(103, la.max() * 0.5, "lr x 0.1", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, color=ps.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Same optimizer, two conventions, one scheduler", color=ps.INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("training loss (log)", color=ps.INK_SECONDARY, fontsize=10)

    ps.save(fig, os.path.join(OUT, "momentum_and_conventions.png"))

    fig, ax = ps.new_axes(7.2, 4.0)
    for i, (label, ls) in enumerate(resume.items()):
        ax.plot(ls, color=ps.SERIES[i], linewidth=1.9, marker="o", markersize=3,
                label=label)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Resuming a run: what the momentum buffers were worth",
              "step after resume", "training loss",
              os.path.join(OUT, "resume.png"))


def main():
    contract_and_match()
    why_no_grad()
    mom = momentum_meaning()
    conv = two_conventions()
    param_groups_demo()
    set_to_none_demo()
    resume = optimizer_state_dict()
    figures(mom, conv, resume)

    path = os.path.join(OUT, "findings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
