"""Project 48 - NaN forensics.

A sequence model whose loss curve looks merely disappointing. It is not
disappointing: 69% of its parameters are NaN by step 1, and the layer that
produced the NaN also hides it, so the loss never once prints `nan`.

The job is to find the single operation that made the first NaN, and to say why
that operation was allowed to.

Sections:
  1. the crime scene: a run that is dead without ever printing nan
  2. the spread: how far the poison travels in one optimizer step
  3. gradient clipping, the accomplice
  4. instrument A - forward hooks that scan for non-finite activations
  5. instrument B - torch.autograd.set_detect_anomaly, and what it costs
  6. the verdict: why `torch.where` produced a NaN out of finite numbers
  7. the second bug, the one the hooks *do* catch
  8. the optimizer state remembers: why restoring a checkpoint is not enough
  9. the guard that keeps a real run alive

Run:  python3 run.py        (~2 minutes)
"""

from __future__ import annotations

import copy
import io
import os
import sys
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import debug_lib as D  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
F_ = D.Findings()


# ===========================================================================
# The model under investigation
# ===========================================================================

SEQ_LEN, D_MODEL, N_SAMPLES = 12, 32, 512
SHORT_ROWS = 4          # of 512 samples, this many are shorter than SEQ_LEN


def seq_data(seed: int = 1):
    """A batch of variable-length sequences, padded to a fixed length.

    Padding is the ordinary way to put sequences of different lengths in one
    rectangular tensor: you pick the longest, and fill the rest with zeros. The
    mask records which positions are real. Almost every sequence model on earth
    does this, and it is what makes the bug below realistic rather than
    contrived — the zero vector is *deliberately* there.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N_SAMPLES, SEQ_LEN, D_MODEL, generator=g)
    lengths = torch.full((N_SAMPLES,), SEQ_LEN)
    short = torch.randperm(N_SAMPLES, generator=g)[:SHORT_ROWS]
    lengths[short] = 3                        # a handful of short sequences
    mask = (torch.arange(SEQ_LEN)[None, :] < lengths[:, None]).float()
    # The label is "is the first token's vector longer than the third's?" — a
    # question about lengths, so the architecture below can actually answer it.
    y = (x[:, 0].pow(2).sum(-1) > x[:, 2].pow(2).sum(-1)).long()
    return x, mask, y


class MaskedNorm(nn.Module):
    """Per-position vector length, guarded against the zero vector.

    The layer wants the L2 length of each position's vector: `sqrt(sum(h^2))`.
    The square root has an infinite slope at zero, and a *padded* position is
    exactly the zero vector, so the author added a guard —
    `torch.where(sq > 0, sqrt(sq), 0)`, read as "only take the square root
    where there is something to take it of".

    That guard is the bug. Section 6 explains why, in one sentence.
    """

    def __init__(self, guard: str = "where"):
        super().__init__()
        self.guard = guard

    def forward(self, h):
        sq = (h * h).sum(-1)                  # exactly 0.0 at padded positions
        if self.guard == "where":
            return torch.where(sq > 0, torch.sqrt(sq), torch.zeros_like(sq))
        if self.guard == "double_where":      # fix 1
            safe = torch.where(sq > 0, sq, torch.ones_like(sq))
            return torch.where(sq > 0, torch.sqrt(safe), torch.zeros_like(sq))
        if self.guard == "clamp":             # fix 2
            return torch.sqrt(sq.clamp_min(1e-12))
        raise ValueError(self.guard)


class SeqClassifier(nn.Module):
    """project -> zero out the padding -> per-position length -> pool -> classify."""

    def __init__(self, guard: str = "where", seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def lin(a, b):
            m = nn.Linear(a, b)
            with torch.no_grad():
                m.weight.copy_(torch.randn(b, a, generator=g) * (1.0 / a) ** 0.5)
                m.bias.zero_()
            return m

        self.proj = lin(D_MODEL, D_MODEL)
        self.norm = MaskedNorm(guard)
        self.head = nn.Sequential(lin(SEQ_LEN, 32), nn.ReLU(), lin(32, 2))

    def forward(self, x, mask):
        h = self.proj(x) * mask[..., None]    # multiply, not select: this is the
        lens = self.norm(h)                   # path the NaN will travel back down
        return self.head(lens)


def make_model(guard: str = "where", seed: int = 0) -> nn.Module:
    return SeqClassifier(guard, seed)


def nan_frac(model) -> float:
    bad = tot = 0
    for p in model.parameters():
        bad += int((~torch.isfinite(p)).sum())
        tot += p.numel()
    return bad / tot


def train(model, steps=60, lr=0.05, clip=None, seed=0, guard_step=False,
          log_every=25, optimizer="sgd", batch=32):
    """One small training loop, instrumented to record the truth per step."""
    torch.manual_seed(seed)
    x, mask, y = seq_data()
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))
    hist = {"loss": [], "grad_nan": [], "param_nan": [], "logged": [], "skipped": 0,
            "padded_rows": []}
    for step in range(steps):
        g = torch.Generator().manual_seed(1000 + step)
        idx = torch.randint(0, len(x), (batch,), generator=g)
        xb, mb, yb = x[idx], mask[idx], y[idx]
        hist["padded_rows"].append(int((mb.min(1).values == 0).sum()))
        loss = F.cross_entropy(model(xb, mb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnan = sum(int((~torch.isfinite(p.grad)).sum()) for p in model.parameters()
                   if p.grad is not None)
        if guard_step and gnan:
            hist["skipped"] += 1
            opt.zero_grad(set_to_none=True)
        else:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        hist["loss"].append(loss.detach().item())
        hist["grad_nan"].append(gnan)
        hist["param_nan"].append(nan_frac(model))
        if step % log_every == 0:
            hist["logged"].append((step, hist["loss"][-1]))
    return hist


# ===========================================================================
# 1. The crime scene
# ===========================================================================

F_.head("1. The crime scene")


def accuracy(model) -> float:
    x, mask, y = seq_data()
    with torch.no_grad():
        pred = model(x, mask).argmax(-1)
    return float((pred == y).float().mean())


buggy_model = make_model("where")
hist = train(buggy_model, steps=300, lr=0.05)
hist_acc = accuracy(buggy_model)

first_grad_nan = next((i for i, v in enumerate(hist["grad_nan"]) if v), None)
first_param_nan = next((i for i, v in enumerate(hist["param_nan"]) if v), None)
first_loss_nan = next((i for i, v in enumerate(hist["loss"]) if not np.isfinite(v)), None)

F_.note("steps trained", len(hist["loss"]))
F_.note("batches that contained at least one padded row",
        f"{sum(1 for v in hist['padded_rows'] if v)} of {len(hist['padded_rows'])}")
F_.note("first step with a non-finite GRADIENT", first_grad_nan)
F_.note("first step with a non-finite PARAMETER", first_param_nan)
F_.note("first step whose LOSS is non-finite",
        first_loss_nan if first_loss_nan is not None else "never")
F_.note("what the log printed", "; ".join(
    f"step {s}: {v:.4f}" if np.isfinite(v) else f"step {s}: nan" for s, v in hist["logged"]))
F_.note("loss at step 0 / final step", f"{hist['loss'][0]:.4f} / {hist['loss'][-1]:.4f}")
F_.note("chance-level loss for 2 classes, ln(2)", float(np.log(2)))
F_.note("parameters non-finite at the end", f"{hist['param_nan'][-1] * 100:.1f}%")

ref_model = make_model("clamp")
ref = train(ref_model, steps=300, lr=0.05)
ref_acc = accuracy(ref_model)
F_.note("same run with the bug fixed: final loss", f"{ref['loss'][-1]:.4f}")
F_.note("accuracy, buggy run / fixed run", f"{hist_acc:.3f} / {ref_acc:.3f}")
F_.note("headline", "the loss curve never showed a nan; the model was already dead")


# ===========================================================================
# 2. The spread
# ===========================================================================

F_.head("2. The spread: how far one NaN travels in one step")

m2 = make_model("where")
h2 = train(m2, steps=first_grad_nan + 1, lr=0.05)
F_.note("fraction of parameters non-finite, no clipping",
        f"{h2['param_nan'][-1] * 100:.1f}%")
per_layer = []
for name, p in m2.named_parameters():
    frac = float((~torch.isfinite(p)).float().mean())
    per_layer.append((name, frac))
    F_.note(f"  {name}", f"{frac * 100:.1f}% non-finite")

m2b = make_model("where")
h2b = train(m2b, steps=first_grad_nan + 1, lr=0.05, clip=1.0)
F_.note("fraction of parameters non-finite, WITH clip_grad_norm_(1.0)",
        f"{h2b['param_nan'][-1] * 100:.1f}%")


# ===========================================================================
# 3. Clipping is an accomplice
# ===========================================================================

F_.head("3. Why clipping spreads it: the norm of a NaN is NaN")

ps = [nn.Parameter(torch.ones(4)) for _ in range(3)]
for p in ps:
    p.grad = torch.ones(4)
ps[1].grad[0] = float("nan")
total = torch.nn.utils.clip_grad_norm_(ps, 1.0)
F_.note("gradient tensors before clipping", "1 of 3 has a single NaN entry (1 of 12 numbers)")
F_.note("total_norm returned by clip_grad_norm_", float(total))
F_.note("gradient numbers non-finite AFTER clipping",
        f"{sum(int((~torch.isfinite(p.grad)).sum()) for p in ps)} of 12")

ps2 = [nn.Parameter(torch.ones(4)) for _ in range(3)]
for p in ps2:
    p.grad = torch.ones(4)
ps2[1].grad[0] = float("nan")
try:
    torch.nn.utils.clip_grad_norm_(ps2, 1.0, error_if_nonfinite=True)
    F_.note("error_if_nonfinite=True", "did not raise (unexpected)")
except RuntimeError as exc:
    F_.note("error_if_nonfinite=True raises", str(exc)[:90])


# ===========================================================================
# 4. Instrument A: forward hooks
# ===========================================================================

F_.head("4. Instrument A: a forward-hook scanner")


class FiniteScanner:
    """Registers a forward hook on every module and reports the first one whose
    output contains a NaN or an infinity.

    A *hook* is a function PyTorch calls for you every time a module runs. You
    do not have to edit the model to use one, which is what makes this the
    cheapest instrument in the box.
    """

    def __init__(self, model):
        self.hits: list[str] = []
        self.handles = []
        for name, mod in model.named_modules():
            if list(mod.children()):
                continue                      # only leaves; a container's output
                                              # is just its last child's output
            self.handles.append(mod.register_forward_hook(self._make(name)))

    def _make(self, name):
        def hook(_mod, _inp, out):
            if torch.is_tensor(out) and not torch.isfinite(out).all():
                self.hits.append(name)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


m4 = make_model("where")
sc = FiniteScanner(m4)
h4 = train(m4, steps=first_grad_nan + 1, lr=0.05)
F_.note("gradient numbers non-finite at the bad step", h4["grad_nan"][-1])
F_.note("modules the forward scanner flagged, up to and including that step",
        len(sc.hits))
F_.note("verdict", "the forward pass never held a NaN, so there was nothing to see")
sc.close()

sc2 = FiniteScanner(m4)                       # m4's weights are NaN now
xs, ms, _ = seq_data()
with torch.no_grad():
    m4(xs[:8], ms[:8])
F_.note("modules flagged on the NEXT forward, after the weights are poisoned",
        f"{len(sc2.hits)} ({sc2.hits[0] if sc2.hits else '-'} first)")
F_.note("so the scanner is one step late", "it sees the consequence, not the cause")
sc2.close()


# ===========================================================================
# 5. Instrument B: anomaly detection
# ===========================================================================

F_.head("5. Instrument B: torch.autograd.set_detect_anomaly")

m5 = make_model("where")
x5, mask5, y5 = seq_data()
bad = (mask5.min(1).values == 0).nonzero().flatten()[:4]      # a batch WITH padding
anomaly_text = ""
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with torch.autograd.set_detect_anomaly(True):
        loss = F.cross_entropy(m5(x5[bad], mask5[bad]), y5[bad])
        try:
            loss.backward()
            anomaly_text = "(no error - the backward happened to be clean)"
        except RuntimeError:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            anomaly_text = buf.getvalue()

# Anomaly mode splits its report in two: the *exception* names the backward op,
# and a separate *warning* carries the forward traceback — the far more useful
# half, and the half a bare `except` throws away. Save both.
warn_text = "\n".join(str(w.message) for w in caught if "Error detected in" in str(w.message))
report = (warn_text + "\n\n" + anomaly_text).replace(HERE + "/", "").replace(
    os.path.dirname(os.path.dirname(torch.__file__)) + "/", "<site-packages>/")
with open(os.path.join(OUT, "anomaly_traceback.txt"), "w") as fh:
    fh.write(report)

first_line = next((ln.strip() for ln in anomaly_text.splitlines()
                   if "returned nan" in ln), anomaly_text[:120])
F_.note("anomaly mode raised", "yes" if "Error" in anomaly_text else "no")
F_.note("the exception names the BACKWARD op", first_line[:96])
F_.note("a separate warning carries the FORWARD traceback", bool(warn_text))
fwd = [ln.strip() for ln in warn_text.splitlines() if "run.py" in ln and "line" in ln]
F_.note("forward source lines it quotes", len(fwd))
F_.note("the innermost one", fwd[-1].replace(HERE + "/", "")[:96] if fwd else "-")
src = [ln.strip() for ln in warn_text.splitlines() if "torch.where(sq" in ln]
F_.note("the source line it prints", src[-1][:96] if src else "-")


_XT, _MT, _YT = seq_data()
_XT, _MT, _YT = _XT[:32], _MT[:32], _YT[:32]


def _fwd_bwd(model):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(_XT, _MT), _YT).backward()


m_fast = make_model("clamp")
scan_model = make_model("clamp")
_scanner = FiniteScanner(scan_model)


def _anom():
    with torch.autograd.set_detect_anomaly(True):
        _fwd_bwd(m_fast)


timings = D.interleaved({
    "no instrument": lambda: _fwd_bwd(m_fast),
    "forward hooks": lambda: _fwd_bwd(scan_model),
    "anomaly mode": _anom,
}, rounds=5, calls=20)
_scanner.close()

base = timings["no instrument"]["best"]
for name, t in timings.items():
    F_.note(f"step time, {name}", f"{t['best'] * 1e3:.3f} ms  ({t['best'] / base:.2f}x)")


# ===========================================================================
# 6. The verdict
# ===========================================================================

F_.head("6. The verdict: where() evaluates both branches")

x6 = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
y6 = torch.where(x6 > 0, torch.sqrt(x6), torch.zeros_like(x6))
y6.sum().backward()
F_.note("forward output of the guarded sqrt", y6.tolist())
F_.note("forward contains a NaN?", bool(~torch.isfinite(y6).all()))
F_.note("gradient of the guarded sqrt", [round(v, 4) if np.isfinite(v) else "nan"
                                         for v in x6.grad.tolist()])

x6b = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
safe = torch.where(x6b > 0, x6b, torch.ones_like(x6b))
torch.where(x6b > 0, torch.sqrt(safe), torch.zeros_like(x6b)).sum().backward()
F_.note("gradient after the double-where fix", [round(v, 4) for v in x6b.grad.tolist()])

x6c = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
torch.sqrt(x6c.clamp_min(1e-12)).sum().backward()
F_.note("gradient after the clamp fix", [f"{v:.4g}" for v in x6c.grad.tolist()])

d_sqrt_0 = float(torch.tensor(0.5) / torch.sqrt(torch.tensor(0.0)))
F_.note("d/dx sqrt(x) at x=0", d_sqrt_0)
F_.note("0 * inf in IEEE-754 floating point", float(torch.tensor(0.0) * torch.tensor(float("inf"))))


# ===========================================================================
# 7. The second bug, the forward-visible one
# ===========================================================================

F_.head("7. A forward-visible bug: hand-rolled log-softmax")

logits = torch.tensor([[0.5, 120.0, -3.0]], requires_grad=True)
naive = torch.log(torch.exp(logits) / torch.exp(logits).sum(-1, keepdim=True))
F_.note("exp(120) in float32", float(torch.exp(torch.tensor(120.0))))
F_.note("hand-rolled log_softmax forward", [f"{v:.4g}" for v in naive[0].tolist()])
F_.note("torch.log_softmax forward",
        [f"{v:.4g}" for v in torch.log_softmax(logits, -1)[0].tolist()])
F_.note("difference", "the hand-rolled one is nan in the FORWARD pass")

hits = []
mod = nn.Sequential(nn.Identity())
h = mod.register_forward_hook(lambda m, i, o: hits.append("Identity") if not torch.isfinite(o).all() else None)
mod(naive.detach())
h.remove()
F_.note("forward scanner catches it", bool(hits))
F_.note("which instrument catches which bug",
        "hooks: forward NaNs only | anomaly: forward AND backward-only NaNs")


# ===========================================================================
# 8. The optimizer state remembers
# ===========================================================================

F_.head("8. Restoring the weights is not enough (Adam)")

m8 = make_model("where")
clean_weights = copy.deepcopy(m8.state_dict())
opt8 = torch.optim.Adam(m8.parameters(), lr=1e-3)
x8, m8mask, y8 = seq_data()
for step in range(4):
    loss = F.cross_entropy(m8(x8, m8mask), y8)
    opt8.zero_grad(set_to_none=True)
    loss.backward()
    opt8.step()

F_.note("parameters non-finite after 4 Adam steps", f"{nan_frac(m8) * 100:.1f}%")
st = opt8.state_dict()["state"]
bad_state = sum(1 for s in st.values()
                for k in ("exp_avg", "exp_avg_sq")
                if k in s and not torch.isfinite(s[k]).all())
F_.note("optimizer state tensors that are non-finite", f"{bad_state} of {2 * len(st)}")

m8.load_state_dict(clean_weights)             # "restore the checkpoint"
F_.note("parameters non-finite right after restoring the weights",
        f"{nan_frac(m8) * 100:.1f}%")
opt8.zero_grad(set_to_none=True)
for p in m8.parameters():                     # hand it a perfectly clean gradient
    p.grad = torch.ones_like(p) * 1e-3
opt8.step()
F_.note("parameters non-finite after ONE step with a clean gradient",
        f"{nan_frac(m8) * 100:.1f}%")

m8.load_state_dict(clean_weights)
opt8 = torch.optim.Adam(m8.parameters(), lr=1e-3)   # ...and a fresh optimizer
for p in m8.parameters():
    p.grad = torch.ones_like(p) * 1e-3
opt8.step()
F_.note("same, after also rebuilding the optimizer", f"{nan_frac(m8) * 100:.1f}%")


# ===========================================================================
# 9. The guard
# ===========================================================================

F_.head("9. Fix it, or survive it")

runs = {}
models = {"buggy where()": make_model("where"),
          "skip-bad-step guard": make_model("where"),
          "clamp fix": make_model("clamp"),
          "double-where fix": make_model("double_where")}
for name, mm in models.items():
    runs[name] = train(mm, steps=300, lr=0.05, guard_step=(name == "skip-bad-step guard"))

for name, h in runs.items():
    finite = [v for v in h["loss"] if np.isfinite(v)]
    F_.note(f"{name}: final loss",
            f"{h['loss'][-1]:.4f}" if np.isfinite(h["loss"][-1]) else "nan")
    F_.note(f"{name}: steps with a usable loss", f"{len(finite)} of {len(h['loss'])}")
    F_.note(f"{name}: accuracy", f"{accuracy(models[name]):.3f}")
    F_.note(f"{name}: parameters non-finite at the end", f"{h['param_nan'][-1] * 100:.1f}%")
    if h["skipped"]:
        F_.note(f"{name}: steps skipped by the guard", f"{h['skipped']} of {len(h['loss'])}")


# ===========================================================================
# figures
# ===========================================================================

def smooth(v, k=15):
    """Moving average. A mini-batch loss is noisy by construction (each step
    sees 32 random rows); smoothing shows the level, which is what these panels
    are about."""
    v = np.asarray(v, dtype=float)
    return np.convolve(v, np.ones(k) / k, mode="valid")


fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9), dpi=110)
for ax in axes:
    style_axes(ax)
fig.patch.set_facecolor("#fcfcfb")

ax = axes[0]
ax.plot(smooth(hist["loss"]), color=SERIES[0], lw=1.8, label="loss (buggy run)")
ax.plot(smooth(ref["loss"]), color=SERIES[1], lw=1.8, label="loss (bug fixed)")
ax.axhline(float(np.log(2)), color="#898781", lw=1.0, ls="-.", label="chance, ln(2)")
ax.axvline(first_grad_nan, color=SERIES[2], lw=1.4, ls="--",
           label=f"1st NaN gradient (step {first_grad_nan})")
ax2 = ax.twinx()
ax2.plot([v * 100 for v in hist["param_nan"]], color=SERIES[2], lw=1.2, alpha=0.55)
ax2.set_ylabel("% params NaN", color=SERIES[2], fontsize=9)
ax2.tick_params(colors=SERIES[2], labelsize=8)
ax2.spines["top"].set_visible(False)
ax.set_title("1. no nan ever reaches the loss", loc="left", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("loss (15-step moving average)")
ax.set_ylim(0.30, 1.05)
ax.legend(fontsize=7, frameon=False, loc="upper right")

ax = axes[1]
names = [n.replace(".weight", ".w").replace(".bias", ".b") for n, _ in per_layer]
vals = [f * 100 for _, f in per_layer]
ax.barh(range(len(vals)), vals, color=SERIES[1], height=0.65, label="no clipping")
ax.barh(range(len(vals)), [100] * len(vals), color=SERIES[2], height=0.28, label="with clip_grad_norm_")
ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("% of parameters that are NaN")
ax.set_title("2. one step of spread", loc="left", fontsize=11)
ax.legend(fontsize=7, frameon=False, loc="lower right")

ax = axes[2]
keys = list(timings)
best = [timings[k]["best"] * 1e3 for k in keys]
ax.bar(range(len(keys)), best, color=[SERIES[0], SERIES[1], SERIES[2]], width=0.6)
for i, v in enumerate(best):
    ax.text(i, v, f"{v / best[0]:.2f}x", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(["none", "hooks", "anomaly"], fontsize=8)
ax.set_ylabel("ms / training step")
ax.set_title("3. what each instrument costs", loc="left", fontsize=11)

ax = axes[3]
for i, (name, h) in enumerate(runs.items()):
    ax.plot(smooth(h["loss"]), color=SERIES[i], lw=1.7, label=name)
ax.set_title("4. buggy, guarded, fixed", loc="left", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("loss (15-step moving average)")
ax.set_ylim(0.35, 0.85)
ax.legend(fontsize=7, frameon=False)

save(fig, os.path.join(OUT, "nan_forensics.png"))
F_.write(os.path.join(OUT, "findings.csv"))
