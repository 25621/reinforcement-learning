"""Project 06 — Micrograd in PyTorch style.

Build a scalar autograd engine (`engine.py`), then hold it against the real
thing:

  1. one expression, every intermediate gradient compared to torch
  2. what breaks if you walk the graph in the wrong order
  3. why `.grad` accumulates, and what `zero_grad()` is actually for
  4. train the same MLP twice -- our engine vs torch -- from identical weights
  5. what the scalar engine costs, and why real frameworks are tensor-valued

Runs in well under a minute. No downloads.
"""

import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-stride-explorer"))
import plot_style as ps
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from engine import Value

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(1)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # match Python floats exactly

FINDINGS = {}


def rec(key, value):
    FINDINGS[key] = value
    return value


# =========================================================================
# 1. One expression, checked against torch node by node
# =========================================================================
def check_against_torch():
    print("=" * 78)
    print("1. THE SAME EXPRESSION, TWICE")
    print("=" * 78)

    # A deliberately tangled expression: x is used three times, so its gradient
    # arrives from three different directions and has to be summed.
    def f_value(xv, wv, bv):
        x, w, b = Value(xv, label="x"), Value(wv, label="w"), Value(bv, label="b")
        h = (x * w + b).tanh()
        g = (x * x).relu()
        out = (h * g + x).sigmoid().log() * -1
        return out, {"x": x, "w": w, "b": b, "h": h, "g": g}

    def f_torch(xv, wv, bv):
        x = torch.tensor(xv, requires_grad=True)
        w = torch.tensor(wv, requires_grad=True)
        b = torch.tensor(bv, requires_grad=True)
        h = torch.tanh(x * w + b)
        g = torch.relu(x * x)
        h.retain_grad()          # h is not a leaf; ask torch to keep its grad
        g.retain_grad()
        out = -torch.log(torch.sigmoid(h * g + x))
        return out, {"x": x, "w": w, "b": b, "h": h, "g": g}

    xv, wv, bv = 0.73, -1.4, 0.31
    ours, ns_o = f_value(xv, wv, bv)
    theirs, ns_t = f_torch(xv, wv, bv)

    ours.backward()
    theirs.backward()

    print(f"  forward   ours {ours.data:.15f}   torch {theirs.item():.15f}")
    fwd_diff = abs(ours.data - theirs.item())
    rec("forward_abs_diff", f"{fwd_diff:.3e}")

    print(f"\n  {'node':<6}{'ours .grad':>22}{'torch .grad':>22}{'abs diff':>14}")
    worst = 0.0
    for name in ["x", "w", "b", "h", "g"]:
        a = ns_o[name].grad
        b_ = ns_t[name].grad.item()
        d = abs(a - b_)
        worst = max(worst, d)
        print(f"  {name:<6}{a:>22.15f}{b_:>22.15f}{d:>14.2e}")
    rec("expr_max_grad_diff", f"{worst:.3e}")
    print(f"\n  worst disagreement anywhere in the graph: {worst:.2e}")

    # The graph itself, printed in the order backward visits it.
    topo = ours.topo_order()
    rec("expr_graph_nodes", len(topo))
    print(f"\n  the graph has {len(topo)} nodes; backward visits them in this order:")
    for i, n in enumerate(reversed(topo)):
        tag = n.label or n._op or "const"
        print(f"    {i:>2}. {tag:<8} data={n.data:>10.5f}  grad={n.grad:>10.5f}")

    # torch names the same nodes; print its chain for comparison.
    gf = theirs.grad_fn
    chain = []
    while gf is not None and len(chain) < 6:
        chain.append(type(gf).__name__)
        nxt = [f for f, _ in gf.next_functions if f is not None]
        gf = nxt[0] if nxt else None
    print(f"\n  torch calls the same nodes: {' -> '.join(chain)}")
    rec("torch_grad_fn_head", chain[0])
    return ns_o, ours


# =========================================================================
# 2. Why the order matters
# =========================================================================
def order_matters():
    print("\n" + "=" * 78)
    print("2. WHAT GOES WRONG WITHOUT A TOPOLOGICAL SORT")
    print("=" * 78)

    def build():
        x = Value(3.0, label="x")
        a = x + 0.0        # a shared intermediate
        b = a * 2.0        # ... used once through b ...
        y = a + b          # ... and once directly. y = a + 2a = 3x
        return x, a, b, y

    true = 3.0             # y = 3x  ->  dy/dx = 3

    # correct: a node runs only after every node that consumes it
    x, a, b, y = build()
    y.backward()
    good = x.grad

    # a very reasonable-looking wrong answer: breadth-first from the output.
    # Start at y, then everything y touches, then everything those touch.
    x2, a2, b2, y2 = build()
    bfs, seen, queue = [], set(), [y2]
    while queue:
        n = queue.pop(0)
        if id(n) in seen:
            continue
        seen.add(id(n))
        bfs.append(n)
        queue.extend(n._prev)
    y2.grad = 1.0
    for node in bfs:
        node._backward()
    bad = x2.grad

    print(f"  y = a + 2a where a = x, at x=3   ->  dy/dx should be {true}")
    print(f"    order visited (breadth-first): "
          f"{[n.label or n._op or f'{n.data:g}' for n in bfs]}")
    print(f"    topological order  : dx = {good}   <- right")
    print(f"    breadth-first order: dx = {bad}   <- wrong, and it did not crash")
    print("\n  Breadth-first reaches `a` one hop from the output, so it runs a's")
    print("  backward while a.grad is only 1.0 -- the +2.0 coming through b has")
    print("  not been delivered yet. a passes on the partial number and is never")
    print("  asked again. The answer is off by exactly the path that was late.")
    print("\n  The rule a topological sort enforces: never run a node's backward")
    print("  until EVERY node that consumed its output has run. PyTorch enforces")
    print("  the same rule with a dependency counter per node in its C++ engine.")
    rec("shared_node_true_grad", true)
    rec("shared_node_topo_grad", good)
    rec("shared_node_bfs_grad", bad)

    # accumulation across the two paths, shown explicitly
    x3, a3, b3, y3 = build()
    y3.backward()
    print(f"\n  With the right order the two paths arrive separately and ADD:")
    print(f"    direct  y = a + b : da += 1.0")
    print(f"    through b = 2a    : da += 2.0")
    print(f"    total a.grad = {a3.grad}  ->  x.grad = {x3.grad}")
    return good, bad, true


# =========================================================================
# 3. Accumulation and zero_grad
# =========================================================================
def accumulate():
    print("\n" + "=" * 78)
    print("3. WHY .grad ACCUMULATES (AND WHY zero_grad EXISTS)")
    print("=" * 78)

    w = Value(2.0, label="w")
    grads = []
    for step in range(3):
        loss = w * w * 3            # 3w^2 -> d/dw = 6w = 12
        loss.backward()
        grads.append(w.grad)
        print(f"  backward #{step + 1}: w.grad = {w.grad:>6.1f}   (one pass is worth 12.0)")

    wt = torch.tensor(2.0, requires_grad=True)
    tgrads = []
    for _ in range(3):
        (wt * wt * 3).backward()
        tgrads.append(wt.grad.item())
    print(f"  torch does exactly the same: {tgrads}")
    rec("accum_after_3_backwards", grads[-1])
    rec("torch_accum_after_3_backwards", tgrads[-1])

    w.zero_grad()
    (w * w * 3).backward()
    print(f"\n  after zero_grad(): w.grad = {w.grad}")
    print("\n  Accumulation is not a wart. A node whose output is used twice gets")
    print("  two gradients and must add them -- that is the sum rule of calculus.")
    print("  The engine cannot tell 'second use in one graph' from 'second graph',")
    print("  so both add. Clearing between steps is YOUR job: optimizer.zero_grad().")
    return grads, tgrads


# =========================================================================
# 4. Train the same MLP twice
# =========================================================================
class MLP:
    """Scalar MLP. Layer sizes like (2, 8, 8, 1); tanh hidden, linear output."""

    def __init__(self, sizes, w_init, b_init):
        self.layers = []
        for li in range(len(sizes) - 1):
            nin, nout = sizes[li], sizes[li + 1]
            W = [[Value(w_init[li][j][i]) for i in range(nin)] for j in range(nout)]
            b = [Value(b_init[li][j]) for j in range(nout)]
            self.layers.append((W, b, li < len(sizes) - 2))

    def __call__(self, xs):
        act = [Value(v) for v in xs]
        for W, b, use_tanh in self.layers:
            nxt = []
            for j in range(len(W)):
                s = b[j]
                for i in range(len(act)):
                    s = s + W[j][i] * act[i]
                nxt.append(s.tanh() if use_tanh else s)
            act = nxt
        return act[0]

    def parameters(self):
        out = []
        for W, b, _ in self.layers:
            for row in W:
                out.extend(row)
            out.extend(b)
        return out


def make_data(n=80, seed=0):
    """Two interleaving half-circles -- not separable by a straight line."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, math.pi, n // 2)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    x1 = np.stack([1 - np.cos(t), 0.4 - np.sin(t)], 1)
    X = np.concatenate([x0, x1]) + rng.normal(0, 0.12, (n, 2))
    y = np.concatenate([-np.ones(n // 2), np.ones(n // 2)])
    return X, y


def train_both(steps=120, lr=0.08):
    print("\n" + "=" * 78)
    print("4. THE SAME MLP, TRAINED BY BOTH ENGINES")
    print("=" * 78)

    X, y = make_data()
    sizes = [2, 8, 8, 1]
    rng = np.random.default_rng(1)
    w_init = [rng.normal(0, 0.7, (sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
    b_init = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # --- our engine -------------------------------------------------------
    t0 = time.perf_counter()
    net = MLP(sizes, [w.tolist() for w in w_init], [b.tolist() for b in b_init])
    params = net.parameters()
    ours_curve = []
    for _ in range(steps):
        outs = [net(row) for row in X.tolist()]
        losses = [(o - yi) ** 2 for o, yi in zip(outs, y.tolist())]
        total = losses[0]
        for l in losses[1:]:
            total = total + l
        loss = total * (1.0 / len(losses))
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
        ours_curve.append(loss.data)
    ours_time = time.perf_counter() - t0

    # how big does the graph get?
    outs = [net(row) for row in X.tolist()]
    total = outs[0] * outs[0]
    for o in outs[1:]:
        total = total + o * o
    graph_nodes = len(total.topo_order())

    # --- torch, identical weights ----------------------------------------
    class TorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ls = torch.nn.ModuleList(
                [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
            with torch.no_grad():
                for i, l in enumerate(self.ls):
                    l.weight.copy_(torch.tensor(w_init[i]))
                    l.bias.copy_(torch.tensor(b_init[i]))

        def forward(self, x):
            for i, l in enumerate(self.ls):
                x = l(x)
                if i < len(self.ls) - 1:
                    x = torch.tanh(x)
            return x.squeeze(-1)

    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    tnet = TorchMLP()
    t0 = time.perf_counter()
    torch_curve = []
    for _ in range(steps):
        loss = ((tnet(Xt) - yt) ** 2).mean()
        tnet.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in tnet.parameters():
                p -= lr * p.grad
        torch_curve.append(loss.item())
    torch_time = time.perf_counter() - t0

    curve_diff = max(abs(a - b) for a, b in zip(ours_curve, torch_curve))
    final_ours = [p.data for p in params]
    final_torch = []
    for l in tnet.ls:
        final_torch.extend(l.weight.detach().flatten().tolist())
        final_torch.extend(l.bias.detach().tolist())
    # our parameter list is W-then-b per layer, same as the loop above
    param_diff = max(abs(a - b) for a, b in zip(final_ours, final_torch))

    # count torch's graph nodes the honest way: walk grad_fn
    tloss = ((tnet(Xt) - yt) ** 2).mean()
    # NOTE: `keep` is not optional. Each access to .grad_fn / .next_functions
    # hands back a fresh Python wrapper; once it is collected, CPython happily
    # reuses its id() for the next one and a plain id()-keyed `seen` set starts
    # reporting nodes it has never visited. Holding a reference pins the ids.
    seen, stack, keep, tnodes = set(), [tloss.grad_fn], [], 0
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keep.append(fn)
        tnodes += 1
        stack.extend(f for f, _ in fn.next_functions)

    acc = float(np.mean(np.sign([net(r).data for r in X.tolist()]) == y))
    print(f"  final loss   ours {ours_curve[-1]:.12f}   torch {torch_curve[-1]:.12f}")
    print(f"  biggest loss disagreement over all {steps} steps : {curve_diff:.2e}")
    print(f"  biggest final-parameter disagreement (105 params): {param_diff:.2e}")
    print(f"  training accuracy after {steps} steps            : {acc:.3f}")
    print(f"\n  wall clock   ours {ours_time:.2f}s   torch {torch_time:.2f}s"
          f"   -> torch is {ours_time / torch_time:.0f}x faster")
    print(f"  one forward pass over {len(X)} points builds {graph_nodes:,} Value nodes;")
    print(f"  torch's graph for the same forward pass has {tnodes} nodes.")
    rec("torch_graph_nodes_per_forward", tnodes)

    rec("mlp_steps", steps)
    rec("mlp_final_loss_ours", round(ours_curve[-1], 12))
    rec("mlp_final_loss_torch", round(torch_curve[-1], 12))
    rec("mlp_max_loss_diff", f"{curve_diff:.3e}")
    rec("mlp_max_param_diff", f"{param_diff:.3e}")
    rec("mlp_train_accuracy", round(acc, 4))
    rec("mlp_seconds_ours", round(ours_time, 3))
    rec("mlp_seconds_torch", round(torch_time, 3))
    rec("mlp_speedup_torch", round(ours_time / torch_time, 1))
    rec("mlp_value_nodes_per_forward", graph_nodes)
    return ours_curve, torch_curve, net, X, y


# =========================================================================
# 5. What a scalar node costs
# =========================================================================
def cost_of_scalars():
    print("\n" + "=" * 78)
    print("5. THE PRICE OF ONE NUMBER PER NODE")
    print("=" * 78)

    n = 20000
    t0 = time.perf_counter()
    acc = Value(0.0)
    xs = [Value(0.001 * i) for i in range(n)]
    for v in xs:
        acc = acc + v * v
    build_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    acc.backward()
    back_t = time.perf_counter() - t0

    nodes = len(acc.topo_order())

    # torch is fast enough that one measurement is mostly noise -> best of 5
    tbuild, tback = 1e9, 1e9
    for _ in range(5):
        xt = torch.arange(n, dtype=torch.float64) * 0.001
        xt.requires_grad_(True)
        t0 = time.perf_counter()
        tacc = (xt * xt).sum()
        tbuild = min(tbuild, time.perf_counter() - t0)
        t0 = time.perf_counter()
        tacc.backward()
        tback = min(tback, time.perf_counter() - t0)

    print(f"  sum of {n:,} squares")
    print(f"    ours : forward {build_t * 1e3:8.1f} ms   backward {back_t * 1e3:8.1f} ms"
          f"   ({nodes:,} graph nodes)")
    print(f"    torch: forward {tbuild * 1e3:8.3f} ms   backward {tback * 1e3:8.3f} ms"
          f"   (2 graph nodes)")
    print(f"    ratio: {(build_t + back_t) / (tbuild + tback):.0f}x")
    print("\n  Same arithmetic, same answer. The difference is bookkeeping:")
    print("  our engine stores a Python object, a closure and a tuple per number.")
    print("  torch stores ONE node for the whole 20,000-element multiply.")
    rec("scalar_sum_n", n)
    rec("scalar_graph_nodes", nodes)
    rec("scalar_ms_ours", round((build_t + back_t) * 1e3, 1))
    rec("scalar_ms_torch", round((tbuild + tback) * 1e3, 3))
    rec("scalar_slowdown", round((build_t + back_t) / (tbuild + tback), 1))
    print(f"    ours grad on x[10] = {xs[10].grad:.6f}   torch = {xt.grad[10].item():.6f}")
    rec("scalar_grad_match", abs(xs[10].grad - xt.grad[10].item()) < 1e-12)

    # The same 20,000-term sum is also 20,000 nodes DEEP, which is what kills a
    # recursive graph walk.
    def recursive_topo(v, order=None, seen=None):
        order, seen = ([] if order is None else order), (set() if seen is None else seen)
        if id(v) in seen:
            return order
        seen.add(id(v))
        for c in v._prev:
            recursive_topo(c, order, seen)
        order.append(v)
        return order

    try:
        recursive_topo(acc)
        msg = "survived"
    except RecursionError:
        msg = f"RecursionError (Python's limit is {sys.getrecursionlimit()} frames)"
    print(f"\n  same graph, recursive topological sort: {msg}")
    print(f"  the iterative one in engine.py: fine, {len(acc.topo_order()):,} nodes")
    rec("recursive_topo_result", msg.split(" (")[0])
    return (build_t + back_t) * 1e3, (tbuild + tback) * 1e3


# =========================================================================
# figures
# =========================================================================
def fig_graph(root):
    """Draw the little expression graph with data and grad on every node."""
    topo = root.topo_order()
    depth = {}
    for n in topo:                       # topo order guarantees inputs first
        depth[id(n)] = 0 if not n._prev else 1 + max(depth[id(c)] for c in n._prev)
    levels = {}
    for n in topo:
        levels.setdefault(depth[id(n)], []).append(n)

    fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax.set_facecolor(ps.SURFACE)
    pos = {}
    maxrow = max(len(v) for v in levels.values())  # widest layer
    for d, nodes in levels.items():
        for i, n in enumerate(nodes):
            y = (i - (len(nodes) - 1) / 2) * 1.25
            pos[id(n)] = (d * 1.9, y)

    for n in topo:
        for c in n._prev:
            x0, y0 = pos[id(c)]
            x1, y1 = pos[id(n)]
            ax.add_patch(FancyArrowPatch((x0 + 0.62, y0), (x1 - 0.62, y1),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=ps.BASELINE, lw=1.1, zorder=1))
    for n in topo:
        x, y = pos[id(n)]
        leaf = not n._prev
        color = ps.SERIES[0] if leaf else ps.SERIES[2] if n._op in ("tanh", "relu", "sigmoid", "log", "exp") else ps.INK_SECONDARY
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.44), 1.2, 0.88,
                                    boxstyle="round,pad=0.02", linewidth=1.4,
                                    edgecolor=color, facecolor="white", zorder=2))
        tag = n.label or n._op or f"{n.data:g}"
        ax.text(x, y + 0.20, tag, ha="center", va="center", fontsize=9,
                color=color, fontweight="bold", zorder=3)
        ax.text(x, y - 0.02, f"{n.data:.3f}", ha="center", va="center",
                fontsize=8, color=ps.INK, zorder=3)
        ax.text(x, y - 0.26, f"g {n.grad:+.3f}", ha="center", va="center",
                fontsize=7.5, color=ps.SERIES[1], zorder=3)

    ys_all = [p[1] for p in pos.values()]
    ax.set_xlim(-0.9, max(levels) * 1.9 + 0.9)
    ax.set_ylim(min(ys_all) - 0.7, max(ys_all) + 0.7)
    ax.axis("off")
    ax.set_title("One expression as a graph — value on top, gradient below\n"
                 "(blue = leaf you can differentiate w.r.t.; red = nonlinearity; arrows point forward)",
                 color=ps.INK, fontsize=11, loc="left", pad=10)
    ps.save(fig, os.path.join(OUT, "expression_graph.png"))


def fig_curves(ours, theirs):
    fig, ax = ps.new_axes(7.2, 4.0)
    ax.plot(ours, color=ps.SERIES[0], lw=3.2, alpha=0.55, label="our scalar engine")
    ax.plot(theirs, color=ps.SERIES[2], lw=1.3, ls="--", label="torch autograd")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Two engines, one curve (they overlap to 1e-15)",
              "gradient-descent step", "mean squared error",
              os.path.join(OUT, "training_curves.png"))


def fig_boundary(net, X, y):
    xs = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 60)
    ys = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 60)
    Z = np.zeros((len(ys), len(xs)))
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            Z[j, i] = net([xx, yy]).data
    fig, ax = ps.new_axes(6.0, 4.4)
    ax.contourf(xs, ys, Z, levels=[-9, 0, 9], colors=[ps.SERIES[0], ps.SERIES[2]], alpha=0.16)
    ax.contour(xs, ys, Z, levels=[0], colors=[ps.INK_SECONDARY], linewidths=1.4)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=22, color=ps.SERIES[0], edgecolor="white", lw=0.6)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=22, color=ps.SERIES[2], edgecolor="white", lw=0.6)
    ax.grid(False)
    ps.finish(fig, ax, "A curved boundary, learned with 105 scalar Values",
              "x₀", "x₁", os.path.join(OUT, "decision_boundary.png"))


def fig_cost(ours_ms, torch_ms):
    fig, ax = ps.new_axes(6.4, 3.6)
    ax.barh([0, 1], [ours_ms, torch_ms],
            color=[ps.SERIES[0], ps.SERIES[2]], height=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["scalar engine\n(60,001 nodes)", "torch\n(2 nodes)"], fontsize=9)
    ax.set_xscale("log")
    for i, v in enumerate([ours_ms, torch_ms]):
        ax.text(v * 1.15, i, f"{v:.2f} ms", va="center", fontsize=9, color=ps.INK)
    ps.finish(fig, ax, "Forward + backward on a sum of 20,000 squares",
              "milliseconds (log scale)", "",
              os.path.join(OUT, "scalar_cost.png"))


# =========================================================================
def main():
    nodes, root = check_against_torch()
    order_matters()
    accumulate()
    ours_curve, torch_curve, net, X, y = train_both()
    ours_ms, torch_ms = cost_of_scalars()

    fig_graph(root)
    fig_curves(ours_curve, torch_curve)
    fig_boundary(net, X, y)
    fig_cost(ours_ms, torch_ms)

    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding", "value"])
        for k, v in FINDINGS.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/findings.csv")


if __name__ == "__main__":
    main()
