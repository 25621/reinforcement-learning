"""Project 22 - turning a Triton kernel into something PyTorch can use.

  A. the naked kernel  - what a direct Triton call silently does not do
  B. registration      - the same kernel as an operator, and what changes
  C. the fake kernel   - missing it, and getting it wrong
  D. autograd          - a backward rule, checked against a CPU reference
  E. mutates_args      - the field that decides whether your call is deleted
  F. opcheck           - the test battery, on a good op and a broken one
  G. the inductor gate - why torch.compile refuses this GPU although Triton
                         itself runs on it
  H. the price         - dispatcher overhead per call
"""

import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                       # noqa: E402
from customop import (raw_silu_mul, silu_mul, double_lie,        # noqa: E402
                      double_honest, relu_badfake, relu_nofake)

N = 1 << 20
R = {}


def err(x, y):
    return (x - y).abs().max().item()


def section_a():
    print("A. the naked kernel: a direct Triton call")
    a, b = gpu.randn(N, seed=1), gpu.randn(N, seed=2)
    ref = F.silu(a.cpu()) * b.cpu()
    fwd = err(raw_silu_mul(a, b).cpu(), ref)

    ar = a.clone().requires_grad_(True)
    y = raw_silu_mul(ar, b)
    grad_lost = not y.requires_grad
    try:
        y.cpu().sum().backward()
        bwd_msg = "backward ran"
    except Exception as e:
        bwd_msg = type(e).__name__ + ": " + str(e).split("\n")[0]

    try:
        torch.compile(lambda x, z: raw_silu_mul(x, z), backend="aot_eager",
                      fullgraph=True)(a, b)
        comp_msg = "compiled"
    except Exception as e:
        comp_msg = type(e).__name__ + ": " + str(e).split("\n")[0]

    R["naked"] = dict(forward_err=fwd, output_requires_grad=y.requires_grad,
                      input_required_grad=True, backward=bwd_msg,
                      fullgraph_compile=comp_msg)
    print("   forward is correct:            max error %.3e" % fwd)
    print("   input requires grad:           True")
    print("   output requires grad:          %s   <- silently dropped"
          % y.requires_grad)
    print("   .backward():                   %s" % bwd_msg[:90])
    print("   torch.compile(fullgraph=True): %s" % comp_msg[:90])


def section_b():
    print("\nB. the same kernel, registered as aihw::silu_mul")
    a, b = gpu.randn(N, seed=1), gpu.randn(N, seed=2)
    ref = F.silu(a.cpu()) * b.cpu()
    fwd = err(silu_mul(a, b).cpu(), ref)
    try:
        out = torch.compile(lambda x, z: silu_mul(x, z), backend="aot_eager",
                            fullgraph=True)(a, b)
        torch.cuda.synchronize()
        comp = "compiled, max error vs eager %.3e" % err(out.cpu(),
                                                         silu_mul(a, b).cpu())
    except Exception as e:
        comp = type(e).__name__ + ": " + str(e).split("\n")[0]

    # what the compiler sees
    graphs = []

    def collect(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    # traced on CPU tensors: the graph is the same, and the surrounding
    # `* 2.0` is a PyTorch kernel this GPU cannot run
    torch.compile(lambda x, z: silu_mul(x, z) * 2.0, backend=collect,
                  fullgraph=True)(torch.randn(64), torch.randn(64))
    lines = [str(n.target) for n in graphs[0].graph.nodes
             if n.op == "call_function"]
    R["registered"] = dict(forward_err=fwd, fullgraph_compile=comp,
                           graph_nodes=lines)
    print("   forward is correct:            max error %.3e" % fwd)
    print("   torch.compile(fullgraph=True): %s" % comp)
    print("   the captured graph contains:   %s" % ", ".join(lines))


def section_c():
    print("\nC. the fake kernel")
    x = torch.arange(8, dtype=torch.float32)
    try:
        torch.compile(lambda z: relu_nofake(z) + 1.0, backend="aot_eager",
                      fullgraph=True)(x)
        nofake = "compiled"
    except Exception as e:
        nofake = type(e).__name__ + ": " + str(e).split("\n")[0]
    try:
        r = torch.compile(lambda z: relu_badfake(z) + 1.0, backend="aot_eager",
                          fullgraph=True)(x)
        badfake = "compiled and returned shape %s (eager gives %s)" % (
            tuple(r.shape), tuple((relu_badfake(x) + 1.0).shape))
    except Exception as e:
        badfake = type(e).__name__ + ": " + str(e).split("\n")[0]
    R["fake"] = dict(no_fake=nofake, bad_fake=badfake)
    print("   no fake registered:  %s" % nofake[:100])
    print("   wrong fake shape:    %s" % badfake[:100])


def section_d():
    print("\nD. autograd")
    a, b = gpu.randn(N, seed=3), gpu.randn(N, seed=4)
    ar, br = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    y = silu_mul(ar, br)
    y.cpu().sum().backward()
    ac = a.cpu().clone().requires_grad_(True)
    bc = b.cpu().clone().requires_grad_(True)
    (F.silu(ac) * bc).sum().backward()
    ga, gb = err(ar.grad.cpu(), ac.grad), err(br.grad.cpu(), bc.grad)
    R["autograd"] = dict(grad_fn=type(y.grad_fn).__name__,
                         requires_grad=y.requires_grad,
                         grad_a_err=ga, grad_b_err=gb)
    print("   output grad_fn:  %s" % type(y.grad_fn).__name__)
    print("   d/da max error:  %.3e" % ga)
    print("   d/db max error:  %.3e" % gb)


def section_e():
    print("\nE. mutates_args: the field that decides whether your call survives")
    x = torch.arange(8, dtype=torch.float32)
    correct = 2 * x.sum().item()

    def f_lie(z):
        y = z.clone()
        double_lie(y)
        return y.sum()

    def f_honest(z):
        y = z.clone()
        double_honest(y)
        return y.sum()

    rows = []
    for nm, fn in (("mutates_args=()   (a lie)", f_lie),
                   ("mutates_args={'x'} (true)", f_honest)):
        eager = fn(x).item()
        try:
            comp = torch.compile(fn, backend="aot_eager",
                                 fullgraph=True)(x).item()
        except Exception as e:
            comp = float("nan")
        rows.append(dict(op=nm, eager=eager, compiled=comp, correct=correct,
                         wrong=abs(comp - correct) > 1e-6))
        print("   %-26s eager %6.1f   compiled %6.1f   (correct %6.1f)%s"
              % (nm, eager, comp, correct,
                 "   <- WRONG, silently" if rows[-1]["wrong"] else ""))
    R["mutation"] = rows


def section_f():
    print("\nF. torch.library.opcheck")
    a = torch.randn(64, requires_grad=True)
    b = torch.randn(64, requires_grad=True)
    try:
        torch.library.opcheck(silu_mul, (a, b))
        good = "PASS"
    except Exception as e:
        good = type(e).__name__ + ": " + str(e).split("\n")[0]
    try:
        torch.library.opcheck(relu_badfake, (torch.randn(8),))
        bad = "PASS - did not catch the wrong fake"
    except Exception as e:
        bad = type(e).__name__ + ": " + str(e).split("\n")[0]
    R["opcheck"] = dict(good=good, bad_fake=bad)
    print("   aihw::silu_mul       %s" % good[:110])
    print("   aihw::relu_badfake   %s" % bad[:110])


def section_g():
    print("\nG. the inductor gate")
    a, b = gpu.randn(1024, seed=5), gpu.randn(1024, seed=6)
    try:
        torch.compile(lambda x, z: silu_mul(x, z), backend="inductor",
                      fullgraph=True)(a, b)
        msg = "compiled"
    except Exception as e:
        msg = type(e).__name__ + ": " + str(e).split("\n")[0]
    # ... while a hand-written Triton kernel runs on the same card
    hand = err(silu_mul(a, b).cpu(), F.silu(a.cpu()) * b.cpu())
    # inductor on the CPU has no such gate
    xc, yc = torch.randn(4096), torch.randn(4096)
    try:
        r = torch.compile(lambda p, q: silu_mul(p, q) * 2.0,
                          backend="inductor", fullgraph=True)(xc, yc)
        cpu_msg = "compiled, max error %.3e" % err(r, (F.silu(xc) * yc) * 2.0)
    except Exception as e:
        cpu_msg = type(e).__name__ + ": " + str(e).split("\n")[0]
    R["inductor"] = dict(gpu=msg, hand_written_triton_err=hand, cpu=cpu_msg)
    print("   torch.compile(backend='inductor') on this GPU:")
    print("     %s" % msg[:160])
    print("   the same Triton kernel, called by hand on the same GPU:")
    print("     ran fine, max error %.3e" % hand)
    print("   torch.compile(backend='inductor') on the CPU:")
    print("     %s" % cpu_msg[:120])


def section_h():
    print("\nH. what registration costs per call")
    rows = []
    for n in [1 << 10, 1 << 14, 1 << 18, 1 << 22]:
        a, b = gpu.randn(n, seed=7), gpu.randn(n, seed=8)
        ms_raw = gpu.bench(lambda: raw_silu_mul(a, b), reps=200)
        ms_op = gpu.bench(lambda: silu_mul(a, b), reps=200)
        rows.append(dict(n=n, raw_us=ms_raw * 1000, op_us=ms_op * 1000,
                         overhead_us=(ms_op - ms_raw) * 1000,
                         ratio=ms_op / ms_raw))
        r = rows[-1]
        print("   n=%-9d raw %8.2f us   registered %8.2f us   "
              "overhead %6.2f us (%.2fx)"
              % (n, r["raw_us"], r["op_us"], r["overhead_us"], r["ratio"]))
    R["overhead"] = rows


def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) capability matrix
    rows_lbl = ["correct forward", "autograd works", "torch.compile\n(fullgraph)",
                "opcheck-able"]
    naked = [1, 0, 0, 0]
    reg = [1, 1, 1, 1]
    ax[0].imshow([[n, r] for n, r in zip(naked, reg)], cmap="RdYlGn",
                 vmin=-0.3, vmax=1.3, aspect="auto")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["raw Triton\ncall", "registered\ncustom op"])
    ax[0].set_yticks(range(len(rows_lbl)))
    ax[0].set_yticklabels(rows_lbl, fontsize=8)
    for i in range(len(rows_lbl)):
        for j, vals in enumerate([naked, reg]):
            ax[0].text(j, i, "yes" if vals[i] else "no", ha="center",
                       va="center", fontsize=10)
    ax[0].set_title("A+B. what registration buys")

    # (2) the mutation result
    m = R["mutation"]
    xs = range(len(m))
    ax[1].bar([x - 0.2 for x in xs], [r["eager"] for r in m], 0.4,
              color="#1f77b4", label="eager")
    ax[1].bar([x + 0.2 for x in xs], [r["compiled"] for r in m], 0.4,
              color="#d62728", label="torch.compile")
    ax[1].axhline(m[0]["correct"], color="black", ls="--", label="correct")
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels(["mutates_args=()\n(a lie)", "mutates_args={'x'}\n(true)"],
                          fontsize=8)
    ax[1].set_ylabel("result")
    ax[1].set_title("E. an undeclared mutation is deleted")
    ax[1].set_ylim(0, 72)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=.3, axis="y")

    # (3) dispatcher overhead
    o = R["overhead"]
    ax[2].plot([r["n"] for r in o], [r["raw_us"] for r in o], "o-",
               color="#2ca02c", label="raw Triton call")
    ax[2].plot([r["n"] for r in o], [r["op_us"] for r in o], "s-",
               color="#9467bd", label="registered custom op")
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log", base=10)
    ax[2].set_xlabel("elements")
    ax[2].set_ylabel("microseconds per call")
    ax[2].set_title("H. registration costs %.1f us per call"
                    % (sum(r["overhead_us"] for r in o) / len(o)))
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    R["device"] = gpu.device_note()
    d = R["device"]
    print("device: %s (cc %s)   torch %s / triton %s\n"
          % (d["name"], d["cc"], torch.__version__, triton.__version__))
    gpu.warm_up()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()

    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value"])
        for k, v in R["naked"].items():
            w.writerow(["A", k, v])
        for k in ("forward_err", "fullgraph_compile"):
            w.writerow(["B", k, R["registered"][k]])
        w.writerow(["B", "graph_nodes", " | ".join(R["registered"]["graph_nodes"])])
        for k, v in R["fake"].items():
            w.writerow(["C", k, v])
        for k, v in R["autograd"].items():
            w.writerow(["D", k, v])
        for r in R["mutation"]:
            w.writerow(["E", r["op"], "eager %.1f compiled %.1f correct %.1f"
                        % (r["eager"], r["compiled"], r["correct"])])
        for k, v in R["opcheck"].items():
            w.writerow(["F", k, v])
        for k, v in R["inductor"].items():
            w.writerow(["G", k, v])
        for r in R["overhead"]:
            w.writerow(["H", "n_%d" % r["n"], "raw %.2f us, op %.2f us, "
                        "overhead %.2f us" % (r["raw_us"], r["op_us"],
                                              r["overhead_us"])])

    p = plot(os.path.join(OUT, "custom_op.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
