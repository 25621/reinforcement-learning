"""Project 31 - what changes when the cluster stops being one box.

There is one machine here, so "two nodes" means two groups of ranks with a
deliberately slow link between the groups. What that buys is the only lesson
that survives the missing hardware: an algorithm that ignores the node boundary
and an algorithm that respects it are the same FLOPs and a different runtime.

Sections
  A  measured: raw TCP over loopback -- latency floor, bandwidth, and Nagle
  B  measured: the rendezvous, and the exact commands for two real boxes
  C  emulated link: flat ring vs hierarchical all-reduce, 2 nodes x 2 ranks
  D  the crossover -- how slow must the link be before the shape matters
  E  arithmetic for a real cluster: NVLink vs InfiniBand on a 7B step

Runtime: ~28 s.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "28-nccl-tests"))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from commlib import run_ranks  # noqa: E402
from netsim import Link, flat_ring, hierarchical, model_time  # noqa: E402
from sockbench import pingpong, two_write  # noqa: E402

MSG = 1 << 22          # 4 MiB, about the size of one bucket of a small model
RANKS_PER_NODE = 2
WORLD = 4
findings: dict = {}


# ------------------------------------------------------------------ A

def section_a():
    res = {}
    for nodelay in [True, False]:
        res["nodelay" if nodelay else "nagle"] = pingpong(
            port=41777 + int(nodelay), nodelay=nodelay)
    split = {k: two_write(port=41800 + i, nodelay=k == "nodelay")
             for i, k in enumerate(["nodelay", "nagle"])}
    small = res["nodelay"][0]
    big = res["nodelay"][-1]
    findings["A_tcp"] = dict(
        points=res,
        latency_us=small["oneway_us"],
        peak_GBs=max(p["GBs"] for p in res["nodelay"]),
        nagle_penalty=res["nagle"][0]["rtt_us"] / res["nodelay"][0]["rtt_us"],
        split_write=split,
        split_nagle_penalty=split["nagle"]["rtt_us"] / split["nodelay"]["rtt_us"],
        hostname=socket.gethostname(),
    )
    print(f"A: 8-byte one-way {small['oneway_us']:.1f} us   "
          f"{big['bytes']/1e6:.0f} MB at {big['GBs']:.2f} GB/s   "
          f"peak {findings['A_tcp']['peak_GBs']:.2f} GB/s   "
          f"Nagle on 8 B costs {findings['A_tcp']['nagle_penalty']:.2f}x")
    print(f"A: two 4-byte writes per round trip: nodelay "
          f"{split['nodelay']['rtt_us']:.1f} us vs Nagle {split['nagle']['rtt_us']:.1f} us "
          f"({findings['A_tcp']['split_nagle_penalty']:.2f}x)")


# ------------------------------------------------------------------ B

def _rendezvous(rank, world, rpn):
    node = rank // rpn
    x = torch.randn(MSG // 4)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(3):
        dist.all_reduce(x)
    t = torch.tensor([(time.perf_counter() - t0) / 3])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return dict(rank=rank, node=node, world=world,
                master=f"{dist.get_rank()}@{socket.gethostname()}",
                baseline_allreduce_s=float(t.item()))


def section_b():
    r = run_ranks(_rendezvous, WORLD, RANKS_PER_NODE)
    r["torchrun_node0"] = (
        "torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 "
        "--master_addr=10.0.0.1 --master_port=29500 train.py")
    r["torchrun_node1"] = (
        "torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 "
        "--master_addr=10.0.0.1 --master_port=29500 train.py")
    r["required_env"] = ["MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME"]
    findings["B_rendezvous"] = r
    print(f"B: {WORLD} ranks over {WORLD // RANKS_PER_NODE} emulated nodes; "
          f"baseline 4 MiB all-reduce {r['baseline_allreduce_s']*1e3:.2f} ms "
          f"(no link penalty yet)")


# ------------------------------------------------------------------ C + D

def _compare(rank, world, rpn, configs, msg, reps):
    out = []
    for lat_us, bw in configs:
        for name, fn in [("flat_ring", flat_ring), ("hierarchical", hierarchical)]:
            link = Link(rpn, lat_us * 1e-6, bw)
            ref = torch.randn(msg // 4)
            gold = ref.clone()
            dist.all_reduce(gold)
            chk = ref.clone()
            fn(chk, rank, world, Link(rpn, 0, 0))     # correctness with a free link
            err = float((chk - gold).abs().max())

            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(ref.clone(), rank, world, link)
            t = torch.tensor([(time.perf_counter() - t0) / reps])
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            # Each rank only charges its *own* sends, and rank 0 may not send
            # across at all (in the 4-rank ring it is ranks 1 and 3 that do),
            # so the counters have to be summed over the whole world.
            cc = torch.tensor([float(link.crossings), float(link.crossed_bytes)])
            dist.all_reduce(cc, op=dist.ReduceOp.SUM)
            out.append(dict(lat_us=lat_us, bw=bw, alg=name, s=float(t.item()),
                            err=err, crossings=float(cc[0]) / reps,
                            crossed_bytes=float(cc[1]) / reps))
    return out


def section_c():
    configs = [(0, 0), (0, 5.0), (100, 5.0), (100, 1.0), (100, 0.2)]
    rows = run_ranks(_compare, WORLD, RANKS_PER_NODE, configs, MSG, 3)
    by = {}
    for r in rows:
        by.setdefault((r["lat_us"], r["bw"]), {})[r["alg"]] = r
    res = []
    for key, d in by.items():
        flat, hier = d["flat_ring"], d["hierarchical"]
        res.append(dict(lat_us=key[0], bw_GBs=key[1],
                        flat_ms=flat["s"] * 1e3, hier_ms=hier["s"] * 1e3,
                        speedup=flat["s"] / hier["s"],
                        flat_crossed_MB=flat["crossed_bytes"] / 1e6,
                        hier_crossed_MB=hier["crossed_bytes"] / 1e6,
                        flat_crossings=flat["crossings"], hier_crossings=hier["crossings"],
                        max_err=max(flat["err"], hier["err"])))
        print(f"C: link lat={key[0]:4d} us bw={key[1] or 'free':>4} GB/s | "
              f"flat {res[-1]['flat_ms']:8.2f} ms ({flat['crossings']:.0f} crossings, "
              f"{res[-1]['flat_crossed_MB']:.2f} MB) | "
              f"hier {res[-1]['hier_ms']:8.2f} ms ({hier['crossings']:.0f}, "
              f"{res[-1]['hier_crossed_MB']:.2f} MB) | {res[-1]['speedup']:.2f}x")
    findings["C_link_sweep"] = res


def section_d():
    bws = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    configs = [(50, b) for b in bws]
    rows = run_ranks(_compare, WORLD, RANKS_PER_NODE, configs, MSG, 2)
    by = {}
    for r in rows:
        by.setdefault(r["bw"], {})[r["alg"]] = r
    res = []
    for bw in bws:
        flat, hier = by[bw]["flat_ring"], by[bw]["hierarchical"]
        pred_f = model_time("flat_ring", WORLD, RANKS_PER_NODE, MSG, 0, 2.0, 50e-6, bw)
        pred_h = model_time("hierarchical", WORLD, RANKS_PER_NODE, MSG, 0, 2.0, 50e-6, bw)
        res.append(dict(bw_GBs=bw, flat_ms=flat["s"] * 1e3, hier_ms=hier["s"] * 1e3,
                        speedup=flat["s"] / hier["s"],
                        predicted_speedup=pred_f / pred_h))
        print(f"D: inter-node {bw:5.1f} GB/s -> measured {res[-1]['speedup']:.2f}x, "
              f"model {res[-1]['predicted_speedup']:.2f}x")
    findings["D_crossover"] = res


# ------------------------------------------------------------------ E

def section_e():
    """Published link speeds, our own algorithm arithmetic. 7B params in bf16 =
    14 GB of gradients per all-reduce."""
    grad_bytes = 7e9 * 2
    scenarios = {
        "NVLink 4 (H100, in-node)": 900e9,
        "InfiniBand NDR 400 Gb/s": 400e9 / 8,
        "InfiniBand HDR 200 Gb/s": 200e9 / 8,
        "100 GbE": 100e9 / 8,
        "10 GbE": 10e9 / 8,
    }
    rows = []
    g, nodes = 8, 2
    world = g * nodes
    for name, bw in scenarios.items():
        flat = model_time("flat_ring", world, g, grad_bytes, 0, 900, 5e-6, bw / 1e9)
        hier = model_time("hierarchical", world, g, grad_bytes, 0, 900, 5e-6, bw / 1e9)
        rows.append(dict(link=name, bw_GBs=bw / 1e9, flat_s=flat, hier_s=hier,
                         speedup=flat / hier))
        print(f"E: {name:28s} {bw/1e9:7.1f} GB/s  flat {flat*1e3:8.1f} ms  "
              f"hier {hier*1e3:8.1f} ms  {flat/hier:.2f}x")
    findings["E_cluster_arithmetic"] = dict(grad_bytes=grad_bytes, world=world,
                                            ranks_per_node=g, rows=rows)


# ------------------------------------------------------------------ plot

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    a = findings["A_tcp"]["points"]
    for k, style in [("nodelay", "o-"), ("nagle", "s--")]:
        ax[0][0].plot([p["bytes"] for p in a[k]], [p["GBs"] for p in a[k]], style, label=k)
    ax[0][0].set_xscale("log", base=2)
    ax[0][0].set_xlabel("message bytes")
    ax[0][0].set_ylabel("GB/s (one way)")
    ax[0][0].set_title("A. raw TCP over loopback")
    ax[0][0].legend(fontsize=7)
    ax[0][0].grid(alpha=.3)

    c = findings["C_link_sweep"]
    labels = [f"{r['lat_us']}us\n{r['bw_GBs'] or 'free'}" for r in c]
    xs = range(len(c))
    ax[0][1].bar([x - .2 for x in xs], [r["flat_ms"] for r in c], width=.4, label="flat ring")
    ax[0][1].bar([x + .2 for x in xs], [r["hier_ms"] for r in c], width=.4, label="hierarchical")
    ax[0][1].set_xticks(list(xs))
    ax[0][1].set_xticklabels(labels, fontsize=6)
    ax[0][1].set_ylabel("ms per 4 MiB all-reduce")
    ax[0][1].set_title("B. the same all-reduce, two shapes")
    ax[0][1].legend(fontsize=7)
    ax[0][1].grid(alpha=.3)

    d = findings["D_crossover"]
    ax[1][0].plot([r["bw_GBs"] for r in d], [r["speedup"] for r in d], "o-", label="measured")
    ax[1][0].plot([r["bw_GBs"] for r in d], [r["predicted_speedup"] for r in d], "s--",
                  label="alpha-beta model")
    ax[1][0].axhline(1.0, color="red", ls=":", lw=1)
    ax[1][0].set_xscale("log")
    ax[1][0].set_xlabel("inter-node bandwidth (GB/s)")
    ax[1][0].set_ylabel("hierarchical speedup (x)")
    ax[1][0].set_title("C. how slow does the link have to be?")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=.3)

    e = findings["E_cluster_arithmetic"]["rows"]
    ys = range(len(e))
    ax[1][1].barh([y - .2 for y in ys], [r["flat_s"] * 1e3 for r in e], height=.4, label="flat")
    ax[1][1].barh([y + .2 for y in ys], [r["hier_s"] * 1e3 for r in e], height=.4, label="hierarchical")
    ax[1][1].set_yticks(list(ys))
    ax[1][1].set_yticklabels([r["link"] for r in e], fontsize=6)
    ax[1][1].set_xscale("log")
    ax[1][1].set_xlabel("ms for one 14 GB all-reduce (16 GPUs, 2 nodes)")
    ax[1][1].set_title("D. arithmetic for hardware we do not have")
    ax[1][1].legend(fontsize=7)
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "multi_node.png", dpi=120)


def main():
    t0 = time.perf_counter()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    plot()
    findings["runtime_s"] = time.perf_counter() - t0
    print(f"total runtime {findings['runtime_s']:.1f} s")
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    with open(OUT / "findings.csv", "w") as f:
        f.write("section,key,flat_ms,hier_ms,speedup\n")
        for r in findings["C_link_sweep"]:
            f.write(f"C,lat{r['lat_us']}us_bw{r['bw_GBs']},{r['flat_ms']:.3f},"
                    f"{r['hier_ms']:.3f},{r['speedup']:.3f}\n")
        for r in findings["D_crossover"]:
            f.write(f"D,bw{r['bw_GBs']},{r['flat_ms']:.3f},{r['hier_ms']:.3f},{r['speedup']:.3f}\n")
        for r in findings["E_cluster_arithmetic"]["rows"]:
            f.write(f"E,{r['link']},{r['flat_s']*1e3:.3f},{r['hier_s']*1e3:.3f},{r['speedup']:.3f}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
