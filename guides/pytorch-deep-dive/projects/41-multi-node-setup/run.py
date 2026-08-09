"""Project 41 — Two nodes: the launch, the identity variables, and the network.

Run:  python3 run.py           (~4 minutes)

We have one machine, so the two "nodes" are two separate `torchrun` commands on
the same host, each with `--node-rank` set, meeting at a c10d rendezvous. That
is the real multi-node launch protocol and the real rendezvous — what it cannot
reproduce is the network between the machines, and section 5 measures exactly
how badly that flatters the result.

Sections
  1. two torchrun commands, one job: who is who
  2. RANK vs LOCAL_RANK vs GROUP_RANK, and the checkpoint bug
  3. rendezvous failures, verbatim
  4. --max-restarts: what torchrun does when a process dies
  5. the latency/bandwidth model of the link, and what it predicts
  6. hierarchical all-reduce: how much has to cross the network
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "36-two-gpu-ddp"))
import dist_lib as D  # noqa: E402

OUT = os.path.join(HERE, "outputs")
WORK = os.path.join(OUT, "work")
os.makedirs(OUT, exist_ok=True)
FINDINGS = []


def record(section, name, value, note=""):
    FINDINGS.append({"section": section, "name": name, "value": value, "note": note})
    print(f"    {name:<52} {value}")


# ---------------------------------------------------------------------------
# launching torchrun
# ---------------------------------------------------------------------------

def torchrun_cmd(node_rank, nnodes, nproc, port, job_id, script_args,
                 max_restarts=0, rdzv_timeout=None, rdzv="c10d"):
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nnodes={nnodes}", f"--node-rank={node_rank}",
           f"--nproc-per-node={nproc}", f"--max-restarts={max_restarts}"]
    if rdzv == "c10d":
        # the modern, elastic path: agents meet at a key-value store
        cmd += ["--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{port}",
                f"--rdzv-id={job_id}"]
        if rdzv_timeout:
            # how long the agents wait at the meeting point before giving up
            cmd.append(f"--rdzv-conf=timeout={rdzv_timeout}")
    else:
        # the older static path: you name the master yourself, and --node-rank
        # is what decides who is who
        cmd += [f"--master-addr=127.0.0.1", f"--master-port={port}"]
    return cmd + [os.path.join(HERE, "train.py")] + script_args


def run_two_nodes(tag, nproc=2, extra=(), max_restarts=0, timeout=120,
                  nnodes_declared=2, launch_nodes=(0, 1), port=None,
                  rdzv_timeout=None, rdzv="c10d"):
    """Start one torchrun per 'node' and wait for both."""
    work = os.path.join(WORK, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    port = port or D.free_port()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")

    procs, logs = [], {}
    for node in launch_nodes:
        log_path = os.path.join(OUT, f"{tag}_node{node}.log")
        logs[node] = log_path
        f = open(log_path, "w")
        cmd = torchrun_cmd(node, nnodes_declared, nproc, port, f"job-{tag}",
                           ["--out", work] + list(extra), max_restarts, rdzv_timeout,
                           rdzv)
        procs.append((node, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                             env=env), f))
    t0 = time.time()
    codes = {}
    for node, p, f in procs:
        try:
            codes[node] = p.wait(timeout=max(5, timeout - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
            codes[node] = "TIMEOUT"
        f.close()

    ranks = {}
    for path in sorted(glob.glob(os.path.join(work, "rank*.json"))):
        with open(path) as fh:
            d = json.load(fh)
            ranks[int(d["RANK"])] = d
    return {"codes": codes, "ranks": ranks, "secs": time.time() - t0,
            "logs": logs, "work": work, "cmd": " ".join(
                torchrun_cmd(0, nnodes_declared, nproc, port, f"job-{tag}",
                             ["--out", work] + list(extra), max_restarts,
                             rdzv_timeout, rdzv))}


def tail(path, n=8, keep=None):
    if not os.path.exists(path):
        return "(no log)"
    lines = [ln.rstrip() for ln in open(path, errors="replace") if ln.strip()]
    if keep:
        lines = [ln for ln in lines if keep in ln] or lines
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# 1 + 2
# ---------------------------------------------------------------------------

def section_1_2():
    print("\n[1] two torchrun commands, one 4-process job")
    r = run_two_nodes("basic", nproc=2, extra=["--checkpoint-mode", "rank0"])
    record("launch", "exit codes (node 0, node 1)",
           f"{r['codes'].get(0)}, {r['codes'].get(1)}")
    record("launch", "processes that reported in", len(r["ranks"]))
    print("\n    RANK  LOCAL_RANK  GROUP_RANK  WORLD_SIZE  LOCAL_WORLD_SIZE  MASTER")
    for rk in sorted(r["ranks"]):
        d = r["ranks"][rk]
        print(f"    {d['RANK']:>4}  {d['LOCAL_RANK']:>10}  {d['GROUP_RANK']:>10}  "
              f"{d['WORLD_SIZE']:>10}  {d['LOCAL_WORLD_SIZE']:>16}  "
              f"{d['MASTER_ADDR']}:{d['MASTER_PORT']}")
    FINDINGS.append({"section": "launch", "name": "rank table",
                     "value": json.dumps({k: {kk: v[kk] for kk in
                                              ("RANK", "LOCAL_RANK", "GROUP_RANK",
                                               "WORLD_SIZE", "LOCAL_WORLD_SIZE")}
                                          for k, v in r["ranks"].items()}),
                     "note": ""})
    wsums = {rk: d["wsum"] for rk, d in r["ranks"].items()}
    spread = max(wsums.values()) - min(wsums.values()) if wsums else float("nan")
    record("launch", "max spread of weight sums across the 4 processes",
           f"{spread:.3e}", "the two 'nodes' really are training one model")

    print("\n[2] who writes the checkpoint?")
    every = run_two_nodes("ckpt_everyone", nproc=2,
                          extra=["--checkpoint-mode", "everyone"])
    n_writers_every = sum(1 for d in every["ranks"].values() if d["wrote_checkpoint"])
    one = run_two_nodes("ckpt_rank0", nproc=2, extra=["--checkpoint-mode", "rank0"])
    n_writers_one = sum(1 for d in one["ranks"].values() if d["wrote_checkpoint"])
    local = run_two_nodes("ckpt_local", nproc=2,
                          extra=["--checkpoint-mode", "local_rank0"])
    record("ckpt", "processes writing to ONE path (mode=everyone)", n_writers_every,
           "4 concurrent writers, same file - last one wins, or a torn file")
    ck = os.path.join(every["work"], "checkpoint.pt")
    who = torch.load(ck, weights_only=False)["by"] if os.path.exists(ck) else "-"
    record("ckpt", "  which rank's copy actually survived", who)
    record("ckpt", "processes writing (mode=rank0)", n_writers_one)
    record("ckpt", "files written (mode=local_rank0, one per node)",
           len(glob.glob(os.path.join(local["work"], "checkpoint_node*.pt"))),
           "the right pattern for per-node caches, not for checkpoints")
    return r


# ---------------------------------------------------------------------------
# 3. rendezvous failures
# ---------------------------------------------------------------------------

def section_3():
    print("\n[3] rendezvous failures")
    missing = run_two_nodes("missing_node", nproc=2, launch_nodes=(0,), timeout=60,
                            rdzv_timeout=25)
    record("rdzv", "declared --nnodes=2, launched only node 0",
           f"exit {missing['codes'].get(0)} after {missing['secs']:.0f}s",
           "the agent waits at the meeting point; nothing tells you who is late")
    err = tail(missing["logs"][0], 3, keep="Error")
    record("rdzv", "  the message",
           err.splitlines()[-1][:170] if err.strip() else "(still waiting when killed)")

    dup = run_two_nodes("dup_rank", nproc=2, launch_nodes=(0, 0), timeout=90)
    groups = sorted(d["GROUP_RANK"] for d in dup["ranks"].values())
    record("rdzv", "both machines launched with --node-rank=0",
           f"exit {dup['codes'].get(0)}, ranks reported: {len(dup['ranks'])}")
    record("rdzv", "  GROUP_RANK each process actually got", groups,
           "with --rdzv-backend=c10d, node ranks are assigned AT the rendezvous "
           "and --node-rank is ignored; it only matters for static rendezvous")

    # an endpoint nobody is listening on
    work = os.path.join(WORK, "bad_endpoint")
    os.makedirs(work, exist_ok=True)
    log = os.path.join(OUT, "bad_endpoint.log")
    with open(log, "w") as f:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=2",
               "--node-rank=1", "--nproc-per-node=1", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:1", "--rdzv-id=nope",
               "--rdzv-conf=timeout=20",
               os.path.join(HERE, "train.py"), "--out", work]
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        try:
            code = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); code = "TIMEOUT"
    record("rdzv", "unreachable rendezvous endpoint (port 1)",
           f"exit {code} after {time.time() - t0:.0f}s")
    wanted = ("Timed out", "Connection refused", "RendezvousConnectionError",
              "timed out")
    lines = [ln.strip() for ln in open(log, errors="replace")
             if any(w in ln for w in wanted) and "frame #" not in ln]
    record("rdzv", "  the message", lines[-1][:170] if lines else "-")
    return missing, dup


# ---------------------------------------------------------------------------
# 4. restarts
# ---------------------------------------------------------------------------

def section_4():
    print("\n[4] a worker dies at step 3")
    # One agent, 2 workers. (Four workers did not recover on this machine even
    # with the static rendezvous - see the README.)
    no_restart = run_two_nodes("crash_norestart", nproc=2,
                               extra=["--crash-at", "3", "--crash-rank", "1"],
                               max_restarts=0, timeout=90,
                               nnodes_declared=1, launch_nodes=(0,), rdzv="static")
    record("restart", "1 agent x 2 workers, --max-restarts=0: exit code",
           f"{no_restart['codes'].get(0)}")
    record("restart", "  workers that finished", len(no_restart["ranks"]),
           "rank 1 died and torchrun tore down both")

    # Restarting is flaky on this machine, and reporting one run would be
    # dishonest either way - so run each backend three times and count.
    trials = {}
    for rdzv in ("static", "c10d"):
        ok, seen, secs = 0, [], []
        for i in range(3):
            r = run_two_nodes(f"crash_restart_{rdzv}_{i}", nproc=2,
                              extra=["--crash-at", "3", "--crash-rank", "1"],
                              max_restarts=2, timeout=120,
                              nnodes_declared=1, launch_nodes=(0,), rdzv=rdzv)
            secs.append(r["secs"])
            if len(r["ranks"]) == 2:
                ok += 1
                seen.append({rk: d["restart_count"] for rk, d in r["ranks"].items()})
        trials[rdzv] = {"ok": ok, "seen": seen, "secs": secs}
        record("restart", f"--max-restarts=2, rdzv={rdzv}: runs that recovered",
               f"{ok} of 3", f"seconds: " + ", ".join(f"{x:.0f}" for x in secs))
    with_restart = None
    for rdzv in ("static", "c10d"):
        if trials[rdzv]["seen"]:
            record("restart", f"  TORCHELASTIC_RESTART_COUNT after recovery ({rdzv})",
                   trials[rdzv]["seen"][0],
                   "every worker restarts from scratch, not just the one that died - "
                   "which is why a restart is only useful with a checkpoint to reload")
            break
    record("restart", "recovered, 2 workers, both rendezvous backends",
           f"static {trials['static']['ok']}/3, c10d {trials['c10d']['ok']}/3",
           "at 4 workers on this machine the replacement group never finished "
           "its second rendezvous and the job hung instead - a restart re-runs "
           "the rendezvous, so it inherits every way a rendezvous can fail")
    return no_restart, trials


# ---------------------------------------------------------------------------
# 5. the link
# ---------------------------------------------------------------------------

def w_pingpong(rank, world, sizes):
    """Measure the time of an all-reduce as a function of message size.

    The classic model of a message-passing link is  T = alpha + D / B :
    a fixed cost alpha (latency: handshakes, syscalls, wake-ups) plus the bytes
    D divided by the bandwidth B. Fitting a straight line through measured
    (D, T) points recovers both numbers.
    """
    out = {}
    for n in sizes:
        t = torch.zeros(n)
        for _ in range(3):
            dist.all_reduce(t)
        reps = 20 if n <= 1 << 16 else 5
        best = float("inf")
        for _ in range(5):               # 5 batches, keep the fastest
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                dist.all_reduce(t)
            dist.barrier()
            best = min(best, (time.perf_counter() - t0) / reps)
        out[n] = best                    # min, not mean: interference only adds
    return {"times": out, "iface": os.environ.get("GLOO_SOCKET_IFNAME", "(default)")}


def fit_alpha_beta(sizes, times):
    """Recover alpha (latency) and B (bandwidth) from measured all-reduce times.

    Not a least-squares fit over all the points: the message sizes span six
    orders of magnitude, so a plain fit is dominated by the biggest messages and
    reports a nonsense latency. Instead we read the two regimes off directly.
    The smallest message is essentially pure overhead, so its time IS alpha; the
    slope between the two largest messages is essentially pure bandwidth.
    """
    ordered = sorted(sizes)
    alpha = times[ordered[0]]
    big, prev = ordered[-1], ordered[-2]
    slope = (times[big] - times[prev]) / ((big - prev) * 4)
    return alpha, (1 / slope if slope > 0 else float("inf"))


def interfaces():
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1] not in names:
            names.append(parts[1])
    return names


def section_5():
    print("\n[5] what does the link between the two 'nodes' actually cost?")
    sizes = [1, 1 << 8, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]
    res_lo = D.launch(w_pingpong, 2, threads=1, args=(sizes,),
                      env={"GLOO_SOCKET_IFNAME": "lo"})[0]
    ifs = [i for i in interfaces() if i != "lo"]
    res_nic = None
    if ifs:
        try:
            res_nic = D.launch(w_pingpong, 2, threads=1, args=(sizes,),
                               env={"GLOO_SOCKET_IFNAME": ifs[0]})[0]
        except RuntimeError:
            res_nic = None

    a_lo, b_lo = fit_alpha_beta(sizes, res_lo["times"])
    record("link", "loopback: latency alpha", f"{a_lo * 1e6:.1f} us")
    record("link", "loopback: bandwidth B", f"{b_lo / 1e9:.2f} GB/s")
    record("link", "loopback: 4 bytes / 4 MB all-reduce",
           f"{res_lo['times'][1] * 1e6:.0f} us / {res_lo['times'][1 << 20] * 1e3:.1f} ms")
    if res_nic:
        a_n, b_n = fit_alpha_beta(sizes, res_nic["times"])
        record("link", f"via interface {ifs[0]}: latency / bandwidth",
               f"{a_n * 1e6:.1f} us / {b_n / 1e9:.2f} GB/s",
               "same host, so Linux still routes it through memory")
        record("link", "ratio to loopback", f"{a_n / a_lo:.2f}x latency, "
               f"{b_lo / b_n:.2f}x bandwidth",
               "a two-node test on one box does NOT test the network")

    # what the model predicts for real links
    print("\n    predicted time to all-reduce 100M parameters (400 MB), 8 ranks:")
    for name, alpha, bw in (("this box (measured)", a_lo, b_lo),
                            ("100 Gb/s InfiniBand", 2e-6, 12.5e9),
                            ("10 Gb/s ethernet", 30e-6, 1.25e9),
                            ("1 Gb/s ethernet", 100e-6, 0.125e9)):
        D_bytes = 400e6
        n_chunks = 2 * (8 - 1)
        t = n_chunks * (alpha + (D_bytes / 8) / bw)
        record("link", f"  {name}", f"{t * 1e3:.0f} ms per step of pure communication")
    return res_lo, res_nic, sizes


# ---------------------------------------------------------------------------
# 6. hierarchical all-reduce
# ---------------------------------------------------------------------------

def w_hier(rank, world, numel, procs_per_node):
    """Two-level all-reduce: inside each node first, then between nodes.

    A flat all-reduce over 8 ranks on 2 machines sends 2*(8-1)/8 = 1.75 D per
    rank, and most of those messages cross the slow link. Reducing inside each
    node first means only ONE rank per node talks across the link, so the slow
    link carries 2*(2-1)/2 = 1 D once per node instead of per rank. NCCL does
    this automatically; here we build it by hand out of subgroups.
    """
    n_nodes = world // procs_per_node
    node_id = rank // procs_per_node
    local_id = rank % procs_per_node

    local_groups = [dist.new_group(list(range(n * procs_per_node,
                                              (n + 1) * procs_per_node)))
                    for n in range(n_nodes)]
    leader_group = dist.new_group([n * procs_per_node for n in range(n_nodes)])

    t = torch.ones(numel) * (rank + 1)
    flat = t.clone()

    def do_flat():
        dist.all_reduce(flat)

    def do_hier():
        dist.all_reduce(t, group=local_groups[node_id])       # inside the node
        if local_id == 0:
            dist.all_reduce(t, group=leader_group)            # across the link
        dist.broadcast(t, src=node_id * procs_per_node, group=local_groups[node_id])

    do_flat(); do_hier()
    expect = world * (world + 1) / 2
    err = abs(float(t[0]) - expect)

    res = D.interleaved({"flat": do_flat, "hier": do_hier}, rounds=5, warmup=1)
    cross_flat = 2 * (world - 1) / world * numel * 4
    cross_hier = 2 * (n_nodes - 1) / n_nodes * numel * 4 / procs_per_node
    return {"err": err, "flat": res["flat"]["min"], "hier": res["hier"]["min"],
            "cross_flat": cross_flat, "cross_hier": cross_hier}


def section_6():
    print("\n[6] hierarchical all-reduce")
    numel = 1 << 20
    r = D.launch(w_hier, 4, threads=1, args=(numel, 2))[0]
    record("hier", "result correct?", f"max error {r['err']:.1e}")
    record("hier", "bytes crossing the link per rank, flat all-reduce",
           D.fmt_bytes(r["cross_flat"]))
    record("hier", "bytes crossing the link per rank, hierarchical",
           D.fmt_bytes(r["cross_hier"]) +
           f"   ({r['cross_flat'] / r['cross_hier']:.1f}x less)")
    record("hier", "measured time flat / hierarchical (one box, fast link)",
           f"{r['flat'] * 1e3:.2f} ms / {r['hier'] * 1e3:.2f} ms",
           "no win here because there is no slow link to avoid")
    return r


def figure(res_lo, res_nic, sizes, hier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    xs = [n * 4 for n in sizes]
    ax[0].loglog(xs, [res_lo["times"][n] * 1e6 for n in sizes], "o-", label="loopback")
    if res_nic:
        ax[0].loglog(xs, [res_nic["times"][n] * 1e6 for n in sizes], "s-",
                     label="via the NIC's address")
    a, b = fit_alpha_beta(sizes, res_lo["times"])
    ax[0].loglog(xs, [(a + x / b) * 1e6 for x in xs], "k--",
                 label=f"fit: {a * 1e6:.0f} us + D/{b / 1e9:.1f} GB/s")
    ax[0].set_xlabel("bytes all-reduced"); ax[0].set_ylabel("microseconds")
    ax[0].set_title("Latency floor, then bandwidth")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    links = ["this box", "100Gb IB", "10Gb eth", "1Gb eth"]
    params = [1e6, 1e7, 1e8]
    for p in params:
        vals = []
        for alpha, bw in ((a, b), (2e-6, 12.5e9), (30e-6, 1.25e9), (100e-6, .125e9)):
            vals.append(2 * 7 * (alpha + (p * 4 / 8) / bw) * 1e3)
        ax[1].plot(links, vals, "o-", label=f"{p / 1e6:.0f}M params")
    ax[1].set_yscale("log"); ax[1].set_ylabel("ms of communication per step")
    ax[1].set_title("Same code, four networks")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    n = ["flat\nall-reduce", "hierarchical\n(node, then link)"]
    v = [hier["cross_flat"] / 1e6, hier["cross_hier"] / 1e6]
    ax[2].bar(n, v, color=["#7f8c8d", "#27ae60"])
    ax[2].set_ylabel("MB per rank crossing the slow link")
    ax[2].set_title("What the link has to carry")
    for i, val in enumerate(v):
        ax[2].text(i, val, f"{val:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    p = os.path.join(OUT, "multinode.png")
    fig.savefig(p, dpi=120)
    print(f"\n  figure -> {p}")


def main():
    t0 = time.time()
    basic = section_1_2()
    section_3()
    section_4()
    res_lo, res_nic, sizes = section_5()
    hier = section_6()
    figure(res_lo, res_nic, sizes, hier)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader(); w.writerows(FINDINGS)
    with open(os.path.join(OUT, "launch_command.txt"), "w") as f:
        f.write(basic["cmd"] + "\n")
    print(f"\ndone in {time.time() - t0:.0f}s -> outputs/findings.csv")


if __name__ == "__main__":
    main()
