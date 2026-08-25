"""Project 46 - the acceptance test you run on a machine you just built.

Sections
  A. link state          - what nvidia-smi reports about the slot, idle vs busy
  B. transfer bandwidth  - pageable vs pinned, host->device and device->host
  C. full duplex         - can both directions run at once?
  D. zero copy           - a kernel reading host RAM straight over the link
  E. all-reduce          - the nccl-tests measurement, done over gloo, with the
                           algbw/busbw arithmetic spelled out
  F. the verdict         - measured GB/s -> "which link do I actually have?"

Everything in A-D is measured with linkprobe.cu on this machine's GTX 1070 Ti.
Section E runs 2 processes over the loopback interface because this machine has
one GPU and no usable NCCL; the *method* is the point, and the modelled PCIe
numbers next to it come from B.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
P45 = os.path.abspath(os.path.join(HERE, "..", "45-2-gpu-build-plan"))
sys.path.insert(0, P45)

import riglib  # noqa: E402

PROBE = os.path.join(OUT, "linkprobe")
LOAD = os.path.join(P45, "outputs", "gpuload")

# PCIe theoretical bandwidth, GB/s per direction, after 128b/130b encoding
# (Gen3+): lanes x transfer rate x 128/130 / 8.
PCIE = {
    ("3.0", 4): 3.94, ("3.0", 8): 7.88, ("3.0", 16): 15.75,
    ("4.0", 4): 7.88, ("4.0", 8): 15.75, ("4.0", 16): 31.5,
    ("5.0", 8): 31.5, ("5.0", 16): 63.0,
}


def run_probe():
    riglib.build_cu(os.path.join(HERE, "linkprobe.cu"), PROBE)
    p = subprocess.run([PROBE], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("linkprobe failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
    d = {"h2d": [], "d2h": []}
    for line in p.stdout.strip().splitlines():
        f = line.split(",")
        if f[0] == "dev":
            d["device"] = dict(name=f[1], cc=f[2], sms=int(f[3]),
                               core_mhz=int(f[4]), mem_mhz=int(f[5]),
                               bus_bits=int(f[6]), vram_mb=int(f[7]))
        elif f[0] in ("h2d", "d2h"):
            d[f[0]].append(dict(bytes=int(f[1]), pageable_gbs=float(f[2]),
                                pinned_gbs=float(f[3])))
        elif f[0] == "duplex":
            d["duplex"] = dict(bytes=int(f[1]), h2d_gbs=float(f[2]),
                               d2h_gbs=float(f[3]), both_gbs=float(f[4]))
        elif f[0] == "zerocopy":
            d["zerocopy"] = dict(bytes=int(f[1]), device_gbs=float(f[2]),
                                 mapped_gbs=float(f[3]))
        elif f[0] == "dram":
            d["dram_gbs"] = float(f[2])
        elif f[0] == "lat":
            d["latency"] = dict(h2d_4b_us=float(f[1]), launch_us=float(f[2]))
    return d


def link_state():
    """What the driver reports about the slot when the GPU is asleep, and
    what it reports two seconds into a real load."""
    idle = riglib.smi()
    proc = subprocess.Popen([LOAD, "compute", "6"], stdout=subprocess.DEVNULL)
    time.sleep(2.5)
    busy = riglib.smi()
    proc.wait()
    time.sleep(1.0)
    keys = ["pcie.link.gen.current", "pcie.link.width.current", "pstate",
            "clocks.sm", "power.draw"]
    return dict(idle={k: idle[k] for k in keys},
                busy={k: busy[k] for k in keys})


# ------------------------------------------------------------------ E
ALLREDUCE_WORKER = r'''
import os, sys, time, json
import torch, torch.distributed as dist

rank = int(sys.argv[1]); world = int(sys.argv[2]); out = sys.argv[3]
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
torch.set_num_threads(2)
dist.init_process_group("gloo", rank=rank, world_size=world)
res = []
for mb in [1, 4, 16, 64, 256]:
    n = mb * 1024 * 1024 // 4
    x = torch.ones(n, dtype=torch.float32)
    for _ in range(2):                       # warm-up
        dist.all_reduce(x); dist.barrier()
    reps = 5 if mb <= 64 else 3
    dist.barrier(); t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x)
    dist.barrier(); t1 = time.perf_counter()
    sec = (t1 - t0) / reps
    # correctness: all-reducing a vector of ones must give exactly `world`
    x.fill_(1.0); dist.all_reduce(x)
    ok = bool(torch.all(x == float(world)))
    res.append(dict(mb=mb, bytes=n * 4, sec=sec, correct=ok))
if rank == 0:
    json.dump(res, open(out, "w"))
dist.destroy_process_group()
'''


def allreduce_bench(world=2):
    """nccl-tests, in miniature, over whatever transport we do have.

    Reports both numbers nccl-tests prints, because the difference between
    them is the single most misread thing in multi-GPU benchmarking:
      algbw = bytes / time                    (what your training loop feels)
      busbw = algbw x 2(N-1)/N                (what each wire actually carries)
    """
    worker = os.path.join(OUT, "_ar_worker.py")
    with open(worker, "w") as f:
        f.write(ALLREDUCE_WORKER)
    res_path = os.path.join(OUT, "_ar.json")
    env = dict(os.environ, MASTER_PORT=str(29517 + world))
    procs = [subprocess.Popen([sys.executable, worker, str(r), str(world),
                               res_path], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=env)
             for r in range(world)]
    for p in procs:
        p.wait(timeout=600)
    rows = json.load(open(res_path))
    for r in rows:
        r["algbw_gbs"] = r["bytes"] / r["sec"] / 1e9
        r["busbw_gbs"] = r["algbw_gbs"] * 2 * (world - 1) / world
    os.remove(res_path); os.remove(worker)
    return rows


def verdict(measured_pinned_gbs, measured_pageable_gbs):
    """Given a measured number, which link is it consistent with? Assume a
    healthy link delivers 75-85% of theory."""
    rows = []
    for (gen, width), theory in sorted(PCIE.items()):
        rows.append(dict(link=f"PCIe {gen} x{width}", theory_gbs=theory,
                         healthy_lo=0.75 * theory, healthy_hi=0.90 * theory,
                         matches_pinned=0.75 * theory <= measured_pinned_gbs
                         <= 0.90 * theory,
                         matches_pageable=0.75 * theory <= measured_pageable_gbs
                         <= 0.90 * theory))
    return rows


def main():
    findings = {}

    print("== A. what the driver says about the slot ==")
    findings["link_state"] = link_state()
    ls = findings["link_state"]
    print(f"   idle: Gen{ls['idle']['pcie.link.gen.current']} "
          f"x{ls['idle']['pcie.link.width.current']}, "
          f"{ls['idle']['pstate']}, {ls['idle']['power.draw']} W")
    print(f"   busy: Gen{ls['busy']['pcie.link.gen.current']} "
          f"x{ls['busy']['pcie.link.width.current']}, "
          f"{ls['busy']['pstate']}, {ls['busy']['power.draw']} W")

    print("== B/C/D. the link, measured ==")
    probe = run_probe()
    findings["probe"] = probe
    big = [r for r in probe["h2d"] if r["bytes"] >= 1 << 24]
    h2d_pin = max(r["pinned_gbs"] for r in big)
    h2d_page = max(r["pageable_gbs"] for r in big)
    d2h_pin = max(r["pinned_gbs"] for r in probe["d2h"] if r["bytes"] >= 1 << 24)
    findings["summary"] = dict(
        h2d_pinned=h2d_pin, h2d_pageable=h2d_page, d2h_pinned=d2h_pin,
        pin_speedup=h2d_pin / h2d_page,
        theory_gbs=PCIE[("3.0", 16)],
        pinned_pct_of_theory=100 * h2d_pin / PCIE[("3.0", 16)],
        pageable_pct_of_theory=100 * h2d_page / PCIE[("3.0", 16)],
        duplex_sum=probe["duplex"]["h2d_gbs"] + probe["duplex"]["d2h_gbs"],
        duplex_both=probe["duplex"]["both_gbs"],
        duplex_efficiency=probe["duplex"]["both_gbs"]
        / (probe["duplex"]["h2d_gbs"] + probe["duplex"]["d2h_gbs"]),
        zerocopy_penalty=probe["zerocopy"]["device_gbs"]
        / probe["zerocopy"]["mapped_gbs"],
        dram_gbs=probe["dram_gbs"],
        dram_over_link=probe["dram_gbs"] / h2d_pin,
    )
    s = findings["summary"]
    print(f"   H2D pinned {h2d_pin:.2f} GB/s ({s['pinned_pct_of_theory']:.0f}% "
          f"of PCIe 3.0 x16), pageable {h2d_page:.2f} GB/s "
          f"({s['pageable_pct_of_theory']:.0f}%) -> pinning is "
          f"{s['pin_speedup']:.2f}x")
    print(f"   duplex: {probe['duplex']['h2d_gbs']:.1f} up + "
          f"{probe['duplex']['d2h_gbs']:.1f} down separately, "
          f"{probe['duplex']['both_gbs']:.1f} together "
          f"({s['duplex_efficiency']*100:.0f}% of the sum)")
    print(f"   zero-copy: kernel reads device DRAM at "
          f"{probe['zerocopy']['device_gbs']:.1f} GB/s but host RAM at "
          f"{probe['zerocopy']['mapped_gbs']:.1f} GB/s "
          f"({s['zerocopy_penalty']:.1f}x slower)")
    print(f"   latency: {probe['latency']['h2d_4b_us']:.2f} us for a 4-byte "
          f"copy, {probe['latency']['launch_us']:.2f} us to launch a kernel")

    print("== E. all-reduce ==")
    findings["allreduce"] = {}
    for world in (2, 4):
        findings["allreduce"][world] = allreduce_bench(world)
        for r in findings["allreduce"][world]:
            print(f"   N={world} {r['mb']:>4} MB: {r['sec']*1000:8.2f} ms  "
                  f"algbw {r['algbw_gbs']:6.2f} GB/s  busbw {r['busbw_gbs']:6.2f}"
                  f" GB/s  correct={r['correct']}")

    print("== F. so which link do I have? ==")
    findings["verdict"] = verdict(h2d_pin, h2d_page)
    for r in findings["verdict"]:
        if r["matches_pinned"] or r["matches_pageable"]:
            who = []
            if r["matches_pinned"]:
                who.append("the pinned measurement")
            if r["matches_pageable"]:
                who.append("the PAGEABLE measurement")
            print(f"   {r['link']:>14} (theory {r['theory_gbs']:5.2f} GB/s): "
                  f"consistent with {' and '.join(who)}")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, acceptance.png")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        for k, v in f["summary"].items():
            w.writerow(["B/C/D link", k, round(v, 3), "", "measured"])
        for k in ("idle", "busy"):
            for kk, vv in f["link_state"][k].items():
                w.writerow(["A link state", f"{k} {kk}", vv, "", "measured"])
        for r in f["probe"]["h2d"]:
            w.writerow(["B h2d", r["bytes"], round(r["pinned_gbs"], 3),
                        "GB/s pinned", "measured"])
            w.writerow(["B h2d", r["bytes"], round(r["pageable_gbs"], 3),
                        "GB/s pageable", "measured"])
        for world, rows in f["allreduce"].items():
            for r in rows:
                w.writerow([f"E allreduce N={world}", f"{r['mb']} MB",
                            round(r["busbw_gbs"], 3), "GB/s busbw", "measured"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    xs = [r["bytes"] / 2**20 for r in f["probe"]["h2d"]]
    a.plot(xs, [r["pinned_gbs"] for r in f["probe"]["h2d"]], "o-",
           label="H2D pinned", color="#27ae60")
    a.plot(xs, [r["pageable_gbs"] for r in f["probe"]["h2d"]], "o-",
           label="H2D pageable", color="#c0392b")
    a.plot(xs, [r["pinned_gbs"] for r in f["probe"]["d2h"]], "s--",
           label="D2H pinned", color="#16a085")
    a.axhline(15.75, ls="--", lw=1, color="#7f8c8d")
    a.text(0.005, 15.9, "PCIe 3.0 x16 theory 15.75 GB/s", fontsize=7)
    a.axhline(7.88, ls=":", lw=1, color="#7f8c8d")
    a.text(0.005, 8.0, "PCIe 3.0 x8 theory 7.88 GB/s", fontsize=7)
    a.set_xscale("log"); a.set_xlabel("transfer size (MiB)")
    a.set_ylabel("GB/s"); a.set_ylim(0, 18)
    a.set_title("B. pinned vs pageable", fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1]
    s = f["summary"]
    names = ["H2D\npageable", "H2D\npinned", "duplex\nboth ways",
             "zero-copy\nkernel read", "device\nDRAM"]
    vals = [s["h2d_pageable"], s["h2d_pinned"], s["duplex_both"],
            f["probe"]["zerocopy"]["mapped_gbs"], s["dram_gbs"]]
    cols = ["#c0392b", "#27ae60", "#2980b9", "#8e44ad", "#34495e"]
    a.bar(names, vals, color=cols)
    a.set_yscale("log"); a.set_ylabel("GB/s (log)")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    a.set_title(f"C/D. the link is {s['dram_over_link']:.1f}x slower than DRAM",
                fontsize=10)
    a.grid(alpha=.3, axis="y")

    a = ax[2]
    for world, (c1, c2) in zip(sorted(f["allreduce"]),
                               [("#e67e22", "#2c3e50"), ("#f1c40f", "#8e44ad")]):
        ar = f["allreduce"][world]
        a.plot([r["mb"] for r in ar], [r["algbw_gbs"] for r in ar], "o-",
               label=f"N={world} algbw (what the app feels)", color=c1)
        a.plot([r["mb"] for r in ar], [r["busbw_gbs"] for r in ar], "s--",
               label=f"N={world} busbw (what the wire carries)", color=c2)
    a.set_xscale("log", base=2); a.set_xlabel("message size (MiB)")
    a.set_ylabel("GB/s")
    a.set_title("E. all-reduce over loopback (gloo)", fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Project 46 - accepting a freshly built machine", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "acceptance.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
