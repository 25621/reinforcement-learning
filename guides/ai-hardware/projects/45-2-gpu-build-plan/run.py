"""Project 45 - specify a multi-GPU workstation, and check the spec against
measurements taken on a real card instead of against spec sheets.

Sections
  A. what one real GPU actually does      (measured: FP32 peak, DRAM bandwidth)
  B. the power transient                  (measured: idle -> full load -> idle)
  C. does the model fit?                  (arithmetic: weights + KV + workspace)
  D. the planner                          (search over a component catalogue
                                           under budget / PSU / wall-circuit /
                                           PCIe-lane / slot constraints)
  E. do the lanes matter?                 (arithmetic on measured link efficiency)

Everything measured comes from this machine's GTX 1070 Ti; everything about
parts we do not own is arithmetic and is labelled as such in findings.json.
"""

import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import riglib  # noqa: E402

LOAD_EXE = os.path.join(OUT, "gpuload")

# Measured in project 46 on this machine: pinned host-to-device transfers reach
# 12.72 GB/s on a PCIe 3.0 x16 link whose theoretical ceiling is 15.75 GB/s.
LINK_EFFICIENCY = 0.80

# ---------------------------------------------------------------- catalogue
# Spec-sheet numbers for parts we do not own: dense FP16 tensor TFLOPs (no
# sparsity), GB of VRAM, GB/s of memory bandwidth, board watts, street price in
# USD (new or typical used, 2026). Prices move; the *shape* of the answer does
# not.
GPUS = [
    # name,                vram, tflops, bw,   watts, price, slots, note
    ("RTX 5080",             16,   225,   960,   360,  1000, 2, "new"),
    ("RTX 4090 (used)",      24,   165,  1008,   450,  1600, 3, "used"),
    ("RTX 5090",             32,   419,  1792,   575,  2200, 3, "new"),
    ("RTX A6000 (used)",     48,   155,   768,   300,  4000, 2, "used"),
    ("L40S",                 48,   181,   864,   350,  8000, 2, "new"),
    ("RTX 6000 Ada",         48,   182,   960,   300,  6800, 2, "new"),
    ("RTX PRO 6000 Blackwell", 96, 503,  1792,   600,  8500, 2, "new"),
    ("A100 80GB (used)",     80,   312,  2039,   400, 12000, 2, "used"),
    ("H100 PCIe (used)",     80,   756,  2000,   350, 22000, 2, "used"),
]
GPU_KEYS = ["name", "vram_gb", "tflops_fp16", "bw_gbs", "watts", "price_usd",
            "slots", "market"]

HOST_WATTS = 200        # CPU + board + drives + fans, under load
HOST_PRICE = 3500       # Threadripper-class host: CPU, board, 128 GB, NVMe, case
PSU_OPTIONS = [(850, 130), (1200, 200), (1600, 400), (2000, 650)]   # W, price


def gpu_dict(row):
    return dict(zip(GPU_KEYS, row))


# ------------------------------------------------------------------ A + B
def build_and_measure():
    """Compile the load kernel, measure peak FP32 and DRAM bandwidth, then
    record the power transient of an idle -> load -> idle cycle."""
    cmd = riglib.build_cu(os.path.join(HERE, "gpuload.cu"), LOAD_EXE)
    # record the command without this checkout's absolute paths
    cmd = re.sub(r"\S*/([\w.]+)(?=\s|$)", r"\1", cmd)
    name, plimit, pmax = riglib.gpu_name()

    def run_load(mode, seconds, extra=()):
        p = subprocess.run([LOAD_EXE, mode, str(seconds), *map(str, extra)],
                           capture_output=True, text=True)
        rows = []
        for line in p.stdout.strip().splitlines():
            f = line.split(",")
            if f[0] == "sample":
                rows.append(dict(t=float(f[1]), dt=float(f[2]),
                                 value=float(f[3]), tag=f[4]))
        return rows

    # One continuous telemetry recording, sampled every 100 ms, covering:
    #   5 s cold idle | 10 s FMA load | 8 s idle | 10 s streaming load | 15 s idle
    # so that "idle" is measured before the card has ever been woken up, and
    # the return to idle is measured too.
    s = riglib.Sampler(period=0.1)
    s.start()
    time.sleep(5.0)
    t_c0 = time.time(); comp = run_load("compute", 10); t_c1 = time.time()
    time.sleep(8.0)
    t_m0 = time.time(); mem = run_load("memory", 10); t_m1 = time.time()
    time.sleep(15.0)
    rows = s.stop()

    def win(a, b, skip=1.5):
        return [r for r in rows if a + skip <= r["t"] <= b - 0.2]

    pw = lambda rs: riglib.mean([r["power.draw"] for r in rs])   # noqa: E731
    cold_idle = [r for r in rows if r["t"] < t_c0 - 0.3]
    comp_win, mem_win = win(t_c0, t_c1), win(t_m0, t_m1)
    tail = [r for r in rows if r["t"] > t_m1 + 12.0]

    peak_gflops = max(r["value"] for r in comp[1:])       # skip warm-up second
    peak_gbs = max(r["value"] for r in mem[1:])
    idle_w, load_w = pw(cold_idle), pw(comp_win)
    mem_w = pw(mem_win)
    peak_w = max(r["power.draw"] for r in rows if r["power.draw"] is not None)

    # how long from load start to within 5% of steady state, and how long
    # after the load stops before the card is back within 10% of cold idle
    rise = next((r["t"] - t_c0 for r in rows
                 if r["t"] >= t_c0 and r["power.draw"] >= 0.95 * load_w), None)
    fall = next((r["t"] - t_m1 for r in rows
                 if r["t"] > t_m1 and r["power.draw"] <= 1.10 * idle_w), None)

    return dict(
        nvcc=cmd, gpu=name, power_limit_w=plimit, power_limit_max_w=pmax,
        peak_gflops=peak_gflops, peak_dram_gbs=peak_gbs,
        idle_w=idle_w, load_w=load_w, mem_load_w=mem_w,
        peak_sampled_w=peak_w, after_w=pw(tail),
        peak_over_steady=peak_w / load_w if load_w else None,
        mem_over_compute_w=mem_w / load_w if load_w else None,
        headroom_pct=100 * load_w / plimit,
        mem_headroom_pct=100 * mem_w / plimit,
        rise_to_95pct_s=rise, fall_to_idle_s=fall,
        compute_temp_c=riglib.mean([r["temperature.gpu"] for r in comp_win]),
        mem_temp_c=riglib.mean([r["temperature.gpu"] for r in mem_win]),
        compute_mhz=riglib.mean([r["clocks.sm"] for r in comp_win]),
        mem_mhz=riglib.mean([r["clocks.sm"] for r in mem_win]),
        sample_period_s=0.1, n_samples=len(rows),
        marks=dict(compute=[t_c0 - rows[0]["t"], t_c1 - rows[0]["t"]],
                   memory=[t_m0 - rows[0]["t"], t_m1 - rows[0]["t"]]),
        trace=[dict(dt=round(r["dt"], 3), w=r["power.draw"],
                    c=r["temperature.gpu"], mhz=r["clocks.sm"],
                    util=r["utilization.gpu"]) for r in rows],
        compute_series=comp, memory_series=mem,
    )


# ---------------------------------------------------------------------- C
MODELS = [
    # name, params (B), layers, hidden, kv_heads, head_dim
    ("Llama-3-8B",   8.0, 32, 4096,  8, 128),
    ("Qwen3-32B",   32.0, 64, 5120,  8, 128),
    ("Llama-3-70B", 70.0, 80, 8192,  8, 128),
    ("Qwen3-235B-A22B", 235.0, 94, 4096, 4, 128),
]


def vram_math(ctx=8192, concurrency=8):
    """Bytes a model actually needs on the card, split into the three parts
    that grow differently: weights (fixed), KV cache (grows with users and
    context), workspace (roughly fixed per GPU)."""
    rows = []
    for name, params, layers, hidden, kv_heads, head_dim in MODELS:
        for dtype, bpw in [("fp16", 2), ("int8", 1), ("int4", 0.5)]:
            weights = params * 1e9 * bpw / 2**30
            # KV cache is 2 (K and V) x layers x kv_heads x head_dim x bytes,
            # per token. Serving engines keep it in fp16 even when weights are
            # quantized, unless you quantize it separately (project 35).
            kv_per_tok = 2 * layers * kv_heads * head_dim * 2
            kv = kv_per_tok * ctx * concurrency / 2**30
            workspace = 1.5      # CUDA context, activations, allocator slack
            rows.append(dict(model=name, dtype=dtype, weights_gib=weights,
                             kv_gib=kv, workspace_gib=workspace,
                             total_gib=weights + kv + workspace,
                             kv_bytes_per_token=kv_per_tok,
                             ctx=ctx, concurrency=concurrency))
    return rows


def fits(total_gib, gpu, n):
    """A model split over n cards needs total/n on each, plus a per-card
    workspace that does NOT get divided."""
    per_card = total_gib / n + 0.8 * (n - 1) / max(n, 1)
    return per_card <= gpu["vram_gb"] * 0.94      # 6% for driver + fragmentation


# ---------------------------------------------------------------------- D
def plan(target_gib, target_name, budget, circuit_amps=15, volts=120,
         max_gpus=4, need_tflops=None, weights_gib=None):
    """Every build that satisfies every constraint, ranked by TFLOPs per dollar.

    Constraints, in the order they usually bite:
      1. the model must fit across the cards
      2. total price <= budget
      3. PSU must cover 1.15 x (GPU + host) watts, and must exist
      4. the WALL must cover it continuously: a US 15 A / 120 V circuit is
         1800 W, of which code allows 80% continuous = 1440 W
      5. slots: 4 double-width cards do not fit a normal motherboard
    """
    wall_continuous = circuit_amps * volts * 0.80
    out = []
    for row in GPUS:
        g = gpu_dict(row)
        for n in range(1, max_gpus + 1):
            r = dict(gpu=g["name"], n=n, target=target_name)
            r["vram_total"] = g["vram_gb"] * n
            r["fits"] = fits(target_gib, g, n)
            r["gpu_price"] = g["price_usd"] * n
            r["price"] = r["gpu_price"] + HOST_PRICE
            r["gpu_watts"] = g["watts"] * n
            r["load_watts"] = r["gpu_watts"] + HOST_WATTS
            psu = next(((w, p) for w, p in PSU_OPTIONS
                        if w >= 1.15 * r["load_watts"]), None)
            r["psu_w"] = psu[0] if psu else None
            r["price"] += psu[1] if psu else 0
            r["tflops"] = g["tflops_fp16"] * n
            r["bw_gbs"] = g["bw_gbs"] * n
            r["slots"] = g["slots"] * n
            r["wall_ok"] = r["load_watts"] <= wall_continuous
            r["psu_ok"] = psu is not None
            r["budget_ok"] = r["price"] <= budget
            r["slots_ok"] = r["slots"] <= 7
            r["tflops_ok"] = need_tflops is None or r["tflops"] >= need_tflops
            r["ok"] = all([r["fits"], r["budget_ok"], r["psu_ok"], r["wall_ok"],
                           r["slots_ok"], r["tflops_ok"]])
            r["tflops_per_kusd"] = r["tflops"] / (r["price"] / 1000)
            r["gbs_per_kusd"] = r["bw_gbs"] / (r["price"] / 1000)
            # Decode speed is a bandwidth question, not a FLOP question: one
            # token needs every weight read once (project 39). Splitting the
            # model over n cards splits the reading too, so bandwidth adds up.
            if weights_gib:
                # n cards each read their own 1/n of the weights at their own
                # bandwidth, at the same time, so the aggregate is what counts.
                r["decode_tok_s"] = r["bw_gbs"] / weights_gib
                r["tok_s_per_kusd"] = r["decode_tok_s"] / (r["price"] / 1000)
            r["wall_headroom_w"] = wall_continuous - r["load_watts"]
            out.append(r)
    return out, wall_continuous


def circuit_limits():
    """How many of each card a wall socket will carry, continuously.
    Electrical code (US NEC 210.19, and the same idea in IEC practice) says a
    circuit feeding a load for more than 3 hours may be loaded to 80% of its
    rating. A training run is the definition of a continuous load."""
    circuits = [("US 15 A / 120 V", 15, 120), ("US 20 A / 120 V", 20, 120),
                ("US 30 A / 240 V (dryer)", 30, 240),
                ("EU 16 A / 230 V", 16, 230)]
    rows = []
    for cname, amps, volts in circuits:
        cont = amps * volts * 0.80
        for row in GPUS:
            g = gpu_dict(row)
            n = int((cont - HOST_WATTS) // g["watts"])
            rows.append(dict(circuit=cname, continuous_w=cont, gpu=g["name"],
                             watts=g["watts"], max_gpus=max(n, 0)))
    return rows


def why_rejected(rows):
    """For every candidate that failed, the first constraint that killed it -
    this is the part a build post never shows you."""
    order = [("fits", "model does not fit"), ("budget_ok", "over budget"),
             ("psu_ok", "no PSU big enough"), ("wall_ok", "over wall circuit"),
             ("slots_ok", "no slots"), ("tflops_ok", "too slow")]
    tally = {}
    for r in rows:
        if r["ok"]:
            continue
        for key, label in order:
            if not r[key]:
                tally[label] = tally.get(label, 0) + 1
                break
    return tally


# ---------------------------------------------------------------------- E
LINKS = [("PCIe 3.0 x8", 7.88), ("PCIe 3.0 x16", 15.75),
         ("PCIe 4.0 x8", 15.75), ("PCIe 4.0 x16", 31.5),
         ("PCIe 5.0 x16", 63.0), ("NVLink 4 (bridge)", 450.0)]


def lane_math(params_b=8.0, dtype_bytes=2, tflops=419.0, mfu=0.40):
    """Data-parallel training: every step all-reduces one full copy of the
    gradients. Does the PCIe width change the step time enough to care?

    Ring all-reduce moves 2(N-1)/N x bytes over each link, so for N=2 it is
    exactly 1 x the gradient size in each direction.

    The answer depends entirely on how many tokens are in a step, so both ends
    of the range are computed: a full 2M-token optimizer step (gradients are
    accumulated over many micro-batches and all-reduced once) and a bare
    8k-token micro-step (all-reduce after every forward/backward).
    """
    grad_bytes = params_b * 1e9 * dtype_bytes
    rows = []
    for sched, toks in [("2M-token optimizer step", 2 * 1024 * 1024),
                        ("8k-token micro-step", 8192)]:
        # 6 FLOPs per parameter per token for forward+backward; 2 GPUs share it
        compute_s = 6 * params_b * 1e9 * toks / (tflops * 1e12 * mfu * 2)
        for link, gbs in LINKS:
            eff = gbs * LINK_EFFICIENCY * 1e9
            allreduce_s = grad_bytes / eff    # N=2: 1 x gradient each way
            rows.append(dict(schedule=sched, tokens=toks, link=link,
                             spec_gbs=gbs, effective_gbs=eff / 1e9,
                             allreduce_s=allreduce_s, compute_s=compute_s,
                             overhead_pct=100 * allreduce_s / compute_s,
                             step_s=compute_s + allreduce_s))
    # The other job the lanes do: streaming weights for a model that does not
    # fit. Every token needs the whole model pulled across the link.
    offload = []
    for link, gbs in [("PCIe 3.0 x16", 15.75), ("PCIe 4.0 x16", 31.5),
                      ("PCIe 5.0 x16", 63.0)]:
        for name, gib in [("Llama-3-70B int4", 35.0), ("Llama-3-8B fp16", 16.0)]:
            eff = gbs * LINK_EFFICIENCY
            offload.append(dict(link=link, model=name, gib=gib,
                                tok_per_s=eff / gib))
    return rows, offload


def main():
    findings = {}

    print("== A/B. measuring this machine's one GPU ==")
    m = build_and_measure()
    findings["machine"] = m
    print(f"   {m['gpu']}: {m['peak_gflops']/1000:.2f} TFLOP/s FP32, "
          f"{m['peak_dram_gbs']:.0f} GB/s DRAM")
    print(f"   power: idle {m['idle_w']:.1f} W -> FMA load {m['load_w']:.1f} W "
          f"({m['headroom_pct']:.0f}% of the {m['power_limit_w']:.0f} W limit) "
          f"-> streaming load {m['mem_load_w']:.1f} W "
          f"({m['mem_over_compute_w']:.2f}x the FMA load)")
    print(f"   peak sample {m['peak_sampled_w']:.1f} W = "
          f"{m['peak_over_steady']:.2f}x steady; 95% of steady after "
          f"{m['rise_to_95pct_s']:.1f} s")
    print(f"   15 s after the load ends it still draws {m['after_w']:.1f} W = "
          f"{m['after_w']/m['idle_w']:.1f}x cold idle"
          + (f" (never returned to idle in the window)"
             if m["fall_to_idle_s"] is None else
             f" (back to idle after {m['fall_to_idle_s']:.1f} s)"))

    print("== C. VRAM arithmetic ==")
    findings["vram"] = vram_math()
    for r in findings["vram"]:
        if r["model"] == "Llama-3-70B":
            print(f"   70B {r['dtype']}: weights {r['weights_gib']:.1f} + "
                  f"KV {r['kv_gib']:.1f} + ws {r['workspace_gib']:.1f} = "
                  f"{r['total_gib']:.1f} GiB")

    print("== D. the planner ==")
    #  name, VRAM needed (GiB), weights alone (GiB), budget, TFLOPs floor
    targets = [
        ("70B int4 serving, 8 users @ 8k", 32.6 + 20.0 + 1.5, 32.6, 12000, None),
        ("70B fp16 serving, 8 users @ 8k", 130.4 + 20.0 + 1.5, 130.4, 30000, None),
        ("8B fp16 fine-tune", 16.0 + 3.7 + 1.5, 16.0, 12000, 300.0),
    ]
    plans = {}
    for name, gib, wgib, budget, need in targets:
        rows, wall = plan(gib, name, budget, need_tflops=need, weights_gib=wgib)
        ok = [r for r in rows if r["ok"]]
        by_flops = sorted(ok, key=lambda r: -r["tflops_per_kusd"])
        by_toks = sorted(ok, key=lambda r: -r.get("tok_s_per_kusd", 0))
        plans[name] = dict(
            target_gib=gib, weights_gib=wgib, budget=budget, need_tflops=need,
            wall_continuous_w=wall, candidates=rows, n_ok=len(ok),
            winners_flops=by_flops[:5], winners_tokens=by_toks[:5],
            rankings_agree=bool(ok) and by_flops[0]["gpu"] == by_toks[0]["gpu"]
            and by_flops[0]["n"] == by_toks[0]["n"],
            rejected=why_rejected(rows))
        if ok:
            a, b = by_flops[0], by_toks[0]
            print(f"   {name}: {len(ok)} legal builds")
            print(f"      by TFLOPs/$k : {a['n']}x {a['gpu']} ${a['price']:,} "
                  f"({a['tflops_per_kusd']:.0f} TFLOP/$k, "
                  f"{a['decode_tok_s']:.0f} tok/s)")
            print(f"      by tokens/$k : {b['n']}x {b['gpu']} ${b['price']:,} "
                  f"({b['tflops_per_kusd']:.0f} TFLOP/$k, "
                  f"{b['decode_tok_s']:.0f} tok/s)")
        else:
            print(f"   {name}: NO legal build under ${budget:,}")
        print(f"      rejections: {plans[name]['rejected']}")
    findings["plans"] = plans

    print("== D2. the wall socket ==")
    findings["circuits"] = circuit_limits()
    for r in findings["circuits"]:
        if r["gpu"] in ("RTX 5090", "RTX A6000 (used)"):
            print(f"   {r['circuit']:>24} ({r['continuous_w']:.0f} W cont.): "
                  f"{r['max_gpus']}x {r['gpu']}")

    print("== E. lanes ==")
    ddp, offload = lane_math()
    findings["lanes"] = dict(ddp=ddp, offload=offload,
                             link_efficiency=LINK_EFFICIENCY)
    for r in ddp:
        print(f"   {r['schedule']:>24} {r['link']:>18}: all-reduce "
              f"{r['allreduce_s']*1000:7.1f} ms = {r['overhead_pct']:6.2f}% "
              f"of a {r['compute_s']:.2f} s step")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, buildplan.png")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        m = f["machine"]
        for k, u in [("peak_gflops", "GFLOP/s"), ("peak_dram_gbs", "GB/s"),
                     ("idle_w", "W"), ("load_w", "W"), ("peak_sampled_w", "W"),
                     ("peak_over_steady", "x"), ("rise_to_95pct_s", "s")]:
            w.writerow(["A/B machine", k, m[k], u, "measured"])
        for r in f["vram"]:
            w.writerow(["C vram", f"{r['model']} {r['dtype']}",
                        round(r["total_gib"], 2), "GiB", "arithmetic"])
        for name, p in f["plans"].items():
            for r in p["winners_flops"]:
                w.writerow(["D plan (by TFLOPs/$k)", f"{name} | {r['n']}x {r['gpu']}",
                            r["price"], "USD", "arithmetic"])
            for r in p["winners_tokens"]:
                w.writerow(["D plan (by tokens/$k)", f"{name} | {r['n']}x {r['gpu']}",
                            round(r.get("decode_tok_s", 0), 1), "tok/s", "arithmetic"])
        for r in f["circuits"]:
            w.writerow(["D2 circuit", f"{r['circuit']} | {r['gpu']}",
                        r["max_gpus"], "cards", "arithmetic"])
        for r in f["lanes"]["ddp"]:
            w.writerow(["E lanes", f"{r['schedule']} | {r['link']}",
                        round(r["overhead_pct"], 3), "% of step", "arithmetic"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # 1. the power transient
    m = f["machine"]
    tr = m["trace"]
    a = ax[0][0]
    a.plot([r["dt"] for r in tr], [r["w"] for r in tr], lw=1.4, color="#c0392b")
    for lbl, (s0, s1) in m.get("marks", {}).items():
        a.axvspan(s0, s1, color="#f39c12", alpha=.15)
        a.text((s0 + s1) / 2, 4, lbl, ha="center", fontsize=7)
    a.axhline(m["power_limit_w"], ls="--", lw=1, color="#7f8c8d")
    a.text(0.5, m["power_limit_w"], f" board power limit {m['power_limit_w']:.0f} W",
           fontsize=7, va="bottom")
    a.set_xlabel("seconds"); a.set_ylabel("board power (W)")
    a.set_ylim(0, m["power_limit_w"] * 1.15)
    a.set_title(f"A/B. idle {m['idle_w']:.0f} W -> FMA {m['load_w']:.0f} W -> "
                f"stream {m['mem_load_w']:.0f} W", fontsize=9)
    a2 = a.twinx()
    a2.plot([r["dt"] for r in tr], [r["c"] for r in tr], lw=1.0, color="#2980b9")
    a2.set_ylabel("temperature (C)", color="#2980b9")
    a.grid(alpha=.3)

    # 2. VRAM stack for the 70B
    a = ax[0][1]
    rows = [r for r in f["vram"] if r["model"] == "Llama-3-70B"]
    x = range(len(rows))
    a.bar(x, [r["weights_gib"] for r in rows], label="weights", color="#34495e")
    a.bar(x, [r["kv_gib"] for r in rows], bottom=[r["weights_gib"] for r in rows],
          label="KV cache (8 users, 8k ctx)", color="#e67e22")
    a.bar(x, [r["workspace_gib"] for r in rows],
          bottom=[r["weights_gib"] + r["kv_gib"] for r in rows],
          label="workspace", color="#95a5a6")
    a.set_xticks(list(x)); a.set_xticklabels([r["dtype"] for r in rows])
    for cap, style in [(24, ":"), (32, "--"), (48, "-."), (96, "-")]:
        a.axhline(cap, ls=style, lw=1, color="#7f8c8d")
        a.text(2.55, cap, f"{cap} GB", fontsize=7, va="bottom", ha="right")
    a.set_ylabel("GiB"); a.set_title("C. Llama-3-70B on one card")
    a.legend(fontsize=7)

    # 3. the plan: price vs capability, legal builds only
    a = ax[1][0]
    name = list(f["plans"])[0]
    cands = f["plans"][name]["candidates"]
    ok = [r for r in cands if r["ok"]]
    bad = [r for r in cands if not r["ok"] and r["fits"]]
    a.scatter([r["price"] for r in bad], [r["tflops"] for r in bad], s=22,
              color="#bdc3c7", label="fits but illegal (power/budget/slots)")
    a.scatter([r["price"] for r in ok], [r["tflops"] for r in ok], s=42,
              color="#27ae60", label="legal build")
    for r in sorted(ok, key=lambda r: -r["tflops_per_kusd"])[:3]:
        a.annotate(f"{r['n']}x {r['gpu']}", (r["price"], r["tflops"]),
                   fontsize=7, xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("build price (USD)"); a.set_ylabel("dense FP16 TFLOPs")
    a.set_title("D. " + name, fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    # 4. lanes
    a = ax[1][1]
    ddp = f["lanes"]["ddp"]
    scheds = sorted({r["schedule"] for r in ddp}, key=lambda s: len(s))
    links = [l for l, _ in LINKS]
    width = 0.38
    for j, sched in enumerate(scheds):
        vals = [next(r["overhead_pct"] for r in ddp
                     if r["schedule"] == sched and r["link"] == l) for l in links]
        ypos = [i + (j - 0.5) * width for i in range(len(links))]
        a.barh(ypos, vals, height=width, label=sched,
               color=["#8e44ad", "#16a085"][j])
        for y, v in zip(ypos, vals):
            a.text(v, y, f" {v:.2f}%", va="center", fontsize=6)
    a.set_yticks(range(len(links))); a.set_yticklabels(links, fontsize=7)
    a.set_xscale("log")
    a.axvline(100, ls="--", lw=1, color="#c0392b")
    a.set_xlabel("all-reduce as % of one training step (8B, 2 GPUs)")
    a.set_title("E. the lanes matter only when the step is small", fontsize=9)
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=.3, axis="x")

    fig.suptitle("Project 45 - a build plan, checked against a measured GPU",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "buildplan.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
