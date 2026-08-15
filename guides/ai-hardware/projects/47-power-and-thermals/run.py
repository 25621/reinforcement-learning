"""Project 47 - what a GPU does to itself when you keep it busy.

Sections
  A. the long run     - 150 s of continuous load: power, temperature, clock,
                        fan, and which throttle reason (if any) fires
  B. two kernels      - arithmetic-bound vs memory-bound, same card:
                        power, clock, and work per joule
  C. how wide?        - the same kernel on 1..152 blocks: how much of the
                        board's power is fixed cost, and what half a GPU costs
  D. duty cycling     - the poor man's power limit, priced
  E. undervolting     - what we could not do (no root) and what the measured
                        numbers say it would be worth

All measurements are from this machine's GTX 1070 Ti through nvidia-smi.
Section E is labelled arithmetic because this card exposes no voltage telemetry
and the power limit cannot be set without root.
"""

import json
import math
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

LOAD = os.path.join(P45, "outputs", "gpuload")

LONG_S = 150      # section A
WIDTH_S = 8       # seconds per width step, section C
DUTY_S = 15       # seconds per duty setting, section D


def load(mode, seconds, extra=()):
    """Run the load kernel, returning its per-interval reports."""
    p = subprocess.run([LOAD, mode, str(seconds), *map(str, extra)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("gpuload failed:\n" + p.stdout[-1500:] + p.stderr[-1500:])
    rows = []
    for line in p.stdout.strip().splitlines():
        f = line.split(",")
        if f[0] == "sample":
            rows.append(dict(t_end=float(f[1]), dt=float(f[2]),
                             value=float(f[3]), tag=f[4]))
    return rows


def pair(sampler, rows):
    """Attach telemetry to each work interval: a report covering [t-dt, t]
    is matched with every nvidia-smi sample taken inside that window."""
    out = []
    for r in rows:
        s = sampler.window(r["t_end"] - r["dt"], r["t_end"])
        if not s:
            continue
        out.append(dict(
            **r,
            w=riglib.mean([x["power.draw"] for x in s]),
            c=riglib.mean([x["temperature.gpu"] for x in s]),
            mhz=riglib.mean([x["clocks.sm"] for x in s]),
            fan=riglib.mean([x["fan.speed"] for x in s]),
            util=riglib.mean([x["utilization.gpu"] for x in s]),
            power_cap=any(x["clocks_event_reasons.sw_power_cap"] for x in s),
            hw_thermal=any(x["clocks_event_reasons.hw_thermal_slowdown"] for x in s),
            sw_thermal=any(x["clocks_event_reasons.sw_thermal_slowdown"] for x in s),
            n=len(s)))
    return out


def fit_tau(times, temps):
    """Newton's law of cooling in reverse: T(t) = Tinf - (Tinf - T0) e^(-t/tau).
    Grid-search tau and the asymptote; no scipy in this environment."""
    t0 = temps[0]
    best = None
    for tinf in [max(temps) + d for d in (0, 1, 2, 3, 5, 8, 12)]:
        for tau in [2 + 0.5 * i for i in range(360)]:
            err = sum((tinf - (tinf - t0) * math.exp(-t / tau) - T) ** 2
                      for t, T in zip(times, temps))
            if best is None or err < best[0]:
                best = (err, tau, tinf)
    return dict(tau_s=best[1], t_inf_c=best[2], t0_c=t0,
                rmse_c=math.sqrt(best[0] / len(temps)))


# --------------------------------------------------------------------- A
def long_run():
    s = riglib.Sampler(period=0.5)
    s.start()
    time.sleep(3)
    rows = pair(s, load("compute", LONG_S))
    time.sleep(2)
    tel = s.stop()

    first = rows[:10]
    last = rows[-10:]
    warm = [r for r in tel if r["utilization.gpu"] and r["utilization.gpu"] > 50]
    t_start = warm[0]["t"]
    fit = fit_tau([r["t"] - t_start for r in warm],
                  [r["temperature.gpu"] for r in warm])
    return dict(
        seconds=LONG_S, series=rows, fit=fit,
        first_gflops=riglib.mean([r["value"] for r in first]),
        last_gflops=riglib.mean([r["value"] for r in last]),
        first_mhz=riglib.mean([r["mhz"] for r in first]),
        last_mhz=riglib.mean([r["mhz"] for r in last]),
        first_c=riglib.mean([r["c"] for r in first]),
        last_c=riglib.mean([r["c"] for r in last]),
        first_w=riglib.mean([r["w"] for r in first]),
        last_w=riglib.mean([r["w"] for r in last]),
        fan_start=rows[0]["fan"], fan_end=rows[-1]["fan"],
        any_power_cap=any(r["power_cap"] for r in rows),
        any_hw_thermal=any(r["hw_thermal"] for r in rows),
        any_sw_thermal=any(r["sw_thermal"] for r in rows),
        perf_drift_pct=100 * (riglib.mean([r["value"] for r in last])
                              / riglib.mean([r["value"] for r in first]) - 1),
        clock_drift_pct=100 * (riglib.mean([r["mhz"] for r in last])
                               / riglib.mean([r["mhz"] for r in first]) - 1),
    )


# --------------------------------------------------------------------- B
def two_kernels():
    out = {}
    for mode, unit in [("compute", "GFLOP/s"), ("memory", "GB/s")]:
        s = riglib.Sampler(period=0.25)
        s.start()
        time.sleep(2)
        rows = pair(s, load(mode, 15))[2:]     # drop warm-up intervals
        s.stop()
        w = riglib.mean([r["w"] for r in rows])
        v = riglib.mean([r["value"] for r in rows])
        out[mode] = dict(unit=unit, value=v, watts=w,
                         mhz=riglib.mean([r["mhz"] for r in rows]),
                         temp=riglib.mean([r["c"] for r in rows]),
                         per_joule=v / w, series=rows)
        time.sleep(5)
    return out


# --------------------------------------------------------------------- C
def width_sweep():
    s = riglib.Sampler(period=0.25)
    s.start()
    time.sleep(2)
    rows = pair(s, load("width", WIDTH_S))
    s.stop()
    for r in rows:
        r["blocks"] = int(r["tag"].split("=")[1])
        r["frac_of_chip"] = min(1.0, r["blocks"] / 152)
        # one block occupies one SM, so up to 19 blocks the block count IS the
        # number of SMs switching; beyond that, extra blocks stack on SMs that
        # are already awake.
        r["active_sms"] = min(r["blocks"], 19)
        r["gflops_per_watt"] = r["value"] / r["w"]
    # Fit power against the number of SMs that are awake, over the range where
    # adding a block really does light up a new SM.
    lin = [r for r in rows if r["blocks"] <= 19]
    xs = [r["active_sms"] for r in lin]
    ys = [r["w"] for r in lin]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    intercept = my - slope * mx
    full_w = max(r["w"] for r in rows)
    return dict(rows=rows, awake_w=intercept, per_sm_w=slope,
                one_block_w=lin[0]["w"], full_w=full_w,
                awake_frac=intercept / full_w,
                best_efficiency=max(r["gflops_per_watt"] for r in rows),
                best_efficiency_blocks=max(
                    rows, key=lambda r: r["gflops_per_watt"])["blocks"])


# --------------------------------------------------------------------- D
def duty_sweep():
    out = []
    s = riglib.Sampler(period=0.25)
    s.start()
    for off in (0, 25, 50, 100, 200):
        time.sleep(2)
        rows = pair(s, load("duty", DUTY_S, extra=(100, off)))[2:]
        v = riglib.mean([r["value"] for r in rows])
        w = riglib.mean([r["w"] for r in rows])
        out.append(dict(off_ms=off, duty=100 / (100 + off), gflops=v, watts=w,
                        gflops_per_watt=v / w))
    s.stop()
    base = out[0]
    for r in out:
        r["rel_perf"] = r["gflops"] / base["gflops"]
        r["rel_power"] = r["watts"] / base["watts"]
        r["rel_efficiency"] = r["gflops_per_watt"] / base["gflops_per_watt"]
    return out


# --------------------------------------------------------------------- E
def undervolt_arithmetic(idle_w, load_w, v_stock=1.05, v_target=0.95):
    """Dynamic switching power is P = C V^2 f. Lowering V at the SAME f leaves
    the work per second unchanged and scales the switching part by (V'/V)^2.
    The static part (leakage, memory, fans, VRMs) does not move.

    v_stock is the typical stock voltage for this chip at its boost clock; this
    card exposes no voltage telemetry, so the section is arithmetic and the
    assumption is stated rather than measured.
    """
    dyn = load_w - idle_w
    scale = (v_target / v_stock) ** 2
    new = idle_w + dyn * scale
    # Try to actually set a power limit, and record what happens.
    p = subprocess.run(["nvidia-smi", "-pl", "150"], capture_output=True,
                       text=True)
    with open(os.path.join(OUT, "power_limit_attempt.log"), "w") as f:
        f.write("$ nvidia-smi -pl 150\n" + p.stdout + p.stderr)
    return dict(v_stock=v_stock, v_target=v_target, static_w=idle_w,
                dynamic_w=dyn, dynamic_scale=scale, new_board_w=new,
                board_saving_w=load_w - new,
                board_saving_pct=100 * (1 - new / load_w),
                efficiency_gain_pct=100 * (load_w / new - 1),
                set_power_limit_rc=p.returncode,
                set_power_limit_msg=(p.stdout + p.stderr).strip().splitlines()[0]
                if (p.stdout + p.stderr).strip() else "")


def cold_idle(seconds=4.0):
    """Board power with nothing running - the floor every other number sits on."""
    s = riglib.Sampler(period=0.25)
    s.start(); time.sleep(seconds); rows = s.stop()
    return dict(watts=riglib.mean([r["power.draw"] for r in rows]),
                temp=riglib.mean([r["temperature.gpu"] for r in rows]),
                mhz=riglib.mean([r["clocks.sm"] for r in rows]),
                pstate=rows[-1]["pstate"])


def main():
    findings = {}

    print("== 0. cold idle ==")
    idle = cold_idle()
    findings["idle"] = idle
    print(f"   {idle['watts']:.1f} W, {idle['temp']:.0f} C, "
          f"{idle['mhz']:.0f} MHz, {idle['pstate']}")

    print("== A. 150 s of continuous load ==")
    a = long_run()
    findings["long_run"] = a
    print(f"   temperature {a['first_c']:.0f} -> {a['last_c']:.0f} C "
          f"(asymptote {a['fit']['t_inf_c']:.0f} C, time constant "
          f"{a['fit']['tau_s']:.0f} s, fit RMSE {a['fit']['rmse_c']:.2f} C)")
    print(f"   clock {a['first_mhz']:.0f} -> {a['last_mhz']:.0f} MHz "
          f"({a['clock_drift_pct']:+.1f}%), perf {a['first_gflops']:.0f} -> "
          f"{a['last_gflops']:.0f} GFLOP/s ({a['perf_drift_pct']:+.1f}%)")
    print(f"   fan {a['fan_start']:.0f} -> {a['fan_end']:.0f} %, "
          f"power {a['first_w']:.0f} -> {a['last_w']:.0f} W")
    print(f"   throttle reasons: power_cap={a['any_power_cap']} "
          f"hw_thermal={a['any_hw_thermal']} sw_thermal={a['any_sw_thermal']}")

    print("== B. arithmetic vs memory ==")
    b = two_kernels()
    findings["two_kernels"] = b
    for m, r in b.items():
        print(f"   {m:>8}: {r['value']:8.1f} {r['unit']:>8}  {r['watts']:6.1f} W  "
              f"{r['mhz']:.0f} MHz  {r['temp']:.0f} C")
    print(f"   the memory-bound kernel draws "
          f"{b['memory']['watts']/b['compute']['watts']:.2f}x the power of the "
          f"arithmetic one")

    print("== C. how much of the chip is switching ==")
    c = width_sweep()
    findings["width"] = c
    for r in c["rows"]:
        print(f"   {r['blocks']:>4} blocks ({100*r['frac_of_chip']:5.1f}% of chip): "
              f"{r['value']:8.1f} GFLOP/s {r['w']:6.1f} W "
              f"{r['gflops_per_watt']:6.1f} GFLOP/J")
    print(f"   one block (1 of 19 SMs) already costs {c['one_block_w']:.1f} W; "
          f"fit over 1..19 SMs: {c['awake_w']:.1f} W just for being awake + "
          f"{c['per_sm_w']:.2f} W per SM")
    print(f"   -> {100*c['awake_frac']:.0f}% of a fully loaded board's "
          f"{c['full_w']:.1f} W is paid before any SM does arithmetic; best "
          f"efficiency {c['best_efficiency']:.1f} GFLOP/J at "
          f"{c['best_efficiency_blocks']} blocks")

    print("== D. duty cycling ==")
    d = duty_sweep()
    findings["duty"] = d
    for r in d:
        print(f"   on 100 ms / off {r['off_ms']:>3} ms (duty {100*r['duty']:5.1f}%): "
              f"{r['gflops']:8.1f} GFLOP/s {r['watts']:6.1f} W "
              f"{r['gflops_per_watt']:6.1f} GFLOP/J "
              f"(efficiency {100*r['rel_efficiency']:5.1f}% of full speed)")

    print("== E. undervolting ==")
    e = undervolt_arithmetic(idle_w=idle["watts"],
                             load_w=b["compute"]["watts"])
    findings["undervolt"] = e
    print(f"   nvidia-smi -pl 150 -> rc={e['set_power_limit_rc']}: "
          f"{e['set_power_limit_msg']}")
    print(f"   arithmetic: {e['v_stock']} V -> {e['v_target']} V at the same "
          f"clock scales the {e['dynamic_w']:.0f} W dynamic part by "
          f"{e['dynamic_scale']:.3f}")
    print(f"   -> {e['new_board_w']:.0f} W instead of "
          f"{b['compute']['watts']:.0f} W: {e['board_saving_pct']:.0f}% less "
          f"power for the same work (+{e['efficiency_gain_pct']:.0f}% "
          f"perf-per-watt)")

    findings["two_card"] = two_card_arithmetic(b["compute"]["watts"],
                                               a["fit"]["t_inf_c"])
    print("== F. two cards in one case (arithmetic) ==")
    for r in findings["two_card"]:
        print(f"   {r['case']}: {r['gpu_w']:.0f} W of GPU -> "
              f"{r['air_rise_c']:.1f} C rise across the case at "
              f"{r['cfm']} CFM; card 2 inlet ~{r['inlet2_c']:.0f} C, "
              f"predicted core ~{r['core2_c']:.0f} C")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, thermals.png")


def two_card_arithmetic(one_card_w, t_inf, ambient=22.0):
    """Air carries heat: Q = m_dot x cp x dT. For airflow in CFM and heat in
    watts the shop formula is dT(C) = W / (1.76 x CFM).

    The second card in a stack breathes the first card's exhaust, so its inlet
    is ambient + rise, and its core sits that much higher too.
    """
    rows = []
    rise_per_w = t_inf - ambient - 0  # measured core rise over ambient, 1 card
    for case, cfm in [("open frame, 120 CFM", 120), ("tower, 60 CFM", 60),
                      ("quiet tower, 30 CFM", 30)]:
        gpu_w = 2 * one_card_w
        dt = gpu_w / (1.76 * cfm)
        inlet2 = ambient + dt
        rows.append(dict(case=case, cfm=cfm, gpu_w=gpu_w, air_rise_c=dt,
                         inlet2_c=inlet2,
                         core2_c=inlet2 + (t_inf - ambient),
                         core_rise_over_inlet_c=t_inf - ambient))
    return rows


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        a = f["long_run"]
        for k, u in [("first_c", "C"), ("last_c", "C"), ("first_mhz", "MHz"),
                     ("last_mhz", "MHz"), ("perf_drift_pct", "%"),
                     ("clock_drift_pct", "%")]:
            w.writerow(["A long run", k, round(a[k], 3), u, "measured"])
        w.writerow(["A long run", "thermal tau", round(a["fit"]["tau_s"], 2),
                    "s", "measured"])
        w.writerow(["A long run", "asymptote", round(a["fit"]["t_inf_c"], 2),
                    "C", "measured"])
        for m, r in f["two_kernels"].items():
            w.writerow(["B kernels", m, round(r["value"], 2), r["unit"], "measured"])
            w.writerow(["B kernels", m + " power", round(r["watts"], 2), "W",
                        "measured"])
        for r in f["width"]["rows"]:
            w.writerow(["C width", f"{r['blocks']} blocks",
                        round(r["gflops_per_watt"], 3), "GFLOP/J", "measured"])
        w.writerow(["C width", "awake power (fit intercept)",
                    round(f["width"]["awake_w"], 2), "W", "measured (fit)"])
        w.writerow(["C width", "per-SM power (fit slope)",
                    round(f["width"]["per_sm_w"], 3), "W/SM", "measured (fit)"])
        for r in f["duty"]:
            w.writerow(["D duty", f"off {r['off_ms']} ms",
                        round(r["gflops_per_watt"], 3), "GFLOP/J", "measured"])
        for k, v in f["undervolt"].items():
            if isinstance(v, (int, float)):
                w.writerow(["E undervolt", k, round(v, 3), "", "arithmetic"])
        for r in f["two_card"]:
            w.writerow(["F two cards", r["case"], round(r["core2_c"], 1), "C",
                        "arithmetic"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    a = ax[0][0]
    s = f["long_run"]["series"]
    t = [r["t_end"] - s[0]["t_end"] for r in s]
    a.plot(t, [r["c"] for r in s], color="#c0392b", label="temperature (C)")
    a.plot(t, [r["fan"] for r in s], color="#2980b9", label="fan (%)")
    fit = f["long_run"]["fit"]
    a.axhline(fit["t_inf_c"], ls="--", lw=1, color="#7f8c8d")
    a.text(t[-1], fit["t_inf_c"], f" asymptote {fit['t_inf_c']:.0f} C",
           fontsize=7, ha="right", va="bottom")
    a2 = a.twinx()
    a2.plot(t, [r["mhz"] for r in s], color="#27ae60", lw=1, label="SM clock")
    a2.set_ylabel("SM clock (MHz)", color="#27ae60")
    a.set_xlabel("seconds of continuous load"); a.set_ylabel("C / %")
    a.set_title(f"A. {f['long_run']['seconds']} s at full load: "
                f"tau = {fit['tau_s']:.0f} s, clock "
                f"{f['long_run']['clock_drift_pct']:+.1f}%", fontsize=9)
    a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.3)

    a = ax[0][1]
    w = f["width"]["rows"]
    one = [r for r in w if r["blocks"] <= 19]
    many = [r for r in w if r["blocks"] > 19]
    a.plot([r["active_sms"] for r in one], [r["w"] for r in one], "o-",
           color="#c0392b", label="board power, 1 block per SM")
    a.plot([r["active_sms"] for r in many], [r["w"] for r in many], "x",
           color="#c0392b", ms=9, label="all 19 SMs, 2/4/8 blocks each")
    fw = f["width"]
    a.plot([0, 19], [fw["awake_w"], fw["awake_w"] + 19 * fw["per_sm_w"]],
           "--", color="#7f8c8d", lw=1,
           label=f"fit: {fw['awake_w']:.0f} W awake + {fw['per_sm_w']:.1f} W/SM")
    a2 = a.twinx()
    a2.plot([r["active_sms"] for r in one],
            [r["gflops_per_watt"] for r in one], "s-", color="#27ae60")
    a2.plot([r["active_sms"] for r in many],
            [r["gflops_per_watt"] for r in many], "s", color="#27ae60",
            mfc="none")
    a2.set_ylabel("GFLOP per joule", color="#27ae60")
    a.set_xlabel("SMs given work (of 19; the last 3 points stack on all 19)")
    a.set_ylabel("W")
    a.set_title("C. the first SM costs 40 W, the next 18 cost 3 W each",
                fontsize=9)
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=.3)

    a = ax[1][0]
    d = f["duty"]
    a.plot([100 * r["duty"] for r in d], [100 * r["rel_perf"] for r in d], "o-",
           label="performance (% of full)", color="#2980b9")
    a.plot([100 * r["duty"] for r in d], [100 * r["rel_power"] for r in d], "s-",
           label="power (% of full)", color="#c0392b")
    a.plot([100 * r["duty"] for r in d], [100 * r["rel_efficiency"] for r in d],
           "^-", label="efficiency (% of full)", color="#27ae60")
    a.axhline(100, ls="--", lw=1, color="#7f8c8d")
    a.set_xlabel("duty cycle (%)"); a.set_ylabel("% of full-speed value")
    a.set_title("D. throttling by waiting never improves efficiency", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1][1]
    b = f["two_kernels"]; u = f["undervolt"]
    names = ["FMA\nkernel", "streaming\nkernel", "FMA at 0.95 V\n(arithmetic)"]
    vals = [b["compute"]["watts"], b["memory"]["watts"], u["new_board_w"]]
    cols = ["#c0392b", "#2980b9", "#27ae60"]
    a.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.0f} W", ha="center", va="bottom", fontsize=9)
    a.axhline(180, ls="--", lw=1, color="#7f8c8d")
    a.text(2.4, 180, "180 W limit", fontsize=7, ha="right", va="bottom")
    a.set_ylabel("board power (W)"); a.set_ylim(0, 200)
    a.set_title("B/E. power is a property of the kernel (and the voltage)",
                fontsize=9)
    a.grid(alpha=.3, axis="y")

    fig.suptitle("Project 47 - power and thermals, measured on one card",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "thermals.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
