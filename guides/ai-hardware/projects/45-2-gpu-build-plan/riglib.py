"""gpuprobe.py - the two instruments this phase uses on real hardware.

  build_cu()  - compile a .cu file with nvcc for this machine's GPU
  smi()       - one nvidia-smi sample (power, temperature, clocks, link state)
  Sampler     - a background thread that logs smi() every `period` seconds
                while something else runs

Imported by project 46 (acceptance testing) and project 47 (power/thermals).
Kept free of any project-specific logic on purpose: it is a driver, not an
experiment.
"""

import os
import shutil
import subprocess
import threading
import time

# nvidia-smi fields sampled by Sampler / smi(). Order matters: it is the CSV
# column order everything downstream assumes.
FIELDS = [
    "power.draw",                       # W, instantaneous board power
    "temperature.gpu",                  # C, core temperature
    "clocks.sm",                        # MHz, shader clock (the one that moves)
    "clocks.mem",                        # MHz, memory clock
    "fan.speed",                        # %, of maximum RPM
    "utilization.gpu",                  # %, of *time* a kernel was resident
    "pstate",                           # P0 (max) .. P8 (idle)
    "pcie.link.gen.current",            # 1..3 here; drops when idle
    "pcie.link.width.current",          # lanes actually trained
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.gpu_idle",
]

NUMERIC = {"power.draw": float, "temperature.gpu": float, "clocks.sm": float,
           "clocks.mem": float, "fan.speed": float, "utilization.gpu": float}


def have_gpu():
    return shutil.which("nvidia-smi") is not None


def build_cu(src, exe, arch="sm_61", extra=()):
    """Compile `src` to `exe`. Returns the nvcc command line actually used."""
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs the CUDA toolkit")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    cmd = ["nvcc", "-O3", "-arch=" + arch, *extra, src, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    return " ".join(cmd)


def smi(fields=FIELDS):
    """One telemetry sample as a dict. Missing/unsupported fields come back
    as the raw string nvidia-smi printed (e.g. '[N/A]')."""
    q = ",".join(fields)
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("nvidia-smi failed:\n" + r.stderr[-1000:])
    vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
    out = {}
    for k, v in zip(fields, vals):
        if k in NUMERIC:
            try:
                out[k] = NUMERIC[k](v)
            except ValueError:
                out[k] = None
        elif k.startswith("clocks_event"):
            out[k] = (v.lower() in ("active", "1", "true"))
        else:
            out[k] = v
    out["t"] = time.time()
    return out


def gpu_name():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,power.max_limit",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    name, lim, maxlim = [v.strip() for v in r.stdout.strip().split(",")]
    return name, float(lim), float(maxlim)


class Sampler(threading.Thread):
    """Logs telemetry in the background while the main thread runs a load.

    Why a thread and not a subprocess loop: we want the samples timestamped on
    the *same* clock as the workload's own timings, so a sample can be matched
    to the iteration that was running when it was taken.
    """

    def __init__(self, period=0.25, fields=FIELDS):
        super().__init__(daemon=True)
        self.period = period
        self.fields = fields
        self.rows = []
        self._halt = threading.Event()
        self.t0 = None

    def run(self):
        self.t0 = time.time()
        while not self._halt.is_set():
            try:
                s = smi(self.fields)
            except SystemExit:
                break
            s["dt"] = s["t"] - self.t0
            self.rows.append(s)
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        return self.rows

    def energy_j(self):
        """Trapezoidal integral of power over the sampled window (joules)."""
        e = 0.0
        for a, b in zip(self.rows, self.rows[1:]):
            if a["power.draw"] is None or b["power.draw"] is None:
                continue
            e += 0.5 * (a["power.draw"] + b["power.draw"]) * (b["t"] - a["t"])
        return e

    def window(self, t_start, t_end):
        """Samples whose wall-clock timestamp lies inside [t_start, t_end]."""
        return [r for r in self.rows if t_start <= r["t"] <= t_end]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
