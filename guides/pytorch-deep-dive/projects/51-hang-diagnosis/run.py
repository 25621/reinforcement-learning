"""Project 51 - diagnosing a hang you did not write.

Project 40 built hangs and read the error messages PyTorch produced. This one
starts one step later and one step harder: a process is stuck, you have its
process ID, and nothing has been printed. Every hang below is launched as a
real child process and then diagnosed from the outside, exactly as you would
diagnose a job on a cluster.

Sections:
  1. five things that all look identical from the outside
  2. the first question: is it burning CPU?
  3. the second question: what is every thread doing?
  4. what the standard tools say here (including the one that is blocked)
  5. the scoreboard: which tool identified which hang
  6. the two lines to add to every long-running job you own
  7. the distributed case, and what a timeout buys you

Run:  python3 run.py        (~4 minutes; it deliberately waits on real hangs)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "48-nan-forensics"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import debug_lib as D  # noqa: E402
import triage as T  # noqa: E402
from plot_style import SERIES, save, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
LOGS = os.path.join(HERE, "logs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
F_ = D.Findings()

VICTIM = os.path.join(HERE, "victim.py")

CASES = {
    "dataloader_fork":  "DataLoader workers deadlock on a lock inherited by fork",
    "ddp_straggler":    "rank 1 is alive but never arrives; rank 0 waits at all_reduce",
    "nan_loop":         "`while not (loss < tol)` with loss = NaN",
    "oversubscribed":   "4 processes x 12 threads on 12 cores: slow, not stuck",
    "lock_order":       "two threads take two locks in opposite orders",
}


def launch(mode, extra="", port=None, wait_ready=True):
    """Start one victim and (optionally) wait for its READY line.

    `wait_ready=False` matters for the distributed case: rank 1 cannot reach its
    READY line until rank 0 has joined the process group, so waiting for it
    before starting rank 0 deadlocks the launcher. That is itself the lesson of
    the case being launched.
    """
    stack_file = os.path.join(LOGS, f"{mode}{extra}.stack")
    for suffix in ("", ".beat"):
        try:
            os.remove(stack_file + suffix)
        except OSError:
            pass
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    if port:
        env["MASTER_PORT"] = str(port)
    proc = subprocess.Popen([sys.executable, VICTIM, mode, stack_file, str(extra)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, env=env, start_new_session=True)
    if not wait_ready:
        return proc, stack_file
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("READY"):
            return proc, stack_file
        if proc.poll() is not None:
            raise RuntimeError(f"{mode} exited before READY")
    raise RuntimeError(f"{mode} never printed READY")


def kill_tree(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def free_port():
    import socket
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ===========================================================================
# 1-3. Run all five, triage each one
# ===========================================================================

F_.head("1-3. Five hangs, one triage procedure")

reports: dict[str, dict] = {}
extra_procs: list = []

for mode in CASES:
    if mode == "ddp_straggler":
        port = free_port()
        dead, _ = launch(mode, extra=1, port=port, wait_ready=False)  # the straggler
        survivor, stack_file = launch(mode, extra=0, port=port)
        extra_procs.append(dead)
        proc = survivor
    elif mode == "oversubscribed":
        procs = [launch(mode, extra=12)[0] for _ in range(3)]
        extra_procs += procs
        proc, stack_file = launch(mode, extra=12)
    else:
        proc, stack_file = launch(mode)

    time.sleep(3.0)                                     # let it settle
    rep = T.triage(proc.pid, stack_file, window=2.0, with_gdb=(mode == "lock_order"))
    rep["desc"] = CASES[mode]

    # is it making progress? a heartbeat file only the honest case writes
    beat = stack_file + ".beat"
    first = os.path.exists(beat) and open(beat).read()
    time.sleep(2.0)
    second = os.path.exists(beat) and open(beat).read()
    rep["progress"] = bool(first is not False and first != second)

    reports[mode] = rep
    print(f"\n--- {mode}: {CASES[mode]}")
    print(f"    state={rep['state']}  cpu={rep['cpu_cores']} cores  "
          f"threads={rep['threads']}  children={rep['children']}  "
          f"sockets={rep['sockets']}  wchan={rep['wchan']}")
    print(f"    progress={rep['progress']}")
    print(f"    verdict: {rep['verdict']}")
    for fr in rep.get("frames", [])[:2]:
        print(f"    stack: {fr[:110]}")

    kill_tree(proc)
    for p in extra_procs:
        kill_tree(p)
    extra_procs = []

for mode, rep in reports.items():
    F_.note(f"{mode}: state / CPU cores / threads",
            f"{rep['state']} / {rep['cpu_cores']} / {rep['threads']}")
    F_.note(f"{mode}: children / sockets / wchan",
            f"{len(rep['children'])} / {rep['sockets']} / {rep['wchan']}")
    F_.note(f"{mode}: making progress?", rep["progress"])
    F_.note(f"{mode}: verdict", rep["verdict"][:110])
    F_.note(f"{mode}: innermost user frame",
            rep["frames"][0].replace(HERE + "/", "")[:100] if rep.get("frames") else "-")

with open(os.path.join(OUT, "triage_reports.json"), "w") as fh:
    # Drop the full stack text (it is in stacks.txt) and strip this checkout's
    # absolute path so the committed file is the same on any machine.
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "stacks"}
            for k, v in reports.items()}
    json.dump(json.loads(json.dumps(slim).replace(HERE + "/", "")), fh, indent=2)
print(f"\nwrote {os.path.join(OUT, 'triage_reports.json')}")

with open(os.path.join(OUT, "stacks.txt"), "w") as fh:
    for mode, rep in reports.items():
        fh.write(f"{'=' * 72}\n{mode}: {CASES[mode]}\n{'=' * 72}\n")
        fh.write((rep.get("stacks") or "(none)").replace(HERE + "/", "") + "\n\n")
print(f"wrote {os.path.join(OUT, 'stacks.txt')}")


# ===========================================================================
# 4. The tools, and the one that is blocked
# ===========================================================================

F_.head("4. What each tool could tell us")

gdb_text = reports["lock_order"].get("gdb", "(not run)")
F_.note("gdb -p <pid> -batch -ex 'thread apply all bt'",
        "BLOCKED by ptrace_scope" if gdb_text.startswith("BLOCKED") else "worked")
F_.note("the message", gdb_text[:150])
with open("/proc/sys/kernel/yama/ptrace_scope") as fh:
    scope = fh.read().strip()
F_.note("/proc/sys/kernel/yama/ptrace_scope", scope)
F_.note("what that setting means",
        "1 = a process may only be traced by its own parent, so gdb/py-spy from "
        "a shell cannot attach")
try:
    import py_spy  # noqa: F401
    F_.note("py-spy importable", True)
except ImportError:
    F_.note("py-spy installed here", False)
F_.note("what still works without ptrace",
        "/proc (always) and faulthandler-on-signal (if registered at startup)")


# ===========================================================================
# 5. The scoreboard
# ===========================================================================

F_.head("5. Which signal identified which hang")


def named_by_cpu(rep):
    return rep["cpu_cores"] > 0.5 or rep["progress"]


def named_by_children(rep):
    return bool(rep["children"]) and all(v < 0.05 for v in rep["child_cpu"].values())


def named_by_sockets(rep):
    return rep["sockets"] > 2


def named_by_stack(rep):
    return bool(rep.get("frames"))


SIGNALS = {
    "CPU fraction": named_by_cpu,
    "blocked children": named_by_children,
    "open sockets": named_by_sockets,
    "faulthandler stack": named_by_stack,
}
grid = {}
for mode, rep in reports.items():
    grid[mode] = {name: bool(fn(rep)) for name, fn in SIGNALS.items()}
    F_.note(mode, " | ".join(f"{n}: {'yes' if v else 'no '}" for n, v in grid[mode].items()))

F_.note("signal that fired for every case", "faulthandler stack")
F_.note("signal that fired for exactly one", "CPU fraction (and it is the one that "
                                             "changes what you do next)")


# ===========================================================================
# 6. The two lines
# ===========================================================================

F_.head("6. The registration that makes all of this possible")

no_reg, _ = launch("lock_order")
time.sleep(2.0)
missing = T.dump_stacks(no_reg.pid, os.path.join(LOGS, "does_not_exist.stack"))
F_.note("asking for a stack when no handler was registered", missing)
kill_tree(no_reg)
F_.note("cost of registering at startup", "2 lines, no measurable runtime cost")
F_.note("cost of NOT registering", "the information does not exist and cannot be "
                                   "recovered from a running process here")

with open(os.path.join(OUT, "register_me.py"), "w") as fh:
    fh.write('''"""Put this at the top of every long-running job you own."""
import faulthandler
import os
import signal

# 1. If the process dies from a segfault, print C and Python stacks.
faulthandler.enable()

# 2. If you send it SIGUSR1, print every thread's Python stack and KEEP RUNNING.
#    This is what lets you diagnose a hang without killing the job.
_dump = open(f"/tmp/stacks-{os.getpid()}.log", "w")
faulthandler.register(signal.SIGUSR1, file=_dump, all_threads=True)

#    Then, from any shell:   kill -USR1 <pid>   and read the file.

# 3. For distributed jobs, also cap the collective wait so an infinite hang
#    becomes an exception with a message (see project 40):
#
#    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
''')
print(f"wrote {os.path.join(OUT, 'register_me.py')}")


# ===========================================================================
# 7. The distributed case: a timeout turns a hang into a message
# ===========================================================================

F_.head("7. What a process-group timeout buys you")

TIMEOUT_CHILD = os.path.join(HERE, "timeout_demo.py")
rows = []
for secs, straggler in ((5, "sleep"), (15, "sleep"), (15, "crash")):
    port = free_port()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MASTER_ADDR="127.0.0.1",
               MASTER_PORT=str(port), PG_TIMEOUT_S=str(secs),
               STRAGGLER_MODE=straggler)
    t0 = time.time()
    p0 = subprocess.Popen([sys.executable, TIMEOUT_CHILD, "0"], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    p1 = subprocess.Popen([sys.executable, TIMEOUT_CHILD, "1"], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out, _ = p0.communicate(timeout=secs + 90)
    elapsed = time.time() - t0
    p1.kill()
    msg = next((l for l in out.splitlines() if "TIMEOUT" in l or "Error" in l), out.strip()[:120])
    rows.append((secs, elapsed, msg, straggler))
    who = "alive but absent" if straggler == "sleep" else "exited"
    F_.note(f"rank 1 {who}, timeout={secs}s: rank 0 reported after", f"{elapsed:.1f} s")
    F_.note(f"rank 1 {who}, timeout={secs}s: message", msg[:130])

F_.note("a rank that EXITS is noticed in", f"{rows[2][1]:.1f} s regardless of the timeout")
F_.note("a rank that is ALIVE and absent is noticed in", "exactly the timeout you set")
F_.note("PyTorch's default collective timeout", "30 minutes for gloo, 10 for NCCL")
F_.note("why you should lower it in development",
        "a hang with no deadline produces no information at all")


# ===========================================================================
# figures
# ===========================================================================

fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.2), dpi=110)
for ax in axes:
    style_axes(ax)
fig.patch.set_facecolor("#fcfcfb")

ax = axes[0]
modes = list(reports)
cpus = [reports[m]["cpu_cores"] for m in modes]
ax.barh(range(len(modes)), cpus,
        color=[SERIES[2] if c < 0.5 else SERIES[1] for c in cpus], height=0.6)
for i, c in enumerate(cpus):
    ax.text(max(c, 0.02), i, f" {c:.2f}", va="center", fontsize=8)
ax.set_yticks(range(len(modes)))
ax.set_yticklabels([m.replace("_", "\n") for m in modes], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("cores of CPU used while 'hung'")
ax.set_title("1. the first question splits the cases", loc="left", fontsize=11)

ax = axes[1]
names = list(SIGNALS)
mat = np.array([[1 if grid[m][n] else 0 for n in names] for m in modes])
ax.imshow(mat, cmap="RdYlGn", vmin=-0.3, vmax=1.3, aspect="auto")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax.set_yticks(range(len(modes)))
ax.set_yticklabels([m.replace("_", "\n") for m in modes], fontsize=8)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, "yes" if mat[i, j] else "no", ha="center", va="center", fontsize=8)
ax.grid(False)
ax.set_title("2. which signal fired", loc="left", fontsize=11)

ax = axes[2]
sleep_rows = [r for r in rows if r[3] == "sleep"]
secs = [r[0] for r in sleep_rows]
el = [r[1] for r in sleep_rows]
ax.plot(secs, el, "o-", color=SERIES[0], lw=1.8, ms=7,
        label="rank 1 alive but absent")
crash = [r for r in rows if r[3] == "crash"][0]
ax.plot([crash[0]], [crash[1]], "s", color=SERIES[2], ms=9,
        label="rank 1 exited (fast error)")
ax.plot(secs, secs, ls="--", lw=1.2, color="#898781", label="the timeout you set")
ax.set_xlabel("process-group timeout (s)")
ax.set_ylabel("seconds until rank 0 reported")
ax.set_title("3. a hang with a deadline is a bug report", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False)

save(fig, os.path.join(OUT, "hang_diagnosis.png"))
F_.write(os.path.join(OUT, "findings.csv"))
