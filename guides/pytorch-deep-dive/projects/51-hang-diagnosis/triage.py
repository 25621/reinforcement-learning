"""Point this at the process ID of something that appears to be stuck.

    python3 triage.py <pid> [stack_file]

It asks five questions, cheapest first, and none of them requires the victim to
cooperate beyond one two-line registration at startup (see `victim.py`):

    1. Is it burning CPU, or asleep?         /proc/<pid>/stat
    2. If asleep, waiting on what?           /proc/<pid>/wchan, /proc/<pid>/status
    3. Does it have children, and are they stuck too?
    4. Is it talking to anybody?             open sockets in /proc/<pid>/fd
    5. What is every thread executing right now?   kill -USR1 -> faulthandler

Question 1 comes first because its answer splits the world in half, and the
tools you reach for on each side share nothing. A process at 0% CPU is *waiting*
— for a lock, a socket, a pipe, a child — and the fix is to find out what it is
waiting for. A process at 100% CPU is *running* and simply never finishing, and
the fix is to find the loop.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")          # scheduler ticks per second


def read_stat(pid: int):
    """Parse /proc/<pid>/stat. The process name can contain spaces and
    parentheses, so split on the LAST ')' — a real bug in many scripts."""
    with open(f"/proc/{pid}/stat") as fh:
        raw = fh.read()
    rest = raw[raw.rindex(")") + 2:].split()
    return {
        "state": rest[0],                    # R running, S sleeping, D uninterruptible
        "utime": int(rest[11]),
        "stime": int(rest[12]),
        "threads": int(rest[17]),
    }


def cpu_fraction(pid: int, window: float = 2.0) -> float:
    """Cores' worth of CPU this process used over `window` seconds.

    0.0 means it did nothing at all. 1.0 means it kept exactly one core busy.
    4.0 means four. This single number is the top of the decision tree.
    """
    a = read_stat(pid)
    t0 = time.time()
    time.sleep(window)
    b = read_stat(pid)
    ticks = (b["utime"] + b["stime"]) - (a["utime"] + a["stime"])
    return ticks / CLK_TCK / (time.time() - t0)


def read_first_line(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip().split("\n")[0]
    except OSError:
        return ""


def children(pid: int) -> list[int]:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            kids = read_first_line(f"/proc/{pid}/task/{tid}/children")
            out += [int(k) for k in kids.split()]
    except OSError:
        pass
    return sorted(set(out))


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def n_sockets(pid: int) -> int:
    n = 0
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if "socket:" in os.readlink(f"/proc/{pid}/fd/{fd}"):
                    n += 1
            except OSError:
                pass
    except OSError:
        pass
    return n


def dump_stacks(pid: int, stack_file: str, wait: float = 1.0) -> str:
    """Ask the process to print every thread's Python stack, without killing it.

    `faulthandler.register(signal.SIGUSR1)` inside the victim turns this signal
    into a stack dump. The victim keeps running afterwards, so you can do this
    to a production job. It only works because the victim registered the
    handler *before* it got stuck.
    """
    size_before = os.path.getsize(stack_file) if os.path.exists(stack_file) else 0
    try:
        os.kill(pid, signal.SIGUSR1)
    except ProcessLookupError:
        return "(process gone)"
    time.sleep(wait)
    if not os.path.exists(stack_file):
        return "(no stack file - was faulthandler registered?)"
    with open(stack_file) as fh:
        fh.seek(size_before)
        return fh.read().strip()


def gdb_backtrace(pid: int, timeout: float = 25.0) -> str:
    """The tool everyone reaches for, and what it says on this machine."""
    try:
        out = subprocess.run(["gdb", "-p", str(pid), "-batch",
                              "-ex", "thread apply all bt"],
                             capture_output=True, text=True, timeout=timeout)
        text = (out.stdout + out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(gdb unavailable: {exc})"
    if "Could not attach" in text or "ptrace" in text:
        return "BLOCKED: " + " ".join(text.split())[:200]
    return text[:2000]


def top_python_frames(stacks: str, n: int = 3) -> list[str]:
    """Pull out the innermost frames that belong to user code, not to torch."""
    out = []
    for line in stacks.splitlines():
        line = line.strip()
        if (line.startswith("File ") and "site-packages" not in line
                and "/usr/lib/python3" not in line and "threading.py" not in line):
            out.append(line)
    return out[:n]


def triage(pid: int, stack_file: str | None = None, window: float = 2.0,
           with_gdb: bool = False) -> dict:
    st = read_stat(pid)
    cpu = cpu_fraction(pid, window)
    kids = children(pid)
    rep = {
        "pid": pid,
        "state": st["state"],
        "threads": st["threads"],
        "cpu_cores": round(cpu, 3),
        "wchan": read_first_line(f"/proc/{pid}/wchan") or "(none)",
        "children": kids,
        "child_states": {k: read_stat(k)["state"] for k in kids
                         if os.path.exists(f"/proc/{k}/stat")},
        "child_cpu": {k: round(cpu_fraction(k, 0.4), 2) for k in kids
                      if os.path.exists(f"/proc/{k}/stat")},
        "sockets": n_sockets(pid),
    }
    if stack_file:
        stacks = dump_stacks(pid, stack_file)
        rep["stacks"] = stacks
        rep["frames"] = top_python_frames(stacks)
    if with_gdb:
        rep["gdb"] = gdb_backtrace(pid)

    # --- the verdict ------------------------------------------------------
    if cpu > 0.5:
        rep["verdict"] = ("NOT BLOCKED - burning %.1f cores. Either an infinite "
                          "loop or genuine slowness; sample the stack twice and "
                          "compare." % cpu)
    elif rep["children"] and any(v == "S" for v in rep["child_states"].values()) \
            and all(c < 0.05 for c in rep["child_cpu"].values()):
        rep["verdict"] = ("BLOCKED, and so is every child - a parent waiting on "
                          "children that are themselves waiting.")
    elif rep["sockets"] > 2:
        rep["verdict"] = ("BLOCKED with %d sockets open - suspect a collective "
                          "the other ranks never reached." % rep["sockets"])
    else:
        rep["verdict"] = "BLOCKED at 0 cores - waiting on a lock or a pipe."
    return rep


if __name__ == "__main__":
    pid = int(sys.argv[1])
    sf = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(triage(pid, sf, with_gdb=True), indent=2))
