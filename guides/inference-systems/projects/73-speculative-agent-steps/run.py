#!/usr/bin/env python3
"""Project 73 — speculating the next agent step.

  A. Where an agent loop's time goes: measured model seconds vs tool seconds
     over 8 real episodes driven by a real 0.5B model.
  B. Is the next tool guessable at all?  Two cheap predictors and an oracle,
     trained on four episodes and scored on the other four.
  C. What speculation is worth: wall time, acceptance, wasted tool-seconds.
  D. Validation: the same episode run for real with threads, against the
     replay's prediction.
  E. The catch that decides whether you may do any of this — side effects.

  python3 run.py           # ~4 minutes
  python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agentlib as A                                           # noqa: E402

OUT = os.path.join(HERE, "outputs")
FINDINGS = os.path.join(OUT, "findings.json")
STEPS_PER_EPISODE = 6


# ---------------------------------------------------------------------------
# A: run the real agent and record what it did and how long it took
# ---------------------------------------------------------------------------

def record_episodes(model, tok):
    eps = []
    for ti, task in enumerate(A.TASKS):
        history, steps = [], []
        for k in range(STEPS_PER_EPISODE):
            ids = A.build_prompt(tok, task, history)
            tool, secs, ntok, text = A.decide(model, tok, ids)
            steps.append(dict(tool=tool, model_s=secs, tokens=ntok,
                              prompt_tokens=len(ids),
                              tool_s=A.TOOLS[tool][0],
                              read_only=A.TOOLS[tool][1],
                              text=text.strip()[:80]))
            history.append((tool, A.RESULTS[tool]))
        eps.append(dict(task=task, steps=steps))
        print(f"  episode {ti + 1}/{len(A.TASKS)}: "
              + " -> ".join(s["tool"] for s in steps)
              + f"   ({sum(s['model_s'] for s in steps):.1f}s of model)",
              flush=True)
    return eps


# ---------------------------------------------------------------------------
# C: replay one episode under a speculation policy
# ---------------------------------------------------------------------------

def replay(ep, predictor, safe_only=True):
    """Wall-clock timeline of one episode with speculation.

    The rule, step by step: the speculated tool starts when the model starts
    thinking.  If the guess was right, the tool has already been running for
    `model_s` seconds, so only the remainder is left; the step costs
    `max(model_s, tool_s)` instead of their sum.  If the guess was wrong, the
    step costs the full sum and the speculated seconds are wasted.
    """
    prev, t, hits, tries = None, 0.0, 0, 0
    wasted, unsafe_exec, saved = 0.0, 0, 0.0
    for s in ep["steps"]:
        guess = predictor.predict(prev) if predictor else None
        if guess is not None and safe_only and not A.TOOLS[guess][1]:
            guess = None                       # refuse to speculate a mutation
        m, w = s["model_s"], s["tool_s"]
        if guess is None:
            t += m + w
        else:
            tries += 1
            g_lat = A.TOOLS[guess][0]
            if not A.TOOLS[guess][1]:
                unsafe_exec += (guess != s["tool"])
            if guess == s["tool"]:
                hits += 1
                step = max(m, w)
                saved += (m + w) - step
                t += step
            else:
                wasted += min(m, g_lat)
                t += m + w
        prev = s["tool"]
    return dict(wall_s=t, hits=hits, tries=tries, wasted_tool_s=wasted,
                saved_s=saved, spurious_side_effects=unsafe_exec)


class Oracle:
    def __init__(self, ep):
        self.tools = [s["tool"] for s in ep["steps"]]
        self.i = 0

    def predict(self, prev=None):
        t = self.tools[min(self.i, len(self.tools) - 1)]
        self.i += 1
        return t


# ---------------------------------------------------------------------------
# D: the same thing for real, with threads
# ---------------------------------------------------------------------------

def real_run(model, tok, task, predictor, safe_only=True, use_spec=True):
    """Actually run the loop, with the speculated tool in a worker thread.

    The clock stops when the agent finishes its last step — *not* when the
    thread pool drains.  A mispredicted tool is still running somewhere when
    the agent moves on; a real server would abandon or cancel it, and it must
    not be charged to the user's wall clock.  Its cost is real and is counted
    separately, as wasted tool-seconds.
    """
    history, prev = [], None
    ex = ThreadPoolExecutor(max_workers=8)
    hits = tries = 0
    t0 = time.perf_counter()
    for k in range(STEPS_PER_EPISODE):
        guess = predictor.predict(prev) if (use_spec and predictor) else None
        if guess is not None and safe_only and not A.TOOLS[guess][1]:
            guess = None
        fut = None
        if guess is not None:
            tries += 1
            fut = ex.submit(time.sleep, A.TOOLS[guess][0])
        ids = A.build_prompt(tok, task, history)
        tool, _, _, _ = A.decide(model, tok, ids)
        if guess == tool and fut is not None:
            fut.result()                     # finish the remainder, if any
            hits += 1
        else:
            time.sleep(A.TOOLS[tool][0])     # the speculated work is dropped
        history.append((tool, A.RESULTS[tool]))
        prev = tool
    wall = time.perf_counter() - t0
    ex.shutdown(wait=False)
    return dict(wall_s=wall, hits=hits, tries=tries)


# ---------------------------------------------------------------------------

def analyse(eps):
    n = len(eps)
    train, test = eps[:n // 2], eps[n // 2:]
    seqs_train = [[s["tool"] for s in e["steps"]] for e in train]

    model_s = [s["model_s"] for e in eps for s in e["steps"]]
    tool_s = [s["tool_s"] for e in eps for s in e["steps"]]
    A_sec = dict(
        episodes=n, steps=len(model_s),
        model_s_total=round(sum(model_s), 1), tool_s_total=round(sum(tool_s), 1),
        model_share=round(sum(model_s) / (sum(model_s) + sum(tool_s)), 3),
        model_mean=round(statistics.mean(model_s), 2),
        model_p95=round(A.pct(model_s, 95), 2),
        tool_mean=round(statistics.mean(tool_s), 2),
        mean_tokens=round(statistics.mean(
            s["tokens"] for e in eps for s in e["steps"]), 1),
        tool_counts={t: sum(1 for e in eps for s in e["steps"]
                            if s["tool"] == t) for t in A.TOOL_NAMES},
        read_only_steps=sum(1 for e in eps for s in e["steps"]
                            if s["read_only"]),
    )

    freq = A.FreqPredictor().fit(seqs_train)
    const = A.ConstantPredictor().fit(seqs_train)

    # --- B: prediction accuracy on held-out episodes -----------------------
    B = {}
    for name, p in (("most-common", const), ("first-order", freq)):
        hit = tot = 0
        for e in test:
            prev = None
            for s in e["steps"]:
                tot += 1
                hit += (p.predict(prev) == s["tool"])
                prev = s["tool"]
        B[name] = dict(accuracy=round(100 * hit / tot, 1), n=tot)
    B["oracle"] = dict(accuracy=100.0, n=B["most-common"]["n"])
    B["table"] = {k: dict(v) for k, v in freq.table.items()}

    # --- C: speculation payoff on the held-out episodes --------------------
    arms = {
        "no speculation": (None, True),
        "most-common, safe tools": (const, True),
        "first-order, safe tools": (freq, True),
        "first-order, all tools": (freq, False),
        "oracle, safe tools": ("oracle", True),
        "oracle, all tools": ("oracle", False),
    }
    C = {}
    for name, (pred, safe) in arms.items():
        tot = dict(wall_s=0.0, hits=0, tries=0, wasted_tool_s=0.0,
                   saved_s=0.0, spurious_side_effects=0)
        for e in test:
            p = Oracle(e) if pred == "oracle" else pred
            r = replay(e, p, safe_only=safe)
            for k in tot:
                tot[k] += r[k]
        C[name] = {k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in tot.items()}
    base = C["no speculation"]["wall_s"]
    for name, d in C.items():
        d["speedup"] = round(base / d["wall_s"], 3)
        d["acceptance"] = (round(100 * d["hits"] / d["tries"], 1)
                           if d["tries"] else None)

    # --- E: how much of the win is only available if you ignore safety -----
    E = dict(
        unsafe_steps=len(model_s) - A_sec["read_only_steps"],
        unsafe_share=round(1 - A_sec["read_only_steps"] / len(model_s), 3),
        extra_speedup=round(C["first-order, all tools"]["speedup"]
                            - C["first-order, safe tools"]["speedup"], 3),
        spurious=C["first-order, all tools"]["spurious_side_effects"],
        ceiling=round(C["oracle, safe tools"]["speedup"], 3),
        theory=dict(),
    )
    # the closed form, checked against the replay
    for name, (pred, safe) in arms.items():
        if pred is None:
            continue
        hits = C[name]["hits"]
        tries = C[name]["tries"]
        pred_saving = 0.0
        for e in test:
            for s in e["steps"]:
                pred_saving += min(s["model_s"], s["tool_s"])
        acc = (hits / tries) if tries else 0
        E["theory"][name] = dict(
            expected_saving_s=round(acc * pred_saving * tries
                                    / max(sum(len(e["steps"]) for e in test), 1),
                                    2),
            measured_saving_s=C[name]["saved_s"])
    return A_sec, B, C, E, (const, freq, test)


# ---------------------------------------------------------------------------

def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A_, B, C, E, D = F["A"], F["B"], F["C"], F["E"], F["D"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Speculative agent steps: guess the tool, run it early, roll "
                 "back if wrong", fontsize=13)

    p = ax[0]
    p.bar([0], [A_["model_s_total"]], color="#4a6fa5", label="model thinking")
    p.bar([0], [A_["tool_s_total"]], bottom=[A_["model_s_total"]],
          color="#e0a458", label="tools running")
    p.set_xticks([0])
    p.set_xticklabels([f"{A_['episodes']} episodes\n{A_['steps']} steps"])
    p.set_ylabel("seconds")
    p.set_title(f"A. Where the loop's time goes\n"
                f"model {A_['model_share'] * 100:.0f}% / tools "
                f"{(1 - A_['model_share']) * 100:.0f}%")
    p.legend(fontsize=8)

    p = ax[1]
    names = [n for n in C]
    sp = [C[n]["speedup"] for n in names]
    colors = ["#8d9db6" if "no spec" in n else
              "#000000" if "oracle" in n else
              "#c0504d" if "all tools" in n else "#4a6fa5" for n in names]
    p.barh(range(len(names)), sp, color=colors)
    for i, (n, v) in enumerate(zip(names, sp)):
        acc = C[n]["acceptance"]
        p.text(v, i, f"  {v:.3f}x" + (f"  ({acc}% accepted)" if acc else ""),
               va="center", fontsize=8)
    p.axvline(1.0, color="k", lw=1)
    p.set_yticks(range(len(names)))
    p.set_yticklabels(names, fontsize=8)
    p.set_xlim(0.95, max(sp) * 1.25)
    p.set_xlabel("wall-clock speedup over no speculation")
    p.set_title("C. What speculation is worth")

    p = ax[2]
    labels = ["replay\nprediction", "real threaded\nrun"]
    vals = [D["predicted_spec_s"], D["measured_spec_s"]]
    base = [D["predicted_base_s"], D["measured_base_s"]]
    w = 0.35
    p.bar([i - w / 2 for i in range(2)], base, w, color="#8d9db6",
          label="no speculation")
    p.bar([i + w / 2 for i in range(2)], vals, w, color="#4a6fa5",
          label="first-order, safe tools")
    p.set_xticks(range(2))
    p.set_xticklabels(labels, fontsize=9)
    p.set_ylabel("seconds for one episode")
    p.set_title(f"D. Does the replay tell the truth?\n"
                f"error {D['error_pct']}%")
    p.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "spec_agent.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    print("loading Qwen2.5-0.5B-Instruct ...", flush=True)
    tok, model = A.load()
    print("A. running 8 real agent episodes ...", flush=True)
    eps = record_episodes(model, tok)

    A_sec, B, C, E, (const, freq, test) = analyse(eps)

    print("D. validating the replay with a real threaded run ...", flush=True)
    task = test[0]["task"]
    real_base = real_run(model, tok, task, None, use_spec=False)
    real_spec = real_run(model, tok, task, freq, safe_only=True)
    pred_base = replay(test[0], None)
    pred_spec = replay(test[0], freq, safe_only=True)
    D = dict(task=task,
             measured_base_s=round(real_base["wall_s"], 2),
             measured_spec_s=round(real_spec["wall_s"], 2),
             predicted_base_s=round(pred_base["wall_s"], 2),
             predicted_spec_s=round(pred_spec["wall_s"], 2),
             measured_speedup=round(real_base["wall_s"] / real_spec["wall_s"],
                                    3),
             predicted_speedup=round(pred_base["wall_s"] / pred_spec["wall_s"],
                                     3),
             real_hits=real_spec["hits"], real_tries=real_spec["tries"])
    D["error_pct"] = round(100 * abs(D["measured_spec_s"] - D["predicted_spec_s"])
                           / D["measured_spec_s"], 1)

    F = dict(A=A_sec, B=B, C=C, D=D, E=E,
             episodes=[[s["tool"] for s in e["steps"]] for e in eps])
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- A. the loop --------------------------------------------------")
    print(f"{A_sec['steps']} steps: {A_sec['model_s_total']}s of model "
          f"({A_sec['model_share'] * 100:.0f}%), {A_sec['tool_s_total']}s of "
          f"tools; mean decision {A_sec['model_mean']}s for "
          f"{A_sec['mean_tokens']} tokens")
    print(f"tool mix {A_sec['tool_counts']}")
    print("\n--- B. can the next tool be guessed? -----------------------------")
    for k in ("most-common", "first-order", "oracle"):
        print(f"  {k:<12} {B[k]['accuracy']}% on {B[k]['n']} held-out steps")
    print("\n--- C. speculation ------------------------------------------------")
    for name, d in C.items():
        print(f"{name:<26} {d['wall_s']:7.2f}s  {d['speedup']:.3f}x  "
              f"accepted {d['acceptance']}%  wasted {d['wasted_tool_s']}s  "
              f"spurious writes {d['spurious_side_effects']}")
    print("\n--- D. validation --------------------------------------------------")
    print(f"real run accepted {D['real_hits']}/{D['real_tries']} guesses")
    print(f"real {D['measured_base_s']}s -> {D['measured_spec_s']}s "
          f"({D['measured_speedup']}x); replay predicted "
          f"{D['predicted_base_s']}s -> {D['predicted_spec_s']}s "
          f"({D['predicted_speedup']}x); error {D['error_pct']}%")
    print("\n--- E. side effects -------------------------------------------------")
    print(f"{E['unsafe_steps']} of {A_sec['steps']} steps call a tool that "
          f"changes something ({E['unsafe_share'] * 100:.0f}%); speculating "
          f"them adds {E['extra_speedup']:+.3f}x and executes "
          f"{E['spurious']} unwanted writes/commits")
    plot(F)


if __name__ == "__main__":
    main()
