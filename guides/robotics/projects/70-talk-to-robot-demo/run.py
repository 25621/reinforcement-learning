"""Talk to a robot: a language model plans, a learned affordance model vetoes.

Five experiments:

1. the affordance model -- train it on robot trials, check it is calibrated
2. free-form generation vs scoring a fixed skill list
3. what the "say" score alone ranks, and the length-bias trap
4. the planner comparison: say only / can only / SayCan / oracle can
5. how much of SayCan's win is the LEARNED affordance and how much is a
   five-line precondition check

Run:  python3 run.py     (about 5 minutes; needs numpy, torch, transformers,
                          matplotlib, and the SmolLM2-360M weights in the HF
                          cache)
"""

import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kitchen as K
from llm import Scorer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
MAX_STEPS = 8
BASE_JITTER = 0.16


_LAST = [None]


def stamp(label):
    """Print how long the previous section took -- section 2 is the surprise."""
    now = time.time()
    if _LAST[0] is not None:
        print(f"      ({now - _LAST[0]:.0f} s)", flush=True)
    _LAST[0] = now
    print(label, flush=True)


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<46s} {value:>10}   {note}")


def new_kitchen(seed):
    """A kitchen with the robot's base jittered, so reach genuinely varies."""
    rng = np.random.default_rng(seed)
    base = K.BASE + rng.uniform(-BASE_JITTER, BASE_JITTER, 2)
    return K.Kitchen(rng, base=base)


# ---------------------------------------------------------------------------
# 1. the affordance model
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_affordance(n_trials=3000, seed=0, epochs=400, lr=0.5):
    """Logistic regression on (distance, clutter, gripper, preconditions).

    This is SayCan's value function in miniature.  It is trained on trials --
    the robot tries a skill, the world says yes or no -- and nothing else.  No
    text, no task.  Whether a skill *works* is a fact about the robot, and the
    only place that fact lives is in the robot's own logs.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n_trials):
        k = new_kitchen(seed * 100000 + i)
        for _ in range(rng.integers(0, 4)):        # random walk into odd states
            k.execute(K.SKILLS[rng.integers(K.N_SKILLS)])
        s = K.SKILLS[rng.integers(K.N_SKILLS)]
        X.append(k.features(s))
        y.append(1.0 if k.execute(s) else 0.0)
    X, y = np.array(X), np.array(y)
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-8
    Xn = np.concatenate([(X[:, :-1] - mu) / sd, X[:, -1:]], 1)
    w = np.zeros(Xn.shape[1])
    for _ in range(epochs):
        p = sigmoid(Xn @ w)
        w -= lr * Xn.T @ (p - y) / len(y)
    return {"w": w, "mu": mu, "sd": sd}, (X, y)


def can(model, kitchen, skills):
    X = np.array([kitchen.features(s) for s in skills])
    Xn = np.concatenate([(X[:, :-1] - model["mu"]) / model["sd"], X[:, -1:]], 1)
    return sigmoid(Xn @ model["w"])


def true_can(kitchen, skills):
    return np.array([kitchen.true_success_prob(s) for s in skills])


# ---------------------------------------------------------------------------
# a cache so that repeated (task, state) prompts cost nothing
# ---------------------------------------------------------------------------
class CachedSay:
    def __init__(self, scorer):
        self.scorer = scorer
        self.cache = {}
        self.hits = self.misses = 0

    def __call__(self, prompt):
        if prompt in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            self.cache[prompt] = self.scorer.score(prompt, K.SKILLS)
        return self.cache[prompt]


# ---------------------------------------------------------------------------
# planners
# ---------------------------------------------------------------------------
def plan_and_run(task, checker, kitchen, say, aff, mode):
    """One episode.  Returns (success, steps, illegal_proposals)."""
    illegal = 0
    for step in range(MAX_STEPS):
        prompt = K.make_prompt(task, kitchen)
        sc, ln = say(prompt)
        sc = np.array(sc, float)
        ln = np.array(ln, float)
        if mode in ("say_norm", "saycan_norm"):
            sc = sc / ln
        if mode == "can_only":
            util = np.log(can(aff, kitchen, K.SKILLS) + 1e-6)
        elif mode in ("say_only", "say_norm"):
            util = sc
        elif mode in ("saycan", "saycan_norm"):
            util = sc + np.log(can(aff, kitchen, K.SKILLS) + 1e-6)
        elif mode == "saycan_oracle":
            util = sc + np.log(true_can(kitchen, K.SKILLS) + 1e-6)
        elif mode == "say_precond":
            legal = np.array([kitchen.preconditions_met(s) for s in K.SKILLS])
            util = sc + np.where(legal, 0.0, -1e6)
        else:
            raise ValueError(mode)
        i = int(np.argmax(util))
        skill = K.SKILLS[i]
        if not kitchen.preconditions_met(skill):
            illegal += 1
        kitchen.execute(skill)
        if checker(kitchen):
            return True, step + 1, illegal
        if skill == "done" and kitchen.finished:
            break
    return bool(checker(kitchen)), MAX_STEPS, illegal


def sweep(mode, say, aff, seeds):
    print(f"    running {mode} ...", flush=True)
    ok, steps, illegal = [], [], []
    for t, (task, checker) in enumerate(K.TASKS):
        for s in seeds:
            k = new_kitchen(9000 + 31 * s + 7 * t)
            a, b, c = plan_and_run(task, checker, k, say, aff, mode)
            ok.append(a)
            steps.append(b)
            illegal.append(c)
    return np.mean(ok), np.mean(steps), np.mean(illegal)


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    scorer = Scorer(threads=min(12, os.cpu_count()))
    say = CachedSay(scorer)

    # -- 1. affordance model -------------------------------------------------
    stamp("\n[1] the affordance model ('can')")
    aff, (X, y) = train_affordance()
    ph = sigmoid(np.concatenate([(X[:, :-1] - aff["mu"]) / aff["sd"],
                                 X[:, -1:]], 1) @ aff["w"])
    acc = float(((ph > 0.5) == (y > 0.5)).mean())
    record("affordance", "trials collected", len(y))
    record("affordance", "base rate of success", round(float(y.mean()), 3))
    record("affordance", "accuracy at 0.5", round(acc, 3))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ph, bins) - 1, 0, 9)
    cal_x, cal_y, cal_n = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 20:
            cal_x.append(float(ph[m].mean()))
            cal_y.append(float(y[m].mean()))
            cal_n.append(int(m.sum()))
    ece = float(np.average(np.abs(np.array(cal_x) - np.array(cal_y)),
                           weights=cal_n))
    record("affordance", "calibration error (weighted)", round(ece, 4),
           "predicted p vs measured p")

    # -- 2. generate vs score ------------------------------------------------
    stamp("\n[2] free-form generation vs scoring a fixed list")
    gen_lines, gen_exec = 0, 0
    examples = []
    for task, _ in K.TASKS[:3]:
        k = new_kitchen(1234)
        txt = scorer.generate(K.make_prompt(task, k), 40)
        lines = [ln.strip(" .0123456789") for ln in txt.strip().splitlines()]
        lines = [ln for ln in lines if len(ln) > 4][:5]
        for ln in lines:
            gen_lines += 1
            if ln.lower() in [s.lower() for s in K.SKILLS]:
                gen_exec += 1
        examples.append((task, lines[:3]))
    record("generation", "generated lines", gen_lines)
    record("generation", "lines that are runnable skills", gen_exec,
           f"{100.0 * gen_exec / max(gen_lines, 1):.0f}%")
    for task, lines in examples[:3]:
        print(f"      '{task}' -> {lines}")

    # -- 3. what the raw score ranks ----------------------------------------
    stamp("\n[3] the say score on one state")
    k0 = new_kitchen(4242)
    sc0, ln0 = say(K.make_prompt("make me a cup of coffee", k0))
    sc0, ln0 = np.array(sc0), np.array(ln0)
    can0 = can(aff, k0, K.SKILLS)
    order = np.argsort(sc0)[::-1]
    for i in order[:5]:
        print(f"      say {sc0[i]:7.2f}  can {can0[i]:.2f}  "
              f"say+logcan {sc0[i] + np.log(can0[i] + 1e-6):7.2f}  {K.SKILLS[i]}")
    corr_len = float(np.corrcoef(sc0, ln0)[0, 1])
    record("say", "corr(say score, token count)", round(corr_len, 3),
           "longer sentences score lower")
    record("say", "top skill by raw say", K.SKILLS[order[0]])
    order_n = np.argsort(sc0 / ln0)[::-1]
    record("say", "top skill by per-token say", K.SKILLS[order_n[0]])

    # -- 4. the planner comparison ------------------------------------------
    stamp("\n[4] planners")
    seeds = list(range(5))
    modes = ["can_only", "say_only", "saycan", "saycan_norm",
             "saycan_oracle"]
    res = {}
    for m in modes:
        s, st, il = sweep(m, say, aff, seeds)
        res[m] = (s, st, il)
        record("planner", f"{m}: task success", round(float(s), 3),
               f"steps {st:.1f}, illegal proposals {il:.2f}")

    # -- 5. where the win comes from ----------------------------------------
    stamp("\n[5] learned affordance vs a five-line precondition check")
    s_pre, st_pre, il_pre = sweep("say_precond", say, aff, seeds)
    res["say_precond"] = (s_pre, st_pre, il_pre)
    record("ablation", "say + precondition filter: task success",
           round(float(s_pre), 3), f"steps {st_pre:.1f}")
    record("ablation", "SayCan minus say-only", round(res["saycan"][0] - res["say_only"][0], 3))
    record("ablation", "precondition filter minus say-only",
           round(s_pre - res["say_only"][0], 3))
    record("ablation", "SayCan minus precondition filter",
           round(res["saycan"][0] - s_pre, 3), "what the LEARNED part bought")

    record("cost", "distinct prompts scored", say.misses)
    record("cost", "seconds per LLM scoring call",
           round(scorer.score_seconds / max(say.misses, 1), 2))
    record("cost", "seconds spent generating free-form plans",
           round(scorer.gen_seconds, 1))
    record("cost", "prompt scorings saved by the cache", say.hits)
    record("cost", "total runtime (s)", round(time.time() - t_start, 1))

    # -- figures -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, (p, grab) in K.OBJECTS.items():
        ax[0].scatter(*p, s=140 if grab else 260,
                      marker="o" if grab else "s",
                      c="#d1495b" if grab else "#5b8c85", zorder=3)
        ax[0].annotate(name, p + np.array([0.015, 0.02]), fontsize=8)
    ax[0].scatter(*k0.base, marker="*", s=320, c="k", zorder=4, label="robot base")
    for r, c, lab in [(K.REACH_OK, "#2a9d8f", "comfortable reach"),
                      (K.REACH_MAX, "#e76f51", "absolute limit")]:
        ax[0].add_patch(plt.Circle(k0.base, r, fill=False, ls="--", ec=c, label=lab))
    ax[0].set_aspect("equal")
    ax[0].set_title("the kitchen (one base draw)")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax[1].plot(cal_x, cal_y, "o-", c="#264653", label="affordance model")
    ax[1].set_xlabel("predicted success probability")
    ax[1].set_ylabel("measured success rate")
    ax[1].set_title(f"is 'can' honest?  error {ece:.3f}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kitchen.png"), dpi=120)
    plt.close(fig)

    top = order[:8]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    yy = np.arange(len(top))
    say_n = (sc0[top] - sc0[top].min()) / (sc0[top].max() - sc0[top].min() + 1e-9)
    ax.barh(yy - 0.26, say_n, 0.25, label="say (rescaled)", color="#457b9d")
    ax.barh(yy, can0[top], 0.25, label="can", color="#e9c46a")
    comb = sc0[top] + np.log(can0[top] + 1e-6)
    comb_n = (comb - comb.min()) / (comb.max() - comb.min() + 1e-9)
    ax.barh(yy + 0.26, comb_n, 0.25, label="say x can", color="#d1495b")
    ax.set_yticks(yy)
    ax.set_yticklabels([K.SKILLS[i] for i in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("'make me a cup of coffee' -- the eight skills the LLM likes most")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "say_can.png"), dpi=120)
    plt.close(fig)

    names = ["can_only", "say_only", "say_precond", "saycan",
             "saycan_norm", "saycan_oracle"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vals = [res[n][0] for n in names]
    cols = ["#adb5bd", "#457b9d", "#2a9d8f", "#d1495b", "#f4a261", "#264653"]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(f"task success over {len(K.TASKS)} tasks x {len(seeds)} seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("what each term is worth")
    plt.xticks(rotation=18, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "planners.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
