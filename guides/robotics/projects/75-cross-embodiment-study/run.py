"""Train on robot A, deploy on robot B, and find out what actually broke.

Six experiments:

1. the gap -- policy on A, policy on B, and B's own ceiling
2. one axis at a time -- add-one-in and leave-one-out main effects
3. calibration -- one measured number
4. task-space retargeting -- the policy keeps A's body, the adapter translates
5. the data fix -- N demonstrations on B, and where it crosses the adapters
6. does the source robot matter? -- the same study run in reverse

Run:  python3 run.py     (about 6 minutes; needs numpy, torch, matplotlib)
"""

import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import nets                # noqa: E402
import embody as E         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
N_EVAL = 100
N_CEIL = 60


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<52s} {value:>8}   {note}")


def plain(policy_fn):
    """The no-adapter transfer: hand B's observation straight to the policy."""
    return lambda env: policy_fn


def main():
    t0 = time.time()
    # Single-observation network calls dominate the run time here, and a
    # 16 x 256 matrix-vector product split across twelve cores spends more
    # time at the thread barrier than on the arithmetic (project 64's lesson,
    # met again).  One thread is measurably faster for this workload.
    import torch
    torch.set_num_threads(1)
    nets.seed_all(0)
    src_arm = E.make_arm(E.SOURCE["lengths"], E.SOURCE["masses"])

    # -- the source policy ---------------------------------------------------
    print("\n[0] cloning a policy on robot A")
    obs, act, n = E.collect(E.SOURCE, 400, seed=0, noise=0.25)
    net, norm, _ = nets.train_bc(obs, act, epochs=150, seed=0)
    pol = nets.make_policy(net, norm)
    record("source", "demonstrations used", n, f"{len(obs)} transitions")

    # -- 1. the gap ----------------------------------------------------------
    print("\n[1] the transfer gap")
    a_on_a = E.evaluate(plain(pol), E.SOURCE, n=N_EVAL)["success"]
    a_on_b = E.evaluate(plain(pol), E.TARGET, n=N_EVAL)["success"]
    ceil_a = E.expert_ceiling(E.SOURCE, n=N_CEIL)
    ceil_b = E.expert_ceiling(E.TARGET, n=N_CEIL)
    record("gap", "policy on robot A (home)", round(a_on_a, 3),
           f"expert ceiling {ceil_a:.3f}")
    record("gap", "policy on robot B (zero shot)", round(a_on_b, 3),
           f"expert ceiling {ceil_b:.3f}")
    record("gap", "the gap", round(a_on_a - a_on_b, 3))
    record("gap", "how much of the gap the robot itself explains",
           round(ceil_a - ceil_b, 3), "ceiling A minus ceiling B")

    # -- 2. one axis at a time ----------------------------------------------
    print("\n[2] main effects: add one in, and leave one out")
    add_in, leave_out = {}, {}
    all_ax = list(E.AXES)
    for ax in all_ax:
        s1 = E.evaluate(plain(pol), E.spec([ax]), n=N_EVAL)["success"]
        s2 = E.evaluate(plain(pol), E.spec([a for a in all_ax if a != ax]),
                        n=N_EVAL)["success"]
        add_in[ax] = a_on_a - s1
        leave_out[ax] = s2 - a_on_b
        # The ceiling for this axis alone: if the SCRIPTED controller also
        # drops, the axis made the task harder rather than making the policy
        # wrong, and the two must not be added to the same total.
        ceil_ax = E.expert_ceiling(E.spec([ax]), n=N_CEIL)
        record("axis", f"{ax}: add-one-in cost", round(add_in[ax], 3),
               f"A alone {a_on_a:.3f} -> {s1:.3f}   expert ceiling {ceil_ax:.3f}")
        record("axis", f"{ax}: leave-one-out gain", round(leave_out[ax], 3),
               f"B minus this axis {s2:.3f}")
    record("axis", "sum of add-one-in costs", round(sum(add_in.values()), 3),
           f"the whole gap is {a_on_a - a_on_b:.3f}")

    # -- 3. calibration ------------------------------------------------------
    print("\n[3] the one-number fix")
    rng = np.random.default_rng(0)
    probe = E.make_env(E.TARGET, rng)
    rt = E.Retargeter(pol, src_arm, probe, scale_only=True)
    g_task, g_joint = rt.calibrate()
    record("calibrate", "measured tool-motion ratio A/B", round(g_task, 3),
           f"the action-scale factor alone would be "
           f"{1 / E.TARGET['dq_scale']:.3f}")
    record("calibrate", "measured joint-command ratio", round(g_joint, 3))

    def mk_scale(env):
        r = E.Retargeter(pol, src_arm, env, scale_only=True)
        r.gain_task = g_task
        return r
    s_cal = E.evaluate(mk_scale, E.TARGET, n=N_EVAL)["success"]
    record("calibrate", "policy on B with calibration", round(s_cal, 3),
           f"was {a_on_b:.3f}")
    # Calibration cannot rescue a policy whose observation is nonsense, so also
    # measure it on the robot B that does NOT have the flipped encoder.  This
    # is what the one-number fix is worth when it is the right fix.
    no_o = [a for a in all_ax if a != "O obs convention"]
    s_noO = E.evaluate(plain(pol), E.spec(no_o), n=N_EVAL)["success"]
    s_noO_cal = E.evaluate(mk_scale, E.spec(no_o), n=N_EVAL)["success"]
    record("calibrate", "on robot B without the encoder flip",
           f"{s_noO:.3f} -> {s_noO_cal:.3f}", "plain -> calibrated")

    # -- 4. task-space retargeting ------------------------------------------
    print("\n[4] retargeting through the tip")
    def mk_rt(env):
        r = E.Retargeter(pol, src_arm, env)
        r.gain_joint = g_joint
        return r
    s_rt = E.evaluate(mk_rt, E.TARGET, n=N_EVAL)["success"]
    record("retarget", "policy on B, task-space retargeting", round(s_rt, 3),
           f"ceiling {ceil_b:.3f}")
    # retargeting on each single axis, to see WHICH gap it closes
    for ax in all_ax:
        sp = E.spec([ax])
        s_plain = E.evaluate(plain(pol), sp, n=N_EVAL)["success"]
        s_ret = E.evaluate(mk_rt, sp, n=N_EVAL)["success"]
        record("retarget", f"{ax}: plain -> retargeted",
               f"{s_plain:.3f} -> {s_ret:.3f}", f"delta {s_ret - s_plain:+.3f}")

    # -- 5. the data fix -----------------------------------------------------
    print("\n[5] demonstrations on robot B")
    curve = {}
    for nd in (5, 10, 25, 100):
        ob, ac, _ = E.collect(E.TARGET, nd, seed=5, noise=0.15)
        # scratch
        n_s, nm_s, _ = nets.train_bc(ob, ac, epochs=200, seed=2)
        s_scratch = E.evaluate(plain(nets.make_policy(n_s, nm_s)), E.TARGET,
                               n=N_EVAL)["success"]
        # fine-tune the source policy
        import copy
        n_f = copy.deepcopy(net)
        import torch
        opt = torch.optim.AdamW(n_f.parameters(), lr=3e-4, weight_decay=1e-4)
        X = torch.tensor(norm(ob), dtype=torch.float32)
        Y = torch.tensor(ac, dtype=torch.float32)
        for _ in range(200):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                b = perm[i:i + 256]
                loss = ((n_f(X[b]) - Y[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        n_f.eval()
        s_ft = E.evaluate(plain(nets.make_policy(n_f, norm)), E.TARGET,
                          n=N_EVAL)["success"]
        curve[nd] = (s_scratch, s_ft)
        record("data", f"{nd} demos on B: from scratch", round(s_scratch, 3),
               f"fine-tuned from A {s_ft:.3f}")

    # -- 6. reverse the study ------------------------------------------------
    print("\n[6] the same study, the other way round")
    ob2, ac2, _ = E.collect(E.TARGET, 400, seed=1, noise=0.25)
    net2, norm2, _ = nets.train_bc(ob2, ac2, epochs=150, seed=0)
    pol2 = nets.make_policy(net2, norm2)
    tgt_arm = E.make_arm(E.TARGET["lengths"], E.TARGET["masses"])
    b_on_b = E.evaluate(plain(pol2), E.TARGET, n=N_EVAL)["success"]
    b_on_a = E.evaluate(plain(pol2), E.SOURCE, n=N_EVAL)["success"]

    probe2 = E.make_env(E.SOURCE, np.random.default_rng(0))
    _, g_joint2 = E.Retargeter(pol2, tgt_arm, probe2,
                               src_spec=E.TARGET).calibrate()

    def mk_rt2(env):
        r = E.Retargeter(pol2, tgt_arm, env, src_spec=E.TARGET)
        r.gain_joint = g_joint2
        return r
    b_rt = E.evaluate(mk_rt2, E.SOURCE, n=N_EVAL)["success"]
    record("reverse", "policy on B (home)", round(b_on_b, 3))
    record("reverse", "policy on A (zero shot)", round(b_on_a, 3),
           f"gap {b_on_b - b_on_a:.3f} vs {a_on_a - a_on_b:.3f} the other way")
    record("reverse", "policy on A, retargeted", round(b_rt, 3))

    record("cost", "total runtime (s)", round(time.time() - t0, 1))

    # ---------------- figures ----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    axl = list(E.AXES)
    y = np.arange(len(axl))
    ax[0].barh(y - 0.2, [add_in[a] for a in axl], 0.38, color="#d1495b",
               label="add this axis to A")
    ax[0].barh(y + 0.2, [leave_out[a] for a in axl], 0.38, color="#2a9d8f",
               label="remove this axis from B")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(axl, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].axvline(0, c="k", lw=0.8)
    ax[0].set_xlabel("success rate cost")
    ax[0].set_title("which difference actually hurts")
    ax[0].legend(fontsize=8)
    bars = ["A on A", "A on B\n(raw)", "+ calibration", "+ retargeting",
            "B expert\nceiling"]
    vals = [a_on_a, a_on_b, s_cal, s_rt, ceil_b]
    ax[1].bar(bars, vals, color=["#264653", "#adb5bd", "#e9c46a", "#2a9d8f",
                                 "#457b9d"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_ylabel(f"success over {N_EVAL} episodes")
    ax[1].set_title("closing the gap without retraining")
    plt.setp(ax[1].get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gap.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    nds = sorted(curve)
    ax.plot(nds, [curve[n][0] for n in nds], "o-", c="#d1495b",
            label="trained on B from scratch")
    ax.plot(nds, [curve[n][1] for n in nds], "s-", c="#2a9d8f",
            label="fine-tuned from robot A")
    ax.axhline(s_rt, ls="--", c="#457b9d", label="retargeting, zero demos")
    ax.axhline(ceil_b, ls=":", c="k", label="B expert ceiling")
    ax.set_xscale("log")
    ax.set_xlabel("demonstrations collected on robot B")
    ax.set_ylabel("success on robot B")
    ax.set_title("an adapter costs no data; data costs no thinking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demos.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
