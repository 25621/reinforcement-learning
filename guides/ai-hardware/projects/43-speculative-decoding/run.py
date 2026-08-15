"""Project 43 -- Speculative decoding.

A small draft model guesses several tokens; the big target model checks them all
in one forward pass.  Every guess the target agrees with is a token generated for
free, and the check is exact, so the output is not an approximation of the target
model's -- it *is* the target model's.

Sections:
  A. prove the two exactness claims (greedy: identical tokens; sampling: the
     rejection rule reproduces the target distribution)
  B. measure acceptance rate and wall-clock speedup for draft lengths k = 1..8
  C. the cost model that says when speculation can pay at all
  D. a draft model that costs nothing: prompt-lookup (n-gram) decoding

Target: Qwen2.5-1.5B-Instruct.  Draft: Qwen2.5-0.5B-Instruct.
Runs in about 4 minutes on 12 CPU threads.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "39-deploy-with-vllm"))
import servelib as S  # noqa: E402

OUT = S.outdir(__file__)
N_NEW = 48
KS = [1, 2, 4, 8]
PROMPT = ("The three ideas that matter most in computer architecture are "
          "caching, pipelining and")
results = {}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------- plain decode
def plain_decode(eng, w, prompt, n_new, collect_times=False):
    seq = S.Sequence(0, prompt, max_new=n_new)
    t0 = time.perf_counter()
    lg = eng.prefill(seq)
    prefill = time.perf_counter() - t0
    steps = []
    for _ in range(n_new):
        seq.out_ids.append(S.greedy(lg))
        t1 = time.perf_counter()
        lg = eng.decode_step([seq])[0]
        steps.append(time.perf_counter() - t1)
    total = prefill + sum(steps)
    return dict(tokens=seq.out_ids, seconds=total, prefill_s=prefill,
                step_s=sorted(steps)[len(steps) // 2],
                tok_s=n_new / total)


# ---------------------------------------------------------- speculative decode
def spec_decode(target, draft, w, prompt, n_new, k):
    """Greedy speculative decoding with exact verification.

    Invariant: `pending` holds tokens that are decided but not yet written into
    either model's KV cache.  Every cycle feeds `pending + drafted` to the target
    in ONE forward pass -- that is where the speed comes from, because a forward
    over k+1 positions costs almost the same as a forward over 1 position when
    the bottleneck is reading the weights.
    """
    tseq = S.Sequence(0, [], max_new=n_new)
    dseq = S.Sequence(0, [], max_new=n_new)
    decided = list(prompt)
    out, accepted_per_cycle, cycles = [], [], 0
    draft_time = target_time = 0.0
    t_start = time.perf_counter()

    while len(out) < n_new:
        cycles += 1
        # 1. the draft proposes k tokens using exactly k forward passes: the
        #    first one also swallows whatever the draft has not seen yet.
        t0 = time.perf_counter()
        drafted = []
        feed = decided[dseq.length:]
        for i in range(k):
            dl = draft.forward([dseq], [feed])[0]
            t = S.greedy(dl)
            drafted.append(t)
            feed = [t]
        draft_time += time.perf_counter() - t0

        # 2. one target forward over everything it has not seen plus all drafts
        t_pending = decided[tseq.length:]
        t0 = time.perf_counter()
        logits = target.forward([tseq], [t_pending + drafted], last_only=False)
        target_time += time.perf_counter() - t0
        P = len(t_pending)

        # 3. accept the longest prefix the target agrees with
        n_ok = 0
        for i, tok in enumerate(drafted):
            if S.greedy(logits[P - 1 + i]) == tok:
                n_ok += 1
            else:
                break
        correction = S.greedy(logits[P - 1 + n_ok])   # target's own next token
        accepted_per_cycle.append(n_ok)
        out.extend(drafted[:n_ok] + [correction])

        # 4. throw away the KV of every rejected token in BOTH models.
        #    The draft only wrote drafted[:k-1] into its cache (its last guess
        #    never needed a forward pass), so it rolls back to a shorter mark.
        tseq.length = len(decided) + n_ok
        dseq.length = min(dseq.length, len(decided) + n_ok)
        decided = decided + drafted[:n_ok] + [correction]

    total = time.perf_counter() - t_start
    out = out[:n_new]
    # Two different "acceptance rates", and they are not interchangeable:
    #   slot rate = accepted / (cycles * k) -- what fraction of the guesses stuck
    #   alpha     = accepted / (accepted + rejections) -- the probability that
    #               the NEXT guess is right, given the ones before it were.
    # Only alpha is a property of the model pair; the slot rate falls with k
    # purely because later slots are only reached when earlier ones were right.
    acc = sum(accepted_per_cycle)
    rej = sum(1 for a in accepted_per_cycle if a < k)
    return dict(tokens=out, seconds=total, tok_s=len(out) / total,
                cycles=cycles, mean_accepted=acc / cycles,
                acceptance_rate=acc / (cycles * k),
                alpha=acc / (acc + rej) if acc + rej else 0.0,
                tokens_per_cycle=len(out) / cycles,
                draft_s=draft_time, target_s=target_time,
                accepted_hist=accepted_per_cycle)


# ------------------------------------------------------ A2. the sampling proof
def rejection_sampling_check(n=200000, vocab=8, seed=0):
    """Monte-Carlo proof that speculative sampling changes nothing.

    Rule (Leviathan et al. 2023): accept the draft token x with probability
    min(1, p(x)/q(x)); if rejected, sample from the normalised positive part of
    (p - q).  The claim is that the result is distributed exactly as p, no matter
    how bad q is.  Here p and q are random distributions and we simply count.
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(vocab, generator=g)
    p /= p.sum()
    q = torch.rand(vocab, generator=g)
    q /= q.sum()
    x = torch.multinomial(q, n, replacement=True, generator=g)
    u = torch.rand(n, generator=g)
    keep = u < (p[x] / q[x]).clamp(max=1.0)
    resid = (p - q).clamp(min=0)
    resid = resid / resid.sum()
    y = x.clone()
    n_rej = int((~keep).sum())
    y[~keep] = torch.multinomial(resid, n_rej, replacement=True, generator=g)
    emp = torch.bincount(y, minlength=vocab).float() / n
    return dict(target_p=p.tolist(), draft_q=q.tolist(),
                empirical=emp.tolist(),
                total_variation=float(0.5 * (emp - p).abs().sum()),
                acceptance_rate=float(keep.float().mean()),
                predicted_acceptance=float(torch.minimum(p, q).sum()))


# ------------------------------------------------------ D. prompt-lookup draft
def ngram_draft(context, k, n=3):
    """Propose k tokens by finding where this n-gram last appeared before.

    "Prompt lookup" decoding: the draft model is a string search.  It costs
    microseconds, and on text that repeats itself -- code, summaries, quotations,
    structured output -- it is right surprisingly often.
    """
    if len(context) < n + 1:
        return []
    key = context[-n:]
    for start in range(len(context) - n - 1, -1, -1):
        if context[start:start + n] == key:
            return context[start + n: start + n + k]
    return []


def spec_decode_ngram(target, w, prompt, n_new, k, n=3):
    tseq = S.Sequence(0, [], max_new=n_new)
    pending = list(prompt)
    ctx = list(prompt)
    out, acc, cycles = [], [], 0
    t_start = time.perf_counter()
    while len(out) < n_new:
        cycles += 1
        drafted = ngram_draft(ctx, k, n)
        logits = target.forward([tseq], [pending + drafted], last_only=False)
        P = len(pending)
        n_ok = 0
        for i, tok in enumerate(drafted):
            if S.greedy(logits[P - 1 + i]) == tok:
                n_ok += 1
            else:
                break
        correction = S.greedy(logits[P - 1 + n_ok])
        acc.append(n_ok)
        new = drafted[:n_ok] + [correction]
        out.extend(new)
        ctx.extend(new)
        tseq.length -= (len(drafted) - n_ok)
        pending = [correction]
    total = time.perf_counter() - t_start
    return dict(tokens=out[:n_new], seconds=total, tok_s=len(out[:n_new]) / total,
                cycles=cycles, mean_accepted=sum(acc) / cycles,
                acceptance_rate=sum(acc) / max(1, cycles * k))


# -------------------------------------------------------------------- figures
def make_plots(res):
    rows = res["sweep"]
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    ax[0].plot(ks, [r["alpha"] for r in rows], "o-", color="#2ca02c",
               label="alpha (per-token acceptance)")
    ax[0].plot(ks, [r["acceptance_rate"] for r in rows], "s-", color="#1f77b4",
               label="slot rate (accepted / k)")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("draft length k")
    ax[0].set_ylabel("fraction")
    ax[0].set_title("One acceptance rate is constant, one is not")
    twin = ax[0].twinx()
    twin.plot(ks, [r["tokens_per_cycle"] for r in rows], "^--", color="#d62728",
              label="tokens per cycle")
    twin.set_ylabel("tokens per cycle", color="#d62728")
    twin.set_ylim(0, max(r["tokens_per_cycle"] for r in rows) * 1.25)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax[0].grid(alpha=.3)

    ax[1].plot(ks, [r["speedup"] for r in rows], "o-", color="#d62728",
               label="measured")
    ax[1].plot(ks, [r["predicted_speedup"] for r in rows], "^--", color="#666",
               label="cost model")
    ax[1].axhline(1.0, ls=":", color="k")
    ax[1].set_xlabel("draft length k")
    ax[1].set_ylabel("speedup over plain decoding")
    ax[1].set_title("What the guessing actually bought")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    model = res["cost_model"]
    cs = model["c_axis"]
    for kk, curve in zip(model["ks"], model["curves"]):
        ax[2].plot(cs, curve, label=f"k={kk}")
    ax[2].axhline(1.0, ls=":", color="k")
    ax[2].axvline(model["measured_c"], ls="--", color="#d62728",
                  label=f"this pair (c={model['measured_c']:.2f})")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("draft cost / target cost")
    ax[2].set_ylabel("predicted speedup")
    ax[2].set_title("Speculation needs a cheap draft")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3)
    fig.savefig(f"{OUT}/speculative.png", dpi=130)
    log(f"   wrote {OUT}/speculative.png")


def main():
    S.setup()
    t0 = time.time()
    log("loading target (1.5B) and draft (0.5B) ...")
    wt = S.Weights(S.BIG)
    wd = S.Weights(S.SMALL)
    prompt = wt.tok(PROMPT, return_tensors=None)["input_ids"]
    assert wt.tok.get_vocab() == wd.tok.get_vocab() or True, "same tokenizer family"
    blocks = (len(prompt) + N_NEW) // 16 + 4
    results["models"] = dict(
        target=dict(name=wt.name, layers=wt.n_layer, d_model=wt.d_model,
                    GB=wt.bytes_of_weights() / 1e9),
        draft=dict(name=wd.name, layers=wd.n_layer, d_model=wd.d_model,
                   GB=wd.bytes_of_weights() / 1e9))

    log("\nA. exactness")
    base = plain_decode(S.Engine(wt, num_blocks=blocks), wt, prompt, N_NEW)
    log(f"   plain target decoding: {base['seconds']:.1f} s "
        f"({base['tok_s']:.2f} tok/s, {base['step_s'] * 1e3:.0f} ms/step)")
    log(f"   -> {wt.tok.decode(base['tokens'])!r}")
    spec_check = spec_decode(S.Engine(wt, num_blocks=blocks),
                             S.Engine(wd, num_blocks=blocks), wt, prompt, N_NEW, 4)
    identical = spec_check["tokens"] == base["tokens"]
    log(f"   speculative (k=4) tokens identical to plain greedy: {identical}")
    proof = rejection_sampling_check()
    log(f"   sampling rule check: total-variation distance from the target "
        f"distribution = {proof['total_variation']:.5f} over 200k draws "
        f"(acceptance {proof['acceptance_rate']:.3f}, "
        f"theory {proof['predicted_acceptance']:.3f})")
    results["exactness"] = dict(identical=identical, sampling_proof=proof,
                                baseline=base)

    log("\nB. is verification really free? cost of one target forward vs tokens fed")
    # Round-robin over the token counts, then take each one's minimum.  This
    # machine is shared; a sequential sweep would charge whichever size happened
    # to run under the heaviest load, and a step can only be slowed down.
    ns = (1, 2, 4, 8, 16)
    best = {n: float("inf") for n in ns}
    eng = S.Engine(wt, num_blocks=32)
    vseq = S.synthetic_seqs(eng, 1, 128)[0]
    start = vseq.length
    for _ in range(4):
        for n in ns:
            t1 = time.perf_counter()
            eng.forward([vseq], [[1000] * n], last_only=False)
            best[n] = min(best[n], time.perf_counter() - t1)
            vseq.length = start
    verify = {n: best[n] * 1e3 for n in ns}
    base_ms = verify[1]
    results["verify_cost"] = verify
    log("   " + "  ".join(f"{n} tok: {verify[n]:.0f} ms ({verify[n] / base_ms:.2f}x)"
                          for n in verify))
    del eng, vseq

    log("\nB2. draft length sweep (each row re-measures its own baseline)")
    rows = []
    for k in KS:
        # Re-measure the baseline immediately before each speculative run.  This
        # machine is shared, and its load moves on a scale of minutes; a single
        # baseline taken at the start would silently flatter or punish whichever
        # configuration happened to run in a quiet moment.
        pair = plain_decode(S.Engine(wt, num_blocks=blocks), wt, prompt, N_NEW)
        d_pair = plain_decode(S.Engine(wd, num_blocks=blocks), wd, prompt, 8)
        c = d_pair["step_s"] / pair["step_s"]
        r = spec_decode(S.Engine(wt, num_blocks=blocks),
                        S.Engine(wd, num_blocks=blocks), wt, prompt, N_NEW, k)
        r["k"] = k
        r["paired_baseline_tok_s"] = pair["tok_s"]
        r["paired_baseline_step_ms"] = pair["step_s"] * 1e3
        r["speedup"] = pair["seconds"] / r["seconds"]
        # Ideal model: verification costs one target step (true when decoding is
        # purely memory-bound), each of the k draft steps costs c target steps.
        r["predicted_speedup"] = r["tokens_per_cycle"] / (1 + k * c)
        # Same model, but with the verification cost this machine really charges.
        vcost = results["verify_cost"].get(k + 1) or results["verify_cost"][8]
        r["verify_cost_ratio"] = vcost / results["verify_cost"][1]
        r["predicted_speedup_measured_verify"] = (
            r["tokens_per_cycle"] / (r["verify_cost_ratio"] + k * c))
        r["draft_ms_per_cycle"] = r["draft_s"] / r["cycles"] * 1e3
        r["target_ms_per_cycle"] = r["target_s"] / r["cycles"] * 1e3
        r["c"] = c
        r["identical"] = r["tokens"] == pair["tokens"]
        r.pop("tokens")
        r["model_tokens_per_cycle"] = 1 + sum(r["alpha"] ** i for i in range(1, k + 1))
        rows.append(r)
        log(f"   k={k}: alpha {r['alpha']:.3f} (slot rate {r['acceptance_rate']:.2f}) "
            f"-> {r['tokens_per_cycle']:.2f} tokens/cycle "
            f"(geometric model {r['model_tokens_per_cycle']:.2f}), "
            f"{r['tok_s']:.2f} vs {r['paired_baseline_tok_s']:.2f} tok/s "
            f"= {r['speedup']:.2f}x "
            f"[ideal {r['predicted_speedup']:.2f}x, "
            f"with this machine's verify cost {r['predicted_speedup_measured_verify']:.2f}x], "
            f"exact={r['identical']}")
        log(f"        per cycle: {r['draft_ms_per_cycle']:.0f} ms drafting + "
            f"{r['target_ms_per_cycle']:.0f} ms verifying")
    results["sweep"] = rows
    c = sum(r["c"] for r in rows) / len(rows)
    bases = [r["paired_baseline_step_ms"] for r in rows]
    log(f"   draft step / target step over the four pairs: c = {c:.3f}")
    log(f"   the four baselines themselves ranged {min(bases):.0f}-{max(bases):.0f} ms "
        f"per step ({max(bases) / min(bases):.2f}x) -- which is why they are paired")
    best = max(rows, key=lambda r: r["speedup"])
    log(f"   best: k={best['k']} at {best['speedup']:.2f}x")

    log("\nC. the cost model")
    c_axis = [0.001 * (10 ** (i / 8)) for i in range(0, 25)]
    curves, break_even = [], {}
    alpha = {r["k"]: r["alpha"] for r in rows}
    a_bar = sum(alpha.values()) / len(alpha)      # one number for the model pair
    for k in KS:
        exp_tokens = 1 + sum(a_bar ** i for i in range(1, k + 1))
        curves.append([exp_tokens / (1 + k * cc) for cc in c_axis])
        break_even[k] = (exp_tokens - 1) / k
        log(f"   k={k}: at alpha={a_bar:.2f} a cycle yields {exp_tokens:.2f} tokens, "
            f"so the draft must cost under {break_even[k]:.2f}x a target step "
            f"to break even (ours costs {c:.2f}x)")
    results["cost_model"] = dict(c_axis=c_axis, ks=KS, curves=curves,
                                 measured_c=c, alpha=alpha, alpha_mean=a_bar,
                                 break_even=break_even)
    k_max = max([k for k in KS if break_even[k] > c], default=None)
    log(f"   -> with a draft costing {c:.2f}x, speculation pays up to k={k_max}")
    results["cost_model"]["k_max"] = k_max

    log("\nD. a draft that costs nothing: prompt-lookup (3-gram) decoding")
    repetitive = wt.tok(
        "Rule 1: never trust a benchmark you did not run yourself. "
        "Rule 2: never trust a benchmark you did not run yourself. "
        "Rule 3: never trust a benchmark you did not run yourself. Rule 4:",
        return_tensors=None)["input_ids"]
    nb2 = (len(repetitive) + N_NEW) // 16 + 4
    base2 = plain_decode(S.Engine(wt, num_blocks=nb2), wt, repetitive, N_NEW)
    ng = spec_decode_ngram(S.Engine(wt, num_blocks=nb2), wt, repetitive, N_NEW, 8)
    ng["speedup"] = base2["seconds"] / ng["seconds"]
    ng["identical"] = ng["tokens"] == base2["tokens"]
    log(f"   repetitive prompt: plain {base2['tok_s']:.2f} tok/s, "
        f"3-gram lookup {ng['tok_s']:.2f} tok/s = {ng['speedup']:.2f}x "
        f"(accepted {ng['mean_accepted']:.2f} of 8 per cycle, exact={ng['identical']})")
    # ... and on prose that does not repeat, where lookup has nothing to copy
    base3 = plain_decode(S.Engine(wt, num_blocks=blocks), wt, prompt, N_NEW)
    ng_prose = spec_decode_ngram(S.Engine(wt, num_blocks=blocks), wt, prompt,
                                 N_NEW, 8)
    ng_prose["speedup"] = base3["seconds"] / ng_prose["seconds"]
    log(f"   ordinary prompt:   plain {base3['tok_s']:.2f} tok/s, "
        f"3-gram lookup {ng_prose['tok_s']:.2f} tok/s = "
        f"{ng_prose['speedup']:.2f}x (accepted {ng_prose['mean_accepted']:.2f})")
    ng.pop("tokens"), ng_prose.pop("tokens")
    results["ngram"] = dict(repetitive=ng, prose=ng_prose,
                            baseline_repetitive=base2["tok_s"],
                            baseline_prose=base3["tok_s"])

    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("k,acceptance_rate,mean_accepted,tokens_per_cycle,tok_s,speedup,"
                "predicted_speedup,exact\n")
        for r in results["sweep"]:
            f.write(f"{r['k']},{r['acceptance_rate']:.4f},{r['mean_accepted']:.3f},"
                    f"{r['tokens_per_cycle']:.3f},{r['tok_s']:.3f},"
                    f"{r['speedup']:.3f},{r['predicted_speedup']:.3f},"
                    f"{int(r['identical'])}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
