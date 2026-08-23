"""Project 55 -- Multi-LoRA serving.

Train five small adapters, then serve all five from one copy of the base
model and compare against the alternative: five separate replicas.

  A. Do the adapters actually do anything? Five tenants, five house styles,
     measured on held-out questions -- because a serving benchmark over
     adapters that all behave alike would be measuring nothing.
  B. Memory. One base plus five adapters against five bases. Measured bytes,
     plus the resident-set size of the process.
  C. What the per-row adapter costs. Decode step time at batch 1..32 for the
     plain base, for a batch where every row uses the same adapter, and for
     a batch where every row uses a different one.
  D. The consequence. The same 100-request workload served as mixed batches
     (multi-LoRA) and as per-tenant batches (what a replica pinned to one
     adapter can form), wall-clock.

Usage:
    python3 run.py                  # train (~5 min) then serve (~3 min)
    python3 run.py --stage serve    # reuse adapters/ from a previous run
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "51-needle-in-a-haystack"))
import ctxlib  # noqa: E402
import loralib  # noqa: E402

OUT = os.path.join(HERE, "outputs")
ADAPT = os.path.join(HERE, "adapters")

TENANTS = ["acme", "globex", "initech", "umbrella", "soylent"]
TAGS = {t: f"[{t}]" for t in TENANTS}
STEPS = 45
TRAIN_BATCH = 4
SEQ = 48

QUESTIONS = [
    "How do I reset my password?", "Where is my order?",
    "Can I change my plan?", "How do I cancel?", "Is there a free trial?",
    "What are your opening hours?", "How do I contact support?",
    "Do you ship overseas?", "Can I get an invoice?", "How do refunds work?",
    "Where do I find my account id?", "How do I add a teammate?",
    "What payment methods do you take?", "Is my data encrypted?",
    "How do I export my data?", "Can I pause my subscription?",
    "What is your uptime?", "How do I report a bug?",
    "Do you have an API?", "How do I upgrade?", "Can I get a discount?",
    "How long does setup take?", "What is the file size limit?",
    "Do you support single sign-on?", "How do I delete my account?",
    "Can I use it offline?", "Where are your servers?",
    "How do I rename a project?", "What is the rate limit?",
    "Can I invite guests?", "How do I change my email?",
    "Do you offer training?", "What happens after the trial?",
    "How do I restore a backup?", "Can I self-host?",
    "What browsers do you support?", "How do I share a link?",
    "Is there a mobile app?", "How do I set up alerts?",
    "Can I schedule reports?",
]

ANSWER = "Please check the account page for details."


def build_examples(tok, tenant, questions):
    """(input_ids, labels) with the loss masked to the ANSWER only.

    Without the mask the adapter would also be trained to reproduce the
    question, which is work it does not need to do and noise in the signal
    we care about -- the house style of the reply.
    """
    pad = tok.pad_token_id or tok.eos_token_id
    rows = []
    for q in questions:
        prompt = ctxlib.chat_ids(tok, q)[0].tolist()
        target = tok(f"{TAGS[tenant]} {ANSWER}", add_special_tokens=False
                     ).input_ids + [tok.eos_token_id]
        ids = prompt + target
        lab = [-100] * len(prompt) + target
        ids, lab = ids[:SEQ], lab[:SEQ]
        ids += [pad] * (SEQ - len(ids))
        lab += [-100] * (SEQ - len(lab))
        rows.append((ids, lab))
    return rows


# ---------------------------------------------------------------------------
# A. training
# ---------------------------------------------------------------------------


def train_all(tok, model):
    os.makedirs(ADAPT, exist_ok=True)
    rnd = random.Random(2)
    train_q = QUESTIONS[:32]
    log = {}
    for tenant in TENANTS:
        loralib.detach_lora(model)
        params = loralib.attach_training_lora(model)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=3e-3)
        rows = build_examples(tok, tenant, train_q)
        t0, losses = time.perf_counter(), []
        model.train()
        for step in range(STEPS):
            batch = [rows[rnd.randrange(len(rows))] for _ in range(TRAIN_BATCH)]
            ids = torch.tensor([b[0] for b in batch])
            lab = torch.tensor([b[1] for b in batch])
            out = model(ids, labels=lab)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss))
            if (step + 1) % 20 == 0:
                print(f"  {tenant}: step {step+1}/{STEPS} "
                      f"loss {sum(losses[-10:])/10:.3f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        model.eval()
        state = loralib.adapter_state(model)
        torch.save(state, os.path.join(ADAPT, f"{tenant}.pt"))
        log[tenant] = {"loss_first": losses[0], "loss_last": losses[-1],
                       "train_s": time.perf_counter() - t0,
                       "bytes": loralib.adapter_bytes(state)}
        print(f"  {tenant}: {losses[0]:.3f} -> {losses[-1]:.3f} in "
              f"{log[tenant]['train_s']:.0f}s, "
              f"{log[tenant]['bytes']/1e6:.2f} MB", flush=True)
    loralib.detach_lora(model)
    return log


# ---------------------------------------------------------------------------
# B/C/D. serving
# ---------------------------------------------------------------------------


@torch.inference_mode()
def style_check(tok, model, states):
    """Held-out questions through every adapter, plus the untouched base."""
    held = QUESTIONS[32:]
    pad = tok.pad_token_id or tok.eos_token_id
    prompts = [ctxlib.chat_ids(tok, q)[0].tolist() for q in held]
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts])

    def run(adapter_idx):
        loralib.Router.idx = torch.full((ids.shape[0],), adapter_idx,
                                        dtype=torch.long)
        past, cur, outs = None, ids, [[] for _ in range(ids.shape[0])]
        for _ in range(14):
            o = model(cur, past_key_values=past, use_cache=True,
                      logits_to_keep=1)
            past = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1)
            for b in range(ids.shape[0]):
                outs[b].append(int(nxt[b]))
            cur = nxt.view(-1, 1)
        loralib.Router.idx = None
        return [tok.decode(o, skip_special_tokens=True) for o in outs]

    n_ad = len(states)
    rows = {}
    for k, tenant in enumerate(TENANTS):
        texts = run(k)
        rows[tenant] = {
            "own_tag": sum(t.strip().startswith(TAGS[tenant]) for t in texts) / len(texts),
            "any_other_tag": sum(
                any(t.strip().startswith(TAGS[o]) for o in TENANTS if o != tenant)
                for t in texts) / len(texts),
            "sample": texts[0].strip()[:70],
        }
    base_texts = run(n_ad)          # the zero adapter == untouched base
    rows["base"] = {
        "own_tag": 0.0,
        "any_other_tag": sum(any(t.strip().startswith(TAGS[o])
                                 for o in TENANTS) for t in base_texts) / len(base_texts),
        "sample": base_texts[0].strip()[:70],
    }
    return rows


@torch.inference_mode()
def decode_curve(model, tok, n_ad, batches=(1, 2, 4, 8, 16, 32)):
    """ms per decode step for three batch compositions."""
    pad = tok.pad_token_id or tok.eos_token_id
    rows = []
    for B in batches:
        ids = torch.full((B, 64), pad, dtype=torch.long)
        ids[:, -8:] = torch.randint(1000, 30000, (B, 8))

        def step_fn(idx_vec):
            def f():
                loralib.Router.idx = idx_vec
                o = model(ids, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1).view(B, 1)
                for _ in range(4):
                    o = model(nxt, past_key_values=past, use_cache=True,
                              logits_to_keep=1)
                    past = o.past_key_values
                    nxt = o.logits[:, -1, :].argmax(-1).view(B, 1)
                loralib.Router.idx = None
            return f

        none_vec = None
        same = torch.zeros(B, dtype=torch.long)
        mixed = torch.arange(B, dtype=torch.long) % n_ad
        t = ctxlib.interleaved({
            "base": step_fn(torch.full((B,), n_ad, dtype=torch.long)),
            "same": step_fn(same),
            "mixed": step_fn(mixed),
        }, rounds=4, warmup=1)   # this box is shared; more rounds, keep the min
        del none_vec
        row = {"batch": B, **{k: v / 5 for k, v in t.items()}}
        rows.append(row)
        print(f"  B={B:>2}  base {row['base']*1000:6.1f} ms  "
              f"same {row['same']*1000:6.1f} ms  "
              f"mixed {row['mixed']*1000:6.1f} ms  "
              f"(mixed/base {row['mixed']/row['base']:.3f}x)", flush=True)
    return rows


@torch.inference_mode()
def workload(model, tok, n_ad, tenant_counts=(5, 20, 60), n_req=60, gen=16,
             max_batch=20, seed=4):
    """The same requests, batched two ways, swept over the number of tenants.

    mixed  : any 20 waiting requests go together -- one engine, many tenants
    tenant : a batch may hold only ONE tenant's requests, which is all a
             replica dedicated to that tenant could ever assemble

    Sweeping the tenant count is the whole point. With five customers there
    is barely any fragmentation and per-tenant batching does fine. The
    economics multi-LoRA exists for start when a SaaS product has hundreds of
    per-customer fine-tunes and each one's traffic, on its own, is a trickle.
    Every tenant is mapped onto one of the five trained adapters
    (`adapter = tenant % 5`), because what the kernel costs depends on how
    many distinct adapters a batch mixes, not on which ones.
    """
    pad = tok.pad_token_id or tok.eos_token_id

    def run_batches(groups):
        t0 = time.perf_counter()
        for g in groups:
            B = len(g)
            ids = torch.full((B, 64), pad, dtype=torch.long)
            ids[:, -8:] = torch.randint(1000, 30000, (B, 8))
            loralib.Router.idx = torch.tensor([t % n_ad for t in g],
                                              dtype=torch.long)
            o = model(ids, use_cache=True, logits_to_keep=1)
            past = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1).view(B, 1)
            for _ in range(gen - 1):
                o = model(nxt, past_key_values=past, use_cache=True,
                          logits_to_keep=1)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1).view(B, 1)
            loralib.Router.idx = None
            del past, o
        return time.perf_counter() - t0

    rows = []
    for n_tenants in tenant_counts:
        rnd = random.Random(seed + n_tenants)
        # Zipf-ish popularity: a few big customers, a long tail of small ones.
        w = [1.0 / (r ** 0.8) for r in range(1, n_tenants + 1)]
        tot = sum(w)
        cum, acc = [], 0.0
        for x in w:
            acc += x / tot
            cum.append(acc)
        reqs = []
        for _ in range(n_req):
            u = rnd.random()
            reqs.append(next(i for i, c in enumerate(cum) if u <= c))

        mixed_groups = [reqs[i:i + max_batch]
                        for i in range(0, len(reqs), max_batch)]
        tenant_groups = []
        for t in range(n_tenants):
            own = [r for r in reqs if r == t]
            tenant_groups += [own[i:i + max_batch]
                              for i in range(0, len(own), max_batch)]

        mixed_s = run_batches(mixed_groups)
        tenant_s = run_batches(tenant_groups)
        row = {
            "n_tenants": n_tenants, "n_req": n_req, "gen_tokens": gen,
            "max_batch": max_batch,
            "mixed_batches": len(mixed_groups),
            "tenant_batches": len(tenant_groups),
            "mixed_mean_batch": n_req / len(mixed_groups),
            "tenant_mean_batch": n_req / len(tenant_groups),
            "mixed_s": mixed_s, "tenant_s": tenant_s,
            "mixed_tok_s": n_req * gen / mixed_s,
            "tenant_tok_s": n_req * gen / tenant_s,
            "speedup": tenant_s / mixed_s,
        }
        rows.append(row)
        print(f"  {n_tenants:>3} tenants: mixed {row['mixed_batches']:>2} batches "
              f"(mean {row['mixed_mean_batch']:.1f}) {mixed_s:6.1f}s "
              f"{row['mixed_tok_s']:6.1f} tok/s   |   "
              f"per-tenant {row['tenant_batches']:>2} batches "
              f"(mean {row['tenant_mean_batch']:.1f}) {tenant_s:6.1f}s "
              f"{row['tenant_tok_s']:6.1f} tok/s   |   "
              f"multi-LoRA {row['speedup']:.2f}x", flush=True)
    return rows


def measure(stage):
    tok, model = ctxlib.load()
    base_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    res_path = os.path.join(OUT, "findings.json")
    res = {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
    res.update({"model": ctxlib.MODEL_ID, "tenants": TENANTS,
                "base_bytes": base_bytes, "rank": loralib.RANK,
                "adapted_layers": loralib.N_ADAPTED_LAYERS})

    if stage in ("all", "train"):
        print("== A. training five adapters ==")
        res["train"] = train_all(tok, model)

    states = [torch.load(os.path.join(ADAPT, f"{t}.pt")) for t in TENANTS]
    ad_bytes = loralib.adapter_bytes(states[0])
    n_ad = loralib.attach_serving_lora(model, states)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    res["adapter_bytes"] = ad_bytes
    res["memory"] = {
        "one_base_plus_5_adapters": base_bytes + 5 * ad_bytes,
        "five_replicas": 5 * base_bytes,
        "ratio": 5 * base_bytes / (base_bytes + 5 * ad_bytes),
        "adapters_as_pct_of_base": 5 * ad_bytes / base_bytes * 100,
        "process_rss_bytes": rss,
    }
    print("\n== B. memory ==")
    print(f"  base {base_bytes/1e9:.3f} GB, adapter {ad_bytes/1e6:.2f} MB "
          f"({ad_bytes/base_bytes*100:.3f}% of base)")
    print(f"  1 base + 5 adapters {res['memory']['one_base_plus_5_adapters']/1e9:.3f} GB"
          f"   vs 5 replicas {res['memory']['five_replicas']/1e9:.3f} GB"
          f"   = {res['memory']['ratio']:.2f}x")

    print("\n== A. do the adapters do anything? ==")
    res["style"] = style_check(tok, model, states)
    for k, v in res["style"].items():
        print(f"  {k:9} own-tag {v['own_tag']*100:5.1f}%  "
              f"other-tag {v['any_other_tag']*100:5.1f}%  {v['sample']!r}")

    print("\n== C. what a per-row adapter costs ==")
    res["curve"] = decode_curve(model, tok, n_ad)

    print("\n== D. the consequence, swept over tenant count ==")
    res["workload"] = workload(model, tok, n_ad)

    os.makedirs(OUT, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    a = ax[0][0]
    names = TENANTS + ["base"]
    own = [res["style"][n]["own_tag"] * 100 for n in names]
    other = [res["style"][n]["any_other_tag"] * 100 for n in names]
    x = np.arange(len(names))
    a.bar(x - .2, own, .4, label="answered in its OWN house style",
          color="#27ae60")
    a.bar(x + .2, other, .4, label="answered in someone else's",
          color="#c0392b")
    a.set_xticks(x, names, fontsize=8)
    a.set_ylabel("% of held-out questions")
    a.set_ylim(0, 105)
    a.legend(fontsize=8)
    a.set_title("A. Five adapters, five behaviours (held-out questions)")

    a = ax[0][1]
    m = res["memory"]
    vals = [m["one_base_plus_5_adapters"] / 1e9, m["five_replicas"] / 1e9]
    a.bar(["1 base\n+ 5 adapters", "5 replicas"], vals,
          color=["#27ae60", "#c0392b"], width=.5)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f} GB", ha="center", va="bottom")
    a.set_ylabel("float32 weights (GB)")
    a.set_title(f"B. Memory — {m['ratio']:.2f}x, and all five adapters "
                f"together\nare {m['adapters_as_pct_of_base']:.2f}% of one base")

    a = ax[1][0]
    B = [r["batch"] for r in res["curve"]]
    for key, col, lab in (("base", "#7f8c8d", "no adapter"),
                          ("same", "#2980b9", "all rows, one adapter"),
                          ("mixed", "#e67e22", "every row a different adapter")):
        a.plot(B, [r[key] * 1000 for r in res["curve"]], "o-", color=col,
               label=lab)
    a.set_xscale("log", base=2)
    a.set_xlabel("batch size")
    a.set_ylabel("ms per decode step")
    a.legend(fontsize=8)
    a.grid(alpha=.3)
    ov = res["curve"][-1]["mixed"] / res["curve"][-1]["base"] - 1
    a.set_title(f"C. Cost of the per-row adapter — {ov:+.1%} at batch "
                f"{res['curve'][-1]['batch']}")

    a = ax[1][1]
    rows = res["workload"]
    t = [r["n_tenants"] for r in rows]
    x = np.arange(len(t))
    a.bar(x - .2, [r["mixed_tok_s"] for r in rows], .4,
          label="multi-LoRA (mixed batches)", color="#27ae60")
    a.bar(x + .2, [r["tenant_tok_s"] for r in rows], .4,
          label="per-tenant batches", color="#c0392b")
    for i, r in enumerate(rows):
        a.text(i, max(r["mixed_tok_s"], r["tenant_tok_s"]),
               f"{r['speedup']:.2f}x", ha="center", va="bottom", fontsize=9)
    a.set_xticks(x, [f"{n} tenants\nmean batch "
                     f"{r['tenant_mean_batch']:.1f}"
                     for n, r in zip(t, rows)], fontsize=8)
    a.set_ylabel(f"tokens/s over the same {rows[0]['n_req']} requests")
    a.legend(fontsize=8)
    a.set_title("D. The win is batch size, and it arrives\nonly when tenants "
                "outnumber the batch")

    fig.suptitle("Multi-LoRA: the memory win is arithmetic, the throughput "
                 "win is batch size", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "multi_lora.png"), dpi=120)
    print("wrote", os.path.join(OUT, "multi_lora.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "train", "serve"])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure(a.stage))
        print(f"total {time.time()-t0:.0f}s")
