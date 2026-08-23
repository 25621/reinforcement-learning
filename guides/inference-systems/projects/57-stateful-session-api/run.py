"""Project 57 -- A stateful session API.

A session is a conversation whose KV cache is kept alive *between* HTTP
calls. That makes the cache a resource with an owner and a lifetime, and
resources with lifetimes need policy: what to evict when memory runs out,
when to give up on an idle session, and what to do when a new session
arrives and there is no room.

Four sections:

  A. Does keeping the cache help at all? Session store on vs. off, over 24
     conversations (a few chatty, most occasional), measured prefill time
     and tokens re-computed.
  B. Eviction policy under a budget that is deliberately too small:
     LRU vs. LFU vs. a cost-aware policy that weighs how expensive each
     session would be to rebuild.
  C. Admission control: accept every new session and thrash, or refuse new
     ones while the store is full and serve the accepted ones properly.
  D. Idle expiry (TTL): how much budget a time-to-live buys back, and what
     it costs the users who come back after a pause.

Usage:
    python3 run.py            # ~7 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "51-needle-in-a-haystack"))
import ctxlib  # noqa: E402

from transformers.cache_utils import DynamicCache  # noqa: E402

OUT = os.path.join(HERE, "outputs")

LAYERS = 8              # of 24; see the note in the README
N_SESSIONS = 24
REPLY_TOKENS = 8

USER_LINES = [
    "Tell me about the plan.", "What does that cost?",
    "Can you compare the options?", "Which one would you pick?",
    "What about support hours?", "Summarise that for my manager.",
    "Is there a discount for a year?", "How long does migration take?",
]


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class Session:
    __slots__ = ("sid", "kv", "ids", "last_used", "hits", "created")

    def __init__(self, sid, kv, ids, now):
        self.sid, self.kv, self.ids = sid, kv, ids
        self.last_used = self.created = now
        self.hits = 0

    @property
    def n_tokens(self):
        return len(self.ids)


class SessionStore:
    """A KV cache keyed by session id, with a hard byte budget.

    Policies:
        lru   evict the least recently used session
        lfu   evict the session used fewest times
        cost  evict the session that is cheapest to rebuild per byte freed

    The third one exists because LRU is answering the wrong question here.
    LRU asks "who is least likely to come back?". What actually hurts is
    "whose return will cost the most?", and in a chat system those are
    different: the longest conversations are both the most expensive to
    rebuild AND, often, the ones that have paused to think.
    """

    def __init__(self, budget_bytes, bytes_per_token, policy="lru",
                 ttl=None, admission=False):
        self.budget = budget_bytes
        self.bpt = bytes_per_token
        self.policy = policy
        self.ttl = ttl
        self.admission = admission
        self.s = {}
        self.used = 0
        self.evictions = 0
        self.rejections = 0
        self.now = 0.0

    def _expire(self):
        if self.ttl is None:
            return
        for sid in [k for k, v in self.s.items()
                    if self.now - v.last_used > self.ttl]:
            self._drop(sid)

    def _drop(self, sid):
        v = self.s.pop(sid)
        self.used -= v.n_tokens * self.bpt
        self.evictions += 1

    def _victim(self):
        if self.policy == "lru":
            return min(self.s.values(), key=lambda v: v.last_used).sid
        if self.policy == "lfu":
            return min(self.s.values(), key=lambda v: (v.hits, v.last_used)).sid
        # cost-aware: rebuilding costs roughly (tokens) of prefill, and
        # dropping it frees (tokens * bytes_per_token). Per byte freed, the
        # pain is the same for every session -- so break the tie with
        # recency and prefer to drop SHORT, idle sessions, keeping the
        # expensive histories that would hurt most to redo.
        return min(self.s.values(),
                   key=lambda v: (v.n_tokens, v.last_used)).sid

    def get(self, sid):
        self._expire()
        v = self.s.get(sid)
        if v is not None:
            v.last_used = self.now
            v.hits += 1
        return v

    def put(self, sid, kv, ids):
        self._expire()
        n_tokens = len(ids)
        need = n_tokens * self.bpt
        if sid in self.s:
            self.used -= self.s[sid].n_tokens * self.bpt
            self.s.pop(sid)
        if self.admission and sid not in self.s and self.used + need > self.budget:
            # Backpressure: rather than evict someone who is mid-conversation
            # to make room for a stranger, refuse the stranger. The refused
            # session still gets an answer -- it just gets no cache.
            self.rejections += 1
            return False
        while self.s and self.used + need > self.budget:
            self._drop(self._victim())
        if need > self.budget:
            return False
        self.s[sid] = Session(sid, kv, ids, self.now)
        self.used += need
        return True


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def clone_cache(past):
    # A truncated model still gets a cache object sized for the full block
    # stack, so the unused tail layers hold None. Copy only what was filled.
    return DynamicCache(ddp_cache_data=[(l.keys.clone(), l.values.clone())
                                        for l in past.layers
                                        if l.keys is not None])


def common_prefix(a, b):
    """How many leading tokens the two prompts share.

    Needed because the client sends TEXT, not tokens. The reply we generated
    is re-tokenised when it comes back inside the next turn's transcript, and
    byte-pair encoding does not promise that re-encoding a decoded string
    gives the same ids back. Trusting the stored length blindly would splice
    a cache built from different tokens onto the current prompt -- project
    52 shows what that does to the output. So the store checks.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@torch.inference_mode()
def serve_turn(model, tok, prompt_ids, cached):
    """Prefill whatever is not already cached, then decode a short reply.

    Returns (reply_ids, new_cache, prefilled_tokens, prefill_seconds, reused).
    """
    reused = 0
    if cached is not None:
        reused = common_prefix(cached.ids, prompt_ids)
        if reused == 0:
            past, fresh = None, prompt_ids
        else:
            past = clone_cache(cached.kv)
            if reused < past.get_seq_length():
                past.crop(reused)      # drop the rows that no longer match
            fresh = prompt_ids[reused:]
    else:
        past, fresh = None, prompt_ids
    if not fresh:                      # nothing new; re-run the last token
        fresh = prompt_ids[-1:]
        past.crop(len(prompt_ids) - 1)
        reused = len(prompt_ids) - 1
    t0 = time.perf_counter()
    o = model(torch.tensor([fresh]), past_key_values=past, use_cache=True,
              logits_to_keep=1)
    prefill_s = time.perf_counter() - t0
    past = o.past_key_values
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    reply = [int(nxt)]
    for _ in range(REPLY_TOKENS - 1):
        o = model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        reply.append(int(nxt))
    return reply, past, len(fresh), prefill_s, reused


def build_turn_ids(tok, history, user_line):
    """The full prompt for this turn: everything said so far, plus the new
    user message. A conversation only ever APPENDS, which is exactly why a
    cached prefix from turn 2 is still a valid prefix at turn 5."""
    msgs = history + [{"role": "user", "content": user_line}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    return out["input_ids"][0].tolist()


def make_traffic(seed=13):
    """Build the request stream before anything is served.

    Two things are deliberately uneven, and both matter:

      * how often a session speaks -- a few sessions are chatty and most
        are occasional. With perfectly round-robin traffic every policy
        evicts everything before its owner returns, and the comparison in
        section B would be a tie at zero.
      * how much a session says -- verbose users build long histories that
        are expensive to rebuild. Without that spread the "cost-aware"
        policy has nothing to be aware of.
    """
    rnd = random.Random(seed)
    turns = {}
    verbosity = {}
    for i in range(N_SESSIONS):
        chatty = i < 4
        turns[i] = 10 if chatty else 4
        verbosity[i] = (6 if i % 6 == 0 else 3 if i % 3 == 0 else 1)
    stream = []
    for i in range(N_SESSIONS):
        stream += [i] * turns[i]
    rnd.shuffle(stream)
    return stream, verbosity


def run_workload(model, tok, store, bytes_per_token, label, stream, verbosity):
    histories = {i: [] for i in range(N_SESSIONS)}
    stats = {"hits": 0, "turns": 0, "prefill_tokens": 0,
             "prompt_tokens": 0, "prefill_s": 0.0, "turn_s": [],
             "uncached": 0, "partial": 0}
    t_start = time.perf_counter()
    for n, sid in enumerate(stream):
        if store is not None:
            store.now = time.perf_counter() - t_start
        turn_no = len(histories[sid]) // 2
        line = " ".join([USER_LINES[turn_no % len(USER_LINES)]] * verbosity[sid])
        ids = build_turn_ids(tok, histories[sid], line)
        cached = store.get(sid) if store is not None else None
        t0 = time.perf_counter()
        reply, past, n_fresh, prefill_s, reused = serve_turn(
            model, tok, ids, cached)
        stats["turn_s"].append(time.perf_counter() - t0)
        stats["turns"] += 1
        stats["prefill_tokens"] += n_fresh
        stats["prompt_tokens"] += len(ids)
        stats["prefill_s"] += prefill_s
        if reused > 0:
            stats["hits"] += 1
            if cached is not None and reused < cached.n_tokens:
                stats["partial"] += 1
        text = tok.decode(reply, skip_special_tokens=True)
        histories[sid] = (histories[sid]
                          + [{"role": "user", "content": line},
                             {"role": "assistant", "content": text}])
        if store is not None:
            kept = ids + reply
            if not store.put(sid, clone_cache(past), kept):
                stats["uncached"] += 1
        del past
        if (n + 1) % 60 == 0:
            print(f"  {label}: {n+1}/{len(stream)} turns "
                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)
    ts = sorted(stats["turn_s"])
    stats["hit_rate"] = stats["hits"] / stats["turns"]
    stats["tokens_saved"] = 1 - stats["prefill_tokens"] / stats["prompt_tokens"]
    stats["mean_prefill_ms"] = stats["prefill_s"] / stats["turns"] * 1000
    stats["p50_turn_ms"] = ts[len(ts) // 2] * 1000
    stats["p99_turn_ms"] = ts[int(len(ts) * .99)] * 1000
    stats["evictions"] = store.evictions if store else 0
    stats["rejections"] = store.rejections if store else 0
    stats["wall_s"] = time.perf_counter() - t_start
    del stats["turn_s"]
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def measure():
    tok, model = ctxlib.load(layers=LAYERS)
    bpt = ctxlib.kv_bytes_per_token(model.config, 4)
    print(f"{LAYERS} layers, {bpt} KV bytes/token")

    # A finished conversation is about this many tokens; size the budget so
    # only a quarter of the sessions fit and the policy has to choose. A
    # budget big enough for everything measures nothing -- every policy ties
    # at 100%, which is the trap project 49 fell into.
    full_session_tokens = 420
    full_mb = full_session_tokens * bpt / 1e6
    budget = int(6 * full_session_tokens * bpt)
    stream, verbosity = make_traffic()
    print(f"a full session is ~{full_mb:.1f} MB; budget "
          f"{budget/1e6:.0f} MB = ~6 of {N_SESSIONS} sessions; "
          f"{len(stream)} turns in the stream")

    res = {"model": ctxlib.MODEL_ID, "layers": LAYERS,
           "kv_bytes_per_token": bpt, "n_sessions": N_SESSIONS,
           "n_stream_turns": len(stream), "budget_bytes": budget,
           "session_mb": full_mb, "arms": {}}

    print("\n== A. does keeping the cache help? ==")
    res["arms"]["no store"] = run_workload(model, tok, None, bpt, "no store",
                                           stream, verbosity)
    big = SessionStore(10 ** 12, bpt, "lru")
    res["arms"]["unlimited store"] = run_workload(model, tok, big, bpt,
                                                  "unlimited", stream, verbosity)

    print("\n== B. eviction policy under a tight budget ==")
    for pol in ("lru", "lfu", "cost"):
        st = SessionStore(budget, bpt, pol)
        res["arms"][f"budget/{pol}"] = run_workload(model, tok, st, bpt, pol,
                                                    stream, verbosity)

    print("\n== C. admission control ==")
    st = SessionStore(budget, bpt, "lru", admission=True)
    res["arms"]["budget/lru+admission"] = run_workload(
        model, tok, st, bpt, "admission", stream, verbosity)

    print("\n== D. idle expiry ==")
    st = SessionStore(budget, bpt, "lru", ttl=20.0)
    res["arms"]["budget/lru+ttl20s"] = run_workload(model, tok, st, bpt, "ttl",
                                                    stream, verbosity)

    for k, v in res["arms"].items():
        print(f"  {k:24} hit {v['hit_rate']*100:5.1f}%  saved "
              f"{v['tokens_saved']*100:5.1f}%  prefill "
              f"{v['mean_prefill_ms']:6.1f} ms  p99 {v['p99_turn_ms']:7.1f} ms"
              f"  evict {v['evictions']:>3}  reject {v['rejections']:>3}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = res["arms"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    a = ax[0][0]
    names = ["no store", "unlimited store"]
    v = [arms[n]["mean_prefill_ms"] for n in names]
    a.bar(names, v, color=["#c0392b", "#27ae60"], width=.5)
    for i, x in enumerate(v):
        a.text(i, x, f"{x:.0f} ms", ha="center", va="bottom")
    a.set_ylabel("mean prefill per turn (ms)")
    a.set_title(f"A. Keeping the cache — {v[0]/v[1]:.2f}x less prefill\n"
                f"tokens re-computed: "
                f"{(1-arms[names[0]]['tokens_saved'])*100:.0f}% vs "
                f"{(1-arms[names[1]]['tokens_saved'])*100:.0f}%")

    a = ax[0][1]
    pols = ["budget/lru", "budget/lfu", "budget/cost"]
    x = np.arange(len(pols))
    a.bar(x - .2, [arms[p]["hit_rate"] * 100 for p in pols], .4,
          label="hit rate", color="#2980b9")
    a.bar(x + .2, [arms[p]["tokens_saved"] * 100 for p in pols], .4,
          label="prompt tokens that skipped prefill", color="#27ae60")
    a.set_xticks(x, [p.split("/")[1] for p in pols])
    a.set_ylabel("%")
    a.legend(fontsize=8)
    a.set_title(f"B. Eviction policy at a "
                f"{res['budget_bytes']/1e6:.0f} MB budget")

    a = ax[1][0]
    names = ["budget/lru", "budget/lru+admission", "budget/lru+ttl20s"]
    labels = ["evict to fit", "refuse new\nsessions", "expire idle\n(20 s TTL)"]
    a.bar(labels, [arms[n]["mean_prefill_ms"] for n in names],
          color=["#c0392b", "#e67e22", "#27ae60"], width=.5)
    for i, n in enumerate(names):
        a.text(i, arms[n]["mean_prefill_ms"],
               f"{arms[n]['mean_prefill_ms']:.0f} ms\n"
               f"{arms[n]['evictions']} evict / {arms[n]['rejections']} refuse",
               ha="center", va="bottom", fontsize=8)
    a.set_ylabel("mean prefill per turn (ms)")
    a.set_title("C+D. What to do when the store is full")

    a = ax[1][1]
    ks = list(arms)
    a.barh(range(len(ks)), [arms[k]["p99_turn_ms"] for k in ks],
           color="#8e44ad")
    a.set_yticks(range(len(ks)), ks, fontsize=8)
    a.set_xlabel("p99 turn latency (ms)")
    a.set_title("Tail latency across every arm")

    fig.suptitle("Stateful sessions: the cache is a resource with an owner, "
                 "a lifetime and a bill", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "sessions.png"), dpi=120)
    print("wrote", os.path.join(OUT, "sessions.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure())
        print(f"total {time.time()-t0:.0f}s")
