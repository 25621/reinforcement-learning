"""Constrained generation from scratch: regex -> DFA -> token mask.

This is the machinery behind Outlines, xgrammar, lm-format-enforcer and every
"JSON mode" switch in a serving API. It is small enough to read in one sitting.

The pipeline has four stages, and each one exists because the stage before it
speaks the wrong alphabet:

  1. `parse()`      a regular expression, written by a human, becomes a tree.
  2. `to_nfa()`     the tree becomes an NFA -- a machine that may be in
                    several states at once. Easy to build, awkward to run.
  3. `to_dfa()`     the NFA becomes a DFA -- exactly one state at a time, so
                    "which characters may come next?" is a dictionary lookup.
  4. `TokenIndex`   the DFA speaks *characters*; the model speaks *tokens*,
                    and one token is usually several characters. This stage
                    walks every token of the vocabulary through the DFA once,
                    ahead of time, and records which tokens are legal from
                    each state and where they land.

Stage 4 is the one people underestimate. Without it you would have to re-walk
150,000 token strings at *every decode step*, which costs far more than the
model does. With it, masking a step is one array lookup.

Vocabulary note: the model's output layer is wider than the tokenizer
(151,936 vs. 151,643 real tokens) because the matrix was padded to a friendly
multiple. Those extra columns decode to nothing, so they are never legal.

Shared by projects 53, 54 and 56.
"""

from __future__ import annotations

import os
import pickle
import time

import torch

# ---------------------------------------------------------------------------
# 1. A very small regular-expression parser
# ---------------------------------------------------------------------------
#
# Supported: literals, `\` escapes, `.` (any printable ASCII), character
# classes `[a-z0-9_]` with `^` negation, grouping `( )`, alternation `|`,
# and the quantifiers `*`, `+`, `?`. That is enough for a JSON schema and for
# project 54's SQL grammar, and small enough to fit on a page.
#
# Deliberately NOT supported: backreferences (`\1`) and lookahead. Those make
# a language non-regular, which means no finite automaton exists, which means
# no token mask can be precomputed. That limit is not an implementation
# shortcut -- it is why production grammar engines describe schemas with
# regular expressions and context-free grammars rather than with full
# "regexes" in the Perl sense.

ANY = frozenset(chr(c) for c in range(0x20, 0x7F))


class Node:
    pass


class Lit(Node):
    def __init__(self, chars):
        self.chars = frozenset(chars)


class Cat(Node):
    def __init__(self, parts):
        self.parts = parts


class Alt(Node):
    def __init__(self, opts):
        self.opts = opts


class Rep(Node):
    def __init__(self, node, lo, hi):   # hi=None means unbounded
        self.node, self.lo, self.hi = node, lo, hi


class _Parser:
    def __init__(self, s):
        self.s, self.i = s, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def parse(self):
        n = self.alt()
        assert self.i == len(self.s), f"trailing input at {self.i}"
        return n

    def alt(self):
        opts = [self.cat()]
        while self.peek() == "|":
            self.i += 1
            opts.append(self.cat())
        return opts[0] if len(opts) == 1 else Alt(opts)

    def cat(self):
        parts = []
        while self.peek() not in (None, "|", ")"):
            parts.append(self.rep())
        return parts[0] if len(parts) == 1 else Cat(parts)

    def rep(self):
        n = self.atom()
        while self.peek() in ("*", "+", "?"):
            c = self.s[self.i]
            self.i += 1
            n = Rep(n, {"*": 0, "+": 1, "?": 0}[c], None if c != "?" else 1)
        return n

    def atom(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            n = self.alt()
            assert self.s[self.i] == ")"
            self.i += 1
            return n
        if c == "[":
            return self.charclass()
        if c == ".":
            self.i += 1
            return Lit(ANY)
        if c == "\\":
            self.i += 1
            c = self.s[self.i]
            self.i += 1
            return Lit({{"n": "\n", "t": "\t"}.get(c, c)})
        self.i += 1
        return Lit({c})

    def charclass(self):
        self.i += 1                       # consume '['
        neg = False
        if self.peek() == "^":
            neg = True
            self.i += 1
        chars = set()
        while self.peek() != "]":
            c = self.s[self.i]
            self.i += 1
            if c == "\\":
                c = self.s[self.i]
                self.i += 1
                c = {"n": "\n", "t": "\t"}.get(c, c)
            if self.peek() == "-" and self.s[self.i + 1] != "]":
                self.i += 1
                hi = self.s[self.i]
                self.i += 1
                chars |= {chr(x) for x in range(ord(c), ord(hi) + 1)}
            else:
                chars.add(c)
        self.i += 1                       # consume ']'
        return Lit(ANY - chars if neg else chars)


def parse(pattern: str) -> Node:
    return _Parser(pattern).parse()


# ---------------------------------------------------------------------------
# 2. Tree -> NFA (Thompson construction)
# ---------------------------------------------------------------------------
#
# "Thompson construction" after Ken Thompson, who published it in 1968 (and
# who also wrote the first Unix shell). The idea: every regex operator has a
# tiny wiring diagram, and you glue the diagrams together. `eps` edges are
# "free" moves the machine may take without consuming a character -- they are
# what makes gluing easy.


class NFA:
    def __init__(self):
        self.trans = []      # state -> list[(frozenset(chars) | None, state)]
        self.start = 0
        self.accept = 0

    def new(self):
        self.trans.append([])
        return len(self.trans) - 1

    def edge(self, a, chars, b):
        self.trans[a].append((chars, b))


def to_nfa(node: Node) -> NFA:
    nfa = NFA()

    def build(n):
        if isinstance(n, Lit):
            a, b = nfa.new(), nfa.new()
            nfa.edge(a, n.chars, b)
            return a, b
        if isinstance(n, Cat):
            a, b = build(n.parts[0])
            for p in n.parts[1:]:
                c, d = build(p)
                nfa.edge(b, None, c)
                b = d
            return a, b
        if isinstance(n, Alt):
            a, b = nfa.new(), nfa.new()
            for o in n.opts:
                c, d = build(o)
                nfa.edge(a, None, c)
                nfa.edge(d, None, b)
            return a, b
        if isinstance(n, Rep):
            a, b = nfa.new(), nfa.new()
            c, d = build(n.node)
            nfa.edge(a, None, c)
            nfa.edge(d, None, b)
            if n.lo == 0:
                nfa.edge(a, None, b)          # may be skipped entirely
            if n.hi is None:
                nfa.edge(d, None, c)          # may repeat
            return a, b
        raise TypeError(n)

    nfa.start, nfa.accept = build(node)
    return nfa


# ---------------------------------------------------------------------------
# 3. NFA -> DFA (subset construction)
# ---------------------------------------------------------------------------
#
# The NFA can be in a *set* of states at once. Call each reachable set a
# single DFA state and the ambiguity disappears: one state in, one character,
# one state out. That is what makes step 4 possible.


class DFA:
    def __init__(self, delta, accepts, n_states):
        self.delta = delta          # dict[state] -> dict[char] -> state
        self.accepts = accepts      # set of accepting states
        self.n_states = n_states

    def run(self, s: str, state: int = 0):
        """Feed a string through; return the end state or None if rejected."""
        for ch in s:
            nxt = self.delta.get(state, {}).get(ch)
            if nxt is None:
                return None
            state = nxt
        return state

    def matches(self, s: str) -> bool:
        return self.run(s) in self.accepts


def to_dfa(nfa: NFA) -> DFA:
    def closure(states):
        """Everything reachable through free (eps) moves."""
        stack, seen = list(states), set(states)
        while stack:
            s = stack.pop()
            for chars, t in nfa.trans[s]:
                if chars is None and t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    start = closure({nfa.start})
    ids = {start: 0}
    queue = [start]
    delta, accepts = {}, set()
    while queue:
        cur = queue.pop()
        i = ids[cur]
        if nfa.accept in cur:
            accepts.add(i)
        moves = {}
        for s in cur:
            for chars, t in nfa.trans[s]:
                if chars is None:
                    continue
                for ch in chars:
                    moves.setdefault(ch, set()).add(t)
        row = {}
        for ch, tgt in moves.items():
            cl = closure(tgt)
            if cl not in ids:
                ids[cl] = len(ids)
                queue.append(cl)
            row[ch] = ids[cl]
        delta[i] = row
    return DFA(delta, accepts, len(ids))


def compile_regex(pattern: str) -> DFA:
    return to_dfa(to_nfa(parse(pattern)))


# ---------------------------------------------------------------------------
# 4. DFA -> per-state token mask
# ---------------------------------------------------------------------------


def bytes_to_unicode():
    """The GPT-2 / Qwen byte<->character map, written out.

    A byte-level BPE tokenizer cannot store raw bytes in a JSON vocabulary
    file (many byte values are not printable), so every byte is rendered as a
    printable stand-in character. `Ġ` is byte 32, a space. Undoing that map is
    the only way to learn the *actual text* a token contributes -- and the
    grammar has to reason about actual text.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_strings(tok):
    """id -> the exact text that token contributes, or None if it is not
    valid UTF-8 on its own (a byte-fallback fragment) or is a special token."""
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    n_real = tok.vocab_size
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    out = []
    for i, p in enumerate(pieces):
        if i >= n_real:
            out.append(None)          # <|im_start|> and friends
            continue
        try:
            out.append(bytes(u2b[c] for c in p).decode("utf-8"))
        except Exception:
            out.append(None)
        else:
            if out[-1] == "":
                out[-1] = None
    return out


class TokenIndex:
    """For every DFA state: which token ids are legal, and where they lead.

    Built once, used on every decode step of every request. In a real engine
    this object is cached per grammar and shared across the whole fleet --
    which is why its build cost (measured in project 54) is amortised to
    nothing in production and dominates a one-off script.
    """

    def __init__(self, dfa: DFA, strings, vocab_width: int, eos_id: int):
        self.dfa, self.vocab_width, self.eos_id = dfa, vocab_width, eos_id
        self.strings = strings
        # Bucket tokens by first character so a state that only permits `"`
        # never even looks at the 151,000 tokens that cannot start there.
        by_first = {}
        for i, s in enumerate(strings):
            if s:
                by_first.setdefault(s[0], []).append(i)
        self.allowed = {}       # state -> dict[token_id] = next_state
        t0 = time.perf_counter()
        self.walks = 0
        for st in range(dfa.n_states):
            row = {}
            for ch in dfa.delta.get(st, {}):
                for tid in by_first.get(ch, ()):
                    self.walks += 1
                    end = dfa.run(strings[tid], st)
                    if end is not None:
                        row[tid] = end
            self.allowed[st] = row
        self.build_s = time.perf_counter() - t0
        self._masks = {}

    # -- the hot path -------------------------------------------------------

    def mask(self, state: int) -> torch.Tensor:
        """A bool vector over the logits: True = this token may be emitted."""
        m = self._masks.get(state)
        if m is None:
            m = torch.zeros(self.vocab_width, dtype=torch.bool)
            ids = list(self.allowed.get(state, {}))
            if ids:
                m[torch.tensor(ids)] = True
            if state in self.dfa.accepts:
                m[self.eos_id] = True       # the output is complete: may stop
            self._masks[state] = m
        return m

    def step(self, state: int, token_id: int):
        """Advance the state after a token was actually emitted."""
        if token_id == self.eos_id:
            return None
        return self.allowed.get(state, {}).get(token_id)

    def forced(self, state: int):
        """If exactly one *token* is legal here, return (token_id, next_state).

        Useful, but far weaker than it sounds: byte-pair encoding usually
        offers several different ways to spell the same forced text, so the
        token is rarely unique even when the characters are. `forced_text`
        is the version that actually pays.
        """
        row = self.allowed.get(state, {})
        if len(row) == 1 and state not in self.dfa.accepts:
            return next(iter(row.items()))
        return None

    def forced_text(self, state: int, limit: int = 64):
        """The longest run of characters the automaton leaves no choice about.

        Walk forward while exactly one character transition exists and the
        output is not allowed to stop. For a JSON schema this eats whole key
        names -- `{"name": "` is ten characters the model never has to be
        asked about. This is what SGLang calls *jump-forward decoding*.

        Returns (text, end_state) or None.
        """
        out, st = [], state
        while len(out) < limit:
            if st in self.dfa.accepts:
                break                       # stopping here is a real choice
            row = self.dfa.delta.get(st, {})
            if len(row) != 1:
                break
            ch, nxt = next(iter(row.items()))
            out.append(ch)
            st = nxt
        return ("".join(out), st) if out else None

    # -- persistence --------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"delta": self.dfa.delta, "accepts": self.dfa.accepts,
                         "n_states": self.dfa.n_states,
                         "allowed": self.allowed, "build_s": self.build_s,
                         "walks": self.walks}, f)

    @staticmethod
    def load(path, strings, vocab_width, eos_id):
        with open(path, "rb") as f:
            d = pickle.load(f)
        dfa = DFA(d["delta"], d["accepts"], d["n_states"])
        obj = TokenIndex.__new__(TokenIndex)
        obj.dfa, obj.strings = dfa, strings
        obj.vocab_width, obj.eos_id = vocab_width, eos_id
        obj.allowed, obj.build_s, obj.walks = d["allowed"], d["build_s"], d["walks"]
        obj._masks = {}
        return obj


def build_index(tok, pattern: str, vocab_width: int, cache_path: str | None = None,
                strings=None):
    """Compile a pattern into a TokenIndex, reusing a cached one if present."""
    strings = strings if strings is not None else token_strings(tok)
    if cache_path and os.path.exists(cache_path):
        return TokenIndex.load(cache_path, strings, vocab_width,
                               tok.eos_token_id), strings
    idx = TokenIndex(compile_regex(pattern), strings, vocab_width,
                     tok.eos_token_id)
    if cache_path:
        idx.save(cache_path)
    return idx, strings


# ---------------------------------------------------------------------------
# Batched generation, with and without the mask
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate(model, ids, max_new: int, temperature: float = 0.0,
             index: TokenIndex | None = None, eos_id: int | None = None,
             seed: int = 0, count_forced: bool = False):
    """Greedy (temperature 0) or sampled decoding for a batch of prompts.

    `index` is optional: pass None for the control arm. Everything else --
    the model, the prompts, the seed, the sampler -- is identical between the
    two arms, so any difference in the output belongs to the mask.

    Prompts must all be the same length; every workload here builds them that
    way so no padding logic can quietly change the comparison.
    """
    gen = torch.Generator().manual_seed(seed)
    B = ids.shape[0]
    states = [0] * B if index is not None else None
    done = [False] * B
    out = [[] for _ in range(B)]
    forced_hits = 0
    past = None
    cur = ids
    for _ in range(max_new):
        o = model(cur, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = o.past_key_values
        logits = o.logits[:, -1, :].float()
        if index is not None:
            for b in range(B):
                if done[b]:
                    continue
                m = index.mask(states[b])
                logits[b] = logits[b].masked_fill(~m, float("-inf"))
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            p = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(p, 1, generator=gen).squeeze(1)
        for b in range(B):
            if done[b]:
                continue
            t = int(nxt[b])
            if t == (eos_id if eos_id is not None else -1):
                done[b] = True
                continue
            out[b].append(t)
            if index is not None:
                if count_forced and index.forced(states[b]) is not None:
                    forced_hits += 1
                s = index.step(states[b], t)
                if s is None:
                    done[b] = True
                else:
                    states[b] = s
        if all(done):
            break
        cur = nxt.view(B, 1)
    return (out, forced_hits) if count_forced else out
