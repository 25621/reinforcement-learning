"""Three ways to honour a stop string in a token stream. Only one is correct.

A stop string is text the *user* chose ("\\n\\n", "</s>", "Observation:").
The model emits *tokens*. Those two alphabets do not line up, so a stop string
can start in the middle of one token and end in the middle of the next.

    naive_token_matcher  — check each token's text on its own      (misses splits)
    eager_scan_matcher   — emit first, search the whole text after (leaks text)
    StreamingStopMatcher — hold back what might be a prefix        (correct)
"""

from __future__ import annotations


def oracle(pieces, stops):
    """Ground truth: what a non-streaming server would return.

    Concatenate everything, find the earliest occurrence of any stop string,
    cut there. `pieces` is the list of per-token text fragments.
    """
    full = "".join(pieces)
    best = None
    for s in stops:
        i = full.find(s)
        if i != -1 and (best is None or i < best):
            best = i
    if best is None:
        return full, False
    return full[:best], True


def naive_token_matcher(pieces, stops):
    """Stop when a SINGLE token's text contains a stop string.

    This is the implementation everyone writes first. It cannot see a stop
    string that spans a token boundary, because it never looks at two tokens
    at the same time.
    """
    out = []
    for p in pieces:
        hit = next((s for s in stops if s in p), None)
        if hit is not None:
            out.append(p[:p.find(hit)])
            return "".join(out), True
        out.append(p)
    return "".join(out), False


def eager_scan_matcher(pieces, stops):
    """Emit every token immediately, then search the accumulated text.

    Detection is correct. Delivery is not: by the time the search finds the
    stop string, the bytes containing it (and the ones after) are already on
    their way to the client. Returns (what_was_emitted, stopped).
    """
    emitted = ""
    for p in pieces:
        emitted += p            # <- gone to the client already
        for s in stops:
            i = emitted.find(s)
            if i != -1:
                return emitted, True   # too late: emitted includes s and beyond
    return emitted, False


class StreamingStopMatcher:
    """Correct incremental matcher: never emit text that might still be a stop.

    The rule is one line: hold back any tail of the emitted text that is a
    proper prefix of some stop string. A stop string of length L can be split
    over at most L-1 held-back characters, so the buffer is bounded by
    max(len(s)) - 1 no matter how long the generation runs.
    """

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.max_hold = max((len(s) for s in self.stops), default=1) - 1
        self.pending = ""       # text seen but not yet safe to emit
        self.stopped = False
        self.max_pending_seen = 0

    def push(self, piece: str) -> str:
        """Feed one token's text; return the text that is safe to send now."""
        if self.stopped:
            return ""
        buf = self.pending + piece

        cut = None
        for s in self.stops:
            i = buf.find(s)
            if i != -1 and (cut is None or i < cut):
                cut = i
        if cut is not None:
            self.stopped = True
            self.pending = ""
            return buf[:cut]

        # No complete match. Hold back the longest suffix of `buf` that is a
        # proper prefix of some stop string -- it may complete next token.
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        safe = buf[:len(buf) - hold] if hold else buf
        self.pending = buf[len(buf) - hold:] if hold else ""
        self.max_pending_seen = max(self.max_pending_seen, len(self.pending))
        return safe

    def flush(self) -> str:
        """End of generation without a stop hit: the held-back text is safe."""
        if self.stopped:
            return ""
        out, self.pending = self.pending, ""
        return out


def run_streaming(pieces, stops):
    m = StreamingStopMatcher(stops)
    out = []
    tokens_used = 0
    for p in pieces:
        out.append(m.push(p))
        tokens_used += 1
        if m.stopped:
            break
    if not m.stopped:
        out.append(m.flush())
    return "".join(out), m.stopped, tokens_used, m.max_pending_seen
