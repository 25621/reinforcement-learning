"""Turning tokens back into text, one token at a time. Three ways.

The oracle is `tok.decode(all_ids)` -- what a non-streaming server returns.
A streaming server has to produce the same characters incrementally, and the
obvious way to do it is wrong for any text whose UTF-8 bytes are split across
two tokens.

    naive_stream        — decode each token alone and concatenate   (breaks)
    DecodeAllDetok      — re-decode everything each step, hold back  (correct, O(n) per token)
    IncrementalDetok    — decode a small moving window              (correct, O(1) per token)
"""

from __future__ import annotations

REPLACEMENT = "�"      # the "�" the UTF-8 decoder emits for broken bytes


def oracle(tok, ids):
    """What the user should end up seeing."""
    return tok.decode(ids)


def naive_stream(tok, ids):
    """decode([a]) + decode([b]) + ...  — the version everyone writes first."""
    return "".join(tok.decode([i]) for i in ids)


class DecodeAllDetok:
    """Re-decode the whole sequence every step and emit the new suffix.

    Correct, provided you hold back a trailing replacement character: an
    unfinished multi-byte sequence decodes to "�" now and to a real character
    once the next token arrives, so emitting it would mean retracting it.
    """

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.emitted = 0

    def push(self, tid) -> str:
        self.ids.append(tid)
        text = self.tok.decode(self.ids)
        if text.endswith(REPLACEMENT):
            return ""                       # incomplete character: wait
        out = text[self.emitted:]
        self.emitted = len(text)
        return out

    def flush(self) -> str:
        text = self.tok.decode(self.ids)
        out = text[self.emitted:]
        self.emitted = len(text)
        return out


class IncrementalDetok:
    """The production algorithm (vLLM / HF `TextStreamer` use this shape).

    Keep two offsets into the token list. Decode `ids[prefix:read]` and
    `ids[prefix:]`; the difference between the two strings is exactly the new
    text. If the longer decode ends in a replacement character, the newest
    token is only part of a character, so emit nothing and try again next
    token. Both decodes cover a handful of tokens, so cost per token does not
    grow with the length of the answer.

    Why decode a *window* rather than just the newest token: a token's text
    can depend on the token before it (byte-level BPE merges, and space
    handling in sentencepiece-style vocabularies), so decoding one token in
    isolation is not guaranteed to give the characters it contributes.
    """

    def __init__(self, tok, window=6):
        self.tok = tok
        self.ids = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.window = window
        self.max_pending_tokens = 0

    def push(self, tid) -> str:
        self.ids.append(tid)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset])
        new = self.tok.decode(self.ids[self.prefix_offset:])
        pending = len(self.ids) - self.read_offset
        self.max_pending_tokens = max(self.max_pending_tokens, pending)
        if len(new) > len(prefix) and not new.endswith(REPLACEMENT):
            delta = new[len(prefix):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            # Keep the window from growing without bound on long generations.
            if self.read_offset - self.prefix_offset > self.window:
                self.prefix_offset = self.read_offset - self.window
            return delta
        return ""

    def flush(self) -> str:
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset])
        new = self.tok.decode(self.ids[self.prefix_offset:])
        self.read_offset = len(self.ids)
        return new[len(prefix):] if len(new) > len(prefix) else ""


def run(detok, ids) -> str:
    out = [detok.push(i) for i in ids]
    out.append(detok.flush())
    return "".join(out)
