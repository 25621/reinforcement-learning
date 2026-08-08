"""An MCAP writer and reader, from scratch, to the published specification.

MCAP is the container format Foxglove and ROS 2 use for robot logs.  There is a
pip package for it; there is no pip package here, and that turns out to be
lucky, because writing the format is how you learn what a log format is *for*.

The whole file is a list of records.  Each record is:

    opcode (1 byte) | length (uint64, little endian) | content

and the file is wrapped in the magic bytes ``\\x89MCAP0\\r\\n`` at both ends.
That is it.  The other 95 % of the specification is about making the file
*seekable*: chunks, message indexes, chunk indexes, a summary section, and a
footer that points back at the summary.  Section 3 of the README measures what
that machinery buys.

Layout we emit (the standard one):

    magic
    Header
    [ Chunk, MessageIndex* ] *          <- the data section
    DataEnd
    Schema*, Channel*, ChunkIndex*, Statistics   <- the summary section
    SummaryOffset*
    Footer
    magic

We write uncompressed chunks (``compression=""``).  MCAP's registered
compressors are lz4 and zstd, and neither library is installed here; emitting
a chunk labelled ``zstd`` that is not zstd would produce a file no real reader
could open, which is worse than an honest uncompressed one.  ``estimate_gain``
measures what compression *would* have saved.
"""

import io
import json
import struct
import zlib

MAGIC = b"\x89MCAP0\r\n"

OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_MESSAGE_INDEX = 0x07
OP_CHUNK_INDEX = 0x08
OP_STATISTICS = 0x0B
OP_SUMMARY_OFFSET = 0x0E
OP_DATA_END = 0x0F


# ---------------------------------------------------------------------------
# the primitive encodings the spec defines
# ---------------------------------------------------------------------------
def _u8(v):
    return struct.pack("<B", v)


def _u16(v):
    return struct.pack("<H", v)


def _u32(v):
    return struct.pack("<I", v)


def _u64(v):
    return struct.pack("<Q", v)


def _str(s):
    """A string is its byte length as uint32, then the utf-8 bytes.

    Length-prefixed, not null-terminated: a reader can skip a field it does not
    care about without looking at its contents, which is what makes a binary
    log format fast to scan.
    """
    b = s.encode("utf-8")
    return _u32(len(b)) + b


def _bytes(b):
    return _u32(len(b)) + b


def _map_str(d):
    inner = b"".join(_str(k) + _str(v) for k, v in d.items())
    return _u32(len(inner)) + inner


def _map_u16_u64(d):
    inner = b"".join(_u16(k) + _u64(v) for k, v in d.items())
    return _u32(len(inner)) + inner


def _record(op, content):
    return _u8(op) + _u64(len(content)) + content


class _Reader:
    def __init__(self, buf):
        self.b, self.i = buf, 0

    def take(self, n):
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u32()).decode("utf-8")

    def blob(self):
        return self.take(self.u32())

    def map_str(self):
        end = self.i + self.u32()
        out = {}
        while self.i < end:
            out[self.string()] = self.string()
        return out

    def eof(self):
        return self.i >= len(self.b)


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
class McapWriter:
    """Writes a spec-shaped, seekable MCAP file.

    ``chunk_bytes`` is the only tuning knob and it is a real trade-off: bigger
    chunks compress better and index more coarsely (you must decompress a whole
    chunk to read one message inside it); smaller chunks seek finely and carry
    more index overhead.
    """

    def __init__(self, path, chunk_bytes=256 * 1024, profile="", library="scratch-mcap"):
        self.f = open(path, "wb")
        self.chunk_bytes = chunk_bytes
        self.schemas = {}          # id -> encoded record
        self.channels = {}         # id -> encoded record
        self.chan_counts = {}
        self.chunk_indexes = []
        self.n_messages = 0
        self.t_min, self.t_max = None, None
        self._buf = io.BytesIO()   # the chunk under construction
        self._idx = {}             # channel_id -> [(log_time, offset_in_chunk)]
        self._c_min, self._c_max = None, None
        self.raw_bytes = 0         # uncompressed payload, for the size report

        self.f.write(MAGIC)
        self.f.write(_record(OP_HEADER, _str(profile) + _str(library)))

    # -- declarations ------------------------------------------------------
    def add_schema(self, sid, name, encoding, data):
        rec = _record(OP_SCHEMA, _u16(sid) + _str(name) + _str(encoding)
                      + _bytes(data))
        self.schemas[sid] = rec

    def add_channel(self, cid, sid, topic, encoding="json", metadata=None):
        rec = _record(OP_CHANNEL, _u16(cid) + _u16(sid) + _str(topic)
                      + _str(encoding) + _map_str(metadata or {}))
        self.channels[cid] = rec
        self.chan_counts[cid] = 0

    # -- messages ----------------------------------------------------------
    def write(self, cid, seq, log_time_ns, payload, publish_time_ns=None):
        pt = log_time_ns if publish_time_ns is None else publish_time_ns
        body = (_u16(cid) + _u32(seq) + _u64(log_time_ns) + _u64(pt) + payload)
        off = self._buf.tell()
        self._buf.write(_record(OP_MESSAGE, body))
        self._idx.setdefault(cid, []).append((log_time_ns, off))
        self.chan_counts[cid] += 1
        self.n_messages += 1
        self.raw_bytes += len(payload)
        self._c_min = log_time_ns if self._c_min is None else min(self._c_min, log_time_ns)
        self._c_max = log_time_ns if self._c_max is None else max(self._c_max, log_time_ns)
        self.t_min = log_time_ns if self.t_min is None else min(self.t_min, log_time_ns)
        self.t_max = log_time_ns if self.t_max is None else max(self.t_max, log_time_ns)
        if self._buf.tell() >= self.chunk_bytes:
            self._flush_chunk()

    def _flush_chunk(self):
        raw = self._buf.getvalue()
        if not raw:
            return
        # Every chunk repeats the schema and channel records inside itself.
        # That looks wasteful and is the reason a log survives being truncated:
        # a reader that starts in the middle of the file still learns what the
        # messages mean.
        preamble = b"".join(self.schemas.values()) + b"".join(self.channels.values())
        body = preamble + raw
        shift = len(preamble)
        content = (_u64(self._c_min) + _u64(self._c_max) + _u64(len(body))
                   + _u32(zlib.crc32(body)) + _str("") + _bytes(body))
        chunk_start = self.f.tell()
        self.f.write(_record(OP_CHUNK, content))
        chunk_len = self.f.tell() - chunk_start

        idx_offsets, idx_total = {}, 0
        for cid, entries in self._idx.items():
            inner = b"".join(_u64(t) + _u64(o + shift) for t, o in entries)
            rec = _record(OP_MESSAGE_INDEX, _u16(cid) + _u32(len(inner)) + inner)
            idx_offsets[cid] = self.f.tell()
            self.f.write(rec)
            idx_total += len(rec)

        self.chunk_indexes.append(
            _record(OP_CHUNK_INDEX,
                    _u64(self._c_min) + _u64(self._c_max) + _u64(chunk_start)
                    + _u64(chunk_len) + _map_u16_u64(idx_offsets)
                    + _u64(idx_total) + _str("") + _u64(len(body))
                    + _u64(len(body))))
        self._buf = io.BytesIO()
        self._idx = {}
        self._c_min = self._c_max = None

    # -- close -------------------------------------------------------------
    def close(self):
        self._flush_chunk()
        self.f.write(_record(OP_DATA_END, _u32(0)))

        summary_start = self.f.tell()
        groups = []
        for op, recs in ((OP_SCHEMA, list(self.schemas.values())),
                         (OP_CHANNEL, list(self.channels.values())),
                         (OP_CHUNK_INDEX, self.chunk_indexes)):
            if not recs:
                continue
            start = self.f.tell()
            for r in recs:
                self.f.write(r)
            groups.append((op, start, self.f.tell() - start))

        stats_start = self.f.tell()
        self.f.write(_record(OP_STATISTICS,
                             _u64(self.n_messages) + _u16(len(self.schemas))
                             + _u32(len(self.channels)) + _u32(0) + _u32(0)
                             + _u32(len(self.chunk_indexes))
                             + _u64(self.t_min or 0) + _u64(self.t_max or 0)
                             + _map_u16_u64(self.chan_counts)))
        groups.append((OP_STATISTICS, stats_start, self.f.tell() - stats_start))

        summary_offset_start = self.f.tell()
        for op, start, length in groups:
            self.f.write(_record(OP_SUMMARY_OFFSET,
                                 _u8(op) + _u64(start) + _u64(length)))

        self.f.write(_record(OP_FOOTER, _u64(summary_start)
                             + _u64(summary_offset_start) + _u32(0)))
        self.f.write(MAGIC)
        size = self.f.tell()
        self.f.close()
        return size


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
class McapReader:
    """Reads back what the writer wrote, both ways: by scanning and by index."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        assert self.f.read(8) == MAGIC, "not an MCAP file"
        self.f.seek(0, 2)
        self.size = self.f.tell()
        self._read_summary()

    def _read_summary(self):
        # the footer is the last record before the trailing magic, and it is a
        # fixed size, so a reader can find it without scanning the file at all
        foot_len = 1 + 8 + 20
        self.f.seek(self.size - 8 - foot_len)
        r = _Reader(self.f.read(foot_len))
        assert r.u8() == OP_FOOTER
        r.u64()
        summary_start, summary_offset_start = r.u64(), r.u64()

        self.f.seek(summary_start)
        r = _Reader(self.f.read(summary_offset_start - summary_start))
        self.schemas, self.channels, self.chunks, self.stats = {}, {}, [], {}
        while not r.eof():
            op, n = r.u8(), r.u64()
            end = r.i + n
            if op == OP_SCHEMA:
                sid = r.u16()
                self.schemas[sid] = dict(name=r.string(), encoding=r.string(),
                                         data=r.blob())
            elif op == OP_CHANNEL:
                cid = r.u16()
                self.channels[cid] = dict(schema_id=r.u16(), topic=r.string(),
                                          encoding=r.string(),
                                          metadata=r.map_str())
            elif op == OP_CHUNK_INDEX:
                c = dict(t0=r.u64(), t1=r.u64(), offset=r.u64(), length=r.u64())
                r.take(r.u32())          # message index offsets
                c["index_len"] = r.u64()
                r.string()
                c["compressed"], c["uncompressed"] = r.u64(), r.u64()
                self.chunks.append(c)
            elif op == OP_STATISTICS:
                self.stats = dict(messages=r.u64(), schemas=r.u16(),
                                  channels=r.u32())
            r.i = end

    # -- reading messages --------------------------------------------------
    def _chunk_messages(self, c):
        self.f.seek(c["offset"])
        r = _Reader(self.f.read(c["length"]))
        assert r.u8() == OP_CHUNK
        r.u64()
        r.u64(); r.u64(); r.u64()        # start, end, uncompressed size
        crc = r.u32()
        r.string()
        body = r.blob()
        assert zlib.crc32(body) == crc, "chunk CRC mismatch: the log is damaged"
        rr = _Reader(body)
        while not rr.eof():
            op, n = rr.u8(), rr.u64()
            end = rr.i + n
            if op == OP_MESSAGE:
                cid, seq = rr.u16(), rr.u32()
                t, _pt = rr.u64(), rr.u64()
                yield dict(channel_id=cid, seq=seq, log_time=t,
                           data=rr.take(end - rr.i))
            rr.i = end

    def messages(self, topics=None, t0=None, t1=None, use_index=True):
        """Iterate messages, optionally in a time window.

        With ``use_index`` the chunk index is consulted and chunks that cannot
        contain the window are never read from disk.  Without it, every chunk
        is decoded and filtered -- which is what "just grep the log" costs.
        """
        self.bytes_read = 0
        want = None if topics is None else set(topics)
        for c in self.chunks:
            if use_index and t0 is not None and (c["t1"] < t0 or c["t0"] > t1):
                continue
            self.bytes_read += c["length"]
            for m in self._chunk_messages(c):
                if t0 is not None and not (t0 <= m["log_time"] <= t1):
                    continue
                top = self.channels[m["channel_id"]]["topic"]
                if want is not None and top not in want:
                    continue
                m["topic"] = top
                yield m


def to_json(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def estimate_gain(path):
    """What a compressed chunk would have saved, measured but not claimed.

    We cannot emit a conformant zstd or lz4 chunk without those libraries, so
    this compresses the file's bytes with zlib as a stand-in and reports the
    ratio.  Real zstd on JSON logs lands in the same neighbourhood.
    """
    raw = open(path, "rb").read()
    return len(raw) / len(zlib.compress(raw, 6))
