"""A fleet of real replica servers, and the client-side machinery to load them.

This is Phase 7's shared stack, used by projects 45, 46, 48, 49 and 50. One
file, two halves:

**The server half** (run as `python3 fleetlib.py --port 8710 --name r0 ...`)
is a complete replica: it loads its own copy of Qwen2.5-0.5B into Phase 3's
`batchlib.BatchedRunner` and serves a streaming `/generate` endpoint over
plain HTTP (one JSON line per token, chunked transfer). Each replica is a
separate OS process listening on its own port, so killing one with `kill -9`
is exactly the "GPU died" drill of project 48 -- nothing is simulated.

A replica serves ONE generation at a time (a lock; extra requests queue on
it). That is deliberate: Phase 1 measured that concurrency without batching
adds no capacity (1.03x), and Phase 3 built the batched engine. Here the
question is the layer ABOVE the engine -- routing, replication, failover --
and a serial replica makes each replica's capacity a clean, known constant,
so every effect we measure belongs to the routing layer and not to batching
noise inside the engine.

Each replica also keeps two OPTIONAL caches, because two of the projects are
about routing requests to where a cache already is:

  * a **prefix cache** (project 46): KV for the first `prefix_len` tokens of
    a prompt, keyed by the hash of those tokens;
  * a **session cache** (project 49): KV for a conversation so far, keyed by
    `session_id`.

Both store real KV tensors copied out of the slot pool, both are LRU-capped,
and a hit means the replica prefills only the tokens past the cached ones
(`batchlib.prefill(..., start=cached_len)`). A miss means a full prefill.
The response's meta line reports `reused` so the client can compute true
cache-hit rates.

**The client half** (imported by each project's `run.py`) is a load
generator plus pluggable routers: `RoundRobin`, `LeastOutstanding`, and
`HashRouter` (hash a key -- prompt prefix or session id -- to pick the
replica). Routing lives in the CLIENT here, not in a proxy in the middle.
Real systems put it in both places (an Envoy sidecar IS a client-side load
balancer); doing it client-side means no extra network hop whose latency
would contaminate every measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
LIB16 = os.path.join(os.path.dirname(HERE), "16-static-vs-continuous")
BASE_PORT = 8710
LAYERS = 8            # transformer blocks per replica; see Replica.__init__


# ===========================================================================
# SERVER half
# ===========================================================================


class KVStore:
    """An LRU store of real KV tensors (one entry = one cached prefix or one
    session). `cap` is a number of entries; at ~24 KB per token a 256-token
    entry is ~6 MB, so the cap is also a memory budget."""

    def __init__(self, cap: int):
        self.cap = cap
        self.d = OrderedDict()
        self.evictions = 0

    def get(self, key):
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, val):
        self.d[key] = val
        self.d.move_to_end(key)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)
            self.evictions += 1

    def bytes(self):
        return sum(v["k"].numel() * 8 for v in self.d.values())  # k and v


def prefix_key(ids, n):
    return hashlib.blake2b(bytes(str(ids[:n]), "utf8"), digest_size=8).hexdigest()


class Replica:
    """The model, the pool, the caches, and one lock around all of them."""

    def __init__(self, name, threads, prefix_cap=16, session_cap=16,
                 max_len=1024, layers=None):
        """`layers` truncates the model to its first N transformer blocks.

        Why that is legitimate here: this phase studies the layer ABOVE the
        engine. A router, a health check and a failover path cannot tell how
        deep the model behind a port is -- they see a port, a queue and a
        token stream. What the depth does control is how much RAM one replica
        costs, and four *full* fp32 copies of Qwen2.5-0.5B (2.8 GB resident
        each) do not fit on this machine; they push it into swap, and a
        swapping box measures the disk, not the fleet. Eight of the 24 blocks
        keeps every real component -- real weights, real attention, a real KV
        cache -- at a third of the memory. Every number in these projects is
        a fleet number, not a model-quality number, so the truncation costs
        the experiment nothing and is stated wherever results are reported.
        """
        import torch
        sys.path.insert(0, LIB16)
        import batchlib
        from batchlib import SlotKV

        self.torch = torch
        self.batchlib = batchlib
        self.name = name
        self.n_layers_used = layers
        if layers is None:
            self.runner, self.tok = batchlib.load_runner(n_threads=threads)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(threads)
            self.tok = AutoTokenizer.from_pretrained(batchlib.MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                batchlib.MODEL_ID, dtype=torch.float32)
            model.eval()
            model.model.layers = model.model.layers[:layers]
            model.config.num_hidden_layers = layers
            self.runner = batchlib.BatchedRunner(model, self.tok)
            del model            # the dropped blocks lose their last reference
        r = self.runner
        self.pool = SlotKV(r.n_layers, 1, r.n_kv_heads, r.d_head, max_len)
        self.max_len = max_len
        self.lock = threading.Lock()
        self.queue = 0          # requests waiting on the lock
        self.busy = 0
        self.done = 0
        self.prefixes = KVStore(prefix_cap)
        self.sessions = KVStore(session_cap)
        self.hits = 0
        self.misses = 0

    # -- KV save/restore ----------------------------------------------------

    def _save_kv(self, length):
        """Copy the pool slot's first `length` positions out as one stacked
        tensor pair. Cloning matters: the pool slot is reused by the next
        request."""
        t = self.torch
        k = t.stack([self.pool.k[l][0, :, :length, :] for l in range(self.runner.n_layers)]).clone()
        v = t.stack([self.pool.v[l][0, :, :length, :] for l in range(self.runner.n_layers)]).clone()
        return k, v

    def _load_kv(self, k, v, length):
        for l in range(self.runner.n_layers):
            self.pool.k[l][0, :, :length, :] = k[l]
            self.pool.v[l][0, :, :length, :] = v[l]

    # -- one request ----------------------------------------------------------

    def generate(self, ids, max_new, prefix_len=None, session_id=None,
                 emit=lambda line: None):
        """Run one request. Calls `emit(dict)` once with a meta line, then
        once per generated token, then once with a done line."""
        t_arrive = time.perf_counter()
        self.queue += 1
        with self.lock:
            self.queue -= 1
            self.busy = 1
            t_start = time.perf_counter()
            reused, start, hit_kind = 0, 0, "none"

            # ---- try the session cache, then the prefix cache -------------
            if session_id is not None:
                ent = self.sessions.get(session_id)
                if ent is not None and ent["ids"] == ids[: len(ent["ids"])]:
                    start = min(len(ent["ids"]), len(ids) - 1)
                    self._load_kv(ent["k"], ent["v"], len(ent["ids"]))
                    reused, hit_kind = start, "session"
            if start == 0 and prefix_len:
                key = prefix_key(ids, prefix_len)
                ent = self.prefixes.get(key)
                if ent is not None:
                    start = min(prefix_len, len(ids) - 1)
                    self._load_kv(ent["k"], ent["v"], start)
                    reused, hit_kind = start, "prefix"
            if reused:
                self.hits += 1
            else:
                self.misses += 1

            # ---- prefill what is not cached --------------------------------
            t = self.torch
            x = t.tensor(ids[start:]).view(1, -1)
            t0 = time.perf_counter()
            logits, _ = self.runner.prefill(self.pool, [0], x, [len(ids)],
                                            start=start, count=False)
            prefill_s = time.perf_counter() - t0

            if prefix_len and hit_kind != "prefix" and prefix_len < len(ids):
                key = prefix_key(ids, prefix_len)
                if self.prefixes.get(key) is None:
                    k, v = self._save_kv(prefix_len)
                    self.prefixes.put(key, {"k": k, "v": v})

            emit({"meta": 1, "replica": self.name, "reused": reused,
                  "hit": hit_kind, "prompt_tokens": len(ids),
                  "prefill_ms": round(prefill_s * 1e3, 1),
                  "queue_ms": round((t_start - t_arrive) * 1e3, 1)})

            # ---- greedy decode ---------------------------------------------
            out = [int(logits.argmax(-1))]
            emit({"t": out[-1]})
            cur = len(ids)
            while len(out) < max_new and cur < self.max_len - 1:
                logits, _ = self.runner.decode_step(self.pool, [0], [out[-1]],
                                                    [cur], count=False)
                cur += 1
                out.append(int(logits.argmax(-1)))
                emit({"t": out[-1]})

            # ---- save the session (prompt + all fed-back tokens) -----------
            if session_id is not None:
                known = ids + out[:-1]     # the last token's KV was never written
                k, v = self._save_kv(len(known))
                self.sessions.put(session_id, {"ids": known, "k": k, "v": v})

            self.busy = 0
            self.done += 1
            emit({"done": 1, "n_out": len(out),
                  "gen_ms": round((time.perf_counter() - t_start) * 1e3, 1)})
            return out

    def reset(self):
        """Drop both caches and zero the counters -- lets one fleet serve
        several cold-cache experiments without reloading 2 GB of weights."""
        with self.lock:
            self.prefixes = KVStore(self.prefixes.cap)
            self.sessions = KVStore(self.sessions.cap)
            self.hits = self.misses = self.done = 0

    def stats(self):
        return {"name": self.name, "layers": self.runner.n_layers,
                "busy": self.busy, "queue": self.queue,
                "done": self.done, "hits": self.hits, "misses": self.misses,
                "prefix_entries": len(self.prefixes.d),
                "session_entries": len(self.sessions.d),
                "prefix_evictions": self.prefixes.evictions,
                "session_evictions": self.sessions.evictions,
                "cache_bytes": self.prefixes.bytes() + self.sessions.bytes()}


def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    rep = Replica(args.name, args.threads, prefix_cap=args.prefix_cap,
                  session_cap=args.session_cap,
                  layers=args.layers or None)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # silence the default access log
            pass

        def do_GET(self):
            if self.path.startswith("/reset"):
                rep.reset()
            body = json.dumps(rep.stats()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(obj):
                data = (json.dumps(obj) + "\n").encode()
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()

            try:
                rep.generate(req["ids"], req.get("max_new", 16),
                             prefix_len=req.get("prefix_len"),
                             session_id=req.get("session_id"), emit=emit)
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                rep.busy = 0

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    print(f"READY {args.name} :{args.port}", flush=True)
    srv.serve_forever()


# ===========================================================================
# CLIENT half
# ===========================================================================


class Fleet:
    """Start/stop N replica processes and expose their URLs."""

    def __init__(self, n, threads=2, base_port=BASE_PORT, prefix_cap=16,
                 session_cap=16, log_dir=None, layers=LAYERS, stagger=True):
        """`threads` may be a single number, or a list giving each replica its
        own thread count -- which is how project 48 builds a *gray failure*: a
        replica that is healthy by every liveness test and simply slow."""
        self.procs = []
        self.ports = [base_port + i for i in range(n)]
        self.urls = [f"http://127.0.0.1:{p}" for p in self.ports]
        self.names = [f"r{i}" for i in range(n)]
        self.threads = threads if isinstance(threads, list) else [threads] * n
        self.layers = layers
        for i, p in enumerate(self.ports):
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(self.threads[i])
            env["MKL_NUM_THREADS"] = str(self.threads[i])
            logf = open(os.path.join(log_dir, f"{self.names[i]}.log"), "w") \
                if log_dir else subprocess.DEVNULL
            self.procs.append(subprocess.Popen(
                [sys.executable, os.path.join(HERE, "fleetlib.py"),
                 "--port", str(p), "--name", self.names[i],
                 "--threads", str(self.threads[i]),
                 "--layers", str(layers) if layers else "0",
                 "--prefix-cap", str(prefix_cap),
                 "--session-cap", str(session_cap)],
                stdout=logf, stderr=subprocess.STDOUT, env=env))
            # Start them ONE AT A TIME. Loading materialises the whole
            # checkpoint before the truncation frees most of it, so four
            # simultaneous loads peak at four times the final footprint --
            # which is exactly the swap storm the truncation exists to avoid.
            if stagger:
                self._wait_one(i)

    def _wait_one(self, i, timeout=300):
        import httpx
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                httpx.get(self.urls[i], timeout=2.0)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"replica {i} never became ready")

    def wait_ready(self, timeout=240):
        import httpx
        t0 = time.time()
        left = set(range(len(self.ports)))
        while left and time.time() - t0 < timeout:
            for i in list(left):
                try:
                    httpx.get(self.urls[i], timeout=2.0)
                    left.discard(i)
                except Exception:
                    time.sleep(1.0)
        if left:
            raise RuntimeError(f"replicas not ready: {left}")

    def stats(self):
        import httpx
        out = []
        for u in self.urls:
            try:
                out.append(httpx.get(u, timeout=5.0).json())
            except Exception:
                out.append(None)
        return out

    def reset(self):
        import httpx
        for u in self.urls:
            try:
                httpx.get(u + "/reset", timeout=10.0)
            except Exception:
                pass

    def kill(self, i, sig=9):
        os.kill(self.procs[i].pid, sig)

    def stop(self):
        """Kill every replica and do not return until they are actually gone.

        Worth the fuss: a replica that survives its owner keeps ~2 GB and its
        threads, and the next fleet then starts on a box that is quietly out
        of memory. That failure looks like "the new replica never became
        ready", which is a long way from its cause.
        """
        for p in self.procs:
            for fn in (p.kill, p.terminate):
                try:
                    fn()
                except Exception:
                    pass
        deadline = time.time() + 20
        for p in self.procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                pass
        for p in self.procs:                      # last resort, by pid
            if p.poll() is None:
                try:
                    os.kill(p.pid, 9)
                except ProcessLookupError:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        self.procs = []


# -- routers -----------------------------------------------------------------


class RoundRobin:
    """Replica i, i+1, i+2, ... in a circle. Ignores everything about the
    request and everything about the replicas."""

    def __init__(self, n):
        self.n, self.i = n, 0
        self.outstanding = [0] * n

    def pick(self, req):
        r = self.i % self.n
        self.i += 1
        self.outstanding[r] += 1
        return r

    def release(self, r):
        self.outstanding[r] -= 1


class LeastOutstanding(RoundRobin):
    """Send to the replica with the fewest requests in flight -- the simplest
    router that actually looks at the fleet's state."""

    def pick(self, req):
        r = min(range(self.n), key=lambda i: (self.outstanding[i], i))
        self.outstanding[r] += 1
        return r


class HashRouter(RoundRobin):
    """hash(key) % n. The key function decides WHAT sticks: hash the prompt's
    first tokens and you have prefix-aware routing (project 46); hash the
    session id and you have session affinity (project 49)."""

    def __init__(self, n, key_fn):
        super().__init__(n)
        self.key_fn = key_fn

    def pick(self, req):
        key = self.key_fn(req)
        h = int(hashlib.blake2b(str(key).encode(), digest_size=8).hexdigest(), 16)
        r = h % self.n
        self.outstanding[r] += 1
        return r


# -- the load generator --------------------------------------------------------


async def request_once(client, url, req, timeout=300.0):
    """One streaming request. Returns a record with the two user clocks
    (TTFT to the first TOKEN line, then per-token gaps) plus whatever the
    replica's meta line said."""
    t0 = time.perf_counter()
    rec = {"rid": req.get("rid"), "ok": False, "ttft_s": None, "itls": [],
           "toks": [], "error": None}
    payload = {k: req[k] for k in ("ids", "max_new", "prefix_len", "session_id")
               if req.get(k) is not None}
    try:
        last = None
        async with client.stream("POST", url + "/generate", json=payload,
                                 timeout=timeout) as r:
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                now = time.perf_counter()
                if "meta" in obj:
                    rec.update({k: obj[k] for k in
                                ("replica", "reused", "hit", "prefill_ms",
                                 "queue_ms", "prompt_tokens")})
                elif "t" in obj:
                    rec["toks"].append(obj["t"])
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = now - t0
                    elif last is not None:
                        rec["itls"].append(now - last)
                    last = now
                elif "done" in obj:
                    rec["ok"] = True
                    rec["n_out"] = obj["n_out"]
    except Exception as e:
        rec["error"] = type(e).__name__
    rec["e2e_s"] = time.perf_counter() - t0
    return rec


async def health_checker(urls, alive, interval, timeout, stop, log):
    """A liveness probe, exactly as most production health checks work: ask
    each replica a cheap question and keep it in the `alive` set while it
    answers. Note what this can and cannot see -- it catches a replica that is
    GONE, and it is blind to one that answers instantly and then generates
    tokens three times too slowly, because answering IS the whole test."""
    import httpx

    async with httpx.AsyncClient() as client:
        while not stop.is_set():
            for i, u in enumerate(urls):
                t0 = time.perf_counter()
                try:
                    await client.get(u, timeout=timeout)
                    ok, dt = True, time.perf_counter() - t0
                except Exception:
                    ok, dt = False, timeout
                was = i in alive
                if ok:
                    alive.add(i)
                else:
                    alive.discard(i)
                if was != ok:
                    log.append({"t": time.perf_counter(), "replica": i,
                                "alive": ok, "probe_s": dt})
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


async def run_load(fleet_urls, requests, router, concurrency=8, retries=0,
                   alive=None, timeout=300.0, open_loop=False,
                   health_interval=None, health_timeout=1.0):
    """Drive `requests` at the fleet.

    Closed loop (default): at most `concurrency` in flight, next starts when
    one finishes -- measures capacity. Open loop (`open_loop=True`): each
    request starts at its own `arrive` time regardless of how the fleet is
    doing -- measures what USERS see, and is the mode the failure drill needs
    (real users do not politely stop arriving because a GPU died).

    `alive`: an optional set of replica indices the router is allowed to use.
    Pass `health_interval` to have a liveness probe maintain it for you.
    `retries`: how many times a failed request is re-sent (to a different
    replica if one is alive).
    """
    import httpx

    sem = asyncio.Semaphore(concurrency if not open_loop else 10 ** 6)
    records = []
    health_log = []
    t_start = time.perf_counter()
    stop = asyncio.Event()
    probe = None
    if health_interval:
        if alive is None:
            alive = set(range(len(fleet_urls)))
        probe = asyncio.create_task(health_checker(
            fleet_urls, alive, health_interval, health_timeout, stop,
            health_log))

    async with httpx.AsyncClient() as client:

        async def one(req):
            async with sem:
                if open_loop:
                    delay = req["arrive"] - (time.perf_counter() - t_start)
                    if delay > 0:
                        await asyncio.sleep(delay)
                tries = retries + 1
                rec = None
                for attempt in range(tries):
                    cand = alive if alive is not None else set(range(router.n))
                    if not cand:
                        break
                    r = router.pick(req)
                    if r not in cand:      # router chose a dead replica
                        router.release(r)
                        r = min(cand)      # failover: any live one
                        router.outstanding[r] += 1
                    t_send = time.perf_counter() - t_start
                    rec = await request_once(client, fleet_urls[r], req,
                                             timeout=timeout)
                    router.release(r)
                    rec.update({"target": r, "attempt": attempt,
                                "t_send": t_send})
                    if rec["ok"]:
                        break
                if rec is None:
                    rec = {"rid": req.get("rid"), "ok": False,
                           "error": "NoReplicaAlive", "ttft_s": None,
                           "itls": [], "e2e_s": 0.0,
                           "t_send": time.perf_counter() - t_start}
                rec["t_end"] = time.perf_counter() - t_start
                records.append(rec)

        await asyncio.gather(*[one(r) for r in requests])
    stop.set()
    if probe is not None:
        await probe
    wall = time.perf_counter() - t_start
    return records, wall, health_log


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))]


def summarize(records, wall, label=""):
    ok = [r for r in records if r["ok"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    itls = [x for r in ok for x in r["itls"]]
    toks = sum(r.get("n_out", 0) for r in ok)
    return {
        "label": label, "requests": len(records), "ok": len(ok),
        "errors": len(records) - len(ok), "wall_s": round(wall, 2),
        "throughput_tok_s": round(toks / wall, 2) if wall else 0.0,
        "ttft_p50_s": round(pct(ttfts, 50), 3),
        "ttft_p99_s": round(pct(ttfts, 99), 3),
        "itl_p50_ms": round(pct(itls, 50) * 1e3, 1),
        "itl_p99_ms": round(pct(itls, 99) * 1e3, 1),
        "e2e_p50_s": round(pct([r["e2e_s"] for r in ok], 50), 3),
        "e2e_p99_s": round(pct([r["e2e_s"] for r in ok], 99), 3),
    }


# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--name", default="r0")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=LAYERS,
                    help="truncate the model to N blocks (0 = full model)")
    ap.add_argument("--prefix-cap", type=int, default=16)
    ap.add_argument("--session-cap", type=int, default=16)
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
