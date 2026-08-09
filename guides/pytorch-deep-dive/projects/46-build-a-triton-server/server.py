"""A minimal inference server that speaks Triton's protocol.

NVIDIA's Triton Inference Server is a large C++ program that needs Docker (and,
in practice, a GPU). What it *is*, though, is three ideas:

  1. a **model repository** on disk - one directory per model, numbered version
     subdirectories, and a `config.pbtxt` describing the inputs and outputs;
  2. the **KServe v2 inference protocol** over HTTP - `/v2/health/ready`,
     `/v2/models/<name>`, `/v2/models/<name>/infer`, with an optional binary
     extension for the tensor bytes;
  3. a **dynamic batcher** - a queue in front of the model that glues concurrent
     requests into one batch before executing them.

This file implements all three in about 250 lines, backed by ONNX Runtime. It is
not a replacement for Triton; it is Triton's shape, small enough to read.

Usage:  python3 server.py --repo model_repository --port 8000 [--no-batching]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------
# config.pbtxt
# --------------------------------------------------------------------------
def parse_pbtxt(text: str) -> dict:
    """Parse the small subset of protobuf-text syntax Triton configs use.

    Handles `key: value`, `key { ... }`, and `key [ {...}, {...} ]`.
    """
    tokens = re.findall(r'"[^"]*"|[\[\]{}:,]|[^\s\[\]{}:,]+', text)
    pos = 0

    def value():
        nonlocal pos
        tok = tokens[pos]
        if tok == "{":
            pos += 1
            return block()
        if tok == "[":
            pos += 1
            items = []
            while tokens[pos] != "]":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                items.append(value())
            pos += 1
            return items
        pos += 1
        if tok.startswith('"'):
            return tok[1:-1]
        if re.fullmatch(r"-?\d+", tok):
            return int(tok)
        return tok

    def block():
        nonlocal pos
        out: dict = {}
        while pos < len(tokens) and tokens[pos] != "}":
            key = tokens[pos]
            pos += 1
            if tokens[pos] == ":":
                pos += 1
            out[key] = value()
        pos += 1
        return out

    return block()


DTYPES = {"TYPE_FP32": np.float32, "TYPE_INT64": np.int64, "TYPE_INT32": np.int32}


class ModelRepository:
    """Discovers `<repo>/<model>/<version>/model.onnx` and their configs."""

    def __init__(self, root: str, force_batch: int | None = None):
        self.models: dict[tuple[str, int], "LoadedModel"] = {}
        self.latest: dict[str, int] = {}
        for name in sorted(os.listdir(root)):
            mdir = os.path.join(root, name)
            cfg_path = os.path.join(mdir, "config.pbtxt")
            if not os.path.isdir(mdir) or not os.path.exists(cfg_path):
                continue
            cfg = parse_pbtxt(open(cfg_path).read())
            if force_batch is not None:
                cfg["max_batch_size"] = force_batch
            for version in sorted(int(v) for v in os.listdir(mdir) if v.isdigit()):
                path = os.path.join(mdir, str(version), "model.onnx")
                self.models[(name, version)] = LoadedModel(name, version, path, cfg)
                self.latest[name] = max(self.latest.get(name, 0), version)
            print(f"[server] loaded {name} versions "
                  f"{sorted(v for n, v in self.models if n == name)}  "
                  f"max_batch_size={cfg.get('max_batch_size')}")

    def get(self, name: str, version: int | None):
        version = version or self.latest.get(name)
        return self.models.get((name, version))


class LoadedModel:
    """One model version: an ORT session plus the batching thread in front of it."""

    def __init__(self, name, version, path, cfg):
        self.name, self.version, self.cfg = name, version, cfg
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "6"))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = cfg["input"] if isinstance(cfg["input"], list) else [cfg["input"]]
        self.outputs = cfg["output"] if isinstance(cfg["output"], list) else [cfg["output"]]
        self.max_batch = int(cfg.get("max_batch_size", 1))
        db = cfg.get("dynamic_batching")
        self.delay_s = (int(db.get("max_queue_delay_microseconds", 0)) / 1e6
                        if isinstance(db, dict) else 0.0)
        self.queue: queue.Queue = queue.Queue()
        # Statistics, in the spirit of Triton's /v2/models/<name>/stats
        self.stats = {"requests": 0, "executions": 0, "batch_sizes": [],
                      "queue_us": [], "compute_us": []}
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- batching
    def _worker(self):
        while True:
            first = self.queue.get()
            batch = [first]
            if self.max_batch > 1:
                # Wait up to max_queue_delay for more requests to arrive, so they
                # can share one execution. This is the entire dynamic batcher.
                deadline = time.perf_counter() + self.delay_s
                while len(batch) < self.max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self.queue.get(timeout=remaining))
                    except queue.Empty:
                        break
            self._execute(batch)

    def _execute(self, batch):
        now = time.perf_counter()
        arrays = np.concatenate([item["array"] for item in batch], axis=0)
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.inputs[0]["name"]: arrays})[0]
        compute_us = (time.perf_counter() - t0) * 1e6

        self.stats["executions"] += 1
        self.stats["batch_sizes"].append(len(batch))
        self.stats["compute_us"].append(compute_us)
        offset = 0
        for item in batch:
            n = item["array"].shape[0]
            item["result"] = outputs[offset:offset + n]
            offset += n
            self.stats["queue_us"].append((now - item["arrived"]) * 1e6)
            item["done"].set()

    def infer(self, array: np.ndarray) -> np.ndarray:
        item = {"array": array, "done": threading.Event(), "result": None,
                "arrived": time.perf_counter()}
        self.stats["requests"] += 1
        self.queue.put(item)
        item["done"].wait()
        return item["result"]

    def metadata(self) -> dict:
        return {
            "name": self.name, "versions": [str(self.version)],
            "platform": self.cfg.get("platform", "onnxruntime_onnx"),
            "inputs": [{"name": i["name"], "datatype": i["data_type"][5:],
                        "shape": [-1] + list(i["dims"])} for i in self.inputs],
            "outputs": [{"name": o["name"], "datatype": o["data_type"][5:],
                         "shape": [-1] + list(o["dims"])} for o in self.outputs],
        }


# --------------------------------------------------------------------------
# the HTTP layer (KServe v2)
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Without this the response header and the response body leave as two small
    # TCP segments, and Nagle's algorithm holds the second one until the client
    # acknowledges the first - a flat 40 ms added to every request.
    disable_nagle_algorithm = True
    repo: ModelRepository = None                       # set in serve()

    def log_message(self, *args):                      # keep stdout for results
        pass

    def _send(self, code, payload=None, raw=b"", header_len=None):
        body = (json.dumps(payload).encode() if payload is not None else b"") + raw
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if header_len is not None:
            self.send_header("Inference-Header-Content-Length", str(header_len))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/v2/health/ready", "/v2/health/live"):
            return self._send(200, {})
        m = re.fullmatch(r"/v2/models/([^/]+)(?:/versions/(\d+))?", self.path)
        if m:
            model = self.repo.get(m.group(1), int(m.group(2)) if m.group(2) else None)
            if model is None:
                return self._send(404, {"error": f"model {m.group(1)} not found"})
            return self._send(200, model.metadata())
        m = re.fullmatch(r"/v2/models/([^/]+)/stats", self.path)
        if m:
            model = self.repo.get(m.group(1), None)
            if model is None:
                return self._send(404, {"error": "not found"})
            s = model.stats
            return self._send(200, {
                "requests": s["requests"], "executions": s["executions"],
                "mean_batch_size": float(np.mean(s["batch_sizes"])) if s["batch_sizes"] else 0,
                "max_batch_size_seen": int(max(s["batch_sizes"])) if s["batch_sizes"] else 0,
                "mean_queue_us": float(np.mean(s["queue_us"])) if s["queue_us"] else 0,
                "mean_compute_us": float(np.mean(s["compute_us"])) if s["compute_us"] else 0,
                "compute_us_per_image": (float(np.sum(s["compute_us"]) / np.sum(s["batch_sizes"]))
                                         if s["batch_sizes"] else 0),
            })
        self._send(404, {"error": "unknown route"})

    def do_POST(self):
        m = re.fullmatch(r"/v2/models/([^/]+)(?:/versions/(\d+))?/infer", self.path)
        if not m:
            return self._send(404, {"error": "unknown route"})
        model = self.repo.get(m.group(1), int(m.group(2)) if m.group(2) else None)
        if model is None:
            return self._send(404, {"error": f"model {m.group(1)} not found"})

        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        hdr_len = self.headers.get("Inference-Header-Content-Length")
        if hdr_len is None:
            req, blob = json.loads(body), b""
        else:
            req, blob = json.loads(body[:int(hdr_len)]), body[int(hdr_len):]

        spec = req["inputs"][0]
        dtype = {"FP32": np.float32}[spec["datatype"]]
        if "data" in spec:
            array = np.asarray(spec["data"], dtype=dtype).reshape(spec["shape"])
        else:
            array = np.frombuffer(blob, dtype=dtype).reshape(spec["shape"])

        want_binary = bool(req.get("outputs", [{}])[0]
                           .get("parameters", {}).get("binary_data", False))
        out = np.ascontiguousarray(model.infer(array))
        meta = {"model_name": model.name, "model_version": str(model.version),
                "outputs": [{"name": model.outputs[0]["name"], "datatype": "FP32",
                             "shape": list(out.shape)}]}
        if want_binary:
            meta["outputs"][0]["parameters"] = {"binary_data_size": out.nbytes}
            header = json.dumps(meta).encode()
            return self._send(200, None, header + out.tobytes(), header_len=len(header))
        meta["outputs"][0]["data"] = out.ravel().tolist()
        self._send(200, meta)


class Server(ThreadingHTTPServer):
    # The default listen backlog is 5. With 16 clients connecting at once the
    # kernel drops the extra SYNs, the client retransmits one second later, and
    # every latency percentile you measure is really TCP's retry timer.
    request_queue_size = 128
    allow_reuse_address = True


def serve(repo_root: str, port: int, force_batch: int | None = None,
          nagle: bool = False):
    Handler.disable_nagle_algorithm = not nagle
    Handler.repo = ModelRepository(repo_root, force_batch=force_batch)
    server = Server(("127.0.0.1", port), Handler)
    print(f"[server] listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="model_repository")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=None,
                    help="override max_batch_size (1 disables dynamic batching)")
    ap.add_argument("--nagle", action="store_true",
                    help="leave Nagle's algorithm on, to show what it costs")
    args = ap.parse_args()
    serve(args.repo, args.port, args.max_batch, args.nagle)
