"""Project 46 - wrap the CNN in a Triton-shaped inference server and query it.

Sections:
  1. the model repository, and what `config.pbtxt` tells the server
  2. the protocol: health, metadata, infer - and are the answers still correct?
  3. the serving tax: how much of a request is not inference
  4. JSON tensors vs the binary extension
  5. dynamic batching under concurrent load
  6. the max_queue_delay knob, and model versions

Run:  python3 run.py       (~3 minutes; starts and stops its own servers)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import torch

torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "42-export-to-onnx"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import deploy_lib as D  # noqa: E402
from plot_style import SERIES, style_axes  # noqa: E402

OUT = os.path.join(HERE, "outputs")
REPO = os.path.join(HERE, "model_repository")
os.makedirs(OUT, exist_ok=True)

# Every server instance gets its own port. A crashed run can leave a process
# holding the old one, and then the "new" server's health check is answered by
# the stale one - which is exactly the bug that produced this comment.
import itertools  # noqa: E402

PORTS = itertools.count(8412)
BASE = ""
FINDINGS: list[tuple] = []


def note(section, name, value):
    FINDINGS.append((section, name, value))
    print(f"    {name:<50} {value}")


# --------------------------------------------------------------------------
# the client side of the protocol
# --------------------------------------------------------------------------
import urllib.request  # noqa: E402


def http(method, path, body=None, headers=None):
    """Return (status, headers, body). A 404 is an answer here, not an exception."""
    req = urllib.request.Request(BASE + path, data=body, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


class Client:
    """One HTTP connection, reused for every request - what a real client does."""

    def __init__(self, nodelay: bool = True):
        import http.client
        import socket

        host, port = BASE.replace("http://", "").split(":")
        self.conn = http.client.HTTPConnection(host, int(port), timeout=60)
        self.conn.connect()
        if nodelay:
            self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _post(self, path, body, headers):
        self.conn.request("POST", path, body=body, headers=headers)
        resp = self.conn.getresponse()
        data = resp.read()
        return dict(resp.getheaders()), data

    def infer_binary(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        header = json.dumps({
            "inputs": [{"name": "x", "shape": list(x.shape), "datatype": "FP32",
                        "parameters": {"binary_data_size": x.nbytes}}],
            "outputs": [{"name": "logits", "parameters": {"binary_data": True}}],
        }).encode()
        hdrs, body = self._post(
            "/v2/models/small_cnn/infer", header + x.tobytes(),
            {"Inference-Header-Content-Length": str(len(header)),
             "Content-Length": str(len(header) + x.nbytes)})
        n = int(hdrs["Inference-Header-Content-Length"])
        meta = json.loads(body[:n])["outputs"][0]
        return np.frombuffer(body[n:], dtype=np.float32).reshape(meta["shape"])

    def infer_json(self, x):
        payload = json.dumps({"inputs": [{"name": "x", "shape": list(x.shape),
                                          "datatype": "FP32",
                                          "data": x.ravel().tolist()}]}).encode()
        _, body = self._post("/v2/models/small_cnn/infer", payload,
                             {"Content-Type": "application/json",
                              "Content-Length": str(len(payload))})
        out = json.loads(body)["outputs"][0]
        return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])

    def close(self):
        self.conn.close()


def infer_binary(x: np.ndarray, model="small_cnn", version=None):
    """The binary tensor extension: a JSON header, then raw bytes."""
    path = f"/v2/models/{model}" + (f"/versions/{version}" if version else "") + "/infer"
    x = np.ascontiguousarray(x, dtype=np.float32)
    header = json.dumps({
        "inputs": [{"name": "x", "shape": list(x.shape), "datatype": "FP32",
                    "parameters": {"binary_data_size": x.nbytes}}],
        "outputs": [{"name": "logits", "parameters": {"binary_data": True}}],
    }).encode()
    _, hdrs, body = http("POST", path, header + x.tobytes(),
                         {"Inference-Header-Content-Length": str(len(header))})
    n = int(hdrs["Inference-Header-Content-Length"])
    meta = json.loads(body[:n])["outputs"][0]
    return np.frombuffer(body[n:], dtype=np.float32).reshape(meta["shape"])


class Server:
    """Start `server.py` in its own process and wait for /v2/health/ready."""

    def __init__(self, max_batch=None, delay_us=None, env=None, nagle=False):
        global BASE
        self.port = next(PORTS)
        BASE = f"http://127.0.0.1:{self.port}"
        cfg = os.path.join(REPO, "small_cnn", "config.pbtxt")
        if delay_us is not None:
            text = open(cfg).read()
            open(cfg, "w").write(re.sub(r"max_queue_delay_microseconds: \d+",
                                        f"max_queue_delay_microseconds: {delay_us}",
                                        text))
        cmd = [sys.executable, os.path.join(HERE, "server.py"), "--repo", REPO,
               "--port", str(self.port)]
        if max_batch is not None:
            cmd += ["--max-batch", str(max_batch)]
        if nagle:
            cmd += ["--nagle"]
        self.log_path = os.path.join(OUT, "server.log")
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **(env or {})})
        for _ in range(200):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited; see {self.log_path}")
            try:
                if http("GET", "/v2/health/ready")[0] == 200:
                    return
            except Exception:                                  # noqa: BLE001
                time.sleep(0.1)
        raise RuntimeError("server did not become ready; see " + self.log_path)

    def stats(self, model="small_cnn"):
        return json.loads(http("GET", f"/v2/models/{model}/stats")[2])

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()


CONFIG_PBTXT = """\
name: "small_cnn"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 2000
}
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
"""


def build_repository(model):
    """Write <repo>/small_cnn/{1,2}/model.onnx + config.pbtxt."""
    shutil.rmtree(REPO, ignore_errors=True)
    sample = torch.randn(1, 3, 32, 32)
    batch = torch.export.Dim("batch")
    for version in (1, 2):
        vdir = os.path.join(REPO, "small_cnn", str(version))
        os.makedirs(vdir, exist_ok=True)
        exported = model
        if version == 2:                       # a "new release": the int8 CNN would go
            exported = model                   # here; we ship the same weights twice
        torch.onnx.export(exported, (sample,), os.path.join(vdir, "model.onnx"),
                          input_names=["x"], output_names=["logits"], dynamo=True,
                          dynamic_shapes={"x": {0: batch}}, external_data=False)
    open(os.path.join(REPO, "small_cnn", "config.pbtxt"), "w").write(CONFIG_PBTXT)


def percentiles(samples):
    a = np.asarray(samples) * 1e3
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


def load_test(kind, x, n_clients, n_per_client, nodelay=True):
    """n_clients threads, each with its own kept-alive connection."""
    lat: list[float] = []
    lock = __import__("threading").Lock()

    def worker():
        client = Client(nodelay=nodelay)
        fn = client.infer_binary if kind == "binary" else client.infer_json
        fn(x)                                   # warm the connection
        mine = []
        for _ in range(n_per_client):
            t0 = time.perf_counter()
            fn(x)
            mine.append(time.perf_counter() - t0)
        client.close()
        with lock:
            lat.extend(mine)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        list(pool.map(lambda _: worker(), range(n_clients)))
    wall = time.perf_counter() - t0
    return lat, n_clients * n_per_client / wall


# ==========================================================================
def main():
    t_start = time.time()
    subprocess.run(["pkill", "-f", os.path.join(HERE, "server.py")], check=False)
    open(os.path.join(OUT, "server.log"), "w").close()
    model = D.get_trained_cnn()
    x, y = D.load_cifar("test", 512)
    one = x[:1].numpy()

    print("\n[1] the model repository")
    build_repository(model)
    listing = []
    for root, _dirs, files in os.walk(REPO):
        for f in sorted(files):
            p = os.path.join(root, f)
            listing.append(f"{os.path.relpath(p, HERE)}  ({D.file_mb(p) * 1000:.0f} KB)")
    for line in listing:
        note(1, "repository file", line)

    server = Server()
    meta = json.loads(http("GET", "/v2/models/small_cnn")[2])
    note(1, "declared input shape (config dims)", "[3, 32, 32]")
    note(1, "shape the server advertises", str(meta["inputs"][0]["shape"]))
    note(1, "versions found on disk", str(sorted(
        int(v) for v in os.listdir(os.path.join(REPO, "small_cnn")) if v.isdigit())))

    print("\n[2] the protocol")
    code, _, _ = http("GET", "/v2/health/ready")
    note(2, "GET /v2/health/ready", code)
    note(2, "GET /v2/models/small_cnn", json.dumps(meta))
    note(2, "GET /v2/models/nope", http("GET", "/v2/models/nope")[0])
    served = np.concatenate([infer_binary(x[i:i + 32].numpy()) for i in range(0, 512, 32)])
    local = D.predict(model, x).numpy()
    note(2, "max |served - local PyTorch|", f"{np.abs(served - local).max():.3e}")
    note(2, "accuracy of the served model",
         f"{(served.argmax(1) == y.numpy()).mean():.4f}")

    print("\n[3] the serving tax")
    ort_only = D.interleaved(
        {"local": lambda: D.predict(model, x[:1])}, rounds=7, calls=20)["local"]
    lat_json, _ = load_test("json", one, 1, 60)
    lat_bin, _ = load_test("binary", one, 1, 60)

    pj, pb = percentiles(lat_json), percentiles(lat_bin)
    st = server.stats()
    note(3, "in-process PyTorch, batch 1 (ms)", f"{ort_only['median_ms']:.3f}")
    note(3, "server-side compute, mean (ms)", f"{st['mean_compute_us'] / 1000:.3f}")
    note(3, "end-to-end over HTTP, binary p50 (ms)", f"{pb['p50']:.3f}")
    note(3, "  of which is not compute",
         f"{pb['p50'] - st['mean_compute_us'] / 1000:.3f} ms "
         f"({100 * (1 - st['mean_compute_us'] / 1000 / pb['p50']):.0f}%)")
    server.close()
    nagle_server = Server(nagle=True)
    lat_nagle, _ = load_test("binary", one, 1, 25)
    pn = percentiles(lat_nagle)
    nagle_server.close()
    server = Server()                      # back to the normal configuration
    note(3, "the same server with Nagle's algorithm left on, p50 (ms)",
         f"{pn['p50']:.3f}")
    note(3, "  what one TCP option costs", f"{pn['p50'] - pb['p50']:+.1f} ms per request")

    print("\n[4] JSON tensors vs the binary extension")
    payload_json = len(json.dumps({"inputs": [{"name": "x", "shape": [1, 3, 32, 32],
                                               "datatype": "FP32",
                                               "data": one.ravel().tolist()}]}))
    note(4, "request body: JSON / binary (bytes)",
         f"{payload_json} / {one.nbytes + 180}   ({payload_json / (one.nbytes + 180):.1f}x)")
    note(4, "latency p50: JSON / binary (ms)", f"{pj['p50']:.3f} / {pb['p50']:.3f}")
    note(4, "latency p99: JSON / binary (ms)", f"{pj['p99']:.3f} / {pb['p99']:.3f}")
    server.close()

    print("\n[5] dynamic batching under load")
    # First: what batching is worth in pure compute, measured in-process with no
    # HTTP in the way. This is the ceiling the batcher is trying to reach.
    import onnxruntime as ort

    sess = ort.InferenceSession(
        os.path.join(REPO, "small_cnn", "2", "model.onnx"),
        providers=["CPUExecutionProvider"])
    batch_curve = {}
    calls = {1: 40, 4: 20, 8: 10, 32: 4}
    variants = {str(b): (lambda b=b: sess.run(None, {"x": x[:b].numpy()}))
                for b in (1, 4, 8, 32)}
    res = D.interleaved(variants, rounds=9, calls=8)
    for b in (1, 4, 8, 32):
        per_image = res[str(b)]["min_ms"] * 1000 / b
        batch_curve[b] = per_image
        note(5, f"in-process compute at batch {b:>2}",
             f"{res[str(b)]['min_ms']:7.3f} ms  =  {per_image:6.1f} us/image")
    note(5, "so batching 32 could cut compute per image by",
         f"{batch_curve[1] / batch_curve[32]:.2f}x")

    rows = []
    for label, max_batch in [("batching off (max_batch_size=1)", 1),
                             ("batching on  (max_batch_size=32)", None)]:
        srv = Server(max_batch=max_batch)
        for clients in (1, 4, 16):
            lat, thr = load_test("binary", one, clients, 80)
            s = srv.stats()
            p = percentiles(lat)
            rows.append({"mode": label, "clients": clients, "throughput": thr,
                         "p50": p["p50"], "p99": p["p99"],
                         "mean_batch": s["mean_batch_size"],
                         "executions": s["executions"], "requests": s["requests"]})
            rows[-1]["compute_per_image_us"] = s["compute_us_per_image"]
            note(5, f"{label}, {clients:>2} clients",
                 f"{thr:7.1f} req/s  p50 {p['p50']:6.2f} ms  p99 {p['p99']:6.2f} ms  "
                 f"mean batch {s['mean_batch_size']:.2f}  "
                 f"compute {s['compute_us_per_image']:6.0f} us/image")
        srv.close()

    print("\n[6] the max_queue_delay knob")
    delay_rows = []
    for delay_us in (0, 500, 2000, 10000):
        srv = Server(delay_us=delay_us)
        lat, thr = load_test("binary", one, 16, 80)
        s = srv.stats()
        p = percentiles(lat)
        delay_rows.append({"delay_us": delay_us, "throughput": thr, "p50": p["p50"],
                           "p99": p["p99"], "mean_batch": s["mean_batch_size"],
                           "queue_ms": s["mean_queue_us"] / 1000,
                           "compute_per_image_us": s["compute_us_per_image"]})
        note(6, f"max_queue_delay = {delay_us:>5} us",
             f"{thr:7.1f} req/s  p50 {p['p50']:6.2f} ms  p99 {p['p99']:6.2f} ms  "
             f"mean batch {s['mean_batch_size']:5.2f}  queued "
             f"{s['mean_queue_us'] / 1000:.2f} ms  compute "
             f"{s['compute_us_per_image']:5.0f} us/image")
        srv.close()
    # restore the committed default
    cfg = os.path.join(REPO, "small_cnn", "config.pbtxt")
    open(cfg, "w").write(CONFIG_PBTXT)

    server.close()
    srv = Server()
    v1 = infer_binary(one, version=1)
    v2 = infer_binary(one, version=2)
    note(6, "version 1 vs version 2 outputs agree",
         f"{np.abs(v1 - v2).max():.3e}")
    note(6, "unversioned request is served by", json.loads(
        http("POST", "/v2/models/small_cnn/infer",
             json.dumps({"inputs": [{"name": "x", "shape": [1, 3, 32, 32],
                                     "datatype": "FP32",
                                     "data": one.ravel().tolist()}]}).encode(),
             {"Content-Type": "application/json"})[2])["model_version"])
    srv.close()

    summary = {"load": rows, "delay": delay_rows,
               "json_p50": pj, "binary_p50": pb, "nagle_p50": pn,
               "local_ms": ort_only["median_ms"]}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    shutil.copy(os.path.join(REPO, "small_cnn", "config.pbtxt"),
                os.path.join(OUT, "config.pbtxt"))
    D.write_csv(os.path.join(OUT, "findings.csv"), FINDINGS, ["section", "name", "value"])
    figure(summary)
    print(f"\ntotal {time.time() - t_start:.0f}s")


def figure(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    modes = sorted({r["mode"] for r in summary["load"]})
    ax = axes[0]
    for i, mode in enumerate(modes):
        rs = [r for r in summary["load"] if r["mode"] == mode]
        ax.plot([r["clients"] for r in rs], [r["throughput"] for r in rs], "o-",
                color=SERIES[i], label=mode.split(" (")[0])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16]); ax.set_xticklabels([1, 4, 16])
    ax.set_xlabel("concurrent clients"); ax.set_ylabel("requests / second")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("throughput", loc="left", fontsize=11)

    ax = axes[1]
    for i, mode in enumerate(modes):
        rs = [r for r in summary["load"] if r["mode"] == mode]
        ax.plot([r["clients"] for r in rs], [r["p50"] for r in rs], "o-",
                color=SERIES[i], label=f"{mode.split(' (')[0]} p50")
        ax.plot([r["clients"] for r in rs], [r["p99"] for r in rs], "o--",
                color=SERIES[i], alpha=0.55, label=f"{mode.split(' (')[0]} p99")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16]); ax.set_xticklabels([1, 4, 16])
    ax.set_xlabel("concurrent clients"); ax.set_ylabel("ms")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("latency", loc="left", fontsize=11)

    ax = axes[2]
    d = summary["delay"]
    xs = np.arange(len(d))
    ax.bar(xs - 0.19, [r["throughput"] for r in d], 0.36, color=SERIES[1],
           label="req/s")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['delay_us'] / 1000:g} ms" for r in d], fontsize=9)
    ax.set_ylabel("requests / second")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["p99"] for r in d], "o-", color=SERIES[2], label="p99 latency")
    ax2.set_ylabel("p99 latency (ms)", color="#52514e")
    ax2.grid(False)
    for i, r in enumerate(d):
        ax.text(i - 0.19, r["throughput"] * 1.02, f"batch\n{r['mean_batch']:.1f}",
                ha="center", fontsize=7.5)
    ax.set_ylim(0, max(r["throughput"] for r in d) * 1.25)
    ax.set_xlabel("max_queue_delay, 16 clients")
    ax.set_title("the batching-delay knob", loc="left", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    path = os.path.join(OUT, "serving.png")
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
