"""A minimal streaming inference server: FastAPI + the loop from project 01.

Two endpoints over the same generation loop:

    POST /generate        -> Server-Sent Events, one token per event
    POST /generate_full   -> plain JSON, sent once the answer is complete

Every request also writes a stage-by-stage trace line (admission, tokenize,
queue, prefill, each decode step, detokenize, done) to $TRACE_PATH if that
environment variable is set. Project 07 turns that on and draws it.

Start it with:
    python3 server.py --port 8117 --threads 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
P01 = os.path.join(os.path.dirname(HERE), "01-manual-inference-loop")
sys.path.insert(0, P01)

import loop_lib as L  # noqa: E402

# Imported at module level on purpose: this file uses `from __future__ import
# annotations`, so FastAPI resolves the endpoints' type hints as strings
# against this module's globals. Importing Request inside build_app() would
# make it invisible there, and FastAPI would treat `request` as a query
# parameter and reject every call with "Field required".
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

TRACE_PATH = os.environ.get("TRACE_PATH")
_trace_lock = threading.Lock()

# The GPU-shaped assumption this server deliberately breaks: only one forward
# pass may touch the model at a time. Real engines fold concurrent requests
# into ONE batched forward pass instead (see project 01 section D, phase 3).
_model_lock = threading.Lock()
SERIALIZE = os.environ.get("SERIALIZE", "1") == "1"


def emit_trace(record: dict) -> None:
    if not TRACE_PATH:
        return
    with _trace_lock, open(TRACE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_app(model, tok):
    app = FastAPI()

    def run_request(prompt: str, max_new_tokens: int, stream: bool):
        """Generator yielding (token_text, stage_marks) as tokens appear."""
        rid = uuid.uuid4().hex[:8]
        t_admit = time.perf_counter()
        marks = {"request_id": rid, "stream": stream, "t_admit": t_admit}

        ids = tok(prompt, return_tensors="pt").input_ids
        marks["t_tokenized"] = time.perf_counter()
        marks["prompt_tokens"] = int(ids.shape[1])

        lock = _model_lock if SERIALIZE else threading.Lock()
        with lock:
            marks["t_scheduled"] = time.perf_counter()
            steps = []
            pieces = []

            def on_token(tid, step):
                steps.append(time.perf_counter())
                pieces.append(tid)

            res = L.generate_with_cache(
                model, ids, max_new_tokens=max_new_tokens,
                eos_id=tok.eos_token_id, on_token=on_token)
        marks["t_first_token"] = steps[0] if steps else time.perf_counter()
        marks["t_last_token"] = steps[-1] if steps else time.perf_counter()
        marks["decode_step_s"] = [round(x, 5) for x in res.decode_step_s]
        marks["prefill_s"] = round(res.prefill_s, 5)
        marks["n_output_tokens"] = len(res.token_ids)
        return res, marks

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "Hello")
        max_new = int(body.get("max_new_tokens", 32))
        chunk_every = int(body.get("chunk_every", 1))

        def sse():
            # Streaming path: we must produce bytes DURING generation, so the
            # loop is driven here rather than collected first.
            rid = uuid.uuid4().hex[:8]
            t_admit = time.perf_counter()
            ids = tok(prompt, return_tensors="pt").input_ids
            t_tokenized = time.perf_counter()
            pending, out_ids, step_times = [], [], []

            lock = _model_lock if SERIALIZE else threading.Lock()
            lock.acquire()
            t_scheduled = time.perf_counter()
            try:
                import torch
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    out = model(ids, use_cache=True)
                    past = out.past_key_values
                    nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
                    prefill_s = time.perf_counter() - t0
                    for step in range(max_new):
                        if step > 0:
                            t1 = time.perf_counter()
                            out = model(nid, past_key_values=past, use_cache=True)
                            past = out.past_key_values
                            nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
                            step_times.append(time.perf_counter() - t1)
                        tid = int(nid)
                        out_ids.append(tid)
                        pending.append(tid)
                        if len(pending) >= chunk_every or tid == tok.eos_token_id:
                            text = tok.decode(pending)
                            pending = []
                            yield ("data: " + json.dumps(
                                {"text": text, "i": step}) + "\n\n").encode()
                        if tid == tok.eos_token_id:
                            break
                    if pending:
                        yield ("data: " + json.dumps(
                            {"text": tok.decode(pending), "i": -1}) + "\n\n").encode()
            finally:
                lock.release()
            yield b"data: [DONE]\n\n"
            emit_trace({"request_id": rid, "stream": True,
                        "chunk_every": chunk_every,
                        "prompt_tokens": int(ids.shape[1]),
                        "t_admit": t_admit, "t_tokenized": t_tokenized,
                        "t_scheduled": t_scheduled, "prefill_s": prefill_s,
                        "decode_step_s": [round(x, 5) for x in step_times],
                        "t_done": time.perf_counter(),
                        "n_output_tokens": len(out_ids)})

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-cache"})

    @app.post("/generate_full")
    async def generate_full(request: Request):
        body = await request.json()
        res, marks = run_request(body.get("prompt", "Hello"),
                                 int(body.get("max_new_tokens", 32)),
                                 stream=False)
        marks["t_done"] = time.perf_counter()
        emit_trace(marks)
        return JSONResponse({"text": tok.decode(res.token_ids),
                             "n_output_tokens": len(res.token_ids)})

    @app.get("/health")
    async def health():
        return {"ok": True, "serialize": SERIALIZE}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8117)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    import uvicorn
    tok, model = L.load(n_threads=args.threads)
    app = build_app(model, tok)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
