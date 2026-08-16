"""One server, two completely different workloads, one HTTP interface.

    POST /generate   -> LLM, Server-Sent Events, one event per token
    POST /imagine    -> Stable Diffusion, Server-Sent Events, one event per
                        denoising step (optionally carrying a preview image)

The point of putting them behind the same interface is that the *client* code
is then identical, so any difference the load generator measures belongs to
the workload rather than to the plumbing.

Start it with:
    python3 server_both.py --port 8131 --threads 6
"""

from __future__ import annotations

import argparse
import base64
import io
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
import torch  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

SD_ID = "Lykon/dreamshaper-8"
_lock = threading.Lock()          # one model at a time, as in project 02
TRACE_PATH = os.environ.get("TRACE_PATH")
_trace_lock = threading.Lock()


def emit_trace(rec):
    if not TRACE_PATH:
        return
    with _trace_lock, open(TRACE_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def load_sd(threads):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    torch.set_num_threads(threads)
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++",
        final_sigmas_type="zero")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_app(tok, model, pipe):
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "Hello")
        max_new = int(body.get("max_new_tokens", 24))

        def sse():
            rid = uuid.uuid4().hex[:8]
            t0 = time.perf_counter()
            ids = tok(prompt, return_tensors="pt").input_ids
            with _lock:
                t_sched = time.perf_counter()
                with torch.inference_mode():
                    out = model(ids, use_cache=True)
                    past = out.past_key_values
                    nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
                    t_first = time.perf_counter()
                    for step in range(max_new):
                        if step > 0:
                            out = model(nid, past_key_values=past, use_cache=True)
                            past = out.past_key_values
                            nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
                        yield ("data: " + json.dumps(
                            {"text": tok.decode([int(nid)]), "i": step,
                             "content": True}) + "\n\n").encode()
                        if int(nid) == tok.eos_token_id:
                            break
            yield b"data: [DONE]\n\n"
            emit_trace({"kind": "llm", "request_id": rid,
                        "queue_s": t_sched - t0,
                        "first_content_s": t_first - t0,
                        "total_s": time.perf_counter() - t0,
                        "units": max_new})

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/imagine")
    async def imagine(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "a red apple on a table")
        steps = int(body.get("steps", 6))
        size = int(body.get("size", 256))
        preview = bool(body.get("preview", False))
        save_as = body.get("save_as")

        def sse():
            # The pipeline runs in a worker thread and pushes events onto a
            # queue, so each denoising step reaches the client WHEN IT HAPPENS.
            # (Building the list first and yielding at the end would make every
            # event arrive at the same moment -- and would quietly turn a
            # streaming measurement into a batched one.)
            import queue as _q
            rid = uuid.uuid4().hex[:8]
            t0 = time.perf_counter()
            q = _q.Queue()
            state = {"first_content_s": None, "t_sched": None}

            def worker():
                try:
                    with _lock:
                        state["t_sched"] = time.perf_counter()

                        def on_step(pipe_, step, timestep, kwargs):
                            """Called after every denoising step.

                            Without preview we can only report PROGRESS: the
                            latents are not an image until the VAE decodes
                            them, and that decode costs more than a denoising
                            step does.
                            """
                            if preview:
                                with torch.inference_mode():
                                    lat = (kwargs["latents"]
                                           / pipe_.vae.config.scaling_factor)
                                    img = pipe_.vae.decode(lat).sample
                                    img = pipe_.image_processor.postprocess(img)[0]
                                payload = {"i": step, "content": True,
                                           "png_b64_len": len(png_b64(img))}
                                if state["first_content_s"] is None:
                                    state["first_content_s"] = \
                                        time.perf_counter() - t0
                            else:
                                payload = {"i": step, "content": False}
                            q.put(payload)
                            return kwargs

                        with torch.inference_mode():
                            result = pipe(prompt, height=size, width=size,
                                          num_inference_steps=steps,
                                          guidance_scale=7.0,
                                          callback_on_step_end=on_step,
                                          callback_on_step_end_tensor_inputs=[
                                              "latents"])
                        image = result.images[0]
                        if state["first_content_s"] is None:
                            state["first_content_s"] = time.perf_counter() - t0
                        if save_as:
                            image.save(os.path.join(HERE, "outputs", save_as))
                        q.put({"i": -1, "content": True, "final": True,
                               "png_b64_len": len(png_b64(image))})
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                item = q.get()
                if item is None:
                    break
                yield ("data: " + json.dumps(item) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            emit_trace({"kind": "sd", "request_id": rid, "steps": steps,
                        "preview": preview,
                        "queue_s": state["t_sched"] - t0,
                        "first_content_s": state["first_content_s"],
                        "total_s": time.perf_counter() - t0,
                        "units": steps})

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    import uvicorn
    tok, model = L.load(n_threads=args.threads)
    pipe = load_sd(args.threads)
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    uvicorn.run(build_app(tok, model, pipe), host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
