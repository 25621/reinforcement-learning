"""Project 08 — Diffusion vs LLM serving.

Same machine, same HTTP interface, same load generator, two workloads that
share almost nothing:

  A. cost anatomy      — what one "unit of work" is for each model
  B. batch scaling     — the same 8x batch, measured on both
  C. under load        — TTFT, first *visible* content, throughput at c=1 and 4
  D. the steps knob    — a latency dial diffusion has and an LLM does not
  E. state             — what each workload keeps in memory per request

Run:  python3 run.py          (~5 min)
      python3 run.py --plot   (redraw from outputs/findings.json)
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
P01 = os.path.join(os.path.dirname(HERE), "01-manual-inference-loop")
sys.path.insert(0, HERE)
sys.path.insert(0, P01)

import torch  # noqa: E402

PORT = 8131
URL = f"http://127.0.0.1:{PORT}"
THREADS = 6
SD_STEPS = 6
SD_SIZE = 256
LLM_TOKENS = 24
LLM_PROMPT = "In one sentence: what is a diffusion model?"
SD_PROMPT = "a red apple on a wooden table, studio photo"


# ---------------------------------------------------------------------------
# In-process measurements (A, B, D)
# ---------------------------------------------------------------------------


def section_a_b_d(f):
    import loop_lib as L
    from server_both import load_sd

    torch.set_num_threads(THREADS)
    tok, model = L.load(n_threads=THREADS)
    pipe = load_sd(THREADS)
    shape = L.model_shape(model)

    ids = tok(LLM_PROMPT, return_tensors="pt").input_ids
    lat = torch.randn(1, 4, SD_SIZE // 8, SD_SIZE // 8)
    ts = torch.tensor([500])
    emb = torch.randn(1, 77, 768)

    with torch.inference_mode():
        # --- A. one unit of work on each side -----------------------------
        warm = model(ids, use_cache=True)
        past, nid = warm.past_key_values, warm.logits[:, -1:].argmax(-1)
        import copy

        fns = {
            "llm_prefill": lambda: model(ids, use_cache=True),
            "llm_decode_step": lambda: model(
                nid, past_key_values=copy.deepcopy(past), use_cache=True),
            "sd_unet_step_1_image": lambda: pipe.unet(
                lat, ts, encoder_hidden_states=emb),
            "sd_unet_step_cfg": lambda: pipe.unet(
                lat.repeat(2, 1, 1, 1), ts,
                encoder_hidden_states=emb.repeat(2, 1, 1)),
            "sd_vae_decode": lambda: pipe.vae.decode(
                lat / pipe.vae.config.scaling_factor),
        }
        best = L.interleaved(fns, rounds=3, warmup=1)
        f["A_unit_costs"] = {k: round(1000 * v, 1) for k, v in best.items()}
        f["A_unit_costs"]["llm_prefill_tokens"] = int(ids.shape[1])
        for k, v in best.items():
            print(f"  A: {k:24s} {1000 * v:8.1f} ms")

        # --- B. batch scaling, both sides, interleaved --------------------
        rows = []
        for b in (1, 2, 4, 8):
            x, e = lat.repeat(b, 1, 1, 1), emb.repeat(b, 1, 1)
            warm_b = model(torch.randint(0, 5000, (b, 32)), use_cache=True)
            past_b = warm_b.past_key_values
            nid_b = torch.randint(0, 5000, (b, 1))
            fns = {
                "sd": lambda x=x, e=e: pipe.unet(x, ts, encoder_hidden_states=e),
                "llm": lambda p=past_b, n=nid_b: model(
                    n, past_key_values=copy.deepcopy(p), use_cache=True),
            }
            bb = L.interleaved(fns, rounds=2, warmup=1)
            rows.append({"batch": b,
                         "sd_step_s": round(bb["sd"], 4),
                         "sd_images_per_s": round(b / bb["sd"], 3),
                         "llm_step_s": round(bb["llm"], 4),
                         "llm_tokens_per_s": round(b / bb["llm"], 1)})
            print(f"  B: batch {b}: SD {bb['sd']:.3f}s/step "
                  f"({b / bb['sd']:.2f} img-steps/s) | LLM {bb['llm']:.3f}s/step "
                  f"({b / bb['llm']:.1f} tok/s)")
        base = rows[0]
        for r in rows:
            r["sd_throughput_gain"] = round(
                r["sd_images_per_s"] / base["sd_images_per_s"], 2)
            r["llm_throughput_gain"] = round(
                r["llm_tokens_per_s"] / base["llm_tokens_per_s"], 2)
        f["B_batch_scaling"] = rows

        # --- D. the steps knob --------------------------------------------
        steps_rows = []
        pipe(SD_PROMPT, height=SD_SIZE, width=SD_SIZE, num_inference_steps=2,
             guidance_scale=7.0)          # warm-up: the first call pays extra
        for n in (4, 8, 12):
            t0 = time.perf_counter()
            img = pipe(SD_PROMPT, height=SD_SIZE, width=SD_SIZE,
                       num_inference_steps=n, guidance_scale=7.0).images[0]
            dt = time.perf_counter() - t0
            img.save(os.path.join(OUT, f"sample_{n}steps.png"))
            steps_rows.append({"steps": n, "seconds": round(dt, 2),
                               "s_per_step": round(dt / n, 3)})
            print(f"  D: {n:2d} steps -> {dt:.2f}s ({dt / n:.3f}s/step)")
        f["D_steps_knob"] = steps_rows

    # --- E. what each request keeps in memory ------------------------------
    kv_per_token = L.kv_bytes_per_token(shape, dtype_bytes=4)
    latent_bytes = 4 * (SD_SIZE // 8) ** 2 * 4
    f["E_state"] = {
        "llm_kv_bytes_per_token": kv_per_token,
        "llm_kv_bytes_at_1k_tokens": kv_per_token * 1024,
        "llm_kv_bytes_at_8k_tokens": kv_per_token * 8192,
        "sd_latent_bytes": latent_bytes,
        "sd_latent_bytes_at_any_step_count": latent_bytes,
        "ratio_at_1k": round(kv_per_token * 1024 / latent_bytes, 1)}
    print(f"  E: LLM KV cache {kv_per_token} B/token (={kv_per_token * 1024 / 1e6:.1f} "
          f"MB at 1k tokens); SD latent {latent_bytes} B, constant")
    del model, pipe


# ---------------------------------------------------------------------------
# Server measurements (C)
# ---------------------------------------------------------------------------


def start_server():
    env = dict(os.environ)
    env["TRACE_PATH"] = os.path.join(OUT, "trace.jsonl")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "server_both.py"),
         "--port", str(PORT), "--threads", str(THREADS)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            if httpx.get(URL + "/health", timeout=2.0).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError("server did not start")


def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


async def one_request(client, path, payload):
    """One streamed request; records when the first EVENT and the first
    piece of user-visible CONTENT arrived. For an LLM they are the same
    event; for diffusion they are not."""
    t0 = time.perf_counter()
    first_event = first_content = None
    n_events = 0
    async with client.stream("POST", URL + path, json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            now = time.perf_counter()
            body = line[6:]
            if body.strip() == "[DONE]":
                break
            ev = json.loads(body)
            n_events += 1
            if first_event is None:
                first_event = now - t0
            if ev.get("content") and first_content is None:
                first_content = now - t0
    return {"first_event_s": first_event, "first_content_s": first_content,
            "e2e_s": time.perf_counter() - t0, "events": n_events}


async def _load(path, payload, concurrency, n):
    sem = asyncio.Semaphore(concurrency)
    res = []
    async with httpx.AsyncClient(timeout=1800.0) as client:
        async def one(i):
            async with sem:
                res.append(await one_request(client, path, dict(payload)))
        t0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(n)])
        wall = time.perf_counter() - t0
    return res, wall


def load(path, payload, concurrency, n, units_per_request, label):
    res, wall = asyncio.run(_load(path, payload, concurrency, n))
    out = {"label": label, "path": path, "concurrency": concurrency,
           "requests": n, "wall_s": round(wall, 2),
           "first_event_p50_s": round(statistics.median(
               [r["first_event_s"] for r in res]), 3),
           "first_content_p50_s": round(statistics.median(
               [r["first_content_s"] for r in res]), 3),
           "e2e_p50_s": round(statistics.median([r["e2e_s"] for r in res]), 3),
           "e2e_max_s": round(max(r["e2e_s"] for r in res), 3),
           "units_per_s": round(n * units_per_request / wall, 3),
           "requests_per_s": round(n / wall, 3),
           "content_share_of_e2e": None}
    out["content_share_of_e2e"] = round(
        out["first_content_p50_s"] / out["e2e_p50_s"], 3)
    print(f"  C: {label:34s} first event {out['first_event_p50_s']:7.3f}s | "
          f"first CONTENT {out['first_content_p50_s']:7.3f}s "
          f"({100 * out['content_share_of_e2e']:5.1f}% of E2E) | "
          f"E2E {out['e2e_p50_s']:7.3f}s | {out['requests_per_s']:.3f} req/s")
    return out


def section_c(f):
    print("starting the two-model server ...")
    proc = start_server()
    try:
        # warm both paths so no measurement pays first-touch costs
        asyncio.run(_load("/generate", {"prompt": LLM_PROMPT,
                                        "max_new_tokens": 2}, 1, 1))
        asyncio.run(_load("/imagine", {"prompt": SD_PROMPT, "steps": 2,
                                       "size": SD_SIZE}, 1, 1))
        rows = [
            load("/generate", {"prompt": LLM_PROMPT,
                               "max_new_tokens": LLM_TOKENS}, 1, 3,
                 LLM_TOKENS, "LLM, concurrency 1"),
            load("/generate", {"prompt": LLM_PROMPT,
                               "max_new_tokens": LLM_TOKENS}, 4, 8,
                 LLM_TOKENS, "LLM, concurrency 4"),
            load("/imagine", {"prompt": SD_PROMPT, "steps": SD_STEPS,
                              "size": SD_SIZE, "save_as": "sample.png"}, 1, 2,
                 1, "diffusion, concurrency 1"),
            load("/imagine", {"prompt": SD_PROMPT, "steps": SD_STEPS,
                              "size": SD_SIZE}, 4, 4, 1,
                 "diffusion, concurrency 4"),
            load("/imagine", {"prompt": SD_PROMPT, "steps": SD_STEPS,
                              "size": SD_SIZE, "preview": True}, 1, 1, 1,
                 "diffusion + per-step preview"),
        ]
        f["C_under_load"] = rows
    finally:
        stop_server(proc)


# ---------------------------------------------------------------------------


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    b = f["B_batch_scaling"]
    bs = [r["batch"] for r in b]
    ax[0].plot(bs, [r["llm_throughput_gain"] for r in b], "o-",
               color="tab:blue", label="LLM decode (memory-bound)")
    ax[0].plot(bs, [r["sd_throughput_gain"] for r in b], "s--",
               color="tab:red", label="diffusion U-Net step (compute-bound)")
    ax[0].plot(bs, bs, ":", color="grey", label="perfect scaling")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("throughput, relative to batch 1")
    ax[0].set_title("B. the same 8x batch is worth\n"
                    f"{b[-1]['llm_throughput_gain']}x to one and "
                    f"{b[-1]['sd_throughput_gain']}x to the other")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")

    c = f["C_under_load"]
    short = {"LLM, concurrency 1": "LLM\nc=1", "LLM, concurrency 4": "LLM\nc=4",
             "diffusion, concurrency 1": "diffusion\nc=1",
             "diffusion, concurrency 4": "diffusion\nc=4",
             "diffusion + per-step preview": "diffusion\n+preview"}
    labels = [short.get(r["label"], r["label"]) for r in c]
    xs = range(len(c))
    ax[1].bar([x - .2 for x in xs], [100 * r["content_share_of_e2e"] for r in c],
              width=.4, color="tab:orange", label="first visible content")
    ax[1].bar([x + .2 for x in xs], [100] * len(c), width=.4,
              color="tab:blue", alpha=.35, label="end to end")
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("% of the request's own end-to-end time")
    ax[1].set_title("C. when does the user first see\nsomething real?")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")
    for i, r in enumerate(c):
        ax[1].text(i - .2, 100 * r["content_share_of_e2e"] + 2,
                   f"{100 * r['content_share_of_e2e']:.0f}%", ha="center",
                   fontsize=8)

    a = f["A_unit_costs"]
    keys = ["llm_decode_step", "llm_prefill", "sd_unet_step_1_image",
            "sd_unet_step_cfg", "sd_vae_decode"]
    ax[2].barh(range(len(keys)), [a[k] for k in keys],
               color=["tab:blue", "tab:cyan", "tab:red", "tab:orange",
                      "tab:purple"])
    ax[2].set_yticks(range(len(keys)))
    ax[2].set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("milliseconds for one unit of work")
    ax[2].set_title("A. one 'step' means very different\nthings in the two stacks")
    ax[2].grid(alpha=.3, axis="x")

    fig.tight_layout()
    p = os.path.join(OUT, "diffusion_vs_llm.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return
    t0 = time.time()
    trace = os.path.join(OUT, "trace.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    f = {"llm": "Qwen/Qwen2.5-0.5B-Instruct", "diffusion": "Lykon/dreamshaper-8",
         "threads": THREADS, "sd_steps": SD_STEPS, "sd_size": SD_SIZE,
         "llm_tokens": LLM_TOKENS}
    section_a_b_d(f)
    section_c(f)
    f["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(f, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        for k, v in f["A_unit_costs"].items():
            fh.write(f"A,{k}_ms,{v}\n")
        for r in f["B_batch_scaling"]:
            fh.write(f"B,llm_gain@b{r['batch']},{r['llm_throughput_gain']}\n")
            fh.write(f"B,sd_gain@b{r['batch']},{r['sd_throughput_gain']}\n")
        for r in f["C_under_load"]:
            key = r["label"].replace(" ", "_").replace(",", "")
            fh.write(f"C,{key}|first_content_s,{r['first_content_p50_s']}\n")
            fh.write(f"C,{key}|e2e_s,{r['e2e_p50_s']}\n")
            fh.write(f"C,{key}|req_per_s,{r['requests_per_s']}\n")
        for r in f["D_steps_knob"]:
            fh.write(f"D,seconds@{r['steps']}steps,{r['seconds']}\n")
        for k, v in f["E_state"].items():
            fh.write(f"E,{k},{v}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
