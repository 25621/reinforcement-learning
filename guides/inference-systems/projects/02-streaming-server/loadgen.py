"""A tiny load generator (the `oha`/`vegeta` role, written in Python).

Single-stream timing is a lie: it never shows queueing. This sends N requests
at a fixed concurrency level and reports, per request, the two clocks a user
feels — TTFT (bytes of the first token) and ITL (gaps between later tokens).
"""

from __future__ import annotations

import asyncio
import statistics
import time

import httpx


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


async def one_stream(client, url, prompt, max_new, chunk_every):
    t0 = time.perf_counter()
    ttft = None
    arrivals = []
    n_chunks = 0
    payload = {"prompt": prompt, "max_new_tokens": max_new,
               "chunk_every": chunk_every}
    async with client.stream("POST", url + "/generate", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            now = time.perf_counter()
            if line.strip() == "data: [DONE]":
                break
            if ttft is None:
                ttft = now - t0
            arrivals.append(now)
            n_chunks += 1
    end = time.perf_counter()
    itls = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft_s": ttft, "e2e_s": end - t0, "itls": itls,
            "n_chunks": n_chunks}


async def one_full(client, url, prompt, max_new):
    t0 = time.perf_counter()
    r = await client.post(url + "/generate_full",
                          json={"prompt": prompt, "max_new_tokens": max_new})
    body = r.json()
    end = time.perf_counter()
    # Nothing arrives until the whole answer does: TTFT == E2E by construction.
    return {"ttft_s": end - t0, "e2e_s": end - t0, "itls": [],
            "n_chunks": body.get("n_output_tokens", 0)}


async def _run(url, prompt, concurrency, n_requests, max_new, stream,
               chunk_every, timeout):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def worker(_i):
            async with sem:
                if stream:
                    res = await one_stream(client, url, prompt, max_new,
                                           chunk_every)
                else:
                    res = await one_full(client, url, prompt, max_new)
                results.append(res)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(n_requests)])
        wall = time.perf_counter() - t0
    return results, wall


def run_load(url, prompt, concurrency=1, n_requests=4, max_new=32,
             stream=True, chunk_every=1, timeout=600.0):
    results, wall = asyncio.run(_run(url, prompt, concurrency, n_requests,
                                     max_new, stream, chunk_every, timeout))
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    itls = [x for r in results for x in r["itls"]]
    toks = sum(r["n_chunks"] for r in results) * chunk_every
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "stream": stream,
        "chunk_every": chunk_every,
        "wall_s": round(wall, 3),
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p99_s": round(pct(ttfts, 99), 3) if ttfts else None,
        "ttft_mean_s": round(statistics.fmean(ttfts), 3) if ttfts else None,
        "itl_p50_s": round(statistics.median(itls), 4) if itls else None,
        "itl_p99_s": round(pct(itls, 99), 4) if itls else None,
        "e2e_p50_s": round(statistics.median([r["e2e_s"] for r in results]), 3),
        "output_tokens": toks,
        "throughput_tok_s": round(toks / wall, 1),
    }
