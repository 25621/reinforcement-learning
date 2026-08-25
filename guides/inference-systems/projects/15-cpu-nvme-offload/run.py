"""Project 15 -- CPU/NVMe offload.

Adds a second, slower tier under the KV cache and measures what it buys and
what it costs.

  A. How fast are the tiers, really? (RAM copy vs NVMe write vs NVMe read,
     with the OS page cache dropped so the disk numbers are honest.)
  B. Offloading blocks of an *active* sequence: what happens to decode.
  C. Offloading a *paused session*: save and restore a whole conversation.
  D. The decision every long-session system faces: when a paused session comes
     back, is it cheaper to reload its cache from disk or to recompute it?

    python3 run.py           # ~3 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

import torch  # noqa: E402
import kvlib  # noqa: E402
from tiered import Tier2, TieredCache  # noqa: E402

BLOCK = 64


def build_prompt(tok, n):
    body = ("A long-running assistant session keeps the whole conversation in "
            "its key-value cache, and that cache has to live somewhere. ") * 400
    return tok(body, return_tensors="pt").input_ids[:, :n]


# ---------------------------------------------------------------------------


def bench_tiers(shape, dtype=torch.float32, rounds=24):
    """Time one block-sized move on each tier.

    The five operations are timed *round-robin* rather than one after the
    other, and the minimum of each is kept. This box is shared and idles at
    load 3-6; timing all the RAM operations first and all the disk ones later
    charges whatever else the machine did in between entirely to one tier,
    which on an earlier draft moved the RAM numbers by 8x between runs.
    """
    k = torch.randn(*shape, dtype=dtype)
    v = torch.randn(*shape, dtype=dtype)
    nb = (k.numel() + v.numel()) * k.element_size()
    out = {"block_bytes": nb, "rounds": rounds}

    ram = Tier2("ram")
    nvme = Tier2("nvme")
    best = {k_: float("inf") for k_ in
            ("ram_write", "ram_read", "nvme_write", "nvme_read_warm",
             "nvme_read_cold")}
    for i in range(rounds + 1):          # round 0 is warm-up, discarded
        t0 = time.perf_counter(); ram.put(i, k, v); a = time.perf_counter() - t0
        t0 = time.perf_counter(); ram.get(i); b = time.perf_counter() - t0
        t0 = time.perf_counter(); nvme.put(i, k, v); c = time.perf_counter() - t0
        _, _, d = nvme.get(i, cold=False)
        _, _, e = nvme.get(i, cold=True)
        if i == 0:
            continue
        for name, dt in (("ram_write", a), ("ram_read", b), ("nvme_write", c),
                         ("nvme_read_warm", d), ("nvme_read_cold", e)):
            best[name] = min(best[name], dt)
    nvme.cleanup()
    for key, dt in best.items():
        out[key + "_s"] = dt
        out[key + "_gbps"] = nb / dt / 1e9
    return out


def main():
    f = {}
    runner, tok, _ = kvlib.load_runner()
    per_tok = kvlib.kv_bytes_per_token(runner.n_layers, runner.n_kv_heads,
                                       runner.d_head, 4)
    f["model"] = {"kv_bytes_per_token": per_tok, "block_tokens": BLOCK}

    # ------------------------------------------------------------------ A
    print("A. how fast is each tier? (one 64-token block of the whole model)")
    shape = (1, runner.n_kv_heads, BLOCK, runner.d_head)
    per_layer = bench_tiers(shape)
    f["A_tiers"] = per_layer
    for key in ("ram_write", "ram_read", "nvme_write", "nvme_read_warm",
                "nvme_read_cold"):
        print(f"   {key:>16}: {per_layer[key+'_s']*1e6:8.1f} us  "
              f"{per_layer[key+'_gbps']:6.2f} GB/s")

    # ------------------------------------------------------------------ B
    print("B. offloading an ACTIVE sequence")
    prompt = build_prompt(tok, 512)
    n_blocks = 512 // BLOCK
    settings = [n_blocks, 4, 2, 1]
    # Round-robin again, for the same reason as section A.
    best = {r: {"median_step_s": float("inf")} for r in settings}
    for _ in range(2):
        for resident in settings:
            cache = TieredCache(runner.n_layers, block=BLOCK,
                                resident_blocks=resident, backend="nvme")
            _, pf, st = runner.generate(prompt, cache, max_new_tokens=9,
                                        stop_on_eos=False)
            med = statistics.median(st)
            if med < best[resident]["median_step_s"]:
                best[resident] = {
                    "resident_blocks": resident,
                    "resident_frac": min(1.0, resident / n_blocks),
                    "median_step_s": med,
                    "fetches": cache.fetches, "fetch_s": cache.fetch_s,
                    "writes_skipped": cache.writes_skipped,
                    "resident_bytes": cache.resident_bytes()}
            cache.t2.cleanup()
    b_rows = [best[r] for r in settings]
    for row in b_rows:
        print(f"   resident {row['resident_blocks']:>2}/{n_blocks} blocks: step "
              f"{row['median_step_s']*1e3:8.1f} ms  "
              f"({row['median_step_s']/b_rows[0]['median_step_s']:5.2f}x)  "
              f"{row['fetches']} fetches", flush=True)
    f["B_active"] = b_rows

    # ------------------------------------------------------------------ C
    print("C. offloading a PAUSED session (whole cache at once)")
    c_rows = []
    for ctx in (256, 512, 1024, 2048):
        p = build_prompt(tok, ctx)
        cache = kvlib.ContiguousCache(runner.n_layers)
        t0 = time.perf_counter()
        runner.forward(p, cache)
        prefill_s = time.perf_counter() - t0

        t2 = Tier2("nvme")
        t0 = time.perf_counter()
        for layer in range(runner.n_layers):
            t2.put(layer, cache.k[layer][0], cache.v[layer][0])
        save_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for layer in range(runner.n_layers):
            t2.get(layer, cold=True)
        load_s = time.perf_counter() - t0
        t2.cleanup()

        c_rows.append({"ctx": ctx, "prefill_s": prefill_s, "save_s": save_s,
                       "load_s": load_s, "bytes": cache.nbytes(),
                       "recompute_vs_reload": prefill_s / load_s})
        print(f"   ctx {ctx:>5}: cache {cache.nbytes()/1e6:6.1f} MB  "
              f"prefill {prefill_s:6.2f} s  save {save_s:5.2f} s  "
              f"reload {load_s:5.2f} s  -> reloading is "
              f"{prefill_s/load_s:5.1f}x cheaper than recomputing", flush=True)
    f["C_session"] = c_rows

    # ------------------------------------------------------------------ D
    # The same decision on hardware this box does not have, done as
    # arithmetic and labelled as such.
    print("D. the same trade on an H100-class box (arithmetic, not measured)")
    scen = []
    for name, kv_per_tok, prefill_tok_s, link_gbps in [
            ("Llama-3.1-8B on 1xH100, NVMe @ 6 GB/s", 128 * 1024, 12000, 6),
            ("Llama-3.1-8B on 1xH100, host RAM @ 25 GB/s", 128 * 1024, 12000, 25),
            ("Llama-3.1-70B on 4xH100, NVMe @ 6 GB/s", 320 * 1024, 4000, 6),
            ("Llama-3.1-70B on 4xH100, host RAM @ 25 GB/s", 320 * 1024, 4000, 25)]:
        for ctx in (2048, 8192, 32768):
            recompute = ctx / prefill_tok_s
            reload_ = ctx * kv_per_tok / (link_gbps * 1e9)
            scen.append({"scenario": name, "ctx": ctx,
                         "recompute_s": recompute, "reload_s": reload_,
                         "ratio": recompute / reload_})
        print(f"   {name}: at 8k, recompute "
              f"{8192/prefill_tok_s*1e3:6.1f} ms vs reload "
              f"{8192*kv_per_tok/(link_gbps*1e9)*1e3:6.1f} ms")
    f["D_scenarios"] = scen

    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "session.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["C_session"][0].keys()))
        w.writeheader()
        for r in f["C_session"]:
            w.writerow(r)
    with open(os.path.join(OUT, "active.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["B_active"][0].keys()))
        w.writeheader()
        for r in f["B_active"]:
            w.writerow(r)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = f["A_tiers"]
    keys = ["ram_write", "ram_read", "nvme_write", "nvme_read_warm",
            "nvme_read_cold"]
    ax[0].bar([k.replace("_", "\n") for k in keys],
              [a[k + "_gbps"] for k in keys],
              color=["tab:blue", "tab:blue", "tab:orange", "tab:green",
                     "tab:red"])
    for i, k in enumerate(keys):
        ax[0].text(i, a[k + "_gbps"], f"{a[k+'_gbps']:.2f}", ha="center",
                   va="bottom", fontsize=8)
    ax[0].set_ylabel("GB/s")
    ax[0].set_yscale("log")
    ax[0].set_title("A. the tiers, measured")
    ax[0].tick_params(labelsize=7)
    ax[0].grid(alpha=.3, axis="y")

    b = f["B_active"]
    base = b[0]["median_step_s"]
    ax[1].plot([r["resident_frac"] * 100 for r in b],
               [r["median_step_s"] / base for r in b], "o-", color="tab:red")
    ax[1].set_xlabel("% of the cache kept resident")
    ax[1].set_ylabel("decode step, relative to no offload")
    ax[1].set_yscale("log")
    ax[1].set_title("B. an active sequence pays a 1.3-1.5x tax\n(it re-reads everything, every step)")
    ax[1].grid(alpha=.3)

    c = f["C_session"]
    x = range(len(c))
    ax[2].bar([i - 0.2 for i in x], [r["prefill_s"] for r in c], width=0.4,
              label="recompute (prefill)")
    ax[2].bar([i + 0.2 for i in x], [r["load_s"] for r in c], width=0.4,
              label="reload from NVMe")
    ax[2].set_xticks(list(x))
    ax[2].set_xticklabels([f"{r['ctx']}" for r in c])
    ax[2].set_xlabel("session context (tokens)")
    ax[2].set_ylabel("seconds")
    ax[2].set_yscale("log")
    ax[2].set_title("C. resuming a paused session")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "offload.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
