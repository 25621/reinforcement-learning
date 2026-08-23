"""The distributed worker for project 44 -- launch with:

    torchrun --nproc_per_node=2 tp_run.py

Both ranks load the full model (2 GB fp32), then slice it into a TPRunner.
Rank 0 additionally keeps an UNSLICED BatchedRunner as the single-device
reference -- slicing makes views, so the reference costs no extra weight
memory. Sections:

  A  correctness  : TP=2 logits vs single-runner logits, 64 greedy steps
  B  step timing  : decode step time + all-reduce share, batch 1/4/16/32
  C  wire micro   : gloo all-reduce latency vs payload size
  D  memory       : per-rank bytes, sharded vs replicated

Writes outputs/findings_dist.json (rank 0 only).
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "16-static-vs-continuous"))

import batchlib  # noqa: E402
from batchlib import SlotKV  # noqa: E402
from tplib import TPRunner  # noqa: E402

MAX_LEN = 320
REPS = 12


def make_pool(runner, n_slots):
    return SlotKV(runner.n_layers, n_slots, runner.n_kv_heads, runner.d_head, MAX_LEN)


def forced_steps(runner, pool, ids_list, n_new, seed=13):
    """Prefill a batch then decode n_new steps, feeding the SAME pre-drawn
    random continuation tokens on every runner (teacher forcing). Driving each
    runner with its own argmax would couple the two trajectories: one tied
    argmax and every later logit legitimately differs. Forcing identical
    inputs keeps every step comparable on its own."""
    b = len(ids_list)
    slots = list(range(b))
    t = max(len(x) for x in ids_list)
    padded = torch.zeros(b, t, dtype=torch.long)
    for i, x in enumerate(ids_list):
        padded[i, : len(x)] = torch.tensor(x)
    lens = [len(x) for x in ids_list]
    logits, _ = runner.prefill(pool, slots, padded, lens, count=False)
    out = [logits]
    rng = torch.Generator().manual_seed(seed)
    cur = list(lens)
    for _ in range(n_new):
        toks = torch.randint(1000, 12000, (b,), generator=rng).tolist()
        logits, _ = runner.decode_step(pool, slots, toks, cur, count=False)
        out.append(logits)
        cur = [c + 1 for c in cur]
    return out


def timed_decode(runner, pool, batch, ctx, reps=REPS):
    """Median decode-step time at batch `batch`, every row at context `ctx`."""
    slots = list(range(batch))
    toks = [11 + i for i in range(batch)]
    lens = [ctx] * batch
    # touch the pool rows so nothing is cold
    runner.decode_step(pool, slots, toks, lens, count=False)
    times = []
    for _ in range(reps):
        if isinstance(runner, TPRunner):
            runner.reset_comm()
            dist.barrier()
        t0 = time.perf_counter()
        _, _ = runner.decode_step(pool, slots, toks, lens, count=False)
        dt = time.perf_counter() - t0
        comm = runner.comm_s if isinstance(runner, TPRunner) else 0.0
        times.append((dt, comm))
    times.sort()
    mid = times[len(times) // 2]
    return mid[0], mid[1]


def main():
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(3)
    tok = AutoTokenizer.from_pretrained(batchlib.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(batchlib.MODEL_ID, dtype=torch.float32)
    model.eval()

    tp = TPRunner(model, tok, rank, world)
    ref = batchlib.BatchedRunner(model, tok) if rank == 0 else None
    find = {}

    # ---- A. correctness --------------------------------------------------
    rng = torch.Generator().manual_seed(7)
    ids_list = [torch.randint(1000, 12000, (n,), generator=rng).tolist()
                for n in (37, 64, 128)]
    pool_tp = make_pool(tp, 4)
    tp_logits = forced_steps(tp, pool_tp, ids_list, 64)
    if rank == 0:
        pool_ref = make_pool(ref, 4)
        ref_logits = forced_steps(ref, pool_ref, ids_list, 64)
        diffs = [float((a - b).abs().max()) for a, b in zip(tp_logits, ref_logits)]
        agree = sum(int(torch.equal(a.argmax(-1), b.argmax(-1)))
                    for a, b in zip(tp_logits, ref_logits))
        find["correct"] = {
            "steps": len(tp_logits),
            "argmax_agree": agree,
            "max_abs_diff": max(diffs),
            "per_step_diff": diffs,
            "ref_logit_scale": float(ref_logits[0].abs().mean()),
        }
        print(f"[A] argmax agree {agree}/{len(tp_logits)}  max|diff| {max(diffs):.2e}",
              flush=True)

    # ---- B. decode step timing ------------------------------------------
    dist.barrier()
    rows = []
    ctx = 256
    for batch in (1, 4, 16, 32):
        # A zero-filled pool is numerically safe for timing (softmax over
        # zeros is uniform, never NaN) and skips a costly warm-up prefill.
        pool_tp = make_pool(tp, batch)
        t_tp, t_comm = timed_decode(tp, pool_tp, batch, ctx)
        if rank == 0:
            torch.set_num_threads(6)
            pool_ref = make_pool(ref, batch)
            t_ref, _ = timed_decode(ref, pool_ref, batch, ctx)
            torch.set_num_threads(3)
            payload = tp.comm_bytes // tp.comm_calls if tp.comm_calls else 0
            rows.append({
                "batch": batch, "ctx": ctx,
                "single_ms": t_ref * 1e3, "tp_ms": t_tp * 1e3,
                "comm_ms": t_comm * 1e3,
                "comm_share": t_comm / t_tp,
                "all_reduces": tp.comm_calls,
                "payload_bytes": payload,
                "speedup": t_ref / t_tp,
            })
            print(f"[B] B={batch:3d} single {t_ref*1e3:7.1f} ms  tp {t_tp*1e3:7.1f} ms "
                  f"(comm {t_comm*1e3:6.1f} ms = {t_comm/t_tp:5.1%})", flush=True)
        dist.barrier()
    if rank == 0:
        find["steps"] = rows

    # ---- also one prefill point (the compute-heavy side) ------------------
    pool_tp = SlotKV(tp.n_layers, 1, tp.n_kv_heads, tp.d_head, 512)
    ids = torch.randint(1000, 12000, (1, 512))
    tp.prefill(pool_tp, [0], ids, [512], count=False)  # warm
    best_tp, best_comm = 1e9, 0.0
    for _ in range(3):
        tp.reset_comm()
        dist.barrier()
        t0 = time.perf_counter()
        tp.prefill(pool_tp, [0], ids, [512], count=False)
        dt = time.perf_counter() - t0
        if dt < best_tp:
            best_tp, best_comm = dt, tp.comm_s
    if rank == 0:
        torch.set_num_threads(6)
        pool_ref = SlotKV(ref.n_layers, 1, ref.n_kv_heads, ref.d_head, 512)
        ref.prefill(pool_ref, [0], ids, [512], count=False)
        best_ref = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            ref.prefill(pool_ref, [0], ids, [512], count=False)
            best_ref = min(best_ref, time.perf_counter() - t0)
        torch.set_num_threads(3)
        find["prefill_512"] = {
            "single_ms": best_ref * 1e3, "tp_ms": best_tp * 1e3,
            "comm_ms": best_comm * 1e3, "comm_share": best_comm / best_tp,
            "speedup": best_ref / best_tp,
        }
        print(f"[B'] prefill 512: single {best_ref*1e3:.0f} ms  tp {best_tp*1e3:.0f} ms "
              f"(comm {best_comm/best_tp:.1%})", flush=True)
    dist.barrier()

    # ---- C. all-reduce microbenchmark -------------------------------------
    sizes = [2 ** p for p in range(10, 24)]  # 1 KiB .. 8 MiB of fp32 bytes
    micro = []
    for nbytes in sizes:
        t = torch.zeros(nbytes // 4)
        for _ in range(3):
            dist.all_reduce(t)  # warm the connection
        dist.barrier()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            dist.all_reduce(t)
        dt = (time.perf_counter() - t0) / n
        if rank == 0:
            micro.append({"bytes": nbytes, "lat_us": dt * 1e6,
                          "gbps": nbytes / dt / 1e9})
    if rank == 0:
        find["allreduce_micro"] = micro
        floor = micro[0]["lat_us"]
        print(f"[C] all-reduce floor {floor:.0f} us at 1 KiB; "
              f"{micro[-1]['gbps']:.1f} GB/s at 8 MiB", flush=True)

    # ---- D. memory accounting ---------------------------------------------
    if rank == 0:
        shard, rep = tp.param_bytes()
        f_shard, f_rep = ref_shard_bytes(ref)
        find["memory"] = {
            "tp_shard_bytes": shard, "tp_replicated_bytes": rep,
            "full_shard_candidates": f_shard, "full_replicated": f_rep,
            "tp_total": shard + rep, "full_total": f_shard + f_rep,
            "kv_bytes_per_token_full": 2 * ref.n_layers * ref.n_kv_heads * ref.d_head * 4,
            "kv_bytes_per_token_rank": 2 * tp.n_layers * tp.n_kv_heads * tp.d_head * 4,
        }
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        with open(os.path.join(HERE, "outputs", "findings_dist.json"), "w") as f:
            json.dump(find, f, indent=1)
        print("[D] wrote findings_dist.json", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def ref_shard_bytes(ref):
    shard = rep = 0
    for p in ref.layers:
        for k in ("wq", "bq", "wk", "bk", "wv", "bv", "wo", "gate", "up", "down"):
            shard += p[k].numel() * 4
        rep += (p["ln1"].numel() + p["ln2"].numel()) * 4
    rep += (ref.embed.numel() + ref.norm.numel()) * 4
    return shard, rep


if __name__ == "__main__":
    main()
