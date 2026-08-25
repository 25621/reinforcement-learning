"""Ring attention: exact attention over a sequence no single device holds.

The problem. A 64k-token prompt's keys and values do not fit on one device.
Splitting the sequence across P devices solves the memory problem and creates
a new one: a query on device 0 still has to see keys on device 3.

The trick. Do not gather the keys -- pass them round. Each device keeps its
own slice of Q for the whole computation and, over P rounds, sees every
other device's K/V block exactly once. After P rounds every query has
attended to every key, and no device ever held more than 2/P of the K/V.

The catch. Softmax is not associative in the obvious way: you cannot average
P separate softmaxes. The fix is the *online softmax* (the same running-max
trick FlashAttention uses):

    running max      m  -- the largest score seen so far
    running sum      l  -- sum of exp(score - m) so far
    running output   o  -- sum of exp(score - m) * V so far

When a new block arrives with a bigger max, rescale what you already have by
exp(m_old - m_new) and carry on. The final output equals computing the whole
thing at once -- this file checks that rather than assuming it.

The second catch, which is the interesting one. Causal masking makes the
work *unequal*. A query near the start of the sequence attends to almost
nothing; a query near the end attends to everything. Hand out contiguous
slices and the device holding the start idles while the device holding the
end sets the pace. `zigzag_chunks` is the standard fix, and this file
measures both.

Launched by `run.py` through torchrun; the backend is gloo, because the GPU
in this machine (sm_61) is not usable from this PyTorch build. The algorithm,
the communication pattern and the imbalance are identical; only the wire is
slower.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Online softmax
# ---------------------------------------------------------------------------


def online_update(m, l, o, scores, v):
    """Fold one block of scores into the running (max, sum, output).

    `scores` is (H, Tq, Tk), already scaled and masked; `v` is (H, Tk, D).
    """
    blk_m = scores.max(dim=-1).values                      # (H, Tq)
    new_m = torch.maximum(m, blk_m)
    # A row that is entirely masked has max -inf. exp(-inf - -inf) is nan, so
    # give those rows a zero correction and let the block contribute nothing.
    finite = torch.isfinite(new_m)
    corr = torch.where(finite, torch.exp(m - new_m), torch.zeros_like(m))
    p = torch.exp(scores - new_m.unsqueeze(-1))
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    l = l * corr + p.sum(-1)
    o = o * corr.unsqueeze(-1) + p @ v
    return new_m, l, o


# ---------------------------------------------------------------------------
# Chunk assignment
# ---------------------------------------------------------------------------


def contiguous_chunks(T, rank, world, chunks_per_rank=2):
    """The obvious split: rank r owns the next slice of the sequence.

    With 2 chunks per rank, rank r owns chunks 2r and 2r+1. Rank 0 therefore
    holds the very beginning of the sequence, whose queries may attend to
    almost nothing -- so most of its work is masked away and skipped, and it
    sits waiting while rank P-1 grinds through everything.
    """
    n = T // (world * chunks_per_rank)
    ids = [rank * chunks_per_rank + i for i in range(chunks_per_rank)]
    return [torch.arange(c * n, (c + 1) * n) for c in ids], ids


def zigzag_chunks(T, rank, world, chunks_per_rank=2):
    """The standard fix: pair an early chunk with a late one.

    Rank r owns chunk r (early, cheap) and chunk 2P-1-r (late, expensive).
    Counting chunk-pairs that survive the causal mask, every rank now owns
    exactly (r+1) + (2P-r) = 2P+1 of them -- the same number for every r.
    "Zigzag" (also "striped") describes the shape of the assignment when you
    draw it: 0, 1, 2, 3, then back 7, 6, 5, 4.
    """
    n = T // (world * chunks_per_rank)
    ids = [rank, world * chunks_per_rank - 1 - rank]
    return [torch.arange(c * n, (c + 1) * n) for c in ids], ids


# ---------------------------------------------------------------------------
# The ring
# ---------------------------------------------------------------------------


def ring_attention(q_chunks, k_chunks, v_chunks, pos_chunks, rank, world):
    """One rank's share. Lists hold this rank's chunks; returns their outputs.

    Round r processes the K/V that started on rank (rank - r) mod P. Within a
    round the rank loops over (its query chunk x the received key chunk) and
    SKIPS any pair the causal mask kills entirely -- that skip is what makes
    the layout matter.
    """
    H, _, D = q_chunks[0].shape
    scale = 1.0 / math.sqrt(D)
    n_local = len(q_chunks)
    m = [torch.full((H, c.shape[1]), float("-inf")) for c in q_chunks]
    l = [torch.zeros(H, c.shape[1]) for c in q_chunks]
    o = [torch.zeros(H, c.shape[1], D) for c in q_chunks]

    cur_k = torch.stack(k_chunks)          # (n_local, H, n, D)
    cur_v = torch.stack(v_chunks)
    cur_p = torch.stack(pos_chunks)        # (n_local, n)
    nxt, prv = (rank + 1) % world, (rank - 1) % world

    comm_s = comp_s = 0.0
    comm_bytes = pairs_done = pairs_skipped = 0
    for r in range(world):
        t0 = time.perf_counter()
        for i in range(n_local):
            for j in range(cur_k.shape[0]):
                if int(cur_p[j].min()) > int(pos_chunks[i].max()):
                    pairs_skipped += 1      # every score would be masked
                    continue
                pairs_done += 1
                s = (q_chunks[i] @ cur_k[j].transpose(-1, -2)) * scale
                mask = pos_chunks[i].view(-1, 1) < cur_p[j].view(1, -1)
                s = s.masked_fill(mask, float("-inf"))
                m[i], l[i], o[i] = online_update(m[i], l[i], o[i], s, cur_v[j])
                del s
        comp_s += time.perf_counter() - t0

        if r == world - 1:
            break
        t0 = time.perf_counter()
        rk, rv, rp = (torch.empty_like(cur_k), torch.empty_like(cur_v),
                      torch.empty_like(cur_p))
        # isend and irecv posted together, so the ring does not serialise
        # into "everyone sends, then everyone receives".
        reqs = [dist.isend(cur_k.contiguous(), nxt),
                dist.isend(cur_v.contiguous(), nxt),
                dist.isend(cur_p.contiguous(), nxt),
                dist.irecv(rk, prv), dist.irecv(rv, prv), dist.irecv(rp, prv)]
        for w in reqs:
            w.wait()
        comm_bytes += sum(t.numel() * t.element_size()
                          for t in (cur_k, cur_v, cur_p))
        cur_k, cur_v, cur_p = rk, rv, rp
        comm_s += time.perf_counter() - t0

    out = [o[i] / l[i].clamp(min=1e-20).unsqueeze(-1) for i in range(n_local)]
    stats = {"comp_s": comp_s, "comm_s": comm_s, "comm_bytes": comm_bytes,
             "pairs_done": pairs_done, "pairs_skipped": pairs_skipped,
             "peak_score_mb": H * q_chunks[0].shape[1] * cur_k.shape[2] * 4 / 1e6,
             "kv_resident_mb": (cur_k.numel() + cur_v.numel()) * 4 / 1e6 * 2}
    return out, stats


# ---------------------------------------------------------------------------
# Entry point under torchrun
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layout", default="contiguous",
                    choices=["contiguous", "zigzag"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    T, H, D = args.seq, args.heads, args.dim
    torch.manual_seed(0)
    # Every rank builds the same full tensors from the same seed and keeps
    # only its own rows. Wasteful, but it means the reference and the ring
    # see identical numbers, and correctness is the point of the exercise.
    Q, K, V = (torch.randn(H, T, D), torch.randn(H, T, D), torch.randn(H, T, D))

    fn = contiguous_chunks if args.layout == "contiguous" else zigzag_chunks
    pos_chunks, chunk_ids = fn(T, rank, world)
    q_chunks = [Q[:, p] for p in pos_chunks]
    k_chunks = [K[:, p] for p in pos_chunks]
    v_chunks = [V[:, p] for p in pos_chunks]

    dist.barrier()
    t0 = time.perf_counter()
    out, stats = ring_attention(q_chunks, k_chunks, v_chunks, pos_chunks,
                                rank, world)
    local_s = time.perf_counter() - t0
    dist.barrier()
    wall = time.perf_counter() - t0

    payload = {"rank": rank, "world": world, "layout": args.layout,
               "seq": T, "heads": H, "dim": D, "threads": args.threads,
               "local_s": local_s, "wall_s": wall, "chunks": chunk_ids,
               **stats}

    if args.check:
        flat = torch.cat(out, dim=1)
        flat_pos = torch.cat(pos_chunks)
        gathered = [torch.zeros_like(flat) for _ in range(world)] if rank == 0 else None
        pos_all = [torch.zeros_like(flat_pos) for _ in range(world)] if rank == 0 else None
        dist.gather(flat.contiguous(), gathered, dst=0)
        dist.gather(flat_pos.contiguous(), pos_all, dst=0)
        if rank == 0:
            import torch.nn.functional as F
            ref = F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0),
                is_causal=True).squeeze(0)
            got = torch.zeros_like(ref)
            for g, p in zip(gathered, pos_all):
                got[:, p] = g
            payload["max_abs_err"] = float((got - ref).abs().max())
            payload["ref_abs_mean"] = float(ref.abs().mean())
            payload["full_score_mb"] = H * T * T * 4 / 1e6

    with open(f"{args.out}.{rank}.json", "w") as f:
        json.dump(payload, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
