"""Project 39 — frames out of the door while the rest is still being made.

    python3 run.py --stage data       # ~2 min  long clips to learn continuation on
    python3 run.py --stage train      # ~7 min  one chunk-causal model
    python3 run.py --stage cache      # ~1 min  correctness + speed of the KV cache
    python3 run.py --stage sweep      # ~4 min  chunk size and step count vs quality
    python3 run.py --stage drift      # ~3 min  its own past vs the real past
    python3 run.py --stage figures    # ~1 min
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from PIL import Image                                          # noqa: E402

import stream_lib as SL                                        # noqa: E402
LL, T, L, FL = SL.LL, SL.T, SL.L, SL.FL

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

STEPS_TRAIN, BATCH, LR = 3000, 16, 6e-4
N_LONG = 512               # long training clips
DROP_PROMPT = 0.1
N_EVAL = 24
SEED = 39
SCHEDULE = [0, 1, 2, 3, 0, 1, 2]          # the same shot list as project 35


# --------------------------------------------------------------------------
# stage: data
# --------------------------------------------------------------------------

@torch.no_grad()
def data():
    """LONG clips, encoded once.

    The obvious training set is project 25's cache of 16-frame clips, and the
    first version of this project used it.  It does not work, and the reason is
    worth keeping: every one of those clips moves in a SINGLE direction, so a
    model trained on (memory, next chunk) pairs cut out of them only ever sees
    a continuation of the motion already under way.  At rollout time we ask it
    to turn a corner at every chunk — a situation it has never been in.  The
    measurement then reports "quality collapses after chunk 1", which sounds
    like [exposure bias](/shared/glossary/#exposure-bias) but is really just
    the model being off its training distribution.

    So the training set is built from 64-frame clips that follow the same shot
    list the rollout will follow.  Turns are now inside the training data.
    """
    t0 = time.time()
    vae, scale = L.load_vae("3d")
    rng = np.random.default_rng(SEED)
    track = LL.latent_direction_track(SCHEDULE)
    lats, digs = [], []
    for _ in range(N_LONG // 8):
        clips, dg, _ = LL.long_real(rng, 8, schedule=SCHEDULE, train=True)
        mean, _ = vae.encode(clips)
        lats.append((mean * scale).clone())
        digs.append(dg)
    out = dict(latents=torch.cat(lats), digit=torch.cat(digs),
               track=torch.from_numpy(track), scale=scale)
    torch.save(out, CK / "long_latents.pt")
    print(f"[data] {tuple(out['latents'].shape)}  {time.time()-t0:.0f}s",
          flush=True)


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train():
    """One model, chunk sizes 1 / 2 / 4 drawn at random each step."""
    torch.manual_seed(0)
    p = CK / "long_latents.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage data`")
    cache = torch.load(p, map_location="cpu", weights_only=False)
    lat, digit, track = cache["latents"], cache["digit"], cache["track"]
    n_lat = lat.shape[2]
    bank = T.TextBank("t5")
    model = SL.StreamDiT()
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    print(f"[train] {L.count_params(model):,} params", flush=True)
    log, t0 = [], time.time()
    for step in range(1, STEPS_TRAIN + 1):
        chunk = SL.CHUNKS[int(torch.randint(0, len(SL.CHUNKS), (1,),
                                            generator=g))]
        # Cut a (memory, next chunk) pair out of a random place in a long
        # clip.  Because the clip turns corners, some of those pairs sit right
        # on a turn — which is exactly the hard case at rollout time.
        idx = torch.randint(0, len(lat), (BATCH,), generator=g)
        start = int(torch.randint(0, n_lat - chunk - SL.MEM + 1, (1,),
                                  generator=g))
        mem = lat[idx][:, :, start:start + SL.MEM]
        x0 = lat[idx][:, :, start + SL.MEM:start + SL.MEM + chunk]
        # the caption describes the chunk being generated, not the memory:
        # the direction of its LAST frame, which is what a shot list would say
        want_dir = int(track[start + SL.MEM + chunk - 1])
        pidx = torch.tensor([T.prompt_index(int(digit[i]), want_dir,
                                            "short", 0) for i in idx])
        text = bank.get(pidx)
        drop = torch.rand(BATCH, generator=g) < DROP_PROMPT
        if drop.any():
            null = bank.null(BATCH)
            for k in text:
                seq, mask = text[k][0].clone(), text[k][1].clone()
                seq[drop] = null[k][0][drop][:, :seq.shape[1]]
                mask[drop] = null[k][1][drop][:, :mask.shape[1]]
                text[k] = (seq, mask)
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        pred = model(flow.interpolate(x0, t, noise), t * flow.T_SCALE, text,
                     mem_lat=mem)
        loss = F.mse_loss(pred, flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[train] {step:5d} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "elapsed": time.time() - t0,
                "params": L.count_params(model)}, CK / "stream.pt")
    np.save(OUT / "log_stream.npy", np.array(log))
    print(f"[train] done {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: cache — is the KV cache correct, and what does it buy?
# --------------------------------------------------------------------------

@torch.no_grad()
def cache():
    model, _ = SL.load_stream()
    bank = T.TextBank("t5")
    n = 8
    digits = torch.arange(n) % 10
    dirs = torch.zeros(n, dtype=torch.long)
    text = LL.text_for(bank, digits, dirs)
    ctx, ctx_n = model.context(text), model.context(bank.null(n))
    mem = torch.randn((n, 4, SL.MEM, 8, 8)) * 0.6
    rows = []
    for chunk in SL.CHUNKS:
        g1 = torch.Generator().manual_seed(4)
        g2 = torch.Generator().manual_seed(4)
        t0 = time.time()
        a = SL.denoise_chunk(model, mem, ctx, ctx_n, chunk, generator=g1,
                             use_cache=True)
        t_cached = time.time() - t0
        t0 = time.time()
        b = SL.denoise_chunk(model, mem, ctx, ctx_n, chunk, generator=g2,
                             use_cache=False)
        t_plain = time.time() - t0
        rows.append(dict(chunk=chunk,
                         max_abs_difference=float((a - b).abs().max()),
                         cached_s=round(t_cached, 3),
                         recomputed_s=round(t_plain, 3),
                         saving=round(1 - t_cached / t_plain, 3)))
        print(f"[cache] {rows[-1]}", flush=True)
    with open(OUT / "kv_cache.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    torch.save({"rows": rows}, CK / "cache.pt")


# --------------------------------------------------------------------------
# stage: sweep — latency against quality
# --------------------------------------------------------------------------

def _prefix(n, digits, seed=0):
    """MEM real latent frames to start the stream from."""
    rng = np.random.default_rng(seed)
    clips, _, _ = LL.long_real(rng, n, schedule=SCHEDULE, digits=digits)
    vae, scale = L.load_vae("3d")
    with torch.no_grad():
        mean, _ = vae.encode(clips)
    return (mean * scale)[:, :, :SL.MEM], clips, (mean * scale)


@torch.no_grad()
def _score(pix, digits, schedule, judge, n_prefix_frames):
    pos, votes = LL.digit_votes(pix, judge)
    _, drift = LL.identity_drift(pix)
    return dict(direction_follow=float(
                    LL.direction_follow(pix, schedule).mean()),
                digit_acc=float((votes == digits[:, None]).float().mean()),
                digit_stable=float((votes == votes[:, :1]).float().mean()),
                identity_drift_end=float(drift[:, -1].mean()),
                path_jerk=float(LL.path_jerk(pix).mean()))


@torch.no_grad()
def sweep():
    t0 = time.time()
    model, _ = SL.load_stream()
    bank = T.TextBank("t5")
    judge, _ = T.load_digit_judge()
    digits = torch.arange(N_EVAL) % 10
    prefix, real_clips, real_lat = _prefix(N_EVAL, digits, seed=SEED)
    n_lat = LL.total_latent(SCHEDULE)
    rows, keep = [], {}
    for chunk in SL.CHUNKS:
        for steps in ([4, 8, 30] if chunk == 2 else [30]):
            # a shot list stretched over however many chunks it takes
            reps = (n_lat - SL.MEM + chunk - 1) // chunk
            sched = [SCHEDULE[min(int(i * chunk / LL.STRIDE),
                                  len(SCHEDULE) - 1)] for i in range(reps)]
            t1 = time.time()
            lat, emit = SL.rollout(model, bank, digits, sched, prefix,
                                   chunk=chunk, steps=steps, seed=SEED)
            wall = time.time() - t1
            pix = LL.decode_long(lat[:, :, :n_lat])
            row = dict(chunk=chunk, steps=steps, chunks_run=reps,
                       first_chunk_s=round(emit[0], 2),
                       total_s=round(wall, 2),
                       s_per_frame=round(wall / (n_lat * LL.PIX_PER_LAT), 3))
            row.update(_score(pix, digits, SCHEDULE, judge, SL.MEM))
            rows.append(row)
            print(f"[sweep] {row}", flush=True)
            keep[f"c{chunk}_s{steps}"] = pix[:4]
    # the whole-clip alternative: project 35's sliding window, same length
    base, bbank, _ = LL.load_base()
    t1 = time.time()
    _, slid = LL.generate_long(base, bbank, digits, "anchored",
                               schedule=SCHEDULE, seed=SEED)
    wall = time.time() - t1
    row = dict(chunk="sliding_window", steps=30, chunks_run=len(SCHEDULE),
               first_chunk_s=round(wall, 2), total_s=round(wall, 2),
               s_per_frame=round(wall / (n_lat * LL.PIX_PER_LAT), 3))
    row.update(_score(slid, digits, SCHEDULE, judge, 0))
    rows.append(row)
    keep["sliding_window"] = slid[:4]
    print(f"[sweep] {row}", flush=True)
    rt = LL.decode_long(real_lat)
    row = dict(chunk="real_vae", steps=0, chunks_run=0, first_chunk_s=0.0,
               total_s=0.0, s_per_frame=0.0)
    row.update(_score(rt, digits, SCHEDULE, judge, 0))
    rows.append(row)
    keep["real_vae"] = rt[:4]
    torch.save({"rows": rows, "keep": keep}, CK / "sweep.pt")
    with open(OUT / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[sweep] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: drift — the cost of reading your own handwriting
# --------------------------------------------------------------------------

@torch.no_grad()
def drift():
    """Roll out twice: once on its own output, once on the real past.

    The difference between the two curves is **exposure bias**.  The name is
    literal: during training the model is only ever *exposed* to real past
    frames, so it never learns what to do when the past is slightly wrong —
    and at generation time the past is always slightly wrong, because the
    model made it.  Small mistakes then feed on themselves.

    Feeding the real past during generation is called **teacher forcing**: the
    "teacher" (the ground-truth data) supplies the history instead of letting
    the student rely on itself.  It is the normal way to train an
    autoregressive model, and Self-Forcing is the fix that stops training that
    way.
    """
    t0 = time.time()
    model, _ = SL.load_stream()
    bank = T.TextBank("t5")
    judge, _ = T.load_digit_judge()
    digits = torch.arange(N_EVAL) % 10
    prefix, real_clips, real_lat = _prefix(N_EVAL, digits, seed=SEED)
    chunk = 2
    n_lat = LL.total_latent(SCHEDULE)
    reps = (n_lat - SL.MEM) // chunk
    sched = [SCHEDULE[min(int(i * chunk / LL.STRIDE), len(SCHEDULE) - 1)]
             for i in range(reps)]
    curves, rows = {}, []
    for mode in ["self", "teacher"]:
        lat, _ = SL.rollout(model, bank, digits, sched, prefix, chunk=chunk,
                            seed=SEED,
                            teacher=(real_lat if mode == "teacher" else None))
        pix = LL.decode_long(lat[:, :, :n_lat])
        # quality as a function of how far into the rollout we are
        per_chunk = []
        for i in range(reps):
            s = (SL.MEM + i * chunk) * LL.PIX_PER_LAT
            e = s + chunk * LL.PIX_PER_LAT
            seg = pix[:, :, s:e]
            probs = F.softmax(judge(seg[:, :, 0]), 1)
            ink = ((seg + 1) / 2).mean(dim=(1, 2, 3, 4))
            per_chunk.append((float((probs.argmax(1) == digits).float().mean()),
                              float(ink.mean()),
                              float(probs.max(1).values.mean())))
        curves[mode] = np.array(per_chunk)
        row = dict(mode=mode, **_score(pix, digits, SCHEDULE, judge, SL.MEM))
        rows.append(row)
        print(f"[drift] {row}", flush=True)
        curves[mode + "_clips"] = pix[:4]
    ink_real = float(((LL.decode_long(real_lat) + 1) / 2).mean())
    torch.save({"curves": {k: v for k, v in curves.items()},
                "rows": rows, "ink_real": ink_real, "reps": reps,
                "chunk": chunk}, CK / "drift.pt")
    with open(OUT / "drift.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[drift] {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def save_gif(clip, path, scale=2, ms=90):
    x = ((clip.clamp(-1, 1) + 1) / 2)[0, 0].numpy()
    fr = [Image.fromarray((f * 255).astype(np.uint8)).resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in x]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=ms, loop=0)


def figures():
    sw = torch.load(CK / "sweep.pt", weights_only=False)
    dr = torch.load(CK / "drift.pt", weights_only=False)
    kv = torch.load(CK / "cache.pt", weights_only=False)
    rows = sw["rows"]

    # ---- 1. latency vs quality -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ps.style_axes(axes[0])
    sel = [r for r in rows if isinstance(r["chunk"], int)]
    for i, r in enumerate(sel):
        axes[0].scatter(r["first_chunk_s"], r["direction_follow"], s=60,
                        color=ps.SERIES[i % len(ps.SERIES)],
                        label=f"chunk {r['chunk']}, {r['steps']} steps")
    full = next(r for r in rows if r["chunk"] == "sliding_window")
    axes[0].scatter(full["first_chunk_s"], full["direction_follow"], s=70,
                    marker="s", color=ps.INK_MUTED, label="whole clip first")
    axes[0].set_xlabel("seconds before the FIRST frame can be shown")
    axes[0].set_ylabel("motion follows the shot list")
    axes[0].legend(frameon=False, fontsize=8)
    ps.style_axes(axes[1])
    kvr = kv["rows"]
    xs = np.arange(len(kvr))
    axes[1].bar(xs - 0.19, [r["recomputed_s"] for r in kvr], 0.36,
                color=ps.SERIES[2], label="memory recomputed every step")
    axes[1].bar(xs + 0.19, [r["cached_s"] for r in kvr], 0.36,
                color=ps.SERIES[1], label="memory keys/values cached")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"chunk {r['chunk']}" for r in kvr], fontsize=9)
    axes[1].set_ylabel("seconds per chunk")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Streaming buys latency; the KV cache buys it back cheaper",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=150)
    plt.close(fig)

    # ---- 2. exposure bias -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for ax, col, lab in zip(axes, [0, 1],
                            ["judge says the right digit",
                             "how much ink is left on screen"]):
        ps.style_axes(ax)
        for i, mode in enumerate(["teacher", "self"]):
            c = dr["curves"][mode]
            ax.plot(np.arange(1, len(c) + 1), c[:, col], "o-", lw=1.8, ms=4,
                    color=ps.SERIES[i],
                    label="teacher-forced (real past)" if mode == "teacher"
                    else "self-rollout (its own past)")
        if col == 1:
            ax.axhline(dr["ink_real"], color=ps.INK_MUTED, ls="--", lw=1.2)
            ax.text(1, dr["ink_real"] + 0.002, "real clips", fontsize=8,
                    color=ps.INK_MUTED)
        ax.set_xlabel("chunk number in the rollout")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Exposure bias: the gap that opens when a model reads its own past",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "exposure_bias.png", dpi=150)
    plt.close(fig)

    # ---- 3. filmstrips ----------------------------------------------------
    order = ["real_vae", "sliding_window", "c1_s30", "c2_s30", "c4_s30",
             "c2_s4"]
    fig, axes = plt.subplots(len(order), 1, figsize=(10.4, 1.3 * len(order)))
    for ax, k in zip(axes, order):
        ax.imshow(LL.contact_sheet(sw["keep"][k][0:1], every=4), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_ylabel(k, rotation=0, ha="right", va="center", fontsize=9,
                      color=ps.INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("The same 64-frame shot list, streamed at different settings",
                 fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "filmstrips.png", dpi=150)
    plt.close(fig)
    for k in ("c2_s30", "sliding_window", "real_vae"):
        save_gif(sw["keep"][k][0:1], OUT / f"stream_{k}.gif")
    print("[figures] wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "train", "cache", "sweep", "drift",
                             "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    globals()[args.stage]()
