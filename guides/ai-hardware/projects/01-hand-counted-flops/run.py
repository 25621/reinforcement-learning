"""Project 01 - Hand-counted FLOPs for one transformer block.

Everything here is a counting exercise, not a training run. We

  1. derive the FLOP count of one transformer block with pen-and-paper formulas,
  2. check those formulas against PyTorch's own FlopCounterMode on a real block,
  3. sweep the sequence length to find where attention overtakes the projections,
  4. count the *bytes* the same block moves, which is the hand-off to project 02.

Runs on CPU in a few seconds. No GPU needed - a FLOP count does not care what
hardware you own.
"""

import csv
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(4)


# --------------------------------------------------------------------------
# 1. The hand formulas
# --------------------------------------------------------------------------
def hand_count(B, S, D, H, Fh):
    """FLOPs for ONE transformer block, forward pass, counted by hand.

    B  batch, S  sequence length, D  model width, H  heads, Fh  MLP hidden width.

    Convention: one multiply-add = 2 FLOPs. A matmul of (M,K) @ (K,N) costs
    2*M*N*K, because each of the M*N outputs is a dot product of length K,
    i.e. K multiplies and K-1 adds ~ 2K FLOPs.
    """
    d = D // H  # head dim
    mm = {}
    # ---- attention ----
    mm["qkv_proj"] = 3 * 2 * B * S * D * D  # x @ Wq, x @ Wk, x @ Wv
    mm["scores_QK"] = 2 * B * H * S * S * d  # per head: (S,d) @ (d,S)
    mm["attn_AV"] = 2 * B * H * S * S * d  # per head: (S,S) @ (S,d)
    mm["out_proj"] = 2 * B * S * D * D
    # ---- MLP ----
    mm["mlp_up"] = 2 * B * S * D * Fh
    mm["mlp_down"] = 2 * B * S * Fh * D

    # ---- the cheap stuff: elementwise + reductions ----
    el = {}
    # LayerNorm x2: mean (S*D adds), var (2 ops/elt), normalise+affine (4 ops/elt)
    el["layernorm"] = 2 * (8 * B * S * D)
    # softmax over S per (batch, head, row): max, subtract, exp, sum, divide
    el["softmax"] = 5 * B * H * S * S
    # GELU on the MLP hidden activations (tanh approximation ~ 8 ops/elt)
    el["gelu"] = 8 * B * S * Fh
    # two residual additions
    el["residual"] = 2 * B * S * D
    # the 1/sqrt(d) scaling of the scores
    el["scale"] = B * H * S * S

    return mm, el


def hand_bytes(B, S, D, H, Fh, dtype_bytes=2):
    """Bytes that must cross the memory bus if every op is a separate kernel.

    This is the pessimistic (unfused) case: each kernel reads its inputs from
    memory and writes its output back. Weights are read once each.
    """
    b = dtype_bytes
    acts = 0
    acts += B * S * D * b * 2  # ln1 read + write
    acts += B * S * D * b + 3 * B * S * D * b  # qkv: read x, write q,k,v
    acts += 2 * B * H * S * S * b + 2 * B * S * D * b  # scores write + q,k read
    acts += 2 * B * H * S * S * b  # softmax read + write
    acts += B * H * S * S * b + B * S * D * b * 2  # AV: read A, read V, write O
    acts += 2 * B * S * D * b  # out proj
    acts += 2 * B * S * D * b  # residual
    acts += 2 * B * S * D * b  # ln2
    acts += B * S * D * b + B * S * Fh * b  # mlp up
    acts += 2 * B * S * Fh * b  # gelu
    acts += B * S * Fh * b + B * S * D * b  # mlp down
    acts += 2 * B * S * D * b  # residual
    weights = (4 * D * D + 2 * D * Fh) * b + (4 * D) * b  # matrices + LN affine
    return acts, weights


# --------------------------------------------------------------------------
# 2. A real block, so the formulas can be checked
# --------------------------------------------------------------------------
class Block(nn.Module):
    """A standard pre-norm transformer block, written so that attention can be
    computed either the fused way (SDPA) or the explicit way (matmul+softmax)."""

    def __init__(self, D, H, Fh, fused: bool):
        super().__init__()
        self.H, self.D, self.fused = H, D, fused
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.fc1 = nn.Linear(D, Fh, bias=False)
        self.fc2 = nn.Linear(Fh, D, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, S, self.H, D // self.H).transpose(1, 2)
        k = k.view(B, S, self.H, D // self.H).transpose(1, 2)
        v = v.view(B, S, self.H, D // self.H).transpose(1, 2)
        if self.fused:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            a = (q @ k.transpose(-2, -1)) * (D // self.H) ** -0.5
            o = a.softmax(dim=-1) @ v
        o = o.transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(o)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


def counted_flops(model, x, backward=False):
    m = FlopCounterMode(display=False)
    with m:
        y = model(x)
        if backward:
            y.sum().backward()
    return m.get_total_flops()


# --------------------------------------------------------------------------
# 3. Run it
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    rows = []
    findings = {}

    # ---- the reference shape: GPT-2 small, one block, 1024 tokens ----
    B, S, D, H, Fh = 1, 1024, 768, 12, 3072
    mm, el = hand_count(B, S, D, H, Fh)
    mm_total, el_total = sum(mm.values()), sum(el.values())
    print(f"=== One block, B={B} S={S} D={D} H={H} F={Fh} ===")
    for kk, vv in mm.items():
        print(f"  {kk:<12} {vv/1e9:9.3f} GFLOP  ({100*vv/mm_total:5.1f}% of matmul)")
        rows.append(dict(section="breakdown", name=kk, kind="matmul", flops=vv))
    for kk, vv in el.items():
        print(f"  {kk:<12} {vv/1e9:9.3f} GFLOP  (elementwise)")
        rows.append(dict(section="breakdown", name=kk, kind="elementwise", flops=vv))
    print(f"  matmul total      {mm_total/1e9:9.3f} GFLOP")
    print(f"  elementwise total {el_total/1e9:9.3f} GFLOP "
          f"({100*el_total/(mm_total+el_total):.2f}% of all FLOPs)")
    findings["matmul_gflop"] = mm_total / 1e9
    findings["elementwise_gflop"] = el_total / 1e9
    findings["elementwise_pct"] = 100 * el_total / (mm_total + el_total)

    # ---- check against PyTorch's own counter ----
    x = torch.randn(B, S, D)
    explicit = Block(D, H, Fh, fused=False).eval()
    fused = Block(D, H, Fh, fused=True).eval()
    with torch.no_grad():
        c_explicit = counted_flops(explicit, x)
        c_fused = counted_flops(fused, x)
    print(f"\nFlopCounterMode, explicit attention : {c_explicit/1e9:.3f} GFLOP")
    print(f"FlopCounterMode, fused SDPA        : {c_fused/1e9:.3f} GFLOP")
    print(f"hand matmul count                  : {mm_total/1e9:.3f} GFLOP")
    print(f"hand == explicit counter           : {c_explicit == mm_total}")
    findings["counter_explicit_gflop"] = c_explicit / 1e9
    findings["counter_fused_gflop"] = c_fused / 1e9
    findings["hand_matches_counter"] = bool(c_explicit == mm_total)
    findings["sdpa_flops_missing_gflop"] = (c_explicit - c_fused) / 1e9
    rows += [
        dict(section="verify", name="hand_matmul", kind="flops", flops=mm_total),
        dict(section="verify", name="counter_explicit", kind="flops", flops=c_explicit),
        dict(section="verify", name="counter_fused_sdpa", kind="flops", flops=c_fused),
    ]

    # ---- forward vs forward+backward ----
    xg = torch.randn(B, S, D, requires_grad=True)
    train_block = Block(D, H, Fh, fused=False)
    c_fwd = counted_flops(train_block, xg)
    train_block.zero_grad()
    c_fwdbwd = counted_flops(train_block, xg, backward=True)
    print(f"\nforward            : {c_fwd/1e9:.3f} GFLOP")
    print(f"forward + backward : {c_fwdbwd/1e9:.3f} GFLOP  "
          f"(ratio {c_fwdbwd/c_fwd:.2f}x)")
    findings["fwd_bwd_ratio"] = c_fwdbwd / c_fwd
    rows += [
        dict(section="verify", name="fwd", kind="flops", flops=c_fwd),
        dict(section="verify", name="fwd_bwd", kind="flops", flops=c_fwdbwd),
    ]

    # ---- the 2*N*D rule ----
    n_params = sum(p.numel() for p in explicit.parameters())
    tokens = B * S
    print(f"\nparameters in the block : {n_params/1e6:.3f} M")
    print(f"matmul FLOPs per token  : {mm_total/tokens/1e6:.3f} M")
    print(f"ratio (the '2N' rule)   : {mm_total/tokens/n_params:.3f}")
    findings["params_M"] = n_params / 1e6
    findings["flops_per_token_over_params"] = mm_total / tokens / n_params

    dense = mm["qkv_proj"] + mm["out_proj"] + mm["mlp_up"] + mm["mlp_down"]
    print(f"ratio using only the weight matmuls: {dense/tokens/n_params:.4f}")
    findings["dense_flops_per_token_over_params"] = dense / tokens / n_params

    # ---- sweep sequence length: when does attention take over? ----
    seqs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    print("\n  seq   attn-share  total GFLOP  arithmetic intensity (bf16)")
    sweep = []
    for s in seqs:
        m2, e2 = hand_count(B, s, D, H, Fh)
        attn = m2["scores_QK"] + m2["attn_AV"]
        tot = sum(m2.values())
        acts, wts = hand_bytes(B, s, D, H, Fh)
        ai = (tot + sum(e2.values())) / (acts + wts)
        share = 100 * attn / tot
        print(f"{s:7d}  {share:8.1f}%  {tot/1e9:10.2f}   {ai:8.1f}")
        sweep.append((s, share, tot, ai, acts + wts))
        rows.append(dict(section="seq_sweep", name=str(s), kind="attn_share_pct",
                         flops=tot, extra=share))
    findings["attn_share_at_1024"] = [r[1] for r in sweep if r[0] == 1024][0]
    findings["attn_share_at_32768"] = [r[1] for r in sweep if r[0] == 32768][0]

    # crossover: attention == the rest, solved exactly from the formula
    #   attention  = 4*B*S^2*D                    (QK^T and A@V)
    #   the rest   = 8*B*S*D^2 + 4*B*S*D*F        (qkv + out proj + both MLP mats)
    #   equal when 4*S = 8*D + 4*F   ->   S = 2*D + F
    s_star = (8 * D * D + 4 * D * Fh) / (4 * D)
    print(f"\nattention == everything else at S = {s_star:.0f} tokens "
          f"(D={D}, F={Fh})")
    findings["crossover_tokens"] = s_star
    # numeric check
    m3, _ = hand_count(B, int(round(s_star)), D, H, Fh)
    a3 = m3["scores_QK"] + m3["attn_AV"]
    r3 = sum(m3.values()) - a3
    findings["crossover_check_ratio"] = a3 / r3
    print(f"check at that S: attention/rest = {a3/r3:.4f}")

    # ---- FLOPs vs bytes: the hand-off to project 02 ----
    acts, wts = hand_bytes(B, S, D, H, Fh)
    total_flops = mm_total + el_total
    print(f"\nbf16 bytes moved (unfused): {(acts+wts)/1e6:.1f} MB "
          f"(activations {acts/1e6:.1f} + weights {wts/1e6:.1f})")
    print(f"arithmetic intensity      : {total_flops/(acts+wts):.1f} FLOPs/byte")
    findings["bytes_MB"] = (acts + wts) / 1e6
    findings["arithmetic_intensity"] = total_flops / (acts + wts)

    # share of FLOPs vs share of bytes for the elementwise ops
    el_bytes = (2 * (2 * B * S * D) + 2 * (2 * B * H * S * S) + 2 * B * S * Fh
                + 2 * B * S * D) * 2
    print(f"elementwise ops: {100*el_total/total_flops:.2f}% of FLOPs but "
          f"{100*el_bytes/(acts+wts):.1f}% of bytes")
    findings["elementwise_byte_pct"] = 100 * el_bytes / (acts + wts)

    # ---- write outputs ----
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "kind", "flops", "extra"])
        w.writeheader()
        for r in rows:
            r.setdefault("extra", "")
            w.writerow(r)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    plot(mm, el, sweep, s_star, D, Fh)
    print(f"\ndone in {time.time()-t0:.1f}s")


def plot(mm, el, sweep, s_star, D, Fh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) where the FLOPs are at S=1024
    names = list(mm.keys()) + ["elementwise\n(all of it)"]
    vals = [v / 1e9 for v in mm.values()] + [sum(el.values()) / 1e9]
    colors = ["#4C78A8"] * 1 + ["#F58518"] * 2 + ["#4C78A8"] + ["#54A24B"] * 2 + ["#B279A2"]
    ax[0].barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax[0].text(v, i, f" {v:.1f}", va="center", fontsize=9)
    ax[0].set_xlabel("GFLOP (one block, forward, S=1024)")
    ax[0].set_title("(a) Where the FLOPs live\norange = attention, green = MLP")
    ax[0].invert_yaxis()

    # (b) attention share vs sequence length
    s = [r[0] for r in sweep]
    share = [r[1] for r in sweep]
    ax[1].semilogx(s, share, "o-", color="#F58518", base=2)
    ax[1].axhline(50, ls="--", c="gray", lw=1)
    ax[1].axvline(s_star, ls="--", c="crimson", lw=1)
    ax[1].text(s_star * 1.1, 15, f"crossover\nS={s_star:.0f}", color="crimson", fontsize=9)
    ax[1].set_xlabel("sequence length (tokens)")
    ax[1].set_ylabel("% of matmul FLOPs spent in attention")
    ax[1].set_title(f"(b) Attention overtakes the weights\nonly past S={s_star:.0f} "
                    f"(D={D}, F={Fh})")
    ax[1].grid(alpha=.3)

    # (c) total FLOPs vs the quadratic term
    tot = [r[2] / 1e9 for r in sweep]
    lin = [(r[2] * (1 - r[1] / 100)) / 1e9 for r in sweep]
    ax[2].loglog(s, tot, "o-", label="total", color="#4C78A8", base=2)
    ax[2].loglog(s, lin, "s--", label="weight matmuls only (linear in S)",
                 color="#54A24B", base=2)
    ax[2].set_xlabel("sequence length (tokens)")
    ax[2].set_ylabel("GFLOP per block, forward")
    ax[2].set_title("(c) The quadratic term is invisible\nuntil suddenly it is not")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flops_breakdown.png"), dpi=110)
    print(f"wrote {OUT}/flops_breakdown.png")


if __name__ == "__main__":
    main()
