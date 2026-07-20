"""Structurally inflate the *real* Stable Diffusion 1.5 U-Net.

No training happens here — fine-tuning an 860M-parameter model is a GPU
job.  What CAN be shown on a CPU, on the genuine article, is the
inflation move itself:

  1. load a real SD 1.5-architecture U-Net (emilianJR/epiCRealism, a
     checkpoint already cached by Phase-3 project 11),
  2. bolt a zero-initialized temporal block onto every down/mid/up block
     via forward hooks,
  3. verify the inflated network is bit-for-bit identical to the
     original run frame-by-frame (identity at initialization),
  4. count what inflation adds: parameters and forward-pass time.

Placement caveat (also in the README): production inflations
(AnimateDiff, SVD) insert temporal layers *inside* each block, after
every spatial sublayer.  Hooking the block boundary is the same
principle at coarser granularity — good enough to demonstrate the
mechanics, not a training-grade placement.

Writes outputs/sd_inflation.txt.
"""

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-tiny-i2v-model"))
from i2v_lib import TemporalBlock  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

MODEL = "emilianJR/epiCRealism"    # SD 1.5 architecture, cached locally
B, T = 1, 8                        # one 8-frame clip
LATENT = 32                        # 32x32 latents = 256x256 pixels


def main():
    from diffusers import UNet2DConditionModel
    print(f"loading {MODEL} U-Net (SD 1.5 architecture)...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL, subfolder="unet", local_files_only=True,
        torch_dtype=torch.float32)
    unet.eval()

    blocks = (list(unet.down_blocks) + [unet.mid_block]
              + list(unet.up_blocks))
    state = {"on": True}
    temporals = torch.nn.ModuleList()
    for blk in blocks:
        c = blk.resnets[-1].out_channels
        tb = TemporalBlock(c)
        tb.eval()
        temporals.append(tb)

        def make_hook(tb):
            def hook(mod, args, out):
                if not state["on"]:
                    return out
                if isinstance(out, tuple):
                    return (tb(out[0], B, T),) + tuple(out[1:])
                return tb(out, B, T)
            return hook

        blk.register_forward_hook(make_hook(tb))

    n_spatial = sum(p.numel() for p in unet.parameters())
    n_temporal = sum(p.numel() for p in temporals.parameters())

    g = torch.Generator().manual_seed(0)
    x = torch.randn(B * T, 4, LATENT, LATENT, generator=g)
    t = torch.tensor(500)
    # dummy text embedding — a structural check needs no real prompt
    text = torch.randn(B * T, 77, 768, generator=g) * 0.1

    with torch.no_grad():
        state["on"] = False
        t0 = time.time()
        base = unet(x, t, encoder_hidden_states=text).sample
        base_s = time.time() - t0
        state["on"] = True
        t0 = time.time()
        inflated = unet(x, t, encoder_hidden_states=text).sample
        infl_s = time.time() - t0

    diff = (base - inflated).abs().max().item()
    lines = [
        f"model: {MODEL} (SD 1.5 U-Net architecture)",
        f"spatial (pretrained, would stay frozen) params: {n_spatial:,}",
        f"temporal (new, zero-initialized) params:        {n_temporal:,}"
        f"  ({100 * n_temporal / n_spatial:.1f}% of spatial)",
        f"temporal blocks inserted: {len(temporals)}"
        f"  (channels: {[tb.conv1.in_channels for tb in temporals]})",
        f"identity at init, max |base - inflated|: {diff:.2e}",
        f"forward pass, {T} frames at {LATENT}x{LATENT} latents "
        f"(= {8 * LATENT}x{8 * LATENT} px):",
        f"  original per-frame U-Net: {base_s:.1f} s",
        f"  inflated video U-Net:     {infl_s:.1f} s",
    ]
    text_out = "\n".join(lines)
    print(text_out)
    (OUT / "sd_inflation.txt").write_text(text_out + "\n")
    assert diff == 0.0, "inflation must start as an exact identity"


if __name__ == "__main__":
    torch.set_num_threads(12)
    main()
