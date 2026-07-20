"""A reconstruction-FID proxy for 64x64 Moving-MNIST frames.

What FID measures
-----------------
FID (Frechet Inception Distance) does not compare images pixel by pixel. It
pushes two *sets* of images through a fixed network, keeps an internal feature
vector for each, and then asks how far apart the two clouds of feature vectors
are — modelling each cloud as one multivariate Gaussian and measuring the
Frechet distance between them (named after Maurice Frechet, who defined this
distance between probability distributions in the 1950s; for Gaussians it has
the closed form below). Lower is better; 0 means the two sets are
indistinguishable in that feature space.

'Reconstruction FID' is that, with the second set being your tokenizer's
reconstructions of the first. It answers "does what comes out still look like
the same *kind* of thing?", which PSNR cannot: PSNR punishes a one-pixel shift
severely and barely notices a digit turning into a different digit.

Why not the real Inception network
----------------------------------
Published FID uses InceptionV3 trained on ImageNet. Its features are tuned to
tell dog breeds and vehicles apart; on 64x64 white-on-black digits it spends
its capacity on distinctions that do not exist here. We instead train a small
CNN to classify the digits in *this* dataset and take its penultimate layer as
the feature space — an in-domain feature extractor, which is what FID variants
for specialised domains do too.

The consequence, stated plainly: **the numbers here are only comparable to
each other.** An rFID of 12 in this project means nothing next to an rFID of
12 in a paper. Within the project it ranks the three quantizers reliably,
which is all it is asked to do.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
import vae3d_lib as V                              # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)
FEAT_CK = CK / "features.pt"
FEAT_DIM = 64


class FeatureNet(nn.Module):
    """Small digit classifier; `features()` returns the penultimate layer."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.SiLU(),      # 64 -> 32
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),     # 32 -> 16
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),     # 16 -> 8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(64, FEAT_DIM)
        self.head = nn.Linear(FEAT_DIM, n_classes)

    def features(self, x):
        return self.proj(self.body(x))

    def forward(self, x):
        return self.head(F.silu(self.features(x)))


def _frames_and_labels(source, batch_size):
    """One frame per clip, with the label of a digit that is in it.

    Moving MNIST clips contain two digits; we train on clips generated with
    `n_digits=1` so the label is unambiguous."""
    clips, labels = source.batch(batch_size, return_labels=True)
    x = clips[:, 0] * 2.0 - 1.0                      # (B, 1, H, W) in [-1, 1]
    return x, labels[:, 0]


def train_features(steps=600, batch=64, lr=2e-3):
    torch.manual_seed(0)
    net = FeatureNet()
    train_src = V.make_source(seed=3, n_digits=1, seq_len=1, train=True)
    test_src = V.make_source(seed=4, n_digits=1, seq_len=1, train=False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, y = _frames_and_labels(train_src, batch)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"[features] step {step} loss {loss.item():.3f}", flush=True)

    net.eval()
    with torch.no_grad():
        correct = total = 0
        for _ in range(8):
            x, y = _frames_and_labels(test_src, 128)
            correct += (net(x).argmax(1) == y).sum().item()
            total += len(y)
    acc = correct / total
    torch.save({"state": net.state_dict(), "acc": acc}, FEAT_CK)
    print(f"[features] held-out accuracy {acc:.1%} -> {FEAT_CK}")
    return acc


def load_features():
    if not FEAT_CK.exists():
        raise SystemExit("run `python3 train.py --stage clf` first")
    ck = torch.load(FEAT_CK, map_location="cpu", weights_only=False)
    net = FeatureNet()
    net.load_state_dict(ck["state"])
    net.eval()
    return net


@torch.no_grad()
def _features_of_clips(net, clips):
    """(B, C, T, H, W) -> (B*T, FEAT_DIM). Every frame is one sample."""
    B, C, T, H, W = clips.shape
    flat = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return net.features(flat)


def _sqrtm_psd(mat, iters=24):
    """Matrix square root by Newton-Schulz iteration.

    scipy.linalg.sqrtm is the usual choice, but scipy is not a dependency of
    these projects; for the small symmetric positive-definite matrices here
    this converges fine and keeps the dependency list short."""
    norm = mat.norm()
    y = mat / norm
    eye = torch.eye(mat.shape[0], dtype=mat.dtype)
    z = eye.clone()
    for _ in range(iters):
        t = 0.5 * (3.0 * eye - z @ y)
        y, z = y @ t, t @ z
    return y * norm.sqrt()


@torch.no_grad()
def frechet(real_clips, fake_clips, net=None):
    """Frechet distance between the two feature clouds.

        d^2 = |mu_r - mu_f|^2 + tr( S_r + S_f - 2 (S_r S_f)^(1/2) )

    The first term compares the average feature (are the two sets centred in
    the same place?); the second compares the spread and correlations (do they
    vary in the same ways?). A model that produced one perfect image over and
    over would score well on the first term and terribly on the second.
    """
    net = net or load_features()
    fr = _features_of_clips(net, real_clips).double()
    ff = _features_of_clips(net, fake_clips).double()

    mu_r, mu_f = fr.mean(0), ff.mean(0)
    cr = torch.cov(fr.t())
    cf = torch.cov(ff.t())

    eps = 1e-6 * torch.eye(cr.shape[0], dtype=cr.dtype)
    covmean = _sqrtm_psd((cr + eps) @ (cf + eps))
    diff = (mu_r - mu_f).pow(2).sum()
    return float(diff + torch.trace(cr + cf - 2.0 * covmean))
