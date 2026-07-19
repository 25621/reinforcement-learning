"""MoCoGAN's content/motion decomposition, implemented inside a small VAE.

One clip gets TWO kinds of latent code:
  z_content — ONE vector for the whole clip (dim 16). Computed by
              averaging the frame features over time, so by construction
              it cannot change within the clip.
  z_motion  — one TINY vector PER FRAME (dim 2). It is the only thing
              that differs from frame to frame.

The decoder rebuilds frame t from (z_content, z_motion[t]). Because
z_content is frozen across the clip and z_motion is only 2 numbers, the
model has no choice: everything that stays constant (which digit, how it
is drawn) must live in z_content, and everything that changes (where the
digit is) must squeeze into z_motion. The split is enforced by
architecture, not by a clever loss.
"""

import torch
import torch.nn as nn


class CoMoVAE(nn.Module):
    def __init__(self, c_dim=16, m_dim=2, ch=32):
        super().__init__()
        self.c_dim, self.m_dim = c_dim, m_dim
        # shared per-frame feature extractor: (1, 32, 32) -> 256-d
        self.features = nn.Sequential(
            nn.Conv2d(1, ch, 4, stride=2, padding=1), nn.ReLU(),    # 16
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1), nn.ReLU(),  # 4
            nn.Flatten(), nn.Linear(ch * 2 * 16, 256), nn.ReLU(),
        )
        self.content_head = nn.Linear(256, 2 * c_dim)   # mu and logvar
        self.motion_head = nn.Linear(256, 2 * m_dim)
        self.decode_fc = nn.Linear(c_dim + m_dim, ch * 2 * 16)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1),
            nn.ReLU(),                                             # 8
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.ReLU(),                                             # 16
            nn.ConvTranspose2d(ch, 1, 4, stride=2, padding=1),     # 32
        )
        self.ch = ch

    def encode(self, clips, generator=None):
        """clips: (B, T, 1, 32, 32) -> content and motion posteriors."""
        B, T = clips.shape[:2]
        feats = self.features(clips.flatten(0, 1)).view(B, T, -1)
        # content path: shift every frame by a random offset BEFORE
        # encoding, so the digit's position is unrecoverable from the
        # content input (identity survives a shift; position does not).
        # Averaging over time alone is NOT enough — the average of the
        # frame features still remembers the clip's MEAN position.
        shifts = torch.randint(-16, 16, (B * T, 2), generator=generator)
        rolled = torch.stack([
            torch.roll(f, tuple(s.tolist()), dims=(-2, -1))
            for f, s in zip(clips.flatten(0, 1), shifts)
        ])
        c_feats = self.features(rolled).view(B, T, -1)
        mu_c, logvar_c = self.content_head(c_feats.mean(dim=1)).chunk(2, -1)
        # motion path: one tiny code per (unshifted) frame
        mu_m, logvar_m = self.motion_head(feats).chunk(2, -1)
        return mu_c, logvar_c, mu_m, logvar_m

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_frames(self, z_c, z_m):
        """z_c: (B, c_dim), z_m: (B, T, m_dim) -> logits (B, T, 1, 32, 32)."""
        B, T = z_m.shape[:2]
        z = torch.cat([z_c[:, None].expand(B, T, self.c_dim), z_m], dim=-1)
        h = self.decode_fc(z.flatten(0, 1)).view(-1, self.ch * 2, 4, 4)
        return self.decode(torch.relu(h)).view(B, T, 1, 32, 32)

    def forward(self, clips):
        mu_c, logvar_c, mu_m, logvar_m = self.encode(clips)
        z_c = self.sample(mu_c, logvar_c)
        z_m = self.sample(mu_m, logvar_m)
        logits = self.decode_frames(z_c, z_m)
        kl_c = -0.5 * (1 + logvar_c - mu_c ** 2 - logvar_c.exp()).sum(-1)
        kl_m = -0.5 * (1 + logvar_m - mu_m ** 2 - logvar_m.exp()).sum((-1, -2))
        return logits, kl_c.mean(), kl_m.mean()
