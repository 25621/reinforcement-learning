"""ConvLSTM future-frame predictor for Moving MNIST.

Architecture (encoder -> ConvLSTM -> decoder):
  frame (1, 32, 32) --conv stride 2--> features (32, 16, 16)
  features          --ConvLSTM------> hidden state (64, 16, 16)
  hidden state      --deconv--------> next frame (1, 32, 32)

The ConvLSTM runs over the 10 context frames to build up its state, then
rolls forward 10 more steps *feeding its own predicted frame back in as
input* — the same closed-loop rollout used at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """An LSTM cell whose state is a (C, H, W) feature map, not a vector.

    Identical to a normal LSTM except that every matrix multiply is
    replaced by a convolution, so the gates are computed locally at each
    spatial position from its neighborhood.
    """

    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        # one conv produces all four gates at once (4 * hid_ch channels)
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel,
                               padding=kernel // 2)
        self.hid_ch = hid_ch

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, (h, c)

    def zero_state(self, batch, size, device):
        z = torch.zeros(batch, self.hid_ch, size, size, device=device)
        return (z, z.clone())


class Predictor(nn.Module):
    def __init__(self, enc_ch=32, hid_ch=64):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, enc_ch, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(enc_ch, enc_ch, 3, padding=1), nn.ReLU(),
        )
        self.cell = ConvLSTMCell(enc_ch, hid_ch)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(hid_ch, enc_ch, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_ch, 1, 3, padding=1),
        )

    def forward(self, context, n_future):
        """Closed-loop rollout: context (B, T_in, 1, 32, 32) ->
        predicted-frame LOGITS (B, n_future, 1, 32, 32); apply sigmoid
        to view them as images. Used at test time and for closed-loop
        fine-tuning."""
        B, T_in = context.shape[:2]
        state = self.cell.zero_state(B, context.shape[-1] // 2, context.device)
        # 1) watch the context frames to build up motion state
        for t in range(T_in):
            h, state = self.cell(self.encode(context[:, t]), state)
        # 2) roll out: predict a frame, feed it back in, repeat
        logits = []
        for _ in range(n_future):
            lg = self.decode(h)
            logits.append(lg)
            h, state = self.cell(self.encode(torch.sigmoid(lg)), state)
        return torch.stack(logits, dim=1)

    def teacher_forced(self, clips):
        """Predict frame t+1 from the REAL frames 0..t, for every t.

        Training this way ("teacher forcing") gives the model a learning
        signal at every step of the sequence and never asks it to build on
        its own possibly-garbage output — which is exactly what makes
        training from scratch stable. The price: it never practices the
        closed-loop rollout it will be asked to do at test time.
        Returns predicted-frame logits for frames 1..T-1.
        """
        B, T = clips.shape[:2]
        state = self.cell.zero_state(B, clips.shape[-1] // 2, clips.device)
        logits = []
        for t in range(T - 1):
            h, state = self.cell(self.encode(clips[:, t]), state)
            logits.append(self.decode(h))
        return torch.stack(logits, dim=1)
