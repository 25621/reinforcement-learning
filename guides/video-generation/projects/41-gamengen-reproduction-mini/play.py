"""Play the model.  There is no game running underneath — only the network.

    python3 play.py                  # 4 denoising steps, playable
    python3 play.py --steps 30       # slow and careful
    python3 play.py --arm noaug      # watch it fall apart
    python3 play.py --real           # the actual game, for comparison

Controls: W A S D (or the arrow keys) to move, R to restart, Q to quit.

The first two frames come from the real game, because the model needs
*something* to look at.  After that nothing real is ever shown to it again:
every frame you see is the model's own previous frame fed back in.  When the
level starts melting, that is the model disagreeing with itself.
"""

import argparse
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import world_lib as W                                          # noqa: E402

import run as R                                                # noqa: E402

KEYS = {"w": 0, "s": 1, "a": 2, "d": 3,
        "A": 0, "B": 1, "D": 2, "C": 3}      # the letters arrow keys send


def paint(frame8, header):
    rgb = (np.clip(np.stack([np.interp(np.clip(frame8, 0, 1), W.LEVELS,
                                       W.PALETTE[:, c]) for c in range(3)],
                            -1), 0, 1) * 255).astype(int)
    out = ["\x1b[H\x1b[2J" + header]
    for r in range(W.GRID):
        line = ""
        for c in range(W.GRID):
            k, g, b = rgb[r, c]
            line += f"\x1b[48;2;{k};{g};{b}m    \x1b[0m"
        out.append(line)
        out.append(line)          # doubled so cells look square in a terminal
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def read_key():
    ch = sys.stdin.read(1)
    if ch == "\x1b":                       # arrow keys arrive as ESC [ A..D
        sys.stdin.read(1)
        ch = sys.stdin.read(1)
    return ch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="aug", choices=R.ARMS)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--real", action="store_true",
                   help="play the real game instead of the model")
    a = p.parse_args()

    env = W.GridGame(seed=int(time.time()) % 10000)
    if not a.real:
        net = R.load_arm(a.arm)
        sig = None if a.arm == "noaug" else torch.full((1,), a.sigma)

    def fresh_context():
        env.reset()
        f0 = W.render(*env.state())
        env.step(np.random.randint(4))
        f1 = W.render(*env.state())
        return torch.from_numpy(np.stack([f0, f1])[None])

    ctx = fresh_context()
    frame = ctx[0, -1].numpy()
    ms = 0.0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        mode = "REAL GAME" if a.real else f"MODEL [{a.arm}]"
        while True:
            paint(frame, f" {mode}   {ms:6.1f} ms/frame   "
                         f"{a.steps} steps   WASD move, R restart, Q quit\n")
            k = read_key()
            if k in ("q", "Q"):
                break
            if k in ("r", "R"):
                ctx = fresh_context()
                frame = ctx[0, -1].numpy()
                continue
            if k not in KEYS:
                continue
            act = KEYS[k]
            t0 = time.time()
            if a.real:
                env.step(act)
                frame = W.render(*env.state())
            else:
                acts = torch.tensor([[act]])
                g = torch.Generator().manual_seed(np.random.randint(1 << 30))
                f = W.sample_frames(net, ctx, acts, steps=a.steps,
                                    generator=g, sigma=sig)
                frame = f[0, -1].numpy()
                ctx = torch.cat([ctx[:, 1:], f[:, -1:]], dim=1)
            ms = (time.time() - t0) * 1000
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


if __name__ == "__main__":
    main()
