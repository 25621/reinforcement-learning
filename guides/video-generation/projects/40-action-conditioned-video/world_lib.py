"""The Phase-9 backbone: a tiny playable game, and models that learn to *be* it.

Projects 41, 42, 43 and 44 all import this file, the same way Phase 6 reused
project 25's `dit_lib`, Phase 7 reused project 30's `text_lib`, and Phase 8
reused project 35's `long_lib`.

Why this phase starts from a new toy instead of the moving-digit clips
---------------------------------------------------------------------
Every earlier phase generated video from a *text prompt*: you say what you want
once, and the model paints five seconds of it.  A world model is a different
animal.  It has to answer a new question at **every single frame**:

    given what the screen looks like now, and given the button the player just
    pressed, what does the screen look like next?

That needs data where one action is attached to one frame.  The moving-MNIST
clips of Phase 1-8 have a single direction fixed for the whole clip, so they
cannot teach a model to *react*.  So this phase records its own data by playing
a tiny game.

The game (deliberately small, deliberately not trivial)
------------------------------------------------------
An 8x8 grid.  The outer ring is wall.  Six more wall cells are scattered inside,
different every episode.  You are the bright square; the dim ring is the coin.
Four buttons: up, down, left, right.

    +--------+     .  = empty          Two rules make this a real world model
    |####### |     #  = wall           problem rather than a bookkeeping
    |#..o..#.|     o  = coin           exercise:
    |#..#..#.|     @  = you
    |#.@...#.|                         1. Walking into a wall does nothing.
    |#...#.#.|                         2. Stepping on the coin scores, and the
    |#######.|                            coin REAPPEARS somewhere random.
    +--------+

Rule 1 means the model cannot just translate the bright square in the direction
of the button; it has to *look* at what is in the way.  Rule 2 means the future
is genuinely uncertain: from the same screen and the same button, several
different next screens are all correct.  That is the whole reason this phase
uses a generative model instead of a plain "predict the next frame" regressor,
and project 40 measures the difference directly.

The cheap exact tokenizer
-------------------------
Phase 5's lesson was: do not run diffusion on raw pixels, run it in a compressed
space that a tokenizer gives you.  This game hands us a perfect tokenizer for
free.  Every 4x4 pixel block is a flat colour, so a 32x32 screen carries exactly
8x8 = 64 numbers of information.  We therefore let all the models work on the
8x8 grid and expand to 32x32 only for display.

This is the same move as encoding with a VAE, with two differences worth being
honest about: our "encoder" is exact (no reconstruction loss at all) and it is
handed to us by the renderer rather than learned.  A real game frame has no such
gift, which is why real systems (GameNGen, Genie, OASIS) spend a large part of
their compute on a learned tokenizer.  We skip that part because Phase 5 already
built one, and because it makes every experiment in this phase 16x cheaper.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
import flow_lib as FL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------

GRID = 8                 # cells per side
CELL = 4                 # display pixels per cell
IMG = GRID * CELL        # 32x32 display frame
N_WALL = 6               # interior walls, resampled every episode
N_ACT = 4                # up, down, left, right
ACT_NAMES = ["up", "down", "left", "right"]
DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # (drow, dcol) per action

# The four grey levels a cell can take.  Kept far apart so that reading a cell
# back out of a generated frame is unambiguous.
EMPTY, WALL, COIN, AGENT = 0.0, 0.30, 0.62, 1.0
LEVELS = np.array([EMPTY, WALL, COIN, AGENT], dtype=np.float32)

EP_LEN = 33              # frames per recorded episode (32 transitions)


def _connected(walls):
    """Flood-fill: are all free cells reachable from one another?

    A random wall scatter can wall a corner off.  An unreachable pocket is not
    wrong, but it wastes coins (they can spawn where the player can never go),
    so we resample instead.
    """
    free = np.argwhere(walls == 0)
    if len(free) == 0:
        return False
    seen = np.zeros_like(walls, dtype=bool)
    stack = [tuple(free[0])]
    seen[stack[0]] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in DELTA:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID and 0 <= nc < GRID and not walls[nr, nc] \
                    and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen.sum() == len(free)


def sample_layout(rng):
    """A border ring plus N_WALL interior blocks, guaranteed connected."""
    while True:
        walls = np.zeros((GRID, GRID), dtype=np.uint8)
        walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = 1
        inner = [(r, c) for r in range(1, GRID - 1) for c in range(1, GRID - 1)]
        pick = rng.choice(len(inner), size=N_WALL, replace=False)
        for i in pick:
            walls[inner[i]] = 1
        if _connected(walls):
            return walls


class GridGame:
    """The environment.  Deterministic movement, random coin respawn."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _free_cells(self, exclude=()):
        cells = np.argwhere(self.walls == 0)
        return [tuple(c) for c in cells if tuple(c) not in exclude]

    def reset(self, walls=None):
        self.walls = sample_layout(self.rng) if walls is None else walls.copy()
        free = self._free_cells()
        i, j = self.rng.choice(len(free), size=2, replace=False)
        self.agent, self.coin = free[i], free[j]
        return self.state()

    def state(self):
        return (self.walls, self.agent, self.coin)

    def step(self, a):
        """Move if the target cell is free; score and respawn on a coin."""
        dr, dc = DELTA[a]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        blocked = self.walls[nr, nc] == 1
        if not blocked:
            self.agent = (nr, nc)
        reward = 0.0
        if self.agent == self.coin:
            reward = 1.0
            free = self._free_cells(exclude=(self.agent,))
            self.coin = free[self.rng.choice(len(free))]
        return self.state(), reward, blocked


def step_symbolic(walls, agent, coin, a, rng):
    """Same transition as GridGame.step but on loose values (used in dreams)."""
    dr, dc = DELTA[a]
    nr, nc = agent[0] + dr, agent[1] + dc
    if walls[nr, nc] == 0:
        agent = (nr, nc)
    reward = 0.0
    if agent == coin:
        reward = 1.0
        free = [tuple(c) for c in np.argwhere(walls == 0) if tuple(c) != agent]
        coin = free[rng.choice(len(free))]
    return agent, coin, reward


# ---------------------------------------------------------------------------
# rendering and reading back
# ---------------------------------------------------------------------------

def render(walls, agent, coin):
    """One 8x8 latent frame.  This is what every model in Phase 9 sees."""
    f = walls.astype(np.float32) * WALL
    f[coin] = COIN
    f[agent] = AGENT
    return f


def render_batch(walls, agents, coins):
    """Vectorised render.  walls (B,8,8) uint8, agents/coins (B,) flat index."""
    b = walls.shape[0]
    f = walls.reshape(b, -1).astype(np.float32) * WALL
    rows = np.arange(b)
    f[rows, coins] = COIN
    f[rows, agents] = AGENT
    return f.reshape(b, GRID, GRID)


def to_pixels(frame8):
    """8x8 latent -> 32x32 display frame (nearest-neighbour, exact inverse)."""
    return np.repeat(np.repeat(frame8, CELL, axis=-2), CELL, axis=-1)


def read_frame(frame8):
    """Read a (possibly generated, possibly messy) 8x8 frame back into state.

    Returns (symbols, agent_idx, coin_idx, snap_error).

    `snap_error` is the mean distance from each cell to the nearest legal grey
    level.  It is the cheapest possible "is this even a valid game screen?"
    check: a perfect frame scores 0, a smeared or hallucinated one scores high.
    """
    d = np.abs(frame8.reshape(-1, 1) - LEVELS.reshape(1, -1))
    sym = d.argmin(axis=1)
    snap = d.min(axis=1).mean()
    flat = frame8.reshape(-1)
    agent = int(flat.argmax())                       # agent is the brightest
    coin_score = -np.abs(flat - COIN)
    coin_score[agent] = -9.9
    coin = int(coin_score.argmax())
    return sym.reshape(GRID, GRID), agent, coin, float(snap)


def walls_from_symbols(sym):
    return (sym == 1).astype(np.uint8)


def channelize(frames):
    """(B, 8, 8) grey -> (B, 3, 8, 8) 'how wall-like / coin-like / player-like'.

    A fixed, hand-written feature map — no learned parameters, applied exactly
    the same way to real frames and to frames a model dreamed up.  Each channel
    is 1 where the cell sits on that grey level and falls linearly to 0 within
    0.2 of it, so a cell the model left half-way between two levels shows up as
    half-belonging to each, rather than being silently rounded off.

    Projects 43 and 44 feed this to their policy and world model instead of the
    raw grey number.  Without it the network has to spend its first layer
    rediscovering that 0.30 means wall and 0.62 means coin, which is wasted
    capacity in a model this small.
    """
    x = frames.unsqueeze(1)
    levels = torch.tensor([WALL, COIN, AGENT]).view(1, 3, 1, 1)
    return (1.0 - (x - levels).abs() / 0.2).clamp(0.0, 1.0)


def frame_to_coords(frames):
    """(B, 8, 8) grey -> (B, 4) the (row, col) of the player and of the coin,
    scaled into [0, 1].  Vectorised, differentiable-free, and applied the same
    way to a real frame or a frame the world model dreamed up.

    Why a policy reads coordinates instead of pixels (project 43 uses this):
    navigating to the coin is a question about the *relative position* of two
    cells, and a small convolutional net turns out to learn that surprisingly
    badly — its receptive field cannot see both the player and a far coin at
    once, so it never reliably answers "which way is the coin?".  Handing the
    policy the four numbers directly removes an accidental difficulty that has
    nothing to do with the model-based-RL idea the project is about.  In the
    dream these numbers are read from the world model's own output, so when the
    model drifts the coordinates go wrong too — which is exactly the failure the
    project measures.
    """
    b = frames.shape[0]
    flat = frames.reshape(b, -1)
    agent = flat.argmax(dim=1)
    coin_score = -(flat - COIN).abs()
    coin_score = coin_score.scatter(1, agent[:, None], -9.9)
    coin = coin_score.argmax(dim=1)
    out = torch.stack([agent // GRID, agent % GRID,
                       coin // GRID, coin % GRID], dim=1).float()
    return out / GRID


# ---------------------------------------------------------------------------
# recording play
# ---------------------------------------------------------------------------

def _greedy_action(walls, agent, coin, rng, eps):
    """A coin-seeking player with eps of pure randomness mixed in.

    Why not a purely random player?  A random walk almost never touches the
    coin, so the recording would contain nearly no respawn events — and the
    respawn is the only genuinely uncertain thing in this world.  Why not a
    purely greedy player?  It would never bump into walls, so the model would
    never learn rule 1.  The mixture covers both.
    """
    if rng.random() < eps:
        return int(rng.integers(N_ACT))
    dr = coin[0] - agent[0]
    dc = coin[1] - agent[1]
    prefs = []
    if abs(dr) >= abs(dc):
        if dr: prefs.append(0 if dr < 0 else 1)
        if dc: prefs.append(2 if dc < 0 else 3)
    else:
        if dc: prefs.append(2 if dc < 0 else 3)
        if dr: prefs.append(0 if dr < 0 else 1)
    for a in prefs:
        d = DELTA[a]
        if walls[agent[0] + d[0], agent[1] + d[1]] == 0:
            return a
    return int(rng.integers(N_ACT))


def record(n_episodes, seed=0, eps=0.35, ep_len=EP_LEN):
    """Play the game n_episodes times and store the episodes compactly.

    We keep symbols (a few bytes per frame), not pixels.  Frames are rendered
    on the fly during training, which costs nothing and keeps 2500 episodes at
    a few hundred kilobytes instead of a hundred megabytes.
    """
    rng = np.random.default_rng(seed)
    env = GridGame(seed=seed + 1)
    W = np.zeros((n_episodes, GRID, GRID), dtype=np.uint8)
    A = np.zeros((n_episodes, ep_len), dtype=np.int64)     # agent, flat index
    C = np.zeros((n_episodes, ep_len), dtype=np.int64)     # coin,  flat index
    ACT = np.zeros((n_episodes, ep_len - 1), dtype=np.int64)
    REW = np.zeros((n_episodes, ep_len - 1), dtype=np.float32)
    BLK = np.zeros((n_episodes, ep_len - 1), dtype=np.uint8)
    for e in range(n_episodes):
        env.reset()
        W[e] = env.walls
        A[e, 0] = env.agent[0] * GRID + env.agent[1]
        C[e, 0] = env.coin[0] * GRID + env.coin[1]
        for t in range(ep_len - 1):
            a = _greedy_action(env.walls, env.agent, env.coin, rng, eps)
            _, r, blocked = env.step(a)
            ACT[e, t], REW[e, t], BLK[e, t] = a, r, blocked
            A[e, t + 1] = env.agent[0] * GRID + env.agent[1]
            C[e, t + 1] = env.coin[0] * GRID + env.coin[1]
    return dict(walls=W, agent=A, coin=C, act=ACT, rew=REW, blocked=BLK)


def frames_of(ep, e_idx, t_idx):
    """Render frames for index arrays of any shape -> (*shape, 8, 8)."""
    e_idx, t_idx = np.asarray(e_idx), np.asarray(t_idx)
    shape = e_idx.shape
    flat = render_batch(ep["walls"][e_idx.reshape(-1)],
                        ep["agent"][e_idx.reshape(-1), t_idx.reshape(-1)],
                        ep["coin"][e_idx.reshape(-1), t_idx.reshape(-1)])
    return flat.reshape(*shape, GRID, GRID)


def save_episodes(ep, name, where=CK):
    np.savez_compressed(Path(where) / f"{name}.npz", **ep)


def load_episodes(name, where=CK):
    z = np.load(Path(where) / f"{name}.npz")
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32)
                      / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class FiLMBlock(nn.Module):
    """Conv block whose normalisation is steered by the conditioning vector.

    "FiLM" = Feature-wise Linear Modulation: the conditioning does not get
    concatenated as extra pixels, it produces a per-channel scale and shift.
    That is the same mechanism as [AdaLN] in the DiT of project 25, written for
    convolutions instead of tokens, and it is why a single vector (timestep +
    which buttons were pressed) can steer the whole image.
    """

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.n2(h) * (1 + scale) + shift)
        return self.skip(x) + self.c2(h)


class ActionUNet(nn.Module):
    """Predicts K future frames from C past frames and K button presses.

    Frames live in the channel axis: the network is an ordinary 2D U-Net on an
    8x8 image with (C + K) input channels.  At this size that is far cheaper
    than 3D convolutions or attention over a token grid, and it loses nothing —
    with only 4 frames of context there is no long-range time structure for a
    fancier temporal mixer to find.

    `use_action=False` builds the identical network with the button input
    zeroed out.  That is project 40's control: same capacity, same data, same
    number of steps, no idea which button was pressed.
    """

    def __init__(self, ctx=2, horizon=4, base=64, cond=128, use_action=True,
                 predict_reward=False, ctx_noise=False):
        super().__init__()
        self.ctx, self.horizon, self.use_action = ctx, horizon, use_action
        self.predict_reward = predict_reward
        self.ctx_noise = ctx_noise
        self.cond_dim = cond
        self.act_emb = nn.Embedding(N_ACT + 1, cond // 2)   # +1 = "unknown"
        self.t_mlp = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(),
                                   nn.Linear(cond, cond))
        # The action embeddings are CONCATENATED, not averaged.  Averaging them
        # would throw away the order — "left then up" and "up then left" would
        # look identical to the network, and they lead to different screens.
        self.a_mlp = nn.Sequential(nn.Linear(horizon * (cond // 2), cond),
                                   nn.SiLU(), nn.Linear(cond, cond))
        cin = ctx + horizon
        self.stem = nn.Conv2d(cin, base, 3, padding=1)
        self.d1 = FiLMBlock(base, base, cond)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)   # 8->4
        self.mid = FiLMBlock(base * 2, base * 2, cond)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)       # 4->8
        self.u1 = FiLMBlock(base * 2, base, cond)
        self.out_n = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, horizon, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if predict_reward:
            self.rew = nn.Linear(base * 2, horizon)
        if ctx_noise:
            # Project 41 corrupts the context on purpose and tells the model
            # how much it corrupted it.  See its README for why.
            self.s_mlp = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(),
                                       nn.Linear(cond, cond))

    def cond_vector(self, t, actions, sigma=None):
        c = self.t_mlp(timestep_embedding(t, self.cond_dim))
        if self.use_action:
            c = c + self.a_mlp(self.act_emb(actions).flatten(1))
        if self.ctx_noise:
            s = torch.zeros_like(t) if sigma is None else sigma
            c = c + self.s_mlp(timestep_embedding(s * 1000.0, self.cond_dim))
        return c

    def forward(self, x_noisy, t, context, actions, return_reward=False,
                sigma=None):
        c = self.cond_vector(t, actions, sigma)
        h = self.stem(torch.cat([context, x_noisy], dim=1))
        h = self.d1(h, c)
        m = self.mid(self.down(h), c)
        u = self.u1(torch.cat([self.up(m), h], dim=1), c)
        v = self.out(F.silu(self.out_n(u)))
        if return_reward:
            return v, self.rew(m.mean(dim=(2, 3)))
        return v


class FlowWrapper(nn.Module):
    """Adapts ActionUNet to the (x, t, **kw) signature flow_lib.sample wants."""

    def __init__(self, net, context, actions, sigma=None):
        super().__init__()
        self.net, self.context, self.actions = net, context, actions
        self.sigma = sigma

    def forward(self, x, t):
        return self.net(x, t, self.context, self.actions, sigma=self.sigma)


def sample_frames(net, context, actions, steps=30, generator=None, sigma=None):
    """Generate `horizon` future latent frames.  Returns (B, K, 8, 8)."""
    flow = FL.RectifiedFlow()
    wrapped = FlowWrapper(net, context, actions, sigma)
    shape = (context.shape[0], net.horizon, GRID, GRID)
    return flow.sample(wrapped, shape, steps=steps, generator=generator)


@torch.no_grad()
def rollout(net, context, actions, steps=30, sigma=None, generator=None,
            clamp=False):
    """Play the model forward: its own output becomes its next context.

    `context` is (B, ctx, 8, 8); `actions` is (B, N) with N >= 1.  Returns
    (B, N, 8, 8) — one generated frame per action.

    This is the loop that turns a next-frame predictor into a playable world,
    and it is also the loop in which small errors get to compound.  Nothing
    here re-reads the real game after the first frame.
    """
    b = context.shape[0]
    n = actions.shape[1]
    ctx = context.clone()
    out = []
    for i in range(n):
        a = actions[:, i:i + 1]
        f = sample_frames(net, ctx, a, steps=steps, generator=generator,
                          sigma=sigma)
        if clamp:
            f = f.clamp(0.0, 1.0)
        out.append(f[:, -1])
        ctx = torch.cat([ctx[:, 1:], f[:, -1:]], dim=1)
    return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def sample_batch(ep, batch, ctx, horizon, rng, device="cpu"):
    """Random windows: `ctx` past frames, `horizon` future frames + actions."""
    n_ep, T = ep["agent"].shape
    e = rng.integers(0, n_ep, size=batch)
    t = rng.integers(ctx - 1, T - horizon, size=batch)
    idx = t[:, None] + np.arange(-ctx + 1, horizon + 1)[None]      # (B, ctx+K)
    ee = np.repeat(e[:, None], idx.shape[1], axis=1)
    frames = frames_of(ep, ee, idx)
    ctx_f = torch.from_numpy(frames[:, :ctx]).to(device)
    tgt_f = torch.from_numpy(frames[:, ctx:]).to(device)
    a_idx = t[:, None] + np.arange(horizon)[None]
    acts = torch.from_numpy(ep["act"][ee[:, :horizon], a_idx]).to(device)
    rews = torch.from_numpy(ep["rew"][ee[:, :horizon], a_idx]).to(device)
    return ctx_f, tgt_f, acts, rews


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ground_truth_rollout(walls, agent, coin, actions):
    """What the game *would* do — the yardstick every prediction is scored on.

    Only the agent path is deterministic; the coin position after a pickup is
    not, so callers compare coin predictions separately.
    """
    path, blocked = [], []
    a = agent
    for act in actions:
        dr, dc = DELTA[act]
        nr, nc = a[0] + dr, a[1] + dc
        hit = walls[nr, nc] == 1
        if not hit:
            a = (nr, nc)
        path.append(a)
        blocked.append(hit)
    return path, blocked


def agent_accuracy(pred_frames, walls, agent0, actions):
    """Fraction of predicted frames whose bright square is in the right cell."""
    path, blocked = ground_truth_rollout(walls, agent0, None, actions)
    ok, ok_blocked, n_blocked = 0, 0, 0
    for k, f in enumerate(pred_frames):
        _, a_idx, _, _ = read_frame(f)
        hit = (a_idx // GRID, a_idx % GRID) == path[k]
        ok += hit
        if blocked[k]:
            n_blocked += 1
            ok_blocked += hit
    return ok / len(pred_frames), ok_blocked, n_blocked


def coin_report(frame8, walls=None, tol=0.1):
    """Does this frame contain a coin at all, and how bright is the best one?

    Returns (has_coin, peak, cell).  `peak` is the brightest cell that is
    neither the player nor a wall; `has_coin` asks whether that peak actually
    lands on the coin grey level rather than somewhere vague.

    This is the sharpest test of whether a model is *generating* or *averaging*.
    When the coin respawns, roughly thirty cells are equally likely to hold it.
    A model that commits picks one and lights it fully (peak ~ 0.62, has_coin
    true, usually the wrong cell).  A model that averages spreads one coin's
    worth of brightness over all thirty (peak ~ 0.02, has_coin false) and hands
    you a screen with no coin on it -- a picture of a game state that cannot
    exist.
    """
    v = frame8.reshape(-1).astype(np.float32).copy()
    v[v.argmax()] = -1.0                              # that is the player
    if walls is not None:
        v[np.asarray(walls).reshape(-1) == 1] = -1.0
    cell = int(v.argmax())
    peak = float(v[cell])
    return bool(abs(peak - COIN) < tol), peak, cell


# ---------------------------------------------------------------------------
# saving pictures
# ---------------------------------------------------------------------------

# Display colours for the four grey levels.  Purely cosmetic — every model in
# this phase sees the grey number, never the colour — but a generated frame
# whose cell lands *between* two levels shows up as a muddy in-between colour,
# which makes "the model is hedging" visible at a glance.
PALETTE = np.array([[0.06, 0.06, 0.08],       # EMPTY
                    [0.24, 0.30, 0.42],       # WALL
                    [0.88, 0.66, 0.23],       # COIN
                    [0.97, 0.97, 0.95]],      # AGENT
                   dtype=np.float32)


def colorize(frame8):
    """(8,8) grey levels -> (32,32,3) RGB for display."""
    v = np.clip(np.asarray(frame8, dtype=np.float32), 0.0, 1.0)
    rgb = np.stack([np.interp(v, LEVELS, PALETTE[:, c]) for c in range(3)],
                   axis=-1)
    return np.repeat(np.repeat(rgb, CELL, axis=0), CELL, axis=1)


def write_gif(frames8, path, scale=4, fps=6):
    """frames8: list/array of (8,8) latent frames -> an animated GIF."""
    from PIL import Image
    imgs = []
    for f in np.asarray(frames8):
        px = np.repeat(np.repeat(colorize(f), scale, axis=0), scale, axis=1)
        imgs.append(Image.fromarray((px * 255).astype(np.uint8), mode="RGB"))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"wrote {path}")


def strip_image(rows, path, scale=3, gap=3):
    """A grid of latent frames as one PNG.  rows: list of lists of (8,8)."""
    from PIL import Image
    h = len(rows)
    w = max(len(r) for r in rows)
    tile = IMG * scale
    canvas = np.full((h * (tile + gap) - gap, w * (tile + gap) - gap, 3), 0.72,
                     dtype=np.float32)
    for i, row in enumerate(rows):
        for j, f in enumerate(row):
            if f is None:
                continue
            px = np.repeat(np.repeat(colorize(f), scale, axis=0), scale,
                           axis=1)
            y, x = i * (tile + gap), j * (tile + gap)
            canvas[y:y + tile, x:x + tile] = px
    Image.fromarray((canvas * 255).astype(np.uint8), mode="RGB").save(path)
    print(f"wrote {path}")


def count_params(m):
    return sum(p.numel() for p in m.parameters())
