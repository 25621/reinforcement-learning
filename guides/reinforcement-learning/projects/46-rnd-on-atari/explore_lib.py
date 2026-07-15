r"""Shared exploration stack for Phase 8: a sparse pixel world, PPO, and RND.

Imported by projects 47 (ICM) and 49 (noisy TV) via sys.path, so the three
projects compare bonuses on *identical* machinery — only the bonus changes.

What is in here
---------------
KeyDoorRoom   a two-room pixel maze whose only reward needs ~31 correct steps in
              a row: thread the tunnel, take the key, thread it back, open the door.
VecEnv        a plain-python vector of those envs (no gym dependency).
ActorCritic   small CNN, one policy head and TWO value heads (see below).
RND           the bonus of project 46: a frozen random target network and a
              predictor trained to copy it. Prediction error = novelty.
train         PPO. Pass `bonus=None` for a pure-extrinsic baseline, or any
              object with `.reward(obs, act, next_obs)` and `.update(...)`.

Why TWO value heads
-------------------
The extrinsic reward is episodic (it ends the episode) but novelty is not: dying
does not make a room you already saw interesting again. Those two signals need
different discounting and different bootstrapping, so they get separate critics
and separate advantages, added at the end with their own weights. This is exactly
what the RND paper does, and mixing them into one head is the most common way to
get a broken RND implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")   # the box's GPU (sm_61) has no kernels in this torch build

# ----------------------------------------------------------------- environment

LAYOUT = [
    "###############",
    "#.....#####...#",
    "#.....#####...#",
    "#.....#####...#",
    "#.....:::::...#",   # ':' = a FIVE-cell tunnel, the only way between the rooms
    "#.....#####...#",
    "#.....#####...#",
    "#.....#####...#",
    "###############",
]
START = (1, 4)     # left room, facing the tunnel
KEY = (13, 1)      # right room, far corner
DOOR = (1, 7)      # left room, the corner furthest from the key
TV = (13, 7)       # only used when tv=True (project 49): a screen showing static

MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # up, down, left, right


class KeyDoorRoom:
    """Sparse-reward pixel maze. +1 for reaching the door WITH the key, else 0.

    The layout is deliberately mean. To collect the +1 the agent must thread a
    five-cell tunnel, cross the far room to its corner, pick up the key, thread the
    tunnel BACK, and reach the opposite corner — about 31 correct moves in a row.
    A uniform random walker solves it 0 times in 3,000 episodes, and so does an
    untrained network (0 in 2,400).

    Observation: a small stack of binary images (channels x 9 x 15), like a heavily
    simplified Atari screen. Nothing in it is a "state number", so counting visits
    (project 45) is not an option — which is the entire reason RND exists.
    """

    n_actions = 4

    def __init__(self, horizon=200, tv=False, seed=0):
        self.grid = np.array([[c for c in row] for row in LAYOUT])
        self.walls = (self.grid == "#").astype(np.float32)
        self.h, self.w = self.walls.shape
        self.horizon = horizon
        self.tv = tv
        self.n_channels = 6 if tv else 5
        self.rng = np.random.default_rng(seed)

    # -- observation ---------------------------------------------------------
    def _obs(self):
        o = np.zeros((self.n_channels, self.h, self.w), dtype=np.float32)
        o[0] = self.walls
        o[1, self.pos[1], self.pos[0]] = 1.0
        # The key is DRAWN where it is: on its shelf, or in the agent's hand once taken —
        # the same courtesy Atari pays you with an inventory bar at the top of the screen.
        # It matters because novelty is only ever measured *in the predictor's eyes*: a
        # state that is meaningfully new has to also LOOK new, or a convolutional network
        # (which generalizes over small input changes by design) will file it as familiar.
        kx, ky = (self.pos[0], self.pos[1]) if self.has_key else KEY
        o[2, ky, kx] = 1.0
        o[3, DOOR[1], DOOR[0]] = 1.0
        o[4, :, :] = float(self.has_key)          # an "inventory bar": am I carrying the key?
        if self.tv:
            # The screen is only *visible* from the neighbouring cells: stand next
            # to it and you get a fresh sheet of random static, every single step.
            if abs(self.pos[0] - TV[0]) <= 1 and abs(self.pos[1] - TV[1]) <= 1:
                o[5, TV[1] - 1:TV[1] + 2, TV[0] - 1:TV[0] + 2] = self.rng.random((3, 3), dtype=np.float32)
        return o

    def near_tv(self):
        return abs(self.pos[0] - TV[0]) <= 1 and abs(self.pos[1] - TV[1]) <= 1

    # -- dynamics ------------------------------------------------------------
    def reset(self):
        self.pos = list(START)
        self.has_key = False
        self.t = 0
        return self._obs()

    def step(self, a):
        dx, dy = MOVES[a]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if self.grid[ny, nx] != "#":              # walls block; ':' is walkable
            self.pos = [nx, ny]
        self.t += 1
        if tuple(self.pos) == KEY:
            self.has_key = True
        r = 0.0
        done = False
        if tuple(self.pos) == DOOR and self.has_key:
            r, done = 1.0, True                   # the ONLY reward in the world
        if self.t >= self.horizon:
            done = True
        return self._obs(), r, done

    def state_id(self):
        """A hashable summary used only for measuring coverage, never by the agent."""
        return (self.pos[0], self.pos[1], int(self.has_key))


class VecEnv:
    """n independent copies, auto-reset on done. Returns stacked numpy arrays."""

    def __init__(self, n, seed=0, **kw):
        self.envs = [KeyDoorRoom(seed=seed * 100 + i, **kw) for i in range(n)]
        self.n = n
        self.n_channels = self.envs[0].n_channels
        self.n_actions = self.envs[0].n_actions

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rew, done = [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d = e.step(int(a))
            if d:
                o = e.reset()                     # the obs we return is the NEXT episode's first
            obs.append(o)
            rew.append(r)
            done.append(d)
        return np.stack(obs), np.array(rew, dtype=np.float32), np.array(done, dtype=np.float32)

    def states(self):
        return [e.state_id() for e in self.envs]

    def near_tv(self):
        return np.array([e.near_tv() for e in self.envs], dtype=np.float32)


# ----------------------------------------------------------------- normalizers

class RunningMeanStd:
    """Streaming mean/variance (Welford). Used to standardize what RND sees."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x, clip=5.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)


class RewardForwardFilter:
    """Running estimate of the DISCOUNTED intrinsic return, used to rescale bonuses.

    Why not just divide by the standard deviation of the raw bonus? Because the
    bonus shrinks as the predictor learns: an unscaled bonus is huge at step 0 and
    tiny at step 100k, so the intrinsic reward would slowly switch itself off and
    the whole run would drift. Dividing by the running std of the intrinsic
    *return* keeps the bonus at a roughly constant scale for the whole run, which
    is what lets one fixed `int_coef` work from beginning to end.
    """

    def __init__(self, gamma):
        self.gamma = gamma
        self.acc = None
        self.rms = RunningMeanStd(())

    def update(self, rews):                        # rews: (T, N)
        for t in range(rews.shape[0]):
            self.acc = rews[t] if self.acc is None else self.acc * self.gamma + rews[t]
            self.rms.update(self.acc.reshape(-1))
        return float(np.sqrt(self.rms.var + 1e-8))


# ----------------------------------------------------------------------- model

def cnn_trunk(in_ch, out_dim, hw=None):
    """Two conv layers then a linear head. `hw` is the input picture size.

    The second conv has stride 2, which halves each side (rounding up), so the
    flattened width is 32 * ceil(h/2) * ceil(w/2). Passing `hw` keeps one function
    usable for the maze AND for the 105x80 Atari frames of project 46's part 2;
    leaving it None reads the size off the current LAYOUT.
    """
    if hw is None:
        hw = (len(LAYOUT), len(LAYOUT[0]))
    h, w = (hw[0] + 1) // 2, (hw[1] + 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * h * w, out_dim), nn.ReLU(),
    )


class ActorCritic(nn.Module):
    def __init__(self, in_ch, n_actions, hidden=128, hw=None):
        super().__init__()
        self.trunk = cnn_trunk(in_ch, hidden, hw)
        self.pi = nn.Linear(hidden, n_actions)
        self.v_ext = nn.Linear(hidden, 1)
        self.v_int = nn.Linear(hidden, 1)          # the second critic (see module docstring)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v_ext(h).squeeze(-1), self.v_int(h).squeeze(-1)

    def act(self, x):
        logits, ve, vi = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), ve, vi


# ------------------------------------------------------------------------- RND

class RND:
    """Random Network Distillation.

    target    : a random network, never trained. It maps each observation to a
                fixed 64-number fingerprint. There is nothing special about the
                fingerprint — the point is only that it is a *deterministic
                function of the observation* that nobody has memorized yet.
    predictor : same shape, trained to copy the target on the states the agent
                actually visits. On a state seen a hundred times it copies well;
                on a state it has never seen it guesses badly.

    bonus = || predictor(s) - target(s) ||^2, which is large exactly on states
    absent from the agent's own history. That is a novelty detector built out of
    nothing but regression, with no counts, no density model, and no memory.
    """

    name = "RND"

    def __init__(self, in_ch, n_actions, obs_shape, out_dim=64, lr=1e-3,
                 hw=None, update_proportion=0.25):
        self.target = cnn_trunk(in_ch, out_dim, hw).to(DEVICE)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = cnn_trunk(in_ch, out_dim, hw).to(DEVICE)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(obs_shape)
        self.update_proportion = update_proportion
        self.rng = np.random.default_rng(0)

    def _prep(self, obs):
        """Standardize the observation before it reaches the two networks.

        Skip this and RND quietly degrades: the target is random, so the scale of its
        output tracks the scale of its input. Whichever pixels happen to be brightest
        would dominate the fingerprint, and the novelty signal would then say more
        about image brightness than about novelty.

        Note this only READS the statistics. They are updated in `update()`, which
        sees real rollout data — the heat-map probe calls `reward()` on hypothetical
        observations the agent never visited, and those must not be counted as
        experience.
        """
        flat = obs.reshape(obs.shape[0], -1)
        return torch.as_tensor(self.obs_rms.norm(flat), device=DEVICE).reshape(obs.shape)

    def reward(self, obs, act, next_obs):
        x = self._prep(next_obs)
        with torch.no_grad():
            err = (self.predictor(x) - self.target(x)).pow(2).mean(-1)
        return err.cpu().numpy()

    def update(self, obs, act, next_obs):
        self.obs_rms.update(next_obs.reshape(next_obs.shape[0], -1))

        # Train the predictor on only a FRACTION of the experience (the paper's
        # `update_proportion`, 0.25). Deliberately handicapping your own predictor
        # looks perverse until you see the race it is in: the predictor learns states
        # far faster than the policy learns to reach the reward, and once every state
        # is familiar the bonus dies and the agent goes back to wandering at random.
        # Feeding the predictor a quarter of the data keeps novelty alive long enough
        # for the policy to convert it into a route.
        keep = self.rng.random(len(next_obs)) < self.update_proportion
        if keep.sum() < 2:
            return 0.0
        x = self._prep(next_obs[keep])
        loss = (self.predictor(x) - self.target(x).detach()).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


class NoBonus:
    """The control arm: PPO with nothing but the environment's own reward."""

    name = "PPO only"

    def reward(self, obs, act, next_obs):
        return np.zeros(len(obs), dtype=np.float32)

    def update(self, obs, act, next_obs):
        return 0.0


# ------------------------------------------------------------------------- PPO

def train(make_bonus=None, total_steps=400_000, n_envs=16, n_steps=128, seed=0,
          lr=2.5e-4, gamma_ext=0.99, gamma_int=0.99, lam=0.95, clip=0.2,
          ent_coef=0.005, ext_coef=2.0, int_coef=1.0, epochs=4, n_minibatch=4,
          tv=False, horizon=200, log_every=20, verbose=True, probe=None):
    """PPO with an optional novelty bonus. Returns a dict of learning curves.

    `make_bonus(in_ch, n_actions, obs_shape)` builds the bonus; None = baseline.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = VecEnv(n_envs, seed=seed, tv=tv, horizon=horizon)
    obs_shape = (envs.n_channels, len(LAYOUT), len(LAYOUT[0]))
    net = ActorCritic(envs.n_channels, envs.n_actions).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    bonus = NoBonus() if make_bonus is None else make_bonus(
        envs.n_channels, envs.n_actions, int(np.prod(obs_shape)))
    rff = RewardForwardFilter(gamma_int)

    obs = envs.reset()
    n_updates = total_steps // (n_envs * n_steps)
    n_states = 2 * int((envs.envs[0].grid != "#").sum())   # (cell, carrying-key?) pairs
    visited, seen_key = set(), False
    hist = {"steps": [], "success": [], "coverage": [], "bonus": [], "tv_frac": [], "probe": []}
    ep_success, ep_count, tv_steps, tot_steps = 0, 0, 0.0, 0
    first_success = None

    for upd in range(n_updates):
        b_obs = np.zeros((n_steps, n_envs) + obs_shape, dtype=np.float32)
        b_next = np.zeros_like(b_obs)
        b_act = np.zeros((n_steps, n_envs), dtype=np.int64)
        b_logp = np.zeros((n_steps, n_envs), dtype=np.float32)
        b_rext = np.zeros((n_steps, n_envs), dtype=np.float32)
        b_done = np.zeros((n_steps, n_envs), dtype=np.float32)
        b_vext = np.zeros((n_steps, n_envs), dtype=np.float32)
        b_vint = np.zeros((n_steps, n_envs), dtype=np.float32)

        for t in range(n_steps):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, _, ve, vi = net.act(torch.as_tensor(obs, device=DEVICE))
            a = a.cpu().numpy()
            if tv:
                tv_steps += envs.near_tv().sum()
            for st in envs.states():
                visited.add(st)
                seen_key = seen_key or st[2] == 1
            next_obs, r, d = envs.step(a)
            b_act[t], b_logp[t] = a, logp.cpu().numpy()
            b_vext[t], b_vint[t] = ve.cpu().numpy(), vi.cpu().numpy()
            b_rext[t], b_done[t] = r, d
            b_next[t] = next_obs
            ep_success += int(r.sum())
            ep_count += int(d.sum())
            obs = next_obs
            tot_steps += n_envs
            if first_success is None and r.sum() > 0:
                first_success = tot_steps

        # ---- intrinsic reward for every transition in the rollout
        flat_obs = b_obs.reshape(-1, *obs_shape)
        flat_next = b_next.reshape(-1, *obs_shape)
        flat_act = b_act.reshape(-1)
        r_int = bonus.reward(flat_obs, flat_act, flat_next).reshape(n_steps, n_envs)
        scale = rff.update(r_int)
        r_int = r_int / scale                     # keeps the bonus scale steady over the run

        with torch.no_grad():
            _, last_ve, last_vi = net(torch.as_tensor(obs, device=DEVICE))
            last_ve, last_vi = last_ve.cpu().numpy(), last_vi.cpu().numpy()

        adv_ext = _gae(b_rext, b_vext, b_done, last_ve, gamma_ext, lam, episodic=True)
        # Intrinsic advantage ignores `done`: novelty does not reset when you die,
        # so an episode boundary must not cut the intrinsic return.
        adv_int = _gae(r_int, b_vint, b_done, last_vi, gamma_int, lam, episodic=False)
        ret_ext, ret_int = adv_ext + b_vext, adv_int + b_vint
        adv = ext_coef * adv_ext + int_coef * adv_int

        # ---- PPO epochs
        t_obs = torch.as_tensor(flat_obs, device=DEVICE)
        t_act = torch.as_tensor(flat_act, device=DEVICE)
        t_logp = torch.as_tensor(b_logp.reshape(-1), device=DEVICE)
        t_adv = torch.as_tensor(adv.reshape(-1), device=DEVICE)
        t_rext = torch.as_tensor(ret_ext.reshape(-1), device=DEVICE)
        t_rint = torch.as_tensor(ret_int.reshape(-1), device=DEVICE)
        batch = n_steps * n_envs
        idx = np.arange(batch)
        mb = batch // n_minibatch
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, batch, mb):
                j = idx[start:start + mb]
                logits, ve, vi = net(t_obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(t_act[j])
                ratio = (logp - t_logp[j]).exp()
                a_j = t_adv[j]
                a_j = (a_j - a_j.mean()) / (a_j.std() + 1e-8)
                pg = -torch.min(ratio * a_j, ratio.clamp(1 - clip, 1 + clip) * a_j).mean()
                v_loss = 0.5 * (F.mse_loss(ve, t_rext[j]) + F.mse_loss(vi, t_rint[j]))
                loss = pg + v_loss - ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
            bonus.update(flat_obs, flat_act, flat_next)

        if (upd + 1) % log_every == 0 or upd == n_updates - 1:
            sr = ep_success / max(ep_count, 1)
            hist["steps"].append(tot_steps)
            hist["success"].append(sr)
            hist["coverage"].append(len(visited))
            hist["bonus"].append(float(r_int.mean()))
            hist["tv_frac"].append(tv_steps / max(tot_steps, 1))
            if probe is not None:
                hist["probe"].append(probe(bonus))   # e.g. "what does the bonus look like
                                                     # in every cell of the maze right now?"
            if verbose:
                print(f"  [{bonus.name}] step {tot_steps:>7d}  success {sr:.3f}  "
                      f"states seen {len(visited):>3d}/{n_states}  bonus {r_int.mean():.3f}"
                      + (f"  tv {tv_steps / tot_steps:.3f}" if tv else ""), flush=True)
            ep_success, ep_count = 0, 0

    hist["first_success"] = first_success
    hist["found_key"] = seen_key
    hist["name"] = bonus.name
    return hist


def _gae(rew, val, done, last_val, gamma, lam, episodic=True):
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t] if episodic else 1.0
        next_val = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv


def random_policy_success(episodes=2000, seed=0, tv=False):
    """How often does a coin-flipping agent solve this maze? (The bar to beat.)"""
    env = KeyDoorRoom(tv=tv, seed=seed)
    rng = np.random.default_rng(seed)
    hits, keys = 0, 0
    for _ in range(episodes):
        env.reset()
        done, got_key = False, False
        while not done:
            _, r, done = env.step(int(rng.integers(4)))
            got_key = got_key or env.has_key
            hits += int(r > 0)
        keys += int(got_key)
    return hits / episodes, keys / episodes


# ------------------------------------------------------------------ bonus probe

def _place(env, x, y, has_key=False):
    env.pos = [x, y]
    env.has_key = has_key
    return env._obs()


def transition_probes(tv=False):
    """One (obs, action, next_obs) triple per walkable cell: STEPPING INTO that cell.

    Both bonuses answer the same question this way — "how much would you pay me to
    arrive here?" — which is what makes the two heat-maps comparable. (Handing RND
    and ICM a fake self-loop `s -> s` instead would be meaningless for ICM, whose
    bonus is the error of predicting a *change*.)

    These are hypothetical transitions built by hand. The agent never takes them and
    nothing is trained on them; they exist only to draw a picture.
    """
    env = KeyDoorRoom(tv=tv)
    env.reset()
    cells, obs, acts, next_obs = [], [], [], []
    for y in range(env.h):
        for x in range(env.w):
            if env.grid[y, x] == "#":
                continue
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = x - dx, y - dy                  # a neighbour that can step INTO (x, y)
                if env.grid[ny, nx] == "#":
                    continue
                cells.append((x, y))
                obs.append(_place(env, nx, ny))
                acts.append(a)
                next_obs.append(_place(env, x, y))
                break
    return cells, np.stack(obs), np.array(acts, dtype=np.int64), np.stack(next_obs)


def bonus_map(bonus, tv=False):
    """Heat-map of what the bonus currently pays for arriving in each cell (NaN = wall)."""
    cells, obs, acts, next_obs = transition_probes(tv=tv)
    vals = bonus.reward(obs, acts, next_obs)
    env = KeyDoorRoom(tv=tv)
    heat = np.full((env.h, env.w), np.nan, dtype=np.float32)
    for (x, y), v in zip(cells, vals):
        heat[y, x] = v
    return heat


def tv_floor_share():
    """Fraction of walkable cells that sit next to the TV.

    This is the honest reference for "not watching television": an agent that spread
    its time evenly over the maze would be next to the TV this often, by accident.
    """
    env = KeyDoorRoom(tv=True)
    walk = [(x, y) for y in range(env.h) for x in range(env.w) if env.grid[y, x] != "#"]
    near = [c for c in walk if abs(c[0] - TV[0]) <= 1 and abs(c[1] - TV[1]) <= 1]
    return len(near) / len(walk)
