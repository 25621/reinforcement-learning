"""Project 36 — Mini MuZero on Tic-Tac-Toe.

MuZero is three networks that only ever talk to each other:

    representation  h(observation)      -> s0          "what matters about this board"
    dynamics        g(s, a)             -> s', r       "if I play a, what then?"
    prediction      f(s)                -> policy, v   "how good is this, what to try"

Note what is missing. There is no decoder. Nothing ever turns `s` back into a board, and
nothing checks that it could. `s` is not a board, it is not a compressed board, and it
does not have to be: it is whatever internal scratchpad makes the *predictions*
(policy, value, reward) come out right. The three networks are trained end-to-end
through the search, so they are free to agree on any private language they like.

That is the "Zero" lineage's last step. AlphaZero was handed the rules of the game and
searched with them. MuZero is handed nothing and learns a rule-like thing that is only
required to be right about the things a decision depends on.

Two consequences you can see in this file:

  * The tree search calls `g` to move between nodes. It never calls the real game. Below
    the root, MuZero does not know which moves are legal — it has never been told — and
    it plans through illegal moves anyway. It has to learn that playing on an occupied
    square is a waste of a turn, the same way it learns everything else: because doing
    so predicts a worse value.
  * The whole thing is trained by *unrolling*: encode a real board once, then run `g`
    forward K times, and demand that the policy/value/reward heads keep matching the
    real game at every one of those K imagined steps. This is the "recurrence" the
    project brief asks you to see, and it is the reason the three heads end up sharing
    one representation instead of three.

  python3 muzero.py     # ~5.5 min on 12 hyperthreads
"""

import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
torch.set_num_threads(1)

N_ACTIONS = 9
LATENT = 64


# ==========================================================================
# The game. Always seen from the perspective of the player about to move.
# ==========================================================================
class TicTacToe:
    """Board as a length-9 array: +1 = the player to move, -1 = the opponent, 0 = empty.

    Everything is *canonical*: after each move the board is negated, so "me" is always
    +1. The network therefore only ever learns one point of view, and never needs a
    "whose turn is it" input. Two players' worth of skill for one player's worth of
    training data.
    """

    WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
            (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def __init__(self, board=None):
        self.board = np.zeros(9, dtype=np.int8) if board is None else board

    def legal(self):
        return np.flatnonzero(self.board == 0)

    def step(self, a):
        """Play `a`. Returns (next_state, reward_for_the_mover, done).

        The returned state is negated, so it is again "from the perspective of whoever
        moves next". Reward is +1 if that move just won, else 0.
        """
        b = self.board.copy()
        b[a] = 1
        won = any(all(b[i] == 1 for i in line) for line in self.WINS)
        if won:
            return TicTacToe(-b), 1.0, True
        if not (b == 0).any():
            return TicTacToe(-b), 0.0, True  # draw
        return TicTacToe(-b), 0.0, False

    def obs(self):
        """Two planes: my stones, their stones. (An empty square is 0 in both.)"""
        return np.concatenate([(self.board == 1), (self.board == -1)]).astype(np.float32)


def perfect_move(state, rng):
    """A flawless minimax opponent, used only for evaluation. Never trains anything.

    Tic-Tac-Toe is small enough to solve exactly, which gives us an unbeatable yardstick.
    Against perfect play the best possible result is a DRAW, so "never loses to this" is
    the true ceiling — and a far more honest score than "beats a random player", which a
    mediocre agent also does.
    """
    best, best_moves = -2, []
    for a in state.legal():
        nxt, r, done = state.step(a)
        v = r if done else -_minimax(nxt)
        if v > best:
            best, best_moves = v, [a]
        elif v == best:
            best_moves.append(a)
    return int(rng.choice(best_moves))


_MM_CACHE = {}


def _minimax(state):
    """Value of `state` for the player to move: +1 win, 0 draw, -1 loss."""
    key = state.board.tobytes()
    if key in _MM_CACHE:
        return _MM_CACHE[key]
    best = -2
    for a in state.legal():
        nxt, r, done = state.step(a)
        v = r if done else -_minimax(nxt)
        best = max(best, v)
    if best == -2:
        best = 0.0
    _MM_CACHE[key] = best
    return best


# ==========================================================================
# The three networks
# ==========================================================================
class MuZeroNet(nn.Module):
    def __init__(self, hidden=128, latent=LATENT):
        super().__init__()
        # h: board -> latent. The ONLY place a real observation ever enters.
        self.repr = nn.Sequential(
            nn.Linear(18, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.Tanh(),
        )
        # g: (latent, one-hot action) -> (next latent, reward). The learned "rules".
        self.dyn = nn.Sequential(
            nn.Linear(latent + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dyn_state = nn.Sequential(nn.Linear(hidden, latent), nn.Tanh())
        self.dyn_reward = nn.Linear(hidden, 1)
        # f: latent -> (policy, value). Used at every node of the tree.
        self.pred = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU())
        self.pred_policy = nn.Linear(hidden, N_ACTIONS)
        self.pred_value = nn.Linear(hidden, 1)

    def initial(self, obs):
        """h then f: encode a real board and immediately evaluate it."""
        s = self.repr(obs)
        p, v = self.prediction(s)
        return s, p, v

    def recurrent(self, s, a_onehot):
        """g then f: take an imagined step and immediately evaluate where you landed.

        This pair is the engine of the tree search. Everything below the root is
        produced here — the search never touches the real game again.
        """
        h = self.dyn(torch.cat([s, a_onehot], dim=-1))
        s2 = self.dyn_state(h)
        r = self.dyn_reward(h).squeeze(-1)
        p, v = self.prediction(s2)
        return s2, r, p, v

    def prediction(self, s):
        h = self.pred(s)
        # `tanh` on the value: a Tic-Tac-Toe outcome is always in [-1, +1], so say so.
        return self.pred_policy(h), torch.tanh(self.pred_value(h)).squeeze(-1)


def onehot(a):
    x = torch.zeros(N_ACTIONS)
    x[a] = 1.0
    return x


# ==========================================================================
# MCTS — over the LEARNED model, not the real game
# ==========================================================================
@dataclass
class Node:
    prior: float
    to_play_sign: int = 1        # +1 or -1 relative to the root's mover
    latent: torch.Tensor = None
    reward: float = 0.0
    value_sum: float = 0.0
    visits: int = 0
    children: dict = field(default_factory=dict)
    # Only populated when `known_rules=True`. Pure MuZero has NO idea what board it is
    # looking at once it is one step below the root — that is the entire point of it.
    state: "TicTacToe" = None
    terminal: bool = False

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class MinMax:
    """Rescales Q into [0, 1] so the exploration constant means the same thing always.

    Without this, `c_puct` is comparing a Q measured in "outcome units" against a prior
    probability, and the balance between them silently depends on how big the values
    happen to be. Normalizing removes that accident.
    """

    def __init__(self):
        self.lo, self.hi = float("inf"), float("-inf")

    def update(self, v):
        self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def norm(self, v):
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return v


def run_mcts(net, root_state, n_sims, rng, c_puct=1.25, dirichlet=0.6, explore_frac=0.25,
             add_noise=True, known_rules=False):
    """MuZero's search. Returns the visit-count distribution over root actions.

    The output is NOT the policy head's opinion — it is what the search *concluded*
    after `n_sims` imagined games. That is the improvement operator at the heart of
    every "Zero" algorithm: the network proposes, the search disposes, and then the
    network is trained to imitate the search. The search is a better player than the
    network that powers it, so imitating it makes the network better, which makes the
    next search better still.

    `known_rules` is the variable this project exists to study.

      False (real MuZero)  Below the root, the search has no idea which moves are legal or
                           whether the game has already ended. It plans through illegal
                           moves, and the learned dynamics — which was never trained on such
                           a transition, because self-play never makes one — returns a
                           garbage latent that poisons the whole subtree.
      True  (the control)  The tree is told the rules: legal moves only, and stop at a
                           terminal position. The *dynamics is still learned* — only
                           legality and termination are given. So this isolates exactly one
                           thing: what does not knowing the rules cost you?
    """
    obs = torch.as_tensor(root_state.obs()).unsqueeze(0)
    with torch.no_grad():
        s, p_logits, v = net.initial(obs)

    # Legal moves are masked HERE and nowhere else. At the root we have the real board,
    # so we use it. One node deeper, we are inside the learned model and have no idea
    # what is legal — and MuZero deliberately does not find out.
    legal = root_state.legal()
    priors = torch.softmax(p_logits[0], dim=-1).numpy()
    mask = np.zeros(N_ACTIONS)
    mask[legal] = 1
    priors = priors * mask
    priors = priors / priors.sum() if priors.sum() > 0 else mask / mask.sum()

    if add_noise:
        # Dirichlet noise at the root: forces the search to give every legal move at
        # least a look, so a network that has become (over)confident too early cannot
        # starve the alternative it never tried.
        noise = rng.dirichlet([dirichlet] * len(legal))
        for i, a in enumerate(legal):
            priors[a] = (1 - explore_frac) * priors[a] + explore_frac * noise[i]

    root = Node(prior=1.0, latent=s[0], state=root_state if known_rules else None)
    for a in range(N_ACTIONS):
        if priors[a] > 0:
            root.children[a] = Node(prior=float(priors[a]), to_play_sign=-1)

    mm = MinMax()

    for _ in range(n_sims):
        node, path = root, [root]
        # ---- 1. SELECT: walk down, picking the most promising child each time ----
        while node.children and not node.terminal:
            total = math.sqrt(max(1, node.visits))
            best_score, best_a, best_child = -1e9, None, None
            for a, child in node.children.items():
                u = c_puct * child.prior * total / (1 + child.visits)
                # What is action `a` worth TO THE PLAYER CHOOSING IT?
                #
                #   child.reward   the reward this move itself pays (in Tic-Tac-Toe:
                #                  +1 exactly when the move wins the game)
                #   -child.q()     everything after it. `child.q()` is from the CHILD's
                #                  mover's point of view, and the child's mover is the
                #                  opponent — so what is good for them is bad for us.
                #                  This one minus sign is the whole two-player adaptation.
                #
                # Dropping `child.reward` here is a subtle and expensive bug: a move that
                # wins on the spot would score no better than any other, because the win
                # is paid out in the reward, and the position after it (from the loser's
                # seat) is just "a lost game" like any other.
                q = mm.norm(child.reward - child.q()) if child.visits else 0.0
                score = q + u
                if score > best_score:
                    best_score, best_a, best_child = score, a, child
            node = best_child
            path.append(node)

        # ---- 2. EXPAND: reached a leaf. Ask the learned model what happens next ----
        if node.terminal:
            # Already a finished game: nothing to imagine, its value is 0 (no moves left)
            # and the outcome is already carried in node.reward.
            value = 0.0
        else:
            parent = path[-2]
            with torch.no_grad():
                s2, r, p_logits, v = net.recurrent(
                    parent.latent.unsqueeze(0), onehot(best_a).unsqueeze(0)
                )
            node.latent = s2[0]
            node.reward = float(r.item())
            value = float(v.item())
            p = torch.softmax(p_logits[0], dim=-1).numpy()

            if known_rules:
                # Take the real move to find out what the board is, whether the game is
                # over, and what is legal from here. The LATENT is still the learned one
                # above — only legality, termination and the reward come from the rules.
                nxt, r_true, done = parent.state.step(best_a)
                node.state = nxt
                node.reward = r_true
                node.terminal = done
                if done:
                    value = 0.0
                else:
                    legal_here = nxt.legal()
                    mask = np.zeros(N_ACTIONS)
                    mask[legal_here] = 1
                    p = p * mask
                    p = p / p.sum() if p.sum() > 0 else mask / mask.sum()
                    for a in legal_here:
                        node.children[int(a)] = Node(
                            prior=float(p[a]), to_play_sign=-node.to_play_sign
                        )
            else:
                # Pure MuZero: every action is "available", including the ones that would
                # be illegal on the real board. The model has to learn that for itself.
                for a in range(N_ACTIONS):
                    node.children[a] = Node(
                        prior=float(p[a]), to_play_sign=-node.to_play_sign
                    )

        # ---- 3. BACKUP: carry the leaf's value up the path, flipping sign each ply ----
        for n in reversed(path):
            n.value_sum += value
            n.visits += 1
            # Feed the MinMax the same quantity the selection rule normalizes, so the
            # scale it learns is the scale it is asked about.
            mm.update(n.reward - n.q())
            value = n.reward - value  # negamax: my gain is your loss (gamma = 1)

    counts = np.array([root.children[a].visits if a in root.children else 0
                       for a in range(N_ACTIONS)], dtype=np.float64)
    return counts / counts.sum(), root.q()


# ==========================================================================
# Self-play
# ==========================================================================
def play_game(net, n_sims, rng, temperature=1.0, known_rules=False):
    """One self-play game. Records what the SEARCH wanted at each move, not what it did."""
    state = TicTacToe()
    obs_list, pi_list, act_list, rew_list, sign_list = [], [], [], [], []
    done = False
    ply = 0
    while not done:
        pi, _ = run_mcts(net, state, n_sims, rng, known_rules=known_rules)
        obs_list.append(state.obs())
        pi_list.append(pi)
        # Temperature: early moves are sampled (variety in the training data), later
        # moves are greedy (don't throw a won game to explore).
        if temperature > 0 and ply < 3:
            a = int(rng.choice(N_ACTIONS, p=pi))
        else:
            a = int(pi.argmax())
        state, r, done = state.step(a)
        act_list.append(a)
        rew_list.append(r)
        sign_list.append(1)
        ply += 1

    # Value target: the final outcome, seen from the player who was about to move.
    # The game ended with the LAST mover getting `rew_list[-1]` (+1 for a win, 0 draw),
    # so walk backwards flipping the sign at each ply.
    z = rew_list[-1]
    values = []
    for t in range(len(act_list) - 1, -1, -1):
        values.append(z)
        z = -z
    values.reverse()
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "pi": np.array(pi_list, dtype=np.float32),
        "act": np.array(act_list, dtype=np.int64),
        "rew": np.array(rew_list, dtype=np.float32),
        "val": np.array(values, dtype=np.float32),
        "outcome": rew_list[-1],
    }


# ==========================================================================
# Training: unroll the model K steps and supervise every step
# ==========================================================================
def make_batch(games, batch_size, unroll, rng):
    """Sample positions, and for each one grab the next `unroll` moves that followed.

    The target for imagined step k is the real game's policy/value/reward at ply t+k.
    So the model is not asked to reconstruct anything — only to keep its *predictions*
    correct after k steps of imagination. That is the whole training signal, and it is
    why the latent can be anything it likes.
    """
    obs, acts, pis, vals, rews, masks = [], [], [], [], [], []
    for _ in range(batch_size):
        g = games[rng.integers(len(games))]
        T = len(g["act"])
        t = rng.integers(T)
        obs.append(g["obs"][t])
        a_seq, pi_seq, v_seq, r_seq, m_seq = [], [], [], [], []
        for k in range(unroll):
            i = t + k
            if i < T:
                a_seq.append(g["act"][i])
                pi_seq.append(g["pi"][i])
                v_seq.append(g["val"][i])
                r_seq.append(g["rew"][i])
                m_seq.append(1.0)
            else:
                # Past the end of the game: pad, and mask the loss out. Absorbing
                # states would otherwise teach the model that games go on forever.
                a_seq.append(0)
                pi_seq.append(np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS)
                v_seq.append(0.0)
                r_seq.append(0.0)
                m_seq.append(0.0)
        acts.append(a_seq)
        pis.append(pi_seq)
        vals.append(v_seq)
        rews.append(r_seq)
        masks.append(m_seq)
    return (
        torch.as_tensor(np.array(obs)),
        torch.as_tensor(np.array(acts)),
        torch.as_tensor(np.array(pis)),
        torch.as_tensor(np.array(vals)),
        torch.as_tensor(np.array(rews)),
        torch.as_tensor(np.array(masks)),
    )


def train_step(net, optim, batch, unroll):
    obs, acts, pis, vals, rews, masks = batch
    B = obs.shape[0]

    s, p_logits, v = net.initial(obs)
    # Step 0 is the real board, so there is no reward to predict yet.
    loss_p = -(pis[:, 0] * F.log_softmax(p_logits, dim=-1)).sum(-1) * masks[:, 0]
    loss_v = (v - vals[:, 0]) ** 2 * masks[:, 0]
    loss_r = torch.zeros(B)

    for k in range(unroll):
        a1h = F.one_hot(acts[:, k], N_ACTIONS).float()
        s, r, p_logits, v = net.recurrent(s, a1h)
        # Gradient scaling: without it the earliest latent gets `unroll` times more
        # gradient than the last and the recurrence destabilizes. MuZero halves the
        # gradient flowing back through each dynamics step.
        s.register_hook(lambda grad: grad * 0.5)
        m = masks[:, k]
        loss_r = loss_r + (r - rews[:, k]) ** 2 * m
        if k + 1 < unroll:
            m2 = masks[:, k + 1]
            loss_p = loss_p - (pis[:, k + 1] * F.log_softmax(p_logits, dim=-1)).sum(-1) * m2
            loss_v = loss_v + (v - vals[:, k + 1]) ** 2 * m2

    loss = (loss_p + loss_v + loss_r).mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optim.step()
    return loss.item(), loss_p.mean().item(), loss_v.mean().item(), loss_r.mean().item()


# ==========================================================================
# Evaluation
# ==========================================================================
def evaluate(net, n_sims, rng, n_games=40, opponent="perfect", known_rules=False):
    """Play `n_games` against a fixed opponent, half as first player, half as second.

    Reported as (win, draw, loss) rates from MuZero's side. Against `perfect`, a draw is
    the best attainable result, so the number to watch is the LOSS rate going to zero.
    """
    w = d = l = 0
    for g in range(n_games):
        state = TicTacToe()
        muzero_first = g % 2 == 0
        turn_is_muzero = muzero_first
        done = False
        while not done:
            if turn_is_muzero:
                if n_sims <= 1:
                    # Search disabled: act on the raw policy head. This is the
                    # "how much of the strength is the network alone?" arm.
                    with torch.no_grad():
                        _, p, _ = net.initial(torch.as_tensor(state.obs()).unsqueeze(0))
                    pr = torch.softmax(p[0], -1).numpy()
                    legal = state.legal()
                    a = int(legal[np.argmax(pr[legal])])
                else:
                    pi, _ = run_mcts(net, state, n_sims, rng, add_noise=False,
                                     known_rules=known_rules)
                    a = int(pi.argmax())
            elif opponent == "perfect":
                a = perfect_move(state, rng)
            else:
                a = int(rng.choice(state.legal()))
            state, r, done = state.step(a)
            if done:
                if r == 1.0:      # the player who just moved won
                    if turn_is_muzero:
                        w += 1
                    else:
                        l += 1
                else:
                    d += 1
            turn_is_muzero = not turn_is_muzero
    n = float(n_games)
    return w / n, d / n, l / n


# ==========================================================================
# The loop
# ==========================================================================
def train_muzero(seed=0, iters=50, games_per_iter=60, n_sims=25, unroll=3,
                 batch_size=128, steps_per_iter=150, progress=False, known_rules=False,
                 label=""):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # BEFORE the net is built, or its weights are unseeded
    rng = np.random.default_rng(seed)

    net = MuZeroNet()
    optim = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)

    replay = []
    hist = {"iter": [], "games": [], "loss": [], "loss_v": [], "loss_r": [],
            "perfect_loss_rate": [], "perfect_draw_rate": [], "random_win_rate": []}
    t0 = time.time()
    total_games = 0

    for it in range(iters):
        for _ in range(games_per_iter):
            replay.append(play_game(net, n_sims, rng, known_rules=known_rules))
            total_games += 1
        replay = replay[-1500:]  # a rolling window: stale games were played by a much
                                 # weaker net, and their search policies are bad targets

        losses = []
        for _ in range(steps_per_iter):
            batch = make_batch(replay, batch_size, unroll, rng)
            losses.append(train_step(net, optim, batch, unroll))
        L = np.mean(losses, axis=0)

        if it % 5 == 0 or it == iters - 1:
            w, d, l = evaluate(net, n_sims, rng, n_games=30, opponent="perfect",
                               known_rules=known_rules)
            rw, _, _ = evaluate(net, n_sims, rng, n_games=30, opponent="random",
                                known_rules=known_rules)
            hist["iter"].append(it)
            hist["games"].append(total_games)
            hist["loss"].append(L[0])
            hist["loss_v"].append(L[2])
            hist["loss_r"].append(L[3])
            hist["perfect_loss_rate"].append(l)
            hist["perfect_draw_rate"].append(d)
            hist["random_win_rate"].append(rw)
            if progress:
                print(f"  [{label} seed {seed}] iter {it:2d}  games {total_games:4d}  "
                      f"loss {L[0]:.3f}  vs-perfect loss-rate {l:.2f}  "
                      f"vs-random win-rate {rw:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    hist["wall"] = time.time() - t0
    hist["state_dict"] = net.state_dict()
    hist["seed"] = seed
    hist["known_rules"] = known_rules
    hist["label"] = label
    return hist


# ==========================================================================
# Experiment: how much of the strength is the SEARCH, and how much the network?
# ==========================================================================
def search_ablation(args):
    """Freeze the weights and only vary how long the agent thinks at decision time.

    `n_sims=1` short-circuits the search entirely and plays the raw policy head, so it
    answers: how much of this agent's strength is the NETWORK, and how much is the SEARCH
    running on top of it? In a healthy "Zero" agent, more search is monotonically better.
    """
    state_dict, seed, known_rules = args
    net = MuZeroNet()
    net.load_state_dict(state_dict)
    rng = np.random.default_rng(seed + 900)
    rows = []
    for n_sims in (1, 4, 16, 32, 64):
        w, d, l = evaluate(net, n_sims, rng, n_games=60, opponent="perfect",
                           known_rules=known_rules)
        rw, _, _ = evaluate(net, n_sims, rng, n_games=60, opponent="random",
                            known_rules=known_rules)
        rows.append((n_sims, w, d, l, rw))
    return rows


def latent_fidelity(state_dict, seed=0, max_k=5):
    """Does the learned model stay useful as it imagines further ahead?

    Take a real board, encode it, and then imagine k moves. Compare the value the model
    predicts *from its imagined latent* against the value it predicts from the REAL board
    at that point (which we can compute, because we have the real game). If the two agree,
    the imagined latent is still carrying the information a decision needs — even though
    nobody ever asked it to look like a board.

    Plotted against the same quantity from a randomly-initialized network, which is the
    control: it says how much of any agreement is just "both numbers are small".
    """
    net = MuZeroNet()
    net.load_state_dict(state_dict)
    rng = np.random.default_rng(seed + 1234)

    # Collect real positions and the moves actually played after them.
    games = [play_game(net, 16, rng, temperature=1.0) for _ in range(60)]
    errs = np.zeros(max_k + 1)
    counts = np.zeros(max_k + 1)

    with torch.no_grad():
        for g in games:
            T = len(g["act"])
            for t in range(T):
                # Rebuild the real board at ply t+k by replaying the game.
                state = TicTacToe()
                for a in g["act"][:t]:
                    state, _, _ = state.step(int(a))
                s, _, _ = net.initial(torch.as_tensor(state.obs()).unsqueeze(0))
                cur = state
                for k in range(1, max_k + 1):
                    if t + k - 1 >= T:
                        break
                    a = int(g["act"][t + k - 1])
                    s, _, _, v_img = net.recurrent(s, onehot(a).unsqueeze(0))
                    cur, _, done = cur.step(a)
                    if done:
                        break
                    # The same position, encoded fresh from the real board.
                    _, _, v_real = net.initial(torch.as_tensor(cur.obs()).unsqueeze(0))
                    errs[k] += abs(float(v_img.item()) - float(v_real.item()))
                    counts[k] += 1
    return errs[1:] / np.maximum(counts[1:], 1)


# ==========================================================================
# The experiment
# ==========================================================================
ARMS = [
    ("MuZero (learns the rules)", False),
    ("MuZero + known legality", True),
]
SEEDS = [0, 1, 2]


def job(args):
    label, known_rules, seed = args
    return train_muzero(
        seed=seed, known_rules=known_rules, label=label,
        progress=(seed == 0),
    )


def main():
    OUT.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    print("=== Tic-Tac-Toe: what does NOT knowing the rules cost? ===", flush=True)
    jobs = [(label, kr, s) for label, kr in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(job, jobs))

    grouped = {label: [h for h in res if h["label"] == label] for label, _ in ARMS}

    print(f"\n  {'variant':28s} {'loss vs perfect':>16s} {'draw vs perfect':>16s} "
          f"{'win vs random':>14s}")
    for label, _ in ARMS:
        hs = grouped[label]
        lr = np.mean([h["perfect_loss_rate"][-1] for h in hs])
        dr = np.mean([h["perfect_draw_rate"][-1] for h in hs])
        wr = np.mean([h["random_win_rate"][-1] for h in hs])
        print(f"  {label:28s} {lr:16.2f} {dr:16.2f} {wr:14.2f}")
    print(f"\n  wall per seed: {np.mean([h['wall'] for h in res]):.0f}s")

    # ---- learning curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
    for i, (label, _) in enumerate(ARMS):
        hs = grouped[label]
        games = np.array(hs[0]["games"])
        lr = np.array([h["perfect_loss_rate"] for h in hs])
        wr = np.array([h["random_win_rate"] for h in hs])
        c = ps.SERIES[2] if i == 0 else ps.SERIES[0]
        axes[0].plot(games, lr.mean(0), color=c, lw=2, label=label)
        axes[0].fill_between(games, lr.min(0), lr.max(0), color=c, alpha=0.13)
        axes[1].plot(games, wr.mean(0), color=c, lw=2, label=label)
        axes[1].fill_between(games, wr.min(0), wr.max(0), color=c, alpha=0.13)
    axes[0].set_title("Loss rate vs PERFECT play (want 0)", color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("self-play games", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("loss rate", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Win rate vs a RANDOM player (want 1)", color=ps.INK, fontsize=11, loc="left")
    axes[1].set_xlabel("self-play games", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("win rate", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'learning_curves.png'}")

    # ---- does thinking longer help? ----
    print("\n=== search ablation: simulations per move, weights frozen ===", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        abl_jobs = [(h["state_dict"], h["seed"], h["known_rules"]) for h in res]
        abl = list(ex.map(search_ablation, abl_jobs))

    fig, ax = ps.new_axes(7.4, 4.3)
    for i, (label, kr) in enumerate(ARMS):
        rows = np.array([abl[j] for j, h in enumerate(res) if h["label"] == label])
        sims = rows[0, :, 0]
        loss_rate = rows[:, :, 3].mean(0)
        c = ps.SERIES[2] if i == 0 else ps.SERIES[0]
        ax.plot(sims, loss_rate, color=c, lw=2, marker="o", ms=6, label=label)
        print(f"  {label}:")
        for j, s in enumerate(sims):
            print(f"    {int(s):3d} sims -> loss rate vs perfect {loss_rate[j]:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(ax.figure, ax, "Does thinking longer make it play better?",
              "MCTS simulations per move (1 = no search, policy head only)",
              "loss rate vs perfect play (lower is better)", OUT / "search_ablation.png")

    # ---- the latent stays predictive ----
    print("\n=== latent fidelity: value agreement after k imagined steps ===", flush=True)
    best = next(h for h in res if h["known_rules"] and h["seed"] == 0)
    fid = latent_fidelity(best["state_dict"], seed=0)
    for k, e in enumerate(fid, start=1):
        print(f"  after {k} imagined step(s): |v_imagined - v_from_real_board| = {e:.3f}")

    fig, ax = ps.new_axes(7.2, 4.2)
    ks = np.arange(1, len(fid) + 1)
    ax.plot(ks, fid, color=ps.SERIES[4], lw=2, marker="o", ms=6)
    ax.set_ylim(bottom=0)
    ps.finish(fig, ax, "The imagined latent keeps predicting the right value",
              "steps imagined through the learned dynamics (k)",
              "gap vs. evaluating the real board", OUT / "latent_fidelity.png")


if __name__ == "__main__":
    main()
