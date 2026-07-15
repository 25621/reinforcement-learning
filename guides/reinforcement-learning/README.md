# Reinforcement Learning: From Beginner to Advanced

A comprehensive guide to reinforcement learning as a *subject* — not just as the toolbox people pull `stable-baselines` from. The goal is to take you from "I have heard of Q-learning" to "I can read a 2026 RL paper, understand why the authors made every design choice, and reproduce its core algorithm." The guide is biased toward what matters in practice now: the things that ship.

> **An honest framing.** Classical RL — the part of the field taught in textbooks — is a beautiful theory built on Markov decision processes, dynamic programming, and stochastic approximation. Most of it does not work directly on the problems people actually care about in 2026. What *does* work is a handful of robust algorithms (PPO, SAC, GRPO, MuZero-style search, DPO) applied with a mountain of practical knowledge about reward shaping, exploration, normalization, and infrastructure. This guide teaches the theory because you cannot debug what you do not understand, and then teaches the practice that the theory does not cover.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: MDPs and the Bellman Equations](#phase-1-mdps-and-the-bellman-equations)
3. [Phase 2: Tabular Methods — DP, Monte Carlo, and TD](#phase-2-tabular-methods--dp-monte-carlo-and-td)
4. [Phase 3: Function Approximation and DQN](#phase-3-function-approximation-and-dqn)
5. [Phase 4: Policy Gradients — REINFORCE to PPO](#phase-4-policy-gradients--reinforce-to-ppo)
6. [Phase 5: Continuous Control — DDPG, TD3, SAC](#phase-5-continuous-control--ddpg-td3-sac)
7. [Phase 6: Model-Based RL](#phase-6-model-based-rl)
8. [Phase 7: Offline RL](#phase-7-offline-rl)
9. [Phase 8: Exploration](#phase-8-exploration)
10. [Phase 9: RL for Language Models — RLHF, DPO, GRPO, RLVR](#phase-9-rl-for-language-models--rlhf-dpo-grpo-rlvr)
11. [Phase 10: Frontier Topics](#phase-10-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls](#common-pitfalls)
15. [Additional Resources](#additional-resources)
16. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

RL combines probability, optimization, and deep learning under a deceptively simple wrapper. You can get started without mastering everything below, but if more than a couple are unfamiliar, slow down before continuing.

### Concepts to Know

- **Probability**: random variables, expectation, conditional expectation, law of total expectation, basic Markov chains
- **Linear algebra**: matrix multiplication, eigenvalues (for contraction arguments), vector norms
- **Calculus**: gradients, chain rule, the log-derivative trick (`∇ log p = ∇p / p`) — you'll use this constantly
- **Optimization**: SGD, Adam, learning rates, the difference between an objective you maximize vs. minimize
- **Deep learning**: training loops, `nn.Module`, basic CNNs and transformers. If shaky, do the [PyTorch Deep Dive](../pytorch-deep-dive/) first
- **Some programming maturity**: you will debug stochastic, asynchronous code where the bug only appears every 50k steps

### The One Equation Everything Comes Back To

```
              ┌─────────────────────────────────────────┐
              │  V(s) = E[ Σ γᵏ r_{t+k} | s_t = s ]    │
              └─────────────────────────────────────────┘

The value of being in state s is the expected sum of (discounted) future rewards
if you act according to your policy from s onward.

Everything in RL — every algorithm, every loss function, every trick —
is some clever way of estimating this expectation when you don't know
the dynamics, the reward function, or even what "state" means.
```

If that sentence is fuzzy now, it will be sharp by the end of Phase 2.

### What You Need Installed

- **Python 3.10+**, PyTorch, NumPy
- **Gymnasium** (the maintained fork of OpenAI Gym) — the de facto environment API
- **Stable-Baselines3** — well-tested reference implementations; read its source
- **CleanRL** — single-file implementations of every major algorithm; the best teaching resource you can pip install
- **MuJoCo** (via `gymnasium[mujoco]`) — for continuous control benchmarks
- **A GPU** — not strictly required for tabular work, but essential by Phase 4

### Resources

- [Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) — the bible, free online. Read it cover to cover. Yes, really.
- [David Silver's RL course (DeepMind/UCL, 2015)](https://www.davidsilver.uk/teaching/) — still the best lecture series for the fundamentals
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) — the cleanest practical on-ramp
- [CleanRL docs and code](https://docs.cleanrl.dev/) — every algorithm in one file

---

## Phase 1: MDPs and the Bellman Equations

The single most important phase. Everything else is variations on the math you learn here.

### Concepts to Learn

- **Markov Decision Process (MDP)**: the tuple `(S, A, P, R, γ)` — states, actions, transition probabilities, reward function, discount factor
- **The Markov property**: the future depends only on the present state, not the past. When this is violated (partial observability), you have a POMDP and life becomes harder
- **Policy** `π(a | s)`: a distribution over actions given a state. Deterministic policies are a special case
- **Return** `G_t = Σ γᵏ r_{t+k}`: the discounted sum of future rewards from step `t`
- **Value functions**:
  - State value `V^π(s) = E_π[G_t | s_t = s]`
  - Action value `Q^π(s, a) = E_π[G_t | s_t = s, a_t = a]`
  - Advantage `A^π(s, a) = Q^π(s, a) − V^π(s)` — "how much better than average is this action?"
- **The Bellman equations** for `V^π` and `Q^π` — recursive consistency conditions
- **The Bellman optimality equations** for `V*` and `Q*` — the same thing but for the best policy
- **Discount factor `γ`**: why `γ < 1` makes the math tractable (geometric series convergence) and what it means in practice (effective horizon ≈ `1/(1−γ)`)
- **Episodic vs continuing tasks**, **finite vs infinite horizon**

### The Two Bellman Equations You Must Know Cold

```
For a fixed policy π:
   V^π(s) = E_a~π(·|s) [ R(s, a) + γ · E_s'~P(·|s,a) [ V^π(s') ] ]

For the optimal policy:
   V*(s) = max_a [ R(s, a) + γ · E_s'~P(·|s,a) [ V*(s') ] ]

In words:
   "The value of a state is the expected reward right now plus the
    discounted value of where you end up."

The max in the optimal version is what makes RL non-trivial.
Without it (policy fixed), value estimation is a linear system.
With it, you're solving a fixed-point problem on a nonlinear operator.
```

### The Geometric Intuition

The Bellman operator is a **contraction** in the supremum norm: applying it repeatedly to any starting `V` converges to `V*` at rate `γ`. This single fact justifies every iterative algorithm in RL. When your training "isn't converging," it's almost always because something has broken contraction — function approximation, off-policy data, or both.

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Build a gridworld](projects/01-build-a-gridworld/README.md) | 5×5 grid with rewards and obstacles; expose `(S, A, P, R, γ)` explicitly | ⭐ |
| [Policy evaluation by matrix inverse](projects/02-policy-evaluation-by-matrix-inverse/README.md) | For a small MDP, solve `V^π = (I − γP^π)⁻¹ r^π` directly; verify against iterative evaluation | ⭐⭐ |
| [Hand-trace Bellman backups](projects/03-hand-trace-bellman-backups/README.md) | On a 3-state MDP, do 10 Bellman backups by hand and plot `V` over iterations | ⭐⭐ |
| [Discount factor study](projects/04-discount-factor-study/README.md) | Same task, sweep `γ ∈ {0.5, 0.9, 0.99, 0.999}`; observe how the optimal policy changes | ⭐⭐ |
| [POMDP exercise](projects/05-pomdp-exercise/README.md) | Build a gridworld where the agent only sees its row, not its column; verify the optimal *Markov* policy is suboptimal | ⭐⭐⭐ |

### Sample Code: Policy Evaluation on a Small MDP

```python
import numpy as np

# 3-state MDP, 2 actions
n_states, n_actions = 3, 2
P = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))  # (S, A, S')
R = np.random.randn(n_states, n_actions)                                # (S, A)
gamma = 0.9

# A uniform random policy
pi = np.full((n_states, n_actions), 1.0 / n_actions)                    # (S, A)

# Iterative policy evaluation
V = np.zeros(n_states)
for _ in range(1000):
    V_new = np.zeros(n_states)
    for s in range(n_states):
        for a in range(n_actions):
            V_new[s] += pi[s, a] * (R[s, a] + gamma * P[s, a] @ V)
    if np.max(np.abs(V_new - V)) < 1e-9:
        break
    V = V_new
print("V^pi =", V)

# Verify via the closed-form solution V = (I - gamma P^pi)^-1 r^pi:
P_pi = np.einsum("sa,sat->st", pi, P)
r_pi = np.einsum("sa,sa->s",   pi, R)
V_closed = np.linalg.solve(np.eye(n_states) - gamma * P_pi, r_pi)
np.testing.assert_allclose(V, V_closed, atol=1e-6)
```

### Key Insight

The Bellman equation is just a consistency check: "what I think about now must agree with what I think about next, plus the reward I get in between." Every RL algorithm is some way of *enforcing* this consistency when you can't compute the right-hand side exactly. Q-learning, DQN, TD(λ), advantage actor-critic — they differ in *which* version of the Bellman equation they're trying to satisfy and *which* approximations they make on the way.

### Resources

- [Sutton & Barto, Chapters 3–4](http://incompleteideas.net/book/RLbook2020.pdf) — MDPs and DP
- [David Silver Lectures 2–3](https://www.davidsilver.uk/teaching/) — same material, video form
- [Csaba Szepesvári, *Algorithms for Reinforcement Learning* (free PDF)](https://sites.ualberta.ca/~szepesva/rlbook.html) — for the theory-minded

---

## Phase 2: Tabular Methods — DP, Monte Carlo, and TD

Tabular methods assume you can keep one number per state (or state-action pair) in a table. This is a toy assumption — but every modern algorithm is the tabular version with neural-network-shaped scaffolding. Master the tabular version and the neural version becomes a footnote.

### Concepts to Learn

- **Dynamic programming** (requires knowing `P` and `R`):
  - **Policy iteration** — alternate policy evaluation and policy improvement
  - **Value iteration** — Bellman optimality backup until convergence; extract the greedy policy at the end
  - **Generalized policy iteration (GPI)** — the unifying picture
- **Monte Carlo (MC) methods** (don't need `P` or `R`, only sampled returns):
  - First-visit vs every-visit MC
  - On-policy MC control with ε-greedy exploration
- **Temporal-difference (TD) learning** — the central idea of RL:
  - TD(0): `V(s) ← V(s) + α(r + γ V(s') − V(s))`
  - SARSA (on-policy): `Q(s,a) ← Q(s,a) + α(r + γ Q(s',a') − Q(s,a))`
  - Q-learning (off-policy): `Q(s,a) ← Q(s,a) + α(r + γ max_{a'} Q(s',a') − Q(s,a))`
- **Bias vs variance**: MC is unbiased high-variance, TD is biased low-variance
- **n-step methods and TD(λ)** — eligibility traces, the unifying parameter
- **Exploration vs exploitation**: ε-greedy, Boltzmann/softmax, UCB
- **The deadly triad**: function approximation + bootstrapping + off-policy training — the combination that breaks convergence guarantees

### The Algorithm Family Tree (Tabular)

```
                     Know dynamics P, R?
                          │
                ┌─────────┴────────┐
               yes                 no
                │                   │
            DP methods       Sample-based methods
            (Phase 2a)          /         \
                              MC            TD
                          (full return)  (bootstrap)
                                          /    \
                                       SARSA   Q-learning
                                      (on-pol) (off-pol)
                                          \    /
                                       n-step / TD(λ)
                                        unifies both
```

### Why TD is the Idea

```
MC update target:    G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...     (a whole trajectory)
TD(0) update target: r_t + γ V(s_{t+1})                            (one step + bootstrap)

MC is unbiased (the target IS the true return),
but high-variance (depends on every future step).

TD is biased (V(s_{t+1}) is an estimate, not the truth),
but low-variance (only one transition's randomness).

For most problems, TD's low variance dominates MC's bias.
This is why every modern algorithm uses bootstrapped targets.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Value iteration on FrozenLake](projects/06-value-iteration-on-frozenlake/README.md) | Solve the Gym/Gymnasium FrozenLake-v1 with value iteration; visualize `V*` and the greedy policy | ⭐⭐ |
| [Policy iteration vs value iteration](projects/07-policy-iteration-vs-value-iteration/README.md) | Same problem, both algorithms; count iterations to convergence | ⭐⭐ |
| [First-visit MC](projects/08-first-visit-mc/README.md) | On Blackjack-v1, learn `V^π` for a fixed policy by MC; verify against an analytic solution if possible | ⭐⭐ |
| [Q-learning on FrozenLake](projects/09-q-learning-on-frozenlake/README.md) | Tabular Q-learning, ε-greedy, decaying ε; report final policy success rate | ⭐⭐ |
| [SARSA vs Q-learning on Cliff Walking](projects/10-sarsa-vs-q-learning-on-cliff-walking/README.md) | Reproduce Sutton & Barto Fig 6.5; explain why SARSA prefers the safe path | ⭐⭐⭐ |
| [Eligibility traces](projects/11-eligibility-traces/README.md) | Implement TD(λ) with replacing traces; sweep `λ ∈ {0, 0.5, 0.9, 1.0}` | ⭐⭐⭐ |

### Sample Code: Tabular Q-Learning

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma = 0.1, 0.99

for ep in range(20000):
    s, _ = env.reset()
    eps = max(0.01, 1.0 - ep / 10000)             # decaying exploration
    done = False
    while not done:
        a = env.action_space.sample() if np.random.rand() < eps else int(Q[s].argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        target = r + gamma * Q[s2].max() * (not term)
        Q[s, a] += alpha * (target - Q[s, a])     # the Bellman update, one row at a time
        s = s2

print("greedy policy =", Q.argmax(axis=1))
```

### Key Insight

The single sentence: **TD learning is gradient descent on the Bellman error, with the gradient through the target term stopped.** That stopped gradient — the fact that `V(s')` in the target is treated as a constant rather than as a function of the same parameters — is what makes TD *work* and what makes it *fragile under function approximation*. Phase 3 is mostly about managing the consequences.

### Resources

- [Sutton & Barto, Chapters 4–7](http://incompleteideas.net/book/RLbook2020.pdf) — DP, MC, TD, n-step
- [David Silver Lectures 3–5](https://www.davidsilver.uk/teaching/)
- [Spinning Up — Key Equations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

---

## Phase 3: Function Approximation and DQN

Tabular methods need one entry per state. Real problems have continuous or astronomically large state spaces — Atari has 256^(84×84) possible screens. The fix is function approximation: replace the table with a neural net. This is where everything gets harder.

### Concepts to Learn

- **Linear function approximation** — features × weights; convergence guarantees mostly survive
- **Nonlinear (neural) function approximation** — convergence guarantees mostly evaporate; empirical care required
- **The deadly triad in detail** — and the tricks that tame it
- **DQN (Deep Q-Network)** — the canonical recipe that made deep RL work on Atari:
  - **Experience replay buffer** — break correlation between consecutive samples; reuse data
  - **Target network** — a periodically-updated copy of the Q-network used in the bootstrap target; prevents instability
  - **ε-greedy with annealed ε**, frame stacking, reward clipping
- **DQN family improvements** (all worth knowing, all small wins individually, large in aggregate):
  - **Double DQN** — decouple action selection and evaluation; reduces overestimation
  - **Dueling DQN** — factor `Q(s, a) = V(s) + A(s, a)`; better value estimation
  - **Prioritized Experience Replay (PER)** — sample transitions by TD-error magnitude
  - **n-step returns** — `r_t + γ r_{t+1} + ... + γⁿ max_a Q(s_{t+n}, a)`
  - **Noisy nets** — parametric exploration noise
  - **Distributional RL (C51, QR-DQN, IQN)** — predict the *distribution* of returns, not just the mean
  - **Rainbow** — all of the above combined
- **When DQN-family algorithms are the right tool**: discrete actions, off-policy data reuse, sample-efficiency matters

### The DQN Loss, Annotated

```python
loss = ((Q(s, a) - target)**2).mean()

           where target = r + γ · max_{a'} Q_target(s', a') · (not done)
                                 │              │
                                 │              └─ frozen target network
                                 └─ greedy action selection
                                    (Double DQN: use online net here, target net to evaluate)

  Crucial implementation details:
   - target = target.detach()           ← no gradients through the target
   - sample (s, a, r, s', d) from a big replay buffer
   - update Q_target ← Q every N steps (or use Polyak averaging)
   - clip rewards to [-1, 1] for Atari
   - Huber loss instead of MSE for robustness to outliers
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [DQN on CartPole](projects/12-dqn-on-cartpole/README.md) | Single-file DQN with no replay buffer and no target network — watch the deadly triad drive the Q-values 10⁸× past what the game can pay | ⭐⭐⭐ |
| [Add a replay buffer](projects/13-add-a-replay-buffer/README.md) | Now add experience replay and a target network; ablate the 2×2 and verify stability | ⭐⭐⭐ |
| [Atari Pong](projects/14-atari-pong/README.md) | Full DQN from pixels with frame stacking and reward clipping — the real ALE preprocessing chain, trained on a Pong that fits in ten minutes | ⭐⭐⭐⭐ |
| [Double + Dueling](projects/15-double-dueling/README.md) | Add both to your DQN; ablate each and measure the overestimation gap they close | ⭐⭐⭐⭐ |
| [Prioritized replay](projects/16-prioritized-replay/README.md) | Implement PER with a sum-tree; verify the priorities improve sample efficiency | ⭐⭐⭐⭐ |
| [Mini Rainbow](projects/17-mini-rainbow/README.md) | Combine Double + Dueling + PER + n-step; reproduce a ~Rainbow-lite ablation | ⭐⭐⭐⭐⭐ |
| [Distributional DQN (C51)](projects/18-distributional-dqn-c51/README.md) | Predict a categorical distribution over returns; verify on a small env | ⭐⭐⭐⭐⭐ |

### Sample Code: A Minimal Working DQN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)

q       = QNet(4, 2)
q_targ  = QNet(4, 2); q_targ.load_state_dict(q.state_dict())
optim   = torch.optim.Adam(q.parameters(), lr=2.5e-4)
buf     = deque(maxlen=50_000)
gamma   = 0.99

def update():
    batch = random.sample(buf, 64)
    s, a, r, s2, d = map(torch.as_tensor, zip(*batch))
    with torch.no_grad():
        # Standard DQN target. For Double DQN:
        #   a_star = q(s2).argmax(-1); target_q = q_targ(s2).gather(-1, a_star[:, None])
        target = r + gamma * q_targ(s2.float()).max(-1).values * (1 - d.float())
    pred = q(s.float()).gather(-1, a[:, None].long()).squeeze(-1)
    loss = F.smooth_l1_loss(pred, target)              # Huber
    optim.zero_grad(); loss.backward(); optim.step()
```

### Key Insight

DQN's three innovations — replay buffer, target network, frame stacking + clipping — were each a fix for a specific instability the underlying TD update has under nonlinear function approximation. The deadly triad doesn't *go away*; it gets *managed*. Every "modern" RL algorithm makes a slightly different bet about how to manage it. SAC bets on entropy regularization. PPO bets on small policy steps. Offline RL bets on staying near the data. None of them are mathematically clean. All of them work empirically because the bets are well-chosen.

### Resources

- [Mnih et al. — *Human-level control through deep reinforcement learning* (Nature, 2015)](https://www.nature.com/articles/nature14236) — the original DQN paper
- [Rainbow paper (Hessel et al., 2017)](https://arxiv.org/abs/1710.02298) — the survey of improvements
- [CleanRL's `dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) — single-file reference
- [Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — same energy as DQN's implementation details; required reading

---

## Phase 4: Policy Gradients — REINFORCE to PPO

Value-based methods estimate `Q` and act greedily. Policy-based methods skip the value function and *directly* parameterize `π_θ(a | s)`. Policy gradients are the foundation of every modern algorithm that ships in production — PPO, GRPO, the entire RLHF stack.

### Concepts to Learn

- **The policy gradient theorem**: `∇_θ J(θ) = E_π[ ∇_θ log π_θ(a | s) · Q^π(s, a) ]`
- **The log-derivative (REINFORCE) trick** — why the policy gradient is a pure expectation; why this matters
- **REINFORCE** — vanilla Monte-Carlo policy gradient with returns `G_t` as the weight
- **Baselines** — subtract a state-dependent baseline `b(s)` (typically `V(s)`) from the return; reduces variance without bias
- **Advantage actor-critic** — use a learned `V` as baseline; the weight becomes `A_t = G_t − V(s_t)` or its bootstrapped version
- **Generalized Advantage Estimation (GAE)** — interpolates between high-bias-low-variance TD and low-bias-high-variance MC via the `λ` parameter
- **A2C / A3C** — synchronous and asynchronous advantage actor-critic
- **Trust regions**:
  - **TRPO** — constrain the KL divergence between old and new policy; works but is awkward to implement
  - **PPO** — clip the importance ratio instead; same spirit, far simpler. *The default on-policy algorithm in 2026*
- **On-policy vs off-policy**: PPO is on-policy, so every gradient step must use fresh data. This is why PPO needs vast amounts of data and parallel envs
- **Entropy regularization** — add `β · H(π)` to the objective to keep the policy from collapsing to a single action prematurely

### The Five Lines That Are PPO

```python
ratio = (new_logp - old_logp).exp()                  # π_new / π_old
clip_ratio = ratio.clamp(1 - eps, 1 + eps)
policy_loss = -torch.min(ratio * adv, clip_ratio * adv).mean()
value_loss  = (V_pred - returns).pow(2).mean()
loss = policy_loss + c1 * value_loss - c2 * entropy
```

That's it. Every other line in a PPO implementation is data wrangling, normalization, or logging.

### Generalized Advantage Estimation, Visualized

```
TD(0) advantage:    A_t = r_t + γ V(s_{t+1}) - V(s_t)       (1-step; low var, high bias)
MC  advantage:      A_t = G_t - V(s_t)                       (full return; high var, low bias)

GAE(γ, λ):
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)                        (the 1-step TD residual)
    A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

    λ = 0  → TD(0) advantage
    λ = 1  → MC advantage (minus value baseline)
    λ ≈ 0.95–0.97 → the sweet spot used in nearly every paper
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [REINFORCE on CartPole](projects/19-reinforce-on-cartpole/README.md) | Vanilla policy gradient, no baseline; observe the variance | ⭐⭐ |
| [Add a value baseline](projects/20-add-a-value-baseline/README.md) | Same task, subtract a learned `V(s)`; verify variance drops | ⭐⭐⭐ |
| [A2C with parallel envs](projects/21-a2c-with-parallel-envs/README.md) | 8 parallel envs, n-step returns, GAE; solve LunarLander | ⭐⭐⭐⭐ |
| [PPO from scratch](projects/22-ppo-from-scratch/README.md) | Reproduce CleanRL's `ppo.py` line by line; explain each detail | ⭐⭐⭐⭐ |
| [The 37 details](projects/23-the-37-details/README.md) | Implement (or audit) every one of the [37 PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); measure each ablation | ⭐⭐⭐⭐⭐ |
| [PPO on Atari](projects/24-ppo-on-atari/README.md) | Apply your PPO to a few Atari games; compare against published numbers | ⭐⭐⭐⭐⭐ |
| [TRPO for comparison](projects/25-trpo-for-comparison/README.md) | Implement TRPO; compare on a Mujoco task; see why nobody uses it anymore | ⭐⭐⭐⭐⭐ |

### Sample Code: The PPO Update Loop (Sketch)

```python
# After collecting `rollout_len` steps from `n_envs` parallel environments...
# obs, actions, log_probs_old, advantages, returns are all shape (rollout_len * n_envs, ...)

for epoch in range(n_epochs):
    for batch in minibatches(data, size=64):
        new_logp, entropy, V_pred = policy.evaluate(batch.obs, batch.actions)
        ratio = (new_logp - batch.log_probs_old).exp()

        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)            # the normalization that always matters

        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * ratio.clamp(1 - eps, 1 + eps)
        pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

        v_loss   = (V_pred - batch.returns).pow(2).mean()
        ent_loss = -entropy.mean()

        loss = pg_loss + 0.5 * v_loss + 0.01 * ent_loss
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)        # the grad clip that always matters
        optim.step()
```

### Key Insight

PPO is famous less for its math than for its **robustness to implementation mistakes**. It survives wrong learning rates, missing normalizations, and bad reward shaping in a way that TRPO, vanilla policy gradients, and most off-policy algorithms do not. This is why it's the default starting point for almost every new RL project — not because it's the best (it usually isn't), but because it works first try while you're still wrong about everything else.

### Resources

- [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/RLbook2020.pdf) — policy gradient methods
- [Schulman et al. — PPO paper (2017)](https://arxiv.org/abs/1707.06347)
- [Schulman et al. — GAE paper (2015)](https://arxiv.org/abs/1506.02438)
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — the most useful single document in applied RL
- [CleanRL's `ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) — one file, fully annotated

---

## Phase 5: Continuous Control — DDPG, TD3, SAC

When actions are continuous (joint torques, steering angles, end-effector deltas), max-over-actions in Q-learning becomes impossible. The fix is a family of *deterministic policy gradient* and *soft actor-critic* methods that learn a continuous-action policy alongside a critic.

### Concepts to Learn

- **Why discrete-action methods don't transfer directly** — `max_a Q(s, a)` over continuous `a` is an optimization, not a lookup
- **The deterministic policy gradient theorem**: `∇_θ J = E_s[ ∇_a Q(s, a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s) ]` — chain rule through the actor
- **DDPG (Deep Deterministic Policy Gradient)**:
  - Actor `μ_θ(s)` and critic `Q_φ(s, a)`
  - Off-policy, replay buffer, target networks (one for actor, one for critic)
  - Exploration via additive noise on the action (Ornstein-Uhlenbeck or Gaussian)
- **TD3 (Twin Delayed DDPG)** — DDPG with three fixes:
  - **Twin critics**: take the min of two Q-networks → reduces overestimation
  - **Delayed actor updates**: critic moves faster than actor → stability
  - **Target policy smoothing**: add noise to the target action → regularization
- **SAC (Soft Actor-Critic)** — the modern default for continuous control:
  - **Maximum entropy RL**: maximize `E[Σ r_t + α H(π(·|s_t))]`
  - **Stochastic policy** with learned mean and std; reparameterization trick for sampling
  - **Automatic temperature tuning** (`α`) to hit a target entropy
- **When to use which**: SAC is the default. TD3 is competitive on deterministic tasks. DDPG is a pedagogical stepping stone — don't ship it
- **Action saturation, tanh squashing, and the log-prob correction** — small implementation details that matter a lot

### The SAC Mental Model

```
Standard RL objective:                  J = E[ Σ γᵗ r_t ]
Maximum entropy RL objective:           J = E[ Σ γᵗ ( r_t + α H(π(·|s_t)) ) ]

The entropy bonus encourages:
  - Exploration (the policy stays stochastic)
  - Robust policies (no commitment to a fragile near-deterministic action)
  - Better critic learning (the Q-function sees a richer action distribution)

The temperature α controls how much you weight entropy.
SAC tunes α automatically by gradient descent on the dual problem.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [DDPG on Pendulum](projects/26-ddpg-on-pendulum/README.md) | Single-file DDPG; verify it learns; observe the instability | ⭐⭐⭐ |
| [TD3 on HalfCheetah](projects/27-td3-on-halfcheetah/README.md) | Full TD3 implementation with twin critics; compare to DDPG | ⭐⭐⭐⭐ |
| [SAC on a Mujoco suite](projects/28-sac-on-a-mujoco-suite/README.md) | Implement SAC; run on HalfCheetah, Walker2d, Ant, Humanoid; report final returns | ⭐⭐⭐⭐ |
| [Automatic temperature tuning](projects/29-automatic-temperature-tuning/README.md) | Add the auto-α update to SAC; verify it stabilizes entropy across tasks | ⭐⭐⭐⭐ |
| [Reparameterization audit](projects/30-reparameterization-audit/README.md) | Verify your `tanh`-squashed Gaussian's log-prob correction is right; off-by-one here silently breaks SAC | ⭐⭐⭐ |
| [Sample efficiency study](projects/31-sample-efficiency-study/README.md) | Compare PPO vs SAC on the same Mujoco task in wall-clock time and samples used | ⭐⭐⭐⭐ |

### Sample Code: The SAC Critic Update

```python
# Sampled (s, a, r, s2, d) batch
with torch.no_grad():
    a2, logp2 = actor.sample(s2)                      # reparameterized + log-prob
    q1_t = q1_targ(s2, a2)
    q2_t = q2_targ(s2, a2)
    q_t  = torch.min(q1_t, q2_t) - alpha * logp2      # entropy bonus baked into target
    target = r + gamma * (1 - d) * q_t

q1_loss = F.mse_loss(q1(s, a), target)
q2_loss = F.mse_loss(q2(s, a), target)
critic_loss = q1_loss + q2_loss
optim_critic.zero_grad(); critic_loss.backward(); optim_critic.step()
```

### Key Insight

SAC's "entropy bonus" looks like a small modification, but it's the deepest practical difference between modern continuous-control RL and the 2014–2016 generation. Maximum-entropy policies are robust to environment perturbations, sample-efficient because the critic learns from a wide action distribution, and naturally explore without hand-tuned noise. Almost everything that worked on humanoid locomotion, dexterous manipulation, and learned legged control between 2018 and 2023 used SAC or one of its near-cousins.

### Resources

- [Lillicrap et al. — DDPG (2015)](https://arxiv.org/abs/1509.02971)
- [Fujimoto et al. — TD3 (2018)](https://arxiv.org/abs/1802.09477)
- [Haarnoja et al. — SAC (2018)](https://arxiv.org/abs/1801.01290) — paper 1 of 2
- [Haarnoja et al. — SAC v2 (2018)](https://arxiv.org/abs/1812.05905) — auto-tuned temperature
- [Spinning Up — SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [CleanRL's `sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

---

## Phase 6: Model-Based RL

Everything so far has been **model-free**: learn a policy or value function from samples, never explicitly modeling the environment. **Model-based RL (MBRL)** learns a model of the dynamics `P(s' | s, a)` and the reward `R(s, a)`, then uses that model to plan, generate synthetic data, or both. The payoff is sample efficiency — orders of magnitude in some cases — and the cost is the new burden of model error.

### Concepts to Learn

- **What "the model" can be**:
  - **A learned forward dynamics network** `f_θ(s, a) → s', r`
  - **An ensemble** of forward dynamics networks (the standard way to estimate uncertainty)
  - **A latent dynamics model**: encoder `s → z`, transition `z, a → z'`, decoder `z → o` (image obs); the **Dreamer** family
  - **A generative video model** doing the same job (world models — see [Phase 9 of the Video Generation Guide](../video-generation/#phase-9-world-models-and-interactive-video))
- **Three ways to use a model**:
  - **Dyna-style** — generate fake transitions to augment a model-free learner's replay buffer (MBPO is the modern version)
  - **Planning** — use the model directly at decision time (MPC, CEM, MPPI)
  - **Search + learning** — combine learned value + learned policy + model-based search (MuZero, AlphaZero, EfficientZero)
- **Model error and pessimism** — short rollouts vs long rollouts; how error compounds
- **MuZero** — the algorithm that learned chess, Go, shogi, and Atari from scratch by combining MCTS with a learned model that doesn't even reconstruct observations
- **Dreamer (V1/V2/V3)** — image-based world models that learn behaviors by imagining rollouts in latent space; V3 is shockingly general across hundreds of tasks with one hyperparameter setting
- **TD-MPC / TD-MPC2** — short-horizon learned-model planning combined with a learned value to bootstrap from the planning horizon
- **The model-based / model-free divide in 2026**: model-based wins on sample efficiency and few-shot adaptation; model-free still wins on raw asymptotic performance in many settings. The line is blurring fast

### The Three Uses of a Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. AUGMENT DATA (Dyna, MBPO)                                        │
│    Train a model. Generate fake (s, a, r, s') tuples. Train SAC/PPO │
│    on real + fake data. Roll the model only k=1–5 steps to limit    │
│    compounding error.                                               │
├─────────────────────────────────────────────────────────────────────┤
│ 2. PLAN AT DECISION TIME (MPC, CEM, MPPI)                           │
│    Don't learn a policy. At each step, sample many action sequences,│
│    roll them out through the model, pick the best one, execute the  │
│    first action, replan. Model is everything; policy is implicit.   │
├─────────────────────────────────────────────────────────────────────┤
│ 3. SEARCH + LEARN (MuZero, EfficientZero, TD-MPC2)                  │
│    Learn a model, value, and policy jointly. Use Monte Carlo Tree   │
│    Search (MuZero) or short-horizon shooting (TD-MPC2) at decision  │
│    time, bootstrapping from the learned value beyond the search     │
│    horizon. The current frontier of model-based RL.                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [PETS / random shooting MPC](projects/32-pets-random-shooting-mpc/README.md) | Learn a 1-step dynamics model on Pendulum; do random-shooting MPC; compare to SAC | ⭐⭐⭐ |
| [CEM-MPC](projects/33-cem-mpc/README.md) | Replace random shooting with Cross-Entropy Method action search | ⭐⭐⭐⭐ |
| [Mini MBPO](projects/34-mini-mbpo/README.md) | Train SAC with short model rollouts mixed into the replay buffer; verify the sample-efficiency win | ⭐⭐⭐⭐ |
| [Dreamer V3 reproduction](projects/35-dreamer-v3-reproduction/README.md) | Port the official Dreamer V3 to a custom env; observe how few hyperparameters it needs | ⭐⭐⭐⭐⭐ |
| [Mini MuZero](projects/36-mini-muzero/README.md) | Implement MuZero on a small game (Tic-Tac-Toe or 4x4 Connect4); see the recurrence between policy, value, and dynamics heads | ⭐⭐⭐⭐⭐ |
| [TD-MPC2 study](projects/37-td-mpc2-study/README.md) | Read TD-MPC2 paper; reproduce its DMC suite results | ⭐⭐⭐⭐⭐ |

### Sample Code: Random-Shooting MPC

```python
def mpc_action(model, s, horizon=20, n_samples=1024, action_dim=1, action_lo=-1, action_hi=1):
    """Sample many action sequences, roll them out through the model, take the best first action."""
    actions = torch.empty(n_samples, horizon, action_dim).uniform_(action_lo, action_hi)
    s_curr = s.expand(n_samples, -1).clone()
    returns = torch.zeros(n_samples)
    for t in range(horizon):
        s_next, r = model(s_curr, actions[:, t])
        returns += (0.99 ** t) * r
        s_curr = s_next
    best = returns.argmax()
    return actions[best, 0]                       # execute only the first action, replan next step
```

### Key Insight

Model-based RL has been "one breakthrough away from taking over" for a decade. What changed in 2023–2025 is that the model side and the policy side started to share representations — Dreamer's latent dynamics, MuZero's abstract state, TD-MPC2's encoder — so the model only needs to be accurate where it matters for the policy. This is the same trick that made transformers work: don't try to model everything, model what you'll be queried on. Expect MBRL to be the default by 2027–2028, especially on data-scarce embodied tasks where every real sample is expensive.

### Resources

- [Sutton & Barto, Chapter 8](http://incompleteideas.net/book/RLbook2020.pdf) — planning and learning
- [Schrittwieser et al. — MuZero (Nature, 2020)](https://www.nature.com/articles/s41586-020-03051-4)
- [Hafner et al. — DreamerV3 (2023)](https://arxiv.org/abs/2301.04104)
- [Hansen et al. — TD-MPC2 (2024)](https://arxiv.org/abs/2310.16828)
- [Janner et al. — MBPO (2019)](https://arxiv.org/abs/1906.08253)

---

## Phase 7: Offline RL

In **[offline RL](/shared/glossary/#offline-rl)** (also called batch RL), you have a fixed dataset of past interactions and cannot collect more. This is the regime that matches most real applications: medical records, recommender system logs, robot-fleet data, historical trading data. The challenge is **[distribution shift](/shared/glossary/#distribution-shift)**: the learned policy will want to take actions that aren't in the dataset, and the [Q-function](/shared/glossary/#q-learning) will hallucinate huge values for those [out-of-distribution](/shared/glossary/#out-of-distribution) actions.

### Concepts to Learn

- **Why "just run Q-learning on the data" fails** — the bootstrapped target uses `max_a Q(s', a)`, which is maximized at unseen actions where `Q` is wildly wrong
- **The two families of fixes**:
  - **[Policy constraint](/shared/glossary/#policy-constraint)**: stay close to the [behavior policy](/shared/glossary/#behavior-policy) that generated the data ([BCQ, BEAR, AWAC, BRAC](/shared/glossary/#policy-constraint))
  - **[Value pessimism](/shared/glossary/#pessimism)**: explicitly penalize Q-values at out-of-distribution actions ([CQL](/shared/glossary/#cql), [IQL](/shared/glossary/#iql) — the modern default)
- **[CQL (Conservative Q-Learning)](/shared/glossary/#cql)** — adds a penalty to the standard [Bellman](/shared/glossary/#bellman-equation) loss that pushes down Q-values at all actions and pulls them up only at the actions in the data
- **[IQL (Implicit Q-Learning)](/shared/glossary/#iql)** — never queries `Q` at out-of-distribution actions at all; uses [expectile regression](/shared/glossary/#expectile-regression) on `V`. Simpler and more robust than CQL
- **[Decision Transformer](/shared/glossary/#decision-transformer) / [Trajectory Transformer](/shared/glossary/#trajectory-transformer)** — reframe offline RL as [autoregressive](/shared/glossary/#autoregressive-model) [sequence modeling](/shared/glossary/#sequence-modeling): condition on a desired [return-to-go](/shared/glossary/#return-to-go), predict the next action. Strong when the dataset is large and diverse
- **[Behavior cloning](/shared/glossary/#bc) baselines** — sometimes BC is shockingly competitive with offline RL, especially when the dataset is high-quality. Always run it as a baseline
- **[D4RL](/shared/glossary/#d4rl)** — the standard offline-RL benchmark suite
- **The relationship to RLHF**: RLHF is a constrained-policy-improvement problem; offline-RL methods (especially DPO-as-offline-RL views) directly inform the RLHF stack

### The Distribution-Shift Picture

```
Dataset D = {(s, a, r, s')} collected by some unknown behavior policy π_β

Naive Q-learning on D:
   target(s, a) = r + γ max_{a'} Q(s', a')   ← evaluated at the GREEDY action a',
                                                which may have NEVER been seen

   ↓
   Q estimates explode at unseen actions
   ↓
   Learned policy preferentially takes those exploded actions
   ↓
   Catastrophic failure on the real environment

Offline RL fix: either keep the policy close to π_β, or pessimistically
under-estimate Q for actions not in the data. Both work; pessimism is
the modern default.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [BC baseline on D4RL](projects/38-bc-baseline-on-d4rl/README.md) | Behavior cloning on a `halfcheetah-medium` dataset; report return | ⭐⭐ |
| [Naive Q-learning on the same dataset](projects/39-naive-q-learning-on-the-same-dataset/README.md) | Verify the catastrophic failure | ⭐⭐⭐ |
| [Implement CQL](projects/40-implement-cql/README.md) | Add the conservative penalty; reproduce CQL's D4RL numbers | ⭐⭐⭐⭐ |
| [Implement IQL](projects/41-implement-iql/README.md) | Verify it works on the same tasks with fewer knobs | ⭐⭐⭐⭐ |
| [Decision Transformer](projects/42-decision-transformer/README.md) | Implement on D4RL; condition on return-to-go; compare to IQL | ⭐⭐⭐⭐ |
| [Dataset-quality study](projects/43-dataset-quality-study/README.md) | Same algorithm, three datasets (random, medium, expert); plot return-vs-data-quality | ⭐⭐⭐ |

### Sample Code: The Heart of IQL

```python
# IQL's three losses:
# 1. Value function: expectile regression on the dataset Q-values
def value_loss(V, Q, s, a, tau=0.7):
    with torch.no_grad():
        q = torch.min(Q1(s, a), Q2(s, a))           # double critic
    v = V(s)
    diff = q - v
    # Asymmetric (expectile) loss: weight positive errors by tau, negative by (1-tau)
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

# 2. Critic: standard TD with NO max over actions
def critic_loss(Q, V, s, a, r, s2, d):
    with torch.no_grad():
        target = r + gamma * V(s2) * (1 - d)
    return F.mse_loss(Q(s, a), target)

# 3. Actor: advantage-weighted regression
def actor_loss(pi, Q, V, s, a, beta=3.0):
    with torch.no_grad():
        adv = torch.min(Q1(s, a), Q2(s, a)) - V(s)
        weight = torch.clamp((beta * adv).exp(), max=100.0)
    return -(weight * pi.log_prob(a)).mean()
```

### Key Insight

The offline-RL field oscillated between policy-constraint and value-pessimism approaches for several years before IQL clarified things: **you don't actually need to query Q at out-of-distribution actions to make a good policy**. By only using in-distribution Q-values to estimate `V`, then using advantage-weighted regression for the policy, IQL sidesteps the whole over-extrapolation problem. This insight — *constrain what you query, not what you output* — is increasingly visible in RLHF and reasoning-model training too.

### Resources

- [Levine et al. — Offline RL Tutorial (2020)](https://arxiv.org/abs/2005.01643) — survey
- [Kumar et al. — CQL (2020)](https://arxiv.org/abs/2006.04779)
- [Kostrikov et al. — IQL (2021)](https://arxiv.org/abs/2110.06169)
- [Chen et al. — Decision Transformer (2021)](https://arxiv.org/abs/2106.01345)
- [D4RL benchmark](https://sites.google.com/view/d4rl/home)

---

## Phase 8: Exploration

Reward-hungry agents in sparse-reward worlds spend almost all of their time wandering. Exploration is the part of RL where mathematical guarantees and practical methods diverge most sharply. Theory says use UCB or Thompson sampling; practice says... it's complicated.

### Concepts to Learn

- **Exploration vs exploitation as a fundamental trade-off** — and why ε-greedy is "good enough" in dense-reward environments and disastrous in sparse-reward ones
- **Optimism in the face of uncertainty** — UCB, Bayesian posteriors over `Q`, bootstrapped DQN
- **Intrinsic motivation**:
  - **Count-based exploration** — bonus for novel states, scaled by `1/√N(s)`
  - **Curiosity-driven exploration (ICM, RND)** — bonus for high prediction error of a learned forward or random model
  - **Empowerment** — maximize mutual information between actions and future state
  - **Disagreement** — bonus for ensemble disagreement on the next state
- **Go-Explore** — separate "go to a known state" from "explore from there"; spectacular results on Montezuma's Revenge
- **Maximum-entropy exploration** — what SAC does implicitly
- **Adversarial / unsupervised exploration** — DIAYN, APT, ProtoRL: learn skills with no reward at all
- **Hard exploration benchmarks**: Montezuma's Revenge, Pitfall, NetHack, MineRL
- **Why this is unsolved**: no algorithm robustly explores arbitrary sparse-reward worlds. Every method works on the benchmarks it was designed for and breaks on adversarial new ones

### A Taxonomy

```
                         How is the bonus computed?
                                 │
          ┌──────────────────────┼─────────────────────────┐
          ↓                      ↓                         ↓
   Count-based            Prediction-error          Information-gain
   (visit counts,         (ICM, RND: train a        (Bayesian / ensemble
   pseudo-counts          predictor; bonus = its    disagreement; bonus
   from density           own error)                = uncertainty in V or P)
   models)
          ↓                      ↓                         ↓
   Works in tabular /      Works on Atari but      Theoretically clean,
   discrete state.         brittle; can latch onto  computationally heavy.
   Hard in continuous /    "noisy TV problem"
   pixel state.            (stochastic noise = high
                            error = high reward, no
                            real exploration value)
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [ε-greedy on a chain](projects/44-epsilon-greedy-on-a-chain/README.md) | A 10-state chain MDP with reward only at one end; observe how ε-greedy fails as chain length grows | ⭐⭐ |
| [Count-based on a small env](projects/45-count-based-on-a-small-env/README.md) | Add a `1/√N(s)` bonus to Q-learning; verify exploration accelerates | ⭐⭐⭐ |
| [RND on Atari](projects/46-rnd-on-atari/README.md) | Implement Random Network Distillation; apply to Montezuma's Revenge; see the famous result | ⭐⭐⭐⭐ |
| [ICM](projects/47-icm/README.md) | Implement the Intrinsic Curiosity Module; compare to RND on the same task | ⭐⭐⭐⭐ |
| [DIAYN](projects/48-diayn/README.md) | Train diverse skills with no extrinsic reward; visualize the skill space | ⭐⭐⭐⭐⭐ |
| [Noisy-TV experiment](projects/49-noisy-tv-experiment/README.md) | Construct an environment with a TV that shows random noise; verify that naive prediction-error methods get stuck staring at it | ⭐⭐⭐ |

### Sample Code: An RND Bonus

```python
# Random Network Distillation: a fixed random target, a trainable predictor.
# Bonus = prediction error. Novel states have high error -> high bonus -> visited more often.
target_net    = MLP(obs_dim, 128).eval()                  # frozen, random init
predictor_net = MLP(obs_dim, 128)                          # trained on observed states
for p in target_net.parameters():
    p.requires_grad = False

def intrinsic_bonus(obs):
    with torch.no_grad():
        t = target_net(obs)
    p = predictor_net(obs)
    err = (t - p).pow(2).mean(dim=-1)
    return err                                             # add this to the extrinsic reward

# Separately, train the predictor on the same observations:
pred_loss = (target_net(obs).detach() - predictor_net(obs)).pow(2).mean()
```

### Key Insight

Exploration is the part of RL where you most need to know the structure of your problem. *General-purpose* exploration is not solved, may not be solvable, and most papers that claim to solve it are evaluated only on the few benchmarks where their assumptions hold. *Problem-specific* exploration — using domain knowledge to design demonstrations, reset distributions, curricula, or shaped intrinsic rewards — almost always wins in practice. The pragmatic advice in 2026: bake exploration into your data collection (curated resets, demonstrations, curricula) rather than into your algorithm. The frontier may shift back the other way as foundation models give us better priors for what's worth exploring.

### Resources

- [Bellemare et al. — Unifying Count-Based Exploration (2016)](https://arxiv.org/abs/1606.01868)
- [Pathak et al. — ICM (2017)](https://arxiv.org/abs/1705.05363)
- [Burda et al. — RND (2018)](https://arxiv.org/abs/1810.12894)
- [Ecoffet et al. — Go-Explore (2019; Nature 2021)](https://arxiv.org/abs/1901.10995)
- [Eysenbach et al. — DIAYN (2018)](https://arxiv.org/abs/1802.06070)
- [Henaff — Curiosity and exploration survey](https://arxiv.org/abs/1907.05388)

---

## Phase 9: RL for Language Models — RLHF, DPO, GRPO, RLVR

This is where the modern field has its center of gravity. The RL ideas you learned in Phases 1–4 directly drive the post-training pipelines for every major frontier LLM. Most innovation in RL between 2023 and 2026 happened in this phase, not in classical control.

### Concepts to Learn

#### The basic RLHF pipeline (post-2022)

1. **Pretraining** — a base language model on web data
2. **Supervised fine-tuning (SFT)** — fine-tune on human-written demonstrations
3. **Reward model (RM) training** — train a model to score completions, supervised by human pairwise preferences
4. **RL fine-tuning** — optimize the policy (the LLM) to maximize the RM's score, with a KL penalty back to the SFT model

#### The RLHF objective

```
J(π) = E_{x ~ data, y ~ π(·|x)} [ R_φ(x, y) ]  −  β · KL( π || π_SFT )

R_φ(x, y) :   reward model's score for completion y on prompt x
π          :   the language model being trained
π_SFT      :   the SFT model, treated as a fixed reference
β          :   how much to penalize drifting from the reference

The KL penalty is crucial. Without it, the policy reward-hacks the RM
(produces gibberish that the RM happens to score high) within hundreds of steps.
With it, the policy gets better at what humans actually want, slowly.
```

#### Algorithms

- **PPO for RLHF** — the original recipe (InstructGPT, ChatGPT). Treat the prompt as the state, the completion as a sequence of actions, the RM score as terminal reward, and run PPO. Notoriously fiddly: value head, GAE on token-level returns, KL scheduling
- **DPO (Direct Preference Optimization)** — closed-form derivation showing that the RLHF objective is equivalent to a *supervised* loss on preference pairs, when the optimal policy is parameterized correctly. No reward model, no rollouts, no PPO. The default in many open-source post-training pipelines
- **IPO, KTO, ORPO, SimPO, R-DPO** — DPO-family variants tweaking the loss to fix specific failure modes (overfitting, length bias, asymmetric preferences)
- **GRPO (Group Relative Policy Optimization)** — DeepSeek's PPO variant. For each prompt, sample a *group* of completions, compute relative advantages within the group (no value baseline needed), and apply PPO-style updates. Memory-efficient, simpler than PPO, the backbone of recent reasoning-model training
- **RLVR (RL with Verifiable Rewards)** — when answers can be checked programmatically (math, code, formal proofs), skip the reward model entirely and use the verifier directly. This is the engine of the reasoning-model wave (o1, R1, etc.)
- **REINFORCE-style algorithms returning** — RLOO and similar; once you have a working verifier, vanilla policy gradients with leave-one-out baselines often match or beat PPO

#### What changed in 2024–2026

- **Reasoning models**: RLVR on math and code at scale produces models that learn to think before answering. The "RL on chain-of-thought" loop turns out to scale beautifully
- **Process reward models** vs **outcome reward models**: scoring each reasoning step vs scoring only the final answer
- **Self-play and self-improvement loops**: model generates problems, attempts solutions, verifies, learns from successes (STaR, V-STaR, RFT, R*)
- **Multi-turn RLHF**: optimizing across full conversations, not just single completions

### The DPO Derivation, in Words

```
Start with the RLHF objective:
   max_π  E[R(x, y)] − β KL(π || π_ref)

Solve for the optimal π analytically (Lagrangian):
   π*(y|x) ∝ π_ref(y|x) · exp(R(x, y) / β)

Rearrange to express R in terms of π* and π_ref:
   R(x, y) = β log( π*(y|x) / π_ref(y|x) ) + const(x)

Substitute this into the Bradley-Terry preference model
P(y_w > y_l | x) = σ(R(x, y_w) − R(x, y_l)).

You get a purely supervised loss on preference pairs:
   L_DPO = −log σ( β log(π(y_w|x)/π_ref(y_w|x)) − β log(π(y_l|x)/π_ref(y_l|x)) )

No reward model. No PPO. Just a clever loss on (prompt, chosen, rejected) triples.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [SFT a small base model](projects/50-sft-a-small-base-model/README.md) | Fine-tune Qwen-0.5B or similar on a small instruction dataset; observe baseline behavior | ⭐⭐ |
| [Train a reward model](projects/51-train-a-reward-model/README.md) | Pairwise classifier over SFT outputs; verify on held-out preferences | ⭐⭐⭐ |
| [PPO-style RLHF](projects/52-ppo-style-rlhf/README.md) | Mini-RLHF on a small model and small RM; track KL to reference; watch for reward hacking | ⭐⭐⭐⭐⭐ |
| [DPO](projects/53-dpo/README.md) | Same dataset, DPO instead of PPO; compare quality, training time, stability | ⭐⭐⭐⭐ |
| [GRPO from scratch](projects/54-grpo-from-scratch/README.md) | Implement GRPO for a small math task (GSM8K-style) with a verifiable reward | ⭐⭐⭐⭐⭐ |
| [RLVR on math](projects/55-rlvr-on-math/README.md) | Train a small reasoning loop on a verifiable math subset; observe the emergence of longer chain-of-thought | ⭐⭐⭐⭐⭐ |
| [Length-bias audit](projects/56-length-bias-audit/README.md) | Plot completion-length distributions of your DPO/PPO models; verify the well-known drift | ⭐⭐⭐ |
| [Reward hacking demo](projects/57-reward-hacking-demo/README.md) | Intentionally over-train against an RM; characterize the gibberish that emerges | ⭐⭐⭐ |

### Sample Code: The DPO Loss

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logits_w, policy_logits_l,
             ref_logits_w,    ref_logits_l,
             chosen_ids,      rejected_ids,
             beta=0.1):
    """
    policy_logits_*: logits from the model being trained, on chosen / rejected sequences
    ref_logits_*:    logits from the frozen reference model
    *_ids:           target token ids
    """
    logp_w_pi  = sequence_logp(policy_logits_w, chosen_ids)
    logp_l_pi  = sequence_logp(policy_logits_l, rejected_ids)
    logp_w_ref = sequence_logp(ref_logits_w,    chosen_ids).detach()
    logp_l_ref = sequence_logp(ref_logits_l,    rejected_ids).detach()

    pi_logratios  = logp_w_pi  - logp_l_pi
    ref_logratios = logp_w_ref - logp_l_ref

    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()

def sequence_logp(logits, ids):
    """Sum of log p(token_t | tokens_<t) along a sequence."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, ids[:, 1:, None]).squeeze(-1).sum(-1)
```

### Key Insight

The RLHF/DPO/GRPO/RLVR progression is the clearest example in recent ML of a research field collapsing complexity. Each step removed a moving part: DPO removed the explicit reward model and PPO. GRPO removed the value function. RLVR (when applicable) removed the reward model and replaced it with a deterministic verifier. The remaining components — preference pairs or verifiers, a reference model, a KL penalty — are mostly irreducible. *When a verifier exists, RL on LLMs becomes embarrassingly straightforward*. When it doesn't (everything involving subjective quality, style, helpfulness), you still need the messy preference pipeline. The grand bet of 2026 is: how many domains can be converted to verifiable form?

### Resources

- [Christiano et al. — Deep RL from Human Preferences (2017)](https://arxiv.org/abs/1706.03741) — the foundational paper
- [Ouyang et al. — InstructGPT (2022)](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. — DPO (2023)](https://arxiv.org/abs/2305.18290)
- [Shao et al. — DeepSeekMath + GRPO (2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 paper (2025)](https://arxiv.org/abs/2501.12948) — the RLVR-for-reasoning playbook
- [Tülu 3 (Allen AI, 2024)](https://arxiv.org/abs/2411.15124) — open-source RLHF/RLVR recipe
- [Hugging Face TRL library](https://huggingface.co/docs/trl/) — the practical implementation
- [Lambert — *Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — the book on this material

---

## Phase 10: Frontier Topics

Where the field is going. Pick one or two threads and follow them; you cannot follow all of them.

### Reasoning Models and the RLVR Wave
The most active research area in RL right now. Long chain-of-thought, RL with verifiable rewards (math, code, formal proofs), self-correction, search-augmented inference. The o1/R1/Claude-3.7-style reasoning trace is the visible part; the post-training recipes that produce it are the iceberg. Expect this thread to dominate 2026–2027.

### Multi-Agent RL
Cooperative (StarCraft II, hide-and-seek), competitive (Go, poker), and mixed (negotiation, market simulation). Self-play, population-based training, league play. Non-stationarity from the other agents' learning is the central technical challenge.

### Meta-RL and Few-Shot Adaptation
Learning to learn: RL² (an LSTM as the inner RL algorithm), MAML, PEARL. Increasingly subsumed by in-context learning in foundation-model policies.

### Hierarchical RL
Options, sub-goals, feudal networks. The dream of compositional skills has been hard to realize cleanly, but VLAs and language-conditioned policies are achieving similar effects through different means (Phase 6 of the [Robotics Guide](../robotics/)).

### World Models and Generative RL
Generative video models that are also simulators. Crosses directly into [Video Generation Guide Phase 9](../video-generation/#phase-9-world-models-and-interactive-video). Genie 2, Sora-style for sim-replacement, Dreamer-class for control. The convergence of generative modeling and RL.

### Inverse RL and Reward Learning
When the reward is unknown but the demonstrations exist: GAIL, AIRL, MaxEnt IRL. Connects to the preference modeling in RLHF and to safety (learning what humans value).

### Constrained RL and Safe RL
Optimize reward subject to safety constraints. Critical for real-world deployment. CMDP formalism, Lagrangian methods, shielded RL. Adjacent to formal-verification approaches for embodied AI.

### RL at Scale on Foundation Models
The infrastructure side: efficient PPO/GRPO on tens of thousands of GPUs, asynchronous generation/training, KV-cache management for rollout, vLLM/SGLang integrations. Most of the engineering work behind reasoning-model training lives here.

### Open-Endedness and Curricula
Procedurally-generated environments (XLand, MineRL BASALT, Crafter), automatic curriculum learning, generative agents. The bet that "general intelligence comes from a diverse stream of tasks."

### Theoretical RL
Sample complexity bounds, regret bounds, PAC-RL, distributional shifts. Currently disconnected from practice but slowly catching up. Worth watching if you like math.

### Resources for the Frontier
- [OpenAI o1 system card and follow-ups](https://openai.com/index/learning-to-reason-with-llms/)
- [DeepSeek-R1, V3, V3.1](https://github.com/deepseek-ai)
- [Tülu 3 (open-source frontier post-training recipe)](https://arxiv.org/abs/2411.15124)
- [GPU Mode RL track](https://github.com/gpu-mode/lectures)
- [NeurIPS, ICML, ICLR](https://nips.cc/) — RL tracks at the top ML venues
- [RLHF book — Lambert](https://rlhfbook.com/)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | Gymnasium, PyTorch, CleanRL installed; Sutton & Barto Ch. 1–2 read |
| 1. MDPs and Bellman | 1 week | Gridworld coded, value iteration converges, Bellman equations are reflex |
| 2. Tabular methods | 1–2 weeks | Q-learning, SARSA, TD(λ) implemented; understand the deadly triad |
| 3. DQN and friends | 2 weeks | DQN on Atari; Double + Dueling + PER ablation done |
| 4. Policy gradients | 2 weeks | PPO from scratch, all 37 details understood |
| 5. Continuous control | 2 weeks | SAC on Mujoco suite; understand why it dominates |
| 6. Model-based | 2–3 weeks | MPC done; mini Dreamer or mini MuZero attempted |
| 7. Offline RL | 1–2 weeks | IQL on D4RL; BC baseline run honestly |
| 8. Exploration | 1–2 weeks | RND on Montezuma's Revenge; opinions on what works and what doesn't |
| 9. RL for LLMs | 3–4 weeks | DPO done; GRPO understood; RLVR loop on a verifiable task working |
| 10. Frontier | Ongoing | Picked one thread and going deep |

**Total to "comfortable practitioner":** ~3–4 months of focused study, longer if combined with real projects (recommended). To "research-comfortable on one frontier thread": 6–12 months beyond that.

---

## Key Advice

1. **Read Sutton & Barto.** Yes, the whole thing. It is the only book in ML where the gap between "skimmed it" and "actually read it" is night-and-day. Most "RL doesn't work" stories trace back to skimming.
2. **Implement from scratch at least once.** Tabular Q-learning, DQN, PPO, SAC, DPO. Not "use the library version"; from scratch. The exercise teaches you which knobs are real and which are accidents.
3. **Read CleanRL before reading anything else.** Each algorithm is in a single self-contained file. The standard reference implementation. Pin its commit when you reproduce a result.
4. **Read the 37 PPO implementation details.** Not optional. The gap between "PPO works" and "PPO works for me" is mostly that document.
5. **Normalize. Everything.** Observations, advantages, returns, rewards. PPO and SAC are weirdly sensitive to scale. The default in every working repo: running mean-and-variance normalization.
6. **Seed and run multiple times.** RL has *enormous* run-to-run variance. A single curve tells you almost nothing. Always plot 3–5 seeds with min/max bands.
7. **Always run a BC / random / hand-coded baseline.** Half the "RL works" claims in the literature are not robust to comparing against a strong non-RL baseline on the same task.
8. **Define your reward signal before your algorithm.** Reward shaping decisions dominate algorithm decisions. A poorly-shaped reward will defeat any algorithm; a well-shaped reward will be solvable by REINFORCE.
9. **`detach()` the target.** The single most common deep-RL bug: gradients flowing through the bootstrap target. Wrap every TD-target computation in `with torch.no_grad():` or sprinkle `.detach()` aggressively.
10. **Profile before scaling.** RL training spends shockingly little time in the forward pass and shockingly much in environment stepping, data movement, and synchronization. Always profile.
11. **In the LLM-RL era, ask: do I have a verifier?** If yes, use RLVR. If no, decide between preference learning (DPO) and full RLHF (PPO/GRPO). The decision dominates everything that follows.
12. **Avoid PPO's value head bug.** A common implementation has the value head share the trunk with the policy and use a separate normalization. The interaction between value-loss clipping and observation normalization is responsible for many silent failures.

---

## Common Pitfalls

- ❌ Forgetting `detach()` on the TD target → gradients flow through both sides of the Bellman equation, training diverges
- ❌ Off-by-one in the discounted return (`G_t` vs `G_{t+1}`) → subtle, takes days to find
- ❌ Wrong `done` handling at episode boundaries → `V(s_{T+1})` should be zero, not predicted
- ❌ Using `float16` for value functions → silent NaNs from the wide value range
- ❌ Not normalizing observations → wildly different scales blow up the policy gradient
- ❌ Tuning hyperparameters on a single seed → publishing results that don't replicate
- ❌ Forgetting the log-prob correction for `tanh`-squashed actions in SAC → silently broken policy
- ❌ Treating the SAC actor's `log_prob` as a categorical when it's Gaussian (or vice versa) → wrong loss, wrong everything
- ❌ Computing GAE across episode boundaries → corrupted advantages
- ❌ Using `optimizer.zero_grad()` once per epoch instead of per minibatch → gradients accumulate incorrectly
- ❌ Reading "PPO with default hyperparameters" and assuming they're the same across papers → they're not; always check
- ❌ Training RLHF without a strong reference model → reward hacking within hours
- ❌ Skipping the KL penalty in RLHF → catastrophic policy collapse
- ❌ Running RL with an environment that has non-Markov observations and assuming the policy will "figure it out" → it won't

---

## Additional Resources

### Books
- [Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) — the bible
- [Szepesvári — *Algorithms for Reinforcement Learning*](https://sites.ualberta.ca/~szepesva/rlbook.html) — concise, mathematical
- [Lambert — *Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — the modern post-training book
- [Bertsekas — *Reinforcement Learning and Optimal Control*](http://www.athenasc.com/RLbook.html) — for the control-theoretically inclined

### Courses
- [David Silver — RL course (UCL/DeepMind, 2015)](https://www.davidsilver.uk/teaching/) — the canonical lecture series
- [Sergey Levine — Deep RL (UC Berkeley, CS285)](https://rail.eecs.berkeley.edu/deeprlcourse/) — the modern course
- [Emma Brunskill — RL (Stanford, CS234)](https://web.stanford.edu/class/cs234/)
- [Pieter Abbeel — Foundations of Deep RL (YouTube)](https://www.youtube.com/playlist?list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37) — 6-lecture series, excellent intro

### Code You Should Read
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file implementations of every major algorithm
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — the production reference
- [TRL (Hugging Face)](https://github.com/huggingface/trl) — RLHF/DPO/GRPO at scale
- [trlX](https://github.com/CarperAI/trlx) — the older RLHF reference; still useful
- [DreamerV3 (official)](https://github.com/danijar/dreamerv3) — model-based reference
- [TD-MPC2](https://github.com/nicklashansen/tdmpc2)

### Environment Libraries
- [Gymnasium](https://gymnasium.farama.org/) — the standard API
- [MuJoCo via Gymnasium](https://gymnasium.farama.org/environments/mujoco/) — continuous control benchmarks
- [Atari via Gymnasium](https://gymnasium.farama.org/environments/atari/) — the classic discrete-action benchmarks
- [DM Control](https://github.com/google-deepmind/dm_control) — DeepMind's continuous control suite
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — massively parallel sim for locomotion and manipulation
- [PettingZoo](https://pettingzoo.farama.org/) — multi-agent environments
- [MineRL](https://minerl.io/) — Minecraft as a hard-exploration benchmark
- [NetHack Learning Environment](https://github.com/heiner/nle)

### Communities
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) — the most active general forum
- [GPU Mode Discord](https://github.com/gpu-mode/lectures) — for the RL-at-scale engineering side
- [Eleuther AI Discord](https://www.eleuther.ai/) — for the RL-for-LLMs work
- [The RL Reading Group](https://rl-reading-group.github.io/) — weekly paper reading

### Talks Worth Watching
- [Sergey Levine — "Underlying Assumptions of Deep RL"](https://www.youtube.com/watch?v=jds0Wh9jTvE)
- [John Schulman — "The Nuts and Bolts of Deep RL Research"](http://joschu.net/docs/nuts-and-bolts.pdf) — the practical advice document
- [Pieter Abbeel — many lectures](https://www.youtube.com/c/PieterAbbeel)

---

## Quick Start Checklist

- [ ] Have written down the Bellman equation for `V^π` and `V*` without looking
- [ ] Have implemented tabular Q-learning and watched it solve FrozenLake
- [ ] Have implemented DQN and watched it solve at least one Atari game
- [ ] Have implemented PPO from scratch and verified all 37 details
- [ ] Have run PPO with 5 seeds and reported min/max/median, not just the best seed
- [ ] Have implemented SAC and understood the entropy bonus
- [ ] Have implemented at least one model-based algorithm (MPC, MBPO, or mini Dreamer)
- [ ] Have run an offline RL algorithm on D4RL and compared it honestly to BC
- [ ] Have built a sparse-reward environment and watched ε-greedy fail
- [ ] Have implemented DPO and trained it on a small instruction dataset
- [ ] Have implemented GRPO or RLVR on a verifiable-reward task
- [ ] Have diagnosed at least one reward-hacking incident
- [ ] Can read a contemporary RL paper and explain which design choices solve which problems

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
