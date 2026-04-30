"""
Pareto Q-Learning (PQL) for multi-objective RL.

Maintains a set of non-dominated value vectors per (state, action) pair.
A single run produces an approximation of the full Pareto front.

Internal coordinate form: (treasure, time_ret)
    treasure  = rew[0]  positive float
    time_ret  = rew[1]  negative float = -time_cost

All metrics computed in maximisation form (-time_cost, treasure)
"""

import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import (
    get_non_dominated,
    compute_hypervolume_2d,
    compute_igd,
    compute_epsilon_indicator,
)
from env import get_true_reference_pf

MAX_POINTS = 50


# Vector-set helpers

def _prune(nd):
    """Limit Pareto set to MAX_POINTS while preserving spread."""
    if len(nd) <= MAX_POINTS:
        return nd
    nd_sorted = sorted(nd, key=lambda p: p[0])
    idx = np.linspace(0, len(nd_sorted) - 1, MAX_POINTS).astype(int)
    return [nd_sorted[i] for i in idx]


def _propagate(immediate, future_nd, gamma):
    """Bellman backup: immediate + gamma * each future vector."""
    return [(immediate[0] + gamma * f[0],
             immediate[1] + gamma * f[1]) for f in future_nd]


def _all_vectors(Q_sets, state, n_actions):
    out = []
    for a in range(n_actions):
        out.extend(Q_sets[state][a])
    return out


def _front_at_state(Q_sets, state, n_actions):
    return get_non_dominated(_all_vectors(Q_sets, state, n_actions))


def _front_to_max(vectors):
    """
    Internal (treasure, time_ret) -> maximisation form (time_ret, treasure).
    time_ret is already negative = -time_cost, so this is a coordinate swap.
    """
    return [(float(t), float(tr)) for tr, t in vectors]


# Action selection

def _best_action(Q_sets, state, n_actions):
    """
    PQL greedy: pick action whose Q-set contributes the most vectors to the
    global non-dominated set at this state. 
    """
    all_vecs  = _all_vectors(Q_sets, state, n_actions)
    global_nd = set(map(tuple, get_non_dominated(all_vecs)))

    best_a, best_count, best_size = 0, -1, -1
    for a in range(n_actions):
        count = sum(1 for v in Q_sets[state][a] if tuple(v) in global_nd)
        size  = len(Q_sets[state][a])
        if count > best_count or (count == best_count and size > best_size):
            best_a, best_count, best_size = a, count, size
    return best_a


# Training

def train_pql(
    total_timesteps = 400_000,
    gamma           = 1.0,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    log_interval    = 5_000,
):
    env       = mo_gym.make("deep-sea-treasure-concave-v0")
    n_actions = env.action_space.n
    true_pf   = get_true_reference_pf()

    start_obs, _ = env.reset()
    start_state  = (int(start_obs[0]), int(start_obs[1]))

    Q_sets = defaultdict(lambda: [[(0.0, 0.0)] for _ in range(n_actions)])

    hv_timesteps,  hv_points  = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []

    global_step, next_log = 0, log_interval

    while global_step < total_timesteps:
        obs, _ = env.reset()
        done   = False

        while not done and global_step < total_timesteps:
            state   = (int(obs[0]), int(obs[1]))
            epsilon = max(epsilon_end,
                          epsilon_start - (epsilon_start - epsilon_end)
                          * global_step / total_timesteps)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = _best_action(Q_sets, state, n_actions)

            next_obs, rew, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            next_state  = (int(next_obs[0]), int(next_obs[1]))
            global_step += 1

            immediate = (float(rew[0]), float(rew[1]))   # (treasure, time_ret)

            future_nd = ([(0.0, 0.0)] if done
                         else get_non_dominated(
                             _all_vectors(Q_sets, next_state, n_actions)))

            new_set = get_non_dominated(_propagate(immediate, future_nd, gamma))
            Q_sets[state][action] = _prune(new_set)

            obs = next_obs

            if global_step % 10_000 == 0:
                print(f"  [PQL] step {global_step:,}")

            # Logging: all metrics via shared utils functions, maximisation form
            while global_step >= next_log:
                front     = _front_at_state(Q_sets, start_state, n_actions)
                front_max = _front_to_max(front)

                hv_timesteps.append(next_log)
                hv_points.append(compute_hypervolume_2d(front_max, ref_point=(-100, 0)))
                igd_timesteps.append(next_log)
                igd_points.append(compute_igd(true_pf, front_max))
                eps_timesteps.append(next_log)
                eps_points.append(compute_epsilon_indicator(true_pf, front_max))

                next_log += log_interval

    # Final front in (time_cost, treasure) form expected by plot_pql_results
    final_vectors = _all_vectors(Q_sets, start_state, n_actions)
    nd            = get_non_dominated(final_vectors)
    final_points  = sorted(
        set((float(-t), float(tr)) for tr, t in nd),
        key=lambda p: p[0],
    )

    env.close()
    return (final_points,
            hv_timesteps, hv_points,
            igd_timesteps, igd_points,
            eps_timesteps, eps_points)