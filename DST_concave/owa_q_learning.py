"""
owa_q_learning.py
=================
OWA Q-Learning: Ordered Weighted Averaging scalarisation.

OWA sorts objective values before applying weights,
so the weight attaches to the rank (best, worst) rather than a fixed objective.
This produces more balanced solutions and can reach some concave front regions.

OWA([v1, v2], [w1, w2]) = w1 * max(v1, v2) + w2 * min(v1, v2)

Coordinate convention
---------------------
    rew[0] = treasure  (positive, maximise)
    rew[1] = time_ret  (negative = -steps, maximise)
    Evaluation returns (time_cost=steps, treasure).
    Metrics logged in maximisation form: (-steps, treasure).
"""

import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d, compute_igd, compute_epsilon_indicator
from env import get_true_reference_pf


# =============================================================================
# OWA scalarisation
# =============================================================================

def _owa(values, weights):
    """
    Ordered Weighted Averaging: sort values descending, then dot with weights.
    weights[0] applies to the largest value, weights[1] to the smallest.
    """
    return sum(w * v for w, v in zip(weights, sorted(values, reverse=True)))


def _best_action(Q1, Q2, state, weights):
    """Greedy action: argmax OWA([Q1[s,a], Q2[s,a]], weights)."""
    return int(np.argmax([
        _owa([Q1[state][a], Q2[state][a]], weights)
        for a in range(4)
    ]))


# =============================================================================
# Policy evaluation
# =============================================================================

def _evaluate(env, Q1, Q2, weights, n_eval=10):
    """
    Run the greedy OWA policy for n_eval episodes.
    Returns (mean_time_cost, mean_treasure).
    """
    times, treasures = [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state  = (int(obs[0]), int(obs[1]))
            action = _best_action(Q1, Q2, state, weights)
            obs, rew, terminated, truncated, _ = env.step(action)
            treasure = float(rew[0])
            steps   += 1
            done     = terminated or truncated
        times.append(steps)
        treasures.append(treasure)
    return float(np.mean(times)), float(np.mean(treasures))


# =============================================================================
# Training
# =============================================================================

def train_owa_q(
    owa_weights,
    total_timesteps = 400_000,
    lr              = 0.1,
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    log_interval    = 1_000,
    n_eval          = 10,
):
    """
    Train OWA Q-Learning with a fixed OWA weight vector.

    Returns: (final_point, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts)
        final_point - (time_cost, treasure) in training form
    """
    env     = mo_gym.make("deep-sea-treasure-concave-v0")
    true_pf = get_true_reference_pf()   # maximisation form

    Q1 = defaultdict(lambda: np.zeros(4))   # time_ret objective
    Q2 = defaultdict(lambda: np.zeros(4))   # treasure objective

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

            action = (env.action_space.sample() if np.random.rand() < epsilon
                      else _best_action(Q1, Q2, state, owa_weights))

            next_obs, rew, terminated, truncated, _ = env.step(action)
            global_step += 1
            done        = terminated or truncated
            next_state  = (int(next_obs[0]), int(next_obs[1]))

            treasure  = float(rew[0])
            time_r    = float(rew[1])
            best_next = _best_action(Q1, Q2, next_state, owa_weights)

            Q1[state][action] += lr * (time_r   + gamma * Q1[next_state][best_next] - Q1[state][action])
            Q2[state][action] += lr * (treasure + gamma * Q2[next_state][best_next] - Q2[state][action])

            obs = next_obs

            while global_step >= next_log:
                tc, tr   = _evaluate(env, Q1, Q2, owa_weights, n_eval)
                pt_max   = (-tc, tr)   # maximisation form

                hv_timesteps.append(next_log)
                hv_points.append(compute_hypervolume_2d([pt_max], ref_point=(-100, 0)))
                igd_timesteps.append(next_log)
                igd_points.append(compute_igd(true_pf, [pt_max]))
                eps_timesteps.append(next_log)
                eps_points.append(compute_epsilon_indicator(true_pf, [pt_max]))

                next_log += log_interval

    tc, tr = _evaluate(env, Q1, Q2, owa_weights, n_eval)
    env.close()
    return (tc, tr), hv_timesteps, hv_points, igd_timesteps, igd_points, eps_timesteps, eps_points