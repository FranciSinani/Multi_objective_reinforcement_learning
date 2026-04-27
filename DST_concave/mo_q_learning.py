"""
mo_q_learning.py
================
MO Q-Learning: weighted-sum scalarisation of two separate Q-tables.

Learns Q1 (time objective) and Q2 (treasure objective) independently,
then combines them as Q[s,a] = timeW * Q1[s,a] + treasureW * Q2[s,a]
for action selection. Each weight setting targets one point on the front.


Coordinate convention
---------------------
    rew[0] = treasure  (positive float, maximise)
    rew[1] = time_ret  (negative float = -steps, maximise)
    Evaluation returns (time_cost=steps, treasure) — positive step count.
    Metric logging converts to maximisation form: pt_max = (-steps, treasure).
    All metric functions (HV, IGD, EPS) receive maximisation-form inputs.
"""

import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d, compute_igd, compute_epsilon_indicator
from env import get_true_reference_pf


# =============================================================================
# Policy evaluation
# =============================================================================

def _evaluate(env, Q, n_eval=10):
    """
    Run the greedy policy (argmax Q[s]) for n_eval episodes.
    Returns (mean_time_cost, mean_treasure).
        time_cost = number of steps taken (positive integer)
        treasure  = final treasure collected (positive float)
    """
    times, treasures = [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state  = (int(obs[0]), int(obs[1]))
            action = int(np.argmax(Q[state]))
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

def train_mo_q(
    timeW,
    treasureW,
    total_timesteps = 400_000,
    lr              = 0.1,
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    log_interval    = 1_000,
    n_eval          = 10,
):
    """
    Train MO Q-Learning with a fixed weighted-sum scalarisation.

    timeW + treasureW should sum to 1.0.
    Returns: (final_point, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts)
        final_point – (time_cost, treasure) tuple in training form
        *_ts        – list of timestep indices at which metrics were logged
        *_pts       – list of metric values at those timesteps
    """
    env     = mo_gym.make("deep-sea-treasure-concave-v0")
    true_pf = get_true_reference_pf()   # maximisation form

    # Separate Q-tables per objective
    Q1 = defaultdict(lambda: np.zeros(4))   # time_ret objective
    Q2 = defaultdict(lambda: np.zeros(4))   # treasure objective
    # Combined scalarised table used for action selection
    Q  = defaultdict(lambda: np.zeros(4))

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
                      else int(np.argmax(Q[state])))

            next_obs, rew, terminated, truncated, _ = env.step(action)
            global_step += 1
            done        = terminated or truncated
            next_state  = (int(next_obs[0]), int(next_obs[1]))

            treasure = float(rew[0])   # positive
            time_r   = float(rew[1])   # negative (= -1 per step)

            # Independent Q-updates per objective (standard Q-learning)
            Q1[state][action] += lr * (time_r   + gamma * np.max(Q1[next_state]) - Q1[state][action])
            Q2[state][action] += lr * (treasure + gamma * np.max(Q2[next_state]) - Q2[state][action])

            # Recompute scalarised table after each update
            for a in range(4):
                Q[state][a] = timeW * Q1[state][a] + treasureW * Q2[state][a]

            obs = next_obs

            while global_step >= next_log:
                tc, tr   = _evaluate(env, Q, n_eval)
                pt_max   = (-tc, tr)   # maximisation form

                hv_timesteps.append(next_log)
                hv_points.append(compute_hypervolume_2d([pt_max], ref_point=(-100, 0)))
                igd_timesteps.append(next_log)
                igd_points.append(compute_igd(true_pf, [pt_max]))
                eps_timesteps.append(next_log)
                eps_points.append(compute_epsilon_indicator(true_pf, [pt_max]))

                next_log += log_interval

    tc, tr = _evaluate(env, Q, n_eval)
    env.close()
    return (tc, tr), hv_timesteps, hv_points, igd_timesteps, igd_points, eps_timesteps, eps_points