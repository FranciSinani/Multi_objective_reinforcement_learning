"""
OWA Q-Learning: Ordered Weighted Averaging scalarisation.

OWA sorts objective values before applying weights,
so the weight attaches to the rank (best, worst) rather than a fixed objective.
This produces more balanced solutions and can reach some concave front regions.

OWA([v1, v2], [w1, w2]) = w1 * max(v1, v2) + w2 * min(v1, v2)

Coordinate convention
    rew[0] = treasure  (positive, maximise)
    rew[1] = time_ret  (negative = -steps, maximise)
    Metrics logged in maximisation form: (-steps, treasure).
"""

from collections import defaultdict

import mo_gymnasium as mo_gym
import numpy as np

from env import get_true_reference_pf
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
    extract_pareto_front,
)


def _archive_to_max(archive):
    return [
        (-float(time_cost), float(treasure))
        for time_cost, treasure in archive
    ]


def _archive_front(archive):
    front_max = extract_pareto_front(_archive_to_max(archive))
    return sorted(
        [(-time_return, treasure) for time_return, treasure in front_max],
        key=lambda point: point[0],
    )


def _owa(values, weights):
    """
    Sort values descending, then apply weights by rank.

    weights[0] applies to the largest value and weights[1] to the smallest.
    """
    return sum(w * v for w, v in zip(weights, sorted(values, reverse=True)))


def _objective_ranges(true_front):
    return {
        "time": (min(point[0] for point in true_front), 0.0),
        "treasure": (0.0, max(point[1] for point in true_front)),
    }


def _normalize(values, lower, upper):
    if np.isclose(lower, upper):
        return np.full_like(values, 0.5, dtype=float)
    return np.clip(
        (np.asarray(values, dtype=float) - lower) / (upper - lower),
        0.0,
        1.0,
    )


def _action_scores(q_time, q_treasure, state, weights, ranges):
    time_values = _normalize(q_time[state], *ranges["time"])
    treasure_values = _normalize(q_treasure[state], *ranges["treasure"])
    return np.asarray([
        _owa([time_values[action], treasure_values[action]], weights)
        for action in range(len(time_values))
    ])


def _best_action(q_time, q_treasure, state, weights, ranges, rng=None):
    """Return an OWA-greedy action, with optional random tie-breaking."""
    scores = _action_scores(
        q_time, q_treasure, state, weights, ranges
    )
    best_actions = np.flatnonzero(np.isclose(scores, scores.max()))
    if rng is None:
        return int(best_actions[0])
    return int(rng.choice(best_actions))


def _evaluate(env, q_time, q_treasure, weights, ranges, n_eval=1):
    """Run the greedy OWA policy and return mean time cost and treasure."""
    times, treasures = [], []
    for _ in range(n_eval):
        observation, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0

        while not done:
            state = (int(observation[0]), int(observation[1]))
            action = _best_action(
                q_time, q_treasure, state, weights, ranges
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated

        times.append(steps)
        treasures.append(treasure)

    return float(np.mean(times)), float(np.mean(treasures))


def train_owa_q(
    owa_weights,
    total_timesteps=200_000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    seed=None,
):
    """
    Train OWA Q-Learning with a fixed OWA weight vector.

    Returns:
        final_point, archive_front,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts
    """
    if (
        len(owa_weights) != 2
        or any(weight < 0 for weight in owa_weights)
        or not np.isclose(sum(owa_weights), 1.0)
    ):
        raise ValueError("OWA weights must be non-negative and add to 1.")
    if epsilon_decay_timesteps is None:
        epsilon_decay_timesteps = total_timesteps
    if epsilon_decay_timesteps <= 0:
        raise ValueError("epsilon_decay_timesteps must be positive.")

    rng = np.random.default_rng(seed)
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    if seed is not None:
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed)
    true_front = get_true_reference_pf()
    ranges = _objective_ranges(true_front)
    n_actions = env.action_space.n

    q_time = defaultdict(lambda: np.zeros(n_actions))
    q_treasure = defaultdict(lambda: np.zeros(n_actions))
    archive = []

    hv_timesteps, hv_points = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []
    global_step, next_log = 0, log_interval

    while global_step < total_timesteps:
        observation, _ = env.reset(
            seed=seed if global_step == 0 else None
        )
        done = False
        episode_steps = 0
        episode_treasure = 0.0

        while not done and global_step < total_timesteps:
            state = (int(observation[0]), int(observation[1]))
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * global_step / epsilon_decay_timesteps,
            )

            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = _best_action(
                    q_time,
                    q_treasure,
                    state,
                    owa_weights,
                    ranges,
                    rng,
                )

            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )
            global_step += 1
            done = terminated or truncated
            next_state = (
                int(next_observation[0]),
                int(next_observation[1]),
            )

            treasure = float(reward[0])
            time_reward = float(reward[1])
            episode_steps += 1
            episode_treasure = treasure
            if done:
                next_time = 0.0
                next_treasure = 0.0
            else:
                best_next = _best_action(
                    q_time,
                    q_treasure,
                    next_state,
                    owa_weights,
                    ranges,
                    rng,
                )
                next_time = q_time[next_state][best_next]
                next_treasure = q_treasure[next_state][best_next]

            q_time[state][action] += lr * (
                time_reward + gamma * next_time - q_time[state][action]
            )
            q_treasure[state][action] += lr * (
                treasure
                + gamma * next_treasure
                - q_treasure[state][action]
            )

            observation = next_observation

            if done:
                archive.append((
                    float(episode_steps),
                    float(episode_treasure),
                ))

            while global_step >= next_log:
                time_cost, eval_treasure = _evaluate(
                    eval_env,
                    q_time,
                    q_treasure,
                    owa_weights,
                    ranges,
                    n_eval,
                )
                archive.append((time_cost, eval_treasure))
                archive_max = extract_pareto_front(
                    _archive_to_max(archive)
                )

                hv_timesteps.append(next_log)
                hv_points.append(compute_hypervolume_2d(
                    archive_max, ref_point=(-100, 0)
                ))
                igd_timesteps.append(next_log)
                igd_points.append(compute_igd(true_front, archive_max))
                eps_timesteps.append(next_log)
                eps_points.append(compute_epsilon_indicator(
                    true_front, archive_max
                ))
                next_log += log_interval

    final_point = _evaluate(
        eval_env,
        q_time,
        q_treasure,
        owa_weights,
        ranges,
        n_eval,
    )
    archive.append(final_point)
    archive_front = _archive_front(archive)
    env.close()
    eval_env.close()

    return (
        final_point,
        archive_front,
        hv_timesteps,
        hv_points,
        igd_timesteps,
        igd_points,
        eps_timesteps,
        eps_points,
    )
