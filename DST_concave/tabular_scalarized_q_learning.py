"""Shared tabular scalarized Q-learning for Deep Sea Treasure.

Weighted Sum, OWA, Chebyshev, and Choquet use the same vector-valued
Q-learning loop. They differ only in the scalarization function that ranks
normalized objective Q-vectors for current action selection and for choosing
the next action used by the Bellman backup. Bellman targets still use raw
rewards and raw Q-values.
"""

from collections import defaultdict

import mo_gymnasium as mo_gym
import numpy as np

from config import (
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LOG_INTERVAL,
    TABULAR_LR,
    TIMESTEPS,
)
from env import get_true_reference_pf
from scalarization import (
    chebyshev_scores,
    choquet_scores,
    normalize_q_vectors,
    objective_ranges,
    owa_scores,
    weighted_sum_scores,
)
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
    extract_pareto_front,
)


def _archive_to_max(archive):
    # Training/evaluation outcomes are stored as (time_cost, treasure).
    # Metrics use maximization form, so time_cost becomes -time_cost.
    return [
        (-float(time_cost), float(treasure))
        for time_cost, treasure in archive
    ]


def _archive_front(archive):
    # Keep only nondominated discovered outcomes, then convert back to
    # (time_cost, treasure) for plotting/saving functions that expect it.
    front_max = extract_pareto_front(_archive_to_max(archive))
    return sorted(
        [(-time_return, treasure) for time_return, treasure in front_max],
        key=lambda point: point[0],
    )


def _q_vectors_for_state(q_time, q_treasure, state):
    # Build one vector per action: [Q_time(s,a), Q_treasure(s,a)].
    return np.column_stack((q_time[state], q_treasure[state]))


def _best_action(q_time, q_treasure, state, score_function, ranges, rng=None):
    q_vectors = _q_vectors_for_state(q_time, q_treasure, state)
    # Scalarization is applied after normalization so objectives are comparable.
    normalized = normalize_q_vectors(q_vectors, ranges)
    scores = score_function(normalized)
    # Random tie-breaking avoids always favoring the lowest action index.
    best_actions = np.flatnonzero(np.isclose(scores, scores.max()))
    if rng is None:
        return int(best_actions[0])
    return int(rng.choice(best_actions))


def _evaluate(
    env,
    q_time,
    q_treasure,
    score_function,
    ranges,
    n_eval=1,
):
    # Greedy evaluation: no exploration, only the learned scalarized policy.
    times, treasures = [], []
    for _ in range(n_eval):
        observation, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0

        while not done:
            state = (int(observation[0]), int(observation[1]))
            action = _best_action(
                q_time, q_treasure, state, score_function, ranges
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            # In DST the terminal reward carries the treasure value.
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated

        times.append(steps)
        treasures.append(treasure)

    return float(np.mean(times)), float(np.mean(treasures))


def train_tabular_scalarized_q(
    score_function,
    total_timesteps=TIMESTEPS,
    lr=TABULAR_LR,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    log_interval=LOG_INTERVAL,
    n_eval=1,
    seed=None,
):
    """Train one tabular vector Q-policy using a scalarized action score."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")

    rng = np.random.default_rng(seed)
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    if seed is not None:
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed)

    true_front = get_true_reference_pf()
    ranges = objective_ranges(true_front)

    # Two Q-tables are learned: one for each objective.
    q_time = defaultdict(lambda: np.zeros(4))
    q_treasure = defaultdict(lambda: np.zeros(4))
    # Archive stores episode/evaluation outcomes as raw (time_cost, treasure).
    archive = []

    hv_timesteps, hv_points = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []
    global_step, next_log = 0, log_interval

    while global_step < total_timesteps:
        observation, _ = env.reset(seed=seed if global_step == 0 else None)
        done = False
        episode_steps = 0
        episode_treasure = 0.0

        while not done and global_step < total_timesteps:
            state = (int(observation[0]), int(observation[1]))
            # Linear epsilon decay over the full training budget.
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end)
                * global_step / total_timesteps,
            )
            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                # Exploitation uses the method-specific scalarization.
                action = _best_action(
                    q_time, q_treasure, state, score_function, ranges, rng
                )

            next_observation, reward, terminated, truncated, _ = env.step(action)
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
                # Terminal transition: no future value is bootstrapped.
                next_time = 0.0
                next_treasure = 0.0
            else:
                # The same scalarization-selected next action is used for
                # both objectives, keeping the backup tied to one policy.
                best_next = _best_action(
                    q_time,
                    q_treasure,
                    next_state,
                    score_function,
                    ranges,
                    rng,
                )
                next_time = q_time[next_state][best_next]
                next_treasure = q_treasure[next_state][best_next]

            # Vector Bellman targets, one target per objective.
            time_target = time_reward + gamma * next_time
            treasure_target = treasure + gamma * next_treasure

            # Standard tabular Q-learning update applied separately per
            # objective, using the same transition and next action.
            q_time[state][action] += lr * (
                time_target - q_time[state][action]
            )
            q_treasure[state][action] += lr * (
                treasure_target - q_treasure[state][action]
            )

            observation = next_observation
            global_step += 1

            if done:
                # Keep actual episode outcomes as discovered solutions.
                archive.append((float(episode_steps), episode_treasure))

            while global_step >= next_log:
                # Periodic greedy evaluation tracks progress during training.
                time_cost, eval_treasure = _evaluate(
                    eval_env,
                    q_time,
                    q_treasure,
                    score_function,
                    ranges,
                    n_eval,
                )
                archive.append((time_cost, eval_treasure))
                archive_max = extract_pareto_front(_archive_to_max(archive))

                # Metrics are always computed on the nondominated archive in
                # maximization form: (-time_cost, treasure).
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

    # Final reported point is a raw environment outcome, not a scalar score.
    final_point = _evaluate(
        eval_env, q_time, q_treasure, score_function, ranges, n_eval
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


def train_mo_q(timeW, treasureW, **training_kwargs):
    # Wrapper: define Weighted Sum scoring, then reuse the shared trainer.
    if (
        timeW < 0
        or treasureW < 0
        or not np.isclose(timeW + treasureW, 1.0)
    ):
        raise ValueError(
            "Weighted-sum weights must be non-negative and add to 1."
        )
    weights = np.asarray((timeW, treasureW), dtype=float)

    def score_function(normalized_q_vectors):
        return weighted_sum_scores(normalized_q_vectors, weights)

    return train_tabular_scalarized_q(score_function, **training_kwargs)


def train_owa_q(owa_weights, **training_kwargs):
    # Wrapper: define OWA scoring, then reuse the shared trainer.
    weights = np.asarray(owa_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("OWA requires exactly two weights.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("OWA weights must be non-negative and add to 1.")

    def score_function(normalized_q_vectors):
        return owa_scores(normalized_q_vectors, weights)

    return train_tabular_scalarized_q(score_function, **training_kwargs)


def train_chebyshev_q(chebyshev_weights, **training_kwargs):
    # Wrapper: define Chebyshev scoring, then reuse the shared trainer.
    weights = np.asarray(chebyshev_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("Chebyshev requires exactly two weights.")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "Chebyshev weights must be positive and add to 1."
        )

    def score_function(normalized_q_vectors):
        return chebyshev_scores(normalized_q_vectors, weights)

    return train_tabular_scalarized_q(score_function, **training_kwargs)


def train_tabular_choquet_q(mu1, mu2, mu12, **training_kwargs):
    # Wrapper: define Choquet scoring, then reuse the shared trainer.
    if not np.isclose(mu12, 1.0):
        raise ValueError("Normalized Choquet capacity requires mu12 = 1.0.")
    if not (0.0 <= mu1 <= 1.0 and 0.0 <= mu2 <= 1.0):
        raise ValueError("mu1 and mu2 must be in [0, 1].")
    capacity = (float(mu1), float(mu2), float(mu12))

    def score_function(normalized_q_vectors):
        return choquet_scores(normalized_q_vectors, capacity)

    return train_tabular_scalarized_q(score_function, **training_kwargs)
