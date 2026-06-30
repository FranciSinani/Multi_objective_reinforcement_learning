"""Shared scalarization and objective-normalization helpers."""

import numpy as np


def objective_ranges(true_front):
    """Return fixed normalization ranges in maximization form."""
    return {
        # Time is represented as -time_cost, so 0 is the best possible value.
        "time": (min(point[0] for point in true_front), 0.0),
        # Treasure cannot be negative; the known front gives the max value.
        "treasure": (0.0, max(point[1] for point in true_front)),
    }


def normalize_values(values, lower, upper):
    if np.isclose(lower, upper):
        # Degenerate range: return a neutral normalized value.
        return np.full_like(values, 0.5, dtype=float)
    return np.clip(
        (np.asarray(values, dtype=float) - lower) / (upper - lower),
        0.0,
        1.0,
    )


def normalize_q_vectors(q_vectors, ranges):
    """Normalize an array shaped (n_actions, 2) as [time, treasure]."""
    q_vectors = np.asarray(q_vectors, dtype=float)
    normalized = np.empty_like(q_vectors, dtype=float)
    # Column 0 is time return, column 1 is treasure.
    for objective, key in enumerate(("time", "treasure")):
        lower, upper = ranges[key]
        normalized[:, objective] = normalize_values(
            q_vectors[:, objective], lower, upper
        )
    return normalized


def weighted_sum_scores(normalized_q_vectors, weights):
    weights = np.asarray(weights, dtype=float)
    # One dot product per action: w_time * time + w_treasure * treasure.
    return np.asarray(normalized_q_vectors, dtype=float) @ weights


def owa_scores(normalized_q_vectors, weights):
    weights = np.asarray(weights, dtype=float)
    # OWA weights the ordered values, not fixed objective names.
    ordered = np.sort(np.asarray(normalized_q_vectors, dtype=float), axis=1)
    return ordered[:, ::-1] @ weights


def chebyshev_scores(normalized_q_vectors, weights):
    weights = np.asarray(weights, dtype=float)
    # In normalized space the ideal point is (1, 1).
    ideal = np.ones(2, dtype=float)
    distances = np.max(
        weights * np.abs(ideal - np.asarray(normalized_q_vectors, dtype=float)),
        axis=1,
    )
    # The training code maximizes scores, so return negative distance.
    return -distances


def choquet_2d(time_value, treasure_value, mu_time, mu_treasure, mu_both=1.0):
    """Two-objective Choquet integral for normalized maximization values."""
    # The two-objective Choquet formula depends on the objective ordering.
    if time_value <= treasure_value:
        return time_value * mu_both + (treasure_value - time_value) * mu_treasure
    return treasure_value * mu_both + (time_value - treasure_value) * mu_time


def choquet_scores(normalized_q_vectors, capacity):
    mu_time, mu_treasure, mu_both = capacity
    # Apply the one-action Choquet formula to every action vector.
    return np.asarray([
        choquet_2d(time_value, treasure_value, mu_time, mu_treasure, mu_both)
        for time_value, treasure_value in np.asarray(normalized_q_vectors, dtype=float)
    ])
