"""Grouped deep scalarized Q-learning methods.

This module defines the method wrappers and shared experiment runner for Deep Weighted Sum, OWA, Chebyshev,
and Choquet so the deep methods are configured in one place.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    CHOQUET_CAPACITIES,
    DEEP_LR,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LOG_INTERVAL,
    OWA_EXPERIMENT_SEED,
    TIMESTEPS,
    WEIGHTS,
)
from deep_morl_common import train_deep_scalarized_q
from env import get_true_reference_pf
from scalarization import (
    chebyshev_scores,
    choquet_scores,
    owa_scores,
    weighted_sum_scores,
)
from utils import extract_pareto_front, save_final_solutions


DEEP_WEIGHTED_SUM_WEIGHTS = WEIGHTS
DEEP_OWA_WEIGHTS = WEIGHTS
DEEP_CHEBYSHEV_WEIGHTS = WEIGHTS
DEEP_CHOQUET_CAPACITIES = CHOQUET_CAPACITIES
DEEP_OWA_EXPERIMENT_SEED = OWA_EXPERIMENT_SEED


def train_deep_weighted_sum_q(time_weight, treasure_weight, **training_kwargs):
    # Method wrapper: validate weights, build the score function, then use
    # the shared deep DQN trainer.
    if time_weight < 0 or treasure_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(time_weight + treasure_weight, 1.0):
        raise ValueError("Weighted-sum weights must add to 1.")

    weights = np.asarray((time_weight, treasure_weight), dtype=float)

    def score_function(normalized_q_vectors):
        return weighted_sum_scores(normalized_q_vectors, weights)

    return train_deep_scalarized_q(score_function, **training_kwargs)


def train_deep_owa_q(owa_weights, **training_kwargs):
    # OWA weights apply to ordered objective values, not fixed objective names.
    weights = np.asarray(owa_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("OWA requires exactly two weights.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("OWA weights must be non-negative and add to 1.")

    def score_function(normalized_q_vectors):
        return owa_scores(normalized_q_vectors, weights)

    return train_deep_scalarized_q(score_function, **training_kwargs)


def train_deep_chebyshev_q(chebyshev_weights, **training_kwargs):
    # Chebyshev weights control the distance from the normalized ideal point.
    weights = np.asarray(chebyshev_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("Chebyshev requires exactly two weights.")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Chebyshev weights must be positive and add to 1.")

    def score_function(normalized_q_vectors):
        return chebyshev_scores(normalized_q_vectors, weights)

    return train_deep_scalarized_q(score_function, **training_kwargs)


def train_deep_choquet_q(mu1, mu2, mu12, **training_kwargs):
    # Choquet capacities allow non-additive interaction between objectives.
    if not np.isclose(mu12, 1.0):
        raise ValueError("The normalized capacity requires mu12 = 1.0.")
    if not (0.0 <= mu1 <= mu12 and 0.0 <= mu2 <= mu12):
        raise ValueError("Capacity must satisfy 0 <= mu1, mu2 <= mu12 = 1.")

    capacity = (float(mu1), float(mu2), float(mu12))

    def score_function(normalized_q_vectors):
        return choquet_scores(normalized_q_vectors, capacity)

    return train_deep_scalarized_q(score_function, **training_kwargs)


def _xy(points):
    # Convenience helper for matplotlib scatter/plot calls.
    return [point[0] for point in points], [point[1] for point in points]


def _coverage(true_front, learned_front, tolerance=0.5):
    # Count true Pareto points recovered within a small numerical tolerance.
    covered = [
        true_point
        for true_point in true_front
        if any(
            np.linalg.norm(np.asarray(true_point) - np.asarray(point))
            <= tolerance
            for point in learned_front
        )
    ]
    return len(covered), len(true_front)


def _save_deep_plots(
    results_by_preference,
    true_front,
    final_policy_front,
    archive_front,
    final_coverage,
    archive_coverage,
    output_dir,
    method_title,
    filename_prefix,
    preference_title,
    archive_color,
):
    os.makedirs(output_dir, exist_ok=True)
    # Final points are returned as (time_cost, treasure); plot as
    # (-time_cost, treasure).
    final_policy_points = [
        (-result[0][0], result[0][1])
        for result in results_by_preference.values()
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        *_xy(true_front),
        color="#555555",
        marker="^",
        s=65,
        facecolors="none",
        linewidths=1.6,
        label="True Pareto solutions",
    )
    ax.scatter(
        *_xy(archive_front),
        color=archive_color,
        s=70,
        alpha=0.85,
        label=(
            "Solutions found during training "
            f"({archive_coverage[0]}/{archive_coverage[1]} covered)"
        ),
        zorder=3,
    )
    ax.scatter(
        *_xy(final_policy_points),
        color="#d62728",
        marker="x",
        s=110,
        linewidth=2.5,
        label=(
            "Final policies solutions "
            f"({final_coverage[0]}/{final_coverage[1]} covered)"
        ),
        zorder=4,
    )
    ax.set_xlabel("- Time Cost")
    ax.set_ylabel("Treasure Value")
    ax.set_title(method_title)
    ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{filename_prefix}_pareto_front.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    metric_specs = [
        # result tuple indices: (final, archive, hv_ts, hv, igd_ts, igd, eps_ts, eps)
        ("Hypervolume", 2, 3, "HV vs Timestep (higher = better)", "hv"),
        ("IGD", 4, 5, "IGD vs Timestep (lower = better)", "igd"),
        (
            "Epsilon indicator",
            6,
            7,
            "Epsilon vs Timestep (lower = better)",
            "epsilon",
        ),
    ]

    for ylabel, timestep_index, value_index, title, suffix in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for preference, result in results_by_preference.items():
            timesteps = np.asarray(result[timestep_index], dtype=float)
            values = np.asarray(result[value_index], dtype=float)
            if len(timesteps) == 0:
                continue
            ax.plot(
                timesteps,
                values,
                linewidth=1.4,
                alpha=0.85,
                label=str(preference),
            )
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, title=preference_title, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"{filename_prefix}_{suffix}.png"),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    print(f"\nSaved {method_title} plots in: {output_dir}")


def _run_all_deep_preferences(
    preferences,
    train_one,
    method_name,
    method_title,
    filename_prefix,
    output_dir,
    preference_label="weights",
    preference_title="Weights",
    preference_prefix="weights",
    total_timesteps=TIMESTEPS,
    lr=DEEP_LR,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    log_interval=LOG_INTERVAL,
    n_eval=1,
    seed=7,
    seed_mode="offset",
    archive_color="#1f77b4",
):
    true_front = get_true_reference_pf()
    results_by_preference = {}

    # Train one independent DQN policy for each weight/capacity setting.
    for index, preference in enumerate(preferences):
        print(
            f"[{index + 1}/{len(preferences)}] "
            f"{preference_prefix}={tuple(preference)}"
        )
        # Different seeds avoid making every preference follow the same random
        # exploration path.
        run_seed = seed * 100 + index if seed_mode == "scaled" else seed + index
        result = train_one(
            preference,
            total_timesteps=total_timesteps,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            log_interval=log_interval,
            n_eval=n_eval,
            seed=run_seed,
        )
        results_by_preference[tuple(preference)] = result
        print(f"  final point: {result[0]}")
        print(f"  archive front size: {len(result[1])}")

    # Combine final policy outcomes across all preferences.
    final_policy_points_max = [
        (-result[0][0], result[0][1])
        for result in results_by_preference.values()
    ]
    final_policy_front = extract_pareto_front(final_policy_points_max)
    final_coverage = _coverage(true_front, final_policy_front)

    # Combine all discovered archive fronts across all preferences.
    archive_points_max = [
        (-time_cost, treasure)
        for result in results_by_preference.values()
        for time_cost, treasure in result[1]
    ]
    archive_front = extract_pareto_front(archive_points_max)
    archive_coverage = _coverage(true_front, archive_front)

    print(f"\nCombined {method_name} final-policy front:")
    for point in final_policy_front:
        print(f"  {point}")
    print(
        f"Final-policy coverage: "
        f"{final_coverage[0]}/{final_coverage[1]}"
    )

    print(f"\nCombined {method_name} archive:")
    for point in archive_front:
        print(f"  {point}")
    print(
        f"Archive coverage: "
        f"{archive_coverage[0]}/{archive_coverage[1]}"
    )

    _save_deep_plots(
        results_by_preference,
        true_front,
        final_policy_front,
        archive_front,
        final_coverage,
        archive_coverage,
        output_dir,
        method_title,
        filename_prefix,
        preference_title,
        archive_color,
    )
    # JSON stores the final front and preference -> final point mapping.
    save_final_solutions(
        output_dir,
        method_title,
        final_policy_front,
        {
            preference: result[0]
            for preference, result in results_by_preference.items()
        },
        preference_label=preference_label,
    )

    return (
        results_by_preference,
        final_policy_front,
        archive_front,
        final_coverage,
        archive_coverage,
    )


def run_all_deep_weighted_sum_weights(
    weights=DEEP_WEIGHTED_SUM_WEIGHTS,
    **kwargs,
):
    # Public runner used by main.py for `deep-ws`.
    def train_one(weight, **training_kwargs):
        return train_deep_weighted_sum_q(
            time_weight=weight[0],
            treasure_weight=weight[1],
            **training_kwargs,
        )

    return _run_all_deep_preferences(
        weights,
        train_one,
        method_name="deep weighted-sum",
        method_title="Deep Weighted-Sum Q-Learning",
        filename_prefix="deep_weighted_sum",
        output_dir=kwargs.pop("output_dir", "results/deep_weighted_sum"),
        archive_color="#1f77b4",
        **kwargs,
    )


def run_all_deep_owa_weights(
    weights=DEEP_OWA_WEIGHTS,
    seed=DEEP_OWA_EXPERIMENT_SEED,
    **kwargs,
):
    # Public runner used by main.py for `deep-owa`.
    def train_one(weight, **training_kwargs):
        return train_deep_owa_q(owa_weights=weight, **training_kwargs)

    return _run_all_deep_preferences(
        weights,
        train_one,
        method_name="deep OWA",
        method_title="Deep OWA Q-Learning",
        filename_prefix="deep_owa",
        output_dir=kwargs.pop("output_dir", "results/deep_owa"),
        seed=seed,
        seed_mode="scaled",
        archive_color="#2ca02c",
        **kwargs,
    )


def run_all_deep_chebyshev_weights(
    weights=DEEP_CHEBYSHEV_WEIGHTS,
    **kwargs,
):
    # Public runner used by main.py for `deep-cheb`.
    def train_one(weight, **training_kwargs):
        return train_deep_chebyshev_q(
            chebyshev_weights=weight, **training_kwargs
        )

    return _run_all_deep_preferences(
        weights,
        train_one,
        method_name="deep Chebyshev",
        method_title="Deep Chebyshev Q-Learning",
        filename_prefix="deep_chebyshev",
        output_dir=kwargs.pop("output_dir", "results/deep_chebyshev"),
        archive_color="#9467bd",
        **kwargs,
    )


def run_all_deep_choquet_capacities(
    capacities=DEEP_CHOQUET_CAPACITIES,
    **kwargs,
):
    # Public runner used by main.py for `deep-choquet`.
    def train_one(capacity, **training_kwargs):
        return train_deep_choquet_q(
            mu1=capacity[0],
            mu2=capacity[1],
            mu12=capacity[2],
            **training_kwargs,
        )

    return _run_all_deep_preferences(
        capacities,
        train_one,
        method_name="deep Choquet",
        method_title="Deep Choquet Q-Learning",
        filename_prefix="deep_choquet",
        output_dir=kwargs.pop("output_dir", "results/deep_choquet"),
        preference_label="capacity",
        preference_title="Capacities",
        preference_prefix="capacity",
        archive_color="#ff7f0e",
        **kwargs,
    )
