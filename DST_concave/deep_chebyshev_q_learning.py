"""
Vector-valued Deep Q-Learning with weighted Chebyshev action selection.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from env import get_true_reference_pf
from deep_morl_common import train_deep_scalarized_q
from utils import extract_pareto_front, save_final_solutions


DEEP_CHEBYSHEV_WEIGHTS = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.35, 0.65),
    (0.3, 0.7),
    (0.25, 0.75),
    (0.2, 0.8),
    (0.15, 0.85),
    (0.1, 0.9),
    (0.05, 0.95),
]


def train_deep_chebyshev_q(
    chebyshev_weights,
    **training_kwargs,
):
    weights = np.asarray(chebyshev_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("Chebyshev requires exactly two weights.")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "Chebyshev weights must be positive and add to 1."
        )

    def score_function(normalized_q_vectors):
        ideal = np.ones(2, dtype=float)
        distances = np.max(
            weights * np.abs(ideal - normalized_q_vectors),
            axis=1,
        )
        return -distances

    return train_deep_scalarized_q(
        score_function=score_function,
        **training_kwargs,
    )


def _xy(points):
    return [point[0] for point in points], [point[1] for point in points]


def _coverage(true_front, learned_front, tolerance=0.5):
    covered = [
        true_point
        for true_point in true_front
        if any(
            np.linalg.norm(
                np.asarray(true_point) - np.asarray(learned_point)
            ) <= tolerance
            for learned_point in learned_front
        )
    ]
    return len(covered), len(true_front)


def _save_deep_chebyshev_plots(
    results_by_weight,
    true_front,
    final_policy_front,
    archive_front,
    final_coverage,
    coverage,
    output_dir="results/deep_chebyshev",
):
    os.makedirs(output_dir, exist_ok=True)
    final_policy_points = [
        (-result[0][0], result[0][1])
        for result in results_by_weight.values()
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
        color="#9467bd",
        s=70,
        alpha=0.85,
        label=(
            "Solutions found during training "
            f"({coverage[0]}/{coverage[1]} covered)"
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
    ax.set_title("Deep Chebyshev Q-Learning")
    ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "deep_chebyshev_pareto_front.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    metric_specs = [
        ("Hypervolume", 2, 3, "HV vs Timestep (higher = better)", "deep_chebyshev_hv.png"),
        ("IGD", 4, 5, "IGD vs Timestep (lower = better)", "deep_chebyshev_igd.png"),
        (
            "Epsilon indicator",
            6,
            7,
            "Epsilon vs Timestep (lower = better)",
            "deep_chebyshev_epsilon.png",
        ),
    ]

    for ylabel, timestep_index, value_index, title, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for weight, result in results_by_weight.items():
            timesteps = np.asarray(result[timestep_index], dtype=float)
            values = np.asarray(result[value_index], dtype=float)
            if len(timesteps) == 0:
                continue
            ax.plot(
                timesteps,
                values,
                linewidth=1.4,
                alpha=0.85,
                label=str(weight),
            )
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, title="Weights", ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    print(f"\nSaved deep Chebyshev plots in: {output_dir}")


def run_all_deep_chebyshev_weights(
    weights=DEEP_CHEBYSHEV_WEIGHTS,
    total_timesteps=200_000,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    seed=7,
    output_dir="results/deep_chebyshev",
):
    true_front = get_true_reference_pf()
    results_by_weight = {}

    for index, weight in enumerate(weights):
        print(
            f"[{index + 1}/{len(weights)}] "
            f"Chebyshev weights=({weight[0]:.2f}, {weight[1]:.2f})"
        )
        result = train_deep_chebyshev_q(
            chebyshev_weights=weight,
            total_timesteps=total_timesteps,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_timesteps=epsilon_decay_timesteps,
            log_interval=log_interval,
            n_eval=n_eval,
            seed=seed + index,
        )
        results_by_weight[weight] = result
        print(f"  final point: {result[0]}")
        print(f"  archive front size: {len(result[1])}")

    final_policy_points_max = [
        (-result[0][0], result[0][1])
        for result in results_by_weight.values()
    ]
    final_policy_front = extract_pareto_front(final_policy_points_max)
    final_coverage = _coverage(true_front, final_policy_front)

    archive_points_max = [
        (-time_cost, treasure)
        for result in results_by_weight.values()
        for time_cost, treasure in result[1]
    ]
    archive_front = extract_pareto_front(archive_points_max)
    coverage = _coverage(true_front, archive_front)

    print("\nCombined deep Chebyshev final-policy front:")
    for point in final_policy_front:
        print(f"  {point}")
    print(
        f"Final-policy coverage: "
        f"{final_coverage[0]}/{final_coverage[1]}"
    )

    print("\nCombined deep Chebyshev archive:")
    for point in archive_front:
        print(f"  {point}")
    print(f"Archive coverage: {coverage[0]}/{coverage[1]}")

    _save_deep_chebyshev_plots(
        results_by_weight,
        true_front,
        final_policy_front,
        archive_front,
        final_coverage,
        coverage,
        output_dir,
    )
    save_final_solutions(
        output_dir,
        "Deep Chebyshev Q-Learning",
        final_policy_front,
        {
            weight: result[0]
            for weight, result in results_by_weight.items()
        },
    )

    return (
        results_by_weight,
        final_policy_front,
        archive_front,
        final_coverage,
        coverage,
    )


if __name__ == "__main__":
    run_all_deep_chebyshev_weights()
