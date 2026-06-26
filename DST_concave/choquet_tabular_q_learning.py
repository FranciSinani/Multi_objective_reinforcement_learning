"""
Clean tabular Choquet Q-Learning baseline for Deep Sea Treasure Concave.

Two Q-tables are learned:
    Q_time[s, a]     = expected cumulative time return
    Q_treasure[s, a] = expected cumulative treasure return

The objectives remain separate during learning. The Choquet integral is used
to select one common greedy action for both Bellman targets.

Results include both one final greedy-policy point per capacity and the
non-dominated outcomes found during training and periodic evaluation.
"""

from collections import defaultdict
import os

import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np

from env import get_true_reference_pf
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
    extract_pareto_front,
    save_final_solutions,
)

TABULAR_CHOQUET_CAPACITIES = [
    # Additive capacities: same preference grid as the weight-based methods.
    (0.90, 0.10, 1.0),
    (0.80, 0.20, 1.0),
    (0.70, 0.30, 1.0),
    (0.60, 0.40, 1.0),
    (0.50, 0.50, 1.0),
    (0.40, 0.60, 1.0),
    (0.35, 0.65, 1.0),
    (0.30, 0.70, 1.0),
    (0.25, 0.75, 1.0),
    (0.20, 0.80, 1.0),
    (0.15, 0.85, 1.0),
    (0.10, 0.90, 1.0),
    (0.05, 0.95, 1.0),

    # Synergy: mu1 + mu2 < 1.
    (0.10, 0.10, 1.0),
    (0.20, 0.20, 1.0),
    (0.30, 0.30, 1.0),
    (0.40, 0.40, 1.0),
    (0.10, 0.30, 1.0),
    (0.30, 0.10, 1.0),
    (0.10, 0.50, 1.0),
    (0.50, 0.10, 1.0),
    (0.20, 0.50, 1.0),
    (0.50, 0.20, 1.0),
    (0.30, 0.50, 1.0),
    (0.50, 0.30, 1.0),
]


def _validate_capacity(mu1, mu2, mu12):
    if not np.isclose(mu12, 1.0):
        raise ValueError("Normalized Choquet capacity requires mu12 = 1.0.")
    if not (0.0 <= mu1 <= 1.0 and 0.0 <= mu2 <= 1.0):
        raise ValueError("mu1 and mu2 must be in [0, 1].")


def _choquet_2d(time_value, treasure_value, mu1, mu2, mu12=1.0):
    """Two-objective Choquet integral for maximization."""
    if time_value <= treasure_value:
        return (
            time_value * mu12
            + (treasure_value - time_value) * mu2
        )
    return (
        treasure_value * mu12
        + (time_value - treasure_value) * mu1
    )


def _objective_ranges(true_front):
    """
    Fixed objective ranges in maximization form.

    The true front contains (time_return, treasure), where time_return is
    already negative. Using one fixed range keeps each capacity consistent
    across states.
    """
    return {
        "time": (min(point[0] for point in true_front), 0.0),
        "treasure": (0.0, max(point[1] for point in true_front)),
    }


def _normalize(value, lower, upper):
    if np.isclose(lower, upper):
        return 0.5
    normalized = (float(value) - lower) / (upper - lower)
    return float(np.clip(normalized, 0.0, 1.0))


def _action_scores(
    q_time,
    q_treasure,
    state,
    n_actions,
    mu1,
    mu2,
    mu12,
    ranges,
):
    time_lower, time_upper = ranges["time"]
    treasure_lower, treasure_upper = ranges["treasure"]

    scores = np.empty(n_actions, dtype=float)
    for action in range(n_actions):
        normalized_time = _normalize(
            q_time[state][action],
            time_lower,
            time_upper,
        )
        normalized_treasure = _normalize(
            q_treasure[state][action],
            treasure_lower,
            treasure_upper,
        )
        scores[action] = _choquet_2d(
            normalized_time,
            normalized_treasure,
            mu1,
            mu2,
            mu12,
        )
    return scores


def _best_action(
    q_time,
    q_treasure,
    state,
    n_actions,
    mu1,
    mu2,
    mu12,
    ranges,
    rng=None,
):
    """
    Select a Choquet-greedy action.

    Random tie-breaking is used during training so equal initial Q-values do
    not always force the same action. Evaluation uses deterministic ties.
    """
    scores = _action_scores(
        q_time,
        q_treasure,
        state,
        n_actions,
        mu1,
        mu2,
        mu12,
        ranges,
    )
    best_actions = np.flatnonzero(np.isclose(scores, scores.max()))
    if rng is None:
        return int(best_actions[0])
    return int(rng.choice(best_actions))


def _evaluate(
    env,
    q_time,
    q_treasure,
    n_actions,
    mu1,
    mu2,
    mu12,
    ranges,
    n_eval,
):
    times, treasures = [], []

    for _ in range(n_eval):
        observation, _ = env.reset()
        done = False
        steps = 0
        treasure = 0.0

        while not done:
            state = (int(observation[0]), int(observation[1]))
            action = _best_action(
                q_time,
                q_treasure,
                state,
                n_actions,
                mu1,
                mu2,
                mu12,
                ranges,
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated

        times.append(steps)
        treasures.append(treasure)

    return float(np.mean(times)), float(np.mean(treasures))


def _archive_to_max(archive):
    """Convert (time_cost, treasure) outcomes to maximization form."""
    return [
        (-float(time_cost), float(treasure))
        for time_cost, treasure in archive
    ]


def _archive_front(archive):
    """Return the non-dominated archive in training form."""
    front_max = extract_pareto_front(_archive_to_max(archive))
    return sorted(
        [(-time_return, treasure) for time_return, treasure in front_max],
        key=lambda point: point[0],
    )


def _xy(points):
    return [point[0] for point in points], [point[1] for point in points]


def _save_tabular_plots(
    results_by_capacity,
    true_front,
    final_policy_front,
    archive_front,
    final_coverage,
    archive_coverage,
    output_dir="results/tabular_choquet",
):
    """Save Pareto-front and metric plots for the tabular experiment."""
    os.makedirs(output_dir, exist_ok=True)

    all_final_points = [
        (-result[0][0], result[0][1])
        for result in results_by_capacity.values()
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
        color="#ff7f0e",
        s=65,
        alpha=0.75,
        label=(
            "Solutions found during training "
            f"({archive_coverage[0]}/{archive_coverage[1]} covered)"
        ),
        zorder=3,
    )
    ax.scatter(
        *_xy(all_final_points),
        color="#1f77b4",
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
    ax.set_title("Tabular Choquet Q-Learning")
    ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "tabular_choquet_pareto_front.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    metric_specs = [
        ("Hypervolume", 2, 3, "higher = better", "tabular_choquet_hv.png"),
        ("IGD", 4, 5, "lower = better", "tabular_choquet_igd.png"),
        (
            "Epsilon Indicator",
            6,
            7,
            "lower = better",
            "tabular_choquet_epsilon.png",
        ),
    ]

    for title, timestep_index, value_index, direction, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for capacity, result in results_by_capacity.items():
            timesteps = np.asarray(result[timestep_index], dtype=float)
            values = np.asarray(result[value_index], dtype=float)
            if len(timesteps) == 0:
                continue
            ax.plot(
                timesteps,
                values,
                linewidth=1,
                alpha=0.25,
                color="#ff7f0e",
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(title)
        ax.set_title(
            f"Tabular Choquet Q-Learning: {title} ({direction})"
        )
        ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    print(f"\nSaved tabular Choquet plots in: {output_dir}")


def train_tabular_choquet_q(
    mu1,
    mu2,
    mu12=1.0,
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
    Train tabular vector Q-learning with Choquet action selection.

    Returns:
        final_point, archive_front,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts

    final_point is the final deterministic greedy-policy result:
        (time_cost, treasure)

    archive_front contains all non-dominated episode outcomes discovered
    during training:
        [(time_cost, treasure), ...]
    """
    _validate_capacity(mu1, mu2, mu12)
    if epsilon_decay_timesteps is None:
        epsilon_decay_timesteps = total_timesteps
    if epsilon_decay_timesteps <= 0:
        raise ValueError("epsilon_decay_timesteps must be positive.")
    rng = np.random.default_rng(seed)

    env = mo_gym.make("deep-sea-treasure-concave-v0")
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    true_front = get_true_reference_pf()
    ranges = _objective_ranges(true_front)
    n_actions = env.action_space.n
    if seed is not None:
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed)

    q_time = defaultdict(lambda: np.zeros(n_actions, dtype=float))
    q_treasure = defaultdict(lambda: np.zeros(n_actions, dtype=float))
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
                    n_actions,
                    mu1,
                    mu2,
                    mu12,
                    ranges,
                    rng,
                )

            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )
            done = terminated or truncated
            next_state = (
                int(next_observation[0]),
                int(next_observation[1]),
            )
            time_reward = float(reward[1])
            treasure_reward = float(reward[0])
            episode_steps += 1
            episode_treasure = treasure_reward

            if done:
                next_time = 0.0
                next_treasure = 0.0
            else:
                next_action = _best_action(
                    q_time,
                    q_treasure,
                    next_state,
                    n_actions,
                    mu1,
                    mu2,
                    mu12,
                    ranges,
                    rng,
                )
                next_time = q_time[next_state][next_action]
                next_treasure = q_treasure[next_state][next_action]

            time_target = time_reward + gamma * next_time
            treasure_target = treasure_reward + gamma * next_treasure

            q_time[state][action] += lr * (
                time_target - q_time[state][action]
            )
            q_treasure[state][action] += lr * (
                treasure_target - q_treasure[state][action]
            )

            observation = next_observation
            global_step += 1

            if done:
                archive.append(
                    (float(episode_steps), float(episode_treasure))
                )

            while global_step >= next_log:
                time_cost, treasure = _evaluate(
                    eval_env,
                    q_time,
                    q_treasure,
                    n_actions,
                    mu1,
                    mu2,
                    mu12,
                    ranges,
                    n_eval,
                )
                archive.append((time_cost, treasure))
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
        n_actions,
        mu1,
        mu2,
        mu12,
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


def run_all_tabular_capacities(
    capacities=TABULAR_CHOQUET_CAPACITIES,
    total_timesteps=200_000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    seed=7,
    output_dir="results/tabular_choquet",
):
    """
    Train one tabular Choquet policy per capacity and combine final points.

    Returns:
        results_by_capacity, learned_front, coverage
    """
    true_front = get_true_reference_pf()
    results_by_capacity = {}

    for index, capacity in enumerate(capacities):
        mu1, mu2, mu12 = capacity
        print(
            f"[{index + 1}/{len(capacities)}] "
            f"capacity=({mu1:.2f}, {mu2:.2f}, {mu12:.1f})"
        )

        result = train_tabular_choquet_q(
            mu1=mu1,
            mu2=mu2,
            mu12=mu12,
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
        results_by_capacity[capacity] = result
        print(f"  final point: {result[0]}")
        print(f"  archived front size: {len(result[1])}")

    final_policy_points_max = [
        (-result[0][0], result[0][1])
        for result in results_by_capacity.values()
    ]
    final_policy_front = extract_pareto_front(final_policy_points_max)

    archived_points_max = [
        (-time_cost, treasure)
        for result in results_by_capacity.values()
        for time_cost, treasure in result[1]
    ]
    archive_front = extract_pareto_front(archived_points_max)

    tolerance = 0.5
    final_covered_points = [
        true_point
        for true_point in true_front
        if any(
            np.linalg.norm(
                np.asarray(true_point) - np.asarray(learned_point)
            ) <= tolerance
            for learned_point in final_policy_front
        )
    ]
    archive_covered_points = [
        true_point
        for true_point in true_front
        if any(
            np.linalg.norm(
                np.asarray(true_point) - np.asarray(learned_point)
            ) <= tolerance
            for learned_point in archive_front
        )
    ]

    final_coverage = (len(final_covered_points), len(true_front))
    archive_coverage = (len(archive_covered_points), len(true_front))

    print("\nCombined final-policy front:")
    for point in final_policy_front:
        print(f"  {point}")
    print(
        f"Final-policy coverage: "
        f"{final_coverage[0]}/{final_coverage[1]}"
    )

    print("\nCombined Pareto archive:")
    for point in archive_front:
        print(f"  {point}")
    print(
        f"Archive coverage: "
        f"{archive_coverage[0]}/{archive_coverage[1]}"
    )

    _save_tabular_plots(
        results_by_capacity,
        true_front,
        final_policy_front,
        archive_front,
        final_coverage,
        archive_coverage,
        output_dir,
    )
    save_final_solutions(
        output_dir,
        "Tabular Choquet Q-Learning",
        final_policy_front,
        {
            capacity: result[0]
            for capacity, result in results_by_capacity.items()
        },
        preference_label="capacity",
    )

    return (
        results_by_capacity,
        final_policy_front,
        archive_front,
        final_coverage,
        archive_coverage,
    )


if __name__ == "__main__":
    run_all_tabular_capacities()
