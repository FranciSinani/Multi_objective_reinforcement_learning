"""
Tabular MO Q-Learning with weighted-sum scalarisation.

Two Q-tables are learned:
    Q_time[s, a]     = expected cumulative time return
    Q_treasure[s, a] = expected cumulative treasure return

The weighted sum is used to select actions. Results are reported in two ways:
    final policy front: one final greedy point per weight
    Pareto archive: all non-dominated episode outcomes discovered in training
"""

import os
from collections import defaultdict

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


WEIGHTED_SUM_WEIGHTS = [
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


def _action_scores(
    q_time,
    q_treasure,
    state,
    time_weight,
    treasure_weight,
    ranges,
):
    time_lower, time_upper = ranges["time"]
    treasure_lower, treasure_upper = ranges["treasure"]
    normalized_time = _normalize(
        q_time[state], time_lower, time_upper
    )
    normalized_treasure = _normalize(
        q_treasure[state], treasure_lower, treasure_upper
    )
    return (
        time_weight * normalized_time
        + treasure_weight * normalized_treasure
    )


def _best_action(
    q_time,
    q_treasure,
    state,
    time_weight,
    treasure_weight,
    ranges,
    rng=None,
):
    scores = _action_scores(
        q_time,
        q_treasure,
        state,
        time_weight,
        treasure_weight,
        ranges,
    )
    best_actions = np.flatnonzero(np.isclose(scores, scores.max()))
    if rng is None:
        return int(best_actions[0])
    return int(rng.choice(best_actions))


def _save_weighted_sum_plots(
    results_by_weight,
    true_front,
    final_policy_front,
    archive_front,
    final_coverage,
    archive_coverage,
    output_dir="results/tabular_weighted_sum",
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
        color="#1f77b4",
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
    ax.set_title("Tabular Weighted-Sum Q-Learning")
    ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "weighted_sum_pareto_front.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    metric_specs = [
        (
            "Hypervolume",
            2,
            3,
            "HV vs Timestep (higher = better)",
            "weighted_sum_hv.png",
        ),
        (
            "IGD",
            4,
            5,
            "IGD vs Timestep (lower = better)",
            "weighted_sum_igd.png",
        ),
        (
            "Epsilon indicator",
            6,
            7,
            "Epsilon vs Timestep (lower = better)",
            "weighted_sum_epsilon.png",
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

    print(f"\nSaved tabular weighted-sum plots in: {output_dir}")


def _evaluate(
    env,
    q_time,
    q_treasure,
    time_weight,
    treasure_weight,
    ranges,
    n_eval=1,
):
    times, treasures = [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state = (int(obs[0]), int(obs[1]))
            action = _best_action(
                q_time,
                q_treasure,
                state,
                time_weight,
                treasure_weight,
                ranges,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated
        times.append(steps)
        treasures.append(treasure)
    return float(np.mean(times)), float(np.mean(treasures))


def train_mo_q(
    timeW,
    treasureW,
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
    Train tabular vector Q-learning with weighted-sum action selection.

    Returns:
        final_point, archive_front,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts
    """
    if (
        timeW < 0
        or treasureW < 0
        or not np.isclose(timeW + treasureW, 1.0)
    ):
        raise ValueError(
            "Weighted-sum weights must be non-negative and add to 1."
        )
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

    q_time = defaultdict(lambda: np.zeros(4))
    q_treasure = defaultdict(lambda: np.zeros(4))
    archive = []

    hv_timesteps, hv_points = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []
    global_step, next_log = 0, log_interval

    while global_step < total_timesteps:
        obs, _ = env.reset(seed=seed if global_step == 0 else None)
        done = False
        episode_steps = 0
        episode_treasure = 0.0

        while not done and global_step < total_timesteps:
            state = (int(obs[0]), int(obs[1]))
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
                    timeW,
                    treasureW,
                    ranges,
                    rng,
                )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = (int(next_obs[0]), int(next_obs[1]))
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
                    timeW,
                    treasureW,
                    ranges,
                    rng,
                )
                next_time = q_time[next_state][best_next]
                next_treasure = q_treasure[next_state][best_next]

            time_target = time_reward + gamma * next_time
            treasure_target = treasure + gamma * next_treasure

            q_time[state][action] += lr * (
                time_target - q_time[state][action]
            )
            q_treasure[state][action] += lr * (
                treasure_target - q_treasure[state][action]
            )

            obs = next_obs
            global_step += 1

            if done:
                archive.append((float(episode_steps), episode_treasure))

            while global_step >= next_log:
                time_cost, eval_treasure = _evaluate(
                    eval_env,
                    q_time,
                    q_treasure,
                    timeW,
                    treasureW,
                    ranges,
                    n_eval,
                )
                archive.append((time_cost, eval_treasure))
                archive_max = extract_pareto_front(_archive_to_max(archive))

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
        timeW,
        treasureW,
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


def run_all_weighted_sum_weights(
    weights=WEIGHTED_SUM_WEIGHTS,
    total_timesteps=200_000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    output_dir="results/tabular_weighted_sum",
):
    true_front = get_true_reference_pf()
    results_by_weight = {}

    for index, weight in enumerate(weights):
        print(
            f"[{index + 1}/{len(weights)}] "
            f"weights=({weight[0]:.2f}, {weight[1]:.2f})"
        )
        result = train_mo_q(
            timeW=weight[0],
            treasureW=weight[1],
            total_timesteps=total_timesteps,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_timesteps=epsilon_decay_timesteps,
            log_interval=log_interval,
            n_eval=n_eval,
            seed=7 + index,
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
    archive_coverage = _coverage(true_front, archive_front)

    print("\nCombined tabular weighted-sum final-policy front:")
    for point in final_policy_front:
        print(f"  {point}")
    print(
        f"Final-policy coverage: "
        f"{final_coverage[0]}/{final_coverage[1]}"
    )

    print("\nCombined tabular weighted-sum archive:")
    for point in archive_front:
        print(f"  {point}")
    print(
        f"Archive coverage: "
        f"{archive_coverage[0]}/{archive_coverage[1]}"
    )

    _save_weighted_sum_plots(
        results_by_weight,
        true_front,
        final_policy_front,
        archive_front,
        final_coverage,
        archive_coverage,
        output_dir,
    )

    save_final_solutions(
        output_dir,
        "Tabular Weighted-Sum Q-Learning",
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
        archive_coverage,
    )


if __name__ == "__main__":
    run_all_weighted_sum_weights()
