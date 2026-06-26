"""
Deep Choquet Q-Learning for Deep Sea Treasure Concave.

The neural network predicts a two-objective Q-vector for every action:
    Q(state, action) = [time_return, treasure_return]

Choquet scalarisation is used only to rank actions. Learning remains
vector-valued, with one TD target per objective.

Capacity interpretation with mu12 fixed to 1:
    mu1 + mu2 < 1  -> synergy / complementarity
    mu1 + mu2 = 1  -> additive case
    mu1 + mu2 > 1  -> redundancy / limited compensation

Metrics and the Pareto archive use maximisation form:
    (-time_cost, treasure)
"""

from collections import deque
import os
import random

import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np
import torch
from torch import nn

from env import get_true_reference_pf
from utils import (
    compute_epsilon_indicator,
    compute_hypervolume_2d,
    compute_igd,
    extract_pareto_front,
    save_final_solutions,
)


DEEP_CHOQUET_CAPACITIES = [
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


class VectorQNetwork(nn.Module):
    """Predict a two-objective Q-vector for every action."""

    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * 2),
        )

    def forward(self, states):
        values = self.network(states)
        return values.view(-1, self.n_actions, 2)


class ReplayBuffer:
    def __init__(self, capacity, rng=None):
        self.data = deque(maxlen=capacity)
        self.rng = rng or random.Random()

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = self.rng.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.data)


def _validate_capacity(mu1, mu2, mu12):
    if not np.isclose(mu12, 1.0):
        raise ValueError("The normalized capacity requires mu12 = 1.0.")
    if not (0.0 <= mu1 <= mu12 and 0.0 <= mu2 <= mu12):
        raise ValueError("Capacity must satisfy 0 <= mu1, mu2 <= mu12 = 1.")


def _choquet_2d(values, mu1, mu2, mu12=1.0):
    """Two-objective Choquet integral for maximisation."""
    x1, x2 = values
    if x1 <= x2:
        return x1 * mu12 + (x2 - x1) * mu2
    return x2 * mu12 + (x1 - x2) * mu1


def _objective_ranges(true_pf):
    return {
        "time": (min(pt[0] for pt in true_pf), 0.0),
        "treasure": (0.0, max(pt[1] for pt in true_pf)),
    }


def _normalise(values, lower, upper):
    if np.isclose(lower, upper):
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _grid_shape(obs_low, obs_high):
    return (
        np.asarray(obs_high, dtype=int)
        - np.asarray(obs_low, dtype=int)
        + 1
    )


def _encode_state(obs, obs_low, obs_high):
    shape = _grid_shape(obs_low, obs_high)
    coordinates = (
        np.asarray(obs, dtype=int)
        - np.asarray(obs_low, dtype=int)
    )
    index = np.ravel_multi_index(tuple(coordinates), tuple(shape))
    encoded = np.zeros(int(np.prod(shape)), dtype=np.float32)
    encoded[index] = 1.0
    return encoded


def _choquet_scores(q_vectors, mu1, mu2, mu12, ranges):
    """Return one Choquet score per action from [action, objective] values."""
    q_vectors = np.asarray(q_vectors, dtype=float)
    time_lo, time_hi = ranges["time"]
    treasure_lo, treasure_hi = ranges["treasure"]
    time_values = _normalise(q_vectors[:, 0], time_lo, time_hi)
    treasure_values = _normalise(q_vectors[:, 1], treasure_lo, treasure_hi)
    return np.array([
        _choquet_2d((time_values[a], treasure_values[a]), mu1, mu2, mu12)
        for a in range(len(q_vectors))
    ])


def _best_action(network, state, mu1, mu2, mu12, ranges, device):
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        q_vectors = network(state_t.unsqueeze(0))[0].cpu().numpy()
    return int(np.argmax(_choquet_scores(
        q_vectors, mu1, mu2, mu12, ranges
    )))


def _best_actions_batch(network, states, mu1, mu2, mu12, ranges):
    """Choquet-greedy actions for a batch of states."""
    with torch.no_grad():
        q_batch = network(states).cpu().numpy()
    actions = [
        int(np.argmax(_choquet_scores(q, mu1, mu2, mu12, ranges)))
        for q in q_batch
    ]
    return torch.as_tensor(actions, dtype=torch.long, device=states.device)


def _evaluate(
    env,
    network,
    mu1,
    mu2,
    mu12,
    ranges,
    obs_low,
    obs_high,
    device,
    n_eval=1,
):
    times, treasures = [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state = _encode_state(obs, obs_low, obs_high)
            action = _best_action(
                network, state, mu1, mu2, mu12, ranges, device
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated
        times.append(steps)
        treasures.append(treasure)
    return float(np.mean(times)), float(np.mean(treasures))


def _archive_to_max(archive):
    return [(-float(time_cost), float(treasure))
            for time_cost, treasure in archive]


def _archive_front(archive):
    front_max = extract_pareto_front(_archive_to_max(archive))
    return sorted([(-time_ret, treasure) for time_ret, treasure in front_max])


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


def _soft_update(target, online, tau):
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(online_parameter, alpha=tau)


def _checkpoint_score(time_cost, treasure, mu1, mu2, mu12, ranges):
    time_lower, time_upper = ranges["time"]
    treasure_lower, treasure_upper = ranges["treasure"]
    normalized_time = float(_normalise(
        np.array([-float(time_cost)]),
        time_lower,
        time_upper,
    )[0])
    normalized_treasure = float(_normalise(
        np.array([float(treasure)]),
        treasure_lower,
        treasure_upper,
    )[0])
    return float(_choquet_2d(
        (normalized_time, normalized_treasure),
        mu1,
        mu2,
        mu12,
    ))


def _copy_state_dict_to_cpu(network):
    return {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }


def _save_deep_choquet_plots(
    results_by_capacity,
    true_front,
    final_policy_front,
    archive_front,
    final_coverage,
    archive_coverage,
    output_dir="results/deep_choquet",
):
    os.makedirs(output_dir, exist_ok=True)
    final_policy_points = [
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
    ax.set_title("Deep Choquet Q-Learning")
    ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "deep_choquet_pareto_front.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    metric_specs = [
        ("Hypervolume", 2, 3, "HV vs Timestep (higher = better)", "deep_choquet_hv.png"),
        ("IGD", 4, 5, "IGD vs Timestep (lower = better)", "deep_choquet_igd.png"),
        (
            "Epsilon indicator",
            6,
            7,
            "Epsilon vs Timestep (lower = better)",
            "deep_choquet_epsilon.png",
        ),
    ]

    for ylabel, timestep_index, value_index, title, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for capacity, result in results_by_capacity.items():
            timesteps = np.asarray(result[timestep_index], dtype=float)
            values = np.asarray(result[value_index], dtype=float)
            if len(timesteps) == 0:
                continue
            ax.plot(
                timesteps,
                values,
                linewidth=1.4,
                alpha=0.85,
                label=str(capacity),
            )
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, title="Capacities", ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    print(f"\nSaved deep Choquet plots in: {output_dir}")


def train_choquet_q(
    mu1,
    mu2,
    mu12=1.0,
    total_timesteps=200_000,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    batch_size=64,
    replay_capacity=100_000,
    learning_starts=1_000,
    train_frequency=4,
    target_tau=0.005,
    hidden_dim=128,
    seed=None,
):
    """
    Train a vector-valued Deep Q-Network with Choquet action selection.

    Returns:
        final_point, archive_front, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts

    final_point is the final deterministic greedy-policy result:
        (time_cost, treasure)

    archive_front is the non-dominated episode archive in training form:
        [(time_cost, treasure), ...]
    """
    _validate_capacity(mu1, mu2, mu12)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if epsilon_decay_timesteps is None:
        epsilon_decay_timesteps = total_timesteps
    if epsilon_decay_timesteps <= 0:
        raise ValueError("epsilon_decay_timesteps must be positive.")
    if not (0.0 < target_tau <= 1.0):
        raise ValueError("target_tau must be in (0, 1].")

    rng = np.random.default_rng(seed)
    replay_rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    if seed is not None:
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed)
    true_pf = get_true_reference_pf()
    ranges = _objective_ranges(true_pf)
    objective_scales = torch.as_tensor(
        [
            ranges["time"][1] - ranges["time"][0],
            ranges["treasure"][1] - ranges["treasure"][0],
        ],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1.0)

    obs_low = np.asarray(env.observation_space.low, dtype=np.float32)
    obs_high = np.asarray(env.observation_space.high, dtype=np.float32)
    state_dim = int(np.prod(_grid_shape(obs_low, obs_high)))
    n_actions = env.action_space.n

    online = VectorQNetwork(state_dim, n_actions, hidden_dim).to(device)
    target = VectorQNetwork(state_dim, n_actions, hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity, replay_rng)
    archive = []

    hv_timesteps, hv_points = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []
    global_step, next_log = 0, log_interval
    best_score = -float("inf")
    best_state_dict = None
    best_timestep = 0

    while global_step < total_timesteps:
        obs, _ = env.reset(seed=seed if global_step == 0 else None)
        state = _encode_state(obs, obs_low, obs_high)
        done = False
        episode_steps = 0
        episode_treasure = 0.0

        while not done and global_step < total_timesteps:
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
                    online, state, mu1, mu2, mu12, ranges, device
                )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = _encode_state(next_obs, obs_low, obs_high)
            treasure = float(reward[0])
            time_return = float(reward[1])
            reward_vector = np.array(
                [time_return, treasure], dtype=np.float32
            )

            replay.add(state, action, reward_vector, next_state, done)
            state = next_state
            global_step += 1
            episode_steps += 1
            episode_treasure = treasure

            if done:
                archive.append((float(episode_steps), episode_treasure))

            if (
                global_step >= learning_starts
                and global_step % train_frequency == 0
                and len(replay) >= batch_size
            ):
                states, actions, rewards, next_states, dones = replay.sample(
                    batch_size, device
                )

                predicted = online(states)
                batch_indices = torch.arange(batch_size, device=device)
                predicted = predicted[batch_indices, actions]

                with torch.no_grad():
                    next_actions = _best_actions_batch(
                        online, next_states, mu1, mu2, mu12, ranges
                    )
                    next_vectors = target(next_states)[
                        batch_indices, next_actions
                    ]
                    targets = rewards + gamma * (1.0 - dones.unsqueeze(1)) * next_vectors

                loss = nn.functional.smooth_l1_loss(
                    predicted / objective_scales,
                    targets / objective_scales,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
                _soft_update(target, online, target_tau)

            while global_step >= next_log:
                time_cost, eval_treasure = _evaluate(
                    eval_env,
                    online,
                    mu1,
                    mu2,
                    mu12,
                    ranges,
                    obs_low,
                    obs_high,
                    device,
                    n_eval,
                )
                evaluation_score = _checkpoint_score(
                    time_cost,
                    eval_treasure,
                    mu1,
                    mu2,
                    mu12,
                    ranges,
                )
                if evaluation_score > best_score:
                    best_score = evaluation_score
                    best_state_dict = _copy_state_dict_to_cpu(online)
                    best_timestep = next_log
                archive.append((time_cost, eval_treasure))
                archive_max = extract_pareto_front(_archive_to_max(archive))

                hv_timesteps.append(next_log)
                hv_points.append(compute_hypervolume_2d(
                    archive_max, ref_point=(-100, 0)
                ))
                igd_timesteps.append(next_log)
                igd_points.append(compute_igd(true_pf, archive_max))
                eps_timesteps.append(next_log)
                eps_points.append(compute_epsilon_indicator(
                    true_pf, archive_max
                ))
                next_log += log_interval

    if best_state_dict is not None:
        online.load_state_dict(best_state_dict)

    final_time, final_treasure = _evaluate(
        eval_env,
        online,
        mu1,
        mu2,
        mu12,
        ranges,
        obs_low,
        obs_high,
        device,
        n_eval,
    )
    archive.append((final_time, final_treasure))
    final_point = (final_time, final_treasure)
    archive_front = _archive_front(archive)
    print(
        f"  restored best checkpoint from timestep {best_timestep:,} "
        f"(score={best_score:.6f})"
    )

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


def run_all_deep_choquet_capacities(
    capacities=DEEP_CHOQUET_CAPACITIES,
    total_timesteps=200_000,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_timesteps=None,
    log_interval=1_000,
    n_eval=1,
    seed=7,
    output_dir="results/deep_choquet",
):
    true_front = get_true_reference_pf()
    results_by_capacity = {}

    for index, capacity in enumerate(capacities):
        mu1, mu2, mu12 = capacity
        print(
            f"[{index + 1}/{len(capacities)}] "
            f"capacity=({mu1:.2f}, {mu2:.2f}, {mu12:.1f})"
        )
        result = train_choquet_q(
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
        print(f"  archive front size: {len(result[1])}")

    final_policy_points_max = [
        (-result[0][0], result[0][1])
        for result in results_by_capacity.values()
    ]
    final_policy_front = extract_pareto_front(final_policy_points_max)
    final_coverage = _coverage(true_front, final_policy_front)

    archive_points_max = [
        (-time_cost, treasure)
        for result in results_by_capacity.values()
        for time_cost, treasure in result[1]
    ]
    archive_front = extract_pareto_front(archive_points_max)
    archive_coverage = _coverage(true_front, archive_front)

    print("\nCombined deep Choquet final-policy front:")
    for point in final_policy_front:
        print(f"  {point}")
    print(
        f"Final-policy coverage: "
        f"{final_coverage[0]}/{final_coverage[1]}"
    )

    print("\nCombined deep Choquet archive:")
    for point in archive_front:
        print(f"  {point}")
    print(
        f"Archive coverage: "
        f"{archive_coverage[0]}/{archive_coverage[1]}"
    )

    _save_deep_choquet_plots(
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
        "Deep Choquet Q-Learning",
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
    run_all_deep_choquet_capacities()
