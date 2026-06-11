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
import random

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
)


class VectorQNetwork(nn.Module):
    """Predict two objective values for each discrete action."""

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
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.data, batch_size)
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


def _normalise_state(obs, obs_low, obs_high):
    obs = np.asarray(obs, dtype=np.float32)
    span = np.maximum(obs_high - obs_low, 1.0)
    return ((obs - obs_low) / span).astype(np.float32)


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
    n_eval=10,
):
    times, treasures = [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state = _normalise_state(obs, obs_low, obs_high)
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


def train_choquet_q(
    mu1,
    mu2,
    mu12=1.0,
    total_timesteps=400_000,
    lr=1e-3,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1_000,
    n_eval=10,
    batch_size=64,
    replay_capacity=100_000,
    learning_starts=1_000,
    train_frequency=4,
    target_update_interval=2_000,
    hidden_dim=128,
    seed=None,
):
    """
    Train a vector-valued Deep Q-Network with Choquet action selection.

    Returns:
        final_points, hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts

    final_points is the non-dominated episode archive in training form:
        [(time_cost, treasure), ...]
    """
    _validate_capacity(mu1, mu2, mu12)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    true_pf = get_true_reference_pf()
    ranges = _objective_ranges(true_pf)

    obs_low = np.asarray(env.observation_space.low, dtype=np.float32)
    obs_high = np.asarray(env.observation_space.high, dtype=np.float32)
    state_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    online = VectorQNetwork(state_dim, n_actions, hidden_dim).to(device)
    target = VectorQNetwork(state_dim, n_actions, hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)
    archive = []

    hv_timesteps, hv_points = [], []
    igd_timesteps, igd_points = [], []
    eps_timesteps, eps_points = [], []
    global_step, next_log = 0, log_interval

    while global_step < total_timesteps:
        obs, _ = env.reset(seed=seed if global_step == 0 else None)
        state = _normalise_state(obs, obs_low, obs_high)
        done = False
        episode_steps = 0
        episode_treasure = 0.0

        while not done and global_step < total_timesteps:
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * global_step / total_timesteps,
            )

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = _best_action(
                    online, state, mu1, mu2, mu12, ranges, device
                )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = _normalise_state(next_obs, obs_low, obs_high)
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

                loss = nn.functional.smooth_l1_loss(predicted, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()

            if global_step % target_update_interval == 0:
                target.load_state_dict(online.state_dict())

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
    final_points = _archive_front(archive)

    env.close()
    eval_env.close()
    return (
        final_points,
        hv_timesteps,
        hv_points,
        igd_timesteps,
        igd_points,
        eps_timesteps,
        eps_points,
    )
