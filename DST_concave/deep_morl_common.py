"""
Shared vector-valued Deep Q-Learning engine for scalarized MORL methods.

Each method supplies a scoring function that maps normalized objective values
for every action to one scalar score per action. The network, replay buffer,
Double-DQN update, target network, evaluation, metrics, and Pareto archive are
otherwise identical across methods.
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


def _objective_ranges(true_front):
    return {
        "time": (min(point[0] for point in true_front), 0.0),
        "treasure": (0.0, max(point[1] for point in true_front)),
    }


def normalize_q_vectors(q_vectors, ranges):
    q_vectors = np.asarray(q_vectors, dtype=float)
    normalized = np.empty_like(q_vectors, dtype=float)
    for objective, key in enumerate(("time", "treasure")):
        lower, upper = ranges[key]
        if np.isclose(lower, upper):
            normalized[:, objective] = 0.5
        else:
            normalized[:, objective] = np.clip(
                (q_vectors[:, objective] - lower) / (upper - lower),
                0.0,
                1.0,
            )
    return normalized


def _grid_shape(observation_low, observation_high):
    return (
        np.asarray(observation_high, dtype=int)
        - np.asarray(observation_low, dtype=int)
        + 1
    )


def _encode_state(observation, observation_low, observation_high):
    shape = _grid_shape(observation_low, observation_high)
    coordinates = (
        np.asarray(observation, dtype=int)
        - np.asarray(observation_low, dtype=int)
    )
    index = np.ravel_multi_index(tuple(coordinates), tuple(shape))
    encoded = np.zeros(int(np.prod(shape)), dtype=np.float32)
    encoded[index] = 1.0
    return encoded


def _best_action(network, state, score_function, ranges, device):
    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        )
        q_vectors = network(state_tensor.unsqueeze(0))[0].cpu().numpy()
    normalized = normalize_q_vectors(q_vectors, ranges)
    return int(np.argmax(score_function(normalized)))


def _best_actions_batch(network, states, score_function, ranges):
    with torch.no_grad():
        q_batch = network(states).cpu().numpy()
    actions = [
        int(np.argmax(score_function(normalize_q_vectors(q, ranges))))
        for q in q_batch
    ]
    return torch.as_tensor(actions, dtype=torch.long, device=states.device)


def _evaluate(
    environment,
    network,
    score_function,
    ranges,
    observation_low,
    observation_high,
    device,
    n_eval,
):
    times, treasures = [], []
    for _ in range(n_eval):
        observation, _ = environment.reset()
        done = False
        steps, treasure = 0, 0.0
        while not done:
            state = _encode_state(
                observation, observation_low, observation_high
            )
            action = _best_action(
                network, state, score_function, ranges, device
            )
            observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            treasure = float(reward[0])
            steps += 1
            done = terminated or truncated
        times.append(steps)
        treasures.append(treasure)
    return float(np.mean(times)), float(np.mean(treasures))


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


def _soft_update(target, online, tau):
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(online_parameter, alpha=tau)


def _checkpoint_score(time_cost, treasure, score_function, ranges):
    outcome = np.array([[-float(time_cost), float(treasure)]])
    normalized = normalize_q_vectors(outcome, ranges)
    return float(score_function(normalized)[0])


def _copy_state_dict_to_cpu(network):
    return {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }


def train_deep_scalarized_q(
    score_function,
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
    Train a vector-valued DQN using the supplied action-scoring function.

    Returns:
        final_point, archive_front,
        hv_ts, hv_pts, igd_ts, igd_pts, eps_ts, eps_pts
    """
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
    environment = mo_gym.make("deep-sea-treasure-concave-v0")
    evaluation_environment = mo_gym.make("deep-sea-treasure-concave-v0")
    if seed is not None:
        environment.action_space.seed(seed)
        evaluation_environment.action_space.seed(seed)
    true_front = get_true_reference_pf()
    ranges = _objective_ranges(true_front)
    objective_scales = torch.as_tensor(
        [
            ranges["time"][1] - ranges["time"][0],
            ranges["treasure"][1] - ranges["treasure"][0],
        ],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1.0)

    observation_low = np.asarray(
        environment.observation_space.low, dtype=np.float32
    )
    observation_high = np.asarray(
        environment.observation_space.high, dtype=np.float32
    )
    state_dim = int(np.prod(_grid_shape(
        observation_low, observation_high
    )))
    n_actions = environment.action_space.n

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
        observation, _ = environment.reset(
            seed=seed if global_step == 0 else None
        )
        state = _encode_state(
            observation, observation_low, observation_high
        )
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
                action = int(environment.action_space.sample())
            else:
                action = _best_action(
                    online, state, score_function, ranges, device
                )

            next_observation, reward, terminated, truncated, _ = (
                environment.step(action)
            )
            done = terminated or truncated
            next_state = _encode_state(
                next_observation, observation_low, observation_high
            )
            reward_vector = np.array(
                [float(reward[1]), float(reward[0])],
                dtype=np.float32,
            )

            replay.add(state, action, reward_vector, next_state, done)
            state = next_state
            global_step += 1
            episode_steps += 1
            episode_treasure = float(reward[0])

            if done:
                archive.append(
                    (float(episode_steps), float(episode_treasure))
                )

            if (
                global_step >= learning_starts
                and global_step % train_frequency == 0
                and len(replay) >= batch_size
            ):
                states, actions, rewards, next_states, dones = replay.sample(
                    batch_size, device
                )
                batch_indices = torch.arange(batch_size, device=device)
                predicted = online(states)[batch_indices, actions]

                with torch.no_grad():
                    next_actions = _best_actions_batch(
                        online, next_states, score_function, ranges
                    )
                    next_vectors = target(next_states)[
                        batch_indices, next_actions
                    ]
                    targets = (
                        rewards
                        + gamma
                        * (1.0 - dones.unsqueeze(1))
                        * next_vectors
                    )

                # Balance objective errors while retaining raw Q-value outputs.
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
                time_cost, treasure = _evaluate(
                    evaluation_environment,
                    online,
                    score_function,
                    ranges,
                    observation_low,
                    observation_high,
                    device,
                    n_eval,
                )
                evaluation_score = _checkpoint_score(
                    time_cost,
                    treasure,
                    score_function,
                    ranges,
                )
                if evaluation_score > best_score:
                    best_score = evaluation_score
                    best_state_dict = _copy_state_dict_to_cpu(online)
                    best_timestep = next_log
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

    if best_state_dict is not None:
        online.load_state_dict(best_state_dict)

    final_time, final_treasure = _evaluate(
        evaluation_environment,
        online,
        score_function,
        ranges,
        observation_low,
        observation_high,
        device,
        n_eval,
    )
    final_point = (final_time, final_treasure)
    archive.append(final_point)
    final_front = _archive_front(archive)
    print(
        f"  restored best checkpoint from timestep {best_timestep:,} "
        f"(score={best_score:.6f})"
    )

    environment.close()
    evaluation_environment.close()
    return (
        final_point,
        final_front,
        hv_timesteps,
        hv_points,
        igd_timesteps,
        igd_points,
        eps_timesteps,
        eps_points,
    )
