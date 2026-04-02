import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d


def train_mo_q(
    timeW,
    treasureW,
    total_timesteps=10000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
):
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    Q1 = defaultdict(lambda: np.zeros(4))
    Q2 = defaultdict(lambda: np.zeros(4))
    Q = defaultdict(lambda: np.zeros(4))

    episode_points = []
    hv_points = []
    hv_timesteps = []
    current_solutions = []

    global_step = 0
    next_log_step = log_interval

    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False

        total_time_cost = 0
        final_treasure = 0.0

        while not done and global_step < total_timesteps:
            state = (int(obs[0]), int(obs[1]))

            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * (global_step / total_timesteps)
            )

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            Q1[state][action] = Q1[state][action] + lr * (
                time_reward + gamma * np.max(Q1[next_state]) - Q1[state][action]
            )

            Q2[state][action] = Q2[state][action] + lr * (
                treasure_reward + gamma * np.max(Q2[next_state]) - Q2[state][action]
            )

            Q[state][action] = timeW * Q1[state][action] + treasureW * Q2[state][action]

            obs = next_obs
            done = terminated or truncated

        episode_points.append((total_time_cost, final_treasure))
        current_solutions.append((-total_time_cost, final_treasure))

        if global_step >= next_log_step:
            hv = compute_hypervolume_2d(current_solutions, ref_point=(-100, 0))
            hv_timesteps.append(global_step)
            hv_points.append(hv)
            next_log_step += log_interval

    env.close()
    return episode_points, hv_timesteps, hv_points