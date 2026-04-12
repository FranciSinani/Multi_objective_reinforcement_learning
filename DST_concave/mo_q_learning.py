import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d


def evaluate_final_mo_policy(env, Q, n_eval_episodes=1):
    total_times = []
    total_treasures = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False

        total_time_cost = 0
        final_treasure = 0.0

        while not done:
            state = (int(obs[0]), int(obs[1]))
            action = int(np.argmax(Q[state]))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)

            treasure_reward = float(reward_vec[0])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            obs = next_obs
            done = terminated or truncated

        total_times.append(total_time_cost)
        total_treasures.append(final_treasure)

    return float(np.mean(total_times)), float(np.mean(total_treasures))


def train_mo_q(
    timeW,
    treasureW,
    total_timesteps=400000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
    n_eval_episodes=1,
):
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    Q1 = defaultdict(lambda: np.zeros(4))
    Q2 = defaultdict(lambda: np.zeros(4))
    Q = defaultdict(lambda: np.zeros(4))

    hv_points = []
    hv_timesteps = []

    global_step = 0
    next_log_step = log_interval

    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False

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

            Q1[state][action] = Q1[state][action] + lr * (
                time_reward + gamma * np.max(Q1[next_state]) - Q1[state][action]
            )

            Q2[state][action] = Q2[state][action] + lr * (
                treasure_reward + gamma * np.max(Q2[next_state]) - Q2[state][action]
            )

            for a in range(4):
                Q[state][a] = timeW * Q1[state][a] + treasureW * Q2[state][a]

            obs = next_obs
            done = terminated or truncated

        # log greedy-policy HV, not archive of exploration episodes
        while global_step >= next_log_step:
            eval_time_cost, eval_treasure = evaluate_final_mo_policy(
                env, Q, n_eval_episodes=n_eval_episodes
            )
            hv = compute_hypervolume_2d([(-eval_time_cost, eval_treasure)], ref_point=(-100, 0))
            hv_timesteps.append(next_log_step)
            hv_points.append(hv)
            next_log_step += log_interval

    final_time_cost, final_treasure = evaluate_final_mo_policy(
        env, Q, n_eval_episodes=n_eval_episodes
    )

    env.close()

    final_point = (final_time_cost, final_treasure)
    return final_point, hv_timesteps, hv_points