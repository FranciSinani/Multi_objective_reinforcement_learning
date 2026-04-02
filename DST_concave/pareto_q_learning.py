import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import get_non_dominated, compute_hypervolume_2d


def compute_hv_for_action_set(vectors, ref_point=(0, -100)):
    """
    vectors are in original reward form:
    (treasure, time_reward)
    both are maximized
    """
    if not vectors:
        return 0.0

    nd = get_non_dominated(vectors)
    nd = sorted(nd, key=lambda p: p[0])

    hv = 0.0
    prev_x = ref_point[0]

    for x, y in nd:
        width = x - prev_x
        height = y - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return hv


def add_vector_to_set(reward_vec, future_set, gamma):
    result = []
    for vec in future_set:
        new_vec = (
            reward_vec[0] + gamma * vec[0],
            reward_vec[1] + gamma * vec[1],
        )
        result.append(new_vec)
    return result


def train_pql(
    total_timesteps=10000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
):
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    num_actions = 4

    # For each state: list of 4 action-sets
    Q_sets = defaultdict(lambda: [[(0.0, 0.0)] for _ in range(num_actions)])

    episode_points = []
    hv_timesteps = []
    hv_points = []

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
                epsilon_start - (epsilon_start - epsilon_end) * (global_step / total_timesteps),
            )

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action_scores = []
                for a in range(num_actions):
                    action_scores.append(compute_hv_for_action_set(Q_sets[state][a]))
                action = int(np.argmax(action_scores))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            immediate_reward = (treasure_reward, time_reward)

            # union of all next-state action sets
            future_candidates = []
            for a in range(num_actions):
                future_candidates.extend(Q_sets[next_state][a])

            future_nd = get_non_dominated(future_candidates)

            new_vectors = add_vector_to_set(immediate_reward, future_nd, gamma)
            Q_sets[state][action] = get_non_dominated(new_vectors)

            obs = next_obs
            done = terminated or truncated

        episode_points.append((total_time_cost, final_treasure))

        if global_step >= next_log_step:
            hv = compute_hypervolume_2d(
                [(-tc, tr) for tc, tr in episode_points],
                ref_point=(-100, 0),
            )
            hv_timesteps.append(global_step)
            hv_points.append(hv)
            next_log_step += log_interval

    env.close()
    return episode_points, hv_timesteps, hv_points