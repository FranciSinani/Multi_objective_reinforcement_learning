import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import get_non_dominated


def compute_hv_for_vector_set(vectors, ref_point=(0.0, -100.0)):
    """
    Hypervolume for vectors stored as:
        (treasure_return, time_return)

    Both objectives are maximized.
    In DST:
        treasure_return >= 0
        time_return <= 0   (because each step gives -1)

    ref_point is in the same coordinate system:
        (treasure_ref, time_ref)
    """
    if not vectors:
        return 0.0

    nd = get_non_dominated(vectors)
    nd = sorted(nd, key=lambda p: p[0])  # sort by treasure

    hv = 0.0
    prev_x = ref_point[0]

    for x, y in nd:
        width = x - prev_x
        height = y - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return hv


def add_vector_to_set(immediate_reward, future_set, gamma):
    """
    immediate_reward: (treasure_reward, time_reward)
    future_set: list of future return vectors
    """
    out = []
    for vec in future_set:
        out.append((
            immediate_reward[0] + gamma * vec[0],
            immediate_reward[1] + gamma * vec[1],
        ))
    return out


def get_state(obs):
    return (int(obs[0]), int(obs[1]))


def get_all_vectors_at_state(Q_sets, state, num_actions):
    all_vectors = []
    for a in range(num_actions):
        all_vectors.extend(Q_sets[state][a])
    return all_vectors


def get_front_at_state(Q_sets, state, num_actions):
    all_vectors = get_all_vectors_at_state(Q_sets, state, num_actions)
    return get_non_dominated(all_vectors)


def convert_vectors_to_plot_points(vectors, round_digits=8):
    """
    Convert internal vectors:
        (treasure_return, time_return)
    into plot points:
        (time_cost, treasure_value)

    So plotting code can still use:
        x = -time_cost
        y = treasure_value

    IMPORTANT:
    This matches real time cost / treasure cleanly when gamma = 1.0.
    """
    points = []
    seen = set()

    for treasure_ret, time_ret in vectors:
        time_cost = -time_ret
        treasure_value = treasure_ret

        key = (round(time_cost, round_digits), round(treasure_value, round_digits))
        if key not in seen:
            seen.add(key)
            points.append((float(time_cost), float(treasure_value)))

    # sort by time cost
    points.sort(key=lambda p: p[0])
    return points


def train_pql(
    total_timesteps=400000,
    gamma=1.0,              # keep 1.0 for DST if you want exact time_cost / treasure points
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
):
    """
    Correct PQL training:
    - learns set-valued Q sets
    - logs hypervolume of the learned Pareto front at the START STATE
    - returns ALL learned final vectors from the START STATE for plotting
    """

    env = mo_gym.make("deep-sea-treasure-concave-v0")
    num_actions = env.action_space.n

    # start state
    start_obs, _ = env.reset()
    start_state = get_state(start_obs)

    # Q_sets[state][action] = list of non-dominated vectors
    Q_sets = defaultdict(lambda: [[(0.0, 0.0)] for _ in range(num_actions)])

    hv_timesteps = []
    hv_points = []

    global_step = 0
    next_log_step = log_interval

    while global_step < total_timesteps:
        obs, _ = env.reset()
        done = False

        while not done and global_step < total_timesteps:
            state = get_state(obs)

            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * (global_step / total_timesteps)
            )

            # epsilon-greedy over action-set hypervolume
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action_scores = [
                    compute_hv_for_vector_set(Q_sets[state][a], ref_point=(0.0, -100.0))
                    for a in range(num_actions)
                ]
                action = int(np.argmax(action_scores))

            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state(next_obs)

            immediate_reward = (float(reward_vec[0]), float(reward_vec[1]))

            # terminal state -> no future return except zero vector
            if done:
                future_nd = [(0.0, 0.0)]
            else:
                future_candidates = get_all_vectors_at_state(Q_sets, next_state, num_actions)
                future_nd = get_non_dominated(future_candidates)

            new_vectors = add_vector_to_set(immediate_reward, future_nd, gamma)

            # keep non-dominated vectors for this state-action
            Q_sets[state][action] = get_non_dominated(new_vectors)

            obs = next_obs
            global_step += 1

            # log HV from the learned front at the START STATE
            while global_step >= next_log_step:
                start_front = get_front_at_state(Q_sets, start_state, num_actions)
                hv = compute_hv_for_vector_set(start_front, ref_point=(0.0, -100.0))

                hv_timesteps.append(next_log_step)
                hv_points.append(hv)

                next_log_step += log_interval

    # FINAL learned vectors from the start state
    final_vectors = get_all_vectors_at_state(Q_sets, start_state, num_actions)

    # convert to (time_cost, treasure) for your existing plotting code
    final_points = convert_vectors_to_plot_points(final_vectors)

    env.close()
    return final_points, hv_timesteps, hv_points