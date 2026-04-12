import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d


def chebyshev_score(values, weights, ideal):
    # compute weighted deviations from the ideal point for chebyshev scalarization
    diffs = [w * (z - v) for v, w, z in zip(values, weights, ideal)]
    # return the maximum weighted deviation (chebyshev norm)
    return max(diffs)


def evaluate_final_chebyshev_policy(env, Q1, Q2, cheb_weights, n_eval_episodes=1):
    # evaluate the final chebyshev policy after training
    # returns one point (time_cost, treasure) for pareto front construction
    total_times = []
    total_treasures = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False

        total_time_cost = 0
        final_treasure = 0.0

        while not done:
            state = (int(obs[0]), int(obs[1]))

            # compute current ideal point (best possible q-values in this state)
            ideal_time = max(Q1[state])
            ideal_treasure = max(Q2[state])
            ideal = [ideal_time, ideal_treasure]

            # compute chebyshev score for every action
            action_scores = []
            for a in range(4):
                score = chebyshev_score(
                    values=[Q1[state][a], Q2[state][a]],
                    weights=cheb_weights,
                    ideal=ideal
                )
                action_scores.append(score)

            # select action with the smallest chebyshev score (min-max approach)
            action = int(np.argmin(action_scores))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)

            treasure_reward = float(reward_vec[0])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            obs = next_obs
            done = terminated or truncated

        total_times.append(total_time_cost)
        total_treasures.append(final_treasure)

    # return format expected by plotting code: (time_cost, treasure)
    return (float(np.mean(total_times)), float(np.mean(total_treasures)))


def train_chebyshev_q(
    cheb_weights,
    total_timesteps=400000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
    n_eval_episodes=1,
):
    # create the concave deep sea treasure environment
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    # separate q-tables for each objective
    Q1 = defaultdict(lambda: np.zeros(4))   # time objective
    Q2 = defaultdict(lambda: np.zeros(4))   # treasure objective

    # lists for hypervolume logging during training
    hv_points = []
    hv_timesteps = []

    global_step = 0
    next_log_step = log_interval

    # training loop
    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False

        while not done and global_step < total_timesteps:
            state = (int(obs[0]), int(obs[1]))

            # linear epsilon decay
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * (global_step / total_timesteps)
            )

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # compute current ideal point for this state
                ideal_time = max(Q1[state])
                ideal_treasure = max(Q2[state])
                ideal = [ideal_time, ideal_treasure]

                # compute chebyshev score for every action
                action_scores = []
                for a in range(4):
                    score = chebyshev_score(
                        values=[Q1[state][a], Q2[state][a]],
                        weights=cheb_weights,
                        ideal=ideal
                    )
                    action_scores.append(score)

                # select action with minimal chebyshev score
                action = int(np.argmin(action_scores))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            # compute ideal point for the next state
            next_ideal_time = max(Q1[next_state])
            next_ideal_treasure = max(Q2[next_state])
            next_ideal = [next_ideal_time, next_ideal_treasure]

            # compute chebyshev scores for next state to select best next action
            next_scores = []
            for a in range(4):
                score = chebyshev_score(
                    values=[Q1[next_state][a], Q2[next_state][a]],
                    weights=cheb_weights,
                    ideal=next_ideal
                )
                next_scores.append(score)

            best_next_action = int(np.argmin(next_scores))

            # update q1 (time objective)
            Q1[state][action] = Q1[state][action] + lr * (
                time_reward + gamma * Q1[next_state][best_next_action] - Q1[state][action]
            )

            # update q2 (treasure objective)
            Q2[state][action] = Q2[state][action] + lr * (
                treasure_reward + gamma * Q2[next_state][best_next_action] - Q2[state][action]
            )

            obs = next_obs
            done = terminated or truncated

        # log hypervolume at regular intervals
        while global_step >= next_log_step:
            eval_time_cost, eval_treasure = evaluate_final_chebyshev_policy(
                env, Q1, Q2, cheb_weights, n_eval_episodes=n_eval_episodes
            )
            hv = compute_hypervolume_2d([(-eval_time_cost, eval_treasure)], ref_point=(-100, 0))
            hv_timesteps.append(next_log_step)
            hv_points.append(hv)
            next_log_step += log_interval

    #  final evaluation
    # evaluate the final policy using chebyshev scalarization
    final_time_cost, final_treasure = evaluate_final_chebyshev_policy(
        env, Q1, Q2, cheb_weights, n_eval_episodes=n_eval_episodes
    )

    env.close()

    # return one final evaluated point (matches the new plotting structure)
    final_point = (final_time_cost, final_treasure)

    return final_point, hv_timesteps, hv_points