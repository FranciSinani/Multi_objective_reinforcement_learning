import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d


def owa_score(values, owa_weights):

    """ Computes the Ordered Weighted Averaging (OWA) scalarization.
        Sorts the values in descending order and applies the weights."""
    
    # Sort the input values (e.g., Q1 and Q2) from highest to lowest
    vals = sorted(values, reverse=True)
    
    # Compute weighted sum: highest value gets the first weight, etc.
    # This allows emphasizing either the best or the worst objective.
    return sum(w * v for w, v in zip(owa_weights, vals))


def evaluate_final_owa_policy(env, Q1, Q2, owa_weights, n_eval_episodes=1):
    """
    Evaluates the final OWA policy after training.
    Uses OWA scalarization on the two Q-tables to select actions.
    Returns one final point (time_cost, treasure) for Pareto front construction.
    """
    total_times = []
    total_treasures = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False

        total_time_cost = 0
        final_treasure = 0.0

        while not done:
            state = (int(obs[0]), int(obs[1]))

            # compute OWA score for every possible action using current Q1 and Q2
            action_scores = []
            for a in range(4):
                score = owa_score([Q1[state][a], Q2[state][a]], owa_weights)
                action_scores.append(score)

            # choose the action with the highest OWA score (greedy policy)
            action = int(np.argmax(action_scores))

            next_obs, reward_vec, terminated, truncated, info = env.step(action)

            treasure_reward = float(reward_vec[0])   # reward_vec[0] = treasure

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            obs = next_obs
            done = terminated or truncated

        total_times.append(total_time_cost)
        total_treasures.append(final_treasure)

    # return format expected by plotting code: (time_cost, treasure)
    return (float(np.mean(total_times)), float(np.mean(total_treasures)))


def train_owa_q(
    owa_weights,
    total_timesteps=400000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
    n_eval_episodes=1,
):
    """Trains an OWA Q-Learning agent.
       Uses Ordered Weighted Averaging instead of linear weights.
       Returns one final evaluated point per OWA weight setting."""
    
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    # Separate Q-tables for each objective
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

            # epsilon-greedy using OWA scalarization on Q-values
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action_scores = []
                for a in range(4):
                    score = owa_score([Q1[state][a], Q2[state][a]], owa_weights)
                    action_scores.append(score)
                action = int(np.argmax(action_scores))

            # take action
            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            # we compute OWA scores again for the next state to choose the best next action
            next_action_scores = []
            for a in range(4):
                score = owa_score([Q1[next_state][a], Q2[next_state][a]], owa_weights)
                next_action_scores.append(score)
            best_next_action = int(np.argmax(next_action_scores))

            # Update Q1 (time objective) using OWA-selected next action
            Q1[state][action] = Q1[state][action] + lr * (
                time_reward + gamma * Q1[next_state][best_next_action] - Q1[state][action]
            )

            # Update Q2 (treasure objective) using OWA-selected next action
            Q2[state][action] = Q2[state][action] + lr * (
                treasure_reward + gamma * Q2[next_state][best_next_action] - Q2[state][action]
            )

            obs = next_obs
            done = terminated or truncated

        # Log hypervolume at regular intervals
        while global_step >= next_log_step:
            eval_time_cost, eval_treasure = evaluate_final_owa_policy(
                env, Q1, Q2, owa_weights, n_eval_episodes=n_eval_episodes
            )
            hv = compute_hypervolume_2d([(-eval_time_cost, eval_treasure)], ref_point=(-100, 0))
            hv_timesteps.append(next_log_step)
            hv_points.append(hv)
            next_log_step += log_interval

    # evaluate the final policy using OWA scalarization
    final_time_cost, final_treasure = evaluate_final_owa_policy(
        env, Q1, Q2, owa_weights, n_eval_episodes=n_eval_episodes
    )

    env.close()

    # return one final evaluated point per OWA weight setting, along with hypervolume logs
    final_point = (final_time_cost, final_treasure)

    return final_point, hv_timesteps, hv_points