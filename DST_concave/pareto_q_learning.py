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


def evaluate_final_pql_policy(env, Q_sets):

    # reset environment
    obs, info = env.reset()
    done = False

    total_time_cost = 0 # will count the number of steps taken
    final_treasure = 0.0 # best treasure value found during the episode

    while not done:

        state = (int(obs[0]), int(obs[1])) # convert observation to discrete state for Q-table

        # select action with highest hypervolume from the set of vectors for each action
        action_scores = []
        for a in range(4):
            action_scores.append(compute_hv_for_action_set(Q_sets[state][a]))
        action = int(np.argmax(action_scores))

        next_obs, reward_vec, terminated, truncated, info = env.step(action)

        # reward_vec = [treasure_reward, time_penalty], this is the order in mo_gymnasium DST environment
        treasure_reward = float(reward_vec[0]) 

        total_time_cost += 1  # we count manually (+1 per step) instead of using reward_vec[1]
        final_treasure = max(final_treasure, treasure_reward) # we want the best treasure found during the episode, not the sum

        obs = next_obs # move to the next observation
        done = terminated or truncated 

    return (total_time_cost, final_treasure) # this order matches the convention expected by plotting


def train_pql(
    total_timesteps=200000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
):
    """Trains a Pareto Q-Learning (PQL) agent that maintains sets of non-dominated vectors for each state-action pair."""
    
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    num_actions = 4

    # for each state: list of 4 action sets (each set contains non-dominated vectors)
    Q_sets = defaultdict(lambda: [[(0.0, 0.0)] for _ in range(num_actions)])

    hv_points = [] # to store hypervolume values at logging intervals
    hv_timesteps = [] # corresponding timestep when hypervolume was computed
    training_points = [] # to store the final evaluated solution (time_cost, treasure) of each episode

    global_step = 0 # counts total steps taken across episodes
    next_log_step = log_interval # when to compute hypervolume next


    # training loop
    while global_step < total_timesteps:     
        obs, info = env.reset()
        done = False

        total_time_cost = 0
        final_treasure = 0.0

        while not done and global_step < total_timesteps:
            state = (int(obs[0]), int(obs[1]))

            # Linear epsilon decay: from epsilon_start down to epsilon_end over the course of training
            epsilon = max( 
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * (global_step / total_timesteps)
            ) 

            # Epsilon-greedy action selection using hypervolume of vector sets
            if np.random.rand() < epsilon: 
                action = env.action_space.sample() # random exploration
            else:
                action_scores = []
                for a in range(num_actions):
                    action_scores.append(compute_hv_for_action_set(Q_sets[state][a]))
                action = int(np.argmax(action_scores)) # select action with best hypervolume

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            immediate_reward = (treasure_reward, time_reward)

            # collect all non-dominated vectors from next state
            future_candidates = []
            for a in range(num_actions):
                future_candidates.extend(Q_sets[next_state][a])

            future_nd = get_non_dominated(future_candidates)

            # compute new vector set by adding immediate reward to future non-dominated vectors
            new_vectors = add_vector_to_set(immediate_reward, future_nd, gamma)
            Q_sets[state][action] = get_non_dominated(new_vectors)

            obs = next_obs
            done = terminated or truncated

        # after each episode we store the current episode's result for hypervolume calculation
        training_points.append((total_time_cost, final_treasure))
        
        # log hypervolume at regular intervals
        if global_step >= next_log_step:
            hv = compute_hypervolume_2d(
                [(-tc, tr) for tc, tr in training_points],
                ref_point=(-100, 0)
            )
            hv_timesteps.append(global_step)
            hv_points.append(hv)
            next_log_step += log_interval

    # after training is complete, evaluate the final greedy policy
    final_time_cost, final_treasure = evaluate_final_pql_policy(env, Q_sets)

    env.close()

    # Return format expected by the new plotting code, one final evaluated point per run
    final_point = [(final_time_cost, final_treasure)]
    return final_point, hv_timesteps, hv_points