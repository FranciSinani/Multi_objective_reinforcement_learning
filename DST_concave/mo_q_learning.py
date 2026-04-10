import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
from utils import compute_hypervolume_2d


def evaluate_final_mo_policy(env, Q):

    # reset environment
    obs, info = env.reset()
    done = False

    total_time_cost = 0 # will count the number of steps taken
    final_treasure = 0.0 # best treasure value found during the episode

    while not done:

        state = (int(obs[0]), int(obs[1])) # convert observation to discrete state for Q-table
        action = int(np.argmax(Q[state])) # select greedy action from Q-table

        next_obs, reward_vec, terminated, truncated, info = env.step(action)

        # reward_vec = [treasure_reward, time_penalty], this is the order in mo_gymnasium DST environment
        treasure_reward = float(reward_vec[0]) 

        total_time_cost += 1  #we count manually (+1 per step) instead of using reward_vec[1]
        final_treasure = max(final_treasure, treasure_reward) # we want the best treasure found during the episode, not the sum

        obs = next_obs # move to the next observation
        done = terminated or truncated 

    return (total_time_cost, final_treasure) # this order matches the convention expected by plotting


def train_mo_q(
    timeW, 
    treasureW,
    total_timesteps=200000,
    lr=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    log_interval=1000,
):
    """Trains a Multi-Objective Q-Learning agent using linear scalarization.
       Uses two separate Q-tables (one per objective) and combines them with given weights."""
    
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    Q1 = defaultdict(lambda: np.zeros(4))  # Q-table for time objective(minimization), each state maps to a vector of 4 Q-values (one per action)
    Q2 = defaultdict(lambda: np.zeros(4))  # Q-table for treasure objective(maximization)
    Q = defaultdict(lambda: np.zeros(4))  # Combined Q-table for linear scalarization

    hv_points = [] # to store hypervolume values at logging intervals
    hv_timesteps = [] # corresponding timestep when hypervolume was computed
    current_solutions = [] # to store the final evaluated solution (time_cost, treasure) for this weight setting

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

            # Epsilon-greedy action selection using the combined Q-table
            if np.random.rand() < epsilon: 
                action = env.action_space.sample() # random exploration
            else:
                action = int(np.argmax(Q[state])) # greedy action based on combined Q-values

            next_obs, reward_vec, terminated, truncated, info = env.step(action)
            global_step += 1

            next_state = (int(next_obs[0]), int(next_obs[1]))

            treasure_reward = float(reward_vec[0])
            time_reward = float(reward_vec[1])

            total_time_cost += 1
            final_treasure = max(final_treasure, treasure_reward)

            # Update Q-values for both objectives separately:

            Q1[state][action] = Q1[state][action] + lr * (
                time_reward + gamma * np.max(Q1[next_state]) - Q1[state][action]
            )

            Q2[state][action] = Q2[state][action] + lr * (
                treasure_reward + gamma * np.max(Q2[next_state]) - Q2[state][action]
            )

            # Update the combined Q-table used for action selection based on the current weights
            for a in range(4):
                Q[state][a] = timeW * Q1[state][a] + treasureW * Q2[state][a]

            obs = next_obs
            done = terminated or truncated

        # after each episode we store the current episode's result for hypervolume calculation
        current_solutions.append((-total_time_cost, final_treasure))
        
        # log hypervolume at regular intervals
        if global_step >= next_log_step:
            hv = compute_hypervolume_2d(current_solutions, ref_point=(-100, 0))
            hv_timesteps.append(global_step)
            hv_points.append(hv)
            next_log_step += log_interval
    # after training is complete, evaluate the final greedy policy derived from the combined Q-table
    final_time_cost, final_treasure = evaluate_final_mo_policy(env, Q)

    env.close()

    # Return format expected by the new plotting code, one final evaluated point per weight setting
    final_point = [(final_time_cost, final_treasure)]
    return final_point, hv_timesteps, hv_points