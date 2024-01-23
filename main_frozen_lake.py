import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from utils import plot_training_history
from replay_buffer import NumpyReplayBuffer
from tqdm import tqdm

""" Frozen Lake
value and policy iteration, 
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_03/chapter-03.ipynb
"""


def policy_iteration(slip):
    """
    Policy Iteration algorithm:
    while true:
    - policy evaluation:
        - update the value for each state until the value matrix stabilizes
    - policy improvement:
        - for each state, choose action with biggest Q[state][action]
    if policy did not update, we have converged and exit

    NOTE: Policy Iteration will have a harder time for the value table to converge,
    recommend setting threshold lower
    :return: policy
    """
    np.set_printoptions(suppress=True)  # pretty print for np arrays
    LR = 0.5
    gamma = 0.99
    epsilon = 0.9
    threshold = 0.005
    env = gym.make("FrozenLake-v1", is_slippery=slip)
    n_states = env.observation_space.n
    n_action = env.action_space.n
    EPOCHS = 1000
    v_table = np.zeros(n_states)
    prev_v_table = np.zeros(n_states)
    q_table = np.zeros((n_states, n_action))
    my_policy = np.argmax(np.random.rand(n_states, n_action), axis=1)  # initialize a random action policy
    # we can't do a full "sweep" when interacting with env online, we use episodes as a sweep instead
    for episode in range(EPOCHS):
        """Policy Evaluation: wait until v_table converge at current policy"""
        update_episode = 0
        while True:
            terminal, truncated = False, False
            state, info = env.reset()
            action = my_policy[state]
            # loop for each step until termination
            loop = 0
            total_reward = 0
            while not (terminal or truncated):
                next_state, reward, terminal, truncated, info = env.step(action)
                # a small negative reward to aid in training
                # reward = 0.0001 if reward == 0 else reward * 1
                # choose action based on epsilon greedy policy:
                if np.random.rand() > epsilon:
                    next_action = my_policy[next_state]
                else:
                    next_action = env.action_space.sample()
                # print(next_state, reward, terminal, truncated, next_action)
                update = reward + gamma * prev_v_table[next_state] * (not terminal)
                # Policy Evaluation: Q update
                q_table[state][action] += (update - q_table[state][action]) * LR
                # Policy Improvement: V update
                v_table[state] += (update - v_table[state]) * LR
                total_reward += reward
                state = next_state
                action = next_action
                loop += 1
            # Policy Evaluation: update the V table and check if V has converged
            value_diff = np.linalg.norm(prev_v_table - v_table, 1)
            print(f"loop {loop} reward: {total_reward}, value_diff: {value_diff:.5f}")
            if (sum(v_table) > 0) and (value_diff < threshold):
                print(f"value table converged after {update_episode} episodes")
                print(v_table.reshape(4, 4))
                break
            prev_v_table = v_table.copy()
            update_episode += 1
            # decaying rates for parameters:
            epsilon = max(0.001, epsilon * 0.999)
            LR = max(0.01, LR * 0.999)

        """ Policy Improvement: re-calculate the policy based on Q table"""
        print(f"updating policy after {update_episode} episodes")
        new_policy = np.argmax(q_table, axis=1)
        policy_diff = np.linalg.norm(new_policy - my_policy, 1)
        if policy_diff < 1:
            print(f"policy converged at episode {episode}")
            print(new_policy)
            continue
        my_policy = new_policy.copy()
        print(f"episode: {episode} epsilon: {epsilon:.2f} reward: {total_reward:.4f}")

    return my_policy


def value_iteration(slip):
    """
    Value Iteration algorithm:
    similar to policy iteration, but we do NOT wait until the state values converge
    Value Iteration goes to policy-evaluation after a single sweep
    while true:
    - policy evaluation:
        - update the value for each state
    - policy improvement:
        - for each state, choose action with biggest Q[state][action]
    if policy did not update, we have converged and exit

    :return: Value, Policy
    """
    np.set_printoptions(suppress=True)  # pretty print for np arrays
    LR = 0.5
    gamma = 0.9
    epsilon = 0.9
    threshold = 0.0001
    env = gym.make("FrozenLake-v1", is_slippery=slip)
    n_states = env.observation_space.n
    n_action = env.action_space.n
    EPOCHS = 5000
    prev_v_table = np.zeros(n_states)
    q_table = np.zeros((n_states, n_action))
    for episode in range(EPOCHS):
        state, info = env.reset()
        action = np.argmax(q_table[state])
        # loop for each step until termination
        loop = 0
        total_reward = 0
        terminal, truncated = False, False
        while not (terminal or truncated):
            next_state, reward, terminal, truncated, info = env.step(action)
            # a small negative reward to aid in training
            # reward = 0.0001 if reward == 0 else reward * 1
            # choose action based on epsilon greedy policy:
            if np.random.rand() > epsilon:
                next_action = np.argmax(q_table[next_state])
            else:
                next_action = env.action_space.sample()
            # print(next_state, reward, terminal, truncated, next_action)
            update = reward + gamma * q_table[next_state][next_action] * (not terminal)
            # Policy Evaluation: Q update
            q_table[state][action] += (update - q_table[state][action]) * LR
            total_reward += reward
            state = next_state
            action = next_action
            loop += 1
        # Value Update for each state or Policy Evaluation
        v_table = np.max(q_table, axis=1)
        delta = np.linalg.norm(v_table - prev_v_table, 1)
        if delta > threshold:
            print(v_table.reshape(4, 4))
        prev_v_table = v_table.copy()
        # decaying rates for parameters:
        epsilon = max(0.001, epsilon * 0.999)
        LR = max(0.01, LR * 0.999)
        print(f"episode {episode}, loop {loop}, reward {total_reward}, LR {LR:.3f}, epslion {epsilon:.3f}, "
              f"delta {delta:.4f}")
    policy = np.argmax(q_table, axis=1)
    return policy


def evaluate_env(policy, slip):
    env = gym.make("FrozenLake-v1", is_slippery=slip, render_mode="human")
    print(policy)
    rewards_list = []
    for episode in range(10):
        observation, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated):
            action = policy[observation]
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        rewards_list.append(total_reward)
    print("final score:", np.average(np.array(rewards_list)))


if __name__ == '__main__':
    slip = False
    policy = policy_iteration(slip)
    # policy = value_iteration(slip)
    slip_policy = [1, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    nonslip_policy = [2, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0]
    evaluate_env(policy, slip)



