import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import os
import numpy as np
import gymnasium as gym
from utils import plot_training_history
from replay_buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_v3, simple_adversary_v3


def show():
    env = simple_v3.parallel_env(render_mode="human")
    # env = simple_adversary_v3.parallel_env(render_mode="human")
    observations, infos = env.reset(seed=42)
    print(f"n_agents: {env.num_agents}")

    while env.agents:
        # this is where you would insert your policy
        actions = {}
        for agent in env.agents:
            action = env.action_space(agent).sample()
            actions[agent] = action
            print(f"agent name: {agent}, action: {action}")

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"observations: {observations}")
        print(f"rewards: {rewards}")
        print(f"terminations: {terminations}, truncations, {truncations}")
        print(f"infos: {infos}")
        print(f"")
    env.close()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    show()
