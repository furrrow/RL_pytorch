import minari
import os
from huggingface_hub import login
import torch
import torch.nn as nn

# access_token = os.environ["HUGGINGFACE_READ_TOKEN"]
# login(token = access_token ,add_to_git_credential=True)


# dataset = minari.load_dataset('mujoco/hopper/simple-v0', download=True)
dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=True)
# dataset = minari.load_dataset('mujoco/hopper/expert-v0', download=True)
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)
env  = dataset.recover_environment()