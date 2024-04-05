import numpy as np
import torch


class RandomStrategy:
    def __init__(self):
        pass

    def select_action(self, model, state):
        q_values = model(state).detach().cpu().data.numpy().squeeze()
        action = q_values.argmax()
        return action
