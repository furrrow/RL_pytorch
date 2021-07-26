import numpy as np
import torch

class RandomStrategy:
    def __init__(self):
        pass

    def select_action(self, model, state):
        return np.random.randint(6)