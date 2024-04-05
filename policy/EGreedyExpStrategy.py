import numpy as np
import torch


class EGreedyExpStrategy:
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        if np.random.rand() > self.epsilon:
            # choose action with the highest estimated value
            # action = model(state).detach().max(1).indices.view(1, 1).item()
            action = q_values.argmax()
            self.exploratory_action_taken = True
        else:
            # action = np.random.randint(model(state).shape[1])
            action = np.random.randint(len(q_values))  # random action

        self._epsilon_update()
        return action
