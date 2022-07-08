import numpy as np
import torch

class LinearDecayStrategy:
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, plateu_step=250):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.plateu_step = plateu_step
        self.min_epsilon = min_epsilon
        self.epsilons = np.linspace(init_epsilon, min_epsilon, num=plateu_step, endpoint=False)
        # self.epsilons = 0.01 / np.logspace(-2, 0, plateu_step, endpoint=False) - 0.01
        # self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        if self.t < self.plateu_step:
            self.epsilon = self.epsilons[self.t]
        else:
            self.epsilon = self.min_epsilon
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        # print(self.epsilon)
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        # will update at every epoch instead!!
        # self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action

# ## plot to see how this works
# import matplotlib.pyplot as plt
# s = LinearDecayStrategy()
# plt.plot([s._epsilon_update() for _ in range(50000)])
# plt.title('plt out decay over time')
# plt.show()