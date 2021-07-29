import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import ReplayBuffer
from torchinfo import summary
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class DuelingSimpleModel(nn.Module):

    def __init__(self, in_features, outputs):
        super(DuelingSimpleModel, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512, 128)
        # self.linear3 = nn.Linear(128, 64)
        # self.final = nn.Linear(128, outputs)
        self.output_value = nn.Linear(128, 1)
        self.output_advantage = nn.Linear(128, outputs)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.final(x))
        a = self.output_advantage(x)
        v = self.output_value(x).expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q

    def load(self, buffer):
        batch = Transition(*zip(*buffer))

        states = np.array(batch.state)
        actions = np.array(batch.action)
        new_states = np.array(batch.next_state)
        rewards = np.array(batch.reward)
        is_terminals = np.array(batch.terminal)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

    def print_model(self, input_size=None):
        if input_size is None:
            print(self)
        else:
            summary(self, input_size=input_size)
