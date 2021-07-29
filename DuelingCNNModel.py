import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class DuelingCNNModel(nn.Module):

    def __init__(self, input_shape, outputs):
        super(DuelingCNNModel, self).__init__()
        self.input_shape = input_shape
        self.outputs = outputs
        c = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, outputs)
        self.output_value = nn.Linear(512, 1)
        self.output_advantage = nn.Linear(512, outputs)

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

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self._format(x)
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
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
            print("input size", input_size)
            summary(self, input_size=input_size)
