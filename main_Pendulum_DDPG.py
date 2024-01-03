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

""" DDPG code implementation,
heavily referencing:
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_12/chapter-12.ipynb


"""


class QVNet(nn.Module):

    def __init__(self, input_dim, n_actions, gamma=0.99):
        super(QVNet, self).__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.logpas = []
        self.entropies = []
        self.gamma = gamma

        self.linear1 = nn.Linear(input_dim+n_actions, 256)
        self.linear2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)
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

    def forward(self, s, a):
        s, a = self._format(s), self._format(a)
        if len(a.shape) < len(s.shape):
            a = a.unsqueeze(1)
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        v = self.value_layer(x)
        return v

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class PNet(nn.Module):

    def __init__(self, input_dim, n_actions, env_bounds: tuple, gamma=0.99):
        super(PNet, self).__init__()
        self.states = []
        self.actions = []
        self.env_min, self.env_max = env_bounds
        self.rewards = []
        self.logpas = []
        self.entropies = []
        self.gamma = gamma

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.policy_layer = nn.Linear(256, n_actions)
        self.device = torch.device(device)
        self.to(self.device)
        self.out_activation_fn = F.tanh

    def rescale_function(self, input):
        nn_min = self.out_activation_fn(torch.Tensor([float('-inf')])).to(self.device)
        nn_max = self.out_activation_fn(torch.Tensor([float('inf')])).to(self.device)
        magnitude = input - nn_min  # tanh goes from -1 to 1
        output = magnitude * (self.env_max - self.env_min) / (nn_max - nn_min) + self.env_min
        return output

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        # TODO: try sigmoid output function and modify rescale function?
        x = self._format(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        pi = self.policy_layer(x)
        pi = self.out_activation_fn(pi)
        pi = self.rescale_function(pi)
        return pi

    def choose_action(self, observation, noise_ratio=0.1):
        action = self.forward(observation).cpu().detach().data.numpy().squeeze()
        noise_scale = (self.env_max - self.env_min)/2 * noise_ratio
        # scale here is a standard dev, not range
        noise = np.random.normal(loc=0, scale=noise_scale, size=action.shape)
        action = action + noise
        action = np.clip(action, self.env_min, self.env_max)
        return action.reshape(-1)


class DDPG:
    def __init__(self, env_id, bounds, batch_size, update_interval, tau, gamma):
        super(DDPG, self).__init__()

        self.target_value_model = QVNet(n_states, n_action)
        self.online_value_model = QVNet(n_states, n_action)
        self.target_policy_model = PNet(n_states, n_action, bounds)
        self.online_policy_model = PNet(n_states, n_action, bounds)
        self.value_optimizer = torch.optim.Adam(self.online_value_model.parameters(), lr=LR)
        # self.value_optimizer = torch.optim.RMSprop(self.online_value_model.parameters(), lr=LR)
        self.policy_optimizer = torch.optim.Adam(self.online_policy_model.parameters(), lr=LR)
        # self.policy_optimizer = torch.optim.RMSprop(self.online_policy_model.parameters(), lr=LR)
        self.env = gym.make(env_id)
        self.buffer = NumpyReplayBuffer(100000, batch_size)
        self.update_interval = update_interval
        self.tau = tau
        self.gamma = gamma

        self.logpas = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.rewards_history = np.array([])
        self.running_timestep = 0
        self.running_reward = 0

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        # value updates:
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        # criterion = torch.nn.MSELoss()
        # value_loss = criterion(q_sa, target_q_sa.detach())
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), 0.75)
        self.value_optimizer.step()

        # policy updates:
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), 0.75)
        self.policy_optimizer.step()

    def interaction_step(self, state, explore=False):
        # altering a little because cart-pole has discrete action space
        with torch.no_grad():
            if explore:
                action = self.online_policy_model.choose_action(state, noise_ratio=1.0)
            else:
                action = self.online_policy_model.choose_action(state, noise_ratio=0.1)
        new_state, reward, terminated, truncated, info = self.env.step(action)
        experience = (state, action, reward, new_state, int(terminated))
        self.buffer.store(experience)
        self.running_reward += reward
        self.running_timestep += 1
        return new_state, terminated, truncated

    def update_networks(self):
        for target, online in zip(self.target_value_model.parameters(),
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(),
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def populate_buffer(self, n_batches=5):
        min_samples = self.buffer.batch_size * n_batches
        while len(self.buffer) <= min_samples:
            state, info = self.env.reset()
            terminal, truncated = False, False
            while not (terminal or truncated):
                new_state, terminal, truncated = self.interaction_step(state, explore=True)
        print(f"{len(self.buffer)} samples populated to buffer")

    def train(self, N_GAMES):
        episode = 0
        self.populate_buffer(10)
        while episode < N_GAMES:
            self.running_reward = 0
            self.running_timestep = 0
            state, info = self.env.reset()
            terminal, truncated = False, False
            while not (terminal or truncated):
                new_state, terminal, truncated = self.interaction_step(state)
                experiences = self.buffer.sample()
                experiences = self.online_value_model.load(experiences)
                self.optimize_model(experiences)
                if self.running_timestep % self.update_interval:
                    self.update_networks()
            self.rewards_history = np.append(self.rewards_history, np.average(self.running_reward))
            print(f"ep: {episode}, t: {self.running_timestep}, reward: {self.running_reward:.2f}, \t"
                  f"running rewards: {np.average(self.rewards_history[-100:]):.2f}")
            episode += 1
        print("training ended")
        self.env.close()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    BATCH_SIZE = 256
    LR = 0.001
    tau = 0.005
    gamma = 0.99
    env_id = "Pendulum-v1"
    env = gym.make(env_id)
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]  # this is a *continuous* action space!
    bounds = (env.action_space.low[0], env.action_space.high[0])
    env.close()
    update_interval = 1
    EPOCHS = 100
    agent = DDPG(env_id, bounds, BATCH_SIZE, update_interval, tau, gamma)
    agent.train(EPOCHS)
    plot_training_history(agent.rewards_history, save=False)


