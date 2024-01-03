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

        self.linear1 = nn.Linear(input_dim+1, 128)
        self.linear2 = nn.Linear(128, 64)
        self.value_layer = nn.Linear(64, 1)
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

        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.policy_layer = nn.Linear(64, n_actions)
        self.device = torch.device(device)
        self.to(self.device)

    # def rescale_function(self, input):
    #     magnitude = input - (-1)  # tanh goes from -1 to 1
    #     output = magnitude * (self.env_max - self.env_min) + self.env_min
    #     return output

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
        # pi = F.tanh(self.policy_layer(x))
        # pi = self.recale_function(pi)
        return pi

    def choose_action(self, observation):
        logits = self.forward(observation)
        dist = Categorical(logits=logits)
        action = dist.sample()
        # log_pa = dist.log_prob(action).unsqueeze(-1)
        # entropy = dist.entropy().unsqueeze(-1)
        # if len(action) == 1:
        #     action = action.item()
        # else:
        #     action = action.data.numpy()
        return action


class DDPG:
    def __init__(self, env_id, bounds, batch_size, update_interval, tau, gamma):
        super(DDPG, self).__init__()

        self.target_value_model = QVNet(n_states, n_action)
        self.online_value_model = QVNet(n_states, n_action)
        self.target_policy_model = PNet(n_states, n_action, bounds)
        self.online_policy_model = PNet(n_states, n_action, bounds)
        # self.value_optimizer = torch.optim.Adam(self.online_value_model.parameters(), lr=LR)
        self.value_optimizer = torch.optim.RMSprop(self.online_value_model.parameters(), lr=LR)
        # self.policy_optimizer = torch.optim.Adam(self.online_policy_model.parameters(), lr=LR)
        self.policy_optimizer = torch.optim.RMSprop(self.online_policy_model.parameters(), lr=LR)
        self.env = gym.make(env_id)
        self.buffer = NumpyReplayBuffer(10000, batch_size)
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
        target_action = self.target_policy_model.choose_action(next_states)
        max_a_q_sp = self.target_value_model(next_states, target_action)
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
        online_action = self.online_policy_model.choose_action(states)
        max_a_q_s = self.online_value_model(states, online_action)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), 0.75)
        self.policy_optimizer.step()

    def interaction_step(self, state):
        # altering a little because cart-pole has discrete action space
        with torch.no_grad():
            action = self.online_policy_model.choose_action(state)
        new_state, reward, terminated, truncated, info = self.env.step(action.item())
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
                new_state, terminal, truncated = self.interaction_step(state)
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
            print(f"episode: {episode}, timesteps: {self.running_timestep}, "
                  f"running rewards: {np.average(self.rewards_history[-100:]):.3f}")
            episode += 1
        print("training ended")
        self.env.close()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    BATCH_SIZE = 64
    LR = 0.005
    tau = 0.005
    gamma = 1.0
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.n
    bounds = (env.action_space.start, n_action-1)
    env.close()
    update_interval = 5
    EPOCHS = 2000
    agent = DDPG(env_id, bounds, BATCH_SIZE, update_interval, tau, gamma)
    agent.train(EPOCHS)
    plot_training_history(agent.rewards_history, save=False)


