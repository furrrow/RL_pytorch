import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from itertools import count
from replay_buffer import NumpyReplayBuffer


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

    def __init__(self, input_dim, env_bounds: tuple, gamma=0.99):
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
        self.policy_layer = nn.Linear(256, 1)
        self.device = torch.device(device)
        self.to(self.device)
        self.out_activation_fn = F.tanh

    def rescale_function(self, input):
        nn_min = self.out_activation_fn(torch.Tensor([float('-inf')])).to(self.device)
        nn_max = self.out_activation_fn(torch.Tensor([float('inf')])).to(self.device)
        self.env_max = self._format(self.env_max)
        self.env_min = self._format(self.env_min)
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


class NormalNoiseStrategy():
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0

    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)

        self.ratio_noise_injected = np.mean(abs((greedy_action - action) / (self.high - self.low)))
        return action


class DDPG():
    def __init__(self,
                 replay_buffer_fn,
                 policy_model_fn,
                 policy_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau):
        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.training_strategy_fn = training_strategy_fn

        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer.step()

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, state, env):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        action = self.training_strategy.select_action(self.online_policy_model,
                                                      state,
                                                      len(self.replay_buffer) < min_samples)
        new_state, reward, is_terminal, is_truncated, info = env.step(action)
        # is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
        return new_state, (is_terminal or is_truncated)

    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model.parameters(),
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(),
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, env, gamma, max_episodes):
        training_start, last_debug_time = time.time(), float('-inf')
        self.gamma = gamma

        env = env

        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model,
                                                       self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model,
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = training_strategy_fn(action_bounds)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        for episode in range(1, max_episodes + 1):
            state, info = env.reset()
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break
            print(f"ep: {episode}, t: {self.episode_timestep[-1]}, reward: {self.episode_reward[-1]:.2f} \t"
                  f"running rewards: {np.average(self.episode_reward[-100:]):.2f}")

        # final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        env.close();
        del env
        # self.get_cleaned_checkpoints()
        return result, wallclock_time


if __name__ == '__main__':
    ddpg_results = []
    best_agent, best_eval_score = None, float('-inf')
    environment_settings = {
        'env_name': 'Pendulum-v1',
        'gamma': 0.99,
        'max_minutes': 20,
        'max_episodes': 500,
        'goal_mean_100_reward': -150
    }
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)

    # policy_model_fn = lambda nS, bounds: FCDP(nS, bounds, hidden_dims=(256, 256))
    policy_model_fn = lambda nS, bounds: PNet(nS, bounds)
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    # value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(256, 256))
    value_model_fn = lambda nS, nA: QVNet(nS, nA)
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0003

    training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1)

    replay_buffer_fn = lambda: NumpyReplayBuffer(max_size=100000, batch_size=256)
    n_warmup_batches = 5
    update_target_every_steps = 1
    tau = 0.005

    env_name, gamma, max_minutes, \
        max_episodes, goal_mean_100_reward = environment_settings.values()

    agent = DDPG(replay_buffer_fn,
                 policy_model_fn,
                 policy_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau)

    env = gym.make(env_name)
    result, wallclock_time = agent.train(env, gamma, max_episodes)
    print(result)
