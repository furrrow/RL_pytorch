import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from utils import plot_training_history
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
from tqdm import tqdm

""" A2C code implementation,
heavily referencing:
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb


"""


class A2CNet(nn.Module):

    def __init__(self, input_dim, n_actions, gamma=0.99, entropy_loss_weight=0.001):
        super(A2CNet, self).__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.logpas = []
        self.entropies = []
        self.gamma = gamma
        self.entropy_loss_weight = entropy_loss_weight

        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.value_layer = nn.Linear(64, 1)
        self.policy_layer = nn.Linear(64, n_actions)
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

    def remember(self, state, action, reward, log_prob_action, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logpas.append(log_prob_action)
        self.entropies.append(entropy)

    def clear_memories(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logpas = []
        self.entropies = []

    def consolidate_memory(self):
        self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        self.states = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        self.actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        self.logpas = [item.reshape(1) for item in self.logpas]
        self.entropies = [item.reshape(1) for item in self.entropies]
        self.logpas = torch.cat(self.logpas).to(self.device)
        self.entropies = torch.cat(self.entropies).to(self.device)

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        pi = self.policy_layer(x)
        v = self.value_layer(x)
        return pi, v

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_pa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        if len(action) == 1:
            action = action.item()
        else:  # len(action) > 1
            action = action.data.numpy()
        return action, log_pa, entropy, value


class MultiprocessEnv(object):
    # TODO: replace this multi-env with gymnasium's built-in vectorized envs!
    def __init__(self, env_name, n_workers):
        self.env_name = env_name
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [
            mp.Process(target=self.work,
                       args=(rank, self.pipes[rank][1]))
            for rank in range(self.n_workers)
        ]
        [w.start() for w in self.workers]
        self.dones = {rank: False for rank in range(self.n_workers)}

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [parent_end.send(msg) for parent_end, _ in self.pipes]

    def reset(self, rank=None, **kwargs):
        if rank is not None:
            parent_end, _ = self.pipes[rank]
            self.send_msg(('reset', {}), rank)
            o, info = parent_end.recv()
            return o, info
        self.broadcast_msg(('reset', kwargs))
        state_list, info_list = [], []
        for parent_end, _ in self.pipes:
            o, info = parent_end.recv()
            state_list.append(o)
            info_list.append(info)
        return np.vstack(state_list), np.vstack(info_list)

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(('step', {'action': actions[rank]}), rank)
         for rank in range(self.n_workers)]
        obs_list = []
        rewards_list = []
        term_list = []
        trunk_list = []
        info_list = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            obs, reward, term, trunc, info = parent_end.recv()
            obs_list.append(obs)
            rewards_list.append(reward)
            term_list.append(term)
            trunk_list.append(trunc)
            info_list.append(info)
        return np.array(obs_list), np.array(rewards_list), np.array(term_list), np.array(trunk_list), np.array(info_list)

    def close(self, **kwargs):
        self.broadcast_msg(("close", kwargs))
        [w.join() for w in self.workers]

    def past_limit(self, **kwargs):
        self.broadcast_msg(("past_limit", kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])

    def work(self, rank, worker_end):
        env = gym.make(self.env_name)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == "reset":
                worker_end.send(env.reset(**kwargs))
            elif cmd == "step":
                worker_end.send(env.step(**kwargs))
            # elif cmd == "past_limit":
            #     worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                env.close(**kwargs)
                del env
                worker_end.close()
                break


class A2C:
    def __init__(self, model, optimizer, env_id, entropy_loss_weight, max_n_steps, n_workers, tau, gamma):
        super(A2C, self).__init__()

        assert n_workers > 0
        self.model = model
        self.optimizer = optimizer
        self.envs = MultiprocessEnv(env_id, n_workers)

        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau
        self.gamma = gamma
        self.env_fn = None

        self.logpas = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.rewards_history = np.array([])
        self.running_timestep = 0
        self.running_reward = 0

    def optimize_model(self):
        # NOTE this is using GAE (generalized advantage estimation)
        logpas = torch.stack(self.logpas).squeeze()  # [15, 2, 1] -> [15, 2]
        entropies = torch.stack(self.entropies).squeeze()  # [15, 2, 1] -> [15, 2]
        values = torch.stack(self.values).squeeze()  # [16, 2, 1] -> [16, 2]

        T = len(self.rewards)  # T=16
        # create discounted returns
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)  # [T]
        rewards = np.array(self.rewards).squeeze()  # [16, 2]
        disc_returns = [[np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)]
                   for w in range(self.n_workers)]
        disc_returns = np.array(disc_returns)  # [2, 16]
        # array of values and (gamma * tau)^t
        np_values = values.data.numpy()  # [16, 2]
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)  # [15]
        # TD errors: R_t + gamma*value_(t+1) - value_t for t=0:T
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]  # [15, 2]
        # GAEs: tau discounts * td errors
        gaes = [[np.sum(tau_discounts[:T - 1 - t] * advs[t:, w]) for t in range(T - 1)]
                for w in range(self.n_workers)]  # [2, 15]
        discounted_gaes = discounts[:-1] * np.array(gaes)  # [2, 15]

        values = values[:-1, ...].view(-1).unsqueeze(1)  # [16, 2] -> [15, 2] -> [30, 1]
        logpas = logpas.view(-1).unsqueeze(1)  # [15, 2] -> [30, 1]
        entropies = entropies.view(-1).unsqueeze(1)  # [15, 2] -> [30, 1]
        returns = disc_returns.T[:-1].reshape(-1, 1)  # [2, 16] -> [15, 2] -> [30, 1]
        returns = torch.tensor(returns)
        discounted_gaes = discounted_gaes.T.reshape(-1, 1)  # [2, 15] -> [30, 1]
        discounted_gaes = torch.tensor(discounted_gaes)

        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = 1.0 * policy_loss + 0.6 * value_loss + entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)

        self.optimizer.step()

    def interaction_step(self, states):
        actions, log_pas, entropies, values = self.model.choose_action_nograd(states)
        new_states, rewards, is_terminals, truncateds, infos = self.envs.step(actions)
        self.logpas.append(log_pas)
        self.rewards.append(rewards)
        self.entropies.append(entropies)
        self.values.append(values)

        self.running_reward += rewards
        self.running_timestep += 1
        return new_states, is_terminals, truncateds

    def train(self, N_GAMES):
        episode = 0
        while episode < N_GAMES:
            self.running_reward = 0
            self.running_timestep = 0
            states, infos = self.envs.reset()
            while True:
                new_states, terminals, truncateds = self.interaction_step(states)
                if (True in terminals) or (True in truncateds):
                    break
                if self.running_timestep > self.max_n_steps:
                    break
                states = new_states
            pi, next_values = self.model.forward(states)  # pi: [1,2,2] v: [1,2,1]
            next_values = next_values.squeeze()  # tensor, shape [2]
            self.values.append(next_values.reshape(self.values[-1].shape))
            self.rewards.append(next_values.detach().numpy() * (1 - terminals))

            self.optimize_model()
            self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
            self.rewards_history = np.append(self.rewards_history, np.average(self.running_reward))
            print(f"episode: {episode}, timesteps: {self.running_timestep}, "
                  f"running rewards: {np.average(self.rewards_history[-100:]):.3f}")
            episode += 1
        print("training ended")

    def close(self):
        self.envs.close()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    LR = 0.005
    entropy_loss_weight = 0.005
    tau = 0.95
    gamma = 0.99
    env_id = "CartPole-v1"
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.n
    env.close()
    MAX_N_STEPS = 1000
    EPOCHS = 2000
    # n_workers = mp.cpu_count()-1
    n_workers = 12
    actor_critic = A2CNet(n_states, n_action)
    # optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LR)
    optimizer = torch.optim.RMSprop(actor_critic.parameters(), lr=LR)
    agents = A2C(actor_critic, optimizer, env_id, entropy_loss_weight,
                 MAX_N_STEPS, n_workers, tau, gamma)
    agents.train(EPOCHS)
    history = agents.rewards_history.copy()
    agents.close()
    plot_training_history(history, save=False)


