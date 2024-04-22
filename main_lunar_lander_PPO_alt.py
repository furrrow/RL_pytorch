import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import plot_training_history
import numpy as np
import gymnasium as gym

""" PPO code implementation using pettingzoo's parallel environments
heavily referencing:
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_12/chapter-12.ipynb

https://gymnasium.farama.org/api/experimental/vector/

"""


class FCCA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 device="cpu",
                 activation_fc=F.relu):
        super(FCCA, self).__init__()
        self.activation_fc = activation_fc
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, output_dim)
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
        return x

    def forward(self, states):
        x = self._format(states)
        x = self.activation_fc(self.linear1(x))
        x = self.activation_fc(self.linear2(x))
        out = self.output_layer(x)
        return out

    def np_pass(self, states):
        states = torch.tensor(states).to(self.device)
        logits = self.forward(states)
        np_logits = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        logpas = dist.log_prob(actions)
        np_logpas = logpas.detach().cpu().numpy()
        is_exploratory = np_actions != np.argmax(np_logits, axis=1)
        return np_actions, np_logpas, is_exploratory

    def select_action(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu().item()

    def get_predictions(self, states, actions):
        states, actions = self._format(states), self._format(actions)
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        return logpas, entropies

    def select_greedy_action(self, states):
        logits = self.forward(states)
        return np.argmax(logits.detach().squeeze().cpu().numpy())


class FCV(nn.Module):
    def __init__(self,
                 input_dim,
                 device="cpu",
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
        return x

    def forward(self, states):
        x = self._format(states)
        # x = torch.tensor(states).to(self.device)
        x = self.activation_fc(self.linear1(x))
        x = self.activation_fc(self.linear2(x))
        out = self.output_layer(x)
        return out


class EpisodeBuffer:
    """
    WARNING, does not work yet with the vectorized env of the gymnasium.vector.VectorEnv
    """

    def __init__(self,
                 state_dim,
                 gamma,
                 tau,
                 n_workers,
                 max_episodes,
                 max_episode_steps,
                 device='cpu'):

        assert max_episodes >= n_workers
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        self.states_mem = np.empty(
            shape=(self.max_episodes, self.max_episode_steps, self.state_dim), dtype=np.float64)
        self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.uint8)
        self.returns_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.logpas_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)

        self.episode_steps = np.zeros(shape=self.max_episodes, dtype=np.uint16)
        self.episode_reward = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_exploration = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_seconds = np.zeros(shape=self.max_episodes, dtype=np.float64)
        self.current_ep_idxs = np.arange(self.n_workers, dtype=np.uint16)
        self.worker_rewards = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.float32)
        self.worker_exploratory = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.bool_)
        self.worker_steps = np.zeros(shape=self.n_workers, dtype=np.uint16)

        self.discounts = np.logspace(
            0, max_episode_steps + 1, num=max_episode_steps + 1, base=gamma, endpoint=False, dtype=np.float128)
        self.tau_discounts = np.logspace(
            0, max_episode_steps + 1, num=max_episode_steps + 1, base=gamma * tau, endpoint=False, dtype=np.float128)

        self.device = torch.device(device)
        self.buffer_full = False
        self.clear()

    def clear(self):
        self.states_mem = np.empty(
            shape=(self.max_episodes, self.max_episode_steps, self.state_dim), dtype=np.float64)
        self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.uint8)
        self.returns_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.logpas_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)

        self.episode_steps = np.zeros(shape=self.max_episodes, dtype=np.uint16)
        self.episode_reward = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_exploration = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_seconds = np.zeros(shape=self.max_episodes, dtype=np.float64)
        self.current_ep_idxs = np.arange(self.n_workers, dtype=np.uint16)
        self.states_mem[:] = 0
        self.actions_mem[:] = 0
        self.returns_mem[:] = 0
        self.gaes_mem[:] = 0
        self.logpas_mem[:] = 0
        self.worker_rewards[:] = 0
        self.worker_exploratory[:] = 0
        self.worker_steps[:] = 0
        self.buffer_full = False
        gc.collect()

    def remember(self, states, actions, logpas, rewards, terminals, truncateds):
        self.states_mem[self.current_ep_idxs, self.worker_steps] = states
        self.actions_mem[self.current_ep_idxs, self.worker_steps] = actions
        self.logpas_mem[self.current_ep_idxs, self.worker_steps] = logpas
        self.worker_rewards[np.arange(self.n_workers), self.worker_steps] = rewards

        for w_idx in range(self.n_workers):
            if self.worker_steps[w_idx] + 1 == self.max_episode_steps:
                truncateds[w_idx] = True

    def process_terminals(self, terminals, truncateds, value_model, next_states):
        terminal_or_truncated = np.logical_or(terminals, truncateds)
        next_values = np.zeros(shape=self.n_workers)
        if True in truncateds:
            # we bootstrap next values for truncated or non-terminal states
            idx_truncated = np.flatnonzero(truncateds)
            with torch.no_grad():
                next_values[idx_truncated] = value_model(next_states[idx_truncated]).squeeze().cpu().numpy()
        if terminal_or_truncated.sum():
            # process each terminal worker at a time:
            for w_idx in range(self.n_workers):
                e_idx = self.current_ep_idxs[w_idx]
                T = self.worker_steps[w_idx]
                self.episode_steps[e_idx] = T
                self.episode_reward[e_idx] = self.worker_rewards[w_idx, :T].sum()
                self.episode_exploration[e_idx] = self.worker_exploratory[w_idx, :T].mean()

                # append the bootstrapping value to reward vector, calculate predicted returns
                ep_rewards = np.concatenate((self.worker_rewards[w_idx, :T], [next_values[w_idx]]))
                ep_discounts = self.discounts[:T + 1]
                ep_returns = np.array(
                    [np.sum(ep_discounts[:T + 1 - t] * ep_rewards[t:]) for t in range(T)])
                self.returns_mem[e_idx, :T] = ep_returns

                ep_states = self.states_mem[e_idx, :T]
                # get the predicted values, append the bootstrapping value to the vector
                with torch.no_grad():
                    value_net_out = value_model(ep_states)
                    if len(value_net_out.shape) > 1:
                        value_net_out = value_net_out.squeeze(-1)
                    ep_values = torch.cat((
                        value_net_out,
                        torch.tensor([next_values[w_idx]], device=value_model.device, dtype=torch.float32)
                    ))
                np_ep_values = ep_values.view(-1).cpu().numpy()
                ep_tau_discounts = self.tau_discounts[:T]
                deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                # calculate the generalized advantage estimators, save to buffer
                gaes = np.array(
                    [np.sum(self.tau_discounts[:T - t] * deltas[t:]) for t in range(T)])
                self.gaes_mem[e_idx, :T] = gaes
                # reset and prepare for next episode
                self.worker_exploratory[w_idx, :] = 0
                self.worker_rewards[w_idx, :] = 0
                self.worker_steps[w_idx] = 0

                new_ep_id = max(self.current_ep_idxs) + 1
                if new_ep_id >= self.max_episodes:
                    self.buffer_full = True
                    break
                # go to next episode if buffer is not full
                self.current_ep_idxs[w_idx] = new_ep_id

    def process_episode_full(self, value_model):
        # episode is full:
        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]
        # remove from memory everything that isn't a number
        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem),
                                        device=value_model.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem),
                                     device=value_model.device)
        self.logpas_mem = [row[:ep_t[i]] for i, row in enumerate(self.logpas_mem[ep_idxs])]
        self.logpas_mem = torch.tensor(np.concatenate(self.logpas_mem),
                                       device=value_model.device)
        # statistics
        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def fill(self, policy_model, value_model):
        states, infos = self.envs.reset()
        idx_resets = None
        next_values = None

        worker_rewards = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.float32)
        worker_exploratory = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.bool_)
        worker_steps = np.zeros(shape=self.n_workers, dtype=np.uint16)

        self.buffer_full = False
        while not self.buffer_full and len(self.episode_steps[self.episode_steps > 0]) < self.max_episodes / 2:
            with torch.no_grad():
                actions, logpas, are_exploratory = policy_model.np_pass(states)
                values = value_model(states)

            next_states, rewards, terminals, truncateds, infos = self.envs.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.logpas_mem[self.current_ep_idxs, worker_steps] = logpas

            worker_exploratory[np.arange(self.n_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards

            for w_idx in range(self.n_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    truncateds[w_idx] = True

            terminal_or_truncated = np.logical_or(terminals, truncateds)
            if True in terminal_or_truncated:
                next_values = np.zeros(shape=self.n_workers)
                if True in truncateds:
                    # we bootstrap next values for truncated or non-terminal states
                    idx_truncated = np.flatnonzero(truncateds)
                    with torch.no_grad():
                        next_values[idx_truncated] = value_model(next_states[idx_truncated]).squeeze().cpu().numpy()

            states = next_states
            worker_steps += 1

            # process the workers if we have terminals
            if terminal_or_truncated.sum():
                # NOTE: gymnasium parallel envs reset themselves, unsure how this affects training
                # new_states, infos = self.envs.reset()
                # states = new_states

                # process each terminal worker at a time:
                for w_idx in range(self.n_workers):
                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()

                    # append the bootstrapping value to reward vector, calculate predicted returns
                    ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_discounts = self.discounts[:T + 1]
                    ep_returns = np.array(
                        [np.sum(ep_discounts[:T + 1 - t] * ep_rewards[t:]) for t in range(T)])
                    self.returns_mem[e_idx, :T] = ep_returns

                    ep_states = self.states_mem[e_idx, :T]
                    # get the predicted values, append the bootstrapping value to the vector
                    with torch.no_grad():
                        value_net_out = value_model(ep_states)
                        if len(value_net_out.shape) > 1:
                            value_net_out = value_net_out.squeeze(-1)
                        ep_values = torch.cat((
                            value_net_out,
                            torch.tensor([next_values[w_idx]], device=value_model.device, dtype=torch.float32)
                        ))
                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    ep_tau_discounts = self.tau_discounts[:T]
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    # calculate the generalized advantage estimators, save to buffer
                    gaes = np.array(
                        [np.sum(self.tau_discounts[:T - t] * deltas[t:]) for t in range(T)])
                    self.gaes_mem[e_idx, :T] = gaes
                    # reset and prepare for next episode
                    worker_exploratory[w_idx, :] = 0
                    worker_rewards[w_idx, :] = 0
                    worker_steps[w_idx] = 0

                    new_ep_id = max(self.current_ep_idxs) + 1
                    if new_ep_id >= self.max_episodes:
                        self.buffer_full = True
                        break
                    # go to next episode if buffer is not full
                    self.current_ep_idxs[w_idx] = new_ep_id

        # episode is full:
        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]
        # remove from memory everything that isn't a number
        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem),
                                        device=value_model.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem),
                                     device=value_model.device)
        self.logpas_mem = [row[:ep_t[i]] for i, row in enumerate(self.logpas_mem[ep_idxs])]
        self.logpas_mem = torch.tensor(np.concatenate(self.logpas_mem),
                                       device=value_model.device)
        # statistics
        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def get_stacks(self):
        return (self.states_mem, self.actions_mem,
                self.returns_mem, self.gaes_mem, self.logpas_mem)

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()


class PPO:
    def __init__(self,
                 env_id,
                 LR,
                 batch_size,
                 update_interval,
                 tau,
                 gamma,
                 max_episodes,
                 max_episode_steps,
                 n_envs,
                 device="cpu"):

        self.lr = LR
        self.env_id = env_id
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.tau = tau
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        self.n_envs = n_envs
        self.n_workers = n_envs
        self.device = device
        self.envs = gym.vector.make(env_id, num_envs=n_envs)
        self.n_states = self.envs.single_observation_space.shape[0]
        self.n_action = self.envs.single_action_space.n
        # self.envs = MultiprocessEnv(env_id, n_envs)

        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.episode_exploration = [], []
        self.evaluation_scores = []
        self.rewards_history = None

        self.policy_model = FCCA(self.n_states, self.n_action, device=self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), self.lr)

        self.value_model = FCV(self.n_states, device=self.device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), self.lr)

        self.episode_buffer = EpisodeBuffer(self.n_states, self.gamma, self.tau, self.n_workers,
                                            self.max_episodes, self.max_episode_steps)
        self.policy_optimizer_lr = 0.0003
        self.policy_optimization_epochs = 80
        self.policy_sample_ratio = 0.8
        self.policy_clip_range = 0.1
        self.policy_stopping_kl = 0.02
        self.value_optimization_epochs = 80
        self.value_sample_ratio = 0.8
        self.value_stopping_mse = 25
        self.entropy_loss_weight = 0.01
        self.policy_model_max_grad_norm = float('inf')
        self.value_model_max_grad_norm = float('inf')
        self.value_clip_range = float('inf')

        self.EPS = 1e-6

    def optimize_model(self):
        states, actions, returns, gaes, logpas = self.episode_buffer.get_stacks()
        values = self.value_model(states).detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + self.EPS)
        n_samples = len(actions)

        # policy updates
        for _ in range(self.policy_optimization_epochs):
            batch_size = int(self.policy_sample_ratio * n_samples)
            # sub sample from the full batch to form a mini-batch
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            logpas_batch = logpas[batch_idxs]

            logpas_pred, entropies_pred = self.policy_model.get_predictions(states_batch,
                                                                            actions_batch)
            # log probabilities to ratio of probabilities
            ratios = (logpas_pred - logpas_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(1.0 - self.policy_clip_range,
                                                       1.0 + self.policy_clip_range)
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight

            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),
                                           self.policy_model_max_grad_norm)
            self.policy_optimizer.step()

            # only optimize again if new policy is within bounds of original policy
            with torch.no_grad():
                logpas_pred_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (logpas - logpas_pred_all).mean()  # KL divergence
                if kl.item() > self.policy_stopping_kl:
                    break
        # value updates
        for _ in range(self.value_optimization_epochs):
            batch_size = int(self.value_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]

            values_pred = self.value_model(states_batch)
            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range,
                                                                                    self.value_clip_range)
            v_loss = (returns_batch - values_pred).pow(2)
            v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(),
                                           self.value_model_max_grad_norm)
            self.value_optimizer.step()

            with torch.no_grad():
                values_pred_all = self.value_model(states)
                mse = (values - values_pred_all).pow(2)
                mse = mse.mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    break

    def train(self, max_episodes):
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        states, infos = self.envs.reset()
        # self.envs = gym.wrappers.RecordEpisodeStatistics(self.envs, deque_size=self.n_envs * max_episodes)

        # collect n_steps rollout
        for episode in range(max_episodes):
            # while not term or truncated:
            episode_length_criteria = \
                len(self.episode_buffer.episode_steps[self.episode_buffer.episode_steps > 0]) < self.max_episodes / 2
            while not self.episode_buffer.buffer_full and episode_length_criteria:
                with torch.no_grad():
                    actions, logpas, are_exploratory = self.policy_model.np_pass(states)
                    values = self.value_model(states)
                next_states, rewards, terminals, truncateds, infos = self.envs.step(actions)
                self.episode_buffer.remember(states, actions, logpas, rewards, terminals, truncateds)
                terminal_or_truncated = np.logical_or(terminals, truncateds)
                if True in terminal_or_truncated:
                    self.episode_buffer.process_terminals(terminals, truncateds, self.value_model, next_states)
                states = next_states
                self.episode_buffer.worker_steps += 1
            episode_timestep, episode_reward, episode_exploration, \
                episode_seconds = self.episode_buffer.process_episode_full(self.value_model)

            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)
            self.optimize_model()
            self.episode_buffer.clear()
            print(f"episode {episode} avg rwd {np.average(episode_reward):.1f}; reward {episode_reward} ")

        print('Training complete.')

        # returns_list = self.envs.return_queue
        returns_list = self.episode_reward
        self.rewards_history = {env_id: np.array(returns_list)}
        self.envs.close()
        del self.envs
        return result



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    BATCH_SIZE = 256
    LR = 0.0005
    tau = 0.97
    gamma = 1.0
    env_id = "LunarLander-v2"
    update_interval = 5
    EPOCHS = 100
    n_workers = 4
    max_episodes = 16
    max_episode_steps = 1000  # truncated step set by env will take precedence
    agent = PPO(env_id, LR, BATCH_SIZE, update_interval, tau, gamma, max_episodes, max_episode_steps, n_workers)
    agent.train(EPOCHS)
    plot_training_history(agent.rewards_history, save=False)
