import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
from replay_buffer import NumpyReplayBuffer, EpisodeBuffer

""" PPO code implementation,
heavily referencing:
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_12/chapter-12.ipynb

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


class MultiprocessEnv(object):
    def __init__(self, env_name, n_workers):
        self.env_name = env_name
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [
            mp.Process(
                target=self.work,
                args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
        [w.start() for w in self.workers]
        self.dones = {rank: False for rank in range(self.n_workers)}

    def reset(self, ranks=None, **kwargs):
        state_list, info_list = [], []
        if not (ranks is None):
            [self.send_msg(('reset', {}), rank) for rank in ranks]
            for rank, (parent_end, _) in enumerate(self.pipes):
                if rank in ranks:
                    o, info = parent_end.recv()
                    state_list.append(o)
                    info_list.append(info)
            return np.vstack(state_list), np.vstack(info_list)
            # return np.stack([parent_end.recv() for rank, (parent_end, _) in enumerate(self.pipes) if rank in ranks])

        self.broadcast_msg(('reset', kwargs))
        for parent_end, _ in self.pipes:
            o, info = parent_end.recv()
            state_list.append(o)
            info_list.append(info)
        return np.vstack(state_list), np.vstack(info_list)
        # return np.stack([parent_end.recv() for parent_end, _ in self.pipes])

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(
            ('step', {'action': actions[rank]}),
            rank) for rank in range(self.n_workers)]
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
        return np.array(obs_list), np.array(rewards_list), np.array(term_list), np.array(trunk_list), np.array(
            info_list)

    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]

    def work(self, rank, worker_end):
        # env = self.make_env_fn(**self.make_env_kargs, seed=self.seed + rank)
        env = gym.make(self.env_name)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            # elif cmd == '_past_limit':
            #     worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                # including close command
                env.close(**kwargs);
                del env;
                worker_end.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [parent_end.send(msg) for parent_end, _ in self.pipes]


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

        assert n_envs > 1
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
        self.env = gym.make(env_id)
        self.n_states = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        self.bounds = (env.action_space.start, n_action - 1)
        self.env.close()
        # self.envs = gym.vector.make(env_id, num_envs=n_envs)
        self.envs = MultiprocessEnv(env_id, n_envs)

        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.episode_exploration = [], []
        self.evaluation_scores = []

        self.policy_model = FCCA(self.n_states, self.n_action, device=self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), self.lr)

        self.value_model = FCV(self.n_states, device=self.device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), self.lr)

        self.episode_buffer = EpisodeBuffer(self.n_states, self.gamma, self.tau, self.n_workers,
                                            self.max_episodes, self.max_episode_steps, self.envs)
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

        # collect n_steps rollout
        for episode in range(max_episodes):
            episode_timestep, episode_reward, episode_exploration, \
                episode_seconds = self.episode_buffer.fill(self.policy_model, self.value_model)

            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)
            self.optimize_model()
            self.episode_buffer.clear()
            print(f"episode {episode} avg rwd {np.average(episode_reward):.1f}; reward {episode_reward} ")

        print('Training complete.')
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
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.n
    bounds = (env.action_space.start, n_action - 1)
    env.close()
    update_interval = 5
    EPOCHS = 300
    n_workers = 2
    max_episodes = 16
    max_episode_steps = 1000  # truncated step set by env will take precedence
    agent = PPO(env_id, LR, BATCH_SIZE, update_interval, tau, gamma, max_episodes, max_episode_steps, n_workers)
    agent.train(EPOCHS)
    # plot_training_history(agent.rewards_history, save=False)
