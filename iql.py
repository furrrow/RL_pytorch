import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_v3
from tqdm import tqdm
import pickle

from policy.EGreedyExpStrategy import EGreedyExpStrategy
from replay_buffer import PrioritizedReplayBuffer

"""
Jim T's version of IQL
- DQN with double Q-learning, dueling network and prioritized experience replay improvements
- Prioritized Replay Buffer module code from https://github.com/mimoralea/gdrl, 
- comments added for more clarity
"""


class FCDuelingQ(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 v_hidden_dims=(32,),
                 a_hidden_dims=(32,),
                 activation_fc=nn.ReLU,
                 device=torch.device("cpu")):
        super(FCDuelingQ, self).__init__()

        # build hidden layers for features, value stream, and advantage stream
        feature_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            feature_hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            feature_hidden_layers.append(activation_fc())

        v_hidden_layers = nn.ModuleList()
        for i in range(len(v_hidden_dims) - 1):
            v_hidden_layers.append(nn.Linear(v_hidden_dims[i], v_hidden_dims[i + 1]))
            v_hidden_layers.append(activation_fc())

        a_hidden_layers = nn.ModuleList()
        for i in range(len(a_hidden_dims) - 1):
            a_hidden_layers.append(nn.Linear(a_hidden_dims[i], a_hidden_dims[i + 1]))
            a_hidden_layers.append(activation_fc())

        # build features, value stream, and advantage stream
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *feature_hidden_layers
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], v_hidden_dims[0]),
            activation_fc(),
            *v_hidden_layers,
            nn.Linear(v_hidden_dims[-1], 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], a_hidden_dims[0]),
            activation_fc(),
            *a_hidden_layers,
            nn.Linear(a_hidden_dims[-1], output_dim)
        )

        if not torch.cuda.is_available():
            device = torch.device("cpu")

        self.device = device
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qs = values + (advantages - advantages.mean())
        return qs

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class DDQN():
    def __init__(self,
                 value_model_fn=lambda num_obs, nA: FCDuelingQ(num_obs, nA),  # state vars, nA -> model
                 value_optimizer_fn=lambda params, lr: optim.RMSprop(params, lr),  # model params, lr -> optimizer
                 value_optimizer_lr=1e-4,  # optimizer learning rate
                 loss_fn=nn.MSELoss(),  # input, target -> loss
                 exploration_strategy_fn=lambda: EGreedyExpStrategy(),
                 # module with select_action function (model, state) -> action
                 replay_buffer_fn=lambda: PrioritizedReplayBuffer(10000),
                 max_gradient_norm=None,
                 tau=0.005,
                 target_update_steps=1
                 ):
        self.gamma = None
        self.memory = None
        self.exploration_strategy = None
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.loss_fn = loss_fn
        self.exploration_strategy_fn = exploration_strategy_fn
        self.memory_fn = replay_buffer_fn
        self.max_gradient_norm = max_gradient_norm
        self.tau = tau
        self.target_update_steps = target_update_steps

    def _init_model(self, env):
        # initialize online and target models
        self.online_model = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model.load_state_dict(
            self.online_model.state_dict())  # copy online model parameters to target model
        # initialize optimizer
        self.optimizer = self.value_optimizer_fn(self.online_model.parameters(), lr=self.value_optimizer_lr)

    def _copy_model(self, env):
        copy = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
        copy.load_state_dict(self.online_model.state_dict())
        return copy

    def marl_copy_model(self, env, agent):
        copy = self.value_model_fn(len(env.observation_space(agent).sample()), env.action_space(agent).n)
        copy.load_state_dict(self.online_model.state_dict())
        return copy

    # initialize independent DQN model for agent given MARL (pettingzoo) environment
    def marl_init_model(self, env, agent, gamma, batch_size=None):
        self.gamma = gamma
        # initialize online and target models
        self.online_model = self.value_model_fn(len(env.observation_space(agent).sample()), env.action_space(agent).n)
        self.target_model = self.value_model_fn(len(env.observation_space(agent).sample()), env.action_space(agent).n)
        self.target_model.load_state_dict(
            self.online_model.state_dict())  # copy online model parameters to target model
        # initialize optimizer
        self.optimizer = self.value_optimizer_fn(self.online_model.parameters(), lr=self.value_optimizer_lr)

        # initialize replay memory
        self.memory = self.memory_fn()
        self.batch_size = batch_size if batch_size else self.memory.batch_size

        # initialize exploration strategy
        self.exploration_strategy = self.exploration_strategy_fn()

    def update_target_network(self, tau=None):
        tau = tau if tau else self.tau
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_weights = tau * online.data + (1 - tau) * target.data
            target.data.copy_(target_weights)

    def optimize_model(self, batch_size=None):
        idxs, weights, experiences = self.memory.sample(batch_size)
        weights = self.online_model.numpy_float_to_device(weights)
        experiences = self.online_model.load(experiences)  # numpy to tensor; move to device
        states, actions, rewards, next_states, is_terminals = experiences

        # select best action of next state according to online model
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        # get values of next states using target network
        max_a_q_sp = self.target_model(next_states).detach()[np.arange(len(idxs)), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))  # calculate q target

        q_sa = self.online_model(states).gather(1, actions)  # get predicted q from model for each state, action pair

        # weigh sample losses by importance sampling for bias correction
        loss = self.loss_fn(weights * q_sa, weights * target_q_sa)  # calculate loss between prediction and target

        # optimize step (gradient descent)
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.optimizer.step()

        # update TD errors
        td_errors = (q_sa - target_q_sa).detach().cpu().numpy()
        self.memory.update(idxs, td_errors)

    def get_action(self, state, greedy=False):
        if greedy:
            return self.online_model(state).detach().max(1).indices.view(1, 1).item()
        else:
            return self.exploration_strategy.select_action(self.online_model, state)

    def train(self, env, gamma=1.0, num_episodes=100, batch_size=None, n_warmup_batches=5, tau=0.005,
              target_update_steps=1, save_models=None):
        self.exploration_strategy = self.exploration_strategy_fn()
        self.memory = self.memory_fn()
        if save_models:  # list of episodes to save models
            save_models.sort()
        self.gamma = gamma
        self._init_model(env)

        saved_models = {}
        best_model = None

        i = 0
        i_prev = 0
        episode_returns = np.zeros(num_episodes)
        for episode in range(num_episodes):
            state = env.reset()[0]
            ep_return = 0
            for t in count():
                i += 1
                action = self.get_action(state)  # use online model to select action
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.memory.store((state, action, reward, next_state, terminated))  # store experience in replay memory

                state = next_state

                if len(self.memory) >= batch_size * n_warmup_batches:  # optimize policy model
                    self.optimize_model(batch_size)

                # update target network with tau
                if i % target_update_steps == 0:
                    self.update_target_network(tau)

                ep_return += reward * gamma ** t  # add discounted reward to return
                if terminated or truncated:
                    # save best model
                    if ep_return >= episode_returns.max():
                        copy = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
                        copy.load_state_dict(self.online_model.state_dict())
                        best_model = copy
                    # copy and save model
                    if save_models and len(saved_models) < len(save_models) and episode + 1 == save_models[
                        len(saved_models)]:
                        copy = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
                        copy.load_state_dict(self.online_model.state_dict())
                        saved_models[episode + 1] = copy

                    episode_returns[episode] = ep_return
                    break
            print(f"ep: {episode}, t: {i - i_prev}, "
                  f"epsilon: {self.epsilon:.3f}, reward: {self.running_reward[-1]:.3f}, "
                  f"running 20 rwd {np.average(np.array(self.rewards_history)[:, -1][-20:]):.3f}")
            i_prev = i

        return episode_returns, best_model, saved_models


class IQL():
    def __init__(self,
                 dqn_fn=lambda agent: DDQN(),  # agent str -> DQN model
                 ):
        self.dqn_fn = dqn_fn
        self.dqns = {}  # dictionary of agent str: dqn model

    def _init_dqns(self, env, gamma, batch_size=None):
        for agent in env.possible_agents:
            dqn = self.dqn_fn(agent)
            dqn.marl_init_model(env, agent, gamma, batch_size)
            self.dqns[agent] = dqn

    def train(self, env, gamma=1.0, num_episodes=100, parallel=False, batch_size=None, n_warmup_batches=5, tau=None,
              target_update_steps=None, save_models=None):
        self._init_dqns(env, gamma, batch_size)

        episode_returns = {agent: [] for agent in self.dqns}
        iter = {agent: 0 for agent in self.dqns}

        saved_models = {}
        best_model = {agent: None for agent in self.dqns}

        for episode in range(num_episodes):
            # previous iter experience variables (state and action) for each agent tracked in dict
            experience = {agent: {'state': None, 't': 0, 'return': 0} for agent in self.dqns}
            env.reset()
            for agent in env.agent_iter():
                iter[agent] += 1
                dqn = self.dqns[agent]
                exp = experience[agent]
                next_state, reward, terminated, truncated, _ = env.last()
                # store experience from last action into replay memory
                if exp['state'] is not None:
                    dqn.memory.store((experience[agent]['state'], exp['action'], reward, next_state, terminated))

                if len(dqn.memory) >= dqn.batch_size * n_warmup_batches:  # optimize policy model
                    dqn.optimize_model(batch_size)

                # update target network with tau
                dqn_target_update_steps = target_update_steps if target_update_steps else dqn.target_update_steps
                if iter[agent] % dqn_target_update_steps == 0:
                    dqn.update_target_network(tau)

                exp['return'] += reward * gamma ** exp['t']  # keep track of individual agent's returns
                exp['t'] += 1
                if terminated or truncated:
                    episode_returns[agent].append(exp['return'])
                    exp['action'] = None
                else:
                    exp['action'] = dqn.get_action(next_state)  # get action from model
                    exp['state'] = next_state

                env.step(exp['action'])

            for agent in self.dqns:
                # save best model
                if exp['return'] >= np.max(episode_returns[agent]):
                    best_model[agent] = self.dqns[agent].marl_copy_model(env, agent)
            # copy and save models
            if save_models and len(saved_models) < len(save_models) and episode + 1 == save_models[len(saved_models)]:
                saved_models[episode + 1] = {agent: self.dqns[agent].marl_copy_model(env, agent) for agent in
                                             self.dqns}

        return episode_returns, best_model, saved_models


if __name__ == '__main__':
    # env = simple_spread_v3.env()
    env = simple_v3.env(max_cycles=75, continuous_actions=False)

    iql = IQL(dqn_fn=lambda _: DDQN(
        value_model_fn=lambda num_obs, nA: FCDuelingQ(num_obs, nA, hidden_dims=(512,), device=torch.device("cuda")),
        value_optimizer_lr=0.0005,
        exploration_strategy_fn=lambda: EGreedyExpStrategy(min_epsilon=0.01),
        replay_buffer_fn=lambda: PrioritizedReplayBuffer()))

    episode_returns, best_model, saved_models = iql.train(env, num_episodes=50, tau=0.01, batch_size=128,
                                                          n_warmup_batches=5,
                                                          save_models=[1, 250, 500, 1000, 2500, 5000])
    results = {'episode_returns': episode_returns, 'best_model': best_model, 'saved_models': saved_models}
    print(results)

    with open('testfiles/iql_simple.results', 'wb') as file:
        pickle.dump(results, file)
