import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from replay_buffer import MultiAgentNumpyBuffer
from pettingzoo.mpe import simple_v3, simple_adversary_v3

""" MADDPG code implementation,
adapting Miguel's DDPG for multi-agent env.

Using the MPE environment, originally from openAI but using the pettingzoo version

"""


def show():
    env = simple_adversary_v3.env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(f"agent name: {agent}")
        print(f"n_agents: {env.num_agents}")
        print(f"observation space: {env.observation_space(agent)}")
        print(f"action_space: {env.action_space(agent)}")
        print(f"observation: {observation}")
        print(f"reward: {reward}")
        print(f"termination: {termination}, truncation, {truncation}")
        print(f"info: {info}")
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        env.step(action)
    env.close()


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc1 = nn.Linear(input_dims + n_agents * 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim, env_bounds, device):
        super(ActorNetwork, self).__init__()
        self.states = []
        self.actions = []
        self.env_min = torch.tensor(env_bounds[0])
        self.env_max = torch.tensor(env_bounds[1])
        self.rewards = []
        self.logpas = []
        self.entropies = []

        self.linear1 = nn.Linear(input_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_layer = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device(device)
        self.to(self.device)
        self.out_activation_fn = F.tanh

    def rescale_function(self, input):
        nn_min = self.out_activation_fn(torch.Tensor([float('-inf')])).to(self.device)
        nn_max = self.out_activation_fn(torch.Tensor([float('inf')])).to(self.device)
        magnitude = input - nn_min  # tanh goes from -1 to 1
        output = magnitude * (self.env_max - self.env_min) / (nn_max - nn_min) + self.env_min
        return output

    def forward(self, state):
        x = torch.Tensor(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        pi = self.policy_layer(x)
        pi = self.out_activation_fn(pi)
        pi = self.rescale_function(pi)
        return pi

    def choose_action(self, observation, explore=True, noise_ratio=0.1):
        with torch.no_grad():
            state = torch.Tensor(observation).to(self.device)
            action = self.forward(state).cpu().detach().data.numpy().squeeze()
        noise_ratio = 1 if explore else noise_ratio
        noise_scale = (self.env_max - self.env_min) / 2 * noise_ratio
        noise = np.random.normal(loc=0, scale=noise_scale.squeeze())
        action = action + noise
        action = np.clip(action, self.env_min, self.env_max)
        return action


class Agent:
    def __init__(self, state_size, total_state_size, action_size, total_action_size, bounds, n_agents, agent_name, device,
                 lr_a=0.01, lr_c=0.01, hidden_dim=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.total_state_size = total_state_size
        self.action_size = action_size
        self.total_action_size = total_action_size
        self.n_agents = n_agents
        self.bounds = bounds
        self.agent_name = agent_name
        self.online_actor = ActorNetwork(state_size, action_size, hidden_dim, bounds, device)
        self.online_critic = CriticNetwork(total_state_size, total_action_size, hidden_dim, device)
        self.target_actor = ActorNetwork(state_size, action_size, hidden_dim, bounds, device)
        self.target_critic = CriticNetwork(total_state_size, total_action_size, hidden_dim, device)
        self.actor_optimizer = torch.optim.Adam(self.online_actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.online_critic.parameters(), lr=lr_c)
        self.device = device

    def update_networks(self):
        for target, online in zip(self.target_critic.parameters(),
                                  self.online_critic.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_actor.parameters(),
                                  self.online_actor.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)


class MADDPG:
    def __init__(self, env, device, lr_a=0.01, lr_b=0.01, fc_dims=64, gamma=0.99,
                 tau=0.01, batch_size=64):
        env.reset()
        self.env = env
        self.agent_names = env.agents
        self.n_agents = len(self.agent_names)
        self.batch_size = batch_size
        self.obs_dims = {}
        self.action_ranges = {}
        self.n_actions = {}
        for agent in self.env.agents:
            dim = env.observation_space(agent).shape[0]
            self.obs_dims[agent] = dim
            self.n_actions[agent] = env.action_space(agent).shape[0]
            action_range = [env.action_space(agent).low[0], env.action_space(agent).high[0]]
            self.action_ranges[agent] = action_range

        self.total_states = sum(self.obs_dims.values())
        self.total_actions = sum(self.n_actions.values())

        self.QAgents = []
        for agent_name in self.agent_names:
            self.QAgents.append(
                Agent(self.obs_dims[agent_name], self.total_states, self.n_actions[agent_name], self.total_actions,
                      self.action_ranges[agent_name], self.n_agents, agent_name, device, lr_a, lr_b, fc_dims, gamma, tau)
            )
        self.buffer = MultiAgentNumpyBuffer(1000000, self.agent_names, batch_size=batch_size)
        self.rewards_history = np.array([])
        self.running_timestep = 0
        self.running_reward = 0
        self.update_interval = 5
        self.device = device

    def choose_actions(self, raw_obs, explore=True):
        actions = {}
        for agent in self.QAgents:
            name = agent.agent_name
            action = agent.online_actor.choose_action(raw_obs[name], explore)
            actions[name] = action
        return actions

    def train(self, n_epochs):
        for episode in range(n_epochs):
            self.running_reward = np.zeros(self.n_agents)
            self.running_timestep = 0
            state, infos = env.reset()
            while env.agents:
                state, terminal, truncated = self.interaction_step(state)
            self.optimize_model()
            if self.running_timestep % self.update_interval == 0:
                [agent.update_networks() for agent in self.QAgents]
            self.rewards_history = np.append(self.rewards_history, np.average(self.running_reward))
            print(f"ep: {episode}, t: {self.running_timestep}, reward: {self.running_reward:.2f}, \t"
                  f"running rewards: {np.average(self.rewards_history[-100:]):.2f}")

    def interaction_step(self, state):
        actions = maddpg_agents.choose_actions(state, explore=self.buffer.size < BATCH_SIZE)
        new_state, reward, done, truncated, info = env.step(actions)
        experience = (state, actions, reward, new_state, done)
        self.buffer.store(experience)
        self.running_reward += list(reward.values())
        self.running_timestep += 1
        return new_state, done, truncated

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            print("buffer not filled yet, buffer count:", len(self.buffer))
            return
        sample = self.buffer.sample()
        all_states = np.concatenate(sample[0], axis=1)
        all_actions = np.concatenate(sample[1], axis=1)
        all_next_states = np.concatenate(sample[3], axis=1)
        all_states = torch.from_numpy(all_states).float().to(self.device)
        all_actions = torch.from_numpy(all_actions).float().to(self.device)
        all_next_states = torch.from_numpy(all_next_states).float().to(self.device)

        # aggregate agent actor policies
        current_state_policies = []
        next_state_policies = []
        for agent_idx, agent in enumerate(self.QAgents):
            agent_state, agent_action, agent_reward, agent_next_state, agent_dones = \
                self.get_agent_experiences(sample, agent_idx)
            argmax_a_q_s = agent.online_actor(agent_state)
            argmax_a_q_sp = agent.target_actor(agent_next_state)
            current_state_policies.append(argmax_a_q_s)
            next_state_policies.append(argmax_a_q_sp)

        current_state_policies = torch.cat(current_state_policies, 1)  # [5,15]
        next_state_policies = torch.cat(next_state_policies, 1)  # [5, 15]
        # value updates
        for agent_idx, agent in enumerate(self.QAgents):
            agent_state, agent_action, agent_reward, agent_next_state, agent_dones = \
                self.get_agent_experiences(sample, agent_idx)

            # value updates
            agent.critic_optimizer.zero_grad()
            # argmax_a_q_sp = agent.target_actor(agent_next_state)
            max_a_q_sp = agent.target_critic(all_next_states, next_state_policies)
            target_q_sa = agent_reward + agent.gamma * max_a_q_sp * (1 - agent_dones)
            q_sa = agent.online_critic(all_states.clone(), all_actions.clone())
            td_error = q_sa - target_q_sa.detach()
            # value_loss = td_error.pow(2).mul(0.5).mean()
            value_loss = F.mse_loss(q_sa, target_q_sa.detach())
            print(agent_idx, "value_loss backward pass:")
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.online_critic.parameters(), float('inf'))
            agent.critic_optimizer.step()

            # policy updates:
            agent.actor_optimizer.zero_grad()
            # argmax_a_q_s = agent.online_actor(agent_state)
            max_a_q_s = agent.online_critic(all_states.clone(), current_state_policies.clone())
            policy_loss = -max_a_q_s.mean()
            print(agent_idx, "policy_loss backward pass:")
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.online_actor.parameters(), float('inf'))
            agent.actor_optimizer.step()

    def get_agent_experiences(self, sample, agent_idx):
        states, actions, rewards, next_states, is_terminals = sample
        agent_state = states[agent_idx]
        agent_action = actions[agent_idx]
        agent_reward = rewards[agent_idx]
        agent_next_state = next_states[agent_idx]
        agent_done = is_terminals[agent_idx]
        agent_state = torch.from_numpy(agent_state).float().to(self.device)
        agent_action = torch.from_numpy(agent_action).float().to(self.device)
        agent_reward = torch.from_numpy(agent_reward).float().to(self.device)
        agent_next_state = torch.from_numpy(agent_next_state).float().to(self.device)
        agent_done = torch.from_numpy(agent_done).float().to(self.device)
        return agent_state, agent_action, agent_reward, agent_next_state, agent_done


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    # show()
    # scenario_name = "simple"
    scenario_name = "simple_adversary"
    # env = simple_v3.parallel_env(render_mode="human", continuous_actions=True)
    # env = simple_v3.parallel_env(render_mode=None, continuous_actions=True)
    env = simple_adversary_v3.parallel_env(render_mode=None, continuous_actions=True)
    PRINT_INTERVAL = 100
    N_GAMES = 30000
    BATCH_SIZE = 5
    torch.autograd.set_detect_anomaly(True)

    maddpg_agents = MADDPG(env, device, lr_a=0.01, lr_b=0.01, fc_dims=64, gamma=0.99, tau=0.01,
                           batch_size=BATCH_SIZE)
    maddpg_agents.train(N_GAMES)
