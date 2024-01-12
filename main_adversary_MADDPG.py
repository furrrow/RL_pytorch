import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import os
import numpy as np
import gymnasium as gym
from utils import plot_training_history
from replay_buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_v3, simple_adversary_v3
from tqdm import tqdm

""" MADDPG code implementation,
heavily referencing:
https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients

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
    def __init__(self, lr, input_dims, hidden_dim, n_agents, n_actions, name, checkpoint_dir, device):
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(self.checkpoint_file)

        # self.fc1 = nn.Linear(input_dims + n_agents * n_actions, hidden_dim)
        self.fc1 = nn.Linear(input_dims + n_agents * 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden_dim, n_actions, name, checkpoint_dir, device):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(self.checkpoint_file)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_name, checkpoint_dir,
                 device, lr_a=0.01, lr_c=0.01, hidden_dim=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agent_name = agent_name
        self.actor = ActorNetwork(lr_a, actor_dims, hidden_dim, n_actions,
                                  f"{self.agent_name}_actor.pt", checkpoint_dir, device)
        self.critic = CriticNetwork(lr_c, critic_dims, hidden_dim, n_agents, n_actions,
                                    f"{self.agent_name}_critic.pt", checkpoint_dir, device)
        self.target_actor = ActorNetwork(lr_a, actor_dims, hidden_dim, n_actions,
                                         f"{self.agent_name}_target_actor.pt", checkpoint_dir, device)
        self.target_critic = CriticNetwork(lr_c, critic_dims, hidden_dim, n_agents, n_actions,
                                           f"{self.agent_name}_target_critic.pt", checkpoint_dir, device)
        self.update_network_parameters(tau=1)
        self.device = device

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_portion = actor_state_dict[name].clone()
            target_portion = target_actor_state_dict[name].clone()
            actor_state_dict[name] = tau * actor_portion + (1 - tau) * target_portion
        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_portion = critic_state_dict[name].clone()
            target_portion = target_critic_state_dict[name].clone()
            critic_state_dict[name] = tau * critic_portion + (1 - tau) * target_portion
        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation, bounds, explore=True):
        noise = torch.Tensor(0)
        state = torch.Tensor(observation).to(self.device)
        actions = self.actor.forward(state)
        if explore:
            noise = torch.rand(1)-0.5
            noise = (noise * self.n_actions).to(self.device)
        action = actions + noise
        action = action.detach().cpu().numpy()[0].round()
        action = np.clip(action, bounds[0], bounds[1]-1)
        return int(action)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class MADDPG:
    def __init__(self, actor_dims:dict, critic_dims:int, agent_list, n_actions, device, action_ranges,
                 scenario_name='simple', lr_a=0.01, lr_b=0.01, fc_dims=64, gamma=0.99,
                 tau=0.01, batch_size=64, checkpoint_dir="./maddpg/"):
        self.agents = []
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.agent_names = agent_list
        self.n_agents = len(agent_list)
        self.n_actions = n_actions
        self.action_ranges = action_ranges
        checkpoint_dir += scenario_name
        self.batch_size = batch_size
        self.buffer = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions,
                                             agent_list, batch_size=batch_size)

        for agent_name in self.agent_names:
            self.agents.append(
                Agent(actor_dims[agent_name], critic_dims, n_actions, self.n_agents, agent_name,
                      checkpoint_dir, device, lr_a, lr_b, fc_dims, gamma, tau)
            )

    def save_checkpoint(self):
        print(f"saving checkpoint...")
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print(f"loading checkpoint...")
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, explore=True):
        actions = {}
        for agent in self.agents:
            name = agent.agent_name
            bounds = self.action_ranges[name]
            action = agent.choose_action(raw_obs[name], bounds, explore)
            actions[name] = action
        return actions

    def learn(self):
        if not self.buffer.ready():
            return
        actor_states, states, actions, rewards, actor_new_states, states_, dones = self.buffer.sample_buffer()
        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        states_ = torch.Tensor(states_).to(device)
        dones = torch.Tensor(dones).to(torch.int).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.Tensor(actor_new_states[agent_idx]).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            mu_states = torch.Tensor(actor_states[agent_idx]).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts.reshape(-1, 1) for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts.reshape(-1, 1) for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts.reshape(-1, 1) for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, agent_idx]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()


def obs_list_to_state_vector(observation):
    # state = np.array([])
    # for obs in observation.values():
    #     state = np.concatenate([state, obs])
    state = np.concatenate(list(observation.values()))
    return state


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("using", device)
    # show()
    scenario_name = "simple"
    # scenario_name = "simple_adversary"
    # env = simple_v3.parallel_env(render_mode="human")
    env = simple_v3.parallel_env(render_mode=None)
    # env = simple_adversary_v3.parallel_env(render_mode="human")
    PRINT_INTERVAL = 500
    N_GAMES = 30000
    MAX_STEPS = 25
    BATCH_SIZE = 1024
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    env.reset()
    n_agents = env.num_agents
    actor_dims = {}
    action_ranges = {}
    n_actions = None
    for agent in env.agents:
        dim = env.observation_space(agent).shape[0]
        min_val = env.action_space(agent).start
        n_actions = env.action_space(agent).n
        action_range = [min_val, min_val + n_actions]
        actor_dims[agent] = dim
        action_ranges[agent] = action_range
    critic_dims = sum(actor_dims.values())
    maddpg_agents = MADDPG(actor_dims, critic_dims, env.agents, n_actions, device, action_ranges,
                           scenario_name, lr_a=0.01, lr_b=0.01, fc_dims=64, gamma=0.99, tau=0.01,
                           batch_size=BATCH_SIZE)
    memory = maddpg_agents.buffer

    if evaluate:
        maddpg_agents.load_checkpoint()
    for i in range(N_GAMES):
        obs, infos = env.reset()
        score = 0
        episode_step = 0
        while env.agents:
            if evaluate:
                env.render()
            # env.render()
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, truncation, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            obs = obs_
            score += sum(reward.values())
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        maddpg_agents.learn()

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f"ep: {i} avg score: {avg_score:.3f}")
