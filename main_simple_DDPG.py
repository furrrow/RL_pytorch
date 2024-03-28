import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import plot_training_history
from replay_buffer import MultiAgentNumpyBuffer
from pettingzoo.mpe import simple_v3, simple_adversary_v3, simple_spread_v3
from gymnasium.utils.save_video import save_video

""" DDPG code implementation,
Using the MPE environment, originally from openAI but using the pettingzoo version
- testing single agent version DDPG on the simple env

"""


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
        x = torch.cat((state, action), dim=-1)
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
            action = self.forward(state).detach().data.numpy().squeeze()
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
        self.online_policy_model = ActorNetwork(state_size, action_size, hidden_dim, bounds, device)
        self.online_value_model = CriticNetwork(total_state_size, total_action_size, hidden_dim, device)
        self.target_policy_model = ActorNetwork(state_size, action_size, hidden_dim, bounds, device)
        self.target_value_model = CriticNetwork(total_state_size, total_action_size, hidden_dim, device)
        self.actor_optimizer = torch.optim.Adam(self.online_policy_model.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.online_value_model.parameters(), lr=lr_c)
        self.device = device

    def update_networks(self):
        for target, online in zip(self.target_value_model.parameters(),
                                  self.online_value_model.parameters()):
            mixed_weights = (1.0 - self.tau) * target.data + self.tau * online.data
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(),
                                  self.online_policy_model.parameters()):
            mixed_weights = (1.0 - self.tau) * target.data + self.tau * online.data
            target.data.copy_(mixed_weights)


class MADDPG:
    def __init__(self, env, device, lr_a=0.01, lr_b=0.01, fc_dims=64, gamma=0.99,
                 tau=0.01, batch_size=64):
        self.env = env
        self.env.reset()
        self.agent_names = self.env.agents
        self.n_agents = len(self.agent_names)
        self.batch_size = batch_size
        self.obs_dims = {}
        self.action_ranges = {}
        self.n_actions = {}
        for agent in self.env.agents:
            dim = self.env.observation_space(agent).shape[0]
            self.obs_dims[agent] = dim
            self.n_actions[agent] = self.env.action_space(agent).shape[0]
            action_range = [self.env.action_space(agent).low[0], self.env.action_space(agent).high[0]]
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
        self.rewards_history = []
        self.running_timestep = 0
        self.running_reward = 0
        self.update_interval = 5
        self.print_interval = 5
        self.video_interval = 20
        self.device = device

    def choose_actions(self, raw_obs, explore=True):
        actions = {}
        for agent in self.QAgents:
            name = agent.agent_name
            action = agent.online_policy_model.choose_action(raw_obs[name], explore)
            actions[name] = action
        return actions

    def video_schedule(self, episode_id: int) -> bool:
        return episode_id % self.video_interval == 0

    def train(self, n_epochs):
        self.populate_buffer(5)
        for episode in range(n_epochs):
            self.running_reward = np.zeros(self.n_agents)
            self.running_timestep = 0
            video_frames = []
            state, infos = self.env.reset()
            while self.env.agents:
                state, terminal, truncated, video_frame = self.interaction_step(state, self.env, explore=False, return_frame=True)
                video_frames.append(video_frame)
            self.optimize_model()  # TODO: when to optimize for ddpg?
            self.rewards_history.append(self.running_reward)
            if episode % self.update_interval == 0:
                [agent.update_networks() for agent in self.QAgents]
            if episode % self.print_interval == 0:
                print(f"ep: {episode}, t: {self.running_timestep}, reward: {self.running_reward[-1]:.3f}, "
                      f"running rwd {np.average(np.array(self.rewards_history)[:,-1][-20:]):.3f}")
            # save video comes with its own "capped_cubic_video_schedule"
            save_video(video_frames, f"videos/{self.env.scenario_name}",
                       fps=30, episode_index=episode)
            # save_video(video_frames, f"videos/{self.env.scenario_name}", episode_trigger=self.video_schedule,
            #            fps=30, episode_index=episode)

    def populate_buffer(self, n_batches=1):
        min_samples = self.batch_size * n_batches
        while len(self.buffer) < self.batch_size:
            state, info = self.env.reset()
            while self.env.agents:
                state, terminal, truncated, video_frame = self.interaction_step(state, self.env, explore=True,
                                                                                return_frame=False)
        print(f"{len(self.buffer)} samples populated to buffer")

    def interaction_step(self, state, myenv, explore, return_frame):
        frame = myenv.render() if return_frame else None
        actions = maddpg_agents.choose_actions(state, explore=explore)
        new_state, reward, done, truncated, info = myenv.step(actions)
        # print(state['agent_0'], actions['agent_0'], new_state['agent_0'])
        experience = (state, actions, reward, new_state, done)
        self.buffer.store(experience)
        self.running_reward = list(reward.values())
        self.running_timestep += 1
        return new_state, done, truncated, frame

    def optimize_model(self):
        sample = self.buffer.sample()
        all_states_np = np.concatenate(sample[0], axis=1)
        all_actions_np = np.concatenate(sample[1], axis=1)
        all_next_states_np = np.concatenate(sample[3], axis=1)
        all_next_states = torch.from_numpy(all_next_states_np).to(self.device)
        all_states = torch.from_numpy(all_states_np).float().to(self.device)
        all_actions = torch.from_numpy(all_actions_np).float().to(self.device)

        for agent_idx, agent in enumerate(self.QAgents):
            agent_state, agent_action, agent_reward, agent_next_state, agent_dones = \
                self.get_agent_experiences(sample, agent_idx)

            # value updates
            argmax_a_q_sp = agent.target_policy_model(all_next_states)
            # next_state_policies = torch.cat([acts for acts in all_actions], dim=1)  # [5, 15]
            max_a_q_sp = agent.target_value_model(all_next_states, argmax_a_q_sp.detach())
            target_q_sa = agent_reward + agent.gamma * max_a_q_sp.detach() * (1 - agent_dones)

            q_sa = agent.online_value_model(all_states, all_actions)
            td_error = q_sa - target_q_sa.detach_()
            # masked_td_error = td_error * (1 - agent_dones)
            value_loss = td_error.pow(2).mul(0.5).mean()
            # value_loss = F.mse_loss(q_sa, target_q_sa.detach_())
            # print(agent_idx, "value_loss backward pass:")
            agent.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online_value_model.parameters(), 0.5)
            agent.critic_optimizer.step()

            # policy updates:
            argmax_a_q_s = agent.online_policy_model(agent_state)
            max_a_q_s = agent.online_value_model(all_states, argmax_a_q_s)
            policy_loss = -max_a_q_s.mean()
            # print(agent_idx, "policy_loss backward pass:")
            agent.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online_policy_model.parameters(), 0.5)
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

    def show(self, n_epochs, show_env):
        for episode in range(n_epochs):
            self.running_reward = np.zeros(self.n_agents)
            self.running_timestep = 0
            state, infos = show_env.reset()
            while show_env.agents:
                state, terminal, truncated, frame = self.interaction_step(state, show_env, explore=False, return_frame=False)
                show_env.render()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    N_GAMES = 500
    BATCH_SIZE = 512
    MAX_CYCLE = 75
    scenario_name = "simple"
    env = simple_v3.parallel_env(max_cycles=MAX_CYCLE, render_mode="rgb_array", continuous_actions=True)
    show_env = simple_v3.parallel_env(max_cycles=MAX_CYCLE, render_mode="human", continuous_actions=True)
    env.scenario_name = scenario_name
    maddpg_agents = MADDPG(env, device, lr_a=0.0005, lr_b=0.0005, fc_dims=256, gamma=0.99, tau=0.01,
                           batch_size=BATCH_SIZE)
    maddpg_agents.train(N_GAMES)
    label_list = ['rewards', '20 episode rolling avg']
    plot_training_history(maddpg_agents.rewards_history, save=True, labels=label_list, filename=scenario_name)
    maddpg_agents.show(2, show_env)
