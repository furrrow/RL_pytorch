import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import plot_training_history
from replay_buffer import MultiAgentNumpyBuffer
from pettingzoo.mpe import simple_v3, simple_adversary_v3, simple_spread_v3
from gymnasium.utils.save_video import save_video

""" DQN code implementation as IQL (independent q-learning)
Using the MPE environment, originally from openAI but using the pettingzoo version

"""


class QNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, device):
        super(QNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class Agent:
    def __init__(self, state_size, total_state_size, action_size, total_action_size, bounds, n_agents, agent_name,
                 device,
                 lr=0.01, hidden_dim=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.total_state_size = total_state_size
        self.action_size = action_size
        self.total_action_size = total_action_size
        self.n_agents = n_agents
        self.bounds = bounds
        self.agent_name = agent_name
        self.lr = lr
        self.online_model = QNetwork(state_size, action_size, hidden_dim, device)
        self.target_model = QNetwork(state_size, action_size, hidden_dim, device)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=lr)
        self.device = device

    def update_networks(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            mixed_weights = (1.0 - self.tau) * target.data + self.tau * online.data
            target.data.copy_(mixed_weights)


class IQL:
    def __init__(self, env, device, lr=0.01, fc_dims=64, gamma=0.99,
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
            self.n_actions[agent] = self.env.action_spaces[agent].n  # for discrete action space
            action_range = [0, self.n_actions[agent]]
            # self.n_actions[agent] = self.env.action_space(agent).shape[0]  # for cont action space
            # action_range = [self.env.action_space(agent).low[0], self.env.action_space(agent).high[0]]
            self.action_ranges[agent] = action_range

        self.total_states = sum(self.obs_dims.values())
        self.total_actions = sum(self.n_actions.values())

        self.QAgents = []
        for agent_name in self.agent_names:
            self.QAgents.append(
                Agent(self.obs_dims[agent_name], self.total_states, self.n_actions[agent_name], self.total_actions,
                      self.action_ranges[agent_name], self.n_agents, agent_name, device, lr, fc_dims, gamma,
                      tau)
            )
        self.buffer = MultiAgentNumpyBuffer(100000, self.agent_names, batch_size=batch_size)
        self.rewards_history = []
        self.running_timestep = 0
        self.running_reward = 0
        self.update_interval = 5
        self.print_interval = 5
        self.video_interval = 20
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.device = device

    def choose_actions(self, raw_obs, epsilon=1):
        actions = {}
        explore = False if np.random.rand() > epsilon else True
        for agent in self.QAgents:
            name = agent.agent_name
            if explore:
                action = self.env.action_spaces[name].sample()
            else:
                state = torch.Tensor(raw_obs[name])
                q_values = agent.online_model(state).cpu().detach().data.numpy().squeeze()
                action = np.argmax(q_values)
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
                state, terminal, truncated, video_frame = self.interaction_step(state, self.env,
                                                                                return_frame=True)
                video_frames.append(video_frame)
            self.optimize_model()
            self.rewards_history.append(self.running_reward)
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(0.005, self.epsilon)
            if episode % self.update_interval == 0:
                [agent.update_networks() for agent in self.QAgents]
            if episode % self.print_interval == 0:
                print(f"ep: {episode}, t: {self.running_timestep}, "
                      f", epsilon: {self.epsilon:.3f}, reward: {self.running_reward[-1]:.3f}, "
                      f"running rwd {np.average(np.array(self.rewards_history)[:, -1][-100:]):.3f}")
            # save video comes with its own "capped_cubic_video_schedule"
            save_video(video_frames, f"videos/{self.env.scenario_name}",
                       # episode_trigger=self.video_schedule,  # comment line for schedule
                       fps=30, episode_index=episode)

    def populate_buffer(self, n_batches=1):
        min_samples = self.batch_size * n_batches
        while len(self.buffer) < self.batch_size:
            state, info = self.env.reset()
            while self.env.agents:
                state, terminal, truncated, video_frame = self.interaction_step(state, self.env,
                                                                                return_frame=False)
        print(f"{len(self.buffer)} samples populated to buffer")

    def interaction_step(self, state, myenv, return_frame):
        frame = myenv.render() if return_frame else None
        actions = self.choose_actions(state, self.epsilon)
        new_state, reward, done, truncated, info = myenv.step(actions)
        # print(state['agent_0'], actions['agent_0'], new_state['agent_0'])
        experience = (state, actions, reward, new_state, done)
        self.buffer.store(experience)
        self.running_reward = list(reward.values())
        self.running_timestep += 1
        return new_state, done, truncated, frame

    def optimize_model(self):
        sample = self.buffer.sample()

        for agent_idx, agent in enumerate(self.QAgents):
            agent_state, agent_action, agent_reward, agent_next_state, agent_dones = \
                self.get_agent_experiences(sample, agent_idx)

            # value updates
            argmax_a_q_sp = agent.online_model(agent_next_state).max(1)[1]  # [batch_size]
            q_sp = agent.target_model(agent_next_state).detach()  # [batch_size, n_action]
            max_a_q_sp = q_sp[np.arange(self.batch_size), argmax_a_q_sp].unsqueeze(-1)  # [batch_size, 1]
            target_q_sa = agent_reward + agent.gamma * max_a_q_sp * (1 - agent_dones)  # [batch_size, 1]
            q_sa = agent.online_model(agent_state).gather(1, agent_action.to(torch.int64))  # [batch_size, 1]

            td_error = q_sa - target_q_sa
            # masked_td_error = td_error * (1 - agent_dones)
            value_loss = td_error.pow(2).mul(0.5).mean()
            # value_loss = F.mse_loss(q_sa, target_q_sa.detach_())
            # print(agent_idx, "value_loss backward pass:")
            agent.optimizer.zero_grad()
            value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.online_model.parameters(), 0.5)
            agent.optimizer.step()

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
                state, terminal, truncated, frame = self.interaction_step(state, show_env,
                                                                          return_frame=False)
                show_env.render()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    N_GAMES = 1000
    BATCH_SIZE = 512
    MAX_CYCLE = 75
    scenario_name = "simple"
    env = simple_v3.parallel_env(max_cycles=MAX_CYCLE, render_mode="rgb_array", continuous_actions=False)
    env.scenario_name = scenario_name
    agents = IQL(env, device, lr=0.001, fc_dims=256, gamma=0.99, tau=0.005,
                 batch_size=BATCH_SIZE)
    agents.train(N_GAMES)
    plot_training_history(agents.rewards_history, save=False)
    agents.show(5, env)
