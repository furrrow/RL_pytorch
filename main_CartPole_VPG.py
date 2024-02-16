import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from utils import plot_training_history
from replay_buffer import NumpyReplayBuffer
from tqdm import tqdm

""" VPG code implementation,
Based on Miguels Vanilla Policy Gradient (aka REINFORCE with baseline)

- Ues action-advantage function
- Uses entropy to encourage more even distribution and encourage exploration

"""


class SimpleModel(nn.Module):

    def __init__(self, in_features, outputs):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(in_features, 32)
        self.linear2 = nn.Linear(32, 32)
        self.final = nn.Linear(32, outputs)
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

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.final(x)
        return x


class VPG:
    def __init__(self, env_id, batch_size, update_interval, tau, gamma):
        super(VPG, self).__init__()
        self.policy_model = SimpleModel(n_states, n_action)
        self.value_model = SimpleModel(n_states, 1)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=LR)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=LR)
        self.env_id = env_id
        self.env = gym.make(env_id, render_mode="rgb_array_list")
        self.buffer = NumpyReplayBuffer(100000, batch_size)
        self.update_interval = update_interval
        self.tau = tau
        self.gamma = gamma

        self.running_timestep = 0
        self.running_reward = 0
        self.reward_record = []
        self.episode_log_p_a = []
        self.episode_entropy = []
        self.episode_values = []
        self.episode_rewards = []
        self.print_interval = 5

    def optimize_model(self):
        # calculate loss
        T = len(self.episode_rewards)
        discount_vector = np.logspace(0, T, num=T, base=gamma, endpoint=False)
        returns = []
        for t in range(T):
            returns.append(np.sum(discount_vector[:(T - t)] * self.episode_rewards[t:]))
        episode_log_p_a = torch.cat(self.episode_log_p_a)  # [18]
        values = torch.cat(self.episode_values).to(device)  # [18]
        returns = torch.tensor(returns).to(device)  # [18]
        entropies = torch.tensor(self.episode_entropy).to(device)  # [18]
        discount_vector = torch.tensor(discount_vector).to(device)  # [18]

        # optimize
        value_error = returns - values
        # multiply by discount optional
        policy_loss = -(episode_log_p_a * value_error.detach() * discount_vector).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        self.reward_record.append(np.sum(self.episode_rewards))

    def interaction_step(self, state):
        # altering a little because cart-pole has discrete action space
        state = torch.Tensor(state)
        logits = self.policy_model.forward(state)
        value = self.value_model.forward(state)
        policy = torch.distributions.Categorical(logits=logits)
        action_tensor = policy.sample()
        log_prob_action = policy.log_prob(action_tensor)
        self.episode_log_p_a.append(log_prob_action.unsqueeze(-1))
        self.episode_entropy.append(policy.entropy())
        self.episode_values.append(value)
        new_state, reward, terminated, truncated, info = self.env.step(action_tensor.item())

        self.episode_rewards.append(reward)
        self.running_reward += reward
        self.running_timestep += 1
        return new_state, terminated, truncated

    def train(self, N_GAMES):
        for episode in range(N_GAMES):
            self.running_reward = 0
            self.running_timestep = 0
            state, info = self.env.reset()
            terminal, truncated = False, False
            self.episode_log_p_a = []
            self.episode_entropy = []
            self.episode_values = []
            self.episode_rewards = []
            while not (terminal or truncated):
                state, terminal, truncated = self.interaction_step(state)

            self.optimize_model()
            if episode % self.print_interval == 0:
                print(f"episode: {episode}, timesteps: {self.running_timestep}, "
                      f"avg running rewards: {np.average(self.reward_record[-100:]):.3f}")
            # save video comes with its own "capped_cubic_video_schedule"
            save_video(self.env.render(), f"videos/{self.env_id}", fps=30, episode_index=episode)

        print("training ended")
        self.env.close()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    BATCH_SIZE = 256
    LR = 0.001
    tau = 0.005
    gamma = 1.0
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.n
    env.close()
    update_interval = 5
    EPOCHS = 500
    agent = VPG(env_id, BATCH_SIZE, update_interval, tau, gamma)
    agent.train(EPOCHS)
    plot_training_history(agent.reward_record, save=False)


