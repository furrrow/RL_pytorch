import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
from tqdm import tqdm
from replay_buffer import NumpyReplayBuffer

""" A3C code implementation,
heavily referencing:
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py#L159
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

the code still has a hard time being trained properly, thus cannot rule out the possibility of a bug
however, A3C has known to be brittle to hyperparameters.
alternatively, I could use entirely different neural networks for both policy and value models (like miguel's notebook)

TODO: this gets stuck on neural network's forward function, unknown why
"""


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)


# miguel has an image based version, I need to start with a state based one...
class ActorCritic(nn.Module):

    def __init__(self, input_dim, n_actions, gamma=0.99, entropy_loss_weight=0.001, device="cpu"):
        super(ActorCritic, self).__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.logpas = []
        self.entropies = []
        self.gamma = gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.pi1 = nn.Linear(input_dim, 128)
        self.pi2 = nn.Linear(128, 64)
        self.pi3 = nn.Linear(64, n_actions)

        self.v1 = nn.Linear(input_dim, 256)
        self.v2 = nn.Linear(256, 128)
        self.v3 = nn.Linear(128, 1)
        self.device = torch.device(device)
        self.to(self.device)
        self.buffer = NumpyReplayBuffer()

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
        pi = F.relu(self.pi1(x))
        pi = F.relu(self.pi2(pi))
        pi = self.pi3(pi)

        v = F.relu(self.v1(x))
        v = F.relu(self.v2(v))
        v = self.v3(v)
        return pi, v

    def calculate_rewards(self, done, next_value):
        next_v = 0 if done else next_value.detach().item()
        R = next_v
        batch_return = []
        for reward in self.rewards[::-1]:  # [::-1] means reversed
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float, device=self.device)
        return batch_return

    def calculate_return_alternative(self, done, next_value):
        # miguel's implementation, same as calculate_rewards() above but cleaner?
        rewards = self.rewards
        next_v = 0 if done else next_value.detach().item()
        rewards.append(next_v)
        n = len(self.rewards)
        discounts = np.logspace(0, n, num=n, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:n - t] * rewards[t:]) for t in range(n)])
        return returns[:-1]

    def calculate_loss(self, done):
        pi, values = self.forward(self.states)

        returns = self.calculate_rewards(done, values[-1])
        values = values.squeeze()  # very important get shape (n, 1)
        value_error = returns - values
        policy_loss = -(self.logpas * value_error.detach()).mean()
        entropy_loss = -self.entropies.mean()
        policy_loss = policy_loss * self.entropy_loss_weight * entropy_loss
        value_loss = value_error.pow(2).mul(0.5).mean()
        total_loss = policy_loss + value_loss

        # actor_loss = -self.logpas * value_error
        # critic_loss = value_error**2
        # total_loss = (critic_loss.detach() + actor_loss).mean()
        return total_loss

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        logits, v = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_pa = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_pa, entropy


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, entropy_loss_weight, gamma, name,
                 global_episode_index, env_id, device, n_games, t_max):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma, entropy_loss_weight, device)
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer
        self.name = f"worker{name:2}"
        self.episode_index = global_episode_index
        self.env = gym.make(env_id)
        self.n_games = n_games
        self.t_max = t_max

    def run(self):
        t_step = 1
        while self.episode_index.value < self.n_games:
            print(self.name, ", step", t_step)
            terminated = False
            truncated = False
            observation, info = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memories()
            while not (terminated or truncated):
                action, log_pa, entropy = self.local_actor_critic.choose_action(observation)
                observation_new, reward, terminated, truncated, info = self.env.step(action)
                print(reward)
                score += reward
                self.local_actor_critic.remember(observation, action, reward, log_pa, entropy)
                if t_step % self.t_max == 0 or (terminated or truncated):
                    self.local_actor_critic.consolidate_memory()
                    loss = self.local_actor_critic.calculate_loss(terminated)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.local_actor_critic.parameters(), max_norm=0.5
                    )
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param.grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memories()
                t_step += 1
                observation = observation_new
            with self.episode_index.get_lock():
                self.episode_index.value += 1
            print(f"{self.name}, episode: {self.episode_index.value}, rewards: {score:.1f}")


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"  # GPU doesn't work yet
    print("using", device)
    LR = 0.0001
    entropy_loss_weight = 0.001
    env_id = "CartPole-v1"
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_action = env.action_space.n
    N_GAMES = 3000
    T_MAX = 5
    # n_workers = mp.cpu_count()-1
    n_workers = 2
    global_actor_critic = ActorCritic(n_states, n_action,
                                      entropy_loss_weight=entropy_loss_weight, device=device)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=LR, betas=(0.92, 0.999))
    global_ep = mp.Value("i", 0)

    workers = []
    for i in range(n_workers):
        workers.append(
            Agent(global_actor_critic, optim, n_states, n_action,
                  entropy_loss_weight=entropy_loss_weight, gamma=1.00, name=i,
                  global_episode_index=global_ep, env_id=env_id, device=device,
                  n_games=N_GAMES, t_max=T_MAX))

    [w.start() for w in workers]
    [w.join() for w in workers]
