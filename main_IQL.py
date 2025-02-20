import copy

import minari
import os
from huggingface_hub import login
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import MultivariateNormal
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
import matplotlib.pyplot as plt
import numpy as np
"""
Implicit Q-learning implementation,
borrowing heavily from:
https://github.com/gwthomas/IQL-PyTorch/tree/main
"""

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, lr, max_steps,
                 tau, beta, device, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(device)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(device)
        self.vf = vf.to(device)
        self.policy = policy.to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.device = device
        self.q_loss_hist = []
        self.v_loss_hist = []
        self.policy_loss_hist = []

    def update(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_loss_hist.append(v_loss.item())
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_loss_hist.append(q_loss.item())
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100) # EXP_ADV_MAX
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.policy_loss_hist.append(policy_loss.item())
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        LOG_STD_MIN = -5.0
        LOG_STD_MAX = 2.0
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

if __name__ == '__main__':
    # access_token = os.environ["HUGGINGFACE_READ_TOKEN"]
    # login(token = access_token ,add_to_git_credential=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using", device)
    BATCH_SIZE = 256
    LR = 0.0003
    tau = 0.005
    alpha = 0.005
    beta = 3.0
    # max_steps = 10**6
    max_steps = 50000
    gamma = 0.99
    hidden_dim = 256
    n_hidden = 2
    eval_period = 10
    max_eval_steps = 1000
    # dataset_id = "mujoco/hopper/simple-v0"
    # dataset_id = "mujoco/hopper/medium-v0"
    dataset_id = "mujoco/hopper/expert-v0"
    dataset = minari.load_dataset(dataset_id, download=True)
    # print("Observation space:", dataset.observation_space)
    # print("Action space:", dataset.action_space)
    # print("Total episodes:", dataset.total_episodes)
    # print("Total steps:", dataset.total_steps)
    env = dataset.recover_environment()

    n_states = dataset.observation_space.shape[0]
    n_action = dataset.action_space.shape[0]
    replay_buffer = MinariExperienceReplay(
        dataset_id,
        split_trajs=False,
        batch_size=BATCH_SIZE,
        sampler=SamplerWithoutReplacement(),
        # transform=DoubleToFloat(),
    )
    update_interval = 1
    EPOCHS = 100
    policy = DeterministicPolicy(n_states, n_action, hidden_dim=hidden_dim, n_hidden=n_hidden)
    # policy = GaussianPolicy(n_states, n_action, hidden_dim=hidden_dim, n_hidden=n_hidden)
    iql = ImplicitQLearning(
        qf=TwinQ(n_states, n_action, hidden_dim=hidden_dim, n_hidden=n_hidden),
        vf=ValueFunction(n_states, hidden_dim=hidden_dim, n_hidden=n_hidden),
        policy=policy,
        lr=LR,
        max_steps=max_steps,
        tau=tau,
        beta=beta,
        device=device,
        alpha=alpha,
        discount=gamma,
    )
    eval_history = []
    for step in range(max_steps):
        # tensordict keys: ['episode', 'action', 'next', 'observation', 'index']
        data = replay_buffer.sample()
        terminals = data['next']['done'] # terminal or truncated?
        data_input = (data['observation'], data['action'], data['next']['observation'], data['next']['reward'], terminals)
        iql.update(data_input[0].to(torch.float32), data_input[1].to(torch.float32), data_input[2].to(torch.float32),
                   data_input[3].to(torch.float32), data_input[4].to(torch.float32))

        if (step + 1) % eval_period == 0:
            # evaluate_policy:
            obs = env.reset()
            obs = obs[0]
            total_reward = 0.
            for _ in range(max_eval_steps):
                with torch.no_grad():
                    obs = torch.from_numpy(obs).to(torch.float32)
                    action = policy.act(obs, deterministic=True).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
                else:
                    obs = next_obs
            # min_score, max_score = 0, 190
            # normalized_returns = (total_reward - min_score) / (max_score - min_score) * 100.0
            eval_history.append([step, total_reward])
            print(f"step {step} total_reward: {total_reward:.3f}")

    x = np.arange(len(iql.v_loss_hist))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, iql.v_loss_hist, label='v_loss')
    ax.plot(x, iql.q_loss_hist, label='q_loss')
    ax.plot(x, iql.policy_loss_hist, label='policy_loss')
    ax.set_xlabel('steps')
    ax.set_title('loss_history')
    ax.legend()

    fig2, ax2 = plt.subplots(1, 1)
    eval_history = np.array(eval_history)
    ax2.plot(eval_history[:, 0], eval_history[:, 1], label='eval rewards')
    ax2.set_xlabel('steps')
    ax2.set_title('rewards during evaluation')
    plt.show()

