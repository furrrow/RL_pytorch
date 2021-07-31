import matplotlib.pyplot as plt
import gym
import numpy as np
from gym.envs import toy_text
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from SimpleModel import SimpleModel
from CNNModel import CNNModel
from EGreedyStrategy import EGreedyStrategy
from EGreedyExpStrategy import EGreedyExpStrategy
from RandomStrategy import RandomStrategy
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer

if __name__ == "__main__":
    #episodes = 1500
    #gamma = 0.995
    #learning_rate = 0.001

    #env = gym.make("Pong-v0")
    #model = SimpleModel(env.observation_space.shape[0], env.action_space.n)
    #criterion = nn.MSELoss()
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    #i = 0
    #converged = False
    #policy = RandomStrategy()
    #while not converged and i < episodes:
        #if i % 100 == 0:
        #    print('Episode ', i+1)
    #    state = torch.from_numpy(env.reset()).float()
    #    action = policy.select_action(model, state)
    #    done = False
    #    iter = 0
    #    while not done:
    #        env.render()
    #        optimizer.zero_grad()
    #        state_prime, reward, done, info = env.step(int(action))
    #        state_prime = torch.from_numpy(state_prime).float()
    #        action_prime = policy.select_action(model, state_prime)
    #        q_sa = model(state.view(-1, 210))
    #        target_sa = torch.clone(target_sa)
    #        if iter < 999:
    #            temp = reward + (gamma * float(model(state_prime.view(-1, 210))[0, action_prime]) * (1 - done))
    #        else:
    #            temp = reward + (gamma * float(model(state_prime.view(-1, 210))[0, action_prime]))
    #        q_sa[0, action] = temp
    #        loss = criterion(target_sa, q_sa)
    #        loss.backward()
    #        optimizer.step()
    #        state = state_prime
    #        action = action_prime
    #        iter += 1
    #    i += 1

    replay_buffer = ReplayBuffer(capacity=25000)
    agent = DQNAgent(replay_buffer,
                     env_name="Pong-v0",
                     model_name="cnn",
                     n_episodes=1000,
                     epsilon=1.0,
                     batch_size=32,
                     learning_rate=0.0005,
                     update_interval=500,
                     gamma=0.995,
                     optimizer="rmsprop",
                     modify_env=True)

    agent.populate_buffer()
    record, rolling_avg, loss_record = agent.train(policy_name="random")
    x = np.arange(len(record))
    print(record)

    fig, rewards_plot = plt.subplots(figsize=(6, 6))
    rewards_plot.plot(x, record, label="rewards")
    rewards_plot.plot(x, rolling_avg, label="rolling_avg")
    rewards_plot.plot(x, loss_record, label="loss")
    rewards_plot.legend()
    plt.savefig('Pong_DuelDDQN.png')
    plt.show()