import torch
import torch.optim as optim
import gym
import numpy as np

import utils
from models.SimpleModel import SimpleModel
from models.CNNModel import CNNModel
from policy.EGreedyStrategy import EGreedyStrategy
from policy.EGreedyExpStrategy import EGreedyExpStrategy
from RandomStrategy import RandomStrategy


class DQNAgent:
    def __init__(self,
                 buffer,
                 env_name,
                 model_name="simple",
                 n_episodes=1500,
                 batch_size=1024,
                 learning_rate=0.0005,
                 epsilon=0.0005,
                 update_interval=10,
                 gamma=0.9995,
                 optimizer="adam",
                 modify_env=False,
                 render=True):
        self.gamma = gamma
        self.buffer = buffer
        self.env_name = env_name
        self.modify_env = modify_env
        if modify_env:
            self.env = utils.mod_env(env_name)
        else:
            self.env = gym.make(env_name)
        self.env.reset()
        self.n_states = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.model_name = model_name
        if model_name == "simple":
            self.online_model = SimpleModel(self.n_states, self.n_action)
            self.target_model = SimpleModel(self.n_states, self.n_action)
            self.online_model.print_model(input_size=(batch_size, self.n_states))
        elif model_name == "cnn":
            (c, h, w) = self.env.observation_space.shape
            self.input_shape = (c, h, w)
            self.online_model = CNNModel(self.input_shape, self.n_action)
            self.target_model = CNNModel(self.input_shape, self.n_action)
            full_shape = (batch_size, c, h, w)
            self.online_model.print_model(input_size=full_shape)
        self.reward_record = []
        self.rolling_average = []
        self.loss_record = []
        self.epoch_loss = []
        self.best_score = -20
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.lr = learning_rate
        self.update_interval = update_interval
        self.solved = False
        self.render = render

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.online_model.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.online_model.parameters(), lr=learning_rate)

    def optimize_miguel(self):
        transitions = self.buffer.sample(self.batch_size)
        (states, actions, next_states, rewards, is_terminals) = self.online_model.load(transitions)
        # states, actions, rewards, next_states, is_terminals = experiences
        # batch_size = len(is_terminals)

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0]  # (batch_size)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))  # (batch_size)
        target_q_sa = target_q_sa.unsqueeze(1)  # (batch_size x 1)
        q_sa = self.online_model(states).gather(1, actions.unsqueeze(1))  # (batch_size x 1)

        td_error = q_sa - target_q_sa
        # value_loss = td_error.pow(2).mul(0.5).mean()
        criterion = torch.nn.SmoothL1Loss()  # MSELoss()
        value_loss = criterion(q_sa, target_q_sa)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        self.epoch_loss.append(value_loss.data.cpu().numpy().copy())

    def optimize_jim(self):
        transitions = self.buffer.sample(self.batch_size)
        (states, actions, new_states, rewards, is_terminals) = self.online_model.load(transitions)
        continue_mask = 1 - is_terminals  # (batch_size)
        q_next = self.target_model(new_states).detach()  # gradient does NOT involve the target
        q_next_max = q_next.max(1)[0]  # (batch_size)
        q_target = rewards + q_next_max * continue_mask * self.gamma  # (batch_size)
        q_target = q_target.unsqueeze(1)  # (batch_size x 1)
        q_values = self.online_model(states).gather(1, actions.unsqueeze(1))  # (batch_size x 1)
        criterion = torch.nn.MSELoss()
        loss = criterion(q_values, q_target)
        self.epoch_loss.append(loss.data.cpu().numpy().copy())
        # may want to return this loss if it has no access to self.epoch_loss
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def populate_buffer(self):
        initial_size = int(self.buffer.capacity * 0.1)

        while self.buffer.length() < initial_size:
            state = self.env.reset()
            terminal = False
            tmp_reward = 0

            while not terminal:
                action = self.env.action_space.sample()
                (next_state, reward, done, info) = self.env.step(action)
                if ("TimeLimit" in info):
                    print(info)
                    print("terminating episode...")
                    done = False
                    terminal = True
                self.buffer.save(state, action, next_state, reward, done)
                terminal = done
                state = next_state
                tmp_reward += reward
        print(self.buffer.length(), "entries saved to ReplayBuffer")

    def train(self, policy_name="egreedy"):
        total_count = 0
        if policy_name == "egreedy":
            policy = EGreedyStrategy(epsilon=self.epsilon)
        elif policy_name == "egreedyexp":
            policy = EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000)
        elif policy_name == "random":
            policy = RandomStrategy()
        else:
            print("policy not yet implemented")
        for episode in range(self.n_episodes):
            count = 0
            state = self.env.reset()
            terminal = False
            tmp_reward = 0
            self.epoch_loss = []
            while not terminal:
                # render if needed
                # if self.render:
                # self.env.render()
                # select action
                action = policy.select_action(self.online_model, state)
                (next_state, reward, done, info) = self.env.step(action)
                # handle cases when it reaches a time limit but not actually terminal
                if "TimeLimit" in info:
                    print(info)
                    print("terminating episode...")
                    done = False
                    terminal = True
                self.buffer.save(state, action, next_state, reward, done)
                terminal = done
                state = next_state
                tmp_reward += reward
                count += 1

                # update target network periodically
                if (count + total_count) % self.update_interval == 0:
                    self.update_target_network()
                    # print("updated target network!")
                self.optimize_jim()
            self.render = False  # reset render flag
            total_count += count
            self.reward_record.append(tmp_reward)
            self.rolling_average.append(np.average(self.reward_record[-100:]))
            avg_loss = np.average(self.epoch_loss)
            self.loss_record.append(avg_loss)
            if episode % 5 == 0:

                print(f"episode {episode:2d} loss {avg_loss:3.3f} epsilon {self.epsilon:3.3f} reward {tmp_reward}")
            if episode % 10 == 0:
                self.render = True  # render the next episode

            if not self.solved:
                if self.rolling_average[-1] > 200:
                    print("!!! exceeded benchmark at epoch", episode)
                    print("!!! exceeded benchmark, last 100 episode avg reward:", round(self.rolling_average[-1], 3))
                    self.solved = True
                    self.n_episodes = episode + 10  # terminate in 10 episodes
            if tmp_reward > self.best_score:
                self.best_score = tmp_reward
                print("episode", episode, "new best score", round(self.best_score, 3), "rolling avg",
                      round(self.rolling_average[-1], 3))

        return self.reward_record, self.rolling_average, self.loss_record