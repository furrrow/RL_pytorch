import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from rl_optimizer import optimize_jim, optimize_miguel
from models.SimpleModel import SimpleModel
from models.CNNModel import CNNModel
from policy.EGreedyStrategy import EGreedyStrategy
from policy.EGreedyExpStrategy import EGreedyExpStrategy
from policy.LinearDecayStrategy import LinearDecayStrategy
from policy.RandomStrategy import RandomStrategy


class DQNAgent:
    def __init__(self,
                 config,
                 buffer,
                 env):
        self.gamma = config['gamma']
        self.buffer = buffer
        self.env = env
        self.env.reset()
        self.n_states = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.model_name = config['model_name']
        self.batch_size = config['batch_size']
        if self.model_name == "simple":
            self.online_model = SimpleModel(self.n_states, self.n_action)
            self.target_model = SimpleModel(self.n_states, self.n_action)
            self.online_model.print_model(input_size=(self.batch_size, self.n_states))
        elif self.model_name == "cnn":
            (c, h, w) = self.env.observation_space.shape
            self.input_shape = (c, h, w)
            self.online_model = CNNModel(self.input_shape, self.n_action)
            self.target_model = CNNModel(self.input_shape, self.n_action)
            full_shape = (self.batch_size, c, h, w)
            self.online_model.print_model(input_size=full_shape)
        self.reward_record = []
        self.rolling_average = []
        self.loss_record = []
        self.epoch_loss = []
        self.best_score = -20
        self.n_episodes = config['episodes']

        self.epsilon = config['epsilon']
        self.lr = config['learning_rate']
        self.update_interval = config['update_interval']
        self.solved = False
        self.optimizer = config['torch_optimizer']

        if config['loss_criterion'] == "MSE":
            self.criterion = torch.nn.MSELoss()
        elif config['loss_criterion'] == "SmoothL1":
            self.criterion = torch.nn.SmoothL1Loss()

        if config['rl_optimizer'] == "jim":
            self.rl_optimizer = optimize_jim
        elif config['rl_optimizer'] == "miguel":
            self.rl_optimizer = optimize_miguel

        if self.optimizer == "adam":
            self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.online_model.parameters(), lr=self.lr)

    def update_target_network(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def populate_buffer(self, factor=0.1):
        initial_size = int(self.buffer.capacity * factor)

        while self.buffer.length() < initial_size:
            state, info = self.env.reset()
            terminated, truncated = False, False
            tmp_reward = 0

            while not (terminated or truncated):
                action = self.env.action_space.sample()
                (next_state, reward, terminated, truncated, info) = self.env.step(action)
                self.buffer.save(state, action, next_state, reward, terminated)
                state = next_state
                tmp_reward += reward
        print(self.buffer.length(), "entries saved to ReplayBuffer")

    def train(self, policy_name="egreedy"):
        total_count = 0
        if policy_name == "egreedy":
            self.policy = EGreedyStrategy(epsilon=self.epsilon)
        elif policy_name == "egreedyexp":
            self.policy = EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000)
        elif policy_name == "linear":
            self.policy = LinearDecayStrategy(init_epsilon=1.0, min_epsilon=0.1, plateu_step=750)
        elif policy_name == "random":
            self.policy = RandomStrategy()
        else:
            print("policy not yet implemented")
        for episode in tqdm(range(self.n_episodes)):
            count = 0
            state, info = self.env.reset()
            terminated, truncated = False, False
            tmp_reward = 0
            self.epoch_loss = []
            while not (terminated or truncated):
                # select action
                action = self.policy.select_action(self.online_model, state)
                (next_state, reward, terminated, truncated, info) = self.env.step(action)
                # handle cases when it reaches a time limit but not actually terminal
                self.buffer.save(state, action, next_state, reward, terminated)
                state = next_state
                tmp_reward += reward
                count += 1

                # update target network periodically, linearly increase update interval
                if (count + total_count) % self.update_interval == 0:
                    self.update_target_network()
                    # print("updated target network!")
                    self.rl_optimizer(self.buffer, self.criterion, self.batch_size, self.online_model, self.target_model,
                                      self.gamma, self.optimizer, self.epoch_loss)
            total_count += count
            self.reward_record.append(tmp_reward)
            self.rolling_average.append(np.average(self.reward_record[-100:]))
            avg_loss = np.average(self.epoch_loss)
            # print(len(self.epoch_loss), self.epoch_loss)
            if policy_name == "linear":  # update epsilon at every epoch instead of step
                self.policy._epsilon_update()
            self.loss_record.append(avg_loss)
            if episode % 5 == 0:
                tqdm.write(f"episode {episode:2d} loss {avg_loss:3.3f} epsilon {self.epsilon:3.3f} reward {tmp_reward}")
            if tmp_reward > self.best_score:
                self.best_score = tmp_reward
                tqdm.write(f"episode {episode} new best score {round(self.best_score, 3)} "
                           f"rolling avg: {round(self.rolling_average[-1], 3)}")
            if not self.solved:
                if self.rolling_average[-1] > 200:
                    tqdm.write(f"!!! exceeded benchmark at epoch {episode} 100 episode avg reward: "
                               f"{round(self.rolling_average[-1], 3)}, press Q to exit")

                    self.solved = True
                    break  # go ahead and exit
        return self.reward_record, self.rolling_average, self.loss_record
