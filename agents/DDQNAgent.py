import torch
import torch.optim as optim
import numpy as np

from rl_optimizer import optimize_jim
from models.SimpleModel import SimpleModel
from models.CNNModel import CNNModel
from policy.EGreedyStrategy import EGreedyStrategy
from policy.EGreedyExpStrategy import EGreedyExpStrategy
from policy.LinearDecayStrategy import LinearDecayStrategy
from policy.RandomStrategy import RandomStrategy


class DDQNAgent:
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
        self.render = config['redner']
        self.optimizer = config['torch_optimizer']

        if config['loss_criterion'] == "MSE":
            self.criterion = torch.nn.MSELoss()
        elif config['loss_criterion'] == "SmoothL1":
            self.criterion = torch.nn.SmoothL1Loss()

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
            state = self.env.reset()
            terminal = False
            tmp_reward = 0

            while not terminal:
                action = self.env.action_space.sample()
                (next_state, reward, done, info) = self.env.step(action)
                if "TimeLimit" in info:
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
        elif policy_name == "linear":
            policy = LinearDecayStrategy(init_epsilon=1.0, min_epsilon=0.1, plateu_step=750)
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
                if self.render:
                    self.env.render()
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

                # update target network periodically, linearly increase update interval
                if (count + total_count) % self.update_interval == 0:
                    self.update_target_network()
                    # print("updated target network!")
                    optimize_jim(self.buffer, self.criterion, self.batch_size, self.online_model, self.target_model,
                                 self.gamma, self.optimizer, self.epoch_loss)
            self.render = False  # reset render flag
            total_count += count
            self.reward_record.append(tmp_reward)
            self.rolling_average.append(np.average(self.reward_record[-100:]))
            avg_loss = np.average(self.epoch_loss)
            if policy_name == "linear":  # update epsilon at every epoch instead of step
                policy._epsilon_update()
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
