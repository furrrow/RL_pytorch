import torch
import torch.optim as optim
import gym
import numpy as np
from SimpleModel import SimpleModel
from EGreedyStrategy import EGreedyStrategy
from EGreedyExpStrategy import EGreedyExpStrategy


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
                 optimizer="adam"):
        self.gamma = gamma
        self.buffer = buffer
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset()
        self.n_states = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.model_name = model_name
        if model_name == "simple":
            self.online_model = SimpleModel(self.n_states, self.n_action)
            self.target_model = SimpleModel(self.n_states, self.n_action)
            self.online_model.print_model(input_size=(batch_size, self.n_states))
        self.reward_record = []
        self.best_score = 75
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.lr = learning_rate
        self.update_interval = update_interval
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.online_model.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.online_model.parameters(), lr=learning_rate)

    def optimize_model(self):
        transitions = self.buffer.sample(self.batch_size)
        (states, actions, next_states, rewards, is_terminals) = self.online_model.load(transitions)
        # states, actions, rewards, next_states, is_terminals = experiences
        # batch_size = len(is_terminals)

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions.unsqueeze(1))

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

    def optimize(self):
        transitions = self.buffer.sample(self.batch_size)
        (states, actions, new_states, rewards, is_terminals) = self.online_model.load(transitions)
        continue_mask = 1 - is_terminals  # if terminal, then = 0
        q_next = self.target_model(new_states).detach()  # gradient does NOT involve the target
        q_next_max = q_next.max(1)[0].unsqueeze(1)
        # target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))

        q_target = rewards + torch.mul(q_next_max, continue_mask) * self.gamma
        q_target = q_target.unsqueeze(1)  # batch_size x 1
        q_values = self.online_model(states).gather(1, actions.unsqueeze(1))  # batch_size x 1
#             criterion = nn.MSELoss()
#             loss = criterion(q_values, q_target)
        td_error = q_values - q_target
        value_loss = td_error.pow(2).mul(0.5).mean()

        # optimize
        self.optimizer.zero_grad()
        value_loss.backward()
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
        count = 0
        if policy_name == "egreedy":
            policy = EGreedyStrategy(epsilon=self.epsilon)
        elif policy_name == "egreedyexp":
            policy = EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000)
        else:
            print("policy not yet implemented")
        for episode in range(self.n_episodes):
            state = self.env.reset()
            terminal = False
            tmp_reward = 0

            while not terminal:
                # select action
                action = policy.select_action(self.online_model, state)
                (next_state, reward, done, info) = self.env.step(action)
                # handle cases when it reaches a time limit but not actually terminal
                if ("TimeLimit" in info):
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
                if count % self.update_interval == 0:
                    self.update_target_network()

                self.optimize_model()

            self.reward_record.append(tmp_reward)
            rolling_average = sum(self.reward_record[-100:]) / 100
            if episode % 10 == 0:
                print("episode", episode, "reward", tmp_reward, "avg", rolling_average)

            if episode > 100:
                if rolling_average > 195:
                    print("!!! last 100 episode avg reward:", rolling_average)
                if tmp_reward > self.best_score:
                    best_score = tmp_reward
                    print("episode", episode, "new best score", best_score, "rolling avg", rolling_average)
        return self.reward_record
