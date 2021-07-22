from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

# create the buffer
replay_buffer = ReplayBuffer(capacity=500000)

# create agent
agent = DQNAgent(replay_buffer,
                 env_name="CartPole-v1",
                 model_name="simple",
                 n_episodes=2000,
                 epsilon=0.5,
                 batch_size=64,
                 learning_rate=0.0005,
                 update_interval=10,
                 optimizer="adam")
agent.populate_buffer()
record = agent.train(policy_name="egreedyexp")
print(record)
plt.plot(record)
