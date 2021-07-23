from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

"""
information I can find on Pong-V0
input shape: (210, 160, 3)
action space: Discrete(6)
action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

"""

# create the buffer
replay_buffer = ReplayBuffer(capacity=500)

# create agent
agent = DQNAgent(replay_buffer,
                 env_name="Pong-v0",
                 model_name="cnn",
                 n_episodes=50,
                 epsilon=0.5,
                 batch_size=32,
                 learning_rate=0.0005,
                 update_interval=150,
                 gamma=1,
                 optimizer="adam",
                 merge_rgb=True)

agent.populate_buffer()
record, rolling_avg, loss_record = agent.train(policy_name="egreedyexp")
x = np.arange(len(record))
# print(record)

fig, rewards_plot = plt.subplots(figsize=(6, 6))
rewards_plot.plot(x, record, label="rewards")
rewards_plot.plot(x, rolling_avg, label="rolling_avg")
rewards_plot.plot(x, loss_record, label="loss")
rewards_plot.legend()
plt.savefig('rewards_plot.png')
plt.show()

