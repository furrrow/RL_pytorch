from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

"""
information I can find on Pong-V0
input shape: (210, 160, 3)
action space: Discrete(6)
action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

world model, scale it to 64 by 64, kept all three channels
for this case, we can try 84 by 84, greyscale, frameskip 4, stack 4


"""

# create the buffer
replay_buffer = ReplayBuffer(capacity=70000)

# create agent
agent = DQNAgent(replay_buffer,
                 env_name="PongNoFrameskip-v4",
                 model_name="cnn",
                 n_episodes=2000,
                 epsilon=0.5,
                 batch_size=32,
                 learning_rate=0.0001,
                 update_interval=1000,
                 gamma=0.99,
                 optimizer="rmsprop",
                 modify_env=True,
                 render=True)

agent.populate_buffer()
record, rolling_avg, loss_record = agent.train(policy_name="egreedyexp")
x = np.arange(len(record))
# print(record)

fig, rewards_plot = plt.subplots(figsize=(6, 6))
rewards_plot.plot(x, record, label="rewards")
rewards_plot.plot(x, rolling_avg, label="rolling_avg")
rewards_plot.plot(x, loss_record, label="loss")
rewards_plot.legend()
plt.savefig('Pong_main.png')
plt.show()

