from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

"""
Pong-ram-v0
the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. 
Each action is repeatedly performed for a duration of kk frames, where kk is uniformly sampled from 
{2,3,4}.
input shape: (128,)
action space: Discrete(6)
action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

reshape to a square shape then apply cnn model?
or just do fully connected?

to be implemented... 


"""

# create the buffer
replay_buffer = ReplayBuffer(capacity=25000)

# create agent
agent = DQNAgent(replay_buffer,
                 # env_name="Pong-v0",
                 env_name="Pong-ram-v0",
                 model_name="simple",
                 n_episodes=1000,
                 epsilon=0.5,
                 batch_size=32,
                 learning_rate=0.0005,
                 update_interval=1000,
                 gamma=0.995,
                 optimizer="adam",
                 modify_env=False)

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

