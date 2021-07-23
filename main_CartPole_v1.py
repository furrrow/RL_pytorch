from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

# create the buffer
replay_buffer = ReplayBuffer(capacity=500000)
# env_name="CartPole-v1 can use SimpleModel
# try Pong-v0 later

# create agent
agent = DQNAgent(replay_buffer,
                 env_name="CartPole-v1",
                 model_name="simple",
                 n_episodes=250,
                 epsilon=0.5,
                 batch_size=64,
                 learning_rate=0.0005,
                 update_interval=150,
                 gamma=1,
                 optimizer="adam")
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

