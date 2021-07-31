from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent
from DDQNAgent import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt

# create the buffer
replay_buffer = ReplayBuffer(capacity=50000)
# env_name="CartPole-v1 can use SimpleModel
# try Pong-v0 later

# create agent
agent = DDQNAgent(replay_buffer,
                  env_name="LunarLander-v2",
                  model_name="simple",
                  n_episodes=1000,
                  epsilon=0.5,
                  batch_size=64,
                  learning_rate=0.0005,
                  update_interval=5000,
                  gamma=0.9995,
                  optimizer="adam",
                  modify_env=False)
agent.populate_buffer(factor=0.25)
record, rolling_avg, loss_record = agent.train(policy_name="linear")
x = np.arange(len(record))
# print(record)

fig, rewards_plot = plt.subplots(figsize=(6, 6))
rewards_plot.plot(x, record, label="rewards")
rewards_plot.plot(x, rolling_avg, label="rolling_avg")
rewards_plot.plot(x, loss_record, label="loss")
rewards_plot.legend()
plt.savefig('LunarLander_DDQN.png')
plt.show()
