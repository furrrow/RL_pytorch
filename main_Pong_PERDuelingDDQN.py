from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from agents.PERDuelingDDQNAgent import PERDuelingDDQNAgent
import numpy as np
import matplotlib.pyplot as plt

"""
Pong-v0, dueling ddqn


"""

# create the buffer
replay_buffer = PrioritizedReplayBuffer(max_samples=50000,
                                        batch_size=32,
                                        rank_based=False,
                                        alpha=0.6,
                                        beta0=0.1,
                                        beta_rate=0.99995)

# create agent
agent = PERDuelingDDQNAgent(replay_buffer,
                            # env_name="Pong-v0",
                            env_name="PongNoFrameskip-v4",
                            model_name="duelcnn",  # "online",
                            n_episodes=2000,
                            epsilon=0.5,
                            batch_size=32,
                            learning_rate=0.0005,
                            update_interval=5000,
                            gamma=0.9995,
                            optimizer="rmsprop",
                            modify_env=True,
                            tau=0.1)

agent.populate_buffer(factor=0.33)
record, rolling_avg, loss_record = agent.train(policy_name="linear")
x = np.arange(len(record))
# print(record)

fig, rewards_plot = plt.subplots(figsize=(6, 6))
rewards_plot.plot(x, record, label="rewards")
rewards_plot.plot(x, rolling_avg, label="rolling_avg")
rewards_plot.plot(x, loss_record, label="loss")
rewards_plot.legend()
plt.savefig('Pong_DuelDDQN.png')
plt.show()
