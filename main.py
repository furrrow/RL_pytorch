import click
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agents.DQNAgent import DQNAgent
from agents.DDQNAgent import DDQNAgent
from replay_buffer import DequeReplayBuffer, NumpyReplayBuffer

from replay_buffer import DequeReplayBuffer
from utils import read_config_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--config_file', default="./configs/cartpole_dqn.yml", help='<path_to_config_file.yml>')
def main(config_file):
    config = read_config_file(config_file)
    save_name = config['env_name'] + "_" + config['agent_type']
    # first make environment
    env = gym.make(config['env_name'], render_mode="rgb_array")
    # replay buffer
    if config['buffer_type'] == "Deque":
        buffer = DequeReplayBuffer(config['replay_capacity'])
    elif config['buffer_type'] == "Numpy":
        buffer = NumpyReplayBuffer(config['replay_capacity'], config['replay_batch_size'])
    # initialize agent
    if config['agent_type'] == "DQN":
        agent = DQNAgent(config, buffer, env)
    elif config['agent_type'] == "DDQN":
        agent = DDQNAgent(config, buffer, env)
    else:
        agent = None
    agent.populate_buffer()
    record, rolling_avg, loss_record = agent.train(policy_name=config['policy_name'])
    x = np.arange(len(record))
    np.save(f"{save_name}_record.npy", x)
    # print(record)

    fig, rewards_plot = plt.subplots(figsize=(6, 6))
    rewards_plot.plot(x, record, label="rewards")
    rewards_plot.plot(x, rolling_avg, label="rolling_avg")
    # rewards_plot.plot(x, loss_record, label="loss")
    rewards_plot.legend()
    plt.savefig(f"{save_name}_results.png")
    plt.show()


if __name__ == '__main__':
    main()
