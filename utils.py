from PIL import Image, ImageOps
import gymnasium as gym
import numpy as np
import os
import yaml
from matplotlib import pyplot as plt


def read_config_file(config_file):
    if config_file is None:
        print("config file is None, exiting...")
        exit()
    if os.path.isfile(config_file):
        print(f"loading configuration file {config_file}")
    else:
        print(f"error, config file {config_file} not found, exiting...")
        exit()
    config = read_yaml(config_file)
    return config


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def plot_training_history(rewards_array: np.array, save: bool=False, filename="history.png"):
    fig = plt.figure()
    rolling_history = []
    for t in range(len(rewards_array)):
        subsection = rewards_array[:t+1]
        rolling_history.append(np.average(subsection[-100:]))
    plt.style.use('seaborn-v0_8')
    plt.plot(rewards_array, alpha=0.5, label='rewards')
    plt.plot(np.array(rolling_history), label='100 episode rolling avg')
    plt.legend()
    if save:
        plt.savefig(filename)
    plt.show()


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])
    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def render_env(env, policy, gif_name):
    tmp_rewards = 0
    state = env.reset()
    frames = []
    terminal = False
    while not terminal:
        action = policy(0, state)
        [next_state, reward, terminal, info] = env.step(action)
        tmp_rewards += reward
        img = env.render(mode='rgb_array')
        frames.append(img)
        state = next_state
        if tmp_rewards > 9000:
            print("reward exceeded 9000!, terminating")
            terminal = True
    print(tmp_rewards)
    env.close()
    image_name = gif_name + ".gif"
    frame_images = [Image.fromarray(frame) for frame in frames]
    frame_images[0].save(image_name, format='GIF', append_images=frame_images[1:], save_all=True, duration=30, loop=0)


"""
gym wrappers!!
credit to work below:
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/utils.py
https://github.com/Arrabonae/openai_DDDQN/blob/master/env.py
"""


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        (max_state, reward, done, info) = (None, 0, False, None)
        for i in range(self.skip):
            (state, reward, done, info) = self.env.step(action)
            if max_state is None:
                max_state = state
            else:
                max_state = np.maximum(max_state, state)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(1, 84, 84)):
        super(ProcessFrame, self).__init__(env)
        self.crop_shape = shape
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(1.0),
                                                shape=self.crop_shape, dtype=np.float32)

    def observation(self, obs):
        img = Image.fromarray(obs)
        greyscale = ImageOps.grayscale(img)
        cropped = greyscale.crop((0, 25, 160, 200))
        reshaped = cropped.resize(self.crop_shape[1:3])
        new_obs = np.asarray(reshaped).reshape(self.crop_shape)
        # plt.imshow(new_obs)
        rescaled = new_obs / 255.0
        return rescaled


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        low = env.observation_space.low.repeat(n_steps, axis=0)  # (4, 84, 84)
        high = env.observation_space.high.repeat(n_steps, axis=0)  # (4, 84, 84)
        self.buffer = np.zeros_like(low, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]  # shift first three img forward
        self.buffer[-1] = observation  # append latest observation into buffer
        return self.buffer


def mod_env(env_name, shape=(1, 84, 84)):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = ProcessFrame(env, shape=shape)
    env = BufferWrapper(env, 4)
    return env
