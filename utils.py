from abc import ABC

from PIL import Image, ImageOps
import gym
import numpy as np


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
credit to youtube video below on how to handle gym wrappers:
https://www.youtube.com/watch?v=a5XbO5Qgy5w&t=2441s&ab_channel=MachineLearningwithPhil

need verifications on this...   
 > some environments will repeate your action n times,
 > but may randomly(25%) select an action other than action chosen

"""


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def step(self, action):
        (state, reward, done, info) = (None, 0, False, None)
        for i in range(self.skip):
            (state, reward, done, info) = self.env.step(action)
            reward += reward
            if done:
                break
        return state, reward, done, info


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84, 1)):
        super(PreProcessFrame, self).__init__(env)
        self.crop_shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=self.crop_shape, dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs, self.crop_shape)

    @staticmethod
    def process(frame, crop_shape):
        img = Image.fromarray(frame)
        greyscale = ImageOps.grayscale(img)
        cropped = greyscale.crop((0, 25, 160, 200))
        reshaped = cropped.resize(crop_shape[0:2])
        return np.asarray(reshaped).reshape(crop_shape)


class MakeChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super(MakeChannelFirst, self).__init__(env)
        shape = self.observation_space.shape
        new_shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        low = env.observation_space.low.repeat(n_steps, axis=0)
        high = env.observation_space.high.repeat(n_steps, axis=0)
        self.buffer = np.zeros_like(low, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def mod_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MakeChannelFirst(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)
