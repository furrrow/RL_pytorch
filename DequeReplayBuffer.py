import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class DequeReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        # self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

    def save(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def transitions(self):
        return self.transition

    def length(self):
        return len(self.memory)
