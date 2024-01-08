import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        # if  self.buffer.count((state, action, reward, next_state, done)) != 0:
        #     return
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class ReplayBufferN:
    def __init__(self, capacity, batch_size):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done, step):
        # if  self.buffer.count((state, action, reward, next_state, done)) != 0:
        #     return
        self.buffer.append((state, action, reward, next_state, done, step))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done, step = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, step

    def size(self):
        return len(self.buffer)