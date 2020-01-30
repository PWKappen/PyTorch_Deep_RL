import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

Transition2 = namedtuple('Transition2',
                        ('state', 'action', 'done', 'reward'))

ReturnTransition = namedtuple('Transition3', ('state', 'action', 'done', 'next_state', 'reward'))

class BasicMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ExtendetMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.memory_reward = []
        self.memory_action = []
        self.memory_done = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        trns = Transition2(*args)
        self.memory[self.position] = trns.state
        self.memory_reward[self.position] = trns.reward
        self.memory_action[self.position] = trns.action
        self.memory_done[self.position] = trns.done
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            choices = random.choice(self.position,batch_size)
        else:
            choices = random.choice(self.capacity, batch_size)
        tmpStates = [self.memory[choices],self.memory[choices-1],self.memory[choices-2],self.memory[choices-3],self.memory[choices-4]]
        states = torch.cat(tmpStates[1:4],dim=1)
        nextStates = torch.cat(tmpStates[0:3],dim=1)
        rewards = self.memory_reward[choices-1]
        actions = self.memory_action[choices-1]
        done = self.memory_done[choices-1]
        return ReturnTransition(states, actions, done, nextStates, rewards)

    def __len__(self):
        return len(self.memory)