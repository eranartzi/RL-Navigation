import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    '''Replay buffer to hold most recent experiences tuples and sampling from that group
    Args:
        action_size: Int, size of action space
        buffer_size: Int, the max number of experience tuples to store in memory
        batch_size: Int, number of exeriences to sample from the buffer
        seed: Int, to set random.seed for reproducable results'''
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.exprience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(a=seed)
        
    def add(self, state, action, reward, next_state, done):
        '''add an experience tuple to memory'''
        e = self.exprience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        '''sample a batch of experience tuples'''
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        nx_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, nx_states, dones)
    
    def __len__(self):
        return len(self.memory)