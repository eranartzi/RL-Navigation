import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from hparams import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    '''An RL agent that implements a DQN to learn an ~optimal policy for RL problems'''
    def __init__(self, state_size, action_size, seed):
        '''Args:
            state_size: Int, number of dims in the state space
            action_size: Int, number of dims in the action space
            seed: Int, to set random seed'''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # create the local and target Qnetworks and Optimizer set to optimize the local network
        self.qn_local = QNetwork(state_size, action_size, seed=seed).to(device)
        self.qn_target = QNetwork(state_size, action_size, seed=seed).to(device)
        self.optimizer = optim.Adam(params=self.qn_local.parameters(), lr=LR)
        
        # create the memory buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # counter for steps to learn
        self.t_step = 0
    
    
    def step(self, state, action, reward, next_state, done):
        '''add experience information to the memory buffer and every UPDATE_EVERY steps, invoke the learn method
        Args:
            state: state vector
            action: Int, preformed action
            reward: Int, reward from action in given state
            next_state: state vector
            done: Bool, if episode is over'''
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    
    def act(self, state, eps=0.):
        '''choose the next action to a given state with an epsilon-greedy policy
        Args:
            state: state vector
            eps: default 0, epsilon for distribution of random action
        return:
            Int, action'''
        # implement an epsilon greedy policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qn_local.eval()
        with torch.no_grad():
            action_values = self.qn_local(state)
        self.qn_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    
    def learn(self, experiences, gamma):
        '''compute the loss between the prediction of Qvalues using the local network and using the target network + rewards
        calculate the gradient on the local network with respect to the loss and use Adam optimizer to change params. Updaste
        the target network with soft_update
        Args:
            experiences: tuple of numpy.arrays of number=batch_size experiences
            gamma: discount rate on future rewards'''
        # collect experience tuples in memory and learn every S steps
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qn_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.qn_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(local_model=self.qn_local, target_model=self.qn_target, tau=TAU)  
    
    
    def soft_update(self, local_model, target_model, tau):
        '''Make a soft update to the target network
        Args:
            local_model: local DQN
            target_model: target DQN
            tau: Float, param for streangth of update, higher means more reliance on the new local network params'''
        # Update the local target network with the local network params
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
            