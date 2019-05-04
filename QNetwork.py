import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    '''A simple MLP with two hidden layers. The QNetowrk class implements the torch.nn.Module architecture. For more information visit pytorch.com
    Args:
        state_size: Int, size of the state space
        action_size: Int, size of the action space
        seed: Int, used to reproduce results between different experiments
        fc1: Int, first hidden layer size/number of hidden nodes
        fc2: Int, second hidden layer size/number of hidden nodes'''
    def __init__(self, state_size, action_size, seed, fc1=64, fc2=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed=seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        
    def forward(self, state):
        '''makes one pass through the net.
        Args:
            state: vector represention of the current envionment state
        returns:
            vector of Qvalues for actions'''
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return self.fc3(x)