import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        
        n_actions = 2
    
        self.Q1 = nn.Sequential(
            nn.Linear(state_size + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(state_size + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # init network weights.
        self.apply(weights_init)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)
        
        return q1, q2

