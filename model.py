import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=800, fc2_units=400, fc3_units=200):
        '''

        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_space)
        self.fcn = out_fcn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fcn(self.fc4(x))


class Critic(nn.Module):
    def __init__(self, state_space, action_space, fc1_units=800, fc2_units=400, fc3_units=200):
        '''

        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_space, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_space)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x, action):
        xs = F.relu(self.fc1(x))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

