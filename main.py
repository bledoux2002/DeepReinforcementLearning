from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DQNAgent(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_layers: list[int] = [24, 24],
                 activation_fn = nn.ReLU(), #example uses tanh
                 dropout: float = 0.5,
                 memory: int = 2000,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                ):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory)
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        if not hidden_layers:
            self.layers = nn.Sequential(
                nn.Linear(state_size, action_size)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(state_size, hidden_layers[0]),
                activation_fn,
                nn.Dropout(p=dropout),
            )
            for i in range(0, len(hidden_layers)-1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                self.layers.append(activation_fn)
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(hidden_layers[-1], action_size))

    def forward(self, x):
        return self.layers(x)