from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Trainer():
    """
    Wrapping class to train model and update Q-Learning hyperparameters
    """
    def __init__(self,
                 model: nn.Module,
                 action_size: int,
                 memory: int = 2000,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                ):
        self.model = model
        self.action_size = action_size
        self.memory = deque(maxlen=memory)
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Saving observations to deque"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Exploration vs Exploitation: Choose action based on epsilon
        (high e -> more likely to choose random/explore;
        low e -> more likely to choose known/exploit)
        """
        if np.random.rand() <= self.epsilon:
            # explore
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return np.argmax(act_values.cpu().numpy()[0])
    
    def replay(self, batch_size):
        """
        Choose random selection of saved observations to train model params on.
        """
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            
            target = reward
            if not done:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_q_values).item()
            
            # Forward pass
            q_values = self.model(state_tensor)
            target_f = q_values.clone().detach()
            target_f[0][action] = target
            
            # Backward pass
            loss = self.loss_fn(q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgent(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_layers: list[int] = [24, 24],
                 activation_fn = nn.ReLU(), #example uses tanh
                 dropout: float = 0.5
                ):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
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
    

    
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent_model = DQNAgent(state_size, action_size, activation_fn=nn.Tanh())
    trainer = Trainer(agent_model, action_size=action_size)
    episodes = 5000

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, 4])
        for time_t in range(500):
            # env.render()
            action = trainer.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            trainer.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state
            if terminated or truncated:
                print(f"Episode: {e}/{episodes} | Score: {time_t}")
                break
        trainer.replay(32)


if __name__ == "__main__":
    main()