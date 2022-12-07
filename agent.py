import torch
import random
from torch.autograd import Variable
import numpy as np


class Agent:
    ''' Deep Q Neural Network Agent. '''

    def __init__(self, agent_id, input_dim, output_dim, hidden_dim=32, agent_learn_rate=0.001):
        print("Initializing Agent %s -- #States: %i -- #PotentialActions: %i" % (agent_id, input_dim, output_dim))

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), agent_learn_rate)
        self.device = torch.device("cuda") if (torch.cuda and torch.cuda.is_available()) else torch.device
        self.model.to(self.device)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device))

    def replay(self, memory, size, gamma=0.9):
        if len(memory) >= size:
            batch = random.sample(memory, size)
            batch_t = list(map(list, zip(*batch)))  # Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]

            actions_tensor = torch.Tensor(actions).long().to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)

            all_q_values = self.predict(np.array(states))  # predicted q_values of all states
            all_q_values_next = self.predict(np.array(next_states))  # predicted q_values of all states of taken action
            # Update q values
            all_q_values[range(len(all_q_values)), actions_tensor] = rewards + \
                gamma*torch.max(all_q_values_next, axis=1).values

            self.update(np.array(states), all_q_values)
            return torch.max(torch.max(all_q_values)).float()

        return 0
