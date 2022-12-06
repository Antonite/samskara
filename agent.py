import torch
import random
from torch.autograd import Variable


class Agent:
    ''' Deep Q Neural Network Agent. '''

    def __init__(self, agent_id, state_dim, action_dim, hidden_dim=32, agent_learn_rate=0.001):
        print("Initializing Agent %s -- #States: %i -- #PotentialActions: %i" % (agent_id, state_dim, action_dim))

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )
        # move to GPU if cuda available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), agent_learn_rate)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device))

    def replay(self, memory, size, gamma=0.9):
        # Try to improve replay speed
        if len(memory) >= size:
            batch = random.sample(memory, size)
            batch_t = list(map(list, zip(*batch)))  # Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]

            states_tensor = torch.Tensor([list(state.values()) for state in states])
            actions_tensor = torch.Tensor(actions).long().to(self.device)
            next_states = torch.Tensor([list(state.values()) for state in next_states])
            rewards = torch.Tensor(rewards).to(self.device)

            all_q_values = self.predict(states_tensor)  # predicted q_values of all states
            all_q_values_next = self.predict(next_states)  # predicted q_values of all states of taken action
            # Update q values
            all_q_values[range(len(all_q_values)), actions_tensor] = rewards + \
                gamma*torch.max(all_q_values_next, axis=1).values

            self.update(states_tensor.tolist(), all_q_values.tolist())
            return max(max(all_q_values.tolist()))

        return 0
