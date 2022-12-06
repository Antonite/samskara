import torch

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
        y_pred = self.model(torch.Tensor(list(state.values())).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(list(state.values())).to(self.device))
