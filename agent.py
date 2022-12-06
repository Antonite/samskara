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

        self.optimizer = torch.optim.Adam(self.model.parameters(), agent_learn_rate)

        try:
            # move to GPU if cuda available
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device
            self.model.to(self.device)
        except TypeError:
            self.device = None
            print("Unable to move model to cuda, continuing on cpu")

    def update(self, state, y):
        """Update the weights of the network given a training sample. """

        if self.device:
            y_pred = self.model(torch.Tensor(list(state.values())).to(self.device))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y).to(self.device)))
        else:
            y_pred = self.model(torch.Tensor(list(state.values())))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y)))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            if self.device:
                return self.model(torch.Tensor(list(state.values())).to(self.device))
            else:
                return self.model(torch.Tensor(list(state.values())))
