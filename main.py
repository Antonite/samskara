import torch
from torch.autograd import Variable
import random
import gym
import time

from envs.env import TestEnv
env = TestEnv(render_mode="human")


# Number of states
n_state = len(env.observation_space)
# Number of actions
n_action = env.action_space.n
# Number of hidden nodes in the DQN
n_hidden = 32
# Learning rate
lr = 0.001

print(n_state)
print(n_action)


class Agent():
    ''' Deep Q Neural Network Agent. '''

    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(list(state.values())))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(list(state.values())))


def q_learning(env, gamma=0.9, epsilon=0.3, eps_decay=0.9999):
    """Deep Q Learning algorithm using the DQN. """
    state = env.reset()
    agents = {}
    for k in state:
        agents[k] = Agent(n_state, n_action, n_hidden, lr)

    steps = 30000
    for i in range(steps):
        actions = {}
        for a in agents:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = agents[a].predict(state[a])
                action = torch.argmax(q_values).item()

            actions[a] = action

        # Take action and add reward to total
        render = i > 10000
        obs, rewards = env.step(actions, render)

        # every expected weighted action for every agent
        q_values_l = {k: agents[k].predict(state[k]).tolist() for k in obs}

        if i % 100 == 0 or i > 10000:
            print("step: " + str(i))
            print("epsilon: " + str(epsilon))
            for a in actions:
                print("agent: " + str(a) + ", action: " + str(actions[a]) + ", reward: " + str(rewards[a]))
                print("qvals: " + str(q_values_l[a]))
                # print("state: " + str(state[a]))

        # Update network weights
        q_values_next = {k: agents[k].predict(obs[k]) for k in obs}
        for a in q_values_l:
            q_values_l[a][action] = rewards[a] + gamma * torch.max(q_values_next[a]).item()
            if i > 10000:
                print("agent: " + str(a) + ", qval: " + str(q_values_l[a][action]))
            agents[a].update(state[a], q_values_l[a])

        state = obs

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        # time.sleep(0.1)


q_learning(env, gamma=.8, epsilon=0.3)
input()


# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     action = None if termination or truncation else env.action_space(
#         agent).sample()
#     env.step(action)
#     print(agent)
#     print(action)
# simple_dqn = Agent(n_state, n_action, n_hidden, lr)
