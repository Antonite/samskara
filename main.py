import os
import random
import sys

from envs.env import TestEnv
from agent import Agent
from torch import argmax

env = TestEnv(render_mode="human")

# Number of states
n_state = len(env.observation_space)
# Number of actions
n_action = env.action_space.n
# Number of hidden nodes in the DQN
n_hidden = 32
# Learning rate
lr = 0.001


class Main:

    def __init__(self, gamma, epsilon, eps_step_decay=0.9999, max_steps=sys.maxsize - 1):
        self.step = 0
        self.states = env.reset()
        self.agents = {}

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_step_decay = eps_step_decay
        self.max_steps = max_steps

        print(max_steps)

        for env_state in self.states:
            self.agents[env_state] = Agent(env_state, n_state, n_action, n_hidden, lr)

    def q_learning(self, render_at_step=1):
        for i in range(self.max_steps):
            self.step = i

            agent_actions = {}
            for agent in self.agents:
                # Epsilon percent chance to take a random movement decayed by eps_decay per step
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    agent_qvals = self.agents[agent].predict(self.states[agent])
                    action = argmax(agent_qvals).item()

                agent_actions[agent] = action

            self.epsilon = max(self.epsilon * self.eps_step_decay, 0.01)

            # Take action and add reward to total
            shouldRender = self.step > render_at_step
            obs, rewards = env.step(agent_actions, shouldRender)

            # every expected weighted action for every agent
            agent_weights_last = {agent: self.agents[agent].predict(self.states[agent]).tolist() for agent in obs}

            # Update network weights based
            agent_weights_next = {agent: self.agents[agent].predict(obs[agent]) for agent in obs}

            for agent in agent_weights_last:
                agent_weights_last[agent][agent_actions[agent]] = rewards[agent] + self.gamma * max(agent_weights_next[agent]).item()
                self.agents[agent].update(self.states[agent], agent_weights_last[agent])

            self.print_step(agent_actions, agent_weights_last, rewards)
            self.states = obs

    def print_step(self, actions, weights, rewards):
        if self.step % 100 == 0 or self.step > 10000:
            print("Step: %s -- Epsilon %s" % (str(self.step), str(self.epsilon)))
            for action in actions:
                print(', '.join(["Agent: " + str(action), "Action: " + env.get_human_readable_action(actions[action]) + " - " + str(weights[action][actions[action]]),
                                 "Reward " + str(rewards[action])]))
                print("Last Action Weights: " + str(weights[action]))
            print(os.linesep)


if __name__ == "__main__":
    main = Main(gamma=.8, epsilon=0.3)
    main.q_learning(10000)
    input()

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     action = None if termination or truncation else env.action_space(
#         agent).sample()
#     env.step(action)
#     print(agent)
#     print(action)
# simple_dqn = Agent(n_state, n_action, n_hidden, lr)
