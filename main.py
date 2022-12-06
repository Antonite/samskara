import os
import random
import sys
import time

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
    def __init__(self, gamma, epsilon, eps_step_decay, max_steps=sys.maxsize - 1):
        self.step = 0
        self.states = env.reset()
        self.agents = {}

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_step_decay = eps_step_decay
        self.max_steps = max_steps

        for env_state in self.states:
            self.agents[env_state] = Agent(env_state, n_state, n_action, n_hidden, lr)

    def q_learning(self, decay_at_reward=10):
        maxReward = 0
        for i in range(self.max_steps):
            self.step = i

            agent_actions = {}
            random_move = []
            for agent in self.agents:
                # Epsilon percent chance to take a random movement decayed by eps_decay per step
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                    random_move.append(True)
                else:
                    agent_qvals = self.agents[agent].predict(self.states[agent])
                    action = argmax(agent_qvals).item()
                    random_move.append(False)

                agent_actions[agent] = action

            # start learning decay if any agent learned something
            if maxReward > decay_at_reward:
                self.epsilon = max(self.epsilon * self.eps_step_decay, 0.01)

            # Take action and add reward to total
            shouldRender = self.epsilon < 0.05
            obs, rewards = env.step(agent_actions, shouldRender)

            # every expected weighted action for every agent
            agent_weights_last = {agent: self.agents[agent].predict(self.states[agent]).tolist() for agent in obs}

            # Update network weights based
            agent_weights_next = {agent: self.agents[agent].predict(obs[agent]) for agent in obs}

            for agent in agent_weights_last:
                # update weights
                expectedReward = rewards[agent] + self.gamma * max(agent_weights_next[agent]).item()
                agent_weights_last[agent][agent_actions[agent]] = expectedReward
                self.agents[agent].update(self.states[agent], agent_weights_last[agent])
                # keep track of biggest reward for rendering
                if expectedReward > maxReward:
                    maxReward = expectedReward

            self.print_step(obs, agent_actions, agent_weights_last, rewards, random_move, shouldRender)
            self.states = obs

    def print_step(self, states, actions, weights, rewards, random_move, rendering):
        if self.step % 1000 == 0 or rendering:
            print("Step: %s -- Epsilon %s" % (str(self.step), str("%.2f" % self.epsilon)))
            for move, agent in enumerate(actions):
                print(', '.join(["Agent: " + str(agent),
                                 "Pos: " + "[" + str(states[agent]["x"]) + "," + str(states[agent]["y"]) + "]",
                                 "Action: (" + ("R" if random_move[move] else "M") + ") " + env.get_human_readable_action(
                                     actions[agent]),
                                 "Weight: " + str("%.2f" % weights[agent][actions[agent]]),
                                 "Reward " + str(rewards[agent])]))
            print(os.linesep)


if __name__ == "__main__":
    main = Main(gamma=0.9, epsilon=0.5, eps_step_decay=0.9999)
    main.q_learning(decay_at_reward=1000)
    input()
