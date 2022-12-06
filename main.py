import os
import random
import sys
import time
import threading

from envs.env import TestEnv
from agent import Agent
from torch import argmax
from copy import copy

size = 15
env = TestEnv(render_mode="human", size=15)

# Number of states
n_state = len(env.observation_space)
# Number of actions
n_action = env.action_space.n
# Number of hidden nodes in the DQN
n_hidden = round(n_state*2/3) + n_action
# Learning rate
lr = 0.001


class Main:
    def __init__(self, gamma, epsilon, eps_step_decay, size, max_steps=sys.maxsize - 1):
        self.step = 0
        self.states = env.reset()
        self.agents = {}
        self._max_q_value = 0
        self.size = size

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_step_decay = eps_step_decay
        self.max_steps = max_steps

        for env_state in self.states:
            self.agents[env_state] = Agent(env_state, n_state, n_action, n_hidden, lr)

    def replay(self, agent, memory, replay_size):
        qval = self.agents[agent].replay(memory, replay_size, self.gamma)
        if qval > self._max_q_value:
            self._max_q_value = qval

    def q_learning(self, replay_size=32):
        ts = time.time()
        tt = time.time()
        self.renderThreshold = self.size ** 2 * 0.95 * 10
        # init memory per agent
        memory = {}
        for agent in self.agents:
            memory[agent] = []

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
                    agent_qvals = self.agents[agent].predict(list(self.states[agent].values()))
                    action = argmax(agent_qvals).item()
                    random_move.append(False)

                agent_actions[agent] = action

            # start learning decay if any agent learned something
            if self._max_q_value > self.renderThreshold:
                self.epsilon = max(self.epsilon * self.eps_step_decay, 0.01)

            # take action and add reward to total
            shouldRender = self.epsilon < 0.1
            if shouldRender:
                print("%.0fs" % (time.time() - ts))
            obs, rewards = env.step(agent_actions, shouldRender)

            # update memory
            for i in obs:
                memory[i].append((copy(self.states[i]), copy(agent_actions[i]), copy(obs[i]), copy(rewards[i])))

            # if self.step % replay_size == 0:
            jobs = []
            for agent in obs:
                # Update network weights using replay memory
                if len(memory[agent]) > 1500:
                    memory[agent] = memory[agent][len(memory)-1500:]
                thread = threading.Thread(target=self.replay(agent, memory[agent], replay_size))
                jobs.append(thread)

            # Start the threads
            for j in jobs:
                j.start()
            # Ensure all of the threads have finished
            for j in jobs:
                j.join()

            if self.step % replay_size == 0:
                te = time.time()
                self.print_step(obs, agent_actions, rewards, random_move, te-tt)
                tt = te

            self.states = obs

    def print_step(self, states, actions, rewards, random_move, batchTime):
        print("Step: %s -- Epsilon %s -- QVal %s/%s -- Time %s" %
              (str(self.step), str("%.2f" % self.epsilon), str("%.0f" % self._max_q_value), str("%.0f" % self.renderThreshold), str("%.1fs" % batchTime)))
        for move, agent in enumerate(actions):
            print(', '.join(["Agent: " + str(agent),
                             "Pos: " + "[" + str(states[agent]["x"]) + "," + str(states[agent]["y"]) + "]",
                             "Action: (" + ("R" if random_move[move] else "M") + ") " + env.get_human_readable_action(
                actions[agent]),
                "Reward: " + str(rewards[agent])]))
        print(os.linesep)


if __name__ == "__main__":
    main = Main(gamma=0.9, epsilon=0.5, eps_step_decay=0.9999, size=15)
    main.q_learning(replay_size=512)
    input()
