import os
import random
import sys
import time
import threading
import numpy as np
import torch
from envs.sequential import SequentialEnv
from agent import Agent
from copy import copy

size = 5
env = SequentialEnv(render_mode="human", size=size)

# Number of states
n_inputs = env.inputs
# Number of actions
n_outputs = env.action_space.n
# Number of hidden nodes in the DQN
n_hidden = round(n_inputs*2/3) + n_outputs

# Hyper params
lr = 0.0001
gamma = 0.9
epsilon = 0.5
eps_step_decay = 0.9999
replay_size = 256

# rendering
maxStoredQVals = 3000


class Main:
    def __init__(self, gamma, epsilon, eps_step_decay, size, max_steps=sys.maxsize - 1):
        env.reset()

        self.step = 0
        self.agents = {}
        self._max_q_values = []
        self.size = size

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_step_decay = eps_step_decay
        self.max_steps = max_steps

        for agent in env.agents:
            self.agents[agent] = Agent(agent, n_inputs, n_outputs, n_hidden, lr)

    def replay(self, agent, memory, replay_size):
        qval = self.agents[agent].replay(memory, replay_size, self.gamma)
        self._max_q_values.append(qval)
        if len(self._max_q_values) > maxStoredQVals:
            self._max_q_values.pop(0)

    def q_learning(self, replay_size=32, forceRender=False, dynamicRender=False):
        ts = time.time()
        tt = time.time()
        self.renderThreshold = self.size ** 2 * 0.95 * 10
        # init memory per agent
        memory = {}
        for agent in self.agents:
            memory[agent] = []

        for i in range(self.max_steps):
            self.step = i

            # conditional rendering
            shouldRender = self.step > 1000 and (forceRender or (
                dynamicRender and round(self.step / 10)*10 % 5000 == 0))

            # step
            for agent in self.agents:
                state = env.state(agent)
                mask = env.action_mask(agent)
                # Epsilon percent chance to take a random movement decayed by eps_decay per step
                if random.random() < self.epsilon:
                    pActs = []
                    for i, v in enumerate(mask):
                        if v == 1:
                            pActs.append(i)
                    action = pActs[random.randrange(0, len(pActs))]
                else:
                    agent_qvals = self.agents[agent].predict(state)
                    maxq = None
                    for i, q in enumerate(agent_qvals):
                        if mask[i] == 1 and (maxq == None or q > maxq):
                            maxq = q
                            action = i

                # take action and add reward to total
                obs, reward = env.step(agent, action, shouldRender)

                # update memory
                memory[agent].append((state, action, obs, reward))

            # learn
            jobs = []
            for agent in self.agents:
                # trim memory
                if len(memory[agent]) > 1500:
                    memory[agent] = memory[agent][len(memory)-1500:]
                # Update network weights using replay memory
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
                self.print_step(te-tt, te-ts)
                tt = te

    def print_step(self, batchTime, totalTime):
        print("Step: %s -- Epsilon %s -- QVal %s/%s -- Batch %s -- Total %s" %
              (str(self.step),
               str("%.2f" % self.epsilon),
               str("%.0f" % max(self._max_q_values)),
               str("%.0f" % self.renderThreshold),
               str("%.1fs" % batchTime),
               str("%.1fs" % totalTime)))


if __name__ == "__main__":
    main = Main(gamma=gamma, epsilon=epsilon, eps_step_decay=eps_step_decay, size=size)
    main.q_learning(replay_size=replay_size, forceRender=False, dynamicRender=True)
    input()
