import gym
from gym import spaces
import numpy as np
import pygame
import random
import torch

color_palette = (
    (240, 163, 255),
    (0, 117, 220),
    (153, 63, 0),
    (76, 0, 92),
    (25, 25, 25),
    (0, 92, 49),
    (43, 206, 72),
    (255, 204, 153),
    (128, 128, 128),
    (148, 255, 181),
    (143, 124, 0),
    (157, 204, 0),
    (194, 0, 136),
    (0, 51, 128),
    (255, 164, 5),
    (255, 168, 187),
    (66, 102, 0),
    (255, 0, 16),
    (94, 241, 242),
    (0, 153, 143),
    (224, 255, 102),
    (116, 10, 255),
    (153, 0, 0),
    (255, 255, 128),
    (255, 255, 0),
    (255, 80, 5)
)

class CustomEnv(gym.Env):
    def __init__(self, num_agents=1):
        super(CustomEnv, self).__init__()

        # Define the dimensions of the grid
        self.grid_size = 5

        # Number of agents
        self.num_agents = num_agents

        # Active agent
        self.active_agent = 0

        # Render
        self.window_size = 768
        self.window = None
        self.clock = None

        # Define the possible actions (left, right, up, down, stay)
        self.action_space = spaces.Discrete(5)

        # Define the observation space (grid size + agent positions)
        low = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        high = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize the agent and reward positions
        self.agent_positions = [(0, 0)] * self.num_agents
        self.reward_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))

    def step(self, action):
        reward = 0
        done = False
        agent = self.active_agent

        # Update the agent position based on the chosen action
        agent_pos = self.agent_positions[agent]

        if action == 0:  # left
            new_pos = (agent_pos[0], max(0, agent_pos[1] - 1))
        elif action == 1:  # right
            new_pos = (agent_pos[0], min(self.grid_size - 1, agent_pos[1] + 1))
        elif action == 2:  # up
            new_pos = (max(0, agent_pos[0] - 1), agent_pos[1])
        elif action == 3:  # down
            new_pos = (min(self.grid_size - 1, agent_pos[0] + 1), agent_pos[1])
        else:  # stay
            new_pos = agent_pos

        # Check for collisions with other agents
        if new_pos in self.agent_positions and new_pos != agent_pos:
            # Agent collided with another agent, stay in the current position
            new_pos = agent_pos

        self.agent_positions[agent] = new_pos

        # Check if the agent reached the reward position
        if new_pos == self.reward_pos:
            reward = 1.0
            self.reset_reward()

        return self.get_state(), reward, done, {}
    
    def reset_reward(self):
        # Generate a new random reward position that is not the same as any agent position
        valid_positions = set([(i, j) for i in range(self.grid_size) for j in range(self.grid_size)])
        valid_positions.difference_update(set(self.agent_positions))
        self.reward_pos = random.choice(list(valid_positions))

    def reset(self):
        random.seed()
        valid_positions = ([(i, j) for i in range(self.grid_size) for j in range(self.grid_size)])
        self.agent_positions = random.sample(valid_positions, k=self.num_agents)
        self.reset_reward()

        return self.get_state()

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("CustomEnv")
            self.clock = pygame.time.Clock()

        self.window.fill((14, 17, 17))

        cell_size = self.window_size // self.grid_size

        # Draw grid lines
        for x in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (90, 90, 90), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (90, 90, 90), (0, y), (self.window_size, y))

        # Draw agent positions
        for i in range(self.num_agents):
            colorOffset = 1 % len(color_palette)
            color_tuple = color_palette[colorOffset]
            colorr = color_tuple[0]
            colorg = color_tuple[1]
            colorb = color_tuple[2]
            agent_pos = self.agent_positions[i]
            pygame.draw.circle(self.window, (colorr,colorg,colorb), ((agent_pos[0]+0.5)*cell_size,(agent_pos[1]+0.5)*cell_size), cell_size / 3)

        # Draw reward position
        pygame.draw.rect(self.window, (0, 255, 0), (self.reward_pos[1] * cell_size, self.reward_pos[0] * cell_size, cell_size, cell_size))

        pygame.display.flip()
        # self.clock.tick(4)

    def close(self):
        if self.window is not None:
            pygame.quit()

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Agent positions
        n = 1
        for agent_pos in self.agent_positions:
            state[agent_pos[0], agent_pos[1]] = n
        # Reward positions
        state[self.reward_pos[0], self.reward_pos[1]] = -1
        state = [torch.Tensor(row) for row in state]
        return state


# Register the environment with Gym
gym.envs.register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv',
)