import gym
from gym import spaces
import numpy as np
import pygame

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
        self.reward_available = True

    def step(self, actions):
        rewards = [0.0] * self.num_agents
        done = False

        # Update the agent positions based on the chosen actions
        for i in range(self.num_agents):
            action = actions[i]

            if action == 0:  # left
                self.agent_positions[i] = (self.agent_positions[i][0], max(0, self.agent_positions[i][1] - 1))
            elif action == 1:  # right
                self.agent_positions[i] = (self.agent_positions[i][0], min(self.grid_size - 1, self.agent_positions[i][1] + 1))
            elif action == 2:  # up
                self.agent_positions[i] = (max(0, self.agent_positions[i][0] - 1), self.agent_positions[i][1])
            elif action == 3:  # down
                self.agent_positions[i] = (min(self.grid_size - 1, self.agent_positions[i][0] + 1), self.agent_positions[i][1])

        # Check if any agent reached the reward position
        for i in range(self.num_agents):
            if self.agent_positions[i] == self.reward_pos:
                if self.reward_available:
                    rewards[i] = 1.0
                    self.reward_available = False

        # Check if all agents reached the reward or the maximum number of steps is reached
        if not self.reward_available or np.sum(rewards) == self.num_agents or np.sum(rewards) > 0.0:
            done = True

        return self.get_state(), rewards, done, {}

    def reset(self):
        # Reset the agent and reward positions
        self.agent_positions = [(0, 0)] * self.num_agents
        self.reward_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        self.reward_available = True

        return self.get_state()

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("CustomEnv")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        cell_size = self.window_size // self.grid_size

        # Draw grid lines
        for x in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (0, 0, 0), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (0, 0, 0), (0, y), (self.window_size, y))

        # Draw agent positions
        for i in range(self.num_agents):
            agent_pos = self.agent_positions[i]
            pygame.draw.rect(self.window, color_palette[i], (agent_pos[1] * cell_size, agent_pos[0] * cell_size, cell_size, cell_size))

        # Draw reward position
        pygame.draw.rect(self.window, (255, 0, 0), (self.reward_pos[1] * cell_size, self.reward_pos[0] * cell_size, cell_size, cell_size))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for agent_pos in self.agent_positions:
            state[agent_pos[0], agent_pos[1]] = 1.0
        return state


# Register the environment with Gym
gym.envs.register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv',
)