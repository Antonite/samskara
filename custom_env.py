import gym
from gym import spaces
import numpy as np
import pygame

color_pallete = (
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
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the dimensions of the grid
        self.grid_size = 5

        # Render
        self.window_size = 768
        self.window = None
        self.clock = None

        # Define the possible actions (left, right, up, down, stay)
        self.action_space = spaces.Discrete(5)

        # Define the observation space (grid size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32
        )

        # Initialize the agent and reward positions
        self.agent_pos = (0, 0)
        self.reward_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))

    def step(self, action):
        # Update the agent's position based on the chosen action
        if action == 0:  # left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 1:  # right
            self.agent_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))
        elif action == 2:  # up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 3:  # down
            self.agent_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 4:  # stay
            pass

        # Check if the agent has reached the reward tile
        if self.agent_pos == self.reward_pos:
            # Calculate the reward
            reward = 1.0

            # Move the reward tile to a random position
            self.reward_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        else:
            reward = 0.0

        # Define whether the episode is done
        done = False

        # Return the updated observation, reward, done flag, and additional information (optional)
        return self._get_observation(), reward, done, {}


    def reset(self):
        # Reset the agent's position
        self.agent_pos = (0, 0)

        # Move the reward tile to a random position
        self.reward_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))

        # Return the initial observation
        return self._get_observation()


    def render(self, mode='human'):
        # grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # grid[self.agent_pos] = 0.5
        # grid[self.reward_pos] = 1.0

        # print(grid)

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((14, 17, 17))
        pix_square_size = self.window_size / self.grid_size  # The size of a single grid square in pixels

        loc = [p * pix_square_size for p in self.reward_pos]
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                loc,
                (pix_square_size, pix_square_size),
            ))

        # for i, a in enumerate(self.agent_locations.values()):
        colorOffset = 1 % len(color_pallete)
        color_tuple = color_pallete[colorOffset]
        colorr = color_tuple[0]
        colorg = color_tuple[1]
        colorb = color_tuple[2]
        loc = [(p + 0.5) * pix_square_size for p in self.agent_pos]
        pygame.draw.circle(
            canvas,
            (colorr, colorg, colorb),
            loc,
            pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (90, 90, 90),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (90, 90, 90),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(10)

        

    def _get_observation(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.agent_pos] = 0.5
        grid[self.reward_pos] = 1.0

        return grid


gym.register(id='CustomEnv-v0', entry_point='custom_env:CustomEnv')