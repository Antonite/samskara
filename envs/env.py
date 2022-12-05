from copy import copy
from gym import spaces
import pygame

import numpy as np
from gymnasium.spaces import Discrete

# from pettingzoo.utils.env import ParallelEnv


class TestEnv():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512  # The size of the PyGame window

        self.foods = ["food_1"]
        self.possible_agents = ["agent_1", "agent_2", "agent_3"]
        # self.action_spaces = {k: spaces.Discrete(5) for k in self.possible_agents}
        self.action_space = spaces.Discrete(5)

        # every location
        self.observation_space = {k: spaces.Discrete(3) for k in self._all_locs()}
        self.observation_space["x"] = spaces.Discrete(size)
        self.observation_space["y"] = spaces.Discrete(size)

        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _all_locs(self):
        # every location
        locKeys = []
        for i in range(self.size):
            for j in range(self.size):
                locKeys.append(str(i)+","+str(j))
        return locKeys

    def _get_obs(self):
        # every location
        grid = {k: 0 for k in self._all_locs()}
        # check agents
        for v in self.agent_locations.values():
            grid[str(v[0])+","+str(v[1])] = 1
        # check foods
        for v in self.food_locations.values():
            grid[str(v[0])+","+str(v[1])] = 2
        observations = {k: copy(grid) for k in self.agent_locations}
        # position agents
        for k in self.agent_locations:
            observations[k]["x"] = self.agent_locations[k][0]
            observations[k]["y"] = self.agent_locations[k][1]

        return observations

    def reset(self):
        self.agents = copy(self.possible_agents)
        self.agent_locations = {k: [0, 0] for k in self.agents}
        self.food_locations = {k: [2, 2] for k in self.foods}
        # if self.render_mode == "human":
        #     self._render_frame()

        return self._get_obs()

    def isOnFood(self, aloc):
        for f in self.food_locations:
            if np.array_equal(aloc, self.food_locations[f]):
                return True
        return False

    def step(self, actions, render):
        directions = {k: self._action_to_direction[actions[k]] for k in actions}
        self.agent_locations = {k: np.clip(
            self.agent_locations[k] + directions[k], 0, self.size - 1) for k in directions}
        reaches = {k: self.isOnFood(self.agent_locations[k]) for k in directions}
        rewards = {k: 1 if reaches[k] else 0 for k in reaches}

        if self.render_mode == "human" and render:
            self._render_frame()

        return self._get_obs(), rewards

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the foods
        for f in self.food_locations.values():
            loc = [p * pix_square_size for p in f]
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    loc,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agents
        color = 255
        i = 0
        for a in self.agent_locations.values():
            colorb = 255 - 80*i
            colorr = 80*i
            i += 1
            loc = [(p + 0.5) * pix_square_size for p in a]
            pygame.draw.circle(
                canvas,
                (colorr, 0, colorb),
                loc,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
