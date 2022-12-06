from copy import copy
import random
import numpy as np
import pygame
from gym import spaces

from resources.colors import color_pallete

_action_to_direction = {
    0: (np.array([0, 0]), "Stay"),
    1: (np.array([1, 0]), "Right"),
    2: (np.array([0, 1]), "Down"),
    3: (np.array([-1, 0]), "Left"),
    4: (np.array([0, -1]), "Up")
}


class TestEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.window_size = 768  # The size of the PyGame window

        self.agents = None
        self.agent_locations = None
        self.food_locations = None

        self.foods = ["food_1"]
        self.possible_agents = ["agent_1", "agent_2"]

        # Up / Down / Left / Right / Stay
        self.action_space = spaces.Discrete(5)

        # every location
        self.observation_space = {k: spaces.Discrete(3) for k in self._all_locs()}
        # agents current location on the board
        self.observation_space["x"] = spaces.Discrete(1)
        self.observation_space["y"] = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _all_locs(self):
        # every location
        locKeys = []
        for i in range(self.size):
            for j in range(self.size):
                locKeys.append(','.join([str(i), str(j)]))
        return locKeys

    """ Returns the 'observable state of the board via a map of (X,Y) coordinates to the following
                (1): Location is occupied
                (2): Location occupied by Reward
        as well as the current location of the agent
    """

    def _get_obs(self):
        # K: (X,Y) - V: 0 All board locations
        grid = {k: 0 for k in self._all_locs()}
        # Encode agent locations
        for v in self.agent_locations.values():
            grid[str(v[0]) + "," + str(v[1])] = 1
        # Encode food locations
        for v in self.food_locations.values():
            grid[str(v[0]) + "," + str(v[1])] = 2
        observations = {k: copy(grid) for k in self.agent_locations}
        # Encode agents current position
        for k in self.agent_locations:
            observations[k]["x"] = self.agent_locations[k][0]
            observations[k]["y"] = self.agent_locations[k][1]

        return observations

    def reset(self):
        self.agents = self.possible_agents  # TODO {gufforda} - I dont think we need this
        self.agent_locations = {k: [0, 0] for k in self.agents}
        self.food_locations = {k: [random.randrange(
            0, self.size - 1), random.randrange(0, self.size - 1)] for k in self.foods}

        return self._get_obs()

    @staticmethod
    def get_human_readable_action(action):
        return _action_to_direction[action][1]

    def isOnFood(self, aloc):
        for f in self.food_locations:
            return (True, f) if np.array_equal(aloc, self.food_locations[f]) else (False, None)

    def step(self, actions, should_render):
        directions = {k: _action_to_direction[actions[k]][0] for k in actions}
        self.agent_locations = {k: np.clip(
            self.agent_locations[k] + directions[k], 0, self.size - 1) for k in directions}

        reaches = {}
        eaten = []
        for a in directions:
            reached, food = self.isOnFood(self.agent_locations[a])
            reaches[a] = reached
            if reached:
                eaten.append(food)

        rewards = {k: self.size ** 2 if reaches[k] else 0 for k in reaches}

        # move food if eaten
        if len(eaten) > 0:
            for f in eaten:
                self.food_locations[f] = [random.randrange(0, self.size - 1), random.randrange(0, self.size - 1)]

        if self.render_mode == "human" and should_render:
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
        canvas.fill((14, 17, 17))
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

        for i, a in enumerate(self.agent_locations.values()):
            colorOffset = i % len(color_pallete)
            color_tuple = color_pallete[colorOffset]
            colorr = color_tuple[0]
            colorg = color_tuple[1]
            colorb = color_tuple[2]

            loc = [(p + 0.5) * pix_square_size for p in a]
            pygame.draw.circle(
                canvas,
                (colorr, colorg, colorb),
                loc,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
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
