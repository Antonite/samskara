from copy import copy
import random
import numpy as np
import pygame
from gym import spaces
import torch

from resources.colors import color_pallete


EMPTY_TILE = 0
AGENT_TILE = 1
FOOD_TILE = 2


ACTION_DOWN = 0
ACTION_UP = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_STAY = 4
ACTION_GATHER = 5

MOVE_ACTIONS = [ACTION_DOWN, ACTION_UP, ACTION_LEFT, ACTION_RIGHT]
ALL_ACTIONS = [ACTION_DOWN, ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_GATHER]

_action_to_direction = {
    ACTION_DOWN: (np.array([0, 1]), "Down"),
    ACTION_UP: (np.array([0, -1]), "Up"),
    ACTION_LEFT: (np.array([-1, 0]), "Left"),
    ACTION_RIGHT: (np.array([1, 0]), "Right"),
    ACTION_STAY: (np.array([0, 0]), "Stay"),
    ACTION_GATHER: (np.array([0, 0]), "Gather")
}


class SequentialEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size, render_mode=None):
        # move to GPU if cuda available
        self.size = size
        self.window_size = 768  # The size of the PyGame window

        self.reward_scaler = 1

        self.agents = None
        self.foods = None
        self.world = np.zeros((size, size))

        self.possible_foods = ["food_1"]
        self.possible_agents = ["agent_1", "agent_2"]

        self.agent_locations = {}
        self.food_locations = {}

        # Up / Down / Left / Right / Stay
        self.action_space = spaces.Discrete(len(ALL_ACTIONS))
        # every location
        self.inputs = len(self.world.flatten()) + 2

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def state(self, agent):
        return np.append(self.world.flatten(), self.agent_locations[agent])

    def _get_empty_location(self):
        # first try a random location a few times
        for i in range(5):
            r = random.randrange(0, self.size - 1)
            c = random.randrange(0, self.size - 1)
            if self.world[r][c] == 0:
                return [r, c]

        # find first empty slot
        for ir, r in enumerate(self.world):
            for ic, c in enumerate(r):
                if c == 0:
                    return [ir, ic]
        return None

    def action_mask(self, agent):
        mask = np.ones(self.action_space.n)
        mask[ACTION_GATHER] = 0
        nearbyFood = []
        for act in MOVE_ACTIONS:
            newl = np.clip(self.agent_locations[agent] + _action_to_direction[act][0], 0, self.size - 1)
            # move outside boundry
            if np.array_equal(newl, self.agent_locations[agent]):
                mask[act] = 0
            # that tile isn't empty
            elif self.world[newl[0]][newl[1]] != EMPTY_TILE:
                mask[act] = 0
                # tile has food
                if self.world[newl[0]][newl[1]] == FOOD_TILE:
                    mask[ACTION_GATHER] = 1
                    nearbyFood.append(newl)
        return mask

    def _nearby_food(self, agent):
        nearbyFood = []
        for act in MOVE_ACTIONS:
            newl = np.clip(self.agent_locations[agent] + _action_to_direction[act][0], 0, self.size - 1)
            if self.world[newl[0]][newl[1]] == FOOD_TILE:
                nearbyFood.append(newl)
        eaten = []
        for floc in nearbyFood:
            for f in self.food_locations:
                if np.array_equal(floc, self.food_locations[f]):
                    eaten.append(f)

        return eaten

    def reset(self):
        self.agents = self.possible_agents
        self.foods = self.possible_foods
        # place foods
        for food in self.foods:
            empty = self._get_empty_location()
            self.food_locations[food] = empty
            self.world[empty[0]][empty[1]] = FOOD_TILE
        # place agents
        for agent in self.agents:
            empty = self._get_empty_location()
            self.agent_locations[agent] = empty
            self.world[empty[0]][empty[1]] = AGENT_TILE

    def step(self, agent, action, should_render):
        reward = 0
        if action == ACTION_GATHER:
            eaten = self._nearby_food(agent)
            reward = len(eaten) * self.reward_scaler
            for f in eaten:
                newl = self._get_empty_location()
                self.world[self.food_locations[f][0]][self.food_locations[f][1]] = EMPTY_TILE
                self.food_locations[f] = newl
                self.world[newl[0]][newl[1]] = FOOD_TILE
        else:
            self.world[self.agent_locations[agent][0]][self.agent_locations[agent][1]] = EMPTY_TILE
            self.agent_locations[agent] = self.agent_locations[agent] + _action_to_direction[action][0]
            self.world[self.agent_locations[agent][0]][self.agent_locations[agent][1]] = AGENT_TILE

        if self.render_mode == "human" and should_render:
            self._render_frame()

        return self.state(agent), reward

    @staticmethod
    def get_human_readable_action(action):
        return _action_to_direction[action][1]

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
