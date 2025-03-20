import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hexcell, agent
import random
import pygame
import math
import resources.colors as color_palette

class SamskaraEnv(gym.Env):
    def __init__(self, num_fighters_per_team=3):
        super().__init__()
        self.num_fighters = num_fighters_per_team
        self.grid = hexcell.HexGrid()
        self.action_space = spaces.Discrete(13 ** self.num_fighters)  # 0: noop, 1-6: move, 7-12: attack
        self.observation_space = spaces.Box(low=0, high=1, shape=(agent.AGENT_FIELDS, 9, 9), dtype=np.float32)
        self.teams = [[], []]
        
        # render
        self.window_size = 1024
        self.window = None
        self.clock = None
        self.last_action = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = hexcell.HexGrid()
        self.teams = [[], []]
        positions = [0, 60]
        dirs = [hexcell.Direction.RIGHT, hexcell.Direction.LEFT]
        for t in range(2):
            cell = self.grid.map[positions[t]]
            for i in range(self.num_fighters):
                fighter = agent.Agent(i, cell.id, agent.Type.FIGHTER, t, 100, 20, 1, 1)
                cell.data = fighter
                self.teams[t].append(fighter)
                cell = cell.neighbors[dirs[t]]
        return self.get_obs(), {}

    def get_obs(self):
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        for t, team in enumerate(self.teams):
            for f in team:
                x, y = divmod(f.cell_id, 9)
                state[:, x, y] = f.normalize(active_team=0)
        return state

    def step(self, action):
        actions = []
        for _ in range(self.num_fighters):
            actions.append(action % 13)
            action //= 13
        actions = [actions, actions]  # same actions for both teams
        rewards = [0, 0]
        dones = [False, False]

        for team_idx, team_actions in enumerate(actions):
            enemy_team = (team_idx + 1) % 2
            for f_idx, action in enumerate(team_actions):
                if f_idx >= len(self.teams[team_idx]):
                    continue  # fighter is dead, skip
                fighter = self.teams[team_idx][f_idx]
                cell = self.grid.map[fighter.cell_id]
                target_cell = None
                if action == 0:
                    continue  # noop
                elif 1 <= action <= 6:
                    direction = hexcell.Direction(action - 1)
                    target_cell = cell.neighbors.get(direction)
                    if target_cell and target_cell.data is None:
                        cell.data, target_cell.data = None, fighter
                        fighter.cell_id = target_cell.id
                elif 7 <= action <= 12:
                    direction = hexcell.Direction(action - 7)
                    target_cell = cell.neighbors.get(direction)
                    if target_cell and target_cell.data and target_cell.data.team == enemy_team:
                        target = target_cell.data
                        target.health -= fighter.power
                        rewards[team_idx] += fighter.power / 100
                        if target.health <= 0:
                            target_cell.data = None
                            self.teams[enemy_team].remove(target)
                            rewards[team_idx] += 0.5
                            if not self.teams[enemy_team]:
                                rewards[team_idx] += 2
                                dones = [True, True]
        return self.get_obs(), rewards[0]-rewards[1], all(dones), False, {}

    def get_coordinates(self, cell_id):
        # Each row has a different number of cells, so we calculate the y-coordinate (row) first
        y = 0
        while cell_id >= 9 - abs(4 - y):
            cell_id -= 9 - abs(4 - y)
            y += 1

        # The remaining cell_id value is the x-coordinate (column)
        x = cell_id

        return x, y

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Samskara")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 21)

        self.window.fill(color_palette.background)

        # For a grid of size=5, we have 9 rows total (2*5 - 1=9).
        # The widest row has 9 hexes, each hex is width = sqrt(3)*R for pointy‑top.
        # So 9*sqrt(3)*R = window_size => R=window_size/(9*sqrt(3)).
        row_count = 2 * self.grid.size - 1  # 9 when size=5
        R = self.window_size / (9 * math.sqrt(3))
        
        # For pointy‑top:
        #   - "width" (corner‑to‑corner horizontally) = sqrt(3)*R
        #   - "height" (corner‑to‑corner vertically) = 2*R
        #   - row spacing = 1.5 * R (centers are 1.5*R apart vertically)
        hex_width = math.sqrt(3) * R
        hex_height = 2.0 * R
        vertical_spacing = 1.5 * R

        # Row lengths for a diamond shape of side=5
        row_lengths = [
            self.grid.size + min(i, (self.grid.size - 1)*2 - i)
            for i in range(row_count)
        ]
        # The longest row has 9 cells => total width = 9*hex_width = window_size

        # Calculate the total height of the diamond:
        # Between consecutive rows, vertical center is 1.5*R apart.
        # For row_count=9, the top row is row=0, bottom row=8 => 8 * (1.5*R)=12*R between top & bottom centers.
        # The hex extends R above the top row center and R below the bottom row center => +2*R
        # => total_grid_height = 12*R + 2*R = 14*R
        total_grid_height = (row_count - 1) * vertical_spacing + hex_height

        # Center everything vertically
        # We'll set row=0 center at y_offset + 0*(1.5*R) = y_offset
        # such that the bottom fits within the window
        y_offset = (self.window_size - total_grid_height) / 2 + R

        cell_idx = 0
        for row, row_length in enumerate(row_lengths):
            # The widest row has 9 cells => total_row_width = 9*hex_width = 1024 => center horizontally
            total_row_width = row_length * hex_width
            x_offset = (self.window_size - total_row_width) / 2 + (hex_width / 2)

            # The row's vertical center
            center_y = y_offset + row * vertical_spacing

            for col in range(row_length):
                center_x = x_offset + col * hex_width

                # Build the polygon (pointy-top: 30° offset means the top vertex points up)
                vertices = []
                for anl in range(6):
                    angle_rad = math.radians(60 * anl + 30)
                    vx = center_x + R * math.cos(angle_rad)
                    vy = center_y + R * math.sin(angle_rad)
                    vertices.append((vx, vy))

                cell = self.grid.map[cell_idx]
                pygame.draw.polygon(self.window, color_palette.cell_borders, vertices, 3)

                if cell.data:
                    agent_color = color_palette.team_colors[cell.data.team]
                    pygame.draw.circle(self.window, agent_color, (center_x, center_y), R / 2)
                    text_surface = self.font.render(str(int(cell.data.health)), True,
                                                    color_palette.background, agent_color)
                    text_rect = text_surface.get_rect(center=(center_x, center_y))
                    self.window.blit(text_surface, text_rect)

                cell_idx += 1

        pygame.display.flip()
        self.clock.tick(10)


