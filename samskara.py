import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hexcell, agent
import random
import pygame
import math
import resources.colors as color_palette
from sb3_contrib import MaskablePPO

class SamskaraEnv(gym.Env):
    def __init__(self, num_fighters_per_team=3):
        super().__init__()
        self.num_fighters = num_fighters_per_team
        self.grid = hexcell.HexGrid()
        self.action_space = spaces.MultiDiscrete([13] * self.num_fighters)  # 0: noop, 1-6: move, 7-12: attack
        self.observation_space = spaces.Box(low=0, high=1, shape=(agent.AGENT_FIELDS, 9, 9), dtype=np.float32)
        self.teams = [[], []]

        # We'll store the average distance from each team's fighters to nearest enemies
        self.prev_distances = [0.0, 0.0]
        
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

        # Optional: random side-flipping to ensure the policy learns both perspectives
        flip = random.random() < 0.5
        if not flip:
            starts = [(0, hexcell.Direction.RIGHT), (60, hexcell.Direction.LEFT)]
        else:
            starts = [(60, hexcell.Direction.LEFT), (0, hexcell.Direction.RIGHT)]

        for t in range(2):
            cell_start, direction = starts[t]
            cell = self.grid.map[cell_start]
            for i in range(self.num_fighters):
                fighter = agent.Agent(
                    id=i,
                    cell_id=cell.id,
                    type=agent.Type.FIGHTER,
                    team=t,
                    health=100,
                    power=20,
                    range=1
                )
                cell.data = fighter
                self.teams[t].append(fighter)
                cell = cell.neighbors[direction]

        # NEW: Initialize prev_distances for each team
        self.prev_distances[0] = self._compute_team_distance(0)
        self.prev_distances[1] = self._compute_team_distance(1)

        return self.get_obs(), {}

    def get_obs(self):
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        for t, team in enumerate(self.teams):
            for f in team:
                x, y = divmod(f.cell_id, 9)
                state[:, x, y] = f.normalize(active_team=0)
        return state

    def step(self, action):
        """
        Single-policy self-play example: same action array for both teams.
        If you do true alternating turns or separate policies, adapt accordingly.
        """
        team_actions = [action, action]
        rewards = [0, 0]
        dones = [False, False]

        # *** Apply actions and compute partial rewards ***
        for team_idx, fighter_actions in enumerate(team_actions):
            enemy_team = (team_idx + 1) % 2
            for f_idx, act in enumerate(fighter_actions):
                if f_idx >= len(self.teams[team_idx]):
                    continue
                fighter = self.teams[team_idx][f_idx]
                if fighter.health <= 0:
                    continue

                cell = self.grid.map[fighter.cell_id]

                # 0 = noop
                if act == 0:
                    continue

                # 1..6 => move
                elif 1 <= act <= 6:
                    direction = hexcell.Direction(act - 1)
                    neighbor = cell.neighbors.get(direction)
                    if neighbor and neighbor.data is None:
                        # Move
                        cell.data = None
                        neighbor.data = fighter
                        fighter.cell_id = neighbor.id

                # 7..12 => attack
                elif 7 <= act <= 12:
                    direction_idx = act - 7
                    direction = hexcell.Direction(direction_idx)
                    current_cell = cell
                    for _ in range(fighter.range):
                        next_cell = current_cell.neighbors.get(direction)
                        if not next_cell:
                            break
                        if next_cell.data:
                            # If it's an enemy, damage them
                            if next_cell.data.team == enemy_team:
                                target = next_cell.data
                                target.health -= fighter.power

                                # PARTIAL REWARD FOR DAMAGE
                                rewards[team_idx] += fighter.power / 100.0

                                # If kill
                                if target.health <= 0:
                                    next_cell.data = None
                                    self.teams[enemy_team].remove(target)

                                    # BUMP KILL REWARD
                                    rewards[team_idx] += 2.0  # was 0.5

                                    # If entire enemy team is wiped
                                    if not self.teams[enemy_team]:
                                        # BUMP WIN REWARD
                                        rewards[team_idx] += 5.0  # was 2
                                        dones = [True, True]
                                # Attack stops after hitting someone
                                break
                        current_cell = next_cell

        # *** Distance-based shaping reward ***
        # After all moves/attacks, compute new distances, see if we moved closer
        for t in [0, 1]:
            new_dist = self._compute_team_distance(t)
            dist_diff = self.prev_distances[t] - new_dist  # positive if we got closer
            # Scale the distance difference, e.g. 0.01 per hex step closer
            rewards[t] += dist_diff * 0.01
            self.prev_distances[t] = new_dist

        # Single scalar reward in SB3 => difference
        reward = rewards[0] - rewards[1]

        return self.get_obs(), reward, all(dones), False, {}
    
    def action_masks(self):
        """
        Return a 1D boolean array of length (13 * num_fighters).
        Indexing: 
        - fighter 0's valid actions in the first 13 slots, 
        - fighter 1 in the next 13, etc.
        """
        raw_mask_2d = self._compute_action_masks(team_idx=0)  # shape (num_fighters, 13)
        # Flatten to shape (13*num_fighters,)
        return raw_mask_2d.flatten()

    def _compute_action_masks(self, team_idx=0):
        """
        Return a (num_fighters, 13) boolean array indicating which actions are valid.
        In this example, we do it for team 0's fighters only (if you're truly self-playing
        with separate steps, you'd do it for whichever team is about to move).
        If you're forcing the same actions on both teams each step, you can just do team_idx=0.
        """
        masks = np.ones((self.num_fighters, 13), dtype=bool)

        # For each fighter on the chosen team
        for f_idx, fighter in enumerate(self.teams[team_idx]):
            if fighter.health <= 0:
                # All actions for a dead fighter are effectively no-op,
                # you could mask everything except 0=noop, or let them do nothing anyway.
                masks[f_idx, 1:] = False
                continue

            cell = self.grid.map[fighter.cell_id]

            # 1..6 => move
            for move_action in range(1, 7):
                direction = hexcell.Direction(move_action - 1)
                neighbor = cell.neighbors.get(direction)
                # invalid if no neighbor or neighbor occupied
                if (neighbor is None) or (neighbor.data is not None):
                    masks[f_idx, move_action] = False

            # 7..12 => attack
            for attack_action in range(7, 13):
                direction_idx = attack_action - 7
                direction = hexcell.Direction(direction_idx)
                # Check if there's any enemy within 'fighter.range' steps in that direction
                # If none, mask it out
                if not self._can_attack_in_direction(fighter, direction):
                    masks[f_idx, attack_action] = False

        return masks

    def _can_attack_in_direction(self, fighter, direction):
        """Return True if there's at least one enemy cell within fighter.range steps in 'direction'."""
        if fighter.health <= 0:
            return False

        start_cell = self.grid.map[fighter.cell_id]
        enemy_team = 1 if fighter.team == 0 else 0

        current = start_cell
        for _ in range(fighter.range):
            nxt = current.neighbors.get(direction)
            if not nxt:
                return False  # end of board
            if nxt.data:  # found a fighter
                return (nxt.data.team == enemy_team)
            current = nxt

        return False
    
      # *** BFS-based distance function for each team's fighters. ***
    # You can adapt to sum or average distance. We'll do average distance to
    # nearest enemy for the entire team.
    def _compute_team_distance(self, team_idx):
        """
        Return the average distance from each of team_idx's fighters
        to the nearest enemy fighter.
        If a fighter is dead, ignore them (or treat distance=0, up to you).
        """
        enemy_idx = 1 - team_idx
        enemies = self.teams[enemy_idx]
        team = self.teams[team_idx]

        if len(enemies) == 0 or len(team) == 0:
            return 0.0  # if no enemies or no fighters, distance is somewhat moot

        total_dist = 0.0
        live_fighters = 0

        for fighter in team:
            if fighter.health <= 0:
                continue
            live_fighters += 1

            dist = self._bfs_nearest_enemy_dist(fighter.cell_id, enemies)
            total_dist += dist

        if live_fighters == 0:
            return 0.0
        return total_dist / live_fighters

    def _bfs_nearest_enemy_dist(self, start_cell_id, enemy_list):
        """
        BFS outward from the start cell until we find an enemy fighter.
        Return the distance in hex steps. If not found, return some large number.
        """
        visited = set()
        queue = [(start_cell_id, 0)]  # (cell_id, distance)
        enemies_cell_ids = {e.cell_id for e in enemy_list}

        while queue:
            cell_id, dist = queue.pop(0)
            if cell_id in enemies_cell_ids:
                return dist
            visited.add(cell_id)

            neighbors = self.grid.map[cell_id].neighbors
            for ncell in neighbors.values():
                if ncell and ncell.id not in visited:
                    queue.append((ncell.id, dist + 1))

        # No enemy found (should be rare if there's at least one alive)
        return 999.0


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


