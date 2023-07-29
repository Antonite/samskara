import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import resources.colors as color_palette 
import agent as ag
import hexcell as hexcell
import math as math

class Samskara(gym.Env):
    def __init__(self, num_agents=1):
        super(Samskara, self).__init__()

        # Define the size of the hex grid
        self.num_cells_row = 9  # Maximum width of the hexagonal grid

        # rewards
        self.REWARD_FOR_WIN = 1
        self.REWARD_FOR_KILL = 0.2
        self.REWARD_FOR_DAMAGE = 0.05

        # Define the dimensions of the field
        self.num_cells = 61

        # Number of agents per team
        self.num_agents = num_agents

        # Agents
        self.active_agent = 0 # index
        self.active_team = 0 # index
        self.agents = []
        
        # Render
        self.window_size = 1024
        self.window = None
        self.clock = None
        self.last_action = []

        # Define the possible actions
        # ---- MOVE ----
        # LEFT = 1
        # TOP_LEFT = 2
        # TOP_RIGHT = 3
        # RIGHT = 4
        # BOTTOM_RIGHT = 5
        # BOTTOM_LEFT = 6
        # ---- ATTACK ----
        # LEFT = 7
        # TOP_LEFT = 8
        # TOP_RIGHT = 9
        # RIGHT = 10
        # BOTTOM_RIGHT = 11
        # BOTTOM_LEFT = 12
        self.num_actions = 12
        self.action_space = spaces.Discrete(self.num_actions)

        # Define the observation space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(ag.AGENT_FIELDS, self.num_cells_row, self.num_cells_row), dtype=np.float32)
        # self.critic_observation_space = spaces.Box(low=0.0, high=1.0, shape=(ag.AGENT_FIELDS + 1, self.num_cells_row, self.num_cells_row), dtype=np.float32)

        self.empty_actor_state = np.zeros((ag.AGENT_FIELDS, self.num_cells_row, self.num_cells_row), dtype=np.float32)

    def invert_action(self,action):
        if action in [1, 2, 3]:
            return action + 3
        elif action in [4, 5, 6]:
            return action - 3
        elif action in [7, 8, 9]:
            return action + 3
        elif action in [10, 11, 12]:
            return action - 3

    # used to render
    # def set_last_actions(self, actions):
    #     self.last_actions = actions
    def set_last_action(self, action):
        self.last_action = action

    # used to figure out who to calculate steps for
    def set_active(self, agent, team):
        self.active_team = team
        self.active_agent = agent

    # used by learning algorithm
    def team_len(self, team):
        return len(self.agents[team])
    
    def active_agent_id(self,team,agent):
        return self.agents[team][agent].id

    def get_coordinates(self, cell_id):
        # Each row has a different number of cells, so we calculate the y-coordinate (row) first
        y = 0
        while cell_id >= 9 - abs(4 - y):
            cell_id -= 9 - abs(4 - y)
            y += 1

        # The remaining cell_id value is the x-coordinate (column)
        x = cell_id

        return x, y

    def get_state(self):
        # Initialize a 3D array with zeros. 
        # The first dimension is the number of features per agent (health, power, speed, etc...)
        # The second and third dimensions are the width and height of the grid.
        # Since the play grid is a hexagon, the invalid cells will always remain zeros.
        state = np.zeros((ag.AGENT_FIELDS, self.num_cells_row, self.num_cells_row), dtype=np.float32)

        # set active agent
        self.agents[self.active_team][self.active_agent].is_active = True

        for team in range(2):
            for agent in self.agents[team]:
                # Flip the state for the purple team
                id = agent.cell_id if self.active_team == 0 else self.num_cells - 1 - agent.cell_id
                # We use the get_coordinates function to get the x, y coordinates of the agent
                x, y = self.get_coordinates(id)

                # Each feature has its own layer in the first dimension
                state[0:ag.AGENT_FIELDS, x, y] = agent.normalize(self.active_team)

        # unset active agent
        self.agents[self.active_team][self.active_agent].is_active = False

        # The state does not need to be flattened since we will be using convolutional layers
        return state
    

    def get_critic_state(self, actions):
        # Initialize a 3D array with zeros. 
        # The first dimension is the number of features per agent (health, power, speed, etc...).
        # The final feature of the agent is the action taken this turn.
        # The second and third dimensions are the width and height of the grid.
        # Since the play grid is a hexagon, the invalid cells will always remain zeros.
        state = np.zeros((ag.AGENT_FIELDS + 1, self.num_cells_row, self.num_cells_row), dtype=np.float32)

        # set active agent
        self.agents[self.active_team][self.active_agent].is_active = True

        for team in range(2):
            for agent_i in range(len(self.agents[team])):
                agent = self.agents[team][agent_i]
                # Flip the state for the purple team
                id = agent.cell_id if self.active_team == 0 else self.num_cells - 1 - agent.cell_id
                # We use the get_coordinates function to get the x, y coordinates of the agent
                x, y = self.get_coordinates(id)

                # Each feature has its own layer in the first dimension
                state[0, x, y],state[1, x, y],state[2, x, y],state[3, x, y],state[4, x, y],state[5, x, y],state[6, x, y] = agent.normalize(self.active_team)
                if team == self.active_team:
                    state[7, x, y] = actions[agent_i] / self.num_actions # normalized action of each agent, ranges from 1/12 to 12/12
                else:
                    state[7, x, y] = 0 # enemy team didn't take any actions

        # unset active agent
        self.agents[self.active_team][self.active_agent].is_active = False

        # The state does not need to be flattened since we will be using convolutional layers
        return state
            

    def reset(self, seed: int | None = None, options: dict[str, object()] | None = None):
        super().reset()
        if options != None and options["fair"]:
            return self.fair_reset()

        self.grid = hexcell.HexGrid()
        self.agents = []

        agent_type = ag.Type.FIGHTER
        nextCell = self.grid.map[random.randrange(self.num_cells)]
        for team in range(2):
            new_team = []
            for i in range(self.num_agents):
                a = ag.Agent(team*self.num_agents+i,nextCell.id,agent_type,team,ag.PROFESSIONS[agent_type].health,ag.PROFESSIONS[agent_type].power,ag.PROFESSIONS[agent_type].speed,ag.PROFESSIONS[agent_type].range)
                nextCell.data = a
                new_team.append(a)
                # new random cell
                rand_cell_id = random.randrange(self.num_cells)
                while self.grid.map[rand_cell_id].data != None:
                    rand_cell_id = random.randrange(self.num_cells)
                nextCell = self.grid.map[rand_cell_id]

            self.agents.append(new_team)

        return self.get_state(), {}

    def fair_reset(self):
        self.grid = hexcell.HexGrid()
        self.agents = []

        # agent_type = random.choice([ag.Type.Runner, ag.Type.Berserker])
        agent_type = ag.Type.FIGHTER
        # nextCells = [self.grid.map[5],self.grid.map[55]]
        nextCells = [self.grid.map[0],self.grid.map[60]]
        for team in range(2):
            new_team = []
            direction = hexcell.Direction.RIGHT if team == 0 else hexcell.Direction.LEFT
            for i in range(self.num_agents):
                a = ag.Agent(team*self.num_agents+i,nextCells[team].id,agent_type,team,ag.PROFESSIONS[agent_type].health,ag.PROFESSIONS[agent_type].power,ag.PROFESSIONS[agent_type].speed,ag.PROFESSIONS[agent_type].range)
                nextCells[team].data = a
                new_team.append(a)
                nextCells[team] = nextCells[team].neighbors[direction]
            self.agents.append(new_team)

        return self.get_state(), {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size - 60))
            pygame.display.set_caption("samskara")
            self.clock = pygame.time.Clock()

        self.window.fill(color_palette.background)
        font = pygame.font.Font(None, 21)

        hexagon_width_pixels = self.window_size / 9
        # r = (2 ** 0.5) * math.cos(30)
        cell_radius = hexagon_width_pixels / ((2 ** 0.5) + math.cos(30)) - 5

        # Find attacked cell id
        attacked_id = None
        agent_cell = self.grid.map[self.agents[self.active_team][self.active_agent].cell_id]
        act = self.last_action
        # Flip action for team 1 since the state is flipped for team 1
        if self.active_team == 1:
            act = self.invert_action(act)
        match act:
            # LEFT
            case 7:
                if agent_cell.neighbors[hexcell.Direction.LEFT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.LEFT].id
            # TOP_LEFT
            case 8:
                if agent_cell.neighbors[hexcell.Direction.TOP_LEFT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.TOP_LEFT].id
            # TOP_RIGHT
            case 9:
                if agent_cell.neighbors[hexcell.Direction.TOP_RIGHT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.TOP_RIGHT].id
            # RIGHT
            case 10:
                if agent_cell.neighbors[hexcell.Direction.RIGHT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.RIGHT].id
            # BOTTOM_RIGHT
            case 11:
                if agent_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT].id
            # BOTTOM_LEFT
            case 12:
                if agent_cell.neighbors[hexcell.Direction.BOTTOM_LEFT] != None:
                    attacked_id = agent_cell.neighbors[hexcell.Direction.BOTTOM_LEFT].id
        
        # Draw agent positions
        for i in range(self.num_cells):
            next_cell = self.grid.map[i]
            render_offset = 0
            row = 0
            # first row
            if 5 > next_cell.id:
                render_offset = hexagon_width_pixels*2 + hexagon_width_pixels*i
            # second row
            elif 11 > next_cell.id:
                render_offset = hexagon_width_pixels*1.5 + hexagon_width_pixels*(i-5)
                row = 1
            # third row
            elif 18 > next_cell.id:
                render_offset = hexagon_width_pixels + hexagon_width_pixels*(i-11)
                row = 2
            # fourth row
            elif 26 > next_cell.id:
                render_offset = hexagon_width_pixels*0.5 + hexagon_width_pixels*(i-18)
                row = 3
            # fifth row
            elif 35 > next_cell.id:
                render_offset = hexagon_width_pixels*(i-26)
                row = 4
            # sixth row
            elif 43 > next_cell.id:
                render_offset = hexagon_width_pixels*0.5 + hexagon_width_pixels*(i-35)
                row = 5
            # seventh row
            elif 50 > next_cell.id:
                render_offset = hexagon_width_pixels + hexagon_width_pixels*(i-43)
                row = 6
            # eigth row
            elif 56 > next_cell.id:
                render_offset = hexagon_width_pixels*1.5 + hexagon_width_pixels*(i-50)
                row = 7
            # ninth row
            elif 61 > next_cell.id:
                render_offset = hexagon_width_pixels*2 + hexagon_width_pixels*(i-56)
                row = 8


            # Draw the hexagon
            center_x = render_offset + hexagon_width_pixels / 2
            center_y = hexagon_width_pixels * row + hexagon_width_pixels / 2 - row*11 + 10
            # Calculate the vertices of the hexagon
            vertices = []
            for anl in range(6):
                angle_rad = (60 * anl + 30) * (3.14159 / 180)
                vertex_x = center_x + cell_radius * math.cos(angle_rad)
                vertex_y = center_y + cell_radius * math.sin(angle_rad)
                vertices.append((vertex_x, vertex_y))
            

            if attacked_id == next_cell.id:
                pygame.draw.polygon(self.window, color_palette.damaged, vertices)
            else:
                pygame.draw.polygon(self.window, color_palette.cell_borders, vertices, 3)

            # draw agent
            if next_cell.data != None:
                pygame.draw.circle(self.window, color_palette.team_colors[next_cell.data.team], (center_x,center_y), cell_radius / 2)
                # agent's health
                text_surface = font.render(str(next_cell.data.health), True, color_palette.background, color_palette.team_colors[next_cell.data.team])
                text_rect = text_surface.get_rect()
                text_rect.center = (center_x,center_y)
                self.window.blit(text_surface, text_rect)

            
            
        pygame.display.flip()
        self.clock.tick(60)

    def step(self, action):
        reward = 0
        done = False
        agent = self.agents[self.active_team][self.active_agent]
        current_cell = self.grid.map[agent.cell_id]
        next_cell = None
        act = action
        # Flip action for team 1 since the state is flipped for team 1
        if self.active_team == 1:
            act = self.invert_action(act)
        match act:
            # LEFT
            case 1 | 7:
                next_cell = current_cell.neighbors[hexcell.Direction.LEFT]
            # TOP_LEFT
            case 2 | 8:
                next_cell = current_cell.neighbors[hexcell.Direction.TOP_LEFT]
            # TOP_RIGHT
            case 3 | 9:
                next_cell = current_cell.neighbors[hexcell.Direction.TOP_RIGHT]
            # RIGHT
            case 4 | 10:
                next_cell = current_cell.neighbors[hexcell.Direction.RIGHT]
            # BOTTOM_RIGHT
            case 5 | 11:
                next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT]
            # BOTTOM_LEFT
            case 6 | 12:
                next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_LEFT]

        # Valid action
        if next_cell != None:
            # ---- MOVE ----
            if act < 7 and next_cell.data == None:
                next_cell.data = agent
                agent.cell_id = next_cell.id
                current_cell.data = None
            # ---- ATTACK ----
            elif act >= 7 and next_cell.data != None and next_cell.data.team != current_cell.data.team:
                    next_cell.data.health -= agent.power
                    reward = self.REWARD_FOR_DAMAGE
                    # dead
                    if next_cell.data.health <= 0.001:
                        # remove from agent list
                        for deadi in range(len(self.agents[next_cell.data.team])):
                            if self.agents[next_cell.data.team][deadi].cell_id == next_cell.id:
                                del self.agents[next_cell.data.team][deadi]
                                if len(self.agents[next_cell.data.team]) == 0:
                                    done = True
                                    reward = self.REWARD_FOR_WIN
                                else:
                                    reward = self.REWARD_FOR_KILL
                                break
                        next_cell.data = None
                    
        return self.empty_actor_state, reward, done, False, {}
    
    
    
    def compute_winner(self):
        alive = [0,0]
        hp = [0,0]

        for team in range(2):
            for agent in self.agents[team]:
                alive[team] += 1
                hp[team] += agent.health
        
        if alive[0] > alive[1]:
            return 0
        elif alive[1] > alive[0]:
            return 1
        
        if hp[0] > hp[1]:
            return 0
        elif hp[1] > hp[0]:
            return 1
        
        return -1


# Register the environment with Gym
gym.envs.register(
    id='Samskara-v0',
    entry_point='samskara:Samskara',
)