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

        # rewards
        self.REWARD_FOR_WIN = 1
        self.REWARD_FOR_KILL = 0.2

        # Define the dimensions of the field
        self.num_cells = 61

        # Number of agents per team
        self.num_agents = num_agents

        # Agents
        self.active_agent = 0 # index
        self.active_team = 0 # index
        self.agents = []
        
        # Render
        self.window_size = 768
        self.window = None
        self.clock = None
        self.last_actions = []

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
        # Define the observation space (grid size * agent parameters)
        self.total_actor_agent_fields = self.num_cells*ag.AGENT_FIELDS
        self.total_critic_agent_fields = self.num_cells*(ag.AGENT_FIELDS + 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.total_actor_agent_fields,), dtype=np.float32)
        self.critic_observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.total_critic_agent_fields,), dtype=np.float32)

        self.empty_actor_state = np.zeros((self.total_actor_agent_fields,), dtype=np.float32)

        # Initialize the agent and reward positions
        self.reset()

    # used to render
    def set_last_actions(self, actions):
        self.last_actions = actions

    # used to figure out who to calculate steps for
    def set_active(self, agent, team):
        self.active_team = team
        self.active_agent = agent

    # used by learning algorithm
    def team_len(self, team):
        return len(self.agents[team])
    
    def active_agent_id(self,team,agent):
        return self.agents[team][agent].id

    def get_state(self):
        state = np.zeros((self.total_actor_agent_fields,), dtype=np.float32)
        # set active agent
        self.agents[self.active_team][self.active_agent].is_active = True

        # Agent positions 
        for team in range(2):
            for agent in self.agents[team]:
                pos = agent.cell_id*ag.AGENT_FIELDS
                state[pos], state[pos+1], state[pos+2], state[pos+3], state[pos+4], state[pos+5], state[pos+6] = agent.normalize(self.active_team)

        # unset active agent
        self.agents[self.active_team][self.active_agent].is_active = False

        return state

    def get_critic_state(self, actions):
        state = np.zeros((self.total_critic_agent_fields,), dtype=np.float32)

        # Agent positions 
        for team in range(2):
            for agent in range(len(self.agents[team])):
                pos = self.agents[team][agent].cell_id*(ag.AGENT_FIELDS+1) # +1 for the extra field for the action taken
                state[pos], state[pos+1], state[pos+2], state[pos+3], state[pos+4], state[pos+5], state[pos+6] = self.agents[team][agent].normalize(self.active_team)
                if team == self.active_team:
                    state[pos+7] = actions[agent] / self.num_actions # normalized action of each agent, ranges from 1/12 to 12/12
                else:
                    state[pos+7] = 0 # enemy team didn't take any actions

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

        # Get attacked id for each agent
        attacked_ids = []
        if len(self.last_actions) > 0:
            for agent_i in range(len(self.agents[self.active_team])): 
                agent_cell = self.grid.map[self.agents[self.active_team][agent_i].cell_id]
                match self.last_actions[agent_i]:
                    # LEFT
                    case 7:
                        if agent_cell.neighbors[hexcell.Direction.LEFT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.LEFT].id)
                    # TOP_LEFT
                    case 8:
                        if agent_cell.neighbors[hexcell.Direction.TOP_LEFT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.TOP_LEFT].id)
                    # TOP_RIGHT
                    case 9:
                        if agent_cell.neighbors[hexcell.Direction.TOP_RIGHT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.TOP_RIGHT].id)
                    # RIGHT
                    case 10:
                        if agent_cell.neighbors[hexcell.Direction.RIGHT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.RIGHT].id)
                    # BOTTOM_RIGHT
                    case 11:
                        if agent_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT].id)
                    # BOTTOM_LEFT
                    case 12:
                        if agent_cell.neighbors[hexcell.Direction.BOTTOM_LEFT] != None:
                            attacked_ids.append(agent_cell.neighbors[hexcell.Direction.BOTTOM_LEFT].id)
        
        
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
            
            attacked = False
            for attacked_id in attacked_ids:
                if attacked_id == i:
                    attacked = True

            if attacked:
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
        self.clock.tick(20)

    def step(self, actions):
        reward = 0
        done = False
        next_positions = [None for _ in range(len(actions))]
        # Find next locations in case agent is moving
        for agent_i in range(len(self.agents[self.active_team])):
            current_cell = self.grid.map[self.agents[self.active_team][agent_i].cell_id]
            next_cell = None
            match actions[agent_i]:
                # LEFT
                case 1:
                    next_cell = current_cell.neighbors[hexcell.Direction.LEFT]
                # TOP_LEFT
                case 2:
                    next_cell = current_cell.neighbors[hexcell.Direction.TOP_LEFT]
                # TOP_RIGHT
                case 3:
                    next_cell = current_cell.neighbors[hexcell.Direction.TOP_RIGHT]
                # RIGHT
                case 4:
                    next_cell = current_cell.neighbors[hexcell.Direction.RIGHT]
                # BOTTOM_RIGHT
                case 5:
                    next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT]
                # BOTTOM_LEFT
                case 6:
                    next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_LEFT]
            # Valid and empty cell?
            if next_cell != None and next_cell.data == None:
                next_positions[agent_i] = next_cell

        # Check for collisions
        for pos in range(len(next_positions)):
            if next_positions[pos] != None:
                # Check all other positions for collision
                collision = False
                for other in range(pos+1,len(next_positions),1):
                    # Collision
                    if next_positions[other] != None and next_positions[other].id == next_positions[pos].id:
                        collision = True
                        next_positions[other] = None
                if collision:
                    next_positions[pos] = None

                    
        for agent_i in range(len(self.agents[self.active_team])):
            agent = self.agents[self.active_team][agent_i]
            current_cell = self.grid.map[agent.cell_id]
            next_cell = None
            match actions[agent_i]:
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
                if actions[agent_i] < 7 and next_positions[agent_i] != None:
                    next_cell.data = agent
                    agent.cell_id = next_cell.id
                    current_cell.data = None
                # ---- ATTACK ----
                elif actions[agent_i] >= 7 and next_cell.data != None and next_cell.data.team != current_cell.data.team:
                        next_cell.data.health -= agent.power
                        # dead
                        if next_cell.data.health <= 0.001:
                            # remove from agent list
                            for deadi in range(len(self.agents[next_cell.data.team])):
                                if self.agents[next_cell.data.team][deadi].cell_id == next_cell.id:
                                    del self.agents[next_cell.data.team][deadi]
                                    if len(self.agents[next_cell.data.team]) == 0:
                                        done = True
                                        reward = self.REWARD_FOR_WIN
                                        return self.empty_actor_state, reward, done, False, {} 
                                    reward += self.REWARD_FOR_KILL
                                    break
                            next_cell.data = None
            
                    
        return self.empty_actor_state, reward, done, False, {}

# Register the environment with Gym
gym.envs.register(
    id='Samskara-v0',
    entry_point='samskara:Samskara',
)