import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import resources.colors as color_palette 
import agent as ag
import hexcell as hexcell

class CustomEnv(gym.Env):
    def __init__(self, num_agents=1):
        super(CustomEnv, self).__init__()

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
        self.last_action = 0

        # Define the possible actions
        # ---- MOVE ----
        # LEFT = 0
        # TOP_LEFT = 1
        # TOP_RIGHT = 2
        # RIGHT = 3
        # BOTTOM_RIGHT = 4
        # BOTTOM_LEFT = 5
        # ---- ATTACK ----
        # LEFT = 6
        # TOP_LEFT = 7
        # TOP_RIGHT = 8
        # RIGHT = 9
        # BOTTOM_RIGHT = 10
        # BOTTOM_LEFT = 11
        self.action_space = spaces.Discrete(12)
        # Define the observation space (grid size * agent parameters + team's turn)
        self.total_agent_fields = self.num_cells*ag.AGENT_FIELDS
        self.state_length = self.total_agent_fields+1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_length,), dtype=np.float32)

        # Initialize the agent and reward positions
        self.reset()

    # used to render
    def set_last_action(self, action):
        self.last_action = action

    # used to figure out who to calculate steps for
    def set_active(self, agent, team):
        self.active_team = team
        self.active_agent = agent

    # used by learning algorithm
    def team_len(self, team):
        return len(self.agents[team])

    def get_state(self):
        state = np.zeros((self.state_length,), dtype=np.float32)
        # Agent positions 
        for team in range(2):
            for agent in self.agents[team]:
                pos = agent.id*ag.AGENT_FIELDS
                state[pos], state[pos+1], state[pos+2], state[pos+3], state[pos+4], state[pos+5], state[pos+6] = agent.normalize()

        # Team's turn
        state[self.total_agent_fields] = self.active_team

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
            for _ in range(self.num_agents):
                a = ag.Agent(nextCell.id,agent_type,team,ag.PROFESSIONS[agent_type].health,ag.PROFESSIONS[agent_type].power,ag.PROFESSIONS[agent_type].speed,ag.PROFESSIONS[agent_type].range)
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
        nextCells = [self.grid.map[0],self.grid.map[60]]
        for team in range(2):
            new_team = []
            direction = hexcell.Direction.RIGHT if team == 0 else hexcell.Direction.LEFT
            for _ in range(self.num_agents):
                a = ag.Agent(nextCells[team].id,agent_type,team,ag.PROFESSIONS[agent_type].health,ag.PROFESSIONS[agent_type].power,ag.PROFESSIONS[agent_type].speed,ag.PROFESSIONS[agent_type].range)
                nextCells[team].data = a
                new_team.append(a)
                nextCells[team] = nextCells[team].neighbors[direction]
            self.agents.append(new_team)

        return self.get_state(), {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("CustomEnv")
            self.clock = pygame.time.Clock()

        self.window.fill(color_palette.background)

        cell_radius = self.window_size // 18
        
        # Draw agent positions
        for i in range(self.num_cells):
            next_cell = self.grid.map[i]
            render_offset = 0
            row = 0
            # first row
            if 5 > next_cell.id:
                render_offset = cell_radius*2
            # second row
            elif 11 > next_cell.id:
                render_offset = cell_radius*1.5
                row = 1
            # third row
            elif 18 > next_cell.id:
                render_offset = cell_radius
                row = 2
            # fourth row
            elif 26 > next_cell.id:
                render_offset = cell_radius*0.5
                row = 3
            # fifth row
            elif 35 > next_cell.id:
                render_offset = 0
                row = 4
            # sixth row
            elif 43 > next_cell.id:
                render_offset = cell_radius*0.5
                row = 5
            # seventh row
            elif 50 > next_cell.id:
                render_offset = cell_radius
                row = 6
            # eigth row
            elif 56 > next_cell.id:
                render_offset = cell_radius*1.5
                row = 7
            # ninth row
            elif 61 > next_cell.id:
                render_offset = cell_radius*2
                row = 8


            # Draw the hexagon
            center_x = render_offset + cell_radius / 2
            center_y = cell_radius * row + cell_radius / 2
            vertices = [
                (center_x + cell_radius * pygame.math.cos(angle), center_y + cell_radius * pygame.math.sin(angle))
                for angle in [0, 60, 120, 180, 240, 300]
            ]
            pygame.draw.polygon(self.window, (90,90,90), vertices, 3)

            # draw agent
            if next_cell.data != None:
                pygame.draw.circle(self.window, color_palette.team_colors[next_cell.data.team], (center_x,center_y), cell_radius / 2)
                # pygame.draw.line(self.window, color_palette.health, (agent.location[1]*cell_size+cell_size/8,agent.location[0]*cell_size+cell_size/12), (agent.location[1]*cell_size+cell_size/8+cell_size/8*6*agent.health/ag.MAX_HEALTH,agent.location[0]*cell_size+cell_size/12), round(cell_size/14))

            # if i == self.active_agent and team == self.active_team and self.last_action == 3:
            #     pygame.draw.rect(self.window, (200, 17, 17), (agent.location[1] * cell_size, agent.location[0] * cell_size, cell_size, cell_size))
            # pygame.draw.circle(self.window, color_palette.team_colors[team], center, cell_size / 3)
            # pygame.draw.line(self.window, color_palette.background, center, end_point, 5)
            # pygame.draw.line(self.window, color_palette.health, (agent.location[1]*cell_size+cell_size/8,agent.location[0]*cell_size+cell_size/12), (agent.location[1]*cell_size+cell_size/8+cell_size/8*6*agent.health/ag.MAX_HEALTH,agent.location[0]*cell_size+cell_size/12), round(cell_size/14))

        pygame.display.flip()
        self.clock.tick(10)

    def step(self, action):
        reward = 0
        done = False
        agent = self.agents[self.active_team][self.active_agent]
        current_cell = self.grid.map[agent.id]
        next_cell = None
        match action:
            # LEFT
            case 0 | 6:
                next_cell = current_cell.neighbors[hexcell.Direction.LEFT]
            # TOP_LEFT
            case 1 | 7:
                next_cell = current_cell.neighbors[hexcell.Direction.TOP_LEFT]
            # TOP_RIGHT
            case 2 | 8:
                next_cell = current_cell.neighbors[hexcell.Direction.TOP_RIGHT]
            # RIGHT
            case 3 | 9:
                next_cell = current_cell.neighbors[hexcell.Direction.RIGHT]
            # BOTTOM_RIGHT
            case 4 | 10:
                next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_RIGHT]
            # BOTTOM_LEFT
            case 5 | 11:
                next_cell = current_cell.neighbors[hexcell.Direction.BOTTOM_LEFT]

        # Invalid action
        if next_cell == None or next_cell.data != None:
            reward -= 1
        else:
            # ---- MOVE ----
            if action < 6:
                reward += 0.05
                next_cell.data = agent
                agent.id = next_cell.id
                current_cell.data = None
            # ---- ATTACK ----
            else:
                next_cell.data.health -= agent.power
                # dead
                if next_cell.data.health <= 0.001:
                    reward += 1
                    # remove from agent list
                    for deadi in range(len(self.agents[next_cell.data.team])):
                        if self.agents[next_cell.data.team][deadi].id == next_cell.id:
                            del self.agents[next_cell.data.team][deadi]
                            if len(self.agents[next_cell.data.team]) == 0:
                                done = True
                            break
                    next_cell.data = None
            
                    
        return self.get_state(), reward, done, False, {}

# Register the environment with Gym
gym.envs.register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv',
)