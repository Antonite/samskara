import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import resources.colors as color_palette 
import agent as ag


class CustomEnv(gym.Env):
    def __init__(self, num_agents=1, grid_size=3):
        super(CustomEnv, self).__init__()

        # Define the dimensions of the grid
        self.grid_size = grid_size

        # Number of agents per team
        self.num_agents = num_agents

        # Agents
        self.active_agent = 0 # index
        self.active_team = 0 # index
        self.agents = []
        self.grid = []
        
        # Render
        self.window_size = 768
        self.window = None
        self.clock = None
        self.last_action = 0

        # Define the possible actions (turn left, turn right, move, attack)
        self.action_space = spaces.Discrete(4)
        # Define the observation space (grid size * agent parameters + team's turn)
        self.state_length = self.grid_size*self.grid_size*ag.AGENT_FIELDS+1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_length,), dtype=np.float32)

        # Initialize the agent and reward positions
        self.reset()

    def step(self, action):
        reward = 0
        done = False
        agent = self.agents[self.active_team][self.active_agent]
        # Update rotation
        if action == 0:
            agent.rotation = ag.Rotation((agent.rotation.value - 1) % 4)
        elif action == 1:
            agent.rotation = ag.Rotation((agent.rotation.value + 1) % 4)
        elif action == 2:
            # Determine the maximum number of tiles the agent can move
            speed = ag.PROFESSIONS[agent.type].speed

            # Calculate the new position based on the agent's rotation
            new_row, new_col = agent.location
            if agent.rotation == ag.Rotation.Left:
                new_col -= speed
            elif agent.rotation == ag.Rotation.Up:
                new_row -= speed
            elif agent.rotation == ag.Rotation.Right:
                new_col += speed
            elif agent.rotation == ag.Rotation.Down:
                new_row += speed


            # Keep track of the last valid position
            last_valid_position = agent.location

            # Calculate the row and column differences
            row_diff = 0
            if new_row > agent.location[0]:
                row_diff = 1
            elif new_row < agent.location[0]:
                row_diff = -1
            
            col_diff = 0
            if new_col > agent.location[1]:
                col_diff = 1
            elif new_col < agent.location[1]:
                col_diff = -1

            # Move towards the new position until blocked or out of bounds
            for _ in range(speed):
                # Calculate the next position
                next_row = last_valid_position[0] + row_diff
                next_col = last_valid_position[1] + col_diff
 
                if 0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size:
                    if self.grid[next_row][next_col] == None:
                        last_valid_position = (next_row, next_col)
                    else:
                        # The path is blocked, stop moving
                        break
                else:
                    # The new position is out of bounds, stop moving
                    break

            # Update the agent's position to the last valid position
            old_loc = agent.location
            agent.location = last_valid_position
            if old_loc != agent.location:
                self.grid[agent.location[0]][agent.location[1]] = agent
                self.grid[old_loc[0]][old_loc[1]] = None
                reward += 0.2
            else:
                reward -= 0.2
        elif action == 3:
            reward -= 0.1
            if agent.type == ag.Type.Berserker:
                # Check all surrounding tiles for other agents
                surrounding_tiles = [
                    (agent.location[0] - 1, agent.location[1] - 1),
                    (agent.location[0] - 1, agent.location[1]),
                    (agent.location[0] - 1, agent.location[1] + 1),
                    (agent.location[0], agent.location[1] - 1),
                    (agent.location[0], agent.location[1] + 1),
                    (agent.location[0] + 1, agent.location[1] - 1),
                    (agent.location[0] + 1, agent.location[1]),
                    (agent.location[0] + 1, agent.location[1] + 1)
                ]
                # Iterate over the surrounding tiles and count the number of agents
                for tile in surrounding_tiles:
                    # Make sure this is a valid tile
                    if 0 <= tile[0] < self.grid_size and 0 <= tile[1] < self.grid_size:
                        # Check if an agent is present on this tile
                        a = self.grid[tile[0]][tile[1]]
                        if a != None:
                            if a.team != self.active_team:
                                a.health -= ag.PROFESSIONS[agent.type].power
                                # float inaccuracies
                                if a.health <= 0.001:
                                    reward += 0.5
                                    # Kill the agent
                                    self.grid[tile[0]][tile[1]] = None
                                    # remove from agent list
                                    for deadi in range(len(self.agents[a.team])):
                                        if self.agents[a.team][deadi].location == tile:
                                            del self.agents[a.team][deadi]
                                            if len(self.agents[a.team]) == 0:
                                                done = True
                                            break
                                else:
                                    reward += 0.1
                            
            else:
                new_row, new_col = agent.location
                if agent.rotation == ag.Rotation.Left:
                    new_col -= 1
                elif agent.rotation == ag.Rotation.Up:
                    new_row -= 1
                elif agent.rotation == ag.Rotation.Right:
                    new_col += 1
                elif agent.rotation == ag.Rotation.Down:
                    new_row += 1
                
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    # Check if an agent is present on this tile
                    a = self.grid[new_row][new_col]
                    if a != None:
                        if a.team != self.active_team:
                            a.health -= ag.PROFESSIONS[agent.type].power
                            if a.health <= 0.001:
                                reward += 0.5
                                # Kill the agent
                                self.grid[new_row][new_col] = None
                                # remove from agent list
                                for deadi in range(len(self.agents[a.team])):
                                    if self.agents[a.team][deadi].location == (new_row, new_col):
                                        del self.agents[a.team][deadi]
                                        if len(self.agents[a.team]) == 0:
                                            done = True
                                        break
                            else:
                                reward += 0.1
            
                    
        return self.get_state(), reward, done, False, {}


    def reset(self, seed: int | None = None, options: dict[str, object()] | None = None):
        super().reset()
        if options != None and options["fair"]:
            return self.fair_reset()

        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.agents = []

        valid_positions = ([(i, j) for i in range(self.grid_size) for j in range(self.grid_size)])
        agent_positions = random.sample(valid_positions, k=self.num_agents*2)
        l = len(agent_positions)
        for team in range(2):
            agents = []
            for agent in range(int(team*l/2), int(l/2+team*l/2)):
                agent_type = random.choice([ag.Type.Runner, ag.Type.Berserker])
                a = ag.Agent(agent_positions[agent], ag.Rotation.Down if team == 0 else ag.Rotation.Up, agent_type, team, self.grid_size)
                agents.append(a)
                self.grid[agent_positions[agent][0]][agent_positions[agent][1]] = a
            self.agents.append(agents)              

        return self.get_state(), {}
    
    def fair_reset(self):
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.agents = []

        agent_type = random.choice([ag.Type.Runner, ag.Type.Berserker])
        # Yellow team
        yellow_agents = []
        n = 0
        for row in range(self.grid_size // 2):
            for col in range(self.grid_size):
                if n >= self.num_agents:
                    break
                a = ag.Agent((row, col), ag.Rotation.Down, agent_type, 0, self.grid_size)
                self.grid[row][col] = a  # Place the agent in the grid
                yellow_agents.append(a)  # Add the agent to the list of agents
                n += 1
        # Purple team
        purple_agents = []
        n = 0
        for row in reversed(range(self.grid_size // 2, self.grid_size)):
            for col in reversed(range(self.grid_size)):
                if n >= self.num_agents:
                    break
                a = ag.Agent((row, col), ag.Rotation.Up, agent_type, 1, self.grid_size)
                self.grid[row][col] = a  # Place the agent in the grid
                purple_agents.append(a)  # Add the agent to the list of agents
                n += 1

        self.agents.append(yellow_agents)
        self.agents.append(purple_agents)

        return self.get_state(), {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("CustomEnv")
            self.clock = pygame.time.Clock()

        self.window.fill(color_palette.background)

        cell_size = self.window_size // self.grid_size

        # Draw grid lines
        for x in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (90, 90, 90), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, cell_size):
            pygame.draw.line(self.window, (90, 90, 90), (0, y), (self.window_size, y))

        # Draw agent positions
        for team in range(2):
            for i in range(len(self.agents[team])):
                agent = self.agents[team][i]
                center = ((agent.location[1]+0.5)*cell_size,(agent.location[0]+0.5)*cell_size)
                if agent.rotation == ag.Rotation.Up:
                    end_point = (center[0], center[1] - (cell_size / 3))
                elif agent.rotation == ag.Rotation.Down:
                    end_point = (center[0], center[1] + (cell_size / 3))
                elif agent.rotation == ag.Rotation.Left:
                    end_point = (center[0] - (cell_size / 3), center[1])
                else:
                    end_point = (center[0] + (cell_size / 3), center[1])
                
                if i == self.active_agent and team == self.active_team and self.last_action == 3:
                    pygame.draw.rect(self.window, (200, 17, 17), (agent.location[1] * cell_size, agent.location[0] * cell_size, cell_size, cell_size))
                pygame.draw.circle(self.window, color_palette.team_colors[team], center, cell_size / 3)
                pygame.draw.line(self.window, color_palette.background, center, end_point, 5)
                pygame.draw.line(self.window, color_palette.health, (agent.location[1]*cell_size+cell_size/8,agent.location[0]*cell_size+cell_size/12), (agent.location[1]*cell_size+cell_size/8+cell_size/8*6*agent.health/ag.MAX_HEALTH,agent.location[0]*cell_size+cell_size/12), round(cell_size/14))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()

    def get_state(self):
        state = np.zeros((self.state_length,), dtype=np.float32)
        # Agent positions
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row*self.grid_size+column)*ag.AGENT_FIELDS
                if self.grid[row][column] == None:
                    state[pos], state[pos+1], state[pos+2], state[pos+3], state[pos+4], state[pos+5], state[pos+6] = 0,0,0,0,0,0,0
                else:
                    state[pos], state[pos+1], state[pos+2], state[pos+3], state[pos+4], state[pos+5] = self.grid[row][column].normalize()
                    state[pos+6] = float(self.agents[self.active_team][self.active_agent].location == (row, column))

        # Team's turn
        state[self.grid_size*self.grid_size*ag.AGENT_FIELDS] = self.active_team

        return state
    
    def set_last_action(self, action):
        self.last_action = action

    def set_active(self, agent, team):
        self.active_team = team
        self.active_agent = agent

    def team_len(self, team):
        return len(self.agents[team])

# Register the environment with Gym
gym.envs.register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv',
)