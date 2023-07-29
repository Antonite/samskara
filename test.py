import gymnasium as gym
import torch
import pygame
import torch.nn as nn
from samskara import Samskara
import warnings

# Ignore unrelated warnings
warnings.filterwarnings('ignore', '.*Box observation space is an image.*', category=UserWarning, module='gym')

training_dir = "training/"

NUM_AGENTS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

env = gym.make('Samskara-v0', num_agents=NUM_AGENTS)  # Set the number of agents
num_input_channels_actor = env.observation_space.shape[0]
num_actions = env.action_space.n

class Actor(nn.Module):
    def __init__(self, num_input_channels, num_actions):
        super(Actor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # The dimensions of the flattened output depend on the size of the 2D grid
        flattened_size = 64 * env.num_cells_row * env.num_cells_row
        self.fc1 = nn.Linear(flattened_size, flattened_size)
        self.fc2 = nn.Linear(flattened_size, num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Create actor and critic networks
actor_network = Actor(num_input_channels_actor, num_actions)
actor_network.load_state_dict(torch.load(f"{training_dir}actor_network.pth"))
# actor_network.eval()

# After training, you can test the agents' performance
while True:
    env.reset(options={"fair": True})
    # state, _ = env.reset()
    env.render()
    done = False
    while not done:
        for team in range(1):
            for agent in range(env.team_len(team)):
                env.set_active(agent, team)
                actor_state = env.get_state()

                # Select an action
                action_probs = actor_network(torch.tensor(actor_state).unsqueeze(0))
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

                sampled_action = action + 1

                # Take a step in the environment with all actions
                _, reward, done, _, _ = env.step(sampled_action)
                env.set_last_action(sampled_action)

                # print(f"agent: {agent} team: {team} action: {action} reward: {reward}")
                pygame.event.pump()
                env.render()
            
                if done:
                    break
            if done:
                break
