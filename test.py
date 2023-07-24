import gymnasium as gym
import torch
import pygame
import torch.nn as nn
from samskara import Samskara

training_dir = "training/"

NUM_AGENTS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
env = gym.make('Samskara-v0', num_agents=NUM_AGENTS)  # Set the number of agents

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

class Actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        hidden_size = round(num_states * 2/3 + num_actions)
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x),  dim=-1)
        return x

actor_network = Actor(num_states, num_actions)
actor_network.load_state_dict(torch.load(f"{training_dir}actor_network.pth"))
# actor_network.eval()

# After training, you can test the agents' performance
while True:
    state, _ = env.reset(options={"fair": True})
    # state, _ = env.reset()
    env.render()
    done = False
    while not done:
        for team in range(2):
            actions = []
            for agent in range(env.team_len(team)):
                env.set_active(agent, team)
                actor_state = env.get_state()

                # Select an action
                action_probs = actor_network(torch.tensor(actor_state))
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

                sampled_action = action + 1
                actions.append(sampled_action)

            # Take a step in the environment with all actions
            _, reward, done, _, _ = env.step(actions)
            env.set_last_actions(actions)

            # print(f"agent: {agent} team: {team} action: {action} reward: {reward}")
            pygame.event.pump()
            env.render()

            if done:
                break
    