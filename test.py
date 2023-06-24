import gymnasium as gym
import torch
import pygame
import torch.nn as nn
from custom_env import CustomEnv

training_dir = "training/"

# Step 1: Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)
# Step 2: Create the environment
env = gym.make('CustomEnv-v0', num_agents=1, grid_size=5)  # Set the number of agents
# Step 3: Define the neural network model for each agent
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        hidden_size = round(num_states * 2/3 + num_actions)
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

king_model = QNetwork(num_states, num_actions)
king_model.load_state_dict(torch.load(f"{training_dir}agent_model.pth"))
king_model.eval()



# After training, you can test the agents' performance
done = False
while not done:
    state, _ = env.reset(options={"fair": True})
    # state, _ = env.reset()
    total_rewards = [0.0] * 2
    env.render()
    # 100 steps at a time
    for i in range(100):
        for team in range(2):
            for agent in range(env.team_len(team)):
                env.set_active(agent,team)
                v = king_model(torch.tensor(state))
                action = torch.argmax(king_model(torch.tensor(state))).item()
                state, reward, _, _, _ = env.step(action)
                env.set_last_action(action)
                # print(f"agent: {agent} team: {team} action: {action} reward: {reward}")
                total_rewards[team] += reward
                pygame.event.pump()
                env.render()

    # Print the total rewards achieved in the test episode
    print(f"Test Episode: Total Rewards = {total_rewards}")
    