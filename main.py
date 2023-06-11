import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from custom_env import CustomEnv

# Step 1: Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Create the environment
env = gym.make('CustomEnv-v0', num_agents=2)  # Set the number of agents

# Step 3: Define the neural network model for each agent
num_states = env.observation_space.shape[0] * env.observation_space.shape[1]  # Update num_states
num_actions = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, num_actions).to(device)

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


models = []
target_models= []
for i in range(env.num_agents):
    model = QNetwork(num_states, num_actions)
    target_model = QNetwork(num_states, num_actions)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    models.append(model)
    target_models.append(target_model)


optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
criterion = nn.MSELoss()

# Step 4: Define the Q-learning parameters
discount_factor = 0.99
num_episodes = 10000
max_steps_per_episode = 100
exploration_rate = 0.3
max_exploration_rate = 1.0
min_exploration_rate = 0.5
exploration_decay_rate = 0.01
batch_size = 64
replay_buffer = deque(maxlen=10000)  # Replay buffer capacity

# Step 5: Implement the Q-learning algorithm using the neural network with experience replay
for episode in range(num_episodes):
    states = env.reset()
    states = [torch.Tensor(state) for state in states]
    done = False
    total_rewards = [0.0] * env.num_agents

    for step in range(max_steps_per_episode):
        actions = []
        for i in range(env.num_agents):
            exploration_threshold = np.random.uniform(0, 1)
            if exploration_threshold > exploration_rate:
                with torch.no_grad():
                    action = torch.argmax(models[i](torch.cat(states))).item()
            else:
                action = env.action_space.sample()
            actions.append(action)


        new_states, rewards, done, _ = env.step(actions)
        new_states = [torch.Tensor(state) for state in new_states]

        # Store the experience in the replay buffer
        replay_buffer.append((states, actions, rewards, new_states, done))

        states = new_states
        total_rewards = [total_rewards[i] + rewards[i] for i in range(env.num_agents)]

        if done:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Update the Q-networks using experience replay
    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat([torch.cat([state.flatten() for state in agent_states]) for agent_states in states]).view(batch_size, -1)
        next_states = torch.cat([torch.cat([state.flatten() for state in agent_states]) for agent_states in next_states]).view(batch_size, -1)
        dones = torch.tensor(dones).to(device)
        
        for i in range(env.num_agents):
            agent_rewards = torch.tensor([reward[i] for reward in rewards]).to(device)
            agent_actions = torch.tensor([action[i] for action in actions]).to(device)

            q_values = models[i](states)
            next_q_values = target_models[i](next_states).detach()

            target_values = agent_rewards + discount_factor * torch.max(next_q_values, dim=1)[0] * (1 - dones.float())

            # Update the Q-values
            q_values[range(batch_size), agent_actions] = target_values

            # Compute the loss and optimize the model
            optimizers[i].zero_grad()
            loss = criterion(q_values, models[i](states))
            loss.backward()
            optimizers[i].step()

            # Update the target networks periodically
            if episode % 10 == 0:
                target_models[i].load_state_dict(models[i].state_dict())

    # Print the episode number and total rewards
    print(f"Episode {episode + 1}: Total Rewards = {total_rewards}")


torch.save(replay_buffer, "replay_buffer.pth")
for i in range(env.num_agents):
    torch.save(models[i].state_dict(), "model"+str(i)+".pth")
    torch.save(target_models[i].state_dict(), "target_model"+str(i)+".pth")

# After training, you can test the agents' performance
states = env.reset()
states = [torch.Tensor(state) for state in states]
done = False
total_rewards = [0.0] * env.num_agents
env.render()
while not done:
    actions = [torch.argmax(models[i](torch.cat(states))).item() for i in range(env.num_agents)]
    states, rewards, done, _ = env.step(actions)
    states = [torch.Tensor(state) for state in states]
    total_rewards = [total_rewards[i] + rewards[i] for i in range(env.num_agents)]
    env.render()

# Print the total rewards achieved in the test episode
print(f"Test Episode: Total Rewards = {total_rewards}")

