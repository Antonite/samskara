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
env = gym.make('CustomEnv-v0')

# Step 3: Define the neural network model
num_states = env.observation_space.shape[0] * env.observation_space.shape[1]
num_actions = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = x.view(-1, num_states)  # Reshape the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = QNetwork(num_states, num_actions).to(device)
# model.load_state_dict(torch.load("model.pth"))
# model.eval()

target_model = QNetwork(num_states, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Step 4: Define the Q-learning parameters
discount_factor = 0.99
num_episodes = 10000
max_steps_per_episode = 100
exploration_rate = 0.3
max_exploration_rate = 1.0
min_exploration_rate = 0.5
exploration_decay_rate = 0.01
batch_size = 32
replay_buffer = deque(maxlen=10000)  # Replay buffer capacity
# replay_buffer = torch.load("replay_buffer.pth")

# Step 5: Implement the Q-learning algorithm using the neural network with experience replay
for episode in range(num_episodes):
    # print(exploration_rate)
    state = env.reset()
    state = torch.Tensor(state).to(device)
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        exploration_threshold = np.random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        new_state = torch.Tensor(new_state).to(device)

        # Store the experience in the replay buffer
        replay_buffer.append((state, action, reward, new_state, done))

        state = new_state
        total_reward += reward

        if done:
            break

    # exploration_rate = min_exploration_rate + \
    #                    (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Update the Q-network using experience replay
    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).to(device)

        q_values = model(states)
        next_q_values = target_model(next_states).detach()

        target_values = rewards + discount_factor * torch.max(next_q_values, dim=1)[0] * (1 - dones.float())

        # Update the Q-values
        q_values[range(batch_size), actions] = target_values

        # Compute the loss and optimize the model
        optimizer.zero_grad()
        loss = criterion(q_values, model(states))
        loss.backward()
        optimizer.step()

        # Update the target network periodically
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

    # Print the episode number and total reward
    print(f"Episode {episode+1}: Total Reward = {total_reward}")


torch.save(model.state_dict(), "model.pth")
torch.save(replay_buffer, "replay_buffer.pth")

# After training, you can test the agent's performance
state = env.reset()
state = torch.Tensor(state).to(device)
done = False
total_reward = 0
env.render()
while not done:
    action = torch.argmax(model(state)).item()
    state, reward, done, _ = env.step(action)
    state = torch.Tensor(state).to(device)
    total_reward += reward
    env.render()

# Print the total reward achieved in the test episode
print(f"Test Episode: Total Reward = {total_reward}")