import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque
from custom_env import CustomEnv

# Step 1: Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)


# Step 2: Create the environment
env = gym.make('CustomEnv-v0', num_agents=2)  # Set the number of agents

# Step 3: Define the neural network model for each agent
num_states = env.observation_space.shape[0] * env.observation_space.shape[1]  # Update num_states
num_actions = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


models = []
target_models = []
replay_buffers = []
for i in range(env.num_agents):
    # model = QNetwork(num_states, num_actions)
    # target_model = QNetwork(num_states, num_actions)
    # target_model.load_state_dict(model.state_dict())
    # target_model.eval()
    # replay_buffer = deque(maxlen=10000)

    model = QNetwork(num_states, num_actions)
    model.load_state_dict(torch.load("model"+str(i)+".pth"))
    model.eval()
    target_model = QNetwork(num_states, num_actions)
    target_model.load_state_dict(torch.load("target_model"+str(i)+".pth"))
    target_model.eval()
    replay_buffer = torch.load("replay_buffer"+str(i)+".pth")

    replay_buffers.append(replay_buffer)
    models.append(model)
    target_models.append(target_model)






optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
criterion = nn.MSELoss()

# Step 4: Define the Q-learning parameters
discount_factor = 0.99
num_episodes = 0
max_steps_per_episode = 100
exploration_rate = 0.3
max_exploration_rate = 1.0
min_exploration_rate = 0.5
exploration_decay_rate = 0.01
batch_size = 64

# Step 5: Implement the Q-learning algorithm using the neural network with experience replay
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_rewards = [0.0] * env.num_agents

    for step in range(max_steps_per_episode * env.num_agents):
        for i in range(env.num_agents):
            exploration_threshold = np.random.uniform(0, 1)
            if exploration_threshold > exploration_rate:
                with torch.no_grad():
                    action = torch.argmax(models[i](torch.cat(state))).item()
            else:
                action = env.action_space.sample()


            env.active_agent = i
            new_state, reward, _, _ = env.step(action)

            # Store the experience in the replay buffer
            replay_buffers[i].append((state, action, reward, new_state, done))

            state = new_state
            total_rewards[i] += reward

        if done:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Update the Q-networks using experience replay
    for i in range(env.num_agents):
        if len(replay_buffers[i]) >= batch_size:
            batch = random.sample(replay_buffers[i], batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat([torch.cat([state.flatten() for state in agent_states]) for agent_states in states]).view(batch_size, -1)
            next_states = torch.cat([torch.cat([state.flatten() for state in agent_states]) for agent_states in next_states]).view(batch_size, -1)
            rewards = torch.tensor(rewards)
            actions = torch.tensor(actions)
            dones = torch.tensor(dones)
            
            q_values = models[i](states)
            next_q_values = target_models[i](next_states).detach()

            target_values = rewards + discount_factor * torch.max(next_q_values, dim=1)[0] * (1 - dones.float())

            # Update the Q-values
            q_values[range(batch_size), actions] = target_values

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


# for i in range(env.num_agents):
#     torch.save(models[i].state_dict(), "model"+str(i)+".pth")
#     torch.save(target_models[i].state_dict(), "target_model"+str(i)+".pth")
#     torch.save(replay_buffers[i], "replay_buffer"+str(i)+".pth")

# After training, you can test the agents' performance
state = env.reset()
done = False
total_rewards = [0.0] * env.num_agents
env.render()
while not done:
    for i in range(env.num_agents):
        action = torch.argmax(models[i](torch.cat(state))).item()
        state, reward, done, _ = env.step(action)
        print("Agent: ", i, " Action: ", action, " Reward: ", reward)
        total_rewards[i] += reward
        env.render()
        time.sleep(0.1)

# Print the total rewards achieved in the test episode
print(f"Test Episode: Total Rewards = {total_rewards}")

