import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from custom_env import CustomEnv
import time
from collections import deque
from datetime import datetime
import copy

training_dir = "training/"
kings_dir = "training/kings/"

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
        self.fc1 = nn.Linear(num_states, num_actions)
        self.fc2 = nn.Linear(num_actions, num_actions)
        self.fc3 = nn.Linear(num_actions, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Step 4: Define the Q-learning parameters
epochs = 1000
num_episodes = 10000

discount_factor = 0.99
max_steps_per_episode = 100000
exploration_rate = 0.5
batch_size = 1000
replay_start_threshold = 50000

agent_replay_buffers = []

king_model = QNetwork(num_states, num_actions)

king_model.load_state_dict(torch.load(f"{training_dir}agent_model.pth"))

agent_models = [copy.deepcopy(king_model) for _ in range(2)]
agent_target_models = []
for i in range(2):
    model = QNetwork(num_states, num_actions)
    model.load_state_dict(agent_models[i].state_dict())
    model.eval()
    agent_target_models.append(model)
agent_replay_buffers = [deque(maxlen=replay_start_threshold) for _ in range(2)]

optimizers = [optim.Adam(agent_models[i].parameters(), lr=0.005) for i in range(2)]
criterions = [nn.MSELoss(),nn.MSELoss()]

start_time = time.time()
total_rewards = [0.0] * 2
total_steps = 0
update = 0
# Step 5: Implement the Q-learning algorithm using the neural network with experience replay
for epoch in range(epochs):
    for episode in range(num_episodes):
        # state, _ = env.reset(options={"fair": True})
        state, _ = env.reset()
        episode_buffer = [[] for _ in range(2)]  # Buffer to store experiences in the episode
        winning_team = 0
        for step in range(max_steps_per_episode):
            # for each team
            for team in range(2):
                # for each agent
                for agent in range(env.team_len(team)):
                    # exploration_threshold = 1
                    exploration_threshold = np.random.uniform(0, 1)
                    if exploration_threshold > exploration_rate:
                        with torch.no_grad():
                            # vv = agent_models[team](torch.tensor(state))
                            action = torch.argmax(agent_models[team](torch.tensor(state))).item()
                    else:
                        action = env.action_space.sample()

                    env.set_active(agent, team)
                    new_state, reward, done, _, _ = env.step(action)

                    # Store the experience in the episode buffer
                    if reward != 0:
                        total_rewards[team] += reward
                        episode_buffer[team].append((state, action, reward, new_state, done))

                    state = new_state
                    
                    if done:
                        winning_team = team
                        break
                if done:
                    break
            if done:
                break
        
        total_steps += step+1

        # Assign retroactive rewards at the end of the episode
        if done:
            for team in range(2):
                if len(episode_buffer[team]) > 0:
                    # m = max(50 / len(episode_buffer[team]), 0.1)
                    # bonus = 0.1 if team == winning_team else -0.1
                    # modified_transitions = [(s, a, r + bonus, n, d) for s, a, r, n, d in episode_buffer[team]]
                    agent_replay_buffers[team].extend(episode_buffer[team])
            
        # Update the Q-networks using experience replay
        for team in range(2):
            if len(agent_replay_buffers[team]) >= replay_start_threshold:
                # Get a random batch from the buffer
                batch = random.choices(agent_replay_buffers[team], k=batch_size)
                # Remove the batch from the buffer
                for _ in range(batch_size):
                    agent_replay_buffers[team].popleft()

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states))
                next_states = torch.tensor(np.array(next_states))
                rewards = torch.tensor(rewards)
                actions = torch.tensor(actions)
                dones = torch.tensor(dones)
                
                q_values = agent_models[team](states)
                next_q_values = agent_target_models[team](next_states).detach()

                target_values = rewards + discount_factor * torch.max(next_q_values, dim=1)[0] * (1 - dones.float())

                # Update the Q-values
                q_values[range(batch_size), actions] = target_values

                # Compute the loss and optimize the model
                optimizers[team].zero_grad()
                loss = criterions[team](q_values, agent_models[team](states))
                loss.backward()
                optimizers[team].step()

                # Update the target networks periodically
                if update == 10:
                    agent_target_models[team].load_state_dict(agent_models[team].state_dict())
                    update = 0
                update += 1

    # Print the episode number and total rewards
    elapsed_time = time.time() - start_time
    total_rewards = [r / num_episodes for r in total_rewards]
    total_steps = round(total_steps / num_episodes)
    print(f"Epoch {epoch}: Total Average Rewards = {total_rewards} Average steps = {total_steps} Elapsed Time = {round(elapsed_time)} seconds")
    start_time = time.time()
    total_rewards = [0.0] * 2
    total_steps = 0

    # Fight
    max_steps = 200
    max_matches = 100
    new_king = False
    # yellow vs king
    won = [0,0]
    fight_models = [copy.deepcopy(agent_models[0]),copy.deepcopy(king_model)]
    fight_models = [model.eval() for model in fight_models]
    # 100 matches
    for _ in range(max_matches):
        state, _ = env.reset(options={"fair": True})
        # cap at 200 steps
        for _ in range(max_steps):
            for team in range(2):
                for agent in range(env.team_len(team)):
                    env.set_active(agent,team)
                    action = torch.argmax(fight_models[team](torch.tensor(state))).item()
                    state, reward, done, _, _ = env.step(action)
                    env.set_last_action(action)
                    if done:
                        won[team] += 1
                        break
                if done:
                    break
            if done:
                break
    if won[0] >= won[1]:
        print(f"Yellow beat The King {won[0]} - {won[1]}")
        winner = "Yellow"
        new_king = True
    else:
        print(f"King beat Yellow {won[1]} - {won[0]}")
        winner = "The King"
    
    # Winner vs purple
    # set winner
    fight_models = [copy.deepcopy(agent_models[1])]
    if won[0] >= won[1]:
        fight_models.append(copy.deepcopy(agent_models[0]))
    else:
        fight_models.append(copy.deepcopy(king_model))
    fight_models = [model.eval() for model in fight_models]
    won = [0,0]
    for _ in range(max_matches):
        state, _ = env.reset(options={"fair": True})
        # cap at 200 steps
        for _ in range(max_steps):
            for team in range(2):
                for agent in range(env.team_len(team)):
                    env.set_active(agent,team)
                    action = torch.argmax(fight_models[team](torch.tensor(state))).item()
                    state, reward, done, _, _ = env.step(action)
                    env.set_last_action(action)
                    if done:
                        won[team] += 1
                        break
                if done:
                    break
            if done:
                break
    if won[0] >= won[1]:
        print(f"Purple beat {winner} {won[0]} - {won[1]}")
        winning_model = fight_models[0]
        new_king = True
    else:
        print(f"{winner} beat purple {won[1]} - {won[0]}")
        winning_model = fight_models[1]
    

    # Save old king
    if new_king:
        # Use winning model
        agent_models = []
        agent_target_models = []
        for i in range(2):
            model = QNetwork(num_states, num_actions)
            model.load_state_dict(winning_model.state_dict())
            agent_models.append(model)

            target_model = QNetwork(num_states, num_actions)
            target_model.load_state_dict(model.state_dict())
            target_model.eval()
            agent_target_models.append(target_model)

        now = datetime.now()
        formatted_time = now.strftime("%B-%d-%H-%M")
        torch.save(king_model.state_dict(), f"{kings_dir}agent_model{formatted_time}.pth")
        king_model = copy.deepcopy(winning_model)

        # Save current state
        torch.save(king_model.state_dict(), f"{training_dir}agent_model.pth")


    
