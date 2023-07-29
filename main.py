import gymnasium as gym
import torch
import torch.nn as nn
import time
import random
import warnings

from samskara import Samskara

# Ignore unrelated warnings
warnings.filterwarnings('ignore', '.*Box observation space is an image.*', category=UserWarning, module='gym')

training_dir = "training/"

NUM_AGENTS = 5

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

# Custom Env
env = gym.make('Samskara-v0', num_agents=NUM_AGENTS)  # Set the number of agents
num_input_channels_actor = env.observation_space.shape[0]
# num_input_channels_critic = env.critic_observation_space.shape[0]
num_actions = env.action_space.n

# Network
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
    
class Critic(nn.Module):
    def __init__(self, num_input_channels, num_actions):
        super(Critic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d((num_input_channels + num_actions)*NUM_AGENTS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        flattened_size = 64 * env.num_cells_row * env.num_cells_row
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x





# Params
epochs = 1000
num_episodes = 10
max_steps_per_episode = 1024
discount_factor = 0.99

# Create actor and critic networks
actor_network = Actor(num_input_channels_actor, num_actions)
critic_network = Critic(num_input_channels_actor, num_actions)
# actor_network.load_state_dict(torch.load(f"{training_dir}actor_network.pth"))
# critic_network.load_state_dict(torch.load(f"{training_dir}critic_network.pth"))

# Optimizers
actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=0.0001)
critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=0.0001)

for epoch in range(epochs):
    total_steps = 0
    critic_loss = [0.0] * 2
    actor_loss = [0.0] * 2
    total_rewards = [0.0] * 2
    start_time = time.time()
    for episode in range(num_episodes):
        env.reset(options={"fair": True})
        # env.reset()
        done = False
        log_probs = [[],[]]
        values = [[],[]]
        rewards = [[],[]]
        masks = [[],[]]
        
        # --- Train ---
        for _ in range(max_steps_per_episode):
            # Count total steps for logging
            total_steps += 1
            for team in range(1):
                actions_tensor = torch.zeros((env.team_len(team), num_actions))
                states = []
                # Sample an action for each agent
                team_length = env.team_len(team)
                for agent in range(team_length):
                    env.set_active(agent, team)
                    actor_state = env.get_state()

                    # Compute the action
                    action_probs = actor_network(torch.tensor(actor_state).unsqueeze(0))
                    action_distribution = torch.distributions.Categorical(action_probs)
                    if team == 0 or random.random() > 0.2:
                        action = action_distribution.sample()
                    else:
                        action = torch.randint(high=12, size=())

                    log_prob = action_distribution.log_prob(action)
                    log_probs[team].append(log_prob.unsqueeze(0))
                    adjusted_action = action + 1

                    # Take a step in the environment
                    _, reward, done, _, _ = env.step(adjusted_action)

                    # Store everything
                    actions_tensor[agent, action] = 1
                    states.append(actor_state)
                    rewards[team].append(torch.tensor([reward], dtype=torch.float).unsqueeze(0))
                    masks[team].append(torch.tensor([1-done], dtype=torch.float).unsqueeze(0))

                    # Sum rewards for logging
                    if reward != 0:
                        total_rewards[team] += reward

                    if done:
                        break

                # Update the critic based on the actions and states of all agents
                for start_i in range(len(states)):
                    # Initialize a tensor filled with zeros to store the modified states of all agents
                    all_states = torch.zeros(NUM_AGENTS, 19, 9, 9)
                    # Iterate over all agents
                    for i in range(start_i,len(states)):
                        agent_state = torch.tensor(states[i])
                        # Embed the action into the state
                        action = actions_tensor[i].unsqueeze(1).unsqueeze(2)  # Shape: (12, 1, 1)
                        action = action.expand(-1, agent_state.shape[1], agent_state.shape[2])  # Shape: (12, 9, 9)
                        agent_state = torch.cat([agent_state, action], dim=0)  # Shape: (19, 9, 9)

                        # Add the modified state to the `all_states` tensor
                        all_states[i] = agent_state
                    global_state = all_states.view(1, -1, 9, 9)
                    value = critic_network(global_state)
                    values[team].append(value)

                if done:
                    break
            if done:
                break

        if not done:
            print("stuck - skipping")
            break
        # winner = env.compute_winner()
        # if winner == -1:
        #     rewards[0][-1] = torch.tensor([-0.2], dtype=torch.float).unsqueeze(0)
        #     rewards[1][-1] = torch.tensor([-0.2], dtype=torch.float).unsqueeze(0)
        # else:
        #     rewards[winner][-1] = torch.tensor([1], dtype=torch.float).unsqueeze(0)
        #     rewards[(winner+1)%2][-1] = torch.tensor([-0.2], dtype=torch.float).unsqueeze(0)
        #     masks[winner][-1] = torch.tensor([0], dtype=torch.float).unsqueeze(0)
        #     masks[(winner+1)%2][-1] = torch.tensor([0], dtype=torch.float).unsqueeze(0)

        for team in range(1):
            # --- Learn ---
            team_log_probs = torch.cat(log_probs[team])
            team_values = torch.cat(values[team])
            team_rewards = torch.cat(rewards[team])
            team_masks = torch.cat(masks[team])

            returns = torch.zeros_like(team_rewards)
            R = 0
            for step in reversed(range(len(team_rewards))):
                R = team_rewards[step] + discount_factor * R * team_masks[step]
                returns[step] = R

            # Compute the value loss
            value_loss = (returns - team_values).pow(2).mean()
            critic_loss[team] += value_loss.item()

            # Optimize the critic network
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            critic_optimizer.step()

            # Compute the policy loss
            advantages = returns - team_values.detach()
            policy_loss = -team_log_probs * advantages
            policy_loss = policy_loss.mean()
            actor_loss[team] += policy_loss.item()

            # Optimize the actor network
            actor_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            actor_optimizer.step()
        

    # Print the episode number and total rewards
    elapsed_time = round((time.time() - start_time)/60,3)
    total_rewards = [round(r / num_episodes,3) for r in total_rewards]
    average_steps = total_steps // num_episodes
    actor_loss = [round(l / num_episodes,3) for l in actor_loss]
    critic_loss = [round(l / num_episodes,3) for l in critic_loss]
    print(f"Epoch {epoch}: Rewards = {total_rewards} Actor Loss = {actor_loss} Critic Loss = {critic_loss} Steps = {average_steps} Elapsed Time = {elapsed_time} minutes")

    torch.save(actor_network.state_dict(), f"{training_dir}actor_network.pth")
    torch.save(critic_network.state_dict(), f"{training_dir}critic_network.pth")
