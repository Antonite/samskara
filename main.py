import gymnasium as gym
import torch
import torch.nn as nn
import time

from samskara import Samskara

training_dir = "training/"

NUM_AGENTS = 5

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor)

# Custom Env
env = gym.make('Samskara-v0', num_agents=NUM_AGENTS)  # Set the number of agents
num_states_actor = env.observation_space.shape[0]
num_states_critic = env.critic_observation_space.shape[0]
num_actions = env.action_space.n

# Network
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

class Critic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        hidden_size = round(num_states * 2/3 + num_actions)
        self.fc1 = nn.Linear(num_states, hidden_size)  # Takes both states and actions as input
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)  # Outputs a single Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Params
epochs = 1000
num_episodes = 100
max_steps_per_episode = 512
discount_factor = 0.99

# Create actor and critic networks
actor_network = Actor(num_states_actor, num_actions)
critic_network = Critic(num_states_critic, num_actions)
actor_network.load_state_dict(torch.load(f"{training_dir}actor_network.pth"))
critic_network.load_state_dict(torch.load(f"{training_dir}critic_network.pth"))

# Optimizers
actor_optimizer = torch.optim.Adam(actor_network.parameters())
critic_optimizer = torch.optim.Adam(critic_network.parameters())


start_time = time.time()
total_rewards = [0.0] * 2
total_loss = 0.0
update = 0
for epoch in range(epochs):
    total_steps = 0
    for episode in range(num_episodes):
        # env.reset(options={"fair": True})
        env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        # --- Train ---
        for _ in range(max_steps_per_episode):
            # Count total steps for logging
            total_steps += 1
            for team in range(2):
                actions = []
                # Sample an action for each agent
                for agent in range(env.team_len(team)):
                    env.set_active(agent, team)
                    actor_state = env.get_state()

                    action_probs = actor_network(torch.tensor(actor_state))
                    action_distribution = torch.distributions.Categorical(action_probs)
                    action = action_distribution.sample()
                    log_prob = action_distribution.log_prob(action)

                    adjusted_action = action + 1
                    actions.append(adjusted_action)
                    log_probs.append(log_prob.unsqueeze(0))

                # Update the critic based on the actions of all agents
                critic_state = env.get_critic_state(actions)
                value = critic_network(torch.tensor(critic_state))
                values.append(value.unsqueeze(0))

                # Take a step in the environment with all actions
                _, reward, done, _, _ = env.step(actions)
                rewards.append(torch.tensor([reward], dtype=torch.float).unsqueeze(0))
                masks.append(torch.tensor([1-done], dtype=torch.float).unsqueeze(0))
                # Sum rewards for logging
                if reward != 0:
                    total_rewards[team] += reward

                if done:
                    break
            if done:
                break


        if not done and len(rewards) >= 2:
            rewards[-1] = torch.tensor([-1.0], dtype=torch.float).unsqueeze(0)
            rewards[-2] = torch.tensor([-1.0], dtype=torch.float).unsqueeze(0)
                
        # --- Learn ---
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        rewards = torch.cat(rewards)
        masks = torch.cat(masks)

        returns = torch.zeros_like(rewards)
        R = 0
        for step in reversed(range(len(rewards))):
            R = rewards[step] + discount_factor * R * masks[step]
            returns[step] = R

        # Compute the value loss
        value_loss = (returns - values).pow(2).mean()

        # Optimize the critic network
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        # Compute the policy loss
        advantages = returns - values.detach()
        policy_loss = -log_probs * advantages
        policy_loss = policy_loss.mean()

        # Optimize the actor network
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        

    # Print the episode number and total rewards
    elapsed_time = round((time.time() - start_time)/60,3)
    total_rewards = [round(r / num_episodes,3) for r in total_rewards]
    average_steps = total_steps // num_episodes
    print(f"Epoch {epoch}: Total Average Rewards = {total_rewards} Average steps = {average_steps} Elapsed Time = {elapsed_time} minutes")
    start_time = time.time()
    total_rewards = [0.0] * 2

    torch.save(actor_network.state_dict(), f"{training_dir}actor_network.pth")
    torch.save(critic_network.state_dict(), f"{training_dir}critic_network.pth")
