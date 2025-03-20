import gymnasium as gym
from stable_baselines3 import PPO
from samskara import SamskaraEnv

env = SamskaraEnv(num_fighters_per_team=3)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_samskara/")
model.learn(total_timesteps=500000)
model.save("ppo_samskara")