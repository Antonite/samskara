import gymnasium as gym
from stable_baselines3 import PPO
from samskara import SamskaraEnv

env = SamskaraEnv(num_fighters_per_team=3)
model = PPO.load("ppo_samskara.zip")

obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()