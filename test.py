# test.py
import torch
from stable_baselines3 import PPO
from samskara import SamskaraEnv

if __name__ == "__main__":
    # Create a single environment instance
    env = SamskaraEnv(num_fighters_per_team=3)

    # Load the trained CNN model (make sure the filename matches your save)
    model = PPO.load("./checkpoints/ppo_samskara_1200000_steps")

    obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        # Predict using the loaded model
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    
    env.close()
