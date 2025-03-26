# test.py
import torch
from sb3_contrib import MaskablePPO
from samskara import SamskaraEnv

if __name__ == "__main__":
    env = SamskaraEnv(num_fighters_per_team=3)

    # Load a model – either your final or a checkpoint.
    # e.g. final saved as: "ppo_samskara_cnn.zip"
    # or a checkpoint like: "./checkpoints/ppo_samskara_1200000_steps.zip"
    model = MaskablePPO.load("ppo_samskara_cnn.zip", env=env) 

    obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()
