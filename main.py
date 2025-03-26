# main.py
import torch
import torch.nn as nn
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import multiprocessing

from samskara import SamskaraEnv

# Custom CNN for the (7, 9, 9) input shape in SamskaraEnv
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 7 in your environment

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Figure out output shape
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 9, 9)
            out = self.cnn(sample)
            cnn_output_size = out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))

def make_env():
    """Factory for SubprocVecEnv."""
    def _init():
        return SamskaraEnv(num_fighters_per_team=3)
    return _init

if __name__ == "__main__":
    # On Windows or if using spawn, you typically need:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    # but only call these if needed. Some systems (Linux fork) won't require them.

    NUM_ENV = 4  # Adjust based on your CPU/GPU resources
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENV)])

    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[512, 512]  # large policy layers
    )

    # Checkpoint every 100,000 steps, storing into "./checkpoints/"
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/",
        name_prefix="ppo_samskara"
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="cuda",                      # ensures GPU usage
        tensorboard_log="./logs_samskara/", # <--- enable TensorBoard
        verbose=1,
        n_steps=2048,
        batch_size=2048
    )

    # tb_log_name: subfolder name inside "logs_samskara"
    model.learn(total_timesteps=5_000_000, tb_log_name="PPO_Samskara", callback=checkpoint_callback)
    model.save("ppo_samskara_cnn")

