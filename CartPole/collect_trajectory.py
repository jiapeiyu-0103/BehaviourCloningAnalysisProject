# collect_expert_data_cartpole.py

import gym
import torch
import numpy as np
from stable_baselines3 import A2C

def collect_expert_trajectories(
    model_path="models/cart_pole_best",
    save_path="data/cartpole_traj/expert_cartpole_trajectories_1000.npz",
    num_episodes=1000
):
    env = gym.make("CartPole-v1")
    model = A2C.load(model_path)

    expert_data = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            expert_data.append((obs, int(action)))
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1}: Total Reward = {total_reward}")

    np.savez_compressed(save_path, trajectories=np.array(expert_data, dtype=object))
    print(f"Saved expert data to: {save_path}")

if __name__ == "__main__":
    collect_expert_trajectories()
