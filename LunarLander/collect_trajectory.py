# collect_expert_data_cartpole.py

import gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import argparse

def collect_expert_trajectories(save_path,num_episodes, rewards_path):

    model_path="models/best_model_lunarlander_2"
    # save_path="data/lunar_lander_traj/expert_lunarLander_trajectories_3000.npz"
    # num_episodes=3000
    # data/traj_rewards/expert_rewards_3000.npy"
    env_str = "LunarLander-v3"
    env = make_vec_env(env_str, n_envs=1)
    model = DQN.load(model_path, env=env, verbose=0)

    expert_data = []
    episode_rewards = []
    collected = 0
    while collected < num_episodes:
        obs = env.reset()
        done = False
        total_reward = 0

        expisode_traj = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            expisode_traj.append((obs, int(action)))
            obs, reward, done_game, _ = env.step(action)
            total_reward += reward
            done = done_game
        
        if total_reward >= 200:
            expert_data.extend(expisode_traj)
            episode_rewards.append(total_reward)
            collected += 1

    np.savez_compressed(save_path, trajectories=np.array(expert_data, dtype=object))
    print(f"Saved expert data to: {save_path}")
    np.save(rewards_path, episode_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data/lunar_lander_expert_traj/expert_lunarLander_trajectories_50.npz")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--rewards_path", type=str, default="data/expert_traj_rewards/expert_rewards_50.npy")
    args = parser.parse_args()

    collect_expert_trajectories(
        save_path = args.save_path,
        num_episodes = args.num_episodes,
        rewards_path = args.rewards_path
    )
