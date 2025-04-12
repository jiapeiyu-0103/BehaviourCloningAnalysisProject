
import gym
import torch
import numpy as np
from train_bc_lunarlander import BCModel
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
import argparse

def evaluate_bc_model(model_path, save_path):

    num_episodes=100
    env_str = "LunarLander-v3"
    env = make_vec_env(env_str, n_envs=1)
    model = BCModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    wins = 0
    
    episode_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0

        while not done:

            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, info = env.step([action])
            obs = obs[0]
            reward = reward[0]

            done = terminated[0]            
            # print(reward)
            total_reward += reward
            episode_rewards.append(total_reward)

        if total_reward >= 200:  # full-length episode = win
            wins += 1

    np.save(save_path, episode_rewards)
    win_rate = wins / num_episodes * 100
    print(f"Model: {model_path} | Win Rate: {win_rate:.2f}%")
    env.close()
    return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/lunar_lander_expert_traj/expert_bc_lunarlander_model_50.pth")
    parser.add_argument("--save_path", type=str, default="data/bc_expert_rewards/bc_expert_rewards_50.npy")
    args = parser.parse_args()
    evaluate_bc_model(model_path=args.model_path, save_path=args.save_path)
