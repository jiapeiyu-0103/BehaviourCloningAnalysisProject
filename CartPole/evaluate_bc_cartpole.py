# evaluate_bc_cartpole.py

import gym
import torch
import numpy as np
from train_bc_cartpole import BCModel

def evaluate_bc_model(
    model_path="models/cartpole_bc/bc_cartpole_model_1000.pth",
    num_episodes=100,
    render=False
):
    env = gym.make("CartPole-v1")

    model = BCModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    wins = 0

    episode_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, done, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        if total_reward >= 500:  # full-length episode = win
            wins += 1
        print(f"Episode {ep + 1}: Reward = {total_reward}")

    win_rate = wins / num_episodes * 100
    np.save("data/bc_expert_rewards/bc_expert_cartpole_rewards_1000.npy", episode_rewards)
    print(f"Model: {model_path} | Win Rate: {win_rate:.2f}%")
    env.close()
    return win_rate

if __name__ == "__main__":
    evaluate_bc_model(render=False)  # Set to False to run without visualization
