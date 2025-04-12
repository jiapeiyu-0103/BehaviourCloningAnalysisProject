# train_bc_cartpole.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import collections
import argparse

class BCModel(nn.Module):
    def __init__(self, input_dim=8, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, num_actions)  
        )

    def forward(self, x):
        return self.net(x)

def train_bc_model(data_path,model_save_path):
    epochs=50
    batch_size=32
    lr=1e-3
    data = np.load(data_path, allow_pickle=True)["trajectories"]

    states = torch.tensor([x[0] for x in data], dtype=torch.float32)
    actions = torch.tensor([x[1] for x in data], dtype=torch.long)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    action_counts = collections.Counter([x[1] for x in data])
    total = sum(action_counts.values())
    weights = [1 / action_counts.get(i, 1) for i in range(4)]  # 4 actions
    weights = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.squeeze(1)
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Saved BC model to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/lunar_lander_expert_traj/expert_lunarLander_trajectories_1.npz")
    parser.add_argument("--model_save_path", type=str, default="models/lunarlander_expert_bc/expert_bc_lunarlander_model_1.pth")
    args = parser.parse_args()
    train_bc_model(data_path=args.data_path, model_save_path=args.model_save_path)
