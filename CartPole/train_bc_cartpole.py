# train_bc_cartpole.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class BCModel(nn.Module):
    def __init__(self, input_dim=4, num_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)

def train_bc_model(
    data_path="data/cartpole_traj/expert_cartpole_trajectories_1000.npz",
    model_save_path="models/cartpole_bc/bc_cartpole_model_1000.pth",
    epochs=10,
    batch_size=32,
    lr=1e-3
):
    data = np.load(data_path, allow_pickle=True)["trajectories"]

    states = torch.tensor([x[0] for x in data], dtype=torch.float32)
    actions = torch.tensor([x[1] for x in data], dtype=torch.long)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in loader:
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
    train_bc_model()
