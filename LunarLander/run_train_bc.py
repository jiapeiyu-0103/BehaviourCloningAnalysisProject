import subprocess

commands = [
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_1.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_1.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_10.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_10.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_50.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_50.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_100.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_100.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_150.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_150.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_300.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_300.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_500.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_500.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_800.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_800.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_1000.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_1000.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_2000.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_2000.pth"
    ],
    [
        "python", "train_bc_lunarlander.py",
        "--data_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_3000.npz",
        "--model_save_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_3000.pth"
    ]
]

for cmd in commands:
    subprocess.run(cmd, check=True)
