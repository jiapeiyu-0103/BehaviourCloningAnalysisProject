import subprocess

commands = [
    # [
    #     "python", "evaluate_bc_lunarlander.py",
    #     "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_1.pth",
    #     "--save_path", "data/bc_expert_rewards/bc_expert_rewards_1.npy"
    # ],
    # [
    #     "python", "evaluate_bc_lunarlander.py",
    #     "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_10.pth",
    #     "--save_path", "data/bc_expert_rewards/bc_expert_rewards_10.npy"
    # ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_50.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_50.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_100.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_100.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_150.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_150.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_300.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_300.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_500.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_500.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_800.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_800.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_1000.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_1000.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_2000.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_2000.npy"
    ],
    [
        "python", "evaluate_bc_lunarlander.py",
        "--model_path", "models/lunarlander_expert_bc/expert_bc_lunarlander_model_3000.pth",
        "--save_path", "data/bc_expert_rewards/bc_expert_rewards_3000.npy"
    ]
]

for cmd in commands:
    subprocess.run(cmd, check=True)
