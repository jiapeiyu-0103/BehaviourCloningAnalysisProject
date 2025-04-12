import subprocess

commands = [
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_1.npz",
        "--num_episodes", "1",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_1.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_10.npz",
        "--num_episodes", "10",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_10.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_50.npz",
        "--num_episodes", "50",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_50.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_100.npz",
        "--num_episodes", "100",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_100.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_150.npz",
        "--num_episodes", "150",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_150.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_300.npz",
        "--num_episodes", "300",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_300.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_500.npz",
        "--num_episodes", "500",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_500.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_800.npz",
        "--num_episodes", "800",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_800.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_1000.npz",
        "--num_episodes", "1000",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_1000.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_2000.npz",
        "--num_episodes", "2000",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_2000.npy"
    ],
    [
        "python", "collect_trajectory.py",
        "--save_path", "data/lunar_lander_expert_traj/expert_lunarLander_trajectories_3000.npz",
        "--num_episodes", "3000",
        "--rewards_path", "data/expert_traj_rewards/expert_rewards_3000.npy"
    ]
]

for cmd in commands:
    subprocess.run(cmd, check=True)
