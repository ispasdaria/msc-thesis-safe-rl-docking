from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

BASE_DIR = Path(__file__).resolve().parent


def angle_from_obs(obs):
    cos_theta = obs[0]
    sin_theta = obs[1]
    return np.arctan2(sin_theta, cos_theta)


def main():
    env = gym.make("Pendulum-v1")
    try:
        model = SAC.load(BASE_DIR / "models" / "sac_pendulum_default_3e-4")
        obs, _ = env.reset(seed=0)

        theta_history = []
        theta_dot_history = []
        action_history = []
        reward_history = []

        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            theta_history.append(angle_from_obs(obs))
            theta_dot_history.append(obs[2])
            action_history.append(action[0])

            obs, reward, terminated, truncated, _ = env.step(action)
            reward_history.append(reward)

            if terminated or truncated:
                break
    finally:
        env.close()

    time = np.arange(len(theta_history))

    plt.figure()
    plt.plot(time, theta_history)
    plt.xlabel("Step")
    plt.ylabel("Angle theta [rad]")
    plt.title("Pendulum angle over time")
    plt.grid(True)

    plt.figure()
    plt.plot(time, theta_dot_history)
    plt.xlabel("Step")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Pendulum angular velocity over time")
    plt.grid(True)

    plt.figure()
    plt.plot(time, action_history)
    plt.xlabel("Step")
    plt.ylabel("Torque")
    plt.title("Control input over time")
    plt.grid(True)

    plt.figure()
    plt.plot(time, reward_history)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.grid(True)

    plt.figure()
    plt.plot(theta_history, theta_dot_history)
    plt.xlabel("Angle theta [rad]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Phase plot")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
