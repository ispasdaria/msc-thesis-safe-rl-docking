import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC

BASE_DIR = Path(__file__).resolve().parent


def angle_from_obs(obs):
    cos_theta = obs[0]
    sin_theta = obs[1]
    return np.arctan2(sin_theta, cos_theta)


env = gym.make("Pendulum-v1")
model = SAC.load(BASE_DIR / "models" / "sac_pendulum_default_3e-4")

obs, info = env.reset(seed=0)

theta_history = []
theta_dot_history = []
action_history = []
reward_history = []

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)

    theta = angle_from_obs(obs)
    theta_dot = obs[2]

    obs, reward, terminated, truncated, info = env.step(action)

    theta_history.append(theta)
    theta_dot_history.append(theta_dot)
    action_history.append(action[0])
    reward_history.append(reward)

    if terminated or truncated:
        break

env.close()

time = np.arange(len(theta_history))

plt.figure()
plt.plot(time, theta_history)
plt.xlabel("Step")
plt.ylabel("Angle theta [rad]")
plt.title("Pendulum angle over time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time, theta_dot_history)
plt.xlabel("Step")
plt.ylabel("Angular velocity [rad/s]")
plt.title("Pendulum angular velocity over time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time, action_history)
plt.xlabel("Step")
plt.ylabel("Torque")
plt.title("Control input over time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time, reward_history)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(theta_history, theta_dot_history)
plt.xlabel("Angle theta [rad]")
plt.ylabel("Angular velocity [rad/s]")
plt.title("Phase plot")
plt.grid(True)
plt.show()
