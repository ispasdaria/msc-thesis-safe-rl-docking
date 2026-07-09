import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC

BASE_DIR = Path(__file__).resolve().parent

env = gym.make("Pendulum-v1", render_mode="human")
model = SAC.load(BASE_DIR / "models" / "sac_pendulum_default_3e-4")

for episode in range(5):
    obs, info = env.reset(seed=episode)

    for step in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()
