from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

BASE_DIR = Path(__file__).resolve().parent

def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    try:
        model = SAC.load(BASE_DIR / "models" / "sac_pendulum_default_3e-4")

        for episode in range(5):
            obs, _ = env.reset(seed=episode)

            for _ in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
