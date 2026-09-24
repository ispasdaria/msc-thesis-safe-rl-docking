from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC


ENV_ID = "MountainCarContinuous-v0"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sac_mountain_car_zoo_params_best" / "best_model.zip"


def main():
    env = gym.make(ENV_ID, render_mode="human")
    model = SAC.load(MODEL_PATH)

    try:
        obs, _ = env.reset(seed=2)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
