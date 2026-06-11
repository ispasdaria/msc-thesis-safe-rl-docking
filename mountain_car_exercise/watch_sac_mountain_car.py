import gymnasium as gym
from stable_baselines3 import SAC


ENV_ID = "MountainCarContinuous-v0"
MODEL_PATH = "mountain_car_exercise/models/sac_mountain_car_zoo_params_bs256_best/best_model.zip"


def main():
    env = gym.make(ENV_ID, render_mode="human")
    model = SAC.load(MODEL_PATH)

    obs, info = env.reset(seed=2)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()