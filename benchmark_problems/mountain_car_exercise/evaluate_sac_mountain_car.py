from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC


ENV_ID = "MountainCarContinuous-v0"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sac_mountain_car_zoo_params_best" / "best_model.zip"
FIGURE_DIR = BASE_DIR / "figures" / "sac_mountain_car_zoo_params_best"


def run_episode(env, model, seed):
    observation, _ = env.reset(seed=seed)
    history = {"positions": [], "velocities": [], "actions": [], "rewards": []}

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)

        history["positions"].append(observation[0])
        history["velocities"].append(observation[1])
        history["actions"].append(action[0])
        history["rewards"].append(reward)

    return {name: np.asarray(values) for name, values in history.items()}


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    try:
        model = SAC.load(MODEL_PATH)
        history = run_episode(env, model, seed=1)
    finally:
        env.close()

    position_history = history["positions"]
    velocity_history = history["velocities"]
    action_history = history["actions"]
    reward_history = history["rewards"]

    time = np.arange(len(position_history))

    print("Evaluation result")
    print("=================")
    print(f"Episode length: {len(reward_history)} steps")
    print(f"Total reward: {np.sum(reward_history):.2f}")
    print(f"Final position: {position_history[-1]:.3f}")
    print(f"Final velocity: {velocity_history[-1]:.3f}")
    print(f"Maximum position: {np.max(position_history):.3f}")
    print(f"Mean absolute action: {np.mean(np.abs(action_history)):.3f}")

    plt.figure()
    plt.plot(time, position_history)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("SAC on MountainCarContinuous-v0: Position")
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "position.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, velocity_history)
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.title("SAC on MountainCarContinuous-v0: Velocity")
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "velocity.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, action_history)
    plt.xlabel("Step")
    plt.ylabel("Action / Force")
    plt.title("SAC on MountainCarContinuous-v0: Action")
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "action.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, np.cumsum(reward_history))
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.title("SAC on MountainCarContinuous-v0: Cumulative Reward")
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "cumulative_reward.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
