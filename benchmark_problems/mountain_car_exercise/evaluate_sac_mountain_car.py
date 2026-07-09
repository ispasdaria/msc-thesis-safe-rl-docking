import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC


ENV_ID = "MountainCarContinuous-v0"
MODEL_PATH = "mountain_car_exercise/models/sac_mountain_car_zoo_params_best/best_model.zip"


def main():
    os.makedirs("mountain_car_exercise/figures/sac_mountain_car_zoo_params_best", exist_ok=True)

    env = gym.make(ENV_ID)
    model = SAC.load(MODEL_PATH)

    obs, info = env.reset(seed=1)

    position_history = []
    velocity_history = []
    action_history = []
    reward_history = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        position_history.append(obs[0])
        velocity_history.append(obs[1])
        action_history.append(action[0])

        obs, reward, terminated, truncated, info = env.step(action)
        reward_history.append(reward)

    env.close()

    position_history = np.array(position_history)
    velocity_history = np.array(velocity_history)
    action_history = np.array(action_history)
    reward_history = np.array(reward_history)

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
    plt.savefig("mountain_car_exercise/figures/sac_mountain_car_zoo_params_best/position.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, velocity_history)
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.title("SAC on MountainCarContinuous-v0: Velocity")
    plt.grid(True)
    plt.savefig("mountain_car_exercise/figures/sac_mountain_car_zoo_params_best/velocity.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, action_history)
    plt.xlabel("Step")
    plt.ylabel("Action / Force")
    plt.title("SAC on MountainCarContinuous-v0: Action")
    plt.grid(True)
    plt.savefig("mountain_car_exercise/figures/sac_mountain_car_zoo_params_best/action.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(time, np.cumsum(reward_history))
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.title("SAC on MountainCarContinuous-v0: Cumulative Reward")
    plt.grid(True)
    plt.savefig("mountain_car_exercise/figures/sac_mountain_car_zoo_params_best/cumulative_reward.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()