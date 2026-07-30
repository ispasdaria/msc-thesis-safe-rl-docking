"""Run a quick smoke check of Gymnasium's continuous Mountain Car environment."""

import gymnasium as gym


def main():
    env = gym.make("MountainCarContinuous-v0")
    try:
        observation, _ = env.reset(seed=0)
        print("Initial observation:", observation)
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
    finally:
        env.close()


if __name__ == "__main__":
    main()
