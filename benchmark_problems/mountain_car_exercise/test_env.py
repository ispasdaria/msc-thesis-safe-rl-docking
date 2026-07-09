import gymnasium as gym

env = gym.make("MountainCarContinuous-v0")
obs, info = env.reset(seed=0)

print("Initial observation:", obs)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

env.close()