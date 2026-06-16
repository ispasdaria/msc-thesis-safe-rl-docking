import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sac_final_approach_rewardv1_50ksteps_seed0"

path = os.path.join(
    SCRIPT_DIR,
    "results",
    "eval_rollouts",
    MODEL_NAME,
    "rollout_001_seed_1.npz",
)

print("Reading:", path)

data = np.load(path, allow_pickle=True)

states = data["states"]
actions = data["actions_clipped"]

# Prefer SAC rewards if available
if "sac_rewards" in data.files:
    rewards = data["sac_rewards"]
else:
    rewards = data["rewards"]

print("available keys:", data.files)

print("states shape:", states.shape)
print("rx min/max:", states[:, 0].min(), states[:, 0].max())
print("ry min/max:", states[:, 1].min(), states[:, 1].max())
print("vx min/max:", states[:, 2].min(), states[:, 2].max())
print("vy min/max:", states[:, 3].min(), states[:, 3].max())
print("psi min/max:", states[:, 4].min(), states[:, 4].max())
print("omega min/max:", states[:, 5].min(), states[:, 5].max())
print("actions min/max:", actions.min(axis=0), actions.max(axis=0))
print("rewards min/max:", rewards.min(), rewards.max())

if "port_distances" in data.files:
    print("port distance min/max:", data["port_distances"].min(), data["port_distances"].max())
    print("port speed min/max:", data["port_speeds"].min(), data["port_speeds"].max())
    print("s_dot min/max:", data["s_dot_values"].min(), data["s_dot_values"].max())