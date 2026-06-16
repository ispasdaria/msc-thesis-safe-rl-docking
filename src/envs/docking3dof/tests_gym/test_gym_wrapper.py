"""
Basic Gymnasium / Stable-Baselines3 compatibility test for Docking3DoFGymEnv.

This script checks:
    - environment creation
    - observation_space and action_space
    - reset() output format
    - step() output format
    - Stable-Baselines3 check_env compatibility
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from stable_baselines3.common.env_checker import check_env
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def test_single_phase(phase):
    print("=" * 80)
    print(f"Testing Gym wrapper for phase: {phase}")
    print("=" * 80)

    env = Docking3DoFGymEnv(
        phase=phase,
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=0,
    )

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    obs, info = env.reset(seed=0)

    print("\nReset output:")
    print("obs:", obs)
    print("obs shape:", obs.shape)
    print("obs dtype:", obs.dtype)
    print("info keys:", list(info.keys()))

    assert isinstance(obs, np.ndarray), "reset() observation must be a NumPy array."
    assert obs.shape == (6,), "Observation must have shape (6,)."
    assert obs.dtype == np.float32, "Observation must have dtype float32."
    assert isinstance(info, dict), "reset() info must be a dictionary."

    assert env.observation_space.contains(obs), (
        "Initial observation is not inside observation_space."
    )

    action = env.action_space.sample()
    step_output = env.step(action)

    assert len(step_output) == 5, (
        "step() must return 5 values: obs, reward, terminated, truncated, info."
    )

    obs, reward, terminated, truncated, info = step_output

    print("\nOne-step output:")
    print("sampled normalized action:", action)
    print("obs:", obs)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("termination reason:", info.get("termination_reason"))
    print("info keys:", list(info.keys()))

    assert isinstance(obs, np.ndarray), "step() observation must be a NumPy array."
    assert obs.shape == (6,), "Step observation must have shape (6,)."
    assert obs.dtype == np.float32, "Step observation must have dtype float32."
    assert isinstance(reward, float), "Reward must be a Python float."
    assert isinstance(terminated, bool), "terminated must be a bool."
    assert isinstance(truncated, bool), "truncated must be a bool."
    assert isinstance(info, dict), "step() info must be a dictionary."

    assert "reward_components" in info, (
        "info must contain reward_components for diagnostics."
    )

    print("\nReward components:")
    for key, value in info["reward_components"].items():
        print(f"  {key}: {value}")

    print("\nRunning Stable-Baselines3 check_env...")
    check_env(env, warn=True)

    env.close()

    print(f"\nGym wrapper test passed for phase: {phase}\n")


def main():
    test_single_phase("closing")
    test_single_phase("final")

    print("=" * 80)
    print("All Gym wrapper tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()