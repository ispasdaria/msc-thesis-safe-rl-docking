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
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from stable_baselines3.common.env_checker import check_env
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def expected_observation(env):
    if env.phase == "closing":
        q = env.core_env.get_derived_quantities(env.core_env.state)
        port_state = np.array(
            [
                q["r_p"][0],
                q["r_p"][1],
                q["v_p"][0],
                q["v_p"][1],
                q["psi_e"],
                q["omega"],
            ]
        )
        error = env.core_env.state - env.core_env.p3_target
        error[4] = env.core_env.wrap_to_pi(error[4])
        return np.concatenate((port_state, error)).astype(np.float32)

    q = env.core_env.get_derived_quantities(env.core_env.state)
    return np.array(
        [
            q["r_p"][0],
            q["r_p"][1],
            q["v_p"][0],
            q["v_p"][1],
            q["psi_e"],
            q["omega"],
        ],
        dtype=np.float32,
    )


def check_closing_observation_is_augmented():
    env = Docking3DoFGymEnv(phase="closing", mode="testing", seed=7)
    obs, info = env.reset(seed=7)

    expected = expected_observation(env)

    assert np.allclose(obs, expected)
    assert info["observation_labels"] == (
        "rpx",
        "rpy",
        "vpx",
        "vpy",
        "psi_e",
        "omegaz",
        "p3_rx_error",
        "p3_ry_error",
        "p3_vx_error",
        "p3_vy_error",
        "p3_psi_error",
        "p3_omega_error",
    )

    env.close()


def check_closing_observation_wraps_attitude_error():
    env = Docking3DoFGymEnv(phase="closing", mode="testing", seed=11)
    env.reset(seed=11)

    env.core_env.state[4] = np.pi - 0.1
    env.core_env.p3_target[4] = -np.pi + 0.1

    obs = env._get_obs()

    assert np.isclose(obs[10], -0.2, atol=1e-6)
    env.close()


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
    expected_shape = (12,) if phase == "closing" else (6,)
    assert obs.shape == expected_shape, (
        f"{phase} observation must have shape {expected_shape}."
    )
    assert obs.dtype == np.float32, "Observation must have dtype float32."
    assert isinstance(info, dict), "reset() info must be a dictionary."

    assert env.observation_space.contains(obs), (
        "Initial observation is not inside observation_space."
    )

    expected_obs = expected_observation(env)
    assert np.allclose(obs, expected_obs), (
        "Observation does not match the phase-specific observation definition."
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
    assert obs.shape == expected_shape, (
        f"{phase} step observation must have shape {expected_shape}."
    )
    assert obs.dtype == np.float32, "Step observation must have dtype float32."
    assert isinstance(reward, float), "Reward must be a Python float."
    assert isinstance(terminated, bool), "terminated must be a bool."
    assert isinstance(truncated, bool), "truncated must be a bool."
    assert isinstance(info, dict), "step() info must be a dictionary."

    expected_obs = expected_observation(env)
    assert np.allclose(obs, expected_obs), (
        "Step observation does not match the phase-specific observation definition."
    )

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
    check_closing_observation_is_augmented()
    check_closing_observation_wraps_attitude_error()
    test_single_phase("closing")
    test_single_phase("final")

    print("=" * 80)
    print("All Gym wrapper tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
