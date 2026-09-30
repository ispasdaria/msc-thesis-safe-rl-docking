"""
Practical rollout test for Docking3DoFGymEnv.

This script checks:
    - reset behavior
    - random-action stepping
    - reward component logging
    - termination reason
    - action scaling
    - history export
"""

from collections import Counter

import numpy as np

from envs.docking3dof.gym_env import Docking3DoFGymEnv


def run_random_rollout(
    phase,
    seed=0,
    max_steps=500,
    mode="training",
):
    print("=" * 80)
    print(f"Random rollout test | phase={phase} | seed={seed}")
    print("=" * 80)

    env = Docking3DoFGymEnv(
        phase=phase,
        mode=mode,
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    print("\nInitial observation:")
    print(obs)

    print("\nInitial reset info:")
    for key in [
        "phase",
        "mode",
        "max_time",
        "safe",
        "warning",
        "unsafe",
        "port_distance",
        "port_speed",
        "s",
        "rho",
        "s_dot",
        "psi_e",
        "omega",
    ]:
        print(f"  {key}: {info.get(key)}")

    total_reward = 0.0
    termination_reason = None

    last_info = info

    for step in range(max_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        last_info = info

        if step < 5 or terminated or truncated:
            print("\nStep:", step)
            print("  normalized_action:", info.get("normalized_action"))
            print("  physical_action:", info.get("physical_action"))
            print("  obs:", obs)
            print("  reward:", reward)
            print("  terminated:", terminated)
            print("  truncated:", truncated)
            print("  safe:", info.get("safe"))
            print("  warning:", info.get("warning"))
            print("  unsafe:", info.get("unsafe"))
            print("  termination_reason:", info.get("termination_reason"))

            reward_components = info.get("reward_components", {})
            print("  reward_total:", reward_components.get("reward_total"))
            print("  reward_position:", reward_components.get("reward_position"))
            print("  reward_velocity:", reward_components.get("reward_velocity"))
            print("  reward_attitude:", reward_components.get("reward_attitude"))
            print("  reward_rate:", reward_components.get("reward_rate"))
            print("  reward_action:", reward_components.get("reward_action"))
            print("  reward_warning:", reward_components.get("reward_warning"))
            print("  reward_failure:", reward_components.get("reward_failure"))
            print("  reward_success:", reward_components.get("reward_success"))
            print("  reward_timeout:", reward_components.get("reward_timeout"))

        if terminated or truncated:
            termination_reason = info.get("termination_reason")
            break

    if termination_reason is None:
        termination_reason = "max_steps_reached_by_test_script"

    print("\nRollout summary:")
    print("  phase:", phase)
    print("  seed:", seed)
    print("  steps executed:", step + 1)
    print("  total reward:", total_reward)
    print("  termination reason:", termination_reason)
    print("  success:", last_info.get("success"))
    print("  failure:", last_info.get("failure"))
    print("  time_limit:", last_info.get("time_limit"))
    print("  violated_constraints:", last_info.get("violated_constraints"))

    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    print("\nHistory arrays:")
    print("  times shape:", times.shape)
    print("  states shape:", states.shape)
    print("  actions_raw shape:", actions_raw.shape)
    print("  actions_clipped shape:", actions_clipped.shape)
    print("  actions_orbital shape:", actions_orbital.shape)

    assert times.ndim == 1, "times should be a 1D array."
    assert states.ndim == 2 and states.shape[1] == 6, (
        "states should have shape (N, 6)."
    )
    assert actions_raw.ndim == 2 and actions_raw.shape[1] == 3, (
        "actions_raw should have shape (N, 3)."
    )
    assert actions_clipped.ndim == 2 and actions_clipped.shape[1] == 3, (
        "actions_clipped should have shape (N, 3)."
    )
    assert actions_orbital.ndim == 2 and actions_orbital.shape[1] == 3, (
        "actions_orbital should have shape (N, 3)."
    )

    env.close()

    print(f"\nRandom rollout test completed for phase: {phase}\n")

    return {
        "phase": phase,
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "success": last_info.get("success"),
        "failure": last_info.get("failure"),
        "time_limit": last_info.get("time_limit"),
        "violated_constraints": last_info.get("violated_constraints"),
    }


def main():
    results = []

    for phase in ["closing", "final"]:
        for seed in [0, 1, 2]:
            result = run_random_rollout(
                phase=phase,
                seed=seed,
                max_steps=500,
                mode="training",
            )
            results.append(result)

    print("=" * 80)
    print("Summary over random rollout tests")
    print("=" * 80)

    reasons = Counter(result["termination_reason"] for result in results)

    print("\nTermination reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print("\nDetailed results:")
    for result in results:
        print(
            f"phase={result['phase']:<8} "
            f"seed={result['seed']} "
            f"steps={result['steps']:<5} "
            f"total_reward={result['total_reward']:.2f} "
            f"reason={result['termination_reason']} "
            f"success={result['success']} "
            f"failure={result['failure']} "
            f"time_limit={result['time_limit']}"
        )

    print("\nEnvironment rollout tests completed.")


if __name__ == "__main__":
    main()
