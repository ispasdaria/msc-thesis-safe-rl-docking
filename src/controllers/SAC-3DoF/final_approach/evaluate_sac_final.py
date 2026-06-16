"""
Evaluate a trained SAC model for the 3-DoF docking final-approach phase.

This version avoids DummyVecEnv auto-reset during rollout history saving.
It runs a normal, non-vectorized environment and manually normalizes
observations using the saved VecNormalize statistics.
"""

import os
import sys
from collections import Counter

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ---------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.gym_env import Docking3DoFGymEnv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sac_final_approach_rewardv1_50ksteps_seed0"

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")

RESULTS_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "eval_rollouts",
    MODEL_NAME,
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def make_raw_eval_env(seed=0, mode="testing"):
    """
    Creates a normal non-vectorized final-approach environment.
    This is better for saving rollout histories because it does not auto-reset.
    """

    env = Docking3DoFGymEnv(
        phase="final",
        mode=mode,
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )

    return env


def load_vecnormalize_stats():
    """
    Loads VecNormalize statistics.

    A dummy environment is needed only because VecNormalize.load expects
    a VecEnv object. We use it only to access the normalization statistics.
    """

    dummy_env = DummyVecEnv([
        lambda: Monitor(make_raw_eval_env(seed=999, mode="testing"))
    ])

    vecnorm = VecNormalize.load(VECNORM_PATH, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    return vecnorm, dummy_env


def normalize_observation(vecnorm, obs):
    """
    Normalizes a single raw observation using VecNormalize statistics.

    Input:
        obs shape: (6,)

    Output:
        normalized obs shape: (1, 6)
    """

    obs_batch = obs.reshape(1, -1)
    norm_obs = vecnorm.normalize_obs(obs_batch)

    return norm_obs


def run_one_rollout(model, vecnorm, seed=0, mode="testing", max_steps=5000):
    """
    Runs one deterministic rollout in a non-vectorized environment.
    """

    env = make_raw_eval_env(seed=seed, mode=mode)

    obs, reset_info = env.reset(seed=seed)

    total_reward = 0.0
    final_info = reset_info

    # Extra diagnostics saved from info at each step
    sac_rewards = []
    port_distances = []
    port_speeds = []
    s_values = []
    rho_values = []
    s_dot_values = []
    psi_e_values = []
    omega_values = []

    reward_totals = []
    reward_positions = []
    reward_velocities = []
    reward_attitudes = []
    reward_rates = []
    reward_actions = []
    reward_warnings = []
    reward_failures = []
    reward_successes = []
    reward_timeouts = []

    normalized_actions = []
    physical_actions = []

    for step in range(max_steps):
        norm_obs = normalize_observation(vecnorm, obs)

        action, _ = model.predict(norm_obs, deterministic=True)

        # model.predict returns shape (1, 3), but the non-vectorized env expects (3,)
        action = action[0]

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += float(reward)
        final_info = info

        rc = info.get("reward_components", {})

        sac_rewards.append(float(reward))

        port_distances.append(info.get("port_distance", np.nan))
        port_speeds.append(info.get("port_speed", np.nan))
        s_values.append(info.get("s", np.nan))
        rho_values.append(info.get("rho", np.nan))
        s_dot_values.append(info.get("s_dot", np.nan))
        psi_e_values.append(info.get("psi_e", np.nan))
        omega_values.append(info.get("omega", np.nan))

        reward_totals.append(rc.get("reward_total", np.nan))
        reward_positions.append(rc.get("reward_position", np.nan))
        reward_velocities.append(rc.get("reward_velocity", np.nan))
        reward_attitudes.append(rc.get("reward_attitude", np.nan))
        reward_rates.append(rc.get("reward_rate", np.nan))
        reward_actions.append(rc.get("reward_action", np.nan))
        reward_warnings.append(rc.get("reward_warning", np.nan))
        reward_failures.append(rc.get("reward_failure", np.nan))
        reward_successes.append(rc.get("reward_success", np.nan))
        reward_timeouts.append(rc.get("reward_timeout", np.nan))

        normalized_actions.append(info.get("normalized_action", np.full(3, np.nan)))
        physical_actions.append(info.get("physical_action", np.full(3, np.nan)))

        if done:
            break

    times, states, actions_raw, actions_clipped, actions_orbital, core_rewards = (
        env.get_history_arrays()
    )

    env.close()

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "success": final_info.get("success", False),
        "failure": final_info.get("failure", False),
        "time_limit": final_info.get("time_limit", False),
        "termination_reason": final_info.get("termination_reason"),
        "violated_constraints": final_info.get("violated_constraints", []),

        "times": times,
        "states": states,
        "actions_raw": actions_raw,
        "actions_clipped": actions_clipped,
        "actions_orbital": actions_orbital,
        "core_rewards": core_rewards,

        "sac_rewards": np.array(sac_rewards),
        "port_distances": np.array(port_distances),
        "port_speeds": np.array(port_speeds),
        "s_values": np.array(s_values),
        "rho_values": np.array(rho_values),
        "s_dot_values": np.array(s_dot_values),
        "psi_e_values": np.array(psi_e_values),
        "omega_values": np.array(omega_values),

        "reward_totals": np.array(reward_totals),
        "reward_positions": np.array(reward_positions),
        "reward_velocities": np.array(reward_velocities),
        "reward_attitudes": np.array(reward_attitudes),
        "reward_rates": np.array(reward_rates),
        "reward_actions": np.array(reward_actions),
        "reward_warnings": np.array(reward_warnings),
        "reward_failures": np.array(reward_failures),
        "reward_successes": np.array(reward_successes),
        "reward_timeouts": np.array(reward_timeouts),

        "normalized_actions": np.array(normalized_actions),
        "physical_actions": np.array(physical_actions),
    }


def save_rollout(result, rollout_id):
    """
    Saves rollout data as .npz.
    """

    filename = f"rollout_{rollout_id:03d}_seed_{result['seed']}.npz"
    filepath = os.path.join(RESULTS_DIR, filename)

    np.savez(
        filepath,
        seed=result["seed"],
        steps=result["steps"],
        total_reward=result["total_reward"],
        success=result["success"],
        failure=result["failure"],
        time_limit=result["time_limit"],
        termination_reason=str(result["termination_reason"]),
        violated_constraints=np.array(result["violated_constraints"], dtype=object),

        times=result["times"],
        states=result["states"],
        actions_raw=result["actions_raw"],
        actions_clipped=result["actions_clipped"],
        actions_orbital=result["actions_orbital"],
        core_rewards=result["core_rewards"],

        sac_rewards=result["sac_rewards"],
        port_distances=result["port_distances"],
        port_speeds=result["port_speeds"],
        s_values=result["s_values"],
        rho_values=result["rho_values"],
        s_dot_values=result["s_dot_values"],
        psi_e_values=result["psi_e_values"],
        omega_values=result["omega_values"],

        reward_totals=result["reward_totals"],
        reward_positions=result["reward_positions"],
        reward_velocities=result["reward_velocities"],
        reward_attitudes=result["reward_attitudes"],
        reward_rates=result["reward_rates"],
        reward_actions=result["reward_actions"],
        reward_warnings=result["reward_warnings"],
        reward_failures=result["reward_failures"],
        reward_successes=result["reward_successes"],
        reward_timeouts=result["reward_timeouts"],

        normalized_actions=result["normalized_actions"],
        physical_actions=result["physical_actions"],
    )

    return filepath


def main():
    print("=" * 80)
    print("Evaluating SAC final-approach model")
    print("=" * 80)
    print(f"Model path:   {MODEL_PATH}")
    print(f"VecNorm path: {VECNORM_PATH}")
    print(f"Results dir:  {RESULTS_DIR}")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(f"VecNormalize file not found: {VECNORM_PATH}")

    vecnorm, dummy_env = load_vecnormalize_stats()

    model = SAC.load(MODEL_PATH)

    results = []

    seeds = list(range(10))
    mode = "testing"

    for rollout_id, seed in enumerate(seeds):
        result = run_one_rollout(
            model=model,
            vecnorm=vecnorm,
            seed=seed,
            mode=mode,
            max_steps=5000,
        )

        filepath = save_rollout(result, rollout_id)
        results.append(result)

        print(
            f"Rollout {rollout_id:03d} | "
            f"seed={seed:<3} | "
            f"steps={result['steps']:<5} | "
            f"success={result['success']} | "
            f"failure={result['failure']} | "
            f"reason={result['termination_reason']} | "
            f"violated={result['violated_constraints']} | "
            f"states_shape={result['states'].shape} | "
            f"saved={filepath}"
        )

    dummy_env.close()

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)

    success_rate = np.mean([r["success"] for r in results])
    failure_rate = np.mean([r["failure"] for r in results])
    time_limit_rate = np.mean([r["time_limit"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])

    reasons = Counter(r["termination_reason"] for r in results)

    print(f"Number of rollouts: {len(results)}")
    print(f"Success rate:       {success_rate:.2f}")
    print(f"Failure rate:       {failure_rate:.2f}")
    print(f"Time-limit rate:    {time_limit_rate:.2f}")
    print(f"Mean steps:         {mean_steps:.1f}")

    print("\nTermination reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()