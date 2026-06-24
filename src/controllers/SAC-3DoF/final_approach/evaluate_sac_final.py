"""
Evaluate a trained SAC model for the 3-DoF docking final-approach phase.

This version avoids DummyVecEnv auto-reset during rollout history saving.
It runs a normal, non-vectorized environment and manually normalizes
observations using the saved VecNormalize statistics.

Use this version for the reward v23 experiment.
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

MODEL_NAME = "sac_final_approach_portobs_rewardv2_100ksteps_seed0"

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
        obs shape: (6,) = [rpx, rpy, vpx, vpy, psi_e, omega]

    Output:
        normalized obs shape: (1, 6)
    """

    obs_batch = obs.reshape(1, -1)
    norm_obs = vecnorm.normalize_obs(obs_batch)

    return norm_obs


def safe_last(array, default=np.nan):
    """
    Returns the last value of an array, or NaN if the array is empty.
    """

    if len(array) == 0:
        return default

    return array[-1]


def safe_min(array, default=np.nan):
    """
    Returns the minimum value of an array, or NaN if the array is empty.
    """

    if len(array) == 0:
        return default

    return np.nanmin(array)


def safe_max(array, default=np.nan):
    """
    Returns the maximum value of an array, or NaN if the array is empty.
    """

    if len(array) == 0:
        return default

    return np.nanmax(array)


def run_one_rollout(model, vecnorm, seed=0, mode="testing", max_steps=5000):
    """
    Runs one deterministic rollout in a non-vectorized environment.
    """

    env = make_raw_eval_env(seed=seed, mode=mode)

    obs, reset_info = env.reset(seed=seed)

    state = env.core_env.state
    print("Initial state:")
    print("psi [deg] =", np.rad2deg(state[4]))
    print("omega [deg/s] =", np.rad2deg(state[5]))

    total_reward = 0.0
    final_info = reset_info

    # -----------------------------------------------------------------
    # Diagnostics saved from info at each step
    # -----------------------------------------------------------------

    sac_rewards = []

    # Physical / safety diagnostics
    port_distances = []
    port_speeds = []
    s_values = []
    rho_values = []
    s_dot_values = []
    psi_e_values = []
    omega_values = []

    # General reward components
    reward_totals = []
    reward_positions = []
    reward_velocities = []
    reward_attitudes = []
    reward_rates = []
    reward_actions = []
    reward_alphas = []
    reward_warnings = []
    reward_failures = []
    reward_successes = []
    reward_timeouts = []

    # Reward-specific diagnostics
    desired_s_dot_values = []
    s_dot_error_values = []
    too_fast_values = []
    moving_away_values = []
    v_lateral_values = []
    progress_port_values = []
    progress_s_values = []
    rho_progress_values = []
    rho_excess_values = []
    cone_radius_at_s_values = []
    rho_ratio_values = []
    center_quality_values = []
    psi_ratio_values = []
    omega_ratio_values = []
    speed_ratio_values = []

    # Progress-gating diagnostics. Missing values remain NaN when evaluating an
    # older reward version.
    rho_gate_values = []
    speed_gate_values = []
    psi_gate_values = []
    omega_gate_values = []
    alignment_gate_values = []
    progress_gate_values = []
    safe_progress_values = []
    unsafe_forward_progress_values = []

    reward_s_dot_values = []
    reward_too_fast_values = []
    reward_moving_away_values = []
    reward_lateral_values = []
    reward_rho_corridor_values = []
    reward_lateral_speed_values = []
    reward_progress_values = []
    reward_unsafe_forward_progress_values = []
    reward_speed_margin_values = []
    reward_rho_keep_values = []
    reward_rho_progress_values = []
    reward_cone_preference_values = []
    reward_psi_margin_values = []
    reward_omega_margin_values = []

    # Action diagnostics
    normalized_actions = []
    physical_actions = []

    for step in range(max_steps):
        norm_obs = normalize_observation(vecnorm, obs)

        action, _ = model.predict(norm_obs, deterministic=True)

        # model.predict returns shape (1, 3), but the non-vectorized env expects (3,)
        action = action[0]

        obs, reward, terminated, truncated, info = env.step(action)

        if step == 0:
            state_next = env.core_env.state
            print("normalized action =", action)
            print("Next state:")
            print("psi [deg] =", np.rad2deg(state_next[4]))
            print("omega [deg/s] =", np.rad2deg(state_next[5]))
            print("psi_safe [deg] =", np.rad2deg(env.core_env.psi_safe))
            print("psi_unsafe [deg] =", np.rad2deg(env.core_env.psi_unsafe))
            print("omega_safe [deg/s] =", np.rad2deg(env.core_env.omega_safe))
            print("omega_unsafe [deg/s] =", np.rad2deg(env.core_env.omega_unsafe))
            print("terminated =", terminated)
            print("truncated =", truncated)
            print("info =", info)

        done = terminated or truncated

        total_reward += float(reward)
        final_info = info

        rc = info.get("reward_components", {})

        # -------------------------------------------------------------
        # Returned SAC reward
        # -------------------------------------------------------------
        sac_rewards.append(float(reward))

        # -------------------------------------------------------------
        # Physical / safety diagnostics
        # -------------------------------------------------------------
        port_distances.append(info.get("port_distance", np.nan))
        port_speeds.append(info.get("port_speed", np.nan))
        s_values.append(info.get("s", np.nan))
        rho_values.append(info.get("rho", np.nan))
        s_dot_values.append(info.get("s_dot", np.nan))
        psi_e_values.append(info.get("psi_e", np.nan))
        omega_values.append(info.get("omega", np.nan))

        # -------------------------------------------------------------
        # General reward components
        # -------------------------------------------------------------
        reward_totals.append(rc.get("reward_total", np.nan))
        reward_positions.append(rc.get("reward_position", np.nan))
        reward_velocities.append(rc.get("reward_velocity", np.nan))
        reward_attitudes.append(rc.get("reward_attitude", np.nan))
        reward_rates.append(rc.get("reward_rate", np.nan))
        reward_actions.append(rc.get("reward_action", np.nan))
        reward_alphas.append(rc.get("reward_alpha", np.nan))
        reward_warnings.append(rc.get("reward_warning", np.nan))
        reward_failures.append(rc.get("reward_failure", np.nan))
        reward_successes.append(rc.get("reward_success", np.nan))
        reward_timeouts.append(rc.get("reward_timeout", np.nan))

        # -------------------------------------------------------------
        # Reward-specific diagnostics
        # -------------------------------------------------------------
        desired_s_dot_values.append(rc.get("desired_s_dot", np.nan))
        s_dot_error_values.append(rc.get("s_dot_error", np.nan))
        too_fast_values.append(rc.get("too_fast", np.nan))
        moving_away_values.append(rc.get("moving_away", np.nan))
        v_lateral_values.append(rc.get("v_lateral", np.nan))
        progress_port_values.append(rc.get("progress_port", np.nan))
        progress_s_values.append(rc.get("progress_s", np.nan))
        rho_progress_values.append(rc.get("rho_progress", np.nan))
        rho_excess_values.append(rc.get("rho_excess", np.nan))
        cone_radius_at_s_values.append(rc.get("cone_radius_at_s", np.nan))
        rho_ratio_values.append(rc.get("rho_ratio", np.nan))
        center_quality_values.append(rc.get("center_quality", np.nan))
        psi_ratio_values.append(rc.get("psi_ratio", np.nan))
        omega_ratio_values.append(rc.get("omega_ratio", np.nan))
        speed_ratio_values.append(rc.get("speed_ratio", np.nan))

        rho_gate_values.append(rc.get("rho_gate", np.nan))
        speed_gate_values.append(rc.get("speed_gate", np.nan))
        psi_gate_values.append(rc.get("psi_gate", np.nan))
        omega_gate_values.append(rc.get("omega_gate", np.nan))
        alignment_gate_values.append(rc.get("alignment_gate", np.nan))
        progress_gate_values.append(rc.get("progress_gate", np.nan))
        safe_progress_values.append(rc.get("safe_progress", np.nan))
        unsafe_forward_progress_values.append(
            rc.get("unsafe_forward_progress", np.nan)
        )

        reward_progress_values.append(rc.get("reward_progress", np.nan))
        reward_unsafe_forward_progress_values.append(
            rc.get("reward_unsafe_forward_progress", np.nan)
        )
        reward_speed_margin_values.append(rc.get("reward_speed_margin", np.nan))
        reward_rho_keep_values.append(rc.get("reward_rho_keep", np.nan))
        reward_s_dot_values.append(rc.get("reward_s_dot", np.nan))
        reward_too_fast_values.append(rc.get("reward_too_fast", np.nan))
        reward_moving_away_values.append(rc.get("reward_moving_away", np.nan))
        reward_lateral_values.append(rc.get("reward_lateral", np.nan))
        reward_rho_corridor_values.append(
            rc.get("reward_rho_corridor", np.nan)
        )
        reward_lateral_speed_values.append(rc.get("reward_lateral_speed", np.nan))
        reward_rho_progress_values.append(rc.get("reward_rho_progress", np.nan))
        reward_cone_preference_values.append(
            rc.get("reward_cone_preference", np.nan)
        )
        reward_psi_margin_values.append(rc.get("reward_psi_margin", np.nan))
        reward_omega_margin_values.append(rc.get("reward_omega_margin", np.nan))

        # -------------------------------------------------------------
        # Action diagnostics
        # -------------------------------------------------------------
        normalized_actions.append(info.get("normalized_action", np.full(3, np.nan)))
        physical_actions.append(info.get("physical_action", np.full(3, np.nan)))

        if done:
            break

    times, states, actions_raw, actions_clipped, actions_orbital, core_rewards = (
        env.get_history_arrays()
    )

    env.close()

    port_distances_array = np.array(port_distances)
    s_values_array = np.array(s_values)
    rho_values_array = np.array(rho_values)

    final_port_distance = safe_last(port_distances_array)
    min_port_distance = safe_min(port_distances_array)
    final_s = safe_last(s_values_array)
    min_s = safe_min(s_values_array)
    final_rho = safe_last(rho_values_array)
    max_rho = safe_max(rho_values_array)

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "success": final_info.get("success", False),
        "failure": final_info.get("failure", False),
        "time_limit": final_info.get("time_limit", False),
        "termination_reason": final_info.get("termination_reason"),
        "violated_constraints": final_info.get("violated_constraints", []),

        # Summary diagnostics
        "final_port_distance": final_port_distance,
        "min_port_distance": min_port_distance,
        "final_s": final_s,
        "min_s": min_s,
        "final_rho": final_rho,
        "max_rho": max_rho,

        # Core history arrays
        "times": times,
        "states": states,
        "actions_raw": actions_raw,
        "actions_clipped": actions_clipped,
        "actions_orbital": actions_orbital,
        "core_rewards": core_rewards,

        # Step-based SAC diagnostics
        "sac_rewards": np.array(sac_rewards),
        "port_distances": port_distances_array,
        "port_speeds": np.array(port_speeds),
        "s_values": s_values_array,
        "rho_values": rho_values_array,
        "s_dot_values": np.array(s_dot_values),
        "psi_e_values": np.array(psi_e_values),
        "omega_values": np.array(omega_values),

        # General reward components
        "reward_totals": np.array(reward_totals),
        "reward_positions": np.array(reward_positions),
        "reward_velocities": np.array(reward_velocities),
        "reward_attitudes": np.array(reward_attitudes),
        "reward_rates": np.array(reward_rates),
        "reward_actions": np.array(reward_actions),
        "reward_alphas": np.array(reward_alphas),
        "reward_warnings": np.array(reward_warnings),
        "reward_failures": np.array(reward_failures),
        "reward_successes": np.array(reward_successes),
        "reward_timeouts": np.array(reward_timeouts),

        # Reward-specific diagnostics
        "desired_s_dot_values": np.array(desired_s_dot_values),
        "s_dot_error_values": np.array(s_dot_error_values),
        "too_fast_values": np.array(too_fast_values),
        "moving_away_values": np.array(moving_away_values),
        "v_lateral_values": np.array(v_lateral_values),
        "progress_port_values": np.array(progress_port_values),
        "progress_s_values": np.array(progress_s_values),
        "rho_progress_values": np.array(rho_progress_values),
        "rho_excess_values": np.array(rho_excess_values),
        "cone_radius_at_s_values": np.array(cone_radius_at_s_values),
        "rho_ratio_values": np.array(rho_ratio_values),
        "center_quality_values": np.array(center_quality_values),
        "psi_ratio_values": np.array(psi_ratio_values),
        "omega_ratio_values": np.array(omega_ratio_values),
        "speed_ratio_values": np.array(speed_ratio_values),
        "rho_gate_values": np.array(rho_gate_values),
        "speed_gate_values": np.array(speed_gate_values),
        "psi_gate_values": np.array(psi_gate_values),
        "omega_gate_values": np.array(omega_gate_values),
        "alignment_gate_values": np.array(alignment_gate_values),
        "progress_gate_values": np.array(progress_gate_values),
        "safe_progress_values": np.array(safe_progress_values),
        "unsafe_forward_progress_values": np.array(unsafe_forward_progress_values),

        "reward_s_dot_values": np.array(reward_s_dot_values),
        "reward_too_fast_values": np.array(reward_too_fast_values),
        "reward_moving_away_values": np.array(reward_moving_away_values),
        "reward_lateral_values": np.array(reward_lateral_values),
        "reward_rho_corridor_values": np.array(reward_rho_corridor_values),
        "reward_lateral_speed_values": np.array(reward_lateral_speed_values),
        "reward_progress_values": np.array(reward_progress_values),
        "reward_unsafe_forward_progress_values": np.array(
            reward_unsafe_forward_progress_values
        ),
        "reward_speed_margin_values": np.array(reward_speed_margin_values),
        "reward_rho_keep_values": np.array(reward_rho_keep_values),
        "reward_rho_progress_values": np.array(reward_rho_progress_values),
        "reward_cone_preference_values": np.array(reward_cone_preference_values),
        "reward_psi_margin_values": np.array(reward_psi_margin_values),
        "reward_omega_margin_values": np.array(reward_omega_margin_values),

        # Action diagnostics
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

        # Summary diagnostics
        final_port_distance=result["final_port_distance"],
        min_port_distance=result["min_port_distance"],
        final_s=result["final_s"],
        min_s=result["min_s"],
        final_rho=result["final_rho"],
        max_rho=result["max_rho"],

        # Core history arrays
        times=result["times"],
        states=result["states"],
        actions_raw=result["actions_raw"],
        actions_clipped=result["actions_clipped"],
        actions_orbital=result["actions_orbital"],
        core_rewards=result["core_rewards"],

        # Step-based SAC diagnostics
        sac_rewards=result["sac_rewards"],
        port_distances=result["port_distances"],
        port_speeds=result["port_speeds"],
        s_values=result["s_values"],
        rho_values=result["rho_values"],
        s_dot_values=result["s_dot_values"],
        psi_e_values=result["psi_e_values"],
        omega_values=result["omega_values"],

        # General reward components
        reward_totals=result["reward_totals"],
        reward_positions=result["reward_positions"],
        reward_velocities=result["reward_velocities"],
        reward_attitudes=result["reward_attitudes"],
        reward_rates=result["reward_rates"],
        reward_actions=result["reward_actions"],
        reward_alphas=result["reward_alphas"],
        reward_warnings=result["reward_warnings"],
        reward_failures=result["reward_failures"],
        reward_successes=result["reward_successes"],
        reward_timeouts=result["reward_timeouts"],

        # Reward-specific diagnostics
        desired_s_dot_values=result["desired_s_dot_values"],
        s_dot_error_values=result["s_dot_error_values"],
        too_fast_values=result["too_fast_values"],
        moving_away_values=result["moving_away_values"],
        v_lateral_values=result["v_lateral_values"],
        progress_port_values=result["progress_port_values"],
        progress_s_values=result["progress_s_values"],
        rho_progress_values=result["rho_progress_values"],
        rho_excess_values=result["rho_excess_values"],
        cone_radius_at_s_values=result["cone_radius_at_s_values"],
        rho_ratio_values=result["rho_ratio_values"],
        center_quality_values=result["center_quality_values"],
        psi_ratio_values=result["psi_ratio_values"],
        omega_ratio_values=result["omega_ratio_values"],
        speed_ratio_values=result["speed_ratio_values"],
        rho_gate_values=result["rho_gate_values"],
        speed_gate_values=result["speed_gate_values"],
        psi_gate_values=result["psi_gate_values"],
        omega_gate_values=result["omega_gate_values"],
        alignment_gate_values=result["alignment_gate_values"],
        progress_gate_values=result["progress_gate_values"],
        safe_progress_values=result["safe_progress_values"],
        unsafe_forward_progress_values=result["unsafe_forward_progress_values"],

        reward_s_dot_values=result["reward_s_dot_values"],
        reward_too_fast_values=result["reward_too_fast_values"],
        reward_moving_away_values=result["reward_moving_away_values"],
        reward_lateral_values=result["reward_lateral_values"],
        reward_rho_corridor_values=result["reward_rho_corridor_values"],
        reward_lateral_speed_values=result["reward_lateral_speed_values"],
        reward_progress_values=result["reward_progress_values"],
        reward_unsafe_forward_progress_values=result[
            "reward_unsafe_forward_progress_values"
        ],
        reward_speed_margin_values=result["reward_speed_margin_values"],
        reward_rho_keep_values=result["reward_rho_keep_values"],
        reward_rho_progress_values=result["reward_rho_progress_values"],
        reward_cone_preference_values=result["reward_cone_preference_values"],
        reward_psi_margin_values=result["reward_psi_margin_values"],
        reward_omega_margin_values=result["reward_omega_margin_values"],

        # Action diagnostics
        normalized_actions=result["normalized_actions"],
        physical_actions=result["physical_actions"],
    )

    return filepath


def print_rollout_line(result, rollout_id, filepath):
    """
    Prints one compact rollout summary.
    """

    print(
        f"Rollout {rollout_id:03d} | "
        f"seed={result['seed']:<3} | "
        f"steps={result['steps']:<5} | "
        f"success={result['success']} | "
        f"failure={result['failure']} | "
        f"reason={result['termination_reason']} | "
        f"violated={result['violated_constraints']} | "
        f"min_port={result['min_port_distance']:.2f} m | "
        f"final_port={result['final_port_distance']:.2f} m | "
        f"min_s={result['min_s']:.2f} m | "
        f"final_s={result['final_s']:.2f} m | "
        f"max_rho={result['max_rho']:.2f} m | "
        f"final_rho={result['final_rho']:.2f} m | "
        f"states_shape={result['states'].shape} | "
        f"saved={filepath}"
    )


def save_evaluation_summary(results):
    """
    Saves a text summary of the evaluation results.
    """

    success_rate = np.mean([r["success"] for r in results])
    failure_rate = np.mean([r["failure"] for r in results])
    time_limit_rate = np.mean([r["time_limit"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])

    mean_min_port_distance = np.nanmean([r["min_port_distance"] for r in results])
    mean_final_port_distance = np.nanmean([r["final_port_distance"] for r in results])
    mean_min_s = np.nanmean([r["min_s"] for r in results])
    mean_final_s = np.nanmean([r["final_s"] for r in results])
    mean_max_rho = np.nanmean([r["max_rho"] for r in results])
    mean_final_rho = np.nanmean([r["final_rho"] for r in results])

    reasons = Counter(r["termination_reason"] for r in results)

    constraint_counter = Counter()
    for result in results:
        for constraint in result["violated_constraints"]:
            constraint_counter[constraint] += 1

    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAC final-approach evaluation summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model name: {MODEL_NAME}\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"VecNorm path: {VECNORM_PATH}\n")
        f.write(f"Results dir: {RESULTS_DIR}\n\n")

        f.write(f"Number of rollouts: {len(results)}\n")
        f.write(f"Success rate:       {success_rate:.2f}\n")
        f.write(f"Failure rate:       {failure_rate:.2f}\n")
        f.write(f"Time-limit rate:    {time_limit_rate:.2f}\n")
        f.write(f"Mean steps:         {mean_steps:.1f}\n\n")

        f.write("Distance / geometry summary:\n")
        f.write(f"Mean min port distance:   {mean_min_port_distance:.2f} m\n")
        f.write(f"Mean final port distance: {mean_final_port_distance:.2f} m\n")
        f.write(f"Mean min s:               {mean_min_s:.2f} m\n")
        f.write(f"Mean final s:             {mean_final_s:.2f} m\n")
        f.write(f"Mean max rho:             {mean_max_rho:.2f} m\n")
        f.write(f"Mean final rho:           {mean_final_rho:.2f} m\n\n")

        f.write("Termination reasons:\n")
        for reason, count in reasons.items():
            f.write(f"  {reason}: {count}\n")

        f.write("\nViolated constraints:\n")
        if constraint_counter:
            for constraint, count in constraint_counter.items():
                f.write(f"  {constraint}: {count}\n")
        else:
            f.write("  none\n")

        f.write("\nPer-rollout summary:\n")
        for result in results:
            f.write(
                f"seed={result['seed']:<3} | "
                f"steps={result['steps']:<5} | "
                f"success={result['success']} | "
                f"failure={result['failure']} | "
                f"reason={result['termination_reason']} | "
                f"violated={result['violated_constraints']} | "
                f"min_port={result['min_port_distance']:.2f} m | "
                f"final_port={result['final_port_distance']:.2f} m | "
                f"min_s={result['min_s']:.2f} m | "
                f"final_s={result['final_s']:.2f} m | "
                f"max_rho={result['max_rho']:.2f} m | "
                f"final_rho={result['final_rho']:.2f} m\n"
            )

    return summary_path


def main():
    print("=" * 80)
    print("Evaluating SAC final-approach model")
    print("=" * 80)
    print(f"Model name:  {MODEL_NAME}")
    print(f"Model path:  {MODEL_PATH}")
    print(f"VecNorm path:{VECNORM_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
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

        print_rollout_line(result, rollout_id, filepath)

    dummy_env.close()

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)

    success_rate = np.mean([r["success"] for r in results])
    failure_rate = np.mean([r["failure"] for r in results])
    time_limit_rate = np.mean([r["time_limit"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])

    mean_min_port_distance = np.nanmean([r["min_port_distance"] for r in results])
    mean_final_port_distance = np.nanmean([r["final_port_distance"] for r in results])
    mean_min_s = np.nanmean([r["min_s"] for r in results])
    mean_final_s = np.nanmean([r["final_s"] for r in results])
    mean_max_rho = np.nanmean([r["max_rho"] for r in results])
    mean_final_rho = np.nanmean([r["final_rho"] for r in results])

    reasons = Counter(r["termination_reason"] for r in results)

    constraint_counter = Counter()
    for result in results:
        for constraint in result["violated_constraints"]:
            constraint_counter[constraint] += 1

    print(f"Number of rollouts: {len(results)}")
    print(f"Success rate:       {success_rate:.2f}")
    print(f"Failure rate:       {failure_rate:.2f}")
    print(f"Time-limit rate:    {time_limit_rate:.2f}")
    print(f"Mean steps:         {mean_steps:.1f}")

    print("\nDistance / geometry summary:")
    print(f"Mean min port distance:   {mean_min_port_distance:.2f} m")
    print(f"Mean final port distance: {mean_final_port_distance:.2f} m")
    print(f"Mean min s:               {mean_min_s:.2f} m")
    print(f"Mean final s:             {mean_final_s:.2f} m")
    print(f"Mean max rho:             {mean_max_rho:.2f} m")
    print(f"Mean final rho:           {mean_final_rho:.2f} m")

    print("\nTermination reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print("\nViolated constraints:")
    if constraint_counter:
        for constraint, count in constraint_counter.items():
            print(f"  {constraint}: {count}")
    else:
        print("  none")

    summary_path = save_evaluation_summary(results)

    print(f"\nSaved summary to: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
