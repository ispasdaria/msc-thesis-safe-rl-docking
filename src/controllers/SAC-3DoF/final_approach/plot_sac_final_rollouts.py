"""
Plot SAC final-approach evaluation rollouts.

This script reads .npz rollout histories saved by evaluate_sac_final.py.

Use this version for the reward v13 longer-training experiment.
"""

import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sac_final_approach_rewardv13_50ksteps_seed0"

ROLLOUT_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "eval_rollouts",
    MODEL_NAME,
)

PLOT_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "plots",
    MODEL_NAME,
)

os.makedirs(PLOT_DIR, exist_ok=True)


def get_array(data, key, length=None, default=np.nan):
    """
    Safely loads an array from the .npz file.

    If the key does not exist, returns an array filled with NaN.
    This makes the plotting script robust while reward diagnostics evolve.
    """

    if key in data.files:
        return data[key]

    if length is None:
        return np.array([])

    return np.full(length, default, dtype=float)


def get_scalar(data, key, default=np.nan):
    """
    Safely loads a scalar from the .npz file.

    np.savez stores Python floats as 0-dimensional arrays, while older
    rollouts may store one-element arrays. This handles both conventions.
    """

    if key not in data.files:
        return default

    value = np.asarray(data[key])

    if value.size == 0:
        return default

    return value.reshape(-1)[0]


def load_rollout(filepath):
    data = np.load(filepath, allow_pickle=True)

    states = data["states"]
    n_steps = len(states) - 1

    return {
        "filepath": filepath,
        "seed": int(data["seed"]),
        "steps": int(data["steps"]),
        "total_reward": float(data["total_reward"]),
        "success": bool(data["success"]),
        "failure": bool(data["failure"]),
        "time_limit": bool(data["time_limit"]),
        "termination_reason": str(data["termination_reason"]),
        "violated_constraints": data["violated_constraints"],

        "final_port_distance": float(get_scalar(data, "final_port_distance")),
        "min_port_distance": float(get_scalar(data, "min_port_distance")),
        "final_s": float(get_scalar(data, "final_s")),
        "min_s": float(get_scalar(data, "min_s")),
        "final_rho": float(get_scalar(data, "final_rho")),
        "max_rho": float(get_scalar(data, "max_rho")),

        "times": data["times"],
        "states": states,
        "actions_raw": data["actions_raw"],
        "actions_clipped": data["actions_clipped"],
        "actions_orbital": data["actions_orbital"],

        "sac_rewards": get_array(data, "sac_rewards", n_steps),
        "port_distances": get_array(data, "port_distances", n_steps),
        "port_speeds": get_array(data, "port_speeds", n_steps),
        "s_values": get_array(data, "s_values", n_steps),
        "rho_values": get_array(data, "rho_values", n_steps),
        "s_dot_values": get_array(data, "s_dot_values", n_steps),
        "psi_e_values": get_array(data, "psi_e_values", n_steps),
        "omega_values": get_array(data, "omega_values", n_steps),

        # Standard reward terms
        "reward_totals": get_array(data, "reward_totals", n_steps),
        "reward_positions": get_array(data, "reward_positions", n_steps),
        "reward_velocities": get_array(data, "reward_velocities", n_steps),
        "reward_attitudes": get_array(data, "reward_attitudes", n_steps),
        "reward_rates": get_array(data, "reward_rates", n_steps),
        "reward_actions": get_array(data, "reward_actions", n_steps),
        "reward_alphas": get_array(data, "reward_alphas", n_steps),
        "reward_warnings": get_array(data, "reward_warnings", n_steps),
        "reward_failures": get_array(data, "reward_failures", n_steps),
        "reward_successes": get_array(data, "reward_successes", n_steps),
        "reward_timeouts": get_array(data, "reward_timeouts", n_steps),

        # Reward-specific terms
        "desired_s_dot": get_array(data, "desired_s_dot_values", n_steps),
        "s_dot_error": get_array(data, "s_dot_error_values", n_steps),
        "too_fast": get_array(data, "too_fast_values", n_steps),
        "moving_away": get_array(data, "moving_away_values", n_steps),
        "v_lateral": get_array(data, "v_lateral_values", n_steps),
        "progress_port": get_array(data, "progress_port_values", n_steps),
        "progress_s": get_array(data, "progress_s_values", n_steps),
        "rho_progress": get_array(data, "rho_progress_values", n_steps),
        "psi_ratio": get_array(data, "psi_ratio_values", n_steps),

        # These are usually NaN for reward v7, but are kept for compatibility.
        "rho_gate": get_array(data, "rho_gate_values", n_steps),
        "psi_gate": get_array(data, "psi_gate_values", n_steps),
        "progress_gate": get_array(data, "progress_gate_values", n_steps),
        "safe_progress": get_array(data, "safe_progress_values", n_steps),
        "unsafe_forward_progress": get_array(
            data,
            "unsafe_forward_progress_values",
            n_steps,
        ),

        "reward_s_dot": get_array(data, "reward_s_dot_values", n_steps),
        "reward_too_fast": get_array(data, "reward_too_fast_values", n_steps),
        "reward_moving_away": get_array(data, "reward_moving_away_values", n_steps),
        "reward_lateral": get_array(data, "reward_lateral_values", n_steps),
        "reward_lateral_speed": get_array(data, "reward_lateral_speed_values", n_steps),
        "reward_progress": get_array(data, "reward_progress_values", n_steps),
        "reward_rho_progress": get_array(data, "reward_rho_progress_values", n_steps),
        "reward_psi_margin": get_array(data, "reward_psi_margin_values", n_steps),

        "normalized_actions": data["normalized_actions"],
        "physical_actions": data["physical_actions"],
    }


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


def plot_one_rollout(result, rollout_id):
    times = result["times"]
    states = result["states"]

    actions_raw = result["actions_raw"]
    actions_clipped = result["actions_clipped"]
    normalized_actions = result["normalized_actions"]
    physical_actions = result["physical_actions"]

    # State components
    rx = states[:, 0]
    ry = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    psi = states[:, 4]
    omega = states[:, 5]

    com_distance = np.sqrt(rx**2 + ry**2)
    com_speed = np.sqrt(vx**2 + vy**2)

    # Step-based diagnostics
    sac_rewards = result["sac_rewards"]
    port_distances = result["port_distances"]
    port_speeds = result["port_speeds"]
    s_values = result["s_values"]
    rho_values = result["rho_values"]
    s_dot_values = result["s_dot_values"]
    psi_e_values = result["psi_e_values"]
    omega_values = result["omega_values"]

    reward_totals = result["reward_totals"]
    reward_positions = result["reward_positions"]
    reward_velocities = result["reward_velocities"]
    reward_attitudes = result["reward_attitudes"]
    reward_rates = result["reward_rates"]
    reward_actions = result["reward_actions"]
    reward_alphas = result["reward_alphas"]
    reward_warnings = result["reward_warnings"]
    reward_failures = result["reward_failures"]

    desired_s_dot = result["desired_s_dot"]
    s_dot_error = result["s_dot_error"]
    too_fast = result["too_fast"]
    moving_away = result["moving_away"]
    v_lateral = result["v_lateral"]
    progress_port = result["progress_port"]
    progress_s = result["progress_s"]
    rho_progress = result["rho_progress"]
    psi_ratio = result["psi_ratio"]
    rho_gate = result["rho_gate"]
    psi_gate = result["psi_gate"]
    progress_gate = result["progress_gate"]
    safe_progress = result["safe_progress"]
    unsafe_forward_progress = result["unsafe_forward_progress"]
    reward_s_dot = result["reward_s_dot"]
    reward_too_fast = result["reward_too_fast"]
    reward_moving_away = result["reward_moving_away"]
    reward_lateral = result["reward_lateral"]
    reward_lateral_speed = result["reward_lateral_speed"]
    reward_progress = result["reward_progress"]
    reward_rho_progress = result["reward_rho_progress"]
    reward_psi_margin = result["reward_psi_margin"]

    # History convention:
    # states has length N+1 because it includes reset state.
    # step-based arrays have length N.
    t_states = times[: len(states)]
    n_steps = len(port_distances)
    t_steps = times[1 : 1 + n_steps]

    title_suffix = (
        f"model={MODEL_NAME}\n"
        f"seed={result['seed']}, "
        f"reason={result['termination_reason']}, "
        f"violated={list(result['violated_constraints'])}"
    )

    # ------------------------------------------------------------
    # 1. 2D trajectory
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(rx, ry, label="trajectory")
    plt.scatter(rx[0], ry[0], marker="o", label="start")
    plt.scatter(rx[-1], ry[-1], marker="x", label="end")
    plt.xlabel("rx [m]")
    plt.ylabel("ry [m]")
    plt.title(f"SAC Final Approach Trajectory\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    save_plot(f"rollout_{rollout_id:03d}_trajectory.png")

    # ------------------------------------------------------------
    # 2. Position states
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_states, rx, label="rx")
    plt.plot(t_states, ry, label="ry")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title(f"SAC Final Approach Position\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_position.png")

    # ------------------------------------------------------------
    # 3. Velocity states
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_states, vx, label="vx")
    plt.plot(t_states, vy, label="vy")
    plt.plot(t_states, com_speed, label="COM speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"SAC Final Approach Velocity States\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_velocity_states.png")

    # ------------------------------------------------------------
    # 4. Docking-port distance and speed
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, port_distances, label="port distance")
    plt.xlabel("Time [s]")
    plt.ylabel("Port distance [m]")
    plt.title(f"SAC Final Approach Port Distance\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_port_distance.png")

    plt.figure()
    plt.plot(t_steps, port_speeds, label="port speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Port speed [m/s]")
    plt.title(f"SAC Final Approach Port Speed\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_port_speed.png")

    # ------------------------------------------------------------
    # 5. Geometry: s and rho
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, s_values, label="s")
    plt.plot(t_steps, rho_values, label="rho")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title(f"SAC Final Approach Geometry\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_geometry_s_rho.png")

    plt.figure()
    plt.plot(t_steps, rho_values, label="rho")
    plt.xlabel("Time [s]")
    plt.ylabel("rho [m]")
    plt.title(f"SAC Final Approach Lateral Offset rho\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_rho.png")

    plt.figure()
    plt.plot(t_steps, rho_values, label="rho")
    plt.plot(t_steps, v_lateral, label="v_lateral")
    plt.xlabel("Time [s]")
    plt.ylabel("Lateral diagnostic")
    plt.title(f"SAC Final Approach Lateral Offset and Velocity\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_lateral_diagnostics.png")

    # ------------------------------------------------------------
    # 6. Axial closing speed and desired s_dot
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, s_dot_values, label="actual s_dot")
    plt.plot(t_steps, desired_s_dot, linestyle="--", label="desired s_dot")
    plt.axhline(0.0, linestyle=":", linewidth=1, label="zero")
    plt.xlabel("Time [s]")
    plt.ylabel("s_dot [m/s]")
    plt.title(f"SAC Final Approach Axial Closing Speed\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_s_dot_vs_desired.png")

    plt.figure()
    plt.plot(t_steps, s_dot_error, label="s_dot error")
    plt.plot(t_steps, too_fast, label="too fast")
    plt.plot(t_steps, moving_away, label="moving away")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed diagnostic [m/s]")
    plt.title(f"SAC Final Approach Speed Diagnostics\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_speed_diagnostics.png")

    # ------------------------------------------------------------
    # 7. Progress diagnostics
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, progress_port, label="progress_port")
    plt.plot(t_steps, progress_s, label="progress_s")
    plt.plot(t_steps, rho_progress, label="rho_progress")
    plt.axhline(0.0, linestyle=":", linewidth=1, label="zero")
    plt.xlabel("Time [s]")
    plt.ylabel("Progress per step [m]")
    plt.title(f"SAC Final Approach Progress Diagnostics\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_progress_diagnostics.png")

    # These are useful only for gated-progress reward versions.
    # For reward v7 they are usually NaN and the plot can be ignored.
    if not np.all(np.isnan(progress_gate)):
        plt.figure()
        plt.plot(t_steps, rho_gate, label="rho_gate")
        plt.plot(t_steps, psi_gate, label="psi_gate")
        plt.plot(t_steps, progress_gate, label="progress_gate")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Time [s]")
        plt.ylabel("Gate value")
        plt.title(f"SAC Final Approach Progress Gates\n{title_suffix}")
        plt.legend()
        plt.grid(True)
        save_plot(f"rollout_{rollout_id:03d}_progress_gates.png")

    if not np.all(np.isnan(safe_progress)):
        plt.figure()
        plt.plot(t_steps, progress_s, label="progress_s")
        plt.plot(t_steps, safe_progress, label="safe_progress")
        plt.plot(t_steps, unsafe_forward_progress, label="unsafe_forward_progress")
        plt.axhline(0.0, linestyle=":", linewidth=1, label="zero")
        plt.xlabel("Time [s]")
        plt.ylabel("Progress per step [m]")
        plt.title(f"SAC Final Approach Gated Progress\n{title_suffix}")
        plt.legend()
        plt.grid(True)
        save_plot(f"rollout_{rollout_id:03d}_gated_progress.png")

    # ------------------------------------------------------------
    # 8. Attitude diagnostics
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, np.rad2deg(psi_e_values), label="psi_e")
    plt.plot(t_steps, np.rad2deg(omega_values), label="omega")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle / rate [deg], [deg/s]")
    plt.title(f"SAC Final Approach Alignment Diagnostics\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_alignment.png")

    plt.figure()
    plt.plot(t_steps, psi_ratio, label="psi_ratio")
    plt.axhline(0.5, linestyle=":", linewidth=1, label="margin threshold")
    plt.axhline(1.0, linestyle="--", linewidth=1, label="safe limit")
    plt.xlabel("Time [s]")
    plt.ylabel("psi_abs / psi_safe")
    plt.title(f"SAC Final Approach Psi Margin Diagnostic\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_psi_margin_diagnostic.png")

    plt.figure()
    plt.plot(t_states, np.rad2deg(psi), label="psi state")
    plt.plot(t_states, np.rad2deg(omega), label="omega state")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle / rate [deg], [deg/s]")
    plt.title(f"SAC Final Approach Attitude States\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_attitude_states.png")

    # ------------------------------------------------------------
    # 9. Physical and normalized actions
    # ------------------------------------------------------------
    labels = ["ax_Bc", "ay_Bc", "alphaz_Bc"]

    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t_steps, physical_actions[:, i], label=label)
    plt.xlabel("Time [s]")
    plt.ylabel("Physical action")
    plt.title(f"SAC Final Approach Physical Actions\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_physical_actions.png")

    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t_steps, normalized_actions[:, i], label=label)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized action [-1, 1]")
    plt.title(f"SAC Final Approach Normalized Actions\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_normalized_actions.png")

    # ------------------------------------------------------------
    # 10. Action smoothness / action changes
    # ------------------------------------------------------------
    if len(physical_actions) > 1:
        delta_actions = np.diff(physical_actions, axis=0)
        t_delta = t_steps[1:]

        plt.figure()
        for i, label in enumerate(labels):
            plt.plot(t_delta, delta_actions[:, i], label=f"delta {label}")
        plt.xlabel("Time [s]")
        plt.ylabel("Action change")
        plt.title(f"SAC Final Approach Action Changes\n{title_suffix}")
        plt.legend()
        plt.grid(True)
        save_plot(f"rollout_{rollout_id:03d}_action_changes.png")

    # ------------------------------------------------------------
    # 11. Raw vs clipped actions from core environment
    # ------------------------------------------------------------
    t_core_actions = times[: len(actions_clipped)]

    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t_core_actions, actions_raw[:, i], linestyle="--", label=f"{label} raw")
        plt.plot(t_core_actions, actions_clipped[:, i], label=f"{label} clipped")
    plt.xlabel("Time [s]")
    plt.ylabel("Action")
    plt.title(f"SAC Final Approach Core Actions: Raw vs Clipped\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_raw_vs_clipped_actions.png")

    # ------------------------------------------------------------
    # 12. Total reward
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, sac_rewards, label="returned SAC reward")
    plt.plot(t_steps, reward_totals, linestyle="--", label="reward_total component")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward")
    plt.title(f"SAC Final Approach Reward\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_reward_total.png")

    # ------------------------------------------------------------
    # 13. Main reward components
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, reward_positions, label="position")
    plt.plot(t_steps, reward_velocities, label="velocity")
    plt.plot(t_steps, reward_s_dot, label="s_dot tracking")
    plt.plot(t_steps, reward_too_fast, label="too fast")
    plt.plot(t_steps, reward_moving_away, label="moving away")
    plt.plot(t_steps, reward_lateral, label="lateral")
    plt.plot(t_steps, reward_progress, label="progress")
    plt.plot(t_steps, reward_rho_progress, label="rho progress")
    plt.plot(t_steps, reward_lateral_speed, label="lateral speed")
    plt.plot(t_steps, reward_attitudes, label="attitude")
    plt.plot(t_steps, reward_rates, label="rate")
    plt.plot(t_steps, reward_actions, label="action")
    plt.plot(t_steps, reward_alphas, label="alpha")
    plt.plot(t_steps, reward_psi_margin, label="psi margin")
    plt.plot(t_steps, reward_warnings, label="warning")
    plt.plot(t_steps, reward_failures, label="failure")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward component")
    plt.title(f"SAC Final Approach Reward Components\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_reward_components.png")

    # ------------------------------------------------------------
    # 14. Reward components without position term
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, reward_velocities, label="velocity")
    plt.plot(t_steps, reward_s_dot, label="s_dot tracking")
    plt.plot(t_steps, reward_too_fast, label="too fast")
    plt.plot(t_steps, reward_moving_away, label="moving away")
    plt.plot(t_steps, reward_lateral, label="lateral")
    plt.plot(t_steps, reward_lateral_speed, label="lateral speed")
    plt.plot(t_steps, reward_progress, label="progress")
    plt.plot(t_steps, reward_rho_progress, label="rho progress")
    plt.plot(t_steps, reward_attitudes, label="attitude")
    plt.plot(t_steps, reward_rates, label="rate")
    plt.plot(t_steps, reward_actions, label="action")
    plt.plot(t_steps, reward_alphas, label="alpha")
    plt.plot(t_steps, reward_psi_margin, label="psi margin")
    plt.plot(t_steps, reward_warnings, label="warning")
    plt.plot(t_steps, reward_failures, label="failure")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward component")
    plt.title(f"SAC Reward Components Without Position\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_reward_components_no_position.png")

    # ------------------------------------------------------------
    # 15. Final-approach diagnostics summary
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_states, com_distance, label="COM distance")
    plt.plot(t_states, com_speed, label="COM speed")
    plt.plot(t_steps, port_distances, label="port distance")
    plt.plot(t_steps, port_speeds, label="port speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance / speed")
    plt.title(f"SAC Final Approach Distance and Speed Summary\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    save_plot(f"rollout_{rollout_id:03d}_distance_speed_summary.png")


def main():
    rollout_files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.npz")))

    if not rollout_files:
        raise FileNotFoundError(f"No rollout files found in: {ROLLOUT_DIR}")

    print("=" * 80)
    print("Plotting SAC final-approach rollouts")
    print("=" * 80)
    print(f"Model name:  {MODEL_NAME}")
    print(f"Rollout dir: {ROLLOUT_DIR}")
    print(f"Plot dir:    {PLOT_DIR}")
    print(f"Found files: {len(rollout_files)}")
    print("=" * 80)

    for rollout_id, filepath in enumerate(rollout_files):
        result = load_rollout(filepath)

        print(
            f"Plotting rollout {rollout_id:03d} | "
            f"seed={result['seed']} | "
            f"steps={result['steps']} | "
            f"reason={result['termination_reason']} | "
            f"violated={list(result['violated_constraints'])} | "
            f"min_port={result['min_port_distance']:.2f} m | "
            f"final_port={result['final_port_distance']:.2f} m | "
            f"max_rho={result['max_rho']:.2f} m"
        )

        plot_one_rollout(result, rollout_id)

    print("\nDone.")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
