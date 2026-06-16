"""
Plot SAC final-approach evaluation rollouts.

This script reads .npz rollout histories saved by evaluate_sac_final.py.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sac_final_approach_rewardv1_50ksteps_seed0"

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


def load_rollout(filepath):
    data = np.load(filepath, allow_pickle=True)

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

        "times": data["times"],
        "states": data["states"],
        "actions_raw": data["actions_raw"],
        "actions_clipped": data["actions_clipped"],
        "actions_orbital": data["actions_orbital"],

        # New SAC diagnostics saved by evaluate_sac_final.py
        "sac_rewards": data["sac_rewards"],
        "port_distances": data["port_distances"],
        "port_speeds": data["port_speeds"],
        "s_values": data["s_values"],
        "rho_values": data["rho_values"],
        "s_dot_values": data["s_dot_values"],
        "psi_e_values": data["psi_e_values"],
        "omega_values": data["omega_values"],

        "reward_totals": data["reward_totals"],
        "reward_positions": data["reward_positions"],
        "reward_velocities": data["reward_velocities"],
        "reward_attitudes": data["reward_attitudes"],
        "reward_rates": data["reward_rates"],
        "reward_actions": data["reward_actions"],
        "reward_warnings": data["reward_warnings"],
        "reward_failures": data["reward_failures"],
        "reward_successes": data["reward_successes"],
        "reward_timeouts": data["reward_timeouts"],

        "normalized_actions": data["normalized_actions"],
        "physical_actions": data["physical_actions"],
    }


def plot_one_rollout(result, rollout_id):
    times = result["times"]
    states = result["states"]

    actions_raw = result["actions_raw"]
    actions_clipped = result["actions_clipped"]

    normalized_actions = result["normalized_actions"]
    physical_actions = result["physical_actions"]

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
    reward_warnings = result["reward_warnings"]
    reward_failures = result["reward_failures"]
    reward_successes = result["reward_successes"]
    reward_timeouts = result["reward_timeouts"]

    # State components
    rx = states[:, 0]
    ry = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    psi = states[:, 4]
    omega = states[:, 5]

    com_distance = np.sqrt(rx**2 + ry**2)
    com_speed = np.sqrt(vx**2 + vy**2)

    # History convention:
    # states has length N+1 because it includes the reset state.
    # step-based arrays have length N.
    t_states = times[: len(states)]
    t_steps = times[1 : 1 + len(port_distances)]

    title_suffix = (
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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_trajectory.png"))
    plt.close()

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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_position.png"))
    plt.close()

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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_velocity_states.png"))
    plt.close()

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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_port_distance.png"))
    plt.close()

    plt.figure()
    plt.plot(t_steps, port_speeds, label="port speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Port speed [m/s]")
    plt.title(f"SAC Final Approach Port Speed\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_port_speed.png"))
    plt.close()

    # ------------------------------------------------------------
    # 5. Final-approach geometry: s, rho, s_dot
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, s_values, label="s")
    plt.plot(t_steps, rho_values, label="rho")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title(f"SAC Final Approach Geometry\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_geometry_s_rho.png"))
    plt.close()

    plt.figure()
    plt.plot(t_steps, s_dot_values, label="s_dot")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("s_dot [m/s]")
    plt.title(f"SAC Final Approach Axial Closing Speed\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_s_dot.png"))
    plt.close()

    # ------------------------------------------------------------
    # 6. Attitude diagnostics
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, np.rad2deg(psi_e_values), label="psi_e")
    plt.plot(t_steps, np.rad2deg(omega_values), label="omega")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle / rate [deg], [deg/s]")
    plt.title(f"SAC Final Approach Alignment Diagnostics\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_alignment.png"))
    plt.close()

    # Also plot raw states psi and omega, just in case you want direct state values.
    plt.figure()
    plt.plot(t_states, np.rad2deg(psi), label="psi state")
    plt.plot(t_states, np.rad2deg(omega), label="omega state")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle / rate [deg], [deg/s]")
    plt.title(f"SAC Final Approach Attitude States\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_attitude_states.png"))
    plt.close()

    # ------------------------------------------------------------
    # 7. Physical body-frame actions
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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_physical_actions.png"))
    plt.close()

    # ------------------------------------------------------------
    # 8. Normalized SAC actions
    # ------------------------------------------------------------
    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(t_steps, normalized_actions[:, i], label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized action [-1, 1]")
    plt.title(f"SAC Final Approach Normalized Actions\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_normalized_actions.png"))
    plt.close()

    # ------------------------------------------------------------
    # 9. Raw vs clipped actions from core environment
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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_raw_vs_clipped_actions.png"))
    plt.close()

    # ------------------------------------------------------------
    # 10. Total SAC reward
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, sac_rewards, label="returned SAC reward")
    plt.plot(t_steps, reward_totals, linestyle="--", label="reward_total component")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward")
    plt.title(f"SAC Final Approach Reward\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_reward_total.png"))
    plt.close()

    # ------------------------------------------------------------
    # 11. Reward components
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, reward_positions, label="position")
    plt.plot(t_steps, reward_velocities, label="velocity")
    plt.plot(t_steps, reward_attitudes, label="attitude")
    plt.plot(t_steps, reward_rates, label="rate")
    plt.plot(t_steps, reward_actions, label="action")
    plt.plot(t_steps, reward_warnings, label="warning")
    plt.plot(t_steps, reward_failures, label="failure")
    plt.plot(t_steps, reward_successes, label="success")
    plt.plot(t_steps, reward_timeouts, label="timeout")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward component")
    plt.title(f"SAC Final Approach Reward Components\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_reward_components.png"))
    plt.close()

    # ------------------------------------------------------------
    # 12. Reward components without dominant position term
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_steps, reward_velocities, label="velocity")
    plt.plot(t_steps, reward_attitudes, label="attitude")
    plt.plot(t_steps, reward_rates, label="rate")
    plt.plot(t_steps, reward_actions, label="action")
    plt.plot(t_steps, reward_warnings, label="warning")
    plt.plot(t_steps, reward_failures, label="failure")
    plt.xlabel("Time [s]")
    plt.ylabel("Reward component")
    plt.title(f"SAC Reward Components Without Position Term\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_reward_components_no_position.png")
    )
    plt.close()

    # ------------------------------------------------------------
    # 13. Distance and speed summary
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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rollout_{rollout_id:03d}_distance_speed_summary.png"))
    plt.close()


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
            f"violated={list(result['violated_constraints'])}"
        )

        plot_one_rollout(result, rollout_id)

    print("\nDone.")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()