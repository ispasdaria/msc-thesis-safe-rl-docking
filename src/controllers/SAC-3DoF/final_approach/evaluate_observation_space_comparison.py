"""Evaluate and plot the three final-approach observation-space variants."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observation_space_env import FinalApproachObservationEnv


SEEDS = tuple(range(10))
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "results" / "observation_comparison" / "evaluation"

CONTROLLERS = {
    "com": {
        "label": "CoM-based observation space",
        "mode": "com",
        "model": MODEL_DIR / "observation_comparison" / "final_sac_com_observation.zip",
        "vecnorm": MODEL_DIR / "observation_comparison" / "final_sac_com_observation_vecnormalize.pkl",
    },
    "docking_port": {
        "label": "Docking-port-based observation space",
        "mode": "docking_port",
        "model": MODEL_DIR / "optuna12_100ksteps_buffer100k" / "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12.zip",
        "vecnorm": MODEL_DIR / "optuna12_100ksteps_buffer100k" / "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12_vecnormalize.pkl",
    },
    "combined": {
        "label": "Combined CoM + docking-port observation space",
        "mode": "combined",
        "model": MODEL_DIR / "observation_comparison" / "final_sac_combined_observation.zip",
        "vecnorm": MODEL_DIR / "observation_comparison" / "final_sac_combined_observation_vecnormalize.pkl",
    },
}


def make_env(observation_mode, seed=0):
    return Monitor(
        FinalApproachObservationEnv(
            observation_mode=observation_mode,
            mode="testing",
            dt=1.0,
            max_time=None,
            integrator="rk4",
            normalize_action=True,
            seed=seed,
        )
    )


def load_controller(spec):
    for key in ("model", "vecnorm"):
        if not spec[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {spec[key]}")

    dummy_env = DummyVecEnv([lambda: make_env(spec["mode"], seed=999)])
    vecnorm = VecNormalize.load(str(spec["vecnorm"]), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(str(spec["model"]), device="cpu")
    return model, vecnorm, dummy_env


def evaluate_one(model, vecnorm, observation_mode, seed, max_steps=5000):
    env = make_env(observation_mode, seed=seed)
    obs, reset_info = env.reset(seed=seed)
    total_reward = 0.0
    final_info = reset_info

    for step in range(max_steps):
        normalized_obs = vecnorm.normalize_obs(np.asarray(obs).reshape(1, -1))
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break

    raw_env = env.unwrapped
    times, states, actions_raw, actions_clipped, actions_orbital = (
        raw_env.get_history_arrays()
    )
    env.close()

    port_positions = []
    port_velocities = []
    com_positions = []
    com_velocities = []
    for state in states:
        quantities = raw_env.core_env.get_derived_quantities(state)
        port_positions.append(quantities["r_p"])
        port_velocities.append(quantities["v_p"])
        com_positions.append(quantities["r_com"])
        com_velocities.append(quantities["v_com"])

    port_positions = np.asarray(port_positions, dtype=float)
    port_velocities = np.asarray(port_velocities, dtype=float)
    com_positions = np.asarray(com_positions, dtype=float)
    com_velocities = np.asarray(com_velocities, dtype=float)

    terminal_position_error = float(np.linalg.norm(port_positions[-1]))
    terminal_velocity_error = float(np.linalg.norm(port_velocities[-1]))
    return {
        "seed": seed,
        "steps": int(step + 1),
        "return": total_reward,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)),
        "termination_reason": str(final_info.get("termination_reason", "")),
        "violated_constraints": list(final_info.get("violated_constraints", [])),
        "terminal_position_error": terminal_position_error,
        "terminal_velocity_error": terminal_velocity_error,
        "terminal_attitude_error_rad": float(abs(states[-1, 4])),
        "terminal_angular_rate_rad_s": float(abs(states[-1, 5])),
        "times": np.asarray(times),
        "states": np.asarray(states),
        "actions_raw": np.asarray(actions_raw),
        "actions_clipped": np.asarray(actions_clipped),
        "actions_orbital": np.asarray(actions_orbital),
        "port_positions": port_positions,
        "port_velocities": port_velocities,
        "com_positions": com_positions,
        "com_velocities": com_velocities,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scalar_row(controller_key, result):
    return {
        "controller": controller_key,
        "observation_space": CONTROLLERS[controller_key]["label"],
        "seed": result["seed"],
        "steps": result["steps"],
        "success": result["success"],
        "failure": result["failure"],
        "time_limit": result["time_limit"],
        "termination_reason": result["termination_reason"],
        "violated_constraints": ";".join(result["violated_constraints"]),
        "terminal_position_error_m": result["terminal_position_error"],
        "terminal_velocity_error_mps": result["terminal_velocity_error"],
        "terminal_attitude_error_deg": np.rad2deg(
            result["terminal_attitude_error_rad"]
        ),
        "terminal_angular_rate_deg_s": np.rad2deg(
            result["terminal_angular_rate_rad_s"]
        ),
        "episode_return": result["return"],
    }


def aggregate(controller_key, results):
    positions = np.asarray([r["terminal_position_error"] for r in results])
    velocities = np.asarray([r["terminal_velocity_error"] for r in results])
    attitude_errors = np.asarray(
        [r["terminal_attitude_error_rad"] for r in results]
    )
    angular_rates = np.asarray(
        [r["terminal_angular_rate_rad_s"] for r in results]
    )
    returns = np.asarray([r["return"] for r in results])
    return {
        "controller": controller_key,
        "observation_space": CONTROLLERS[controller_key]["label"],
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_terminal_position_error_m": float(np.mean(positions)),
        "median_terminal_position_error_m": float(np.median(positions)),
        "std_terminal_position_error_m": float(np.std(positions)),
        "mean_terminal_velocity_error_mps": float(np.mean(velocities)),
        "median_terminal_velocity_error_mps": float(np.median(velocities)),
        "std_terminal_velocity_error_mps": float(np.std(velocities)),
        "mean_terminal_attitude_error_deg": float(np.rad2deg(np.mean(attitude_errors))),
        "median_terminal_attitude_error_deg": float(np.rad2deg(np.median(attitude_errors))),
        "std_terminal_attitude_error_deg": float(np.rad2deg(np.std(attitude_errors))),
        "mean_terminal_angular_rate_deg_s": float(np.rad2deg(np.mean(angular_rates))),
        "median_terminal_angular_rate_deg_s": float(np.rad2deg(np.median(angular_rates))),
        "std_terminal_angular_rate_deg_s": float(np.rad2deg(np.std(angular_rates))),
        "mean_episode_return": float(np.mean(returns)),
        "median_episode_return": float(np.median(returns)),
        "std_episode_return": float(np.std(returns)),
        "unsuccessful_termination_causes": dict(
            Counter(
                r["termination_reason"]
                for r in results
                if not r["success"]
            )
        ),
    }


def choose_representative_seed(results_by_controller):
    port_results = results_by_controller["docking_port"]
    median_position = float(
        np.median([r["terminal_position_error"] for r in port_results])
    )
    return min(
        port_results,
        key=lambda result: (
            abs(result["terminal_position_error"] - median_position),
            abs(result["return"] - np.median([r["return"] for r in port_results])),
        ),
    )["seed"], median_position


def save_trajectory(result, path):
    np.savez_compressed(
        path,
        seed=result["seed"],
        times=result["times"],
        states=result["states"],
        port_positions=result["port_positions"],
        port_velocities=result["port_velocities"],
        com_positions=result["com_positions"],
        com_velocities=result["com_velocities"],
    )


def plot_trajectories(results_by_controller, representative_seed, path):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )
    colors = {
        "com": "#0072B2",  # Okabe-Ito blue
        "docking_port": "#D55E00",  # Okabe-Ito vermilion
        "combined": "#009E73",  # Okabe-Ito bluish green
    }
    panel_titles = {
        "com": "(a) CoM-based observation space",
        "docking_port": "(b) Docking-port-based observation space",
        "combined": "(c) Combined CoM + docking-port observation space",
    }
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    selected = {
        key: next(
            result
            for result in results
            if result["seed"] == representative_seed
        )
        for key, results in results_by_controller.items()
    }
    all_port = np.concatenate([result["port_positions"] for result in selected.values()])
    x_values = np.concatenate((all_port[:, 0], np.array([0.0])))
    y_values = np.concatenate((all_port[:, 1], np.array([0.0])))
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    x_pad = max(0.05 * (x_max - x_min), 1.0)
    y_pad = max(0.05 * (y_max - y_min), 1.0)
    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    docking_position_radius = 0.1

    for axis, (key, result) in zip(axes, selected.items()):
        port = result["port_positions"]
        color = colors[key]
        axis.plot(
            port[:, 0],
            port[:, 1],
            color=color,
            linewidth=2.0,
            label="Docking-port trajectory",
            zorder=3,
        )
        axis.scatter(
            port[0, 0],
            port[0, 1],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            marker="o",
            s=48,
            zorder=5,
            label="Start",
        )
        axis.scatter(
            port[-1, 0],
            port[-1, 1],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            marker="X",
            s=62,
            zorder=5,
            label="End",
        )
        axis.scatter(
            0.0,
            0.0,
            color="black",
            marker="+",
            s=90,
            linewidth=1.5,
            zorder=6,
            label="Docking target",
        )
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                docking_position_radius,
                facecolor="#56B4E9",
                edgecolor="#0072B2",
                linewidth=1.1,
                alpha=0.25,
                label=r"Docking position zone $||r_p|| <= 0.1$ m",
                zorder=1,
            )
        )
        zoom = inset_axes(axis, width="27%", height="27%", loc="lower right", borderpad=1.0)
        zoom.plot(port[:, 0], port[:, 1], color=color, linewidth=1.4, zorder=3)
        zoom.add_patch(
            Circle(
                (0.0, 0.0),
                docking_position_radius,
                facecolor="#56B4E9",
                edgecolor="#0072B2",
                linewidth=0.9,
                alpha=0.3,
                zorder=1,
            )
        )
        zoom.scatter(0.0, 0.0, color="black", marker="+", s=35, linewidth=1.0, zorder=4)
        zoom.scatter(port[-1, 0], port[-1, 1], color=color, marker="X", s=28, zorder=5)
        zoom.set_xlim(-0.3, 0.3)
        zoom.set_ylim(-0.3, 0.3)
        zoom.set_aspect("equal", adjustable="box")
        zoom.set_title("Docking zone", fontsize=7, pad=2)
        zoom.tick_params(labelsize=6, length=2)
        zoom.grid(True, color="0.85", linewidth=0.5)
        termination = result["termination_reason"].replace("_", " ")
        axis.set_title(
            f"{panel_titles[key]}\n(seed {representative_seed}; {termination})",
            pad=9,
            fontweight="normal",
        )
        axis.set_xlabel(r"Relative axial position $r_{p,x}$ [m]")
        axis.grid(True, color="0.82", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=3.5, width=0.7)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_facecolor("white")

    axes[0].set_ylabel(r"Relative lateral position $r_{p,y}$ [m]")
    legend_handles = [
        Line2D([0], [0], color="#0072B2", linewidth=2.0, label="Docking-port trajectory"),
        Line2D([0], [0], color="0.2", marker="o", markerfacecolor="white", linestyle="None", markersize=6, label="Start"),
        Line2D([0], [0], color="0.2", marker="X", markerfacecolor="0.2", linestyle="None", markersize=6, label="End"),
        Line2D([0], [0], color="black", marker="+", linestyle="None", markersize=9, label="Docking target"),
        Line2D([0], [0], color="#0072B2", marker="o", markerfacecolor="#56B4E9", alpha=0.7, linestyle="None", markersize=8, label=r"Docking position zone $||r_p|| <= 0.1$ m"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.20, wspace=0.12)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_port_distance_vs_timesteps(results_by_controller, representative_seed, path):
    colors = {
        "com": "#0072B2",
        "docking_port": "#D55E00",
        "combined": "#009E73",
    }
    labels = {
        "com": "CoM observation",
        "docking_port": "Docking-port observation",
        "combined": "Combined CoM + docking-port observation",
    }
    figure, axis = plt.subplots(figsize=(11.5, 6.2))

    selected = {
        key: next(
            result
            for result in results
            if result["seed"] == representative_seed
        )
        for key, results in results_by_controller.items()
    }
    for key, result in selected.items():
        steps = np.arange(len(result["port_positions"]))
        distance = np.linalg.norm(result["port_positions"], axis=1)
        axis.plot(
            steps,
            distance,
            color=colors[key],
            linewidth=2.0,
            label=labels[key],
        )

    axis.axhline(
        0.1,
        color="#56B4E9",
        linestyle=(0, (5, 2)),
        linewidth=1.2,
        label=r"Docking requirement $||r_p|| <= 0.1$ m",
    )
    axis.fill_between(
        [0, max(len(result["port_positions"]) for result in selected.values())],
        0,
        0.1,
        color="#56B4E9",
        alpha=0.12,
        zorder=0,
    )
    axis.set_xlabel("Time, t [s]")
    axis.set_ylabel(r"Docking-port distance $||r_p||$ [m]")
    axis.set_title(
        f"Docking-port distance during final-approach (testing seed {representative_seed})",
        pad=10,
        fontweight="normal",
    )
    axis.grid(True, color="0.82", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.18, 0.99),
        fontsize=9,
        borderaxespad=0.0,
    )

    terminal_steps = {
        key: len(result["port_positions"]) - 1
        for key, result in selected.items()
    }
    latest_terminal_step = max(terminal_steps.values())
    terminal_window = max(
        500,
        latest_terminal_step - min(terminal_steps.values()) + 150,
    )
    terminal_x_min = max(0, latest_terminal_step - terminal_window)
    terminal_x_max = latest_terminal_step + 50
    terminal_distance_max = max(
        float(np.linalg.norm(result["port_positions"][-1]))
        for result in selected.values()
    )
    terminal_y_max = max(0.3, terminal_distance_max * 1.2)

    zoom = inset_axes(axis, width="34%", height="35%", loc="upper right", borderpad=2.5)
    for key, result in selected.items():
        steps = np.arange(len(result["port_positions"]))
        distance = np.linalg.norm(result["port_positions"], axis=1)
        zoom.plot(steps, distance, color=colors[key], linewidth=1.3)
    zoom.axhline(0.1, color="#56B4E9", linestyle=(0, (5, 2)), linewidth=0.9)
    zoom.fill_between(
        [0, max(len(result["port_positions"]) for result in selected.values())],
        0,
        0.1,
        color="#56B4E9",
        alpha=0.12,
    )
    zoom.set_xlim(terminal_x_min, terminal_x_max)
    zoom.set_ylim(0.0, terminal_y_max)
    zoom.set_title("Terminal distance", fontsize=8, pad=2)
    zoom.tick_params(labelsize=7, length=2)
    zoom.grid(True, color="0.85", linewidth=0.5)

    figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.12)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "trajectories").mkdir(parents=True, exist_ok=True)
    results_by_controller = {}
    raw_rows = []
    aggregate_rows = []

    for controller_key, spec in CONTROLLERS.items():
        model, vecnorm, dummy_env = load_controller(spec)
        try:
            results = []
            for seed in SEEDS:
                result = evaluate_one(model, vecnorm, spec["mode"], seed)
                results.append(result)
                raw_rows.append(scalar_row(controller_key, result))
                save_trajectory(
                    result,
                    OUTPUT_DIR / "trajectories" / f"{controller_key}_seed{seed}.npz",
                )
            results_by_controller[controller_key] = results
            aggregate_rows.append(aggregate(controller_key, results))
        finally:
            dummy_env.close()

    raw_fields = list(raw_rows[0])
    write_csv(OUTPUT_DIR / "per_seed_results.csv", raw_rows, raw_fields)

    aggregate_fields = [
        key for key, value in aggregate_rows[0].items() if not isinstance(value, dict)
    ]
    aggregate_csv_rows = []
    for row in aggregate_rows:
        csv_row = {key: row[key] for key in aggregate_fields}
        csv_row["unsuccessful_termination_causes"] = json.dumps(
            row["unsuccessful_termination_causes"], sort_keys=True
        )
        aggregate_csv_rows.append(csv_row)
    write_csv(OUTPUT_DIR / "aggregate_results.csv", aggregate_csv_rows, aggregate_fields + ["unsuccessful_termination_causes"])

    representative_seed, port_median = choose_representative_seed(results_by_controller)
    representative = {
        "seed": representative_seed,
        "selection_rule": "Docking-port controller seed nearest the median terminal port-position error; return used as tie-break.",
        "docking_port_median_terminal_position_error_m": port_median,
    }
    (OUTPUT_DIR / "representative_selection.json").write_text(
        json.dumps(representative, indent=2), encoding="utf-8"
    )
    representative_rows = []
    for key, results in results_by_controller.items():
        result = next(item for item in results if item["seed"] == representative_seed)
        save_trajectory(result, OUTPUT_DIR / f"representative_{key}_seed{representative_seed}.npz")
        for index, (time, port, com) in enumerate(zip(result["times"], result["port_positions"], result["com_positions"])):
            representative_rows.append({
                "controller": key,
                "seed": representative_seed,
                "step": index,
                "time_s": time,
                "port_rx_m": port[0],
                "port_ry_m": port[1],
                "com_rx_m": com[0],
                "com_ry_m": com[1],
            })
    write_csv(
        OUTPUT_DIR / "representative_trajectory.csv",
        representative_rows,
        list(representative_rows[0]),
    )
    plot_trajectories(
        results_by_controller,
        representative_seed,
        OUTPUT_DIR / "observation_space_trajectory_comparison.pdf",
    )
    plot_port_distance_vs_timesteps(
        results_by_controller,
        representative_seed,
        OUTPUT_DIR / "docking_port_distance_vs_timesteps.pdf",
    )

    summary = {
        "models": {
            key: {"model": str(spec["model"]), "vecnormalize": str(spec["vecnorm"])}
            for key, spec in CONTROLLERS.items()
        },
        "testing_seeds": list(SEEDS),
        "aggregate": aggregate_rows,
        "representative": representative,
        "outputs": {
            "per_seed": str(OUTPUT_DIR / "per_seed_results.csv"),
            "aggregate": str(OUTPUT_DIR / "aggregate_results.csv"),
            "figure_pdf": str(OUTPUT_DIR / "observation_space_trajectory_comparison.pdf"),
            "distance_figure_pdf": str(OUTPUT_DIR / "docking_port_distance_vs_timesteps.pdf"),
        },
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
