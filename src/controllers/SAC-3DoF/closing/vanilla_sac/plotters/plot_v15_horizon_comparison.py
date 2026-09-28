"""Compare one frozen closing-SAC policy at three evaluation horizons.

The representative seed is selected from the existing 12,000-s, seeds 0--49
evaluation as the rollout whose terminal position error is closest to the
50-rollout mean.  Only that seed is then replayed deterministically; no model
parameters or normalization statistics are changed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOSING_DIR = os.path.dirname(SCRIPT_DIR)
SRC_PATH = os.path.abspath(os.path.join(CLOSING_DIR, "..", "..", ".."))
for path in (SCRIPT_DIR, CLOSING_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluate_sac_closing as evaluator


MODEL_NAME = "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k"
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "reward_shaping_trials")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")
METRICS_12000 = os.path.join(
    SCRIPT_DIR,
    "results",
    "timestep_transfer",
    "article_trial16_400k_seeds0_49",
    "native_1hz",
    "rollout_metrics.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "retained_figures",
    "horizon_comparison",
)
HORIZONS = (12000, 13000, 20000)

TRAJECTORY_COLOR = "#0072B2"
START_COLOR = "#009E73"
END_COLOR = "#D55E00"
HORIZON_COLORS = {
    12000: "#0072B2",
    13000: "#D55E00",
    20000: "#009E73",
}
CONE_COLOR = "#E69F00"
CORRIDOR_COLOR = "#A06AB4"
AXIS_COLOR = "#6B6B6B"


def select_representative_seed(metrics_path: str) -> tuple[int, float, float]:
    """Select the 12,000-s rollout closest to the mean terminal distance."""

    with open(metrics_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rollout rows found in {metrics_path}")

    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    distances = np.asarray(
        [float(row["final_position_error_m"]) for row in rows], dtype=float
    )
    expected = np.arange(50, dtype=int)
    if not np.array_equal(np.sort(seeds), expected):
        raise ValueError("The 12,000-s metrics must contain exactly seeds 0--49.")

    mean_distance = float(np.mean(distances))
    selected_index = int(np.argmin(np.abs(distances - mean_distance)))
    return int(seeds[selected_index]), mean_distance, float(distances[selected_index])


def run_rollout(model, vecnorm, seed: int, horizon: int) -> dict:
    """Run one deterministic testing rollout and return its full history."""

    env = evaluator.Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=float(horizon),
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version="v15",
    )
    obs, _ = env.reset(seed=seed)
    p3_target = np.asarray(env.core_env.p3_target, dtype=float).copy()
    p3_port = np.asarray(
        env.core_env.get_derived_quantities(p3_target)["r_p"], dtype=float
    )
    final_info = {}

    for _ in range(horizon):
        normalized_obs = evaluator.normalize_observation(vecnorm, obs)
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, _, terminated, truncated, final_info = env.step(action[0])
        if terminated or truncated:
            break

    times, states, *_ = env.get_history_arrays()
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    errors = states[:, :2] - p3_target[:2]
    distances = np.linalg.norm(errors, axis=1)
    terminal_attitude = float(
        np.rad2deg(
            abs(
                np.arctan2(
                    np.sin(states[-1, 4] - p3_target[4]),
                    np.cos(states[-1, 4] - p3_target[4]),
                )
            )
        )
    )
    result = {
        "times": times,
        "states": states,
        "errors": errors,
        "p3_target": p3_target,
        "p3_port": p3_port,
        "minimum_position_error_m": float(np.min(distances)),
        "terminal_position_error_m": float(distances[-1]),
        "terminal_attitude_error_deg": terminal_attitude,
        "termination_reason": str(final_info.get("termination_reason", "horizon")),
        "cone_slope": float(env.core_env.cone_slope),
        "cone_length": float(env.core_env.cone_length),
        "corridor_radius": float(env.core_env.corridor_radius),
        "d_p3": float(env.core_env.d_P3),
    }
    env.close()
    return result


def geometry_lines(record: dict, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express physical docking-cone boundaries in P3-relative coordinates."""

    absolute_axial = x_values + record["p3_port"][0]
    upper = record["cone_slope"] * absolute_axial - record["p3_port"][1]
    lower = -record["cone_slope"] * absolute_axial - record["p3_port"][1]
    valid = (absolute_axial >= 0.0) & (absolute_axial <= record["cone_length"])
    return np.where(valid, upper, np.nan), np.where(valid, lower, np.nan)


def save_trajectory_data(path: str, seed: int, records: dict[int, dict]) -> None:
    payload = {"seed": np.asarray(seed, dtype=int), "horizons_s": np.asarray(HORIZONS)}
    for horizon, record in records.items():
        prefix = f"h{horizon}"
        payload[f"{prefix}_times_s"] = record["times"]
        payload[f"{prefix}_states"] = record["states"]
        payload[f"{prefix}_p3_relative_xy_m"] = record["errors"]
        payload[f"{prefix}_p3_target"] = record["p3_target"]
    np.savez_compressed(path, **payload)


def plot(records: dict[int, dict], seed: int, png_path: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )

    all_errors = np.vstack([record["errors"] for record in records.values()])
    x_min, x_max = float(np.min(all_errors[:, 0])), float(np.max(all_errors[:, 0]))
    x_padding = 0.035 * (x_max - x_min)
    display_x_min = min(x_min - x_padding, -10.0)
    display_x_max = x_max + x_padding

    # Include trajectories and relevant cone/corridor geometry in shared y limits.
    geometry_x = np.linspace(display_x_min, display_x_max, 1500)
    y_candidates = [all_errors[:, 1]]
    first = records[HORIZONS[0]]
    cone_upper, cone_lower = geometry_lines(first, geometry_x)
    y_candidates.extend([cone_upper[np.isfinite(cone_upper)], cone_lower[np.isfinite(cone_lower)]])
    y_all = np.concatenate(y_candidates)
    y_limit = 1.08 * max(abs(float(np.min(y_all))), abs(float(np.max(y_all))), 8.0)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        record = records[horizon]
        errors = record["errors"]
        cone_upper, cone_lower = geometry_lines(record, geometry_x)

        axis.axhline(0.0, color=AXIS_COLOR, linewidth=1.0, linestyle="-.", zorder=0)
        axis.plot(geometry_x, cone_upper, color=CONE_COLOR, linewidth=1.35, linestyle=":", zorder=1)
        axis.plot(geometry_x, cone_lower, color=CONE_COLOR, linewidth=1.35, linestyle=":", zorder=1)

        corridor_x = np.asarray(
            [-record["p3_port"][0], record["d_p3"] - record["p3_port"][0]]
        )
        for sign in (-1.0, 1.0):
            corridor_y = sign * record["corridor_radius"] - record["p3_port"][1]
            axis.plot(
                corridor_x,
                [corridor_y, corridor_y],
                color=CORRIDOR_COLOR,
                linewidth=1.35,
                linestyle=":",
                zorder=1,
            )

        axis.plot(errors[:, 0], errors[:, 1], color=TRAJECTORY_COLOR, linewidth=1.8, label="trajectory", zorder=3)
        axis.scatter(errors[0, 0], errors[0, 1], marker="D", s=42, facecolor="white", edgecolor=START_COLOR, linewidth=1.4, zorder=5)
        axis.scatter(errors[-1, 0], errors[-1, 1], marker="X", s=54, color=END_COLOR, edgecolor="white", linewidth=0.5, zorder=6)
        axis.scatter(0.0, 0.0, marker="*", s=95, color="#CC3311", edgecolor="white", linewidth=0.5, zorder=7)
        axis.set_title(f"{horizon:,}-s horizon", fontweight="bold", pad=8)
        axis.set_xlabel("Axial position error [m]")
        axis.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
        axis.set_axisbelow(True)
        axis.text(
            0.035,
            0.965,
            "$d_{\\min}$ = " + f"{record['minimum_position_error_m']:.1f} m\n"
            "$d_T$ = " + f"{record['terminal_position_error_m']:.1f} m\n"
            "$|e_{\\psi,T}|$ = " + f"{record['terminal_attitude_error_deg']:.1f}°",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88, "pad": 3.0},
            zorder=8,
        )
        axis.set_xlim(display_x_max, display_x_min)
        axis.set_ylim(-y_limit, y_limit)

    axes[0].set_ylabel("Lateral position error [m]")
    fig.suptitle(
        "Closing-phase vanilla SAC trajectory under extended evaluation horizons\n"
        f"Representative deterministic rollout (testing seed {seed})",
        fontweight="bold",
        y=1.04,
    )
    handles = [
        Line2D([0], [0], color=TRAJECTORY_COLOR, linewidth=1.8, label="trajectory"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="white", markeredgecolor=START_COLOR, label="start"),
        Line2D([0], [0], marker="X", linestyle="None", color=END_COLOR, label="terminal point"),
        Line2D([0], [0], marker="*", linestyle="None", color="#CC3311", markersize=10, label="$P_3$ target"),
        Line2D([0], [0], color=AXIS_COLOR, linestyle="-.", label="docking axis"),
        Line2D([0], [0], color=CONE_COLOR, linestyle=":", linewidth=1.5, label="docking cone"),
        Line2D([0], [0], color=CORRIDOR_COLOR, linestyle=":", linewidth=1.5, label="docking corridor"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=7, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.13, top=0.79, wspace=0.08)
    fig.savefig(png_path, dpi=350, facecolor="white")
    plt.close(fig)


def plot_overlay(records: dict[int, dict], seed: int, png_path: str) -> None:
    """Plot all evaluation horizons together, with horizon-consistent markers."""

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )

    all_errors = np.vstack([record["errors"] for record in records.values()])
    x_min, x_max = float(np.min(all_errors[:, 0])), float(np.max(all_errors[:, 0]))
    x_padding = 0.035 * (x_max - x_min)
    display_x_min = min(x_min - x_padding, -10.0)
    display_x_max = x_max + x_padding

    geometry_x = np.linspace(display_x_min, display_x_max, 1500)
    first = records[HORIZONS[0]]
    cone_upper, cone_lower = geometry_lines(first, geometry_x)
    finite_geometry = [
        cone_upper[np.isfinite(cone_upper)],
        cone_lower[np.isfinite(cone_lower)],
    ]
    y_all = np.concatenate([all_errors[:, 1], *finite_geometry])
    y_limit = 1.08 * max(abs(float(np.min(y_all))), abs(float(np.max(y_all))), 8.0)

    fig, axis = plt.subplots(figsize=(8.7, 5.6))
    axis.axhline(0.0, color=AXIS_COLOR, linewidth=1.0, linestyle="-.", zorder=0)
    axis.plot(geometry_x, cone_upper, color=CONE_COLOR, linewidth=1.35, linestyle=":", zorder=1)
    axis.plot(geometry_x, cone_lower, color=CONE_COLOR, linewidth=1.35, linestyle=":", zorder=1)

    corridor_x = np.asarray(
        [-first["p3_port"][0], first["d_p3"] - first["p3_port"][0]]
    )
    for sign in (-1.0, 1.0):
        corridor_y = sign * first["corridor_radius"] - first["p3_port"][1]
        axis.plot(
            corridor_x,
            [corridor_y, corridor_y],
            color=CORRIDOR_COLOR,
            linewidth=1.35,
            linestyle=":",
            zorder=1,
        )

    # Draw longer rollouts first so each shorter horizon remains visible along
    # the otherwise identical prefix of the deterministic trajectory.
    for horizon in reversed(HORIZONS):
        errors = records[horizon]["errors"]
        color = HORIZON_COLORS[horizon]
        axis.plot(errors[:, 0], errors[:, 1], color=color, linewidth=1.8, zorder=3)
        axis.scatter(
            errors[0, 0], errors[0, 1], marker="D",
            s={12000: 34, 13000: 52, 20000: 72}[horizon],
            facecolor="white", edgecolor=color, linewidth=1.4, zorder=5,
        )
        axis.scatter(
            errors[-1, 0], errors[-1, 1], marker="X", s=58,
            color=color, edgecolor="white", linewidth=0.5, zorder=6,
        )

    axis.scatter(
        0.0, 0.0, marker="*", s=95, color="#CC3311",
        edgecolor="white", linewidth=0.5, zorder=7,
    )
    axis.set_xlabel(r"Axial position error to $P_3$ [m]")
    axis.set_ylabel(r"Lateral position error to $P_3$ [m]")
    axis.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlim(display_x_max, display_x_min)
    axis.set_ylim(-y_limit, y_limit)

    fig.suptitle(
        "Closing-phase vanilla SAC trajectories under extended evaluation horizons\n"
        f"(testing seed {seed})",
        fontweight="normal",
        y=0.98,
    )
    handles = [
        *[
            Line2D(
                [0], [0], color=HORIZON_COLORS[horizon], linewidth=1.8,
                label=f"{horizon:,}-s horizon",
            )
            for horizon in HORIZONS
        ],
        Line2D(
            [0], [0], marker="D", linestyle="None", markerfacecolor="white",
            markeredgecolor="0.25", label="start",
        ),
        Line2D(
            [0], [0], marker="X", linestyle="None", color="0.25",
            label="terminal point",
        ),
        Line2D([0], [0], marker="*", linestyle="None", color="#CC3311", markersize=10, label="$P_3$ target"),
        Line2D([0], [0], color=AXIS_COLOR, linestyle="-.", label="docking axis"),
        Line2D([0], [0], color=CONE_COLOR, linestyle=":", linewidth=1.5, label="docking cone"),
        Line2D([0], [0], color=CORRIDOR_COLOR, linestyle=":", linewidth=1.5, label="docking corridor"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=5, frameon=False,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.76)
    fig.savefig(png_path, dpi=350, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for required_path in (MODEL_PATH, VECNORM_PATH, METRICS_12000):
        if not os.path.exists(required_path):
            raise FileNotFoundError(required_path)
    os.makedirs(args.output_dir, exist_ok=True)

    seed, mean_12000, selected_12000 = select_representative_seed(METRICS_12000)
    evaluator.MODEL_NAME = MODEL_NAME
    evaluator.MODEL_DIR = MODEL_DIR
    evaluator.MODEL_PATH = MODEL_PATH
    evaluator.VECNORM_PATH = VECNORM_PATH
    with contextlib.redirect_stdout(io.StringIO()):
        vecnorm, dummy_env = evaluator.load_vecnormalize_stats()
        model = evaluator.SAC.load(MODEL_PATH)
    records = {horizon: run_rollout(model, vecnorm, seed, horizon) for horizon in HORIZONS}
    dummy_env.close()

    stem = os.path.join(args.output_dir, "closing_vanilla_sac_horizon_trajectory_comparison")
    png_path, data_path = stem + ".png", stem + "_data.npz"
    save_trajectory_data(data_path, seed, records)
    plot_overlay(records, seed, png_path)

    print(f"selected_seed={seed}")
    print("selection_reason=12,000-s terminal position error closest to the seeds 0--49 mean")
    print(f"mean_12000_terminal_position_error_m={mean_12000:.9f}")
    print(f"selected_12000_csv_terminal_position_error_m={selected_12000:.9f}")
    for horizon in HORIZONS:
        record = records[horizon]
        print(
            f"horizon_{horizon}: minimum_position_error_m="
            f"{record['minimum_position_error_m']:.9f}, terminal_position_error_m="
            f"{record['terminal_position_error_m']:.9f}, terminal_attitude_error_deg="
            f"{record['terminal_attitude_error_deg']:.9f}, termination_reason="
            f"{record['termination_reason']}"
        )
    print(f"png={png_path}")
    print(f"trajectory_data={data_path}")
    print(f"model={MODEL_PATH}")
    print(f"vecnormalize={VECNORM_PATH}")


if __name__ == "__main__":
    main()
