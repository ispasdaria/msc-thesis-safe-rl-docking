"""Overlay docking-port distance histories for 50 final-approach rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
SWEEP_DIR = SCRIPT_DIR / "tuning_lqr_final_approach"
for import_path in (SCRIPT_DIR, SRC_DIR, SWEEP_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_final import (
    LQRFinalController,
    build_linearized_3dof_matrices,
    build_port_error_mapping,
    build_Q_final,
    build_R_final,
    compute_lqr_gain,
)
from lqr_sweep_final import compute_final_error_history, run_single_rollout


RESULT_DIR = SCRIPT_DIR / "results" / "lqr_final"
DEFAULT_CACHE = RESULT_DIR / "final_approach_0_49_port_distance_histories.npz"
DEFAULT_OUTPUT = RESULT_DIR / "final_approach_convergence"
TRAJECTORY_COLOR = "#0072B2"
MEDIAN_COLOR = "#003B5C"
TOLERANCE_COLOR = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all final-approach docking-port distance histories."
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=5000.0)
    parser.add_argument(
        "--mode", choices=("nominal", "testing", "training"), default="testing"
    )
    parser.add_argument("--dock-tolerance", type=float, default=0.1)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rerun all simulations even if the trajectory cache exists.",
    )
    return parser.parse_args()


def build_controller(max_time: float) -> LQRFinalController:
    env = Docking3DoFEnv(phase="final", max_time=max_time, seed=0)
    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_port = build_Q_final()
    Q_com = H.T @ Q_port @ H
    gain, _ = compute_lqr_gain(A, B, Q_com, build_R_final(env))
    return LQRFinalController(K=gain, H=H)


def collect_histories(args: argparse.Namespace) -> dict[str, np.ndarray]:
    controller = build_controller(args.max_time)
    seeds: list[int] = []
    times: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    axial_velocities: list[np.ndarray] = []
    successes: list[bool] = []
    reasons: list[str] = []

    for rollout_index in range(args.rollouts):
        seed = args.seed_start + rollout_index
        env, _, info, _ = run_single_rollout(
            controller=controller,
            max_time=args.max_time,
            seed=seed,
            mode=args.mode,
        )
        rollout_times, states, _, _, _ = env.get_history_arrays()
        error_history = compute_final_error_history(env, states)
        port_distance = np.linalg.norm(error_history[:, :2], axis=1)
        axial_port_velocity = error_history[:, 2]

        seeds.append(seed)
        times.append(np.asarray(rollout_times, dtype=float))
        distances.append(np.asarray(port_distance, dtype=float))
        axial_velocities.append(np.asarray(axial_port_velocity, dtype=float))
        successes.append(bool(info.get("success", False)))
        reasons.append(str(info.get("termination_reason", "unknown")))
        if (rollout_index + 1) % 5 == 0 or rollout_index + 1 == args.rollouts:
            print(f"Completed {rollout_index + 1}/{args.rollouts} rollouts")

    max_length = max(array.size for array in times)
    padded_times = np.full((args.rollouts, max_length), np.nan, dtype=float)
    padded_distances = np.full((args.rollouts, max_length), np.nan, dtype=float)
    padded_axial_velocities = np.full(
        (args.rollouts, max_length), np.nan, dtype=float
    )
    for index, (rollout_times, rollout_distances, rollout_velocities) in enumerate(
        zip(times, distances, axial_velocities)
    ):
        padded_times[index, : rollout_times.size] = rollout_times
        padded_distances[index, : rollout_distances.size] = rollout_distances
        padded_axial_velocities[index, : rollout_velocities.size] = rollout_velocities

    return {
        "seeds": np.asarray(seeds, dtype=int),
        "times": padded_times,
        "port_distances": padded_distances,
        "axial_port_velocities": padded_axial_velocities,
        "success": np.asarray(successes, dtype=bool),
        "termination_reasons": np.asarray(reasons, dtype="U32"),
        "dock_tolerance": np.asarray(args.dock_tolerance, dtype=float),
    }


def write_cache(path: Path, histories: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **histories)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def make_figure(histories: dict[str, np.ndarray]) -> plt.Figure:
    seeds = np.asarray(histories["seeds"], dtype=int)
    times = np.asarray(histories["times"], dtype=float)
    distances = np.asarray(histories["port_distances"], dtype=float)
    axial_velocities = np.asarray(histories["axial_port_velocities"], dtype=float)
    success = np.asarray(histories["success"], dtype=bool)
    tolerance = float(np.asarray(histories["dock_tolerance"]))

    valid_counts = np.sum(np.isfinite(times), axis=1)
    completion_seconds = np.asarray(
        [times[index, count - 1] for index, count in enumerate(valid_counts)],
        dtype=float,
    )
    median_seconds = float(np.median(completion_seconds))
    median_index = int(np.argmin(np.abs(completion_seconds - median_seconds)))

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (time_ax, phase_ax) = plt.subplots(
        1, 2, figsize=(13.2, 5.25), sharey=True
    )

    for index in range(seeds.size):
        count = valid_counts[index]
        time_ax.plot(
            times[index, :count] / 60.0,
            distances[index, :count],
            color=TRAJECTORY_COLOR,
            linewidth=0.85,
            alpha=0.18,
            zorder=2,
        )
        time_ax.scatter(
            times[index, count - 1] / 60.0,
            distances[index, count - 1],
            s=8,
            color=TRAJECTORY_COLOR,
            alpha=0.24,
            edgecolor="none",
            zorder=3,
        )
        phase_ax.plot(
            axial_velocities[index, :count],
            distances[index, :count],
            color=TRAJECTORY_COLOR,
            linewidth=0.85,
            alpha=0.18,
            zorder=2,
        )
        phase_ax.scatter(
            axial_velocities[index, count - 1],
            distances[index, count - 1],
            s=8,
            color=TRAJECTORY_COLOR,
            alpha=0.24,
            edgecolor="none",
            zorder=3,
        )

    median_count = valid_counts[median_index]
    time_ax.plot(
        times[median_index, :median_count] / 60.0,
        distances[median_index, :median_count],
        color=MEDIAN_COLOR,
        linewidth=2.2,
        alpha=0.95,
        zorder=4,
    )
    time_ax.scatter(
        times[median_index, median_count - 1] / 60.0,
        distances[median_index, median_count - 1],
        s=42,
        marker="o",
        facecolor=MEDIAN_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    phase_ax.plot(
        axial_velocities[median_index, :median_count],
        distances[median_index, :median_count],
        color=MEDIAN_COLOR,
        linewidth=2.2,
        alpha=0.95,
        zorder=4,
    )
    phase_ax.scatter(
        axial_velocities[median_index, median_count - 1],
        distances[median_index, median_count - 1],
        s=42,
        marker="o",
        facecolor=MEDIAN_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    lower_limit = min(0.07, 0.8 * np.nanmin(distances))
    for ax in (time_ax, phase_ax):
        ax.axhspan(
            lower_limit, tolerance, color=TOLERANCE_COLOR, alpha=0.13, zorder=1
        )
        ax.axhline(
            tolerance,
            color="#007A59",
            linewidth=1.4,
            linestyle="--",
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_ylim(lower_limit, 1.12 * np.nanmax(distances))
        ax.grid(True, which="both", color="#D9D9D9", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)

    time_ax.set_xlim(0.0, 1.03 * np.max(completion_seconds) / 60.0)
    time_ax.set_xlabel("Time [min]")
    time_ax.set_ylabel(r"Docking-port distance $\|\mathbf{r}_p\|$ [m]")
    time_ax.set_title(
        "(a) Docking-port distance versus time", loc="left", fontweight="bold"
    )
    phase_ax.axvline(0.0, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
    phase_ax.set_xlabel(r"Axial port velocity $v_{p,x}$ [m/s]")
    phase_ax.set_title(
        "(b) Docking-port distance versus axial port velocity",
        loc="left",
        fontweight="bold",
    )

    completion_minutes = completion_seconds / 60.0
    time_ax.text(
        0.035,
        0.055,
        (
            f"{int(np.sum(success))}/{success.size} successful\n"
            f"Completion time: median {np.median(completion_minutes):.1f} min "
            f"(range {np.min(completion_minutes):.1f}–{np.max(completion_minutes):.1f} min)"
        ),
        transform=time_ax.transAxes,
        fontsize=9.2,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4},
        zorder=6,
    )

    legend_handles = [
        Line2D(
            [], [], color=TRAJECTORY_COLOR, linewidth=1.2, alpha=0.3,
            label="Individual rollouts"
        ),
        Line2D(
            [], [], color=MEDIAN_COLOR, linewidth=2.2,
            label=f"Median-duration rollout (seed {seeds[median_index]})"
        ),
        Patch(
            facecolor=TOLERANCE_COLOR, edgecolor="#007A59", linestyle="--",
            alpha=0.13, label="0.1 m docking tolerance"
        ),
    ]
    fig.suptitle(
        "Final-approach convergence across 50 rollouts",
        x=0.075,
        y=0.985,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.94),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.82, wspace=0.12)
    return fig


def main() -> None:
    args = parse_args()
    histories = None
    if not args.recompute and args.cache.exists():
        histories = load_cache(args.cache)

    if histories is None or "axial_port_velocities" not in histories:
        histories = collect_histories(args)
        write_cache(args.cache, histories)
        print(f"Wrote {args.cache}")
    else:
        print(f"Loaded {args.cache}")

    figure = make_figure(histories)
    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    success = np.asarray(histories["success"], dtype=bool)
    lengths = np.sum(np.isfinite(histories["times"]), axis=1)
    completion = np.asarray(
        [histories["times"][index, count - 1] for index, count in enumerate(lengths)]
    )
    print(f"Successful rollouts: {np.sum(success)}/{success.size}")
    print(
        f"Completion time [min]: median={np.median(completion) / 60.0:.3f}, "
        f"range=[{np.min(completion) / 60.0:.3f}, {np.max(completion) / 60.0:.3f}]"
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
