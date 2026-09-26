"""Plot final-approach docking-requirement histories across many rollouts.

The four panels show the docking-port position and velocity norms, absolute
attitude error, and absolute angular-rate error.  Each panel uses a linear
scale fitted to its rollout histories.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
DEFAULT_CACHE = RESULT_DIR / "final_approach_seeds_0_49_requirement_histories.npz"
DEFAULT_OUTPUT = RESULT_DIR / "final_approach_seeds_0_49_requirements"

TRAJECTORY_COLOR = "tab:red"
REPRESENTATIVE_COLOR = "tab:orange"

REQUIRED_CACHE_FIELDS = {
    "seeds",
    "times",
    "port_position_errors",
    "port_velocity_errors",
    "attitude_errors_deg",
    "angular_rate_errors_deg_s",
    "success",
    "termination_reasons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot four docking-requirement histories for final-approach LQR "
            "rollouts."
        )
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=5000.0)
    parser.add_argument(
        "--mode", choices=("nominal", "testing", "training"), default="testing"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; PNG and PDF are written.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rerun the simulations even when a compatible cache exists.",
    )
    args = parser.parse_args()
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")
    if args.max_time <= 0.0:
        parser.error("--max-time must be positive")
    return args


def build_controller(max_time: float) -> LQRFinalController:
    env = Docking3DoFEnv(phase="final", max_time=max_time, seed=0)
    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_com = H.T @ build_Q_final() @ H
    gain, _ = compute_lqr_gain(A, B, Q_com, build_R_final(env))
    return LQRFinalController(K=gain, H=H)


def pad_histories(histories: Sequence[np.ndarray]) -> np.ndarray:
    max_length = max(history.size for history in histories)
    padded = np.full((len(histories), max_length), np.nan, dtype=float)
    for index, history in enumerate(histories):
        padded[index, : history.size] = history
    return padded


def collect_histories(args: argparse.Namespace) -> dict[str, np.ndarray]:
    controller = build_controller(args.max_time)
    seeds: list[int] = []
    times: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    attitude_errors: list[np.ndarray] = []
    angular_rate_errors: list[np.ndarray] = []
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
        errors = compute_final_error_history(env, states)

        seeds.append(seed)
        times.append(np.asarray(rollout_times, dtype=float))
        position_errors.append(np.linalg.norm(errors[:, 0:2], axis=1))
        velocity_errors.append(np.linalg.norm(errors[:, 2:4], axis=1))
        attitude_errors.append(np.abs(np.rad2deg(errors[:, 4])))
        angular_rate_errors.append(np.abs(np.rad2deg(errors[:, 5])))
        successes.append(bool(info.get("success", False)))
        reasons.append(str(info.get("termination_reason", "unknown")))

        completed = rollout_index + 1
        if completed % 5 == 0 or completed == args.rollouts:
            print(f"Completed {completed}/{args.rollouts} rollouts")

    return {
        "seeds": np.asarray(seeds, dtype=int),
        "times": pad_histories(times),
        "port_position_errors": pad_histories(position_errors),
        "port_velocity_errors": pad_histories(velocity_errors),
        "attitude_errors_deg": pad_histories(attitude_errors),
        "angular_rate_errors_deg_s": pad_histories(angular_rate_errors),
        "success": np.asarray(successes, dtype=bool),
        "termination_reasons": np.asarray(reasons, dtype="U32"),
        "mode": np.asarray(args.mode),
        "max_time": np.asarray(args.max_time, dtype=float),
    }


def write_cache(path: Path, histories: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **histories)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def cache_matches(
    histories: dict[str, np.ndarray], args: argparse.Namespace
) -> bool:
    if not REQUIRED_CACHE_FIELDS.issubset(histories):
        return False
    expected_seeds = np.arange(
        args.seed_start, args.seed_start + args.rollouts, dtype=int
    )
    if not np.array_equal(np.asarray(histories["seeds"], dtype=int), expected_seeds):
        return False
    if "mode" in histories and str(np.asarray(histories["mode"]).item()) != args.mode:
        return False
    if "max_time" in histories and not np.isclose(
        float(np.asarray(histories["max_time"])), args.max_time
    ):
        return False
    return True


def representative_rollout_index(times: np.ndarray) -> int:
    valid_counts = np.sum(np.isfinite(times), axis=1)
    completion_times = np.asarray(
        [times[index, count - 1] for index, count in enumerate(valid_counts)],
        dtype=float,
    )
    return int(np.argmin(np.abs(completion_times - np.median(completion_times))))


def make_figure(histories: dict[str, np.ndarray]) -> plt.Figure:
    seeds = np.asarray(histories["seeds"], dtype=int)
    times = np.asarray(histories["times"], dtype=float)
    success = np.asarray(histories["success"], dtype=bool)
    valid_counts = np.sum(np.isfinite(times), axis=1)
    representative_index = representative_rollout_index(times)

    panel_specs = (
        (
            "port_position_errors",
            r"Docking-port position error $\|\mathbf{r}_p(t)\|$ [m]",
            "(a) Docking-port position error",
        ),
        (
            "port_velocity_errors",
            r"Docking-port velocity $\|\mathbf{v}_p(t)\|$ [m/s]",
            "(b) Docking-port velocity",
        ),
        (
            "attitude_errors_deg",
            r"Attitude error $|e_\psi(t)|$ [deg]",
            "(c) Attitude error",
        ),
        (
            "angular_rate_errors_deg_s",
            r"Angular-rate error $|e_\omega(t)|$ [deg/s]",
            "(d) Angular-rate error",
        ),
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes_grid = plt.subplots(
        2, 2, figsize=(9.4, 6.8), sharex=True, constrained_layout=False
    )
    axes = axes_grid.ravel()

    for ax, (field, ylabel, title) in zip(axes, panel_specs):
        values = np.asarray(histories[field], dtype=float)
        for rollout_index in range(seeds.size):
            count = valid_counts[rollout_index]
            ax.plot(
                times[rollout_index, :count] / 60.0,
                values[rollout_index, :count],
                color=TRAJECTORY_COLOR,
                linewidth=0.8,
                alpha=0.14,
                zorder=2,
            )

        representative_count = valid_counts[representative_index]
        ax.plot(
            times[representative_index, :representative_count] / 60.0,
            values[representative_index, :representative_count],
            color=REPRESENTATIVE_COLOR,
            linestyle="--",
            linewidth=1.8,
            alpha=1.0,
            zorder=3,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.28)
        ax.set_axisbelow(True)

    for ax in axes_grid[-1, :]:
        ax.set_xlabel("Time [min]")
    axes[-1].set_xlim(
        0.0,
        1.01
        * max(
            times[index, count - 1]
            for index, count in enumerate(valid_counts)
        )
        / 60.0,
    )

    fig.suptitle(
        (
            "Final-approach docking performance: "
            f"{int(np.sum(success))}/{success.size} successful rollouts "
            f"(seeds {seeds[0]}-{seeds[-1]})"
        ),
        fontsize=12,
        y=0.985,
    )
    figure_handles = (
        Line2D(
            [],
            [],
            color=TRAJECTORY_COLOR,
            linewidth=1.5,
            alpha=0.55,
            label="Individual rollouts",
        ),
        Line2D(
            [],
            [],
            color=REPRESENTATIVE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=(
                "Rollout nearest median duration "
                f"(seed {seeds[representative_index]})"
            ),
        ),
    )
    fig.legend(
        handles=figure_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.16,
        top=0.91,
        wspace=0.28,
        hspace=0.30,
    )
    return fig


def print_summary(histories: dict[str, np.ndarray]) -> None:
    seeds = np.asarray(histories["seeds"], dtype=int)
    times = np.asarray(histories["times"], dtype=float)
    success = np.asarray(histories["success"], dtype=bool)
    valid_counts = np.sum(np.isfinite(times), axis=1)
    completion_minutes = np.asarray(
        [times[index, count - 1] for index, count in enumerate(valid_counts)]
    ) / 60.0

    print(f"Evaluation seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Successful rollouts: {int(np.sum(success))}/{success.size}")
    print(
        "Completion time [min]: "
        f"median={np.median(completion_minutes):.3f}, "
        f"range=[{np.min(completion_minutes):.3f}, "
        f"{np.max(completion_minutes):.3f}]"
    )


def main() -> None:
    args = parse_args()
    histories = None
    if not args.recompute and args.cache.exists():
        candidate = load_cache(args.cache)
        if cache_matches(candidate, args):
            histories = candidate
            print(f"Loaded {args.cache}")
        else:
            print(f"Ignoring incompatible cache: {args.cache}")

    if histories is None:
        histories = collect_histories(args)
        write_cache(args.cache, histories)
        print(f"Wrote {args.cache}")

    figure = make_figure(histories)
    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print_summary(histories)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
