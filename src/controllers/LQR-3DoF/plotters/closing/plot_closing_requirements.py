"""Plot translational P3 tracking errors for the final closing LQR.

The four panels show the signed component errors requested for the closing
handover assessment: ``e_x``, ``e_y``, ``e_vx``, and ``e_vy``.  The acceptance
interval for each component is drawn explicitly from the environment's P3
tolerance vector.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
LQR_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = LQR_ROOT.parents[1]
for import_path in (LQR_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
)
from tuning.closing.sweep_closing_lqr_weights import run_single_rollout


RESULT_DIR = (
    LQR_ROOT / "results" / "lqr_closing"
    / "closing_controller_dt0p1_evaluation_and_tuning"
)
DEFAULT_CACHE = RESULT_DIR / "closing_phase_seeds_0_49_dt0p1_error_histories.npz"
DEFAULT_OUTPUT = RESULT_DIR / "closing_phase_seeds_0_49_dt0p1_errors"

SUCCESS_COLOR = "tab:red"
UNSUCCESSFUL_COLOR = "tab:blue"
REPRESENTATIVE_COLOR = "tab:orange"

REQUIRED_CACHE_FIELDS = {
    "seeds",
    "times",
    "ex",
    "ey",
    "evx",
    "evy",
    "epsi_deg",
    "eomega_deg_s",
    "p3_target_tolerance",
    "success",
    "termination_reasons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ex, ey, evx, and evy histories for the final closing LQR."
        )
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Maximum retained points per rollout for plotting and caching.",
    )
    parser.add_argument(
        "--mode", choices=("nominal", "testing", "training"), default="testing"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; a PNG file is written.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rerun simulations even when a compatible cache exists.",
    )
    args = parser.parse_args()
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.max_time <= 0.0:
        parser.error("--max-time must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_points < 2:
        parser.error("--max-points must be at least 2")
    return args


def build_controller(max_time: float, dt: float) -> LQRClosingController:
    env = Docking3DoFEnv(
        phase="closing", dt=dt, max_time=max_time, seed=0
    )
    A, B = build_linearized_3dof_matrices(env)
    gain, _ = compute_lqr_gain(
        A, B, build_Q_closing(dt=env.dt), build_R_closing(env)
    )
    return LQRClosingController(gain)


def pad_histories(histories: Sequence[np.ndarray]) -> np.ndarray:
    max_length = max(history.size for history in histories)
    padded = np.full((len(histories), max_length), np.nan, dtype=float)
    for index, history in enumerate(histories):
        padded[index, : history.size] = history
    return padded


def collect_one_rollout(
    task: tuple[int, float, float, str, int]
) -> dict[str, object]:
    seed, max_time, dt, mode, max_points = task
    controller = build_controller(max_time, dt)
    env, _, info, _ = run_single_rollout(
        controller=controller,
        max_time=max_time,
        seed=seed,
        mode=mode,
        dt=dt,
    )
    rollout_times, states, _, _, _ = env.get_history_arrays()
    if rollout_times.size > max_points:
        indices = np.unique(
            np.linspace(0, rollout_times.size - 1, max_points, dtype=np.int64)
        )
        rollout_times = rollout_times[indices]
        states = states[indices]

    errors = np.asarray(states, dtype=float) - np.asarray(
        env.p3_target, dtype=float
    ).reshape(1, -1)
    errors[:, 4] = (errors[:, 4] + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "seed": seed,
        "times": np.asarray(rollout_times, dtype=float),
        "ex": errors[:, 0],
        "ey": errors[:, 1],
        "evx": errors[:, 2],
        "evy": errors[:, 3],
        "epsi_deg": np.abs(np.rad2deg(errors[:, 4])),
        "eomega_deg_s": np.abs(np.rad2deg(errors[:, 5])),
        "p3_target_tolerance": np.asarray(
            env.p3_target_tolerance, dtype=float
        ),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
    }


def collect_histories(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tasks = [
        (seed, args.max_time, args.dt, args.mode, args.max_points)
        for seed in range(args.seed_start, args.seed_start + args.rollouts)
    ]
    if args.workers == 1:
        iterator = map(collect_one_rollout, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(collect_one_rollout, tasks)

    records: list[dict[str, object]] = []
    try:
        for completed, record in enumerate(iterator, start=1):
            records.append(record)
            if completed % 5 == 0 or completed == args.rollouts:
                print(f"Completed {completed}/{args.rollouts} rollouts", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    records.sort(key=lambda record: int(record["seed"]))

    return {
        "seeds": np.asarray([record["seed"] for record in records], dtype=int),
        "times": pad_histories([record["times"] for record in records]),
        "ex": pad_histories([record["ex"] for record in records]),
        "ey": pad_histories([record["ey"] for record in records]),
        "evx": pad_histories([record["evx"] for record in records]),
        "evy": pad_histories([record["evy"] for record in records]),
        "epsi_deg": pad_histories([record["epsi_deg"] for record in records]),
        "eomega_deg_s": pad_histories(
            [record["eomega_deg_s"] for record in records]
        ),
        "p3_target_tolerance": np.asarray(
            records[0]["p3_target_tolerance"], dtype=float
        ),
        "success": np.asarray(
            [record["success"] for record in records], dtype=bool
        ),
        "termination_reasons": np.asarray(
            [record["termination_reason"] for record in records], dtype="U32"
        ),
        "mode": np.asarray(args.mode),
        "dt": np.asarray(args.dt, dtype=float),
        "max_time": np.asarray(args.max_time, dtype=float),
        "max_points": np.asarray(args.max_points, dtype=int),
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
    scalar_settings = {
        "dt": args.dt,
        "max_time": args.max_time,
        "max_points": args.max_points,
    }
    for field, expected in scalar_settings.items():
        if field in histories and not np.isclose(
            float(np.asarray(histories[field])), expected
        ):
            return False
    if "mode" in histories and str(np.asarray(histories["mode"]).item()) != args.mode:
        return False
    return True


def representative_success_index(
    times: np.ndarray, success: np.ndarray
) -> int:
    valid_counts = np.sum(np.isfinite(times), axis=1)
    completion_times = np.asarray(
        [times[index, count - 1] for index, count in enumerate(valid_counts)],
        dtype=float,
    )
    candidates = np.flatnonzero(success)
    if candidates.size == 0:
        candidates = np.arange(times.shape[0])
    candidate_times = completion_times[candidates]
    return int(
        candidates[
            np.argmin(np.abs(candidate_times - np.median(candidate_times)))
        ]
    )


def make_figure(histories: dict[str, np.ndarray]) -> plt.Figure:
    seeds = np.asarray(histories["seeds"], dtype=int)
    times = np.asarray(histories["times"], dtype=float)
    success = np.asarray(histories["success"], dtype=bool)
    valid_counts = np.sum(np.isfinite(times), axis=1)
    representative_index = representative_success_index(times, success)
    p3_tolerance = np.asarray(
        histories["p3_target_tolerance"], dtype=float
    )
    if p3_tolerance.shape != (6,):
        raise ValueError(
            "p3_target_tolerance must have shape (6,), "
            f"got {p3_tolerance.shape}"
        )

    panel_specs = (
        (
            "ex",
            r"Axial position error $e_x(t)$ [m]",
            r"(a) $e_x(t)$",
            p3_tolerance[0],
            "m",
        ),
        (
            "ey",
            r"Lateral position error $e_y(t)$ [m]",
            r"(b) $e_y(t)$",
            p3_tolerance[1],
            "m",
        ),
        (
            "evx",
            r"Axial velocity error $e_{v_x}(t)$ [m/s]",
            r"(c) $e_{v_x}(t)$",
            p3_tolerance[2],
            "m/s",
        ),
        (
            "evy",
            r"Lateral velocity error $e_{v_y}(t)$ [m/s]",
            r"(d) $e_{v_y}(t)$",
            p3_tolerance[3],
            "m/s",
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

    for ax, (field, ylabel, title, tolerance, unit) in zip(axes, panel_specs):
        values = np.asarray(histories[field], dtype=float)
        for rollout_index in range(seeds.size):
            count = valid_counts[rollout_index]
            ax.plot(
                times[rollout_index, :count] / 60.0,
                values[rollout_index, :count],
                color=(
                    SUCCESS_COLOR
                    if success[rollout_index]
                    else UNSUCCESSFUL_COLOR
                ),
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
            zorder=3,
        )
        ax.axhspan(
            -tolerance,
            tolerance,
            color="#009E73",
            alpha=0.12,
            zorder=1,
        )
        ax.axhline(
            tolerance,
            color="#007A59",
            linestyle=":",
            linewidth=1.4,
            zorder=4,
        )
        ax.axhline(
            -tolerance,
            color="#007A59",
            linestyle=":",
            linewidth=1.4,
            zorder=4,
        )
        ax.axhline(0.0, color="#555555", linewidth=0.7, alpha=0.65, zorder=1)
        ax.text(
            0.98,
            0.95,
            rf"$P_3$: $\pm {tolerance:g}$ {unit}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#007A59",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
            zorder=5,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        lower, upper = ax.get_ylim()
        ax.set_ylim(
            min(lower, -1.5 * tolerance),
            max(upper, 1.5 * tolerance),
        )
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
            f"Closing-phase P3 tracking performance "
            f"($\\Delta t={float(np.asarray(histories['dt'])):g}$ s): "
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
            color=SUCCESS_COLOR,
            linewidth=1.5,
            alpha=0.55,
            label="Successful rollouts",
        ),
        Line2D(
            [],
            [],
            color=UNSUCCESSFUL_COLOR,
            linewidth=1.5,
            alpha=0.55,
            label="Unsuccessful rollouts",
        ),
        Line2D(
            [],
            [],
            color=REPRESENTATIVE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=(
                "Successful rollout nearest median duration "
                f"(seed {seeds[representative_index]})"
            ),
        ),
        Line2D(
            [],
            [],
            color="#007A59",
            linestyle=":",
            linewidth=1.4,
            label=r"$P_3$ tolerance ($\pm$ bound)",
        ),
    )
    fig.legend(
        handles=figure_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.17,
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
        "Rollout duration [min]: "
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
