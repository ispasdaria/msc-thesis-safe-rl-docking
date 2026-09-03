"""Compare representative final-controller closing rollouts.

The default successful case (seed 20) has the median completion time among the
eligible successes in the final ``dt = 0.1 s`` 50-rollout evaluation.  The
default timeout (seed 1) finishes inside every P3 tolerance except lateral
position, so it directly illustrates the final controller's dominant terminal
limitation.  Seeds 41 and 13 from the earlier controller are not reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_DIR = (
    SCRIPT_DIR / "results" / "lqr_closing" / "dt_0p1" / "rollout_histories"
)
DEFAULT_SUCCESS = (
    HISTORY_DIR / "q1_r1_seed0020_final_dt0p1_median_completion_success.npz"
)
DEFAULT_TIMEOUT = (
    HISTORY_DIR / "q1_r1_seed0001_final_dt0p1_lateral_position_timeout.npz"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "results" / "lqr_closing" / "closing_representative_trajectories"
)
DEFAULT_XY_OUTPUT = (
    SCRIPT_DIR / "results" / "lqr_closing" / "closing_representative_xy_trajectories"
)

SUCCESS_COLOR = "#0072B2"
TIMEOUT_COLOR = "#D55E00"
TOLERANCE_COLOR = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot representative successful and timeout closing trajectories."
    )
    parser.add_argument(
        "--success",
        type=Path,
        default=DEFAULT_SUCCESS,
        help="Saved successful-rollout NPZ history.",
    )
    parser.add_argument(
        "--timeout",
        type=Path,
        default=DEFAULT_TIMEOUT,
        help="Saved timeout-rollout NPZ history.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension (both .png and .pdf are written).",
    )
    parser.add_argument(
        "--xy-output",
        type=Path,
        default=DEFAULT_XY_OUTPUT,
        help="Spatial-trajectory output path without an extension.",
    )
    return parser.parse_args()


def load_history(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        required = {"times", "states", "p3_target", "p3_target_tolerance"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

        times = np.asarray(data["times"], dtype=float)
        states = np.asarray(data["states"], dtype=float)
        target = np.asarray(data["p3_target"], dtype=float)
        tolerance = np.asarray(data["p3_target_tolerance"], dtype=float)
        metadata = (
            json.loads(str(data["metadata_json"]))
            if "metadata_json" in data.files
            else {}
        )

    if times.ndim != 1 or states.shape != (times.size, 6):
        raise ValueError(
            f"Inconsistent history shapes in {path}: times={times.shape}, "
            f"states={states.shape}"
        )
    if target.shape != (6,) or tolerance.shape != (6,):
        raise ValueError(f"P3 target and tolerance must have shape (6,) in {path}")

    error = states - target
    error[:, 4] = (error[:, 4] + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "time_hours": times / 3600.0,
        "abs_axial_error": np.abs(error[:, 0]),
        "abs_lateral_error": np.abs(error[:, 1]),
        "x": states[:, 0],
        "y": states[:, 1],
        "target": target,
        "tolerance": tolerance,
        "metadata": metadata,
    }


def plot_series(
    ax: plt.Axes,
    success: dict[str, object],
    timeout: dict[str, object],
    field: str,
    tolerance: float,
) -> tuple[plt.Line2D, plt.Line2D]:
    success_line, = ax.plot(
        success["time_hours"],
        success[field],
        color=SUCCESS_COLOR,
        linewidth=2.0,
        label=f"Successful (seed {success['metadata'].get('seed', '?')})",
        zorder=3,
    )
    timeout_line, = ax.plot(
        timeout["time_hours"],
        timeout[field],
        color=TIMEOUT_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=f"Timeout (seed {timeout['metadata'].get('seed', '?')})",
        zorder=3,
    )
    ax.scatter(
        success["time_hours"][-1],
        success[field][-1],
        s=50,
        marker="o",
        color=SUCCESS_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.scatter(
        timeout["time_hours"][-1],
        timeout[field][-1],
        s=65,
        marker="X",
        color=TIMEOUT_COLOR,
        edgecolor="#7F2704",
        linewidth=0.7,
        zorder=5,
    )
    ax.axhspan(0.0, tolerance, color=TOLERANCE_COLOR, alpha=0.13, zorder=1)
    ax.axhline(
        tolerance,
        color="#007A59",
        linewidth=1.4,
        linestyle="--",
        zorder=2,
    )
    return success_line, timeout_line


def add_terminal_inset(
    ax: plt.Axes,
    success: dict[str, object],
    timeout: dict[str, object],
    field: str,
    tolerance: float,
    ylim: tuple[float, float],
) -> None:
    inset = ax.inset_axes([0.43, 0.38, 0.53, 0.54])
    inset.plot(
        success["time_hours"],
        success[field],
        color=SUCCESS_COLOR,
        linewidth=1.6,
    )
    inset.plot(
        timeout["time_hours"],
        timeout[field],
        color=TIMEOUT_COLOR,
        linewidth=1.6,
        linestyle="--",
    )
    inset.scatter(
        success["time_hours"][-1],
        success[field][-1],
        s=30,
        marker="o",
        color=SUCCESS_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    inset.scatter(
        timeout["time_hours"][-1],
        timeout[field][-1],
        s=40,
        marker="X",
        color=TIMEOUT_COLOR,
        edgecolor="#7F2704",
        linewidth=0.5,
        zorder=5,
    )
    inset.axhspan(0.0, tolerance, color=TOLERANCE_COLOR, alpha=0.13)
    inset.axhline(tolerance, color="#007A59", linewidth=1.0, linestyle="--")
    inset.set_xlim(2.75, max(timeout["time_hours"]) + 0.025)
    inset.set_ylim(*ylim)
    inset.set_title("Terminal interval", fontsize=8.5, pad=3)
    inset.tick_params(labelsize=8)
    inset.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)


def make_figure(
    success: dict[str, object], timeout: dict[str, object]
) -> plt.Figure:
    success_tolerance = np.asarray(success["tolerance"], dtype=float)
    timeout_tolerance = np.asarray(timeout["tolerance"], dtype=float)
    if not np.allclose(success_tolerance, timeout_tolerance):
        raise ValueError("The two histories use different P3 tolerance vectors")

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 11.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.35))
    success_line, timeout_line = plot_series(
        axes[0],
        success,
        timeout,
        "abs_axial_error",
        success_tolerance[0],
    )
    plot_series(
        axes[1],
        success,
        timeout,
        "abs_lateral_error",
        success_tolerance[1],
    )

    axes[0].set_title("(a) Remaining axial distance to $P_3$", loc="left", fontweight="bold")
    axes[0].set_ylabel(r"Axial error $|e_x|$ [m]")
    axes[1].set_title("(b) Lateral error", loc="left", fontweight="bold")
    axes[1].set_ylabel(r"Lateral error $|e_y|$ [m]")

    max_time = max(max(success["time_hours"]), max(timeout["time_hours"]))
    for ax in axes:
        ax.set_xlim(0.0, max_time + 0.025)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Time [h]")
        ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    add_terminal_inset(
        axes[0],
        success,
        timeout,
        "abs_axial_error",
        success_tolerance[0],
        (0.0, 105.0),
    )
    add_terminal_inset(
        axes[1],
        success,
        timeout,
        "abs_lateral_error",
        success_tolerance[1],
        (0.0, 5.0),
    )

    tolerance_patch = Patch(
        facecolor=TOLERANCE_COLOR,
        edgecolor="#007A59",
        linestyle="--",
        alpha=0.13,
        label=r"$P_3$ tolerance region",
    )
    fig.legend(
        [success_line, timeout_line, tolerance_patch],
        [
            success_line.get_label(),
            timeout_line.get_label(),
            tolerance_patch.get_label(),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.82, wspace=0.25)
    return fig


def draw_xy_trajectory(
    ax: plt.Axes,
    rollout: dict[str, object],
    color: str,
    linestyle: str,
    outcome: str,
    terminal_marker: str,
) -> plt.Line2D:
    seed = rollout["metadata"].get("seed", "?")
    line, = ax.plot(
        rollout["x"],
        rollout["y"],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=f"{outcome} (seed {seed})",
        zorder=3,
    )
    ax.scatter(
        rollout["x"][0],
        rollout["y"][0],
        marker="D",
        s=48,
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        rollout["x"][-1],
        rollout["y"][-1],
        marker=terminal_marker,
        s=70,
        facecolor=color,
        edgecolor="white" if terminal_marker == "o" else "#7F2704",
        linewidth=0.8,
        zorder=6,
    )
    ax.scatter(
        rollout["target"][0],
        rollout["target"][1],
        marker="*",
        s=115,
        facecolor=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    tolerance = np.asarray(rollout["tolerance"], dtype=float)
    target = np.asarray(rollout["target"], dtype=float)
    ax.add_patch(
        Rectangle(
            (target[0] - tolerance[0], target[1] - tolerance[1]),
            2.0 * tolerance[0],
            2.0 * tolerance[1],
            facecolor="none",
            edgecolor=color,
            alpha=0.38,
            linewidth=1.2,
            linestyle="-",
            zorder=1,
        )
    )
    return line


def make_xy_figure(
    success: dict[str, object], timeout: dict[str, object]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    success_line = draw_xy_trajectory(
        ax, success, SUCCESS_COLOR, "-", "Successful", "o"
    )
    timeout_line = draw_xy_trajectory(
        ax, timeout, TIMEOUT_COLOR, "--", "Timeout", "X"
    )

    ax.set_xlabel(r"Axial CoM-relative position $x$ [m]")
    ax.set_ylabel(r"Lateral CoM-relative position $y$ [m]")
    ax.set_title("Closing trajectories in the docking plane", loc="left", fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.03, y=0.08)
    # Show the approach in the natural reading direction: initial states on
    # the left and convergence toward P3 on the right.  This changes only the
    # display direction; the physical axial coordinate remains unchanged.
    ax.invert_xaxis()

    # The full closing path spans about 1.5 km, so a terminal inset is needed
    # to make the sampled P3 targets and their 40 m x 4 m tolerance boxes clear.
    inset = ax.inset_axes([0.45, 0.47, 0.47, 0.47])
    draw_xy_trajectory(inset, success, SUCCESS_COLOR, "-", "Successful", "o")
    draw_xy_trajectory(inset, timeout, TIMEOUT_COLOR, "--", "Timeout", "X")

    for rollout, color, label in (
        (
            success,
            SUCCESS_COLOR,
            f"Seed {success['metadata'].get('seed', '?')} tolerance",
        ),
        (
            timeout,
            TIMEOUT_COLOR,
            f"Seed {timeout['metadata'].get('seed', '?')} tolerance",
        ),
    ):
        target = np.asarray(rollout["target"], dtype=float)
        tolerance = np.asarray(rollout["tolerance"], dtype=float)
        inset.text(
            target[0] + tolerance[0] - 1.2,
            target[1] + tolerance[1] - 0.25,
            label,
            color=color,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=7,
        )

    all_targets = np.vstack((success["target"][:2], timeout["target"][:2]))
    inset.set_xlim(np.max(all_targets[:, 0]) + 32.0, np.min(all_targets[:, 0]) - 28.0)
    inset.set_ylim(np.min(all_targets[:, 1]) - 5.0, np.max(all_targets[:, 1]) + 5.0)
    inset.set_title("$P_3$ terminal region", fontsize=10.5, pad=3)
    inset.set_xlabel("$x$ [m]", fontsize=9)
    inset.set_ylabel("$y$ [m]", fontsize=9)
    inset.tick_params(labelsize=9)
    inset.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)

    start_handle = plt.Line2D(
        [], [], marker="D", linestyle="none", markerfacecolor="white",
        markeredgecolor="#555555", markersize=6, label="Initial state"
    )
    target_handle = plt.Line2D(
        [], [], marker="*", linestyle="none", markerfacecolor="#666666",
        markeredgecolor="white", markersize=10, label="Sampled $P_3$ target"
    )
    fig.legend(
        handles=[success_line, timeout_line, start_handle, target_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.82)
    return fig


def main() -> None:
    args = parse_args()
    success = load_history(args.success)
    timeout = load_history(args.timeout)

    if not bool(success["metadata"].get("success", False)):
        raise ValueError(f"Expected a successful history: {args.success}")
    if str(timeout["metadata"].get("termination_reason", "")) != "timeout":
        raise ValueError(f"Expected a timeout history: {args.timeout}")

    figure = make_figure(success, timeout)
    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    xy_figure = make_xy_figure(success, timeout)
    xy_output_base = args.xy_output.with_suffix("")
    xy_output_base.parent.mkdir(parents=True, exist_ok=True)
    xy_png_path = xy_output_base.with_suffix(".png")
    xy_pdf_path = xy_output_base.with_suffix(".pdf")
    xy_figure.savefig(xy_png_path, bbox_inches="tight", facecolor="white")
    xy_figure.savefig(xy_pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(xy_figure)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {xy_png_path}")
    print(f"Wrote {xy_pdf_path}")


if __name__ == "__main__":
    main()
