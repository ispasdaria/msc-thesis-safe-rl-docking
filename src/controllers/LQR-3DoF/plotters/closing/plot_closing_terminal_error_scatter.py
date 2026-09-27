"""Plot closing-phase terminal errors against the P3 acceptance limits.

The input is the per-rollout CSV written by ``sweep_closing_lqr_weights.py``. By
default, the script uses the 50-rollout evaluation of the selected, refined
``dt = 0.1 s`` controller and writes a PNG to the canonical closing-results
directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
LQR_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = (
    LQR_ROOT
    / "results"
    / "lqr_closing"
    / "closing_controller_dt0p1_evaluation_and_tuning"
    / "fine_lateral_velocity_weight_sweep"
    / "my_1_mvy_0p85"
    / "rollouts_seeds_0_49.csv"
)
DEFAULT_OUTPUT = (
    LQR_ROOT
    / "results"
    / "lqr_closing"
    / "closing_terminal_error_scatter"
)

# Default Docking3DoFEnv P3 tolerances in state order [rx, ry, vx, vy, psi, omega].
# See src/envs/docking3dof/core.py.
DEFAULT_P3_TOLERANCE = np.array(
    [20.0, 2.0, 0.01, 0.01, np.deg2rad(0.5), np.deg2rad(0.05)],
    dtype=float,
)

REQUIRED_COLUMNS = {
    "success",
    "termination_reason",
    "abs_final_ex",
    "abs_final_ey",
    "abs_final_evx",
    "abs_final_evy",
    "abs_final_epsi",
    "abs_final_eomega",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot terminal closing errors against the P3 tolerances."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Per-rollout closing evaluation CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; a PNG file is written.",
    )
    parser.add_argument(
        "--position-tolerances",
        type=float,
        nargs=2,
        metavar=("EX", "EY"),
        default=DEFAULT_P3_TOLERANCE[:2],
        help="P3 absolute position-error limits in metres (default: 20 2).",
    )
    parser.add_argument(
        "--velocity-tolerances",
        type=float,
        nargs=2,
        metavar=("EVX", "EVY"),
        default=DEFAULT_P3_TOLERANCE[2:4],
        help="P3 absolute velocity-error limits in m/s (default: 0.01 0.01).",
    )
    parser.add_argument(
        "--attitude-tolerances",
        type=float,
        nargs=2,
        metavar=("EPSI", "EOMEGA"),
        default=np.rad2deg(DEFAULT_P3_TOLERANCE[4:6]),
        help="P3 attitude and angular-rate limits in deg and deg/s (default: 0.5 0.05).",
    )
    return parser.parse_args()


def load_rollouts(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rollout records found in {path}")

    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    def numeric(column: str) -> np.ndarray:
        try:
            return np.asarray([float(row[column]) for row in rows], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Column {column!r} contains a non-numeric value") from exc

    success = np.asarray(
        [row["success"].strip().lower() == "true" for row in rows], dtype=bool
    )
    timeout = np.asarray(
        [row["termination_reason"].strip().lower() == "timeout" for row in rows],
        dtype=bool,
    )

    if np.any(~(success | timeout)):
        reasons = sorted(
            {
                row["termination_reason"]
                for row, is_success, is_timeout in zip(rows, success, timeout)
                if not (is_success or is_timeout)
            }
        )
        raise ValueError(
            "The requested success/timeout figure cannot classify termination "
            f"reason(s): {reasons}"
        )

    return {
        "success": success,
        "timeout": timeout,
        "ex": numeric("abs_final_ex"),
        "ey": numeric("abs_final_ey"),
        "evx": numeric("abs_final_evx"),
        "evy": numeric("abs_final_evy"),
        "epsi": numeric("abs_final_epsi"),
        "eomega": numeric("abs_final_eomega"),
    }


def draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    success: np.ndarray,
    timeout: np.ndarray,
    tolerance: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    tol_x, tol_y = tolerance

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            tol_x,
            tol_y,
            facecolor="#009E73",
            edgecolor="#007A59",
            alpha=0.13,
            linewidth=1.8,
            linestyle="--",
            label=r"$P_3$ tolerance region",
            zorder=1,
        )
    )
    ax.scatter(
        x[success],
        y[success],
        s=45,
        marker="o",
        facecolor="#0072B2",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.9,
        label=f"Successful ({success.sum()})",
        zorder=3,
    )
    ax.scatter(
        x[timeout],
        y[timeout],
        s=70,
        marker="X",
        facecolor="#D55E00",
        edgecolor="#7F2704",
        linewidth=0.65,
        label=f"Timeout ({timeout.sum()})",
        zorder=4,
    )

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def make_figure(
    data: dict[str, np.ndarray],
    position_tolerance: tuple[float, float],
    velocity_tolerance: tuple[float, float],
    attitude_tolerance: tuple[float, float],
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 11.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.35))
    draw_panel(
        axes[0],
        data["ex"],
        data["ey"],
        data["success"],
        data["timeout"],
        position_tolerance,
        r"Axial error $|e_x|$ [m]",
        r"Lateral error $|e_y|$ [m]",
        r"(a) CoM position error relative to the $P_3$ target",
    )
    draw_panel(
        axes[1],
        data["evx"],
        data["evy"],
        data["success"],
        data["timeout"],
        velocity_tolerance,
        r"Axial-velocity error $|e_{v_x}|$ [m/s]",
        r"Lateral-velocity error $|e_{v_y}|$ [m/s]",
        r"(b) CoM velocity error relative to the $P_3$ target",
    )
    draw_panel(
        axes[2],
        np.rad2deg(data["epsi"]),
        np.rad2deg(data["eomega"]),
        data["success"],
        data["timeout"],
        attitude_tolerance,
        r"Attitude error $|e_{\psi}|$ [deg]",
        r"Angular-rate error $|e_{\omega}|$ [deg/s]",
        r"(c) Attitude-alignment error relative to the $P_3$ target",
    )

    # One shared legend avoids repeating identical outcome counts in each panel.
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.062, right=0.99, bottom=0.16, top=0.82, wspace=0.29
    )
    return fig


def main() -> None:
    args = parse_args()
    data = load_rollouts(args.input)
    figure = make_figure(
        data,
        tuple(args.position_tolerances),
        tuple(args.velocity_tolerances),
        tuple(args.attitude_tolerances),
    )

    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    success_count = int(data["success"].sum())
    timeout_count = int(data["timeout"].sum())
    print(f"Plotted {success_count + timeout_count} rollouts: "
          f"{success_count} successful, {timeout_count} timeouts")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
