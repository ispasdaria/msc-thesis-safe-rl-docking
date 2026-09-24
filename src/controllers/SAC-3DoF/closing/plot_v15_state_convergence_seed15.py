"""Plot six P3-relative errors for the seed-15 extended-horizon rollout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np


DATA_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "publication_figures"
    / "v15_trial16_400k_horizon_comparison"
    / "closing_vanilla_sac_horizon_trajectory_comparison_data.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_PATH = OUTPUT_DIR / "closing_vanilla_sac_state_convergence_seed15.png"

HORIZON = 20_000
CUTOFFS = (12_000.0, 13_000.0)
EXPECTED_CLOSEST_TIME = 13_616.0
TOLERANCES = np.array(
    [20.0, 2.0, 0.01, 0.01, np.deg2rad(0.5), np.deg2rad(0.05)],
    dtype=float,
)

TRAJECTORY_COLOR = "#003B5C"
TOLERANCE_COLOR = "#009E73"
CLOSEST_COLOR = "#9C2F2F"
CUTOFF_COLOR = "#9A9A9A"
GRID_COLOR = "#D9D9D9"


def load_errors() -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=False) as data:
        saved_seed = int(data["seed"])
        if saved_seed != 15:
            raise ValueError(f"Expected testing seed 15, found {saved_seed}.")
        times = np.asarray(data[f"h{HORIZON}_times_s"], dtype=float)
        states = np.asarray(data[f"h{HORIZON}_states"], dtype=float)
        target = np.asarray(data[f"h{HORIZON}_p3_target"], dtype=float)

    errors = states - target
    errors[:, 4] = np.arctan2(np.sin(errors[:, 4]), np.cos(errors[:, 4]))
    return times, errors


def nearest_index(times: np.ndarray, requested_time: float) -> int:
    return int(np.argmin(np.abs(times - requested_time)))


def print_summary(times: np.ndarray, errors: np.ndarray) -> int:
    absolute = np.abs(errors)
    position_error = np.linalg.norm(errors[:, :2], axis=1)
    closest_index = int(np.argmin(position_error))
    closest_time = float(times[closest_index])

    if not np.isclose(closest_time, EXPECTED_CLOSEST_TIME, atol=0.51):
        raise ValueError(
            f"Closest approach was expected near {EXPECTED_CLOSEST_TIME:.0f} s, "
            f"but the saved data gives {closest_time:.1f} s."
        )

    display_scale = np.array([1.0, 1.0, 1.0, 1.0, 180.0 / np.pi, 180.0 / np.pi])
    labels = ("|e_x| [m]", "|e_y| [m]", "|e_vx| [m/s]", "|e_vy| [m/s]",
              "|e_psi| [deg]", "|e_omega| [deg/s]")
    print("State-error summary (absolute values)")
    print("time_s," + ",".join(labels))
    for requested_time in (*CUTOFFS, EXPECTED_CLOSEST_TIME, 20_000.0):
        index = nearest_index(times, requested_time)
        values = absolute[index] * display_scale
        print(f"{times[index]:.0f}," + ",".join(f"{value:.9g}" for value in values))

    ratios = absolute[closest_index] / TOLERANCES
    outside = np.flatnonzero(ratios > 1.0)
    limiting = np.flatnonzero(np.isclose(ratios, np.max(ratios)))
    simultaneous = np.all(absolute <= TOLERANCES, axis=1)

    print(f"minimum_position_error_m={position_error[closest_index]:.9f}")
    print(f"minimum_position_error_time_s={closest_time:.0f}")
    print(f"complete_P3_terminal_set_ever_entered={bool(np.any(simultaneous))}")
    print("outside_tolerance_at_closest=" + ", ".join(labels[i] for i in outside))
    print("limiting_component_at_closest=" + ", ".join(labels[i] for i in limiting))
    print("tolerance_crossing_summary")
    for i, label in enumerate(labels):
        inside = absolute[:, i] <= TOLERANCES[i]
        transitions = int(np.count_nonzero(inside[1:] != inside[:-1]))
        print(
            f"  {label}: ever_within={bool(np.any(inside))}, "
            f"within_samples={int(np.count_nonzero(inside))}, transitions={transitions}"
        )
    return closest_index


def make_figure(times: np.ndarray, errors: np.ndarray, closest_index: int) -> None:
    values = np.abs(errors).copy()
    values[:, 4:] = np.rad2deg(values[:, 4:])
    plotted_tolerances = TOLERANCES.copy()
    plotted_tolerances[4:] = np.rad2deg(plotted_tolerances[4:])

    labels = (
        r"Axial position error, $|e_x|$ [m]",
        r"Lateral position error, $|e_y|$ [m]",
        r"Axial velocity error, $|e_{v_x}|$ [m/s]",
        r"Lateral velocity error, $|e_{v_y}|$ [m/s]",
        r"Attitude error, $|e_\psi|$ [deg]",
        r"Angular-rate error, $|e_\omega|$ [deg/s]",
    )
    tolerance_labels = (
        "Tolerance: 20 m", "Tolerance: 2 m",
        "Tolerance: 0.01 m/s", "Tolerance: 0.01 m/s",
        "Tolerance: 0.5 deg", "Tolerance: 0.05 deg/s",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes_grid = plt.subplots(2, 3, figsize=(12.4, 6.4), sharex=True)
    axes = axes_grid.ravel()
    closest_time = float(times[closest_index])

    for panel, (axis, series, label, tolerance, tolerance_label) in enumerate(
        zip(axes, values.T, labels, plotted_tolerances, tolerance_labels), start=1
    ):
        axis.plot(times, series, color=TRAJECTORY_COLOR, linewidth=1.35, zorder=3)
        axis.axhline(
            tolerance, color=TOLERANCE_COLOR, linewidth=1.1,
            linestyle="--", zorder=2,
        )
        for cutoff in CUTOFFS:
            axis.axvline(
                cutoff, color=CUTOFF_COLOR, linewidth=0.75,
                linestyle=":", alpha=0.65, zorder=1,
            )
        axis.axvline(
            closest_time, color=CLOSEST_COLOR, linewidth=0.95,
            linestyle="--", alpha=0.85, zorder=2,
        )
        axis.set_ylabel(label)
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.72)
        axis.set_axisbelow(True)
        axis.text(
            0.012, 0.94, f"({chr(96 + panel)})", transform=axis.transAxes,
            ha="left", va="top", fontsize=9.5, fontweight="bold",
        )
        annotation_transform = blended_transform_factory(axis.transAxes, axis.transData)
        axis.text(
            0.035, tolerance, tolerance_label, transform=annotation_transform,
            ha="left", va="bottom", color=TOLERANCE_COLOR, fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.7},
        )

    for axis in axes_grid[-1, :]:
        axis.set_xlabel("Time [s]")

    fig.suptitle(
        "Closing-phase vanilla SAC state convergence (testing seed 15)",
        fontsize=13.5, fontweight="normal", y=0.985,
    )
    reference_handles = [
        Line2D([], [], color=CUTOFF_COLOR, linewidth=0.8, linestyle=":",
               label="12,000-s and 13,000-s cut-offs"),
        Line2D([], [], color=CLOSEST_COLOR, linewidth=1.0, linestyle="--",
               label=f"Closest approach ({closest_time:,.0f} s)"),
    ]
    fig.legend(
        handles=reference_handles, loc="upper center", bbox_to_anchor=(0.5, 0.947),
        ncol=2, frameon=False,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.09, top=0.855,
        hspace=0.22, wspace=0.25,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure={OUTPUT_PATH}")


def main() -> None:
    times, errors = load_errors()
    closest_index = print_summary(times, errors)
    make_figure(times, errors, closest_index)


if __name__ == "__main__":
    main()
