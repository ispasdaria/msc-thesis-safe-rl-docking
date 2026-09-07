"""Create matched four-panel terminal-state convergence figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
SAC_CACHE = (
    HERE / "SAC-3DoF" / "final_approach" / "results" / "eval_lqr_phase_50"
    / "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
    / "testing_seeds_0_49" / "trajectory_histories.npz"
)
LQR_FINAL_CACHE = (
    HERE / "LQR-3DoF" / "results" / "lqr_final"
    / "final_approach_seeds_0_49_requirement_histories.npz"
)
LQR_CLOSING_CACHE = (
    HERE / "LQR-3DoF" / "results" / "lqr_closing" / "dt_0p1"
    / "closing_phase_seeds_0_49_dt0p1_error_histories.npz"
)
OUTPUT_DIR = HERE / "results" / "terminal_state_convergence"

BLUE = "#0072B2"
DARK_BLUE = "#003B5C"
GREEN = "#009E73"
GRID = "#D9D9D9"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate(data: dict[str, np.ndarray], fields: tuple[str, ...]) -> None:
    missing = set(fields) - set(data)
    if missing:
        raise KeyError(f"{missing} missing from trajectory cache")
    if not np.array_equal(data["seeds"].astype(int), np.arange(50)):
        raise ValueError("Trajectory cache is not the seeds 0..49 evaluation set")


def representative_index(times: np.ndarray, success: np.ndarray) -> int:
    counts = np.sum(np.isfinite(times), axis=1)
    completion = np.array([times[i, n - 1] for i, n in enumerate(counts)])
    candidates = np.flatnonzero(success)
    if not candidates.size:
        candidates = np.arange(times.shape[0])
    median = np.median(completion[candidates])
    return int(candidates[np.argmin(np.abs(completion[candidates] - median))])


def plot_case(
    data: dict[str, np.ndarray], title: str, output_stem: str,
    fields: tuple[str, ...], labels: tuple[str, ...], tolerances: tuple[float, ...],
    tolerance_labels: tuple[str, ...], annotation_x: tuple[float, ...],
    annotation_y_factor: tuple[float, ...],
) -> tuple[Path, Path]:
    validate(data, ("seeds", "times", "success", *fields))
    seeds = data["seeds"].astype(int)
    times = data["times"].astype(float)
    success = data["success"].astype(bool)
    counts = np.sum(np.isfinite(times), axis=1)
    representative = representative_index(times, success)

    plt.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 9, "figure.dpi": 130, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes_grid = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    axes = axes_grid.ravel()
    for panel, (ax, field, label, tolerance, tolerance_label, text_x, text_y_factor) in enumerate(
        zip(axes, fields, labels, tolerances, tolerance_labels,
            annotation_x, annotation_y_factor), start=1
    ):
        values = np.asarray(data[field], dtype=float)
        for i in range(50):
            ax.plot(times[i, :counts[i]], values[i, :counts[i]], color=BLUE,
                    lw=0.65, alpha=0.13, rasterized=True, zorder=2)
        ax.plot(times[representative, :counts[representative]],
                values[representative, :counts[representative]], color=DARK_BLUE,
                lw=1.55, alpha=0.95, zorder=3)
        ax.axhline(tolerance, color=GREEN, lw=1.25, ls="--", zorder=1)
        ax.set_ylabel(label)
        if panel == 1:
            # A small nonphysical display margin keeps a sub-pixel terminal
            # tolerance visibly distinct from the zero-valued axis spine.
            upper = float(np.nanmax(values))
            ax.set_ylim(bottom=-0.025 * upper)
        else:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.75)
        ax.set_axisbelow(True)
        ax.text(0.01, 0.92, f"({chr(96 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        annotation_transform = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(text_x, tolerance * text_y_factor, tolerance_label,
                transform=annotation_transform, ha="left", va="bottom",
                color=GREEN, fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.0})
    for ax in axes_grid[-1, :]:
        ax.set_xlabel("Time [s]")
    axes[-1].set_xlim(0.0, 1.01 * np.nanmax(times))
    fig.suptitle(title, fontsize=12.2, fontweight="bold", y=0.985)
    handles = [
        Line2D([], [], color=BLUE, lw=1.0, alpha=0.35, label="Individual rollouts"),
        Line2D([], [], color=DARK_BLUE, lw=1.7,
               label=f"Median-duration successful rollout (seed {seeds[representative]})"),
        Line2D([], [], color=GREEN, lw=1.25, ls="--", label="Terminal tolerance"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False)
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.09, top=0.86, hspace=0.22, wspace=0.20
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{output_stem}.png"
    pdf = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    final_fields = ("port_position_errors", "port_velocity_errors",
                    "attitude_errors_deg", "angular_rate_errors_deg_s")
    final_labels = ("Docking-port distance [m]", "Relative docking-port speed [m/s]",
                    "Attitude error [deg]", "Angular-rate error [deg/s]")
    final_tolerances = (0.1, 0.1, 4.0, 0.2)
    final_tolerance_labels = ("Tolerance: 0.1 m", "Tolerance: 0.1 m/s",
                              "Tolerance: 4 deg", "Tolerance: 0.2 deg/s")

    outputs = []
    outputs += plot_case(load(SAC_CACHE),
        "Terminal-state convergence across 50 rollouts for the retained final-approach SAC policy",
        "figure_a_retained_sac_final_approach", final_fields, final_labels,
        final_tolerances, final_tolerance_labels,
        (0.56, 0.50, 0.78, 0.73), (8.0, 0.94, 0.92, 0.92))
    outputs += plot_case(load(LQR_FINAL_CACHE),
        "Terminal-state convergence across 50 rollouts for the final-approach LQR controller",
        "figure_b_lqr_final_approach", final_fields, final_labels,
        final_tolerances, final_tolerance_labels,
        (0.08, 0.70, 0.78, 0.73), (8.0, 1.035, 0.92, 0.92))

    closing = load(LQR_CLOSING_CACHE)
    tolerance = np.asarray(closing["p3_target_tolerance"], dtype=float)
    closing["position_error_norm"] = np.hypot(closing["ex"], closing["ey"])
    closing["velocity_error_norm"] = np.hypot(closing["evx"], closing["evy"])
    # Component-wise rectangular tolerances imply the Euclidean norm bounds below.
    closing_tolerances = (float(np.hypot(tolerance[0], tolerance[1])),
                          float(np.hypot(tolerance[2], tolerance[3])),
                          float(np.rad2deg(tolerance[4])),
                          float(np.rad2deg(tolerance[5])))
    outputs += plot_case(closing,
        "Terminal-state convergence across 50 rollouts for the closing LQR controller",
        "figure_c_lqr_closing", ("position_error_norm", "velocity_error_norm",
        "epsi_deg", "eomega_deg_s"), (r"Position error relative to $P_3$ [m]",
        r"Velocity error relative to $P_3$ [m/s]", "Attitude error [deg]",
        "Angular-rate error [deg/s]"), closing_tolerances,
        (r"Corner norm: $\sqrt{20^2+2^2}=20.10$ m",
         r"Corner norm: $\sqrt{0.01^2+0.01^2}=0.0141$ m/s",
         "Tolerance: 0.5 deg", "Tolerance: 0.05 deg/s"),
        (0.08, 0.08, 0.69, 0.71), (1.35, 1.10, 1.08, 1.10))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
