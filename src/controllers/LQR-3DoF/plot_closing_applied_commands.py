"""Plot requested and applied closing commands.

The default history is the median successful rollout selected by the
50-rollout closing-phase evaluation. One figure summarizes the full-duration
signed translational commands, while a separate detail figure retains the
native samples over a short window to expose alternation and clipping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_DIR = SCRIPT_DIR / "results" / "lqr_closing" / "rollout_histories"
DEFAULT_HISTORY = HISTORY_DIR / "q1_r1_seed0041_median_success.npz"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "results" / "lqr_closing" / "closing_applied_commands"
)
DEFAULT_DETAIL_OUTPUT = (
    SCRIPT_DIR / "results" / "lqr_closing" / "closing_applied_commands_detail"
)
DEFAULT_VELOCITY_OUTPUT = (
    SCRIPT_DIR / "results" / "lqr_closing" / "closing_velocity_history"
)

REQUESTED_COLOR = "#D55E00"
APPLIED_COLOR = "#0072B2"
AXIAL_COLOR = "#0072B2"
LATERAL_COLOR = "#D55E00"
LIMIT_COLOR = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw and clipped translational commands for a representative "
            "successful closing rollout."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="Saved successful-rollout NPZ history.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; PNG and PDF are written.",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=DEFAULT_DETAIL_OUTPUT,
        help="Detail-figure output path without an extension.",
    )
    parser.add_argument(
        "--velocity-output",
        type=Path,
        default=DEFAULT_VELOCITY_OUTPUT,
        help="Velocity-history output path without an extension.",
    )
    parser.add_argument(
        "--window-start",
        "--inset-start",
        dest="window_start",
        type=float,
        default=100.0,
        help="Start time of the signed-command detail panel in seconds.",
    )
    parser.add_argument(
        "--window-duration",
        "--inset-duration",
        dest="window_duration",
        type=float,
        default=10.0,
        help="Duration of the signed-command detail panel in seconds.",
    )
    return parser.parse_args()


def load_history(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "times",
            "states",
            "actions_body_raw",
            "actions_body_clipped",
            "p3_target",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

        times = np.asarray(data["times"], dtype=float)
        states = np.asarray(data["states"], dtype=float)
        requested = np.asarray(data["actions_body_raw"], dtype=float)
        applied = np.asarray(data["actions_body_clipped"], dtype=float)
        p3_target = np.asarray(data["p3_target"], dtype=float)
        metadata = (
            json.loads(str(data["metadata_json"]))
            if "metadata_json" in data.files
            else {}
        )

    if times.ndim != 1:
        raise ValueError(f"Expected one-dimensional times, got {times.shape}")
    expected_shape = (times.size, 3)
    if requested.shape != expected_shape or applied.shape != expected_shape:
        raise ValueError(
            "Inconsistent command history shapes: "
            f"times={times.shape}, requested={requested.shape}, "
            f"applied={applied.shape}"
        )
    if states.shape != (times.size, 6):
        raise ValueError(
            f"Inconsistent state history shapes: times={times.shape}, "
            f"states={states.shape}"
        )
    if p3_target.shape != (6,):
        raise ValueError(f"Expected P3 target shape (6,), got {p3_target.shape}")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("History must contain at least two increasing timestamps.")
    if not (
        np.all(np.isfinite(times))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(requested))
        and np.all(np.isfinite(applied))
        and np.all(np.isfinite(p3_target))
    ):
        raise ValueError(f"Non-finite values found in {path}")

    dt = float(np.median(np.diff(times)))
    return {
        "times": times,
        "states": states,
        "requested": requested,
        "applied": applied,
        "p3_target": p3_target,
        "dt": dt,
        "metadata": metadata,
    }


def select_window(
    history: dict[str, object], start: float, duration: float
) -> np.ndarray:
    if duration <= 0.0:
        raise ValueError("Detail-window duration must be positive.")
    times = history["times"]
    mask = (times >= start) & (times <= start + duration)
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            f"Detail window [{start:g}, {start + duration:g}] s contains fewer than "
            "three samples."
        )
    return mask


def binned_percentiles(
    times: np.ndarray,
    values: np.ndarray,
    bin_width: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 10th, 50th, and 90th percentiles in fixed time bins."""
    if bin_width <= 0.0:
        raise ValueError("Bin width must be positive.")

    edges = np.arange(times[0], times[-1] + bin_width, bin_width)
    if edges[-1] <= times[-1]:
        edges = np.append(edges, edges[-1] + bin_width)

    centers: list[float] = []
    lower: list[float] = []
    median: list[float] = []
    upper: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (times >= left) & (times < right)
        if right == edges[-1]:
            mask |= times == right
        samples = values[mask]
        if samples.size == 0:
            continue
        centers.append(float(np.mean(times[mask])))
        q10, q50, q90 = np.percentile(samples, (10.0, 50.0, 90.0))
        lower.append(float(q10))
        median.append(float(q50))
        upper.append(float(q90))

    return (
        np.asarray(centers),
        np.asarray(lower),
        np.asarray(median),
        np.asarray(upper),
    )


def make_figures(
    history: dict[str, object], window_start: float, window_duration: float
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    times = history["times"]
    time_hours = times / 3600.0
    states = history["states"]
    requested = history["requested"]
    applied = history["applied"]
    p3_target = history["p3_target"]
    window_mask = select_window(history, window_start, window_duration)
    summary_time, _, requested_ax_median, _ = binned_percentiles(
        times, requested[:, 0], bin_width=180.0
    )
    summary_time_y, _, requested_ay_median, _ = binned_percentiles(
        times, requested[:, 1], bin_width=180.0
    )
    if not np.array_equal(summary_time, summary_time_y):
        raise ValueError("Axial and lateral summaries use inconsistent time bins.")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    command_figure, (axial_ax, lateral_ax) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.6),
        sharex=True,
        sharey=True,
    )
    command_figure.subplots_adjust(
        left=0.13,
        right=0.975,
        bottom=0.11,
        top=0.88,
        hspace=0.24,
    )
    detail_figure, detail_ax = plt.subplots(figsize=(8.4, 5.0))
    detail_figure.subplots_adjust(
        left=0.13,
        right=0.975,
        bottom=0.13,
        top=0.88,
    )
    velocity_figure, (vx_ax, vy_ax) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.6),
        sharex=True,
    )
    velocity_figure.subplots_adjust(
        left=0.13,
        right=0.975,
        bottom=0.11,
        top=0.88,
        hspace=0.24,
    )

    # Three-minute component medians prevent the requested 1 Hz oscillation
    # from becoming opaque at full-rollout scale. Applied signed components are
    # connected at their original one-second sampling interval.
    component_specs = (
        (
            axial_ax,
            requested_ax_median,
            applied[:, 0],
            AXIAL_COLOR,
            r"(a) Axial command, $a_{x,b_c}$",
        ),
        (
            lateral_ax,
            requested_ay_median,
            applied[:, 1],
            LATERAL_COLOR,
            r"(b) Lateral command, $a_{y,b_c}$",
        ),
    )
    for axis, requested_median, applied_component, color, title in component_specs:
        axis.plot(
            summary_time / 3600.0,
            requested_median,
            color=color,
            linestyle="--",
            linewidth=1.55,
            alpha=0.88,
            label="Requested (180-s binned median)",
            zorder=2,
        )
        axis.plot(
            time_hours,
            applied_component,
            color=color,
            linewidth=0.35,
            alpha=0.38,
            rasterized=True,
            label="Applied (1-s history)",
            zorder=3,
        )
        axis.set_title(title, fontsize=11)
        axis.axhline(0.0, color="#888888", linewidth=0.65, alpha=0.65)
        axis.grid(True, alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(
            loc="best",
            fontsize=9,
            frameon=True,
        )

    command_figure.supylabel(
        r"Chaser body-frame acceleration command, $a_{b_c}$ [m/s$^2$]",
        x=0.035,
        fontsize=11,
    )
    lateral_ax.set_xlabel(r"Elapsed closing time, $t$ [h]", fontsize=11)
    lateral_ax.set_xlim(time_hours[0], time_hours[-1])
    displayed_min = min(
        float(np.min(requested_ax_median)),
        float(np.min(requested_ay_median)),
        float(np.min(applied[:, :2])),
    )
    displayed_max = max(
        float(np.max(requested_ax_median)),
        float(np.max(requested_ay_median)),
        float(np.max(applied[:, :2])),
    )
    displayed_span = displayed_max - displayed_min
    axial_ax.set_ylim(
        displayed_min - 0.12 * displayed_span,
        displayed_max + 0.72 * displayed_span,
    )
    command_figure.suptitle(
        "Closing-phase requested and applied translational commands",
        fontsize=12,
    )

    detail_time = times[window_mask]
    detail_ax.plot(
        detail_time,
        requested[window_mask, 0],
        color=AXIAL_COLOR,
        linestyle="--",
        linewidth=1.35,
        marker="o",
        markersize=3.5,
        label=r"Requested $a_{x,b_c}$",
        zorder=2,
    )
    detail_ax.plot(
        detail_time,
        applied[window_mask, 0],
        color=AXIAL_COLOR,
        linewidth=1.9,
        marker="o",
        markersize=4.0,
        label=r"Applied $a_{x,b_c}$",
        zorder=4,
    )
    detail_ax.plot(
        detail_time,
        requested[window_mask, 1],
        color=LATERAL_COLOR,
        linestyle="--",
        linewidth=1.35,
        marker="s",
        markersize=3.3,
        label=r"Requested $a_{y,b_c}$",
        zorder=2,
    )
    detail_ax.plot(
        detail_time,
        applied[window_mask, 1],
        color=LATERAL_COLOR,
        linewidth=1.9,
        marker="s",
        markersize=3.8,
        label=r"Applied $a_{y,b_c}$",
        zorder=5,
    )
    detail_ax.axhline(
        0.01,
        color=LIMIT_COLOR,
        linestyle=":",
        linewidth=1.15,
        zorder=1,
    )
    detail_ax.axhline(
        -0.01,
        color=LIMIT_COLOR,
        linestyle=":",
        linewidth=1.15,
        zorder=1,
    )
    detail_ax.set_xlim(window_start, window_start + window_duration)
    detail_ax.set_xlabel(r"Elapsed closing time, $t$ [s]", fontsize=11)
    detail_ax.set_ylabel(
        r"Chaser body-frame acceleration command, $a_{b_c}$ [m/s$^2$]",
        fontsize=11,
    )
    detail_figure.suptitle(
        "Ten-second detail of requested and applied translational commands",
        fontsize=12,
    )
    detail_ax.grid(True, alpha=0.25)
    detail_ax.set_axisbelow(True)
    detail_ax.set_ylim(-0.14, 0.17)
    detail_ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=9,
        frameon=True,
        columnspacing=1.0,
    )

    velocity_specs = (
        (
            vx_ax,
            states[:, 2],
            float(p3_target[2]),
            AXIAL_COLOR,
            r"(a) Axial relative velocity, $v_x$",
        ),
        (
            vy_ax,
            states[:, 3],
            float(p3_target[3]),
            LATERAL_COLOR,
            r"(b) Lateral relative velocity, $v_y$",
        ),
    )
    for axis, velocity, target_velocity, color, title in velocity_specs:
        velocity_summary_time, _, velocity_median, _ = binned_percentiles(
            times, velocity, bin_width=180.0
        )
        axis.plot(
            time_hours,
            velocity,
            color=color,
            linewidth=0.30,
            alpha=0.20,
            rasterized=True,
            label="Native history (1-s samples)",
            zorder=2,
        )
        axis.plot(
            velocity_summary_time / 3600.0,
            velocity_median,
            color=color,
            linewidth=1.85,
            label="180-s median",
            zorder=3,
        )
        axis.axhline(
            target_velocity,
            color="#555555",
            linestyle="--",
            linewidth=1.15,
            label=r"$P_3$ terminal target",
            zorder=1,
        )
        axis.scatter(
            time_hours[-1],
            velocity[-1],
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        axis.set_title(title, fontsize=11)
        axis.grid(True, alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(
            loc="best",
            fontsize=9,
            frameon=True,
        )

    velocity_figure.supylabel(
        r"Relative velocity in $\mathcal{F}_{o}$ [m/s]",
        x=0.035,
        fontsize=11,
    )
    vy_ax.set_xlabel(r"Elapsed closing time, $t$ [h]", fontsize=11)
    vy_ax.set_xlim(time_hours[0], time_hours[-1])
    velocity_figure.suptitle(
        "Closing-phase relative-velocity history",
        fontsize=12,
    )

    return command_figure, detail_figure, velocity_figure


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    command_figure, detail_figure, velocity_figure = make_figures(
        history, args.window_start, args.window_duration
    )

    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    command_figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    command_figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(command_figure)

    detail_output_base = args.detail_output.with_suffix("")
    detail_output_base.parent.mkdir(parents=True, exist_ok=True)
    detail_png_path = detail_output_base.with_suffix(".png")
    detail_pdf_path = detail_output_base.with_suffix(".pdf")
    detail_figure.savefig(
        detail_png_path, bbox_inches="tight", facecolor="white"
    )
    detail_figure.savefig(
        detail_pdf_path, bbox_inches="tight", facecolor="white"
    )
    plt.close(detail_figure)

    velocity_output_base = args.velocity_output.with_suffix("")
    velocity_output_base.parent.mkdir(parents=True, exist_ok=True)
    velocity_png_path = velocity_output_base.with_suffix(".png")
    velocity_pdf_path = velocity_output_base.with_suffix(".pdf")
    velocity_figure.savefig(
        velocity_png_path, bbox_inches="tight", facecolor="white"
    )
    velocity_figure.savefig(
        velocity_pdf_path, bbox_inches="tight", facecolor="white"
    )
    plt.close(velocity_figure)

    metadata = history["metadata"]
    print(f"Loaded {args.history}")
    print(f"Seed: {metadata.get('seed', '?')}")
    print(f"Duration: {history['times'][-1]:.1f} s")
    print(f"Sampling interval: {history['dt']:.1f} s")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {detail_png_path}")
    print(f"Wrote {detail_pdf_path}")
    print(f"Wrote {velocity_png_path}")
    print(f"Wrote {velocity_pdf_path}")


if __name__ == "__main__":
    main()
