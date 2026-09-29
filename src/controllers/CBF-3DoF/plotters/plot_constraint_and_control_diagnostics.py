"""Plot logged closing barriers and final-approach CBF control interventions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "trajectory_figures"
CLOSING_NOMINAL = OUT / "selected_nominal_rollouts/campaign_phase_balanced_100k_seeds_0_49/step_timeseries_unfiltered_eval_seed0025.npz"
CLOSING_FILTERED = HERE / "results/closing/evaluations/controller_campaigns/campaign_phase_balanced_100k_seeds_0_49/step_timeseries_cone_speed_and_alignment_cbfs_eval_seed0025.npz"
FINAL_CASES = (
    (
        "Final-approach round 5, seed 10, 490k checkpoint (eval. seed 7)",
        HERE / "results/final_approach/evaluations/controller_campaigns/campaign_round5_seed10_best490k_seeds_0_49",
        7,
    ),
    (
        "Final-approach round 1, seed 12, 100k checkpoint (eval. seed 0)",
        HERE / "results/final_approach/evaluations/controller_campaigns/campaign_round1_seed12_final100k_seeds_0_49",
        0,
    ),
)

ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.9",
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
        }
    )


def plot_barrier_pair(ax, nominal, filtered, keys, labels, ylabel, scale=1.0):
    styles = ("-", "--", ":")
    for index, (key, label) in enumerate(zip(keys, labels)):
        ax.plot(nominal["time"], scale * nominal[key], color=ORANGE, ls=styles[index], lw=1.15, alpha=0.9)
        ax.plot(filtered["time"], scale * filtered[key], color=BLUE, ls=styles[index], lw=1.25, label=label)
    ax.axhline(0.0, color="0.15", lw=0.9, ls="-.")
    ax.set_ylabel(ylabel)
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=len(keys),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )


def closing_figure() -> dict:
    nominal, filtered = load(CLOSING_NOMINAL), load(CLOSING_FILTERED)
    fig = plt.figure(figsize=(10.2, 10.0))
    grid = fig.add_gridspec(
        3,
        4,
        left=0.09,
        right=0.97,
        bottom=0.07,
        top=0.87,
        hspace=0.48,
        wspace=0.32,
    )
    cone_ax = fig.add_subplot(grid[0, 0:2])
    axial_ax = fig.add_subplot(grid[0, 2:4], sharex=cone_ax)
    speed_ax = fig.add_subplot(grid[1, 0:2], sharex=cone_ax)
    attitude_ax = fig.add_subplot(grid[1, 2:4], sharex=cone_ax)
    rate_ax = fig.add_subplot(grid[2, 1:3], sharex=cone_ax)
    plot_barrier_pair(
        cone_ax, nominal, filtered,
        ("h_cone_upper", "h_cone_lower"),
        ("Upper cone", "Lower cone"), r"Cone barrier $h$ [m]",
    )
    plot_barrier_pair(
        axial_ax, nominal, filtered,
        ("h_cone_axial_upper",), ("Finite axial bound",),
        r"Axial-bound $h$ [m]",
    )
    plot_barrier_pair(
        speed_ax, nominal, filtered,
        ("h_port_speed", "h_axial_port_speed"),
        ("Total port speed", "Axial speed"), r"Speed barrier $h$ [m$^2$/s$^2$]",
    )
    plot_barrier_pair(
        attitude_ax, nominal, filtered,
        ("h_attitude_upper", "h_attitude_lower"),
        ("Positive attitude", "Negative attitude"), r"Attitude barrier $h$ [deg]",
        scale=180.0 / np.pi,
    )
    plot_barrier_pair(
        rate_ax,
        nominal,
        filtered,
        ("h_angular_rate_upper", "h_angular_rate_lower"),
        ("Positive angular rate", "Negative angular rate"),
        r"Angular-rate barrier $h$ [deg/s]",
        scale=180.0 / np.pi,
    )
    for ax in (cone_ax, axial_ax, speed_ax, attitude_ax):
        ax.tick_params(labelbottom=False)
    fig.supxlabel("Time [s]", fontsize=10.5)
    fig.suptitle("Closing phase-balanced policy: constraint values versus time (seed 25)", fontsize=12)
    fig.legend(
        handles=(
            Line2D([], [], color=ORANGE, lw=1.5, label="Nominal SAC"),
            Line2D([], [], color=BLUE, lw=1.5, label="SAC + CBF"),
            Line2D([], [], color="0.15", lw=0.9, ls="-.", label=r"Safety boundary $h=0$"),
        ),
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False,
    )
    png = OUT / "closing_phase_balanced_seed25_constraint_values_vs_time.png"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    names = {
        "h_cone_upper": "upper cone",
        "h_cone_lower": "lower cone",
        "h_cone_axial_upper": "finite cone axial bound",
        "h_port_speed": "total port-relative speed",
        "h_axial_port_speed": "axial velocity",
        "h_attitude_upper": "positive attitude",
        "h_attitude_lower": "negative attitude",
        "h_angular_rate_upper": "positive angular rate",
        "h_angular_rate_lower": "negative angular rate",
    }
    nominal_violated = [label for key, label in names.items() if float(np.min(nominal[key])) < 0.0]
    filtered_violated = [label for key, label in names.items() if float(np.min(filtered[key])) < 0.0]
    duration = float(max(nominal["time"][-1], filtered["time"][-1]))
    print("Closing constraints: evaluation seed 25")
    print(f"  episode duration: {duration:.0f} s")
    print(f"  nominal violated constraints: {', '.join(nominal_violated) if nominal_violated else 'none'}")
    print(f"  filtered admissible for every plotted barrier: {'yes' if not filtered_violated else 'no'}")
    if filtered_violated:
        print(f"  filtered barrier crossings: {', '.join(filtered_violated)}")
    return {"seed": 25, "duration_s": duration, "nominal_violations": nominal_violated, "filtered_violations": filtered_violated}


def contiguous_intervals(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[cuts + 1]]
    ends = np.r_[indices[cuts], indices[-1]]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return [(float(times[start]), float(times[end] + dt)) for start, end in zip(starts, ends)]


def _closing_physical_quantities(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    denominator = 0.1 * np.maximum(data["s"].astype(float), 1.0e-12)
    attitude = 0.5 * (data["h_attitude_lower"] - data["h_attitude_upper"])
    angular_rate = 0.5 * (
        data["h_angular_rate_lower"] - data["h_angular_rate_upper"]
    )
    return {
        "time": data["time"].astype(float),
        "cone_ratio": np.abs(data["y_p"].astype(float)) / denominator,
        "axial_position": data["s"].astype(float),
        "total_speed": data["port_speed"].astype(float),
        "axial_speed": np.abs(data["s_dot"].astype(float)),
        "attitude_error_deg": np.abs(np.rad2deg(attitude)),
        "angular_rate_deg_s": np.abs(np.rad2deg(angular_rate)),
    }


def closing_physical_figure() -> dict:
    nominal = _closing_physical_quantities(load(CLOSING_NOMINAL))
    filtered = _closing_physical_quantities(load(CLOSING_FILTERED))
    fig = plt.figure(figsize=(10.2, 10.0))
    grid = fig.add_gridspec(
        3, 4, left=0.09, right=0.97, bottom=0.07, top=0.87,
        hspace=0.39, wspace=0.38,
    )
    axes = (
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 1:3]),
    )

    def panel(ax, key, ylabel, limit, limit_label, *, labelpad=8):
        ax.plot(nominal["time"], nominal[key], color=ORANGE, lw=1.15)
        ax.plot(filtered["time"], filtered[key], color=BLUE, lw=1.25)
        ax.axhline(limit, color="0.2", lw=1.0, ls="--", label=limit_label)
        ax.set_ylabel(ylabel, fontsize=11.5, labelpad=labelpad)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(
            frameon=False, fontsize=9, loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
        )

    panel(
        axes[0], "cone_ratio",
        r"Normalized cone ratio $|r_{p,y}|/(0.1s)$", 1.0,
        r"Cone boundary: ratio $=1$",
    )
    panel(
        axes[1], "axial_position", r"Axial position $s$ [m]", 2000.0,
        r"Finite cone length $L_c=2000$ m",
        labelpad=2,
    )
    axes[2].plot(nominal["time"], nominal["total_speed"], color=ORANGE, lw=1.15, ls="-")
    axes[2].plot(filtered["time"], filtered["total_speed"], color=BLUE, lw=1.25, ls="-")
    axes[2].plot(nominal["time"], nominal["axial_speed"], color=ORANGE, lw=1.15, ls="--")
    axes[2].plot(filtered["time"], filtered["axial_speed"], color=BLUE, lw=1.25, ls="--")
    axes[2].axhline(0.25, color="0.2", lw=1.0, ls=":", label=r"Total-speed limit $0.25$ m/s")
    axes[2].axhline(0.15, color="0.2", lw=1.0, ls="-.", label=r"Axial-speed limit $0.15$ m/s")
    axes[2].set_ylabel("Speed [m/s]", fontsize=11.5, labelpad=8)
    axes[2].tick_params(axis="both", labelsize=10)
    axes[2].legend(
        handles=(
            Line2D([], [], color=BLUE, ls="-", label=r"Total speed $\|v_p\|$"),
            Line2D([], [], color=BLUE, ls="--", label=r"Axial speed $|\dot{s}|$"),
            Line2D([], [], color="0.2", ls=":", label="0.25 m/s limit"),
            Line2D([], [], color="0.2", ls="-.", label="0.15 m/s limit"),
        ),
        frameon=False, fontsize=8.2, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    panel(
        axes[3], "attitude_error_deg", r"Attitude error $|\psi_e|$ [deg]",
        3.0, r"Attitude limit $3^\circ$",
    )
    panel(
        axes[4], "angular_rate_deg_s", r"Angular rate $|\omega|$ [deg/s]",
        0.15, r"Angular-rate limit $0.15^\circ$/s",
    )
    for ax in axes[:4]:
        ax.tick_params(labelbottom=False)
    axes[4].set_xlabel("Time [s]", fontsize=11.5)
    fig.suptitle(
        "Closing phase-balanced policy: physical constraint quantities versus time (seed 25)",
        fontsize=13,
    )
    fig.legend(
        handles=(
            Line2D([], [], color=ORANGE, lw=1.5, label="Nominal SAC"),
            Line2D([], [], color=BLUE, lw=1.5, label="SAC + CBF"),
        ),
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2,
        frameon=False, fontsize=10,
    )
    png = OUT / "closing_phase_balanced_seed25_physical_constraints_vs_time.png"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    limits = {
        "cone_ratio": 1.0,
        "axial_position": 2000.0,
        "total_speed": 0.25,
        "axial_speed": 0.15,
        "attitude_error_deg": 3.0,
        "angular_rate_deg_s": 0.15,
    }
    labels = {
        "cone_ratio": "normalized cone ratio",
        "axial_position": "finite axial cone bound",
        "total_speed": "total port-relative speed",
        "axial_speed": "axial closing speed",
        "attitude_error_deg": "attitude error",
        "angular_rate_deg_s": "angular-rate error",
    }
    nominal_exceeded = [labels[key] for key, limit in limits.items() if float(np.max(nominal[key])) > limit]
    filtered_exceeded = [labels[key] for key, limit in limits.items() if float(np.max(filtered[key])) > limit]
    maxima_keys = ("cone_ratio", "total_speed", "axial_speed", "attitude_error_deg", "angular_rate_deg_s")
    maxima = {
        key: {"nominal": float(np.max(nominal[key])), "sac_cbf": float(np.max(filtered[key]))}
        for key in maxima_keys
    }
    print("Closing physical constraints: evaluation seed 25")
    print(f"  nominal quantities exceeding limits: {', '.join(nominal_exceeded) if nominal_exceeded else 'none'}")
    print(f"  SAC + CBF within all plotted physical limits: {'yes' if not filtered_exceeded else 'no'}")
    if filtered_exceeded:
        print(f"  SAC + CBF quantities exceeding limits: {', '.join(filtered_exceeded)}")
    for key in maxima_keys:
        print(f"  maximum {labels[key]}: nominal={maxima[key]['nominal']:.6g}, SAC+CBF={maxima[key]['sac_cbf']:.6g}")
    return {
        "seed": 25,
        "nominal_limit_exceedances": nominal_exceeded,
        "sac_cbf_limit_exceedances": filtered_exceeded,
        "maxima": maxima,
        "figure": str(png.resolve()),
    }


def final_control_figure() -> list[dict]:
    fig = plt.figure(figsize=(11.0, 8.2))
    outer = fig.add_gridspec(1, 2, left=0.08, right=0.985, bottom=0.08, top=0.86, wspace=0.18)
    reports = []
    channel_names = (r"$a_{x,b_c}$ [m/s$^2$]", r"$a_{y,b_c}$ [m/s$^2$]", r"$\alpha_{z,b_c}$ [deg/s$^2$]")
    actuator_bounds = np.array([0.01, 0.01, 0.003], dtype=float)
    component_colors = ("#009E73", "#CC79A7", "#E69F00")
    for column, (title, campaign, seed) in enumerate(FINAL_CASES):
        path = campaign / f"step_timeseries_all_cbf_eval_seed{seed:04d}.npz"
        data = load(path)
        subgrid = outer[column].subgridspec(4, 1, hspace=0.08, height_ratios=(1, 1, 1, 0.72))
        axes = [fig.add_subplot(subgrid[index, 0]) for index in range(4)]
        time = data["time"].astype(float)
        intervened = data["intervened"].astype(bool)
        intervals = contiguous_intervals(time, intervened)
        for channel, ax in enumerate(axes[:3]):
            conversion = 180.0 / np.pi if channel == 2 else 1.0
            ax.plot(time, conversion * data["action_nominal"][:, channel], color=ORANGE, lw=1.0)
            ax.plot(time, conversion * data["action_filtered"][:, channel], color=BLUE, lw=1.1)
            for start, end in intervals:
                ax.axvspan(start, end, color=GREEN, alpha=0.13, lw=0)
            ax.set_ylabel(channel_names[channel], fontsize=8.5)
            ax.tick_params(labelbottom=False)
        axes[0].set_title(title, fontsize=10.5, pad=8)
        component_magnitudes = np.abs(data["action_filtered"] - data["action_nominal"]) / actuator_bounds
        component_labels = (r"$\Delta u_{x,\rm norm}$", r"$\Delta u_{y,\rm norm}$", r"$\Delta u_{\alpha,\rm norm}$")
        for component, (label, color) in enumerate(zip(component_labels, component_colors)):
            axes[3].plot(time, component_magnitudes[:, component], color=color, lw=1.0, label=label)
        axes[3].set_ylabel("Normalized\ncorrection", fontsize=8.5)
        axes[3].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
        axes[3].set_xlabel("Time [s]", fontsize=9.5)
        count = int(np.sum(intervened))
        rate = float(np.mean(intervened))
        maxima = np.max(component_magnitudes, axis=0)
        pattern = "sparse/localized" if rate < 0.05 else "frequent"
        print(f"{title}")
        print(f"  policy evaluation seed: {seed}")
        print(f"  intervention time steps: {count}")
        print(f"  intervention rate: {100.0 * rate:.3f}%")
        print(
            "  maximum component-normalized corrections: "
            f"x={maxima[0]:.6g}, y={maxima[1]:.6g}, alpha={maxima[2]:.6g}"
        )
        print(f"  intervention pattern: {pattern}")
        reports.append({
            "policy": title,
            "eval_seed": seed,
            "intervention_steps": count,
            "intervention_rate": rate,
            "maximum_component_normalized_corrections": {
                "x": float(maxima[0]), "y": float(maxima[1]), "alpha": float(maxima[2])
            },
            "pattern": pattern,
        })

    fig.suptitle("Final-approach control outputs and CBF interventions versus time", fontsize=12, y=0.975)
    fig.legend(
        handles=(
            Line2D([], [], color=ORANGE, lw=1.5, label="Nominal SAC command"),
            Line2D([], [], color=BLUE, lw=1.5, label="CBF-filtered command"),
            Patch(facecolor=GREEN, alpha=0.18, label="CBF intervention interval"),
        ),
        loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False, fontsize=8.2,
    )
    png = OUT / "final_approach_control_outputs_and_cbf_interventions_vs_time.png"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return reports


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    report = {
        "closing_physical_constraints": closing_physical_figure(),
        "final_controls": final_control_figure(),
    }
    with (OUT / "constraint_and_control_figure_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Figures written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
