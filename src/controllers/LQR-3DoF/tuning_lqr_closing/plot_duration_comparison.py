"""Plot matched closing-LQR rollouts under three phase-duration limits.

The figure contrasts a seed that succeeds before every duration limit with a
seed that times out at every limit.  Each panel overlays the trajectories from
12,000 s, 13,000 s, and 14,000 s evaluations using the selected closing LQR.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LQR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    LQR_ROOT,
    "results",
    "lqr_closing",
    "dt_0p1",
    "duration_comparison",
)


def build_controller(dt):
    """Build the frozen selected closing controller at the requested timestep."""
    gain_env = Docking3DoFEnv(
        phase="closing",
        dt=dt,
        max_time=12_000.0,
        seed=0,
    )
    A, B = build_linearized_3dof_matrices(gain_env)
    Q = build_Q_closing(dt=dt)
    R = build_R_closing(gain_env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    return LQRClosingController(K)


def run_rollout(seed, max_time, dt, mode):
    """Run one full-history rollout and return its axial-error trajectory."""
    env = Docking3DoFEnv(
        phase="closing",
        dt=dt,
        max_time=max_time,
        seed=seed,
    )
    controller = build_controller(dt)
    state = env.reset(mode=mode)

    done = False
    info = {}
    while not done:
        state, done, info = env.step(controller.compute_action(env, state))

    times, states, _, _, _ = env.get_history_arrays()
    axial_error = states[:, 0] - float(env.p3_target[0])
    return {
        "times": np.asarray(times, dtype=float),
        "axial_error": np.asarray(axial_error, dtype=float),
        "status": "success" if info.get("success") else info.get(
            "termination_reason", "unknown"
        ),
        "elapsed_time": float(env.time),
        "terminal_axial_error": float(axial_error[-1]),
        "axial_tolerance": float(env.p3_target_tolerance[0]),
    }


def decimate_for_plot(times, values, max_points=6_000):
    """Limit vector-figure size while preserving both trajectory endpoints."""
    if len(times) <= max_points:
        return times, values
    indices = np.linspace(0, len(times) - 1, max_points, dtype=np.int64)
    indices = np.unique(indices)
    return times[indices], values[indices]


def add_duration_curves(ax, rollouts, durations):
    # Plot the longest history first. Shorter histories then remain visible over
    # the common interval, while the extra-duration tails retain their colors.
    styles = {
        12_000.0: ("tab:blue", ":", "o"),
        13_000.0: ("tab:orange", "--", "s"),
        14_000.0: ("tab:red", "-", "D"),
    }
    for zorder, duration in enumerate(sorted(durations, reverse=True), start=2):
        rollout = rollouts[duration]
        color, linestyle, marker = styles[duration]
        times, axial_error = decimate_for_plot(
            rollout["times"], rollout["axial_error"]
        )
        ax.plot(
            times,
            axial_error,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"limit = {duration:,.0f} s",
            zorder=zorder,
        )
        ax.scatter(
            rollout["elapsed_time"],
            rollout["terminal_axial_error"],
            color=color,
            marker=marker,
            s=34,
            edgecolor="white",
            linewidth=0.6,
            zorder=8,
        )


def format_panel(ax, tolerance):
    ax.axhspan(
        -tolerance,
        tolerance,
        color="tab:green",
        alpha=0.10,
        label=r"axial tolerance ($|e_x|\leq20$ m)",
        zorder=0,
    )
    ax.axhline(0.0, color="0.35", linewidth=0.9, zorder=1)
    ax.axhline(tolerance, color="tab:green", linestyle="--", linewidth=1.0)
    ax.axhline(-tolerance, color="tab:green", linestyle="--", linewidth=1.0)
    ax.set_ylabel(r"Axial-position error $e_x$ [m]")
    ax.grid(True, alpha=0.28)


def terminal_summary(rollouts, durations):
    return "\n".join(
        f"{duration / 1000:.0f}k seconds: "
        f"$e_x={rollouts[duration]['terminal_axial_error']:.1f}$ m"
        for duration in durations
    )


def make_figure(success_seed, timeout_seed, durations, dt, mode, output_dir):
    cases = {
        "success": {
            duration: run_rollout(success_seed, duration, dt, mode)
            for duration in durations
        },
        "timeout": {
            duration: run_rollout(timeout_seed, duration, dt, mode)
            for duration in durations
        },
    }

    success_statuses = {item["status"] for item in cases["success"].values()}
    timeout_statuses = {item["status"] for item in cases["timeout"].values()}
    if success_statuses != {"success"}:
        raise RuntimeError(
            f"Seed {success_seed} was expected to succeed at all limits; "
            f"observed {sorted(success_statuses)}."
        )
    if timeout_statuses != {"timeout"}:
        raise RuntimeError(
            f"Seed {timeout_seed} was expected to time out at all limits; "
            f"observed {sorted(timeout_statuses)}."
        )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)

    tolerance = cases["success"][durations[0]]["axial_tolerance"]
    for ax in axes:
        format_panel(ax, tolerance)

    add_duration_curves(axes[0], cases["success"], durations)
    success_time = cases["success"][durations[0]]["elapsed_time"]
    axes[0].set_title(f"(a) Successful closing rollout (seed {success_seed})")
    axes[0].text(
        0.985,
        0.94,
        f"All trajectories coincide\nand succeed at {success_time:,.0f} s",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )

    add_duration_curves(axes[1], cases["timeout"], durations)
    axes[1].set_title(f"(b) Persistent closing timeout (seed {timeout_seed})")
    axes[1].text(
        0.985,
        0.94,
        terminal_summary(cases["timeout"], durations),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    axes[1].set_xlabel("Physical time [s]")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.15, hspace=0.25)

    os.makedirs(output_dir, exist_ok=True)
    base = f"closing_duration_comparison_seed{success_seed:04d}_seed{timeout_seed:04d}"
    png_path = os.path.join(output_dir, f"{base}.png")
    pdf_path = os.path.join(output_dir, f"{base}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    for case_name, seed in (("success", success_seed), ("timeout", timeout_seed)):
        print(f"{case_name} seed {seed}:")
        for duration in durations:
            item = cases[case_name][duration]
            print(
                f"  limit={duration:.0f} s status={item['status']} "
                f"elapsed={item['elapsed_time']:.1f} s "
                f"terminal_ex={item['terminal_axial_error']:.6f} m"
            )

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot successful and timed-out closing rollouts at matched duration limits."
    )
    parser.add_argument("--success-seed", type=int, default=41)
    parser.add_argument("--timeout-seed", type=int, default=36)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--mode", choices=["nominal", "testing", "training"], default="testing")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    make_figure(
        success_seed=args.success_seed,
        timeout_seed=args.timeout_seed,
        durations=(12_000.0, 13_000.0, 14_000.0),
        dt=args.dt,
        mode=args.mode,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
