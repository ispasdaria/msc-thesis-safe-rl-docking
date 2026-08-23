"""Plot a matched closing-phase rollout comparison with and without scheduling.

The two nonlinear simulations use the same initial state, sampled P3 target,
LQR gain, actuator limits, integration method, and time limit.  The only
difference is the axial-velocity reference:

* fixed reference: ``vx_ref = vx_P3`` during the complete closing phase;
* scheduled reference: the implemented distance-dependent closing reference.

This is intended as an illustrative controller-development figure, not as a
replacement for stochastic controller evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LQR_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    closing_error,
    compute_lqr_gain,
    orbital_to_body_action,
)


class FixedP3ReferenceController:
    """Closing LQR that tracks the complete sampled P3 state directly."""

    def __init__(self, gain: np.ndarray):
        self.K = np.asarray(gain, dtype=float)
        if self.K.shape != (3, 6):
            raise ValueError("gain must have shape (3, 6)")

    def compute_action(self, env: Docking3DoFEnv, state: np.ndarray) -> np.ndarray:
        if env.phase != "closing":
            raise ValueError("The fixed-reference controller requires the closing phase.")
        if env.p3_target is None:
            raise RuntimeError("Reset the closing environment before computing an action.")

        state = np.asarray(state, dtype=float)
        error = closing_error(env, state, env.p3_target)
        action_orbital = -self.K @ error
        return orbital_to_body_action(env, action_orbital, psi=state[4])


def scheduled_axial_reference(
    state: np.ndarray,
    p3_target: np.ndarray,
    controller: LQRClosingController,
) -> float:
    """Evaluate the same axial-reference law used by LQRClosingController."""
    distance = abs(float(state[0] - p3_target[0]))
    v_p3 = float(p3_target[2])

    if distance >= controller.d_fast:
        return controller.v_fast
    if distance <= controller.d_slow:
        return v_p3

    alpha = (distance - controller.d_slow) / (
        controller.d_fast - controller.d_slow
    )
    return (1.0 - alpha) * v_p3 + alpha * controller.v_fast


def build_gain(env: Docking3DoFEnv) -> np.ndarray:
    """Construct the implemented closing-phase LQR gain."""
    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    return K


def run_rollout(
    env: Docking3DoFEnv,
    controller,
    mode: str,
    reference_kind: str,
) -> dict:
    """Run one nonlinear closing rollout and collect the plotted quantities."""
    state = env.reset(mode=mode)
    initial_state = state.copy()
    p3_target = env.p3_target.copy()

    references = []
    done = False
    info = {}
    max_steps = int(np.ceil(env.max_time / env.dt)) + 1

    for _ in range(max_steps):
        if reference_kind == "fixed":
            references.append(float(p3_target[2]))
        elif reference_kind == "scheduled":
            references.append(
                scheduled_axial_reference(state, p3_target, controller)
            )
        else:
            raise ValueError(f"Unknown reference kind: {reference_kind}")

        action = controller.compute_action(env, state)
        state, done, info = env.step(action)
        if done:
            break

    times, states, actions_raw, actions_clipped, _ = env.get_history_arrays()

    # The environment logs the initial condition before the first action.  Add
    # its reference value so that the reference and state arrays have equal size.
    if reference_kind == "fixed":
        initial_reference = float(p3_target[2])
    else:
        initial_reference = scheduled_axial_reference(
            initial_state, p3_target, controller
        )
    reference_history = np.asarray([initial_reference, *references], dtype=float)
    reference_history = reference_history[: len(times)]

    distance = np.abs(states[:, 0] - p3_target[0])
    saturation = np.any(
        np.abs(actions_raw - actions_clipped) > 1e-12,
        axis=1,
    )

    return {
        "times": times,
        "states": states,
        "distance": distance,
        "vx_reference": reference_history,
        "initial_state": initial_state,
        "p3_target": p3_target,
        "terminal_state": state.copy(),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "elapsed_time": float(env.time),
        "saturation_fraction": float(np.mean(saturation)),
    }


def assert_matched_cases(fixed: dict, scheduled: dict) -> None:
    """Guard against accidentally comparing different sampled episodes."""
    if not np.array_equal(fixed["initial_state"], scheduled["initial_state"]):
        raise RuntimeError("The two cases did not receive the same initial state.")
    if not np.array_equal(fixed["p3_target"], scheduled["p3_target"]):
        raise RuntimeError("The two cases did not receive the same sampled P3 target.")


def status_text(result: dict) -> str:
    if result["success"]:
        return f"success at {result['elapsed_time']:.0f} s"
    return f"{result['termination_reason'].replace('_', ' ')} at {result['elapsed_time']:.0f} s"


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Return an edge-padded moving average without shortening the history."""
    values = np.asarray(values, dtype=float)
    window_samples = max(1, int(window_samples))
    if window_samples == 1 or values.size < 2:
        return values.copy()
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = min(window_samples, 2 * (values.size // 2) + 1)
    half_window = window_samples // 2
    padded = np.pad(values, (half_window, half_window), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(padded, kernel, mode="valid")


def plot_comparison(
    fixed: dict,
    scheduled: dict,
    output_stem: Path,
    smoothing_window_s: float,
) -> None:
    """Create and save the standalone two-panel development comparison."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    colors = {"fixed": "#777777", "scheduled": "#1769aa"}

    fixed_label = f"Fixed $P_3$ reference ({status_text(fixed)})"
    scheduled_label = f"Scheduled reference ({status_text(scheduled)})"

    ax = axes[0]
    ax.plot(
        fixed["times"],
        fixed["distance"],
        color=colors["fixed"],
        linewidth=2.0,
        label=fixed_label,
    )
    ax.plot(
        scheduled["times"],
        scheduled["distance"],
        color=colors["scheduled"],
        linewidth=2.2,
        label=scheduled_label,
    )
    axial_tolerance = 20.0
    ax.axhline(
        axial_tolerance,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label=r"Axial $P_3$ tolerance ($|e_x|=20$ m)",
    )
    ax.set_xlabel("Time, $t$ [s]")
    ax.set_ylabel(r"Remaining axial distance, $|r_x-r_{x,P_3}|$ [m]")
    ax.set_title("(a) Progress toward the sampled $P_3$ target")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    fixed_dt = float(np.median(np.diff(fixed["times"])))
    scheduled_dt = float(np.median(np.diff(scheduled["times"])))
    fixed_window = max(1, round(smoothing_window_s / fixed_dt))
    scheduled_window = max(1, round(smoothing_window_s / scheduled_dt))
    fixed_vx_mean = moving_average(fixed["states"][:, 2], fixed_window)
    scheduled_vx_mean = moving_average(
        scheduled["states"][:, 2], scheduled_window
    )

    # Show the unsmoothed simulator histories faintly beneath the moving means.
    ax.plot(
        fixed["distance"],
        fixed["states"][:, 2],
        color=colors["fixed"],
        linewidth=0.45,
        alpha=0.16,
        label="_nolegend_",
    )
    ax.plot(
        scheduled["distance"],
        scheduled["states"][:, 2],
        color=colors["scheduled"],
        linewidth=0.45,
        alpha=0.16,
        label="_nolegend_",
    )
    ax.plot(
        fixed["distance"],
        fixed_vx_mean,
        color=colors["fixed"],
        linewidth=2.0,
        label=(
            f"Without scheduled reference ({smoothing_window_s:g} s mean)"
        ),
    )
    ax.plot(
        scheduled["distance"],
        scheduled_vx_mean,
        color=colors["scheduled"],
        linewidth=2.2,
        label=(
            f"With scheduled reference ({smoothing_window_s:g} s mean)"
        ),
    )
    ax.plot(
        fixed["distance"],
        fixed["vx_reference"],
        color=colors["fixed"],
        linewidth=1.3,
        linestyle="--",
        alpha=0.9,
        label=r"$v_x^{\mathrm{ref}}=v_{x,P_3}$",
    )
    ax.plot(
        scheduled["distance"],
        scheduled["vx_reference"],
        color=colors["scheduled"],
        linewidth=1.3,
        linestyle="--",
        alpha=0.9,
        label=r"Scheduled $v_x^{\mathrm{ref}}$",
    )
    ax.set_xlabel(r"Remaining axial distance, $|r_x-r_{x,P_3}|$ [m]")
    ax.set_ylabel(r"Axial velocity, $v_x$ [m/s]")
    ax.set_title("(b) Axial velocity over the closing distance")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.suptitle(
        "Closing-phase LQR development comparison on the nonlinear simulator",
        fontsize=12,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight")
    plt.close(fig)


def save_summary(fixed: dict, scheduled: dict, output_stem: Path, args) -> None:
    """Save the exact paired-run metadata used to generate the figure."""
    summary = {
        "purpose": "illustrative controller-development comparison",
        "controlled_difference": {
            "fixed": "vx_ref equals the sampled P3 axial velocity",
            "scheduled": "implemented distance-dependent axial-velocity reference",
        },
        "mode": args.mode,
        "seed": args.seed,
        "dt_s": args.dt,
        "max_time_s": args.max_time,
        "integrator": args.integrator,
        "velocity_plot_moving_average_window_s": args.smoothing_window,
        "initial_state": fixed["initial_state"].tolist(),
        "p3_target": fixed["p3_target"].tolist(),
        "fixed": {
            "success": fixed["success"],
            "termination_reason": fixed["termination_reason"],
            "elapsed_time_s": fixed["elapsed_time"],
            "final_axial_distance_m": float(fixed["distance"][-1]),
            "saturation_fraction": fixed["saturation_fraction"],
        },
        "scheduled": {
            "success": scheduled["success"],
            "termination_reason": scheduled["termination_reason"],
            "elapsed_time_s": scheduled["elapsed_time"],
            "final_axial_distance_m": float(scheduled["distance"][-1]),
            "saturation_fraction": scheduled["saturation_fraction"],
        },
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed and distance-scheduled closing LQR references in "
            "matched nonlinear-simulator rollouts."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("nominal", "testing", "training"),
        default="nominal",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument("--integrator", choices=("euler", "rk4"), default="rk4")
    parser.add_argument(
        "--smoothing-window",
        type=float,
        default=60.0,
        help=(
            "Moving-average window in seconds for the velocity-distance panel. "
            "The unsmoothed histories remain visible as faint lines."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(LQR_ROOT)
        / "results"
        / "development_comparisons"
        / "closing_velocity_schedule_comparison",
        help="Output path without an extension. PNG, PDF, and JSON are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing_window <= 0.0:
        raise ValueError("--smoothing-window must be positive")
    env_kwargs = {
        "phase": "closing",
        "dt": args.dt,
        "max_time": args.max_time,
        "integrator": args.integrator,
        "seed": args.seed,
    }

    fixed_env = Docking3DoFEnv(**env_kwargs)
    scheduled_env = Docking3DoFEnv(**env_kwargs)
    K = build_gain(fixed_env)

    fixed = run_rollout(
        fixed_env,
        FixedP3ReferenceController(K),
        mode=args.mode,
        reference_kind="fixed",
    )
    scheduled = run_rollout(
        scheduled_env,
        LQRClosingController(K),
        mode=args.mode,
        reference_kind="scheduled",
    )
    assert_matched_cases(fixed, scheduled)

    output_stem = args.output.resolve()
    plot_comparison(
        fixed,
        scheduled,
        output_stem,
        smoothing_window_s=args.smoothing_window,
    )
    save_summary(fixed, scheduled, output_stem, args)

    print(f"Fixed reference:    {status_text(fixed)}")
    print(f"Scheduled reference: {status_text(scheduled)}")
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {output_stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
