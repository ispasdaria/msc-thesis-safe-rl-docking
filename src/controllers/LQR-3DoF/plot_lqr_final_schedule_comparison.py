"""Plot matched final-approach rollouts with and without velocity scheduling.

Both cases use the same nonlinear simulator, initial P3 state, port-shaped LQR
gain, actuator limits, integration method, and time limit.  The only controlled
difference is the axial docking-port velocity reference:

* without scheduling: ``v_px_ref = 0`` over the complete final approach;
* with scheduling: the implemented distance-dependent far/creep reference.

The resulting figure is an illustrative controller-development comparison,
not a replacement for stochastic controller evaluation.
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
from lqr_3dof_final import (
    LQRFinalController,
    build_Q_final,
    build_R_final,
    build_linearized_3dof_matrices,
    build_port_error_mapping,
    compute_lqr_gain,
    final_error,
    orbital_to_body_action,
)


V_FAR = -0.07
D_FAST = 30.0
D_SLOW = 1.0
V_CREEP = -0.005


class ZeroPortVelocityReferenceController:
    """Port-shaped final LQR with an unscheduled zero velocity reference."""

    def __init__(self, gain: np.ndarray, port_mapping: np.ndarray):
        self.K = np.asarray(gain, dtype=float)
        self.H = np.asarray(port_mapping, dtype=float)
        if self.K.shape != (3, 6):
            raise ValueError("gain must have shape (3, 6)")
        if self.H.shape != (6, 6):
            raise ValueError("port_mapping must have shape (6, 6)")

    def compute_action(self, env: Docking3DoFEnv, state: np.ndarray) -> np.ndarray:
        if env.phase != "final":
            raise ValueError("The zero-reference controller requires the final phase.")

        state = np.asarray(state, dtype=float)
        e_port = final_error(env, state)
        # final_error already defines the axial velocity error as v_px - 0.
        e_com_equiv = np.linalg.solve(self.H, e_port)
        action_orbital = -self.K @ e_com_equiv
        return orbital_to_body_action(env, action_orbital, psi=state[4])


def scheduled_port_velocity_reference(state: np.ndarray, env: Docking3DoFEnv) -> float:
    """Evaluate the axial port-velocity schedule used by LQRFinalController."""
    r_px = float(env.get_port_relative_position(state)[0])
    distance = abs(r_px)

    if distance >= D_FAST:
        return V_FAR
    if distance <= D_SLOW:
        return V_CREEP

    alpha = (distance - D_SLOW) / (D_FAST - D_SLOW)
    return (1.0 - alpha) * V_CREEP + alpha * V_FAR


def build_gain_and_mapping(env: Docking3DoFEnv) -> tuple[np.ndarray, np.ndarray]:
    """Construct the implemented port-shaped final-approach LQR design."""
    Q_port = build_Q_final()
    R = build_R_final(env)
    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_com = H.T @ Q_port @ H
    K, _ = compute_lqr_gain(A, B, Q_com, R)
    return K, H


def run_rollout(
    env: Docking3DoFEnv,
    controller,
    mode: str,
    reference_kind: str,
) -> dict:
    """Run one nonlinear final-approach rollout and collect plot quantities."""
    state = env.reset(mode=mode)
    initial_state = state.copy()

    references = []
    done = False
    info = {}
    max_steps = int(np.ceil(env.max_time / env.dt)) + 1

    for _ in range(max_steps):
        if reference_kind == "zero":
            references.append(0.0)
        elif reference_kind == "scheduled":
            references.append(scheduled_port_velocity_reference(state, env))
        else:
            raise ValueError(f"Unknown reference kind: {reference_kind}")

        action = controller.compute_action(env, state)
        state, done, info = env.step(action)
        if done:
            break

    times, states, actions_raw, actions_clipped, _ = env.get_history_arrays()
    if reference_kind == "zero":
        initial_reference = 0.0
    else:
        initial_reference = scheduled_port_velocity_reference(initial_state, env)
    reference_history = np.asarray([initial_reference, *references], dtype=float)
    reference_history = reference_history[: len(times)]

    port_position = np.asarray(
        [env.get_port_relative_position(sample) for sample in states], dtype=float
    )
    port_velocity = np.asarray(
        [env.get_port_relative_velocity(sample) for sample in states], dtype=float
    )
    saturation = np.any(
        np.abs(actions_raw - actions_clipped) > 1e-12,
        axis=1,
    )

    return {
        "times": times,
        "states": states,
        "port_position": port_position,
        "port_velocity": port_velocity,
        "port_distance": np.linalg.norm(port_position, axis=1),
        "axial_port_distance": np.abs(port_position[:, 0]),
        "vpx_reference": reference_history,
        "initial_state": initial_state,
        "terminal_state": state.copy(),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "elapsed_time": float(env.time),
        "saturation_fraction": float(np.mean(saturation)),
    }


def assert_matched_cases(without_schedule: dict, with_schedule: dict) -> None:
    """Guard against accidentally comparing different sampled P3 states."""
    if not np.array_equal(
        without_schedule["initial_state"], with_schedule["initial_state"]
    ):
        raise RuntimeError("The two cases did not receive the same initial P3 state.")


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
    without_schedule: dict,
    with_schedule: dict,
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
    colors = {"without": "#777777", "with": "#d97706"}

    without_label = f"Without scheduled reference ({status_text(without_schedule)})"
    with_label = f"With scheduled reference ({status_text(with_schedule)})"

    ax = axes[0]
    ax.plot(
        without_schedule["times"],
        without_schedule["port_distance"],
        color=colors["without"],
        linewidth=2.0,
        label=without_label,
    )
    ax.plot(
        with_schedule["times"],
        with_schedule["port_distance"],
        color=colors["with"],
        linewidth=2.2,
        label=with_label,
    )
    docking_tolerance = 0.1
    ax.axhline(
        docking_tolerance,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label=r"Docking-port distance tolerance ($\|r_p\|=0.1$ m)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time, $t$ [s]")
    ax.set_ylabel(r"Docking-port distance, $\|r_p\|$ [m]")
    ax.set_title("(a) Progress toward the docking interface")
    ax.set_xlim(left=0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    without_dt = float(np.median(np.diff(without_schedule["times"])))
    with_dt = float(np.median(np.diff(with_schedule["times"])))
    without_window = max(1, round(smoothing_window_s / without_dt))
    with_window = max(1, round(smoothing_window_s / with_dt))
    without_vpx_mean = moving_average(
        without_schedule["port_velocity"][:, 0], without_window
    )
    with_vpx_mean = moving_average(
        with_schedule["port_velocity"][:, 0], with_window
    )

    # Keep the raw nonlinear-simulator velocities faintly visible.
    ax.plot(
        without_schedule["axial_port_distance"],
        without_schedule["port_velocity"][:, 0],
        color=colors["without"],
        linewidth=0.45,
        alpha=0.16,
        label="_nolegend_",
    )
    ax.plot(
        with_schedule["axial_port_distance"],
        with_schedule["port_velocity"][:, 0],
        color=colors["with"],
        linewidth=0.45,
        alpha=0.16,
        label="_nolegend_",
    )
    ax.plot(
        without_schedule["axial_port_distance"],
        without_vpx_mean,
        color=colors["without"],
        linewidth=2.0,
        label=f"Without scheduled reference ({smoothing_window_s:g} s mean)",
    )
    ax.plot(
        with_schedule["axial_port_distance"],
        with_vpx_mean,
        color=colors["with"],
        linewidth=2.2,
        label=f"With scheduled reference ({smoothing_window_s:g} s mean)",
    )
    ax.plot(
        without_schedule["axial_port_distance"],
        without_schedule["vpx_reference"],
        color=colors["without"],
        linewidth=1.3,
        linestyle="--",
        alpha=0.9,
        label=r"$v_{p,x}^{\mathrm{ref}}=0$",
    )
    ax.plot(
        with_schedule["axial_port_distance"],
        with_schedule["vpx_reference"],
        color=colors["with"],
        linewidth=1.3,
        linestyle="--",
        alpha=0.9,
        label=r"Scheduled $v_{p,x}^{\mathrm{ref}}$",
    )
    ax.set_xlabel(r"Remaining axial port distance, $|r_{p,x}|$ [m]")
    ax.set_ylabel(r"Axial port velocity, $v_{p,x}$ [m/s]")
    ax.set_title("(b) Axial port velocity over the final-approach distance")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.suptitle(
        "Final-approach LQR development comparison on the nonlinear simulator",
        fontsize=12,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight")
    plt.close(fig)


def save_summary(
    without_schedule: dict,
    with_schedule: dict,
    output_stem: Path,
    args,
) -> None:
    """Save exact paired-run metadata accompanying the generated figure."""
    summary = {
        "purpose": "illustrative controller-development comparison",
        "controlled_difference": {
            "without_schedule": "v_px_ref equals zero",
            "with_schedule": "implemented distance-dependent v_far/v_creep reference",
        },
        "schedule": {
            "v_far_m_per_s": V_FAR,
            "d_fast_m": D_FAST,
            "d_slow_m": D_SLOW,
            "v_creep_m_per_s": V_CREEP,
        },
        "mode": args.mode,
        "seed": args.seed,
        "dt_s": args.dt,
        "max_time_s": args.max_time,
        "integrator": args.integrator,
        "velocity_plot_moving_average_window_s": args.smoothing_window,
        "initial_state": without_schedule["initial_state"].tolist(),
        "without_schedule": {
            "success": without_schedule["success"],
            "termination_reason": without_schedule["termination_reason"],
            "elapsed_time_s": without_schedule["elapsed_time"],
            "final_port_distance_m": float(without_schedule["port_distance"][-1]),
            "final_port_speed_m_per_s": float(
                np.linalg.norm(without_schedule["port_velocity"][-1])
            ),
            "saturation_fraction": without_schedule["saturation_fraction"],
        },
        "with_schedule": {
            "success": with_schedule["success"],
            "termination_reason": with_schedule["termination_reason"],
            "elapsed_time_s": with_schedule["elapsed_time"],
            "final_port_distance_m": float(with_schedule["port_distance"][-1]),
            "final_port_speed_m_per_s": float(
                np.linalg.norm(with_schedule["port_velocity"][-1])
            ),
            "saturation_fraction": with_schedule["saturation_fraction"],
        },
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare zero and distance-scheduled port-velocity references in "
            "matched nonlinear final-approach rollouts."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("nominal", "testing", "training"),
        default="nominal",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=5000.0)
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
        / "final_velocity_schedule_comparison",
        help="Output path without an extension. PNG, PDF, and JSON are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing_window <= 0.0:
        raise ValueError("--smoothing-window must be positive")

    env_kwargs = {
        "phase": "final",
        "dt": args.dt,
        "max_time": args.max_time,
        "integrator": args.integrator,
        "seed": args.seed,
    }
    without_env = Docking3DoFEnv(**env_kwargs)
    with_env = Docking3DoFEnv(**env_kwargs)
    K, H = build_gain_and_mapping(without_env)

    without_schedule = run_rollout(
        without_env,
        ZeroPortVelocityReferenceController(K, H),
        mode=args.mode,
        reference_kind="zero",
    )
    with_schedule = run_rollout(
        with_env,
        LQRFinalController(K, H),
        mode=args.mode,
        reference_kind="scheduled",
    )
    assert_matched_cases(without_schedule, with_schedule)

    output_stem = args.output.resolve()
    plot_comparison(
        without_schedule,
        with_schedule,
        output_stem,
        smoothing_window_s=args.smoothing_window,
    )
    save_summary(without_schedule, with_schedule, output_stem, args)

    print(f"Without schedule: {status_text(without_schedule)}")
    print(f"With schedule:    {status_text(with_schedule)}")
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {output_stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
