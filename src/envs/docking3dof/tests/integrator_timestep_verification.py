"""Numerical verification study for the 3-DoF propagators.

This script compares the fixed-step Euler and RK4 implementations in
``Docking3DoFEnv`` with an adaptive, high-accuracy DOP853 reference solution.
It is an offline verification study, not the integrator used by the runtime
environment.

The study covers representative closing and final-approach states with zero
and nonzero initial attitude/angular rate. A constant nonzero body-frame input
is used so that every run solves the same continuous-time initial-value
problem. The script also runs a terminal-event resolution test to distinguish
integration accuracy from the temporal resolution of safety/success checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.integrate import solve_ivp


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from envs.docking3dof.core import Docking3DoFEnv


DEFAULT_DTS = (4.0, 2.0, 1.0, 0.5, 0.25)
DEFAULT_ACTION_BODY = np.array(
    [2.0e-4, 1.0e-4, np.deg2rad(1.0e-4)],
    dtype=float,
)


@dataclass(frozen=True)
class Scenario:
    """One representative continuous-time propagation case."""

    name: str
    phase: str
    rotational_case: str
    duration: float
    initial_state: tuple[float, float, float, float, float, float]


SCENARIOS = (
    Scenario(
        name="closing_zero_rotation",
        phase="closing",
        rotational_case="zero",
        duration=200.0,
        initial_state=(1700.0, 20.0, -0.10, 0.0, 0.0, 0.0),
    ),
    Scenario(
        name="closing_nonzero_rotation",
        phase="closing",
        rotational_case="nonzero",
        duration=200.0,
        initial_state=(
            1700.0,
            20.0,
            -0.10,
            0.0,
            np.deg2rad(3.0),
            np.deg2rad(0.05),
        ),
    ),
    Scenario(
        name="final_zero_rotation",
        phase="final",
        rotational_case="zero",
        duration=100.0,
        initial_state=(300.0, 2.0, -0.03, 0.0, 0.0, 0.0),
    ),
    Scenario(
        name="final_nonzero_rotation",
        phase="final",
        rotational_case="nonzero",
        duration=100.0,
        initial_state=(
            300.0,
            2.0,
            -0.03,
            0.0,
            np.deg2rad(1.0),
            np.deg2rad(0.02),
        ),
    ),
)


def _time_grid(duration: float, dt: float) -> np.ndarray:
    """Return a grid ending exactly at ``duration``."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Every timestep must be finite and positive.")

    step_count_float = duration / dt
    step_count = int(round(step_count_float))
    if not np.isclose(step_count_float, step_count, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Duration {duration} s is not an integer multiple of dt={dt} s."
        )

    return np.arange(step_count + 1, dtype=float) * dt


def run_fixed_step_trajectory(
    scenario: Scenario,
    integrator: str,
    dt: float,
    action_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one scenario with the environment's fixed-step integrator."""

    times = _time_grid(scenario.duration, dt)
    states = np.empty((len(times), 6), dtype=float)
    states[0] = np.asarray(scenario.initial_state, dtype=float)

    env = Docking3DoFEnv(
        phase=scenario.phase,
        dt=dt,
        max_time=scenario.duration,
        integrator=integrator,
        seed=0,
    )

    for index in range(len(times) - 1):
        states[index + 1] = env.integrate(states[index], action_body)

    return times, states


def build_reference_solution(
    scenario: Scenario,
    action_body: np.ndarray,
    rtol: float,
    atol: float,
):
    """Build an adaptive DOP853 dense reference solution for one scenario."""

    env = Docking3DoFEnv(
        phase=scenario.phase,
        dt=1.0,
        max_time=scenario.duration,
        integrator="rk4",
        seed=0,
    )
    initial_state = np.asarray(scenario.initial_state, dtype=float)

    solution = solve_ivp(
        fun=lambda _time, state: env.dynamics(state, action_body),
        t_span=(0.0, scenario.duration),
        y0=initial_state,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(
            f"DOP853 failed for scenario {scenario.name}: {solution.message}"
        )

    return solution


def compute_error_metrics(
    states: np.ndarray,
    reference_states: np.ndarray,
) -> dict[str, float]:
    """Compute component-wise final and maximum trajectory errors."""

    difference = states - reference_states
    position_error = np.linalg.norm(difference[:, 0:2], axis=1)
    velocity_error = np.linalg.norm(difference[:, 2:4], axis=1)
    attitude_error_deg = np.abs(np.rad2deg(difference[:, 4]))
    angular_rate_error_deg_s = np.abs(np.rad2deg(difference[:, 5]))

    return {
        "max_position_error_m": float(np.max(position_error)),
        "final_position_error_m": float(position_error[-1]),
        "max_velocity_error_m_s": float(np.max(velocity_error)),
        "final_velocity_error_m_s": float(velocity_error[-1]),
        "max_attitude_error_deg": float(np.max(attitude_error_deg)),
        "final_attitude_error_deg": float(attitude_error_deg[-1]),
        "max_angular_rate_error_deg_s": float(
            np.max(angular_rate_error_deg_s)
        ),
        "final_angular_rate_error_deg_s": float(
            angular_rate_error_deg_s[-1]
        ),
    }


def run_convergence_study(
    dts: list[float],
    action_body: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run all scenarios and return detailed error rows."""

    rows: list[dict[str, Any]] = []
    reference_evaluations: dict[str, int] = {}

    for scenario in SCENARIOS:
        reference = build_reference_solution(
            scenario=scenario,
            action_body=action_body,
            rtol=rtol,
            atol=atol,
        )
        reference_evaluations[scenario.name] = int(reference.nfev)

        for integrator in ("euler", "rk4"):
            for dt in dts:
                times, states = run_fixed_step_trajectory(
                    scenario=scenario,
                    integrator=integrator,
                    dt=dt,
                    action_body=action_body,
                )
                reference_states = reference.sol(times).T
                metrics = compute_error_metrics(states, reference_states)

                rows.append(
                    {
                        "scenario": scenario.name,
                        "phase": scenario.phase,
                        "rotational_case": scenario.rotational_case,
                        "duration_s": scenario.duration,
                        "integrator": integrator,
                        "dt_s": dt,
                        **metrics,
                    }
                )

    return rows, reference_evaluations


def build_worst_case_summary(
    detailed_rows: list[dict[str, Any]],
    dts: list[float],
) -> list[dict[str, Any]]:
    """Aggregate the worst error over the four scenarios for each method/dt."""

    summary: list[dict[str, Any]] = []

    for integrator in ("euler", "rk4"):
        method_rows: list[dict[str, Any]] = []
        for dt in dts:
            subset = [
                row
                for row in detailed_rows
                if row["integrator"] == integrator and row["dt_s"] == dt
            ]
            position_worst = max(
                subset,
                key=lambda row: row["max_position_error_m"],
            )
            method_rows.append(
                {
                    "integrator": integrator,
                    "dt_s": dt,
                    "max_position_error_m": max(
                        row["max_position_error_m"] for row in subset
                    ),
                    "max_position_error_scenario": position_worst["scenario"],
                    "max_velocity_error_m_s": max(
                        row["max_velocity_error_m_s"] for row in subset
                    ),
                    "max_attitude_error_deg": max(
                        row["max_attitude_error_deg"] for row in subset
                    ),
                    "max_angular_rate_error_deg_s": max(
                        row["max_angular_rate_error_deg_s"] for row in subset
                    ),
                    "position_order_to_next_finer": None,
                }
            )

        for index in range(len(method_rows) - 1):
            coarse = method_rows[index]
            fine = method_rows[index + 1]
            step_ratio = coarse["dt_s"] / fine["dt_s"]
            error_ratio = (
                coarse["max_position_error_m"]
                / fine["max_position_error_m"]
            )
            coarse["position_order_to_next_finer"] = float(
                np.log(error_ratio) / np.log(step_ratio)
            )

        summary.extend(method_rows)

    return summary


def run_terminal_event_resolution_test(
    dts: list[float],
) -> list[dict[str, Any]]:
    """Check whether each timestep resolves a narrow terminal docking event."""

    # At psi=0, the implemented geometry gives s = rx - 24.1 m. This state
    # begins 0.15 m from contact at the maximum accepted docking speed.
    initial_state = np.array([24.25, 0.0, -0.10, 0.0, 0.0, 0.0])
    zero_action = np.zeros(3, dtype=float)
    rows: list[dict[str, Any]] = []
    event_horizon = max(5.0, max(dts))

    for integrator in ("euler", "rk4"):
        for dt in dts:
            env = Docking3DoFEnv(
                phase="final",
                dt=dt,
                max_time=event_horizon,
                integrator=integrator,
                seed=0,
            )
            env.reset(initial_state=initial_state, mode="nominal")

            done = False
            step_count = 0
            info: dict[str, Any] = {}
            max_steps = int(np.ceil(event_horizon / dt)) + 1
            while not done and step_count < max_steps:
                _, _, done, info = env.step(zero_action)
                step_count += 1

            rows.append(
                {
                    "integrator": integrator,
                    "dt_s": dt,
                    "steps": step_count,
                    "event_time_s": env.time,
                    "final_axial_separation_m": float(info["s"]),
                    "final_port_distance_m": float(info["port_distance"]),
                    "termination_reason": info.get("termination_reason"),
                    "success": bool(info.get("success")),
                    "failure": bool(info.get("failure")),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty list of dictionaries as a CSV file."""

    if not rows:
        raise ValueError(f"Cannot write an empty result table to {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    dts: list[float],
    reference_evaluations: dict[str, int],
) -> None:
    """Record the complete study configuration needed for reproduction."""

    metadata = {
        "reference_method": "DOP853",
        "reference_rtol": args.rtol,
        "reference_atol": args.atol,
        "reference_function_evaluations": reference_evaluations,
        "fixed_step_integrators": ["euler", "rk4"],
        "timesteps_s": dts,
        "constant_action_body": {
            "ax_m_s2": float(DEFAULT_ACTION_BODY[0]),
            "ay_m_s2": float(DEFAULT_ACTION_BODY[1]),
            "alpha_rad_s2": float(DEFAULT_ACTION_BODY[2]),
            "alpha_deg_s2": float(np.rad2deg(DEFAULT_ACTION_BODY[2])),
        },
        "scenarios": [
            {
                "name": scenario.name,
                "phase": scenario.phase,
                "rotational_case": scenario.rotational_case,
                "duration_s": scenario.duration,
                "initial_state": list(scenario.initial_state),
            }
            for scenario in SCENARIOS
        ],
        "terminal_event_case": {
            "initial_state": [24.25, 0.0, -0.10, 0.0, 0.0, 0.0],
            "initial_axial_separation_m": 0.15,
            "initial_closing_speed_m_s": 0.10,
            "action": [0.0, 0.0, 0.0],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write(os.linesep)


def print_worst_case_summary(rows: list[dict[str, Any]]) -> None:
    """Print the thesis-facing aggregate convergence table."""

    print("\nWorst error across the four representative scenarios")
    print("-" * 111)
    print(
        f"{'method':>8} | {'dt [s]':>7} | {'position [m]':>15} | "
        f"{'velocity [m/s]':>16} | {'attitude [deg]':>16} | "
        f"{'position order':>14}"
    )
    print("-" * 111)
    for row in rows:
        order = row["position_order_to_next_finer"]
        order_text = "-" if order is None else f"{order:.6f}"
        print(
            f"{row['integrator']:>8} | {row['dt_s']:7.3f} | "
            f"{row['max_position_error_m']:15.6e} | "
            f"{row['max_velocity_error_m_s']:16.6e} | "
            f"{row['max_attitude_error_deg']:16.6e} | "
            f"{order_text:>14}"
        )
    print("-" * 111)
    print(
        "Note: a non-monotone RK4 order near zero indicates that its error has "
        "reached the reference/roundoff floor."
    )


def print_terminal_event_results(rows: list[dict[str, Any]]) -> None:
    """Print the timestep event-resolution table."""

    print("\nTerminal-event resolution test")
    print("-" * 91)
    print(
        f"{'method':>8} | {'dt [s]':>7} | {'time [s]':>8} | "
        f"{'final s [m]':>13} | {'outcome':>20}"
    )
    print("-" * 91)
    for row in rows:
        print(
            f"{row['integrator']:>8} | {row['dt_s']:7.3f} | "
            f"{row['event_time_s']:8.3f} | "
            f"{row['final_axial_separation_m']:13.6f} | "
            f"{str(row['termination_reason']):>20}"
        )
    print("-" * 91)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the fixed-step Euler and RK4 propagators against an "
            "adaptive DOP853 reference and test terminal-event resolution."
        )
    )
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=list(DEFAULT_DTS),
        help="Fixed timesteps in seconds, ordered from coarse to fine.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-12,
        help="Relative tolerance for the DOP853 reference.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-14,
        help="Absolute tolerance for the DOP853 reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for detailed_convergence.csv, "
            "worst_case_summary.csv, terminal_event_resolution.csv, and "
            "study_metadata.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dts = [float(dt) for dt in args.dts]
    if dts != sorted(dts, reverse=True):
        raise ValueError("--dts must be ordered from coarse to fine.")
    if len(set(dts)) != len(dts):
        raise ValueError("--dts must not contain duplicate values.")
    if not np.isfinite(args.rtol) or args.rtol <= 0.0:
        raise ValueError("--rtol must be finite and positive.")
    if not np.isfinite(args.atol) or args.atol <= 0.0:
        raise ValueError("--atol must be finite and positive.")

    detailed_rows, reference_evaluations = run_convergence_study(
        dts=dts,
        action_body=DEFAULT_ACTION_BODY,
        rtol=args.rtol,
        atol=args.atol,
    )
    summary_rows = build_worst_case_summary(detailed_rows, dts)
    event_rows = run_terminal_event_resolution_test(dts)

    print_worst_case_summary(summary_rows)
    print_terminal_event_results(event_rows)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        write_csv(output_dir / "detailed_convergence.csv", detailed_rows)
        write_csv(output_dir / "worst_case_summary.csv", summary_rows)
        write_csv(output_dir / "terminal_event_resolution.csv", event_rows)
        write_metadata(
            output_dir / "study_metadata.json",
            args=args,
            dts=dts,
            reference_evaluations=reference_evaluations,
        )
        print(f"\nSaved verification artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
