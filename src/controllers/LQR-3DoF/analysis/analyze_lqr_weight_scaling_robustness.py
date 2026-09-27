"""Check LQR stability under moderate Q/R weight scaling.

For every phase and every requested pair of Q and R scale factors, this
script repeats the continuous-time CARE synthesis.  It then checks:

* the eigenvalues of the redesigned continuous closed loop, ``A - B K``;
* the eigenvalues of the corresponding sampled-data closed loop,
  ``A_d - B_d K``, at the control interval used by that phase.

The sampled plant is formed with an exact zero-order hold.  The gain remains
the continuous-time CARE gain because that is how the implemented phase
controllers are synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm


SCRIPT_DIR = Path(__file__).resolve().parent
LQR_ROOT = SCRIPT_DIR.parent
SRC_DIR = LQR_ROOT.parents[1]
for import_path in (LQR_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices as build_closing_matrices,
    compute_lqr_gain as compute_closing_gain,
)
from lqr_3dof_final import (
    build_Q_final,
    build_R_final,
    build_linearized_3dof_matrices as build_final_matrices,
    build_port_error_mapping,
    compute_lqr_gain as compute_final_gain,
)


DEFAULT_SCALES = (0.8, 1.0, 1.2)
DEFAULT_CLOSING_DT = 0.1
DEFAULT_FINAL_DT = 1.0
DEFAULT_OUTPUT = (
    LQR_ROOT
    / "results"
    / "lqr_mission"
    / "lqr_weight_scaling_stability_analysis"
    / "lqr_weight_scaling_robustness"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat phase-specific CARE synthesis after independent Q/R "
            "scaling and check continuous and sampled-data stability."
        )
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        metavar="FACTOR",
        help="Positive Q and R scale factors (default: 0.8 1.0 1.2).",
    )
    parser.add_argument(
        "--closing-dt",
        type=float,
        default=DEFAULT_CLOSING_DT,
        help="Implemented closing control interval in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--final-dt",
        type=float,
        default=DEFAULT_FINAL_DT,
        help="Implemented final-approach control interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; CSV and JSON are written.",
    )
    args = parser.parse_args()

    if any(scale <= 0.0 for scale in args.scales):
        parser.error("all --scales values must be positive")
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales must not contain duplicates")
    if args.closing_dt <= 0.0 or args.final_dt <= 0.0:
        parser.error("phase control intervals must be positive")
    return args


def exact_zoh(
    A: np.ndarray, B: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact zero-order-hold discretization of ``(A, B)``."""
    n, m = B.shape
    block = np.zeros((n + m, n + m), dtype=float)
    block[:n, :n] = A
    block[:n, n:] = B
    discrete = expm(block * dt)
    return discrete[:n, :n], discrete[:n, n:]


def encode_eigenvalues(values: np.ndarray) -> list[dict[str, float]]:
    """Encode complex eigenvalues without losing their imaginary parts."""
    ordered = sorted(
        np.asarray(values, dtype=complex),
        key=lambda value: (float(value.real), float(value.imag)),
    )
    return [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in ordered
    ]


def build_phase_problem(
    phase: str, dt: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
]:
    """Build the selected continuous LQR problem for one mission phase."""
    env = Docking3DoFEnv(max_time=12_000.0 if phase == "closing" else 5_000.0, phase=phase, dt=dt, seed=0)
    if phase == "closing":
        A, B = build_closing_matrices(env)
        Q = build_Q_closing(dt=dt)
        R = build_R_closing(env)
        gain_solver = compute_closing_gain
    elif phase == "final":
        A, B = build_final_matrices(env)
        H = build_port_error_mapping(env)
        Q = H.T @ build_Q_final() @ H
        R = build_R_final(env)
        gain_solver = compute_final_gain
    else:
        raise ValueError(f"Unsupported phase: {phase!r}")
    return A, B, Q, R, gain_solver


def analyze_phase(
    phase: str, dt: float, scales: list[float] | tuple[float, ...]
) -> list[dict[str, object]]:
    """Analyze every Cartesian product of Q and R scales for one phase."""
    A, B, selected_Q, selected_R, gain_solver = build_phase_problem(phase, dt)
    Ad, Bd = exact_zoh(A, B, dt)
    records: list[dict[str, object]] = []

    for q_scale in scales:
        for r_scale in scales:
            gain, _ = gain_solver(
                A,
                B,
                float(q_scale) * selected_Q,
                float(r_scale) * selected_R,
            )
            continuous_eigenvalues = np.linalg.eigvals(A - B @ gain)
            sampled_eigenvalues = np.linalg.eigvals(Ad - Bd @ gain)
            max_continuous_real_part = float(
                np.max(np.real(continuous_eigenvalues))
            )
            sampled_spectral_radius = float(
                np.max(np.abs(sampled_eigenvalues))
            )
            continuous_stable = max_continuous_real_part < 0.0
            sampled_stable = sampled_spectral_radius < 1.0
            records.append(
                {
                    "phase": phase,
                    "control_interval_seconds": float(dt),
                    "q_scale": float(q_scale),
                    "r_scale": float(r_scale),
                    "nominal": bool(
                        np.isclose(q_scale, 1.0) and np.isclose(r_scale, 1.0)
                    ),
                    "max_continuous_real_part": max_continuous_real_part,
                    "continuous_stable": continuous_stable,
                    "sampled_spectral_radius": sampled_spectral_radius,
                    "sampled_data_stable": sampled_stable,
                    "both_stable": continuous_stable and sampled_stable,
                    "continuous_closed_loop_eigenvalues": encode_eigenvalues(
                        continuous_eigenvalues
                    ),
                    "sampled_closed_loop_eigenvalues": encode_eigenvalues(
                        sampled_eigenvalues
                    ),
                }
            )
    return records


def csv_record(record: dict[str, object]) -> dict[str, object]:
    """Select scalar result fields for the compact CSV output."""
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "continuous_closed_loop_eigenvalues",
            "sampled_closed_loop_eigenvalues",
        }
    }


def write_outputs(
    output: Path,
    records: list[dict[str, object]],
    scales: list[float] | tuple[float, ...],
    closing_dt: float,
    final_dt: float,
) -> tuple[Path, Path]:
    """Write compact CSV results and a full eigenvalue audit in JSON."""
    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_records = [csv_record(record) for record in records]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_records[0]))
        writer.writeheader()
        writer.writerows(scalar_records)

    summary = {
        "configuration": {
            "method": "continuous-time CARE redesign with exact-ZOH sampled check",
            "q_scale_factors": [float(value) for value in scales],
            "r_scale_factors": [float(value) for value in scales],
            "closing_control_interval_seconds": float(closing_dt),
            "final_control_interval_seconds": float(final_dt),
            "continuous_stability_criterion": "max(real(eig(A-BK))) < 0",
            "sampled_stability_criterion": "spectral_radius(Ad-BdK) < 1",
        },
        "summary": {
            "num_cases": len(records),
            "num_continuous_stable": sum(
                bool(record["continuous_stable"]) for record in records
            ),
            "num_sampled_data_stable": sum(
                bool(record["sampled_data_stable"]) for record in records
            ),
            "num_both_stable": sum(bool(record["both_stable"]) for record in records),
            "all_cases_both_stable": all(
                bool(record["both_stable"]) for record in records
            ),
        },
        "records": records,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return csv_path, json_path


def print_summary(records: list[dict[str, object]]) -> None:
    print("LQR WEIGHT-SCALING ROBUSTNESS")
    print("=" * 79)
    print(
        "phase    dt [s]  Q scale  R scale  max Re(lambda_c)  "
        "rho(lambda_d)  stable"
    )
    for record in records:
        print(
            f"{record['phase']:<8} "
            f"{record['control_interval_seconds']:>6.1f}  "
            f"{record['q_scale']:>7.2f}  "
            f"{record['r_scale']:>7.2f}  "
            f"{record['max_continuous_real_part']:>16.9e}  "
            f"{record['sampled_spectral_radius']:>13.9f}  "
            f"{str(record['both_stable']):>6}"
        )

    print("-" * 79)
    for phase in ("closing", "final"):
        phase_records = [record for record in records if record["phase"] == phase]
        worst_continuous = max(
            phase_records, key=lambda record: record["max_continuous_real_part"]
        )
        worst_sampled = max(
            phase_records, key=lambda record: record["sampled_spectral_radius"]
        )
        print(
            f"{phase}: {sum(bool(r['both_stable']) for r in phase_records)}/"
            f"{len(phase_records)} cases stable; "
            f"worst max Re(lambda_c)={worst_continuous['max_continuous_real_part']:.9e} "
            f"at Qx{worst_continuous['q_scale']:.2f}/Rx{worst_continuous['r_scale']:.2f}; "
            f"worst rho(lambda_d)={worst_sampled['sampled_spectral_radius']:.9f} "
            f"at Qx{worst_sampled['q_scale']:.2f}/Rx{worst_sampled['r_scale']:.2f}"
        )


def main() -> None:
    args = parse_args()
    records = analyze_phase("closing", args.closing_dt, args.scales)
    records.extend(analyze_phase("final", args.final_dt, args.scales))
    print_summary(records)
    csv_path, json_path = write_outputs(
        args.output,
        records,
        args.scales,
        args.closing_dt,
        args.final_dt,
    )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    if not all(bool(record["both_stable"]) for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
