"""Run a held-out closing-phase LQR validation without changing the controller."""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.linalg import expm


LQR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
)
from tuning.closing.sweep_closing_lqr_weights import (
    run_rollouts,
    write_rollout_records_csv,
)


def exact_zoh(A, B, dt):
    """Return the exact zero-order-hold discretization of (A, B)."""
    n, m = B.shape
    block = np.zeros((n + m, n + m), dtype=float)
    block[:n, :n] = A
    block[:n, n:] = B
    discrete = expm(block * dt)
    return discrete[:n, :n], discrete[:n, n:]


def wilson_interval(successes, total, z=1.959963984540054):
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total**2))
    half_width /= denominator
    return center - half_width, center + half_width


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def count(records, key, value=True):
    return sum(record.get(key) == value for record in records)


def build_summary(args, env, Q, R, K, eigenvalues, spectral_radius, result, records):
    successes = count(records, "success")
    success_times = [record["elapsed_time"] for record in records if record["success"]]
    wilson_low, wilson_high = wilson_interval(successes, len(records))

    safety_records = [
        record
        for record in records
        if record["termination_reason"] == "safety_violation"
    ]

    def safety_count(category):
        return sum(category in record["violated_constraints"] for record in safety_records)

    tolerance_keys = {
        "axial_position": "fail_rx_tolerance",
        "lateral_position": "fail_ry_tolerance",
        "axial_velocity": "fail_vx_tolerance",
        "lateral_velocity": "fail_vy_tolerance",
        "attitude": "fail_psi_tolerance",
        "angular_rate": "fail_omega_tolerance",
    }

    terminal_error_keys = {
        "axial_position_m": "ex",
        "lateral_position_m": "ey",
        "axial_velocity_mps": "evx",
        "lateral_velocity_mps": "evy",
        "attitude_rad": "epsi",
        "angular_rate_radps": "eomega",
    }

    return {
        "configuration": {
            "phase": "closing",
            "mode": args.mode,
            "dt_seconds": float(args.dt),
            "max_time_seconds": float(args.max_time),
            "maximum_control_steps": int(math.ceil(args.max_time / args.dt)),
            "seed_start": int(args.seed_start),
            "seed_end": int(args.seed_start + args.rollouts - 1),
            "rollouts": int(args.rollouts),
            "Q_closing": Q,
            "R_closing": R,
            "K_closing": K,
            "lateral_gains": {
                "k_y": float(K[1, 1]),
                "k_vy": float(K[1, 3]),
                "k_vy_over_k_y_seconds": float(K[1, 3] / K[1, 1]),
            },
            "controller_schedule": {
                "v_fast_mps": float(LQRClosingController(K).v_fast),
                "d_fast_m": float(LQRClosingController(K).d_fast),
                "d_slow_m": float(LQRClosingController(K).d_slow),
            },
            "actuator_limits": {
                "ax_max_mps2": float(env.ax_max),
                "ay_max_mps2": float(env.ay_max),
                "alphaz_max_radps2": float(env.alphaz_max),
            },
            "p3_target_tolerance": np.asarray(env.p3_target_tolerance, dtype=float),
        },
        "sampled_data_stability": {
            "method": "exact_zero_order_hold",
            "eigenvalues": [
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                    "magnitude": float(abs(value)),
                }
                for value in eigenvalues
            ],
            "spectral_radius": float(spectral_radius),
            "stable": bool(spectral_radius < 1.0),
        },
        "outcomes": {
            "successes": successes,
            "success_rate": successes / float(len(records)),
            "success_rate_wilson_95": {
                "lower": wilson_low,
                "upper": wilson_high,
            },
            "timeouts": count(records, "termination_reason", "timeout"),
            "hard_safety_violations": len(safety_records),
            "hard_safety_by_type": {
                "velocity": safety_count("velocity"),
                "geometry": safety_count("geometry"),
                "alignment": safety_count("alignment"),
            },
        },
        "timing": {
            "mean_elapsed_time_seconds": float(result["elapsed_time"]),
            "mean_successful_completion_time_seconds": (
                float(np.mean(success_times)) if success_times else float("nan")
            ),
        },
        "mean_absolute_terminal_errors": {
            name: float(result[key]) for name, key in terminal_error_keys.items()
        },
        "p3_tolerance_failures": {
            name: {
                "count": count(records, key),
                "fraction": count(records, key) / float(len(records)),
            }
            for name, key in tolerance_keys.items()
        },
        "mean_actuator_saturation_fractions": {
            "any_channel": float(result["sat_fraction"]),
            "ax_body": float(result["sat_ax_fraction"]),
            "ay_body": float(result["sat_ay_fraction"]),
            "alphaz_body": float(result["sat_alphaz_fraction"]),
        },
        "mean_control_effort": {
            "translational_delta_v_mps": float(result["delta_v_proxy"]),
            "angular_rate_change_radps": float(result["angular_effort"]),
        },
        "observed_maxima": {
            "docking_port_speed_mps": float(result["peak_port_speed_max"]),
            "absolute_axial_closing_speed_mps": float(result["peak_s_dot_abs_max"]),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Held-out validation of the selected closing LQR controller."
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=300)
    parser.add_argument("--mode", choices=("testing",), default="testing")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dedicated result directory (derived from dt/seeds when omitted).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the selected matrices/stability and exit without rollouts.",
    )
    args = parser.parse_args()
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")
    if args.max_time <= 0.0:
        parser.error("--max-time must be positive")
    return args


def main():
    args = parse_args()
    seed_end = args.seed_start + args.rollouts - 1
    dt_label = f"{args.dt:g}".replace("-", "m").replace(".", "p")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            LQR_ROOT,
            "results",
            "lqr_closing",
            f"dt_{dt_label}",
            f"heldout_validation_seeds_{args.seed_start}_{seed_end}",
        )
    stem = f"closing_heldout_seeds_{args.seed_start}_{seed_end}_dt{dt_label}"
    csv_path = os.path.join(args.output_dir, f"{stem}_rollouts.csv")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.json")

    env = Docking3DoFEnv(
        phase="closing",
        dt=args.dt,
        max_time=args.max_time,
        seed=args.seed_start,
    )
    Q = build_Q_closing(dt=args.dt)
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    Ad, Bd = exact_zoh(A, B, args.dt)
    eigenvalues = np.linalg.eigvals(Ad - Bd @ K)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    print("Q_closing =")
    print(Q)
    print("K_closing =")
    print(K)
    print(f"k_y={K[1, 1]:.9f}, k_vy={K[1, 3]:.9f}")
    print(f"exact-ZOH spectral radius={spectral_radius:.12f}")
    print(f"sampled-data stable={spectral_radius < 1.0}")
    print(f"physical horizon={args.max_time:g} s, maximum steps={math.ceil(args.max_time / args.dt)}")
    print(f"seed range={args.seed_start}-{seed_end}")
    print(f"rollout CSV={csv_path}")
    print(f"summary JSON={summary_path}")

    if spectral_radius >= 1.0:
        raise RuntimeError("Refusing evaluation: sampled closed loop is unstable.")
    if args.check_only:
        print("Check-only mode: no rollouts were run and no output files were written.")
        return

    existing = [path for path in (csv_path, summary_path) if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing held-out results: " + ", ".join(existing)
        )

    controller = LQRClosingController(K)
    result, records = run_rollouts(
        controller,
        num_rollouts=args.rollouts,
        max_time=args.max_time,
        seed_start=args.seed_start,
        mode=args.mode,
        dt=args.dt,
    )
    summary = build_summary(
        args, env, Q, R, K, eigenvalues, spectral_radius, result, records
    )

    write_rollout_records_csv(records, csv_path)
    with open(summary_path, "x", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, allow_nan=True)
        handle.write("\n")

    print(f"Completed {args.rollouts} held-out rollouts.")
    print(f"Successes: {summary['outcomes']['successes']}/{args.rollouts}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
