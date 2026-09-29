"""Evaluate the repository LQR controllers through the full hard CBF-QP."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


CONTROLLER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
LQR_PATH = SRC_PATH / "controllers" / "LQR-3DoF"
for path in (CONTROLLER_DIR, SRC_PATH, LQR_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cbf_3dof.barriers import FinalApproachCBFConfig
from cbf_3dof.closing_barriers import ClosingCBFConfig
from cbf_3dof.controller import CBFQPController
from controllers.evaluation_initialization import (
    add_initial_range_arguments,
    initial_ranges_from_args,
    sample_initial_state,
)
from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices as build_closing_matrices,
    compute_lqr_gain as compute_closing_gain,
)
from lqr_3dof_final import (
    LQRFinalController,
    build_Q_final,
    build_R_final,
    build_port_error_mapping,
    build_linearized_3dof_matrices as build_final_matrices,
    compute_lqr_gain as compute_final_gain,
)


DEFAULT_RESULTS_DIR = CONTROLLER_DIR / "results" / "lqr"
FAMILY_RESIDUAL_NAMES = {
    "cone": ("cone_upper", "cone_lower", "cone_axial_upper"),
    "total_relative_velocity": ("port_speed",),
    "axial_velocity": ("axial_port_speed",),
    "attitude": ("attitude_upper", "attitude_lower"),
    "angular_rate": ("angular_rate_upper", "angular_rate_lower"),
}


def build_lqr(env: Docking3DoFEnv):
    """Build the unchanged LQR selected by the existing phase evaluations."""
    if env.phase == "closing":
        gain, _ = compute_closing_gain(
            *build_closing_matrices(env),
            build_Q_closing(dt=env.dt),
            build_R_closing(env),
        )
        return LQRClosingController(gain)

    q_port = build_Q_final()
    mapping = build_port_error_mapping(env)
    gain, _ = compute_final_gain(
        *build_final_matrices(env),
        mapping.T @ q_port @ mapping,
        build_R_final(env),
    )
    return LQRFinalController(gain, mapping)


def full_cbf_config(phase: str):
    if phase == "closing":
        return ClosingCBFConfig(
            cone_k1=0.05,
            cone_k2=0.05,
            cone_interior_margin=1.0e-4,
            port_speed_max=0.25,
            axial_port_speed_max=0.15,
            port_speed_alpha=0.25,
            axial_port_speed_alpha=0.5,
            attitude_max=np.deg2rad(3.0),
            angular_rate_max=np.deg2rad(0.15),
            attitude_k1=0.5,
            attitude_k2=0.5,
            angular_rate_alpha=1.0,
            allow_slack=False,
            enable_cone=True,
            enable_port_speed=True,
            enable_axial_port_speed=True,
            enable_alignment=True,
        )
    return FinalApproachCBFConfig(
        kappa=0.1,
        port_speed_max=0.15,
        cone_k1=0.1,
        cone_k2=0.1,
        speed_alpha=0.5,
        attitude_max=np.deg2rad(3.0),
        angular_rate_max=np.deg2rad(0.15),
        attitude_k1=0.3,
        attitude_k2=0.3,
        angular_rate_alpha=0.5,
        allow_slack=False,
        enable_cone=True,
        enable_speed=True,
        enable_alignment=True,
    )


def run_rollout(
    phase: str,
    seed: int,
    mode: str,
    max_time: float,
    initial_ranges: dict[str, tuple[float, float]] | None = None,
    use_cbf: bool = True,
):
    dt = 0.1 if phase == "closing" else 1.0
    env = Docking3DoFEnv(
        phase=phase,
        dt=dt,
        max_time=max_time,
        integrator="rk4",
        seed=seed,
    )
    initial_state = (
        None if initial_ranges is None else sample_initial_state(seed, initial_ranges)
    )
    state = env.reset(initial_state=initial_state, mode=mode)
    p3_target = None if env.p3_target is None else env.p3_target.copy()
    nominal_controller = build_lqr(env)
    controller = (
        CBFQPController(nominal_controller, full_cbf_config(phase))
        if use_cbf
        else nominal_controller
    )

    interventions = 0
    qp_failures = 0
    family_intervention_steps = Counter()
    hard = cone = total_speed = axial_speed = attitude = angular_rate = False
    closest_p3 = np.inf
    steps = 0
    info = {}
    done = False
    max_steps = int(np.ceil(max_time / dt))

    while not done and steps < max_steps:
        filtered_action = controller.compute_action(env, state)
        result = controller.last_result if use_cbf else None
        # CBFQPController computes LQR nominal -> CBF filtered; only the latter
        # is passed to the base environment here.
        state, done, info = env.step(filtered_action)
        steps += 1
        interventions += int(result.intervened) if result is not None else 0
        qp_failures += int(result.qp_failed) if result is not None else 0
        if result is not None and result.intervened:
            # A family is attributed independently when one of its filtered
            # hard rows is binding to the intervention tolerance. Multiple
            # families may therefore be counted at the same control step.
            for family, names in FAMILY_RESIDUAL_NAMES.items():
                residuals = [
                    result.cbf_residuals[name]
                    for name in names
                    if name in result.cbf_residuals
                ]
                if residuals and min(residuals) <= controller.config.intervention_eps:
                    family_intervention_steps[family] += 1

        q = env.get_derived_quantities(state)
        monitor = env.monitor_safety(state)
        geometry = monitor["geometry"]
        hard |= bool(monitor["unsafe"])
        cone |= not bool(geometry["in_C_dock"])
        if phase == "closing":
            total_speed |= q["port_speed"] > env.vp_unsafe_close
            axial_speed |= abs(q["s_dot"]) > env.sdot_unsafe_close
            if p3_target is not None:
                closest_p3 = min(closest_p3, float(np.linalg.norm(state[:2] - p3_target[:2])))
        else:
            total_speed |= q["port_speed"] > env.vp_unsafe_final
            axial_speed |= abs(q["s_dot"]) > env.sdot_unsafe_final
        attitude |= abs(q["psi_e"]) > env.psi_unsafe
        angular_rate |= abs(q["omega"]) > env.omega_unsafe

    q = env.get_derived_quantities(state)
    row = {
        "phase": phase,
        "eval_seed": seed,
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "time_limit": bool(info.get("time_limit", False)) or not done,
        "termination_reason": info.get("termination_reason", "evaluation_step_limit"),
        "steps": steps,
        "phase_duration_seconds": float(env.time),
        "hard_safety_violation": hard,
        "docking_cone_violation": cone,
        "total_relative_velocity_violation": total_speed,
        "axial_velocity_violation": axial_speed,
        "alignment_attitude_violation": attitude,
        "angular_rate_violation": angular_rate,
        "intervention_count": interventions,
        "intervention_fraction": interventions / steps,
        "qp_failure_count": qp_failures,
    }
    for family in FAMILY_RESIDUAL_NAMES:
        count = int(family_intervention_steps[family])
        row[f"{family}_binding_intervention_steps"] = count
        row[f"{family}_active_during_intervention"] = count > 0
    if phase == "closing":
        error = state - p3_target
        error[4] = env.wrap_to_pi(error[4])
        row.update(
            closest_p3_position_error=float(closest_p3),
            terminal_p3_position_error=float(np.linalg.norm(error[:2])),
            terminal_axial_error=float(abs(error[0])),
            terminal_lateral_error=float(abs(error[1])),
            terminal_velocity_error=float(np.linalg.norm(error[2:4])),
            terminal_attitude_error=float(abs(error[4])),
            terminal_angular_rate_error=float(abs(error[5])),
        )
    else:
        row.update(
            terminal_port_distance=float(q["port_distance"]),
            terminal_port_speed=float(q["port_speed"]),
            terminal_axial_position=float(q["s"]),
            terminal_lateral_position=float(q["rho"]),
            terminal_axial_velocity=float(q["s_dot"]),
            terminal_lateral_velocity=float(q["v_p"][1]),
            terminal_attitude_error=float(abs(q["psi_e"])),
            terminal_angular_rate_error=float(abs(q["omega"])),
        )
    return row


def summarize(rows):
    summary = {
        "phase": rows[0]["phase"],
        "episodes": len(rows),
        "success_count": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "hard_safety_violation_count": int(sum(row["hard_safety_violation"] for row in rows)),
        "docking_cone_violation_count": int(sum(row["docking_cone_violation"] for row in rows)),
        "total_relative_velocity_violation_count": int(sum(row["total_relative_velocity_violation"] for row in rows)),
        "axial_velocity_violation_count": int(sum(row["axial_velocity_violation"] for row in rows)),
        "alignment_attitude_violation_count": int(sum(row["alignment_attitude_violation"] for row in rows)),
        "angular_rate_violation_count": int(sum(row["angular_rate_violation"] for row in rows)),
        "mean_intervention_fraction": float(np.mean([row["intervention_fraction"] for row in rows])),
        "intervention_count": int(sum(row["intervention_count"] for row in rows)),
        "qp_failure_count": int(sum(row["qp_failure_count"] for row in rows)),
        "mean_phase_duration_seconds": float(np.mean([row["phase_duration_seconds"] for row in rows])),
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
    }
    for family in FAMILY_RESIDUAL_NAMES:
        summary[f"{family}_active_intervention_rollout_count"] = int(
            sum(row[f"{family}_active_during_intervention"] for row in rows)
        )
        summary[f"{family}_binding_intervention_steps"] = int(
            sum(row[f"{family}_binding_intervention_steps"] for row in rows)
        )
    excluded = set(summary) | {"eval_seed", "failure", "time_limit", "steps", "termination_reason"}
    for key in rows[0]:
        if key.startswith("terminal_") or key == "closest_p3_position_error":
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    return summary


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("closing", "final"), required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--mode", choices=("nominal", "testing", "training"), default="testing")
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-cbf",
        action="store_true",
        help="Evaluate the unchanged nominal LQR without the CBF-QP filter.",
    )
    parser.add_argument(
        "--native-initialization",
        action="store_true",
        help=(
            "Use the environment's phase-specific seeded reset distribution "
            "instead of the explicit evaluation-range arguments."
        ),
    )
    add_initial_range_arguments(parser)
    return parser.parse_args()


def resolve_initial_ranges(args):
    """Return None when the environment should perform its native reset."""
    if args.native_initialization:
        return None
    return initial_ranges_from_args(args)


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    default_max_time = 12_000.0 if args.phase == "closing" else 5_000.0
    max_time = default_max_time if args.max_time is None else args.max_time
    if max_time <= 0.0:
        raise ValueError("--max-time must be positive")
    initial_ranges = resolve_initial_ranges(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        row = run_rollout(
            args.phase,
            seed,
            args.mode,
            max_time,
            initial_ranges=initial_ranges,
            use_cbf=not args.no_cbf,
        )
        rows.append(row)
        print(f"{args.phase} seed={seed} success={row['success']} reason={row['termination_reason']}")

    summary = summarize(rows)
    write_csv(args.output_dir / "per_episode_metrics.csv", rows)
    with (args.output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (args.output_dir / "run_configuration.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "phase": args.phase,
                "controller": type(build_lqr(Docking3DoFEnv(max_time=12_000.0 if args.phase == "closing" else 5_000.0, phase=args.phase, dt=0.1 if args.phase == 'closing' else 1.0))).__name__,
                "dt_seconds": 0.1 if args.phase == "closing" else 1.0,
                "max_time_seconds": max_time,
                "integrator": "rk4",
                "mode": args.mode,
                "seeds": list(range(args.seed_start, args.seed_start + args.num_seeds)),
                "initial_ranges": initial_ranges,
                "cbf": None if args.no_cbf else asdict(full_cbf_config(args.phase)),
                "control_path": (
                    "LQR nominal -> base environment"
                    if args.no_cbf
                    else "LQR nominal -> full hard CBF-QP -> base environment"
                ),
                "sac_or_vecnormalize_used": False,
            },
            stream,
            indent=2,
        )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
