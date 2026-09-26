"""Targeted closing-phase Q-only sweep for lateral position/velocity weights.

The sweep keeps the selected controller architecture, R matrix, reference
construction, actuator limits, dt, horizon, environment, and evaluation seeds
unchanged.  Each candidate is saved to a separate directory and can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.linalg import expm


LQR_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LQR_ROOT.parents[1]
for path in (str(LQR_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)
from tuning_lqr_closing.lqr_sweep_closing import (
    run_rollouts,
    write_rollout_records_csv,
)


DT = 0.1
MAX_TIME = 12_000.0
NUM_ROLLOUTS = 50
SEED_START = 0
MODE = "testing"
PORT_SPEED_SAFE = 0.25
PORT_SPEED_UNSAFE = 0.35
S_DOT_SAFE = 0.15
S_DOT_UNSAFE = 0.25
MY_VALUES = (1.0, 1.5, 2.0, 3.0)
MVY_VALUES = (1.0, 0.75, 0.5)
CURRENT_SELECTED_Q_SCALE = 6.5
CURRENT_SELECTED_Q_MULTIPLIER_Y = 6.5
CURRENT_SELECTED_Q_MULTIPLIER_VY = 6.5


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        if abs(value.imag) < 1e-14:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def candidate_tag(my, mvy):
    return f"my_{my:g}_mvy_{mvy:g}".replace(".", "p")


def exact_zoh(A, B, dt):
    n, m = B.shape
    block = np.zeros((n + m, n + m), dtype=float)
    block[:n, :n] = A
    block[:n, n:] = B
    discrete = expm(block * dt)
    return discrete[:n, :n], discrete[:n, n:]


def build_candidate(my, mvy):
    env = Docking3DoFEnv(
        phase="closing", dt=DT, max_time=MAX_TIME, seed=SEED_START
    )
    A, B = build_linearized_3dof_matrices(env)
    # Sweep factors are defined relative to the original dt=1.0-selected Q.
    Q_original = build_Q_closing(dt=1.0)
    Q = Q_original.copy()
    Q[1, 1] *= my
    Q[3, 3] *= mvy
    R = build_R_closing(env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    Ad, Bd = exact_zoh(A, B, DT)
    sampled_eigenvalues = np.linalg.eigvals(Ad - Bd @ K)
    spectral_radius = float(np.max(np.abs(sampled_eigenvalues)))
    return {
        "my": float(my),
        "mvy": float(mvy),
        "tag": candidate_tag(my, mvy),
        "Q": Q,
        "Q_original": Q_original,
        "R": R,
        "K": K,
        "A": A,
        "B": B,
        "Ad": Ad,
        "Bd": Bd,
        "sampled_eigenvalues": sampled_eigenvalues,
        "spectral_radius": spectral_radius,
        "sampled_stable": bool(spectral_radius < 1.0),
        "ky": float(K[1, 1]),
        "kvy": float(K[1, 3]),
        "gain_ratio_seconds": float(K[1, 3] / K[1, 1]),
        "effective_selected_q_multiplier_y": float(
            CURRENT_SELECTED_Q_MULTIPLIER_Y * my
        ),
        "effective_selected_q_multiplier_vy": float(
            CURRENT_SELECTED_Q_MULTIPLIER_VY * mvy
        ),
    }


def candidate_precheck_record(candidate):
    return {
        "my": candidate["my"],
        "mvy": candidate["mvy"],
        "tag": candidate["tag"],
        "effective_selected_q_multiplier_y": candidate[
            "effective_selected_q_multiplier_y"
        ],
        "effective_selected_q_multiplier_vy": candidate[
            "effective_selected_q_multiplier_vy"
        ],
        "Q_diagonal": np.diag(candidate["Q"]),
        "R_diagonal_unchanged": np.diag(candidate["R"]),
        "K": candidate["K"],
        "ky": candidate["ky"],
        "kvy": candidate["kvy"],
        "kvy_over_ky_seconds": candidate["gain_ratio_seconds"],
        "sampled_eigenvalues_dt_0p1": candidate["sampled_eigenvalues"],
        "spectral_radius_dt_0p1": candidate["spectral_radius"],
        "sampled_stable_dt_0p1": candidate["sampled_stable"],
    }


def evaluation_summary(candidate, result, rollout_records):
    count = len(rollout_records)
    failure_counts = {
        "axial_position": sum(item["fail_rx_tolerance"] for item in rollout_records),
        "lateral_position": sum(item["fail_ry_tolerance"] for item in rollout_records),
        "axial_velocity": sum(item["fail_vx_tolerance"] for item in rollout_records),
        "lateral_velocity": sum(item["fail_vy_tolerance"] for item in rollout_records),
        "attitude": sum(item["fail_psi_tolerance"] for item in rollout_records),
        "angular_rate": sum(item["fail_omega_tolerance"] for item in rollout_records),
    }
    peak_port_speeds = np.asarray(
        [item["peak_port_speed"] for item in rollout_records], dtype=float
    )
    peak_s_dot_abs = np.asarray(
        [item["peak_s_dot_abs"] for item in rollout_records], dtype=float
    )
    return {
        **candidate_precheck_record(candidate),
        "configuration": {
            "dt_seconds": DT,
            "max_time_seconds": MAX_TIME,
            "seeds": f"{SEED_START}-{SEED_START + NUM_ROLLOUTS - 1}",
            "num_rollouts": NUM_ROLLOUTS,
            "mode": MODE,
            "Q_only_lateral_retuning": True,
            "R_unchanged": True,
            "controller_architecture_unchanged": True,
            "reference_construction_unchanged": True,
        },
        "outcomes": {
            "successes": int(round(result["success_rate"] * count)),
            "success_rate": result["success_rate"],
            "timeouts": int(round(result["timeout_rate"] * count)),
            "timeout_rate": result["timeout_rate"],
            "hard_safety_violations": int(round(result["safety_rate"] * count)),
            "hard_safety_violation_types": {
                "geometry": int(round(result["geom_rate"] * count)),
                "velocity": int(round(result["vel_rate"] * count)),
                "alignment": int(round(result["align_rate"] * count)),
            },
        },
        "mean_elapsed_physical_time_seconds": result["elapsed_time"],
        "mean_absolute_terminal_errors": {
            "axial_position_m": result["ex"],
            "lateral_position_m": result["ey"],
            "axial_velocity_mps": result["evx"],
            "lateral_velocity_mps": result["evy"],
            "attitude_rad": result["epsi"],
            "angular_rate_radps": result["eomega"],
        },
        "p3_tolerance_failures": {
            name: {"count": int(value), "fraction": value / count}
            for name, value in failure_counts.items()
        },
        "mean_actuator_saturation_fractions": {
            "any_channel": result["sat_fraction"],
            "ax_bc": result["sat_ax_fraction"],
            "ay_bc": result["sat_ay_fraction"],
            "alphaz_bc": result["sat_alphaz_fraction"],
        },
        "mean_control_effort": {
            "translational_sum_norm_a_dt_mps": result["delta_v_proxy"],
            "angular_sum_abs_alpha_dt_radps": result["angular_effort"],
        },
        "trajectory_peak_velocity": {
            "maximum_port_speed_mps": float(np.max(peak_port_speeds)),
            "maximum_abs_axial_closing_speed_mps": float(
                np.max(peak_s_dot_abs)
            ),
            "port_speed_safe_threshold_mps": PORT_SPEED_SAFE,
            "port_speed_unsafe_threshold_mps": PORT_SPEED_UNSAFE,
            "abs_axial_speed_safe_threshold_mps": S_DOT_SAFE,
            "abs_axial_speed_unsafe_threshold_mps": S_DOT_UNSAFE,
            "rollouts_exceeding_port_speed_safe_threshold": int(
                np.count_nonzero(peak_port_speeds > PORT_SPEED_SAFE)
            ),
            "rollouts_exceeding_port_speed_unsafe_threshold": int(
                np.count_nonzero(peak_port_speeds > PORT_SPEED_UNSAFE)
            ),
            "rollouts_exceeding_abs_axial_speed_safe_threshold": int(
                np.count_nonzero(peak_s_dot_abs > S_DOT_SAFE)
            ),
            "rollouts_exceeding_abs_axial_speed_unsafe_threshold": int(
                np.count_nonzero(peak_s_dot_abs > S_DOT_UNSAFE)
            ),
        },
        "raw_result": result,
    }


def evaluate_one(args):
    my, mvy, output_root_text, resume = args
    output_root = Path(output_root_text)
    candidate = build_candidate(my, mvy)
    candidate_dir = output_root / candidate["tag"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = candidate_dir / "summary.json"
    csv_path = candidate_dir / "rollouts_seeds_0_49.csv"

    if resume and summary_path.exists() and csv_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        print(f"resume: {candidate['tag']} already complete", flush=True)
        return summary

    if not candidate["sampled_stable"]:
        summary = {
            **candidate_precheck_record(candidate),
            "discarded_before_rollouts": True,
            "discard_reason": "sampled-data unstable at dt=0.1 s",
        }
    else:
        controller = LQRClosingController(candidate["K"])
        result, rollout_records = run_rollouts(
            controller=controller,
            num_rollouts=NUM_ROLLOUTS,
            max_time=MAX_TIME,
            seed_start=SEED_START,
            mode=MODE,
            dt=DT,
        )
        write_rollout_records_csv(rollout_records, str(csv_path))
        summary = evaluation_summary(candidate, result, rollout_records)
        summary["discarded_before_rollouts"] = False

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    print(
        f"complete: {candidate['tag']} rho={candidate['spectral_radius']:.9f}"
        + (
            f" success={summary['outcomes']['successes']}/{NUM_ROLLOUTS}"
            if "outcomes" in summary
            else " discarded"
        ),
        flush=True,
    )
    return json_safe(summary)


def rank_summaries(summaries):
    evaluated = [item for item in summaries if "outcomes" in item]
    return sorted(
        evaluated,
        key=lambda item: (
            item["outcomes"]["hard_safety_violations"] > 0,
            -item["outcomes"]["successes"],
            item["outcomes"]["timeouts"],
            item["p3_tolerance_failures"]["lateral_position"]["count"],
            item["mean_actuator_saturation_fractions"]["any_channel"],
            item["mean_control_effort"]["translational_sum_norm_a_dt_mps"],
            item["outcomes"]["hard_safety_violations"],
        ),
    )


def write_aggregate_csv(path, ranked):
    rows = []
    for rank, item in enumerate(ranked, start=1):
        outcome = item["outcomes"]
        failures = item["p3_tolerance_failures"]
        errors = item["mean_absolute_terminal_errors"]
        saturation = item["mean_actuator_saturation_fractions"]
        effort = item["mean_control_effort"]
        peaks = item["trajectory_peak_velocity"]
        rows.append(
            {
                "safety_first_rank": rank,
                "tag": item["tag"],
                "my": item["my"],
                "mvy": item["mvy"],
                "ky": item["ky"],
                "kvy": item["kvy"],
                "kvy_over_ky_seconds": item["kvy_over_ky_seconds"],
                "spectral_radius_dt_0p1": item["spectral_radius_dt_0p1"],
                "successes": outcome["successes"],
                "success_rate": outcome["success_rate"],
                "timeouts": outcome["timeouts"],
                "timeout_rate": outcome["timeout_rate"],
                "hard_safety_violations": outcome["hard_safety_violations"],
                "safety_geometry": outcome["hard_safety_violation_types"]["geometry"],
                "safety_velocity": outcome["hard_safety_violation_types"]["velocity"],
                "safety_alignment": outcome["hard_safety_violation_types"]["alignment"],
                "mean_elapsed_seconds": item["mean_elapsed_physical_time_seconds"],
                "mean_abs_terminal_ex_m": errors["axial_position_m"],
                "mean_abs_terminal_ey_m": errors["lateral_position_m"],
                "mean_abs_terminal_evx_mps": errors["axial_velocity_mps"],
                "mean_abs_terminal_evy_mps": errors["lateral_velocity_mps"],
                "mean_abs_terminal_epsi_rad": errors["attitude_rad"],
                "mean_abs_terminal_eomega_radps": errors["angular_rate_radps"],
                "fail_rx_count": failures["axial_position"]["count"],
                "fail_ry_count": failures["lateral_position"]["count"],
                "fail_vx_count": failures["axial_velocity"]["count"],
                "fail_vy_count": failures["lateral_velocity"]["count"],
                "fail_psi_count": failures["attitude"]["count"],
                "fail_omega_count": failures["angular_rate"]["count"],
                "sat_any_fraction": saturation["any_channel"],
                "sat_ax_fraction": saturation["ax_bc"],
                "sat_ay_fraction": saturation["ay_bc"],
                "sat_alphaz_fraction": saturation["alphaz_bc"],
                "translational_effort_mps": effort[
                    "translational_sum_norm_a_dt_mps"
                ],
                "angular_effort_radps": effort["angular_sum_abs_alpha_dt_radps"],
                "peak_port_speed_mps": peaks["maximum_port_speed_mps"],
                "peak_abs_axial_closing_speed_mps": peaks[
                    "maximum_abs_axial_closing_speed_mps"
                ],
                "port_speed_safe_exceedance_rollouts": peaks[
                    "rollouts_exceeding_port_speed_safe_threshold"
                ],
                "port_speed_unsafe_exceedance_rollouts": peaks[
                    "rollouts_exceeding_port_speed_unsafe_threshold"
                ],
                "axial_speed_safe_exceedance_rollouts": peaks[
                    "rollouts_exceeding_abs_axial_speed_safe_threshold"
                ],
                "axial_speed_unsafe_exceedance_rollouts": peaks[
                    "rollouts_exceeding_abs_axial_speed_unsafe_threshold"
                ],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--my-values",
        default=",".join(str(value) for value in MY_VALUES),
        help="Comma-separated relative lateral-position Q multipliers.",
    )
    parser.add_argument(
        "--mvy-values",
        default=",".join(str(value) for value in MVY_VALUES),
        help="Comma-separated relative lateral-velocity Q multipliers.",
    )
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=LQR_ROOT
        / "results"
        / "lqr_closing"
        / "tuning_diagnostics_and_evaluation_dt0p1"
        / "targeted_lateral_q_sweep",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    my_values = tuple(float(value) for value in args.my_values.split(","))
    mvy_values = tuple(float(value) for value in args.mvy_values.split(","))
    candidates = [
        build_candidate(my, mvy) for my in my_values for mvy in mvy_values
    ]
    precheck = {
        "current_selected_Q_definition": {
            "global_q_scale": CURRENT_SELECTED_Q_SCALE,
            "lateral_position_internal_multiplier": CURRENT_SELECTED_Q_MULTIPLIER_Y,
            "lateral_velocity_internal_multiplier": CURRENT_SELECTED_Q_MULTIPLIER_VY,
            "original_Q_diagonal": np.diag(candidates[0]["Q_original"]),
            "original_Q_y_diagonal": candidates[0]["Q_original"][1, 1],
            "original_Q_vy_diagonal": candidates[0]["Q_original"][3, 3],
        },
        "unchanged_configuration": {
            "R_diagonal": np.diag(candidates[0]["R"]),
            "dt_seconds": DT,
            "max_time_seconds": MAX_TIME,
            "seeds": "0-49",
            "mode": MODE,
        },
        "candidates": [candidate_precheck_record(item) for item in candidates],
    }
    with (args.output_root / "gain_and_stability_precheck.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(precheck), handle, indent=2)

    for candidate in candidates:
        print(
            f"{candidate['tag']}: ky={candidate['ky']:.9g} "
            f"kvy={candidate['kvy']:.9g} "
            f"ratio={candidate['gain_ratio_seconds']:.6f}s "
            f"rho={candidate['spectral_radius']:.9f} "
            f"stable={candidate['sampled_stable']}",
            flush=True,
        )

    if args.precheck_only:
        return

    jobs = [
        (item["my"], item["mvy"], str(args.output_root), not args.no_resume)
        for item in candidates
        if item["sampled_stable"]
    ]
    summaries = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(evaluate_one, job) for job in jobs]
        for future in as_completed(futures):
            summaries.append(future.result())

    # Preserve discarded candidates in the aggregate as well.
    for item in candidates:
        if not item["sampled_stable"]:
            summaries.append(
                {
                    **candidate_precheck_record(item),
                    "discarded_before_rollouts": True,
                    "discard_reason": "sampled-data unstable at dt=0.1 s",
                }
            )

    ranked = rank_summaries(summaries)
    aggregate = {
        "precheck_file": "gain_and_stability_precheck.json",
        "candidate_count": len(candidates),
        "stable_candidate_count": sum(item["sampled_stable"] for item in candidates),
        "evaluated_candidate_count": len(ranked),
        "ranking_rule": (
            "zero hard-safety violations first, successes descending, timeouts "
            "ascending, lateral tolerance failures ascending, any-channel "
            "saturation ascending, translational effort ascending"
        ),
        "ranked_candidates": ranked,
    }
    with (args.output_root / "aggregate_ranked_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(aggregate), handle, indent=2)
    write_aggregate_csv(args.output_root / "aggregate_candidate_table.csv", ranked)

    print("ranking:", flush=True)
    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {item['tag']} success={item['outcomes']['successes']}/50 "
            f"lateral_fail={item['p3_tolerance_failures']['lateral_position']['count']} "
            f"safety={item['outcomes']['hard_safety_violations']} "
            f"elapsed={item['mean_elapsed_physical_time_seconds']:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
