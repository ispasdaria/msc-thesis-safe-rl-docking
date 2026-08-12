"""Paired evaluation of a closing-phase SAC policy with speed CBF-QP variants.

Every requested variant uses the same deterministic policy, evaluation seeds,
initial-state distribution, VecNormalize statistics, and rollout limit.  The
baseline evaluates the CBF rows in shadow mode without changing the action.
P3 tolerances are performance diagnostics only; they are never CBF rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from controllers.cbf_3DoF.closing_barriers import (
    ClosingCBFConfig,
    build_closing_constraints,
)
from controllers.cbf_3DoF.diagnostics import hard_cbf_feasibility_margin
from controllers.cbf_3DoF.qp_filter import ClosingCBFFilter
from envs.docking3dof.gym_env import Docking3DoFGymEnv


SAC_CLOSING_DIR = SRC_PATH / "controllers" / "SAC-3DoF" / "closing"
DEFAULT_MODEL_STEM = (
    "sac_closing_p3augobs_rewardv15_optuna_trial16_"
    "400ksteps_seed0_buffer300k"
)
DEFAULT_MODEL_DIR = SAC_CLOSING_DIR / "models" / "reward_shaping_trials"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}.zip"
DEFAULT_VECNORM_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}_vecnormalize.pkl"
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "closing" / "evaluations"
)

VARIANT_FLAGS = {
    "baseline": (False, False),
    "total_speed": (True, False),
    "axial_speed": (False, True),
    "both": (True, True),
}
STEP_LOG_LABELS = {
    "baseline": "unfiltered",
    "total_speed": "total_speed_cbf",
    "axial_speed": "axial_speed_cbf",
    "both": "both_speed_cbfs",
}
CONSTRAINT_FAMILY_FIELDS = {
    "total_speed": "enable_port_speed",
    "axial_speed": "enable_axial_port_speed",
}

RUN_CONFIG_FILENAME = "run_configuration.json"
EPISODE_METRICS_FILENAME = "per_episode_metrics.csv"
VARIANT_SUMMARY_FILENAME = "aggregate_metrics_by_filter_variant.csv"
PAIRED_COMPARISON_FILENAME = "paired_baseline_vs_both.csv"
PROTECTED_BARRIER_TOLERANCE = 1.0e-6
RESULT_SCHEMA_VERSION = 1


def make_env(seed: int, mode: str) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="closing",
        mode=mode,
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_vecnormalize(path: Path) -> tuple[VecNormalize, DummyVecEnv]:
    dummy = DummyVecEnv([lambda: Monitor(make_env(999_999, "testing"))])
    vecnormalize = VecNormalize.load(str(path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize, dummy


def normalize_observation(vecnormalize: VecNormalize, observation: np.ndarray):
    return vecnormalize.normalize_obs(observation.reshape(1, -1))


def classify_protected_barrier_violations(post_values):
    """Return strict and tolerance-aware post-step protected-set crossings."""
    strict = tuple(name for name, value in post_values.items() if value < 0.0)
    beyond_tolerance = tuple(
        name
        for name, value in post_values.items()
        if value < -PROTECTED_BARRIER_TOLERANCE
    )
    return strict, beyond_tolerance


def diagnose_qp_failure(core_env, state, config, solver_status):
    """Classify a hard-QP failure using independent actuator-bounded LPs."""
    empty = {
        "reason": "",
        "hard_feasibility_margin": np.nan,
        "hard_feasibility_solver_status": "not_checked",
        "minimal_infeasible_constraint_groups": "",
    }
    try:
        enabled_families = tuple(
            family
            for family, field in CONSTRAINT_FAMILY_FIELDS.items()
            if getattr(config, field)
        )
        full_constraints = build_closing_constraints(core_env, state, config)
        full_feasibility = hard_cbf_feasibility_margin(
            full_constraints,
            core_env.action_high,
        )

        minimal_infeasible = []
        for size in range(1, len(enabled_families) + 1):
            for subset in combinations(enabled_families, size):
                subset_set = set(subset)
                if any(
                    set(previous).issubset(subset_set)
                    for previous in minimal_infeasible
                ):
                    continue
                subset_config = replace(
                    config,
                    **{
                        field: family in subset_set
                        for family, field in CONSTRAINT_FAMILY_FIELDS.items()
                    },
                )
                feasibility = hard_cbf_feasibility_margin(
                    build_closing_constraints(core_env, state, subset_config),
                    core_env.action_high,
                )
                if not feasibility.feasible:
                    minimal_infeasible.append(subset)

        groups = ",".join("+".join(group) for group in minimal_infeasible)
        status = str(solver_status).replace("[", "(").replace("]", ")")
        if not full_feasibility.feasible:
            reason = f"hard_constraints_infeasible[{groups or 'unclassified'}]"
        else:
            reason = f"solver_failed_despite_hard_feasibility[{status}]"
        return {
            "reason": reason,
            "hard_feasibility_margin": float(full_feasibility.margin),
            "hard_feasibility_solver_status": full_feasibility.solver_status,
            "minimal_infeasible_constraint_groups": groups,
        }
    except Exception as exc:
        diagnosis = dict(empty)
        diagnosis["reason"] = (
            f"feasibility_diagnostic_error[{type(exc).__name__}:{exc}]"
        )
        diagnosis["hard_feasibility_solver_status"] = "diagnostic_error"
        return diagnosis


def _baseline_diagnostics(core_env, state, nominal, config):
    """Evaluate both CBF families without modifying the baseline action."""
    constraints = build_closing_constraints(core_env, state, config)
    residuals = constraints.A @ nominal - constraints.b
    return {
        "action": nominal,
        "status": "not_run",
        "backend": "none",
        "intervened": False,
        "scaled_norm": 0.0,
        "values": constraints.barrier_values,
        "residuals": dict(zip(constraints.names, residuals)),
        "failed": False,
        "failure_reason": "",
        "hard_feasibility_margin": np.nan,
        "hard_feasibility_solver_status": "not_checked",
        "minimal_infeasible_constraint_groups": "",
        "kinematics": constraints.kinematics,
    }


def _serialize_step_records(records: list[dict[str, object]], path: Path) -> None:
    if not records:
        np.savez_compressed(path, step=np.empty(0, dtype=np.int64))
        return
    arrays = {}
    for key in records[0]:
        values = [record[key] for record in records]
        first = values[0]
        if isinstance(first, str):
            arrays[key] = np.asarray(values, dtype=np.str_)
        elif isinstance(first, (bool, np.bool_)):
            arrays[key] = np.asarray(values, dtype=np.bool_)
        else:
            arrays[key] = np.asarray(values)
    np.savez_compressed(path, **arrays)


def _format_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def _update_counter(counter: Counter, summary: object) -> None:
    for item in str(summary).split(";"):
        if item:
            key, count = item.rsplit(":", 1)
            counter[key] += int(count)


def run_rollout(
    *,
    model,
    vecnormalize,
    eval_seed: int,
    mode: str,
    max_steps: int,
    variant: str,
    base_config: ClosingCBFConfig,
    output_dir: Path,
) -> dict[str, object]:
    env = make_env(eval_seed, mode)
    observation, reset_info = env.reset(seed=eval_seed)
    initial_state = env.get_physical_state()
    p3_target = np.asarray(env.core_env.p3_target, dtype=np.float64).copy()
    total_reward = 0.0
    records: list[dict[str, object]] = []
    total_enabled, axial_enabled = VARIANT_FLAGS[variant]

    # Baseline runs both barriers in shadow mode for a like-for-like comparison.
    config = replace(
        base_config,
        enable_port_speed=True if variant == "baseline" else total_enabled,
        enable_axial_port_speed=True if variant == "baseline" else axial_enabled,
    )
    cbf_filter = None if variant == "baseline" else ClosingCBFFilter(
        env.core_env,
        config,
    )
    initial_constraints = build_closing_constraints(
        env.core_env,
        initial_state,
        config,
    )
    initial_barriers = initial_constraints.barrier_values
    initial_protected_set_satisfied = all(
        value >= 0.0 for value in initial_barriers.values()
    )

    final_info = reset_info
    for step_index in range(max_steps):
        state = env.get_physical_state()
        normalized_observation = normalize_observation(vecnormalize, observation)
        action_normalized, _ = model.predict(
            normalized_observation,
            deterministic=True,
        )
        nominal_physical = env.normalized_action_to_physical(action_normalized[0])

        if cbf_filter is None:
            diagnostics = _baseline_diagnostics(
                env.core_env,
                state,
                nominal_physical,
                config,
            )
        else:
            result = cbf_filter.filter(state, nominal_physical)
            constraints = build_closing_constraints(env.core_env, state, config)
            failure = (
                diagnose_qp_failure(
                    env.core_env,
                    state,
                    config,
                    result.solver_status,
                )
                if result.qp_failed
                else {
                    "reason": "",
                    "hard_feasibility_margin": np.nan,
                    "hard_feasibility_solver_status": "not_checked",
                    "minimal_infeasible_constraint_groups": "",
                }
            )
            diagnostics = {
                "action": result.action_filtered,
                "status": result.solver_status,
                "backend": result.solver_backend,
                "intervened": result.intervened,
                "scaled_norm": result.intervention_scaled_norm,
                "values": result.barrier_values,
                # Raw hard-row residuals, never relaxed residuals.
                "residuals": result.cbf_residuals,
                "failed": result.qp_failed,
                "failure_reason": failure["reason"],
                "hard_feasibility_margin": failure["hard_feasibility_margin"],
                "hard_feasibility_solver_status": failure[
                    "hard_feasibility_solver_status"
                ],
                "minimal_infeasible_constraint_groups": failure[
                    "minimal_infeasible_constraint_groups"
                ],
                "kinematics": constraints.kinematics,
            }

        filtered_physical = np.asarray(diagnostics["action"], dtype=np.float64)
        filtered_normalized = env.physical_action_to_normalized(filtered_physical)
        observation, reward, terminated, truncated, final_info = env.step(
            filtered_normalized
        )
        total_reward += float(reward)

        post_constraints = build_closing_constraints(
            env.core_env,
            env.get_physical_state(),
            config,
        )
        post_values = post_constraints.barrier_values
        strict, beyond_tolerance = classify_protected_barrier_violations(
            post_values
        )
        values = diagnostics["values"]
        residuals = diagnostics["residuals"]
        kinematics = diagnostics["kinematics"]
        reward_components = final_info.get("reward_components", {})
        geometry = final_info.get("geometry", {})
        velocity = final_info.get("velocity", {})
        alignment = final_info.get("alignment", {})

        records.append(
            {
                "step": step_index,
                "time": float(env.core_env.time),
                "state": state,
                "action_nominal": nominal_physical,
                "action_filtered": filtered_physical,
                "intervened": diagnostics["intervened"],
                "intervention_scaled_norm": diagnostics["scaled_norm"],
                "solver_status": str(diagnostics["status"]),
                "solver_backend": str(diagnostics["backend"]),
                "qp_failed": diagnostics["failed"],
                "qp_failure_reason": str(diagnostics["failure_reason"]),
                "qp_failure_hard_feasibility_margin": float(
                    diagnostics["hard_feasibility_margin"]
                ),
                "qp_failure_hard_feasibility_solver_status": str(
                    diagnostics["hard_feasibility_solver_status"]
                ),
                "qp_failure_minimal_infeasible_constraint_groups": str(
                    diagnostics["minimal_infeasible_constraint_groups"]
                ),
                "s": float(kinematics.s),
                "y_p": float(kinematics.y_p),
                "s_dot": float(kinematics.s_dot),
                "y_p_dot": float(kinematics.y_p_dot),
                "port_speed": float(np.linalg.norm(kinematics.v_p)),
                "h_port_speed": float(values.get("port_speed", np.nan)),
                "h_axial_port_speed": float(
                    values.get("axial_port_speed", np.nan)
                ),
                "post_h_port_speed": float(
                    post_values.get("port_speed", np.nan)
                ),
                "post_h_axial_port_speed": float(
                    post_values.get("axial_port_speed", np.nan)
                ),
                "strict_protected_barrier_violation": bool(strict),
                "strict_protected_barriers_violated": ",".join(strict),
                "protected_barrier_violation_beyond_tolerance": bool(
                    beyond_tolerance
                ),
                "protected_barriers_violated_beyond_tolerance": ",".join(
                    beyond_tolerance
                ),
                "residual_port_speed": float(
                    residuals.get("port_speed", np.nan)
                ),
                "residual_axial_port_speed": float(
                    residuals.get("axial_port_speed", np.nan)
                ),
                "post_port_distance": float(
                    final_info.get("port_distance", np.nan)
                ),
                "post_port_speed": float(final_info.get("port_speed", np.nan)),
                "post_s_dot_abs": abs(float(final_info.get("s_dot", np.nan))),
                "post_velocity_safe": bool(velocity.get("safe", False)),
                "post_velocity_warning": bool(velocity.get("warn", False)),
                "post_velocity_unsafe": bool(velocity.get("unsafe", False)),
                "post_geometry_safe": bool(geometry.get("safe", False)),
                "post_geometry_warning": bool(geometry.get("warn", False)),
                "post_geometry_unsafe": bool(geometry.get("unsafe", False)),
                "post_alignment_safe": bool(alignment.get("safe", False)),
                "post_alignment_warning": bool(
                    alignment.get("warn", False)
                ),
                "post_alignment_unsafe": bool(
                    alignment.get("unsafe", False)
                ),
                "p3_position_error": float(
                    reward_components.get("pos_error", np.nan)
                ),
                "p3_velocity_error": float(
                    reward_components.get("vel_error", np.nan)
                ),
                "p3_attitude_error": float(
                    reward_components.get("attitude_error", np.nan)
                ),
                "p3_rate_error": float(
                    reward_components.get("rate_error", np.nan)
                ),
                "p3_position_ratio": float(
                    reward_components.get("p3_position_ratio", np.nan)
                ),
                "p3_max_target_ratio": float(
                    np.nanmax(
                        [
                            reward_components.get("p3_position_ratio", np.nan),
                            reward_components.get("p3_vx_ratio", np.nan),
                            reward_components.get("p3_vy_ratio", np.nan),
                            reward_components.get("p3_attitude_ratio", np.nan),
                            reward_components.get("p3_rate_ratio", np.nan),
                        ]
                    )
                ),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    env.close()
    rollout_path = output_dir / (
        f"step_timeseries_{STEP_LOG_LABELS[variant]}_eval_seed{eval_seed:04d}.npz"
    )
    _serialize_step_records(records, rollout_path)

    def array(key):
        return np.asarray([record[key] for record in records], dtype=np.float64)

    def finite_min(key):
        values = array(key)
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else np.nan

    def finite_last(key):
        values = array(key)
        finite = values[np.isfinite(values)]
        return float(finite[-1]) if finite.size else np.nan

    def finite_max(key):
        values = array(key)
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else np.nan

    def barrier_names(key):
        return ",".join(
            sorted(
                {
                    name
                    for record in records
                    for name in str(record[key]).split(",")
                    if name
                }
            )
        )

    qp_reasons = Counter(
        record["qp_failure_reason"]
        for record in records
        if record["qp_failed"] and record["qp_failure_reason"]
    )
    qp_groups = Counter(
        record["qp_failure_minimal_infeasible_constraint_groups"]
        for record in records
        if record["qp_failed"]
        and record["qp_failure_minimal_infeasible_constraint_groups"]
    )
    ended = bool(
        final_info.get("success")
        or final_info.get("failure")
        or final_info.get("time_limit")
    )
    reason = final_info.get("termination_reason") if ended else "evaluation_step_limit"
    intervention = array("intervention_scaled_norm")

    return {
        "variant": variant,
        "eval_seed": eval_seed,
        "initial_rx": float(initial_state[0]),
        "initial_ry": float(initial_state[1]),
        "initial_vx": float(initial_state[2]),
        "initial_vy": float(initial_state[3]),
        "initial_psi": float(initial_state[4]),
        "initial_omega": float(initial_state[5]),
        "p3_target_rx": float(p3_target[0]),
        "p3_target_ry": float(p3_target[1]),
        "p3_target_vx": float(p3_target[2]),
        "p3_target_vy": float(p3_target[3]),
        "p3_target_psi": float(p3_target[4]),
        "p3_target_omega": float(p3_target[5]),
        "initial_h_port_speed": float(
            initial_barriers.get("port_speed", np.nan)
        ),
        "initial_h_axial_port_speed": float(
            initial_barriers.get("axial_port_speed", np.nan)
        ),
        "initial_protected_set_satisfied": initial_protected_set_satisfied,
        "steps": len(records),
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)) or not ended,
        "termination_reason": reason,
        "violated_constraints": ",".join(
            final_info.get("violated_constraints", [])
        ),
        "total_reward": total_reward,
        "final_port_distance": finite_last("post_port_distance"),
        "minimum_port_distance": finite_min("post_port_distance"),
        "final_port_speed": finite_last("post_port_speed"),
        "maximum_port_speed": finite_max("post_port_speed"),
        "final_abs_axial_speed": finite_last("post_s_dot_abs"),
        "maximum_abs_axial_speed": finite_max("post_s_dot_abs"),
        "final_p3_position_error": finite_last("p3_position_error"),
        "minimum_p3_position_error": finite_min("p3_position_error"),
        "final_p3_velocity_error": finite_last("p3_velocity_error"),
        "final_p3_attitude_error": finite_last("p3_attitude_error"),
        "final_p3_rate_error": finite_last("p3_rate_error"),
        "final_p3_max_target_ratio": finite_last("p3_max_target_ratio"),
        "velocity_safe_fraction": float(np.mean(array("post_velocity_safe"))),
        "velocity_warning_fraction": float(
            np.mean(array("post_velocity_warning"))
        ),
        "velocity_unsafe_fraction": float(np.mean(array("post_velocity_unsafe"))),
        "geometry_safe_fraction": float(np.mean(array("post_geometry_safe"))),
        "geometry_warning_fraction": float(
            np.mean(array("post_geometry_warning"))
        ),
        "geometry_unsafe_fraction": float(
            np.mean(array("post_geometry_unsafe"))
        ),
        "alignment_safe_fraction": float(np.mean(array("post_alignment_safe"))),
        "alignment_warning_fraction": float(
            np.mean(array("post_alignment_warning"))
        ),
        "alignment_unsafe_fraction": float(
            np.mean(array("post_alignment_unsafe"))
        ),
        "intervention_count": int(sum(r["intervened"] for r in records)),
        "intervention_fraction": float(np.mean([r["intervened"] for r in records])),
        "mean_intervention_scaled_norm": float(np.mean(intervention)),
        "max_intervention_scaled_norm": float(np.max(intervention)),
        "qp_failure_count": int(sum(r["qp_failed"] for r in records)),
        "qp_failure_reasons": _format_counter(qp_reasons),
        "qp_failure_minimal_infeasible_constraint_groups": _format_counter(
            qp_groups
        ),
        "minimum_qp_failure_hard_feasibility_margin": finite_min(
            "qp_failure_hard_feasibility_margin"
        ),
        "strict_protected_barrier_violation_count": int(
            sum(r["strict_protected_barrier_violation"] for r in records)
        ),
        "strict_protected_barriers_violated": barrier_names(
            "strict_protected_barriers_violated"
        ),
        "protected_barrier_violation_beyond_tolerance_count": int(
            sum(r["protected_barrier_violation_beyond_tolerance"] for r in records)
        ),
        "protected_barriers_violated_beyond_tolerance": barrier_names(
            "protected_barriers_violated_beyond_tolerance"
        ),
        "min_h_port_speed": finite_min("h_port_speed"),
        "min_h_axial_port_speed": finite_min("h_axial_port_speed"),
        "min_post_h_port_speed": finite_min("post_h_port_speed"),
        "min_post_h_axial_port_speed": finite_min("post_h_axial_port_speed"),
        "min_raw_residual_port_speed": finite_min("residual_port_speed"),
        "min_raw_residual_axial_port_speed": finite_min(
            "residual_axial_port_speed"
        ),
        "step_log": rollout_path.name,
    }


def summarize_variant(variant: str, rows: list[dict[str, object]]):
    reasons = Counter(row["termination_reason"] for row in rows)
    violations = Counter()
    qp_reasons = Counter()
    qp_groups = Counter()
    strict = Counter()
    tolerant = Counter()
    for row in rows:
        violations.update(
            item for item in str(row["violated_constraints"]).split(",") if item
        )
        _update_counter(qp_reasons, row["qp_failure_reasons"])
        _update_counter(
            qp_groups,
            row["qp_failure_minimal_infeasible_constraint_groups"],
        )
        strict.update(
            item
            for item in str(row["strict_protected_barriers_violated"]).split(",")
            if item
        )
        tolerant.update(
            item
            for item in str(
                row["protected_barriers_violated_beyond_tolerance"]
            ).split(",")
            if item
        )

    finite_margins = [
        row["minimum_qp_failure_hard_feasibility_margin"]
        for row in rows
        if np.isfinite(row["minimum_qp_failure_hard_feasibility_margin"])
    ]
    return {
        "variant": variant,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "failure_rate": float(np.mean([row["failure"] for row in rows])),
        "time_limit_rate": float(np.mean([row["time_limit"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_final_p3_position_error": float(
            np.mean([row["final_p3_position_error"] for row in rows])
        ),
        "mean_minimum_p3_position_error": float(
            np.mean([row["minimum_p3_position_error"] for row in rows])
        ),
        "mean_final_p3_max_target_ratio": float(
            np.mean([row["final_p3_max_target_ratio"] for row in rows])
        ),
        "mean_velocity_safe_fraction": float(
            np.mean([row["velocity_safe_fraction"] for row in rows])
        ),
        "mean_velocity_warning_fraction": float(
            np.mean([row["velocity_warning_fraction"] for row in rows])
        ),
        "mean_velocity_unsafe_fraction": float(
            np.mean([row["velocity_unsafe_fraction"] for row in rows])
        ),
        "mean_geometry_warning_fraction": float(
            np.mean([row["geometry_warning_fraction"] for row in rows])
        ),
        "mean_geometry_unsafe_fraction": float(
            np.mean([row["geometry_unsafe_fraction"] for row in rows])
        ),
        "mean_alignment_warning_fraction": float(
            np.mean([row["alignment_warning_fraction"] for row in rows])
        ),
        "mean_alignment_unsafe_fraction": float(
            np.mean([row["alignment_unsafe_fraction"] for row in rows])
        ),
        "mean_intervention_fraction": float(
            np.mean([row["intervention_fraction"] for row in rows])
        ),
        "initial_protected_set_violation_count": int(
            sum(not row["initial_protected_set_satisfied"] for row in rows)
        ),
        "qp_failure_count": int(sum(row["qp_failure_count"] for row in rows)),
        "qp_failure_reasons": _format_counter(qp_reasons),
        "qp_failure_minimal_infeasible_constraint_groups": _format_counter(
            qp_groups
        ),
        "minimum_qp_failure_hard_feasibility_margin": (
            float(np.min(finite_margins)) if finite_margins else np.nan
        ),
        "strict_protected_barrier_violation_count": int(
            sum(row["strict_protected_barrier_violation_count"] for row in rows)
        ),
        "protected_barrier_violation_beyond_tolerance_count": int(
            sum(
                row["protected_barrier_violation_beyond_tolerance_count"]
                for row in rows
            )
        ),
        "termination_reasons": _format_counter(reasons),
        "violated_constraints": _format_counter(violations),
        "strict_protected_barriers_violated": _format_counter(strict),
        "protected_barriers_violated_beyond_tolerance": _format_counter(
            tolerant
        ),
    }


def build_paired_comparison(rows: list[dict[str, object]]):
    """Build seed-matched baseline-versus-both rows when both are present."""
    indexed = {(row["variant"], row["eval_seed"]): row for row in rows}
    seeds = sorted(
        seed
        for variant, seed in indexed
        if variant == "baseline" and ("both", seed) in indexed
    )
    paired = []
    for seed in seeds:
        baseline = indexed[("baseline", seed)]
        cbf = indexed[("both", seed)]
        paired.append(
            {
                "eval_seed": seed,
                "baseline_success": baseline["success"],
                "cbf_success": cbf["success"],
                "baseline_failure": baseline["failure"],
                "cbf_failure": cbf["failure"],
                "baseline_final_p3_position_error": baseline[
                    "final_p3_position_error"
                ],
                "cbf_final_p3_position_error": cbf["final_p3_position_error"],
                "cbf_minus_baseline_final_p3_position_error": (
                    cbf["final_p3_position_error"]
                    - baseline["final_p3_position_error"]
                ),
                "baseline_minimum_p3_position_error": baseline[
                    "minimum_p3_position_error"
                ],
                "cbf_minimum_p3_position_error": cbf[
                    "minimum_p3_position_error"
                ],
                "baseline_velocity_warning_fraction": baseline[
                    "velocity_warning_fraction"
                ],
                "cbf_velocity_warning_fraction": cbf[
                    "velocity_warning_fraction"
                ],
                "baseline_protected_violations_beyond_tolerance": baseline[
                    "protected_barrier_violation_beyond_tolerance_count"
                ],
                "cbf_protected_violations_beyond_tolerance": cbf[
                    "protected_barrier_violation_beyond_tolerance_count"
                ],
                "cbf_intervention_fraction": cbf["intervention_fraction"],
                "cbf_qp_failure_count": cbf["qp_failure_count"],
            }
        )
    return paired


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired closing SAC versus hard speed-CBF evaluation."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORM_PATH)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FLAGS),
        default=["baseline", "both"],
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("nominal", "testing", "training"),
        default="testing",
    )
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--port-speed-max", type=float, default=0.25)
    parser.add_argument("--axial-port-speed-max", type=float, default=0.15)
    parser.add_argument("--port-speed-alpha", type=float, default=0.5)
    parser.add_argument("--axial-port-speed-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0 or args.max_steps <= 0:
        raise ValueError("--num-seeds and --max-steps must be positive.")
    if not args.model.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {args.model}")
    if not args.vecnormalize.is_file():
        raise FileNotFoundError(
            f"VecNormalize statistics not found: {args.vecnormalize}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_RESULTS_DIR / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Closing thesis evaluations deliberately use hard constraints.
    config = ClosingCBFConfig(
        port_speed_max=args.port_speed_max,
        axial_port_speed_max=args.axial_port_speed_max,
        port_speed_alpha=args.port_speed_alpha,
        axial_port_speed_alpha=args.axial_port_speed_alpha,
        allow_slack=False,
        solver_backend=args.solver,
    )
    model = SAC.load(str(args.model))
    vecnormalize, dummy_env = load_vecnormalize(args.vecnormalize)

    rows = []
    try:
        for variant in args.variants:
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                row = run_rollout(
                    model=model,
                    vecnormalize=vecnormalize,
                    eval_seed=seed,
                    mode=args.mode,
                    max_steps=args.max_steps,
                    variant=variant,
                    base_config=config,
                    output_dir=output_dir,
                )
                rows.append(row)
                print(
                    f"{variant:12s} seed={seed:4d} success={row['success']} "
                    f"reason={row['termination_reason']} steps={row['steps']} "
                    f"P3 error={row['final_p3_position_error']:.2f} m "
                    f"intervention={row['intervention_fraction']:.3f}"
                )
    finally:
        dummy_env.close()

    summaries = [
        summarize_variant(
            variant,
            [row for row in rows if row["variant"] == variant],
        )
        for variant in args.variants
    ]
    paired = build_paired_comparison(rows)
    write_csv(output_dir / EPISODE_METRICS_FILENAME, rows)
    write_csv(output_dir / VARIANT_SUMMARY_FILENAME, summaries)
    write_csv(output_dir / PAIRED_COMPARISON_FILENAME, paired)

    metadata_env = make_env(999_998, args.mode)
    try:
        metadata_env.reset(seed=999_998)
        environment_metadata = {
            "vp_safe_close": float(metadata_env.core_env.vp_safe_close),
            "vp_unsafe_close": float(metadata_env.core_env.vp_unsafe_close),
            "sdot_safe_close": float(metadata_env.core_env.sdot_safe_close),
            "sdot_unsafe_close": float(metadata_env.core_env.sdot_unsafe_close),
            "p3_target_tolerance": np.asarray(
                metadata_env.core_env.p3_target_tolerance,
                dtype=float,
            ).tolist(),
        }
    finally:
        metadata_env.close()

    with (output_dir / RUN_CONFIG_FILENAME).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "model": str(args.model.resolve()),
                "vecnormalize": str(args.vecnormalize.resolve()),
                "variants": args.variants,
                "seed_start": args.seed_start,
                "num_seeds": args.num_seeds,
                "mode": args.mode,
                "max_steps": args.max_steps,
                "paired_evaluation": True,
                "protected_barrier_tolerance": PROTECTED_BARRIER_TOLERANCE,
                "constraint_policy": "hard_no_slack",
                "qp_failure_behavior": (
                    "record_feasibility_diagnosis_and_retain_nominal_action_"
                    "fallback_not_cbf_certified"
                ),
                "p3_tolerances_used_as_cbf_constraints": False,
                "p3_target_sampling": (
                    "sampled_per_episode_and_saved_in_per_episode_metrics"
                ),
                "closing_reward_version": metadata_env.closing_reward_version,
                "cbf": asdict(config),
                "environment": environment_metadata,
            },
            stream,
            indent=2,
        )

    print(f"\nResults written to: {output_dir.resolve()}")
    for summary in summaries:
        print(
            f"{summary['variant']:12s}: success={summary['success_rate']:.3f}, "
            f"failure={summary['failure_rate']:.3f}, "
            f"P3 error={summary['mean_final_p3_position_error']:.2f} m, "
            f"intervention={summary['mean_intervention_fraction']:.3f}, "
            f"QP failures={summary['qp_failure_count']}, strict crossings="
            f"{summary['strict_protected_barrier_violation_count']}, beyond "
            f"{PROTECTED_BARRIER_TOLERANCE:.0e}="
            f"{summary['protected_barrier_violation_beyond_tolerance_count']}"
        )


if __name__ == "__main__":
    main()
