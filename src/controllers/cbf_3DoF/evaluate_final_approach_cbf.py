"""Paired evaluation of a final-approach SAC policy with CBF-QP variants.

The same evaluation seeds are used for every requested variant, which makes
the baseline-versus-filter comparison paired rather than dependent on a new
set of randomly sampled initial states.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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

from controllers.cbf_3DoF.barriers import (
    FinalApproachCBFConfig,
    build_final_approach_constraints,
)
from controllers.cbf_3DoF.diagnostics import hard_cbf_feasibility_margin
from controllers.cbf_3DoF.qp_filter import FinalApproachCBFFilter
from controllers.evaluation_initialization import (
    add_initial_range_arguments,
    initial_ranges_from_args,
    sample_initial_state,
)
from envs.docking3dof.gym_env import Docking3DoFGymEnv


SAC_FINAL_DIR = SRC_PATH / "controllers" / "SAC-3DoF" / "final_approach"
DEFAULT_MODEL_STEM = (
    "sac_final_approach_portobs_rewardv9_600ksteps_"
    "seed10_optuna12_buffer600k_best"
)
DEFAULT_MODEL_DIR = (
    SAC_FINAL_DIR / "models" / "retained_policies"
    / "optuna12_600ksteps_buffer600k"
)
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}.zip"
DEFAULT_VECNORM_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}_vecnormalize.pkl"
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "final_approach" / "evaluations"
)

VARIANT_FLAGS = {
    "baseline": (False, False, False),
    "cone": (True, False, False),
    "speed": (False, True, False),
    "alignment": (False, False, True),
    "both": (True, True, False),
    "all": (True, True, True),
}

STEP_LOG_LABELS = {
    "baseline": "unfiltered",
    "cone": "cone_cbf",
    "speed": "speed_cbf",
    "alignment": "alignment_cbf",
    "both": "combined_cbf",
    "all": "all_cbf",
}
RUN_CONFIG_FILENAME = "run_configuration.json"
EPISODE_METRICS_FILENAME = "per_episode_metrics.csv"
VARIANT_SUMMARY_FILENAME = "aggregate_metrics_by_filter_variant.csv"
PROTECTED_BARRIER_TOLERANCE = 1.0e-6
RESULT_SCHEMA_VERSION = 2

CONSTRAINT_FAMILY_FIELDS = {
    "cone": "enable_cone",
    "speed": "enable_speed",
    "alignment": "enable_alignment",
}


def make_env(seed: int, mode: str) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final",
        mode=mode,
        dt=1.0,
        max_time=5_000.0,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_vecnormalize(path: Path) -> tuple[VecNormalize, DummyVecEnv]:
    dummy = DummyVecEnv([lambda: Monitor(make_env(seed=999_999, mode="testing"))])
    vecnormalize = VecNormalize.load(str(path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize, dummy


def normalize_observation(vecnormalize: VecNormalize, observation: np.ndarray):
    return vecnormalize.normalize_obs(observation.reshape(1, -1))


def classify_protected_barrier_violations(post_values):
    """Return strict and tolerance-aware post-step CBF violations.

    The strict set matches the mathematical barrier condition ``h >= 0``.
    The tolerance-aware set separately identifies crossings larger than the
    evaluator's numerical reporting tolerance.  Neither set changes simulator
    termination or the action applied to the environment.
    """
    strict = tuple(name for name, value in post_values.items() if value < 0.0)
    beyond_tolerance = tuple(
        name
        for name, value in post_values.items()
        if value < -PROTECTED_BARRIER_TOLERANCE
    )
    return strict, beyond_tolerance


def diagnose_qp_failure(core_env, state, config, solver_status):
    """Explain a failed QP without changing its existing fallback behavior.

    An independent LP checks the raw hard CBF rows for the complete enabled
    constraint set and for all enabled family subsets.  Reporting only minimal
    infeasible subsets distinguishes a single-family limitation (for example,
    ``cone``) from a conflict that appears only when families are combined.
    """
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
        full_constraints = build_final_approach_constraints(core_env, state, config)
        full_feasibility = hard_cbf_feasibility_margin(
            full_constraints,
            core_env.action_high,
        )

        minimal_infeasible = []
        for size in range(1, len(enabled_families) + 1):
            for subset in combinations(enabled_families, size):
                subset_set = set(subset)
                if any(set(previous).issubset(subset_set) for previous in minimal_infeasible):
                    continue
                subset_config = replace(
                    config,
                    **{
                        field: family in subset_set
                        for family, field in CONSTRAINT_FAMILY_FIELDS.items()
                    },
                )
                subset_constraints = build_final_approach_constraints(
                    core_env,
                    state,
                    subset_config,
                )
                subset_feasibility = hard_cbf_feasibility_margin(
                    subset_constraints,
                    core_env.action_high,
                )
                if not subset_feasibility.feasible:
                    minimal_infeasible.append(subset)

        group_labels = tuple("+".join(group) for group in minimal_infeasible)
        groups = ",".join(group_labels)
        status = str(solver_status).replace("[", "(").replace("]", ")")
        if config.allow_slack:
            reason = f"soft_qp_solver_failure[{status}]"
            if groups:
                reason += f"|hard_rows_infeasible[{groups}]"
        elif not full_feasibility.feasible:
            detail = groups or "unclassified"
            reason = f"hard_constraints_infeasible[{detail}]"
        else:
            reason = f"solver_failed_despite_hard_feasibility[{status}]"

        return {
            "reason": reason,
            "hard_feasibility_margin": float(full_feasibility.margin),
            "hard_feasibility_solver_status": full_feasibility.solver_status,
            "minimal_infeasible_constraint_groups": groups,
        }
    except Exception as exc:  # diagnostics must not hide the original QP failure
        diagnosis = dict(empty)
        diagnosis["reason"] = (
            f"feasibility_diagnostic_error[{type(exc).__name__}:{exc}]"
        )
        diagnosis["hard_feasibility_solver_status"] = "diagnostic_error"
        return diagnosis


def _serialize_step_records(records: list[dict[str, object]], path: Path) -> None:
    """Store fixed-schema scalar/vector step records in a compact NPZ file."""
    if not records:
        np.savez_compressed(path, step=np.empty(0, dtype=np.int64))
        return

    keys = records[0].keys()
    arrays = {}
    for key in keys:
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


def _update_counter_from_summary(counter: Counter, summary: object) -> None:
    for item in str(summary).split(";"):
        if not item:
            continue
        key, count = item.rsplit(":", 1)
        counter[key] += int(count)


def _baseline_diagnostics(core_env, state, nominal, config):
    constraints = build_final_approach_constraints(core_env, state, config)
    residual = constraints.A @ nominal - constraints.b
    return {
        "action": nominal,
        "status": "not_run",
        "backend": "none",
        "intervened": False,
        "scaled_norm": 0.0,
        "values": constraints.barrier_values,
        "residuals": dict(zip(constraints.names, residual)),
        "slacks": {name: 0.0 for name in constraints.names},
        "failed": False,
        "failure_reason": "",
        "hard_feasibility_margin": np.nan,
        "hard_feasibility_solver_status": "not_checked",
        "minimal_infeasible_constraint_groups": "",
        "kinematics": constraints.kinematics,
    }


def run_rollout(
    *,
    model,
    vecnormalize,
    eval_seed: int,
    mode: str,
    max_steps: int,
    variant: str,
    base_config: FinalApproachCBFConfig,
    output_dir: Path,
    initial_ranges: dict[str, tuple[float, float]],
) -> dict[str, object]:
    env = make_env(eval_seed, mode)
    initial_state = sample_initial_state(eval_seed, initial_ranges)
    observation, info = env.reset(
        seed=eval_seed,
        options={"initial_state": initial_state, "mode": mode},
    )
    total_reward = 0.0
    records: list[dict[str, object]] = []
    cone_enabled, speed_enabled, alignment_enabled = VARIANT_FLAGS[variant]

    # Baseline constraints are evaluated in shadow mode for diagnostics only.
    config = replace(
        base_config,
        enable_cone=True if variant == "baseline" else cone_enabled,
        enable_speed=True if variant == "baseline" else speed_enabled,
        enable_alignment=True if variant == "baseline" else alignment_enabled,
    )
    cbf_filter = (
        None if variant == "baseline" else FinalApproachCBFFilter(env.core_env, config)
    )

    final_info = info
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
            constraints = build_final_approach_constraints(env.core_env, state, config)
            failure_diagnosis = (
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
                "residuals": result.cbf_residuals,
                "slacks": result.slack_values,
                "failed": result.qp_failed,
                "failure_reason": failure_diagnosis["reason"],
                "hard_feasibility_margin": failure_diagnosis[
                    "hard_feasibility_margin"
                ],
                "hard_feasibility_solver_status": failure_diagnosis[
                    "hard_feasibility_solver_status"
                ],
                "minimal_infeasible_constraint_groups": failure_diagnosis[
                    "minimal_infeasible_constraint_groups"
                ],
                "kinematics": constraints.kinematics,
            }

        filtered_physical = np.asarray(diagnostics["action"], dtype=np.float64)
        filtered_normalized = env.physical_action_to_normalized(filtered_physical)
        observation, reward, terminated, truncated, final_info = env.step(
            filtered_normalized
        )
        total_reward += reward

        post_constraints = build_final_approach_constraints(
            env.core_env,
            env.get_physical_state(),
            config,
        )
        post_values = post_constraints.barrier_values
        strict_post_violations, tolerance_post_violations = (
            classify_protected_barrier_violations(post_values)
        )

        values = diagnostics["values"]
        residuals = diagnostics["residuals"]
        slacks = diagnostics["slacks"]
        kinematics = diagnostics["kinematics"]
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
                "psi_error": float(env.core_env.wrap_to_pi(state[4])),
                "angular_rate": float(state[5]),
                "post_time": float(env.core_env.time),
                "post_s": float(final_info.get("s", np.nan)),
                "post_rho": float(final_info.get("rho", np.nan)),
                "post_s_dot": float(final_info.get("s_dot", np.nan)),
                "post_y_p_dot": float(post_constraints.kinematics.y_p_dot),
                "post_port_distance": float(final_info.get("port_distance", np.nan)),
                "post_port_speed": float(final_info.get("port_speed", np.nan)),
                "post_psi_error": abs(float(final_info.get("psi_e", np.nan))),
                "post_angular_rate": abs(float(final_info.get("omega", np.nan))),
                "post_hard_safety_violation": bool(final_info.get("unsafe", False)),
                "post_docking_cone_violation": bool(not geometry.get("in_C_dock", False)),
                "post_total_speed_violation": bool(float(final_info.get("port_speed", np.nan)) > env.core_env.vp_unsafe_final),
                "post_axial_speed_violation": bool(abs(float(final_info.get("s_dot", np.nan))) > env.core_env.sdot_unsafe_final),
                "post_attitude_violation": bool(abs(float(final_info.get("psi_e", np.nan))) > env.core_env.psi_unsafe),
                "post_angular_rate_violation": bool(abs(float(final_info.get("omega", np.nan))) > env.core_env.omega_unsafe),
                "h_cone_upper": float(values.get("cone_upper", np.nan)),
                "h_cone_lower": float(values.get("cone_lower", np.nan)),
                "h_port_speed": float(values.get("port_speed", np.nan)),
                "h_attitude_upper": float(values.get("attitude_upper", np.nan)),
                "h_attitude_lower": float(values.get("attitude_lower", np.nan)),
                "h_angular_rate_upper": float(
                    values.get("angular_rate_upper", np.nan)
                ),
                "h_angular_rate_lower": float(
                    values.get("angular_rate_lower", np.nan)
                ),
                "post_h_cone_upper": float(
                    post_values.get("cone_upper", np.nan)
                ),
                "post_h_cone_lower": float(
                    post_values.get("cone_lower", np.nan)
                ),
                "post_h_port_speed": float(
                    post_values.get("port_speed", np.nan)
                ),
                "post_h_attitude_upper": float(
                    post_values.get("attitude_upper", np.nan)
                ),
                "post_h_attitude_lower": float(
                    post_values.get("attitude_lower", np.nan)
                ),
                "post_h_angular_rate_upper": float(
                    post_values.get("angular_rate_upper", np.nan)
                ),
                "post_h_angular_rate_lower": float(
                    post_values.get("angular_rate_lower", np.nan)
                ),
                "strict_protected_barrier_violation": bool(
                    strict_post_violations
                ),
                "strict_protected_barriers_violated": ",".join(
                    strict_post_violations
                ),
                "protected_barrier_violation_beyond_tolerance": bool(
                    tolerance_post_violations
                ),
                "protected_barriers_violated_beyond_tolerance": ",".join(
                    tolerance_post_violations
                ),
                "residual_cone_upper": float(residuals.get("cone_upper", np.nan)),
                "residual_cone_lower": float(residuals.get("cone_lower", np.nan)),
                "residual_port_speed": float(residuals.get("port_speed", np.nan)),
                "residual_attitude_upper": float(
                    residuals.get("attitude_upper", np.nan)
                ),
                "residual_attitude_lower": float(
                    residuals.get("attitude_lower", np.nan)
                ),
                "residual_angular_rate_upper": float(
                    residuals.get("angular_rate_upper", np.nan)
                ),
                "residual_angular_rate_lower": float(
                    residuals.get("angular_rate_lower", np.nan)
                ),
                "slack_cone_upper": float(slacks.get("cone_upper", np.nan)),
                "slack_cone_lower": float(slacks.get("cone_lower", np.nan)),
                "slack_port_speed": float(slacks.get("port_speed", np.nan)),
                "slack_attitude_upper": float(
                    slacks.get("attitude_upper", np.nan)
                ),
                "slack_attitude_lower": float(
                    slacks.get("attitude_lower", np.nan)
                ),
                "slack_angular_rate_upper": float(
                    slacks.get("angular_rate_upper", np.nan)
                ),
                "slack_angular_rate_lower": float(
                    slacks.get("angular_rate_lower", np.nan)
                ),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    env.close()
    steps = len(records)
    rollout_path = output_dir / (
        f"step_timeseries_{STEP_LOG_LABELS[variant]}_eval_seed{eval_seed:04d}.npz"
    )
    _serialize_step_records(records, rollout_path)

    def values(key):
        return np.asarray([record[key] for record in records], dtype=np.float64)

    def finite_min(key):
        array = values(key)
        finite = array[np.isfinite(array)]
        return float(np.min(finite)) if finite.size else np.nan

    def finite_last(key):
        array = values(key)
        finite = array[np.isfinite(array)]
        return float(finite[-1]) if finite.size else np.nan

    intervention = values("intervention_scaled_norm")
    slacks = np.column_stack(
        [
            values("slack_cone_upper"),
            values("slack_cone_lower"),
            values("slack_port_speed"),
            values("slack_attitude_upper"),
            values("slack_attitude_lower"),
            values("slack_angular_rate_upper"),
            values("slack_angular_rate_lower"),
        ]
    )
    finite_slacks = slacks[np.isfinite(slacks)]
    ended = bool(final_info.get("success") or final_info.get("failure") or final_info.get("time_limit"))
    reason = final_info.get("termination_reason") if ended else "evaluation_step_limit"
    qp_failure_reasons = Counter(
        str(record["qp_failure_reason"])
        for record in records
        if record["qp_failed"] and record["qp_failure_reason"]
    )
    qp_failure_groups = Counter(
        str(record["qp_failure_minimal_infeasible_constraint_groups"])
        for record in records
        if record["qp_failed"]
        and record["qp_failure_minimal_infeasible_constraint_groups"]
    )

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

    return {
        "variant": variant,
        "eval_seed": eval_seed,
        "steps": steps,
        "phase_duration_seconds": finite_last("post_time"),
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)) or not ended,
        "termination_reason": reason,
        "violated_constraints": ",".join(final_info.get("violated_constraints", [])),
        "total_reward": total_reward,
        "final_port_distance": finite_last("post_port_distance"),
        "final_port_speed": finite_last("post_port_speed"),
        "final_axial_position": finite_last("post_s"),
        "final_lateral_position": finite_last("post_rho"),
        "final_axial_velocity": finite_last("post_s_dot"),
        "final_lateral_velocity": finite_last("post_y_p_dot"),
        "final_attitude_error": finite_last("post_psi_error"),
        "final_angular_rate_error": finite_last("post_angular_rate"),
        "hard_safety_violation": bool(any(record["post_hard_safety_violation"] for record in records)),
        "docking_cone_violation": bool(any(record["post_docking_cone_violation"] for record in records)),
        "total_relative_velocity_violation": bool(any(record["post_total_speed_violation"] for record in records)),
        "axial_velocity_violation": bool(any(record["post_axial_speed_violation"] for record in records)),
        "alignment_attitude_violation": bool(any(record["post_attitude_violation"] for record in records)),
        "angular_rate_violation": bool(any(record["post_angular_rate_violation"] for record in records)),
        "intervention_count": int(sum(record["intervened"] for record in records)),
        "intervention_fraction": float(np.mean([record["intervened"] for record in records])),
        "mean_intervention_scaled_norm": float(np.mean(intervention)),
        "max_intervention_scaled_norm": float(np.max(intervention)),
        "qp_failure_count": int(sum(record["qp_failed"] for record in records)),
        "qp_failure_reasons": _format_counter(qp_failure_reasons),
        "qp_failure_minimal_infeasible_constraint_groups": _format_counter(
            qp_failure_groups
        ),
        "minimum_qp_failure_hard_feasibility_margin": finite_min(
            "qp_failure_hard_feasibility_margin"
        ),
        "strict_protected_barrier_violation_count": int(
            sum(record["strict_protected_barrier_violation"] for record in records)
        ),
        "strict_protected_barriers_violated": barrier_names(
            "strict_protected_barriers_violated"
        ),
        "protected_barrier_violation_beyond_tolerance_count": int(
            sum(
                record["protected_barrier_violation_beyond_tolerance"]
                for record in records
            )
        ),
        "protected_barriers_violated_beyond_tolerance": barrier_names(
            "protected_barriers_violated_beyond_tolerance"
        ),
        "max_slack": float(np.max(finite_slacks)) if finite_slacks.size else 0.0,
        "min_h_cone_upper": finite_min("h_cone_upper"),
        "min_h_cone_lower": finite_min("h_cone_lower"),
        "min_h_port_speed": finite_min("h_port_speed"),
        "min_h_attitude_upper": finite_min("h_attitude_upper"),
        "min_h_attitude_lower": finite_min("h_attitude_lower"),
        "min_h_angular_rate_upper": finite_min("h_angular_rate_upper"),
        "min_h_angular_rate_lower": finite_min("h_angular_rate_lower"),
        "min_post_h_cone_upper": finite_min("post_h_cone_upper"),
        "min_post_h_cone_lower": finite_min("post_h_cone_lower"),
        "min_post_h_port_speed": finite_min("post_h_port_speed"),
        "min_post_h_attitude_upper": finite_min("post_h_attitude_upper"),
        "min_post_h_attitude_lower": finite_min("post_h_attitude_lower"),
        "min_post_h_angular_rate_upper": finite_min(
            "post_h_angular_rate_upper"
        ),
        "min_post_h_angular_rate_lower": finite_min(
            "post_h_angular_rate_lower"
        ),
        "step_log": rollout_path.name,
    }


def summarize_variant(variant: str, rows: list[dict[str, object]]):
    reasons = Counter(row["termination_reason"] for row in rows)
    violations = Counter()
    qp_failure_reasons = Counter()
    qp_failure_groups = Counter()
    strict_protected_violations = Counter()
    tolerance_protected_violations = Counter()
    for row in rows:
        violations.update(item for item in str(row["violated_constraints"]).split(",") if item)
        _update_counter_from_summary(qp_failure_reasons, row["qp_failure_reasons"])
        _update_counter_from_summary(
            qp_failure_groups,
            row["qp_failure_minimal_infeasible_constraint_groups"],
        )
        strict_protected_violations.update(
            item
            for item in str(row["strict_protected_barriers_violated"]).split(",")
            if item
        )
        tolerance_protected_violations.update(
            item
            for item in str(
                row["protected_barriers_violated_beyond_tolerance"]
            ).split(",")
            if item
        )
    return {
        "variant": variant,
        "episodes": len(rows),
        "success_count": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "failure_rate": float(np.mean([row["failure"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_phase_duration_seconds": float(np.mean([row.get("phase_duration_seconds", row["steps"]) for row in rows])),
        "mean_final_port_distance": float(np.mean([row.get("final_port_distance", np.nan) for row in rows])),
        "mean_final_port_speed": float(np.mean([row.get("final_port_speed", np.nan) for row in rows])),
        "mean_final_axial_position": float(np.mean([row.get("final_axial_position", np.nan) for row in rows])),
        "mean_final_lateral_position": float(np.mean([row.get("final_lateral_position", np.nan) for row in rows])),
        "mean_final_axial_velocity": float(np.mean([row.get("final_axial_velocity", np.nan) for row in rows])),
        "mean_final_lateral_velocity": float(np.mean([row.get("final_lateral_velocity", np.nan) for row in rows])),
        "mean_final_attitude_error": float(np.mean([row.get("final_attitude_error", np.nan) for row in rows])),
        "mean_final_angular_rate_error": float(np.mean([row.get("final_angular_rate_error", np.nan) for row in rows])),
        "mean_intervention_fraction": float(
            np.mean([row["intervention_fraction"] for row in rows])
        ),
        "intervention_count": int(sum(row.get("intervention_count", 0) for row in rows)),
        "hard_safety_violation_count": int(sum(row.get("hard_safety_violation", False) for row in rows)),
        "docking_cone_violation_count": int(sum(row.get("docking_cone_violation", False) for row in rows)),
        "total_relative_velocity_violation_count": int(sum(row.get("total_relative_velocity_violation", False) for row in rows)),
        "axial_velocity_violation_count": int(sum(row.get("axial_velocity_violation", False) for row in rows)),
        "alignment_attitude_violation_count": int(sum(row.get("alignment_attitude_violation", False) for row in rows)),
        "angular_rate_violation_count": int(sum(row.get("angular_rate_violation", False) for row in rows)),
        "qp_failure_count": int(sum(row["qp_failure_count"] for row in rows)),
        "qp_failure_reasons": _format_counter(qp_failure_reasons),
        "qp_failure_minimal_infeasible_constraint_groups": _format_counter(
            qp_failure_groups
        ),
        "minimum_qp_failure_hard_feasibility_margin": float(
            np.nanmin(
                [
                    row["minimum_qp_failure_hard_feasibility_margin"]
                    for row in rows
                ]
            )
        )
        if any(
            np.isfinite(row["minimum_qp_failure_hard_feasibility_margin"])
            for row in rows
        )
        else np.nan,
        "strict_protected_barrier_violation_count": int(
            sum(row["strict_protected_barrier_violation_count"] for row in rows)
        ),
        "protected_barrier_violation_beyond_tolerance_count": int(
            sum(
                row["protected_barrier_violation_beyond_tolerance_count"]
                for row in rows
            )
        ),
        "maximum_slack": float(max(row["max_slack"] for row in rows)),
        "termination_reasons": ";".join(f"{key}:{value}" for key, value in reasons.items()),
        "violated_constraints": ";".join(f"{key}:{value}" for key, value in violations.items()),
        "strict_protected_barriers_violated": ";".join(
            f"{key}:{value}" for key, value in strict_protected_violations.items()
        ),
        "protected_barriers_violated_beyond_tolerance": ";".join(
            f"{key}:{value}" for key, value in tolerance_protected_violations.items()
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired final-approach SAC versus CBF-QP evaluation."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORM_PATH)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FLAGS),
        default=["baseline", "all"],
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--mode", choices=("nominal", "testing", "training"), default="testing")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--hard-constraints", action="store_true")
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--port-speed-max", type=float, default=0.15)
    parser.add_argument("--cone-k1", type=float, default=0.1)
    parser.add_argument("--cone-k2", type=float, default=0.1)
    parser.add_argument("--speed-alpha", type=float, default=0.5)
    parser.add_argument("--attitude-max-deg", type=float, default=3.0)
    parser.add_argument("--angular-rate-max-deg-s", type=float, default=0.15)
    parser.add_argument("--attitude-k1", type=float, default=0.3)
    parser.add_argument("--attitude-k2", type=float, default=0.3)
    parser.add_argument("--angular-rate-alpha", type=float, default=0.5)
    parser.add_argument("--cone-slack-penalty", type=float, default=1.0e8)
    parser.add_argument("--speed-slack-penalty", type=float, default=1.0e8)
    parser.add_argument("--alignment-slack-penalty", type=float, default=1.0e8)
    parser.add_argument("--output-dir", type=Path, default=None)
    add_initial_range_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0 or args.max_steps <= 0:
        raise ValueError("--num-seeds and --max-steps must be positive.")
    if not args.model.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {args.model}")
    if not args.vecnormalize.is_file():
        raise FileNotFoundError(f"VecNormalize statistics not found: {args.vecnormalize}")
    initial_ranges = initial_ranges_from_args(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_RESULTS_DIR / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = FinalApproachCBFConfig(
        kappa=args.kappa,
        port_speed_max=args.port_speed_max,
        cone_k1=args.cone_k1,
        cone_k2=args.cone_k2,
        speed_alpha=args.speed_alpha,
        attitude_max=np.deg2rad(args.attitude_max_deg),
        angular_rate_max=np.deg2rad(args.angular_rate_max_deg_s),
        attitude_k1=args.attitude_k1,
        attitude_k2=args.attitude_k2,
        angular_rate_alpha=args.angular_rate_alpha,
        cone_slack_penalty=args.cone_slack_penalty,
        speed_slack_penalty=args.speed_slack_penalty,
        alignment_slack_penalty=args.alignment_slack_penalty,
        allow_slack=not args.hard_constraints,
        solver_backend=args.solver,
    )
    model = SAC.load(str(args.model))
    vecnormalize, dummy_env = load_vecnormalize(args.vecnormalize)

    rows = []
    try:
        for variant in args.variants:
            for eval_seed in range(args.seed_start, args.seed_start + args.num_seeds):
                row = run_rollout(
                    model=model,
                    vecnormalize=vecnormalize,
                    eval_seed=eval_seed,
                    mode=args.mode,
                    max_steps=args.max_steps,
                    variant=variant,
                    base_config=config,
                    output_dir=output_dir,
                    initial_ranges=initial_ranges,
                )
                rows.append(row)
                print(
                    f"{variant:8s} seed={eval_seed:4d} "
                    f"success={row['success']} reason={row['termination_reason']} "
                    f"steps={row['steps']} intervention={row['intervention_fraction']:.3f}"
                )
    finally:
        dummy_env.close()

    summaries = [
        summarize_variant(variant, [row for row in rows if row["variant"] == variant])
        for variant in args.variants
    ]
    write_csv(output_dir / EPISODE_METRICS_FILENAME, rows)
    write_csv(output_dir / VARIANT_SUMMARY_FILENAME, summaries)
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
                "initial_ranges": initial_ranges,
                "protected_barrier_tolerance": PROTECTED_BARRIER_TOLERANCE,
                "qp_failure_behavior": (
                    "record_diagnosis_and_retain_nominal_action_fallback"
                ),
                "cbf": asdict(config),
            },
            stream,
            indent=2,
        )

    print(f"\nResults written to: {output_dir.resolve()}")
    for summary in summaries:
        print(
            f"{summary['variant']:8s}: success={summary['success_rate']:.3f}, "
            f"failure={summary['failure_rate']:.3f}, "
            f"intervention={summary['mean_intervention_fraction']:.3f}, "
            f"QP failures={summary['qp_failure_count']}, "
            "strict protected post-step violations="
            f"{summary['strict_protected_barrier_violation_count']}, "
            f"beyond {PROTECTED_BARRIER_TOLERANCE:.0e}="
            f"{summary['protected_barrier_violation_beyond_tolerance_count']}"
        )


if __name__ == "__main__":
    main()
