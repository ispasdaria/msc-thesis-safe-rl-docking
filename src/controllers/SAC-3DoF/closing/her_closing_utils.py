"""Shared evaluation helpers for goal-conditioned closing SAC experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingEnv,
    GoalConditionedClosingV2Env,
    GoalConditionedClosingV3Env,
)
from her_residual_env import ResidualV15GoalConditionedClosingEnv
from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)


HER_ENV_CLASSES = {
    "v1": GoalConditionedClosingEnv,
    "v2": GoalConditionedClosingV2Env,
    "v3": GoalConditionedClosingV3Env,
    "v4": ResidualV15GoalConditionedClosingEnv,
    "lqr_residual": ResidualLQRGoalConditionedClosingEnv,
    "lqr_reference_residual": ReferenceResidualLQRGoalConditionedClosingEnv,
}

GOAL_COMPONENTS = ("axial", "lateral", "vx", "vy", "attitude", "rate")


def make_her_env(
    seed,
    mode,
    max_time=12_000.0,
    reward_version="v1",
    **env_kwargs,
):
    try:
        env_class = HER_ENV_CLASSES[reward_version]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HER reward version {reward_version!r}; expected one of "
            f"{sorted(HER_ENV_CLASSES)}."
        ) from exc
    return env_class(
        mode=mode,
        dt=1.0,
        max_time=max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        **env_kwargs,
    )


def physical_goal_metrics(env):
    state = np.asarray(env.core_env.state, dtype=np.float64)
    target = np.asarray(env.core_env.p3_target, dtype=np.float64)
    error = state - target
    error[4] = env.core_env.wrap_to_pi(error[4])
    tolerance = np.asarray(
        env.core_env.p3_target_tolerance,
        dtype=np.float64,
    )
    ratios = np.abs(error) / np.maximum(tolerance, 1e-12)
    metrics = {
        "position_error": float(np.linalg.norm(error[:2])),
        "axial_error": float(abs(error[0])),
        "lateral_error": float(abs(error[1])),
        "vx_error": float(abs(error[2])),
        "vy_error": float(abs(error[3])),
        "velocity_error": float(np.linalg.norm(error[2:4])),
        "attitude_error_deg": float(np.rad2deg(abs(error[4]))),
        "rate_error_degs": float(np.rad2deg(abs(error[5]))),
        "max_target_ratio": float(np.max(ratios)),
        "limiting_component": GOAL_COMPONENTS[int(np.argmax(ratios))],
    }
    metrics.update(
        {
            f"{component}_ratio": float(ratio)
            for component, ratio in zip(GOAL_COMPONENTS, ratios)
        }
    )
    return metrics


def _snapshot_fields(prefix, metrics):
    fields = {
        f"{prefix}_limiting_component": metrics["limiting_component"],
        f"{prefix}_position_error": metrics["position_error"],
    }
    fields.update(
        {
            f"{prefix}_{component}_ratio": metrics[f"{component}_ratio"]
            for component in GOAL_COMPONENTS
        }
    )
    return fields


def evaluate_model(
    model,
    seeds,
    max_steps=12_000,
    mode="testing",
    reward_version="v1",
    env_kwargs=None,
):
    rows = []
    for seed in seeds:
        environment_options = dict(env_kwargs or {})
        env = make_her_env(
            seed=int(seed),
            mode=mode,
            max_time=max_steps,
            reward_version=reward_version,
            **environment_options,
        )
        observation, _ = env.reset(seed=int(seed))
        initial = physical_goal_metrics(env)
        best_ratio = initial["max_target_ratio"]
        best_joint = dict(initial)
        minimum_position_error = initial["position_error"]
        closest_position = dict(initial)
        target_2x_steps = 0
        longest_target_2x_dwell_steps = 0
        current_target_2x_dwell_steps = 0
        total_reward = 0.0
        final_info = {}
        steps = 0

        for steps in range(1, int(max_steps) + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(
                action
            )
            total_reward += float(reward)
            metrics = physical_goal_metrics(env)
            if metrics["max_target_ratio"] < best_ratio:
                best_ratio = metrics["max_target_ratio"]
                best_joint = dict(metrics)
            if metrics["position_error"] < minimum_position_error:
                minimum_position_error = metrics["position_error"]
                closest_position = dict(metrics)
            if metrics["max_target_ratio"] <= 2.0:
                target_2x_steps += 1
                current_target_2x_dwell_steps += 1
                longest_target_2x_dwell_steps = max(
                    longest_target_2x_dwell_steps,
                    current_target_2x_dwell_steps,
                )
            else:
                current_target_2x_dwell_steps = 0
            if bool(final_info.get("is_success", False)) or terminated or truncated:
                break

        final = physical_goal_metrics(env)
        success = bool(final_info.get("is_success", False))
        failure = bool(final_info.get("failure", False))
        time_limit = bool(
            final_info.get("time_limit", False)
            or (steps >= int(max_steps) and not success and not failure)
        )
        row = {
                "evaluation_seed": int(seed),
                "success": int(success),
                "failure": int(failure),
                "time_limit": int(time_limit),
                "steps": int(steps),
                "return": float(total_reward),
                "initial_position_error": initial["position_error"],
                "final_position_error": final["position_error"],
                "minimum_position_error": minimum_position_error,
                "best_max_target_ratio": best_ratio,
                "target_2x_steps": int(target_2x_steps),
                "longest_target_2x_dwell_steps": int(
                    longest_target_2x_dwell_steps
                ),
                "final_lateral_error": final["lateral_error"],
                "final_velocity_error": final["velocity_error"],
                "final_attitude_error_deg": final["attitude_error_deg"],
                "final_rate_error_degs": final["rate_error_degs"],
                "final_max_target_ratio": final["max_target_ratio"],
            }
        row.update(_snapshot_fields("best_joint", best_joint))
        row.update(_snapshot_fields("closest", closest_position))
        row.update(_snapshot_fields("final", final))
        rows.append(row)
        env.close()
    return rows


def summarize_rollouts(rows):
    if not rows:
        raise ValueError("At least one rollout is required.")

    def mean(field):
        return float(np.mean([float(row[field]) for row in rows]))

    summary = {
        "n_episodes": len(rows),
        "n_success": int(sum(row["success"] for row in rows)),
        "n_failure": int(sum(row["failure"] for row in rows)),
        "n_timeout": int(sum(row["time_limit"] for row in rows)),
        "success_rate": mean("success"),
        "failure_rate": mean("failure"),
        "time_limit_rate": mean("time_limit"),
        "mean_steps": mean("steps"),
        "mean_return": mean("return"),
        "mean_final_position_error": mean("final_position_error"),
        "mean_minimum_position_error": mean("minimum_position_error"),
        "mean_best_max_target_ratio": mean("best_max_target_ratio"),
        "mean_target_2x_steps": mean("target_2x_steps"),
        "mean_longest_target_2x_dwell_steps": mean(
            "longest_target_2x_dwell_steps"
        ),
        "mean_final_lateral_error": mean("final_lateral_error"),
        "mean_final_velocity_error": mean("final_velocity_error"),
        "mean_final_attitude_error_deg": mean("final_attitude_error_deg"),
        "mean_final_rate_error_degs": mean("final_rate_error_degs"),
        "mean_final_max_target_ratio": mean("final_max_target_ratio"),
    }
    for prefix in ("best_joint", "closest", "final"):
        for component in GOAL_COMPONENTS:
            field = f"{prefix}_{component}_ratio"
            if field in rows[0]:
                summary[f"mean_{field}"] = mean(field)
    if "best_joint_limiting_component" in rows[0]:
        for component in GOAL_COMPONENTS:
            summary[f"best_joint_limiting_{component}_fraction"] = float(
                np.mean(
                    [
                        row["best_joint_limiting_component"] == component
                        for row in rows
                    ]
                )
            )
    return summary


def checkpoint_rank(summary):
    return (
        float(summary["success_rate"]),
        -float(summary["failure_rate"]),
        -float(summary["mean_best_max_target_ratio"]),
        float(summary["mean_longest_target_2x_dwell_steps"]),
        -float(summary["mean_final_max_target_ratio"]),
        float(summary["mean_return"]),
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
