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


HER_ENV_CLASSES = {
    "v1": GoalConditionedClosingEnv,
    "v2": GoalConditionedClosingV2Env,
    "v3": GoalConditionedClosingV3Env,
    "v4": ResidualV15GoalConditionedClosingEnv,
}


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
    return {
        "position_error": float(np.linalg.norm(error[:2])),
        "lateral_error": float(abs(error[1])),
        "velocity_error": float(np.linalg.norm(error[2:4])),
        "attitude_error_deg": float(np.rad2deg(abs(error[4]))),
        "rate_error_degs": float(np.rad2deg(abs(error[5]))),
        "max_target_ratio": float(np.max(ratios)),
    }


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
        minimum_position_error = initial["position_error"]
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
            best_ratio = min(best_ratio, metrics["max_target_ratio"])
            minimum_position_error = min(
                minimum_position_error,
                metrics["position_error"],
            )
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
        rows.append(
            {
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
        )
        env.close()
    return rows


def summarize_rollouts(rows):
    if not rows:
        raise ValueError("At least one rollout is required.")

    def mean(field):
        return float(np.mean([float(row[field]) for row in rows]))

    return {
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
