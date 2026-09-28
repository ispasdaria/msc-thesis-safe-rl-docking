"""Evaluate the retained final-approach SAC on LQR phase seeds 0--49.

This is evaluation-only: it loads the final 100k seed-12 policy and its
matching frozen VecNormalize statistics, executes deterministic actions, and
writes per-rollout, aggregate, and ranking artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.gym_env import Docking3DoFGymEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
MODEL_DIR = os.path.join(
    SCRIPT_DIR, "models", "retained_policies", "optuna12_100ksteps_buffer100k"
)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "results", "retained_policy_evaluations", "eval_lqr_phase_50",
    MODEL_NAME, "testing_seeds_0_49"
)
MAX_STEPS = 5000
DT_SECONDS = 1.0


def make_env(seed: int) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final", mode="testing", dt=DT_SECONDS, max_time=5_000.0,
        integrator="rk4", normalize_action=True, seed=seed,
    )


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return centre - radius, centre + radius


def rollout(model: SAC, vecnorm: VecNormalize, seed: int) -> dict:
    env = make_env(seed)
    obs, info = env.reset(seed=seed)
    final_info = info
    port_distances: list[float] = []
    saturation_steps = 0

    for step_index in range(MAX_STEPS):
        norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        normalized_action = np.asarray(action[0], dtype=np.float64)
        saturation_steps += int(np.any(np.abs(normalized_action) >= 1.0 - 1e-6))
        obs, _, terminated, truncated, info = env.step(normalized_action)
        final_info = info
        port_distances.append(float(info["port_distance"]))
        if terminated or truncated:
            break

    state = np.asarray(env.core_env.state, dtype=np.float64)
    port_position = np.asarray(final_info["port_relative_position"], dtype=np.float64)
    port_velocity = np.asarray(final_info["port_relative_velocity"], dtype=np.float64)
    steps = step_index + 1
    result = {
        "seed": seed,
        "success": bool(final_info.get("success", False)),
        "timeout": bool(final_info.get("time_limit", False)),
        "hard_safety_termination": bool(
            final_info.get("failure", False)
            and final_info.get("termination_reason") == "safety_violation"
        ),
        "termination_reason": final_info.get("termination_reason"),
        "violated_constraints": ";".join(final_info.get("violated_constraints", [])),
        "episode_steps": steps,
        "elapsed_time_s": steps * DT_SECONDS,
        "final_port_distance_m": float(final_info["port_distance"]),
        "min_port_distance_m": float(np.min(port_distances)),
        "final_port_speed_mps": float(final_info["port_speed"]),
        "final_axial_position_error_m": float(port_position[0]),
        "final_lateral_position_error_m": float(port_position[1]),
        "final_axial_velocity_error_mps": float(port_velocity[0]),
        "final_lateral_velocity_error_mps": float(port_velocity[1]),
        "final_attitude_error_deg": float(np.rad2deg(final_info["psi_e"])),
        "final_angular_rate_error_deg_s": float(np.rad2deg(final_info["omega"])),
        "action_saturation_step_fraction": saturation_steps / steps,
        "final_com_rx_m": float(state[0]),
        "final_com_ry_m": float(state[1]),
    }
    env.close()
    return result


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std_sample": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    n_success = sum(row["success"] for row in results)
    low, high = wilson_interval(n_success, n)
    fields = {
        "episode_steps": "episode_steps",
        "elapsed_time_s": "elapsed_time_s",
        "final_port_distance_m": "final_port_distance_m",
        "min_port_distance_m": "min_port_distance_m",
        "final_port_speed_mps": "final_port_speed_mps",
        "abs_final_axial_position_error_m": "final_axial_position_error_m",
        "abs_final_lateral_position_error_m": "final_lateral_position_error_m",
        "abs_final_axial_velocity_error_mps": "final_axial_velocity_error_mps",
        "abs_final_lateral_velocity_error_mps": "final_lateral_velocity_error_mps",
        "abs_final_attitude_error_deg": "final_attitude_error_deg",
        "abs_final_angular_rate_error_deg_s": "final_angular_rate_error_deg_s",
        "action_saturation_step_fraction": "action_saturation_step_fraction",
    }
    distributions = {}
    for output_name, source_name in fields.items():
        values = [float(row[source_name]) for row in results]
        if output_name.startswith("abs_"):
            values = [abs(value) for value in values]
        distributions[output_name] = stats(values)

    return {
        "model_path": MODEL_PATH,
        "vecnormalize_path": VECNORM_PATH,
        "evaluation_seeds": list(range(50)),
        "deterministic": True,
        "environment": {
            "phase": "final", "mode": "testing", "dt_s": DT_SECONDS,
            "integrator": "rk4", "normalize_action": True, "max_steps": MAX_STEPS,
        },
        "n_rollouts": n,
        "n_success": n_success,
        "success_percentage": 100.0 * n_success / n,
        "success_wilson_95_low": low,
        "success_wilson_95_high": high,
        "n_timeouts": sum(row["timeout"] for row in results),
        "n_hard_safety_terminations": sum(row["hard_safety_termination"] for row in results),
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in results)),
        "distributions": distributions,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_flat_summary(path: str, summary: dict, ranking: dict) -> None:
    rows = [
        ("n_rollouts", summary["n_rollouts"]),
        ("n_success", summary["n_success"]),
        ("success_percentage", summary["success_percentage"]),
        ("success_wilson_95_low_percentage", 100 * summary["success_wilson_95_low"]),
        ("success_wilson_95_high_percentage", 100 * summary["success_wilson_95_high"]),
        ("n_timeouts", summary["n_timeouts"]),
        ("n_hard_safety_terminations", summary["n_hard_safety_terminations"]),
    ]
    for name, values in summary["distributions"].items():
        rows.extend((f"{name}_{key}", value) for key, value in values.items())
    rows.extend((name, value) for name, value in ranking.items())
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> None:
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECNORM_PATH):
        raise FileNotFoundError(f"Missing model/statistics: {MODEL_PATH} ; {VECNORM_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dummy_env = DummyVecEnv([lambda: Monitor(make_env(999))])
    vecnorm = VecNormalize.load(VECNORM_PATH, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(MODEL_PATH)

    results = []
    for seed in range(50):
        row = rollout(model, vecnorm, seed)
        results.append(row)
        print(f"seed={seed:02d} success={row['success']} steps={row['episode_steps']} "
              f"final_port={row['final_port_distance_m']:.6f} m")
    dummy_env.close()

    successful = sorted(
        (row for row in results if row["success"]),
        key=lambda row: (row["final_port_distance_m"], row["seed"]),
    )
    if successful:
        median = successful[(len(successful) - 1) // 2]
        ranking = {
            "representative_median_successful_seed": median["seed"],
            "representative_median_final_port_distance_m": median["final_port_distance_m"],
            "best_successful_seed": successful[0]["seed"],
            "best_successful_final_port_distance_m": successful[0]["final_port_distance_m"],
            "worst_successful_seed": successful[-1]["seed"],
            "worst_successful_final_port_distance_m": successful[-1]["final_port_distance_m"],
        }
    else:
        ranking = {"representative_median_successful_seed": None,
                   "best_successful_seed": None, "worst_successful_seed": None}

    summary = aggregate(results)
    summary["successful_seed_ranking"] = ranking
    write_csv(os.path.join(OUTPUT_DIR, "rollout_results.csv"), results)
    with open(os.path.join(OUTPUT_DIR, "aggregate_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    write_flat_summary(os.path.join(OUTPUT_DIR, "aggregate_summary.csv"), summary, ranking)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
