"""Evaluate the selected final-approach SAC checkpoints on seeds 0-49.

Evaluation only. Every checkpoint is loaded with the VecNormalize file sharing
its exact basename. Results are written to a dedicated reconstruction folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from envs.docking3dof.gym_env import Docking3DoFGymEnv


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR / "models"
OUTPUT_DIR = (
    SCRIPT_DIR / "results" / "retained_policy_evaluations"
    / "reconstructed_selected_policies_50rollouts"
)
MAX_STEPS = 5_000
DT_SECONDS = 1.0

SPECS = [
    (1, 12, 100_000, 100_000, False),
    (2, 3, 300_000, 100_000, True),
    (2, 9, 300_000, 100_000, True),
    (2, 15, 300_000, 100_000, True),
    (3, 6, 300_000, 300_000, True),
    (3, 12, 300_000, 300_000, True),
    (4, 5, 600_000, 100_000, True),
    (4, 10, 600_000, 100_000, True),
    (4, 18, 600_000, 100_000, True),
    (5, 11, 600_000, 600_000, False),
    (5, 3, 600_000, 600_000, True),
    (5, 15, 600_000, 600_000, True),
    (5, 16, 600_000, 600_000, True),
    (5, 19, 600_000, 600_000, True),
]

SUPPLEMENTAL_SPECS = [
    (2, 8, 300_000, 100_000, True),
    (3, 1, 300_000, 300_000, True),
    (3, 16, 300_000, 300_000, True),
    (3, 19, 300_000, 300_000, True),
    (5, 1, 600_000, 600_000, True),
    (5, 10, 600_000, 600_000, True),
    (5, 13, 600_000, 600_000, True),
    (5, 14, 600_000, 600_000, True),
]


def names(seed: int, budget: int, buffer_size: int, best: bool) -> tuple[str, str]:
    group = f"optuna12_{budget // 1000}ksteps_buffer{buffer_size // 1000}k"
    stem = f"sac_final_approach_portobs_rewardv9_{budget // 1000}ksteps_seed{seed}_optuna12"
    if budget != 100_000:
        stem += f"_buffer{buffer_size // 1000}k"
    if best:
        stem += "_best"
    return group, stem


def model_group_directory(group: str) -> Path:
    """Resolve a campaign from its retained or development category."""

    category = (
        "retained_policies"
        if group in {"optuna12_100ksteps_buffer100k", "optuna12_600ksteps_buffer600k"}
        else "development_experiments"
    )
    return MODEL_ROOT / category / group


def make_env(seed: int) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final", mode="testing", dt=DT_SECONDS, max_time=5_000.0,
        integrator="rk4", normalize_action=True, seed=seed,
    )


def evaluate_one(model: SAC, vecnorm: VecNormalize, eval_seed: int) -> dict:
    env = make_env(eval_seed)
    obs, info = env.reset(seed=eval_seed)
    final_info = info
    saturation_steps = 0
    port_distances = []

    for step_index in range(MAX_STEPS):
        normalized_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(normalized_obs, deterministic=True)
        normalized_action = np.asarray(action[0], dtype=np.float64)
        saturation_steps += int(np.any(np.abs(normalized_action) >= 1.0 - 1e-6))
        obs, _, terminated, truncated, info = env.step(normalized_action)
        final_info = info
        port_distances.append(float(info["port_distance"]))
        if terminated or truncated:
            break

    steps = step_index + 1
    row = {
        "eval_seed": eval_seed,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)),
        "termination_reason": final_info.get("termination_reason"),
        "violated_constraints": ";".join(final_info.get("violated_constraints", [])),
        "episode_steps": steps,
        "elapsed_time_s": steps * DT_SECONDS,
        "terminal_port_position_error_m": float(final_info["port_distance"]),
        "max_terminal_port_position_error_helper_m": float(np.max(port_distances)),
        "terminal_port_velocity_error_mps": float(final_info["port_speed"]),
        "terminal_attitude_error_deg": abs(float(np.rad2deg(final_info["psi_e"]))),
        "terminal_angular_rate_error_deg_s": abs(float(np.rad2deg(final_info["omega"]))),
        "action_saturation_fraction": saturation_steps / steps,
    }
    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    specs = SUPPLEMENTAL_SPECS if args.supplemental else SPECS
    output_dir = (
        OUTPUT_DIR.parent / "reconstructed_supplemental_policies_50rollouts"
        if args.supplemental else OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []

    for index, (round_number, seed, budget, buffer_size, best) in enumerate(specs, 1):
        group, stem = names(seed, budget, buffer_size, best)
        group_directory = model_group_directory(group)
        model_path = group_directory / f"{stem}.zip"
        vecnorm_path = group_directory / f"{stem}_vecnormalize.pkl"
        if not model_path.is_file() or not vecnorm_path.is_file():
            raise FileNotFoundError(f"Missing exact model pair: {model_path}; {vecnorm_path}")

        dummy_env = DummyVecEnv([lambda: Monitor(make_env(999))])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        model = SAC.load(str(model_path))

        model_rows = []
        print(f"[{index}/{len(specs)}] {stem}", flush=True)
        for eval_seed in range(50):
            row = evaluate_one(model, vecnorm, eval_seed)
            row.update({
                "round": round_number,
                "training_seed": seed,
                "training_budget_steps": budget,
                "replay_buffer_size": buffer_size,
                "policy_type": "Intermediate checkpoint" if best else "Final policy",
                "model_name": stem,
                "model_path": str(model_path),
                "vecnormalize_path": str(vecnorm_path),
            })
            model_rows.append(row)
            all_rows.append(row)
        dummy_env.close()

        def mean(field: str) -> float:
            return float(np.mean([float(row[field]) for row in model_rows]))

        summary = {
            "round": round_number,
            "training_seed": seed,
            "training_budget_steps": budget,
            "replay_buffer_size": buffer_size,
            "policy_type": "Intermediate checkpoint" if best else "Final policy",
            "model_name": stem,
            "model_path": str(model_path),
            "vecnormalize_path": str(vecnorm_path),
            "n_evaluation_rollouts": len(model_rows),
            "success_rate_percent": 100.0 * mean("success"),
            "mean_episode_duration_s": mean("elapsed_time_s"),
            "mean_action_saturation_fraction_percent": 100.0 * mean("action_saturation_fraction"),
            "mean_terminal_docking_port_position_error_m": mean("terminal_port_position_error_m"),
            "maximum_terminal_docking_port_position_error_m": float(max(row["terminal_port_position_error_m"] for row in model_rows)),
            "mean_terminal_docking_port_velocity_error_mps": mean("terminal_port_velocity_error_mps"),
            "mean_terminal_attitude_error_deg": mean("terminal_attitude_error_deg"),
            "mean_terminal_angular_rate_error_deg_s": mean("terminal_angular_rate_error_deg_s"),
        }
        summaries.append(summary)
        print(f"  success={summary['success_rate_percent']:.1f}%", flush=True)

    with (output_dir / "all_rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)

    print(f"Saved {len(all_rows)} rollouts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
