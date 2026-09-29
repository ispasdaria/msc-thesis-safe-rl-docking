"""Run retained LQR/SAC full-mission evaluations, optionally with closing CBF.

Seed ranges are inclusive. Results for each range are written to a separate
directory, preserving earlier campaign blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
LQR_ROOT = SRC_ROOT / "controllers" / "LQR-3DoF"
CLOSING_CODE_ROOT = SRC_ROOT / "controllers" / "SAC-3DoF" / "closing"
CBF_ROOT = SRC_ROOT / "controllers" / "CBF-3DoF"
for path in (SRC_ROOT, LQR_ROOT, CLOSING_CODE_ROOT, CBF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from cbf_3dof.closing_barriers import ClosingCBFConfig
from cbf_3dof.qp_filter import ClosingCBFFilter
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


SAC_ROOT = SRC_ROOT / "controllers" / "SAC-3DoF"
FINAL_ROOT = SAC_ROOT / "final_approach" / "models" / "retained_policies"
CLOSING_ROOT = SAC_ROOT / "closing" / "models"
DEFAULT_PHASE_HORIZONS = {"closing": 12000.0, "final": 5000.0}

FINAL_100K_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
FINAL_100K_DIR = FINAL_ROOT / "optuna12_100ksteps_buffer100k"
FINAL_600K_NAME = "sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k"
FINAL_600K_DIR = FINAL_ROOT / "optuna12_600ksteps_buffer600k"

BALANCED_DIR = (
    CLOSING_ROOT / "multiseed"
    / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
    / "seed_0"
)
VANILLA_DIR = CLOSING_ROOT / "reward_shaping_trials"
VANILLA_NAME = "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k"


@dataclass(frozen=True)
class PolicyFiles:
    model: Path
    vecnorm: Path


@dataclass(frozen=True)
class MissionConfiguration:
    key: str
    label: str
    closing_kind: str
    closing_dt: float
    final_policy: PolicyFiles
    closing_policy: PolicyFiles | None = None
    closing_cbf: bool = False


def paired(directory: Path, stem: str) -> PolicyFiles:
    return PolicyFiles(directory / f"{stem}.zip", directory / f"{stem}_vecnormalize.pkl")


CONFIGURATIONS = {
    "lqr_final_r1_s12_100k": MissionConfiguration(
        "lqr_final_r1_s12_100k", "LQR dt=0.1 + final SAC round 1 seed 12 100k/buffer100k",
        "lqr", 0.1, paired(FINAL_100K_DIR, FINAL_100K_NAME),
    ),
    "lqr_final_r5_s11_600k": MissionConfiguration(
        "lqr_final_r5_s11_600k", "LQR dt=0.1 + final SAC round 5 seed 11 600k/buffer600k",
        "lqr", 0.1, paired(FINAL_600K_DIR, FINAL_600K_NAME),
    ),
    "balanced100k_final_r1_s12_100k": MissionConfiguration(
        "balanced100k_final_r1_s12_100k", "phase-balanced closing SAC 100k + final SAC round 1 seed 12 100k",
        "sac", 1.0, paired(FINAL_100K_DIR, FINAL_100K_NAME),
        paired(BALANCED_DIR, "model_0100000_steps"),
    ),
    "vanilla400k_final_r1_s12_100k": MissionConfiguration(
        "vanilla400k_final_r1_s12_100k", "vanilla closing SAC reward v15/trial 16 final 400k + final SAC round 1 seed 12 100k",
        "sac", 1.0, paired(FINAL_100K_DIR, FINAL_100K_NAME),
        paired(VANILLA_DIR, VANILLA_NAME),
    ),
    "vanilla400k_cbf_final_r1_s12_100k": MissionConfiguration(
        "vanilla400k_cbf_final_r1_s12_100k",
        "vanilla closing SAC reward v15/trial 16 final 400k + full closing CBF + final SAC round 1 seed 12 100k",
        "sac", 1.0, paired(FINAL_100K_DIR, FINAL_100K_NAME),
        paired(VANILLA_DIR, VANILLA_NAME), True,
    ),
}


class LoadedSAC:
    def __init__(self, files: PolicyFiles, phase: str, dt: float):
        self.files = files
        dummy = DummyVecEnv([
            lambda: Monitor(make_env(phase, 999_999, dt, DEFAULT_PHASE_HORIZONS[phase]))
        ])
        self.vecnorm = VecNormalize.load(str(files.vecnorm), dummy)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False
        self.model = SAC.load(str(files.model), device="auto")
        self.dummy = dummy

    def action(self, observation: np.ndarray) -> np.ndarray:
        normalized = self.vecnorm.normalize_obs(observation.reshape(1, -1))
        action, _ = self.model.predict(normalized, deterministic=True)
        return np.asarray(action[0], dtype=np.float64)

    def close(self) -> None:
        self.dummy.close()


def make_env(phase: str, seed: int, dt: float, max_time: float) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase=phase, mode="testing", dt=dt, max_time=max_time, integrator="rk4",
        normalize_action=True, seed=seed,
    )


def build_lqr(env: Docking3DoFGymEnv) -> LQRClosingController:
    core = env.core_env
    gain, _ = compute_lqr_gain(
        *build_linearized_3dof_matrices(core), build_Q_closing(dt=core.dt), build_R_closing(core)
    )
    return LQRClosingController(gain)


def run_phase(env, observation, action_function, max_steps, cbf_filter=None):
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    info = {}
    saturation_steps = 0
    intervention_steps = 0
    qp_failure_count = 0
    for step_index in range(max_steps):
        action = np.asarray(action_function(observation), dtype=np.float64)
        if cbf_filter is not None:
            nominal_physical = env.normalized_action_to_physical(action)
            filtered = cbf_filter.filter(env.get_physical_state(), nominal_physical)
            intervention_steps += int(filtered.intervened)
            qp_failure_count += int(filtered.qp_failed)
            action = env.physical_action_to_normalized(filtered.action_filtered)
        saturation_steps += int(np.any(np.abs(action) >= 1.0 - 1e-6))
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    steps = step_index + 1
    state = np.asarray(env.core_env.state, dtype=float).copy()
    q = env.core_env.get_derived_quantities(state)
    if env.phase == "closing":
        error = state - np.asarray(env.core_env.p3_target, dtype=float)
        error[4] = env.core_env.wrap_to_pi(error[4])
        position_error = error[:2]
        velocity_error = error[2:4]
        attitude_error = float(error[4])
        rate_error = float(error[5])
    else:
        position_error = np.asarray(q["r_p"], dtype=float)
        velocity_error = np.asarray(q["v_p"], dtype=float)
        attitude_error = float(q["psi_e"])
        rate_error = float(q["omega"])
    return {
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "time_limit": bool(info.get("time_limit", False)),
        "termination_reason": info.get("termination_reason", "runner_max_steps"),
        "steps": steps,
        "elapsed_time_s": steps * env.core_env.dt,
        "saturation_fraction": saturation_steps / steps,
        "cbf_intervention_count": intervention_steps,
        "cbf_intervention_fraction": intervention_steps / steps,
        "cbf_qp_failure_count": qp_failure_count,
        "terminal_state": state,
        "final_port_distance_m": float(info.get("port_distance", math.nan)),
        "final_port_speed_mps": float(info.get("port_speed", math.nan)),
        "axial_position_error_m": float(position_error[0]),
        "lateral_position_error_m": float(position_error[1]),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "axial_velocity_error_mps": float(velocity_error[0]),
        "lateral_velocity_error_mps": float(velocity_error[1]),
        "velocity_error_norm_mps": float(np.linalg.norm(velocity_error)),
        "attitude_error_deg": float(np.rad2deg(attitude_error)),
        "angular_rate_error_deg_s": float(np.rad2deg(rate_error)),
    }


def run_mission(config, seed, closing_sac, final_sac, closing_max_time, final_max_time):
    closing_env = make_env("closing", seed, config.closing_dt, closing_max_time)
    try:
        closing_obs, _ = closing_env.reset(seed=seed)
        closing_steps = math.ceil(closing_max_time / config.closing_dt)
        if config.closing_kind == "lqr":
            lqr = build_lqr(closing_env)

            def closing_action(_):
                physical = lqr.compute_action(closing_env.core_env, closing_env.core_env.state)
                return closing_env.physical_action_to_normalized(physical)
        else:
            closing_action = closing_sac.action
        closing_cbf = None
        if config.closing_cbf:
            closing_cbf = ClosingCBFFilter(
                closing_env.core_env,
                ClosingCBFConfig(allow_slack=False),
            )
        closing = run_phase(
            closing_env, closing_obs, closing_action, closing_steps, closing_cbf
        )
    finally:
        closing_env.close()

    final = None
    if closing["success"]:
        final_env = make_env("final", seed, 1.0, final_max_time)
        try:
            final_obs, _ = final_env.reset(
                seed=seed, options={"initial_state": closing["terminal_state"], "mode": "testing"}
            )
            final = run_phase(final_env, final_obs, final_sac.action, math.ceil(final_max_time))
        finally:
            final_env.close()

    return flatten_result(config, seed, closing, final)


def flatten_result(config, seed, closing, final):
    mission_success = bool(final and final["success"])
    row = {
        "configuration": config.key, "seed": seed, "mission_success": mission_success,
        "failed_phase": "" if mission_success else ("closing" if not closing["success"] else "final"),
        "mission_elapsed_time_s": closing["elapsed_time_s"] + (
            0.0 if final is None else final["elapsed_time_s"]
        ),
        "mission_steps": closing["steps"] + (0 if final is None else final["steps"]),
    }
    for prefix, result in (("closing", closing), ("final", final)):
        for field in ("success", "failure", "time_limit", "termination_reason", "steps",
                      "elapsed_time_s", "saturation_fraction", "final_port_distance_m",
                      "final_port_speed_mps", "axial_position_error_m",
                      "lateral_position_error_m", "position_error_norm_m",
                      "axial_velocity_error_mps", "lateral_velocity_error_mps",
                      "velocity_error_norm_mps", "attitude_error_deg",
                      "angular_rate_error_deg_s", "cbf_intervention_count",
                      "cbf_intervention_fraction", "cbf_qp_failure_count"):
            row[f"{prefix}_{field}"] = "" if result is None else result[field]
        for index, state_name in enumerate(("rx_m", "ry_m", "vx_mps", "vy_mps", "psi_deg", "omega_deg_s")):
            value = "" if result is None else float(result["terminal_state"][index])
            if result is not None and index >= 4:
                value = float(np.rad2deg(value))
            row[f"{prefix}_terminal_{state_name}"] = value
    return row


def distribution(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "std_sample": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std_sample": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summary(config, rows, seed_start, seed_end):
    count = len(rows)
    successes = sum(row["mission_success"] for row in rows)
    result = {
        "configuration": config.key, "label": config.label,
        "seed_start": seed_start, "seed_end": seed_end, "num_missions": count,
        "mission_successes": successes, "mission_success_rate": successes / count,
        "closing_successes": sum(row["closing_success"] for row in rows),
        "final_attempts": sum(row["final_success"] != "" for row in rows),
        "final_successes": sum(row["final_success"] is True for row in rows),
        "failed_phase_counts": dict(Counter(row["failed_phase"] or "none" for row in rows)),
        "closing_termination_counts": dict(Counter(row["closing_termination_reason"] for row in rows)),
        "final_termination_counts": dict(Counter(row["final_termination_reason"] for row in rows if row["final_termination_reason"])),
        "closing_dt_s": config.closing_dt, "final_dt_s": 1.0,
        "closing_cbf_enabled": config.closing_cbf,
        "closing_model": None if config.closing_policy is None else str(config.closing_policy.model),
        "closing_vecnormalize": None if config.closing_policy is None else str(config.closing_policy.vecnorm),
        "final_model": str(config.final_policy.model),
        "final_vecnormalize": str(config.final_policy.vecnorm),
    }
    distribution_fields = (
        "elapsed_time_s", "steps", "saturation_fraction", "final_port_distance_m",
        "final_port_speed_mps", "axial_position_error_m", "lateral_position_error_m",
        "position_error_norm_m", "axial_velocity_error_mps",
        "lateral_velocity_error_mps", "velocity_error_norm_mps",
        "attitude_error_deg", "angular_rate_error_deg_s",
        "cbf_intervention_count", "cbf_intervention_fraction",
        "cbf_qp_failure_count",
    )
    result["distributions"] = {}
    result["distributions"]["mission_elapsed_time_s"] = distribution(
        [row["mission_elapsed_time_s"] for row in rows]
    )
    result["distributions"]["mission_steps"] = distribution(
        [row["mission_steps"] for row in rows]
    )
    for phase in ("closing", "final"):
        for field in distribution_fields:
            key = f"{phase}_{field}"
            values = [row[key] for row in rows if row[key] != ""]
            # Signed components retain their signed distribution; absolute-error
            # distributions are also supplied for direct accuracy reporting.
            result["distributions"][key] = distribution(values)
            if "_error_" in field and not field.endswith("norm_m") and not field.endswith("norm_mps"):
                result["distributions"][f"{phase}_absolute_{field}"] = distribution(
                    [abs(float(value)) for value in values]
                )
    return result


def validate(configs):
    missing = []
    for config in configs:
        if config.closing_kind == "sac" and config.closing_policy is None:
            raise ValueError(f"SAC configuration {config.key} requires a closing policy")
        policies = [config.final_policy] + ([] if config.closing_policy is None else [config.closing_policy])
        missing.extend(str(path) for policy in policies for path in (policy.model, policy.vecnorm) if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing required policy artifacts:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True, help="Inclusive final seed.")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--output-root", type=Path, default=SAC_ROOT / "results" / "full_mission_mixed")
    parser.add_argument("--closing-max-time", type=float, default=12000.0)
    parser.add_argument("--final-max-time", type=float, default=5000.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_end < args.seed_start:
        parser.error("Require 0 <= --seed-start <= --seed-end")
    if not math.isfinite(args.closing_max_time) or args.closing_max_time <= 0:
        parser.error("--closing-max-time must be finite and positive")
    if not math.isfinite(args.final_max_time) or args.final_max_time <= 0:
        parser.error("--final-max-time must be finite and positive")
    configs = [CONFIGURATIONS[key] for key in args.configs]
    validate(configs)
    print("Validated configurations:")
    for config in configs:
        print(f"  {config.key}: {config.label}")
    if args.check_only:
        return

    output_dir = args.output_root / f"seeds_{args.seed_start}_{args.seed_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in configs:
        csv_path = output_dir / f"{config.key}.csv"
        json_path = output_dir / f"{config.key}_summary.json"
        if (csv_path.exists() or json_path.exists()) and not args.overwrite:
            raise FileExistsError(f"Results already exist for {config.key}; pass --overwrite to replace them")
        final_sac = LoadedSAC(config.final_policy, "final", 1.0)
        closing_sac = LoadedSAC(config.closing_policy, "closing", config.closing_dt) if config.closing_policy else None
        rows = []
        try:
            for seed in range(args.seed_start, args.seed_end + 1):
                row = run_mission(config, seed, closing_sac, final_sac, args.closing_max_time, args.final_max_time)
                rows.append(row)
                print(f"{config.key} seed={seed} success={row['mission_success']} failed_phase={row['failed_phase'] or '-'}")
        finally:
            final_sac.close()
            if closing_sac:
                closing_sac.close()
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result_summary = summary(config, rows, args.seed_start, args.seed_end)
        json_path.write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
        summaries.append(result_summary)
        print(f"wrote {csv_path}")
    (output_dir / "campaign_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
