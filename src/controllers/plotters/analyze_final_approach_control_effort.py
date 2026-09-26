"""Paired final-approach effort analysis using final applied physical commands.

The retained full-CBF seed-0--49 campaign is reused. With
``--generate-missing``, only absent LQR and retained-SAC canonical histories
are evaluated.  ``--analyze`` validates all histories and writes thesis-ready
tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROLLERS_ROOT = HERE.parent
REPO_ROOT = CONTROLLERS_ROOT.parents[1]
SRC = REPO_ROOT / "src"
LQR_DIR = CONTROLLERS_ROOT / "LQR-3DoF"
LQR_TUNING_DIR = LQR_DIR / "tuning_lqr_final_approach"
SAC_FINAL_DIR = CONTROLLERS_ROOT / "SAC-3DoF" / "final_approach"
CBF_DIR = CONTROLLERS_ROOT / "cbf_3DoF"
for import_path in (SRC, LQR_DIR, LQR_TUNING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

SEEDS = tuple(range(50))
DT_SECONDS = 1.0
ACTION_LIMITS = np.array([0.01, 0.01, 0.003], dtype=np.float64)
CONTROLLERS = (
    "LQR",
    "SAC R1-S12-final100k",
    "SAC+CBF R5-S10-best490k",
)
RETAINED_STEM = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
RETAINED_MODEL_DIR = SAC_FINAL_DIR / "models" / "optuna12_100ksteps_buffer100k"
RETAINED_MODEL = RETAINED_MODEL_DIR / f"{RETAINED_STEM}.zip"
RETAINED_VECNORM = RETAINED_MODEL_DIR / f"{RETAINED_STEM}_vecnormalize.pkl"
CBF_CAMPAIGN = (
    CBF_DIR
    / "results/final_approach/evaluations"
    / "campaign_round5_seed10_best490k_seeds_0_49"
)
DEFAULT_OUTPUT = CONTROLLERS_ROOT / "results" / "final_approach_control_effort"
LEGACY_LQR = LQR_DIR / "results/lqr_final/rollout_histories"
LQR_PROVENANCE = "selected-final-lqr:q1-r1-base-tuning"
SAC_PROVENANCE = f"model={RETAINED_MODEL.resolve()}|vecnormalize={RETAINED_VECNORM.resolve()}"


def _as_float_array(value, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def compute_rollout_metrics(
    times: np.ndarray,
    port_positions: np.ndarray,
    applied_actions: np.ndarray,
    action_limits: np.ndarray,
    saturation_atol: float = 1.0e-10,
) -> dict[str, float | int]:
    """Compute physical effort, geometric path, and control variation."""
    times = np.asarray(times, dtype=np.float64)
    positions = _as_float_array(port_positions, (2,), "port_positions")
    actions = _as_float_array(applied_actions, (3,), "applied_actions")
    limits = np.asarray(action_limits, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array.")
    if times.size != positions.shape[0]:
        raise ValueError("times and port_positions must have the same sample count.")
    if positions.shape[0] != actions.shape[0] + 1:
        raise ValueError("State history must contain one more sample than applied actions.")
    if actions.shape[0] == 0:
        raise ValueError("A rollout must contain at least one applied action.")
    if limits.shape != (3,) or np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("action_limits must be three positive finite values.")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if np.any(np.abs(actions) > limits + saturation_atol):
        raise ValueError("Applied command exceeds the physical actuator limits.")

    path_segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(np.sum(path_segments))
    direct_distance = float(np.linalg.norm(positions[-1] - positions[0]))
    if path_length == 0.0:
        path_efficiency = 1.0 if direct_distance == 0.0 else math.nan
    else:
        path_efficiency = direct_distance / path_length
    action_changes = np.diff(actions, axis=0)
    saturated = np.any(np.isclose(np.abs(actions), limits, rtol=0.0, atol=saturation_atol), axis=1)
    result = {
        "num_steps": int(actions.shape[0]),
        "duration_s": float(times[-1] - times[0]),
        "delta_v_eq_mps": float(np.sum(np.linalg.norm(actions[:, :2], axis=1) * dt)),
        "J_alpha_radps": float(np.sum(np.abs(actions[:, 2]) * dt)),
        "saturation_fraction": float(np.mean(saturated)),
        "saturation_steps": int(np.sum(saturated)),
        "path_length_m": path_length,
        "path_efficiency": float(path_efficiency),
        "J_delta_a_mps2": float(np.sum(np.linalg.norm(action_changes[:, :2], axis=1))),
        "J_delta_alpha_radps2": float(np.sum(np.abs(action_changes[:, 2]))),
    }
    if any(not np.isfinite(float(value)) for key, value in result.items() if key != "path_efficiency"):
        raise ValueError("Computed rollout metrics must be finite.")
    if result["delta_v_eq_mps"] < 0.0 or result["J_alpha_radps"] < 0.0:
        raise ValueError("Integrated effort metrics must be nonnegative.")
    return result


def distribution_stats(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Statistics require a nonempty finite one-dimensional sample.")
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0], method="linear")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def json_safe(value):
    """Recursively convert NumPy values to standard JSON-compatible types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def paired_stats(lqr_values, method_values, tie_atol: float = 1.0e-12) -> dict[str, float | int]:
    lqr = np.asarray(lqr_values, dtype=np.float64)
    method = np.asarray(method_values, dtype=np.float64)
    if lqr.shape != method.shape or lqr.ndim != 1 or lqr.size == 0:
        raise ValueError("Paired samples must be nonempty one-dimensional arrays of equal shape.")
    difference = method - lqr
    lower = difference < -tie_atol
    higher = difference > tie_atol
    ties = ~(lower | higher)
    lqr_mean = float(np.mean(lqr))
    return {
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "std_difference": float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0,
        "lower_count": int(np.sum(lower)),
        "lower_percent": float(100.0 * np.mean(lower)),
        "higher_count": int(np.sum(higher)),
        "higher_percent": float(100.0 * np.mean(higher)),
        "tie_count": int(np.sum(ties)),
        "change_vs_lqr_percent": (
            float(100.0 * (np.mean(method) - lqr_mean) / lqr_mean)
            if lqr_mean != 0.0 else math.nan
        ),
    }


def _port_positions(states: np.ndarray) -> np.ndarray:
    from envs.docking3dof.core import Docking3DoFEnv

    env = Docking3DoFEnv(phase="final", dt=DT_SECONDS, max_time=5000.0, seed=0)
    return np.asarray([env.get_port_relative_position(state) for state in states], dtype=np.float64)


def _save_history(path: Path, *, seed: int, times, states, actions, success, reason,
                  controller_key: str, source: str, provenance_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = _port_positions(np.asarray(states, dtype=np.float64))
    applied = np.asarray(actions, dtype=np.float64)
    np.savez_compressed(
        path,
        evaluation_seed=int(seed),
        time=np.asarray(times, dtype=np.float64),
        port_position=positions,
        r_p_x=positions[:, 0],
        r_p_y=positions[:, 1],
        applied_action_body=applied,
        a_x_Bc_applied=applied[:, 0],
        a_y_Bc_applied=applied[:, 1],
        alpha_z_Bc_applied=applied[:, 2],
        success=bool(success),
        termination_reason=str(reason),
        dt_seconds=DT_SECONDS,
        action_limits=ACTION_LIMITS,
        controller_key=controller_key,
        source=source,
        provenance_id=provenance_id,
    )


def _canonical_path(output_dir: Path, controller_key: str, seed: int) -> Path:
    return output_dir / "histories" / controller_key / f"seed_{seed:04d}.npz"


def _import_legacy_histories(output_dir: Path) -> set[tuple[str, int]]:
    imported: set[tuple[str, int]] = set()
    # Older detailed SAC files do not carry a sufficiently strong record of
    # their model/VecNormalize pair, so they are deliberately not imported.
    for path in sorted(LEGACY_LQR.glob("q1_r1_seed*.npz")):
        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata_json"]))
            seed = int(metadata["seed"])
            destination = _canonical_path(output_dir, "lqr", seed)
            if destination.is_file():
                continue
            if metadata.get("q_scale") != 1.0 or metadata.get("r_scale") != 1.0 \
                    or metadata.get("mode") != "testing" or metadata.get("dt") != DT_SECONDS:
                continue
            times = np.asarray(data["times"], dtype=np.float64)
            states = np.asarray(data["states"], dtype=np.float64)
            actions = np.asarray(data["actions_body_clipped"], dtype=np.float64)
            if actions.shape[0] == states.shape[0]:
                actions = actions[1:]
            _save_history(
                destination, seed=seed, times=times,
                states=states, actions=actions, success=metadata["success"],
                reason=metadata["termination_reason"], controller_key="lqr",
                source=str(path.resolve()), provenance_id=LQR_PROVENANCE,
            )
            imported.add(("lqr", seed))
    return imported


def _build_lqr():
    from envs.docking3dof.core import Docking3DoFEnv
    from lqr_3dof_final import (
        LQRFinalController, build_linearized_3dof_matrices,
        build_port_error_mapping, build_Q_final, build_R_final, compute_lqr_gain,
    )

    env = Docking3DoFEnv(phase="final", dt=DT_SECONDS, max_time=5000.0, seed=0)
    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    K, _ = compute_lqr_gain(A, B, H.T @ build_Q_final() @ H, build_R_final(env))
    return LQRFinalController(K, H)


def _collect_lqr(seed: int, path: Path, controller) -> None:
    from envs.docking3dof.core import Docking3DoFEnv

    env = Docking3DoFEnv(phase="final", dt=DT_SECONDS, max_time=5000.0,
                         integrator="rk4", seed=seed)
    state = env.reset(mode="testing")
    info: dict = {}
    for _ in range(5000):
        state, done, info = env.step(controller.compute_action(env, state))
        if done:
            break
    times, states, _, clipped, _ = env.get_history_arrays()
    _save_history(path, seed=seed, times=times, states=states, actions=clipped[1:],
                  success=info.get("success", False),
                  reason=info.get("termination_reason", "evaluation_step_limit"),
                  controller_key="lqr", source="fresh exact LQR evaluation",
                  provenance_id=LQR_PROVENANCE)


def _collect_sac(seed: int, path: Path, model, vecnormalize) -> None:
    from envs.docking3dof.gym_env import Docking3DoFGymEnv

    env = Docking3DoFGymEnv(phase="final", mode="testing", dt=DT_SECONDS,
                            max_time=5000.0, integrator="rk4",
                            normalize_action=True, seed=seed)
    try:
        obs, info = env.reset(seed=seed)
        for _ in range(5000):
            normalized_obs = vecnormalize.normalize_obs(obs.reshape(1, -1))
            action, _ = model.predict(normalized_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(np.asarray(action[0]))
            if terminated or truncated:
                break
        times, states, _, clipped, _ = env.core_env.get_history_arrays()
        _save_history(path, seed=seed, times=times, states=states, actions=clipped[1:],
                      success=info.get("success", False),
                      reason=info.get("termination_reason", "evaluation_step_limit"),
                      controller_key="sac",
                      source=f"fresh deterministic evaluation: {RETAINED_MODEL}; {RETAINED_VECNORM}",
                      provenance_id=SAC_PROVENANCE)
    finally:
        env.close()


def generate_missing(output_dir: Path) -> None:
    imported = _import_legacy_histories(output_dir)
    missing_lqr = [seed for seed in SEEDS if not _canonical_path(output_dir, "lqr", seed).is_file()]
    missing_sac = [seed for seed in SEEDS if not _canonical_path(output_dir, "sac", seed).is_file()]
    print(f"Imported {len(imported)} compatible legacy histories; "
          f"missing LQR={len(missing_lqr)}, SAC={len(missing_sac)}.")
    if missing_lqr:
        controller = _build_lqr()
        for index, seed in enumerate(missing_lqr, 1):
            _collect_lqr(seed, _canonical_path(output_dir, "lqr", seed), controller)
            print(f"LQR {index}/{len(missing_lqr)} seed={seed}", flush=True)
    if missing_sac:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from envs.docking3dof.gym_env import Docking3DoFGymEnv

        if not RETAINED_MODEL.is_file() or not RETAINED_VECNORM.is_file():
            raise FileNotFoundError(f"Missing exact SAC pair: {RETAINED_MODEL}; {RETAINED_VECNORM}")
        dummy = DummyVecEnv([lambda: Monitor(Docking3DoFGymEnv(
            phase="final", mode="testing", dt=DT_SECONDS, max_time=5000.0,
            integrator="rk4", normalize_action=True, seed=999_999))])
        vecnormalize = VecNormalize.load(str(RETAINED_VECNORM), dummy)
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        model = SAC.load(str(RETAINED_MODEL))
        try:
            for index, seed in enumerate(missing_sac, 1):
                _collect_sac(seed, _canonical_path(output_dir, "sac", seed), model, vecnormalize)
                print(f"SAC {index}/{len(missing_sac)} seed={seed}", flush=True)
        finally:
            dummy.close()


def audit_sources(output_dir: Path) -> dict:
    cbf_files = sorted(CBF_CAMPAIGN.glob("step_timeseries_all_cbf_eval_seed*.npz"))
    config_path = CBF_CAMPAIGN / "run_configuration.json"
    episode_path = CBF_CAMPAIGN / "per_episode_metrics.csv"
    if not config_path.is_file() or not episode_path.is_file():
        raise FileNotFoundError("The retained SAC+CBF campaign metadata is incomplete.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_cbf_model = "seed10_optuna12_buffer600k_best.zip"
    expected_vecnorm = "seed10_optuna12_buffer600k_best_vecnormalize.pkl"
    cbf_file_validations = []
    required_fields = {"step", "time", "state", "action_nominal", "action_filtered", "intervened"}
    for seed in SEEDS:
        path = CBF_CAMPAIGN / f"step_timeseries_all_cbf_eval_seed{seed:04d}.npz"
        valid = path.is_file()
        if valid:
            with np.load(path, allow_pickle=True) as data:
                valid = required_fields.issubset(data.files)
                if valid:
                    count = len(data["step"])
                    filtered = np.asarray(data["action_filtered"], dtype=np.float64)
                    valid = (
                        np.array_equal(np.asarray(data["step"]), np.arange(count))
                        and filtered.shape == (count, 3)
                        and np.asarray(data["state"]).shape == (count, 6)
                        and np.asarray(data["action_nominal"]).shape == (count, 3)
                        and np.asarray(data["intervened"]).shape == (count,)
                        and np.asarray(data["time"]).shape == (count,)
                        and np.allclose(np.diff(np.asarray(data["time"], dtype=float)), DT_SECONDS)
                        and np.all(np.isfinite(filtered))
                        and np.all(np.abs(filtered) <= ACTION_LIMITS + 1.0e-10)
                    )
        cbf_file_validations.append(bool(valid))
    checks = {
        "cbf_history_count": len(cbf_files),
        "cbf_seeds_complete": len(cbf_files) == 50,
        "cbf_model_is_retained_best": str(config["model"]).endswith(expected_cbf_model),
        "cbf_vecnormalize_is_matching_retained_best": str(config["vecnormalize"]).endswith(expected_vecnorm),
        "cbf_full_filter": config["variants"] == ["all"]
        and all(config["cbf"][key] for key in ("enable_cone", "enable_speed", "enable_alignment"))
        and config["cbf"]["allow_slack"] is False,
        "cbf_testing_seeds_0_49": config["seed_start"] == 0
        and config["num_seeds"] == 50 and config["mode"] == "testing",
        "lqr_history_count": sum(_canonical_path(output_dir, "lqr", s).is_file() for s in SEEDS),
        "sac_history_count": sum(_canonical_path(output_dir, "sac", s).is_file() for s in SEEDS),
        "dt_seconds": DT_SECONDS,
        "action_limits": ACTION_LIMITS.tolist(),
        "all_cbf_history_schemas_times_and_bounds_valid": all(cbf_file_validations),
        "cbf_applied_command_equivalence": (
            "action_filtered is already within physical bounds; "
            "physical_action_to_normalized followed by normalized_action_to_physical "
            "and core clipping therefore leaves it unchanged"
        ),
    }
    if cbf_files:
        with np.load(cbf_files[0]) as sample:
            checks["cbf_filtered_command_field"] = "action_filtered" in sample.files
            checks["cbf_nominal_command_field"] = "action_nominal" in sample.files
    return checks


def _load_canonical(path: Path, expected_seed: int, expected_controller: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float]:
    with np.load(path, allow_pickle=True) as data:
        seed = int(data["evaluation_seed"])
        dt = float(data["dt_seconds"])
        limits = np.asarray(data["action_limits"], dtype=np.float64)
        controller = str(data["controller_key"])
        provenance = str(data["provenance_id"])
        times = np.asarray(data["time"], dtype=np.float64)
        positions = np.asarray(data["port_position"], dtype=np.float64)
        actions = np.asarray(data["applied_action_body"], dtype=np.float64)
        if seed != expected_seed or controller != expected_controller:
            raise ValueError(f"Canonical history identity mismatch: {path}")
        expected_provenance = LQR_PROVENANCE if expected_controller == "lqr" else SAC_PROVENANCE
        if provenance != expected_provenance:
            raise ValueError(f"Canonical history controller provenance mismatch: {path}")
        if dt != DT_SECONDS or not np.array_equal(limits, ACTION_LIMITS):
            raise ValueError(f"Canonical history environment metadata mismatch: {path}")
        if times.size != actions.shape[0] + 1 or positions.shape != (times.size, 2):
            raise ValueError(f"Canonical history state/action alignment mismatch: {path}")
        if not np.allclose(np.diff(times), DT_SECONDS, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Canonical history timestep mismatch: {path}")
        return (
            times, positions, actions,
            bool(data["success"]), str(data["termination_reason"]), math.nan,
        )


def _load_cbf(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float]:
    from envs.docking3dof.core import Docking3DoFEnv

    path = CBF_CAMPAIGN / f"step_timeseries_all_cbf_eval_seed{seed:04d}.npz"
    with np.load(path, allow_pickle=True) as data:
        states = np.asarray(data["state"], dtype=np.float64)
        filtered = np.asarray(data["action_filtered"], dtype=np.float64)
        actions = np.clip(filtered, -ACTION_LIMITS, ACTION_LIMITS)
        if not np.allclose(actions, filtered, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"CBF filtered commands require unexpected actuator clipping: {path}")
        times_after = np.asarray(data["time"], dtype=np.float64)
        interventions = np.asarray(data["intervened"], dtype=bool)
        # Stored states are start-of-step. Reconstruct the single terminal state
        # with the same core RK4 step; this does not rerun a controller or CBF.
        env = Docking3DoFEnv(phase="final", dt=DT_SECONDS, max_time=5000.0,
                             integrator="rk4", seed=seed)
        terminal_state = env.integrate(states[-1], actions[-1])
        all_states = np.vstack((states, terminal_state))
        times = np.concatenate(([times_after[0] - DT_SECONDS], times_after))
    with (CBF_CAMPAIGN / "per_episode_metrics.csv").open(newline="", encoding="utf-8") as stream:
        episode = next(row for row in csv.DictReader(stream)
                       if int(row["eval_seed"]) == seed and row["variant"] == "all")
    return times, _port_positions(all_states), actions, episode["success"] == "True", \
        episode["termination_reason"], float(np.mean(interventions))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict], primary_seeds: set[int]) -> list[dict]:
    output = []
    metric_names = (
        "duration_s", "delta_v_eq_mps", "J_alpha_radps", "path_length_m",
        "path_efficiency", "J_delta_a_mps2", "J_delta_alpha_radps2",
    )
    for controller in CONTROLLERS:
        all_rows = [row for row in rows if row["controller"] == controller]
        selected = [row for row in all_rows if row["evaluation_seed"] in primary_seeds]
        stats = {name: distribution_stats([row[name] for row in selected]) for name in metric_names}
        total_steps = sum(row["num_steps"] for row in selected)
        summary = {
            "Controller": controller,
            "Success": f"{sum(row['success'] for row in all_rows)}/50",
            "Primary N": len(selected),
            "Mean duration [s]": stats["duration_s"]["mean"],
            "Std duration [s]": stats["duration_s"]["std"],
            "Median duration [s]": stats["duration_s"]["median"],
            "Mean Delta_v_eq [m/s]": stats["delta_v_eq_mps"]["mean"],
            "Std Delta_v_eq [m/s]": stats["delta_v_eq_mps"]["std"],
            "Median Delta_v_eq [m/s]": stats["delta_v_eq_mps"]["median"],
            "IQR Delta_v_eq [m/s]": stats["delta_v_eq_mps"]["iqr"],
            "Mean J_alpha [rad/s]": stats["J_alpha_radps"]["mean"],
            "Std J_alpha [rad/s]": stats["J_alpha_radps"]["std"],
            "Median J_alpha [rad/s]": stats["J_alpha_radps"]["median"],
            "IQR J_alpha [rad/s]": stats["J_alpha_radps"]["iqr"],
            "Saturation [%]": 100.0 * sum(row["saturation_steps"] for row in selected) / total_steps,
        }
        for name, values in stats.items():
            for stat_name, value in values.items():
                summary[f"{name}_{stat_name}"] = value
        output.append(summary)
    return output


def _paired_rows(rows: list[dict], primary_seeds: set[int]) -> list[dict]:
    lookup = {(row["controller"], row["evaluation_seed"]): row for row in rows}
    metrics = ("delta_v_eq_mps", "J_alpha_radps", "duration_s", "path_length_m",
               "path_efficiency", "J_delta_a_mps2", "J_delta_alpha_radps2")
    output = []
    seeds = sorted(primary_seeds)
    for method in CONTROLLERS[1:]:
        result = {"Comparison": f"{method} - LQR", "Paired N": len(seeds)}
        for metric in metrics:
            lqr = np.array([lookup[("LQR", seed)][metric] for seed in seeds])
            candidate = np.array([lookup[(method, seed)][metric] for seed in seeds])
            for key, value in paired_stats(lqr, candidate).items():
                result[f"{metric}_{key}"] = value
        result.update({
            "Mean Delta-v difference [m/s]": result["delta_v_eq_mps_mean_difference"],
            "Delta-v change vs LQR [%]": result["delta_v_eq_mps_change_vs_lqr_percent"],
            "Seeds with lower Delta-v than LQR": result["delta_v_eq_mps_lower_count"],
            "Mean J_alpha difference [rad/s]": result["J_alpha_radps_mean_difference"],
            "J_alpha change vs LQR [%]": result["J_alpha_radps_change_vs_lqr_percent"],
            "Seeds with lower J_alpha than LQR": result["J_alpha_radps_lower_count"],
            "Duration change vs LQR [%]": result["duration_s_change_vs_lqr_percent"],
        })
        output.append(result)
    return output


def _plot_paired(rows: list[dict], primary_seeds: set[int], metric: str,
                 ylabel: str, basename: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lookup = {(row["controller"], row["evaluation_seed"]): row for row in rows}
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for seed in sorted(primary_seeds):
        values = [lookup[(controller, seed)][metric] for controller in CONTROLLERS]
        ax.plot(x, values, color="0.72", linewidth=0.55, alpha=0.38, zorder=1)
        ax.scatter(x, values, s=9, color=("#333333", "#2878B5", "#D95319"), alpha=0.55, zorder=2)
    distributions = [[lookup[(controller, seed)][metric] for seed in sorted(primary_seeds)]
                     for controller in CONTROLLERS]
    ax.boxplot(distributions, positions=x, widths=0.42, showfliers=False,
               medianprops={"color": "black", "linewidth": 1.5},
               boxprops={"linewidth": 1.2}, whiskerprops={"linewidth": 1.1},
               capprops={"linewidth": 1.1})
    ax.set_xticks(x, ["LQR", "SAC\nR1-S12-100k", "SAC+CBF\nR5-S10-490k"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / f"{basename}.png", dpi=300)
    plt.close(fig)


def analyze(output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    audit = audit_sources(output_dir)
    if audit["cbf_history_count"] != 50 or audit["lqr_history_count"] != 50 \
            or audit["sac_history_count"] != 50:
        raise RuntimeError(f"Histories incomplete; run with --generate-missing first: {audit}")
    if not all(audit[key] for key in ("cbf_seeds_complete", "cbf_model_is_retained_best",
                                      "cbf_vecnormalize_is_matching_retained_best",
                                      "cbf_full_filter", "cbf_testing_seeds_0_49",
                                      "cbf_filtered_command_field",
                                      "all_cbf_history_schemas_times_and_bounds_valid")):
        raise RuntimeError(f"CBF provenance validation failed: {audit}")
    rows = []
    loaders = {
        "LQR": lambda seed: _load_canonical(_canonical_path(output_dir, "lqr", seed), seed, "lqr"),
        "SAC R1-S12-final100k": lambda seed: _load_canonical(_canonical_path(output_dir, "sac", seed), seed, "sac"),
        "SAC+CBF R5-S10-best490k": _load_cbf,
    }
    for controller, loader in loaders.items():
        for seed in SEEDS:
            times, positions, actions, success, reason, intervention = loader(seed)
            metrics = compute_rollout_metrics(times, positions, actions, ACTION_LIMITS)
            rows.append({"controller": controller, "evaluation_seed": seed,
                         "success": success, "termination_reason": reason,
                         **metrics, "cbf_intervention_fraction": intervention})
    successful = {
        controller: {row["evaluation_seed"] for row in rows
                     if row["controller"] == controller and row["success"]}
        for controller in CONTROLLERS
    }
    primary_seeds = set.intersection(*successful.values())
    if not primary_seeds:
        raise RuntimeError(f"No common successful seeds; success counts: "
                           f"{ {controller: len(seeds) for controller, seeds in successful.items()} }")
    summaries = _summary_rows(rows, primary_seeds)
    paired = _paired_rows(rows, primary_seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_rollout_control_effort.csv", rows)
    _write_csv(output_dir / "summary_statistics.csv", summaries)
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    # Independent round-trip recomputation of the first LQR row.
    manual = loaders["LQR"](0)
    recomputed = compute_rollout_metrics(manual[0], manual[1], manual[2], ACTION_LIMITS)
    first = next(row for row in rows if row["controller"] == "LQR" and row["evaluation_seed"] == 0)
    audit.update({
        "success_counts": {c: len(successful[c]) for c in CONTROLLERS},
        "primary_common_successful_seeds": sorted(primary_seeds),
        "primary_population": "intersection of successful evaluation seeds",
        "sample_standard_deviation_ddof": 1,
        "quartile_method": "numpy linear",
        "saturation_definition": "fraction of applied timesteps with >=1 final applied component at its physical bound (atol=1e-10)",
        "manual_seed0_delta_v_matches": np.isclose(recomputed["delta_v_eq_mps"], first["delta_v_eq_mps"]),
        "manual_seed0_J_alpha_matches": np.isclose(recomputed["J_alpha_radps"], first["J_alpha_radps"]),
    })
    (output_dir / "validation_report.json").write_text(
        json.dumps(json_safe(audit), indent=2), encoding="utf-8"
    )
    configuration = {
        "evaluation_seeds": list(SEEDS), "dt_seconds": DT_SECONDS,
        "action_limits": ACTION_LIMITS.tolist(), "retained_sac_model": str(RETAINED_MODEL),
        "retained_sac_vecnormalize": str(RETAINED_VECNORM),
        "cbf_source_campaign": str(CBF_CAMPAIGN),
        "cbf_command_integrated": "action_filtered (post-CBF physical body-frame command)",
        "lqr_and_sac_command_integrated": "core action_body_clipped",
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(json_safe(configuration), indent=2), encoding="utf-8"
    )
    _plot_paired(rows, primary_seeds, "delta_v_eq_mps",
                 r"Equivalent translational $\Delta v$ [m/s]",
                 "equivalent_translational_delta_v_paired", output_dir)
    _plot_paired(rows, primary_seeds, "J_alpha_radps",
                 r"Integrated rotational effort $J_\alpha$ [rad/s]",
                 "integrated_rotational_effort_paired", output_dir)
    print_table(summaries, paired)
    print_interpretation(summaries, paired)
    return rows, summaries, paired


def print_table(summaries: list[dict], paired: list[dict]) -> None:
    summary_cols = ("Controller", "Success", "Mean duration [s]", "Std duration [s]",
                    "Mean Delta_v_eq [m/s]", "Std Delta_v_eq [m/s]",
                    "Mean J_alpha [rad/s]", "Std J_alpha [rad/s]", "Saturation [%]")
    print("\nSUMMARY TABLE")
    print(" | ".join(summary_cols))
    for row in summaries:
        print(" | ".join(str(row[c]) if isinstance(row[c], str) else f"{row[c]:.8g}" for c in summary_cols))
    paired_cols = ("Comparison", "Mean Delta-v difference [m/s]",
                   "Delta-v change vs LQR [%]", "Seeds with lower Delta-v than LQR",
                   "Mean J_alpha difference [rad/s]", "J_alpha change vs LQR [%]",
                   "Seeds with lower J_alpha than LQR", "Duration change vs LQR [%]")
    print("\nPAIRED COMPARISON TABLE")
    print(" | ".join(paired_cols))
    for row in paired:
        print(" | ".join(str(row[c]) if isinstance(row[c], (str, int)) else f"{row[c]:.8g}" for c in paired_cols))


def print_interpretation(summaries: list[dict], paired: list[dict]) -> None:
    dv_best = min(summaries, key=lambda row: row["Mean Delta_v_eq [m/s]"])
    rot_best = min(summaries, key=lambda row: row["Mean J_alpha [rad/s]"])
    lookup = {row["Comparison"].replace(" - LQR", ""): row for row in paired}
    print("\nFACTUAL INTERPRETATION")
    print(f"Lowest mean equivalent translational Delta-v: {dv_best['Controller']} "
          f"({dv_best['Mean Delta_v_eq [m/s]']:.6g} m/s).")
    print(f"Lowest mean integrated rotational effort: {rot_best['Controller']} "
          f"({rot_best['Mean J_alpha [rad/s]']:.6g} rad/s).")
    for method in CONTROLLERS[1:]:
        row = lookup[method]
        print(f"{method}: Delta-v change versus LQR {row['Delta-v change vs LQR [%]']:+.3f}% "
              f"with {row['Seeds with lower Delta-v than LQR']} paired seeds lower; "
              f"J_alpha change {row['J_alpha change vs LQR [%]']:+.3f}% with "
              f"{row['Seeds with lower J_alpha than LQR']} paired seeds lower; "
              f"duration change {row['Duration change vs LQR [%]']:+.3f}%.")
    sac = next(row for row in summaries if row["Controller"] == CONTROLLERS[1])
    cbf = next(row for row in summaries if row["Controller"] == CONTROLLERS[2])
    print(f"SAC+CBF minus retained SAC mean Delta-v: "
          f"{cbf['Mean Delta_v_eq [m/s]'] - sac['Mean Delta_v_eq [m/s]']:+.6g} m/s; "
          f"mean J_alpha: {cbf['Mean J_alpha [rad/s]'] - sac['Mean J_alpha [rad/s]']:+.6g} rad/s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true",
                        help="Inspect saved sources without running any controller.")
    parser.add_argument("--generate-missing", action="store_true",
                        help="Generate only absent LQR/SAC histories; never rerun CBF.")
    parser.add_argument("--analyze", action="store_true",
                        help="Validate complete histories and create tables/figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_missing:
        generate_missing(args.output_dir)
    audit = audit_sources(args.output_dir)
    print(json.dumps(audit, indent=2))
    if args.analyze:
        analyze(args.output_dir)
    elif not args.audit_only and not args.generate_missing:
        raise SystemExit("Choose --audit-only, --generate-missing, and/or --analyze.")


if __name__ == "__main__":
    main()
