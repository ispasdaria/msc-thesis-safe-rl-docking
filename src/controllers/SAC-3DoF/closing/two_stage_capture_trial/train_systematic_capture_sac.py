"""Train a fresh SAC for the systematic ``X_H -> validated X_P3`` task.

This script does not read the old handover dataset and writes only beneath
``artifacts/systematic``. Every episode is sampled directly from the complete
150 m six-state handover distribution; no distance curriculum is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import SRC_ROOT, TRIAL_DIR, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from systematic_capture_env import SystematicCaptureEnv  # noqa: E402
from systematic_capture_observation import (  # noqa: E402
    MarkovCaptureObservationScale,
)
from systematic_capture_reset_env import (  # noqa: E402
    SystematicCaptureResetWrapper,
)
from systematic_interface import DEFAULT_CONTRACT  # noqa: E402


SYSTEMATIC_ROOT = TRIAL_DIR / "artifacts" / "systematic"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=2_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing experiment with the same name under "
            "artifacts/systematic."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("Training and checkpoint timesteps must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.experiment_name is not None:
        name = args.experiment_name
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError(
                "--experiment-name must be a single directory name, not a path."
            )


def prepare_output_directories(paths, overwrite):
    systematic_root = SYSTEMATIC_ROOT.resolve()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(systematic_root):
            raise ValueError(f"Refusing output outside {systematic_root}: {resolved}")
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to reuse experiment directory: {path}. "
                    "Pass --overwrite to replace this isolated experiment."
                )
            shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True)


def make_capture_env(args):
    base = SystematicCaptureEnv(
        phase="closing",
        mode="training",
        max_time=args.max_time,
        seed=args.training_seed,
    )
    wrapped = SystematicCaptureResetWrapper(
        base,
        capture_distance_m=DEFAULT_CONTRACT.handover_distance_m,
        enable_lateral_divergence_termination=True,
    )
    return Monitor(MarkovCaptureObservationScale(wrapped))


def build_config(args, experiment):
    return {
        "experiment_name": experiment,
        "isolated_systematic_trial": True,
        "task": "X_H_to_validated_X_P3_testing",
        "handover_contract": DEFAULT_CONTRACT.as_dict(),
        "success": "direct membership in the validated final-controller P3 testing box",
        "reset_distribution": "synthetic full-six-state X_H pairs",
        "curriculum_learning": False,
        "capture_reset_distance_m": DEFAULT_CONTRACT.handover_distance_m,
        "reward": {
            "scheme": SystematicCaptureEnv.reward_scheme,
            "axial_progress_clip_m": [
                -SystematicCaptureEnv.axial_progress_negative_clip_m,
                SystematicCaptureEnv.axial_progress_positive_clip_m,
            ],
            "axial_progress_scale": SystematicCaptureEnv.axial_progress_scale,
            "positive_axial_progress_gate_components": ["ry", "vy", "attitude"],
            "axial_velocity_tracking_target": "sampled_p3_target_vx",
            "distance_dependent_vx_reference": False,
            "distance_dependent_vx_weight": True,
            "vx_tracking_weight_far": SystematicCaptureEnv.vx_tracking_weight_far,
            "vx_tracking_weight_near": SystematicCaptureEnv.vx_tracking_weight_near,
            "vx_full_weight_axial_gap_m": (
                SystematicCaptureEnv.vx_full_weight_axial_gap_m
            ),
            "tracking_cost": "distance_weighted_pseudo_huber_with_linear_tail",
            "tracking_scales": SystematicCaptureEnv.tracking_scales.tolist(),
            "tracking_weights_far": (
                SystematicCaptureEnv.tracking_weights_far.tolist()
            ),
            "tracking_weights_near": (
                SystematicCaptureEnv.tracking_weights_near.tolist()
            ),
            "progress_gate_floor": SystematicCaptureEnv.progress_gate_floor,
            "progress_gate_error_bounds": (
                SystematicCaptureEnv.progress_gate_error_bounds.tolist()
            ),
            "coupled_lateral_energy": {
                "components": ["ry", "vy"],
                "scales": SystematicCaptureEnv.lateral_energy_scales.tolist(),
                "progress_negative_clip": (
                    SystematicCaptureEnv.lateral_energy_progress_negative_clip
                ),
                "progress_positive_clip": (
                    SystematicCaptureEnv.lateral_energy_progress_positive_clip
                ),
                "progress_scale": (
                    SystematicCaptureEnv.lateral_energy_progress_scale
                ),
                "state_cost_weight": (
                    SystematicCaptureEnv.lateral_energy_state_weight
                ),
                "state_cost_bounded": False,
            },
            "terminal_box_progress": {
                "success_box": "global_validated_p3_box",
                "active_only_in_terminal_sync_zone": True,
                "negative_clip": (
                    SystematicCaptureEnv.terminal_box_progress_negative_clip
                ),
                "positive_clip": (
                    SystematicCaptureEnv.terminal_box_progress_positive_clip
                ),
                "scale": SystematicCaptureEnv.terminal_box_progress_scale,
            },
            "terminal_readiness": {
                "zone_outside_upper_p3_face_m": (
                    SystematicCaptureEnv.terminal_sync_zone_m
                ),
                "inward_speed_scale_mps": (
                    SystematicCaptureEnv.unready_inward_speed_scale_mps
                ),
                "penalty_weight": (
                    SystematicCaptureEnv.unready_inward_penalty_weight
                ),
                "lateral_position_weight": (
                    SystematicCaptureEnv.terminal_lateral_unready_weight
                ),
                "lateral_position_depth_clip": (
                    SystematicCaptureEnv.terminal_lateral_unready_depth_clip
                ),
            },
            "box_state_cost_weight": SystematicCaptureEnv.box_state_cost_weight,
            "continuous_geometry_warning_depth": True,
            "passed_p3_warning_boundary_rx_m": float(
                DEFAULT_CONTRACT.p3_lower[0]
            ),
            "warning_weights": {
                "geometry": SystematicCaptureEnv.geometry_warning_weight,
                "velocity": SystematicCaptureEnv.velocity_warning_weight,
                "alignment": SystematicCaptureEnv.alignment_warning_weight,
            },
            "attitude_margin": {
                "start_deg": float(
                    np.round(
                        np.rad2deg(SystematicCaptureEnv.attitude_margin_start),
                        12,
                    )
                ),
                "full_cost_deg": float(
                    np.round(
                        np.rad2deg(SystematicCaptureEnv.attitude_margin_full),
                        12,
                    )
                ),
                "weight": SystematicCaptureEnv.attitude_margin_weight,
            },
            "action_weights": SystematicCaptureEnv.action_weights.tolist(),
            "action_delta_weights": (
                SystematicCaptureEnv.action_delta_weights.tolist()
            ),
            "success_bonus": SystematicCaptureEnv.success_bonus,
            "safety_failure_penalty": SystematicCaptureEnv.safety_failure_penalty,
            "timeout_penalty": SystematicCaptureEnv.timeout_penalty,
            "failed_approach_penalty": (
                SystematicCaptureResetWrapper.failed_approach_penalty
            ),
            "missed_handover_penalty": (
                SystematicCaptureResetWrapper.missed_handover_penalty
            ),
        },
        "hard_safety_termination": {
            "geometry": True,
            "velocity": True,
            "alignment": True,
            "attitude_warning_deg": 3.0,
            "attitude_unsafe_deg": 4.0,
            "angular_rate_warning_degs": 0.15,
            "angular_rate_unsafe_degs": 0.20,
        },
        "failed_approach_termination": {
            "inward_below_validated_p3_box": True,
            "inward_boundary_rx_m": float(DEFAULT_CONTRACT.p3_lower[0]),
            "position_growth_from_best_m": 50.0,
            "moving_away_rate_mps": 0.01,
            "consecutive_steps": 60,
            "lateral_divergence": True,
            "maximum_lateral_error_m": 50.0,
            "maximum_lateral_velocity_error_mps": 0.15,
        },
        "training_seed": args.training_seed,
        "max_time_s": args.max_time,
        "total_timesteps": args.total_timesteps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "sac": {
            "learning_rate": 1e-4,
            "buffer_size": args.total_timesteps,
            "learning_starts": 5_000,
            "batch_size": 512,
            "tau": 0.005,
            "gamma": 0.999,
            "train_freq": 16,
            "gradient_steps": 8,
            "ent_coef": 0.01,
        },
        "observation_scaling": MarkovCaptureObservationScale.as_dict(),
        "vecnormalize": {
            "norm_obs": False,
            "norm_reward": False,
            "clip_obs": MarkovCaptureObservationScale.clip,
        },
        "device": args.device,
    }


def main():
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = args.experiment_name or (
        f"systematic_capture_v9_h150_seed{args.training_seed}_{timestamp}"
    )
    model_dir = SYSTEMATIC_ROOT / "models" / experiment
    result_dir = SYSTEMATIC_ROOT / "results" / experiment
    log_dir = SYSTEMATIC_ROOT / "logs" / experiment

    config = build_config(args, experiment)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    prepare_output_directories(
        (model_dir, result_dir, log_dir), overwrite=args.overwrite
    )
    write_json(result_dir / "run_config.json", config)

    vector_env = DummyVecEnv([lambda: make_capture_env(args)])
    vector_env = VecNormalize(
        vector_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.999,
    )
    model = SAC(
        "MlpPolicy",
        vector_env,
        learning_rate=1e-4,
        buffer_size=args.total_timesteps,
        learning_starts=5_000,
        batch_size=512,
        tau=0.005,
        gamma=0.999,
        train_freq=16,
        gradient_steps=8,
        ent_coef=0.01,
        target_update_interval=1,
        tensorboard_log=str(log_dir),
        seed=args.training_seed,
        device=args.device,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(model_dir),
        name_prefix="systematic_capture_sac",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint,
            tb_log_name=experiment,
            progress_bar=args.progress_bar,
        )
        model.save(str(model_dir / "final_model"))
        vector_env.save(str(model_dir / "final_vecnormalize.pkl"))
    finally:
        vector_env.close()

    print(f"Capture model: {model_dir / 'final_model.zip'}")
    print(f"Normalization: {model_dir / 'final_vecnormalize.pkl'}")
    print(f"Configuration: {result_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
