"""Train a new capture-only SAC from isolated handover reset data.

The script creates new actor/critic networks, replay data, normalization state,
and output directories.  It never loads or overwrites the frozen long-range
checkpoint and does not modify any existing closing training script.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULT_DIR,
    SRC_ROOT,
    write_json,
)
from capture_reset_env import CaptureResetWrapper

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from capture_safety_terminal_env import CaptureSafetyTerminalEnv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR / "guarded100k_handover_sweep.npz",
    )
    parser.add_argument("--capture-radius", type=float, default=100.0)
    parser.add_argument("--empirical-fraction", type=float, default=0.80)
    parser.add_argument("--lateral-error", type=float, default=15.0)
    parser.add_argument("--velocity-error", type=float, default=0.03)
    parser.add_argument("--attitude-error-deg", type=float, default=3.0)
    parser.add_argument("--rate-error-degs", type=float, default=0.10)
    parser.add_argument("--maximum-radial-rate", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=25_000)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=2_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not args.dry_run and not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.capture_radius <= 0.0:
        raise ValueError("--capture-radius must be positive.")
    if not 0.0 <= args.empirical_fraction <= 1.0:
        raise ValueError("--empirical-fraction must lie in [0, 1].")
    if args.total_timesteps <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("Training and checkpoint timesteps must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")


def make_capture_env(args):
    base = CaptureSafetyTerminalEnv(
        phase="closing",
        mode="training",
        max_time=args.max_time,
        seed=args.training_seed,
    )
    wrapped = CaptureResetWrapper(
        base,
        dataset_path=args.dataset,
        radius_m=args.capture_radius,
        empirical_fraction=args.empirical_fraction,
        lateral_error_m=args.lateral_error,
        velocity_error_mps=args.velocity_error,
        attitude_error_deg=args.attitude_error_deg,
        rate_error_degs=args.rate_error_degs,
        maximum_radial_rate_mps=args.maximum_radial_rate,
    )
    return Monitor(wrapped)


def main():
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = args.experiment_name or (
        f"capture_sac_r{args.capture_radius:g}_seed{args.training_seed}_{timestamp}"
    )
    model_dir = DEFAULT_MODEL_DIR / experiment
    result_dir = DEFAULT_RESULT_DIR / experiment
    log_dir = DEFAULT_LOG_DIR / experiment
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse trial output directory: {path}")

    config = {
        "experiment_name": experiment,
        "isolated_trial": True,
        "dataset": str(args.dataset.resolve()),
        "capture_radius_m": args.capture_radius,
        "empirical_fraction": args.empirical_fraction,
        "component_bounds": {
            "lateral_error_m": args.lateral_error,
            "velocity_error_mps": args.velocity_error,
            "attitude_error_deg": args.attitude_error_deg,
            "rate_error_degs": args.rate_error_degs,
            "maximum_radial_rate_mps": args.maximum_radial_rate,
        },
        "reward": CaptureSafetyTerminalEnv.reward_scheme,
        "reward_parameters": {
            "progress_metric": (
                "0.5 * mean(log1p(P3 tolerance ratios)) + "
                "0.5 * smooth-worst(log1p(P3 tolerance ratios))"
            ),
            "progress_scale": CaptureSafetyTerminalEnv.progress_scale,
            "progress_clip": CaptureSafetyTerminalEnv.progress_clip,
            "action_weight": CaptureSafetyTerminalEnv.action_weight,
            "warning_weight": CaptureSafetyTerminalEnv.warning_weight,
            "success_bonus": CaptureSafetyTerminalEnv.success_bonus,
            "safety_failure_penalty": (
                CaptureSafetyTerminalEnv.safety_failure_penalty
            ),
            "timeout_penalty": CaptureSafetyTerminalEnv.timeout_penalty,
            "persistent_state_cost": False,
        },
        "success": "existing six-component P3 target tolerance",
        "hard_safety_termination": True,
        "capture_alignment_safety": {
            "enabled": True,
            "attitude_safe_deg": 3.0,
            "attitude_unsafe_deg": 4.0,
            "angular_rate_safe_degs": 0.15,
            "angular_rate_unsafe_degs": 0.20,
            "warning_semantics": "between safe and unsafe limits",
            "unsafe_semantics": "immediate terminal safety failure",
        },
        "final_approach_default_max_time_s": 5_000.0,
        "training_seed": args.training_seed,
        "max_time_s": args.max_time,
        "total_timesteps": args.total_timesteps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "sac": {
            "learning_rate": 1e-4,
            "buffer_size": 500_000,
            "learning_starts": 5_000,
            "batch_size": 512,
            "tau": 0.005,
            "gamma": 0.999,
            "train_freq": 16,
            "gradient_steps": 8,
            "ent_coef": "auto_0.05",
        },
        "vecnormalize": {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0},
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    write_json(result_dir / "run_config.json", config)

    vector_env = DummyVecEnv([lambda: make_capture_env(args)])
    vector_env = VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.999,
    )
    model = SAC(
        "MlpPolicy",
        vector_env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=512,
        tau=0.005,
        gamma=0.999,
        train_freq=16,
        gradient_steps=8,
        ent_coef="auto_0.05",
        target_update_interval=1,
        tensorboard_log=str(log_dir),
        seed=args.training_seed,
        device=args.device,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(model_dir),
        name_prefix="capture_sac",
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
