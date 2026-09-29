"""Train a fresh long-range SAC from sampled P2 states to the shared X_H set.

Run this only after the systematic capture controller demonstrates reliable
150 m capture.  The script is isolated from all existing closing trainers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import SRC_ROOT, TRIAL_DIR, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from systematic_interface import DEFAULT_CONTRACT  # noqa: E402
from systematic_long_range_env import (  # noqa: E402
    SystematicLongRangeEnv,
    SystematicLongRangeResetWrapper,
)


SYSTEMATIC_ROOT = TRIAL_DIR / "artifacts" / "systematic"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=800_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_env(args):
    base = SystematicLongRangeEnv(
        phase="closing",
        mode="training",
        max_time=args.max_time,
        seed=args.training_seed,
    )
    return Monitor(SystematicLongRangeResetWrapper(base, p2_mode="training"))


def main():
    args = parse_args()
    if args.total_timesteps <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("Training and checkpoint timesteps must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = args.experiment_name or (
        f"systematic_long_range_XH_seed{args.training_seed}_{timestamp}"
    )
    model_dir = SYSTEMATIC_ROOT / "models" / experiment
    result_dir = SYSTEMATIC_ROOT / "results" / experiment
    log_dir = SYSTEMATIC_ROOT / "logs" / experiment
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse experiment directory: {path}")

    config = {
        "experiment_name": experiment,
        "isolated_systematic_trial": True,
        "task": "sampled_X_P2_to_systematic_X_H",
        "handover_contract": DEFAULT_CONTRACT.as_dict(),
        "success": "all six X_H tolerances satisfied simultaneously",
        "hard_safety_termination": ["geometry", "velocity", "alignment"],
        "training_seed": args.training_seed,
        "max_time_s": args.max_time,
        "total_timesteps": args.total_timesteps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "sac": {
            "learning_rate": 1e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "batch_size": 512,
            "tau": 0.005,
            "gamma": 0.999,
            "train_freq": 16,
            "gradient_steps": 8,
            "ent_coef": 0.01,
        },
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    write_json(result_dir / "run_config.json", config)
    vector_env = VecNormalize(
        DummyVecEnv([lambda: make_env(args)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.999,
    )
    model = SAC(
        "MlpPolicy",
        vector_env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
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
    callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(model_dir),
        name_prefix="systematic_long_range_sac",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            tb_log_name=experiment,
            progress_bar=args.progress_bar,
        )
        model.save(str(model_dir / "final_model"))
        vector_env.save(str(model_dir / "final_vecnormalize.pkl"))
    finally:
        vector_env.close()

    print(f"Long-range model: {model_dir / 'final_model.zip'}")
    print(f"Normalization: {model_dir / 'final_vecnormalize.pkl'}")
    print(f"Configuration: {result_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
