"""Train closing SAC from scratch with the isolated P3-reaching reward.

This experiment starts every rollout from the normal P2 training distribution.
It creates a new actor, new critics, an empty replay buffer, and new observation
and reward normalization statistics. Safety classifications and penalties are
retained, but a safety violation does not end an exploration episode.

No pretrained model or VecNormalize file is loaded or modified.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationV2GymEnv,
)
from train_sac_closing_no_safety_termination import (
    ExperimentCheckpointCallback,
    TrainingSafetyDiagnosticsCallback,
    make_env,
    write_json,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_FAMILY = "p3reach_from_scratch"
EXPERIMENT_ENV_CLASS = ClosingSafetyExplorationV2GymEnv
EXPERIMENT_REWARD_SCHEME = EXPERIMENT_ENV_CLASS.reward_scheme
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", EXPERIMENT_FAMILY)
LOG_ROOT = os.path.join(SCRIPT_DIR, "logs", EXPERIMENT_FAMILY)
RESULT_ROOT = os.path.join(SCRIPT_DIR, "results", EXPERIMENT_FAMILY)

# Keep the best-known trial-16 SAC optimizer settings fixed. The controlled
# changes are initialization, the experiment reward, and safety termination.
SAC_PARAMETERS = {
    "learning_rate": 0.00026276728418436003,
    "buffer_size": 300_000,
    "learning_starts": 10_000,
    "batch_size": 512,
    "tau": 0.01,
    "gamma": 0.99,
    "train_freq": 16,
    "gradient_steps": 8,
    "ent_coef": "auto_0.05",
    "target_update_interval": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a new closing SAC from scratch with p3_reach_v2 and "
            "non-terminal safety exploration."
        )
    )
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[100_000, 200_000, 300_000, 400_000, 500_000, 600_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def resolve_checkpoint_steps(args):
    checkpoints = sorted(
        {
            int(step)
            for step in [*args.checkpoint_steps, args.total_timesteps]
            if 0 < int(step) <= args.total_timesteps
        }
    )
    if not checkpoints or checkpoints[-1] != args.total_timesteps:
        raise RuntimeError("The final training step must be a checkpoint.")
    return checkpoints


def build_config(args, checkpoints, experiment_name):
    return {
        "experiment_name": experiment_name,
        "purpose": "closing SAC P3-reaching training from scratch",
        "phase": "closing",
        "initial_distribution": "normal P2 training distribution",
        "initialization": {
            "actor": "new random initialization",
            "critics": "new random initialization",
            "replay_buffer": "new empty buffer",
            "vecnormalize": "new statistics",
            "pretrained_model_loaded": False,
            "pretrained_vecnormalize_loaded": False,
        },
        "experiment_reward_scheme": EXPERIMENT_REWARD_SCHEME,
        "shared_gym_reward_version_used_for_experiment": False,
        "shared_reward_files_modified": False,
        "experiment_reward_weights": EXPERIMENT_ENV_CLASS.reward_weights,
        "capture_scales": {
            "axial_position_m": float(EXPERIMENT_ENV_CLASS.capture_scales[0]),
            "lateral_position_m": float(EXPERIMENT_ENV_CLASS.capture_scales[1]),
            "axial_velocity_mps": float(EXPERIMENT_ENV_CLASS.capture_scales[2]),
            "lateral_velocity_mps": float(EXPERIMENT_ENV_CLASS.capture_scales[3]),
            "attitude_deg": float(
                EXPERIMENT_ENV_CLASS.capture_scales[4] * 180.0 / 3.141592653589793
            ),
            "rate_degs": float(
                EXPERIMENT_ENV_CLASS.capture_scales[5] * 180.0 / 3.141592653589793
            ),
        },
        "precision_gate_distance_m": (
            EXPERIMENT_ENV_CLASS.precision_gate_distance_m
        ),
        "precision_gate_width_m": (
            EXPERIMENT_ENV_CLASS.precision_gate_width_m
        ),
        "safety_classification_enabled": True,
        "safety_penalties_enabled": True,
        "safety_violation_termination_enabled": False,
        "terminal_conditions": ["P3 success", "time limit", "non-finite state"],
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": checkpoints,
        "max_time": args.max_time,
        "device": args.device,
        "sac_parameters": SAC_PARAMETERS,
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": SAC_PARAMETERS["gamma"],
        },
        "checkpoint_selection": (
            "success, target ratio, position error, unsafe fraction, return"
        ),
        "training_learning_curve": {
            "file": "episode_return_learning_curve.png",
            "raw_episode_return": True,
            "moving_average_window_episodes": 10,
            "refresh_interval_episodes": 5,
            "evaluation_checkpoints_marked": True,
        },
    }


def main():
    args = parse_args()
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or (
        "sac_closing_p3reachv2_from_scratch_"
        "no_safety_termination_"
        f"{args.total_timesteps // 1000}k_seed{args.training_seed}_{timestamp}"
    )
    config = build_config(args, checkpoints, experiment_name)

    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = os.path.join(MODEL_ROOT, experiment_name)
    result_dir = os.path.join(RESULT_ROOT, experiment_name)
    log_dir = os.path.join(LOG_ROOT, experiment_name)
    for path in (model_dir, result_dir, log_dir):
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        os.makedirs(path)
    write_json(os.path.join(result_dir, "run_config.json"), config)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    env_class=EXPERIMENT_ENV_CLASS,
                )
            )
        ]
    )
    training_env = VecNormalize(
        training_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=SAC_PARAMETERS["gamma"],
    )

    model = SAC(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=SAC_PARAMETERS["learning_rate"],
        buffer_size=SAC_PARAMETERS["buffer_size"],
        learning_starts=SAC_PARAMETERS["learning_starts"],
        batch_size=SAC_PARAMETERS["batch_size"],
        tau=SAC_PARAMETERS["tau"],
        gamma=SAC_PARAMETERS["gamma"],
        train_freq=SAC_PARAMETERS["train_freq"],
        gradient_steps=SAC_PARAMETERS["gradient_steps"],
        ent_coef=SAC_PARAMETERS["ent_coef"],
        target_update_interval=SAC_PARAMETERS["target_update_interval"],
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=log_dir,
        device=args.device,
    )

    training_callback = TrainingSafetyDiagnosticsCallback(
        output_path=os.path.join(
            result_dir,
            "training_episode_safety_metrics.csv",
        ),
        plot_path=os.path.join(
            result_dir,
            "episode_return_learning_curve.png",
        ),
        checkpoint_steps=checkpoints,
    )
    checkpoint_callback = ExperimentCheckpointCallback(
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_time=args.max_time,
        model_dir=model_dir,
        result_dir=result_dir,
        evaluate_initial=False,
        env_class=EXPERIMENT_ENV_CLASS,
    )

    print("=" * 80)
    print("Closing SAC from scratch: P3-reaching reward")
    print("=" * 80)
    print(f"Experiment:          {experiment_name}")
    print("Initialization:      new actor, critics, replay buffer, normalization")
    print("Initial states:      normal P2 training distribution")
    print(f"Reward:              isolated {EXPERIMENT_REWARD_SCHEME}")
    print("Safety violations:   classified and penalized, but non-terminal")
    print(f"Training steps:      {args.total_timesteps:,}")
    print(f"Checkpoints:         {checkpoints}")
    print(f"Validation seeds:    {args.validation_seeds}")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=True,
            tb_log_name=experiment_name,
            progress_bar=args.progress_bar,
            callback=[training_callback, checkpoint_callback],
        )
        model.save(os.path.join(model_dir, "final_model"))
        training_env.save(os.path.join(model_dir, "final_vecnormalize.pkl"))
    finally:
        training_env.close()

    print(f"Best checkpoint: {checkpoint_callback.best_step}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
