"""Train full-closing SAC with protected, scheduled LQR replay mixing.

The LQR demonstrations remain in a separate immutable buffer. Online SAC
experience is stored independently. Every gradient batch contains a fixed
demonstration fraction: 50% initially and 25% after the configured switch by
default. LQR is never called during online interaction or evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
for import_path in (SRC_PATH, SCRIPT_DIR):
    import_path_string = str(import_path)
    if import_path_string not in sys.path:
        sys.path.insert(0, import_path_string)

from generate_lqr_replay_dataset import (  # noqa: E402
    load_demonstration_dataset,
)
from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_replay import (  # noqa: E402
    FullClosingCheckpointCallback,
    initialize_vecnormalize_from_demonstrations,
    make_closing_env,
    resolve_checkpoint_steps,
    write_json,
)
from train_sac_closing_multiseed import (  # noqa: E402
    TRIAL_16_PARAMS,
    TrainingReturnCallback,
)


MODEL_ROOT = SCRIPT_DIR / "models" / "lqr_mixed_replay"
LOG_ROOT = SCRIPT_DIR / "logs" / "lqr_mixed_replay"
RESULTS_ROOT = SCRIPT_DIR / "results" / "lqr_mixed_replay"
DEFAULT_VALIDATION_SEEDS = list(range(1000, 1020))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train full-closing SAC with protected LQR demonstrations and "
            "scheduled mixed replay batches."
        )
    )
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--reward-version", default=None)
    parser.add_argument(
        "--online-buffer-size",
        type=int,
        default=None,
        help=(
            "Capacity for online transitions only. By default, online steps "
            "+ margin rounded up to 100k. Demonstrations are stored separately."
        ),
    )
    parser.add_argument("--online-buffer-margin", type=int, default=50_000)
    parser.add_argument("--initial-demo-fraction", type=float, default=0.50)
    parser.add_argument("--later-demo-fraction", type=float, default=0.25)
    parser.add_argument(
        "--demo-fraction-switch-step",
        type=int,
        default=None,
        help="Online transition count for switching fractions (default: halfway).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_VALIDATION_SEEDS,
    )
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-initialize-normalization",
        action="store_true",
    )
    args = parser.parse_args()

    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if args.online_buffer_margin < 0:
        parser.error("--online-buffer-margin must be non-negative.")
    if args.online_buffer_size is not None and args.online_buffer_size <= 0:
        parser.error("--online-buffer-size must be positive.")
    for name in ("initial_demo_fraction", "later_demo_fraction"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if args.demo_fraction_switch_step is not None:
        if not 0 <= args.demo_fraction_switch_step <= args.total_timesteps:
            parser.error(
                "--demo-fraction-switch-step must lie between 0 and total timesteps."
            )
    if args.max_eval_steps <= 0:
        parser.error("--max-eval-steps must be positive.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values.")
    if not args.validation_seeds:
        parser.error("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error("--validation-seeds must contain unique values.")
    if set(args.seeds) & set(args.validation_seeds):
        parser.error("Training and validation seeds must be disjoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    return args


def resolve_online_buffer_size(requested, total_timesteps, margin):
    required = total_timesteps + margin
    if requested is None:
        return int(np.ceil(required / 100_000) * 100_000)
    if requested < total_timesteps:
        raise ValueError(
            "The online replay buffer would overwrite online transitions "
            f"before training ends: {requested} < {total_timesteps}."
        )
    return int(requested)


def safe_output_paths(experiment_name, seed, overwrite):
    model_dir = MODEL_ROOT / experiment_name / f"seed_{seed}"
    result_dir = RESULTS_ROOT / experiment_name / f"seed_{seed}"
    protected = (
        model_dir / "final_model.zip",
        result_dir / "checkpoint_metrics.csv",
        result_dir / "replay_mixing_initial.json",
    )
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Experiment output already exists. Choose another experiment name "
            f"or pass --overwrite: {existing}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, result_dir


def train_one_seed(
    args,
    dataset,
    dataset_metadata,
    reward_version,
    online_buffer_size,
    switch_step,
    checkpoint_steps,
    experiment_name,
    seed,
):
    model_dir, result_dir = safe_output_paths(
        experiment_name,
        seed,
        args.overwrite,
    )
    train_env = None
    model = None
    try:
        train_env = DummyVecEnv(
            [
                lambda seed=seed: Monitor(
                    make_closing_env(
                        mode="training",
                        seed=seed,
                        reward_version=reward_version,
                        monitor=False,
                    )
                )
            ]
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=TRIAL_16_PARAMS["gamma"],
        )
        if not args.no_initialize_normalization:
            initialize_vecnormalize_from_demonstrations(
                train_env,
                dataset,
                gamma=TRIAL_16_PARAMS["gamma"],
            )

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=TRIAL_16_PARAMS["learning_rate"],
            buffer_size=online_buffer_size,
            learning_starts=0,
            batch_size=TRIAL_16_PARAMS["batch_size"],
            tau=TRIAL_16_PARAMS["tau"],
            gamma=TRIAL_16_PARAMS["gamma"],
            train_freq=TRIAL_16_PARAMS["train_freq"],
            gradient_steps=TRIAL_16_PARAMS["gradient_steps"],
            ent_coef=TRIAL_16_PARAMS["ent_coef"],
            target_update_interval=1,
            replay_buffer_class=ProtectedMixedReplayBuffer,
            replay_buffer_kwargs={
                "initial_demo_fraction": args.initial_demo_fraction,
                "later_demo_fraction": args.later_demo_fraction,
                "demo_fraction_switch_step": switch_step,
            },
            verbose=1,
            seed=seed,
            tensorboard_log=str(LOG_ROOT / experiment_name),
            device=args.device,
        )
        replay_buffer = model.replay_buffer
        if not isinstance(replay_buffer, ProtectedMixedReplayBuffer):
            raise RuntimeError("Protected mixed replay buffer was not created.")
        replay_buffer.set_demonstrations(dataset)
        write_json(
            result_dir / "replay_mixing_initial.json",
            {
                "dataset_path": str(args.demonstrations.resolve()),
                "dataset_metadata": dataset_metadata,
                "normalization_initialized_from_demonstrations": (
                    not args.no_initialize_normalization
                ),
                **replay_buffer.mixing_statistics(),
            },
        )

        checkpoint_callback = FullClosingCheckpointCallback(
            training_seed=seed,
            checkpoint_steps=checkpoint_steps,
            validation_seeds=args.validation_seeds,
            max_eval_steps=args.max_eval_steps,
            reward_version=reward_version,
            model_dir=model_dir,
            result_dir=result_dir,
        )
        training_return_callback = TrainingReturnCallback(
            training_seed=seed,
            result_dir=str(result_dir),
            model_name=f"{experiment_name} - seed {seed}",
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=f"{experiment_name}_seed_{seed}",
            progress_bar=args.progress_bar,
            callback=[training_return_callback, checkpoint_callback],
        )
        model.save(model_dir / "final_model")
        train_env.save(model_dir / "final_vecnormalize.pkl")

        write_json(
            result_dir / "replay_mixing_final.json",
            replay_buffer.mixing_statistics(),
        )
        best_checkpoint, _, best_checkpoint_episode = (
            checkpoint_callback.get_best_checkpoint()
        )
        training_return_callback.set_best_checkpoint(
            best_checkpoint_episode,
            best_checkpoint,
        )
        training_return_callback.save_outputs()
    finally:
        if train_env is not None:
            train_env.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    dataset, dataset_metadata = load_demonstration_dataset(args.demonstrations)
    dataset_reward_version = dataset_metadata["reward_version"]
    reward_version = args.reward_version or dataset_reward_version
    if reward_version != dataset_reward_version:
        raise ValueError(
            "Training and demonstration reward versions differ: "
            f"{reward_version!r} != {dataset_reward_version!r}."
        )

    online_buffer_size = resolve_online_buffer_size(
        args.online_buffer_size,
        args.total_timesteps,
        args.online_buffer_margin,
    )
    switch_step = (
        args.demo_fraction_switch_step
        if args.demo_fraction_switch_step is not None
        else args.total_timesteps // 2
    )
    checkpoint_steps = resolve_checkpoint_steps(
        args.checkpoint_steps,
        args.total_timesteps,
    )
    if args.experiment_name is None:
        experiment_name = (
            f"sac_closing_reward{reward_version}_lqr_mixed_replay_"
            f"{args.total_timesteps}steps_"
            f"{int(100 * args.initial_demo_fraction)}to"
            f"{int(100 * args.later_demo_fraction)}pct"
        )
    else:
        experiment_name = args.experiment_name

    config = {
        "experiment_name": experiment_name,
        "demonstration_dataset": str(args.demonstrations.resolve()),
        "demonstration_metadata": dataset_metadata,
        "training_seeds": args.seeds,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "reward_version": reward_version,
        "checkpoint_steps": checkpoint_steps,
        "max_eval_steps": args.max_eval_steps,
        "online_buffer_size": online_buffer_size,
        "online_buffer_margin": args.online_buffer_margin,
        "initial_demo_fraction": args.initial_demo_fraction,
        "later_demo_fraction": args.later_demo_fraction,
        "demo_fraction_switch_step": switch_step,
        "demonstrations_protected": True,
        "demonstration_sampling": "uniform_all_transitions",
        "full_p2_resets_only": True,
        "lqr_used_during_online_training": False,
        "learning_starts": 0,
        "trial_16_params": TRIAL_16_PARAMS,
    }
    write_json(
        RESULTS_ROOT / experiment_name / "experiment_config.json",
        config,
    )

    print("=" * 80)
    print("Full-closing SAC with protected mixed LQR replay")
    print("=" * 80)
    print(f"Experiment:             {experiment_name}")
    print(f"Demonstrations:         {dataset_metadata['transition_count']:,}")
    print(f"Successful episodes:    {dataset_metadata['successful_episodes']}")
    print(f"Online timesteps:       {args.total_timesteps:,} per seed")
    print(f"Online replay capacity: {online_buffer_size:,}")
    print(
        "Demo batch schedule:    "
        f"{args.initial_demo_fraction:.0%} -> "
        f"{args.later_demo_fraction:.0%} at {switch_step:,} online steps"
    )
    print("Demonstration sampling: uniform across every complete trajectory")
    print("Online resets:          ordinary full training P2 distribution")
    print("Online LQR use:         none")
    print("=" * 80)

    for seed in args.seeds:
        train_one_seed(
            args=args,
            dataset=dataset,
            dataset_metadata=dataset_metadata,
            reward_version=reward_version,
            online_buffer_size=online_buffer_size,
            switch_step=switch_step,
            checkpoint_steps=checkpoint_steps,
            experiment_name=experiment_name,
            seed=seed,
        )


if __name__ == "__main__":
    main()
