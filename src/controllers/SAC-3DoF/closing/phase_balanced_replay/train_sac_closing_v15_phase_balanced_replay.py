"""Continue the terminal V15 trial-16 SAC with phase-balanced replay.

The source actor, critics, entropy state, and VecNormalize statistics are loaded
from the 400k Optuna trial-16 checkpoint.  Its unsaved uniform replay buffer is
replaced by online global/capture replay and filled with the source stochastic
policy before gradient updates resume.  The standard closing environment is
used, so hard safety violations terminate exactly as they did during the
source policy's training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLOSING_DIR = os.path.dirname(SCRIPT_DIR)
VANILLA_DIR = os.path.join(CLOSING_DIR, "vanilla_sac")
for path in (SCRIPT_DIR, CLOSING_DIR, VANILLA_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from phase_balanced_replay_buffer_implementation import (  # noqa: E402
    PhaseBalancedReplayBuffer,
)
from train_sac_closing_multiseed import (  # noqa: E402
    EvaluationCheckpointCallback,
    RESULTS_ROOT,
    TRIAL_16_PARAMS,
    TrainingReturnCallback,
    seed_output_paths,
    write_aggregate_outputs,
    write_json,
)
from tune_sac_closing_optuna import make_training_env  # noqa: E402


MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", "multiseed")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "multiseed")


def seed_output_paths(experiment_name, seed):
    return (
        os.path.join(MODEL_ROOT, experiment_name, f"seed_{seed}"),
        os.path.join(RESULTS_ROOT, experiment_name, f"seed_{seed}"),
    )


DEFAULT_SOURCE_MODEL = os.path.join(
    VANILLA_DIR,
    "models",
    "reward_shaping_trials",
    "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_"
    "seed0_buffer300k.zip",
)
DEFAULT_SOURCE_VECNORMALIZE = os.path.join(
    VANILLA_DIR,
    "models",
    "reward_shaping_trials",
    "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_"
    "seed0_buffer300k_vecnormalize.pkl",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue terminal V15 Optuna trial 16 with online "
            "phase-balanced SAC replay."
        )
    )
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument("--training-timesteps", type=int, default=50_000)
    parser.add_argument(
        "--max-replay-warmup-steps",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[10_000, 25_000, 50_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--global-buffer-size", type=int, default=300_000)
    parser.add_argument("--capture-buffer-size", type=int, default=100_000)
    parser.add_argument("--priority-buffer-size", type=int, default=25_000)
    parser.add_argument("--capture-distance-m", type=float, default=200.0)
    parser.add_argument("--capture-batch-fraction", type=float, default=0.25)
    parser.add_argument(
        "--priority-capture-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-capture-transitions",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--minimum-capture-episodes",
        type=int,
        default=5,
    )
    parser.add_argument("--continuation-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--continuation-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--experiment-name",
        default=(
            "sac_closing_v15_trial16_terminal_phase_balanced_guarded50k_seed0"
        ),
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.training_timesteps <= 0:
        raise ValueError("--training-timesteps must be positive.")
    if args.max_replay_warmup_steps <= 0:
        raise ValueError("--max-replay-warmup-steps must be positive.")
    if args.max_eval_steps <= 0:
        raise ValueError("--max-eval-steps must be positive.")
    for name in (
        "global_buffer_size",
        "capture_buffer_size",
        "priority_buffer_size",
        "minimum_capture_transitions",
        "minimum_capture_episodes",
        "continuation_gradient_steps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.capture_distance_m <= 0.0:
        raise ValueError("--capture-distance-m must be positive.")
    if args.continuation_learning_rate <= 0.0:
        raise ValueError("--continuation-learning-rate must be positive.")
    for name in ("capture_batch_fraction", "priority_capture_fraction"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if not os.path.isfile(args.source_model):
        raise FileNotFoundError(args.source_model)
    if not os.path.isfile(args.source_vecnormalize):
        raise FileNotFoundError(args.source_vecnormalize)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")


def resolve_checkpoint_steps(args):
    checkpoints = sorted(
        {
            int(step)
            for step in [*args.checkpoint_steps, args.training_timesteps]
            if 0 < int(step) <= args.training_timesteps
        }
    )
    if not checkpoints or checkpoints[-1] != args.training_timesteps:
        raise RuntimeError("The final training timestep must be a checkpoint.")
    return checkpoints


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safety_capture_checkpoint_rank(summary, mean_return, score):
    """Prefer success, then safety and actual P3 approach quality."""
    return (
        float(summary["success_rate"]),
        -float(summary["failure_rate"]),
        -float(summary["mean_min_position_error"]),
        -float(summary["mean_final_max_target_ratio"]),
        float(score),
        float(mean_return),
    )


class ReplayReadinessCallback(BaseCallback):
    """Stop frozen-policy collection once replay diversity is sufficient."""

    def __init__(self):
        super().__init__()
        self.ready = False
        self.ready_at_timestep = None

    def _on_step(self):
        replay_buffer = self.model.replay_buffer
        if replay_buffer.phase_balancing_active:
            self.ready = True
            self.ready_at_timestep = int(self.num_timesteps)
            return False
        return True


class PhaseBalancedCheckpointCallback(EvaluationCheckpointCallback):
    """Include the unchanged source policy and replay state at every check."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_evaluation_complete = False
        self.replay_rows = []

    def _on_training_start(self):
        if not self.initial_evaluation_complete:
            self._evaluate_and_save(0)
            self.initial_evaluation_complete = True

    def _evaluate_and_save(self, checkpoint):
        super()._evaluate_and_save(checkpoint)
        statistics = self.model.replay_buffer.replay_statistics()
        self.replay_rows.append(
            {"additional_timestep": int(checkpoint), **statistics}
        )
        write_csv(
            os.path.join(
                self.result_dir,
                "replay_buffer_checkpoint_metrics.csv",
            ),
            self.replay_rows,
        )


def build_config(args, checkpoints, source_num_timesteps):
    return {
        "experiment_name": args.experiment_name,
        "purpose": (
            "continue terminal V15 Optuna trial 16 with online "
            "phase-balanced replay"
        ),
        "training_seeds": [args.training_seed],
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "reward_version": "v15",
        "source_model": os.path.abspath(args.source_model),
        "source_vecnormalize": os.path.abspath(args.source_vecnormalize),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize_sha256": file_sha256(
            args.source_vecnormalize
        ),
        "source_model_num_timesteps": int(source_num_timesteps),
        "policy_initialization": "loaded V15 trial-16 400k checkpoint",
        "critic_initialization": "loaded V15 trial-16 400k checkpoint",
        "replay_initialization": "new empty phase-balanced online buffer",
        "gradient_training_timesteps": args.training_timesteps,
        "total_timesteps": args.training_timesteps,
        "maximum_replay_warmup_interaction_steps": (
            args.max_replay_warmup_steps
        ),
        "replay_warmup_action_source": "loaded stochastic SAC policy",
        "gradient_updates_during_replay_warmup": False,
        "checkpoint_additional_steps": [0, *checkpoints],
        "checkpoint_steps": [0, *checkpoints],
        "max_eval_steps": args.max_eval_steps,
        "device": args.device,
        "trial_16_params": TRIAL_16_PARAMS,
        "continuation_optimizer": {
            "learning_rate": args.continuation_learning_rate,
            "gradient_steps": args.continuation_gradient_steps,
            "train_freq": TRIAL_16_PARAMS["train_freq"],
            "source_learning_rate": TRIAL_16_PARAMS["learning_rate"],
            "source_gradient_steps": TRIAL_16_PARAMS["gradient_steps"],
        },
        "safety_violation_termination_enabled": True,
        "training_environment": (
            "standard Docking3DoFGymEnv closing V15 with hard safety "
            "termination"
        ),
        "terminal_conditions": [
            "P3 success",
            "hard safety violation",
            "time limit",
            "numerical failure",
        ],
        "cbf_filter_used_during_training": False,
        "lqr_policy_or_transitions_used": False,
        "curriculum_used": False,
        "phase_balanced_replay": {
            "global_partition": "all current-agent online transitions",
            "global_buffer_size": args.global_buffer_size,
            "capture_partition": (
                "current-agent online transitions at or below the P3 "
                "capture distance"
            ),
            "capture_distance_m": args.capture_distance_m,
            "capture_buffer_size": args.capture_buffer_size,
            "capture_batch_fraction": args.capture_batch_fraction,
            "minimum_capture_transitions_before_balancing": (
                args.minimum_capture_transitions
            ),
            "minimum_completed_capture_episodes_before_balancing": (
                args.minimum_capture_episodes
            ),
            "protected_priority_buffer_size": args.priority_buffer_size,
            "priority_fraction_within_capture": (
                args.priority_capture_fraction
            ),
            "priority_events": [
                "one true closest-approach transition per completed episode",
                "velocity stopping-margin crossing",
                "axial overshoot onset",
                "successful-braking onset while approaching",
                "P3 success",
            ],
            "fallback_before_activation": "ordinary global uniform replay",
        },
        "vecnormalize": {
            "loaded_from_source": True,
            "training_updates_enabled": False,
            "statistics_frozen_during_warmup_and_training": True,
            "norm_obs": True,
            "norm_reward": True,
            "gamma": TRIAL_16_PARAMS["gamma"],
        },
        "checkpoint_selection": [
            "highest success rate",
            "lowest safety-failure rate",
            "lowest mean minimum P3 position error",
            "lowest final max target ratio",
            "highest safety-first score",
            "highest mean return",
        ],
    }


def make_phase_buffer(model, args):
    kwargs = {
        "capture_buffer_size": args.capture_buffer_size,
        "priority_buffer_size": args.priority_buffer_size,
        "capture_distance_m": args.capture_distance_m,
        "capture_batch_fraction": args.capture_batch_fraction,
        "priority_capture_fraction": args.priority_capture_fraction,
        "minimum_capture_transitions": args.minimum_capture_transitions,
        "minimum_capture_episodes": args.minimum_capture_episodes,
    }
    replay_buffer = PhaseBalancedReplayBuffer(
        buffer_size=args.global_buffer_size,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        **kwargs,
    )
    model.buffer_size = args.global_buffer_size
    model.replay_buffer_class = PhaseBalancedReplayBuffer
    model.replay_buffer_kwargs = kwargs
    model.replay_buffer = replay_buffer
    return replay_buffer


def main():
    args = parse_args()
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)

    source_probe = SAC.load(args.source_model, device="cpu")
    source_num_timesteps = int(source_probe.num_timesteps)
    del source_probe
    config = build_config(args, checkpoints, source_num_timesteps)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir, result_dir = seed_output_paths(
        args.experiment_name,
        args.training_seed,
    )
    log_dir = os.path.join(
        SCRIPT_DIR,
        "logs",
        "multiseed",
        args.experiment_name,
    )
    for path in (model_dir, result_dir, log_dir):
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        os.makedirs(path)
    write_json(
        os.path.join(
            RESULTS_ROOT,
            args.experiment_name,
            "experiment_config.json",
        ),
        config,
    )

    training_env = DummyVecEnv(
        [
            lambda: make_training_env(
                seed=args.training_seed,
                closing_reward_version="v15",
            )
        ]
    )
    training_env = VecNormalize.load(
        args.source_vecnormalize,
        training_env,
    )
    # Freezing these statistics is part of freezing the source policy: changing
    # observation normalization changes the action even when weights do not.
    training_env.training = False
    training_env.norm_reward = True

    model = SAC.load(
        args.source_model,
        env=training_env,
        device=args.device,
        seed=args.training_seed,
        tensorboard_log=log_dir,
    )
    replay_buffer = make_phase_buffer(model, args)

    checkpoint_callback = PhaseBalancedCheckpointCallback(
        training_seed=args.training_seed,
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        closing_reward_version="v15",
        model_dir=model_dir,
        result_dir=result_dir,
        model_name=f"{args.experiment_name} - seed {args.training_seed}",
        rank_function=safety_capture_checkpoint_rank,
        selection_metric=(
            "success, safety failure rate, closest P3 approach, target ratio, "
            "safety-first score, return"
        ),
    )
    training_return_callback = TrainingReturnCallback(
        training_seed=args.training_seed,
        result_dir=result_dir,
        model_name=f"{args.experiment_name} - seed {args.training_seed}",
    )
    callbacks = [training_return_callback, checkpoint_callback]

    print("=" * 80)
    print("Terminal V15 SAC continuation with phase-balanced online replay")
    print("=" * 80)
    print(f"Experiment:        {args.experiment_name}")
    print(f"Source steps:      {source_num_timesteps:,}")
    print(f"Training steps:    {args.training_timesteps:,}")
    print(
        "Replay warmup:     adaptive, at most "
        f"{args.max_replay_warmup_steps:,} interactions"
    )
    print(
        "Warmup gate:       "
        f"{args.minimum_capture_transitions:,} capture transitions from "
        f"{args.minimum_capture_episodes} completed capture episodes"
    )
    print("Safety violations: terminal, matching source training")
    print("VecNormalize:       frozen at source statistics")
    print(
        "Replay batches:     "
        f"{1.0 - args.capture_batch_fraction:.0%} global / "
        f"{args.capture_batch_fraction:.0%} capture after activation"
    )
    print(f"Capture boundary:  {args.capture_distance_m:g} m")
    print(
        "Continuation:      "
        f"lr={args.continuation_learning_rate:g}, "
        f"gradient_steps={args.continuation_gradient_steps}"
    )
    print(f"Checkpoints:       {[0, *checkpoints]}")
    print(f"Validation seeds:  {args.validation_seeds}")
    print("=" * 80)

    try:
        readiness_callback = ReplayReadinessCallback()
        model.learning_starts = 0
        model.gradient_steps = 0
        model.learn(
            total_timesteps=args.max_replay_warmup_steps,
            reset_num_timesteps=True,
            tb_log_name=f"{args.experiment_name}_frozen_replay_warmup",
            progress_bar=args.progress_bar,
            callback=readiness_callback,
        )
        warmup_steps = int(model.num_timesteps)
        warmup_statistics = replay_buffer.replay_statistics()
        warmup_record = {
            "ready_for_gradient_training": bool(readiness_callback.ready),
            "warmup_interaction_steps": warmup_steps,
            **warmup_statistics,
        }
        write_json(
            os.path.join(result_dir, "replay_buffer_after_warmup.json"),
            warmup_record,
        )
        config["actual_replay_warmup_interaction_steps"] = warmup_steps
        config["replay_ready_after_warmup"] = bool(
            readiness_callback.ready
        )
        write_json(
            os.path.join(
                RESULTS_ROOT,
                args.experiment_name,
                "experiment_config.json",
            ),
            config,
        )
        if not readiness_callback.ready:
            raise RuntimeError(
                "Replay warmup reached its maximum without collecting "
                f"{args.minimum_capture_transitions:,} capture transitions "
                f"from {args.minimum_capture_episodes} completed capture "
                "episodes. No gradient update was performed."
            )

        model.learning_starts = 0
        model.gradient_steps = args.continuation_gradient_steps
        model.learning_rate = args.continuation_learning_rate
        model.lr_schedule = get_schedule_fn(
            args.continuation_learning_rate
        )
        model.learn(
            total_timesteps=args.training_timesteps,
            reset_num_timesteps=True,
            tb_log_name=args.experiment_name,
            progress_bar=args.progress_bar,
            callback=callbacks,
        )
        model.save(os.path.join(model_dir, "final_model"))
        training_env.save(
            os.path.join(model_dir, "final_vecnormalize.pkl")
        )
        write_json(
            os.path.join(result_dir, "replay_buffer_final.json"),
            replay_buffer.replay_statistics(),
        )
    finally:
        training_env.close()

    best_checkpoint, best_mean_return, best_episode = (
        checkpoint_callback.get_best_checkpoint()
    )
    training_return_callback.set_best_checkpoint(
        best_episode,
        best_checkpoint,
    )
    training_return_callback.save_outputs()
    write_aggregate_outputs(args.experiment_name, [args.training_seed])
    print(f"Best additional checkpoint: {best_checkpoint}")
    print(f"Best validation return:      {best_mean_return:.3f}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
