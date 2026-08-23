"""Train HER V3 after distilling and replaying the safe V15 closing policy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)
from her_closing_utils import (
    checkpoint_rank,
    evaluate_model,
    make_her_env,
    summarize_rollouts,
    write_csv,
    write_json,
)
from her_source_warmstart import (
    behavior_distill_source_actor,
    collect_safe_source_trajectories,
    load_source_policy,
    prefill_her_replay_buffer,
)
from train_sac_closing_her import HerEvaluationCheckpointCallback


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "her"
RESULT_ROOT = CLOSING_DIR / "results" / "her"
LOG_ROOT = CLOSING_DIR / "logs" / "her"
SOURCE_STEM = (
    "sac_closing_p3augobs_rewardv15_optuna_trial16_"
    "400ksteps_seed0_buffer300k"
)
DEFAULT_SOURCE_MODEL = (
    CLOSING_DIR / "models" / "reward_shaping_trials" / f"{SOURCE_STEM}.zip"
)
DEFAULT_SOURCE_VECNORMALIZE = (
    CLOSING_DIR
    / "models"
    / "reward_shaping_trials"
    / f"{SOURCE_STEM}_vecnormalize.pkl"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Safe-V15-initialized goal-conditioned SAC with HER V3."
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_goal_her_v3_source45m_guarded50k_seed0",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
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
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        type=Path,
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument("--source-transition-count", type=int, default=60_000)
    parser.add_argument("--source-seed-start", type=int, default=0)
    parser.add_argument("--source-max-episodes", type=int, default=12)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-validation-fraction", type=float, default=0.2)
    parser.add_argument("--bc-log-std-target", type=float, default=-3.0)
    parser.add_argument("--bc-log-std-weight", type=float, default=0.01)
    parser.add_argument(
        "--max-bc-validation-failure-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-bc-validation-min-position-error",
        type=float,
        default=250.0,
    )
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--train-freq", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.05")
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument(
        "--goal-selection-strategy",
        choices=("future", "final", "episode"),
        default="future",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(parser, args):
    for path in (args.source_model, args.source_vecnormalize):
        if not path.is_file():
            parser.error(f"Required source artifact does not exist: {path}")
    if args.total_timesteps <= 0 or args.source_transition_count <= 0:
        parser.error("Training and source transition counts must be positive.")
    if args.buffer_size <= args.source_transition_count:
        parser.error("Replay buffer must exceed the source prefill size.")
    if not 0.0 < args.bc_validation_fraction < 1.0:
        parser.error("--bc-validation-fraction must lie in (0, 1).")
    checkpoints = sorted(
        {
            step
            for step in args.checkpoint_steps
            if 0 < step <= args.total_timesteps
        }
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)
    return checkpoints


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoints = validate_args(parser, args)
    reward_class = GoalConditionedClosingV3Env
    reward_parameters = {
        key: getattr(reward_class, key)
        for key in (
            "goal_ratio_clip",
            "goal_log_reference_ratio",
            "goal_cost_weight",
            "success_bonus",
            "safety_failure_penalty",
            "velocity_warning_penalty",
            "geometry_warning_penalty",
            "alignment_warning_penalty",
            "timeout_penalty",
        )
    }
    config = {
        "experiment_name": args.experiment_name,
        "method": "safe V15 behavior initialized goal-conditioned SAC+HER",
        "reward_version": "v3",
        "reward_scheme": reward_class.reward_scheme,
        "reward_parameters": reward_parameters,
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize": str(args.source_vecnormalize.resolve()),
        "source_vecnormalize_sha256": file_sha256(args.source_vecnormalize),
        "source_behavior_reference": (
            "V15 Optuna trial16 400k; approximately 44-45 m final P3 error"
        ),
        "source_transition_count_requested": args.source_transition_count,
        "source_seed_start": args.source_seed_start,
        "source_max_episodes": args.source_max_episodes,
        "actor_initialization": "behavior distillation from safe V15 actions",
        "replay_initialization": "complete non-failure V15 trajectories",
        "bc": {
            "epochs": args.bc_epochs,
            "batch_size": args.bc_batch_size,
            "learning_rate": args.bc_learning_rate,
            "validation_fraction": args.bc_validation_fraction,
            "log_std_target": args.bc_log_std_target,
            "log_std_weight": args.bc_log_std_weight,
            "maximum_validation_failure_rate": (
                args.max_bc_validation_failure_rate
            ),
            "maximum_validation_min_position_error": (
                args.max_bc_validation_min_position_error
            ),
        },
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "goal_success_termination_during_training": False,
        "near_p3_resets_used": False,
        "curriculum_used": False,
        "cbf_filter_used_during_training": False,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": 0,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "ent_coef": args.ent_coef,
        },
        "her": {
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
            "copy_info_dict": True,
        },
    }
    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = MODEL_ROOT / args.experiment_name
    result_dir = RESULT_ROOT / args.experiment_name
    log_dir = LOG_ROOT / args.experiment_name
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    write_json(result_dir / "experiment_config.json", config)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_her_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    reward_version="v3",
                )
            )
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        training_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
            "copy_info_dict": True,
        },
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_update_interval=1,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    model.save(model_dir / "random_uninitialized_model")

    source_model, source_vecnormalize, source_dummy = load_source_policy(
        args.source_model,
        args.source_vecnormalize,
        device=args.device,
    )
    try:
        transitions, source_episode_rows = collect_safe_source_trajectories(
            source_model=source_model,
            source_vecnormalize=source_vecnormalize,
            minimum_transitions=args.source_transition_count,
            seed_start=args.source_seed_start,
            max_episodes=args.source_max_episodes,
            max_steps=args.max_time,
        )
    finally:
        source_dummy.close()
    write_csv(result_dir / "source_episode_collection.csv", source_episode_rows)
    prefill_metrics = prefill_her_replay_buffer(model, transitions)
    bc_metrics = behavior_distill_source_actor(
        model=model,
        transitions=transitions,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        learning_rate=args.bc_learning_rate,
        validation_fraction=args.bc_validation_fraction,
        split_seed=args.training_seed + 10_000,
        log_std_target=args.bc_log_std_target,
        log_std_weight=args.bc_log_std_weight,
        result_path=result_dir / "behavior_distillation_learning_curve.csv",
    )
    model.save(model_dir / "distilled_source_model")

    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version="v3",
    )
    initial_summary = summarize_rollouts(initial_rows)
    write_csv(result_dir / "distilled_validation_rollouts.csv", initial_rows)
    write_json(
        result_dir / "source_initialization.json",
        {
            **prefill_metrics,
            "behavior_distillation": bc_metrics,
            "distilled_validation": initial_summary,
        },
    )
    if initial_summary["failure_rate"] > args.max_bc_validation_failure_rate:
        raise RuntimeError(
            "Distilled actor failed the safety gate: validation failure rate "
            f"{initial_summary['failure_rate']:.3f} exceeds "
            f"{args.max_bc_validation_failure_rate:.3f}."
        )
    if (
        initial_summary["mean_minimum_position_error"]
        > args.max_bc_validation_min_position_error
    ):
        raise RuntimeError(
            "Distilled actor failed the approach gate: mean minimum P3 error "
            f"{initial_summary['mean_minimum_position_error']:.2f} m exceeds "
            f"{args.max_bc_validation_min_position_error:.2f} m."
        )

    initial_rank = checkpoint_rank(initial_summary)
    model.save(model_dir / "checkpoint_0_model")
    model.save(model_dir / "best_model")
    write_json(
        result_dir / "best_checkpoint.json",
        {
            "checkpoint": 0,
            "selection": (
                "success, safety, best simultaneous target ratio, 2x dwell, "
                "final target ratio, return"
            ),
            "summary": initial_summary,
        },
    )
    callback = HerEvaluationCheckpointCallback(
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        model_dir=model_dir,
        result_dir=result_dir,
        initial_rank=initial_rank,
        initial_summary=initial_summary,
        initial_rows=initial_rows,
        reward_version="v3",
    )
    callback._save_tables()

    print(f"Experiment:       {args.experiment_name}")
    print(
        f"Safe prefill:      {prefill_metrics['prefill_transition_count']:,} "
        f"transitions / {prefill_metrics['prefill_episode_count']} episodes"
    )
    print(
        "Distilled gate:   "
        f"failure={initial_summary['failure_rate']:.3f}, "
        f"min P3 error={initial_summary['mean_minimum_position_error']:.2f} m"
    )
    print(f"Online HER steps: {args.total_timesteps:,}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=args.experiment_name,
        progress_bar=args.progress_bar,
    )
    model.save(model_dir / "final_model")
    config["actual_timesteps"] = int(model.num_timesteps)
    config["best_checkpoint"] = int(callback.best_checkpoint)
    config["source_prefill"] = prefill_metrics
    config["behavior_distillation"] = bc_metrics
    write_json(result_dir / "experiment_config.json", config)
    training_env.close()


if __name__ == "__main__":
    main()
