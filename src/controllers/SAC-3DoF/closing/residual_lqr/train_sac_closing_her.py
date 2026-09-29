"""Train a goal-conditioned closing SAC agent with future-goal HER."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from her_closing_utils import (
    HER_ENV_CLASSES,
    checkpoint_rank,
    evaluate_model,
    make_her_env,
    summarize_rollouts,
    write_csv,
    write_json,
)


MODEL_ROOT = Path(__file__).resolve().parent / "models" / "her"
RESULT_ROOT = Path(__file__).resolve().parent / "results" / "her"
LOG_ROOT = Path(__file__).resolve().parent / "logs" / "her"


class HerEvaluationCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        checkpoint_steps,
        validation_seeds,
        max_eval_steps,
        model_dir,
        result_dir,
        initial_rank,
        initial_summary,
        initial_rows,
        reward_version,
        env_kwargs=None,
    ):
        super().__init__(verbose=0)
        self.checkpoint_steps = list(checkpoint_steps)
        self.validation_seeds = list(validation_seeds)
        self.max_eval_steps = int(max_eval_steps)
        self.model_dir = Path(model_dir)
        self.result_dir = Path(result_dir)
        self.next_checkpoint_index = 0
        self.best_rank = initial_rank
        self.best_checkpoint = 0
        self.checkpoint_summaries = [{"checkpoint": 0, **initial_summary}]
        self.rollout_rows = [
            {"checkpoint": 0, **row} for row in initial_rows
        ]
        self.reward_version = reward_version
        self.env_kwargs = dict(env_kwargs or {})

    def _save_tables(self):
        write_csv(
            self.result_dir / "checkpoint_metrics.csv",
            self.checkpoint_summaries,
        )
        write_csv(
            self.result_dir / "validation_rollout_metrics.csv",
            self.rollout_rows,
        )

    def _evaluate_checkpoint(self, checkpoint):
        self.model.save(self.model_dir / f"checkpoint_{checkpoint}_model")
        rows = evaluate_model(
            self.model,
            self.validation_seeds,
            max_steps=self.max_eval_steps,
            reward_version=self.reward_version,
            env_kwargs=self.env_kwargs,
        )
        summary = summarize_rollouts(rows)
        rank = checkpoint_rank(summary)
        self.checkpoint_summaries.append(
            {"checkpoint": int(checkpoint), **summary}
        )
        self.rollout_rows.extend(
            {"checkpoint": int(checkpoint), **row} for row in rows
        )
        if rank > self.best_rank:
            self.best_rank = rank
            self.best_checkpoint = int(checkpoint)
            self.model.save(self.model_dir / "best_model")
            write_json(
                self.result_dir / "best_checkpoint.json",
                {
                    "checkpoint": self.best_checkpoint,
                    "selection": (
                        "success, safety, best simultaneous target ratio, "
                        "2x dwell, final target ratio, return"
                    ),
                    "summary": summary,
                },
            )
        self._save_tables()
        print(
            f"HER checkpoint {checkpoint:,}: "
            f"success={summary['success_rate']:.3f}, "
            f"failure={summary['failure_rate']:.3f}, "
            f"best ratio={summary['mean_best_max_target_ratio']:.2f}, "
            f"final ratio={summary['mean_final_max_target_ratio']:.2f}"
        )

    def _on_step(self):
        while (
            self.next_checkpoint_index < len(self.checkpoint_steps)
            and self.num_timesteps
            >= self.checkpoint_steps[self.next_checkpoint_index]
        ):
            checkpoint = self.checkpoint_steps[self.next_checkpoint_index]
            self._evaluate_checkpoint(checkpoint)
            self.next_checkpoint_index += 1
        return True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train standard SAC with future-goal HER for P3 closing."
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--reward-version",
        choices=tuple(HER_ENV_CLASSES),
        default="v1",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[25_000, 50_000, 100_000],
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
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=12_001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.6276728418436003e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--train-freq", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=8)
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
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if args.max_time <= 0 or args.max_eval_steps <= 0:
        parser.error("Episode limits must be positive.")
    maximum_episode_steps = int(args.max_time)
    if args.learning_starts <= maximum_episode_steps:
        parser.error(
            "HER requires --learning-starts to exceed the maximum episode "
            f"length ({maximum_episode_steps} steps)."
        )
    if args.buffer_size <= maximum_episode_steps:
        parser.error("--buffer-size must exceed the maximum episode length.")
    if args.n_sampled_goal <= 0:
        parser.error("--n-sampled-goal must be positive.")
    checkpoints = sorted(
        set(
            step
            for step in args.checkpoint_steps
            if 0 < step <= args.total_timesteps
        )
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)
    return checkpoints


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoints = validate_args(parser, args)
    experiment_name = args.experiment_name or (
        f"sac_closing_goal_her_{args.reward_version}_pilot_seed"
        f"{args.training_seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    reward_env_class = HER_ENV_CLASSES[args.reward_version]

    config = {
        "experiment_name": experiment_name,
        "method": "goal-conditioned SAC with Hindsight Experience Replay",
        "algorithm": "stable_baselines3.SAC MultiInputPolicy",
        "reward_version": args.reward_version,
        "reward_scheme": reward_env_class.reward_scheme,
        "reward_parameters": {
            "goal_ratio_clip": reward_env_class.goal_ratio_clip,
            "goal_log_reference_ratio": (
                reward_env_class.goal_log_reference_ratio
            ),
            "goal_cost_weight": reward_env_class.goal_cost_weight,
            "success_bonus": reward_env_class.success_bonus,
            "safety_failure_penalty": (
                reward_env_class.safety_failure_penalty
            ),
            "velocity_warning_penalty": (
                reward_env_class.velocity_warning_penalty
            ),
            "geometry_warning_penalty": (
                reward_env_class.geometry_warning_penalty
            ),
            "alignment_warning_penalty": (
                reward_env_class.alignment_warning_penalty
            ),
            "timeout_penalty": reward_env_class.timeout_penalty,
        },
        "reward_history_dependent": False,
        "training_environment": (
            "original P2 distribution with hard safety termination"
        ),
        "goal_success_termination_during_training": False,
        "goal_success_evaluation": "first entry into the exact P3 set",
        "near_p3_resets_used": False,
        "curriculum_used": False,
        "cbf_filter_used_during_training": False,
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
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

    model_dir = MODEL_ROOT / experiment_name
    result_dir = RESULT_ROOT / experiment_name
    log_dir = LOG_ROOT / experiment_name
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
                    reward_version=args.reward_version,
                )
            )
        ]
    )
    model = SAC(
        policy="MultiInputPolicy",
        env=training_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
            "copy_info_dict": True,
        },
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
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

    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version=args.reward_version,
    )
    initial_summary = summarize_rollouts(initial_rows)
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
        reward_version=args.reward_version,
    )
    callback._save_tables()

    print(f"Experiment:       {experiment_name}")
    print("Environment:      original P2 reset, hard safety termination")
    print(
        f"Reward:           {reward_env_class.reward_scheme} "
        "bounded recomputable goal cost"
    )
    print(
        f"HER:              {args.n_sampled_goal} future goals per real goal"
    )
    print(f"Training steps:   {args.total_timesteps:,}")
    print(f"Checkpoints:      {checkpoints}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=experiment_name,
        progress_bar=args.progress_bar,
    )
    model.save(model_dir / "final_model")
    config["actual_timesteps"] = int(model.num_timesteps)
    config["best_checkpoint"] = int(callback.best_checkpoint)
    write_json(result_dir / "experiment_config.json", config)
    training_env.close()


if __name__ == "__main__":
    main()
