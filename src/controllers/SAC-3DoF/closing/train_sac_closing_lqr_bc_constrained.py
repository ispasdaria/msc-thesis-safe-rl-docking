"""Train closing SAC with persistent behavior-cloning regularization.

This experiment preserves the successful behavior-cloned warm start while
allowing gradual online SAC fine-tuning. It is intentionally separate from the
original warm-start script so existing results and defaults remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
for import_path in (SRC_PATH, SCRIPT_DIR):
    import_path_string = str(import_path)
    if import_path_string not in sys.path:
        sys.path.insert(0, import_path_string)

from behavior_cloned_constrained_sac import (  # noqa: E402
    BehaviorClonedConstrainedSAC,
)
from generate_lqr_replay_dataset import load_demonstration_dataset  # noqa: E402
from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_bc_warmstart import (  # noqa: E402
    behavior_clone_actor,
    normalized_demonstration_observations,
    split_demonstration_episodes,
    write_rows,
)
from train_sac_closing_lqr_replay import (  # noqa: E402
    evaluate_full_closing_policy,
    initialize_vecnormalize_from_demonstrations,
    make_closing_env,
    summarize_evaluation,
    write_json,
)
from train_sac_closing_multiseed import (  # noqa: E402
    TRIAL_16_PARAMS,
    TrainingReturnCallback,
)


MODEL_ROOT = SCRIPT_DIR / "models" / "lqr_bc_constrained"
LOG_ROOT = SCRIPT_DIR / "logs" / "lqr_bc_constrained"
RESULTS_ROOT = SCRIPT_DIR / "results" / "lqr_bc_constrained"
DEFAULT_DEMONSTRATIONS = (
    SCRIPT_DIR
    / "demonstrations"
    / "lqr_closing_training_rewardv15_expanded_200successes.npz"
)
DEFAULT_VALIDATION_SEEDS = list(range(1000, 1020))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Behavior-clone SAC, warm its critics with a frozen actor, then "
            "fine-tune with a persistent behavior-cloning constraint."
        )
    )
    parser.add_argument(
        "--demonstrations",
        type=Path,
        default=DEFAULT_DEMONSTRATIONS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--reward-version", default=None)

    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--bc-validation-fraction", type=float, default=0.20)
    parser.add_argument("--bc-patience", type=int, default=5)
    parser.add_argument("--bc-min-delta", type=float, default=1e-5)
    parser.add_argument("--bc-action-clip", type=float, default=0.999)
    parser.add_argument("--bc-log-std-target", type=float, default=-3.0)
    parser.add_argument("--bc-log-std-weight", type=float, default=0.01)
    parser.add_argument("--bc-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--bc-split-seed", type=int, default=31_415)

    parser.add_argument("--critic-warmup-steps", type=int, default=50_000)
    parser.add_argument(
        "--deterministic-finetune-steps",
        type=int,
        default=50_000,
        help="Post-warmup steps that still collect deterministic actor actions.",
    )
    parser.add_argument("--bc-finetune-weight-start", type=float, default=10.0)
    parser.add_argument("--bc-finetune-weight-end", type=float, default=1.0)
    parser.add_argument(
        "--bc-finetune-decay-steps",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--bc-finetune-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--actor-learning-rate-scale",
        type=float,
        default=0.1,
        help="Actor rate as a fraction of the configured SAC learning rate.",
    )
    parser.add_argument(
        "--initial-ent-coef",
        type=float,
        default=0.001,
        help="Initial value for SAC's automatically learned entropy coefficient.",
    )

    parser.add_argument("--online-buffer-size", type=int, default=None)
    parser.add_argument("--online-buffer-margin", type=int, default=50_000)
    parser.add_argument("--initial-demo-fraction", type=float, default=0.50)
    parser.add_argument("--later-demo-fraction", type=float, default=0.50)
    parser.add_argument(
        "--demo-fraction-switch-step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--update-normalization-after-warmup",
        action="store_true",
        help="Opt into online VecNormalize updates after warmup; frozen is safer.",
    )

    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10_000,
        help="Checkpoint interval after critic warmup when explicit steps are absent.",
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
    parser.add_argument("--skip-bc-policy-evaluation", action="store_true")
    args = parser.parse_args()

    positive_values = {
        "total-timesteps": args.total_timesteps,
        "bc-epochs": args.bc_epochs,
        "bc-batch-size": args.bc_batch_size,
        "bc-learning-rate": args.bc_learning_rate,
        "bc-patience": args.bc_patience,
        "bc-action-clip": args.bc_action_clip,
        "bc-max-grad-norm": args.bc_max_grad_norm,
        "bc-finetune-batch-size": args.bc_finetune_batch_size,
        "actor-learning-rate-scale": args.actor_learning_rate_scale,
        "initial-ent-coef": args.initial_ent_coef,
        "evaluation-interval": args.evaluation_interval,
        "max-eval-steps": args.max_eval_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive.")
    non_negative_values = {
        "critic-warmup-steps": args.critic_warmup_steps,
        "deterministic-finetune-steps": args.deterministic_finetune_steps,
        "bc-finetune-weight-start": args.bc_finetune_weight_start,
        "bc-finetune-weight-end": args.bc_finetune_weight_end,
        "bc-finetune-decay-steps": args.bc_finetune_decay_steps,
        "bc-min-delta": args.bc_min_delta,
        "bc-log-std-weight": args.bc_log_std_weight,
        "online-buffer-margin": args.online_buffer_margin,
    }
    for name, value in non_negative_values.items():
        if value < 0:
            parser.error(f"--{name} must be non-negative.")
    if not 0.0 < args.bc_validation_fraction < 1.0:
        parser.error("--bc-validation-fraction must lie strictly between 0 and 1.")
    if not 0.0 < args.bc_action_clip < 1.0:
        parser.error("--bc-action-clip must lie strictly between 0 and 1.")
    for name in ("initial_demo_fraction", "later_demo_fraction"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if not 0 <= args.critic_warmup_steps <= args.total_timesteps:
        parser.error("--critic-warmup-steps must lie within the training run.")
    if args.demo_fraction_switch_step is not None and not (
        0 <= args.demo_fraction_switch_step <= args.total_timesteps
    ):
        parser.error("--demo-fraction-switch-step must lie within the run.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values.")
    if not args.validation_seeds or len(set(args.validation_seeds)) != len(
        args.validation_seeds
    ):
        parser.error("--validation-seeds must contain unique values.")
    if set(args.seeds) & set(args.validation_seeds):
        parser.error("Training and validation seeds must be disjoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    return args


def resolve_online_buffer_size(requested, total_timesteps, margin):
    if requested is not None:
        if requested < total_timesteps:
            raise ValueError(
                "The online buffer must retain all online transitions: "
                f"{requested} < {total_timesteps}."
            )
        return int(requested)
    required = total_timesteps + margin
    return int(np.ceil(required / 100_000) * 100_000)


def resolve_checkpoint_steps(args):
    if args.checkpoint_steps is not None:
        checkpoints = sorted(set(args.checkpoint_steps))
    else:
        first = max(args.critic_warmup_steps, args.evaluation_interval)
        checkpoints = list(
            range(first, args.total_timesteps + 1, args.evaluation_interval)
        )
        checkpoints.extend((args.critic_warmup_steps, args.total_timesteps))
        checkpoints = sorted(set(checkpoints))
    checkpoints = [step for step in checkpoints if step > 0]
    if not checkpoints or checkpoints[-1] > args.total_timesteps:
        raise ValueError("Checkpoint steps must lie inside the training run.")
    return checkpoints


def safe_output_paths(experiment_name, seed, overwrite):
    model_dir = MODEL_ROOT / experiment_name / f"seed_{seed}"
    result_dir = RESULTS_ROOT / experiment_name / f"seed_{seed}"
    protected = (
        model_dir / "bc_actor_state.pt",
        model_dir / "final_model.zip",
        result_dir / "checkpoint_metrics.csv",
    )
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Experiment output already exists. Choose another name or pass "
            f"--overwrite: {existing}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, result_dir


def evaluation_rank(summary):
    """Prefer success, then safety, then completion, then shaped return."""

    return (
        float(summary["success_rate"]),
        -float(summary["failure_rate"]),
        -float(summary["time_limit_rate"]),
        float(summary["mean_return"]),
    )


class SuccessFirstCheckpointCallback(BaseCallback):
    """Save exact checkpoints and choose the best by task outcome first."""

    def __init__(
        self,
        training_seed,
        checkpoint_steps,
        validation_seeds,
        max_eval_steps,
        reward_version,
        model_dir,
        result_dir,
    ):
        super().__init__()
        self.training_seed = int(training_seed)
        self.checkpoint_steps = list(checkpoint_steps)
        self.validation_seeds = list(validation_seeds)
        self.max_eval_steps = int(max_eval_steps)
        self.reward_version = reward_version
        self.model_dir = Path(model_dir)
        self.result_dir = Path(result_dir)
        self.next_checkpoint_index = 0
        self.completed_episodes = 0
        self.rows = []
        self.rollout_rows = []
        self.best_rank = None
        self.best_mean_return = -np.inf
        self.best_checkpoint = None
        self.best_checkpoint_episode = None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("episode") is not None:
                self.completed_episodes += 1
        while self.next_checkpoint_index < len(self.checkpoint_steps):
            checkpoint = self.checkpoint_steps[self.next_checkpoint_index]
            if self.num_timesteps < checkpoint:
                break
            self._evaluate_and_save(checkpoint)
            self.next_checkpoint_index += 1
        return True

    def _evaluate_and_save(self, checkpoint):
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is None:
            raise RuntimeError("VecNormalize is required for evaluation.")
        stem = f"model_{checkpoint:07d}_steps"
        self.model.save(self.model_dir / stem)
        vecnormalize.save(self.model_dir / f"{stem}_vecnormalize.pkl")

        results = evaluate_full_closing_policy(
            model=self.model,
            vecnormalize=vecnormalize,
            seeds=self.validation_seeds,
            max_steps=self.max_eval_steps,
            reward_version=self.reward_version,
        )
        summary = summarize_evaluation(results)
        row = {
            "training_seed": self.training_seed,
            "checkpoint": int(checkpoint),
            "episode": self.completed_episodes,
            **summary,
        }
        self.rows.append(row)
        self.rollout_rows.extend(
            {
                "training_seed": self.training_seed,
                "checkpoint": int(checkpoint),
                **result,
            }
            for result in results
        )
        self._write_csv(self.result_dir / "checkpoint_metrics.csv", self.rows)
        self._write_csv(
            self.result_dir / "validation_rollout_metrics.csv",
            self.rollout_rows,
        )

        rank = evaluation_rank(summary)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            self.best_mean_return = float(summary["mean_return"])
            self.best_checkpoint = int(checkpoint)
            self.best_checkpoint_episode = self.completed_episodes
            self.model.save(self.model_dir / "best_model")
            vecnormalize.save(self.model_dir / "best_vecnormalize.pkl")
            write_json(self.result_dir / "best_checkpoint.json", row)

        print(
            f"Checkpoint {checkpoint:,} | "
            f"return={summary['mean_return']:.3f} | "
            f"success={summary['success_rate']:.3f} | "
            f"failure={summary['failure_rate']:.3f} | "
            f"timeout={summary['time_limit_rate']:.3f}",
            flush=True,
        )

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def get_best_checkpoint(self):
        return (
            self.best_checkpoint,
            self.best_mean_return,
            self.best_checkpoint_episode,
        )


class NormalizationPolicyCallback(BaseCallback):
    """Keep demo normalization fixed, unless online updates are requested."""

    def __init__(self, warmup_steps, release_after_warmup):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.release_after_warmup = bool(release_after_warmup)
        self.released = False

    def _on_training_start(self):
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is None:
            raise RuntimeError("VecNormalize is required.")
        vecnormalize.training = False

    def _on_step(self):
        if (
            self.release_after_warmup
            and not self.released
            and self.num_timesteps >= self.warmup_steps
        ):
            vecnormalize = self.model.get_vec_normalize_env()
            if vecnormalize is None:
                raise RuntimeError("VecNormalize is required.")
            vecnormalize.training = True
            self.released = True
            print("VecNormalize online updates enabled.", flush=True)
        return True


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
        initialize_vecnormalize_from_demonstrations(
            train_env,
            dataset,
            gamma=TRIAL_16_PARAMS["gamma"],
        )

        model = BehaviorClonedConstrainedSAC(
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
            ent_coef=f"auto_{args.initial_ent_coef}",
            target_update_interval=1,
            replay_buffer_class=ProtectedMixedReplayBuffer,
            replay_buffer_kwargs={
                "initial_demo_fraction": args.initial_demo_fraction,
                "later_demo_fraction": args.later_demo_fraction,
                "demo_fraction_switch_step": switch_step,
            },
            critic_warmup_steps=args.critic_warmup_steps,
            deterministic_finetune_steps=args.deterministic_finetune_steps,
            bc_finetune_weight_start=args.bc_finetune_weight_start,
            bc_finetune_weight_end=args.bc_finetune_weight_end,
            bc_finetune_decay_steps=args.bc_finetune_decay_steps,
            bc_finetune_batch_size=args.bc_finetune_batch_size,
            bc_action_clip=args.bc_action_clip,
            bc_log_std_target=args.bc_log_std_target,
            bc_log_std_weight=args.bc_log_std_weight,
            actor_learning_rate_scale=args.actor_learning_rate_scale,
            verbose=1,
            seed=seed,
            tensorboard_log=str(LOG_ROOT / experiment_name),
            device=args.device,
        )
        replay_buffer = model.replay_buffer
        if not isinstance(replay_buffer, ProtectedMixedReplayBuffer):
            raise RuntimeError("Protected mixed replay buffer was not created.")
        replay_buffer.set_demonstrations(dataset)

        training_indices, validation_indices, split_metadata = (
            split_demonstration_episodes(
                dataset,
                validation_fraction=args.bc_validation_fraction,
                split_seed=args.bc_split_seed + seed,
            )
        )
        write_json(result_dir / "bc_episode_split.json", split_metadata)
        normalized_observations = normalized_demonstration_observations(
            train_env,
            dataset,
        )
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        bc_metrics = behavior_clone_actor(
            model=model,
            normalized_observations=normalized_observations,
            actions=actions,
            training_indices=training_indices,
            validation_indices=validation_indices,
            args=args,
            result_dir=result_dir,
        )
        write_json(result_dir / "bc_action_metrics.json", bc_metrics)
        torch.save(model.actor.state_dict(), model_dir / "bc_actor_state.pt")
        model.save(model_dir / "bc_initialized_model")
        train_env.save(model_dir / "bc_vecnormalize.pkl")

        if not args.skip_bc_policy_evaluation:
            bc_rollouts = evaluate_full_closing_policy(
                model=model,
                vecnormalize=train_env,
                seeds=args.validation_seeds,
                max_steps=args.max_eval_steps,
                reward_version=reward_version,
            )
            bc_summary = summarize_evaluation(bc_rollouts)
            write_rows(
                result_dir / "bc_policy_validation_rollouts.csv",
                bc_rollouts,
            )
            write_json(
                result_dir / "bc_policy_validation_summary.json",
                bc_summary,
            )
            print(
                "Pure BC evaluation | "
                f"return={bc_summary['mean_return']:.3f} | "
                f"success={bc_summary['success_rate']:.3f} | "
                f"failure={bc_summary['failure_rate']:.3f} | "
                f"timeout={bc_summary['time_limit_rate']:.3f}",
                flush=True,
            )

        write_json(
            result_dir / "stage_initialization.json",
            {
                "dataset_path": str(args.demonstrations.resolve()),
                "dataset_metadata": dataset_metadata,
                "bc_metrics": bc_metrics,
                "critic_warmup_steps": args.critic_warmup_steps,
                "normalization_frozen_for_full_run": not (
                    args.update_normalization_after_warmup
                ),
                **replay_buffer.mixing_statistics(),
            },
        )

        checkpoint_callback = SuccessFirstCheckpointCallback(
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
        normalization_callback = NormalizationPolicyCallback(
            warmup_steps=args.critic_warmup_steps,
            release_after_warmup=args.update_normalization_after_warmup,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=f"{experiment_name}_seed_{seed}",
            progress_bar=args.progress_bar,
            callback=[
                normalization_callback,
                training_return_callback,
                checkpoint_callback,
            ],
        )
        model.save(model_dir / "final_model")
        train_env.save(model_dir / "final_vecnormalize.pkl")
        write_json(
            result_dir / "final_training_state.json",
            {
                "critic_warmup_gradient_updates": int(
                    model.critic_warmup_gradient_updates
                ),
                "constrained_actor_gradient_updates": int(
                    model.constrained_actor_gradient_updates
                ),
                "final_bc_finetune_weight": float(model.bc_finetune_weight),
                **replay_buffer.mixing_statistics(),
            },
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
    checkpoint_steps = resolve_checkpoint_steps(args)
    experiment_name = args.experiment_name or (
        f"sac_closing_reward{reward_version}_lqr_bc_constrained_"
        f"{args.total_timesteps}steps"
    )

    config = {
        "experiment_name": experiment_name,
        "demonstration_dataset": str(args.demonstrations.resolve()),
        "demonstration_metadata": dataset_metadata,
        "training_seeds": args.seeds,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "reward_version": reward_version,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_learning_rate,
        "bc_validation_fraction": args.bc_validation_fraction,
        "bc_patience": args.bc_patience,
        "bc_action_clip": args.bc_action_clip,
        "bc_log_std_target": args.bc_log_std_target,
        "bc_log_std_weight": args.bc_log_std_weight,
        "critic_warmup_steps": args.critic_warmup_steps,
        "deterministic_finetune_steps": args.deterministic_finetune_steps,
        "bc_finetune_weight_start": args.bc_finetune_weight_start,
        "bc_finetune_weight_end": args.bc_finetune_weight_end,
        "bc_finetune_decay_steps": args.bc_finetune_decay_steps,
        "bc_finetune_batch_size": args.bc_finetune_batch_size,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "initial_ent_coef": args.initial_ent_coef,
        "normalization_frozen_for_full_run": not (
            args.update_normalization_after_warmup
        ),
        "checkpoint_selection": [
            "success_rate",
            "negative_failure_rate",
            "negative_timeout_rate",
            "mean_return",
        ],
        "checkpoint_steps": checkpoint_steps,
        "online_buffer_size": online_buffer_size,
        "initial_demo_fraction": args.initial_demo_fraction,
        "later_demo_fraction": args.later_demo_fraction,
        "demo_fraction_switch_step": switch_step,
        "full_p2_resets_only": True,
        "lqr_used_during_online_training": False,
        "trial_16_params": {
            **TRIAL_16_PARAMS,
            "ent_coef": f"auto_{args.initial_ent_coef}",
        },
    }
    write_json(RESULTS_ROOT / experiment_name / "experiment_config.json", config)

    print("=" * 88)
    print("Behavior-constrained LQR warm-start SAC")
    print("=" * 88)
    print(f"Experiment:              {experiment_name}")
    print(f"Demonstrations:          {dataset_metadata['transition_count']:,}")
    print(f"Successful episodes:     {dataset_metadata['successful_episodes']}")
    print(f"Critic warm-up:          {args.critic_warmup_steps:,} steps")
    print(
        "BC fine-tune weight:    "
        f"{args.bc_finetune_weight_start:g} -> "
        f"{args.bc_finetune_weight_end:g} over "
        f"{args.bc_finetune_decay_steps:,} steps"
    )
    print(f"Actor LR scale:          {args.actor_learning_rate_scale:g}")
    print(
        "Deterministic handoff:  "
        f"{args.deterministic_finetune_steps:,} post-warmup steps"
    )
    print(
        "Demo batch schedule:    "
        f"{args.initial_demo_fraction:.0%} -> "
        f"{args.later_demo_fraction:.0%} at {switch_step:,}"
    )
    print(
        "Normalization:          "
        + (
            "released after warmup"
            if args.update_normalization_after_warmup
            else "frozen for full run"
        )
    )
    print("Checkpoint selection:    success -> safety -> completion -> return")
    print("=" * 88, flush=True)

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
