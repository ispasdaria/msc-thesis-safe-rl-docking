"""Behavior-clone the LQR policy, warm up SAC critics, then train online.

Experiment stages
-----------------
1. Split complete successful LQR episodes into BC train/validation groups.
2. Train the SAC actor directly on normalized observation/action pairs.
3. Evaluate and save the pure behavior-cloned neural policy.
4. Keep the actor and normalization fixed while warming SAC critics online.
5. Continue ordinary SAC with protected 50% -> 25% mixed LQR replay.

LQR is never called during online interaction, evaluation, or deployment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
for import_path in (SRC_PATH, SCRIPT_DIR):
    import_path_string = str(import_path)
    if import_path_string not in sys.path:
        sys.path.insert(0, import_path_string)

from behavior_cloned_warmup_sac import (  # noqa: E402
    BehaviorClonedWarmupSAC,
)
from generate_lqr_replay_dataset import (  # noqa: E402
    load_demonstration_dataset,
)
from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_mixed_replay import (  # noqa: E402
    resolve_online_buffer_size,
)
from train_sac_closing_lqr_replay import (  # noqa: E402
    FullClosingCheckpointCallback,
    evaluate_full_closing_policy,
    initialize_vecnormalize_from_demonstrations,
    make_closing_env,
    resolve_checkpoint_steps,
    summarize_evaluation,
    write_json,
)
from train_sac_closing_multiseed import (  # noqa: E402
    TRIAL_16_PARAMS,
    TrainingReturnCallback,
)


MODEL_ROOT = SCRIPT_DIR / "models" / "lqr_bc_warmstart"
LOG_ROOT = SCRIPT_DIR / "logs" / "lqr_bc_warmstart"
RESULTS_ROOT = SCRIPT_DIR / "results" / "lqr_bc_warmstart"
DEFAULT_VALIDATION_SEEDS = list(range(1000, 1020))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Behavior-clone SAC from successful LQR trajectories, warm its "
            "critics with a frozen actor, then continue ordinary online SAC."
        )
    )
    parser.add_argument("--demonstrations", type=Path, required=True)
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
    parser.add_argument("--online-buffer-size", type=int, default=None)
    parser.add_argument("--online-buffer-margin", type=int, default=50_000)
    parser.add_argument("--initial-demo-fraction", type=float, default=0.50)
    parser.add_argument("--later-demo-fraction", type=float, default=0.25)
    parser.add_argument(
        "--demo-fraction-switch-step",
        type=int,
        default=None,
    )
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)
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
        "--skip-bc-policy-evaluation",
        action="store_true",
        help="Skip the pre-online closed-loop evaluation (useful for smoke tests).",
    )
    args = parser.parse_args()

    positive_values = {
        "total-timesteps": args.total_timesteps,
        "bc-epochs": args.bc_epochs,
        "bc-batch-size": args.bc_batch_size,
        "bc-learning-rate": args.bc_learning_rate,
        "bc-patience": args.bc_patience,
        "bc-action-clip": args.bc_action_clip,
        "bc-max-grad-norm": args.bc_max_grad_norm,
        "max-eval-steps": args.max_eval_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive.")
    if not 0.0 < args.bc_validation_fraction < 1.0:
        parser.error("--bc-validation-fraction must lie strictly between 0 and 1.")
    if not 0.0 < args.bc_action_clip < 1.0:
        parser.error("--bc-action-clip must lie strictly between 0 and 1.")
    if args.bc_min_delta < 0 or args.bc_log_std_weight < 0:
        parser.error("BC delta and log-standard-deviation weight must be non-negative.")
    if not 0 <= args.critic_warmup_steps <= args.total_timesteps:
        parser.error("--critic-warmup-steps must lie within online training.")
    if args.online_buffer_margin < 0:
        parser.error("--online-buffer-margin must be non-negative.")
    if args.online_buffer_size is not None and args.online_buffer_size <= 0:
        parser.error("--online-buffer-size must be positive.")
    for name in ("initial_demo_fraction", "later_demo_fraction"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if args.demo_fraction_switch_step is not None:
        if not 0 <= args.demo_fraction_switch_step <= args.total_timesteps:
            parser.error("The replay fraction switch must lie within training.")
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


def safe_output_paths(experiment_name, seed, overwrite):
    model_dir = MODEL_ROOT / experiment_name / f"seed_{seed}"
    result_dir = RESULTS_ROOT / experiment_name / f"seed_{seed}"
    protected = (
        model_dir / "bc_actor_state.pt",
        model_dir / "final_model.zip",
        result_dir / "bc_learning_curve.csv",
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


def split_demonstration_episodes(dataset, validation_fraction, split_seed):
    offsets = np.asarray(dataset["episode_offsets"], dtype=np.int64)
    episode_seeds = np.asarray(dataset["episode_seeds"], dtype=np.int64)
    episode_count = len(episode_seeds)
    if episode_count < 2:
        raise ValueError("Behavior cloning requires at least two complete episodes.")

    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(episode_count)
    validation_count = int(round(episode_count * validation_fraction))
    validation_count = int(np.clip(validation_count, 1, episode_count - 1))
    validation_episodes = np.sort(permutation[:validation_count])
    training_episodes = np.sort(permutation[validation_count:])

    def transition_indices(episode_indices):
        return np.concatenate(
            [
                np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
                for index in episode_indices
            ]
        )

    training_indices = transition_indices(training_episodes)
    validation_indices = transition_indices(validation_episodes)
    metadata = {
        "split_seed": int(split_seed),
        "validation_fraction_requested": float(validation_fraction),
        "training_episode_ids": training_episodes.tolist(),
        "validation_episode_ids": validation_episodes.tolist(),
        "training_episode_seeds": episode_seeds[training_episodes].tolist(),
        "validation_episode_seeds": episode_seeds[validation_episodes].tolist(),
        "training_transition_count": int(len(training_indices)),
        "validation_transition_count": int(len(validation_indices)),
    }
    return training_indices, validation_indices, metadata


def normalized_demonstration_observations(vecnormalize, dataset):
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    return np.asarray(
        vecnormalize.normalize_obs(observations),
        dtype=np.float32,
    )


def evaluate_actor_actions(actor, observations, actions, indices, batch_size):
    squared_error_sum = 0.0
    absolute_error_sum = np.zeros(actions.shape[1], dtype=np.float64)
    value_count = 0
    row_count = 0
    actor.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            observation_batch = torch.as_tensor(
                observations[batch_indices],
                device=actor.device,
            )
            target_batch = torch.as_tensor(
                actions[batch_indices],
                device=actor.device,
            )
            prediction = actor(observation_batch, deterministic=True)
            error = prediction - target_batch
            squared_error_sum += float(torch.sum(error.square()).item())
            absolute_error_sum += np.sum(
                np.abs(error.detach().cpu().numpy()),
                axis=0,
            )
            value_count += int(error.numel())
            row_count += len(batch_indices)
    return {
        "action_mse": squared_error_sum / value_count,
        "action_mae": float(np.sum(absolute_error_sum) / value_count),
        "action_mae_ax": float(absolute_error_sum[0] / row_count),
        "action_mae_ay": float(absolute_error_sum[1] / row_count),
        "action_mae_alpha": float(absolute_error_sum[2] / row_count),
    }


def behavior_clone_actor(
    model,
    normalized_observations,
    actions,
    training_indices,
    validation_indices,
    args,
    result_dir,
):
    actor = model.actor
    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=args.bc_learning_rate,
    )
    target_actions = np.asarray(actions, dtype=np.float32)
    clipped_actions = np.clip(
        target_actions,
        -args.bc_action_clip,
        args.bc_action_clip,
    )
    latent_targets = np.arctanh(clipped_actions).astype(np.float32)
    action_loss_scales = np.asarray(
        getattr(args, "resolved_bc_action_loss_scales", np.ones(3)),
        dtype=np.float32,
    )
    rng = np.random.default_rng(args.bc_split_seed + model.seed)
    rows = []
    best_validation_mse = np.inf
    best_epoch = None
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.bc_epochs + 1):
        actor.set_training_mode(True)
        shuffled = rng.permutation(training_indices)
        latent_loss_sum = 0.0
        log_std_loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(shuffled), args.bc_batch_size):
            batch_indices = shuffled[start : start + args.bc_batch_size]
            observation_batch = torch.as_tensor(
                normalized_observations[batch_indices],
                device=model.device,
            )
            latent_target_batch = torch.as_tensor(
                latent_targets[batch_indices],
                device=model.device,
            )
            mean_actions, log_std, _ = actor.get_action_dist_params(
                observation_batch
            )
            scale_batch = torch.as_tensor(
                action_loss_scales,
                dtype=mean_actions.dtype,
                device=model.device,
            )
            scaled_error = (mean_actions - latent_target_batch) / scale_batch
            latent_loss = F.smooth_l1_loss(
                scaled_error,
                torch.zeros_like(scaled_error),
            )
            desired_log_std = torch.full_like(
                log_std,
                args.bc_log_std_target,
            )
            log_std_loss = F.mse_loss(log_std, desired_log_std)
            loss = latent_loss + args.bc_log_std_weight * log_std_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                args.bc_max_grad_norm,
            )
            optimizer.step()
            latent_loss_sum += float(latent_loss.item())
            log_std_loss_sum += float(log_std_loss.item())
            batch_count += 1

        validation_metrics = evaluate_actor_actions(
            actor,
            normalized_observations,
            target_actions,
            validation_indices,
            args.bc_batch_size,
        )
        row = {
            "epoch": epoch,
            "mean_latent_huber_loss": latent_loss_sum / batch_count,
            "mean_log_std_loss": log_std_loss_sum / batch_count,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        rows.append(row)
        print(
            f"BC epoch {epoch:02d}/{args.bc_epochs:02d} | "
            f"latent={row['mean_latent_huber_loss']:.6f} | "
            f"validation MSE={validation_metrics['action_mse']:.6f} | "
            f"validation MAE={validation_metrics['action_mae']:.6f}",
            flush=True,
        )

        improved = (
            validation_metrics["action_mse"]
            < best_validation_mse - args.bc_min_delta
        )
        if improved:
            best_validation_mse = validation_metrics["action_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.bc_patience:
                print(
                    f"BC early stopping after epoch {epoch}; "
                    f"best epoch was {best_epoch}.",
                    flush=True,
                )
                break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Behavior cloning did not produce a valid actor state.")
    actor.load_state_dict(best_state)
    final_training_metrics = evaluate_actor_actions(
        actor,
        normalized_observations,
        target_actions,
        training_indices,
        args.bc_batch_size,
    )
    final_validation_metrics = evaluate_actor_actions(
        actor,
        normalized_observations,
        target_actions,
        validation_indices,
        args.bc_batch_size,
    )

    curve_path = Path(result_dir) / "bc_learning_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_bc_learning_curve(rows, Path(result_dir) / "bc_learning_curve.png")
    return {
        "best_epoch": int(best_epoch),
        "epochs_completed": len(rows),
        "action_loss_scales": action_loss_scales.tolist(),
        "training": final_training_metrics,
        "validation": final_validation_metrics,
    }


def plot_bc_learning_curve(rows, output_path):
    epochs = np.asarray([row["epoch"] for row in rows])
    latent = np.asarray([row["mean_latent_huber_loss"] for row in rows])
    validation_mse = np.asarray(
        [row["validation_action_mse"] for row in rows]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, latent, marker="o")
    axes[0].set_xlabel("BC epoch")
    axes[0].set_ylabel("Mean pre-squash Huber loss")
    axes[0].set_title("Behavior-cloning optimization")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, validation_mse, marker="o", color="tab:orange")
    axes[1].set_xlabel("BC epoch")
    axes[1].set_ylabel("Validation action MSE")
    axes[1].set_title("Held-out complete episodes")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NormalizationFreezeCallback(BaseCallback):
    """Freeze demo-initialized normalization until critic warm-up ends."""

    def __init__(self, warmup_steps):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.released = self.warmup_steps == 0

    def _on_training_start(self):
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is None:
            raise RuntimeError("VecNormalize is required.")
        vecnormalize.training = self.released

    def _on_step(self):
        if not self.released and self.num_timesteps >= self.warmup_steps:
            vecnormalize = self.model.get_vec_normalize_env()
            if vecnormalize is None:
                raise RuntimeError("VecNormalize is required.")
            vecnormalize.training = True
            self.released = True
            print(
                "Critic warm-up complete: actor unfrozen by SAC and "
                "VecNormalize online updates enabled.",
                flush=True,
            )
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

        model = BehaviorClonedWarmupSAC(
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
            critic_warmup_steps=args.critic_warmup_steps,
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
                "Pure BC policy evaluation | "
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
                "normalization_frozen_during_critic_warmup": True,
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
        normalization_callback = NormalizationFreezeCallback(
            args.critic_warmup_steps
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
    checkpoint_steps = resolve_checkpoint_steps(
        args.checkpoint_steps,
        args.total_timesteps,
    )
    if 0 < args.critic_warmup_steps <= args.total_timesteps:
        checkpoint_steps = sorted(
            set(checkpoint_steps + [args.critic_warmup_steps])
        )
    if args.experiment_name is None:
        experiment_name = (
            f"sac_closing_reward{reward_version}_lqr_bc_warmstart_"
            f"mixed_replay_{args.total_timesteps}steps"
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
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_learning_rate,
        "bc_validation_fraction": args.bc_validation_fraction,
        "bc_patience": args.bc_patience,
        "bc_action_clip": args.bc_action_clip,
        "bc_log_std_target": args.bc_log_std_target,
        "bc_log_std_weight": args.bc_log_std_weight,
        "critic_warmup_steps": args.critic_warmup_steps,
        "normalization_frozen_during_critic_warmup": True,
        "checkpoint_steps": checkpoint_steps,
        "online_buffer_size": online_buffer_size,
        "initial_demo_fraction": args.initial_demo_fraction,
        "later_demo_fraction": args.later_demo_fraction,
        "demo_fraction_switch_step": switch_step,
        "full_p2_resets_only": True,
        "lqr_used_during_online_training": False,
        "trial_16_params": TRIAL_16_PARAMS,
    }
    write_json(
        RESULTS_ROOT / experiment_name / "experiment_config.json",
        config,
    )

    print("=" * 88)
    print("LQR behavior-cloned SAC with critic warm-up and protected mixed replay")
    print("=" * 88)
    print(f"Experiment:             {experiment_name}")
    print(f"Demonstrations:         {dataset_metadata['transition_count']:,}")
    print(f"Successful episodes:    {dataset_metadata['successful_episodes']}")
    print(
        f"BC:                     up to {args.bc_epochs} epochs, "
        f"{args.bc_validation_fraction:.0%} held-out episodes"
    )
    print(f"Critic warm-up:         {args.critic_warmup_steps:,} online steps")
    print(f"Online SAC timesteps:   {args.total_timesteps:,} per seed")
    print(
        "Demo batch schedule:    "
        f"{args.initial_demo_fraction:.0%} -> "
        f"{args.later_demo_fraction:.0%} at {switch_step:,} steps"
    )
    print("Online resets:          ordinary full training P2 distribution")
    print("Online LQR use:         none")
    print("=" * 88)

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
