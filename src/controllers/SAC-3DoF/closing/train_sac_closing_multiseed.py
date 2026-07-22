"""Validate the Optuna trial-16 SAC configuration across training seeds.

This script trains independent closing-phase SAC agents with the best
configuration from Optuna study ``sac_closing_p3augobs_rewardv15_optuna_v2``.
It is intentionally a multi-seed validation experiment rather than a new
hyperparameter search.

Defaults:
    - training seeds: 0, 1, 2, 3, 4, 5
    - closing reward: v18 scale-corrected tolerance-normalized terminal progress
    - training steps: 500,000 per seed
    - checkpoints: every 100k through 500k
    - validation seeds: 100 through 109 (disjoint from Optuna seeds 0--4)

Each checkpoint records both deterministic evaluation return and the
safety-first score used by the Optuna study. As in the original training
script, the best intermediate model is selected by highest mean evaluation
return. Models, VecNormalize statistics, per-rollout diagnostics, and
aggregate cross-training-seed metrics are saved separately so runs cannot
overwrite one another.

Example:
    python train_sac_closing_multiseed.py --device cuda --progress-bar
"""

import argparse
import csv
import gc
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tune_sac_closing_optuna import (  # noqa: E402
    evaluate_model,
    make_training_env,
    score_summary,
    summarize_results,
)


MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", "multiseed")
LOG_ROOT = os.path.join(SCRIPT_DIR, "logs", "multiseed")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "multiseed")

# Exact Optuna trial-16 configuration. Keep these values fixed during the
# multi-seed experiment so it tests robustness rather than a new search.
TRIAL_16_PARAMS = {
    "learning_rate": 0.00026276728418436003,
    "batch_size": 512,
    "tau": 0.01,
    "gamma": 0.99,
    "learning_starts": 10_000,
    "train_freq": 16,
    "gradient_steps": 8,
    "ent_coef": "auto_0.05",
}

AGGREGATE_METRICS = (
    "score",
    "mean_return",
    "std_return",
    "success_rate",
    "failure_rate",
    "time_limit_rate",
    "mean_steps",
    "mean_final_position_error",
    "mean_min_position_error",
    "mean_progress_fraction",
    "mean_final_lateral_error",
    "mean_final_velocity_error",
    "mean_final_attitude_error_deg",
    "mean_final_rate_error_degs",
    "mean_final_position_ratio",
    "mean_final_velocity_ratio",
    "mean_final_alignment_ratio",
    "mean_final_max_target_ratio",
    "mean_warning_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate the closing-phase Optuna trial-16 SAC "
            "configuration over multiple training seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(6)),
        help="Independent SAC training seeds (default: 0 1 2 3 4 5).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Training steps per seed (default: 500000).",
    )
    parser.add_argument(
        "--reward-version",
        choices=["v15", "v16", "v17", "v18"],
        default="v18",
        help=(
            "Closing reward implementation to use. v18 uses v17's smooth "
            "tolerance-normalized P3 progress at scale 500 (default: v18)."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Evaluation/model checkpoint steps. By default, checkpoints are "
            "made every 100k and at the final timestep."
        ),
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
        help=(
            "Fixed held-out environment seeds used for checkpoint selection "
            "(default: 100 through 109)."
        ),
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=12_000,
        help="Maximum steps in each deterministic validation rollout.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300_000,
        help="Replay-buffer capacity used by the Optuna study.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Output-directory name. A descriptive name is generated by default.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip seeds whose final model and checkpoint metrics already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow existing output files for incomplete/requested seeds to be "
            "overwritten. Combine with --skip-completed after an interrupted "
            "multi-seed experiment."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.max_eval_steps <= 0:
        raise ValueError("--max-eval-steps must be positive.")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive.")
    if not args.seeds:
        raise ValueError("At least one training seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be unique.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if set(args.seeds) & set(args.validation_seeds):
        raise ValueError(
            "Training and validation seeds must be disjoint to avoid direct "
            "seed reuse."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )


def resolve_checkpoint_steps(args):
    if args.checkpoint_steps is None:
        requested = [
            100_000,
            200_000,
            300_000,
            400_000,
            500_000,
            args.total_timesteps,
        ]
    else:
        requested = [*args.checkpoint_steps, args.total_timesteps]

    checkpoints = sorted(
        {
            step
            for step in requested
            if 0 < step <= args.total_timesteps
        }
    )
    if not checkpoints or checkpoints[-1] != args.total_timesteps:
        raise RuntimeError("The final timestep must be a checkpoint.")
    return checkpoints


def format_step_count(total_timesteps):
    if total_timesteps % 1_000_000 == 0:
        return f"{total_timesteps // 1_000_000}M"
    if total_timesteps % 1_000 == 0:
        return f"{total_timesteps // 1_000}k"
    return str(total_timesteps)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


class TrainingReturnCallback(BaseCallback):
    """Persist episode-number versus raw Monitor return learning curves."""

    def __init__(
        self,
        training_seed,
        result_dir,
        model_name,
        moving_average_window=10,
        save_every_episodes=10,
    ):
        super().__init__()
        self.training_seed = int(training_seed)
        self.result_dir = result_dir
        self.model_name = model_name
        self.moving_average_window = int(moving_average_window)
        self.save_every_episodes = int(save_every_episodes)
        self.rows = []
        self.best_checkpoint_episode = None
        self.best_checkpoint_timestep = None
        self.csv_path = os.path.join(
            result_dir,
            "learning_curve.csv",
        )
        self.plot_path = os.path.join(
            result_dir,
            "episode_return_learning_curve.png",
        )

    def set_best_checkpoint(self, episode, timestep):
        if episode is None or timestep is None:
            return
        self.best_checkpoint_episode = int(episode)
        self.best_checkpoint_timestep = int(timestep)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            episode_return = float(episode_info["r"])
            previous_returns = [row["episode_return"] for row in self.rows]
            rolling_values = (previous_returns + [episode_return])[
                -self.moving_average_window :
            ]
            self.rows.append(
                {
                    "training_seed": self.training_seed,
                    "episode": len(self.rows) + 1,
                    "timestep": int(self.num_timesteps),
                    "episode_length": int(episode_info["l"]),
                    "episode_return": episode_return,
                    "rolling_mean_return": float(np.mean(rolling_values)),
                    "termination_reason": info.get("termination_reason"),
                }
            )

            if len(self.rows) % self.save_every_episodes == 0:
                self.save_outputs()

        return True

    def _on_training_end(self):
        self.save_outputs()

    def save_outputs(self):
        if not self.rows:
            return

        output_rows = [
            {
                **row,
                "is_best_checkpoint": int(
                    row["episode"] == self.best_checkpoint_episode
                ),
            }
            for row in self.rows
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

        episodes = np.asarray(
            [row["episode"] for row in self.rows],
            dtype=np.int64,
        )
        returns = np.asarray(
            [row["episode_return"] for row in self.rows],
            dtype=np.float64,
        )
        if len(returns) >= self.moving_average_window:
            weights = (
                np.ones(self.moving_average_window)
                / self.moving_average_window
            )
            averaged_returns = np.convolve(returns, weights, mode="valid")
            averaged_episodes = episodes[self.moving_average_window - 1 :]
        else:
            averaged_returns = np.asarray([], dtype=np.float64)
            averaged_episodes = np.asarray([], dtype=np.int64)
        plt.figure(figsize=(10, 6))
        plt.plot(
            episodes,
            returns,
            color="tab:blue",
            linewidth=1.0,
            alpha=0.8,
            label="Episode return",
        )
        if len(averaged_returns) > 0:
            plt.plot(
                averaged_episodes,
                averaged_returns,
                color="tab:orange",
                linewidth=2.0,
                label=f"{self.moving_average_window}-episode moving average",
            )
        if self.best_checkpoint_episode is not None:
            plt.axvline(
                self.best_checkpoint_episode,
                color="tab:red",
                linestyle="--",
                linewidth=1.6,
                label="Best deterministic checkpoint",
            )
            matching_rows = [
                row
                for row in self.rows
                if row["episode"] == self.best_checkpoint_episode
            ]
            if matching_rows:
                plt.scatter(
                    [self.best_checkpoint_episode],
                    [matching_rows[0]["episode_return"]],
                    color="tab:red",
                    s=45,
                    zorder=5,
                )
        plt.xlabel("Episode number")
        plt.ylabel("Total reward per episode")
        plt.title(f"Total Reward per Training Episode\n{self.model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


class EvaluationCheckpointCallback(BaseCallback):
    """Evaluate exact-timestep checkpoints and retain the best mean return."""

    def __init__(
        self,
        training_seed,
        checkpoint_steps,
        validation_seeds,
        max_eval_steps,
        closing_reward_version,
        model_dir,
        result_dir,
        model_name,
    ):
        super().__init__()
        self.training_seed = int(training_seed)
        self.checkpoint_steps = list(checkpoint_steps)
        self.validation_seeds = list(validation_seeds)
        self.max_eval_steps = int(max_eval_steps)
        self.closing_reward_version = closing_reward_version
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.model_name = model_name

        self.next_checkpoint_index = 0
        self.completed_episodes = 0
        self.checkpoint_rows = []
        self.evaluation_rows = []
        self.evaluation_results = []
        self.evaluation_lengths = []
        self.rollout_rows = []
        self.best_mean_return = -np.inf
        self.best_checkpoint = None
        self.best_checkpoint_episode = None

        self.checkpoint_csv = os.path.join(
            self.result_dir,
            "checkpoint_metrics.csv",
        )
        self.rollout_csv = os.path.join(
            self.result_dir,
            "validation_rollout_metrics.csv",
        )
        self.best_json = os.path.join(
            self.result_dir,
            "best_checkpoint.json",
        )
        self.plot_path = os.path.join(
            self.result_dir,
            "evaluation_learning_curve.png",
        )
        self.evaluation_csv = os.path.join(
            self.result_dir,
            "evaluation_learning_curve.csv",
        )
        self.evaluations_npz = os.path.join(
            self.result_dir,
            "evaluations.npz",
        )

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
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is None:
            raise RuntimeError("VecNormalize is required for validation.")

        checkpoint_stem = f"model_{checkpoint:07d}_steps"
        model_path = os.path.join(self.model_dir, checkpoint_stem)
        vecnorm_path = os.path.join(
            self.model_dir,
            f"{checkpoint_stem}_vecnormalize.pkl",
        )

        self.model.save(model_path)
        vecnorm.save(vecnorm_path)

        results = evaluate_model(
            model=self.model,
            vecnorm=vecnorm,
            seeds=self.validation_seeds,
            max_steps=self.max_eval_steps,
            closing_reward_version=self.closing_reward_version,
        )
        summary = summarize_results(results)
        score = score_summary(summary)
        returns = np.asarray(
            [result["episode_return"] for result in results],
            dtype=np.float64,
        )
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        lengths = np.asarray(
            [result["steps"] for result in results],
            dtype=np.int64,
        )

        checkpoint_row = {
            "training_seed": self.training_seed,
            "checkpoint": checkpoint,
            "episode": self.completed_episodes,
            "score": score,
            "mean_return": mean_return,
            "std_return": std_return,
            **summary,
        }
        self.checkpoint_rows.append(checkpoint_row)

        evaluation_row = {
            "episode": self.completed_episodes,
            "timestep": checkpoint,
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_length": float(np.mean(lengths)),
        }
        for result in results:
            seed = result["seed"]
            evaluation_row[f"seed_{seed}_return"] = result["episode_return"]
            evaluation_row[f"seed_{seed}_length"] = result["steps"]
        self.evaluation_rows.append(evaluation_row)
        self.evaluation_results.append(returns)
        self.evaluation_lengths.append(lengths)

        for result in results:
            self.rollout_rows.append(
                {
                    "training_seed": self.training_seed,
                    "checkpoint": checkpoint,
                    **result,
                }
            )

        # Match the original training script: the best intermediate model is
        # selected by the highest mean deterministic evaluation return. The
        # safety-first score remains recorded as a separate diagnostic.
        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self.best_checkpoint = checkpoint
            self.best_checkpoint_episode = self.completed_episodes
            self.model.save(os.path.join(self.model_dir, "best_model"))
            vecnorm.save(
                os.path.join(self.model_dir, "best_vecnormalize.pkl")
            )
            write_json(
                self.best_json,
                {
                    "training_seed": self.training_seed,
                    "checkpoint": checkpoint,
                    "episode": self.completed_episodes,
                    "selection_metric": "mean_return",
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "score": score,
                    "summary": summary,
                },
            )

        self._write_outputs()

        print(
            f"Training seed {self.training_seed} | "
            f"checkpoint={checkpoint} | "
            f"mean return={mean_return:.3f} | "
            f"score={score:.3f} | "
            f"success={summary['success_rate']:.2f} | "
            f"failure={summary['failure_rate']:.2f} | "
            f"timeout={summary['time_limit_rate']:.2f} | "
            "P3 error="
            f"{summary['mean_final_position_error']:.2f} m | "
            "target ratio="
            f"{summary['mean_final_max_target_ratio']:.2f}"
        )

    def _write_outputs(self):
        if self.checkpoint_rows:
            with open(
                self.checkpoint_csv,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(self.checkpoint_rows[0]),
                )
                writer.writeheader()
                writer.writerows(self.checkpoint_rows)

        if self.rollout_rows:
            with open(
                self.rollout_csv,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(self.rollout_rows[0]),
                )
                writer.writeheader()
                writer.writerows(self.rollout_rows)

        if self.evaluation_rows:
            with open(
                self.evaluation_csv,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(self.evaluation_rows[0]),
                )
                writer.writeheader()
                writer.writerows(self.evaluation_rows)

            results = np.asarray(self.evaluation_results, dtype=np.float64)
            mean_returns = np.mean(results, axis=1)
            best_index = int(np.argmax(mean_returns))
            np.savez(
                self.evaluations_npz,
                timesteps=np.asarray(
                    [row["timestep"] for row in self.evaluation_rows],
                    dtype=np.int64,
                ),
                episodes=np.asarray(
                    [row["episode"] for row in self.evaluation_rows],
                    dtype=np.int64,
                ),
                results=results,
                ep_lengths=np.asarray(
                    self.evaluation_lengths,
                    dtype=np.int64,
                ),
                eval_seeds=np.asarray(
                    self.validation_seeds,
                    dtype=np.int64,
                ),
                best_eval_index=np.asarray(best_index, dtype=np.int64),
                best_episode=np.asarray(
                    self.evaluation_rows[best_index]["episode"],
                    dtype=np.int64,
                ),
                best_timestep=np.asarray(
                    self.evaluation_rows[best_index]["timestep"],
                    dtype=np.int64,
                ),
                best_mean_return=np.asarray(
                    mean_returns[best_index],
                    dtype=np.float64,
                ),
            )

        if self.checkpoint_rows:
            episodes = np.asarray(
                [row["episode"] for row in self.checkpoint_rows],
                dtype=np.int64,
            )
            means = np.asarray(
                [row["mean_return"] for row in self.checkpoint_rows],
                dtype=np.float64,
            )
            stds = np.asarray(
                [row["std_return"] for row in self.checkpoint_rows],
                dtype=np.float64,
            )
            plt.figure(figsize=(10, 6))
            plt.plot(
                episodes,
                means,
                color="tab:green",
                marker="o",
                linewidth=2.0,
                markersize=4,
                label="Mean evaluation return",
            )
            plt.fill_between(
                episodes,
                means - stds,
                means + stds,
                color="tab:green",
                alpha=0.2,
                label="1 std across seeds",
            )
            plt.xlabel("Completed training episodes")
            plt.ylabel("Mean total evaluation reward")
            plt.title(f"Deterministic Evaluation Learning Curve\n{self.model_name}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=200)
            plt.close()

    def get_best_checkpoint(self):
        return (
            self.best_checkpoint,
            self.best_mean_return,
            self.best_checkpoint_episode,
        )


def seed_output_paths(experiment_name, seed):
    model_dir = os.path.join(
        MODEL_ROOT,
        experiment_name,
        f"seed_{seed}",
    )
    result_dir = os.path.join(
        RESULTS_ROOT,
        experiment_name,
        f"seed_{seed}",
    )
    return model_dir, result_dir


def seed_is_complete(model_dir, result_dir):
    return (
        os.path.exists(os.path.join(model_dir, "final_model.zip"))
        and os.path.exists(
            os.path.join(result_dir, "checkpoint_metrics.csv")
        )
    )


def ensure_seed_outputs_are_safe(model_dir, result_dir, args):
    existing_paths = [
        os.path.join(model_dir, "final_model.zip"),
        os.path.join(result_dir, "checkpoint_metrics.csv"),
        os.path.join(result_dir, "validation_rollout_metrics.csv"),
    ]
    if any(os.path.exists(path) for path in existing_paths) and not args.overwrite:
        raise FileExistsError(
            "Seed output already exists. Use --skip-completed to retain a "
            "completed run or --overwrite to replace its known output files: "
            f"{result_dir}"
        )


def train_one_seed(args, experiment_name, checkpoint_steps, seed):
    model_dir, result_dir = seed_output_paths(experiment_name, seed)

    if args.skip_completed and seed_is_complete(model_dir, result_dir):
        print(f"Skipping completed training seed {seed}.")
        return

    ensure_seed_outputs_are_safe(model_dir, result_dir, args)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_env = None
    model = None
    try:
        train_env = DummyVecEnv(
            [
                lambda seed=seed: make_training_env(
                    seed=seed,
                    closing_reward_version=args.reward_version,
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

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=TRIAL_16_PARAMS["learning_rate"],
            buffer_size=args.buffer_size,
            learning_starts=TRIAL_16_PARAMS["learning_starts"],
            batch_size=TRIAL_16_PARAMS["batch_size"],
            tau=TRIAL_16_PARAMS["tau"],
            gamma=TRIAL_16_PARAMS["gamma"],
            train_freq=TRIAL_16_PARAMS["train_freq"],
            gradient_steps=TRIAL_16_PARAMS["gradient_steps"],
            ent_coef=TRIAL_16_PARAMS["ent_coef"],
            target_update_interval=1,
            verbose=1,
            seed=seed,
            tensorboard_log=os.path.join(LOG_ROOT, experiment_name),
            device=args.device,
        )

        checkpoint_callback = EvaluationCheckpointCallback(
            training_seed=seed,
            checkpoint_steps=checkpoint_steps,
            validation_seeds=args.validation_seeds,
            max_eval_steps=args.max_eval_steps,
            closing_reward_version=args.reward_version,
            model_dir=model_dir,
            result_dir=result_dir,
            model_name=f"{experiment_name} - seed {seed}",
        )
        training_return_callback = TrainingReturnCallback(
            training_seed=seed,
            result_dir=result_dir,
            model_name=f"{experiment_name} - seed {seed}",
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=f"{experiment_name}_seed_{seed}",
            progress_bar=args.progress_bar,
            callback=[training_return_callback, checkpoint_callback],
        )

        model.save(os.path.join(model_dir, "final_model"))
        train_env.save(
            os.path.join(model_dir, "final_vecnormalize.pkl")
        )

        best_checkpoint, best_mean_return, best_checkpoint_episode = (
            checkpoint_callback.get_best_checkpoint()
        )
        training_return_callback.set_best_checkpoint(
            best_checkpoint_episode,
            best_checkpoint,
        )
        training_return_callback.save_outputs()
        print(
            f"Completed training seed {seed} | "
            f"best checkpoint={best_checkpoint} | "
            f"best mean return={best_mean_return:.3f}"
        )
    finally:
        if train_env is not None:
            train_env.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def read_checkpoint_rows(experiment_name, seeds):
    rows = []
    for seed in seeds:
        _, result_dir = seed_output_paths(experiment_name, seed)
        csv_path = os.path.join(result_dir, "checkpoint_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        learning_curve_path = os.path.join(result_dir, "learning_curve.csv")
        if not os.path.exists(learning_curve_path):
            learning_curve_path = os.path.join(
                result_dir,
                "training_episode_returns.csv",
            )
        episode_timesteps = []
        if os.path.exists(learning_curve_path):
            with open(
                learning_curve_path,
                newline="",
                encoding="utf-8",
            ) as file:
                episode_timesteps = [
                    int(row["timestep"]) for row in csv.DictReader(file)
                ]

        with open(csv_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                converted = {
                    key: (
                        int(value)
                        if key in {"training_seed", "checkpoint", "episode"}
                        else float(value)
                    )
                    for key, value in row.items()
                }
                if "episode" not in converted:
                    converted["episode"] = int(
                        np.searchsorted(
                            episode_timesteps,
                            converted["checkpoint"],
                            side="right",
                        )
                    )
                rows.append(converted)
    return rows


def write_learning_curve_plots(
    experiment_name,
    seeds,
    highlight_checkpoint=None,
):
    """Create cross-seed training and deterministic-evaluation plots."""
    result_root = os.path.join(RESULTS_ROOT, experiment_name)
    plot_dir = os.path.join(result_root, "learning_curves")
    os.makedirs(plot_dir, exist_ok=True)
    checkpoint_rows = read_checkpoint_rows(experiment_name, seeds)
    highlighted_rows = [
        row
        for row in checkpoint_rows
        if highlight_checkpoint is not None
        and row["checkpoint"] == highlight_checkpoint
    ]

    training_curves = []
    plt.figure(figsize=(10, 6))
    for seed in seeds:
        _, result_dir = seed_output_paths(experiment_name, seed)
        csv_path = os.path.join(result_dir, "learning_curve.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(
                result_dir,
                "training_episode_returns.csv",
            )
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            episodes = [int(row["episode"]) for row in rows]
            returns = [float(row["episode_return"]) for row in rows]
        else:
            continue
        if not episodes:
            continue
        per_seed_window = 10
        if len(returns) >= per_seed_window:
            weights = np.ones(per_seed_window) / per_seed_window
            rolling = np.convolve(returns, weights, mode="valid")
            rolling_episodes = episodes[per_seed_window - 1 :]
            plt.plot(
                rolling_episodes,
                rolling,
                linewidth=1.5,
                alpha=0.65,
                label=f"Seed {seed}",
            )
        training_curves.append((episodes, returns))

    if training_curves:
        common_episodes = min(len(returns) for _, returns in training_curves)
        stacked_returns = np.asarray(
            [returns[:common_episodes] for _, returns in training_curves],
            dtype=np.float64,
        )
        episode_axis = np.arange(1, common_episodes + 1, dtype=np.int64)
        mean_returns = np.mean(stacked_returns, axis=0)
        std_returns = np.std(stacked_returns, axis=0)
        aggregate_window = 10
        if common_episodes >= aggregate_window:
            weights = np.ones(aggregate_window) / aggregate_window
            smoothed_mean = np.convolve(mean_returns, weights, mode="valid")
            smoothed_episodes = episode_axis[aggregate_window - 1 :]
            plt.plot(
                smoothed_episodes,
                smoothed_mean,
                color="black",
                linewidth=2.5,
                label="Across-seed 10-episode moving average",
            )
        plt.fill_between(
            episode_axis,
            mean_returns - std_returns,
            mean_returns + std_returns,
            color="black",
            alpha=0.10,
            label="Across-seed 1 std",
        )
        for index, row in enumerate(highlighted_rows):
            seed = row["training_seed"]
            plt.axvline(
                row["episode"],
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
                label=(
                    f"Highlighted checkpoint: {highlight_checkpoint // 1000}k "
                    f"(Seed {seed})"
                    if index == 0
                    else None
                ),
            )
        plt.xlabel("Episode number")
        plt.ylabel("Total reward per episode")
        plt.title(
            "Total Reward per Training Episode Across Seeds\n"
            f"{experiment_name}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir,
                "episode_return_learning_curves.png",
            ),
            dpi=200,
        )
    plt.close()

    if not checkpoint_rows:
        return

    plt.figure(figsize=(10, 6))
    for seed in seeds:
        seed_rows = sorted(
            (
                row
                for row in checkpoint_rows
                if row["training_seed"] == seed
            ),
            key=lambda row: row["checkpoint"],
        )
        if not seed_rows:
            continue
        plt.plot(
            [row["episode"] for row in seed_rows],
            [row["mean_return"] for row in seed_rows],
            marker="o",
            alpha=0.65,
            linewidth=1.4,
            label=f"Seed {seed}",
        )

    checkpoints = sorted({row["checkpoint"] for row in checkpoint_rows})
    aggregate_episodes = []
    aggregate_means = []
    aggregate_stds = []
    for checkpoint in checkpoints:
        values = np.asarray(
            [
                row["mean_return"]
                for row in checkpoint_rows
                if row["checkpoint"] == checkpoint
            ],
            dtype=np.float64,
        )
        episode_values = [
            row["episode"]
            for row in checkpoint_rows
            if row["checkpoint"] == checkpoint
        ]
        aggregate_episodes.append(float(np.mean(episode_values)))
        aggregate_means.append(float(np.mean(values)))
        aggregate_stds.append(float(np.std(values)))

    aggregate_episodes = np.asarray(aggregate_episodes, dtype=np.float64)
    aggregate_means = np.asarray(aggregate_means, dtype=np.float64)
    aggregate_stds = np.asarray(aggregate_stds, dtype=np.float64)
    plt.plot(
        aggregate_episodes,
        aggregate_means,
        color="black",
        linewidth=2.5,
        marker="o",
        label="Across-seed mean",
    )
    plt.fill_between(
        aggregate_episodes,
        aggregate_means - aggregate_stds,
        aggregate_means + aggregate_stds,
        color="black",
        alpha=0.12,
        label="Across-seed 1 std",
    )
    for index, row in enumerate(highlighted_rows):
        seed = row["training_seed"]
        plt.scatter(
            row["episode"],
            row["mean_return"],
            marker="*",
            s=180,
            color="tab:red",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label=(
                f"Highlighted checkpoint: {highlight_checkpoint // 1000}k "
                f"(Seed {seed})"
                if index == 0
                else None
            ),
        )
    plt.xlabel("Completed training episodes")
    plt.ylabel("Mean total evaluation reward")
    plt.title(
        "Deterministic Evaluation Learning Curves Across Training Seeds\n"
        f"{experiment_name}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, "evaluation_learning_curves.png"),
        dpi=200,
    )
    plt.close()


def write_aggregate_outputs(experiment_name, seeds):
    rows = read_checkpoint_rows(experiment_name, seeds)
    if not rows:
        return

    result_root = os.path.join(RESULTS_ROOT, experiment_name)
    checkpoints = sorted({row["checkpoint"] for row in rows})
    aggregate_rows = []

    for checkpoint in checkpoints:
        checkpoint_rows = [
            row for row in rows if row["checkpoint"] == checkpoint
        ]
        aggregate = {
            "checkpoint": checkpoint,
            "n_training_seeds": len(checkpoint_rows),
        }
        for metric in AGGREGATE_METRICS:
            values = np.asarray(
                [row[metric] for row in checkpoint_rows],
                dtype=np.float64,
            )
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values))
            aggregate[f"min_{metric}"] = float(np.min(values))
            aggregate[f"max_{metric}"] = float(np.max(values))
        aggregate_rows.append(aggregate)

    aggregate_csv = os.path.join(
        result_root,
        "aggregate_checkpoint_metrics.csv",
    )
    with open(aggregate_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(aggregate_rows[0]),
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    best_rows = []
    for seed in seeds:
        seed_rows = [row for row in rows if row["training_seed"] == seed]
        if not seed_rows:
            continue
        best = max(seed_rows, key=lambda row: row["mean_return"])
        best_rows.append(best)

    best_csv = os.path.join(result_root, "best_checkpoint_per_seed.csv")
    with open(best_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"Aggregate checkpoint metrics: {aggregate_csv}")
    print(f"Best checkpoint per seed:     {best_csv}")
    write_learning_curve_plots(experiment_name, seeds)


def main():
    args = parse_args()
    validate_args(args)
    checkpoint_steps = resolve_checkpoint_steps(args)

    if args.experiment_name is None:
        step_label = format_step_count(args.total_timesteps)
        experiment_name = (
            f"sac_closing_reward{args.reward_version}_trial16_multiseed_"
            f"{step_label}steps"
        )
    else:
        experiment_name = args.experiment_name

    result_root = os.path.join(RESULTS_ROOT, experiment_name)
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(os.path.join(MODEL_ROOT, experiment_name), exist_ok=True)
    os.makedirs(os.path.join(LOG_ROOT, experiment_name), exist_ok=True)

    config = {
        "experiment_name": experiment_name,
        "training_seeds": args.seeds,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "reward_version": args.reward_version,
        "checkpoint_steps": checkpoint_steps,
        "max_eval_steps": args.max_eval_steps,
        "buffer_size": args.buffer_size,
        "device": args.device,
        "trial_16_params": TRIAL_16_PARAMS,
    }
    write_json(os.path.join(result_root, "experiment_config.json"), config)

    print("=" * 80)
    print("SAC closing multi-seed validation: Optuna trial 16")
    print("=" * 80)
    print(f"Experiment:       {experiment_name}")
    print(f"Training seeds:   {args.seeds}")
    print(f"Validation seeds: {args.validation_seeds}")
    print(f"Training steps:   {args.total_timesteps}")
    print(f"Reward version:   {args.reward_version}")
    print(f"Checkpoints:      {checkpoint_steps}")
    print(f"Buffer size:      {args.buffer_size}")
    print(f"Device:           {args.device}")
    print(f"Parameters:       {TRIAL_16_PARAMS}")
    print("=" * 80)

    for seed in args.seeds:
        print("=" * 80)
        print(f"Starting training seed {seed}")
        print("=" * 80)
        train_one_seed(
            args=args,
            experiment_name=experiment_name,
            checkpoint_steps=checkpoint_steps,
            seed=seed,
        )
        write_aggregate_outputs(experiment_name, args.seeds)

    print("=" * 80)
    print("Multi-seed experiment complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
