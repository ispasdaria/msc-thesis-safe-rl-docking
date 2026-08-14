"""Continue the best closing SAC while safety violations remain non-terminal.

This is an isolated experiment.  It does not modify the ordinary closing
environment, existing reward-v15 code, or source checkpoints.  Training and
validation use the experiment-only ``p3_reach_v1`` reward in
``ClosingSafetyExplorationGymEnv``. Every safety state is classified and
penalized, while only P3 success, timeout, or a numerical error ends an episode.

The default source is the trial-16 seed-0 400k model that reached roughly
45 m from P3.  Checkpoint 0 is evaluated before any continuation updates, which
provides a direct within-run baseline for the later checkpoints.

Example:
    python train_sac_closing_no_safety_termination.py --device cuda --progress-bar
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.closing_safety_exploration_env import (  # noqa: E402
    ClosingSafetyExplorationGymEnv,
)


SOURCE_STEM = (
    "sac_closing_p3augobs_rewardv15_optuna_trial16_"
    "400ksteps_seed0_buffer300k"
)
SOURCE_DIR = os.path.join(SCRIPT_DIR, "models", "reward_shaping_trials")
DEFAULT_SOURCE_MODEL = os.path.join(SOURCE_DIR, f"{SOURCE_STEM}.zip")
DEFAULT_SOURCE_VECNORMALIZE = os.path.join(
    SOURCE_DIR,
    f"{SOURCE_STEM}_vecnormalize.pkl",
)

EXPERIMENT_FAMILY = "no_safety_termination"
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", EXPERIMENT_FAMILY)
LOG_ROOT = os.path.join(SCRIPT_DIR, "logs", EXPERIMENT_FAMILY)
RESULT_ROOT = os.path.join(SCRIPT_DIR, "results", EXPERIMENT_FAMILY)
SOURCE_REWARD_VERSION = "v15"
EXPERIMENT_REWARD_SCHEME = ClosingSafetyExplorationGymEnv.reward_scheme


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue closing reward-v15 SAC from normal P2 while recording "
            "and penalizing, but not terminating on, safety violations."
        )
    )
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        default=300_000,
        help="Continuation steps after the source checkpoint (default: 300000).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 150_000, 200_000, 250_000, 300_000],
        help="Additional-step checkpoints. Step 0 is always evaluated first.",
    )
    parser.add_argument(
        "--replay-warmup-steps",
        type=int,
        default=10_000,
        help=(
            "Steps collected with the loaded stochastic policy and no gradient "
            "updates before continuation training (default: 10000)."
        ),
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
        help="Fixed held-out P2 rollout seeds (default: 100 through 109).",
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved experiment without creating outputs.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.additional_timesteps <= 0:
        raise ValueError("--additional-timesteps must be positive.")
    if not 0 <= args.replay_warmup_steps < args.additional_timesteps:
        raise ValueError(
            "--replay-warmup-steps must be non-negative and smaller than "
            "--additional-timesteps."
        )
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
    for path in (args.source_model, args.source_vecnormalize):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)


def resolve_checkpoint_steps(args):
    checkpoints = sorted(
        {
            int(step)
            for step in [*args.checkpoint_steps, args.additional_timesteps]
            if 0 < int(step) <= args.additional_timesteps
        }
    )
    if checkpoints[-1] != args.additional_timesteps:
        raise RuntimeError("The final additional timestep must be a checkpoint.")
    return checkpoints


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_env(
    seed,
    mode,
    max_time,
    env_class=ClosingSafetyExplorationGymEnv,
):
    return env_class(
        phase="closing",
        mode=mode,
        max_time=max_time,
        seed=seed,
        # Required by the parent constructor only. The experiment subclass
        # replaces reward computation and never invokes this shared version.
        closing_reward_version=SOURCE_REWARD_VERSION,
    )


def normalize_observation(vecnormalize, observation):
    batch = np.asarray(observation, dtype=np.float32)[None, :]
    return vecnormalize.normalize_obs(batch)[0]


def p3_errors(core_env):
    error = np.asarray(core_env.state - core_env.p3_target, dtype=np.float64)
    error[4] = core_env.wrap_to_pi(error[4])
    tolerance = np.asarray(core_env.p3_target_tolerance, dtype=np.float64)
    ratios = np.abs(error) / np.maximum(tolerance, 1e-12)
    return error, ratios


def evaluate_rollout(
    model,
    vecnormalize,
    seed,
    max_time,
    env_class=ClosingSafetyExplorationGymEnv,
):
    env = make_env(
        seed=seed,
        mode="testing",
        max_time=max_time,
        env_class=env_class,
    )
    observation, _ = env.reset(seed=seed)
    initial_error, _ = p3_errors(env.core_env)
    initial_position_error = float(np.linalg.norm(initial_error[:2]))
    minimum_position_error = initial_position_error

    classification_counts = Counter()
    violated_counts = Counter()
    episode_return = 0.0
    failure_penalty_total = 0.0
    steps = 0
    terminal_info = {}

    while True:
        normalized_observation = normalize_observation(vecnormalize, observation)
        action, _ = model.predict(normalized_observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        episode_return += float(reward)
        terminal_info = info

        for name in ("safe", "warning", "unsafe"):
            if bool(info.get(name, False)):
                classification_counts[name] += 1
        for name in ("geometry", "velocity", "alignment"):
            classification = info.get(name, {})
            for level in ("safe", "warn", "unsafe"):
                if bool(classification.get(level, False)):
                    classification_counts[f"{name}_{level}"] += 1
        for constraint in info.get("violated_constraints", []):
            violated_counts[str(constraint)] += 1

        components = info.get("reward_components", {})
        failure_penalty_total += float(components.get("reward_failure", 0.0))
        error, _ = p3_errors(env.core_env)
        minimum_position_error = min(
            minimum_position_error,
            float(np.linalg.norm(error[:2])),
        )

        if terminated or truncated:
            break

    final_error, final_ratios = p3_errors(env.core_env)
    denominator = max(steps, 1)
    result = {
        "seed": int(seed),
        "steps": int(steps),
        "episode_return": float(episode_return),
        "success": int(bool(terminal_info.get("success", False))),
        "timeout": int(bool(terminal_info.get("time_limit", False))),
        "numerical_failure": int(bool(terminal_info.get("failure", False))),
        "termination_reason": terminal_info.get("termination_reason"),
        "success_after_safety_violation": int(
            bool(terminal_info.get("success", False))
            and classification_counts["unsafe"] > 0
        ),
        "unsafe_ever": int(classification_counts["unsafe"] > 0),
        "unsafe_steps": int(classification_counts["unsafe"]),
        "unsafe_fraction": classification_counts["unsafe"] / denominator,
        "warning_steps": int(classification_counts["warning"]),
        "warning_fraction": classification_counts["warning"] / denominator,
        "safe_steps": int(classification_counts["safe"]),
        "geometry_unsafe_steps": int(classification_counts["geometry_unsafe"]),
        "velocity_unsafe_steps": int(classification_counts["velocity_unsafe"]),
        "alignment_unsafe_steps": int(classification_counts["alignment_unsafe"]),
        "safety_failure_penalty_total": float(failure_penalty_total),
        "violated_constraint_counts": json.dumps(
            dict(sorted(violated_counts.items())), sort_keys=True
        ),
        "initial_position_error_m": initial_position_error,
        "minimum_position_error_m": float(minimum_position_error),
        "final_position_error_m": float(np.linalg.norm(final_error[:2])),
        "final_lateral_error_m": float(abs(final_error[1])),
        "final_velocity_error_mps": float(np.linalg.norm(final_error[2:4])),
        "final_attitude_error_deg": float(np.rad2deg(abs(final_error[4]))),
        "final_rate_error_degs": float(np.rad2deg(abs(final_error[5]))),
        "final_position_ratio": float(np.linalg.norm(final_ratios[:2])),
        "final_velocity_ratio": float(np.linalg.norm(final_ratios[2:4])),
        "final_alignment_ratio": float(np.max(final_ratios[4:6])),
        "final_max_target_ratio": float(np.max(final_ratios)),
    }
    env.close()
    return result


def summarize_rollouts(results):
    def mean(key):
        return float(np.mean([row[key] for row in results]))

    return {
        "rollouts": len(results),
        "success_rate": mean("success"),
        "success_after_violation_rate": mean("success_after_safety_violation"),
        "timeout_rate": mean("timeout"),
        "numerical_failure_rate": mean("numerical_failure"),
        "unsafe_episode_rate": mean("unsafe_ever"),
        "mean_unsafe_fraction": mean("unsafe_fraction"),
        "mean_warning_fraction": mean("warning_fraction"),
        "mean_velocity_unsafe_steps": mean("velocity_unsafe_steps"),
        "mean_geometry_unsafe_steps": mean("geometry_unsafe_steps"),
        "mean_alignment_unsafe_steps": mean("alignment_unsafe_steps"),
        "mean_return": mean("episode_return"),
        "mean_steps": mean("steps"),
        "mean_minimum_position_error_m": mean("minimum_position_error_m"),
        "mean_final_position_error_m": mean("final_position_error_m"),
        "mean_final_lateral_error_m": mean("final_lateral_error_m"),
        "mean_final_velocity_error_mps": mean("final_velocity_error_mps"),
        "mean_final_attitude_error_deg": mean("final_attitude_error_deg"),
        "mean_final_rate_error_degs": mean("final_rate_error_degs"),
        "mean_final_max_target_ratio": mean("final_max_target_ratio"),
    }


def checkpoint_rank(summary):
    """Task-first selection; safety is measured, not used as a success gate."""
    return (
        summary["success_rate"],
        -summary["mean_final_max_target_ratio"],
        -summary["mean_final_position_error_m"],
        -summary["mean_unsafe_fraction"],
        summary["mean_return"],
    )


class TrainingSafetyDiagnosticsCallback(BaseCallback):
    """Save episode diagnostics and refresh the return learning curve."""

    def __init__(
        self,
        output_path,
        plot_path,
        checkpoint_steps,
        moving_average_window=10,
        plot_every_episodes=5,
    ):
        super().__init__()
        self.output_path = output_path
        self.plot_path = plot_path
        self.checkpoint_steps = list(checkpoint_steps)
        self.moving_average_window = int(moving_average_window)
        self.plot_every_episodes = int(plot_every_episodes)
        self.rows = []
        self.counts = Counter()
        self.violated = Counter()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for level in ("safe", "warning", "unsafe"):
                if bool(info.get(level, False)):
                    self.counts[level] += 1
            for name in ("geometry", "velocity", "alignment"):
                classification = info.get(name, {})
                for level in ("safe", "warn", "unsafe"):
                    if bool(classification.get(level, False)):
                        self.counts[f"{name}_{level}"] += 1
            for constraint in info.get("violated_constraints", []):
                self.violated[str(constraint)] += 1

            episode = info.get("episode")
            if episode is None:
                continue
            length = int(episode["l"])
            self.rows.append(
                {
                    "episode": len(self.rows) + 1,
                    "additional_timestep": int(self.num_timesteps),
                    "episode_length": length,
                    "episode_return": float(episode["r"]),
                    "termination_reason": info.get("termination_reason"),
                    "success": int(bool(info.get("success", False))),
                    "timeout": int(bool(info.get("time_limit", False))),
                    "unsafe_steps": int(self.counts["unsafe"]),
                    "unsafe_fraction": self.counts["unsafe"] / max(length, 1),
                    "warning_steps": int(self.counts["warning"]),
                    "geometry_unsafe_steps": int(self.counts["geometry_unsafe"]),
                    "velocity_unsafe_steps": int(self.counts["velocity_unsafe"]),
                    "alignment_unsafe_steps": int(self.counts["alignment_unsafe"]),
                    "violated_constraint_counts": json.dumps(
                        dict(sorted(self.violated.items())), sort_keys=True
                    ),
                }
            )
            self.counts.clear()
            self.violated.clear()
            write_csv(self.output_path, self.rows)
            if len(self.rows) % self.plot_every_episodes == 0:
                self._write_learning_curve()
        return True

    def _on_training_end(self):
        write_csv(self.output_path, self.rows)
        self._write_learning_curve()

    def _write_learning_curve(self):
        if not self.rows:
            return

        episodes = np.arange(1, len(self.rows) + 1, dtype=np.int64)
        returns = np.asarray(
            [row["episode_return"] for row in self.rows],
            dtype=np.float64,
        )
        timesteps = np.asarray(
            [row["additional_timestep"] for row in self.rows],
            dtype=np.int64,
        )

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            episodes,
            returns,
            color="tab:blue",
            linewidth=0.9,
            alpha=0.55,
            label="Episode return",
        )

        window = self.moving_average_window
        if len(returns) >= window:
            weights = np.ones(window, dtype=np.float64) / window
            moving_average = np.convolve(returns, weights, mode="valid")
            axis.plot(
                episodes[window - 1 :],
                moving_average,
                color="tab:orange",
                linewidth=2.0,
                label=f"{window}-episode moving average",
            )

        checkpoint_label_used = False
        for checkpoint in self.checkpoint_steps:
            reached = np.flatnonzero(timesteps >= checkpoint)
            if len(reached) == 0:
                continue
            checkpoint_episode = int(episodes[reached[0]])
            axis.axvline(
                checkpoint_episode,
                color="tab:gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=(
                    "Evaluation checkpoint"
                    if not checkpoint_label_used
                    else None
                ),
            )
            checkpoint_label_used = True

        axis.set_xlabel("Completed training episode")
        axis.set_ylabel("Raw cumulative reward")
        axis.set_title(
            "Closing SAC learning curve\n"
            "P3-reaching reward with non-terminal safety exploration"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=180)
        plt.close(figure)


class ExperimentCheckpointCallback(BaseCallback):
    """Evaluate the source and exact continuation checkpoints."""

    def __init__(
        self,
        checkpoint_steps,
        validation_seeds,
        max_time,
        model_dir,
        result_dir,
        evaluate_initial=True,
        env_class=ClosingSafetyExplorationGymEnv,
    ):
        super().__init__()
        self.checkpoint_steps = list(checkpoint_steps)
        self.validation_seeds = list(validation_seeds)
        self.max_time = float(max_time)
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.evaluate_initial = bool(evaluate_initial)
        self.env_class = env_class
        self.next_checkpoint = 0
        self.summary_rows = []
        self.rollout_rows = []
        self.best_rank = None
        self.best_step = None

    def _on_training_start(self):
        if self.evaluate_initial and not self.summary_rows:
            self._evaluate_and_save(0)

    def _on_step(self):
        while self.next_checkpoint < len(self.checkpoint_steps):
            step = self.checkpoint_steps[self.next_checkpoint]
            if self.num_timesteps < step:
                break
            self._evaluate_and_save(step)
            self.next_checkpoint += 1
        return True

    def _evaluate_and_save(self, step):
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is None:
            raise RuntimeError("VecNormalize is required.")

        stem = f"model_additional_{step:07d}_steps"
        self.model.save(os.path.join(self.model_dir, stem))
        vecnormalize.save(os.path.join(self.model_dir, f"{stem}_vecnormalize.pkl"))

        results = [
            evaluate_rollout(
                model=self.model,
                vecnormalize=vecnormalize,
                seed=seed,
                max_time=self.max_time,
                env_class=self.env_class,
            )
            for seed in self.validation_seeds
        ]
        summary = summarize_rollouts(results)
        summary_row = {"additional_timestep": int(step), **summary}
        self.summary_rows.append(summary_row)
        self.rollout_rows.extend(
            {"additional_timestep": int(step), **result} for result in results
        )
        write_csv(
            os.path.join(self.result_dir, "checkpoint_metrics.csv"),
            self.summary_rows,
        )
        write_csv(
            os.path.join(self.result_dir, "validation_rollout_metrics.csv"),
            self.rollout_rows,
        )

        rank = checkpoint_rank(summary)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            self.best_step = int(step)
            self.model.save(os.path.join(self.model_dir, "best_model"))
            vecnormalize.save(os.path.join(self.model_dir, "best_vecnormalize.pkl"))
            write_json(
                os.path.join(self.result_dir, "best_checkpoint.json"),
                {
                    "additional_timestep": self.best_step,
                    "selection_order": [
                        "highest success rate",
                        "lowest mean final max target ratio",
                        "lowest mean final position error",
                        "lowest mean unsafe fraction",
                        "highest mean return",
                    ],
                    "summary": summary,
                },
            )

        print(
            f"Additional step {step:,} | "
            f"success={summary['success_rate']:.2%} | "
            f"success after violation={summary['success_after_violation_rate']:.2%} | "
            f"unsafe episodes={summary['unsafe_episode_rate']:.2%} | "
            f"unsafe steps={summary['mean_unsafe_fraction']:.2%} | "
            f"final P3 error={summary['mean_final_position_error_m']:.2f} m | "
            f"max target ratio={summary['mean_final_max_target_ratio']:.2f}",
            flush=True,
        )


def build_config(args, checkpoints, experiment_name):
    return {
        "experiment_name": experiment_name,
        "purpose": "closing SAC exploration without safety termination",
        "phase": "closing",
        "initial_distribution": "normal P2",
        "source_reward_version": SOURCE_REWARD_VERSION,
        "experiment_reward_scheme": EXPERIMENT_REWARD_SCHEME,
        "shared_gym_reward_version_used_for_experiment": False,
        "shared_reward_files_modified": False,
        "experiment_reward_weights": ClosingSafetyExplorationGymEnv.reward_weights,
        "precision_gate_distance_m": (
            ClosingSafetyExplorationGymEnv.precision_gate_distance_m
        ),
        "precision_gate_width_m": (
            ClosingSafetyExplorationGymEnv.precision_gate_width_m
        ),
        "safety_classification_enabled": True,
        "safety_penalties_enabled": True,
        "safety_violation_termination_enabled": False,
        "terminal_conditions": ["P3 success", "time limit", "non-finite state"],
        "source_model": os.path.abspath(args.source_model),
        "source_vecnormalize": os.path.abspath(args.source_vecnormalize),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize_sha256": file_sha256(args.source_vecnormalize),
        "source_files_are_read_only_inputs": True,
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "additional_timesteps": args.additional_timesteps,
        "replay_warmup_steps": args.replay_warmup_steps,
        "replay_warmup_action_source": "loaded stochastic SAC policy",
        "gradient_updates_during_replay_warmup": False,
        "checkpoint_additional_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "device": args.device,
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
        "sac_closing_p3reachv1_no_safety_termination_"
        f"from_trial16_400k_add{args.additional_timesteps // 1000}k_"
        f"seed{args.training_seed}_{timestamp}"
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
                )
            )
        ]
    )
    training_env = VecNormalize.load(args.source_vecnormalize, training_env)
    training_env.training = True
    training_env.norm_reward = True

    model = SAC.load(
        args.source_model,
        env=training_env,
        device=args.device,
        seed=args.training_seed,
        tensorboard_log=log_dir,
    )
    # SAC checkpoints do not include replay-buffer content.  Fill the fresh
    # buffer with the loaded stochastic policy, rather than discarding that
    # policy and reverting to uniformly random warm-up actions.
    source_num_timesteps = int(model.num_timesteps)
    original_learning_starts = int(model.learning_starts)
    original_gradient_steps = int(model.gradient_steps)
    config["source_model_num_timesteps"] = source_num_timesteps
    config["continuation_replay_buffer"] = "fresh"
    config["source_learning_starts"] = original_learning_starts
    config["source_gradient_steps"] = original_gradient_steps
    config["vecnormalize_updates_during_continuation"] = True
    write_json(os.path.join(result_dir, "run_config.json"), config)

    safety_callback = TrainingSafetyDiagnosticsCallback(
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
    )

    print("=" * 80)
    print("Closing SAC experiment: safety violations are penalized but non-terminal")
    print("=" * 80)
    print(f"Experiment:          {experiment_name}")
    print(f"Source checkpoint:   {args.source_model}")
    print(f"Additional steps:    {args.additional_timesteps:,}")
    print(f"Checkpoints:         {[0, *checkpoints]}")
    print(f"Validation seeds:    {args.validation_seeds}")
    print(f"Reward:              isolated {EXPERIMENT_REWARD_SCHEME}")
    print("Shared reward code:  unchanged; source policy was trained with v15")
    print("Episode boundaries:  P3 success, timeout, or numerical failure")
    print("=" * 80)

    try:
        model.learning_starts = 0
        model.gradient_steps = 0
        model.learn(
            total_timesteps=args.replay_warmup_steps,
            reset_num_timesteps=True,
            tb_log_name=experiment_name,
            progress_bar=args.progress_bar,
            callback=[safety_callback, checkpoint_callback],
        )
        model.learning_starts = original_learning_starts
        model.gradient_steps = original_gradient_steps
        model.learn(
            total_timesteps=(
                args.additional_timesteps - args.replay_warmup_steps
            ),
            reset_num_timesteps=False,
            tb_log_name=experiment_name,
            progress_bar=args.progress_bar,
            callback=[safety_callback, checkpoint_callback],
        )
        model.save(os.path.join(model_dir, "final_model"))
        training_env.save(os.path.join(model_dir, "final_vecnormalize.pkl"))
    finally:
        training_env.close()

    print(f"Best additional step: {checkpoint_callback.best_step}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
