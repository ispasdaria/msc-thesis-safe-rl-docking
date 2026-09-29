"""Tune vanilla SAC hyperparameters for the isolated P3-reaching v12 reward.

Every trial starts from a new randomly initialized SAC model, an empty replay
buffer, and new observation-normalization statistics.  No curriculum,
demonstrations, behavior cloning, pretrained weights, or replay prefill are
used.

The objective remains informative before the first success: success dominates
the score, followed by simultaneous proximity to all six P3 tolerances,
approach progress, terminal precision, and safety diagnostics.  Intermediate
evaluation at fixed validation seeds permits median pruning without evaluating
on the training seed.

Example:
    python tune_p3_reach_v12_from_scratch_optuna.py --fresh --save-models
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (SCRIPT_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationV12GymEnv,
)
from train_p3_reach_nonterminal_safety_continuation import (
    make_env,
    normalize_observation,
    p3_errors,
)


STUDY_VERSION = "p3_reach_v12_optuna"
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", STUDY_VERSION)
RESULTS_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    STUDY_VERSION,
)

SEARCH_SPACE = {
    "learning_rate": {"low": 3e-5, "high": 3e-4, "log": True},
    "batch_size": [256, 512, 1024],
    "tau": [0.002, 0.005, 0.01],
    "gamma": [0.995, 0.997, 0.999, 0.9995],
    "learning_starts": [5_000, 10_000, 25_000],
    "ent_coef": [
        "auto_0.01",
        "auto_0.03",
        "auto_0.05",
        "auto_0.1",
    ],
    # Encoded categorically so Optuna samples compatible pairs together.
    "train_update": ["8_8", "16_8", "16_16", "32_16"],
}


def make_training_env(seed, max_time):
    return Monitor(
        make_env(
            seed=seed,
            mode="training",
            max_time=max_time,
            env_class=ClosingSafetyExplorationV12GymEnv,
        )
    )


def target_metrics(core_env):
    error, ratios = p3_errors(core_env)
    return {
        "position_error_m": float(np.linalg.norm(error[:2])),
        "lateral_error_m": float(abs(error[1])),
        "velocity_error_mps": float(np.linalg.norm(error[2:4])),
        "attitude_error_deg": float(np.rad2deg(abs(error[4]))),
        "rate_error_degs": float(np.rad2deg(abs(error[5]))),
        "position_ratio": float(np.linalg.norm(ratios[:2])),
        "velocity_ratio": float(np.linalg.norm(ratios[2:4])),
        "alignment_ratio": float(np.max(ratios[4:6])),
        "max_target_ratio": float(np.max(ratios)),
    }


def evaluate_model(model, vecnormalize, seeds, max_time):
    """Evaluate deterministically while preserving VecNormalize state."""
    results = []
    previous_training = vecnormalize.training
    previous_norm_reward = vecnormalize.norm_reward
    vecnormalize.training = False
    vecnormalize.norm_reward = False

    try:
        for seed in seeds:
            env = make_env(
                seed=seed,
                mode="testing",
                max_time=max_time,
                env_class=ClosingSafetyExplorationV12GymEnv,
            )
            try:
                observation, _ = env.reset(seed=seed)
                initial = target_metrics(env.core_env)
                minimum_position_error = initial["position_error_m"]
                minimum_max_target_ratio = initial["max_target_ratio"]
                unsafe_steps = 0
                warning_steps = 0
                episode_return = 0.0
                final_info = {}
                steps = 0

                while True:
                    normalized_observation = normalize_observation(
                        vecnormalize,
                        observation,
                    )
                    action, _ = model.predict(
                        normalized_observation,
                        deterministic=True,
                    )
                    observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    steps += 1
                    episode_return += float(reward)
                    final_info = info
                    unsafe_steps += int(bool(info.get("unsafe", False)))
                    warning_steps += int(bool(info.get("warning", False)))

                    current = target_metrics(env.core_env)
                    minimum_position_error = min(
                        minimum_position_error,
                        current["position_error_m"],
                    )
                    minimum_max_target_ratio = min(
                        minimum_max_target_ratio,
                        current["max_target_ratio"],
                    )

                    if terminated or truncated:
                        break

                final = target_metrics(env.core_env)
                initial_distance = max(initial["position_error_m"], 1e-12)
                approach_progress = np.clip(
                    (initial_distance - minimum_position_error)
                    / initial_distance,
                    -1.0,
                    1.0,
                )
                denominator = max(steps, 1)
                results.append(
                    {
                        "seed": int(seed),
                        "steps": int(steps),
                        "episode_return": float(episode_return),
                        "success": int(bool(final_info.get("success", False))),
                        "timeout": int(
                            bool(final_info.get("time_limit", False))
                        ),
                        "failed_approach_reset": int(
                            bool(final_info.get("failed_approach_reset", False))
                        ),
                        "numerical_failure": int(
                            bool(final_info.get("failure", False))
                        ),
                        "unsafe_ever": int(unsafe_steps > 0),
                        "unsafe_fraction": unsafe_steps / denominator,
                        "warning_fraction": warning_steps / denominator,
                        "initial_position_error_m": initial[
                            "position_error_m"
                        ],
                        "minimum_position_error_m": minimum_position_error,
                        "minimum_max_target_ratio": minimum_max_target_ratio,
                        "approach_progress_fraction": float(approach_progress),
                        **{f"final_{key}": value for key, value in final.items()},
                    }
                )
            finally:
                env.close()
    finally:
        vecnormalize.training = previous_training
        vecnormalize.norm_reward = previous_norm_reward

    return results


def summarize_results(results):
    def mean(key):
        return float(np.mean([result[key] for result in results]))

    minimum_ratios = np.asarray(
        [result["minimum_max_target_ratio"] for result in results],
        dtype=np.float64,
    )
    return {
        "rollouts": int(len(results)),
        "success_rate": mean("success"),
        "timeout_rate": mean("timeout"),
        "failed_approach_reset_rate": mean("failed_approach_reset"),
        "numerical_failure_rate": mean("numerical_failure"),
        "unsafe_episode_rate": mean("unsafe_ever"),
        "mean_unsafe_fraction": mean("unsafe_fraction"),
        "mean_warning_fraction": mean("warning_fraction"),
        "mean_return": mean("episode_return"),
        "mean_steps": mean("steps"),
        "mean_approach_progress_fraction": mean(
            "approach_progress_fraction"
        ),
        "mean_minimum_position_error_m": mean("minimum_position_error_m"),
        "mean_minimum_max_target_ratio": mean("minimum_max_target_ratio"),
        "reach_10x_rate": float(np.mean(minimum_ratios <= 10.0)),
        "reach_5x_rate": float(np.mean(minimum_ratios <= 5.0)),
        "reach_3x_rate": float(np.mean(minimum_ratios <= 3.0)),
        "mean_final_position_error_m": mean("final_position_error_m"),
        "mean_final_lateral_error_m": mean("final_lateral_error_m"),
        "mean_final_velocity_error_mps": mean("final_velocity_error_mps"),
        "mean_final_attitude_error_deg": mean("final_attitude_error_deg"),
        "mean_final_rate_error_degs": mean("final_rate_error_degs"),
        "mean_final_position_ratio": mean("final_position_ratio"),
        "mean_final_velocity_ratio": mean("final_velocity_ratio"),
        "mean_final_alignment_ratio": mean("final_alignment_ratio"),
        "mean_final_max_target_ratio": mean("final_max_target_ratio"),
    }


def score_summary(summary):
    """Rank success first, then near-success and dense terminal quality."""
    dense_quality = (
        5_000.0 * summary["reach_3x_rate"]
        + 2_000.0 * summary["reach_5x_rate"]
        + 500.0 * summary["reach_10x_rate"]
        + 1_000.0 * summary["mean_approach_progress_fraction"]
        - 100.0 * np.log1p(summary["mean_minimum_max_target_ratio"])
        - 200.0 * np.log1p(summary["mean_final_max_target_ratio"])
        - 50.0 * np.log1p(summary["mean_final_velocity_ratio"])
        - 20.0 * np.log1p(summary["mean_final_alignment_ratio"])
        - 250.0 * summary["failed_approach_reset_rate"]
        - 250.0 * summary["mean_unsafe_fraction"]
        - 50.0 * summary["mean_warning_fraction"]
    )
    return float(100_000.0 * summary["success_rate"] + dense_quality)


def suggest_sac_params(trial):
    update_choice = trial.suggest_categorical(
        "train_update",
        SEARCH_SPACE["train_update"],
    )
    train_freq, gradient_steps = (
        int(value) for value in update_choice.split("_")
    )
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate"]["low"],
            SEARCH_SPACE["learning_rate"]["high"],
            log=SEARCH_SPACE["learning_rate"]["log"],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            SEARCH_SPACE["batch_size"],
        ),
        "tau": trial.suggest_categorical("tau", SEARCH_SPACE["tau"]),
        "gamma": trial.suggest_categorical("gamma", SEARCH_SPACE["gamma"]),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            SEARCH_SPACE["learning_starts"],
        ),
        "ent_coef": trial.suggest_categorical(
            "ent_coef",
            SEARCH_SPACE["ent_coef"],
        ),
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }


def evaluation_steps(train_steps, eval_interval):
    steps = list(range(eval_interval, train_steps + 1, eval_interval))
    if not steps or steps[-1] != train_steps:
        steps.append(train_steps)
    return steps


def append_metrics(path, trial, trained_steps, score, summary):
    row = {
        "trial": trial.number,
        "trained_steps": trained_steps,
        "score": score,
        **summary,
        "params": json.dumps(trial.params, sort_keys=True),
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_rollouts(path, trial_number, trained_steps, results):
    rows = [
        {
            "trial": trial_number,
            "trained_steps": trained_steps,
            **result,
        }
        for result in results
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_objective(args, study_dir, metrics_csv, rollouts_csv):
    checkpoints = evaluation_steps(args.train_steps, args.eval_interval)

    def objective(trial):
        params = suggest_sac_params(trial)
        training_env = DummyVecEnv(
            [lambda: make_training_env(args.training_seed, args.max_time)]
        )
        training_env = VecNormalize(
            training_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=params["gamma"],
        )
        model = SAC(
            policy="MlpPolicy",
            env=training_env,
            learning_rate=params["learning_rate"],
            buffer_size=args.buffer_size,
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
            ent_coef=params["ent_coef"],
            target_update_interval=1,
            verbose=0,
            seed=args.training_seed,
            tensorboard_log=LOG_DIR,
            device=args.device,
        )

        trained_steps = 0
        try:
            for checkpoint in checkpoints:
                segment_steps = checkpoint - trained_steps
                model.learn(
                    total_timesteps=segment_steps,
                    reset_num_timesteps=(trained_steps == 0),
                    tb_log_name=(
                        f"{args.study_name}_trial_{trial.number:03d}"
                    ),
                    progress_bar=args.progress_bar,
                )
                trained_steps = checkpoint

                results = evaluate_model(
                    model=model,
                    vecnormalize=training_env,
                    seeds=args.validation_seeds,
                    max_time=args.max_time,
                )
                summary = summarize_results(results)
                score = score_summary(summary)
                append_metrics(
                    metrics_csv,
                    trial,
                    trained_steps,
                    score,
                    summary,
                )
                save_rollouts(
                    rollouts_csv,
                    trial.number,
                    trained_steps,
                    results,
                )

                for key, value in summary.items():
                    trial.set_user_attr(key, value)
                trial.set_user_attr("trained_steps", trained_steps)
                trial.report(score, step=trained_steps)

                print(
                    f"Trial {trial.number:03d} | step={trained_steps:,} | "
                    f"score={score:.2f} | "
                    f"success={summary['success_rate']:.1%} | "
                    f"reach<=5x={summary['reach_5x_rate']:.1%} | "
                    f"min ratio={summary['mean_minimum_max_target_ratio']:.2f} | "
                    f"final ratio={summary['mean_final_max_target_ratio']:.2f}",
                    flush=True,
                )

                if trained_steps < args.train_steps and trial.should_prune():
                    trial.set_user_attr("pruned_at_steps", trained_steps)
                    raise optuna.TrialPruned(
                        f"Median-pruned after {trained_steps:,} steps."
                    )

            if args.save_models:
                trial_dir = os.path.join(
                    study_dir,
                    f"trial_{trial.number:03d}",
                )
                os.makedirs(trial_dir, exist_ok=True)
                model.save(os.path.join(trial_dir, "model"))
                training_env.save(
                    os.path.join(trial_dir, "vecnormalize.pkl")
                )

            return score
        finally:
            training_env.close()

    return objective


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna tuning for vanilla SAC trained from scratch on "
            "p3_reach_v12."
        )
    )
    parser.add_argument(
        "--study-name",
        default="sac_closing_p3reach_v12_vanilla_sac_600k",
    )
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=600_000)
    parser.add_argument("--eval-interval", type=int, default=200_000)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 105)),
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL; defaults to study_dir/study.db.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete an existing study with the same name before starting.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    study_dir = os.path.join(RESULTS_DIR, args.study_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = args.storage
    if storage is None:
        database_path = Path(study_dir, "study.db").resolve().as_posix()
        storage = f"sqlite:///{database_path}"

    metrics_csv = os.path.join(study_dir, "evaluation_metrics.csv")
    rollouts_csv = os.path.join(study_dir, "evaluation_rollouts.csv")
    config_path = os.path.join(study_dir, "study_config.json")

    if args.fresh:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
        except KeyError:
            pass
        for output_path in (metrics_csv, rollouts_csv):
            if os.path.exists(output_path):
                os.remove(output_path)

    config = {
        "study_version": STUDY_VERSION,
        "study_name": args.study_name,
        "reward_scheme": ClosingSafetyExplorationV12GymEnv.reward_scheme,
        "vanilla_sac_from_scratch": True,
        "excluded_methods": [
            "curriculum",
            "demonstrations",
            "behavior cloning",
            "pretrained weights",
            "replay prefill",
        ],
        "storage": storage,
        "log_dir": LOG_DIR,
        "results_dir": study_dir,
        "arguments": vars(args),
        "search_space": SEARCH_SPACE,
        "evaluation_steps": evaluation_steps(
            args.train_steps,
            args.eval_interval,
        ),
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
        },
        "objective_priority": [
            "success rate",
            "fraction reaching simultaneous 3x/5x/10x P3 tolerances",
            "approach progress",
            "minimum and final maximum P3 target ratio",
            "terminal velocity and alignment ratios",
            "failed approaches and safety diagnostics",
        ],
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    sampler = optuna.samplers.TPESampler(seed=args.training_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=min(400_000, args.train_steps),
        interval_steps=args.eval_interval,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=not args.fresh,
    )

    print("=" * 80)
    print("Vanilla SAC Optuna tuning: p3_reach_v12")
    print("=" * 80)
    print(f"Study:              {args.study_name}")
    print(f"Trials this run:     {args.n_trials}")
    print(f"Steps per trial:     {args.train_steps:,}")
    print(f"Evaluation steps:    {config['evaluation_steps']}")
    print(f"Training seed:       {args.training_seed}")
    print(f"Validation seeds:    {args.validation_seeds}")
    print(f"Reward normalization: disabled")
    print(f"Storage:             {storage}")
    print(f"Results:             {study_dir}")
    print("=" * 80)

    study.optimize(
        build_objective(
            args,
            study_dir,
            metrics_csv,
            rollouts_csv,
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    best = study.best_trial
    print("=" * 80)
    print("Best completed trial")
    print("=" * 80)
    print(f"Trial:  {best.number}")
    print(f"Score:  {best.value:.6f}")
    print(f"Params: {best.params}")
    for key, value in sorted(best.user_attrs.items()):
        print(f"{key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
