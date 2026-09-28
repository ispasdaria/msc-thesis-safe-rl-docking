"""Evaluate multi-seed SAC closing models and saved checkpoints.

The evaluator supports three saved-model types from
``train_sac_closing_multiseed.py``:

* ``best``: the intermediate checkpoint with the best validation return;
* ``final``: the model at the final training timestep;
* ``checkpoint``: one or more explicit milestone checkpoints.

By default, the six best models are evaluated deterministically on testing
seeds 1000--1099. These seeds are disjoint from the training seeds and the
checkpoint-selection seeds used by the default multi-seed training run.

Examples:
    python evaluate_sac_closing_multiseed.py --model-types best
    python evaluate_sac_closing_multiseed.py --model-types final
    python evaluate_sac_closing_multiseed.py --model-types checkpoint \
        --checkpoint-steps 300000 400000 500000 600000
    python evaluate_sac_closing_multiseed.py \
        --model-types best final checkpoint --checkpoint-steps 300000 500000
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tune_sac_closing_optuna import (  # noqa: E402
    evaluate_model,
    make_eval_env,
    score_summary,
    summarize_results,
)


DEFAULT_EXPERIMENT = (
    "sac_closing_rewardv15_trial16_multiseed_600ksteps"
)
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", "multiseed")
TRAINING_RESULTS_ROOT = os.path.join(
    SCRIPT_DIR,
    "results",
    "multiseed",
)
EVALUATION_RESULTS_ROOT = os.path.join(
    SCRIPT_DIR,
    "results",
    "multiseed_evaluation",
)

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
    "mean_best_max_target_ratio",
    "mean_target_2x_steps",
    "mean_target_2x_fraction",
    "mean_longest_target_2x_dwell_steps",
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
            "Evaluate final, best-intermediate, or milestone models from a "
            "multi-seed SAC closing experiment."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help="Training experiment directory name.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["best", "final", "checkpoint"],
        default=["best"],
        help=(
            "Saved-model types to evaluate. Multiple types may be supplied "
            "in one invocation (default: best)."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[300_000, 400_000, 500_000, 600_000],
        help=(
            "Milestones evaluated when 'checkpoint' is requested "
            "(default: 300000 400000 500000 600000)."
        ),
    )
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Trained-policy seeds to evaluate. Defaults to the seeds stored "
            "in experiment_config.json."
        ),
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit testing environment seeds. Overrides --test-seed-start "
            "and --n-test-seeds."
        ),
    )
    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=1000,
        help="First testing seed when --test-seeds is omitted.",
    )
    parser.add_argument(
        "--n-test-seeds",
        type=int,
        default=100,
        help="Number of consecutive testing seeds (default: 100).",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=12_000,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional evaluation run directory name.",
    )
    parser.add_argument(
        "--allow-validation-seed-reuse",
        action="store_true",
        help=(
            "Allow testing seeds to overlap checkpoint-selection seeds. This "
            "should not be used for a final unbiased evaluation."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already completed policy summaries in this output run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite requested policy outputs. Combine with --skip-existing "
            "to restart only incomplete policies."
        ),
    )
    return parser.parse_args()


def load_experiment_config(experiment_name):
    config_path = os.path.join(
        TRAINING_RESULTS_ROOT,
        experiment_name,
        "experiment_config.json",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Training experiment configuration was not found: "
            f"{config_path}"
        )
    with open(config_path, encoding="utf-8") as file:
        return json.load(file)


def unique_in_order(values):
    return list(dict.fromkeys(values))


def resolve_arguments(args, experiment_config):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    if args.max_eval_steps <= 0:
        raise ValueError("--max-eval-steps must be positive.")
    if args.n_test_seeds <= 0:
        raise ValueError("--n-test-seeds must be positive.")

    model_types = unique_in_order(args.model_types)
    training_seeds = (
        unique_in_order(args.training_seeds)
        if args.training_seeds is not None
        else list(experiment_config["training_seeds"])
    )
    if not training_seeds:
        raise ValueError("At least one training seed is required.")

    if args.test_seeds is None:
        test_seeds = list(
            range(
                args.test_seed_start,
                args.test_seed_start + args.n_test_seeds,
            )
        )
    else:
        test_seeds = unique_in_order(args.test_seeds)
    if not test_seeds:
        raise ValueError("At least one testing seed is required.")

    checkpoints = sorted(set(args.checkpoint_steps))
    if "checkpoint" in model_types:
        if not checkpoints or any(step <= 0 for step in checkpoints):
            raise ValueError("Checkpoint steps must be positive.")

    validation_seeds = set(experiment_config.get("validation_seeds", []))
    overlap = validation_seeds & set(test_seeds)
    if overlap and not args.allow_validation_seed_reuse:
        raise ValueError(
            "Testing seeds overlap the checkpoint-selection seeds: "
            f"{sorted(overlap)}. Use genuinely held-out seeds or explicitly "
            "pass --allow-validation-seed-reuse."
        )

    return model_types, training_seeds, test_seeds, checkpoints


def checkpoint_label(step):
    return f"checkpoint_{step:07d}_steps"


def build_variants(model_types, checkpoints):
    variants = []
    for model_type in model_types:
        if model_type == "best":
            variants.append(
                {
                    "variant": "best",
                    "model_filename": "best_model.zip",
                    "vecnorm_filename": "best_vecnormalize.pkl",
                }
            )
        elif model_type == "final":
            variants.append(
                {
                    "variant": "final",
                    "model_filename": "final_model.zip",
                    "vecnorm_filename": "final_vecnormalize.pkl",
                }
            )
        elif model_type == "checkpoint":
            for step in checkpoints:
                stem = f"model_{step:07d}_steps"
                variants.append(
                    {
                        "variant": checkpoint_label(step),
                        "model_filename": f"{stem}.zip",
                        "vecnorm_filename": (
                            f"{stem}_vecnormalize.pkl"
                        ),
                    }
                )
    return variants


def default_output_name(test_seeds, variants):
    consecutive = test_seeds == list(
        range(test_seeds[0], test_seeds[0] + len(test_seeds))
    )
    if consecutive:
        seed_label = f"seeds_{test_seeds[0]}_{test_seeds[-1]}"
    else:
        seed_label = f"{len(test_seeds)}_explicit_seeds"

    variant_parts = []
    checkpoint_parts = []
    for variant in variants:
        name = variant["variant"]
        if name.startswith("checkpoint_"):
            step = int(name.split("_")[1])
            checkpoint_parts.append(
                f"{step // 1000}k" if step % 1000 == 0 else str(step)
            )
        else:
            variant_parts.append(name)
    if checkpoint_parts:
        variant_parts.append("ckpt_" + "-".join(checkpoint_parts))
    variant_label = "-".join(variant_parts)
    return f"test_{seed_label}__{variant_label}"


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def write_rows_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_paths(experiment_name, training_seed, variant):
    seed_dir = os.path.join(
        MODEL_ROOT,
        experiment_name,
        f"seed_{training_seed}",
    )
    return (
        os.path.join(seed_dir, variant["model_filename"]),
        os.path.join(seed_dir, variant["vecnorm_filename"]),
    )


def load_model_and_vecnorm(
    model_path,
    vecnorm_path,
    device,
    closing_reward_version,
    action_hold_steps,
    adaptive_action_hold_distance_m,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"VecNormalize statistics not found: {vecnorm_path}"
        )

    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_eval_env(
                    seed=999_999,
                    closing_reward_version=closing_reward_version,
                    action_hold_steps=action_hold_steps,
                    adaptive_action_hold_distance_m=(
                        adaptive_action_hold_distance_m
                    ),
                )
            )
        ]
    )
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(model_path, device=device)
    return model, vecnorm


def summarize_policy_results(
    experiment_name,
    variant_name,
    training_seed,
    model_path,
    vecnorm_path,
    results,
    action_hold_steps,
    adaptive_action_hold_distance_m,
):
    summary = summarize_results(results)
    returns = np.asarray(
        [result["episode_return"] for result in results],
        dtype=np.float64,
    )
    n_success = int(sum(result["success"] for result in results))
    n_failure = int(sum(result["failure"] for result in results))
    n_timeout = int(sum(result["time_limit"] for result in results))
    return {
        "experiment_name": experiment_name,
        "variant": variant_name,
        "training_seed": int(training_seed),
        "action_hold_steps": int(action_hold_steps),
        "control_interval_seconds": float(action_hold_steps),
        "adaptive_action_hold_distance_m": (
            adaptive_action_hold_distance_m
        ),
        "n_test_episodes": len(results),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_timeout": n_timeout,
        "score": score_summary(summary),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        **summary,
        "model_path": str(Path(model_path).resolve()),
        "vecnorm_path": str(Path(vecnorm_path).resolve()),
    }


def evaluate_one_policy(
    args,
    experiment_name,
    variant,
    training_seed,
    test_seeds,
    policy_output_dir,
    closing_reward_version,
    action_hold_steps,
    adaptive_action_hold_distance_m,
):
    summary_json = os.path.join(policy_output_dir, "summary.json")
    if args.skip_existing and os.path.exists(summary_json):
        print(
            f"Skipping completed {variant['variant']} / "
            f"training seed {training_seed}."
        )
        with open(summary_json, encoding="utf-8") as file:
            return json.load(file)

    if os.path.exists(policy_output_dir) and os.listdir(policy_output_dir):
        if not args.overwrite:
            raise FileExistsError(
                "Policy evaluation output already exists. Use "
                "--skip-existing, --overwrite, or a new --output-name: "
                f"{policy_output_dir}"
            )
    os.makedirs(policy_output_dir, exist_ok=True)

    model_path, vecnorm_path = policy_paths(
        experiment_name,
        training_seed,
        variant,
    )
    model = None
    vecnorm = None
    try:
        model, vecnorm = load_model_and_vecnorm(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            device=args.device,
            closing_reward_version=closing_reward_version,
            action_hold_steps=action_hold_steps,
            adaptive_action_hold_distance_m=(
                adaptive_action_hold_distance_m
            ),
        )
        results = evaluate_model(
            model=model,
            vecnorm=vecnorm,
            seeds=test_seeds,
            max_steps=args.max_eval_steps,
            closing_reward_version=closing_reward_version,
            action_hold_steps=action_hold_steps,
            adaptive_action_hold_distance_m=(
                adaptive_action_hold_distance_m
            ),
        )
        summary = summarize_policy_results(
            experiment_name=experiment_name,
            variant_name=variant["variant"],
            training_seed=training_seed,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            results=results,
            action_hold_steps=action_hold_steps,
            adaptive_action_hold_distance_m=(
                adaptive_action_hold_distance_m
            ),
        )

        rollout_rows = [
            {
                "experiment_name": experiment_name,
                "variant": variant["variant"],
                "training_seed": training_seed,
                **result,
            }
            for result in results
        ]
        write_rows_csv(
            os.path.join(policy_output_dir, "rollout_metrics.csv"),
            rollout_rows,
        )
        write_json(summary_json, summary)

        print(
            f"{variant['variant']} | training seed {training_seed} | "
            f"success={summary['success_rate']:.1%} | "
            f"failure={summary['failure_rate']:.1%} | "
            f"timeout={summary['time_limit_rate']:.1%} | "
            f"P3 error={summary['mean_final_position_error']:.2f} m | "
            "target ratio="
            f"{summary['mean_final_max_target_ratio']:.2f} | "
            f"mean return={summary['mean_return']:.2f}"
        )
        return summary
    finally:
        if vecnorm is not None:
            vecnorm.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_aggregate_outputs(run_dir, policy_summaries):
    if not policy_summaries:
        return

    write_rows_csv(
        os.path.join(run_dir, "per_policy_summary.csv"),
        policy_summaries,
    )

    aggregate_rows = []
    variants = unique_in_order(
        summary["variant"] for summary in policy_summaries
    )
    for variant in variants:
        rows = [
            summary
            for summary in policy_summaries
            if summary["variant"] == variant
        ]
        n_rollouts = int(sum(row["n_test_episodes"] for row in rows))
        n_success = int(sum(row["n_success"] for row in rows))
        n_failure = int(sum(row["n_failure"] for row in rows))
        n_timeout = int(sum(row["n_timeout"] for row in rows))
        aggregate = {
            "variant": variant,
            "n_training_seeds": len(rows),
            "n_test_rollouts": n_rollouts,
            "n_success": n_success,
            "n_failure": n_failure,
            "n_timeout": n_timeout,
            "pooled_success_rate": n_success / n_rollouts,
            "pooled_failure_rate": n_failure / n_rollouts,
            "pooled_time_limit_rate": n_timeout / n_rollouts,
        }
        for metric in AGGREGATE_METRICS:
            values = np.asarray(
                [row[metric] for row in rows],
                dtype=np.float64,
            )
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values))
            aggregate[f"min_{metric}"] = float(np.min(values))
            aggregate[f"max_{metric}"] = float(np.max(values))
        aggregate_rows.append(aggregate)

    write_rows_csv(
        os.path.join(run_dir, "aggregate_summary.csv"),
        aggregate_rows,
    )


def main():
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_name)
    closing_reward_version = experiment_config.get("reward_version", "v15")
    action_hold_steps = int(experiment_config.get("action_hold_steps", 1))
    if action_hold_steps <= 0:
        raise ValueError(
            "Experiment action_hold_steps must be a positive integer."
        )
    adaptive_action_hold_distance_m = experiment_config.get(
        "adaptive_action_hold_distance_m"
    )
    if adaptive_action_hold_distance_m is not None:
        adaptive_action_hold_distance_m = float(
            adaptive_action_hold_distance_m
        )
        if adaptive_action_hold_distance_m <= 0.0:
            raise ValueError(
                "Experiment adaptive action-hold distance must be positive."
            )
    (
        model_types,
        training_seeds,
        test_seeds,
        checkpoints,
    ) = resolve_arguments(args, experiment_config)
    variants = build_variants(model_types, checkpoints)

    output_name = args.output_name or default_output_name(
        test_seeds,
        variants,
    )
    run_dir = os.path.join(
        EVALUATION_RESULTS_ROOT,
        args.experiment_name,
        output_name,
    )
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "experiment_name": args.experiment_name,
        "model_types": model_types,
        "variants": [variant["variant"] for variant in variants],
        "training_seeds": training_seeds,
        "test_seeds": test_seeds,
        "max_eval_steps": args.max_eval_steps,
        "device": args.device,
        "reward_version": closing_reward_version,
        "action_hold_steps": action_hold_steps,
        "control_interval_seconds": float(action_hold_steps),
        "adaptive_action_hold_distance_m": (
            adaptive_action_hold_distance_m
        ),
        "output_name": output_name,
        "training_experiment_config": experiment_config,
    }
    write_json(os.path.join(run_dir, "evaluation_manifest.json"), manifest)

    print("=" * 80)
    print("SAC closing multi-seed evaluation")
    print("=" * 80)
    print(f"Experiment:     {args.experiment_name}")
    print(f"Variants:       {[v['variant'] for v in variants]}")
    print(f"Training seeds: {training_seeds}")
    print(f"Reward version: {closing_reward_version}")
    print(f"Control cadence: {action_hold_steps} s per decision")
    if adaptive_action_hold_distance_m is not None:
        print(
            "Cadence override: 1 s near P3 (<= "
            f"{adaptive_action_hold_distance_m:g} m) or on warning"
        )
    print(
        f"Testing seeds:  {test_seeds[0]} ... {test_seeds[-1]} "
        f"({len(test_seeds)} total)"
    )
    print(f"Output:         {run_dir}")
    print("=" * 80)

    policy_summaries = []
    for variant in variants:
        for training_seed in training_seeds:
            policy_output_dir = os.path.join(
                run_dir,
                variant["variant"],
                f"seed_{training_seed}",
            )
            summary = evaluate_one_policy(
                args=args,
                experiment_name=args.experiment_name,
                variant=variant,
                training_seed=training_seed,
                test_seeds=test_seeds,
                policy_output_dir=policy_output_dir,
                closing_reward_version=closing_reward_version,
                action_hold_steps=action_hold_steps,
                adaptive_action_hold_distance_m=(
                    adaptive_action_hold_distance_m
                ),
            )
            policy_summaries.append(summary)
            write_aggregate_outputs(run_dir, policy_summaries)

    print("=" * 80)
    print("Evaluation complete")
    print(f"Per-policy summary: {os.path.join(run_dir, 'per_policy_summary.csv')}")
    print(f"Aggregate summary:  {os.path.join(run_dir, 'aggregate_summary.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
