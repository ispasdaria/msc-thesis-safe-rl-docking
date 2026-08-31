"""Evaluate fixed-goal LQR-reference-residual SAC on held-out seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from her_closing_utils import (
    evaluate_model,
    summarize_rollouts,
    write_csv,
    write_json,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "lqr_reference_residual_no_her"
RESULT_ROOT = CLOSING_DIR / "results" / "lqr_reference_residual_no_her"
DEFAULT_EXPERIMENT = (
    "sac_closing_lqr_reference_residual_"
    "pilot_c_no_her_1hz_50k_seed0"
)
STRATIFIED_EXPERIMENT = (
    "sac_closing_lqr_reference_residual_"
    "pilot_d_stratified_1hz_50k_seed0"
)
TERMINAL_EXPERIMENT = (
    "sac_closing_lqr_reference_residual_"
    "pilot_f_terminal_r2_h600_1hz_50k_seed0"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a fixed-goal reference-residual SAC model against its "
            "zero-residual LQR baseline on held-out environment seeds."
        )
    )
    parser.add_argument(
        "--artifact-family",
        choices=("no_her", "stratified", "terminal"),
        default="no_her",
        help="Use Pilot C, Pilot D/E, or Pilot F model/result directories.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=("baseline", "best", "final", "checkpoint"),
        default=("baseline", "best"),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=(),
        help="Required only when model-types includes checkpoint.",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seeds; overrides seed-start and n-seeds.",
    )
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--allow-validation-seed-reuse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_models(args, config):
    pilot_label = f"pilot_{str(config.get('pilot', 'c')).lower()}"
    models = []
    for model_type in dict.fromkeys(args.model_types):
        if model_type == "baseline":
            models.append(("lqr_zero_residual", 0, "checkpoint_0_model.zip"))
        elif model_type == "best":
            models.append(
                (
                    f"{pilot_label}_best",
                    int(config["best_checkpoint"]),
                    "best_model.zip",
                )
            )
        elif model_type == "final":
            models.append(
                (
                    f"{pilot_label}_final",
                    int(config["actual_timesteps"]),
                    "final_model.zip",
                )
            )
        else:
            if not args.checkpoint_steps:
                raise ValueError(
                    "--checkpoint-steps is required for checkpoint models."
                )
            for checkpoint in sorted(set(args.checkpoint_steps)):
                if checkpoint < 0:
                    raise ValueError("Checkpoint steps must be non-negative.")
                models.append(
                    (
                        f"checkpoint_{checkpoint}",
                        checkpoint,
                        f"checkpoint_{checkpoint}_model.zip",
                    )
                )
    return models


def environment_kwargs(config):
    bounds = config["reference_residual_bounds"]
    reward = config["reward"]
    residual_fade = config.get("terminal_residual_fade", {})
    return {
        "dt": float(config.get("dt_seconds", 1.0)),
        "reference_residual_bounds": (
            float(bounds["vx_mps"]),
            float(bounds["vy_mps"]),
            np.deg2rad(float(bounds["attitude_deg"])),
        ),
        "progress_gamma": float(config["sac"]["gamma"]),
        "progress_weight": float(reward["progress_weight"]),
        "goal_component_weights": tuple(
            float(value)
            for value in reward.get("goal_component_weights", [1.0] * 6)
        ),
        "worst_component_weight": float(
            reward.get("worst_component_weight", 0.5)
        ),
        "lateral_constraint_progress_weight": float(
            reward.get("lateral_constraint_progress_weight", 0.0)
        ),
        "reference_magnitude_weight": float(reward["magnitude_weight"]),
        "reference_smoothness_weight": float(reward["smoothness_weight"]),
        "success_bonus": float(reward["success_bonus"]),
        "safety_failure_penalty": float(reward["failure_penalty"]),
        "timeout_penalty": float(reward["timeout_penalty"]),
        "timeout_violation_weight": float(
            reward.get("timeout_violation_weight", 0.0)
        ),
        "time_scale_persistent_reward": bool(
            reward.get("persistent_terms_time_scaled", False)
        ),
        "reward_reference_dt": 1.0,
        "time_consistent_smoothness_reward": bool(
            reward.get("smoothness_is_action_rate_cost", False)
        ),
        "reward_design": str(reward.get("design", "v1")),
        "terminal_residual_fade_full_ratio": (
            residual_fade.get("full_authority_at_or_above_ratio")
            if residual_fade.get("enabled", False)
            else None
        ),
        "terminal_residual_fade_zero_ratio": float(
            residual_fade.get("zero_authority_at_or_below_ratio", 1.0)
        ),
        "terminate_on_success": bool(config["actual_p3_success_terminal"]),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_seeds <= 0 or args.max_eval_steps <= 0:
        raise ValueError("Seed count and evaluation horizon must be positive.")
    if args.artifact_family == "terminal":
        model_root = CLOSING_DIR / "models" / "lqr_reference_residual_terminal"
        result_root = CLOSING_DIR / "results" / "lqr_reference_residual_terminal"
        experiment_name = args.experiment_name or TERMINAL_EXPERIMENT
    elif args.artifact_family == "stratified":
        model_root = CLOSING_DIR / "models" / "lqr_reference_residual_stratified"
        result_root = CLOSING_DIR / "results" / "lqr_reference_residual_stratified"
        experiment_name = args.experiment_name or STRATIFIED_EXPERIMENT
    else:
        model_root = MODEL_ROOT
        result_root = RESULT_ROOT
        experiment_name = args.experiment_name or DEFAULT_EXPERIMENT
    result_dir = result_root / experiment_name
    model_dir = model_root / experiment_name
    config_path = result_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if bool(config.get("her_used", True)):
        raise ValueError("This evaluator requires a no-HER experiment.")
    if bool(config.get("cbf_filter_used_during_training", True)):
        raise ValueError("The recorded experiment unexpectedly used CBF.")

    seeds = (
        list(dict.fromkeys(args.test_seeds))
        if args.test_seeds is not None
        else list(range(args.seed_start, args.seed_start + args.n_seeds))
    )
    if not seeds:
        raise ValueError("At least one test seed is required.")
    overlap = sorted(set(seeds) & set(config.get("validation_seeds", [])))
    if overlap and not args.allow_validation_seed_reuse:
        raise ValueError(
            "Test seeds overlap checkpoint-selection seeds: "
            f"{overlap}. Choose held-out seeds or explicitly pass "
            "--allow-validation-seed-reuse."
        )
    models = resolve_models(args, config)
    output_name = args.output_name or (
        f"heldout_seeds{min(seeds)}_{max(seeds)}_n{len(seeds)}"
    )
    output_dir = result_dir / "evaluations" / output_name
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_dir}. Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = environment_kwargs(config)
    all_rows = []
    summaries = {}
    for label, checkpoint, filename in models:
        model_path = model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = ZeroResidualConstrainedSAC.load(
            model_path,
            device=args.device,
        )
        rows = evaluate_model(
            model,
            seeds,
            max_steps=args.max_eval_steps,
            reward_version=config["reward_version"],
            env_kwargs=kwargs,
            progress_callback=lambda row, completed, total, label=label: print(
                f"{label}: [{completed}/{total}] "
                f"seed={row['evaluation_seed']} "
                f"success={row['success']} failure={row['failure']} "
                f"timeout={row['time_limit']}",
                flush=True,
            ),
        )
        summary = {
            "label": label,
            "checkpoint": checkpoint,
            "model_path": str(model_path),
            **summarize_rollouts(rows),
        }
        summary["dt_seconds"] = float(config.get("dt_seconds", 1.0))
        summary["mean_elapsed_time_seconds"] = float(
            summary["mean_steps"] * summary["dt_seconds"]
        )
        summaries[label] = summary
        all_rows.extend(
            {"controller": label, "checkpoint": checkpoint, **row}
            for row in rows
        )
        write_csv(output_dir / label / "rollout_metrics.csv", rows)
        write_json(output_dir / label / "summary.json", summary)
        print(
            f"{label}: success={summary['n_success']}/{summary['n_episodes']}, "
            f"failure={summary['n_failure']}, timeout={summary['n_timeout']}, "
            f"best_ratio={summary['mean_best_max_target_ratio']:.4f}",
            flush=True,
        )

    write_csv(output_dir / "combined_rollout_metrics.csv", all_rows)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "experiment_name": experiment_name,
            "test_seeds": seeds,
            "selection_seed_overlap": overlap,
            "max_eval_steps": args.max_eval_steps,
            "dt_seconds": float(config.get("dt_seconds", 1.0)),
            "her_used": False,
            "cbf_used": False,
            "controllers": summaries,
        },
    )
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
