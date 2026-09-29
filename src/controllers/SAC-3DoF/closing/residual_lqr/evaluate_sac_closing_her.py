"""Evaluate best/final goal-conditioned SAC+HER closing policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from her_closing_utils import (
    HER_ENV_CLASSES,
    evaluate_model,
    make_her_env,
    summarize_rollouts,
    write_csv,
    write_json,
)


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "her"
RESULT_ROOT = CLOSING_DIR / "results" / "her_evaluation"
TRAINING_RESULT_ROOT = CLOSING_DIR / "results" / "her"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate goal-conditioned closing SAC+HER models."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=("best", "final"),
        default=["best", "final"],
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=list(range(200, 210)),
    )
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--output-name", default="test_seeds200_209")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--reward-version",
        choices=tuple(HER_ENV_CLASSES),
        default=None,
        help="Override the reward version recorded by the trainer.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = (
        TRAINING_RESULT_ROOT / args.experiment_name / "experiment_config.json"
    )
    recorded_reward_version = "v1"
    recorded_config = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            recorded_config = json.load(stream)
            recorded_reward_version = recorded_config.get(
                "reward_version",
                "v1",
            )
    reward_version = args.reward_version or recorded_reward_version
    env_kwargs = {}
    if reward_version == "v4":
        env_kwargs["residual_scale"] = float(
            recorded_config.get("residual_scale", 0.05)
        )
    output_dir = RESULT_ROOT / args.experiment_name / args.output_name
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_type in args.model_types:
        model_path = (
            MODEL_ROOT / args.experiment_name / f"{model_type}_model.zip"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        # SB3 restores the saved HerReplayBuffer configuration while loading
        # SAC.  HerReplayBuffer requires a compatible VecEnv even when the
        # model will only be used for inference.
        load_env = DummyVecEnv(
            [
                lambda: Monitor(
                    make_her_env(
                        seed=args.test_seeds[0],
                        mode="testing",
                        max_time=args.max_eval_steps,
                        reward_version=reward_version,
                        **env_kwargs,
                    )
                )
            ]
        )
        model = SAC.load(model_path, env=load_env, device=args.device)
        rows = evaluate_model(
            model,
            args.test_seeds,
            max_steps=args.max_eval_steps,
            reward_version=reward_version,
            env_kwargs=env_kwargs,
        )
        summary = {
            "experiment_name": args.experiment_name,
            "model_type": model_type,
            "model_path": str(model_path),
            **summarize_rollouts(rows),
        }
        summaries.append(summary)
        write_csv(output_dir / model_type / "rollout_metrics.csv", rows)
        write_json(output_dir / model_type / "summary.json", summary)
        print(
            f"{model_type}: success={summary['success_rate']:.3f}, "
            f"failure={summary['failure_rate']:.3f}, "
            f"P3 error={summary['mean_final_position_error']:.2f} m, "
            f"best ratio={summary['mean_best_max_target_ratio']:.2f}, "
            f"final ratio={summary['mean_final_max_target_ratio']:.2f}"
        )
        load_env.close()

    write_csv(output_dir / "per_policy_summary.csv", summaries)
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "experiment_name": args.experiment_name,
            "model_types": args.model_types,
            "test_seeds": args.test_seeds,
            "max_eval_steps": args.max_eval_steps,
            "reward_version": reward_version,
        },
    )


if __name__ == "__main__":
    main()
