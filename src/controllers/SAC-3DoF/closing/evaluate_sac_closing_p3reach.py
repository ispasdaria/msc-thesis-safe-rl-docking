"""Evaluate an isolated P3-reaching closing SAC checkpoint."""

import argparse
import json
import os
import sys
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationV6GymEnv,
    ClosingSafetyExplorationV7GymEnv,
    ClosingSafetyExplorationV8GymEnv,
    ClosingSafetyExplorationV9GymEnv,
    ClosingSafetyExplorationV10GymEnv,
    ClosingSafetyExplorationV11GymEnv,
    ClosingSafetyExplorationV12GymEnv,
    ClosingSafetyExplorationV13GymEnv,
    ClosingSafetyExplorationV14GymEnv,
    ClosingSafetyExplorationV15GymEnv,
    ClosingSafetyExplorationV16GymEnv,
    ClosingSafetyExplorationV17GymEnv,
    ClosingSafetyExplorationV18GymEnv,
)
from train_sac_closing_no_safety_termination import (
    evaluate_rollout,
    make_env,
    summarize_rollouts,
    write_csv,
    write_json,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_FAMILY = "p3reach_from_scratch"
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", EXPERIMENT_FAMILY)
RESULT_ROOT = os.path.join(SCRIPT_DIR, "results", EXPERIMENT_FAMILY)
ENV_CLASSES = {
    ClosingSafetyExplorationV6GymEnv.reward_scheme: (
        ClosingSafetyExplorationV6GymEnv
    ),
    ClosingSafetyExplorationV7GymEnv.reward_scheme: (
        ClosingSafetyExplorationV7GymEnv
    ),
    ClosingSafetyExplorationV8GymEnv.reward_scheme: (
        ClosingSafetyExplorationV8GymEnv
    ),
    ClosingSafetyExplorationV9GymEnv.reward_scheme: (
        ClosingSafetyExplorationV9GymEnv
    ),
    ClosingSafetyExplorationV10GymEnv.reward_scheme: (
        ClosingSafetyExplorationV10GymEnv
    ),
    ClosingSafetyExplorationV11GymEnv.reward_scheme: (
        ClosingSafetyExplorationV11GymEnv
    ),
    ClosingSafetyExplorationV12GymEnv.reward_scheme: (
        ClosingSafetyExplorationV12GymEnv
    ),
    ClosingSafetyExplorationV13GymEnv.reward_scheme: (
        ClosingSafetyExplorationV13GymEnv
    ),
    ClosingSafetyExplorationV14GymEnv.reward_scheme: (
        ClosingSafetyExplorationV14GymEnv
    ),
    ClosingSafetyExplorationV15GymEnv.reward_scheme: (
        ClosingSafetyExplorationV15GymEnv
    ),
    ClosingSafetyExplorationV16GymEnv.reward_scheme: (
        ClosingSafetyExplorationV16GymEnv
    ),
    ClosingSafetyExplorationV17GymEnv.reward_scheme: (
        ClosingSafetyExplorationV17GymEnv
    ),
    ClosingSafetyExplorationV18GymEnv.reward_scheme: (
        ClosingSafetyExplorationV18GymEnv
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a P3-reaching checkpoint on selected seeds."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-type",
        choices=["best", "final", "checkpoint"],
        default="best",
    )
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_checkpoint(args, model_dir):
    if args.model_type == "best":
        stem = "best_model"
        vecnormalize_name = "best_vecnormalize.pkl"
    elif args.model_type == "final":
        stem = "final_model"
        vecnormalize_name = "final_vecnormalize.pkl"
    else:
        if args.checkpoint_step is None or args.checkpoint_step <= 0:
            raise ValueError(
                "--checkpoint-step must be positive for --model-type checkpoint."
            )
        stem = f"model_additional_{args.checkpoint_step:07d}_steps"
        vecnormalize_name = f"{stem}_vecnormalize.pkl"

    model_path = os.path.join(model_dir, f"{stem}.zip")
    vecnormalize_path = os.path.join(model_dir, vecnormalize_name)
    for path in (model_path, vecnormalize_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return stem, model_path, vecnormalize_path


def resolve_env_class(result_dir):
    config_path = os.path.join(result_dir, "run_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    reward_scheme = config.get("experiment_reward_scheme")
    if reward_scheme not in ENV_CLASSES:
        raise ValueError(
            f"Unsupported experiment reward scheme: {reward_scheme!r}. "
            f"Supported schemes: {sorted(ENV_CLASSES)}"
        )
    return reward_scheme, ENV_CLASSES[reward_scheme]


def main():
    args = parse_args()
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Evaluation seeds must be unique.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")

    model_dir = os.path.join(MODEL_ROOT, args.experiment_name)
    result_dir = os.path.join(RESULT_ROOT, args.experiment_name)
    reward_scheme, env_class = resolve_env_class(result_dir)
    stem, model_path, vecnormalize_path = resolve_checkpoint(args, model_dir)

    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed=999,
                    mode="testing",
                    max_time=args.max_time,
                    env_class=env_class,
                )
            )
        ]
    )
    vecnormalize = VecNormalize.load(vecnormalize_path, dummy_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(model_path, device=args.device)

    try:
        results = [
            evaluate_rollout(
                model=model,
                vecnormalize=vecnormalize,
                seed=seed,
                max_time=args.max_time,
                env_class=env_class,
            )
            for seed in args.seeds
        ]
    finally:
        dummy_env.close()

    summary = summarize_rollouts(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = f"evaluation_{stem}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"{output_stem}_rollouts.csv")
    summary_path = os.path.join(result_dir, f"{output_stem}_summary.json")
    write_csv(csv_path, results)
    write_json(
        summary_path,
        {
            "experiment_name": args.experiment_name,
            "model_type": args.model_type,
            "checkpoint_step": args.checkpoint_step,
            "model_path": model_path,
            "vecnormalize_path": vecnormalize_path,
            "reward_scheme": reward_scheme,
            "safety_violation_termination_enabled": False,
            "seeds": args.seeds,
            "summary": summary,
        },
    )

    for result in results:
        print(
            f"seed={result['seed']} success={bool(result['success'])} "
            f"termination={result['termination_reason']} "
            f"min_position_error={result['minimum_position_error_m']:.2f} m "
            f"final_max_ratio={result['final_max_target_ratio']:.2f} "
            f"unsafe_fraction={result['unsafe_fraction']:.2%} "
            f"cone_exit_fraction={result['cone_exit_fraction']:.2%} "
            f"max_cone_depth={result['max_cone_exit_depth']:.3f}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Rollouts: {csv_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
