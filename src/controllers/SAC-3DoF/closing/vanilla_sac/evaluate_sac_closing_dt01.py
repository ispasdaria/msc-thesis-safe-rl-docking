"""Evaluate a trained 0.1 s V15-dt closing SAC on held-out seeds."""

import argparse
import json
import os
import sys
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..")
)
CLOSING_DIR = os.path.dirname(SCRIPT_DIR)
CURRICULUM_DIR = os.path.join(CLOSING_DIR, "curriculum_and_p3_reach")
SRC_PATH = os.path.join(REPO_ROOT, "src")
for path in (SCRIPT_DIR, CLOSING_DIR, CURRICULUM_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from train_p3_reach_nonterminal_safety_continuation import (
    evaluate_rollout,
    summarize_rollouts,
    write_csv,
    write_json,
)


DT = 0.1
REWARD_VERSION = "v15_dt"
SUPPORTED_REWARD_VERSIONS = ("v15_dt", "v15_dt_safe")
DEFAULT_EXPERIMENT_NAME = (
    "sac_closing_v15dt_dt0p1_trial16_500ksteps_seed0_pilot"
)
MODEL_ROOT = os.path.join(
    SCRIPT_DIR,
    "models",
    "reward_shaping_trials",
    "dt_0p1",
)
RESULT_ROOT = os.path.join(
    SCRIPT_DIR,
    "results",
    "reward_shaping_trials",
    "dt_0p1",
)


class ClosingV15Dt01EvaluationEnv(Docking3DoFGymEnv):
    """Force diagnostic rollouts to use the training MDP exactly."""

    def __init__(self, *args, **kwargs):
        kwargs["dt"] = DT
        kwargs["closing_reward_version"] = REWARD_VERSION
        kwargs["integrator"] = "rk4"
        kwargs["action_hold_steps"] = 1
        super().__init__(*args, **kwargs)


class ClosingV15DtSafe01EvaluationEnv(Docking3DoFGymEnv):
    """Reproduce the failure-safe 10 Hz training MDP."""

    def __init__(self, *args, **kwargs):
        kwargs["dt"] = DT
        kwargs["closing_reward_version"] = "v15_dt_safe"
        kwargs["integrator"] = "rk4"
        kwargs["action_hold_steps"] = 1
        super().__init__(*args, **kwargs)


EVALUATION_ENV_CLASSES = {
    "v15_dt": ClosingV15Dt01EvaluationEnv,
    "v15_dt_safe": ClosingV15DtSafe01EvaluationEnv,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained 10 Hz closing SAC deterministically on "
            "held-out testing seeds."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--model-type",
        choices=["best", "final"],
        default="best",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(200, 210)),
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional evaluation output directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and configuration without loading the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique.")


def resolve_artifacts(args, require_model=True):
    experiment_name = args.experiment_name
    model_dir = os.path.join(MODEL_ROOT, experiment_name)
    result_dir = os.path.join(RESULT_ROOT, experiment_name)
    config_path = os.path.join(result_dir, "experiment_config.json")
    stem = f"{experiment_name}_{args.model_type}"
    model_path = os.path.join(model_dir, f"{stem}.zip")
    vecnormalize_path = os.path.join(
        model_dir,
        f"{stem}_vecnormalize.pkl",
    )

    required_paths = [config_path]
    if require_model:
        required_paths.extend([model_path, vecnormalize_path])
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    return {
        "experiment_name": experiment_name,
        "model_dir": model_dir,
        "result_dir": result_dir,
        "config_path": config_path,
        "model_path": model_path,
        "vecnormalize_path": vecnormalize_path,
    }


def load_and_validate_training_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not abs(float(config.get("dt_seconds", -1.0)) - DT) <= 1e-12:
        raise ValueError(
            f"Expected training dt={DT}, got {config.get('dt_seconds')!r}."
        )
    reward_version = config.get("reward", {}).get("version")
    if reward_version not in SUPPORTED_REWARD_VERSIONS:
        raise ValueError(
            f"Unsupported dt=0.1 reward version: {reward_version!r}."
        )
    if int(config.get("action_hold_steps", -1)) != 1:
        raise ValueError("The evaluated policy must have action_hold_steps=1.")
    return config


def make_eval_env(seed, max_time, env_class=ClosingV15Dt01EvaluationEnv):
    return env_class(
        phase="closing",
        mode="testing",
        max_time=max_time,
        normalize_action=True,
        seed=seed,
    )


def print_rollout(result):
    print(
        f"seed={result['seed']} "
        f"success={bool(result['success'])} "
        f"termination={result['termination_reason']} "
        f"time={result['elapsed_time_seconds']:.1f} s "
        f"return={result['episode_return']:.2f} "
        f"final_max_ratio={result['final_max_target_ratio']:.3f} "
        f"unsafe_fraction={result['unsafe_fraction']:.2%}"
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    artifacts = resolve_artifacts(args, require_model=not args.dry_run)
    training_config = load_and_validate_training_config(
        artifacts["config_path"]
    )
    reward_version = training_config["reward"]["version"]
    env_class = EVALUATION_ENV_CLASSES[reward_version]

    evaluation_name = args.output_name
    if evaluation_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_name = (
            f"heldout_{args.model_type}_seeds{args.seeds[0]}_"
            f"{args.seeds[-1]}_{timestamp}"
        )
    output_dir = os.path.join(
        artifacts["result_dir"],
        "evaluations",
        evaluation_name,
    )

    manifest = {
        "experiment_name": args.experiment_name,
        "model_type": args.model_type,
        "model_path": artifacts["model_path"],
        "vecnormalize_path": artifacts["vecnormalize_path"],
        "training_config_path": artifacts["config_path"],
        "dt_seconds": DT,
        "control_frequency_hz": 1.0 / DT,
        "reward_version": reward_version,
        "deterministic": True,
        "testing_distribution": True,
        "seeds": list(args.seeds),
        "max_time_seconds": args.max_time,
        "max_steps": int(round(args.max_time / DT)),
        "device": args.device,
        "output_dir": output_dir,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest

    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Refusing to reuse evaluation directory: {output_dir}"
        )
    os.makedirs(output_dir)

    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_eval_env(
                    seed=999,
                    max_time=args.max_time,
                    env_class=env_class,
                )
            )
        ]
    )
    vecnormalize = VecNormalize.load(
        artifacts["vecnormalize_path"],
        dummy_env,
    )
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(artifacts["model_path"], device=args.device)

    try:
        results = [
            evaluate_rollout(
                model=model,
                vecnormalize=vecnormalize,
                seed=seed,
                max_time=args.max_time,
                env_class=env_class,
                action_hold_steps=1,
            )
            for seed in args.seeds
        ]
    finally:
        dummy_env.close()

    summary = summarize_rollouts(results)
    write_csv(os.path.join(output_dir, "rollout_metrics.csv"), results)
    write_json(
        os.path.join(output_dir, "evaluation_summary.json"),
        {**manifest, "summary": summary, "training": training_config},
    )

    for result in results:
        print_rollout(result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved evaluation: {output_dir}")
    return summary


if __name__ == "__main__":
    main()
