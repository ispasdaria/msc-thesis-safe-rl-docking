"""Compare a frozen 1 Hz closing SAC at 1.0 s and 0.1 s integration.

This evaluator does not train or modify the source policy.  It executes the
same deterministic actor on the same testing seeds in three environments:

* its native 1.0 s integration and 1.0 s decision cadence;
* 0.1 s integration with each action held for ten simulator steps;
* 0.1 s integration with the actor recomputed after every simulator step.

The held-action case isolates integration accuracy.  Comparing the two
0.1 s cases then isolates the effect of changing the policy cadence to 10 Hz.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
CURRICULUM_DIR = SCRIPT_DIR / "curriculum_and_p3_reach"
for path in (SCRIPT_DIR, CURRICULUM_DIR, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from train_sac_closing_no_safety_termination import (
    evaluate_rollout,
    file_sha256,
    summarize_rollouts,
    write_csv,
    write_json,
)


SOURCE_EXPERIMENT = (
    "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
)
SOURCE_SEED_DIR = (
    SCRIPT_DIR / "models" / "multiseed" / SOURCE_EXPERIMENT / "seed_0"
)
DEFAULT_MODEL_PATH = SOURCE_SEED_DIR / "model_0100000_steps.zip"
DEFAULT_VECNORMALIZE_PATH = (
    SOURCE_SEED_DIR / "model_0100000_steps_vecnormalize.pkl"
)
RESULT_ROOT = SCRIPT_DIR / "results" / "timestep_transfer"
DEFAULT_VARIANTS = (
    "native_1hz",
    "dt0p1_hold_1s",
    "dt0p1_recompute_10hz",
)


@dataclass(frozen=True)
class EvaluationVariant:
    name: str
    integration_dt_seconds: float
    action_hold_steps: int
    reward_version: str
    description: str

    @property
    def decision_interval_seconds(self):
        return self.integration_dt_seconds * self.action_hold_steps


VARIANTS = {
    "native_1hz": EvaluationVariant(
        name="native_1hz",
        integration_dt_seconds=1.0,
        action_hold_steps=1,
        reward_version="v15",
        description="Native 1 s integration and native 1 Hz policy cadence.",
    ),
    "dt0p1_hold_1s": EvaluationVariant(
        name="dt0p1_hold_1s",
        integration_dt_seconds=0.1,
        action_hold_steps=10,
        reward_version="v15_dt",
        description=(
            "0.1 s integration with one deterministic policy action held "
            "for ten simulator steps."
        ),
    ),
    "dt0p1_recompute_10hz": EvaluationVariant(
        name="dt0p1_recompute_10hz",
        integration_dt_seconds=0.1,
        action_hold_steps=1,
        reward_version="v15_dt",
        description=(
            "0.1 s integration with a new deterministic policy action "
            "computed after every simulator step."
        ),
    ),
}


class NativeOneHzEvaluationEnv(Docking3DoFGymEnv):
    """Force the frozen policy's native integration and reward definition."""

    def __init__(self, *args, **kwargs):
        kwargs["dt"] = 1.0
        kwargs["integrator"] = "rk4"
        kwargs["closing_reward_version"] = "v15"
        super().__init__(*args, **kwargs)


class Dt01EvaluationEnv(Docking3DoFGymEnv):
    """Force 0.1 s RK4 integration and the time-scaled V15 reward."""

    def __init__(self, *args, **kwargs):
        kwargs["dt"] = 0.1
        kwargs["integrator"] = "rk4"
        kwargs["closing_reward_version"] = "v15_dt"
        super().__init__(*args, **kwargs)


ENV_CLASSES = {
    "native_1hz": NativeOneHzEvaluationEnv,
    "dt0p1_hold_1s": Dt01EvaluationEnv,
    "dt0p1_recompute_10hz": Dt01EvaluationEnv,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen 1 Hz closing SAC under native, fine-integration "
            "held-action, and true 10 Hz policy cadences."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Frozen SAC model to evaluate.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=DEFAULT_VECNORMALIZE_PATH,
        help="VecNormalize statistics paired with --model-path.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(200, 210)),
        help="Held-out testing seeds used identically for every variant.",
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
        help=(
            "New directory name under results/timestep_transfer. Existing "
            "directories are never reused."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the experiment without loading the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique.")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must be unique.")
    if args.output_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_name
    ):
        raise ValueError(
            "--output-name may contain only letters, numbers, '.', '_', and "
            "'-', and must begin with a letter or number."
        )


def resolve_input_path(path):
    return Path(path).expanduser().resolve()


def validate_input_artifacts(model_path, vecnormalize_path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(
            f"VecNormalize statistics not found: {vecnormalize_path}"
        )


def make_output_dir_name(args):
    if args.output_name is not None:
        return args.output_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"guarded100k_cadence_comparison_{timestamp}"


def variant_manifest(variant):
    return {
        "name": variant.name,
        "description": variant.description,
        "integration_dt_seconds": variant.integration_dt_seconds,
        "integration_frequency_hz": 1.0 / variant.integration_dt_seconds,
        "action_hold_steps": variant.action_hold_steps,
        "decision_interval_seconds": variant.decision_interval_seconds,
        "policy_frequency_hz": 1.0 / variant.decision_interval_seconds,
        "reward_version": variant.reward_version,
    }


def build_manifest(args, model_path, vecnormalize_path, output_dir):
    return {
        "purpose": "deterministic closing SAC evaluation",
        "source_experiment": model_path.stem,
        "source_checkpoint": model_path.stem,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path),
        "model_sha256": file_sha256(model_path),
        "vecnormalize_sha256": file_sha256(vecnormalize_path),
        "source_artifacts_modified": False,
        "training_performed": False,
        "deterministic": True,
        "testing_distribution": True,
        "seeds": list(args.seeds),
        "max_time_seconds": float(args.max_time),
        "device": args.device,
        "output_dir": str(output_dir),
        "variants": [variant_manifest(VARIANTS[name]) for name in args.variants],
        "return_comparison_warning": (
            "Native uses V15 while 0.1 s variants use time-scaled V15-dt. "
            "Compare task, safety, and state-error metrics; do not rank "
            "variants by episode return."
        ),
    }


def load_frozen_policy(model_path, vecnormalize_path, device):
    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                NativeOneHzEvaluationEnv(
                    phase="closing",
                    mode="testing",
                    max_time=1.0,
                    normalize_action=True,
                    seed=999_999,
                )
            )
        ]
    )
    try:
        vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy_env)
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        model = SAC.load(str(model_path), device=device)
    except Exception:
        dummy_env.close()
        raise
    return model, vecnormalize, dummy_env


def evaluate_variant(model, vecnormalize, variant, seeds, max_time):
    env_class = ENV_CLASSES[variant.name]
    return [
        evaluate_rollout(
            model=model,
            vecnormalize=vecnormalize,
            seed=seed,
            max_time=max_time,
            env_class=env_class,
            action_hold_steps=variant.action_hold_steps,
        )
        for seed in seeds
    ]


COMPARISON_METRICS = (
    "success_rate",
    "safety_failure_rate",
    "timeout_rate",
    "unsafe_episode_rate",
    "mean_minimum_position_error_m",
    "mean_final_position_error_m",
    "mean_final_max_target_ratio",
    "mean_terminal_port_speed_mps",
    "mean_steps",
)


def build_comparison_rows(variant_names, summaries):
    native = summaries.get("native_1hz")
    rows = []
    for name in variant_names:
        variant = VARIANTS[name]
        summary = summaries[name]
        row = {
            **variant_manifest(variant),
            **{metric: summary[metric] for metric in COMPARISON_METRICS},
        }
        if native is not None:
            for metric in COMPARISON_METRICS:
                row[f"delta_from_native__{metric}"] = (
                    summary[metric] - native[metric]
                )
        rows.append(row)
    return rows


PAIRED_METRICS = (
    "success",
    "safety_failure",
    "timeout",
    "termination_reason",
    "elapsed_time_seconds",
    "decision_steps",
    "simulator_steps",
    "minimum_position_error_m",
    "final_position_error_m",
    "final_max_target_ratio",
    "max_port_speed_mps",
)


def build_paired_rows(seeds, variant_names, results_by_variant):
    indexed = {
        name: {int(row["seed"]): row for row in results_by_variant[name]}
        for name in variant_names
    }
    rows = []
    for seed in seeds:
        row = {"seed": int(seed)}
        for name in variant_names:
            result = indexed[name][int(seed)]
            for metric in PAIRED_METRICS:
                row[f"{name}__{metric}"] = result[metric]
        rows.append(row)
    return rows


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(name, summary):
    variant = VARIANTS[name]
    print(
        f"{name}: success={summary['success_rate']:.1%}, "
        f"safety_failure={summary['safety_failure_rate']:.1%}, "
        f"timeout={summary['timeout_rate']:.1%}, "
        f"minimum_P3_error={summary['mean_minimum_position_error_m']:.2f} m, "
        f"final_ratio={summary['mean_final_max_target_ratio']:.3f}, "
        f"policy_hz={1.0 / variant.decision_interval_seconds:.1f}"
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    model_path = resolve_input_path(args.model_path)
    vecnormalize_path = resolve_input_path(args.vecnormalize_path)
    validate_input_artifacts(model_path, vecnormalize_path)

    output_dir = (RESULT_ROOT / make_output_dir_name(args)).resolve()
    manifest = build_manifest(
        args,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        output_dir=output_dir,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse evaluation directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    write_json(str(output_dir / "evaluation_manifest.json"), manifest)

    model = None
    vecnormalize = None
    dummy_env = None
    results_by_variant = {}
    summaries = {}
    try:
        model, vecnormalize, dummy_env = load_frozen_policy(
            model_path,
            vecnormalize_path,
            args.device,
        )
        for name in args.variants:
            variant = VARIANTS[name]
            print(f"Evaluating {name}: {variant.description}", flush=True)
            results = evaluate_variant(
                model=model,
                vecnormalize=vecnormalize,
                variant=variant,
                seeds=args.seeds,
                max_time=args.max_time,
            )
            summary = summarize_rollouts(results)
            results_by_variant[name] = results
            summaries[name] = summary

            variant_dir = output_dir / name
            variant_dir.mkdir()
            write_csv(str(variant_dir / "rollout_metrics.csv"), results)
            write_json(
                str(variant_dir / "summary.json"),
                {**variant_manifest(variant), "summary": summary},
            )
            print_summary(name, summary)
    finally:
        if dummy_env is not None:
            dummy_env.close()

    comparison_rows = build_comparison_rows(args.variants, summaries)
    paired_rows = build_paired_rows(
        args.seeds,
        args.variants,
        results_by_variant,
    )
    write_rows(output_dir / "comparison_summary.csv", comparison_rows)
    write_rows(output_dir / "paired_seed_metrics.csv", paired_rows)
    write_json(
        str(output_dir / "comparison_summary.json"),
        {
            "manifest": manifest,
            "summaries": summaries,
            "comparison_rows": comparison_rows,
        },
    )
    print(f"Saved isolated comparison: {output_dir}")
    return summaries


if __name__ == "__main__":
    main()
