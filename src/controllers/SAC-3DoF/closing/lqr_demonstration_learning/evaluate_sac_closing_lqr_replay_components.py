"""Diagnose component-wise P3 errors for an LQR-replay SAC policy.

This is a post-training analysis entrypoint.  It loads one saved SAC policy
and its matching VecNormalize statistics, runs deterministic full-closing
rollouts, and reports both the final P3 error and the best simultaneous
P3-target approach reached anywhere in each rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sac_closing_lqr_replay import make_closing_env  # noqa: E402


MODEL_ROOTS = {
    "lqr_replay": SCRIPT_DIR / "models" / "lqr_replay",
    "lqr_mixed_replay": SCRIPT_DIR / "models" / "lqr_mixed_replay",
    "lqr_bc_warmstart": SCRIPT_DIR / "models" / "lqr_bc_warmstart",
    "lqr_bc_constrained": SCRIPT_DIR / "models" / "lqr_bc_constrained",
}
RESULTS_ROOTS = {
    "lqr_replay": SCRIPT_DIR / "results" / "lqr_replay",
    "lqr_mixed_replay": SCRIPT_DIR / "results" / "lqr_mixed_replay",
    "lqr_bc_warmstart": SCRIPT_DIR / "results" / "lqr_bc_warmstart",
    "lqr_bc_constrained": SCRIPT_DIR / "results" / "lqr_bc_constrained",
}
COMPONENTS = ("rx", "ry", "vx", "vy", "psi", "omega")
DISPLAY_LABELS = ("rx", "ry", "vx", "vy", "psi", "omega")
UNITS = ("m", "m", "m/s", "m/s", "rad", "rad/s")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved LQR-replay SAC model and diagnose which P3 "
            "target components prevent successful closing."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_rewardv15_lqr_replay_600000steps",
    )
    parser.add_argument(
        "--experiment-family",
        choices=tuple(MODEL_ROOTS),
        default="lqr_replay",
        help="Select the training family that owns the saved model outputs.",
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--model-type",
        choices=("best", "final", "checkpoint"),
        default="best",
    )
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("testing", "training", "nominal"),
        default="testing",
    )
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--reward-version", default="v15")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_rollouts <= 0:
        parser.error("--num-rollouts must be positive.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive.")
    if args.model_type == "checkpoint" and args.checkpoint_step is None:
        parser.error("--checkpoint-step is required for --model-type checkpoint.")
    if args.model_type != "checkpoint" and args.checkpoint_step is not None:
        parser.error("--checkpoint-step is only valid for checkpoint models.")
    return args


def resolve_model_paths(args):
    seed_dir = (
        MODEL_ROOTS[args.experiment_family]
        / args.experiment_name
        / f"seed_{args.training_seed}"
    )
    if args.model_type == "best":
        label = "best"
        model_path = seed_dir / "best_model.zip"
        vecnormalize_path = seed_dir / "best_vecnormalize.pkl"
    elif args.model_type == "final":
        label = "final"
        model_path = seed_dir / "final_model.zip"
        vecnormalize_path = seed_dir / "final_vecnormalize.pkl"
    else:
        label = f"checkpoint_{args.checkpoint_step}"
        stem = f"model_{args.checkpoint_step:07d}_steps"
        model_path = seed_dir / f"{stem}.zip"
        vecnormalize_path = seed_dir / f"{stem}_vecnormalize.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(
            f"VecNormalize statistics not found: {vecnormalize_path}"
        )
    return label, model_path, vecnormalize_path


def p3_error(core_env):
    error = np.asarray(
        core_env.state - core_env.p3_target,
        dtype=np.float64,
    )
    error[4] = core_env.wrap_to_pi(error[4])
    tolerance = np.asarray(core_env.p3_target_tolerance, dtype=np.float64)
    ratios = np.abs(error) / tolerance
    return error, ratios, tolerance


def component_fields(prefix, error, ratios):
    fields = {}
    for index, component in enumerate(COMPONENTS):
        fields[f"{prefix}_{component}_error"] = float(error[index])
        fields[f"{prefix}_{component}_abs_error"] = float(abs(error[index]))
        fields[f"{prefix}_{component}_ratio"] = float(ratios[index])
    fields[f"{prefix}_max_ratio"] = float(np.max(ratios))
    failed = [
        component
        for component, ratio in zip(COMPONENTS, ratios)
        if ratio > 1.0
    ]
    fields[f"{prefix}_failed_component_count"] = len(failed)
    fields[f"{prefix}_failed_components"] = ";".join(failed)
    fields[f"{prefix}_dominant_component"] = (
        COMPONENTS[int(np.argmax(ratios))] if failed else "none"
    )
    return fields


def evaluate_one_rollout(
    model,
    vecnormalize,
    seed,
    mode,
    max_steps,
    reward_version,
):
    env = make_closing_env(
        mode=mode,
        seed=seed,
        reward_version=reward_version,
        monitor=False,
    )
    try:
        observation, _ = env.reset(seed=seed)
        initial_state = env.core_env.state.copy()
        p3_target = env.core_env.p3_target.copy()
        closest_error, closest_ratios, tolerance = p3_error(env.core_env)
        closest_step = 0
        closest_score = float(np.max(closest_ratios))

        episode_return = 0.0
        terminated = False
        truncated = False
        info = {}
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            normalized_observation = vecnormalize.normalize_obs(
                np.asarray(observation, dtype=np.float32).reshape(1, -1)
            )
            action, _ = model.predict(
                normalized_observation,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            episode_return += float(reward)
            steps += 1

            error, ratios, _ = p3_error(env.core_env)
            score = float(np.max(ratios))
            if score < closest_score:
                closest_score = score
                closest_step = steps
                closest_error = error.copy()
                closest_ratios = ratios.copy()

        final_error, final_ratios, _ = p3_error(env.core_env)
        success = bool(info.get("success", False))
        failure = bool(info.get("failure", False))
        timeout = bool(
            info.get("time_limit", False)
            or (steps >= max_steps and not terminated)
        )
        record = {
            "seed": int(seed),
            "success": success,
            "failure": failure,
            "timeout": timeout,
            "termination_reason": info.get("termination_reason"),
            "steps": int(steps),
            "episode_return": float(episode_return),
            "closest_step": int(closest_step),
        }
        for index, component in enumerate(COMPONENTS):
            record[f"initial_{component}"] = float(initial_state[index])
            record[f"target_{component}"] = float(p3_target[index])
            record[f"tolerance_{component}"] = float(tolerance[index])
        record.update(component_fields("final", final_error, final_ratios))
        record.update(
            component_fields("closest", closest_error, closest_ratios)
        )
        return record
    finally:
        env.close()


def summarize(records, args, model_path, vecnormalize_path):
    count = len(records)
    final_dominant = Counter(
        record["final_dominant_component"]
        for record in records
        if record["final_dominant_component"] != "none"
    )
    closest_dominant = Counter(
        record["closest_dominant_component"]
        for record in records
        if record["closest_dominant_component"] != "none"
    )
    final_max = np.asarray(
        [record["final_max_ratio"] for record in records],
        dtype=np.float64,
    )
    closest_max = np.asarray(
        [record["closest_max_ratio"] for record in records],
        dtype=np.float64,
    )

    component_summary = {}
    for index, component in enumerate(COMPONENTS):
        final_abs = np.asarray(
            [record[f"final_{component}_abs_error"] for record in records],
            dtype=np.float64,
        )
        final_ratios = np.asarray(
            [record[f"final_{component}_ratio"] for record in records],
            dtype=np.float64,
        )
        closest_abs = np.asarray(
            [record[f"closest_{component}_abs_error"] for record in records],
            dtype=np.float64,
        )
        closest_ratios = np.asarray(
            [record[f"closest_{component}_ratio"] for record in records],
            dtype=np.float64,
        )
        component_summary[component] = {
            "unit": UNITS[index],
            "tolerance": float(records[0][f"tolerance_{component}"]),
            "final_mean_abs_error": float(np.mean(final_abs)),
            "final_median_abs_error": float(np.median(final_abs)),
            "final_median_ratio": float(np.median(final_ratios)),
            "final_within_tolerance_rate": float(np.mean(final_ratios <= 1.0)),
            "closest_mean_abs_error": float(np.mean(closest_abs)),
            "closest_median_abs_error": float(np.median(closest_abs)),
            "closest_median_ratio": float(np.median(closest_ratios)),
            "closest_within_tolerance_rate": float(
                np.mean(closest_ratios <= 1.0)
            ),
            "final_dominant_violation_count": int(
                final_dominant.get(component, 0)
            ),
            "closest_dominant_violation_count": int(
                closest_dominant.get(component, 0)
            ),
        }

    return {
        "experiment_name": args.experiment_name,
        "experiment_family": args.experiment_family,
        "training_seed": args.training_seed,
        "model_type": args.model_type,
        "model_path": str(model_path.resolve()),
        "vecnormalize_path": str(vecnormalize_path.resolve()),
        "mode": args.mode,
        "seed_start": args.seed_start,
        "num_rollouts": count,
        "max_steps": args.max_steps,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "failure_rate": float(np.mean([r["failure"] for r in records])),
        "timeout_rate": float(np.mean([r["timeout"] for r in records])),
        "mean_return": float(np.mean([r["episode_return"] for r in records])),
        "final_max_ratio_mean": float(np.mean(final_max)),
        "final_max_ratio_median": float(np.median(final_max)),
        "closest_max_ratio_mean": float(np.mean(closest_max)),
        "closest_max_ratio_median": float(np.median(closest_max)),
        "closest_max_ratio_le_2_rate": float(np.mean(closest_max <= 2.0)),
        "closest_max_ratio_le_5_rate": float(np.mean(closest_max <= 5.0)),
        "final_single_component_miss_rate": float(
            np.mean(
                [r["final_failed_component_count"] == 1 for r in records]
            )
        ),
        "closest_single_component_miss_rate": float(
            np.mean(
                [r["closest_failed_component_count"] == 1 for r in records]
            )
        ),
        "final_dominant_violation_counts": dict(final_dominant),
        "closest_dominant_violation_counts": dict(closest_dominant),
        "component_summary": component_summary,
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(records, output_path):
    final_ratios = np.asarray(
        [
            [record[f"final_{component}_ratio"] for component in COMPONENTS]
            for record in records
        ],
        dtype=np.float64,
    )
    closest_ratios = np.asarray(
        [
            [record[f"closest_{component}_ratio"] for component in COMPONENTS]
            for record in records
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, values, title in (
        (axes[0, 0], final_ratios, "Final component-error ratios"),
        (
            axes[0, 1],
            closest_ratios,
            "Component ratios at best simultaneous approach",
        ),
    ):
        axis.boxplot(
            [np.maximum(values[:, index], 1e-6) for index in range(6)],
            tick_labels=DISPLAY_LABELS,
            showfliers=False,
        )
        axis.axhline(1.0, color="tab:red", linestyle="--", label="Tolerance")
        axis.set_yscale("log")
        axis.set_ylabel("Absolute error / P3 tolerance")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    positions = np.arange(len(COMPONENTS))
    width = 0.38
    final_within = np.mean(final_ratios <= 1.0, axis=0)
    closest_within = np.mean(closest_ratios <= 1.0, axis=0)
    axes[1, 0].bar(
        positions - width / 2,
        final_within,
        width,
        label="Final",
    )
    axes[1, 0].bar(
        positions + width / 2,
        closest_within,
        width,
        label="Best simultaneous approach",
    )
    axes[1, 0].set_xticks(positions, DISPLAY_LABELS)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Fraction within tolerance")
    axes[1, 0].set_title("Per-component tolerance satisfaction")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend()

    dominant_counts = Counter(
        record["closest_dominant_component"]
        for record in records
        if record["closest_dominant_component"] != "none"
    )
    axes[1, 1].bar(
        DISPLAY_LABELS,
        [dominant_counts.get(component, 0) for component in COMPONENTS],
        color="tab:purple",
    )
    axes[1, 1].set_ylabel("Number of rollouts")
    axes[1, 1].set_title("Dominant miss at best simultaneous approach")
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    figure.suptitle("SAC closing P3 component-error diagnosis")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def print_summary(summary):
    print("=" * 88)
    print("Component-wise P3 diagnostic summary")
    print("=" * 88)
    print(f"Rollouts:                    {summary['num_rollouts']}")
    print(f"Success rate:                {summary['success_rate']:.3f}")
    print(f"Failure rate:                {summary['failure_rate']:.3f}")
    print(f"Timeout rate:                {summary['timeout_rate']:.3f}")
    print(
        "Median final max ratio:      "
        f"{summary['final_max_ratio_median']:.3f}"
    )
    print(
        "Median closest max ratio:    "
        f"{summary['closest_max_ratio_median']:.3f}"
    )
    print(
        "Closest max ratio <= 2 rate: "
        f"{summary['closest_max_ratio_le_2_rate']:.3f}"
    )
    print()
    print(
        "component | tolerance | final median ratio | "
        "closest median ratio | closest within tolerance"
    )
    for component in COMPONENTS:
        values = summary["component_summary"][component]
        print(
            f"{component:9s} | "
            f"{values['tolerance']:9.5g} {values['unit']:5s} | "
            f"{values['final_median_ratio']:18.3f} | "
            f"{values['closest_median_ratio']:20.3f} | "
            f"{values['closest_within_tolerance_rate']:24.3f}"
        )
    print("=" * 88)


def main():
    args = parse_args()
    model_label, model_path, vecnormalize_path = resolve_model_paths(args)
    seed_end = args.seed_start + args.num_rollouts - 1
    if args.output_dir is None:
        output_dir = (
            RESULTS_ROOTS[args.experiment_family]
            / args.experiment_name
            / f"seed_{args.training_seed}"
            / "component_evaluations"
            / f"{model_label}_{args.mode}_seeds_{args.seed_start}_{seed_end}"
        )
    else:
        output_dir = args.output_dir.resolve()

    output_files = {
        "rollouts": output_dir / "rollout_component_errors.csv",
        "summary": output_dir / "component_error_summary.json",
        "plot": output_dir / "component_error_diagnostics.png",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist: {existing}. "
            "Pass --overwrite or choose another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_env = DummyVecEnv(
        [
            lambda: make_closing_env(
                mode=args.mode,
                seed=args.seed_start,
                reward_version=args.reward_version,
                monitor=False,
            )
        ]
    )
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(str(model_path), device=args.device)

    records = []
    try:
        for rollout_index, seed in enumerate(
            range(args.seed_start, args.seed_start + args.num_rollouts),
            start=1,
        ):
            record = evaluate_one_rollout(
                model=model,
                vecnormalize=vecnormalize,
                seed=seed,
                mode=args.mode,
                max_steps=args.max_steps,
                reward_version=args.reward_version,
            )
            records.append(record)
            print(
                f"rollout={rollout_index:03d}/{args.num_rollouts:03d} | "
                f"seed={seed} | reason={record['termination_reason']} | "
                f"final_max={record['final_max_ratio']:.3f} | "
                f"closest_max={record['closest_max_ratio']:.3f} | "
                f"closest_dominant={record['closest_dominant_component']}",
                flush=True,
            )
    finally:
        vecnormalize.close()

    summary = summarize(
        records,
        args=args,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
    )
    write_csv(output_files["rollouts"], records)
    with output_files["summary"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    plot_diagnostics(records, output_files["plot"])
    print_summary(summary)
    print(f"Rollout CSV: {output_files['rollouts']}")
    print(f"Summary JSON: {output_files['summary']}")
    print(f"Diagnostic plot: {output_files['plot']}")


if __name__ == "__main__":
    main()
