"""Evaluate the pre-Optuna reward-v15 closing SAC and plot seed 0."""

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import evaluate_sac_closing as evaluator


MODEL_NAME = "sac_closing_p3augobs_rewardv15_300ksteps_seed0_buffer300k"
ARTIFACT_GROUP = "reward_shaping_trials"
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", ARTIFACT_GROUP)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")
PLOT_PATH = os.path.join(
    SCRIPT_DIR,
    "results",
    ARTIFACT_GROUP,
    "plots",
    MODEL_NAME,
    f"{MODEL_NAME}_p3_relative_trajectory_seed0.png",
)


def evaluate_seed(model, vecnorm, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        env = evaluator.make_raw_eval_env(seed=seed, mode="testing")
        obs, _ = env.reset(seed=seed)
        p3_target = env.core_env.p3_target.copy()
        for step in range(12000):
            normalized_obs = evaluator.normalize_observation(vecnorm, obs)
            action, _ = model.predict(normalized_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            if terminated or truncated:
                break
        states = np.asarray(env.get_history_arrays()[1], dtype=float)
        env.close()

    position_errors = states[:, :2] - p3_target[:2]
    angle_errors = np.arctan2(
        np.sin(states[:, 4] - p3_target[4]),
        np.cos(states[:, 4] - p3_target[4]),
    )
    rate_errors = states[:, 5] - p3_target[5]
    position_norms = np.linalg.norm(position_errors, axis=1)

    return {
        "seed": seed,
        "steps": step + 1,
        "position_errors": position_errors,
        "position_norms": position_norms,
        "attitude_errors": np.abs(angle_errors),
        "rate_errors": np.abs(rate_errors),
        "termination_reason": info.get("termination_reason"),
        "min_position_error": float(np.min(position_norms)),
        "terminal_position_error": float(position_norms[-1]),
        "terminal_axial_error": float(abs(position_errors[-1, 0])),
        "terminal_lateral_error": float(abs(position_errors[-1, 1])),
        "terminal_attitude_error": float(abs(angle_errors[-1])),
        "terminal_rate_error": float(abs(rate_errors[-1])),
    }


def plot_seed_zero(result, output_path):
    errors = result["position_errors"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(errors[:, 0], errors[:, 1], linewidth=1.8, label="Trajectory")
    axis.scatter(
        errors[0, 0], errors[0, 1], s=65, marker="o", label="Start", zorder=3
    )
    axis.scatter(
        errors[-1, 0], errors[-1, 1], s=75, marker="x", label="End", zorder=3
    )
    axis.scatter(0.0, 0.0, s=65, marker="+", label="$P_3$ target", zorder=4)
    axis.set_xlabel(r"Axial $P_3$-relative position error, $e_{P_3,x}$ [m]")
    axis.set_ylabel("Lateral $P_3$-relative position\nerror, $e_{P_3,y}$ [m]")
    axis.set_title("Selected Pre-Optimization Reward Candidate\n(Testing Seed 0)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.invert_xaxis()
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.56),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.73)
    figure.savefig(output_path, dpi=200)
    figure.savefig(os.path.splitext(output_path)[0] + ".pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.join(MODEL_DIR, model_name + ".zip")
    vecnorm_path = os.path.join(MODEL_DIR, model_name + "_vecnormalize.pkl")
    output_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, "results", ARTIFACT_GROUP, "baseline_evaluation", model_name
    )
    plot_path = os.path.join(
        output_dir, f"{model_name}_selected_reward_candidate_testing_seed0.png"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(vecnorm_path)
    if args.plot_only and 0 not in args.seeds:
        raise ValueError("--plot-only requires testing seed 0 in --seeds")
    os.makedirs(output_dir, exist_ok=True)

    evaluator.MODEL_DIR = MODEL_DIR
    evaluator.MODEL_PATH = model_path
    evaluator.VECNORM_PATH = vecnorm_path

    with contextlib.redirect_stdout(io.StringIO()):
        vecnorm, dummy_env = evaluator.load_vecnormalize_stats()
        model = evaluator.SAC.load(model_path)

    results = [evaluate_seed(model, vecnorm, seed) for seed in args.seeds]
    dummy_env.close()
    seed_zero = next((result for result in results if result["seed"] == 0), None)
    if not args.no_plot and seed_zero is not None:
        plot_seed_zero(seed_zero, plot_path)

    if args.plot_only:
        print(f"plot_path={plot_path}")
        return

    metric_names = [
        "min_position_error",
        "terminal_position_error",
        "terminal_axial_error",
        "terminal_lateral_error",
        "terminal_attitude_error",
        "terminal_rate_error",
    ]

    success_rate = np.mean([r["termination_reason"] == "success" for r in results])
    timeout_rate = np.mean([r["termination_reason"] == "timeout" for r in results])
    hard_safety_rate = np.mean(
        [r["termination_reason"] == "safety_violation" for r in results]
    )
    summary = {
        "model_name": model_name,
        "testing_seeds": args.seeds,
        "number_of_rollouts": len(results),
        "mean_closest_p3_position_error_m": float(np.mean([r["min_position_error"] for r in results])),
        "mean_terminal_position_error_m": float(np.mean([r["terminal_position_error"] for r in results])),
        "mean_terminal_absolute_axial_error_m": float(np.mean([r["terminal_axial_error"] for r in results])),
        "mean_terminal_absolute_lateral_error_m": float(np.mean([r["terminal_lateral_error"] for r in results])),
        "success_rate": float(success_rate),
        "timeout_rate": float(timeout_rate),
        "hard_safety_violation_rate": float(hard_safety_rate),
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    per_seed_path = os.path.join(output_dir, "per_seed_metrics.csv")
    with open(per_seed_path, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "seed", "steps", "min_position_error", "terminal_position_error",
            "terminal_axial_error", "terminal_lateral_error", "termination_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result[field] for field in fields} for result in results)

    print(f"model_path={model_path}")
    print(f"vecnormalize_path={vecnorm_path}")
    if not args.no_plot and seed_zero is not None:
        print(f"plot_path={plot_path}")
    print("PER_SEED")
    print(
        "seed | min_er | terminal_er | terminal_abs_ex | "
        "terminal_abs_ey | terminal_abs_epsi | terminal_abs_eomega | termination"
    )
    for result in results:
        print(
            f"{result['seed']:4d} | "
            f"{result['min_position_error']:.6f} | "
            f"{result['terminal_position_error']:.6f} | "
            f"{result['terminal_axial_error']:.6f} | "
            f"{result['terminal_lateral_error']:.6f} | "
            f"{result['terminal_attitude_error']:.6f} | "
            f"{result['terminal_rate_error']:.6f} | "
            f"{result['termination_reason']}"
        )

    print("AGGREGATES")
    for metric in metric_names:
        values = np.asarray([result[metric] for result in results])
        print(
            f"{metric} | mean={np.mean(values):.6f} | "
            f"std={np.std(values):.6f} | min={np.min(values):.6f} | "
            f"max={np.max(values):.6f}"
        )
    print("OUTCOME_RATES")
    print(f"success_rate={success_rate:.6f}")
    print(f"timeout_rate={timeout_rate:.6f}")
    print(f"hard_safety_violation_rate={hard_safety_rate:.6f}")
    print(f"summary_path={summary_path}")
    print(f"per_seed_path={per_seed_path}")


if __name__ == "__main__":
    main()
