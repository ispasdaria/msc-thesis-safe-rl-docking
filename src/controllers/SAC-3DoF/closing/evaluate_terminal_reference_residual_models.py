"""Evaluate baseline and trained residual policies on paired terminal resets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyze_terminal_reference_residual_oracle import select_paired_indices
from evaluate_sac_closing_reference_residual_no_her import environment_kwargs
from her_closing_utils import make_her_env, physical_goal_metrics
from terminal_reference_residual_env import load_terminal_reset_dataset
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = (
    "sac_closing_lqr_reference_residual_"
    "pilot_f_terminal_r2_h600_1hz_50k_seed0"
)
DEFAULT_DATASET = (
    CLOSING_DIR
    / "demonstrations"
    / "terminal_reference_residual_oracle_seeds400_449.npz"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired held-out terminal evaluation for Pilot F."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=92_000)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name", default="heldout_terminal_n50_h600")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _outcome(info):
    if info.get("success", False):
        return "success"
    if info.get("failure", False):
        return "failure"
    return "time_limit"


def rollout(model, env, state, target, horizon):
    observation, _ = env.reset(
        options={
            "initial_state": np.asarray(state, dtype=np.float64),
            "p3_target": np.asarray(target, dtype=np.float64),
            "mode": "testing",
        }
    )
    best_ratio = physical_goal_metrics(env)["max_target_ratio"]
    total_reward = 0.0
    action_square_sum = 0.0
    final_info = {}
    for step in range(horizon):
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        action_square_sum += float(np.dot(action, action))
        best_ratio = min(
            best_ratio, physical_goal_metrics(env)["max_target_ratio"]
        )
        if terminated or truncated:
            break
    final = physical_goal_metrics(env)
    return {
        "outcome": _outcome(final_info),
        "steps": step + 1,
        "return": total_reward,
        "best_ratio": float(best_ratio),
        "final_ratio": float(final["max_target_ratio"]),
        "action_rms": float(np.sqrt(action_square_sum / max(step + 1, 1))),
        "termination_reason": str(final_info.get("termination_reason", "")),
    }


def exact_paired_p_value(rescues, losses):
    discordant = rescues + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(rescues, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result_root = CLOSING_DIR / "results" / "lqr_reference_residual_terminal"
    model_root = CLOSING_DIR / "models" / "lqr_reference_residual_terminal"
    result_dir = result_root / args.experiment_name
    model_dir = model_root / args.experiment_name
    output_dir = result_dir / "evaluations" / args.output_name
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite.")
    config = json.loads(
        (result_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    dataset = load_terminal_reset_dataset(args.dataset)
    indices = select_paired_indices(
        dataset,
        args.n_states,
        args.sample_seed,
        nonsuccess_fraction=0.5,
    )
    models = {
        "lqr_zero_residual": ZeroResidualConstrainedSAC.load(
            model_dir / "checkpoint_0_model.zip", device=args.device
        ),
        "pilot_f_best": ZeroResidualConstrainedSAC.load(
            model_dir / "best_model.zip", device=args.device
        ),
    }
    env = make_her_env(
        seed=args.sample_seed,
        mode="testing",
        max_time=args.horizon,
        reward_version=config["reward_version"],
        **environment_kwargs(config),
    )
    rows = []
    try:
        for ordinal, index in enumerate(indices, start=1):
            common = {
                "ordinal": ordinal,
                "dataset_index": int(index),
                "source_seed": int(dataset["source_seeds"][index]),
                "source_step": int(dataset["source_steps"][index]),
                "source_outcome": str(dataset["source_outcomes"][index]),
                "initial_max_target_ratio": float(
                    dataset["max_target_ratios"][index]
                ),
            }
            for label, model in models.items():
                metrics = rollout(
                    model,
                    env,
                    dataset["states"][index],
                    dataset["p3_targets"][index],
                    args.horizon,
                )
                rows.append({"controller": label, **common, **metrics})
            print(f"[{ordinal:02d}/{len(indices):02d}] source_seed={common['source_seed']}")
    finally:
        env.close()

    grouped = {
        label: [row for row in rows if row["controller"] == label]
        for label in models
    }
    summaries = {}
    for label, controller_rows in grouped.items():
        successes = sum(row["outcome"] == "success" for row in controller_rows)
        failures = sum(row["outcome"] == "failure" for row in controller_rows)
        summaries[label] = {
            "n_states": len(controller_rows),
            "n_success": successes,
            "n_failure": failures,
            "n_timeout": len(controller_rows) - successes - failures,
            "success_rate": successes / len(controller_rows),
            "mean_best_ratio": float(
                np.mean([row["best_ratio"] for row in controller_rows])
            ),
            "mean_final_ratio": float(
                np.mean([row["final_ratio"] for row in controller_rows])
            ),
            "mean_action_rms": float(
                np.mean([row["action_rms"] for row in controller_rows])
            ),
        }
    baseline = {row["dataset_index"]: row for row in grouped["lqr_zero_residual"]}
    learned = {row["dataset_index"]: row for row in grouped["pilot_f_best"]}
    rescues = [
        index
        for index in baseline
        if baseline[index]["outcome"] != "success"
        and learned[index]["outcome"] == "success"
    ]
    losses = [
        index
        for index in baseline
        if baseline[index]["outcome"] == "success"
        and learned[index]["outcome"] != "success"
    ]
    paired = {
        "rescues": len(rescues),
        "losses": len(losses),
        "rescue_source_seeds": [baseline[index]["source_seed"] for index in rescues],
        "loss_source_seeds": [baseline[index]["source_seed"] for index in losses],
        "discordant_pairs": len(rescues) + len(losses),
        "two_sided_exact_p_value": exact_paired_p_value(
            len(rescues), len(losses)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": str(args.dataset.resolve()),
        "sample_seed": args.sample_seed,
        "horizon": args.horizon,
        "controllers": summaries,
        "paired_comparison": paired,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
