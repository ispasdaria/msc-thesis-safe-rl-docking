"""Paired zero-residual versus simulator-search terminal controllability study."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from her_closing_utils import make_her_env, physical_goal_metrics
from terminal_reference_residual_env import load_terminal_reset_dataset
from train_sac_closing_reference_residual_lqr_no_her import (
    build_parser as build_reference_parser,
    environment_kwargs,
)


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    CLOSING_DIR / "demonstrations" / "terminal_reference_residual_oracle_bank.npz"
)
DEFAULT_OUTPUT = (
    CLOSING_DIR / "results" / "lqr_reference_residual_terminal" / "oracle_study"
)
REWARD_VERSION = "lqr_reference_residual"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a paired terminal residual-action oracle diagnostic."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-states", type=int, default=30)
    parser.add_argument(
        "--nonsuccess-fraction",
        type=float,
        default=0.5,
        help="Requested fraction from failure/time-limit source episodes.",
    )
    parser.add_argument("--sample-seed", type=int, default=91_000)
    parser.add_argument("--rollout-horizon", type=int, default=1_500)
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument(
        "--action-levels",
        type=float,
        nargs="+",
        default=[-1.0, 0.0, 1.0],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.set_defaults(_reference_defaults=build_reference_parser().parse_args([]))
    return parser


def candidate_action_grid(levels):
    values = np.asarray(levels, dtype=np.float32)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("action levels must contain at least two finite values.")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("action levels must lie in [-1, 1].")
    values = np.unique(values)
    if not np.any(values == 0.0):
        raise ValueError("action levels must include zero as the paired baseline.")
    return np.asarray(list(itertools.product(values, repeat=3)), dtype=np.float32)


def _reset_explicit(env, state, target):
    return env.reset(
        options={
            "initial_state": np.asarray(state, dtype=np.float64),
            "p3_target": np.asarray(target, dtype=np.float64),
            "mode": "testing",
        }
    )


def _outcome(info):
    if info.get("success", False):
        return "success"
    if info.get("failure", False):
        return "failure"
    return "time_limit"


def rollout_zero(env, state, target, horizon):
    _reset_explicit(env, state, target)
    best_ratio = physical_goal_metrics(env)["max_target_ratio"]
    total_reward = 0.0
    final_info = {}
    for step in range(horizon):
        _, reward, terminated, truncated, final_info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        total_reward += float(reward)
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
        "termination_reason": str(final_info.get("termination_reason", "")),
    }


def score_constant_action(planning_env, state, target, action, lookahead_steps):
    _reset_explicit(planning_env, state, target)
    final_info = {}
    for step in range(lookahead_steps):
        _, _, terminated, truncated, final_info = planning_env.step(action)
        if terminated or truncated:
            break
    if final_info.get("success", False):
        return -1_000.0 + 0.01 * (step + 1)
    if final_info.get("failure", False):
        return 1_000.0
    metrics = physical_goal_metrics(planning_env)
    # The tiny norm term resolves numerically equivalent candidates in favor
    # of the smallest intervention. Zero is always present in the grid.
    return float(metrics["max_target_ratio"] + 1e-5 * np.dot(action, action))


def rollout_grid_oracle(
    env,
    planning_env,
    state,
    target,
    horizon,
    lookahead_steps,
    candidates,
):
    _reset_explicit(env, state, target)
    best_ratio = physical_goal_metrics(env)["max_target_ratio"]
    total_reward = 0.0
    action_square_sum = 0.0
    nonzero_steps = 0
    final_info = {}
    for step in range(horizon):
        current_state = np.asarray(env.core_env.state, dtype=np.float64).copy()
        scores = np.asarray(
            [
                score_constant_action(
                    planning_env,
                    current_state,
                    target,
                    action,
                    lookahead_steps,
                )
                for action in candidates
            ],
            dtype=np.float64,
        )
        action = candidates[int(np.argmin(scores))]
        _, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        action_square_sum += float(np.dot(action, action))
        nonzero_steps += int(np.any(action != 0.0))
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
        "termination_reason": str(final_info.get("termination_reason", "")),
        "action_rms": float(np.sqrt(action_square_sum / max(step + 1, 1))),
        "nonzero_action_fraction": float(nonzero_steps / max(step + 1, 1)),
    }


def select_paired_indices(dataset, n_states, seed, nonsuccess_fraction=0.5):
    if n_states <= 0:
        raise ValueError("n_states must be positive.")
    if not 0.0 <= nonsuccess_fraction <= 1.0:
        raise ValueError("nonsuccess_fraction must lie in [0, 1].")
    rng = np.random.default_rng(seed)
    source_seeds = np.asarray(dataset["source_seeds"], dtype=np.int64)
    source_outcomes = np.asarray(dataset["source_outcomes"]).astype(str)
    unique_seeds = np.unique(source_seeds)
    if n_states > unique_seeds.size:
        raise ValueError(
            f"Requested {n_states} states but dataset has only "
            f"{unique_seeds.size} source episodes."
        )
    successful_seeds = np.unique(source_seeds[source_outcomes == "success"])
    nonsuccessful_seeds = np.unique(source_seeds[source_outcomes != "success"])
    requested_nonsuccess = min(
        int(round(n_states * nonsuccess_fraction)), len(nonsuccessful_seeds)
    )
    requested_success = min(n_states - requested_nonsuccess, len(successful_seeds))
    # Fill any shortfall from whichever outcome group still has unused episodes.
    shortfall = n_states - requested_nonsuccess - requested_success
    if shortfall:
        nonsuccess_spare = len(nonsuccessful_seeds) - requested_nonsuccess
        add_nonsuccess = min(shortfall, nonsuccess_spare)
        requested_nonsuccess += add_nonsuccess
        shortfall -= add_nonsuccess
    if shortfall:
        requested_success += shortfall
    chosen_parts = []
    if requested_nonsuccess:
        chosen_parts.append(
            rng.choice(
                nonsuccessful_seeds, size=requested_nonsuccess, replace=False
            )
        )
    if requested_success:
        chosen_parts.append(
            rng.choice(successful_seeds, size=requested_success, replace=False)
        )
    chosen_seeds = np.concatenate(chosen_parts)
    rng.shuffle(chosen_seeds)
    indices = []
    for source_seed in chosen_seeds:
        choices = np.flatnonzero(source_seeds == source_seed)
        indices.append(int(rng.choice(choices)))
    return np.asarray(indices, dtype=np.int64)


def summarize(rows):
    count = len(rows)
    baseline_successes = sum(row["baseline_outcome"] == "success" for row in rows)
    oracle_successes = sum(row["oracle_outcome"] == "success" for row in rows)
    rescued = sum(
        row["baseline_outcome"] != "success" and row["oracle_outcome"] == "success"
        for row in rows
    )
    lost = sum(
        row["baseline_outcome"] == "success" and row["oracle_outcome"] != "success"
        for row in rows
    )
    return {
        "state_count": count,
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / count,
        "oracle_successes": oracle_successes,
        "oracle_success_rate": oracle_successes / count,
        "oracle_rescues": rescued,
        "oracle_losses": lost,
        "mean_baseline_best_ratio": float(
            np.mean([row["baseline_best_ratio"] for row in rows])
        ),
        "mean_oracle_best_ratio": float(
            np.mean([row["oracle_best_ratio"] for row in rows])
        ),
        "mean_oracle_nonzero_action_fraction": float(
            np.mean([row["oracle_nonzero_action_fraction"] for row in rows])
        ),
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_states <= 0 or args.rollout_horizon <= 0 or args.lookahead_steps <= 0:
        parser.error("state count and horizons must be positive.")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output_dir}. Pass --overwrite to replace it."
        )
    dataset = load_terminal_reset_dataset(args.dataset)
    indices = select_paired_indices(
        dataset,
        args.n_states,
        args.sample_seed,
        nonsuccess_fraction=args.nonsuccess_fraction,
    )
    candidates = candidate_action_grid(args.action_levels)
    env_kwargs = environment_kwargs(args._reference_defaults)
    env = make_her_env(
        seed=args.sample_seed,
        mode="testing",
        max_time=args.rollout_horizon,
        reward_version=REWARD_VERSION,
        **env_kwargs,
    )
    planning_env = make_her_env(
        seed=args.sample_seed + 1,
        mode="testing",
        max_time=max(args.lookahead_steps + 1, 10),
        reward_version=REWARD_VERSION,
        **env_kwargs,
    )
    rows = []
    try:
        for ordinal, index in enumerate(indices, start=1):
            state = dataset["states"][index]
            target = dataset["p3_targets"][index]
            baseline = rollout_zero(
                env, state, target, horizon=args.rollout_horizon
            )
            oracle = rollout_grid_oracle(
                env,
                planning_env,
                state,
                target,
                horizon=args.rollout_horizon,
                lookahead_steps=args.lookahead_steps,
                candidates=candidates,
            )
            row = {
                "ordinal": ordinal,
                "dataset_index": int(index),
                "source_seed": int(dataset["source_seeds"][index]),
                "source_step": int(dataset["source_steps"][index]),
                "source_outcome": str(dataset["source_outcomes"][index]),
                "initial_max_target_ratio": float(
                    dataset["max_target_ratios"][index]
                ),
            }
            row.update({f"baseline_{key}": value for key, value in baseline.items()})
            row.update({f"oracle_{key}": value for key, value in oracle.items()})
            rows.append(row)
            print(
                f"[{ordinal:02d}/{len(indices):02d}] seed={row['source_seed']} "
                f"ratio={row['initial_max_target_ratio']:.2f} "
                f"zero={baseline['outcome']} oracle={oracle['outcome']}"
            )
    finally:
        env.close()
        planning_env.close()

    summary = summarize(rows)
    config = {
        "dataset": str(args.dataset.resolve()),
        "sample_seed": args.sample_seed,
        "n_states": args.n_states,
        "nonsuccess_fraction_requested": args.nonsuccess_fraction,
        "rollout_horizon": args.rollout_horizon,
        "lookahead_steps": args.lookahead_steps,
        "action_levels": [float(x) for x in args.action_levels],
        "candidate_count": len(candidates),
        "interpretation_limit": (
            "Success demonstrates useful bounded residual action authority. "
            "Failure does not prove mathematical uncontrollability because the "
            "oracle uses a finite grid and constant-action lookahead."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
