"""Collect real near-P3 LQR states for terminal-reset experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from her_closing_utils import make_her_env, physical_goal_metrics
from train_sac_closing_reference_residual_lqr_no_her import environment_kwargs


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CLOSING_DIR / "demonstrations" / "terminal_reference_residual_bank.npz"
REWARD_VERSION = "lqr_reference_residual"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect empirical terminal reset states from full LQR closings."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=1_000)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--mode", choices=("training", "testing"), default="training")
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--minimum-ratio", type=float, default=1.05)
    parser.add_argument("--maximum-ratio", type=float, default=5.0)
    parser.add_argument("--snapshot-stride", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    # Reuse exactly the Pilot C/D reward and interface defaults.
    from train_sac_closing_reference_residual_lqr_no_her import build_parser as base

    defaults = base().parse_args([])
    parser.set_defaults(_reference_defaults=defaults)
    return parser


def collect_terminal_reset_dataset(
    *,
    seed_start,
    n_seeds,
    mode,
    max_steps,
    minimum_ratio,
    maximum_ratio,
    snapshot_stride,
    env_kwargs,
):
    if n_seeds <= 0 or max_steps <= 0 or snapshot_stride <= 0:
        raise ValueError("n_seeds, max_steps, and snapshot_stride must be positive.")
    if not 1.0 < minimum_ratio < maximum_ratio:
        raise ValueError("Ratio limits must satisfy 1 < minimum < maximum.")

    env = make_her_env(
        seed=seed_start,
        mode=mode,
        max_time=max_steps,
        reward_version=REWARD_VERSION,
        **env_kwargs,
    )
    records = []
    episode_rows = []
    try:
        for offset in range(n_seeds):
            seed = int(seed_start + offset)
            env.reset(seed=seed)
            episode_records = []
            eligible_counter = 0
            final_info = {}
            for step in range(max_steps):
                metrics = physical_goal_metrics(env)
                ratio = float(metrics["max_target_ratio"])
                if minimum_ratio <= ratio <= maximum_ratio:
                    if eligible_counter % snapshot_stride == 0:
                        episode_records.append(
                            {
                                "state": np.asarray(
                                    env.core_env.state, dtype=np.float64
                                ).copy(),
                                "p3_target": np.asarray(
                                    env.core_env.p3_target, dtype=np.float64
                                ).copy(),
                                "max_target_ratio": ratio,
                                "source_seed": seed,
                                "source_step": step,
                            }
                        )
                    eligible_counter += 1
                _, _, terminated, truncated, final_info = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                if terminated or truncated:
                    break
            if final_info.get("success", False):
                outcome = "success"
            elif final_info.get("failure", False):
                outcome = "failure"
            else:
                outcome = "time_limit"
            for record in episode_records:
                record["source_outcome"] = outcome
            records.extend(episode_records)
            episode_rows.append(
                {
                    "seed": seed,
                    "steps": step + 1,
                    "outcome": outcome,
                    "eligible_steps": eligible_counter,
                    "stored_snapshots": len(episode_records),
                    "termination_reason": str(
                        final_info.get("termination_reason", "")
                    ),
                }
            )
            print(
                f"seed={seed} outcome={outcome} steps={step + 1} "
                f"terminal_steps={eligible_counter} snapshots={len(episode_records)}"
            )
    finally:
        env.close()

    if not records:
        raise RuntimeError("No eligible terminal snapshots were collected.")
    dataset = {
        "states": np.stack([record["state"] for record in records]),
        "p3_targets": np.stack([record["p3_target"] for record in records]),
        "max_target_ratios": np.asarray(
            [record["max_target_ratio"] for record in records], dtype=np.float64
        ),
        "source_seeds": np.asarray(
            [record["source_seed"] for record in records], dtype=np.int64
        ),
        "source_steps": np.asarray(
            [record["source_step"] for record in records], dtype=np.int64
        ),
        "source_outcomes": np.asarray(
            [record["source_outcome"] for record in records], dtype="U16"
        ),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "controller": "zero-reference-residual LQR",
                    "seed_start": int(seed_start),
                    "n_seeds": int(n_seeds),
                    "mode": mode,
                    "max_steps": int(max_steps),
                    "minimum_ratio": float(minimum_ratio),
                    "maximum_ratio": float(maximum_ratio),
                    "snapshot_stride": int(snapshot_stride),
                    "snapshot_count": len(records),
                    "episode_rows": episode_rows,
                },
                sort_keys=True,
            )
        ),
    }
    return dataset, episode_rows


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Pass --overwrite.")
    env_kwargs = environment_kwargs(args._reference_defaults)
    dataset, rows = collect_terminal_reset_dataset(
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        mode=args.mode,
        max_steps=args.max_steps,
        minimum_ratio=args.minimum_ratio,
        maximum_ratio=args.maximum_ratio,
        snapshot_stride=args.snapshot_stride,
        env_kwargs=env_kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    outcomes = {name: sum(row["outcome"] == name for row in rows) for name in (
        "success", "failure", "time_limit"
    )}
    print(f"Saved {len(dataset['states']):,} snapshots to {args.output}")
    print(f"Source outcomes: {outcomes}")


if __name__ == "__main__":
    main()
