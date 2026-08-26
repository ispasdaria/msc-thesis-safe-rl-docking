"""Collect first-entry states from the frozen 100k long-range SAC.

For each deterministic P2/P3 rollout, the script records the first state that
enters each requested capture radius while satisfying the component-specific
envelope.  Continuing a rollout after a larger-radius entry lets every radius
use exactly the same initial state and trajectory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import (
    CLOSING_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_LONG_RANGE_MODEL,
    DEFAULT_LONG_RANGE_VECNORMALIZE,
    REPO_ROOT,
    SRC_ROOT,
    CaptureEnvelope,
    handover_metrics,
    is_capture_envelope_entry,
    save_handover_dataset,
    write_json,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(CLOSING_DIR) not in sys.path:
    sys.path.insert(0, str(CLOSING_DIR))

from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_LONG_RANGE_MODEL)
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=DEFAULT_LONG_RANGE_VECNORMALIZE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "guarded100k_handover_sweep.npz",
    )
    parser.add_argument("--radii", type=float, nargs="+", default=[50, 75, 100, 150])
    parser.add_argument("--lateral-error", type=float, default=15.0)
    parser.add_argument("--velocity-error", type=float, default=0.03)
    parser.add_argument("--attitude-error-deg", type=float, default=3.0)
    parser.add_argument("--rate-error-degs", type=float, default=0.10)
    parser.add_argument(
        "--maximum-radial-rate",
        type=float,
        default=0.0,
        help="Maximum P3-relative radial rate at entry; zero requires approach.",
    )
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["training", "testing", "nominal"],
        default="training",
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    args.radii = sorted(set(float(radius) for radius in args.radii))
    if not args.radii:
        raise ValueError("At least one capture radius is required.")
    for path in (args.model, args.vecnormalize):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )


def make_env(mode, max_time, seed):
    return Docking3DoFGymEnv(
        phase="closing",
        mode=mode,
        max_time=max_time,
        seed=seed,
        closing_reward_version="v15",
    )


def load_policy(model_path, vecnormalize_path, mode, max_time, device):
    dummy = DummyVecEnv([lambda: make_env(mode, max_time, seed=0)])
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(str(model_path), device=device)
    return model, vecnormalize, dummy


def normalized_action(model, vecnormalize, observation):
    normalized = vecnormalize.normalize_obs(
        np.asarray(observation, dtype=np.float32)[None, :]
    )
    action, _ = model.predict(normalized, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def record_for_entry(env, seed, step, radius, metrics):
    return {
        "state": env.core_env.state.copy(),
        "p3_target": env.core_env.p3_target.copy(),
        "radius_m": float(radius),
        "seed": int(seed),
        "step": int(step),
        "time_s": float(env.core_env.time),
        **metrics,
    }


def collect_rollout(model, vecnormalize, seed, args, envelopes):
    env = make_env(args.mode, args.max_time, seed)
    observation, _ = env.reset(seed=seed)
    pending = set(envelopes)
    records = []
    terminated = truncated = False
    final_info = {}
    step = 0

    try:
        while not (terminated or truncated) and pending:
            action = normalized_action(model, vecnormalize, observation)
            observation, _, terminated, truncated, final_info = env.step(action)
            step += 1
            metrics = handover_metrics(
                env.core_env.state,
                env.core_env.p3_target,
                env.core_env.p3_target_tolerance,
            )
            safety = env.core_env.monitor_safety(env.core_env.state)
            for radius in sorted(pending, reverse=True):
                if is_capture_envelope_entry(metrics, envelopes[radius], safety):
                    records.append(record_for_entry(env, seed, step, radius, metrics))
                    pending.remove(radius)
    finally:
        env.close()

    return records, {
        "seed": int(seed),
        "steps": int(step),
        "hit_radii_m": sorted(record["radius_m"] for record in records),
        "missed_radii_m": sorted(pending),
        "source_success": bool(final_info.get("success", False)),
        "source_failure": bool(final_info.get("failure", False)),
        "source_time_limit": bool(final_info.get("time_limit", False)),
        "source_termination_reason": final_info.get("termination_reason"),
    }


def write_records_csv(path, records):
    path = Path(path)
    fieldnames = [
        "seed",
        "radius_m",
        "step",
        "time_s",
        "position_error_m",
        "lateral_error_m",
        "velocity_error_mps",
        "attitude_error_deg",
        "rate_error_degs",
        "radial_rate_mps",
        "max_target_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def main():
    args = parse_args()
    validate_args(args)
    envelopes = {
        radius: CaptureEnvelope(
            radius_m=radius,
            lateral_error_m=args.lateral_error,
            velocity_error_mps=args.velocity_error,
            attitude_error_deg=args.attitude_error_deg,
            rate_error_degs=args.rate_error_degs,
            maximum_radial_rate_mps=args.maximum_radial_rate,
        )
        for radius in args.radii
    }
    for envelope in envelopes.values():
        envelope.validate()

    model, vecnormalize, dummy = load_policy(
        args.model,
        args.vecnormalize,
        args.mode,
        args.max_time,
        args.device,
    )
    all_records = []
    rollout_summaries = []
    try:
        for index in range(args.num_seeds):
            seed = args.seed_start + index
            records, summary = collect_rollout(
                model, vecnormalize, seed, args, envelopes
            )
            all_records.extend(records)
            rollout_summaries.append(summary)
            print(
                f"seed={seed} entries={summary['hit_radii_m']} "
                f"missed={summary['missed_radii_m']} steps={summary['steps']}"
            )
    finally:
        dummy.close()

    hit_counts = {
        str(radius): int(
            sum(np.isclose(record["radius_m"], radius) for record in all_records)
        )
        for radius in args.radii
    }
    metadata = {
        "format_version": 1,
        "repository_root": str(REPO_ROOT),
        "source_model": str(args.model.resolve()),
        "source_vecnormalize": str(args.vecnormalize.resolve()),
        "mode": args.mode,
        "seed_start": args.seed_start,
        "num_rollouts": args.num_seeds,
        "max_time_s": args.max_time,
        "radii_m": args.radii,
        "envelopes": {
            str(radius): envelope.to_dict()
            for radius, envelope in envelopes.items()
        },
        "hit_counts": hit_counts,
        "entry_rates": {
            key: count / args.num_seeds for key, count in hit_counts.items()
        },
    }
    save_handover_dataset(args.output, all_records, metadata)
    write_records_csv(args.output.with_suffix(".csv"), all_records)
    write_json(
        args.output.with_name(args.output.stem + "_summary.json"),
        {**metadata, "rollouts": rollout_summaries},
    )
    print(f"Saved {len(all_records)} handover records to {args.output}")


if __name__ == "__main__":
    main()
