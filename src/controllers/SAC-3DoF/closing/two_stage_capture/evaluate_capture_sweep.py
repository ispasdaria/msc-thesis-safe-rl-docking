"""Evaluate capture SAC success conditional on real long-range handovers."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import (
    CLOSING_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_RESULT_DIR,
    SRC_ROOT,
    handover_metrics,
    load_handover_dataset,
    write_json,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(CLOSING_DIR) not in sys.path:
    sys.path.insert(0, str(CLOSING_DIR))

from capture_safety_terminal_env import CaptureSafetyTerminalEnv  # noqa: E402
from envs.docking3dof.core import Docking3DoFEnv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR / "guarded100k_handover_sweep.npz",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=None,
        help="Defaults to every radius present in the dataset.",
    )
    parser.add_argument("--max-episodes-per-radius", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=2_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to trial artifacts/results/eval_<model stem>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_env(max_time):
    return CaptureSafetyTerminalEnv(
        phase="closing",
        mode="testing",
        max_time=max_time,
    )


def normalize_action(model, vecnormalize, observation):
    normalized = vecnormalize.normalize_obs(
        np.asarray(observation, dtype=np.float32)[None, :]
    )
    action, _ = model.predict(normalized, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def final_entry_classification(state):
    final_env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    final_env.reset(initial_state=np.asarray(state), mode="testing")
    monitor = final_env.monitor_safety(final_env.state)
    return {
        "final_entry_safe": bool(monitor.get("safe", False)),
        "final_entry_warning": bool(monitor.get("warning", False)),
        "final_entry_unsafe": bool(monitor.get("unsafe", False)),
    }


def evaluate_entry(model, vecnormalize, row, max_time):
    env = make_env(max_time)
    observation, _ = env.reset(
        options={
            "initial_state": row["state"],
            "p3_target": row["p3_target"],
            "mode": "testing",
        }
    )
    terminated = truncated = False
    info = {}
    steps = 0
    episode_return = 0.0
    try:
        while not (terminated or truncated):
            action = normalize_action(model, vecnormalize, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps += 1
        terminal_state = env.core_env.state.copy()
        terminal_metrics = handover_metrics(
            terminal_state,
            row["p3_target"],
            env.core_env.p3_target_tolerance,
        )
    finally:
        env.close()

    success = bool(info.get("success", False))
    final_entry = final_entry_classification(terminal_state)
    valid_p3 = bool(success and not final_entry["final_entry_unsafe"])
    return {
        "radius_m": float(row["radius_m"]),
        "source_seed": int(row["seed"]),
        "source_entry_step": int(row["step"]),
        "capture_steps": int(steps),
        "capture_return": float(episode_return),
        "success": success,
        "valid_p3_success": valid_p3,
        "failure": bool(info.get("failure", False)),
        "time_limit": bool(info.get("time_limit", False)),
        "termination_reason": info.get("termination_reason"),
        "initial_position_error_m": float(row["position_error_m"]),
        "initial_lateral_error_m": float(row["lateral_error_m"]),
        "initial_velocity_error_mps": float(row["velocity_error_mps"]),
        "initial_attitude_error_deg": float(row["attitude_error_deg"]),
        "initial_rate_error_degs": float(row["rate_error_degs"]),
        "initial_radial_rate_mps": float(row["radial_rate_mps"]),
        "initial_max_target_ratio": float(row["max_target_ratio"]),
        "final_position_error_m": terminal_metrics["position_error_m"],
        "final_lateral_error_m": terminal_metrics["lateral_error_m"],
        "final_velocity_error_mps": terminal_metrics["velocity_error_mps"],
        "final_attitude_error_deg": terminal_metrics["attitude_error_deg"],
        "final_rate_error_degs": terminal_metrics["rate_error_degs"],
        "final_max_target_ratio": terminal_metrics["max_target_ratio"],
        **final_entry,
    }


def dataset_rows(dataset, radius, limit):
    indices = np.flatnonzero(
        np.isclose(dataset["radii_m"], radius, rtol=0.0, atol=1e-9)
    )
    if limit is not None:
        indices = indices[:limit]
    scalar_map = {
        "radius_m": "radii_m",
        "seed": "seeds",
        "step": "steps",
        "position_error_m": "position_errors_m",
        "lateral_error_m": "lateral_errors_m",
        "velocity_error_mps": "velocity_errors_mps",
        "attitude_error_deg": "attitude_errors_deg",
        "rate_error_degs": "rate_errors_degs",
        "radial_rate_mps": "radial_rates_mps",
        "max_target_ratio": "max_target_ratios",
    }
    for index in indices:
        row = {
            "state": dataset["states"][index],
            "p3_target": dataset["p3_targets"][index],
        }
        row.update(
            {name: dataset[key][index] for name, key in scalar_map.items()}
        )
        yield row


def summarize_radius(records, dataset_metadata, radius):
    n = len(records)
    success_count = sum(record["success"] for record in records)
    valid_count = sum(record["valid_p3_success"] for record in records)
    source_entry_rate = dataset_metadata.get("entry_rates", {}).get(str(radius))
    if source_entry_rate is None:
        source_entry_rate = dataset_metadata.get("entry_rates", {}).get(f"{radius:g}")
    conditional = valid_count / n if n else 0.0
    return {
        "radius_m": float(radius),
        "n_handover_entries_evaluated": n,
        "capture_success_count": int(success_count),
        "valid_p3_success_count": int(valid_count),
        "capture_success_given_entry": success_count / n if n else 0.0,
        "valid_p3_success_given_entry": conditional,
        "long_range_entry_rate": source_entry_rate,
        "estimated_long_range_times_valid_capture_rate": (
            None if source_entry_rate is None else source_entry_rate * conditional
        ),
        "failure_rate": sum(record["failure"] for record in records) / n if n else 0.0,
        "time_limit_rate": sum(record["time_limit"] for record in records) / n if n else 0.0,
        "mean_capture_steps": float(np.mean([r["capture_steps"] for r in records])) if n else None,
        "mean_initial_max_target_ratio": float(
            np.mean([r["initial_max_target_ratio"] for r in records])
        ) if n else None,
        "mean_final_max_target_ratio": float(
            np.mean([r["final_max_target_ratio"] for r in records])
        ) if n else None,
        "termination_reasons": dict(Counter(r["termination_reason"] for r in records)),
    }


def main():
    args = parse_args()
    for path in (args.model, args.vecnormalize, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.max_episodes_per_radius is not None and args.max_episodes_per_radius <= 0:
        raise ValueError("--max-episodes-per-radius must be positive.")

    dataset = load_handover_dataset(args.dataset)
    available = sorted(set(float(value) for value in dataset["radii_m"]))
    radii = available if args.radii is None else sorted(set(args.radii))
    missing = [radius for radius in radii if not any(np.isclose(radius, available))]
    if missing:
        raise ValueError(f"Requested radii absent from dataset: {missing}")

    output_dir = args.output_dir or DEFAULT_RESULT_DIR / f"eval_{args.model.stem}"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory exists: {output_dir}. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy = DummyVecEnv([lambda: make_env(args.max_time)])
    vecnormalize = VecNormalize.load(str(args.vecnormalize), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(str(args.model), device=args.device)
    all_records = []
    summaries = []
    try:
        for radius in radii:
            radius_records = []
            for row in dataset_rows(dataset, radius, args.max_episodes_per_radius):
                result = evaluate_entry(model, vecnormalize, row, args.max_time)
                radius_records.append(result)
                print(
                    f"radius={radius:g} seed={result['source_seed']} "
                    f"valid_p3={result['valid_p3_success']} "
                    f"reason={result['termination_reason']}"
                )
            all_records.extend(radius_records)
            summaries.append(
                summarize_radius(radius_records, dataset["metadata"], radius)
            )
    finally:
        dummy.close()

    csv_path = output_dir / "capture_rollouts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        if all_records:
            writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
            writer.writeheader()
            writer.writerows(all_records)
    write_json(
        output_dir / "capture_radius_summary.json",
        {
            "model": str(args.model.resolve()),
            "vecnormalize": str(args.vecnormalize.resolve()),
            "dataset": str(args.dataset.resolve()),
            "max_time_s": args.max_time,
            "radius_summaries": summaries,
            "selection_rule": (
                "Choose the largest radius whose valid_p3_success_given_entry "
                "meets the predeclared reliability threshold."
            ),
        },
    )
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
