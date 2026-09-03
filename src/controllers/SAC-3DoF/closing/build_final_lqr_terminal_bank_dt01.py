"""Build a provenance-locked terminal reset bank from the final 10 Hz LQR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from her_closing_utils import make_her_env, physical_goal_metrics
from residual_lqr_config import (
    CLOSING_LQR_CONTROL_HZ,
    CLOSING_LQR_DT_SECONDS,
    steps_for_duration,
)
from terminal_reference_residual_env import (
    PROVENANCE_SCHEMA_VERSION,
    load_terminal_reset_dataset,
    sha256_file,
    terminal_reset_metadata,
    validate_terminal_reset_provenance,
)
from validate_residual_lqr_dt01_baseline import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BASELINE_DIR,
    validate_saved_baseline_manifest,
)


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    CLOSING_DIR
    / "demonstrations"
    / "terminal_reference_residual_final_lqr_dt0p1_bank.npz"
)
DEFAULT_BASELINE_MANIFEST = DEFAULT_BASELINE_DIR / "baseline_manifest.json"
REWARD_VERSION = "lqr_reference_residual"


def checkpoint_paths(output_path):
    output_path = Path(output_path)
    return (
        output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}.progress.json"),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Collect success and timeout terminal states using only the "
            "fingerprinted final dt=0.1 closing LQR."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST
    )
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--target-success-episodes", type=int, default=10)
    parser.add_argument("--target-timeout-episodes", type=int, default=10)
    parser.add_argument("--mode", choices=("training", "testing"), default="training")
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--minimum-ratio", type=float, default=1.05)
    parser.add_argument("--maximum-ratio", type=float, default=5.0)
    parser.add_argument("--snapshot-stride-steps", type=int, default=25)
    parser.add_argument("--max-snapshots-per-episode", type=int, default=1_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(parser, args):
    for name in (
        "max_attempts",
        "target_success_episodes",
        "target_timeout_episodes",
        "snapshot_stride_steps",
        "max_snapshots_per_episode",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_time <= 0.0:
        parser.error("--max-time must be positive.")
    if not 1.0 < args.minimum_ratio < args.maximum_ratio:
        parser.error("ratio limits must satisfy 1 < minimum < maximum.")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")


def _outcome(info, terminated, truncated):
    if bool(info.get("success", False) or info.get("is_success", False)):
        return "success"
    if bool(info.get("failure", False)):
        return "failure"
    if bool(info.get("time_limit", False)) or truncated:
        return "time_limit"
    if terminated:
        return str(info.get("termination_reason", "terminated"))
    return "step_limit"


def _limit_snapshots(records, maximum):
    if len(records) <= maximum:
        return records
    indices = np.unique(
        np.linspace(0, len(records) - 1, num=maximum, dtype=np.int64)
    )
    return [records[int(index)] for index in indices]


def _episode_counts(dataset):
    outcomes = np.asarray(dataset["source_outcomes"]).astype(str)
    seeds = np.asarray(dataset["source_seeds"], dtype=np.int64)
    return {
        outcome: int(len(np.unique(seeds[outcomes == outcome])))
        for outcome in ("success", "time_limit")
    }


def _episode_counts_from_records(records):
    counts = {"success": set(), "time_limit": set()}
    for record in records:
        outcome = str(record["source_outcome"])
        if outcome in counts:
            counts[outcome].add(int(record["source_seed"]))
    return {outcome: len(seeds) for outcome, seeds in counts.items()}


def _dataset_from_records(records, metadata):
    return {
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
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _records_from_dataset(dataset):
    return [
        {
            "state": dataset["states"][index].copy(),
            "p3_target": dataset["p3_targets"][index].copy(),
            "max_target_ratio": float(dataset["max_target_ratios"][index]),
            "source_seed": int(dataset["source_seeds"][index]),
            "source_step": int(dataset["source_steps"][index]),
            "source_outcome": str(dataset["source_outcomes"][index]),
        }
        for index in range(len(dataset["states"]))
    ]


def _write_progress(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    output_path = args.output.resolve()
    manifest_path = args.baseline_manifest.resolve()
    partial_path, progress_path = checkpoint_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (output_path, partial_path, progress_path):
            if path.exists():
                path.unlink()
    elif output_path.exists():
        raise FileExistsError(f"Completed bank already exists: {output_path}")

    baseline = validate_saved_baseline_manifest(manifest_path)
    controller = baseline["controller"]
    maximum_steps = steps_for_duration(args.max_time)
    records = []
    episode_rows = []
    first_seed = int(args.seed_start)

    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError("Resume requires the progress JSON.")
        if partial_path.is_file():
            existing = load_terminal_reset_dataset(partial_path)
            validate_terminal_reset_provenance(
                existing,
                dataset_path=partial_path,
                baseline_manifest=baseline,
                baseline_manifest_path=manifest_path,
                required_outcomes=(),
            )
            metadata = terminal_reset_metadata(existing)
            if (
                int(metadata["target_success_episodes"])
                != args.target_success_episodes
            ):
                raise ValueError("Resume success quota differs from the saved bank.")
            if (
                int(metadata["target_timeout_episodes"])
                != args.target_timeout_episodes
            ):
                raise ValueError("Resume timeout quota differs from the saved bank.")
            records = _records_from_dataset(existing)
            episode_rows = list(metadata.get("episode_rows", []))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        first_seed = int(progress["last_attempted_seed"]) + 1
    elif partial_path.exists() or progress_path.exists():
        raise FileExistsError(
            "A partial bank exists. Pass --resume or choose another output."
        )

    manifest_hash = sha256_file(manifest_path)

    def current_metadata(last_attempted_seed, complete=False):
        counts = _episode_counts_from_records(records)
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "dataset_role": "final dt=0.1 LQR terminal reset bank",
            "controller": "zero-reference-residual final closing LQR",
            "dt_seconds": CLOSING_LQR_DT_SECONDS,
            "control_frequency_hz": CLOSING_LQR_CONTROL_HZ,
            "integrator": str(controller["integrator"]),
            "controller_numeric_fingerprint_sha256": controller[
                "numeric_fingerprint_sha256"
            ],
            "baseline_manifest_path": str(manifest_path),
            "baseline_manifest_sha256": manifest_hash,
            "controller_payload": controller,
            "mode": args.mode,
            "seed_start": int(args.seed_start),
            "last_attempted_seed": int(last_attempted_seed),
            "max_time_seconds": float(args.max_time),
            "max_steps": int(maximum_steps),
            "minimum_ratio": float(args.minimum_ratio),
            "maximum_ratio": float(args.maximum_ratio),
            "snapshot_stride_steps": int(args.snapshot_stride_steps),
            "max_snapshots_per_episode": int(args.max_snapshots_per_episode),
            "target_success_episodes": int(args.target_success_episodes),
            "target_timeout_episodes": int(args.target_timeout_episodes),
            "source_outcome_episode_counts": counts,
            "snapshot_count": len(records),
            "episode_rows": episode_rows,
            "complete": bool(complete),
        }

    def save_partial(last_attempted_seed, complete=False):
        metadata = current_metadata(last_attempted_seed, complete=complete)
        dataset = _dataset_from_records(records, metadata)
        np.savez_compressed(partial_path, **dataset)
        _write_progress(
            progress_path,
            {
                "complete": bool(complete),
                "last_attempted_seed": int(last_attempted_seed),
                "source_outcome_episode_counts": metadata[
                    "source_outcome_episode_counts"
                ],
                "snapshot_count": len(records),
                "partial_path": str(partial_path),
            },
        )
        return metadata

    existing_counts = _episode_counts_from_records(records)
    print("=" * 88)
    print("Final-LQR provenance-locked dt=0.1 terminal bank")
    print("=" * 88)
    print(f"Controller fingerprint: {controller['numeric_fingerprint_sha256']}")
    print(f"Full episode horizon:    {args.max_time:g} s / {maximum_steps:,} steps")
    print(
        "Existing outcomes:      "
        f"success={existing_counts['success']}, "
        f"timeout={existing_counts['time_limit']}"
    )
    print(
        "Target outcomes:        "
        f"success={args.target_success_episodes}, "
        f"timeout={args.target_timeout_episodes}"
    )
    print(f"First attempted seed:    {first_seed}")
    print(f"Output:                  {output_path}")
    print("=" * 88, flush=True)

    env = make_her_env(
        seed=first_seed,
        mode=args.mode,
        max_time=args.max_time,
        reward_version=REWARD_VERSION,
        dt=CLOSING_LQR_DT_SECONDS,
        terminate_on_success=True,
    )
    last_attempted_seed = first_seed - 1
    interrupted = False
    try:
        for attempt in range(args.max_attempts):
            counts = _episode_counts_from_records(records)
            if (
                counts["success"] >= args.target_success_episodes
                and counts["time_limit"] >= args.target_timeout_episodes
            ):
                break

            seed = first_seed + attempt
            last_attempted_seed = seed
            env.reset(seed=seed)
            episode_records = []
            eligible_steps = 0
            final_info = {}
            terminated = truncated = False
            for step in range(maximum_steps):
                metrics = physical_goal_metrics(env)
                ratio = float(metrics["max_target_ratio"])
                if args.minimum_ratio <= ratio <= args.maximum_ratio:
                    if eligible_steps % args.snapshot_stride_steps == 0:
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
                    eligible_steps += 1
                _, _, terminated, truncated, final_info = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                if terminated or truncated:
                    break

            outcome = _outcome(final_info, terminated, truncated)
            quota = {
                "success": args.target_success_episodes,
                "time_limit": args.target_timeout_episodes,
            }.get(outcome, 0)
            retained = bool(
                quota > counts.get(outcome, 0) and len(episode_records) > 0
            )
            stored = _limit_snapshots(
                episode_records, args.max_snapshots_per_episode
            )
            if retained:
                for record in stored:
                    record["source_outcome"] = outcome
                records.extend(stored)
            episode_rows.append(
                {
                    "seed": seed,
                    "steps": step + 1,
                    "elapsed_seconds": float((step + 1) * CLOSING_LQR_DT_SECONDS),
                    "outcome": outcome,
                    "eligible_steps": eligible_steps,
                    "stored_snapshots": len(stored) if retained else 0,
                    "retained": retained,
                    "termination_reason": str(
                        final_info.get("termination_reason", "")
                    ),
                }
            )
            if retained:
                metadata = save_partial(last_attempted_seed, complete=False)
                counts = metadata["source_outcome_episode_counts"]
            print(
                f"attempt={attempt + 1:04d} seed={seed} outcome={outcome} "
                f"steps={step + 1} eligible={eligible_steps} "
                f"stored={len(stored) if retained else 0} retained={retained} "
                f"totals=success:{counts['success']}/"
                f"{args.target_success_episodes},timeout:"
                f"{counts['time_limit']}/{args.target_timeout_episodes}",
                flush=True,
            )
            _write_progress(
                progress_path,
                {
                    "complete": False,
                    "last_attempted_seed": last_attempted_seed,
                    "source_outcome_episode_counts": counts,
                    "snapshot_count": len(records),
                    "partial_path": str(partial_path) if records else None,
                },
            )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        env.close()

    counts = _episode_counts_from_records(records)
    complete = bool(
        counts["success"] >= args.target_success_episodes
        and counts["time_limit"] >= args.target_timeout_episodes
    )
    if not complete:
        message = (
            f"Bank incomplete: success={counts['success']}/"
            f"{args.target_success_episodes}, timeout={counts['time_limit']}/"
            f"{args.target_timeout_episodes}."
        )
        if interrupted:
            print(message + " Resume with --resume.", flush=True)
            return None
        raise RuntimeError(message + " Increase --max-attempts and resume.")

    save_partial(last_attempted_seed, complete=True)
    partial_path.replace(output_path)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update({"complete": True, "output": str(output_path), "partial_path": None})
    _write_progress(progress_path, progress)
    completed = load_terminal_reset_dataset(output_path)
    audit = validate_terminal_reset_provenance(
        completed,
        dataset_path=output_path,
        baseline_manifest=baseline,
        baseline_manifest_path=manifest_path,
    )
    print(f"Completed terminal bank: {output_path}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return output_path


if __name__ == "__main__":
    main()
