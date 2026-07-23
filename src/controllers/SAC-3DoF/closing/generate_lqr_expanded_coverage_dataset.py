"""Expand a successful LQR demonstration dataset with new training seeds.

This is an isolated data-coverage experiment.  It imports the established LQR
collector and dataset validation code without modifying the simulator, Gym
environment, LQR controller, or earlier demonstration dataset.

The collector periodically writes a validated ``.partial.npz`` checkpoint.
If collection is interrupted, rerun the same command with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_lqr_replay_dataset import (
    DEFAULT_OUTPUT_DIR,
    _metadata_from_dataset,
    build_lqr_controller,
    collect_lqr_episode,
    combine_successful_episodes,
    load_demonstration_dataset,
    make_demonstration_env,
    save_demonstration_dataset,
    validate_demonstration_dataset,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DATASET = (
    DEFAULT_OUTPUT_DIR
    / "lqr_closing_training_rewardv15_28successes.npz"
)
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "lqr_closing_training_rewardv15_expanded_200successes.npz"
)

TRANSITION_ARRAY_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
)
EPISODE_ARRAY_KEYS = (
    "episode_seeds",
    "episode_returns",
    "saturation_fractions",
    "initial_states",
    "p3_targets",
    "final_states",
    "termination_reasons",
)


def merge_demonstration_datasets(datasets):
    """Merge validated datasets while preserving complete episode boundaries."""

    datasets = list(datasets)
    if not datasets:
        raise ValueError("At least one demonstration dataset is required.")

    metadata_values = [validate_demonstration_dataset(data) for data in datasets]
    reference = metadata_values[0]
    compatibility_keys = (
        "schema_version",
        "phase",
        "mode",
        "reward_version",
        "normalize_action",
        "observation_size",
        "action_size",
        "controller",
    )
    for metadata in metadata_values[1:]:
        mismatches = {
            key: (reference.get(key), metadata.get(key))
            for key in compatibility_keys
            if reference.get(key) != metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                "Cannot merge incompatible demonstration datasets: "
                f"{mismatches}."
            )

    merged = {
        key: np.concatenate([np.asarray(data[key]) for data in datasets], axis=0)
        for key in TRANSITION_ARRAY_KEYS + EPISODE_ARRAY_KEYS
    }
    episode_lengths = np.concatenate(
        [
            np.diff(np.asarray(data["episode_offsets"], dtype=np.int64))
            for data in datasets
        ]
    )
    merged["episode_offsets"] = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(episode_lengths, dtype=np.int64),
        )
    )
    merged["episode_ids"] = np.concatenate(
        [
            np.full(length, episode_id, dtype=np.int32)
            for episode_id, length in enumerate(episode_lengths)
        ]
    )

    metadata = dict(reference)
    metadata["successful_episodes"] = int(len(merged["episode_seeds"]))
    metadata["transition_count"] = int(len(merged["rewards"]))
    metadata["coverage_expanded"] = True
    merged["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))
    validate_demonstration_dataset(merged)
    return merged


def checkpoint_paths(output_path):
    output_path = Path(output_path)
    partial_path = output_path.with_name(f"{output_path.stem}.partial.npz")
    progress_path = output_path.with_name(f"{output_path.stem}.progress.json")
    return partial_path, progress_path


def write_progress(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(values, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand the successful training-range LQR dataset using new seeds, "
            "with resumable partial checkpoints."
        )
    )
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--target-successes", type=int, default=200)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=20_300,
        help="First new training-environment seed (default: 20300).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2_500,
        help="Maximum new LQR rollouts attempted in this invocation.",
    )
    parser.add_argument(
        "--checkpoint-every-successes",
        type=int,
        default=10,
        help="Save a validated partial dataset after this many new successes.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from this output's partial NPZ and progress JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing completed output dataset.",
    )
    args = parser.parse_args()

    if args.target_successes <= 0:
        parser.error("--target-successes must be positive.")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive.")
    if args.checkpoint_every_successes <= 0:
        parser.error("--checkpoint-every-successes must be positive.")
    return args


def main():
    args = parse_args()
    output_path = args.output.resolve()
    partial_path, progress_path = checkpoint_paths(output_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Completed output already exists: {output_path}. "
            "Use a different path or pass --overwrite."
        )

    if args.resume:
        if not partial_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError(
                "Resume requires both checkpoint files: "
                f"{partial_path} and {progress_path}."
            )
        base_dataset, base_metadata = load_demonstration_dataset(partial_path)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress["target_successes"]) != args.target_successes:
            raise ValueError(
                "The resume checkpoint target does not match "
                f"--target-successes={args.target_successes}."
            )
        seed_start = int(progress["last_attempted_seed"]) + 1
        print(f"Resuming from validated checkpoint: {partial_path}", flush=True)
    else:
        if partial_path.exists() or progress_path.exists():
            raise FileExistsError(
                "Partial collection files already exist. Pass --resume to "
                "continue them, or choose a different --output path."
            )
        base_dataset, base_metadata = load_demonstration_dataset(
            args.base_dataset
        )
        seed_start = args.seed_start

    if base_metadata["mode"] != "training":
        raise ValueError(
            "The expanded coverage experiment requires a training-mode base "
            "dataset."
        )
    base_count = int(base_metadata["successful_episodes"])
    if base_count >= args.target_successes:
        raise ValueError(
            f"Base dataset already has {base_count} episodes, which meets or "
            f"exceeds the target of {args.target_successes}."
        )

    existing_seeds = set(
        np.asarray(base_dataset["episode_seeds"], dtype=np.int64).tolist()
    )
    reward_version = str(base_metadata["reward_version"])
    env = make_demonstration_env(
        mode="training",
        seed=seed_start,
        reward_version=reward_version,
    )
    controller = build_lqr_controller(env.core_env)
    new_successful_episodes = []
    last_attempted_seed = seed_start - 1

    def current_dataset():
        if not new_successful_episodes:
            return base_dataset
        additions = combine_successful_episodes(
            new_successful_episodes,
            reward_version=reward_version,
            mode="training",
        )
        return merge_demonstration_datasets((base_dataset, additions))

    def save_checkpoint():
        combined = current_dataset()
        save_demonstration_dataset(partial_path, combined, overwrite=True)
        checkpoint_metadata = _metadata_from_dataset(combined)
        write_progress(
            progress_path,
            {
                "target_successes": args.target_successes,
                "last_attempted_seed": last_attempted_seed,
                "successful_episodes": checkpoint_metadata[
                    "successful_episodes"
                ],
                "transition_count": checkpoint_metadata["transition_count"],
            },
        )
        print(
            "Checkpoint saved | "
            f"episodes={checkpoint_metadata['successful_episodes']}/"
            f"{args.target_successes} | path={partial_path}",
            flush=True,
        )

    print("=" * 88)
    print("Expanded LQR demonstration coverage collection")
    print("=" * 88)
    print(f"Starting successful episodes: {base_count}")
    print(f"Target successful episodes:   {args.target_successes}")
    print(f"First new seed:               {seed_start}")
    print(f"Maximum attempts this run:    {args.max_attempts}")
    print("Distribution:                 training")
    print("Stored episodes:              complete LQR successes only")
    print("=" * 88, flush=True)

    interrupted = False
    try:
        for attempt_index in range(args.max_attempts):
            seed = seed_start + attempt_index
            last_attempted_seed = seed
            if seed in existing_seeds:
                print(f"seed={seed} already stored; skipping", flush=True)
                continue

            episode = collect_lqr_episode(env, controller, seed)
            print(
                f"attempt={attempt_index + 1:04d} | seed={seed} | "
                f"success={episode['success']} | "
                f"reason={episode['termination_reason']} | "
                f"steps={len(episode['rewards'])} | "
                f"total_successes={base_count + len(new_successful_episodes)}",
                flush=True,
            )
            if episode["success"]:
                new_successful_episodes.append(episode)
                total_successes = base_count + len(new_successful_episodes)
                print(
                    f"  retained {total_successes}/{args.target_successes}",
                    flush=True,
                )
                if (
                    len(new_successful_episodes)
                    % args.checkpoint_every_successes
                    == 0
                ):
                    save_checkpoint()
                if total_successes == args.target_successes:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Collection interrupted; saving retained successes...", flush=True)
    finally:
        env.close()

    total_successes = base_count + len(new_successful_episodes)
    if total_successes < args.target_successes:
        save_checkpoint()
        if interrupted:
            print(
                "Partial progress is safe. Rerun the command with --resume.",
                flush=True,
            )
            return
        raise RuntimeError(
            f"Collected {total_successes}/{args.target_successes} successful "
            f"episodes after {args.max_attempts} new attempts. The validated "
            f"partial dataset is at {partial_path}; continue with --resume."
        )

    final_dataset = current_dataset()
    saved_path = save_demonstration_dataset(
        output_path,
        final_dataset,
        overwrite=args.overwrite,
    )
    final_metadata = _metadata_from_dataset(final_dataset)
    write_progress(
        progress_path,
        {
            "complete": True,
            "target_successes": args.target_successes,
            "last_attempted_seed": last_attempted_seed,
            "successful_episodes": final_metadata["successful_episodes"],
            "transition_count": final_metadata["transition_count"],
            "output": str(saved_path),
        },
    )
    print("=" * 88)
    print("Expanded LQR dataset complete")
    print("=" * 88)
    print(f"Path:                {saved_path}")
    print(f"Successful episodes: {final_metadata['successful_episodes']}")
    print(f"Transitions:         {final_metadata['transition_count']:,}")


if __name__ == "__main__":
    main()
