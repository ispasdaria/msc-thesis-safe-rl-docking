"""Resumably generate canonical 10 Hz LQR demonstrations for BC-SAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_lqr_expanded_coverage_dataset import (
    checkpoint_paths,
    merge_demonstration_datasets,
    write_progress,
)
from generate_lqr_replay_dataset import (
    _metadata_from_dataset,
    build_lqr_controller,
    collect_lqr_episode,
    combine_successful_episodes,
    load_demonstration_dataset,
    make_demonstration_env,
    save_demonstration_dataset,
)


DT_SECONDS = 0.1
MAX_TIME_SECONDS = 12_000.0
ACTION_HOLD_STEPS = 1
REWARD_VERSION = "v15_dt"
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "demonstrations"
    / "lqr_closing_training_rewardv15_dt_dt0p1_200successes.npz"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Collect complete successful dt=0.1 s closing-LQR episodes with "
            "validated, resumable checkpoints."
        )
    )
    parser.add_argument("--target-successes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--max-attempts", type=int, default=1_000)
    parser.add_argument("--checkpoint-every-successes", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    for field in (
        "target_successes",
        "max_attempts",
        "checkpoint_every_successes",
    ):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    return args


def timing_metadata():
    return {
        "dt_seconds": DT_SECONDS,
        "control_interval_seconds": DT_SECONDS * ACTION_HOLD_STEPS,
        "control_frequency_hz": 1.0 / (DT_SECONDS * ACTION_HOLD_STEPS),
        "action_hold_steps": ACTION_HOLD_STEPS,
        "max_time_seconds": MAX_TIME_SECONDS,
        "lqr_q_timestep_seconds": DT_SECONDS,
        "integrator": "rk4",
        "dataset_role": "BC supervision and protected SAC replay",
    }


def validate_dt01_metadata(metadata):
    expected = timing_metadata()
    if metadata.get("reward_version") != REWARD_VERSION:
        raise ValueError("Resume dataset reward version is not v15_dt.")
    for key in (
        "dt_seconds",
        "control_interval_seconds",
        "control_frequency_hz",
        "lqr_q_timestep_seconds",
    ):
        if key not in metadata or not np.isclose(metadata[key], expected[key]):
            raise ValueError(
                f"Resume dataset has incompatible {key}: "
                f"{metadata.get(key)!r} != {expected[key]!r}."
            )
    if int(metadata.get("action_hold_steps", -1)) != ACTION_HOLD_STEPS:
        raise ValueError("Resume dataset action_hold_steps is not one.")


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output.resolve()
    partial_path, progress_path = checkpoint_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Completed output already exists: {output_path}.")

    base_dataset = None
    base_count = 0
    existing_seeds = set()
    seed_start = args.seed_start
    if args.resume:
        if not partial_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError(
                "Resume requires both files: "
                f"{partial_path} and {progress_path}."
            )
        base_dataset, metadata = load_demonstration_dataset(partial_path)
        validate_dt01_metadata(metadata)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress["target_successes"]) != args.target_successes:
            raise ValueError("Resume target_successes does not match the CLI.")
        base_count = int(metadata["successful_episodes"])
        existing_seeds = set(
            np.asarray(base_dataset["episode_seeds"], dtype=np.int64).tolist()
        )
        seed_start = int(progress["last_attempted_seed"]) + 1
    elif partial_path.exists() or progress_path.exists():
        raise FileExistsError(
            "Partial collection exists. Pass --resume or choose a new output."
        )

    if base_count >= args.target_successes:
        raise ValueError("The partial dataset already meets the requested target.")

    env = make_demonstration_env(
        mode="training",
        seed=seed_start,
        reward_version=REWARD_VERSION,
        dt=DT_SECONDS,
        max_time=MAX_TIME_SECONDS,
        action_hold_steps=ACTION_HOLD_STEPS,
    )
    controller = build_lqr_controller(env.core_env)
    additions = []
    last_attempted_seed = seed_start - 1

    def current_dataset():
        if not additions:
            if base_dataset is None:
                raise RuntimeError("No successful demonstrations collected yet.")
            return base_dataset
        new_dataset = combine_successful_episodes(
            additions,
            reward_version=REWARD_VERSION,
            mode="training",
            metadata_overrides=timing_metadata(),
        )
        if base_dataset is None:
            return new_dataset
        merged = merge_demonstration_datasets((base_dataset, new_dataset))
        metadata = _metadata_from_dataset(merged)
        metadata.update(timing_metadata())
        merged["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))
        return merged

    def save_checkpoint(complete=False):
        dataset = current_dataset()
        save_demonstration_dataset(partial_path, dataset, overwrite=True)
        metadata = _metadata_from_dataset(dataset)
        record = {
            "complete": bool(complete),
            "target_successes": args.target_successes,
            "last_attempted_seed": last_attempted_seed,
            "successful_episodes": metadata["successful_episodes"],
            "transition_count": metadata["transition_count"],
            "partial_path": str(partial_path),
        }
        write_progress(progress_path, record)
        print(
            "Validated checkpoint saved | "
            f"successes={metadata['successful_episodes']}/"
            f"{args.target_successes} | transitions="
            f"{metadata['transition_count']:,}",
            flush=True,
        )
        return metadata

    print("=" * 88)
    print("Canonical dt=0.1 s closing-LQR demonstration collection")
    print("=" * 88)
    print(f"Existing successes: {base_count}")
    print(f"Target successes:   {args.target_successes}")
    print(f"First seed:          {seed_start}")
    print(f"Maximum attempts:    {args.max_attempts}")
    print(f"Output:              {output_path}")
    print("Held-out seeds 300-899 are not used by this default campaign.")
    print("=" * 88, flush=True)

    interrupted = False
    try:
        for attempt in range(args.max_attempts):
            seed = seed_start + attempt
            last_attempted_seed = seed
            if seed in existing_seeds:
                continue
            episode = collect_lqr_episode(env, controller, seed)
            total = base_count + len(additions)
            print(
                f"attempt={attempt + 1:04d} seed={seed} "
                f"success={episode['success']} "
                f"reason={episode['termination_reason']} "
                f"steps={len(episode['rewards'])} retained={total}",
                flush=True,
            )
            if not episode["success"]:
                continue
            additions.append(episode)
            total = base_count + len(additions)
            if (
                len(additions) % args.checkpoint_every_successes == 0
                or total == args.target_successes
            ):
                save_checkpoint(complete=total == args.target_successes)
            if total == args.target_successes:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        env.close()

    total = base_count + len(additions)
    if total < args.target_successes:
        if additions:
            save_checkpoint(complete=False)
        if interrupted:
            print("Interrupted safely; rerun with --resume.", flush=True)
            return None
        raise RuntimeError(
            f"Collected {total}/{args.target_successes} successes. "
            "Continue the validated partial dataset with --resume."
        )

    if not partial_path.is_file():
        save_checkpoint(complete=True)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        output_path.unlink()
    partial_path.replace(output_path)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["output"] = str(output_path)
    progress["partial_path"] = None
    write_progress(progress_path, progress)
    print(f"Completed dataset: {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    main()
