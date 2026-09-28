"""Create separate behavior-cloning and critic-replay views of 10 Hz LQR data.

The behavior-cloning view keeps every successful episode but thins its state-
action pairs in time.  The replay view keeps a smaller, archive-spanning set
of complete episodes at the native 10 Hz resolution so Bellman transitions
remain consecutive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_lqr_replay_dataset import (
    _metadata_from_dataset,
    load_demonstration_dataset,
    save_demonstration_dataset,
    validate_demonstration_dataset,
)


DEMONSTRATION_DIR = SCRIPT_DIR / "demonstrations"
DEFAULT_SOURCE = (
    DEMONSTRATION_DIR
    / "lqr_closing_training_rewardv15_dt_dt0p1_200successes.partial.npz"
)
DEFAULT_BC_OUTPUT = (
    DEMONSTRATION_DIR
    / "lqr_closing_training_rewardv15_dt_dt0p1_bc_stride10_78episodes.npz"
)
DEFAULT_REPLAY_OUTPUT = (
    DEMONSTRATION_DIR
    / "lqr_closing_training_rewardv15_dt_dt0p1_replay_20episodes.npz"
)

TRANSITION_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
)
EPISODE_FIELDS = (
    "episode_seeds",
    "episode_returns",
    "saturation_fractions",
    "initial_states",
    "p3_targets",
    "final_states",
    "termination_reasons",
)


def evenly_spaced_episode_indices(episode_count, selected_count):
    """Select deterministic indices spanning the full source archive."""

    if episode_count <= 0:
        raise ValueError("episode_count must be positive.")
    if not 1 <= selected_count <= episode_count:
        raise ValueError("selected_count must lie in [1, episode_count].")
    if selected_count == episode_count:
        return np.arange(episode_count, dtype=np.int64)
    indices = np.rint(
        np.linspace(0, episode_count - 1, selected_count)
    ).astype(np.int64)
    if len(np.unique(indices)) != selected_count:
        raise RuntimeError("Evenly spaced episode selection produced duplicates.")
    return indices


def select_episode_view(dataset, episode_indices, temporal_stride, role):
    """Return a validated episode-aware view of a demonstration dataset."""

    source_metadata = validate_demonstration_dataset(dataset)
    offsets = np.asarray(dataset["episode_offsets"], dtype=np.int64)
    source_episode_count = len(offsets) - 1
    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    if episode_indices.ndim != 1 or len(episode_indices) == 0:
        raise ValueError("episode_indices must be a non-empty one-dimensional list.")
    if len(np.unique(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices must be unique.")
    if np.any(episode_indices < 0) or np.any(
        episode_indices >= source_episode_count
    ):
        raise ValueError("episode_indices contain an out-of-range index.")
    episode_indices = np.sort(episode_indices)
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive.")

    selections = []
    selected_lengths = []
    for episode_index in episode_indices:
        start = int(offsets[episode_index])
        stop = int(offsets[episode_index + 1])
        selected = np.arange(start, stop, temporal_stride, dtype=np.int64)
        terminal_index = stop - 1
        if selected[-1] != terminal_index:
            selected = np.concatenate(
                (selected, np.asarray([terminal_index], dtype=np.int64))
            )
        selections.append(selected)
        selected_lengths.append(len(selected))

    transition_indices = np.concatenate(selections)
    selected_lengths = np.asarray(selected_lengths, dtype=np.int64)
    selected_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(selected_lengths))
    )
    selected_ids = np.concatenate(
        [
            np.full(length, new_id, dtype=np.int32)
            for new_id, length in enumerate(selected_lengths)
        ]
    )

    result = {
        key: np.asarray(dataset[key])[transition_indices]
        for key in TRANSITION_FIELDS
    }
    result["episode_ids"] = selected_ids
    result["episode_offsets"] = selected_offsets
    for key in EPISODE_FIELDS:
        if key in dataset:
            result[key] = np.asarray(dataset[key])[episode_indices]

    metadata = dict(source_metadata)
    metadata.update(
        {
            "successful_episodes": int(len(episode_indices)),
            "transition_count": int(len(transition_indices)),
            "dataset_role": str(role),
            "source_successful_episodes": int(source_episode_count),
            "source_transition_count": int(source_metadata["transition_count"]),
            "source_episode_ids": episode_indices.tolist(),
            "source_episode_seeds": np.asarray(
                dataset["episode_seeds"], dtype=np.int64
            )[episode_indices].tolist(),
            "temporal_stride": int(temporal_stride),
            "contains_consecutive_transitions": bool(temporal_stride == 1),
        }
    )
    dt_seconds = metadata.get("dt_seconds")
    if dt_seconds is not None:
        metadata["sample_spacing_seconds"] = float(dt_seconds) * temporal_stride
    result["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))
    validate_demonstration_dataset(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Curate separate 10 Hz LQR datasets for BC and SAC replay."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bc-output", type=Path, default=DEFAULT_BC_OUTPUT)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY_OUTPUT)
    parser.add_argument("--bc-stride", type=int, default=10)
    parser.add_argument("--replay-episodes", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bc_stride <= 0:
        parser.error("--bc-stride must be positive.")
    if args.replay_episodes <= 0:
        parser.error("--replay-episodes must be positive.")
    if args.bc_output.resolve() == args.replay_output.resolve():
        parser.error("BC and replay outputs must be different files.")
    return args


def main(argv=None):
    args = parse_args(argv)
    source, metadata = load_demonstration_dataset(args.source)
    episode_count = int(metadata["successful_episodes"])
    if args.replay_episodes > episode_count:
        raise ValueError(
            "Replay episode count exceeds the available successful episodes: "
            f"{args.replay_episodes} > {episode_count}."
        )

    all_episodes = np.arange(episode_count, dtype=np.int64)
    bc_dataset = select_episode_view(
        source,
        episode_indices=all_episodes,
        temporal_stride=args.bc_stride,
        role="behavior_cloning_supervision",
    )
    save_demonstration_dataset(
        args.bc_output,
        bc_dataset,
        overwrite=args.overwrite,
    )

    replay_episode_indices = evenly_spaced_episode_indices(
        episode_count,
        args.replay_episodes,
    )
    replay_dataset = select_episode_view(
        source,
        episode_indices=replay_episode_indices,
        temporal_stride=1,
        role="protected_critic_replay",
    )
    save_demonstration_dataset(
        args.replay_output,
        replay_dataset,
        overwrite=args.overwrite,
    )

    bc_metadata = _metadata_from_dataset(bc_dataset)
    replay_metadata = _metadata_from_dataset(replay_dataset)
    print("Curated dt=0.1 LQR demonstrations")
    print(
        "BC supervision: "
        f"{bc_metadata['successful_episodes']} episodes, "
        f"{bc_metadata['transition_count']:,} state-action pairs, "
        f"stride={bc_metadata['temporal_stride']}"
    )
    print(
        "Protected replay: "
        f"{replay_metadata['successful_episodes']} complete episodes, "
        f"{replay_metadata['transition_count']:,} consecutive transitions"
    )
    print(f"BC output:       {args.bc_output.resolve()}")
    print(f"Replay output:   {args.replay_output.resolve()}")
    return args.bc_output.resolve(), args.replay_output.resolve()


if __name__ == "__main__":
    main()
