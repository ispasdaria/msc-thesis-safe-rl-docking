"""Tests for the isolated expanded-demonstration collector."""

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_lqr_expanded_coverage_dataset import (  # noqa: E402
    merge_demonstration_datasets,
)
from generate_lqr_replay_dataset import (  # noqa: E402
    combine_successful_episodes,
    validate_demonstration_dataset,
)


def make_episode(seed, length, offset):
    observations = np.full((length, 12), offset, dtype=np.float32)
    terminated = np.zeros(length, dtype=np.bool_)
    terminated[-1] = True
    return {
        "seed": seed,
        "success": True,
        "termination_reason": "success_P3_target",
        "observations": observations,
        "actions": np.zeros((length, 3), dtype=np.float32),
        "rewards": np.ones(length, dtype=np.float32),
        "next_observations": observations + 1.0,
        "terminated": terminated,
        "truncated": np.zeros(length, dtype=np.bool_),
        "initial_state": np.zeros(6),
        "p3_target": np.ones(6),
        "final_state": np.ones(6),
        "episode_return": float(length),
        "saturation_fraction": 0.0,
    }


def test_merge_preserves_all_complete_episode_boundaries():
    first = combine_successful_episodes(
        [make_episode(20_000, 3, 1.0), make_episode(20_001, 4, 2.0)],
        reward_version="v15",
        mode="training",
    )
    second = combine_successful_episodes(
        [make_episode(20_300, 5, 3.0)],
        reward_version="v15",
        mode="training",
    )

    merged = merge_demonstration_datasets((first, second))
    metadata = validate_demonstration_dataset(merged)

    assert metadata["successful_episodes"] == 3
    assert metadata["transition_count"] == 12
    assert metadata["coverage_expanded"] is True
    assert np.array_equal(merged["episode_offsets"], [0, 3, 7, 12])
    assert np.array_equal(merged["episode_seeds"], [20_000, 20_001, 20_300])
    assert np.array_equal(
        merged["episode_ids"],
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    )


def test_merge_rejects_different_collection_modes():
    training = combine_successful_episodes(
        [make_episode(1, 2, 0.0)],
        reward_version="v15",
        mode="training",
    )
    testing = combine_successful_episodes(
        [make_episode(2, 2, 0.0)],
        reward_version="v15",
        mode="testing",
    )

    with pytest.raises(ValueError, match="incompatible"):
        merge_demonstration_datasets((training, testing))
