from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from gymnasium import spaces


CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from p3_stratified_dict_replay_buffer import P3StratifiedDictReplayBuffer


def make_buffer():
    observation_space = spaces.Dict(
        {
            "observation": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    return P3StratifiedDictReplayBuffer(
        buffer_size=100,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        outer_buffer_size=20,
        inner_buffer_size=20,
    )


def add_transition(buffer, ratio, marker):
    observation = {
        "observation": np.full((1, 3), marker, dtype=np.float32),
        "achieved_goal": np.full((1, 2), marker, dtype=np.float32),
        "desired_goal": np.zeros((1, 2), dtype=np.float32),
    }
    next_observation = {
        key: value + 0.1 for key, value in observation.items()
    }
    buffer.add(
        observation,
        next_observation,
        np.full((1, 3), marker, dtype=np.float32),
        np.asarray([marker], dtype=np.float32),
        np.asarray([False]),
        [{"p3_max_target_ratio": ratio}],
    )


def test_partitions_are_nested_and_batch_is_exact_50_25_25():
    buffer = make_buffer()
    add_transition(buffer, ratio=8.0, marker=0.1)
    add_transition(buffer, ratio=4.0, marker=0.2)
    add_transition(buffer, ratio=2.0, marker=0.3)
    add_transition(buffer, ratio=1.0, marker=0.4)

    assert buffer.size() == 4
    assert buffer.outer_buffer.size() == 3
    assert buffer.inner_buffer.size() == 2

    samples = buffer.sample(256)
    assert samples.actions.shape == (256, 3)
    assert samples.observations["observation"].shape == (256, 3)
    assert buffer.last_sample_global_count == 128
    assert buffer.last_sample_outer_count == 64
    assert buffer.last_sample_inner_count == 64
    statistics = buffer.replay_statistics()
    assert statistics["realized_global_sample_fraction"] == pytest.approx(0.5)
    assert statistics["realized_outer_sample_fraction"] == pytest.approx(0.25)
    assert statistics["realized_inner_sample_fraction"] == pytest.approx(0.25)
    assert statistics["target_partitions_are_nested"] is True


def test_missing_target_partition_falls_back_to_global_sampling():
    buffer = make_buffer()
    add_transition(buffer, ratio=8.0, marker=0.1)
    samples = buffer.sample(256)

    assert samples.rewards.shape == (256, 1)
    assert buffer.last_sample_global_count == 256
    assert buffer.last_sample_outer_count == 0
    assert buffer.last_sample_inner_count == 0
    assert buffer.fallback_to_global_transitions == 128


def test_ratio_is_required_for_every_transition():
    buffer = make_buffer()
    observation = {
        "observation": np.zeros((1, 3), dtype=np.float32),
        "achieved_goal": np.zeros((1, 2), dtype=np.float32),
        "desired_goal": np.zeros((1, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="p3_max_target_ratio"):
        buffer.add(
            observation,
            observation,
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            [{}],
        )
