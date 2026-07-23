"""Tests for immutable demonstrations and scheduled mixed replay batches."""

import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_mixed_replay import (  # noqa: E402
    resolve_online_buffer_size,
)


def demonstration_dataset(count=20):
    observations = np.arange(count * 12, dtype=np.float32).reshape(count, 12)
    terminated = np.zeros(count, dtype=np.bool_)
    terminated[-1] = True
    return {
        "observations": observations,
        "actions": np.linspace(-1.0, 1.0, count * 3, dtype=np.float32).reshape(
            count,
            3,
        ),
        "rewards": np.linspace(-2.0, 3.0, count, dtype=np.float32),
        "next_observations": observations + 0.5,
        "terminated": terminated,
        "truncated": np.zeros(count, dtype=np.bool_),
    }


def add_online_transition(buffer, value):
    observation = np.full((1, 12), value, dtype=np.float32)
    buffer.add(
        observation,
        observation + 0.25,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([value], dtype=np.float32),
        np.asarray([False]),
        [{}],
    )


def test_exact_mixed_batch_schedule_and_protected_demonstrations():
    buffer = ProtectedMixedReplayBuffer(
        buffer_size=8,
        observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
        device="cpu",
        initial_demo_fraction=0.50,
        later_demo_fraction=0.25,
        demo_fraction_switch_step=10,
    )
    dataset = demonstration_dataset()
    original_demonstrations = dataset["observations"].copy()
    buffer.set_demonstrations(dataset)

    assert buffer.demonstration_count == 20
    assert not buffer.demonstration_observations.flags.writeable

    for value in range(4):
        add_online_transition(buffer, -100.0 - value)
    samples = buffer.sample(batch_size=12)
    assert samples.observations.shape == (12, 12)
    assert buffer.last_sample_demo_count == 6
    assert buffer.last_sample_online_count == 6
    assert buffer.active_demo_fraction == 0.50

    for value in range(4, 10):
        add_online_transition(buffer, -100.0 - value)
    samples = buffer.sample(batch_size=12)
    assert samples.observations.shape == (12, 12)
    assert buffer.last_sample_demo_count == 3
    assert buffer.last_sample_online_count == 9
    assert buffer.active_demo_fraction == 0.25

    # The online ring has wrapped, but demonstrations remain byte-for-byte intact.
    assert buffer.size() == 8
    assert np.array_equal(
        buffer.demonstration_observations,
        original_demonstrations,
    )


def test_demo_only_fallback_before_online_data_exists():
    buffer = ProtectedMixedReplayBuffer(
        buffer_size=8,
        observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
        device="cpu",
    )
    buffer.set_demonstrations(demonstration_dataset(count=5))
    samples = buffer.sample(batch_size=16)
    assert samples.observations.shape == (16, 12)
    assert buffer.last_sample_demo_count == 16
    assert buffer.last_sample_online_count == 0


def test_online_buffer_capacity_is_independent_of_demonstration_count():
    assert resolve_online_buffer_size(None, 600_000, 50_000) == 700_000
    assert resolve_online_buffer_size(None, 300_000, 0) == 300_000
