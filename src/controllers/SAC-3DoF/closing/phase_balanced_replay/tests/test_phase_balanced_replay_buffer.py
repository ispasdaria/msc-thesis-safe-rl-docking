"""Tests for online global/capture phase-balanced replay."""

import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_balanced_replay_buffer_implementation import (  # noqa: E402
    PhaseBalancedReplayBuffer,
)
from train_sac_closing_v15_phase_balanced_replay import (  # noqa: E402
    safety_capture_checkpoint_rank,
)


def make_buffer(minimum_capture_transitions=2):
    return PhaseBalancedReplayBuffer(
        buffer_size=16,
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
        capture_buffer_size=8,
        priority_buffer_size=4,
        capture_distance_m=400.0,
        capture_batch_fraction=0.50,
        priority_capture_fraction=0.50,
        minimum_capture_transitions=minimum_capture_transitions,
    )


def add_transition(
    buffer,
    value,
    position_error,
    *,
    position_progress=0.0,
    minimum_position_error=None,
    axial_overshoot_m=0.0,
    radial_velocity_mps=0.0,
    port_speed=None,
    velocity=None,
    success=False,
    done=False,
):
    observation = np.full((1, 12), value, dtype=np.float32)
    reward_components = {
        "position_error": position_error,
        "position_progress": position_progress,
        "axial_overshoot_m": axial_overshoot_m,
        "radial_velocity_mps": radial_velocity_mps,
    }
    info = {
        "reward_components": reward_components,
        "success": success,
    }
    if minimum_position_error is not None:
        info["minimum_position_error_m"] = minimum_position_error
    if port_speed is not None:
        info["port_speed"] = port_speed
    if velocity is not None:
        info["velocity"] = velocity
    buffer.add(
        observation,
        observation + 0.25,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([value], dtype=np.float32),
        np.asarray([done]),
        [info],
    )


def test_global_fallback_then_exact_balanced_batch_counts():
    buffer = make_buffer(minimum_capture_transitions=2)
    add_transition(buffer, -10.0, 1000.0)
    add_transition(buffer, 1.0, 350.0)

    samples = buffer.sample(batch_size=20)
    assert samples.observations.shape == (20, 12)
    assert buffer.last_sample_global_count == 20
    assert buffer.last_sample_capture_count == 0
    assert not buffer.phase_balancing_active

    add_transition(buffer, 2.0, 300.0)
    samples = buffer.sample(batch_size=20)
    assert samples.observations.shape == (20, 12)
    assert buffer.last_sample_global_count == 10
    assert buffer.last_sample_capture_count == 10
    assert buffer.last_sample_priority_count == 0
    assert buffer.phase_balancing_active


def test_priority_events_form_half_of_capture_samples_when_available():
    buffer = make_buffer(minimum_capture_transitions=2)
    add_transition(buffer, 1.0, 350.0)
    add_transition(
        buffer,
        2.0,
        300.0,
        position_progress=1.0,
        minimum_position_error=300.0,
        axial_overshoot_m=0.5,
        success=True,
        done=True,
    )

    samples = buffer.sample(batch_size=20)
    assert samples.observations.shape == (20, 12)
    assert buffer.last_sample_global_count == 10
    assert buffer.last_sample_capture_count == 10
    assert buffer.last_sample_priority_count == 5

    statistics = buffer.replay_statistics()
    assert statistics["total_online_transitions"] == 2
    assert statistics["total_capture_transitions_seen"] == 2
    assert statistics["total_priority_transitions_seen"] == 2
    assert statistics["event_counts"]["closest_approach"] == 1
    assert statistics["event_counts"]["overshoot"] == 1
    assert statistics["event_counts"]["success"] == 1
    assert statistics["realized_capture_sample_fraction"] == pytest.approx(0.5)


def test_velocity_crossing_and_successful_braking_are_priority_events():
    buffer = make_buffer(minimum_capture_transitions=1)
    safe = {"safe": True, "warn": False, "unsafe": False}
    warning = {"safe": False, "warn": True, "unsafe": False}
    add_transition(
        buffer,
        1.0,
        450.0,
        port_speed=0.30,
        velocity=safe,
    )
    add_transition(
        buffer,
        2.0,
        350.0,
        radial_velocity_mps=-0.02,
        port_speed=0.20,
        velocity=warning,
    )

    statistics = buffer.replay_statistics()
    assert statistics["capture_buffer_size"] == 1
    assert statistics["priority_buffer_size"] == 1
    assert statistics["event_counts"]["stopping_margin_crossing"] == 1
    assert statistics["event_counts"]["successful_braking"] == 1


def test_shared_v15_keys_and_internal_closest_approach_are_supported():
    buffer = make_buffer(minimum_capture_transitions=1)
    observation = np.zeros((1, 12), dtype=np.float32)

    def add_v15(position_error, port_speed, overshoot=0.0):
        next_observation = observation.copy()
        next_observation[0, 6] = position_error
        buffer.add(
            observation,
            next_observation,
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.asarray([False]),
            [
                {
                    "port_speed": port_speed,
                    "reward_components": {
                        "pos_error": position_error,
                        "axial_overshoot": overshoot,
                    },
                }
            ],
        )

    add_v15(450.0, 0.30)
    add_v15(190.0, 0.20, overshoot=1.0)
    buffer.add(
        observation,
        observation.copy(),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.asarray([True]),
        [{"reward_components": {"pos_error": 180.0}}],
    )

    statistics = buffer.replay_statistics()
    assert statistics["capture_buffer_size"] == 2
    assert statistics["priority_buffer_size"] == 2
    assert statistics["event_counts"]["closest_approach"] == 1
    assert statistics["event_counts"]["overshoot"] == 1
    assert statistics["event_counts"]["successful_braking"] == 1


def test_activation_can_require_capture_from_multiple_completed_episodes():
    buffer = PhaseBalancedReplayBuffer(
        buffer_size=16,
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
        capture_buffer_size=8,
        priority_buffer_size=4,
        capture_distance_m=400.0,
        minimum_capture_transitions=2,
        minimum_capture_episodes=2,
    )
    add_transition(buffer, 1.0, 300.0, done=True)
    assert not buffer.phase_balancing_active
    assert buffer.capture_episodes_seen == 1

    add_transition(buffer, 2.0, 250.0, done=True)
    assert buffer.phase_balancing_active
    assert buffer.capture_episodes_seen == 2


def test_only_one_true_closest_approach_is_prioritized_per_episode():
    buffer = make_buffer(minimum_capture_transitions=1)
    add_transition(buffer, 1.0, 350.0)
    add_transition(buffer, 2.0, 300.0)
    add_transition(buffer, 3.0, 250.0, done=True)

    statistics = buffer.replay_statistics()
    assert statistics["capture_buffer_size"] == 3
    assert statistics["priority_buffer_size"] == 1
    assert statistics["event_counts"]["closest_approach"] == 1


def test_safety_capture_rank_does_not_reward_short_unsafe_episodes():
    safe_source = {
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "mean_min_position_error": 25.0,
        "mean_final_max_target_ratio": 21.0,
    }
    unsafe_short_episode = {
        "success_rate": 0.0,
        "failure_rate": 1.0,
        "mean_min_position_error": 500.0,
        "mean_final_max_target_ratio": 200.0,
    }
    source_rank = safety_capture_checkpoint_rank(
        safe_source,
        mean_return=-21_500.0,
        score=536.0,
    )
    unsafe_rank = safety_capture_checkpoint_rank(
        unsafe_short_episode,
        mean_return=-16_000.0,
        score=-1_100_000.0,
    )
    assert source_rank > unsafe_rank


@pytest.mark.parametrize(
    "argument,value",
    [
        ("capture_buffer_size", 0),
        ("priority_buffer_size", 0),
        ("capture_distance_m", 0.0),
        ("capture_batch_fraction", 1.1),
        ("priority_capture_fraction", -0.1),
        ("minimum_capture_transitions", 0),
        ("minimum_capture_episodes", -1),
    ],
)
def test_invalid_configuration_is_rejected(argument, value):
    kwargs = {argument: value}
    with pytest.raises(ValueError):
        PhaseBalancedReplayBuffer(
            buffer_size=16,
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
            **kwargs,
        )
