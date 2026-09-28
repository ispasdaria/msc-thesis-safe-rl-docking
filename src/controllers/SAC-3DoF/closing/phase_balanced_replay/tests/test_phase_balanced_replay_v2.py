"""Tests for guarded simultaneous-target replay and reset augmentation."""

import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402
from near_p3_reset_wrapper import FixedNearP3ResetWrapper  # noqa: E402
from phase_balanced_replay_buffer_v2 import (  # noqa: E402
    SimultaneousTargetReplayBuffer,
)
from train_sac_closing_v15_phase_balanced_replay_v2 import (  # noqa: E402
    simultaneous_capture_checkpoint_rank,
)


def make_buffer(**overrides):
    kwargs = {
        "buffer_size": 32,
        "observation_space": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        ),
        "action_space": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
        "device": "cpu",
        "anchor_buffer_size": 8,
        "capture_buffer_size": 8,
        "priority_buffer_size": 12,
        "target_ratio_threshold": 10.0,
        "anchor_batch_fraction": 0.25,
        "capture_batch_fraction": 0.25,
        "priority_capture_fraction": 0.50,
        "minimum_anchor_transitions": 2,
        "minimum_capture_transitions": 2,
        "minimum_capture_episodes": 2,
    }
    kwargs.update(overrides)
    return SimultaneousTargetReplayBuffer(**kwargs)


def add_transition(
    buffer,
    value,
    ratios,
    *,
    done=False,
    success=False,
    reset_distribution="standard_p2",
    port_speed=None,
    overshoot=0.0,
):
    obs = np.full((1, 12), value, dtype=np.float32)
    ratio_values = np.asarray(ratios, dtype=np.float64)
    components = {
        "pos_error": float(abs(value)),
        "p3_axial_ratio": ratio_values[0],
        "p3_lateral_ratio": ratio_values[1],
        "p3_vx_ratio": ratio_values[2],
        "p3_vy_ratio": ratio_values[3],
        "p3_attitude_ratio": ratio_values[4],
        "p3_rate_ratio": ratio_values[5],
        "axial_overshoot": overshoot,
    }
    info = {
        "reward_components": components,
        "success": success,
        "reset_distribution": reset_distribution,
    }
    if port_speed is not None:
        info["port_speed"] = port_speed
    buffer.add(
        obs,
        obs + 0.1,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([value], dtype=np.float32),
        np.asarray([done]),
        [info],
    )


def test_anchor_is_immutable_and_batch_is_50_25_25():
    buffer = make_buffer()
    add_transition(buffer, 1.0, [5, 1, 1, 1, 1, 1], done=True)
    add_transition(buffer, 2.0, [4, 1, 1, 1, 1, 1], done=True)
    assert buffer.replay_ready
    buffer.freeze_anchor()
    frozen_anchor_size = buffer.anchor_size()

    for value in range(3, 20):
        add_transition(
            buffer,
            float(value),
            [20, 20, 20, 20, 20, 20],
            done=True,
        )
    assert buffer.anchor_size() == frozen_anchor_size

    samples = buffer.sample(20)
    assert samples.observations.shape == (20, 12)
    assert buffer.last_sample_global_count == 10
    assert buffer.last_sample_anchor_count == 5
    assert buffer.last_sample_capture_count == 5


def test_near_p3_warmup_transitions_do_not_enter_anchor():
    buffer = make_buffer(minimum_anchor_transitions=1, minimum_capture_episodes=1)
    add_transition(
        buffer,
        1.0,
        [2, 2, 2, 2, 2, 2],
        reset_distribution="near_p3",
        done=True,
    )
    assert buffer.anchor_size() == 0
    assert buffer.capture_size() == 1

    add_transition(
        buffer,
        2.0,
        [20, 20, 20, 20, 20, 20],
        reset_distribution="standard_p2",
    )
    assert buffer.anchor_size() == 1


def test_readiness_can_require_complete_standard_anchor_episodes():
    buffer = make_buffer(
        minimum_anchor_transitions=2,
        minimum_anchor_episodes=2,
        minimum_capture_transitions=2,
        minimum_capture_episodes=2,
    )
    add_transition(buffer, 1.0, [2, 1, 1, 1, 1, 1], done=True)
    add_transition(
        buffer,
        2.0,
        [2, 1, 1, 1, 1, 1],
        reset_distribution="near_p3",
        done=True,
    )
    assert buffer.anchor_episodes_seen == 1
    assert not buffer.replay_ready

    add_transition(buffer, 3.0, [2, 1, 1, 1, 1, 1], done=True)
    assert buffer.anchor_episodes_seen == 2
    assert buffer.capture_episodes_seen == 3
    assert buffer.replay_ready


def test_events_are_limited_per_episode_and_sampled_by_category():
    buffer = make_buffer(
        minimum_anchor_transitions=1,
        minimum_capture_transitions=1,
        minimum_capture_episodes=1,
    )
    add_transition(buffer, 5.0, [5, 1, 1, 1, 1, 1], port_speed=0.30)
    add_transition(buffer, 4.0, [4, 1, 1, 1, 1, 1], port_speed=0.20)
    add_transition(buffer, 3.0, [3, 1, 1, 1, 1, 1], port_speed=0.10)
    add_transition(
        buffer,
        2.0,
        [2, 1, 1, 1, 1, 1],
        port_speed=0.05,
        overshoot=1.0,
        done=True,
    )
    assert buffer.event_counts[buffer.EVENT_BRAKING] == 1
    assert buffer.event_counts[buffer.EVENT_OVERSHOOT] == 1
    assert buffer.event_counts[buffer.EVENT_BEST_TARGET] == 1
    assert buffer.event_counts[buffer.EVENT_CLOSEST_APPROACH] == 1

    buffer.freeze_anchor()
    buffer.sample(24)
    sampled = list(buffer.sampled_event_counts.values())
    assert sampled
    assert max(sampled) - min(sampled) <= 1


def test_success_partition_does_not_overwrite_when_full():
    buffer = make_buffer(
        priority_buffer_size=6,
        minimum_anchor_transitions=1,
        minimum_capture_transitions=1,
        minimum_capture_episodes=1,
    )
    for value in (1.0, 2.0, 3.0):
        add_transition(
            buffer,
            value,
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            success=True,
            done=True,
        )
    assert buffer.event_size(buffer.EVENT_SUCCESS) == 1
    assert buffer.event_counts[buffer.EVENT_SUCCESS] == 1


def test_fixed_near_p3_reset_is_safe_and_outside_terminal_set():
    base_env = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=17,
        closing_reward_version="v15",
    )
    env = FixedNearP3ResetWrapper(
        base_env,
        near_p3_fraction=1.0,
        minimum_target_ratio=1.2,
        maximum_target_ratio=4.0,
    )
    try:
        _, info = env.reset(seed=17)
        core = env.core_env
        error = core.state - core.p3_target
        error[4] = core.wrap_to_pi(error[4])
        ratio = float(
            np.max(np.abs(error) / np.asarray(core.p3_target_tolerance))
        )
        assert info["reset_distribution"] == "near_p3"
        assert 1.2 <= ratio <= 4.0
        assert not core.is_at_P3_target(core.state)
        assert not core.monitor_safety(core.state)["unsafe"]
    finally:
        env.close()


def test_v21_reset_mixture_is_deterministic_and_only_one_component_is_outside():
    base_env = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=23,
        closing_reward_version="v15",
    )
    env = FixedNearP3ResetWrapper(
        base_env,
        near_p3_fraction=0.20,
        minimum_target_ratio=1.05,
        maximum_target_ratio=1.50,
        other_component_max_ratio=0.50,
        deterministic_mixture=True,
    )
    try:
        distributions = []
        near_ratios = None
        for reset_index in range(5):
            _, info = env.reset(seed=23 if reset_index == 0 else None)
            distributions.append(info["reset_distribution"])
            if info["reset_distribution"] == "near_p3":
                core = env.core_env
                error = core.state - core.p3_target
                error[4] = core.wrap_to_pi(error[4])
                near_ratios = np.abs(error) / np.asarray(
                    core.p3_target_tolerance
                )

        assert distributions == [
            "standard_p2",
            "standard_p2",
            "standard_p2",
            "standard_p2",
            "near_p3",
        ]
        assert near_ratios is not None
        assert 1.05 <= float(np.max(near_ratios)) <= 1.50
        assert np.count_nonzero(near_ratios > 1.0) == 1
        assert np.count_nonzero(near_ratios > 0.50 + 1e-10) == 1
    finally:
        env.close()


def test_checkpoint_rank_prefers_simultaneous_capture_over_flyby():
    balanced = {
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "mean_best_max_target_ratio": 2.0,
        "mean_target_2x_fraction": 0.01,
        "mean_longest_target_2x_dwell_steps": 10.0,
        "mean_final_max_target_ratio": 4.0,
    }
    flyby = {
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "mean_best_max_target_ratio": 10.0,
        "mean_target_2x_fraction": 0.0,
        "mean_longest_target_2x_dwell_steps": 0.0,
        "mean_final_max_target_ratio": 20.0,
    }
    assert simultaneous_capture_checkpoint_rank(
        balanced,
        mean_return=-10_000.0,
        score=500.0,
    ) > simultaneous_capture_checkpoint_rank(
        flyby,
        mean_return=1_000.0,
        score=900.0,
    )
