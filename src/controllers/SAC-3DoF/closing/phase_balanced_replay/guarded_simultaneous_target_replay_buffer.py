"""Guarded simultaneous-target replay for closing SAC.

V2 preserves a non-overwriting anchor partition from the frozen source policy,
defines capture by the true six-component P3 tolerance ratio, and balances
event categories so repeated braking detections cannot dominate replay.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_balanced_replay_buffer_implementation import PhaseBalancedReplayBuffer


class SimultaneousTargetReplayBuffer(PhaseBalancedReplayBuffer):
    """Replay 50% recent, 25% immutable anchor, and 25% target experience."""

    EVENT_BEST_TARGET = "best_simultaneous_target"
    EVENT_CLOSEST_APPROACH = "closest_approach"
    EVENT_STOPPING_MARGIN = "stopping_margin_crossing"
    EVENT_OVERSHOOT = "overshoot_onset"
    EVENT_BRAKING = "braking_onset"
    EVENT_SUCCESS = "success"
    EVENT_NAMES = (
        EVENT_BEST_TARGET,
        EVENT_CLOSEST_APPROACH,
        EVENT_STOPPING_MARGIN,
        EVENT_OVERSHOOT,
        EVENT_BRAKING,
        EVENT_SUCCESS,
    )

    TARGET_RATIO_KEYS = (
        "p3_axial_ratio",
        "p3_lateral_ratio",
        "p3_vx_ratio",
        "p3_vy_ratio",
        "p3_attitude_ratio",
        "p3_rate_ratio",
    )

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        anchor_buffer_size=100_000,
        capture_buffer_size=100_000,
        priority_buffer_size=25_000,
        target_ratio_threshold=10.0,
        anchor_batch_fraction=0.25,
        capture_batch_fraction=0.25,
        priority_capture_fraction=0.50,
        minimum_anchor_transitions=20_000,
        minimum_anchor_episodes=0,
        minimum_capture_transitions=5_000,
        minimum_capture_episodes=5,
        braking_speed_improvement_mps=1e-4,
    ):
        if int(anchor_buffer_size) <= 0:
            raise ValueError("anchor_buffer_size must be positive.")
        if int(minimum_anchor_transitions) <= 0:
            raise ValueError("minimum_anchor_transitions must be positive.")
        if int(minimum_anchor_episodes) < 0:
            raise ValueError("minimum_anchor_episodes cannot be negative.")
        if float(target_ratio_threshold) <= 1.0:
            raise ValueError("target_ratio_threshold must exceed one.")
        for name, value in (
            ("anchor_batch_fraction", anchor_batch_fraction),
            ("capture_batch_fraction", capture_batch_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if float(anchor_batch_fraction) + float(capture_batch_fraction) > 1.0:
            raise ValueError(
                "anchor_batch_fraction + capture_batch_fraction cannot exceed one."
            )

        # The parent provides storage/sample helpers and the ordinary capture
        # ring.  Its single priority ring is replaced below by category rings.
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            capture_buffer_size=capture_buffer_size,
            priority_buffer_size=priority_buffer_size,
            capture_distance_m=1.0,
            capture_batch_fraction=capture_batch_fraction,
            priority_capture_fraction=priority_capture_fraction,
            minimum_capture_transitions=minimum_capture_transitions,
            minimum_capture_episodes=minimum_capture_episodes,
            braking_speed_improvement_mps=braking_speed_improvement_mps,
        )
        self.anchor_buffer_size = int(anchor_buffer_size)
        self.minimum_anchor_transitions = int(minimum_anchor_transitions)
        self.minimum_anchor_episodes = int(minimum_anchor_episodes)
        self.target_ratio_threshold = float(target_ratio_threshold)
        self.anchor_batch_fraction = float(anchor_batch_fraction)

        self._anchor = self._allocate_partition(self.anchor_buffer_size)
        self.anchor_pos = 0
        self.anchor_frozen = False
        self.anchor_episodes_seen = 0

        per_event_capacity = max(
            1,
            int(np.ceil(self.priority_buffer_size / len(self.EVENT_NAMES))),
        )
        self.event_partition_capacity = per_event_capacity
        self._event_partitions = {
            event: self._allocate_partition(per_event_capacity)
            for event in self.EVENT_NAMES
        }
        self._event_positions = {event: 0 for event in self.EVENT_NAMES}
        self._event_full = {event: False for event in self.EVENT_NAMES}

        self.sampled_anchor_transitions = 0
        self.sampled_event_transitions = 0
        self.sampled_event_counts = Counter()
        self.last_sample_anchor_count = 0
        self.last_sample_event_count = 0

        self._episode_has_capture = False
        self._episode_best_ratio = np.inf
        self._episode_best_ratio_transition = None
        self._episode_closest_position_error_m = np.inf
        self._episode_closest_transition = None
        self._episode_recorded_events = set()
        self._previous_port_speed_mps = None
        self._previous_position_error_m = None
        self._previous_velocity_zone = None
        self._previous_overshoot_active = False

    def anchor_size(self):
        return self.anchor_pos

    def event_size(self, event=None):
        if event is not None:
            return (
                self.event_partition_capacity
                if self._event_full[event]
                else self._event_positions[event]
            )
        return sum(self.event_size(name) for name in self.EVENT_NAMES)

    @property
    def replay_ready(self):
        return bool(
            self.anchor_size() >= self.minimum_anchor_transitions
            and self.anchor_episodes_seen >= self.minimum_anchor_episodes
            and self.capture_size() >= self.minimum_capture_transitions
            and self.capture_episodes_seen >= self.minimum_capture_episodes
        )

    @property
    def phase_balancing_active(self):
        return bool(self.anchor_frozen and self.replay_ready)

    def freeze_anchor(self):
        if not self.replay_ready:
            raise RuntimeError("Cannot freeze anchor before replay is ready.")
        self.anchor_frozen = True

    def _append_anchor(self, transition):
        if self.anchor_frozen or self.anchor_pos >= self.anchor_buffer_size:
            return
        self._store_in_partition(self._anchor, self.anchor_pos, transition)
        self.anchor_pos += 1

    def _append_event(self, event, transition):
        if event == self.EVENT_SUCCESS and self._event_full[event]:
            return
        partition = self._event_partitions[event]
        position = self._event_positions[event]
        self._store_in_partition(partition, position, transition)
        position += 1
        if position >= self.event_partition_capacity:
            if event == self.EVENT_SUCCESS:
                # Successful terminal transitions are immutable once retained.
                self._event_full[event] = True
                position = self.event_partition_capacity
            else:
                self._event_full[event] = True
                position = 0
        self._event_positions[event] = position
        self.total_priority_transitions_seen += 1
        self.event_counts.update([event])

    def _target_ratio(self, info):
        components = self._reward_components(info)
        ratios = []
        for key in self.TARGET_RATIO_KEYS:
            value = components.get(key)
            if value is None or not np.isfinite(value):
                return np.inf
            ratios.append(abs(float(value)))
        return float(max(ratios))

    def _record_once(self, event, transition):
        if event in self._episode_recorded_events:
            return
        self._append_event(event, transition)
        self._episode_recorded_events.add(event)

    def _finish_episode(self):
        self.completed_episodes += 1
        if self._episode_has_capture:
            self.capture_episodes_seen += 1
        if self._episode_best_ratio_transition is not None:
            self._append_event(
                self.EVENT_BEST_TARGET,
                self._episode_best_ratio_transition,
            )
        if self._episode_closest_transition is not None:
            self._append_event(
                self.EVENT_CLOSEST_APPROACH,
                self._episode_closest_transition,
            )

        self._episode_has_capture = False
        self._episode_best_ratio = np.inf
        self._episode_best_ratio_transition = None
        self._episode_closest_position_error_m = np.inf
        self._episode_closest_transition = None
        self._episode_recorded_events.clear()
        self._previous_port_speed_mps = None
        self._previous_position_error_m = None
        self._previous_velocity_zone = None
        self._previous_overshoot_active = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Bypass the V1 add method: its distance-only capture and shared
        # priority ring are intentionally replaced here.
        ReplayBuffer.add(self, obs, next_obs, action, reward, done, infos)
        self.total_online_transitions += 1

        info = infos[0]
        timeout = float(info.get("TimeLimit.truncated", False))
        transition = (
            np.asarray(obs[0]).copy(),
            np.asarray(next_obs[0]).copy(),
            np.asarray(action[0]).copy(),
            float(reward[0]),
            float(done[0]),
            timeout,
        )

        if info.get("reset_distribution") != "near_p3":
            self._append_anchor(transition)

        target_ratio = self._target_ratio(info)
        position_error = self._position_error(info, next_obs[0])
        in_capture = bool(target_ratio <= self.target_ratio_threshold)
        if in_capture:
            self._append_capture(transition)
            self.total_capture_transitions_seen += 1
            self._episode_has_capture = True
            if target_ratio < self._episode_best_ratio:
                self._episode_best_ratio = target_ratio
                self._episode_best_ratio_transition = transition
            if position_error < self._episode_closest_position_error_m:
                self._episode_closest_position_error_m = position_error
                self._episode_closest_transition = transition

            velocity_zone = self._velocity_zone(info)
            if (
                self._previous_velocity_zone is not None
                and velocity_zone is not None
                and velocity_zone != self._previous_velocity_zone
            ):
                self._record_once(self.EVENT_STOPPING_MARGIN, transition)

            components = self._reward_components(info)
            overshoot = float(components.get("axial_overshoot", 0.0)) > 0.0
            if overshoot and not self._previous_overshoot_active:
                self._record_once(self.EVENT_OVERSHOOT, transition)

            port_speed = info.get("port_speed")
            approaching = bool(
                self._previous_position_error_m is not None
                and position_error < self._previous_position_error_m
            )
            braking = bool(
                port_speed is not None
                and np.isfinite(port_speed)
                and self._previous_port_speed_mps is not None
                and approaching
                and float(port_speed)
                <= self._previous_port_speed_mps
                - self.braking_speed_improvement_mps
            )
            if braking:
                self._record_once(self.EVENT_BRAKING, transition)

            if bool(info.get("success", False)):
                self._append_event(self.EVENT_SUCCESS, transition)

            self._previous_velocity_zone = velocity_zone
            self._previous_overshoot_active = overshoot

        port_speed = info.get("port_speed")
        self._previous_port_speed_mps = (
            float(port_speed)
            if port_speed is not None and np.isfinite(port_speed)
            else None
        )
        self._previous_position_error_m = position_error

        if bool(done[0]):
            if info.get("reset_distribution") != "near_p3":
                self.anchor_episodes_seen += 1
            self._finish_episode()

    def _event_samples(self, batch_size, env=None):
        available = [name for name in self.EVENT_NAMES if self.event_size(name) > 0]
        if not available or batch_size <= 0:
            return None

        counts = {name: batch_size // len(available) for name in available}
        remainder = batch_size - sum(counts.values())
        for name in np.random.permutation(available)[:remainder]:
            counts[name] += 1

        groups = []
        for name in available:
            count = counts[name]
            if count <= 0:
                continue
            groups.append(
                self._get_partition_samples(
                    self._event_partitions[name],
                    self.event_size(name),
                    count,
                    env=env,
                )
            )
            self.sampled_event_counts[name] += count
        return (
            groups[0]
            if len(groups) == 1
            else self._concatenate_and_shuffle(groups)
        )

    def sample(self, batch_size, env=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size() <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if not self.phase_balancing_active:
            return ReplayBuffer.sample(self, batch_size, env=env)

        anchor_count = int(round(batch_size * self.anchor_batch_fraction))
        capture_count = int(round(batch_size * self.capture_batch_fraction))
        anchor_count = int(np.clip(anchor_count, 0, batch_size))
        capture_count = int(
            np.clip(capture_count, 0, batch_size - anchor_count)
        )
        event_count = 0
        if self.event_size() > 0:
            event_count = int(
                round(capture_count * self.priority_capture_fraction)
            )
        ordinary_capture_count = capture_count - event_count
        global_count = batch_size - anchor_count - capture_count

        groups = []
        if global_count:
            groups.append(ReplayBuffer.sample(self, global_count, env=env))
        if anchor_count:
            groups.append(
                self._get_partition_samples(
                    self._anchor,
                    self.anchor_size(),
                    anchor_count,
                    env=env,
                )
            )
        if ordinary_capture_count:
            groups.append(
                self._get_partition_samples(
                    self._capture,
                    self.capture_size(),
                    ordinary_capture_count,
                    env=env,
                )
            )
        if event_count:
            groups.append(self._event_samples(event_count, env=env))

        samples = self._concatenate_and_shuffle(groups)
        self.sample_calls += 1
        self.balanced_sample_calls += 1
        self.sampled_global_transitions += global_count
        self.sampled_anchor_transitions += anchor_count
        self.sampled_capture_transitions += capture_count
        self.sampled_priority_transitions += event_count
        self.sampled_event_transitions += event_count
        self.last_sample_global_count = global_count
        self.last_sample_anchor_count = anchor_count
        self.last_sample_capture_count = capture_count
        self.last_sample_priority_count = event_count
        self.last_sample_event_count = event_count
        return samples

    def replay_statistics(self):
        sampled_total = (
            self.sampled_global_transitions
            + self.sampled_anchor_transitions
            + self.sampled_capture_transitions
        )
        return {
            "global_buffer_capacity": int(self.buffer_size),
            "global_buffer_size": int(self.size()),
            "anchor_buffer_capacity": self.anchor_buffer_size,
            "anchor_buffer_size": int(self.anchor_size()),
            "anchor_frozen": bool(self.anchor_frozen),
            "capture_buffer_capacity": self.capture_buffer_size,
            "capture_buffer_size": int(self.capture_size()),
            "event_buffer_configured_capacity": self.priority_buffer_size,
            "event_partition_capacity": self.event_partition_capacity,
            "event_buffer_size": int(self.event_size()),
            "event_partition_sizes": {
                name: int(self.event_size(name)) for name in self.EVENT_NAMES
            },
            "target_ratio_threshold": self.target_ratio_threshold,
            "anchor_batch_fraction_target": self.anchor_batch_fraction,
            "capture_batch_fraction_target": self.capture_batch_fraction,
            "priority_capture_fraction_target": self.priority_capture_fraction,
            "minimum_anchor_transitions": self.minimum_anchor_transitions,
            "minimum_anchor_episodes": self.minimum_anchor_episodes,
            "minimum_capture_transitions": self.minimum_capture_transitions,
            "minimum_capture_episodes": self.minimum_capture_episodes,
            "replay_ready": bool(self.replay_ready),
            "phase_balancing_active": bool(self.phase_balancing_active),
            "completed_episodes": int(self.completed_episodes),
            "anchor_episodes_seen": int(self.anchor_episodes_seen),
            "capture_episodes_seen": int(self.capture_episodes_seen),
            "total_online_transitions": int(self.total_online_transitions),
            "total_capture_transitions_seen": int(
                self.total_capture_transitions_seen
            ),
            "total_priority_transitions_seen": int(
                self.total_priority_transitions_seen
            ),
            "capture_transition_rate": float(
                self.total_capture_transitions_seen
                / max(self.total_online_transitions, 1)
            ),
            "event_counts": dict(sorted(self.event_counts.items())),
            "sample_calls": int(self.sample_calls),
            "balanced_sample_calls": int(self.balanced_sample_calls),
            "sampled_global_transitions": int(self.sampled_global_transitions),
            "sampled_anchor_transitions": int(self.sampled_anchor_transitions),
            "sampled_capture_transitions": int(self.sampled_capture_transitions),
            "sampled_event_transitions": int(self.sampled_event_transitions),
            "sampled_event_counts": dict(sorted(self.sampled_event_counts.items())),
            "realized_anchor_sample_fraction": float(
                self.sampled_anchor_transitions / max(sampled_total, 1)
            ),
            "realized_capture_sample_fraction": float(
                self.sampled_capture_transitions / max(sampled_total, 1)
            ),
            "last_sample_global_count": int(self.last_sample_global_count),
            "last_sample_anchor_count": int(self.last_sample_anchor_count),
            "last_sample_capture_count": int(self.last_sample_capture_count),
            "last_sample_event_count": int(self.last_sample_event_count),
        }
