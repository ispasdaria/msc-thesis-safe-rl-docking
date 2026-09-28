"""Online phase-balanced replay for closing SAC.

The ordinary Stable-Baselines3 replay ring remains the global partition.  A
second ring stores the agent's own near-P3 transitions.  Once that ring has a
minimum amount of data, every sampled batch contains a fixed global/capture
mixture.  No LQR actions or offline demonstrations are used.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PhaseBalancedReplayBuffer(ReplayBuffer):
    """Stratify SAC batches between global and near-capture experience.

    The inherited replay arrays contain every online transition and preserve
    the ordinary SAC distribution.  Near-P3 transitions are also copied into
    a smaller capture ring.  Important capture events are copied once more to
    a protected priority ring so that closest approaches, velocity-warning
    crossings, overshoots, and successful braking remain sampleable.
    """

    EVENT_CLOSEST_APPROACH = "closest_approach"
    EVENT_STOPPING_MARGIN = "stopping_margin_crossing"
    EVENT_OVERSHOOT = "overshoot"
    EVENT_SUCCESSFUL_BRAKING = "successful_braking"
    EVENT_SUCCESS = "success"

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        capture_buffer_size=100_000,
        priority_buffer_size=25_000,
        capture_distance_m=400.0,
        capture_batch_fraction=0.50,
        priority_capture_fraction=0.50,
        minimum_capture_transitions=256,
        minimum_capture_episodes=0,
        braking_speed_improvement_mps=1e-4,
    ):
        if n_envs != 1:
            raise ValueError(
                "PhaseBalancedReplayBuffer currently requires one environment."
            )
        if optimize_memory_usage:
            raise ValueError(
                "PhaseBalancedReplayBuffer does not support "
                "optimize_memory_usage=True."
            )
        if int(capture_buffer_size) <= 0:
            raise ValueError("capture_buffer_size must be positive.")
        if int(priority_buffer_size) <= 0:
            raise ValueError("priority_buffer_size must be positive.")
        if float(capture_distance_m) <= 0.0:
            raise ValueError("capture_distance_m must be positive.")
        for name, value in (
            ("capture_batch_fraction", capture_batch_fraction),
            ("priority_capture_fraction", priority_capture_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if int(minimum_capture_transitions) <= 0:
            raise ValueError("minimum_capture_transitions must be positive.")
        if int(minimum_capture_episodes) < 0:
            raise ValueError("minimum_capture_episodes cannot be negative.")
        if float(braking_speed_improvement_mps) < 0.0:
            raise ValueError(
                "braking_speed_improvement_mps must be non-negative."
            )

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.capture_buffer_size = int(capture_buffer_size)
        self.priority_buffer_size = int(priority_buffer_size)
        self.capture_distance_m = float(capture_distance_m)
        self.capture_batch_fraction = float(capture_batch_fraction)
        self.priority_capture_fraction = float(priority_capture_fraction)
        self.minimum_capture_transitions = int(minimum_capture_transitions)
        self.minimum_capture_episodes = int(minimum_capture_episodes)
        self.braking_speed_improvement_mps = float(
            braking_speed_improvement_mps
        )

        self._capture = self._allocate_partition(self.capture_buffer_size)
        self._priority = self._allocate_partition(self.priority_buffer_size)
        self.capture_pos = 0
        self.capture_full = False
        self.priority_pos = 0
        self.priority_full = False

        self.total_online_transitions = 0
        self.total_capture_transitions_seen = 0
        self.total_priority_transitions_seen = 0
        self.event_counts = Counter()
        self.sample_calls = 0
        self.balanced_sample_calls = 0
        self.sampled_global_transitions = 0
        self.sampled_capture_transitions = 0
        self.sampled_priority_transitions = 0
        self.last_sample_global_count = 0
        self.last_sample_capture_count = 0
        self.last_sample_priority_count = 0
        self._previous_port_speed_mps = None
        self._previous_velocity_zone = None
        self._previous_position_error_m = None
        self._previous_overshoot_active = False
        self._previous_successful_braking = False
        self._episode_has_capture = False
        self._episode_closest_position_error_m = np.inf
        self._episode_closest_transition = None
        self.completed_episodes = 0
        self.capture_episodes_seen = 0

    def _allocate_partition(self, capacity):
        return {
            "observations": np.zeros(
                (capacity, *self.obs_shape),
                dtype=self.observations.dtype,
            ),
            "actions": np.zeros(
                (capacity, self.action_dim),
                dtype=self.actions.dtype,
            ),
            "rewards": np.zeros(capacity, dtype=np.float32),
            "next_observations": np.zeros(
                (capacity, *self.obs_shape),
                dtype=self.next_observations.dtype,
            ),
            "dones": np.zeros(capacity, dtype=np.float32),
            "timeouts": np.zeros(capacity, dtype=np.float32),
        }

    def capture_size(self):
        return self.capture_buffer_size if self.capture_full else self.capture_pos

    def priority_size(self):
        return self.priority_buffer_size if self.priority_full else self.priority_pos

    @property
    def phase_balancing_active(self):
        return bool(
            self.capture_size() >= self.minimum_capture_transitions
            and self.capture_episodes_seen >= self.minimum_capture_episodes
        )

    @staticmethod
    def _reward_components(info):
        components = info.get("reward_components", {})
        return components if isinstance(components, dict) else {}

    def _position_error(self, info, next_obs):
        components = self._reward_components(info)
        for key in ("position_error", "pos_error"):
            value = components.get(key)
            if value is not None and np.isfinite(value):
                return float(value)

        # Every closing observation begins with six docking-port quantities,
        # followed by the six-dimensional P3 error.  Additional observations
        # used by later reward versions are appended after those 12 values.
        flat_next_obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)
        if flat_next_obs.size < 8:
            raise ValueError(
                "Closing observations need at least eight values to infer "
                "P3 position error."
            )
        return float(np.linalg.norm(flat_next_obs[6:8]))

    @staticmethod
    def _velocity_zone(info):
        velocity = info.get("velocity", {})
        if not isinstance(velocity, dict):
            return None
        if bool(velocity.get("unsafe", False)):
            return "unsafe"
        if bool(velocity.get("warn", False)):
            return "warning"
        if bool(velocity.get("safe", False)):
            return "safe"
        return None

    def _capture_events(self, info, position_error):
        components = self._reward_components(info)
        events = set()

        velocity_zone = self._velocity_zone(info)
        if (
            self._previous_velocity_zone is not None
            and velocity_zone is not None
            and velocity_zone != self._previous_velocity_zone
        ):
            events.add(self.EVENT_STOPPING_MARGIN)

        axial_overshoot = components.get(
            "axial_overshoot_m",
            components.get("axial_overshoot", 0.0),
        )
        overshoot_active = float(axial_overshoot) > 0.0
        if overshoot_active and not self._previous_overshoot_active:
            events.add(self.EVENT_OVERSHOOT)

        port_speed = info.get("port_speed")
        radial_velocity = components.get("radial_velocity_mps")
        approaching = (
            float(radial_velocity) < 0.0
            if radial_velocity is not None
            else (
                self._previous_position_error_m is not None
                and position_error < self._previous_position_error_m
            )
        )
        successful_braking = bool(
            port_speed is not None
            and np.isfinite(port_speed)
            and self._previous_port_speed_mps is not None
            and approaching
            and float(port_speed)
            <= self._previous_port_speed_mps
            - self.braking_speed_improvement_mps
        )
        if successful_braking and not self._previous_successful_braking:
            events.add(self.EVENT_SUCCESSFUL_BRAKING)

        if bool(info.get("success", False)):
            events.add(self.EVENT_SUCCESS)
        return (
            events,
            velocity_zone,
            port_speed,
            overshoot_active,
            successful_braking,
        )

    @staticmethod
    def _store_in_partition(partition, index, transition):
        obs, next_obs, action, reward, done, timeout = transition
        partition["observations"][index] = obs
        partition["next_observations"][index] = next_obs
        partition["actions"][index] = action
        partition["rewards"][index] = reward
        partition["dones"][index] = done
        partition["timeouts"][index] = timeout

    def _append_capture(self, transition):
        self._store_in_partition(self._capture, self.capture_pos, transition)
        self.capture_pos += 1
        if self.capture_pos >= self.capture_buffer_size:
            self.capture_full = True
            self.capture_pos = 0

    def _append_priority(self, transition):
        self._store_in_partition(self._priority, self.priority_pos, transition)
        self.priority_pos += 1
        if self.priority_pos >= self.priority_buffer_size:
            self.priority_full = True
            self.priority_pos = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # The global partition is ordinary SB3 replay and receives everything.
        super().add(obs, next_obs, action, reward, done, infos)
        self.total_online_transitions += 1

        info = infos[0]
        position_error = self._position_error(info, next_obs[0])
        (
            events,
            velocity_zone,
            port_speed,
            overshoot_active,
            successful_braking,
        ) = self._capture_events(
            info,
            position_error,
        )
        timeout = float(info.get("TimeLimit.truncated", False))
        transition = (
            np.asarray(obs[0]).copy(),
            np.asarray(next_obs[0]).copy(),
            np.asarray(action[0]).copy(),
            float(reward[0]),
            float(done[0]),
            timeout,
        )

        if position_error <= self.capture_distance_m:
            self._append_capture(transition)
            self.total_capture_transitions_seen += 1
            self._episode_has_capture = True
            if position_error < self._episode_closest_position_error_m:
                self._episode_closest_position_error_m = position_error
                self._episode_closest_transition = transition
            if events:
                self._append_priority(transition)
                self.total_priority_transitions_seen += 1
                self.event_counts.update(events)

        if bool(done[0]):
            self.completed_episodes += 1
            if self._episode_has_capture:
                self.capture_episodes_seen += 1
            if self._episode_closest_transition is not None:
                self._append_priority(self._episode_closest_transition)
                self.total_priority_transitions_seen += 1
                self.event_counts.update([self.EVENT_CLOSEST_APPROACH])
            self._previous_port_speed_mps = None
            self._previous_velocity_zone = None
            self._previous_position_error_m = None
            self._previous_overshoot_active = False
            self._previous_successful_braking = False
            self._episode_has_capture = False
            self._episode_closest_position_error_m = np.inf
            self._episode_closest_transition = None
        else:
            self._previous_port_speed_mps = (
                float(port_speed)
                if port_speed is not None and np.isfinite(port_speed)
                else None
            )
            self._previous_velocity_zone = velocity_zone
            self._previous_position_error_m = position_error
            self._previous_overshoot_active = overshoot_active
            self._previous_successful_braking = successful_braking

    def _get_partition_samples(self, partition, size, batch_size, env=None):
        indices = np.random.randint(0, size, size=batch_size)
        dones = (
            partition["dones"][indices]
            * (1.0 - partition["timeouts"][indices])
        ).reshape(-1, 1)
        data = (
            self._normalize_obs(partition["observations"][indices], env),
            partition["actions"][indices],
            self._normalize_obs(partition["next_observations"][indices], env),
            dones,
            self._normalize_reward(
                partition["rewards"][indices].reshape(-1, 1),
                env,
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _concatenate_and_shuffle(sample_groups):
        fields = (
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
        )
        combined = ReplayBufferSamples(
            *(
                torch.cat(
                    tuple(getattr(samples, field) for samples in sample_groups),
                    dim=0,
                )
                for field in fields
            )
        )
        permutation = torch.randperm(
            combined.observations.shape[0],
            device=combined.observations.device,
        )
        return ReplayBufferSamples(
            *(getattr(combined, field)[permutation] for field in fields)
        )

    def sample(self, batch_size, env=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size() <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        capture_count = 0
        priority_count = 0
        if self.phase_balancing_active:
            capture_count = int(round(batch_size * self.capture_batch_fraction))
            capture_count = int(np.clip(capture_count, 0, batch_size))
            if self.priority_size() > 0:
                priority_count = int(
                    round(capture_count * self.priority_capture_fraction)
                )
                priority_count = int(np.clip(priority_count, 0, capture_count))
        ordinary_capture_count = capture_count - priority_count
        global_count = batch_size - capture_count

        groups = []
        if global_count:
            groups.append(super().sample(global_count, env=env))
        if ordinary_capture_count:
            groups.append(
                self._get_partition_samples(
                    self._capture,
                    self.capture_size(),
                    ordinary_capture_count,
                    env=env,
                )
            )
        if priority_count:
            groups.append(
                self._get_partition_samples(
                    self._priority,
                    self.priority_size(),
                    priority_count,
                    env=env,
                )
            )

        samples = (
            groups[0]
            if len(groups) == 1
            else self._concatenate_and_shuffle(groups)
        )
        self.sample_calls += 1
        if capture_count:
            self.balanced_sample_calls += 1
        self.sampled_global_transitions += global_count
        self.sampled_capture_transitions += capture_count
        self.sampled_priority_transitions += priority_count
        self.last_sample_global_count = global_count
        self.last_sample_capture_count = capture_count
        self.last_sample_priority_count = priority_count
        return samples

    def replay_statistics(self):
        sampled_total = (
            self.sampled_global_transitions + self.sampled_capture_transitions
        )
        return {
            "global_buffer_capacity": int(self.buffer_size),
            "global_buffer_size": int(self.size()),
            "capture_buffer_capacity": self.capture_buffer_size,
            "capture_buffer_size": int(self.capture_size()),
            "priority_buffer_capacity": self.priority_buffer_size,
            "priority_buffer_size": int(self.priority_size()),
            "capture_distance_m": self.capture_distance_m,
            "capture_batch_fraction_target": self.capture_batch_fraction,
            "priority_capture_fraction_target": self.priority_capture_fraction,
            "minimum_capture_transitions": self.minimum_capture_transitions,
            "minimum_capture_episodes": self.minimum_capture_episodes,
            "phase_balancing_active": bool(self.phase_balancing_active),
            "completed_episodes": int(self.completed_episodes),
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
            "sampled_global_transitions": int(
                self.sampled_global_transitions
            ),
            "sampled_capture_transitions": int(
                self.sampled_capture_transitions
            ),
            "sampled_priority_transitions": int(
                self.sampled_priority_transitions
            ),
            "realized_capture_sample_fraction": float(
                self.sampled_capture_transitions / max(sampled_total, 1)
            ),
            "last_sample_global_count": int(self.last_sample_global_count),
            "last_sample_capture_count": int(self.last_sample_capture_count),
            "last_sample_priority_count": int(self.last_sample_priority_count),
        }
