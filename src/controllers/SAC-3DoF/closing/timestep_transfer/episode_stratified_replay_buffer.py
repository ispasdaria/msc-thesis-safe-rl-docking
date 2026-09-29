"""Episode-aware P3-ratio replay for uniform-timestep closing SAC.

Every transition is assigned to exactly one physical-quality stratum using the
worst normalized P3 component error.  Sampling first chooses an episode and
then a transition from that episode, preventing a single long trajectory from
dominating a stratum merely because it contains more simulator steps.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class EpisodeStratifiedReplayBuffer(ReplayBuffer):
    """Replay balanced by target-quality stratum and completed episode."""

    STRATUM_TARGET_LE_2 = "target_le_2"
    STRATUM_TARGET_2_TO_5 = "target_2_to_5"
    STRATUM_TARGET_5_TO_10 = "target_5_to_10"
    STRATUM_APPROACH = "approach_gt_10"
    STRATA = (
        STRATUM_TARGET_LE_2,
        STRATUM_TARGET_2_TO_5,
        STRATUM_TARGET_5_TO_10,
        STRATUM_APPROACH,
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
        target_le_2_capacity=25_000,
        target_2_to_5_capacity=75_000,
        target_5_to_10_capacity=75_000,
        approach_capacity=400_000,
        target_le_2_fraction=0.10,
        target_2_to_5_fraction=0.30,
        target_5_to_10_fraction=0.25,
        approach_fraction=0.35,
        minimum_completed_episodes=20,
        minimum_standard_episodes=4,
        minimum_near_p3_episodes=16,
        minimum_target_le_2_episodes=0,
        minimum_target_2_to_5_episodes=8,
        minimum_target_5_to_10_episodes=4,
        minimum_approach_episodes=4,
    ):
        if n_envs != 1:
            raise ValueError(
                "EpisodeStratifiedReplayBuffer currently requires one environment."
            )
        if optimize_memory_usage:
            raise ValueError(
                "EpisodeStratifiedReplayBuffer does not support "
                "optimize_memory_usage=True."
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

        capacities = {
            self.STRATUM_TARGET_LE_2: target_le_2_capacity,
            self.STRATUM_TARGET_2_TO_5: target_2_to_5_capacity,
            self.STRATUM_TARGET_5_TO_10: target_5_to_10_capacity,
            self.STRATUM_APPROACH: approach_capacity,
        }
        fractions = {
            self.STRATUM_TARGET_LE_2: target_le_2_fraction,
            self.STRATUM_TARGET_2_TO_5: target_2_to_5_fraction,
            self.STRATUM_TARGET_5_TO_10: target_5_to_10_fraction,
            self.STRATUM_APPROACH: approach_fraction,
        }
        minimum_episodes = {
            self.STRATUM_TARGET_LE_2: minimum_target_le_2_episodes,
            self.STRATUM_TARGET_2_TO_5: minimum_target_2_to_5_episodes,
            self.STRATUM_TARGET_5_TO_10: minimum_target_5_to_10_episodes,
            self.STRATUM_APPROACH: minimum_approach_episodes,
        }
        if any(int(value) <= 0 for value in capacities.values()):
            raise ValueError("All stratum capacities must be positive.")
        if any(float(value) < 0.0 for value in fractions.values()):
            raise ValueError("Stratum sampling fractions cannot be negative.")
        if not np.isclose(sum(fractions.values()), 1.0):
            raise ValueError("Stratum sampling fractions must sum to one.")
        if any(int(value) < 0 for value in minimum_episodes.values()):
            raise ValueError("Minimum stratum episode counts cannot be negative.")
        for name, value in (
            ("minimum_completed_episodes", minimum_completed_episodes),
            ("minimum_standard_episodes", minimum_standard_episodes),
            ("minimum_near_p3_episodes", minimum_near_p3_episodes),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} cannot be negative.")

        self.capacities = {key: int(value) for key, value in capacities.items()}
        self.sampling_fractions = {
            key: float(value) for key, value in fractions.items()
        }
        self.minimum_stratum_episodes = {
            key: int(value) for key, value in minimum_episodes.items()
        }
        self.minimum_completed_episodes = int(minimum_completed_episodes)
        self.minimum_standard_episodes = int(minimum_standard_episodes)
        self.minimum_near_p3_episodes = int(minimum_near_p3_episodes)

        self._partitions = {
            name: self._allocate_partition(capacity)
            for name, capacity in self.capacities.items()
        }
        self._positions = {name: 0 for name in self.STRATA}
        self._full = {name: False for name in self.STRATA}
        self._episode_indices = {
            name: defaultdict(set) for name in self.STRATA
        }
        self._completed_episode_ids: set[int] = set()
        self._readiness_latched = False
        self._current_episode_id = 0
        self._current_episode_strata: set[str] = set()
        self._current_reset_distribution = "unknown"

        self.completed_episodes = 0
        self.standard_episodes = 0
        self.near_p3_episodes = 0
        self.completed_stratum_episode_counts = Counter()
        self.total_stratum_transitions_seen = Counter()
        self.sampled_stratum_transitions = Counter()
        self.sample_calls = 0
        self.episode_balanced_sample_calls = 0
        self.last_sample_counts = {name: 0 for name in self.STRATA}

    def _allocate_partition(self, capacity):
        return {
            "observations": np.zeros(
                (capacity, *self.obs_shape), dtype=self.observations.dtype
            ),
            "actions": np.zeros(
                (capacity, self.action_dim), dtype=self.actions.dtype
            ),
            "rewards": np.zeros(capacity, dtype=np.float32),
            "next_observations": np.zeros(
                (capacity, *self.obs_shape), dtype=self.next_observations.dtype
            ),
            "dones": np.zeros(capacity, dtype=np.float32),
            "timeouts": np.zeros(capacity, dtype=np.float32),
            "episode_ids": np.full(capacity, -1, dtype=np.int64),
        }

    @staticmethod
    def _reward_components(info):
        components = info.get("reward_components", {})
        return components if isinstance(components, dict) else {}

    def _target_ratio(self, info):
        components = self._reward_components(info)
        ratios = []
        for key in self.TARGET_RATIO_KEYS:
            value = components.get(key)
            if value is None or not np.isfinite(value):
                return np.inf
            ratios.append(abs(float(value)))
        return float(max(ratios))

    def _stratum(self, target_ratio):
        if target_ratio <= 2.0:
            return self.STRATUM_TARGET_LE_2
        if target_ratio <= 5.0:
            return self.STRATUM_TARGET_2_TO_5
        if target_ratio <= 10.0:
            return self.STRATUM_TARGET_5_TO_10
        return self.STRATUM_APPROACH

    def partition_size(self, stratum):
        return self.capacities[stratum] if self._full[stratum] else self._positions[stratum]

    def _remove_overwritten_index(self, stratum, index):
        partition = self._partitions[stratum]
        old_episode_id = int(partition["episode_ids"][index])
        if old_episode_id < 0:
            return
        indices = self._episode_indices[stratum].get(old_episode_id)
        if indices is None:
            return
        indices.discard(index)
        if not indices:
            del self._episode_indices[stratum][old_episode_id]

    def _append(self, stratum, transition):
        partition = self._partitions[stratum]
        index = self._positions[stratum]
        self._remove_overwritten_index(stratum, index)
        obs, next_obs, action, reward, done, timeout = transition
        partition["observations"][index] = obs
        partition["next_observations"][index] = next_obs
        partition["actions"][index] = action
        partition["rewards"][index] = reward
        partition["dones"][index] = done
        partition["timeouts"][index] = timeout
        partition["episode_ids"][index] = self._current_episode_id
        self._episode_indices[stratum][self._current_episode_id].add(index)

        index += 1
        if index >= self.capacities[stratum]:
            self._full[stratum] = True
            index = 0
        self._positions[stratum] = index
        self._current_episode_strata.add(stratum)
        self.total_stratum_transitions_seen[stratum] += 1

    def _completed_episode_ids_for(self, stratum):
        return tuple(
            episode_id
            for episode_id, indices in self._episode_indices[stratum].items()
            if indices and episode_id in self._completed_episode_ids
        )

    def completed_episode_count(self, stratum):
        return len(self._completed_episode_ids_for(stratum))

    @property
    def replay_ready(self):
        if self._readiness_latched:
            return True
        if self.completed_episodes < self.minimum_completed_episodes:
            return False
        if self.standard_episodes < self.minimum_standard_episodes:
            return False
        if self.near_p3_episodes < self.minimum_near_p3_episodes:
            return False
        ready = all(
            self.completed_episode_count(stratum) >= minimum
            for stratum, minimum in self.minimum_stratum_episodes.items()
        )
        if ready:
            self._readiness_latched = True
        return ready

    @property
    def episode_balancing_active(self):
        return bool(self.replay_ready)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        ReplayBuffer.add(self, obs, next_obs, action, reward, done, infos)
        info = infos[0]
        reset_distribution = info.get("reset_distribution", "unknown")
        self._current_reset_distribution = str(reset_distribution)
        transition = (
            np.asarray(obs[0]).copy(),
            np.asarray(next_obs[0]).copy(),
            np.asarray(action[0]).copy(),
            float(reward[0]),
            float(done[0]),
            float(info.get("TimeLimit.truncated", False)),
        )
        stratum = self._stratum(self._target_ratio(info))
        self._append(stratum, transition)

        if bool(done[0]):
            episode_id = self._current_episode_id
            self._completed_episode_ids.add(episode_id)
            self.completed_episodes += 1
            if self._current_reset_distribution == "standard_p2":
                self.standard_episodes += 1
            elif self._current_reset_distribution == "near_p3":
                self.near_p3_episodes += 1
            self.completed_stratum_episode_counts.update(
                self._current_episode_strata
            )
            self._current_episode_id += 1
            self._current_episode_strata.clear()
            self._current_reset_distribution = "unknown"

    @staticmethod
    def _allocate_counts(batch_size, fractions):
        names = tuple(fractions)
        raw = {name: batch_size * fractions[name] for name in names}
        counts = {name: int(np.floor(raw[name])) for name in names}
        remainder = batch_size - sum(counts.values())
        order = sorted(names, key=lambda name: raw[name] - counts[name], reverse=True)
        for name in order[:remainder]:
            counts[name] += 1
        return counts

    def _active_sampling_fractions(self):
        available = {
            stratum: self.sampling_fractions[stratum]
            for stratum in self.STRATA
            if self.completed_episode_count(stratum) > 0
            and self.sampling_fractions[stratum] > 0.0
        }
        if not available:
            return {}
        total = sum(available.values())
        return {name: value / total for name, value in available.items()}

    def _episode_balanced_indices(self, stratum, batch_size):
        episodes = self._completed_episode_ids_for(stratum)
        if not episodes:
            raise RuntimeError(f"No completed episodes in replay stratum {stratum}.")
        selected_episodes = np.random.choice(episodes, size=batch_size, replace=True)
        indices = np.empty(batch_size, dtype=np.int64)
        for position, episode_id in enumerate(selected_episodes):
            episode_indices = tuple(self._episode_indices[stratum][int(episode_id)])
            indices[position] = episode_indices[np.random.randint(len(episode_indices))]
        return indices

    def _get_stratum_samples(self, stratum, batch_size, env=None):
        partition = self._partitions[stratum]
        indices = self._episode_balanced_indices(stratum, batch_size)
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
                partition["rewards"][indices].reshape(-1, 1), env
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
            combined.observations.shape[0], device=combined.observations.device
        )
        return ReplayBufferSamples(
            *(getattr(combined, field)[permutation] for field in fields)
        )

    def sample(self, batch_size, env=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size() <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if not self.episode_balancing_active:
            return ReplayBuffer.sample(self, batch_size, env=env)

        active_fractions = self._active_sampling_fractions()
        counts = self._allocate_counts(batch_size, active_fractions)
        groups = [
            self._get_stratum_samples(stratum, count, env=env)
            for stratum, count in counts.items()
            if count > 0
        ]
        samples = groups[0] if len(groups) == 1 else self._concatenate_and_shuffle(groups)
        self.sample_calls += 1
        self.episode_balanced_sample_calls += 1
        self.last_sample_counts = {name: counts.get(name, 0) for name in self.STRATA}
        for stratum, count in counts.items():
            self.sampled_stratum_transitions[stratum] += count
        return samples

    def replay_statistics(self):
        sampled_total = sum(self.sampled_stratum_transitions.values())
        return {
            "global_buffer_capacity": int(self.buffer_size),
            "global_buffer_size": int(self.size()),
            "episode_balancing_active": bool(self.episode_balancing_active),
            "replay_ready": bool(self.replay_ready),
            "completed_episodes": int(self.completed_episodes),
            "standard_episodes": int(self.standard_episodes),
            "near_p3_episodes": int(self.near_p3_episodes),
            "minimum_completed_episodes": self.minimum_completed_episodes,
            "minimum_standard_episodes": self.minimum_standard_episodes,
            "minimum_near_p3_episodes": self.minimum_near_p3_episodes,
            "stratum_capacities": dict(self.capacities),
            "stratum_sizes": {
                name: int(self.partition_size(name)) for name in self.STRATA
            },
            "minimum_stratum_episodes": dict(self.minimum_stratum_episodes),
            "completed_stratum_episodes": {
                name: int(self.completed_episode_count(name)) for name in self.STRATA
            },
            "historical_completed_stratum_episodes": {
                name: int(self.completed_stratum_episode_counts[name])
                for name in self.STRATA
            },
            "total_stratum_transitions_seen": {
                name: int(self.total_stratum_transitions_seen[name])
                for name in self.STRATA
            },
            "sampling_fraction_targets": dict(self.sampling_fractions),
            "sample_calls": int(self.sample_calls),
            "episode_balanced_sample_calls": int(
                self.episode_balanced_sample_calls
            ),
            "sampled_stratum_transitions": {
                name: int(self.sampled_stratum_transitions[name])
                for name in self.STRATA
            },
            "realized_stratum_sample_fractions": {
                name: float(
                    self.sampled_stratum_transitions[name] / max(sampled_total, 1)
                )
                for name in self.STRATA
            },
            "last_sample_counts": dict(self.last_sample_counts),
        }
