"""Fixed-mixture replay for goal-conditioned closing SAC.

Every transition is stored in the ordinary global DictReplayBuffer.  A
transition is additionally copied into nested near-P3 rings according to
the worst normalized P3 target ratio reported by the environment:

* inner ring: ratio <= 2;
* outer ring: ratio <= 5 (including the inner-ring transitions).

Once both rings contain data, a training batch is sampled as 50% global,
25% outer-ring, and 25% inner-ring experience.  Missing targeted samples fall
back to global replay during early collection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class P3StratifiedDictReplayBuffer(DictReplayBuffer):
    """Dict replay with an auditable global/near-P3 sampling mixture."""

    RATIO_INFO_KEY = "p3_max_target_ratio"

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        outer_buffer_size=100_000,
        inner_buffer_size=100_000,
        global_batch_fraction=0.50,
        outer_batch_fraction=0.25,
        inner_batch_fraction=0.25,
        outer_ratio_limit=5.0,
        inner_ratio_limit=2.0,
    ):
        if n_envs != 1:
            raise ValueError(
                "P3StratifiedDictReplayBuffer currently requires one environment."
            )
        if optimize_memory_usage:
            raise ValueError(
                "P3StratifiedDictReplayBuffer does not support "
                "optimize_memory_usage=True."
            )
        if int(outer_buffer_size) <= 0 or int(inner_buffer_size) <= 0:
            raise ValueError("Near-P3 buffer sizes must be positive.")
        fractions = np.asarray(
            [global_batch_fraction, outer_batch_fraction, inner_batch_fraction],
            dtype=np.float64,
        )
        if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("Replay batch fractions must be non-negative and sum to 1.")
        if not 0.0 < float(inner_ratio_limit) < float(outer_ratio_limit):
            raise ValueError(
                "Ratio limits must satisfy 0 < inner_ratio_limit < outer_ratio_limit."
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
        partition_kwargs = {
            "observation_space": observation_space,
            "action_space": action_space,
            "device": device,
            "n_envs": n_envs,
            "optimize_memory_usage": False,
            "handle_timeout_termination": handle_timeout_termination,
        }
        self.outer_buffer = DictReplayBuffer(
            buffer_size=int(outer_buffer_size), **partition_kwargs
        )
        self.inner_buffer = DictReplayBuffer(
            buffer_size=int(inner_buffer_size), **partition_kwargs
        )
        self.outer_buffer_size = int(outer_buffer_size)
        self.inner_buffer_size = int(inner_buffer_size)
        self.global_batch_fraction = float(global_batch_fraction)
        self.outer_batch_fraction = float(outer_batch_fraction)
        self.inner_batch_fraction = float(inner_batch_fraction)
        self.outer_ratio_limit = float(outer_ratio_limit)
        self.inner_ratio_limit = float(inner_ratio_limit)

        self.total_transitions_seen = 0
        self.total_outer_transitions_seen = 0
        self.total_inner_transitions_seen = 0
        self.sample_calls = 0
        self.fully_stratified_sample_calls = 0
        self.sampled_global_transitions = 0
        self.sampled_outer_transitions = 0
        self.sampled_inner_transitions = 0
        self.fallback_to_global_transitions = 0
        self.last_sample_global_count = 0
        self.last_sample_outer_count = 0
        self.last_sample_inner_count = 0

    @staticmethod
    def _batch_counts(batch_size, fractions):
        """Largest-remainder allocation that always sums to batch_size."""
        exact = np.asarray(fractions, dtype=np.float64) * int(batch_size)
        counts = np.floor(exact).astype(int)
        remainder = int(batch_size) - int(counts.sum())
        if remainder:
            order = np.argsort(-(exact - counts), kind="stable")
            counts[order[:remainder]] += 1
        return tuple(int(value) for value in counts)

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if len(infos) != 1:
            raise ValueError("Exactly one info dictionary is required.")
        ratio = infos[0].get(self.RATIO_INFO_KEY)
        if ratio is None or not np.isfinite(ratio):
            raise ValueError(
                f"Every transition must provide finite '{self.RATIO_INFO_KEY}'."
            )
        ratio = float(ratio)

        super().add(obs, next_obs, action, reward, done, infos)
        self.total_transitions_seen += 1
        if ratio <= self.outer_ratio_limit:
            self.outer_buffer.add(obs, next_obs, action, reward, done, infos)
            self.total_outer_transitions_seen += 1
        if ratio <= self.inner_ratio_limit:
            self.inner_buffer.add(obs, next_obs, action, reward, done, infos)
            self.total_inner_transitions_seen += 1

    @staticmethod
    def _concatenate_and_shuffle(sample_groups):
        observations = {
            key: torch.cat(
                [samples.observations[key] for samples in sample_groups], dim=0
            )
            for key in sample_groups[0].observations
        }
        next_observations = {
            key: torch.cat(
                [samples.next_observations[key] for samples in sample_groups], dim=0
            )
            for key in sample_groups[0].next_observations
        }
        combined = DictReplayBufferSamples(
            observations=observations,
            actions=torch.cat([samples.actions for samples in sample_groups], dim=0),
            next_observations=next_observations,
            dones=torch.cat([samples.dones for samples in sample_groups], dim=0),
            rewards=torch.cat([samples.rewards for samples in sample_groups], dim=0),
        )
        first_tensor = next(iter(combined.observations.values()))
        permutation = torch.randperm(first_tensor.shape[0], device=first_tensor.device)
        return DictReplayBufferSamples(
            observations={
                key: value[permutation]
                for key, value in combined.observations.items()
            },
            actions=combined.actions[permutation],
            next_observations={
                key: value[permutation]
                for key, value in combined.next_observations.items()
            },
            dones=combined.dones[permutation],
            rewards=combined.rewards[permutation],
        )

    def sample(self, batch_size, env=None):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size() <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        global_count, outer_count, inner_count = self._batch_counts(
            int(batch_size),
            (
                self.global_batch_fraction,
                self.outer_batch_fraction,
                self.inner_batch_fraction,
            ),
        )
        requested_targeted = outer_count + inner_count
        if self.outer_buffer.size() == 0:
            global_count += outer_count
            outer_count = 0
        if self.inner_buffer.size() == 0:
            global_count += inner_count
            inner_count = 0

        groups = []
        if global_count:
            groups.append(super().sample(global_count, env=env))
        if outer_count:
            groups.append(self.outer_buffer.sample(outer_count, env=env))
        if inner_count:
            groups.append(self.inner_buffer.sample(inner_count, env=env))
        samples = (
            groups[0]
            if len(groups) == 1
            else self._concatenate_and_shuffle(groups)
        )

        self.sample_calls += 1
        if outer_count and inner_count:
            self.fully_stratified_sample_calls += 1
        self.sampled_global_transitions += global_count
        self.sampled_outer_transitions += outer_count
        self.sampled_inner_transitions += inner_count
        self.fallback_to_global_transitions += (
            requested_targeted - outer_count - inner_count
        )
        self.last_sample_global_count = global_count
        self.last_sample_outer_count = outer_count
        self.last_sample_inner_count = inner_count
        return samples

    def replay_statistics(self):
        sampled_total = (
            self.sampled_global_transitions
            + self.sampled_outer_transitions
            + self.sampled_inner_transitions
        )
        return {
            "global_buffer_capacity": int(self.buffer_size),
            "global_buffer_size": int(self.size()),
            "outer_buffer_capacity": self.outer_buffer_size,
            "outer_buffer_size": int(self.outer_buffer.size()),
            "inner_buffer_capacity": self.inner_buffer_size,
            "inner_buffer_size": int(self.inner_buffer.size()),
            "outer_ratio_band": f"ratio <= {self.outer_ratio_limit:g}",
            "inner_ratio_band": f"ratio <= {self.inner_ratio_limit:g}",
            "target_partitions_are_nested": True,
            "global_batch_fraction_target": self.global_batch_fraction,
            "outer_batch_fraction_target": self.outer_batch_fraction,
            "inner_batch_fraction_target": self.inner_batch_fraction,
            "total_transitions_seen": int(self.total_transitions_seen),
            "total_outer_transitions_seen": int(
                self.total_outer_transitions_seen
            ),
            "total_inner_transitions_seen": int(
                self.total_inner_transitions_seen
            ),
            "sample_calls": int(self.sample_calls),
            "fully_stratified_sample_calls": int(
                self.fully_stratified_sample_calls
            ),
            "sampled_global_transitions": int(self.sampled_global_transitions),
            "sampled_outer_transitions": int(self.sampled_outer_transitions),
            "sampled_inner_transitions": int(self.sampled_inner_transitions),
            "fallback_to_global_transitions": int(
                self.fallback_to_global_transitions
            ),
            "realized_global_sample_fraction": float(
                self.sampled_global_transitions / max(sampled_total, 1)
            ),
            "realized_outer_sample_fraction": float(
                self.sampled_outer_transitions / max(sampled_total, 1)
            ),
            "realized_inner_sample_fraction": float(
                self.sampled_inner_transitions / max(sampled_total, 1)
            ),
            "last_sample_global_count": int(self.last_sample_global_count),
            "last_sample_outer_count": int(self.last_sample_outer_count),
            "last_sample_inner_count": int(self.last_sample_inner_count),
        }
