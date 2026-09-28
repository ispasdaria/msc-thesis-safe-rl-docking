"""Protected demonstration/online replay mixing for Stable-Baselines3 SAC."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ProtectedMixedReplayBuffer(ReplayBuffer):
    """Keep demonstrations immutable and mix them into every sampled batch.

    The inherited replay arrays contain online transitions only. Demonstration
    arrays are held separately, never overwritten, and sampled uniformly over
    all stored transitions. The demonstration fraction changes after a chosen
    number of online transitions have been added.
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        initial_demo_fraction=0.50,
        later_demo_fraction=0.25,
        demo_fraction_switch_step=300_000,
    ):
        if n_envs != 1:
            raise ValueError(
                "ProtectedMixedReplayBuffer currently requires one environment."
            )
        if optimize_memory_usage:
            raise ValueError(
                "ProtectedMixedReplayBuffer does not support "
                "optimize_memory_usage=True."
            )
        for name, value in (
            ("initial_demo_fraction", initial_demo_fraction),
            ("later_demo_fraction", later_demo_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if int(demo_fraction_switch_step) < 0:
            raise ValueError("demo_fraction_switch_step must be non-negative.")

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.initial_demo_fraction = float(initial_demo_fraction)
        self.later_demo_fraction = float(later_demo_fraction)
        self.demo_fraction_switch_step = int(demo_fraction_switch_step)
        self.total_online_transitions = 0
        self.demonstration_count = 0
        self.demonstration_observations = None
        self.demonstration_actions = None
        self.demonstration_rewards = None
        self.demonstration_next_observations = None
        self.demonstration_dones = None
        self.demonstration_timeouts = None
        self.behavior_cloning_count = 0
        self.behavior_cloning_observations = None
        self.behavior_cloning_actions = None
        self.sample_calls = 0
        self.sampled_demo_transitions = 0
        self.sampled_online_transitions = 0
        self.last_sample_demo_count = 0
        self.last_sample_online_count = 0

    @staticmethod
    def _readonly_copy(values, dtype):
        copied = np.asarray(values, dtype=dtype).copy()
        copied.setflags(write=False)
        return copied

    def set_demonstrations(self, dataset):
        """Attach a validated SAC-format dataset as immutable demonstrations."""

        observations = self._readonly_copy(
            dataset["observations"],
            np.float32,
        )
        actions = self._readonly_copy(dataset["actions"], np.float32)
        rewards = self._readonly_copy(dataset["rewards"], np.float32)
        next_observations = self._readonly_copy(
            dataset["next_observations"],
            np.float32,
        )
        terminated = np.asarray(dataset["terminated"], dtype=np.bool_)
        truncated = np.asarray(dataset["truncated"], dtype=np.bool_)

        count = len(rewards)
        if count <= 0:
            raise ValueError("The demonstration dataset is empty.")
        if observations.shape != (count, *self.obs_shape):
            raise ValueError(
                "Demonstration observation shape does not match the replay "
                f"space: {observations.shape} != {(count, *self.obs_shape)}."
            )
        if next_observations.shape != observations.shape:
            raise ValueError("Demonstration next-observation shape is invalid.")
        if actions.shape != (count, self.action_dim):
            raise ValueError(
                "Demonstration action shape does not match the replay space: "
                f"{actions.shape} != {(count, self.action_dim)}."
            )
        if terminated.shape != (count,) or truncated.shape != (count,):
            raise ValueError("Demonstration terminal arrays must have shape (N,).")

        dones = self._readonly_copy(
            np.logical_or(terminated, truncated).astype(np.float32),
            np.float32,
        )
        timeouts = self._readonly_copy(
            np.logical_and(truncated, np.logical_not(terminated)).astype(
                np.float32
            ),
            np.float32,
        )
        self.demonstration_observations = observations
        self.demonstration_actions = actions
        self.demonstration_rewards = rewards
        self.demonstration_next_observations = next_observations
        self.demonstration_dones = dones
        self.demonstration_timeouts = timeouts
        self.demonstration_count = count

    def set_behavior_cloning_demonstrations(self, dataset):
        """Attach a potentially thinned state-action pool for actor cloning."""

        observations = self._readonly_copy(
            dataset["observations"],
            np.float32,
        )
        actions = self._readonly_copy(dataset["actions"], np.float32)
        count = len(actions)
        if count <= 0:
            raise ValueError("The behavior-cloning dataset is empty.")
        if observations.shape != (count, *self.obs_shape):
            raise ValueError(
                "Behavior-cloning observation shape does not match the replay "
                f"space: {observations.shape} != {(count, *self.obs_shape)}."
            )
        if actions.shape != (count, self.action_dim):
            raise ValueError(
                "Behavior-cloning action shape does not match the replay "
                f"space: {actions.shape} != {(count, self.action_dim)}."
            )
        self.behavior_cloning_observations = observations
        self.behavior_cloning_actions = actions
        self.behavior_cloning_count = count

    @property
    def active_demo_fraction(self):
        if self.total_online_transitions < self.demo_fraction_switch_step:
            return self.initial_demo_fraction
        return self.later_demo_fraction

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        self.total_online_transitions += self.n_envs

    def _get_demonstration_samples(self, batch_size, env=None):
        if self.demonstration_count <= 0:
            raise RuntimeError("No demonstrations have been attached.")
        indices = np.random.randint(
            0,
            self.demonstration_count,
            size=batch_size,
        )
        dones = (
            self.demonstration_dones[indices]
            * (1.0 - self.demonstration_timeouts[indices])
        ).reshape(-1, 1)
        data = (
            self._normalize_obs(
                self.demonstration_observations[indices],
                env,
            ),
            self.demonstration_actions[indices],
            self._normalize_obs(
                self.demonstration_next_observations[indices],
                env,
            ),
            dones,
            self._normalize_reward(
                self.demonstration_rewards[indices].reshape(-1, 1),
                env,
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_behavior_cloning_samples(self, batch_size, env=None):
        """Sample actor targets from the dedicated BC pool when configured."""

        if self.behavior_cloning_count <= 0:
            samples = self._get_demonstration_samples(batch_size, env=env)
            return samples.observations, samples.actions
        indices = np.random.randint(
            0,
            self.behavior_cloning_count,
            size=batch_size,
        )
        observations = self._normalize_obs(
            self.behavior_cloning_observations[indices],
            env,
        )
        actions = self.behavior_cloning_actions[indices]
        return self.to_torch(observations), self.to_torch(actions)

    @staticmethod
    def _concatenate_and_shuffle(demo_samples, online_samples):
        tensor_fields = (
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
        )
        combined = ReplayBufferSamples(
            *(
                torch.cat(
                    (
                        getattr(demo_samples, field),
                        getattr(online_samples, field),
                    ),
                    dim=0,
                )
                for field in tensor_fields
            )
        )
        permutation = torch.randperm(
            combined.observations.shape[0],
            device=combined.observations.device,
        )
        return ReplayBufferSamples(
            *(getattr(combined, field)[permutation] for field in tensor_fields)
        )

    def sample(self, batch_size, env=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        online_available = self.size() > 0
        demos_available = self.demonstration_count > 0
        if not online_available and not demos_available:
            raise RuntimeError("Cannot sample from an empty mixed replay buffer.")

        if not demos_available:
            demo_count = 0
        elif not online_available:
            demo_count = batch_size
        else:
            demo_count = int(round(batch_size * self.active_demo_fraction))
            demo_count = int(np.clip(demo_count, 0, batch_size))
        online_count = batch_size - demo_count

        if demo_count == batch_size:
            samples = self._get_demonstration_samples(batch_size, env=env)
        elif online_count == batch_size:
            samples = super().sample(batch_size, env=env)
        else:
            demo_samples = self._get_demonstration_samples(
                demo_count,
                env=env,
            )
            online_samples = super().sample(online_count, env=env)
            samples = self._concatenate_and_shuffle(
                demo_samples,
                online_samples,
            )

        self.sample_calls += 1
        self.sampled_demo_transitions += demo_count
        self.sampled_online_transitions += online_count
        self.last_sample_demo_count = demo_count
        self.last_sample_online_count = online_count
        return samples

    def mixing_statistics(self):
        total_sampled = (
            self.sampled_demo_transitions + self.sampled_online_transitions
        )
        realized_fraction = (
            self.sampled_demo_transitions / total_sampled
            if total_sampled
            else 0.0
        )
        return {
            "demonstration_count": int(self.demonstration_count),
            "behavior_cloning_count": int(self.behavior_cloning_count),
            "online_buffer_capacity": int(self.buffer_size),
            "online_buffer_size": int(self.size()),
            "total_online_transitions": int(self.total_online_transitions),
            "initial_demo_fraction": self.initial_demo_fraction,
            "later_demo_fraction": self.later_demo_fraction,
            "demo_fraction_switch_step": self.demo_fraction_switch_step,
            "active_demo_fraction": self.active_demo_fraction,
            "sample_calls": int(self.sample_calls),
            "sampled_demo_transitions": int(
                self.sampled_demo_transitions
            ),
            "sampled_online_transitions": int(
                self.sampled_online_transitions
            ),
            "realized_demo_fraction": float(realized_fraction),
        }
