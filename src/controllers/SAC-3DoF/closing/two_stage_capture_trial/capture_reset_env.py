"""Empirical-plus-synthetic reset wrapper for the capture-only SAC task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from common import (
    CaptureEnvelope,
    handover_metrics,
    is_capture_envelope_entry,
    load_handover_dataset,
)


class CaptureResetWrapper(gym.Wrapper):
    """Reset closing episodes near P3 without changing the base environment.

    Empirical resets reproduce the first-entry state and exact P3 target saved
    by the long-range collector.  Synthetic resets use separate component
    bounds and rejection sampling, avoiding an unrealistic uniform ``8x`` box.
    """

    RESET_EMPIRICAL = "empirical_handover"
    RESET_SYNTHETIC = "synthetic_component_bounds"

    def __init__(
        self,
        env,
        dataset_path,
        radius_m,
        empirical_fraction=0.80,
        lateral_error_m=15.0,
        velocity_error_mps=0.03,
        attitude_error_deg=3.0,
        rate_error_degs=0.10,
        maximum_radial_rate_mps=0.0,
        minimum_max_target_ratio=1.05,
        maximum_sampling_attempts=512,
    ):
        super().__init__(env)
        self.dataset = load_handover_dataset(dataset_path, radius_m=radius_m)
        self.envelope = CaptureEnvelope(
            radius_m=float(radius_m),
            lateral_error_m=float(lateral_error_m),
            velocity_error_mps=float(velocity_error_mps),
            attitude_error_deg=float(attitude_error_deg),
            rate_error_degs=float(rate_error_degs),
            maximum_radial_rate_mps=float(maximum_radial_rate_mps),
        )
        self.envelope.validate()
        self.empirical_fraction = float(empirical_fraction)
        self.minimum_max_target_ratio = float(minimum_max_target_ratio)
        self.maximum_sampling_attempts = int(maximum_sampling_attempts)
        if not 0.0 <= self.empirical_fraction <= 1.0:
            raise ValueError("empirical_fraction must lie in [0, 1].")
        if self.minimum_max_target_ratio <= 1.0:
            raise ValueError("minimum_max_target_ratio must exceed one.")
        if self.maximum_sampling_attempts <= 0:
            raise ValueError("maximum_sampling_attempts must be positive.")

        eligible = np.flatnonzero(
            self.dataset["max_target_ratios"] >= self.minimum_max_target_ratio
        )
        if self.empirical_fraction > 0.0 and eligible.size == 0:
            raise ValueError(
                "No empirical handovers remain after excluding immediate P3 states."
            )
        self.eligible_empirical_indices = eligible
        self.current_reset_source = None
        self.current_dataset_index = None

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def _reset_to_state(self, state, target, mode):
        return self.env.reset(
            options={
                "initial_state": np.asarray(state, dtype=np.float64),
                "p3_target": np.asarray(target, dtype=np.float64),
                "mode": mode,
            }
        )

    def _empirical_reset(self, rng, mode):
        for _ in range(self.maximum_sampling_attempts):
            index = int(rng.choice(self.eligible_empirical_indices))
            observation, info = self._reset_to_state(
                self.dataset["states"][index],
                self.dataset["p3_targets"][index],
                mode,
            )
            if not bool(info.get("unsafe", False)):
                self.current_dataset_index = index
                return observation, info
        raise RuntimeError("Could not select a non-unsafe empirical handover reset.")

    def _sample_synthetic_error(self, rng):
        radius = self.envelope.radius_m
        attitude = np.deg2rad(self.envelope.attitude_error_deg)
        rate = np.deg2rad(self.envelope.rate_error_degs)
        return np.array(
            [
                rng.uniform(-radius, radius),
                rng.uniform(-self.envelope.lateral_error_m, self.envelope.lateral_error_m),
                rng.uniform(-self.envelope.velocity_error_mps, self.envelope.velocity_error_mps),
                rng.uniform(-self.envelope.velocity_error_mps, self.envelope.velocity_error_mps),
                rng.uniform(-attitude, attitude),
                rng.uniform(-rate, rate),
            ],
            dtype=np.float64,
        )

    def _synthetic_reset(self, rng, target, mode):
        tolerance = self.core_env.p3_target_tolerance
        for _ in range(self.maximum_sampling_attempts):
            error = self._sample_synthetic_error(rng)
            state = np.asarray(target, dtype=np.float64) + error
            state[4] = self.core_env.wrap_to_pi(state[4])
            metrics = handover_metrics(state, target, tolerance)
            if metrics["max_target_ratio"] < self.minimum_max_target_ratio:
                continue
            preliminary_safety = self.core_env.monitor_safety(state)
            if not is_capture_envelope_entry(
                metrics, self.envelope, preliminary_safety
            ):
                continue
            observation, info = self._reset_to_state(state, target, mode)
            if bool(info.get("unsafe", False)):
                continue
            return observation, info
        raise RuntimeError(
            "Could not sample a valid synthetic capture reset. Relax the "
            "component bounds or increase maximum_sampling_attempts."
        )

    def reset(self, seed=None, options=None):
        # Explicit resets are used by evaluation and always take precedence.
        if options and options.get("initial_state") is not None:
            observation, info = self.env.reset(seed=seed, options=options)
            info = dict(info)
            info["capture_reset_source"] = "explicit"
            info["capture_dataset_index"] = None
            return observation, info

        observation, info = self.env.reset(seed=seed, options=options)
        mode = (options or {}).get("mode", self.env.unwrapped.mode)
        rng = self.core_env.rng
        target = self.core_env.p3_target.copy()
        self.current_dataset_index = None

        use_empirical = bool(
            self.empirical_fraction > 0.0
            and rng.random() < self.empirical_fraction
        )
        if use_empirical:
            observation, info = self._empirical_reset(rng, mode)
            self.current_reset_source = self.RESET_EMPIRICAL
        else:
            observation, info = self._synthetic_reset(rng, target, mode)
            self.current_reset_source = self.RESET_SYNTHETIC

        info = dict(info)
        info["capture_reset_source"] = self.current_reset_source
        info["capture_dataset_index"] = self.current_dataset_index
        info["capture_radius_m"] = self.envelope.radius_m
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["capture_reset_source"] = self.current_reset_source
        info["capture_dataset_index"] = self.current_dataset_index
        info["capture_radius_m"] = self.envelope.radius_m
        return observation, reward, terminated, truncated, info
