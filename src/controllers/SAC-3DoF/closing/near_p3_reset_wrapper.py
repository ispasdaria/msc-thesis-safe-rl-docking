"""Fixed near-P3 reset augmentation for closing-policy training.

The wrapper keeps the ordinary P2 reset distribution and replaces a fixed
fraction of episode resets with safe states just outside the six-dimensional
P3 terminal set.  The fraction is not scheduled as a curriculum.  Evaluation
environments do not use this wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FixedNearP3ResetWrapper(gym.Wrapper):
    """Mix normal P2 resets with safe states in a normalized P3-error shell."""

    RESET_STANDARD = "standard_p2"
    RESET_NEAR_P3 = "near_p3"

    def __init__(
        self,
        env,
        near_p3_fraction=0.20,
        minimum_target_ratio=1.20,
        maximum_target_ratio=4.00,
        other_component_max_ratio=None,
        deterministic_mixture=False,
        maximum_sampling_attempts=128,
    ):
        super().__init__(env)
        self.minimum_target_ratio = float(minimum_target_ratio)
        self.maximum_target_ratio = float(maximum_target_ratio)
        self.other_component_max_ratio = (
            None
            if other_component_max_ratio is None
            else float(other_component_max_ratio)
        )
        self.deterministic_mixture = bool(deterministic_mixture)
        self.maximum_sampling_attempts = int(maximum_sampling_attempts)
        self._mixture_accumulator = 0.0
        self._near_p3_fraction = 0.0
        self.set_near_p3_fraction(near_p3_fraction)

        if self.minimum_target_ratio <= 1.0:
            raise ValueError("minimum_target_ratio must be greater than one.")
        if self.maximum_target_ratio <= self.minimum_target_ratio:
            raise ValueError(
                "maximum_target_ratio must exceed minimum_target_ratio."
            )
        if (
            self.other_component_max_ratio is not None
            and not 0.0 <= self.other_component_max_ratio < 1.0
        ):
            raise ValueError(
                "other_component_max_ratio must lie in [0, 1)."
            )
        if self.maximum_sampling_attempts <= 0:
            raise ValueError("maximum_sampling_attempts must be positive.")

        self.total_resets = 0
        self.standard_resets = 0
        self.near_p3_resets = 0
        self.near_p3_sampling_failures = 0
        self.current_reset_distribution = self.RESET_STANDARD
        self.current_initial_max_target_ratio = None

    @property
    def near_p3_fraction(self):
        return self._near_p3_fraction

    def set_near_p3_fraction(self, fraction, reset_schedule=True):
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("near_p3_fraction must lie in [0, 1].")
        self._near_p3_fraction = fraction
        if reset_schedule:
            self._mixture_accumulator = 0.0

    def _select_near_p3_reset(self, rng):
        if self.near_p3_fraction <= 0.0:
            return False
        if not self.deterministic_mixture:
            return bool(rng.random() < self.near_p3_fraction)

        # A fractional accumulator gives an exact long-run mixture without
        # clustering.  For 0.2, every fifth eligible reset is near-P3.
        self._mixture_accumulator += self.near_p3_fraction
        if self._mixture_accumulator + 1e-12 < 1.0:
            return False
        self._mixture_accumulator -= 1.0
        return True

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def _sample_near_p3_state(self, target):
        core = self.core_env
        tolerance = np.asarray(core.p3_target_tolerance, dtype=np.float64)
        rng = core.rng

        for _ in range(self.maximum_sampling_attempts):
            shell_ratio = float(
                rng.uniform(
                    self.minimum_target_ratio,
                    self.maximum_target_ratio,
                )
            )
            if self.other_component_max_ratio is None:
                ratios = rng.uniform(-shell_ratio, shell_ratio, size=6)
            else:
                ratios = rng.uniform(
                    -self.other_component_max_ratio,
                    self.other_component_max_ratio,
                    size=6,
                )
            dominant_index = int(rng.integers(0, 6))
            dominant_sign = -1.0 if rng.random() < 0.5 else 1.0
            ratios[dominant_index] = dominant_sign * shell_ratio

            state = np.asarray(target, dtype=np.float64) + ratios * tolerance
            state[4] = core.wrap_to_pi(state[4])
            monitor = core.monitor_safety(state)
            if bool(monitor.get("unsafe", False)):
                continue
            if core.is_at_P3_target(state):
                continue
            return state, float(np.max(np.abs(ratios)))

        return None, None

    def reset(self, seed=None, options=None):
        # First perform the ordinary reset.  This initializes Gymnasium's RNG
        # and samples the exact P3 target used to construct an augmented state.
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_resets += 1

        explicit_state = bool(options and options.get("initial_state") is not None)
        rng = self.core_env.rng
        use_near_p3 = bool(
            not explicit_state and self._select_near_p3_reset(rng)
        )

        self.current_reset_distribution = self.RESET_STANDARD
        self.current_initial_max_target_ratio = None
        if use_near_p3:
            target = np.asarray(self.core_env.p3_target, dtype=np.float64).copy()
            state, max_ratio = self._sample_near_p3_state(target)
            if state is not None:
                reset_options = dict(options or {})
                reset_options.update(
                    {
                        "initial_state": state,
                        "p3_target": target,
                        "mode": reset_options.get("mode", self.env.unwrapped.mode),
                    }
                )
                obs, info = self.env.reset(options=reset_options)
                self.current_reset_distribution = self.RESET_NEAR_P3
                self.current_initial_max_target_ratio = max_ratio
                self.near_p3_resets += 1
            else:
                self.near_p3_sampling_failures += 1

        if self.current_reset_distribution == self.RESET_STANDARD:
            self.standard_resets += 1

        info = dict(info)
        info["reset_distribution"] = self.current_reset_distribution
        info["initial_max_target_ratio"] = self.current_initial_max_target_ratio
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["reset_distribution"] = self.current_reset_distribution
        info["initial_max_target_ratio"] = self.current_initial_max_target_ratio
        return obs, reward, terminated, truncated, info

    def reset_statistics(self):
        return {
            "configured_near_p3_fraction": self.near_p3_fraction,
            "minimum_target_ratio": self.minimum_target_ratio,
            "maximum_target_ratio": self.maximum_target_ratio,
            "other_component_max_ratio": self.other_component_max_ratio,
            "deterministic_mixture": self.deterministic_mixture,
            "total_resets": int(self.total_resets),
            "standard_resets": int(self.standard_resets),
            "near_p3_resets": int(self.near_p3_resets),
            "near_p3_sampling_failures": int(self.near_p3_sampling_failures),
            "realized_near_p3_fraction": float(
                self.near_p3_resets / max(self.total_resets, 1)
            ),
        }
