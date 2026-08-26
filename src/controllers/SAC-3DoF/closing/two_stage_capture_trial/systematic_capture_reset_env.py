"""Direct full-distance reset distribution for systematic capture training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from systematic_interface import DEFAULT_CONTRACT, state_error


class SystematicCaptureResetWrapper(gym.Wrapper):
    """Reset directly in the explicit six-state ``X_H`` handover set.

    This removes the circular dependency on states produced by an inadequate
    long-range policy.  Once capture is validated, its demonstrated region of
    attraction can be used to finalize the long-range handover contract.
    """

    failed_approach_penalty = 500.0
    missed_handover_penalty = 500.0

    def __init__(
        self,
        env,
        capture_distance_m=150.0,
        maximum_sampling_attempts=512,
        flyaway_margin_m=50.0,
        flyaway_radial_rate_mps=0.01,
        flyaway_consecutive_steps=60,
        enable_lateral_divergence_termination=False,
        maximum_lateral_error_m=50.0,
        maximum_lateral_velocity_error_mps=0.15,
    ):
        super().__init__(env)
        self.maximum_sampling_attempts = int(maximum_sampling_attempts)
        self.flyaway_margin_m = float(flyaway_margin_m)
        self.flyaway_radial_rate_mps = float(flyaway_radial_rate_mps)
        self.flyaway_consecutive_steps = int(flyaway_consecutive_steps)
        self.enable_lateral_divergence_termination = bool(
            enable_lateral_divergence_termination
        )
        self.maximum_lateral_error_m = float(maximum_lateral_error_m)
        self.maximum_lateral_velocity_error_mps = float(
            maximum_lateral_velocity_error_mps
        )
        if self.maximum_sampling_attempts <= 0:
            raise ValueError("maximum_sampling_attempts must be positive.")
        if self.flyaway_margin_m <= 0.0:
            raise ValueError("flyaway_margin_m must be positive.")
        if self.flyaway_consecutive_steps <= 0:
            raise ValueError("flyaway_consecutive_steps must be positive.")
        if self.maximum_lateral_error_m <= 0.0:
            raise ValueError("maximum_lateral_error_m must be positive.")
        if self.maximum_lateral_velocity_error_mps <= 0.0:
            raise ValueError(
                "maximum_lateral_velocity_error_mps must be positive."
            )
        self.set_capture_distance(capture_distance_m)
        self.p3_target = None
        self.handover_target = None
        self.best_position_error_m = np.inf
        self.flyaway_counter = 0

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def set_capture_distance(self, distance_m):
        distance = float(distance_m)
        if not 0.0 < distance <= DEFAULT_CONTRACT.handover_distance_m:
            raise ValueError(
                "capture distance must lie in "
                f"(0, {DEFAULT_CONTRACT.handover_distance_m:g}] m."
            )
        self.capture_distance_m = distance

    def _position_error(self, state):
        return float(np.linalg.norm(state_error(state, self.p3_target)[0:2]))

    def _reset_diagnostics(self, info):
        info = dict(info)
        info.update(
            {
                "capture_reset_source": "systematic_X_H",
                "capture_distance_m": self.capture_distance_m,
                "p3_target": self.p3_target.copy(),
                "handover_target": self.handover_target.copy(),
                "handover_tolerance": DEFAULT_CONTRACT.handover_tolerance.copy(),
                "initial_in_handover_set": DEFAULT_CONTRACT.in_handover_set(
                    self.core_env.state,
                    self.p3_target,
                    distance_m=self.capture_distance_m,
                ),
                "lateral_divergence_termination_enabled": (
                    self.enable_lateral_divergence_termination
                ),
            }
        )
        return info

    def reset(self, seed=None, options=None):
        # Explicit states remain available for deterministic evaluation.
        if options and options.get("initial_state") is not None:
            observation, info = self.env.reset(seed=seed, options=options)
            self.p3_target = np.asarray(options["p3_target"], dtype=np.float64)
            distance = options.get("capture_distance_m", self.capture_distance_m)
            self.handover_target = DEFAULT_CONTRACT.intermediate_target(
                self.p3_target, distance_m=distance
            )
            self.best_position_error_m = self._position_error(self.core_env.state)
            self.flyaway_counter = 0
            return observation, self._reset_diagnostics(info)

        # Seed the core RNG through the ordinary Gym reset, then replace its
        # sampled P2 state with a controlled X_H/P3 pair.
        self.env.reset(seed=seed, options={"mode": "training"})
        rng = self.core_env.rng
        for _ in range(self.maximum_sampling_attempts):
            p3_target = DEFAULT_CONTRACT.sample_p3(rng)
            state, handover_target = DEFAULT_CONTRACT.sample_handover_state(
                rng, p3_target, distance_m=self.capture_distance_m
            )
            if DEFAULT_CONTRACT.in_validated_p3(state):
                continue
            if self.core_env.monitor_safety(state)["unsafe"]:
                continue
            observation, info = self.env.reset(
                options={
                    "initial_state": state,
                    "p3_target": p3_target,
                    "mode": "training",
                }
            )
            if info.get("unsafe", False):
                continue
            self.p3_target = p3_target
            self.handover_target = handover_target
            self.best_position_error_m = self._position_error(state)
            self.flyaway_counter = 0
            return observation, self._reset_diagnostics(info)
        raise RuntimeError("Could not sample a safe systematic X_H capture reset.")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        current_error = self._position_error(self.core_env.state)
        self.best_position_error_m = min(self.best_position_error_m, current_error)
        error = state_error(self.core_env.state, self.p3_target)
        if current_error > 1e-12:
            radial_rate = float(np.dot(error[0:2], error[2:4]) / current_error)
        else:
            radial_rate = 0.0

        # Crossing inward through the near face of the validated P3 box is a
        # failed handover unless the base environment already declared valid
        # P3 membership. Existing hard-safety failures retain their more
        # informative geometry/velocity/alignment termination reason.
        missed_p3 = (
            float(self.core_env.state[0]) < float(DEFAULT_CONTRACT.p3_lower[0])
            and not info.get("success", False)
        )
        if not terminated and not truncated and missed_p3:
            terminated = True
            reward = float(reward) - self.missed_handover_penalty
            info.update(
                {
                    "success": False,
                    "failure": True,
                    "failure_reason": "failed_capture_missed_p3",
                    "termination_reason": "failed_capture_missed_p3",
                }
            )
            components = dict(info.get("reward_components", {}))
            components["reward_missed_handover"] = -self.missed_handover_penalty
            components["reward_total"] = float(reward)
            info["reward_components"] = components

        lateral_error = abs(float(error[1]))
        lateral_velocity_error = abs(float(error[3]))
        lateral_divergence = self.enable_lateral_divergence_termination and (
            lateral_error > self.maximum_lateral_error_m
            or lateral_velocity_error > self.maximum_lateral_velocity_error_mps
        )
        if not terminated and not truncated and lateral_divergence:
            terminated = True
            reward = float(reward) - self.failed_approach_penalty
            reasons = []
            if lateral_error > self.maximum_lateral_error_m:
                reasons.append("lateral_position")
            if lateral_velocity_error > self.maximum_lateral_velocity_error_mps:
                reasons.append("lateral_velocity")
            info.update(
                {
                    "success": False,
                    "failure": True,
                    "failure_reason": "failed_capture_lateral_divergence",
                    "termination_reason": "failed_capture_lateral_divergence",
                    "lateral_divergence_reasons": reasons,
                }
            )
            components = dict(info.get("reward_components", {}))
            components["reward_lateral_divergence"] = (
                -self.failed_approach_penalty
            )
            components["reward_total"] = float(reward)
            info["reward_components"] = components

        moving_away = (
            current_error > self.best_position_error_m + self.flyaway_margin_m
            and radial_rate > self.flyaway_radial_rate_mps
        )
        self.flyaway_counter = self.flyaway_counter + 1 if moving_away else 0
        if (
            not terminated
            and not truncated
            and self.flyaway_counter >= self.flyaway_consecutive_steps
        ):
            terminated = True
            reward = float(reward) - self.failed_approach_penalty
            info.update(
                {
                    "success": False,
                    "failure": True,
                    "failure_reason": "failed_capture_flyaway",
                    "termination_reason": "failed_capture_flyaway",
                }
            )
            components = dict(info.get("reward_components", {}))
            components["reward_failed_capture"] = -self.failed_approach_penalty
            components["reward_total"] = float(reward)
            info["reward_components"] = components

        info.update(
            {
                "capture_reset_source": "systematic_X_H",
                "capture_distance_m": self.capture_distance_m,
                "position_error_m": current_error,
                "best_position_error_m": self.best_position_error_m,
                "target_radial_rate_mps": radial_rate,
                "flyaway_counter": self.flyaway_counter,
                "lateral_error_m": lateral_error,
                "lateral_velocity_error_mps": lateral_velocity_error,
                "lateral_divergence_termination_enabled": (
                    self.enable_lateral_divergence_termination
                ),
            }
        )
        if terminated or truncated:
            # DummyVecEnv auto-resets immediately after this return. Preserve
            # the actual physical terminal state and targets for evaluators.
            info["terminal_physical_state"] = self.core_env.state.copy()
            info["terminal_p3_target"] = self.p3_target.copy()
            info["terminal_handover_target"] = self.handover_target.copy()
        return observation, float(reward), terminated, truncated, info
