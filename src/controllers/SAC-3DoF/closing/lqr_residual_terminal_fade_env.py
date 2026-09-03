"""Terminal-fade reward/control variant for direct residual SAC."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lqr_residual_lateral_env import (
    LateralProgressResidualLQRGoalConditionedClosingEnv,
)


class TerminalFadeLateralResidualLQRGoalConditionedClosingEnv(
    LateralProgressResidualLQRGoalConditionedClosingEnv
):
    """Keep residual authority until the state enters the exact P3 set."""

    reward_scheme = (
        "goal_conditioned_p3_direct_residual_lqr_dt01_terminal_fade_v2"
    )

    def __init__(
        self,
        *,
        terminal_fade_full_ratio=1.2,
        terminal_fade_zero_ratio=1.0,
        residual_magnitude_weight=0.002,
        residual_smoothness_weight=0.001,
        time_consistent_smoothness_reward=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.terminal_fade_full_ratio = float(terminal_fade_full_ratio)
        self.terminal_fade_zero_ratio = float(terminal_fade_zero_ratio)
        if (
            not np.isfinite(self.terminal_fade_zero_ratio)
            or self.terminal_fade_zero_ratio < 0.0
        ):
            raise ValueError(
                "terminal_fade_zero_ratio must be finite and non-negative."
            )
        if (
            not np.isfinite(self.terminal_fade_full_ratio)
            or self.terminal_fade_full_ratio <= self.terminal_fade_zero_ratio
        ):
            raise ValueError(
                "terminal_fade_full_ratio must exceed terminal_fade_zero_ratio."
            )
        for name, value in {
            "residual_magnitude_weight": residual_magnitude_weight,
            "residual_smoothness_weight": residual_smoothness_weight,
        }.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        self.residual_magnitude_weight = float(residual_magnitude_weight)
        self.residual_smoothness_weight = float(residual_smoothness_weight)
        self.time_consistent_smoothness_reward = bool(
            time_consistent_smoothness_reward
        )
        self.previous_effective_residual_action = np.zeros(3, dtype=np.float32)
        self._online_effective_residual_action = None
        self._online_previous_effective_residual_action = None

    def reset(self, seed=None, options=None):
        self.previous_effective_residual_action = np.zeros(3, dtype=np.float32)
        self._online_effective_residual_action = None
        self._online_previous_effective_residual_action = None
        return super().reset(seed=seed, options=options)

    def terminal_residual_authority(self):
        achieved = self._scale_state(self.core_env.state)
        desired = self._scale_state(self.core_env.p3_target)
        maximum_ratio = float(
            np.max(self.goal_error_ratios(achieved, desired))
        )
        fraction = np.clip(
            (maximum_ratio - self.terminal_fade_zero_ratio)
            / (
                self.terminal_fade_full_ratio
                - self.terminal_fade_zero_ratio
            ),
            0.0,
            1.0,
        )
        authority = fraction * fraction * (3.0 - 2.0 * fraction)
        return float(authority), maximum_ratio

    def residual_gate(self):
        """Replace the distance cutoff with a smooth exact-P3 terminal fade."""
        return self.terminal_residual_authority()[0]

    @staticmethod
    def _action_values(info, key, achieved_shape, online_value):
        leading_shape = achieved_shape[:-1]
        default = np.zeros(3, dtype=np.float64)
        if isinstance(info, dict):
            value = info.get(key, online_value)
            if value is None:
                value = default
            return np.broadcast_to(
                np.asarray(value, dtype=np.float64),
                (*leading_shape, 3),
            )
        if isinstance(info, (Sequence, np.ndarray)):
            values = []
            for item in info:
                if isinstance(item, dict) and key in item:
                    values.append(np.asarray(item[key], dtype=np.float64))
                else:
                    values.append(default.copy())
            return np.asarray(values, dtype=np.float64).reshape(
                (*leading_shape, 3)
            )
        return np.broadcast_to(default, (*leading_shape, 3))

    def compute_reward(self, achieved_goal, desired_goal, info):
        base_reward = np.asarray(
            super().compute_reward(achieved_goal, desired_goal, info),
            dtype=np.float64,
        )
        achieved_shape = np.asarray(achieved_goal).shape
        action = self._action_values(
            info,
            "effective_residual_normalized_action",
            achieved_shape,
            self._online_effective_residual_action,
        )
        previous_action = self._action_values(
            info,
            "previous_effective_residual_normalized_action",
            achieved_shape,
            self._online_previous_effective_residual_action,
        )
        magnitude_cost = np.sum(np.square(action), axis=-1)
        smoothness_cost = np.sum(np.square(action - previous_action), axis=-1)
        magnitude_scale = float(self.core_env.dt) / self.reward_reference_dt
        smoothness_scale = (
            self.reward_reference_dt / float(self.core_env.dt)
            if self.time_consistent_smoothness_reward
            else magnitude_scale
        )
        reward = (
            base_reward
            - magnitude_scale
            * self.residual_magnitude_weight
            * magnitude_cost
            - smoothness_scale
            * self.residual_smoothness_weight
            * smoothness_cost
        )
        reward = np.asarray(reward, dtype=np.float32)
        if reward.ndim == 0:
            return np.float32(reward)
        return reward

    def _step_once(self, residual_action):
        authority, maximum_ratio = self.terminal_residual_authority()
        effective_action = (
            authority * np.asarray(residual_action, dtype=np.float32)
        ).astype(np.float32)
        previous_effective_action = (
            self.previous_effective_residual_action.copy()
        )
        self._online_effective_residual_action = effective_action
        self._online_previous_effective_residual_action = (
            previous_effective_action
        )
        try:
            observation, reward, terminated, truncated, info = super()._step_once(
                residual_action
            )
        finally:
            self._online_effective_residual_action = None
            self._online_previous_effective_residual_action = None
        self.previous_effective_residual_action = effective_action
        info = dict(info)
        info.update(
            {
                "residual_gate_design": "p3_ratio_terminal_fade",
                "terminal_residual_authority": authority,
                "terminal_residual_maximum_ratio": maximum_ratio,
                "terminal_fade_full_ratio": self.terminal_fade_full_ratio,
                "terminal_fade_zero_ratio": self.terminal_fade_zero_ratio,
                "effective_residual_normalized_action": effective_action,
                "previous_effective_residual_normalized_action": (
                    previous_effective_action
                ),
                "residual_magnitude_weight": self.residual_magnitude_weight,
                "residual_smoothness_weight": self.residual_smoothness_weight,
                "magnitude_reward_time_scale": float(
                    self.core_env.dt / self.reward_reference_dt
                ),
                "smoothness_reward_time_scale": float(
                    self.reward_reference_dt / self.core_env.dt
                    if self.time_consistent_smoothness_reward
                    else self.core_env.dt / self.reward_reference_dt
                ),
            }
        )
        return observation, reward, terminated, truncated, info
