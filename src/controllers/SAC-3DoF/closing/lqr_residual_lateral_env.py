"""Lateral-progress reward for direct residual SAC around the 10 Hz LQR."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv


class LateralProgressResidualLQRGoalConditionedClosingEnv(
    ResidualLQRGoalConditionedClosingEnv
):
    """Use HER-compatible P3 and lateral progress for direct residual actions.

    The controller and action-combination law are inherited unchanged.  Only
    the reward differs from the validated absolute-cost residual wrapper.
    """

    reward_scheme = "goal_conditioned_p3_direct_residual_lqr_dt01_lateral_v1"

    def __init__(
        self,
        *,
        progress_gamma=0.9998999549714793,
        progress_weight=2.0,
        goal_component_weights=(1.0, 4.0, 1.0, 2.0, 1.0, 1.0),
        worst_component_weight=0.75,
        lateral_constraint_progress_weight=4.0,
        success_bonus=5.0,
        safety_failure_penalty=25.0,
        timeout_penalty=2.0,
        timeout_violation_weight=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 < float(progress_gamma) <= 1.0:
            raise ValueError("progress_gamma must lie in (0, 1].")
        nonnegative = {
            "progress_weight": progress_weight,
            "lateral_constraint_progress_weight": (
                lateral_constraint_progress_weight
            ),
            "success_bonus": success_bonus,
            "safety_failure_penalty": safety_failure_penalty,
            "timeout_penalty": timeout_penalty,
            "timeout_violation_weight": timeout_violation_weight,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        component_weights = np.asarray(goal_component_weights, dtype=np.float64)
        if (
            component_weights.shape != (6,)
            or not np.all(np.isfinite(component_weights))
            or np.any(component_weights <= 0.0)
        ):
            raise ValueError(
                "goal_component_weights must contain six finite positives."
            )
        if not 0.0 <= float(worst_component_weight) <= 1.0:
            raise ValueError("worst_component_weight must lie in [0, 1].")

        self.progress_gamma = float(progress_gamma)
        self.progress_weight = float(progress_weight)
        self.goal_component_weights = component_weights
        self.worst_component_weight = float(worst_component_weight)
        self.lateral_constraint_progress_weight = float(
            lateral_constraint_progress_weight
        )
        self.success_bonus = float(success_bonus)
        self.safety_failure_penalty = float(safety_failure_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.timeout_violation_weight = float(timeout_violation_weight)
        self._online_previous_achieved_goal = None

    def _goal_cost(self, achieved_goal, desired_goal):
        ratios = self.goal_error_ratios(achieved_goal, desired_goal)
        normalized = np.log1p(ratios) / np.log1p(
            self.goal_log_reference_ratio
        )
        weighted_mean = np.sum(
            normalized * self.goal_component_weights,
            axis=-1,
        ) / np.sum(self.goal_component_weights)
        return (
            (1.0 - self.worst_component_weight) * weighted_mean
            + self.worst_component_weight * np.max(normalized, axis=-1)
        )

    @staticmethod
    def _lateral_constraint_cost(ratios):
        lateral_excess = np.maximum(
            np.asarray(ratios, dtype=np.float64)[..., 1] - 1.0,
            0.0,
        )
        return np.log1p(lateral_excess) / np.log(2.0)

    @staticmethod
    def _worst_constraint_cost(ratios):
        worst_excess = np.maximum(
            np.max(np.asarray(ratios, dtype=np.float64), axis=-1) - 1.0,
            0.0,
        )
        return np.log1p(worst_excess) / np.log(2.0)

    def _previous_goals(self, achieved_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        online_previous = self._online_previous_achieved_goal
        if isinstance(info, dict):
            if "previous_achieved_goal" in info:
                previous = np.asarray(
                    info["previous_achieved_goal"], dtype=np.float64
                )
            elif online_previous is not None:
                previous = np.asarray(online_previous, dtype=np.float64)
            else:
                previous = achieved
            return np.broadcast_to(previous, achieved.shape)
        if isinstance(info, (Sequence, np.ndarray)):
            flattened_achieved = achieved.reshape(-1, 6)
            previous_rows = []
            for index, item in enumerate(info):
                if isinstance(item, dict) and "previous_achieved_goal" in item:
                    previous_rows.append(
                        np.asarray(item["previous_achieved_goal"], dtype=np.float64)
                    )
                else:
                    previous_rows.append(flattened_achieved[index])
            return np.asarray(previous_rows, dtype=np.float64).reshape(achieved.shape)
        return achieved

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Return potential progress plus explicit lateral P3 progress."""
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        leading_shape = achieved.shape[:-1]
        previous_achieved = self._previous_goals(achieved_goal, info)
        next_cost = self._goal_cost(achieved_goal, desired_goal)
        previous_cost = self._goal_cost(previous_achieved, desired_goal)
        ratios = self.goal_error_ratios(achieved_goal, desired_goal)
        previous_ratios = self.goal_error_ratios(
            previous_achieved,
            desired_goal,
        )
        progress = previous_cost - self.progress_gamma * next_cost
        lateral_progress = (
            self._lateral_constraint_cost(previous_ratios)
            - self.progress_gamma * self._lateral_constraint_cost(ratios)
        )
        success = np.all(ratios <= 1.0, axis=-1).astype(np.float64)
        failure = self._failure_mask(info, leading_shape)
        time_limit = self._time_limit_mask(info, leading_shape)
        timeout_multiplier = (
            1.0
            + self.timeout_violation_weight
            * self._worst_constraint_cost(ratios)
        )
        reward = (
            self.progress_weight * progress
            + self.lateral_constraint_progress_weight * lateral_progress
            + self.success_bonus * success
            - self.safety_failure_penalty * failure
            - self.timeout_penalty
            * timeout_multiplier
            * time_limit
            * (1.0 - success)
        )
        reward = np.asarray(reward, dtype=np.float32)
        if reward.ndim == 0:
            return np.float32(reward)
        return reward

    def _step_once(self, residual_action):
        previous_achieved = self._scale_state(self.core_env.state).copy()
        self._online_previous_achieved_goal = previous_achieved
        try:
            observation, reward, terminated, truncated, info = super()._step_once(
                residual_action
            )
        finally:
            self._online_previous_achieved_goal = None
        info = dict(info)
        info.update(
            {
                "previous_achieved_goal": previous_achieved,
                "reward_design": "dt01_lateral_progress_v1",
                "progress_gamma": self.progress_gamma,
                "progress_weight": self.progress_weight,
                "goal_component_weights": self.goal_component_weights.copy(),
                "worst_component_weight": self.worst_component_weight,
                "lateral_constraint_progress_weight": (
                    self.lateral_constraint_progress_weight
                ),
                "timeout_violation_weight": self.timeout_violation_weight,
            }
        )
        return observation, reward, terminated, truncated, info
