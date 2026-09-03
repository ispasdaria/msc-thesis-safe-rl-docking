"""Goal-conditioned SAC reference residuals around the closing LQR.

The SAC action does not add actuator acceleration.  Instead, it applies
bounded offsets to the axial-velocity, lateral-velocity, and attitude
references tracked by the unchanged closing LQR gain.  A zero SAC action
therefore reproduces the established LQR controller exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)


CLOSING_DIR = Path(__file__).resolve().parent
LQR_DIR = CLOSING_DIR.parents[1] / "LQR-3DoF"
if str(LQR_DIR) not in sys.path:
    sys.path.insert(0, str(LQR_DIR))

from lqr_3dof_closing import (  # noqa: E402
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    closing_error,
    compute_lqr_gain,
    orbital_to_body_action,
)


class ReferenceResidualLQRGoalConditionedClosingEnv(
    GoalConditionedClosingV3Env
):
    """Let SAC adjust three references tracked by the configured closing LQR."""

    reward_scheme = "goal_conditioned_p3_reference_residual_lqr_v1"

    def __init__(
        self,
        *,
        reference_residual_bounds=(0.02, 0.01, np.deg2rad(1.0)),
        progress_gamma=0.9999,
        progress_weight=100.0,
        reference_magnitude_weight=0.01,
        reference_smoothness_weight=0.01,
        success_bonus=100.0,
        safety_failure_penalty=1000.0,
        timeout_penalty=100.0,
        time_scale_persistent_reward=False,
        time_consistent_smoothness_reward=False,
        reward_reference_dt=1.0,
        goal_component_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        worst_component_weight=0.5,
        lateral_constraint_progress_weight=0.0,
        timeout_violation_weight=0.0,
        reward_design="v1",
        terminal_residual_fade_full_ratio=None,
        terminal_residual_fade_zero_ratio=1.0,
        lqr_controller=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        bounds = np.asarray(reference_residual_bounds, dtype=np.float64)
        if bounds.shape != (3,) or not np.all(np.isfinite(bounds)):
            raise ValueError(
                "reference_residual_bounds must contain three finite values."
            )
        if np.any(bounds <= 0.0):
            raise ValueError("reference residual bounds must be positive.")
        if not 0.0 < float(progress_gamma) <= 1.0:
            raise ValueError("progress_gamma must lie in (0, 1].")
        nonnegative = {
            "progress_weight": progress_weight,
            "reference_magnitude_weight": reference_magnitude_weight,
            "reference_smoothness_weight": reference_smoothness_weight,
            "success_bonus": success_bonus,
            "safety_failure_penalty": safety_failure_penalty,
            "timeout_penalty": timeout_penalty,
            "lateral_constraint_progress_weight": (
                lateral_constraint_progress_weight
            ),
            "timeout_violation_weight": timeout_violation_weight,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

        self.reference_residual_bounds = bounds
        self.progress_gamma = float(progress_gamma)
        self.progress_weight = float(progress_weight)
        self.reference_magnitude_weight = float(reference_magnitude_weight)
        self.reference_smoothness_weight = float(reference_smoothness_weight)
        self.success_bonus = float(success_bonus)
        self.safety_failure_penalty = float(safety_failure_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.time_scale_persistent_reward = bool(
            time_scale_persistent_reward
        )
        self.time_consistent_smoothness_reward = bool(
            time_consistent_smoothness_reward
        )
        self.reward_reference_dt = float(reward_reference_dt)
        if (
            not np.isfinite(self.reward_reference_dt)
            or self.reward_reference_dt <= 0.0
        ):
            raise ValueError("reward_reference_dt must be finite and positive.")
        component_weights = np.asarray(
            goal_component_weights,
            dtype=np.float64,
        )
        if (
            component_weights.shape != (6,)
            or not np.all(np.isfinite(component_weights))
            or np.any(component_weights <= 0.0)
        ):
            raise ValueError(
                "goal_component_weights must contain six finite positive "
                "values."
            )
        if not 0.0 <= float(worst_component_weight) <= 1.0:
            raise ValueError("worst_component_weight must lie in [0, 1].")
        if reward_design not in {"v1", "dt01_lateral_v2"}:
            raise ValueError(
                "reward_design must be 'v1' or 'dt01_lateral_v2'."
            )
        self.goal_component_weights = component_weights
        self.worst_component_weight = float(worst_component_weight)
        self.lateral_constraint_progress_weight = float(
            lateral_constraint_progress_weight
        )
        self.timeout_violation_weight = float(timeout_violation_weight)
        self.reward_design = str(reward_design)
        if self.reward_design == "dt01_lateral_v2":
            self.reward_scheme = (
                "goal_conditioned_p3_reference_residual_lqr_dt01_lateral_v2"
            )
        self.terminal_residual_fade_full_ratio = (
            None
            if terminal_residual_fade_full_ratio is None
            else float(terminal_residual_fade_full_ratio)
        )
        self.terminal_residual_fade_zero_ratio = float(
            terminal_residual_fade_zero_ratio
        )
        if not np.isfinite(self.terminal_residual_fade_zero_ratio) or (
            self.terminal_residual_fade_zero_ratio < 0.0
        ):
            raise ValueError(
                "terminal_residual_fade_zero_ratio must be finite and "
                "non-negative."
            )
        if self.terminal_residual_fade_full_ratio is not None and (
            not np.isfinite(self.terminal_residual_fade_full_ratio)
            or self.terminal_residual_fade_full_ratio
            <= self.terminal_residual_fade_zero_ratio
        ):
            raise ValueError(
                "terminal_residual_fade_full_ratio must be finite and above "
                "terminal_residual_fade_zero_ratio."
            )
        self.previous_reference_residual_action = np.zeros(3, dtype=np.float32)

        if lqr_controller is None:
            A, B = build_linearized_3dof_matrices(self.core_env)
            Q = build_Q_closing(dt=self.core_env.dt)
            R = build_R_closing(self.core_env)
            K, _ = compute_lqr_gain(A=A, B=B, Q=Q, R=R)
            lqr_controller = LQRClosingController(K)
        if not isinstance(lqr_controller, LQRClosingController):
            raise TypeError(
                "lqr_controller must be an LQRClosingController instance."
            )
        self.lqr_controller = lqr_controller

    @staticmethod
    def _info_values(info, key, leading_shape, width=None):
        default = np.zeros(width, dtype=np.float64) if width else 0.0

        def value(item):
            if not isinstance(item, dict) or key not in item:
                return default.copy() if width else default
            result = np.asarray(item[key], dtype=np.float64)
            if width:
                return result.reshape(width)
            return float(result)

        if isinstance(info, dict):
            result = value(info)
            if width:
                return np.broadcast_to(result, (*leading_shape, width))
            return np.full(leading_shape or (), result, dtype=np.float64)
        if isinstance(info, (Sequence, np.ndarray)):
            result = np.asarray([value(item) for item in info], dtype=np.float64)
            suffix = (width,) if width else ()
            return result.reshape((*leading_shape, *suffix))
        shape = (*leading_shape, width) if width else (leading_shape or ())
        return np.zeros(shape, dtype=np.float64)

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
        """Smoothly measure only lateral-position target-set violation."""
        lateral_excess = np.maximum(
            np.asarray(ratios, dtype=np.float64)[..., 1] - 1.0,
            0.0,
        )
        return np.log1p(lateral_excess) / np.log(2.0)

    @staticmethod
    def _worst_constraint_cost(ratios):
        """Smoothly measure the worst violation outside the exact P3 set."""
        worst_excess = np.maximum(
            np.max(np.asarray(ratios, dtype=np.float64), axis=-1) - 1.0,
            0.0,
        )
        return np.log1p(worst_excess) / np.log(2.0)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute relabelable P3 progress and reference-control costs."""
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        leading_shape = achieved.shape[:-1]
        previous_achieved = self._info_values(
            info,
            "previous_achieved_goal",
            leading_shape,
            width=6,
        )
        next_cost = self._goal_cost(achieved_goal, desired_goal)
        previous_cost = self._goal_cost(previous_achieved, desired_goal)
        ratios = self.goal_error_ratios(achieved_goal, desired_goal)
        previous_ratios = self.goal_error_ratios(
            previous_achieved,
            desired_goal,
        )
        success = np.all(ratios <= 1.0, axis=-1).astype(np.float64)
        failure = self._failure_mask(info, leading_shape)
        time_limit = self._time_limit_mask(info, leading_shape)
        action = self._info_values(
            info,
            "reference_residual_normalized_action",
            leading_shape,
            width=3,
        )
        previous_action = self._info_values(
            info,
            "previous_reference_residual_normalized_action",
            leading_shape,
            width=3,
        )
        magnitude_cost = np.sum(np.square(action), axis=-1)
        smoothness_cost = np.sum(np.square(action - previous_action), axis=-1)
        magnitude_scale = (
            float(self.core_env.dt) / self.reward_reference_dt
            if self.time_scale_persistent_reward
            else 1.0
        )
        smoothness_scale = (
            self.reward_reference_dt / float(self.core_env.dt)
            if self.time_consistent_smoothness_reward
            else magnitude_scale
        )
        progress = previous_cost - self.progress_gamma * next_cost
        lateral_progress = (
            self._lateral_constraint_cost(previous_ratios)
            - self.progress_gamma * self._lateral_constraint_cost(ratios)
        )
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
            - magnitude_scale * self.reference_magnitude_weight * magnitude_cost
            - smoothness_scale
            * self.reference_smoothness_weight
            * smoothness_cost
        )
        reward = np.asarray(reward, dtype=np.float32)
        if reward.ndim == 0:
            return np.float32(reward)
        return reward

    def scheduled_lqr_reference(self):
        """Return the reference used by the established zero-residual LQR."""
        state = np.asarray(self.core_env.state, dtype=np.float64)
        reference = np.asarray(self.core_env.p3_target, dtype=np.float64).copy()
        axial_distance = abs(float(state[0] - reference[0]))
        controller = self.lqr_controller
        if axial_distance >= controller.d_fast:
            vx_reference = controller.v_fast
        elif axial_distance <= controller.d_slow:
            vx_reference = reference[2]
        else:
            fraction = (
                (axial_distance - controller.d_slow)
                / (controller.d_fast - controller.d_slow)
            )
            vx_reference = (
                (1.0 - fraction) * reference[2]
                + fraction * controller.v_fast
            )
        reference[2] = vx_reference
        return reference

    def lqr_action_for_reference_residual(self, residual_action):
        """Return LQR actions and nominal/modified reference diagnostics."""
        residual_action = np.asarray(residual_action, dtype=np.float64)
        nominal_reference = self.scheduled_lqr_reference()
        offsets = self.reference_residual_bounds * residual_action
        modified_reference = nominal_reference.copy()
        modified_reference[2] += offsets[0]
        modified_reference[3] += offsets[1]
        modified_reference[4] = self.core_env.wrap_to_pi(
            modified_reference[4] + offsets[2]
        )
        state = np.asarray(self.core_env.state, dtype=np.float64)
        error = closing_error(self.core_env, state, modified_reference)
        orbital_action = -self.lqr_controller.K @ error
        raw_physical = orbital_to_body_action(
            self.core_env,
            orbital_action,
            psi=state[4],
        )
        limits = self.env.get_physical_action_limits()
        clipped_physical = np.clip(raw_physical, -limits, limits)
        normalized = self.env.physical_action_to_normalized(clipped_physical)
        return (
            raw_physical,
            clipped_physical,
            normalized.astype(np.float32),
            nominal_reference,
            modified_reference,
            offsets,
        )

    def source_action(self):
        return np.zeros(3, dtype=np.float32)

    def terminal_residual_authority(self):
        """Return residual authority and the current worst P3 error ratio."""
        state = np.asarray(self.core_env.state, dtype=np.float64)
        target = np.asarray(self.core_env.p3_target, dtype=np.float64)
        error = state - target
        error[4] = self.core_env.wrap_to_pi(error[4])
        tolerance = np.asarray(
            self.core_env.p3_target_tolerance,
            dtype=np.float64,
        )
        maximum_ratio = float(
            np.max(np.abs(error) / np.maximum(tolerance, 1e-12))
        )
        full_ratio = self.terminal_residual_fade_full_ratio
        if full_ratio is None:
            return 1.0, maximum_ratio
        zero_ratio = self.terminal_residual_fade_zero_ratio
        authority = np.clip(
            (maximum_ratio - zero_ratio) / (full_ratio - zero_ratio),
            0.0,
            1.0,
        )
        return float(authority), maximum_ratio

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.previous_reference_residual_action = np.zeros(3, dtype=np.float32)
        return observation, info

    def step(self, action):
        commanded_residual_action = np.asarray(
            action, dtype=np.float32
        ).reshape(-1)
        if commanded_residual_action.shape != (3,):
            raise ValueError("Reference residual action must have shape (3,).")
        if not np.all(np.isfinite(commanded_residual_action)):
            raise ValueError("Reference residual action must be finite.")
        commanded_residual_action = np.clip(
            commanded_residual_action,
            -1.0,
            1.0,
        )
        residual_authority, pre_step_max_target_ratio = (
            self.terminal_residual_authority()
        )
        residual_action = (
            commanded_residual_action * residual_authority
        ).astype(np.float32)
        previous_achieved_goal = self._scale_state(self.core_env.state)
        previous_residual_action = self.previous_reference_residual_action.copy()
        (
            raw_physical,
            clipped_physical,
            normalized_action,
            nominal_reference,
            modified_reference,
            offsets,
        ) = self.lqr_action_for_reference_residual(residual_action)
        observation, _, terminated, truncated, info = super().step(
            normalized_action
        )
        info = dict(info)
        info.update(
            {
                "reward_scheme": self.reward_scheme,
                "reference_residual_controller_enabled": True,
                "reference_residual_components": ("vx", "vy", "attitude"),
                "reference_residual_bounds": self.reference_residual_bounds.copy(),
                "reference_residual_commanded_action": (
                    commanded_residual_action.copy()
                ),
                "reference_residual_normalized_action": residual_action.copy(),
                "previous_reference_residual_normalized_action": (
                    previous_residual_action
                ),
                "reference_residual_physical_offsets": offsets.copy(),
                "terminal_residual_fade_enabled": bool(
                    self.terminal_residual_fade_full_ratio is not None
                ),
                "terminal_residual_authority": float(residual_authority),
                "terminal_residual_pre_step_max_target_ratio": float(
                    pre_step_max_target_ratio
                ),
                "terminal_residual_fade_full_ratio": (
                    self.terminal_residual_fade_full_ratio
                ),
                "terminal_residual_fade_zero_ratio": float(
                    self.terminal_residual_fade_zero_ratio
                ),
                "previous_achieved_goal": previous_achieved_goal.copy(),
                "nominal_lqr_reference": nominal_reference.copy(),
                "modified_lqr_reference": modified_reference.copy(),
                "lqr_raw_physical_action": raw_physical.copy(),
                "lqr_clipped_physical_action": clipped_physical.copy(),
                "lqr_saturated": bool(
                    np.any(np.abs(raw_physical - clipped_physical) > 1e-12)
                ),
                "executed_normalized_action": normalized_action.copy(),
                "sac_decision_interval_seconds": float(self.core_env.dt),
                "lqr_update_interval_seconds": float(self.core_env.dt),
                "persistent_reward_time_scale": float(
                    self.core_env.dt / self.reward_reference_dt
                    if self.time_scale_persistent_reward
                    else 1.0
                ),
                "smoothness_reward_time_scale": float(
                    self.reward_reference_dt / self.core_env.dt
                    if self.time_consistent_smoothness_reward
                    else (
                        self.core_env.dt / self.reward_reference_dt
                        if self.time_scale_persistent_reward
                        else 1.0
                    )
                ),
                "reward_design": self.reward_design,
                "cbf_filter_applied": False,
            }
        )
        reward = self.compute_reward(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )
        info["her_reward"] = float(reward)
        self.previous_reference_residual_action = residual_action.copy()
        return observation, float(reward), terminated, truncated, info
