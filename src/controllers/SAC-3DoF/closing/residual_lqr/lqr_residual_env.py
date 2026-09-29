"""Goal-conditioned residual SAC environment around the closing LQR."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)


CLOSING_DIR = Path(__file__).resolve().parent
LQR_DIR = CLOSING_DIR.parents[2] / "LQR-3DoF"
if str(LQR_DIR) not in sys.path:
    sys.path.insert(0, str(LQR_DIR))

from lqr_3dof_closing import (  # noqa: E402
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


class ResidualLQRGoalConditionedClosingEnv(GoalConditionedClosingV3Env):
    """Execute ``clipped LQR action + scale * SAC residual action``.

    SAC acts only in the normalized residual action space.  The LQR receives
    the physical simulator state and sampled P3 target and is recomputed on
    every environment step.  Its raw physical command is clipped before the
    residual is added so that a residual can move the realizable command away
    from an actuator boundary even when the unconstrained LQR is saturated.
    """

    reward_scheme = "goal_conditioned_p3_residual_lqr_dt01_v2"

    def __init__(
        self,
        *,
        residual_scale=0.05,
        residual_zero_distance=None,
        residual_full_distance=None,
        residual_hold_steps=1,
        reward_reference_dt=1.0,
        time_scale_persistent_reward=True,
        lqr_controller=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.residual_scale = float(residual_scale)
        if not np.isfinite(self.residual_scale):
            raise ValueError("residual_scale must be finite.")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must lie in (0, 1].")
        if (
            not isinstance(residual_hold_steps, (int, np.integer))
            or isinstance(residual_hold_steps, (bool, np.bool_))
            or residual_hold_steps <= 0
        ):
            raise ValueError("residual_hold_steps must be a positive integer.")
        self.residual_hold_steps = int(residual_hold_steps)
        self.reward_reference_dt = float(reward_reference_dt)
        if not np.isfinite(self.reward_reference_dt) or self.reward_reference_dt <= 0.0:
            raise ValueError("reward_reference_dt must be finite and positive.")
        self.time_scale_persistent_reward = bool(time_scale_persistent_reward)
        if (residual_zero_distance is None) != (residual_full_distance is None):
            raise ValueError(
                "residual_zero_distance and residual_full_distance must be "
                "provided together."
            )
        self.residual_zero_distance = (
            None
            if residual_zero_distance is None
            else float(residual_zero_distance)
        )
        self.residual_full_distance = (
            None
            if residual_full_distance is None
            else float(residual_full_distance)
        )
        if self.residual_zero_distance is not None:
            if not np.isfinite(self.residual_zero_distance):
                raise ValueError("residual_zero_distance must be finite.")
            if not np.isfinite(self.residual_full_distance):
                raise ValueError("residual_full_distance must be finite.")
            if self.residual_zero_distance < 0.0:
                raise ValueError("residual_zero_distance must be non-negative.")
            if self.residual_full_distance <= self.residual_zero_distance:
                raise ValueError(
                    "residual_full_distance must exceed residual_zero_distance."
                )

        if lqr_controller is None:
            A, B = build_linearized_3dof_matrices(self.core_env)
            Q = build_Q_closing(dt=self.core_env.dt)
            R = build_R_closing(self.core_env)
            K, _ = compute_lqr_gain(A=A, B=B, Q=Q, R=R)
            lqr_controller = LQRClosingController(K)
        self.lqr_controller = lqr_controller

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Scale persistent V3 shaping by elapsed physical time.

        Goal-error and warning costs were designed at a one-second reference
        interval.  At 10 Hz they are multiplied by ``dt / reference_dt`` so
        their approximate cost per physical second is unchanged.  Terminal
        success, hard-failure, and timeout terms remain one-shot events.
        """
        base_reward = np.asarray(
            super().compute_reward(achieved_goal, desired_goal, info),
            dtype=np.float64,
        )
        ratios = self.goal_error_ratios(achieved_goal, desired_goal)
        leading_shape = np.shape(base_reward)
        success = np.all(ratios <= 1.0, axis=-1).astype(np.float64)
        failure = self._failure_mask(info, leading_shape)
        time_limit = self._time_limit_mask(info, leading_shape)
        terminal_reward = (
            self.success_bonus * success
            - self.safety_failure_penalty * failure
            - self.timeout_penalty * time_limit * (1.0 - success)
        )
        persistent_scale = (
            float(self.core_env.dt) / self.reward_reference_dt
            if self.time_scale_persistent_reward
            else 1.0
        )
        reward = terminal_reward + persistent_scale * (
            base_reward - terminal_reward
        )
        reward = np.asarray(reward, dtype=np.float32)
        if reward.ndim == 0:
            return np.float32(reward)
        return reward

    def lqr_action(self):
        """Return raw physical, clipped physical, and normalized LQR actions."""
        state = np.asarray(self.core_env.state, dtype=np.float64)
        raw_physical = np.asarray(
            self.lqr_controller.compute_action(self.core_env, state),
            dtype=np.float64,
        )
        if raw_physical.shape != (3,) or not np.all(np.isfinite(raw_physical)):
            raise ValueError("The LQR controller returned an invalid action.")

        limits = self.env.get_physical_action_limits()
        clipped_physical = np.clip(raw_physical, -limits, limits)
        normalized = self.env.physical_action_to_normalized(clipped_physical)
        return raw_physical, clipped_physical, normalized

    def source_action(self):
        """Return the normalized actuator-realizable LQR action."""
        return self.lqr_action()[2].astype(np.float32)

    def residual_gate(self):
        """Return smooth residual authority based on axial P3 distance."""
        if self.residual_zero_distance is None:
            return 1.0
        axial_distance = abs(
            float(self.core_env.state[0] - self.core_env.p3_target[0])
        )
        fraction = np.clip(
            (axial_distance - self.residual_zero_distance)
            / (self.residual_full_distance - self.residual_zero_distance),
            0.0,
            1.0,
        )
        # Smoothstep avoids a derivative discontinuity in applied authority.
        return float(fraction * fraction * (3.0 - 2.0 * fraction))

    def _step_once(self, residual_action):
        """Execute one simulator step with a freshly recomputed LQR action."""
        raw_physical, clipped_physical, source_action = self.lqr_action()
        residual_gate = self.residual_gate()
        scaled_residual = (
            self.residual_scale * residual_gate * residual_action
        ).astype(np.float32)
        executed_action = np.clip(
            source_action + scaled_residual,
            -1.0,
            1.0,
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = super().step(
            executed_action
        )
        info = dict(info)
        info.update(
            {
                "residual_controller_enabled": True,
                "residual_baseline": "closing_lqr",
                "residual_scale": self.residual_scale,
                "residual_gate": residual_gate,
                "residual_zero_distance": self.residual_zero_distance,
                "residual_full_distance": self.residual_full_distance,
                "persistent_reward_time_scale": float(
                    self.core_env.dt / self.reward_reference_dt
                    if self.time_scale_persistent_reward
                    else 1.0
                ),
                "reward_reference_dt": self.reward_reference_dt,
                "lqr_raw_physical_action": raw_physical.copy(),
                "lqr_clipped_physical_action": clipped_physical.copy(),
                "lqr_saturated": bool(
                    np.any(np.abs(raw_physical - clipped_physical) > 1e-12)
                ),
                "source_normalized_action": source_action.copy(),
                "residual_normalized_action": residual_action.copy(),
                "scaled_residual_normalized_action": scaled_residual.copy(),
                "executed_normalized_action": executed_action.copy(),
            }
        )
        return observation, reward, terminated, truncated, info

    def step(self, action):
        residual_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if residual_action.shape != (3,):
            raise ValueError("Residual action must have shape (3,).")
        if not np.all(np.isfinite(residual_action)):
            raise ValueError("Residual action must contain only finite values.")

        diagnostic_reward_sum = 0.0
        executed_steps = 0
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.residual_hold_steps):
            observation, reward, terminated, truncated, info = self._step_once(
                residual_action
            )
            diagnostic_reward_sum += float(reward)
            executed_steps += 1
            if (
                terminated
                or truncated
                or bool(info.get("is_success", False))
            ):
                break

        info = dict(info)
        info.update(
            {
                "residual_hold_steps_requested": self.residual_hold_steps,
                "residual_hold_steps_executed": executed_steps,
                "residual_decision_interval_seconds_requested": float(
                    self.residual_hold_steps * self.core_env.dt
                ),
                "residual_decision_interval_seconds_executed": float(
                    executed_steps * self.core_env.dt
                ),
                "lqr_recomputed_each_simulator_step": True,
                "residual_hold_reward_aggregation": (
                    "final_substep_for_her_compatibility"
                ),
                "residual_hold_reward_sum_diagnostic": diagnostic_reward_sum,
            }
        )
        return observation, reward, terminated, truncated, info
