"""Goal-conditioned closing environment for SAC with HER.

The environment preserves the original closing dynamics, P2 reset
distribution, hard safety termination, and exact six-component P3 success
set.  It replaces the history-dependent closing reward with a bounded reward
that can be recomputed after HER changes ``desired_goal``.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.docking3dof.gym_env import Docking3DoFGymEnv


class GoalConditionedClosingEnv(gym.Wrapper):
    """Expose closing as a Gymnasium goal-conditioned environment.

    All three observation entries use fixed, shared physical scales.  In
    particular, ``achieved_goal`` and ``desired_goal`` are transformed with
    the same affine map.  This avoids applying unrelated running statistics
    to real P3 goals and HER-relabeled future goals.
    """

    reward_scheme = "goal_conditioned_p3_v1"
    goal_ratio_clip = 10.0
    goal_log_reference_ratio = 10.0
    goal_cost_weight = 1.0
    success_bonus = 1.0
    safety_failure_penalty = 1.0
    velocity_warning_penalty = 0.0
    geometry_warning_penalty = 0.0
    alignment_warning_penalty = 0.0
    timeout_penalty = 0.0

    # Representative closing-phase scales.  They are used only to condition
    # neural-network inputs; success and reward are computed in physical units.
    state_scale = np.array(
        [
            2000.0,
            200.0,
            0.25,
            0.05,
            np.deg2rad(5.0),
            np.deg2rad(0.10),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        terminate_on_success=False,
        seed=None,
    ):
        env = Docking3DoFGymEnv(
            phase="closing",
            mode=mode,
            dt=dt,
            max_time=max_time,
            integrator=integrator,
            normalize_action=normalize_action,
            seed=seed,
            # The inherited reward is deliberately discarded in ``step``.
            closing_reward_version="v15",
        )
        super().__init__(env)
        self.terminate_on_success = bool(terminate_on_success)
        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": vector_space,
                "achieved_goal": vector_space,
                "desired_goal": vector_space,
            }
        )

    @property
    def core_env(self):
        return self.env.core_env

    def _scale_state(self, state):
        state = np.asarray(state, dtype=np.float64).copy()
        state[4] = self.core_env.wrap_to_pi(state[4])
        return (state / self.state_scale).astype(np.float32)

    def _goal_observation(self, base_observation):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")
        port_state = np.asarray(base_observation[:6], dtype=np.float64)
        return {
            "observation": self._scale_state(port_state),
            "achieved_goal": self._scale_state(self.core_env.state),
            "desired_goal": self._scale_state(self.core_env.p3_target),
        }

    @classmethod
    def _failure_mask(cls, info, leading_shape):
        if isinstance(info, dict):
            return np.full(
                leading_shape or (),
                float(bool(info.get("failure", False))),
                dtype=np.float64,
            )
        if isinstance(info, (Sequence, np.ndarray)):
            flat = np.asarray(
                [
                    float(bool(item.get("failure", False)))
                    if isinstance(item, dict)
                    else 0.0
                    for item in info
                ],
                dtype=np.float64,
            )
            return flat.reshape(leading_shape)
        return np.zeros(leading_shape or (), dtype=np.float64)

    def _velocity_warning_depth(self, info, leading_shape):
        """Return a goal-independent velocity warning depth in ``[0, 1]``."""

        def depth(item):
            if not isinstance(item, dict):
                return 0.0
            port_speed = float(item.get("port_speed", 0.0))
            axial_speed = abs(float(item.get("s_dot", 0.0)))
            core = self.core_env
            port_depth = np.clip(
                (port_speed - core.vp_safe_close)
                / max(core.vp_unsafe_close - core.vp_safe_close, 1e-12),
                0.0,
                1.0,
            )
            axial_depth = np.clip(
                (axial_speed - core.sdot_safe_close)
                / max(core.sdot_unsafe_close - core.sdot_safe_close, 1e-12),
                0.0,
                1.0,
            )
            return float(max(port_depth, axial_depth))

        if isinstance(info, dict):
            return np.full(
                leading_shape or (),
                depth(info),
                dtype=np.float64,
            )
        if isinstance(info, (Sequence, np.ndarray)):
            flat = np.asarray([depth(item) for item in info], dtype=np.float64)
            return flat.reshape(leading_shape)
        return np.zeros(leading_shape or (), dtype=np.float64)

    def _geometry_warning_depth(self, info, leading_shape):
        """Return continuous approach-ellipse/cone warning depth."""

        def depth(item):
            if not isinstance(item, dict):
                return 0.0
            core = self.core_env
            state = item.get("state")
            if state is not None:
                q = core.get_derived_quantities(
                    np.asarray(state, dtype=np.float64)
                )
                rx, ry = q["r_com"]
                s = float(q["s"])
                rho = float(q["rho"])
            else:
                # Ordinary environment infos expose the required derived
                # quantities except the CoM pair.  Recover it from current
                # state for online rewards; HER prefills explicitly store it.
                q = core.get_derived_quantities(core.state)
                rx, ry = q["r_com"]
                s = float(item.get("s", q["s"]))
                rho = float(item.get("rho", q["rho"]))
            ellipse_value = (rx / core.ae_x) ** 2 + (ry / core.ae_y) ** 2
            ellipse_depth = np.clip(
                (ellipse_value - 0.90) / 0.10,
                0.0,
                1.0,
            )
            allowed_radius = core.cone_slope * np.clip(
                s,
                0.0,
                core.cone_length,
            )
            cone_depth = np.clip(
                (rho - allowed_radius) / max(core.cone_base_radius, 1e-12),
                0.0,
                1.0,
            )
            return float(max(ellipse_depth, cone_depth))

        if isinstance(info, dict):
            return np.full(leading_shape or (), depth(info), dtype=np.float64)
        if isinstance(info, (Sequence, np.ndarray)):
            flat = np.asarray([depth(item) for item in info], dtype=np.float64)
            return flat.reshape(leading_shape)
        return np.zeros(leading_shape or (), dtype=np.float64)

    def _alignment_warning_depth(self, info, leading_shape):
        """Return continuous attitude/rate warning depth in ``[0, 1]``."""

        def depth(item):
            if not isinstance(item, dict):
                return 0.0
            core = self.core_env
            psi = abs(float(item.get("psi_e", 0.0)))
            omega = abs(float(item.get("omega", 0.0)))
            psi_depth = np.clip(
                (psi - core.psi_safe)
                / max(core.psi_unsafe - core.psi_safe, 1e-12),
                0.0,
                1.0,
            )
            omega_depth = np.clip(
                (omega - core.omega_safe)
                / max(core.omega_unsafe - core.omega_safe, 1e-12),
                0.0,
                1.0,
            )
            return float(max(psi_depth, omega_depth))

        if isinstance(info, dict):
            return np.full(leading_shape or (), depth(info), dtype=np.float64)
        if isinstance(info, (Sequence, np.ndarray)):
            flat = np.asarray([depth(item) for item in info], dtype=np.float64)
            return flat.reshape(leading_shape)
        return np.zeros(leading_shape or (), dtype=np.float64)

    @staticmethod
    def _time_limit_mask(info, leading_shape):
        def value(item):
            return float(bool(item.get("time_limit", False))) if isinstance(item, dict) else 0.0

        if isinstance(info, dict):
            return np.full(leading_shape or (), value(info), dtype=np.float64)
        if isinstance(info, (Sequence, np.ndarray)):
            flat = np.asarray([value(item) for item in info], dtype=np.float64)
            return flat.reshape(leading_shape)
        return np.zeros(leading_shape or (), dtype=np.float64)

    def goal_error_ratios(self, achieved_goal, desired_goal):
        """Return exact P3-tolerance ratios for scalar or batched goals."""
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        desired = np.asarray(desired_goal, dtype=np.float64)
        if achieved.shape != desired.shape or achieved.shape[-1:] != (6,):
            raise ValueError(
                "achieved_goal and desired_goal must have matching (..., 6) "
                "shapes."
            )
        physical_error = (achieved - desired) * self.state_scale
        physical_error = physical_error.copy()
        physical_error[..., 4] = np.arctan2(
            np.sin(physical_error[..., 4]),
            np.cos(physical_error[..., 4]),
        )
        tolerance = np.asarray(
            self.core_env.p3_target_tolerance,
            dtype=np.float64,
        )
        return np.abs(physical_error) / np.maximum(tolerance, 1e-12)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the bounded, HER-compatible reward.

        The goal cost lies in ``[0, 1]`` and equally combines the mean and
        worst tolerance-normalized component.  A relabeled transition that
        reaches its alternative goal receives the same success bonus as a
        real P3 transition.  Safety failure remains goal-independent and is
        copied through HER's info dictionaries.
        """
        ratios = self.goal_error_ratios(achieved_goal, desired_goal)
        if self.goal_ratio_clip is None:
            transformed_ratios = ratios
        else:
            transformed_ratios = np.minimum(ratios, self.goal_ratio_clip)
        log_scale = np.log1p(self.goal_log_reference_ratio)
        normalized = np.log1p(transformed_ratios) / log_scale
        goal_cost = 0.5 * np.mean(normalized, axis=-1) + 0.5 * np.max(
            normalized,
            axis=-1,
        )
        success = np.all(ratios <= 1.0, axis=-1).astype(np.float64)
        failure = self._failure_mask(info, np.shape(goal_cost))
        warning_depth = self._velocity_warning_depth(
            info,
            np.shape(goal_cost),
        )
        geometry_depth = self._geometry_warning_depth(
            info,
            np.shape(goal_cost),
        )
        alignment_depth = self._alignment_warning_depth(
            info,
            np.shape(goal_cost),
        )
        time_limit = self._time_limit_mask(info, np.shape(goal_cost))
        reward = (
            -self.goal_cost_weight * goal_cost
            + self.success_bonus * success
            - self.safety_failure_penalty * failure
            - self.velocity_warning_penalty * warning_depth
            - self.geometry_warning_penalty * geometry_depth
            - self.alignment_warning_penalty * alignment_depth
            - self.timeout_penalty * time_limit * (1.0 - success)
        )
        reward = np.asarray(reward, dtype=np.float32)
        if reward.ndim == 0:
            return np.float32(reward)
        return reward

    def reset(self, seed=None, options=None):
        base_observation, info = self.env.reset(seed=seed, options=options)
        observation = self._goal_observation(base_observation)
        ratios = self.goal_error_ratios(
            observation["achieved_goal"],
            observation["desired_goal"],
        )
        info = dict(info)
        info.update(
            {
                "reward_scheme": self.reward_scheme,
                "is_success": float(np.all(ratios <= 1.0)),
                "p3_max_target_ratio": float(np.max(ratios)),
            }
        )
        return observation, info

    def step(self, action):
        base_observation, inherited_reward, terminated, truncated, info = (
            self.env.step(action)
        )
        info = dict(info)
        # HER recomputes rewards long after this transition was collected.
        # Store the physical next state so geometry warning depth does not
        # accidentally depend on whichever state the live env has later.
        info["state"] = np.asarray(
            self.core_env.state,
            dtype=np.float64,
        ).copy()
        observation = self._goal_observation(base_observation)
        reward = self.compute_reward(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )
        ratios = self.goal_error_ratios(
            observation["achieved_goal"],
            observation["desired_goal"],
        )
        original_success = bool(info.get("success", False))
        if original_success and not self.terminate_on_success:
            # SB3's HER buffer recomputes rewards but intentionally preserves
            # done flags.  P3 success must therefore be non-terminal while
            # gathering HER experience; safety and time-limit endings remain.
            terminated = bool(info.get("failure", False))
        info.update(
            {
                "reward_scheme": self.reward_scheme,
                "is_success": float(np.all(ratios <= 1.0)),
                "original_p3_success": original_success,
                "goal_success_termination_enabled": (
                    self.terminate_on_success
                ),
                "p3_max_target_ratio": float(np.max(ratios)),
                "inherited_v15_reward_ignored": float(inherited_reward),
                "her_reward": float(reward),
            }
        )
        return observation, float(reward), terminated, truncated, info


class GoalConditionedClosingV2Env(GoalConditionedClosingEnv):
    """Guarded HER reward with visible far-field progress.

    V1 clipped every normalized goal error at ten tolerances.  The initial
    axial P3 error is roughly 71 tolerances, so this removed the axial reward
    gradient over most of closing.  V1 also assigned only ``-1`` to a hard
    failure while accumulating up to ``-1`` goal cost per step, making early
    termination an attractive shortcut.  V2 corrects only those diagnosed
    scales and adds a bounded, goal-independent warning-depth penalty.
    """

    reward_scheme = "goal_conditioned_p3_v2"
    goal_ratio_clip = 100.0
    goal_log_reference_ratio = 100.0
    safety_failure_penalty = 1000.0
    velocity_warning_penalty = 10.0


class GoalConditionedClosingV3Env(GoalConditionedClosingEnv):
    """Long-horizon HER reward used with safe V15 behavior initialization."""

    reward_scheme = "goal_conditioned_p3_v3_source_initialized"
    goal_ratio_clip = None
    goal_log_reference_ratio = 100.0
    goal_cost_weight = 0.01
    success_bonus = 100.0
    safety_failure_penalty = 1000.0
    velocity_warning_penalty = 1.0
    geometry_warning_penalty = 1.0
    alignment_warning_penalty = 1.0
    timeout_penalty = 100.0
