"""Closing-only P3-recovery experiment with non-terminal safety violations.

This module is intentionally separate from :mod:`envs.docking3dof.core` and
:mod:`envs.docking3dof.gym_env`.  Importing it does not change the termination
logic used by any existing training or evaluation script.

The experiment changes two features in this isolated subclass:

* a closing safety violation is observed and penalized, but is not an episode
  boundary;
* an experiment-specific P3-reaching reward replaces the inherited closing
  reward without registering or changing a shared ``gym_env.py`` version.

P3 success, the time limit, and numerical failures retain their usual terminal
semantics.
"""

import numpy as np

from envs.docking3dof.core import Docking3DoFEnv
from envs.docking3dof.gym_env import Docking3DoFGymEnv


class ClosingSafetyExplorationCoreEnv(Docking3DoFEnv):
    """Closing core that records hard safety violations without terminating."""

    def __init__(self, *args, phase="closing", **kwargs):
        if phase != "closing":
            raise ValueError(
                "ClosingSafetyExplorationCoreEnv is restricted to phase='closing'."
            )
        super().__init__(*args, phase=phase, **kwargs)

    def check_done(self, state):
        """Apply normal checks, then suppress only a safety-violation boundary."""
        done, info = super().check_done(state)
        safety_violation = (
            info.get("failure_reason") == "safety_violation"
            or info.get("termination_reason") == "safety_violation"
        )

        info["safety_termination_disabled"] = True
        info["safety_violation_observed"] = bool(safety_violation)
        info["safety_violation_ignored_for_termination"] = False

        if not safety_violation:
            return done, info

        # Retain the original event as diagnostics, but it must not be exposed
        # to Gym/SB3 as a terminal failure in this experiment.
        info["observed_failure_reason"] = info.pop("failure_reason", None)
        info["failure"] = False
        info["safety_violation_ignored_for_termination"] = True

        # The standard core checks safety before task completion.  Once safety
        # termination is disabled, P3 and timeout must therefore be rechecked.
        if self.is_at_P3_target(state):
            info["success"] = True
            info["terminal_set"] = "P3_target"
            info["termination_reason"] = "success_P3_target"
            info["success_after_safety_violation"] = True
            return True, info

        if self.time >= self.max_time:
            info["time_limit"] = True
            info["termination_reason"] = "timeout"
            return True, info

        info["termination_reason"] = None
        return False, info


class ClosingSafetyExplorationGymEnv(Docking3DoFGymEnv):
    """Gym wrapper with a P3-reaching reward and non-terminal unsafe states."""

    reward_scheme = "p3_reach_v1"
    precision_beta = 4.0
    precision_gate_distance_m = 100.0
    precision_gate_width_m = 20.0

    # Kept explicit so the experiment is reproducible from its run config.
    reward_weights = {
        "position_state": 0.005,
        "position_progress": 100.0,
        "lateral_state": 0.02,
        "velocity_state": 2.0,
        "attitude_state": 5.0,
        "rate_state": 0.5,
        "precision_state": 5.0,
        "precision_progress": 100.0,
        "action": 0.01,
        "time": 0.01,
        "warning": 10.0,
        "unsafe": 1000.0,
        "success": 1000.0,
        "timeout": 200.0,
    }

    def __init__(
        self,
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=None,
        closing_reward_version="v15",
    ):
        if phase != "closing":
            raise ValueError(
                "ClosingSafetyExplorationGymEnv is restricted to phase='closing'."
            )

        super().__init__(
            phase=phase,
            mode=mode,
            dt=dt,
            max_time=max_time,
            integrator=integrator,
            normalize_action=normalize_action,
            seed=seed,
            closing_reward_version=closing_reward_version,
        )

        # Replace only this wrapper's core.  The ordinary Gym environment still
        # constructs Docking3DoFEnv and retains its original safety termination.
        self.core_env = ClosingSafetyExplorationCoreEnv(
            dt=dt,
            max_time=self.max_time,
            integrator=integrator,
            phase="closing",
            seed=seed,
        )

    def _p3_reward_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        ratios = np.abs(error) / tolerance
        log_ratios = np.log1p(ratios)
        maximum = float(np.max(log_ratios))
        smooth_worst = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(
                        self.precision_beta * (log_ratios - maximum)
                    )
                )
            )
            / self.precision_beta
        )
        precision_cost = 0.5 * float(np.mean(log_ratios)) + 0.5 * smooth_worst
        position_error = float(np.linalg.norm(error[:2]))
        gate_argument = np.clip(
            (position_error - self.precision_gate_distance_m)
            / self.precision_gate_width_m,
            -60.0,
            60.0,
        )
        precision_gate = float(1.0 / (1.0 + np.exp(gate_argument)))
        return {
            "error": error,
            "ratios": ratios,
            "position_error": position_error,
            "lateral_error": float(abs(error[1])),
            "velocity_error": float(np.linalg.norm(error[2:4])),
            "attitude_error": float(abs(error[4])),
            "rate_error": float(abs(error[5])),
            "precision_cost": precision_cost,
            "precision_smooth_worst": smooth_worst,
            "precision_gate": precision_gate,
        }

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        reward_state = self._p3_reward_state(self.core_env.state)
        self._previous_p3_position_error = reward_state["position_error"]
        self._previous_p3_precision_cost = reward_state["precision_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward_state = self._p3_reward_state(state)
        position_progress = (
            self._previous_p3_position_error - reward_state["position_error"]
        )
        precision_progress = (
            self._previous_p3_precision_cost - reward_state["precision_cost"]
        )
        self._previous_p3_position_error = reward_state["position_error"]
        self._previous_p3_precision_cost = reward_state["precision_cost"]

        action_scale = np.asarray(
            [
                self.core_env.ax_max,
                self.core_env.ay_max,
                self.core_env.alphaz_max,
            ],
            dtype=np.float64,
        )
        action_norm = float(
            np.linalg.norm(np.asarray(physical_action, dtype=np.float64) / action_scale)
        )

        weights = self.reward_weights
        terms = {
            "reward_position_state": (
                -weights["position_state"] * reward_state["position_error"]
            ),
            "reward_position_progress": (
                weights["position_progress"] * position_progress
            ),
            "reward_lateral_state": (
                -weights["lateral_state"] * reward_state["lateral_error"]
            ),
            "reward_velocity_state": (
                -weights["velocity_state"] * reward_state["velocity_error"]
            ),
            "reward_attitude_state": (
                -weights["attitude_state"] * reward_state["attitude_error"]
            ),
            "reward_rate_state": (
                -weights["rate_state"] * reward_state["rate_error"]
            ),
            "reward_precision_state": (
                -weights["precision_state"]
                * reward_state["precision_gate"]
                * reward_state["precision_cost"]
            ),
            "reward_precision_progress": (
                weights["precision_progress"]
                * reward_state["precision_gate"]
                * precision_progress
            ),
            "reward_action": -weights["action"] * action_norm,
            "reward_time": -weights["time"],
            "reward_warning": (
                -weights["warning"] if info.get("warning", False) else 0.0
            ),
            "reward_failure": (
                -weights["unsafe"]
                if info.get("safety_violation_observed", False)
                else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": (
                -weights["timeout"] if truncated else 0.0
            ),
        }
        reward = float(sum(terms.values()))
        ratios = reward_state["ratios"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": bool(
                info.get("safety_violation_observed", False)
            ),
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "position_error": reward_state["position_error"],
            "lateral_error": reward_state["lateral_error"],
            "velocity_error": reward_state["velocity_error"],
            "attitude_error": reward_state["attitude_error"],
            "rate_error": reward_state["rate_error"],
            "position_progress": float(position_progress),
            "precision_cost": reward_state["precision_cost"],
            "precision_smooth_worst": reward_state["precision_smooth_worst"],
            "precision_progress": float(precision_progress),
            "precision_gate": reward_state["precision_gate"],
            "p3_axial_ratio": float(ratios[0]),
            "p3_lateral_ratio": float(ratios[1]),
            "p3_vx_ratio": float(ratios[2]),
            "p3_vy_ratio": float(ratios[3]),
            "p3_attitude_ratio": float(ratios[4]),
            "p3_rate_ratio": float(ratios[5]),
            "p3_max_target_ratio": float(np.max(ratios)),
            "action_norm": action_norm,
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        if "capture_ratios" in reward_state:
            capture_ratios = reward_state["capture_ratios"]
            components.update(
                {
                    "capture_axial_position_ratio": float(capture_ratios[0]),
                    "capture_lateral_position_ratio": float(capture_ratios[1]),
                    "capture_axial_velocity_ratio": float(capture_ratios[2]),
                    "capture_lateral_velocity_ratio": float(capture_ratios[3]),
                    "capture_attitude_ratio": float(capture_ratios[4]),
                    "capture_rate_ratio": float(capture_ratios[5]),
                    "capture_max_ratio": float(np.max(capture_ratios)),
                }
            )
        return reward, components


class ClosingSafetyExplorationV2GymEnv(ClosingSafetyExplorationGymEnv):
    """Earlier terminal-capture shaping based on physically meaningful scales."""

    reward_scheme = "p3_reach_v2"
    precision_gate_distance_m = 400.0
    precision_gate_width_m = 100.0
    capture_scales = np.array(
        [
            100.0,
            20.0,
            0.05,
            0.02,
            np.deg2rad(10.0),
            np.deg2rad(0.1),
        ],
        dtype=np.float64,
    )
    reward_weights = {
        "position_state": 0.005,
        "position_progress": 100.0,
        "lateral_state": 0.05,
        "lateral_progress": 30.0,
        "velocity_state": 3.0,
        "velocity_progress": 30.0,
        "attitude_state": 10.0,
        "attitude_progress": 30.0,
        "rate_state": 1.0,
        "precision_state": 10.0,
        "precision_progress": 100.0,
        "action": 0.01,
        "time": 0.01,
        "warning": 10.0,
        "unsafe": 1000.0,
        "success": 1000.0,
        "timeout": 200.0,
    }

    def _p3_reward_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        exact_tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        exact_ratios = np.abs(error) / exact_tolerance
        capture_ratios = np.abs(error) / self.capture_scales
        log_capture_ratios = np.log1p(capture_ratios)
        maximum = float(np.max(log_capture_ratios))
        smooth_worst = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(
                        self.precision_beta
                        * (log_capture_ratios - maximum)
                    )
                )
            )
            / self.precision_beta
        )
        precision_cost = (
            0.5 * float(np.mean(log_capture_ratios)) + 0.5 * smooth_worst
        )
        position_error = float(np.linalg.norm(error[:2]))
        gate_argument = np.clip(
            (position_error - self.precision_gate_distance_m)
            / self.precision_gate_width_m,
            -60.0,
            60.0,
        )
        return {
            "error": error,
            "ratios": exact_ratios,
            "capture_ratios": capture_ratios,
            "position_error": position_error,
            "lateral_error": float(abs(error[1])),
            "velocity_error": float(np.linalg.norm(error[2:4])),
            "attitude_error": float(abs(error[4])),
            "rate_error": float(abs(error[5])),
            "precision_cost": precision_cost,
            "precision_smooth_worst": smooth_worst,
            "precision_gate": float(1.0 / (1.0 + np.exp(gate_argument))),
        }

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        reward_state = self._p3_reward_state(self.core_env.state)
        self._previous_p3_lateral_error = reward_state["lateral_error"]
        self._previous_p3_velocity_error = reward_state["velocity_error"]
        self._previous_p3_attitude_error = reward_state["attitude_error"]
        return observation, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        previous_lateral_error = self._previous_p3_lateral_error
        previous_velocity_error = self._previous_p3_velocity_error
        previous_attitude_error = self._previous_p3_attitude_error

        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        lateral_progress = previous_lateral_error - components["lateral_error"]
        velocity_progress = previous_velocity_error - components["velocity_error"]
        attitude_progress = previous_attitude_error - components["attitude_error"]
        self._previous_p3_lateral_error = components["lateral_error"]
        self._previous_p3_velocity_error = components["velocity_error"]
        self._previous_p3_attitude_error = components["attitude_error"]

        gate = components["precision_gate"]
        extra_terms = {
            "reward_lateral_progress": (
                self.reward_weights["lateral_progress"] * lateral_progress
            ),
            "reward_velocity_progress": (
                self.reward_weights["velocity_progress"]
                * gate
                * velocity_progress
            ),
            "reward_attitude_progress": (
                self.reward_weights["attitude_progress"]
                * gate
                * attitude_progress
            ),
        }
        reward += float(sum(extra_terms.values()))
        components.update(
            {
                "capture_scale_axial_position_m": float(self.capture_scales[0]),
                "capture_scale_lateral_position_m": float(self.capture_scales[1]),
                "capture_scale_axial_velocity_mps": float(self.capture_scales[2]),
                "capture_scale_lateral_velocity_mps": float(self.capture_scales[3]),
                "capture_scale_attitude_deg": float(
                    np.rad2deg(self.capture_scales[4])
                ),
                "capture_scale_rate_degs": float(
                    np.rad2deg(self.capture_scales[5])
                ),
                "lateral_progress": float(lateral_progress),
                "velocity_progress": float(velocity_progress),
                "attitude_progress": float(attitude_progress),
                **{key: float(value) for key, value in extra_terms.items()},
                "reward_total": float(reward),
            }
        )
        return float(reward), components


class ClosingSafetyExplorationV3GymEnv(ClosingSafetyExplorationGymEnv):
    """Goal-progress reward without a prescribed closing schedule.

    Position, velocity, attitude, and rate progress are active throughout the
    closing phase. Velocity safety shaping is exactly zero in the safe set and
    rises smoothly only through the simulator's warning bands.
    """

    reward_scheme = "p3_reach_v3"
    reward_weights = {
        "position_progress": 20.0,
        "velocity_progress": 50.0,
        "attitude_progress": 100.0,
        "rate_progress": 20.0,
        "position_state": 0.002,
        "velocity_state": 0.5,
        "attitude_state": 5.0,
        "rate_state": 0.1,
        "total_speed_warning": 20.0,
        "axial_speed_warning": 20.0,
        "geometry_warning": 10.0,
        "action": 0.01,
        "time": 0.01,
        "unsafe": 1000.0,
        "success": 1000.0,
        "timeout": 200.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        reward_state = self._goal_state(self.core_env.state)
        self._previous_goal_position_error = reward_state["position_error"]
        self._previous_goal_velocity_error = reward_state["velocity_error"]
        self._previous_goal_attitude_error = reward_state["attitude_error"]
        self._previous_goal_rate_error = reward_state["rate_error"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _goal_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        return {
            "error": error,
            "ratios": np.abs(error) / tolerance,
            "position_error": float(np.linalg.norm(error[:2])),
            "velocity_error": float(np.linalg.norm(error[2:4])),
            "attitude_error": float(abs(error[4])),
            "rate_error": float(abs(error[5])),
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        goal = self._goal_state(state)
        position_progress = (
            self._previous_goal_position_error - goal["position_error"]
        )
        velocity_progress = (
            self._previous_goal_velocity_error - goal["velocity_error"]
        )
        attitude_progress = (
            self._previous_goal_attitude_error - goal["attitude_error"]
        )
        rate_progress = self._previous_goal_rate_error - goal["rate_error"]
        self._previous_goal_position_error = goal["position_error"]
        self._previous_goal_velocity_error = goal["velocity_error"]
        self._previous_goal_attitude_error = goal["attitude_error"]
        self._previous_goal_rate_error = goal["rate_error"]

        q = self.core_env.get_derived_quantities(state)
        total_speed_warning_depth = float(
            np.clip(
                (q["port_speed"] - self.core_env.vp_safe_close)
                / (
                    self.core_env.vp_unsafe_close
                    - self.core_env.vp_safe_close
                ),
                0.0,
                1.0,
            )
        )
        axial_speed_warning_depth = float(
            np.clip(
                (abs(q["s_dot"]) - self.core_env.sdot_safe_close)
                / (
                    self.core_env.sdot_unsafe_close
                    - self.core_env.sdot_safe_close
                ),
                0.0,
                1.0,
            )
        )
        action_scale = np.asarray(
            [
                self.core_env.ax_max,
                self.core_env.ay_max,
                self.core_env.alphaz_max,
            ],
            dtype=np.float64,
        )
        action_norm = float(
            np.linalg.norm(np.asarray(physical_action, dtype=np.float64) / action_scale)
        )

        weights = self.reward_weights
        geometry_warning = bool(info.get("geometry", {}).get("warn", False))
        terms = {
            "reward_position_progress": (
                weights["position_progress"] * position_progress
            ),
            "reward_velocity_progress": (
                weights["velocity_progress"] * velocity_progress
            ),
            "reward_attitude_progress": (
                weights["attitude_progress"] * attitude_progress
            ),
            "reward_rate_progress": weights["rate_progress"] * rate_progress,
            "reward_position_state": (
                -weights["position_state"] * goal["position_error"]
            ),
            "reward_velocity_state": (
                -weights["velocity_state"] * goal["velocity_error"]
            ),
            "reward_attitude_state": (
                -weights["attitude_state"] * goal["attitude_error"]
            ),
            "reward_rate_state": -weights["rate_state"] * goal["rate_error"],
            "reward_total_speed_warning": (
                -weights["total_speed_warning"]
                * total_speed_warning_depth**2
            ),
            "reward_axial_speed_warning": (
                -weights["axial_speed_warning"]
                * axial_speed_warning_depth**2
            ),
            "reward_geometry_warning": (
                -weights["geometry_warning"] if geometry_warning else 0.0
            ),
            "reward_action": -weights["action"] * action_norm,
            "reward_time": -weights["time"],
            "reward_failure": (
                -weights["unsafe"]
                if info.get("safety_violation_observed", False)
                else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        ratios = goal["ratios"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": bool(
                info.get("safety_violation_observed", False)
            ),
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "position_error": goal["position_error"],
            "velocity_error": goal["velocity_error"],
            "attitude_error": goal["attitude_error"],
            "rate_error": goal["rate_error"],
            "position_progress": float(position_progress),
            "velocity_progress": float(velocity_progress),
            "attitude_progress": float(attitude_progress),
            "rate_progress": float(rate_progress),
            "total_speed_warning_depth": total_speed_warning_depth,
            "axial_speed_warning_depth": axial_speed_warning_depth,
            "geometry_warning": geometry_warning,
            "p3_axial_ratio": float(ratios[0]),
            "p3_lateral_ratio": float(ratios[1]),
            "p3_vx_ratio": float(ratios[2]),
            "p3_vy_ratio": float(ratios[3]),
            "p3_attitude_ratio": float(ratios[4]),
            "p3_rate_ratio": float(ratios[5]),
            "p3_max_target_ratio": float(np.max(ratios)),
            "action_norm": action_norm,
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV4GymEnv(ClosingSafetyExplorationGymEnv):
    """Balanced component-wise P3 reward with no prescribed trajectory.

    Each physical error is mapped through ``log1p(abs(error) / scale)`` before
    it contributes to the reward.  The broad characteristic scales balance
    metres, metres per second, radians, and radians per second; they are not
    CBF limits, success tolerances, waypoints, or desired state schedules.
    """

    reward_scheme = "p3_reach_v4"
    goal_component_names = ("axial", "lateral", "vx", "vy", "attitude", "rate")
    goal_scales = np.array(
        [
            100.0,
            20.0,
            0.05,
            0.02,
            np.deg2rad(10.0),
            np.deg2rad(0.1),
        ],
        dtype=np.float64,
    )
    reward_weights = {
        "axial_progress": 5000.0,
        "lateral_progress": 7500.0,
        "vx_progress": 5000.0,
        "vy_progress": 5000.0,
        "attitude_progress": 5000.0,
        "rate_progress": 2000.0,
        "axial_state": 0.25,
        "lateral_state": 1.0,
        "vx_state": 0.75,
        "vy_state": 1.0,
        "attitude_state": 0.75,
        "rate_state": 2.0,
        "total_speed_warning": 20.0,
        "axial_speed_warning": 20.0,
        "geometry_warning": 1.0,
        "action": 0.01,
        "time": 0.01,
        "unsafe": 1000.0,
        "success": 5000.0,
        "timeout": 200.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._balanced_goal_state(self.core_env.state)
        self._previous_goal_potentials = goal["potentials"].copy()
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _balanced_goal_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        absolute_error = np.abs(error)
        exact_tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        scaled_error = absolute_error / self.goal_scales
        return {
            "error": error,
            "absolute_error": absolute_error,
            "exact_ratios": absolute_error / exact_tolerance,
            "scaled_error": scaled_error,
            "potentials": np.log1p(scaled_error),
            "position_error": float(np.linalg.norm(error[:2])),
            "velocity_error": float(np.linalg.norm(error[2:4])),
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        goal = self._balanced_goal_state(state)
        progress = self._previous_goal_potentials - goal["potentials"]
        self._previous_goal_potentials = goal["potentials"].copy()

        q = self.core_env.get_derived_quantities(state)
        total_speed_warning_depth = float(
            np.clip(
                (q["port_speed"] - self.core_env.vp_safe_close)
                / (
                    self.core_env.vp_unsafe_close
                    - self.core_env.vp_safe_close
                ),
                0.0,
                1.0,
            )
        )
        axial_speed_warning_depth = float(
            np.clip(
                (abs(q["s_dot"]) - self.core_env.sdot_safe_close)
                / (
                    self.core_env.sdot_unsafe_close
                    - self.core_env.sdot_safe_close
                ),
                0.0,
                1.0,
            )
        )

        geometry_warning = bool(info.get("geometry", {}).get("warn", False))
        cone_radius = self.core_env.cone_slope * max(float(q["s"]), 1e-6)
        cone_excess_ratio = max(0.0, float(q["rho"]) / cone_radius - 1.0)
        geometry_warning_depth = (
            float(np.clip(cone_excess_ratio, 0.0, 1.0))
            if geometry_warning
            else 0.0
        )

        action_scale = np.asarray(
            [
                self.core_env.ax_max,
                self.core_env.ay_max,
                self.core_env.alphaz_max,
            ],
            dtype=np.float64,
        )
        action_norm = float(
            np.linalg.norm(np.asarray(physical_action, dtype=np.float64) / action_scale)
        )

        weights = self.reward_weights
        potentials = goal["potentials"]
        terms = {
            "reward_axial_progress": weights["axial_progress"] * progress[0],
            "reward_lateral_progress": (
                weights["lateral_progress"] * progress[1]
            ),
            "reward_vx_progress": weights["vx_progress"] * progress[2],
            "reward_vy_progress": weights["vy_progress"] * progress[3],
            "reward_attitude_progress": (
                weights["attitude_progress"] * progress[4]
            ),
            "reward_rate_progress": weights["rate_progress"] * progress[5],
            "reward_axial_state": -weights["axial_state"] * potentials[0],
            "reward_lateral_state": -weights["lateral_state"] * potentials[1],
            "reward_vx_state": -weights["vx_state"] * potentials[2],
            "reward_vy_state": -weights["vy_state"] * potentials[3],
            "reward_attitude_state": (
                -weights["attitude_state"] * potentials[4]
            ),
            "reward_rate_state": -weights["rate_state"] * potentials[5],
            "reward_total_speed_warning": (
                -weights["total_speed_warning"]
                * total_speed_warning_depth**2
            ),
            "reward_axial_speed_warning": (
                -weights["axial_speed_warning"]
                * axial_speed_warning_depth**2
            ),
            "reward_geometry_warning": (
                -weights["geometry_warning"] * geometry_warning_depth**2
            ),
            "reward_action": -weights["action"] * action_norm,
            "reward_time": -weights["time"],
            "reward_failure": (
                -weights["unsafe"]
                if info.get("safety_violation_observed", False)
                else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        exact_ratios = goal["exact_ratios"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": bool(
                info.get("safety_violation_observed", False)
            ),
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "position_error": goal["position_error"],
            "lateral_error": float(goal["absolute_error"][1]),
            "velocity_error": goal["velocity_error"],
            "attitude_error": float(goal["absolute_error"][4]),
            "rate_error": float(goal["absolute_error"][5]),
            "total_speed_warning_depth": total_speed_warning_depth,
            "axial_speed_warning_depth": axial_speed_warning_depth,
            "geometry_warning": geometry_warning,
            "geometry_warning_depth": geometry_warning_depth,
            "cone_excess_ratio": float(cone_excess_ratio),
            "action_norm": action_norm,
            "p3_max_target_ratio": float(np.max(exact_ratios)),
            **{
                f"goal_scale_{name}": float(scale)
                for name, scale in zip(self.goal_component_names, self.goal_scales)
            },
            **{
                f"goal_potential_{name}": float(value)
                for name, value in zip(self.goal_component_names, potentials)
            },
            **{
                f"goal_progress_{name}": float(value)
                for name, value in zip(self.goal_component_names, progress)
            },
            **{
                f"p3_{name}_ratio": float(value)
                for name, value in zip(self.goal_component_names, exact_ratios)
            },
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV5GymEnv(ClosingSafetyExplorationGymEnv):
    """Far-field translation plus smoothly activated terminal capture.

    Axial/lateral P3 progress and attitude/rate stabilization are active from
    P2.  Only matching the sampled P3 terminal velocity is distance-gated, so
    the far-field policy is not rewarded for slowing to terminal velocity
    before it has closed the separation.  No desired velocity profile,
    waypoint, braking point, or reference action is supplied.
    """

    reward_scheme = "p3_reach_v5"
    terminal_velocity_gate_distance_m = 400.0
    terminal_velocity_gate_width_m = 100.0
    reward_weights = {
        "axial_progress": 40.0,
        "lateral_progress": 80.0,
        "attitude_progress": 100.0,
        "rate_progress": 500.0,
        "terminal_velocity_progress": 100.0,
        "axial_state": 0.004,
        "lateral_state": 0.05,
        "attitude_state": 10.0,
        "rate_state": 500.0,
        "terminal_velocity_state": 5.0,
        "total_speed_warning": 20.0,
        "axial_speed_warning": 20.0,
        "geometry_warning": 1.0,
        "action": 0.01,
        "time": 0.01,
        "unsafe": 1000.0,
        "success": 5000.0,
        "timeout": 200.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._v5_goal_state(self.core_env.state)
        self._previous_axial_error = goal["axial_error"]
        self._previous_lateral_error = goal["lateral_error"]
        self._previous_velocity_error = goal["velocity_error"]
        self._previous_attitude_error = goal["attitude_error"]
        self._previous_rate_error = goal["rate_error"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v5_goal_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        absolute_error = np.abs(error)
        exact_tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        position_error = float(np.linalg.norm(error[:2]))
        gate_argument = np.clip(
            (position_error - self.terminal_velocity_gate_distance_m)
            / self.terminal_velocity_gate_width_m,
            -60.0,
            60.0,
        )
        return {
            "error": error,
            "absolute_error": absolute_error,
            "exact_ratios": absolute_error / exact_tolerance,
            "position_error": position_error,
            "axial_error": float(absolute_error[0]),
            "lateral_error": float(absolute_error[1]),
            "velocity_error": float(np.linalg.norm(error[2:4])),
            "attitude_error": float(absolute_error[4]),
            "rate_error": float(absolute_error[5]),
            "terminal_velocity_gate": float(
                1.0 / (1.0 + np.exp(gate_argument))
            ),
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        goal = self._v5_goal_state(state)
        axial_progress = self._previous_axial_error - goal["axial_error"]
        lateral_progress = self._previous_lateral_error - goal["lateral_error"]
        velocity_progress = self._previous_velocity_error - goal["velocity_error"]
        attitude_progress = (
            self._previous_attitude_error - goal["attitude_error"]
        )
        rate_progress = self._previous_rate_error - goal["rate_error"]
        self._previous_axial_error = goal["axial_error"]
        self._previous_lateral_error = goal["lateral_error"]
        self._previous_velocity_error = goal["velocity_error"]
        self._previous_attitude_error = goal["attitude_error"]
        self._previous_rate_error = goal["rate_error"]

        q = self.core_env.get_derived_quantities(state)
        total_speed_warning_depth = float(
            np.clip(
                (q["port_speed"] - self.core_env.vp_safe_close)
                / (
                    self.core_env.vp_unsafe_close
                    - self.core_env.vp_safe_close
                ),
                0.0,
                1.0,
            )
        )
        axial_speed_warning_depth = float(
            np.clip(
                (abs(q["s_dot"]) - self.core_env.sdot_safe_close)
                / (
                    self.core_env.sdot_unsafe_close
                    - self.core_env.sdot_safe_close
                ),
                0.0,
                1.0,
            )
        )

        geometry_warning = bool(info.get("geometry", {}).get("warn", False))
        cone_radius = self.core_env.cone_slope * max(float(q["s"]), 1e-6)
        cone_excess_ratio = max(0.0, float(q["rho"]) / cone_radius - 1.0)
        geometry_warning_depth = (
            float(np.clip(cone_excess_ratio, 0.0, 1.0))
            if geometry_warning
            else 0.0
        )

        action_scale = np.asarray(
            [
                self.core_env.ax_max,
                self.core_env.ay_max,
                self.core_env.alphaz_max,
            ],
            dtype=np.float64,
        )
        action_norm = float(
            np.linalg.norm(np.asarray(physical_action, dtype=np.float64) / action_scale)
        )

        weights = self.reward_weights
        velocity_gate = goal["terminal_velocity_gate"]
        terms = {
            "reward_axial_progress": weights["axial_progress"] * axial_progress,
            "reward_lateral_progress": (
                weights["lateral_progress"] * lateral_progress
            ),
            "reward_attitude_progress": (
                weights["attitude_progress"] * attitude_progress
            ),
            "reward_rate_progress": weights["rate_progress"] * rate_progress,
            "reward_terminal_velocity_progress": (
                weights["terminal_velocity_progress"]
                * velocity_gate
                * velocity_progress
            ),
            "reward_axial_state": -weights["axial_state"] * goal["axial_error"],
            "reward_lateral_state": (
                -weights["lateral_state"] * goal["lateral_error"]
            ),
            "reward_attitude_state": (
                -weights["attitude_state"] * goal["attitude_error"]
            ),
            "reward_rate_state": -weights["rate_state"] * goal["rate_error"],
            "reward_terminal_velocity_state": (
                -weights["terminal_velocity_state"]
                * velocity_gate
                * goal["velocity_error"]
            ),
            "reward_total_speed_warning": (
                -weights["total_speed_warning"]
                * total_speed_warning_depth**2
            ),
            "reward_axial_speed_warning": (
                -weights["axial_speed_warning"]
                * axial_speed_warning_depth**2
            ),
            "reward_geometry_warning": (
                -weights["geometry_warning"] * geometry_warning_depth**2
            ),
            "reward_action": -weights["action"] * action_norm,
            "reward_time": -weights["time"],
            "reward_failure": (
                -weights["unsafe"]
                if info.get("safety_violation_observed", False)
                else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        exact_ratios = goal["exact_ratios"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": bool(
                info.get("safety_violation_observed", False)
            ),
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "position_error": goal["position_error"],
            "axial_error": goal["axial_error"],
            "lateral_error": goal["lateral_error"],
            "velocity_error": goal["velocity_error"],
            "attitude_error": goal["attitude_error"],
            "rate_error": goal["rate_error"],
            "axial_progress": float(axial_progress),
            "lateral_progress": float(lateral_progress),
            "velocity_progress": float(velocity_progress),
            "attitude_progress": float(attitude_progress),
            "rate_progress": float(rate_progress),
            "terminal_velocity_gate": velocity_gate,
            "total_speed_warning_depth": total_speed_warning_depth,
            "axial_speed_warning_depth": axial_speed_warning_depth,
            "geometry_warning": geometry_warning,
            "geometry_warning_depth": geometry_warning_depth,
            "cone_excess_ratio": float(cone_excess_ratio),
            "action_norm": action_norm,
            "p3_axial_ratio": float(exact_ratios[0]),
            "p3_lateral_ratio": float(exact_ratios[1]),
            "p3_vx_ratio": float(exact_ratios[2]),
            "p3_vy_ratio": float(exact_ratios[3]),
            "p3_attitude_ratio": float(exact_ratios[4]),
            "p3_rate_ratio": float(exact_ratios[5]),
            "p3_max_target_ratio": float(np.max(exact_ratios)),
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV6GymEnv(ClosingSafetyExplorationGymEnv):
    """Tolerance-normalized smooth worst-component P3 reward.

    Every P3 state error is divided by its corresponding terminal tolerance.
    A numerically stable log-mean-exp then provides a smooth approximation of
    the largest normalized error.  Unlike the earlier experiment rewards,
    this objective has no distance gate, reference trajectory, or separately
    hand-tuned weights for quantities expressed in different physical units.

    Safety violations keep the experiment's non-terminal semantics.  Their
    penalty is therefore applied on every step whose resulting state is
    classified unsafe; it is not a terminal penalty.
    """

    reward_scheme = "p3_reach_v6"
    smooth_max_beta = 5.0
    progress_clip = 1.0
    state_cost_clip = 10.0
    goal_component_names = (
        "axial",
        "lateral",
        "vx",
        "vy",
        "attitude",
        "rate",
    )
    reward_weights = {
        "smoothmax_progress": 10.0,
        "smoothmax_state": 0.01,
        "action": 0.001,
        "time": 0.001,
        "unsafe": 100.0,
        "success": 100.0,
        "timeout": 0.0,
    }

    def reset(self, seed=None, options=None):
        # Call the shared wrapper directly so v1's reward-state reset is not
        # mixed into this isolated reward implementation.
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._v6_goal_state(self.core_env.state)
        self._previous_smooth_max_cost = goal["smooth_max_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v6_goal_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        ratios = np.abs(error) / tolerance

        # Subtracting the maximum prevents overflow. Using the mean rather
        # than the sum makes the metric exactly zero at the P3 target and
        # equal to c when all six normalized errors equal c.
        maximum = float(np.max(ratios))
        smooth_max_cost = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(self.smooth_max_beta * (ratios - maximum))
                )
            )
            / self.smooth_max_beta
        )
        return {
            "error": error,
            "ratios": ratios,
            "max_ratio": maximum,
            "smooth_max_cost": smooth_max_cost,
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        del physical_action, terminated
        goal = self._v6_goal_state(state)
        raw_progress = (
            self._previous_smooth_max_cost - goal["smooth_max_cost"]
        )
        clipped_progress = float(
            np.clip(raw_progress, -self.progress_clip, self.progress_clip)
        )
        self._previous_smooth_max_cost = goal["smooth_max_cost"]

        normalized_action = np.asarray(normalized_action, dtype=np.float64)
        action_norm_squared = float(np.sum(np.square(normalized_action)))
        clipped_state_cost = float(
            min(goal["smooth_max_cost"], self.state_cost_clip)
        )

        weights = self.reward_weights
        terms = {
            "reward_smoothmax_progress": (
                weights["smoothmax_progress"] * clipped_progress
            ),
            "reward_smoothmax_state": (
                -weights["smoothmax_state"] * clipped_state_cost
            ),
            "reward_action": -weights["action"] * action_norm_squared,
            "reward_time": -weights["time"],
            "reward_failure": (
                -weights["unsafe"]
                if info.get("safety_violation_observed", False)
                else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        ratios = goal["ratios"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": bool(
                info.get("safety_violation_observed", False)
            ),
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "smooth_max_beta": float(self.smooth_max_beta),
            "smooth_max_cost": goal["smooth_max_cost"],
            "smooth_max_progress_raw": float(raw_progress),
            "smooth_max_progress_clipped": clipped_progress,
            "smooth_max_state_cost_clipped": clipped_state_cost,
            "p3_max_target_ratio": goal["max_ratio"],
            "action_norm_squared": action_norm_squared,
            **{
                f"p3_{name}_ratio": float(value)
                for name, value in zip(self.goal_component_names, ratios)
            },
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV7GymEnv(ClosingSafetyExplorationGymEnv):
    """Globally active, balanced P3 capture reward.

    Exact P3-tolerance ratios are compressed with ``log1p`` before combining
    their mean and a smooth worst-component value.  The mean keeps gradients
    active for every P3 requirement, while the smooth maximum still emphasizes
    the current bottleneck.  Safety remains non-terminal and is penalized only
    when the trajectory enters the hard-unsafe set.
    """

    reward_scheme = "p3_reach_v7"
    smooth_max_beta = 2.0
    aggregate_mean_weight = 0.5
    aggregate_smooth_worst_weight = 0.5
    progress_clip = 0.2
    goal_component_names = (
        "axial",
        "lateral",
        "vx",
        "vy",
        "attitude",
        "rate",
    )
    reward_weights = {
        "goal_progress": 10.0,
        "goal_state": 0.05,
        "action": 0.001,
        "time": 0.001,
        "unsafe_entry": 25.0,
        "success": 100.0,
        "timeout": 0.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._v7_goal_state(self.core_env.state)
        self._previous_goal_cost = goal["goal_cost"]
        initial_monitor = self.core_env.monitor_safety(self.core_env.state)
        self._was_safety_unsafe = bool(initial_monitor["unsafe"])
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v7_goal_state(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("The closing P3 target is not initialized.")

        error = np.asarray(state - self.core_env.p3_target, dtype=np.float64)
        error[4] = self.core_env.wrap_to_pi(error[4])
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )
        ratios = np.abs(error) / tolerance
        log_components = np.log1p(ratios)

        maximum = float(np.max(log_components))
        smooth_worst = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(
                        self.smooth_max_beta * (log_components - maximum)
                    )
                )
            )
            / self.smooth_max_beta
        )
        component_mean = float(np.mean(log_components))
        goal_cost = float(
            self.aggregate_mean_weight * component_mean
            + self.aggregate_smooth_worst_weight * smooth_worst
        )
        return {
            "error": error,
            "ratios": ratios,
            "log_components": log_components,
            "max_ratio": float(np.max(ratios)),
            "component_mean": component_mean,
            "smooth_worst": smooth_worst,
            "goal_cost": goal_cost,
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        del physical_action, terminated
        goal = self._v7_goal_state(state)
        raw_progress = self._previous_goal_cost - goal["goal_cost"]
        clipped_progress = float(
            np.clip(raw_progress, -self.progress_clip, self.progress_clip)
        )
        self._previous_goal_cost = goal["goal_cost"]

        unsafe_now = bool(info.get("safety_violation_observed", False))
        unsafe_entry = bool(unsafe_now and not self._was_safety_unsafe)
        self._was_safety_unsafe = unsafe_now

        normalized_action = np.asarray(normalized_action, dtype=np.float64)
        action_norm_squared = float(np.sum(np.square(normalized_action)))

        weights = self.reward_weights
        terms = {
            "reward_goal_progress": weights["goal_progress"] * clipped_progress,
            "reward_goal_state": -weights["goal_state"] * goal["goal_cost"],
            "reward_action": -weights["action"] * action_norm_squared,
            "reward_time": -weights["time"],
            "reward_failure": (
                -weights["unsafe_entry"] if unsafe_entry else 0.0
            ),
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        ratios = goal["ratios"]
        log_components = goal["log_components"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": unsafe_now,
            "safety_unsafe_entry": unsafe_entry,
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "smooth_max_beta": float(self.smooth_max_beta),
            "goal_component_mean": goal["component_mean"],
            "goal_smooth_worst": goal["smooth_worst"],
            "goal_cost": goal["goal_cost"],
            "goal_progress_raw": float(raw_progress),
            "goal_progress_clipped": clipped_progress,
            "p3_max_target_ratio": goal["max_ratio"],
            "action_norm_squared": action_norm_squared,
            **{
                f"p3_{name}_ratio": float(value)
                for name, value in zip(self.goal_component_names, ratios)
            },
            **{
                f"p3_{name}_log_component": float(value)
                for name, value in zip(
                    self.goal_component_names,
                    log_components,
                )
            },
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV8GymEnv(ClosingSafetyExplorationV7GymEnv):
    """V7 goal metric with symmetric progress and unsafe occupancy cost.

    Progress uses the complete change in the log-compressed goal cost, so a
    large departure is penalized by the same rule that rewards an approach.
    Every hard-unsafe resulting state receives a small bounded penalty while
    safety violations retain their non-terminal experiment semantics.
    """

    reward_scheme = "p3_reach_v8"
    progress_clip = None
    reward_weights = {
        "goal_progress": 10.0,
        "goal_state": 0.05,
        "action": 0.001,
        "time": 0.001,
        "unsafe_step": 0.1,
        "success": 100.0,
        "timeout": 0.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._v8_goal_state(self.core_env.state)
        self._previous_goal_cost = goal["goal_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v8_goal_state(self, state):
        return self._v7_goal_state(state)

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        del physical_action, terminated
        goal = self._v8_goal_state(state)
        goal_progress = self._previous_goal_cost - goal["goal_cost"]
        self._previous_goal_cost = goal["goal_cost"]

        unsafe_now = bool(info.get("safety_violation_observed", False))
        normalized_action = np.asarray(normalized_action, dtype=np.float64)
        action_norm_squared = float(np.sum(np.square(normalized_action)))

        weights = self.reward_weights
        terms = {
            "reward_goal_progress": weights["goal_progress"] * goal_progress,
            "reward_goal_state": -weights["goal_state"] * goal["goal_cost"],
            "reward_action": -weights["action"] * action_norm_squared,
            "reward_time": -weights["time"],
            "reward_failure": -weights["unsafe_step"] if unsafe_now else 0.0,
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        ratios = goal["ratios"]
        log_components = goal["log_components"]
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": unsafe_now,
            "unsafe_occupancy_penalty_active": unsafe_now,
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "smooth_max_beta": float(self.smooth_max_beta),
            "goal_component_mean": goal["component_mean"],
            "goal_smooth_worst": goal["smooth_worst"],
            "goal_cost": goal["goal_cost"],
            "goal_progress": float(goal_progress),
            "p3_max_target_ratio": goal["max_ratio"],
            "action_norm_squared": action_norm_squared,
            **{
                f"p3_{name}_ratio": float(value)
                for name, value in zip(self.goal_component_names, ratios)
            },
            **{
                f"p3_{name}_log_component": float(value)
                for name, value in zip(
                    self.goal_component_names,
                    log_components,
                )
            },
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV9GymEnv(ClosingSafetyExplorationV8GymEnv):
    """Global guidance with distance-gated terminal-velocity capture.

    Position and attitude/rate objectives remain active throughout closing.
    Exact P3 velocity matching is activated smoothly near P3, preventing the
    far-field policy from being rewarded for adopting terminal velocity while
    it is still more than a kilometre away.  The gate multiplies the already
    computed velocity progress instead of being included inside its temporal
    difference.
    """

    reward_scheme = "p3_reach_v9"
    terminal_velocity_gate_distance_m = 400.0
    terminal_velocity_gate_width_m = 100.0
    position_group_weight = 0.5
    attitude_group_weight = 0.25
    velocity_group_weight = 0.25

    def reset(self, seed=None, options=None):
        observation, info = Docking3DoFGymEnv.reset(
            self,
            seed=seed,
            options=options,
        )
        goal = self._v9_goal_state(self.core_env.state)
        self._previous_position_cost = goal["position_cost"]
        self._previous_attitude_cost = goal["attitude_cost"]
        self._previous_velocity_cost = goal["velocity_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v9_group_cost(self, log_components):
        log_components = np.asarray(log_components, dtype=np.float64)
        maximum = float(np.max(log_components))
        smooth_worst = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(
                        self.smooth_max_beta * (log_components - maximum)
                    )
                )
            )
            / self.smooth_max_beta
        )
        component_mean = float(np.mean(log_components))
        return float(
            self.aggregate_mean_weight * component_mean
            + self.aggregate_smooth_worst_weight * smooth_worst
        )

    def _v9_goal_state(self, state):
        goal = self._v7_goal_state(state)
        error = goal["error"]
        position_error = float(np.linalg.norm(error[:2]))
        gate_argument = np.clip(
            (position_error - self.terminal_velocity_gate_distance_m)
            / self.terminal_velocity_gate_width_m,
            -60.0,
            60.0,
        )
        log_components = goal["log_components"]
        goal.update(
            {
                "position_error": position_error,
                "terminal_velocity_gate": float(
                    1.0 / (1.0 + np.exp(gate_argument))
                ),
                "position_cost": self._v9_group_cost(log_components[:2]),
                "velocity_cost": self._v9_group_cost(log_components[2:4]),
                "attitude_cost": self._v9_group_cost(log_components[4:6]),
            }
        )
        return goal

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        del physical_action, terminated
        goal = self._v9_goal_state(state)
        position_progress = (
            self._previous_position_cost - goal["position_cost"]
        )
        attitude_progress = (
            self._previous_attitude_cost - goal["attitude_cost"]
        )
        velocity_progress = (
            self._previous_velocity_cost - goal["velocity_cost"]
        )
        self._previous_position_cost = goal["position_cost"]
        self._previous_attitude_cost = goal["attitude_cost"]
        self._previous_velocity_cost = goal["velocity_cost"]

        velocity_gate = goal["terminal_velocity_gate"]
        unsafe_now = bool(info.get("safety_violation_observed", False))
        normalized_action = np.asarray(normalized_action, dtype=np.float64)
        action_norm_squared = float(np.sum(np.square(normalized_action)))

        weights = self.reward_weights
        terms = {
            "reward_position_progress": (
                weights["goal_progress"]
                * self.position_group_weight
                * position_progress
            ),
            "reward_attitude_progress": (
                weights["goal_progress"]
                * self.attitude_group_weight
                * attitude_progress
            ),
            "reward_terminal_velocity_progress": (
                weights["goal_progress"]
                * self.velocity_group_weight
                * velocity_gate
                * velocity_progress
            ),
            "reward_position_state": (
                -weights["goal_state"]
                * self.position_group_weight
                * goal["position_cost"]
            ),
            "reward_attitude_state": (
                -weights["goal_state"]
                * self.attitude_group_weight
                * goal["attitude_cost"]
            ),
            "reward_terminal_velocity_state": (
                -weights["goal_state"]
                * self.velocity_group_weight
                * velocity_gate
                * goal["velocity_cost"]
            ),
            "reward_action": -weights["action"] * action_norm_squared,
            "reward_time": -weights["time"],
            "reward_failure": -weights["unsafe_step"] if unsafe_now else 0.0,
            "reward_success": (
                weights["success"] if info.get("success", False) else 0.0
            ),
            "reward_timeout": -weights["timeout"] if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        ratios = goal["ratios"]
        log_components = goal["log_components"]
        weighted_goal_cost = float(
            self.position_group_weight * goal["position_cost"]
            + self.attitude_group_weight * goal["attitude_cost"]
            + self.velocity_group_weight
            * velocity_gate
            * goal["velocity_cost"]
        )
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "shared_gym_reward_version_used": False,
            "safety_termination_disabled": True,
            "safety_violation_observed": unsafe_now,
            "unsafe_occupancy_penalty_active": unsafe_now,
            "safety_penalty_preserved": bool(terms["reward_failure"] != 0.0),
            "smooth_max_beta": float(self.smooth_max_beta),
            "position_error": goal["position_error"],
            "position_cost": goal["position_cost"],
            "velocity_cost": goal["velocity_cost"],
            "attitude_cost": goal["attitude_cost"],
            "weighted_goal_cost": weighted_goal_cost,
            "position_progress": float(position_progress),
            "velocity_progress": float(velocity_progress),
            "attitude_progress": float(attitude_progress),
            "terminal_velocity_gate": velocity_gate,
            "p3_max_target_ratio": goal["max_ratio"],
            "action_norm_squared": action_norm_squared,
            **{
                f"p3_{name}_ratio": float(value)
                for name, value in zip(self.goal_component_names, ratios)
            },
            **{
                f"p3_{name}_log_component": float(value)
                for name, value in zip(
                    self.goal_component_names,
                    log_components,
                )
            },
            **{key: float(value) for key, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class ClosingSafetyExplorationV10GymEnv(ClosingSafetyExplorationV9GymEnv):
    """V9 guidance with dense terminal capture and departure deterrence.

    V9 can approach P3 without ever observing its sparse success bonus.  V10
    therefore adds a smooth near-target cost based on the worst of all six P3
    tolerance ratios.  The cost is zero only at the complete P3 target, so it
    also trains velocity and alignment before the exact terminal set is hit.

    Two state-based costs discourage the failure modes seen in V9 rollouts:
    crossing beyond the axial P3 tolerance and moving radially away from P3.
    The fly-away term uses only the current P3-relative position and velocity,
    rather than a hidden best-distance history, keeping the reward Markov.
    """

    reward_scheme = "p3_reach_v10"
    near_target_gate_distance_m = 400.0
    near_target_gate_width_m = 100.0
    flyaway_speed_scale_mps = 0.01
    reward_weights = {
        **ClosingSafetyExplorationV9GymEnv.reward_weights,
        "near_target_capture": 0.25,
        "axial_overshoot": 1.0,
        "flyaway": 1.0,
    }

    def _v10_goal_state(self, state):
        goal = self._v9_goal_state(state)
        error = goal["error"]
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )

        gate_argument = np.clip(
            (goal["position_error"] - self.near_target_gate_distance_m)
            / self.near_target_gate_width_m,
            -60.0,
            60.0,
        )
        near_target_gate = float(1.0 / (1.0 + np.exp(gate_argument)))

        axial_overshoot_m = float(max(0.0, -error[0] - tolerance[0]))
        axial_overshoot_ratio = axial_overshoot_m / tolerance[0]
        axial_overshoot_cost = float(np.log1p(axial_overshoot_ratio))

        if goal["position_error"] > 1e-12:
            radial_velocity_mps = float(
                np.dot(error[:2], error[2:4]) / goal["position_error"]
            )
        else:
            radial_velocity_mps = 0.0
        flyaway_speed_mps = float(max(0.0, radial_velocity_mps))
        flyaway_cost = float(
            near_target_gate
            * np.log1p(flyaway_speed_mps / self.flyaway_speed_scale_mps)
        )

        goal.update(
            {
                "near_target_gate": near_target_gate,
                "near_target_capture_cost": goal["smooth_worst"],
                "axial_overshoot_m": axial_overshoot_m,
                "axial_overshoot_ratio": axial_overshoot_ratio,
                "axial_overshoot_cost": axial_overshoot_cost,
                "radial_velocity_mps": radial_velocity_mps,
                "flyaway_speed_mps": flyaway_speed_mps,
                "flyaway_cost": flyaway_cost,
            }
        )
        return goal

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        goal = self._v10_goal_state(state)
        weights = self.reward_weights
        v10_terms = {
            "reward_near_target_capture": (
                -weights["near_target_capture"]
                * goal["near_target_gate"]
                * goal["near_target_capture_cost"]
            ),
            "reward_axial_overshoot": (
                -weights["axial_overshoot"] * goal["axial_overshoot_cost"]
            ),
            "reward_flyaway": -weights["flyaway"] * goal["flyaway_cost"],
        }
        reward = float(reward + sum(v10_terms.values()))
        components.update(
            {
                "near_target_gate": goal["near_target_gate"],
                "near_target_capture_cost": goal["near_target_capture_cost"],
                "axial_overshoot_m": goal["axial_overshoot_m"],
                "axial_overshoot_ratio": goal["axial_overshoot_ratio"],
                "axial_overshoot_cost": goal["axial_overshoot_cost"],
                "radial_velocity_mps": goal["radial_velocity_mps"],
                "flyaway_speed_mps": goal["flyaway_speed_mps"],
                "flyaway_cost": goal["flyaway_cost"],
                **{key: float(value) for key, value in v10_terms.items()},
                "reward_total": reward,
            }
        )
        return reward, components


class ClosingSafetyExplorationV11GymEnv(ClosingSafetyExplorationV9GymEnv):
    """V9 guidance with bounded near-target capture progress.

    V10's persistent near-target and axial-overshoot occupancy costs made
    avoiding P3 more rewarding than attempting capture.  V11 removes both
    costs.  Instead, it rewards improvement and penalizes regression in the
    smooth worst P3 component only while near P3.  The symmetric progress term
    is clipped to the same per-step magnitude as the unsafe occupancy penalty.

    Positive P3-relative radial velocity remains an explicit fly-away signal,
    but its penalty is also bounded to the unsafe-step penalty magnitude.  The
    axial overshoot distance is retained solely as a rollout diagnostic.
    """

    reward_scheme = "p3_reach_v11"
    near_target_gate_distance_m = 400.0
    near_target_gate_width_m = 100.0
    capture_progress_clip = 0.1
    flyaway_speed_scale_mps = 0.01
    flyaway_penalty_clip = 0.1
    reward_weights = {
        **ClosingSafetyExplorationV9GymEnv.reward_weights,
        "capture_progress": 1.0,
        "flyaway": 1.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        goal = self._v11_goal_state(self.core_env.state)
        self._previous_capture_cost = goal["capture_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v11_goal_state(self, state):
        goal = self._v9_goal_state(state)
        error = goal["error"]
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )

        gate_argument = np.clip(
            (goal["position_error"] - self.near_target_gate_distance_m)
            / self.near_target_gate_width_m,
            -60.0,
            60.0,
        )
        near_target_gate = float(1.0 / (1.0 + np.exp(gate_argument)))

        axial_overshoot_m = float(max(0.0, -error[0] - tolerance[0]))
        if goal["position_error"] > 1e-12:
            radial_velocity_mps = float(
                np.dot(error[:2], error[2:4]) / goal["position_error"]
            )
        else:
            radial_velocity_mps = 0.0
        flyaway_speed_mps = float(max(0.0, radial_velocity_mps))
        flyaway_cost_raw = float(
            near_target_gate
            * np.log1p(flyaway_speed_mps / self.flyaway_speed_scale_mps)
        )
        flyaway_cost_clipped = float(
            min(flyaway_cost_raw, self.flyaway_penalty_clip)
        )

        goal.update(
            {
                "near_target_gate": near_target_gate,
                "capture_cost": goal["smooth_worst"],
                "axial_overshoot_m": axial_overshoot_m,
                "radial_velocity_mps": radial_velocity_mps,
                "flyaway_speed_mps": flyaway_speed_mps,
                "flyaway_cost_raw": flyaway_cost_raw,
                "flyaway_cost_clipped": flyaway_cost_clipped,
            }
        )
        return goal

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        goal = self._v11_goal_state(state)
        capture_progress_raw = float(
            self._previous_capture_cost - goal["capture_cost"]
        )
        capture_progress_clipped = float(
            np.clip(
                capture_progress_raw,
                -self.capture_progress_clip,
                self.capture_progress_clip,
            )
        )
        self._previous_capture_cost = goal["capture_cost"]

        weights = self.reward_weights
        v11_terms = {
            "reward_capture_progress": (
                weights["capture_progress"]
                * goal["near_target_gate"]
                * capture_progress_clipped
            ),
            "reward_flyaway": (
                -weights["flyaway"] * goal["flyaway_cost_clipped"]
            ),
        }
        reward = float(reward + sum(v11_terms.values()))
        components.update(
            {
                "near_target_gate": goal["near_target_gate"],
                "capture_cost": goal["capture_cost"],
                "capture_progress_raw": capture_progress_raw,
                "capture_progress_clipped": capture_progress_clipped,
                "capture_progress_clip": float(self.capture_progress_clip),
                "near_target_occupancy_penalty_active": False,
                "axial_overshoot_m": goal["axial_overshoot_m"],
                "axial_overshoot_penalty_active": False,
                "radial_velocity_mps": goal["radial_velocity_mps"],
                "flyaway_speed_mps": goal["flyaway_speed_mps"],
                "flyaway_cost_raw": goal["flyaway_cost_raw"],
                "flyaway_cost_clipped": goal["flyaway_cost_clipped"],
                "flyaway_penalty_clip": float(self.flyaway_penalty_clip),
                **{key: float(value) for key, value in v11_terms.items()},
                "reward_total": reward,
            }
        )
        return reward, components


class ClosingSafetyExplorationV12GymEnv(ClosingSafetyExplorationV9GymEnv):
    """Stable v11 successor with balanced capture progress and early resets.

    Capture progress is computed from separate position, attitude/rate, and
    velocity group costs.  Their weighted combination is clipped before the
    near-target gate is applied, keeping the complete additional progress term
    in ``[-0.1, 0.1]``.  The fly-away penalty is independently capped at 0.05.

    An approach that has entered the near-target region is truncated, without
    an added reward penalty, only after it has lost substantial ground while
    moving away continuously.  This replaces long post-failure trajectories
    with fresh P2 episodes while preserving non-terminal safety exploration.
    """

    reward_scheme = "p3_reach_v12"
    position_group_weight = 0.4
    attitude_group_weight = 0.4
    velocity_group_weight = 0.2
    near_target_gate_distance_m = 400.0
    near_target_gate_width_m = 100.0
    capture_progress_clip = 0.1
    flyaway_speed_scale_mps = 0.01
    flyaway_penalty_clip = 0.05
    failed_approach_entry_distance_m = 400.0
    failed_approach_distance_growth_m = 200.0
    failed_approach_radial_speed_mps = 0.01
    failed_approach_patience_steps = 60
    reward_weights = {
        **ClosingSafetyExplorationV9GymEnv.reward_weights,
        "capture_progress": 1.0,
        "flyaway": 1.0,
    }

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        goal = self._v12_goal_state(self.core_env.state)
        self._previous_capture_position_cost = goal["position_cost"]
        self._previous_capture_attitude_cost = goal["attitude_cost"]
        self._previous_capture_velocity_cost = goal["velocity_cost"]
        self._entered_near_target = bool(
            goal["position_error"] <= self.failed_approach_entry_distance_m
        )
        self._minimum_position_error = goal["position_error"]
        self._failed_approach_flyaway_steps = 0
        info["reward_scheme"] = self.reward_scheme
        info["failed_approach_reset_enabled"] = True
        return observation, info

    def _v12_goal_state(self, state):
        goal = self._v9_goal_state(state)
        error = goal["error"]
        tolerance = np.maximum(
            np.asarray(self.core_env.p3_target_tolerance, dtype=np.float64),
            1e-12,
        )

        gate_argument = np.clip(
            (goal["position_error"] - self.near_target_gate_distance_m)
            / self.near_target_gate_width_m,
            -60.0,
            60.0,
        )
        near_target_gate = float(1.0 / (1.0 + np.exp(gate_argument)))

        axial_overshoot_m = float(max(0.0, -error[0] - tolerance[0]))
        if goal["position_error"] > 1e-12:
            radial_velocity_mps = float(
                np.dot(error[:2], error[2:4]) / goal["position_error"]
            )
        else:
            radial_velocity_mps = 0.0
        flyaway_speed_mps = float(max(0.0, radial_velocity_mps))
        flyaway_cost_raw = float(
            near_target_gate
            * np.log1p(flyaway_speed_mps / self.flyaway_speed_scale_mps)
        )
        flyaway_cost_clipped = float(
            min(flyaway_cost_raw, self.flyaway_penalty_clip)
        )

        goal.update(
            {
                "near_target_gate": near_target_gate,
                "axial_overshoot_m": axial_overshoot_m,
                "radial_velocity_mps": radial_velocity_mps,
                "flyaway_speed_mps": flyaway_speed_mps,
                "flyaway_cost_raw": flyaway_cost_raw,
                "flyaway_cost_clipped": flyaway_cost_clipped,
            }
        )
        return goal

    def _update_failed_approach_state(self, goal):
        position_error = float(goal["position_error"])
        self._minimum_position_error = min(
            self._minimum_position_error,
            position_error,
        )
        if position_error <= self.failed_approach_entry_distance_m:
            self._entered_near_target = True

        lost_ground = (
            position_error - self._minimum_position_error
            >= self.failed_approach_distance_growth_m
        )
        moving_away = (
            goal["radial_velocity_mps"]
            >= self.failed_approach_radial_speed_mps
        )
        if self._entered_near_target and lost_ground and moving_away:
            self._failed_approach_flyaway_steps += 1
        else:
            self._failed_approach_flyaway_steps = 0

        return bool(
            self._failed_approach_flyaway_steps
            >= self.failed_approach_patience_steps
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        goal = self._v12_goal_state(self.core_env.state)
        failed_approach_reset = self._update_failed_approach_state(goal)

        if not terminated and not truncated and failed_approach_reset:
            truncated = True
            info["failed_approach_reset"] = True
            info["termination_reason"] = "failed_approach_reset"
        else:
            info["failed_approach_reset"] = False

        info["entered_near_target"] = self._entered_near_target
        info["minimum_position_error_m"] = self._minimum_position_error
        info["failed_approach_flyaway_steps"] = (
            self._failed_approach_flyaway_steps
        )
        reward_components = info.get("reward_components", {})
        reward_components["failed_approach_reset"] = failed_approach_reset
        reward_components["failed_approach_reset_reward"] = 0.0
        return observation, reward, terminated, truncated, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        goal = self._v12_goal_state(state)
        position_progress = float(
            self._previous_capture_position_cost - goal["position_cost"]
        )
        attitude_progress = float(
            self._previous_capture_attitude_cost - goal["attitude_cost"]
        )
        velocity_progress = float(
            self._previous_capture_velocity_cost - goal["velocity_cost"]
        )
        self._previous_capture_position_cost = goal["position_cost"]
        self._previous_capture_attitude_cost = goal["attitude_cost"]
        self._previous_capture_velocity_cost = goal["velocity_cost"]

        balanced_progress_raw = float(
            self.position_group_weight * position_progress
            + self.attitude_group_weight * attitude_progress
            + self.velocity_group_weight * velocity_progress
        )
        balanced_progress_clipped = float(
            np.clip(
                balanced_progress_raw,
                -self.capture_progress_clip,
                self.capture_progress_clip,
            )
        )

        weights = self.reward_weights
        v12_terms = {
            "reward_capture_progress": (
                weights["capture_progress"]
                * goal["near_target_gate"]
                * balanced_progress_clipped
            ),
            "reward_flyaway": (
                -weights["flyaway"] * goal["flyaway_cost_clipped"]
            ),
        }
        reward = float(reward + sum(v12_terms.values()))
        components.update(
            {
                "near_target_gate": goal["near_target_gate"],
                "capture_position_progress": position_progress,
                "capture_attitude_progress": attitude_progress,
                "capture_velocity_progress": velocity_progress,
                "capture_balanced_progress_raw": balanced_progress_raw,
                "capture_balanced_progress_clipped": (
                    balanced_progress_clipped
                ),
                "capture_progress_clip": float(self.capture_progress_clip),
                "near_target_occupancy_penalty_active": False,
                "axial_overshoot_m": goal["axial_overshoot_m"],
                "axial_overshoot_penalty_active": False,
                "radial_velocity_mps": goal["radial_velocity_mps"],
                "flyaway_speed_mps": goal["flyaway_speed_mps"],
                "flyaway_cost_raw": goal["flyaway_cost_raw"],
                "flyaway_cost_clipped": goal["flyaway_cost_clipped"],
                "flyaway_penalty_clip": float(self.flyaway_penalty_clip),
                **{key: float(value) for key, value in v12_terms.items()},
                "reward_total": reward,
            }
        )
        return reward, components


class ClosingSafetyExplorationV13GymEnv(ClosingSafetyExplorationV12GymEnv):
    """V12 dynamics with worst-unsatisfied-component capture progress.

    V12 can trade a severe error in one P3 group against progress in another.
    V13 instead removes already-satisfied components from its near-target
    capture cost and blends a nominal group-weighted cost with a smooth maximum
    over the three remaining group deficits.  Position and terminal velocity
    receive most of the nominal weight, while the smooth-worst term still makes
    alignment decisive whenever it becomes the limiting unsatisfied group.

    Only bounded progress through this cost is rewarded.  V13 preserves V12's
    absence of near-target occupancy and axial-overshoot penalties, its bounded
    fly-away penalty, and its zero-penalty failed-approach reset.
    """

    reward_scheme = "p3_reach_v13"
    capture_position_group_weight = 0.60
    capture_velocity_group_weight = 0.30
    capture_attitude_group_weight = 0.10
    capture_weighted_cost_fraction = 0.35
    capture_worst_cost_fraction = 0.65
    capture_worst_beta = 4.0
    capture_satisfied_log_threshold = float(np.log(2.0))
    capture_satisfied_ratio_epsilon = 1e-12

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        goal = self._v13_goal_state(self.core_env.state)
        self._previous_v13_capture_cost = goal["capture_cost"]
        info["reward_scheme"] = self.reward_scheme
        return observation, info

    def _v13_smooth_max(self, values):
        values = np.asarray(values, dtype=np.float64)
        maximum = float(np.max(values))
        return float(
            maximum
            + np.log(
                np.mean(
                    np.exp(self.capture_worst_beta * (values - maximum))
                )
            )
            / self.capture_worst_beta
        )

    def _v13_goal_state(self, state):
        goal = self._v12_goal_state(state)
        log_components = np.asarray(goal["log_components"], dtype=np.float64)
        ratios = np.asarray(goal["ratios"], dtype=np.float64)
        deficits = np.maximum(
            log_components - self.capture_satisfied_log_threshold,
            0.0,
        )
        deficits = np.where(
            ratios <= 1.0 + self.capture_satisfied_ratio_epsilon,
            0.0,
            deficits,
        )

        position_deficit = self._v13_smooth_max(deficits[:2])
        velocity_deficit = self._v13_smooth_max(deficits[2:4])
        attitude_deficit = self._v13_smooth_max(deficits[4:6])
        nominal_cost = float(
            self.capture_position_group_weight * position_deficit
            + self.capture_velocity_group_weight * velocity_deficit
            + self.capture_attitude_group_weight * attitude_deficit
        )
        worst_group_cost = self._v13_smooth_max(
            [position_deficit, velocity_deficit, attitude_deficit]
        )
        capture_cost = float(
            self.capture_weighted_cost_fraction * nominal_cost
            + self.capture_worst_cost_fraction * worst_group_cost
        )

        goal.update(
            {
                "capture_component_deficits": deficits,
                "capture_position_deficit": position_deficit,
                "capture_velocity_deficit": velocity_deficit,
                "capture_attitude_deficit": attitude_deficit,
                "capture_nominal_cost": nominal_cost,
                "capture_worst_group_cost": worst_group_cost,
                "capture_cost": capture_cost,
            }
        )
        return goal

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        # Deliberately bypass V12's additive capture term while retaining the
        # complete V9 global guidance reward underneath the new V13 shaping.
        reward, components = ClosingSafetyExplorationV9GymEnv._compute_sac_reward(
            self,
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        goal = self._v13_goal_state(state)
        capture_progress_raw = float(
            self._previous_v13_capture_cost - goal["capture_cost"]
        )
        capture_progress_clipped = float(
            np.clip(
                capture_progress_raw,
                -self.capture_progress_clip,
                self.capture_progress_clip,
            )
        )
        self._previous_v13_capture_cost = goal["capture_cost"]

        weights = self.reward_weights
        v13_terms = {
            "reward_capture_progress": (
                weights["capture_progress"]
                * goal["near_target_gate"]
                * capture_progress_clipped
            ),
            "reward_flyaway": (
                -weights["flyaway"] * goal["flyaway_cost_clipped"]
            ),
        }
        reward = float(reward + sum(v13_terms.values()))
        component_names = self.goal_component_names
        components.update(
            {
                "near_target_gate": goal["near_target_gate"],
                "capture_cost": goal["capture_cost"],
                "capture_nominal_cost": goal["capture_nominal_cost"],
                "capture_worst_group_cost": goal[
                    "capture_worst_group_cost"
                ],
                "capture_position_deficit": goal[
                    "capture_position_deficit"
                ],
                "capture_velocity_deficit": goal[
                    "capture_velocity_deficit"
                ],
                "capture_attitude_deficit": goal[
                    "capture_attitude_deficit"
                ],
                "capture_progress_raw": capture_progress_raw,
                "capture_progress_clipped": capture_progress_clipped,
                "capture_progress_clip": float(self.capture_progress_clip),
                "capture_satisfied_log_threshold": (
                    self.capture_satisfied_log_threshold
                ),
                "capture_satisfied_ratio_epsilon": (
                    self.capture_satisfied_ratio_epsilon
                ),
                "capture_weighted_cost_fraction": (
                    self.capture_weighted_cost_fraction
                ),
                "capture_worst_cost_fraction": (
                    self.capture_worst_cost_fraction
                ),
                "capture_worst_beta": self.capture_worst_beta,
                "near_target_occupancy_penalty_active": False,
                "axial_overshoot_m": goal["axial_overshoot_m"],
                "axial_overshoot_penalty_active": False,
                "radial_velocity_mps": goal["radial_velocity_mps"],
                "flyaway_speed_mps": goal["flyaway_speed_mps"],
                "flyaway_cost_raw": goal["flyaway_cost_raw"],
                "flyaway_cost_clipped": goal["flyaway_cost_clipped"],
                "flyaway_penalty_clip": float(self.flyaway_penalty_clip),
                **{
                    f"capture_{name}_deficit": float(value)
                    for name, value in zip(
                        component_names,
                        goal["capture_component_deficits"],
                    )
                },
                **{key: float(value) for key, value in v13_terms.items()},
                "reward_total": reward,
            }
        )
        return reward, components


class ClosingSafetyExplorationV14GymEnv(ClosingSafetyExplorationV12GymEnv):
    """V12 with a bounded physical angle/rate soft constraint.

    This is intentionally a direct V12 successor rather than a V13 successor.
    All V12 task shaping is preserved.  The only new reward term is zero while
    both physical docking alignment quantities are in their existing safe
    bands, rises continuously through the warning bands, and saturates once
    either quantity reaches its existing unsafe threshold.  Violations remain
    non-terminal and the initial-state distribution is unchanged.
    """

    reward_scheme = "p3_reach_v14"
    reward_weights = {
        **ClosingSafetyExplorationV12GymEnv.reward_weights,
        "angle_constraint": 0.1,
    }
    angle_constraint_psi_safe_deg = 3.0
    angle_constraint_psi_unsafe_deg = 4.0
    angle_constraint_omega_safe_degs = 0.15
    angle_constraint_omega_unsafe_degs = 0.20

    @staticmethod
    def _bounded_warning_depth(value, safe_limit, unsafe_limit):
        width = max(float(unsafe_limit - safe_limit), 1e-12)
        return float(np.clip((value - safe_limit) / width, 0.0, 1.0))

    def _v14_angle_constraint(self, state):
        q = self.core_env.get_derived_quantities(state)
        psi_abs = float(abs(q["psi_e"]))
        omega_abs = float(abs(q["omega"]))
        psi_depth = self._bounded_warning_depth(
            psi_abs,
            np.deg2rad(self.angle_constraint_psi_safe_deg),
            np.deg2rad(self.angle_constraint_psi_unsafe_deg),
        )
        omega_depth = self._bounded_warning_depth(
            omega_abs,
            np.deg2rad(self.angle_constraint_omega_safe_degs),
            np.deg2rad(self.angle_constraint_omega_unsafe_degs),
        )
        # Smooth bounded union: zero only when both depths are zero and one
        # whenever either physical alignment quantity is fully unsafe.
        combined_depth = float(
            1.0 - (1.0 - psi_depth) * (1.0 - omega_depth)
        )
        return {
            "psi_abs": psi_abs,
            "omega_abs": omega_abs,
            "psi_warning_depth": psi_depth,
            "omega_warning_depth": omega_depth,
            "combined_warning_depth": combined_depth,
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        constraint = self._v14_angle_constraint(state)
        reward_angle_constraint = float(
            -self.reward_weights["angle_constraint"]
            * constraint["combined_warning_depth"]
        )
        reward = float(reward + reward_angle_constraint)
        components.update(
            {
                "angle_constraint_penalty_enabled": True,
                "angle_constraint_non_terminal": True,
                "angle_constraint_psi_abs_deg": float(
                    np.rad2deg(constraint["psi_abs"])
                ),
                "angle_constraint_omega_abs_degs": float(
                    np.rad2deg(constraint["omega_abs"])
                ),
                "angle_constraint_psi_safe_deg": float(
                    self.angle_constraint_psi_safe_deg
                ),
                "angle_constraint_psi_unsafe_deg": float(
                    self.angle_constraint_psi_unsafe_deg
                ),
                "angle_constraint_omega_safe_degs": float(
                    self.angle_constraint_omega_safe_degs
                ),
                "angle_constraint_omega_unsafe_degs": float(
                    self.angle_constraint_omega_unsafe_degs
                ),
                "angle_constraint_psi_warning_depth": constraint[
                    "psi_warning_depth"
                ],
                "angle_constraint_omega_warning_depth": constraint[
                    "omega_warning_depth"
                ],
                "angle_constraint_combined_warning_depth": constraint[
                    "combined_warning_depth"
                ],
                "angle_constraint_penalty_clip": float(
                    self.reward_weights["angle_constraint"]
                ),
                "reward_angle_constraint": reward_angle_constraint,
                "reward_total": reward,
            }
        )
        return reward, components


class ClosingSafetyExplorationV15GymEnv(ClosingSafetyExplorationV12GymEnv):
    """V12 with a bounded lateral docking-cone soft constraint.

    The seeds 200--209 V12 diagnostic showed no axial cone exits: all observed
    containment failures were lateral.  V15 therefore penalizes only physical
    lateral excess beyond ``rho <= cone_slope * s``.  The excess is normalized
    by the cone base radius and clipped, so the new term is continuous at the
    cone boundary and never exceeds the existing 0.1 hard-safety penalty per
    step.  It is zero everywhere inside the cone, remains non-terminal, and
    does not change the initial-state distribution or V12 task shaping.

    No separate axial-overshoot cost is reintroduced; V11 removed that cost
    because persistent target-relative overshoot shaping harmed approach.
    """

    reward_scheme = "p3_reach_v15"
    reward_weights = {
        **ClosingSafetyExplorationV12GymEnv.reward_weights,
        "cone_constraint": 0.1,
    }

    def _v15_cone_constraint(self, state):
        q = self.core_env.get_derived_quantities(state)
        cone_length = float(self.core_env.cone_length)
        cone_base_radius = float(self.core_env.cone_base_radius)
        axial_position = float(np.clip(q["s"], 0.0, cone_length))
        allowed_radius = float(self.core_env.cone_slope * axial_position)
        lateral_excess = float(max(0.0, q["rho"] - allowed_radius))
        warning_depth = float(
            np.clip(
                lateral_excess / max(cone_base_radius, 1e-12),
                0.0,
                1.0,
            )
        )
        return {
            "s": float(q["s"]),
            "rho": float(q["rho"]),
            "allowed_radius": allowed_radius,
            "lateral_excess": lateral_excess,
            "warning_depth": warning_depth,
        }

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        reward, components = super()._compute_sac_reward(
            state=state,
            normalized_action=normalized_action,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        constraint = self._v15_cone_constraint(state)
        reward_cone_constraint = float(
            -self.reward_weights["cone_constraint"]
            * constraint["warning_depth"]
        )
        reward = float(reward + reward_cone_constraint)
        components.update(
            {
                "cone_constraint_penalty_enabled": True,
                "cone_constraint_non_terminal": True,
                "cone_constraint_lateral_only": True,
                "cone_constraint_s_m": constraint["s"],
                "cone_constraint_rho_m": constraint["rho"],
                "cone_constraint_allowed_radius_m": constraint[
                    "allowed_radius"
                ],
                "cone_constraint_lateral_excess_m": constraint[
                    "lateral_excess"
                ],
                "cone_constraint_warning_depth": constraint[
                    "warning_depth"
                ],
                "cone_constraint_penalty_clip": float(
                    self.reward_weights["cone_constraint"]
                ),
                "reward_cone_constraint": reward_cone_constraint,
                "reward_total": reward,
            }
        )
        return reward, components
