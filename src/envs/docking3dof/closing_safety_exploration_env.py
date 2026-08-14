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
