"""Capture-specific reward with ordinary hard closing-safety termination.

This isolated environment keeps :class:`Docking3DoFGymEnv` dynamics, safety
classification, and terminal conditions.  Only the reward is replaced.  In
particular, a hard closing safety violation still terminates the episode.
"""

from __future__ import annotations

import numpy as np

from envs.docking3dof.core import Docking3DoFEnv
from envs.docking3dof.gym_env import Docking3DoFGymEnv


class CaptureSafetyTerminalCoreEnv(Docking3DoFEnv):
    """Closing core that also treats alignment as safety-critical."""

    def __init__(self, *args, phase="closing", **kwargs):
        if phase != "closing":
            raise ValueError(
                "CaptureSafetyTerminalCoreEnv requires phase='closing'."
            )
        super().__init__(*args, phase=phase, **kwargs)

    def monitor_safety(self, state):
        monitor = super().monitor_safety(state)
        geometry = monitor["geometry"]
        velocity = monitor["velocity"]
        alignment = monitor["alignment"]

        unsafe = bool(
            geometry["unsafe"]
            or velocity["unsafe"]
            or alignment["unsafe"]
        )
        safe = bool(
            geometry["safe"]
            and velocity["safe"]
            and alignment["safe"]
        )
        warning = bool(
            (
                geometry["warn"]
                or velocity["warn"]
                or alignment["warn"]
            )
            and not unsafe
        )
        alignment_warning_reasons = []
        alignment_unsafe_reasons = []
        if self.psi_safe < alignment["psi_abs"] <= self.psi_unsafe:
            alignment_warning_reasons.append("attitude")
        if self.omega_safe < alignment["omega_abs"] <= self.omega_unsafe:
            alignment_warning_reasons.append("angular_rate")
        if alignment["psi_abs"] > self.psi_unsafe:
            alignment_unsafe_reasons.append("attitude")
        if alignment["omega_abs"] > self.omega_unsafe:
            alignment_unsafe_reasons.append("angular_rate")
        violated_constraints = list(monitor["violated_constraints"])
        if alignment["unsafe"] and "alignment" not in violated_constraints:
            violated_constraints.append("alignment")
        for reason in alignment_unsafe_reasons:
            if reason not in violated_constraints:
                violated_constraints.append(reason)

        monitor.update(
            {
                "safe": safe,
                "warning": warning,
                "unsafe": unsafe,
                "violated_constraints": violated_constraints,
                "capture_alignment_safety_enabled": True,
                "alignment_warning_reasons": alignment_warning_reasons,
                "alignment_unsafe_reasons": alignment_unsafe_reasons,
            }
        )
        return monitor

    def check_done(self, state):
        done, info = super().check_done(state)
        alignment = info["alignment"]
        info["capture_alignment_safety_enabled"] = True
        info["alignment_warning_reasons"] = [
            name
            for name, active in (
                (
                    "attitude",
                    self.psi_safe < alignment["psi_abs"] <= self.psi_unsafe,
                ),
                (
                    "angular_rate",
                    self.omega_safe
                    < alignment["omega_abs"]
                    <= self.omega_unsafe,
                ),
            )
            if active
        ]
        info["alignment_unsafe_reasons"] = [
            name
            for name, active in (
                ("attitude", alignment["psi_abs"] > self.psi_unsafe),
                ("angular_rate", alignment["omega_abs"] > self.omega_unsafe),
            )
            if active
        ]
        return done, info


class CaptureSafetyTerminalEnv(Docking3DoFGymEnv):
    """Closing environment tailored to the 50-150 m P3 capture task."""

    reward_scheme = "capture_progress_safety_terminal_v1"
    progress_scale = 100.0
    progress_clip = 0.05
    smooth_max_beta = 4.0
    aggregate_weight = 0.5
    worst_component_weight = 0.5
    action_weight = 0.01
    warning_weight = 0.5
    success_bonus = 1_000.0
    safety_failure_penalty = 1_500.0
    timeout_penalty = 100.0

    def __init__(
        self,
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=None,
        closing_reward_version="v20",
        action_hold_steps=1,
    ):
        if phase != "closing":
            raise ValueError("CaptureSafetyTerminalEnv requires phase='closing'.")
        # The inherited reward version is not used by _compute_sac_reward, but
        # V20 keeps the parent reset diagnostics internally consistent.
        super().__init__(
            phase=phase,
            mode=mode,
            dt=dt,
            max_time=max_time,
            integrator=integrator,
            normalize_action=normalize_action,
            seed=seed,
            closing_reward_version=closing_reward_version,
            action_hold_steps=action_hold_steps,
        )
        # Replace only this isolated trial's core. The shared closing core and
        # every existing controller retain their original safety definition.
        self.core_env = CaptureSafetyTerminalCoreEnv(
            dt=dt,
            max_time=self.max_time,
            integrator=integrator,
            phase="closing",
            seed=seed,
        )

    def _capture_error(self, state):
        if self.core_env.p3_target is None:
            raise RuntimeError("Capture reward requires an episode-specific P3 target.")
        error = np.asarray(state, dtype=np.float64) - self.core_env.p3_target
        error = error.copy()
        error[4] = self.core_env.wrap_to_pi(error[4])
        return error

    def _capture_cost(self, state):
        error = self._capture_error(state)
        tolerance = np.maximum(self.core_env.p3_target_tolerance, 1e-12)
        ratios = np.abs(error) / tolerance
        log_components = np.log1p(ratios)
        aggregate = float(np.mean(log_components))
        maximum = float(np.max(log_components))
        smooth_worst = float(
            maximum
            + np.log(
                np.mean(
                    np.exp(self.smooth_max_beta * (log_components - maximum))
                )
            )
            / self.smooth_max_beta
        )
        cost = (
            self.aggregate_weight * aggregate
            + self.worst_component_weight * smooth_worst
        )
        return {
            "error": error,
            "ratios": ratios,
            "log_components": log_components,
            "aggregate": aggregate,
            "smooth_worst": smooth_worst,
            "cost": float(cost),
        }

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        capture = self._capture_cost(self.core_env.state)
        monitor = self.core_env.monitor_safety(self.core_env.state)
        self.previous_capture_cost = capture["cost"]
        info = dict(info)
        info.update(
            {
                "reward_scheme": self.reward_scheme,
                "capture_cost": capture["cost"],
                "safety_violation_termination_enabled": True,
                "capture_alignment_safety_enabled": True,
                "geometry": monitor["geometry"],
                "velocity": monitor["velocity"],
                "alignment": monitor["alignment"],
                "violated_constraints": monitor["violated_constraints"],
                "alignment_warning_reasons": monitor[
                    "alignment_warning_reasons"
                ],
                "alignment_unsafe_reasons": monitor[
                    "alignment_unsafe_reasons"
                ],
            }
        )
        return observation, info

    def _warning_depth(self, derived, info):
        port_depth = np.clip(
            (derived["port_speed"] - self.core_env.vp_safe_close)
            / max(
                self.core_env.vp_unsafe_close - self.core_env.vp_safe_close,
                1e-12,
            ),
            0.0,
            1.0,
        )
        axial_depth = np.clip(
            (abs(derived["s_dot"]) - self.core_env.sdot_safe_close)
            / max(
                self.core_env.sdot_unsafe_close
                - self.core_env.sdot_safe_close,
                1e-12,
            ),
            0.0,
            1.0,
        )
        geometry_depth = float(bool(info.get("geometry", {}).get("warn", False)))
        attitude_depth = np.clip(
            (abs(derived["psi_e"]) - self.core_env.psi_safe)
            / max(self.core_env.psi_unsafe - self.core_env.psi_safe, 1e-12),
            0.0,
            1.0,
        )
        angular_rate_depth = np.clip(
            (abs(derived["omega"]) - self.core_env.omega_safe)
            / max(
                self.core_env.omega_unsafe - self.core_env.omega_safe,
                1e-12,
            ),
            0.0,
            1.0,
        )
        return float(
            max(
                port_depth,
                axial_depth,
                geometry_depth,
                attitude_depth,
                angular_rate_depth,
            )
        )

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
        capture = self._capture_cost(state)
        previous_cost = getattr(self, "previous_capture_cost", capture["cost"])
        raw_progress = float(previous_cost - capture["cost"])
        clipped_progress = float(
            np.clip(raw_progress, -self.progress_clip, self.progress_clip)
        )
        self.previous_capture_cost = capture["cost"]

        action = np.asarray(normalized_action, dtype=np.float64)
        action_squared_norm = float(np.dot(action, action))
        derived = self.core_env.get_derived_quantities(state)
        warning_depth = self._warning_depth(derived, info)

        reward_capture_progress = self.progress_scale * clipped_progress
        reward_action = -self.action_weight * action_squared_norm
        reward_warning_margin = -self.warning_weight * warning_depth**2
        reward_success = self.success_bonus if info.get("success", False) else 0.0
        reward_safety_failure = (
            -self.safety_failure_penalty if info.get("failure", False) else 0.0
        )
        reward_timeout = -self.timeout_penalty if truncated else 0.0
        reward = (
            reward_capture_progress
            + reward_action
            + reward_warning_margin
            + reward_success
            + reward_safety_failure
            + reward_timeout
        )

        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "safety_violation_termination_enabled": True,
            "capture_alignment_safety_enabled": True,
            "capture_cost": capture["cost"],
            "capture_cost_aggregate": capture["aggregate"],
            "capture_cost_smooth_worst": capture["smooth_worst"],
            "capture_cost_progress_raw": raw_progress,
            "capture_cost_progress_clipped": clipped_progress,
            "capture_normalized_error_mean": float(np.mean(capture["ratios"])),
            "capture_normalized_error_max": float(np.max(capture["ratios"])),
            "capture_axial_ratio": float(capture["ratios"][0]),
            "capture_lateral_ratio": float(capture["ratios"][1]),
            "capture_vx_ratio": float(capture["ratios"][2]),
            "capture_vy_ratio": float(capture["ratios"][3]),
            "capture_attitude_ratio": float(capture["ratios"][4]),
            "capture_rate_ratio": float(capture["ratios"][5]),
            "warning_depth": warning_depth,
            "action_squared_norm": action_squared_norm,
            "reward_capture_progress": float(reward_capture_progress),
            "reward_action": float(reward_action),
            "reward_warning_margin": float(reward_warning_margin),
            "reward_success": float(reward_success),
            "reward_safety_failure": float(reward_safety_failure),
            "reward_timeout": float(reward_timeout),
            "reward_total": float(reward),
        }
        return float(reward), components
