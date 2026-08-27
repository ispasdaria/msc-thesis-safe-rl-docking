"""Capture environment for the explicit ``X_H -> validated X_P3`` task."""

from __future__ import annotations

import numpy as np

from capture_safety_terminal_env import (
    CaptureSafetyTerminalCoreEnv,
    CaptureSafetyTerminalEnv,
)
from systematic_interface import DEFAULT_CONTRACT


class SystematicCaptureCoreEnv(CaptureSafetyTerminalCoreEnv):
    """Use validated final-controller P3 membership as the only success set."""

    def is_at_P3_target(self, state):
        return DEFAULT_CONTRACT.in_validated_p3(state)

    def check_done(self, state):
        done, info = super().check_done(state)
        info["validated_p3_membership"] = DEFAULT_CONTRACT.in_validated_p3(state)
        if info.get("success", False):
            info["terminal_set"] = "validated_X_P3_testing"
            info["termination_reason"] = "success_validated_X_P3"
        return done, info


class SystematicCaptureEnv(CaptureSafetyTerminalEnv):
    """Systematic capture reward plus hard geometry/velocity/alignment safety."""

    reward_scheme = "systematic_capture_filtered_axial_consistency_v10"

    # Axial progress is measured in metres toward the validated P3 rx box.
    axial_progress_negative_clip_m = 1.0
    axial_progress_positive_clip_m = 0.25
    axial_progress_scale = 1.5

    # Pseudo-Huber costs retain a linear recovery gradient at large errors
    # without the hard plateau of v3 or the vanishing log gradient of v4.
    tracking_scales = np.array(
        [0.015, 3.0, 0.01, np.deg2rad(1.0), np.deg2rad(0.05)],
        dtype=np.float64,
    )
    tracking_weights_far = np.array([0.0, 0.05, 0.02, 0.02, 0.01])
    # Preserve v5 lateral/attitude shaping throughout the maneuver. Only vx
    # receives stronger terminal-zone tracking in this controlled ablation.
    tracking_weights_near = np.array([0.0, 0.05, 0.02, 0.02, 0.01])
    vx_tracking_weight_far = 0.003
    vx_tracking_weight_near = 0.150
    vx_full_weight_axial_gap_m = 30.0
    box_state_cost_weight = 0.05

    # Potential-based progress toward the actual global P3 success box.
    terminal_box_progress_negative_clip = 0.25
    terminal_box_progress_positive_clip = 0.25
    terminal_box_progress_scale = 2.0

    # Inside 30 m of the upper P3 face, continuing inward while any other
    # terminal component is invalid receives an explicit readiness cost.
    terminal_sync_zone_m = 30.0
    unready_inward_speed_scale_mps = 0.10
    unready_inward_penalty_weight = 0.20

    # Positive progress is reduced only after errors exceed maneuver-level
    # bounds, and never disappears entirely.
    progress_gate_floor = 0.05
    progress_gate_error_bounds = np.array(
        [10.0, 0.03, np.deg2rad(3.0)], dtype=np.float64
    )

    # Treat lateral position and velocity as one stabilization problem. The
    # robust energy retains a useful gradient far from P3, while its bounded
    # state cost cannot dominate every other objective. Symmetric progress
    # clipping avoids rewarding a reduction in one component while the other
    # component is diverging.
    lateral_energy_scales = np.array([3.0, 0.01], dtype=np.float64)
    lateral_energy_progress_negative_clip = 0.10
    lateral_energy_progress_positive_clip = 0.10
    lateral_energy_progress_scale = 1.00
    lateral_energy_state_weight = 0.02

    # Close to P3, lateral-position invalidity receives an additional inward
    # readiness cost. Unlike v8's generic readiness depth, this term retains
    # magnitude information beyond the first P3 half-width.
    terminal_lateral_unready_weight = 0.15
    terminal_lateral_unready_depth_clip = 4.0

    # Penalize reversal immediately rather than waiting for the delayed
    # flyaway terminal condition. This does not prescribe a distance-varying
    # vx reference; it only rejects outward motion during capture.
    outward_velocity_scale_mps = 0.05
    outward_velocity_depth_clip = 3.0
    outward_velocity_weight = 0.20

    geometry_warning_weight = 1.00
    velocity_warning_weight = 0.25
    alignment_warning_weight = 0.50
    attitude_margin_start = np.deg2rad(2.0)
    attitude_margin_full = np.deg2rad(3.0)
    attitude_margin_weight = 0.25
    action_weights = np.array([0.001, 0.001, 0.004], dtype=np.float64)
    action_delta_weights = np.array([0.005, 0.020, 0.040], dtype=np.float64)
    success_bonus = 500.0
    safety_failure_penalty = 500.0
    timeout_penalty = 200.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inherited_core = self.core_env
        seed = kwargs.get("seed", None)
        self.core_env = SystematicCaptureCoreEnv(
            dt=inherited_core.dt,
            max_time=self.max_time,
            integrator=inherited_core.integrator,
            phase="closing",
            p3_target_tolerance=DEFAULT_CONTRACT.capture_shaping_scale,
            seed=seed,
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.previous_axial_gap_m = self._axial_box_gap(self.core_env.state)
        tracking = self._tracking_terms(self.core_env.state)
        self.previous_lateral_energy = self._lateral_energy_terms(
            tracking["errors"]
        )["energy"]
        self.previous_terminal_box_cost = self._terminal_box_terms(
            self.core_env.state
        )["cost"]
        self.previous_normalized_action = np.zeros(3, dtype=np.float64)
        info = dict(info)
        info.update(
            {
                "success_set": "validated_X_P3_testing",
                "validated_p3_membership": DEFAULT_CONTRACT.in_validated_p3(
                    self.core_env.state
                ),
            }
        )
        return observation, info

    @staticmethod
    def _axial_box_gap(state):
        rx = float(np.asarray(state, dtype=np.float64)[0])
        lower = float(DEFAULT_CONTRACT.p3_lower[0])
        upper = float(DEFAULT_CONTRACT.p3_upper[0])
        return float(max(lower - rx, 0.0, rx - upper))

    def _tracking_terms(self, state):
        state = np.asarray(state, dtype=np.float64)
        target = np.asarray(self.core_env.p3_target, dtype=np.float64)
        errors = np.array(
            [
                state[2] - target[2],
                state[1] - target[1],
                state[3] - target[3],
                self.core_env.wrap_to_pi(state[4] - target[4]),
                state[5] - target[5],
            ],
            dtype=np.float64,
        )
        ratios = np.abs(errors) / self.tracking_scales
        axial_gap = self._axial_box_gap(state)
        vx_near_fraction = float(
            np.clip(1.0 - axial_gap / self.vx_full_weight_axial_gap_m, 0.0, 1.0)
        )
        vx_weight = float(
            self.vx_tracking_weight_far
            + (self.vx_tracking_weight_near - self.vx_tracking_weight_far)
            * vx_near_fraction
        )
        component_weights = (
            self.tracking_weights_far
            + (self.tracking_weights_near - self.tracking_weights_far)
            * vx_near_fraction
        )
        component_weights[0] = vx_weight
        robust_ratios = np.sqrt(1.0 + ratios**2) - 1.0
        component_costs = component_weights * robust_ratios

        # Gating begins only beyond maneuver-level ry/vy/attitude bounds.
        # Its floor retains an approach signal while recovery is in progress.
        gate_errors = np.abs(errors[[1, 2, 3]])
        gate_excess = np.maximum(
            gate_errors / self.progress_gate_error_bounds - 1.0, 0.0
        )
        raw_progress_gate = float(1.0 / (1.0 + np.dot(gate_excess, gate_excess)))
        progress_gate = float(
            self.progress_gate_floor
            + (1.0 - self.progress_gate_floor) * raw_progress_gate
        )
        return {
            "errors": errors,
            "ratios": ratios,
            "robust_ratios": robust_ratios,
            "component_weights": component_weights,
            "component_costs": component_costs,
            "cost": float(np.sum(component_costs)),
            "vx_weight": vx_weight,
            "vx_near_fraction": vx_near_fraction,
            "gate_excess": gate_excess,
            "raw_progress_gate": raw_progress_gate,
            "progress_gate": progress_gate,
        }

    def _lateral_energy_terms(self, tracking_errors):
        """Return a coupled robust energy for target-relative ry and vy."""
        lateral_errors = np.asarray(tracking_errors, dtype=np.float64)[1:3]
        ratios = lateral_errors / self.lateral_energy_scales
        energy = float(np.sqrt(1.0 + np.dot(ratios, ratios)) - 1.0)
        return {
            "errors": lateral_errors,
            "ratios": ratios,
            "energy": energy,
            "bounded_energy": float(np.tanh(energy)),
        }

    @staticmethod
    def _terminal_box_terms(state):
        ratios = DEFAULT_CONTRACT.p3_box_ratios(state)
        robust_ratios = np.sqrt(1.0 + ratios**2) - 1.0
        return {
            "ratios": ratios,
            "robust_ratios": robust_ratios,
            "cost": float(np.sum(robust_ratios)),
        }

    def _terminal_readiness_terms(self, state, box_ratios):
        axial_gap = self._axial_box_gap(state)
        zone_fraction = float(
            np.clip(1.0 - axial_gap / self.terminal_sync_zone_m, 0.0, 1.0)
        )
        nonaxial_indices = np.array([1, 2, 3, 4, 5])
        unready_depth = float(
            np.clip(np.max(box_ratios[nonaxial_indices]), 0.0, 1.0)
        )
        inward_speed = max(-float(np.asarray(state)[2]), 0.0)
        inward_speed_depth = float(
            np.clip(
                inward_speed / self.unready_inward_speed_scale_mps,
                0.0,
                1.0,
            )
        )
        generic_cost = (
            self.unready_inward_penalty_weight
            * zone_fraction
            * unready_depth
            * inward_speed_depth
        )
        lateral_position_unready_depth = float(
            np.clip(
                box_ratios[1],
                0.0,
                self.terminal_lateral_unready_depth_clip,
            )
        )
        lateral_position_cost = (
            self.terminal_lateral_unready_weight
            * zone_fraction
            * lateral_position_unready_depth
            * inward_speed_depth
        )
        cost = generic_cost + lateral_position_cost
        return {
            "zone_fraction": zone_fraction,
            "unready_depth": unready_depth,
            "inward_speed_mps": inward_speed,
            "inward_speed_depth": inward_speed_depth,
            "generic_cost": float(generic_cost),
            "lateral_position_unready_depth": lateral_position_unready_depth,
            "lateral_position_cost": float(lateral_position_cost),
            "cost": float(cost),
        }

    def _geometry_warning_depth(self, derived, geometry):
        if not geometry.get("warn", False):
            return 0.0

        s = float(derived["s"])
        rho = float(derived["rho"])
        cone_radius = max(self.core_env.cone_slope * max(s, 0.0), 1e-12)
        cone_excess = max(rho / cone_radius - 1.0, 0.0)
        cone_depth = 1.0 - np.exp(-cone_excess)

        kos_depth = 0.0
        if geometry.get("in_S_KOS", False):
            kos_depth = np.clip(
                (self.core_env.r_kos - derived["com_distance"])
                / max(0.25 * self.core_env.r_kos, 1e-12),
                0.0,
                1.0,
            )

        passed_depth = 0.0
        # The inherited closing monitor uses the sampled target rx as its P3
        # boundary. This systematic task accepts the full global [280, 320] m
        # box, so do not penalize valid lower-box states as "passed P3".
        if (
            geometry.get("passed_P3_region", False)
            and float(derived["r_com"][0]) < float(DEFAULT_CONTRACT.p3_lower[0])
        ):
            passed_depth = np.clip(
                (float(DEFAULT_CONTRACT.p3_lower[0]) - derived["r_com"][0])
                / max(DEFAULT_CONTRACT.capture_shaping_scale[0], 1e-12),
                0.0,
                1.0,
            )
        return float(np.clip(max(cone_depth, kos_depth, passed_depth), 0.0, 1.0))

    def _warning_costs(self, derived, info):
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
                self.core_env.sdot_unsafe_close - self.core_env.sdot_safe_close,
                1e-12,
            ),
            0.0,
            1.0,
        )
        attitude_depth = np.clip(
            (abs(derived["psi_e"]) - self.core_env.psi_safe)
            / max(self.core_env.psi_unsafe - self.core_env.psi_safe, 1e-12),
            0.0,
            1.0,
        )
        rate_depth = np.clip(
            (abs(derived["omega"]) - self.core_env.omega_safe)
            / max(self.core_env.omega_unsafe - self.core_env.omega_safe, 1e-12),
            0.0,
            1.0,
        )
        geometry_depth = self._geometry_warning_depth(
            derived, info.get("geometry", {})
        )
        velocity_depth = float(max(port_depth, axial_depth))
        alignment_depth = float(max(attitude_depth, rate_depth))
        attitude_margin_depth = float(
            np.clip(
                (abs(derived["psi_e"]) - self.attitude_margin_start)
                / max(
                    self.attitude_margin_full - self.attitude_margin_start,
                    1e-12,
                ),
                0.0,
                1.0,
            )
        )
        return {
            "geometry_depth": geometry_depth,
            "velocity_depth": velocity_depth,
            "alignment_depth": alignment_depth,
            "attitude_margin_depth": attitude_margin_depth,
            "geometry_cost": self.geometry_warning_weight * geometry_depth,
            "velocity_cost": self.velocity_warning_weight * velocity_depth**2,
            "alignment_cost": self.alignment_warning_weight
            * alignment_depth**2,
            "attitude_margin_cost": self.attitude_margin_weight
            * attitude_margin_depth**2,
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
        axial_gap = self._axial_box_gap(state)
        previous_axial_gap = getattr(self, "previous_axial_gap_m", axial_gap)
        raw_axial_progress = float(previous_axial_gap - axial_gap)
        clipped_axial_progress = float(
            np.clip(
                raw_axial_progress,
                -self.axial_progress_negative_clip_m,
                self.axial_progress_positive_clip_m,
            )
        )
        self.previous_axial_gap_m = axial_gap

        tracking = self._tracking_terms(state)
        lateral = self._lateral_energy_terms(tracking["errors"])
        previous_lateral_energy = getattr(
            self, "previous_lateral_energy", lateral["energy"]
        )
        raw_lateral_energy_progress = float(
            previous_lateral_energy - lateral["energy"]
        )
        clipped_lateral_energy_progress = float(
            np.clip(
                raw_lateral_energy_progress,
                -self.lateral_energy_progress_negative_clip,
                self.lateral_energy_progress_positive_clip,
            )
        )
        self.previous_lateral_energy = lateral["energy"]
        terminal_box = self._terminal_box_terms(state)
        box_ratios = terminal_box["ratios"]
        previous_terminal_box_cost = getattr(
            self, "previous_terminal_box_cost", terminal_box["cost"]
        )
        raw_terminal_box_progress = float(
            previous_terminal_box_cost - terminal_box["cost"]
        )
        clipped_terminal_box_progress = float(
            np.clip(
                raw_terminal_box_progress,
                -self.terminal_box_progress_negative_clip,
                self.terminal_box_progress_positive_clip,
            )
        )
        self.previous_terminal_box_cost = terminal_box["cost"]
        readiness = self._terminal_readiness_terms(state, box_ratios)
        box_cost = float(np.mean(np.log1p(box_ratios)))
        bounded_box_cost = float(np.tanh(box_cost))
        derived = self.core_env.get_derived_quantities(state)
        warnings = self._warning_costs(derived, info)
        action = np.asarray(normalized_action, dtype=np.float64)
        previous_action = getattr(
            self,
            "previous_normalized_action",
            np.zeros_like(action),
        )
        action_delta = action - previous_action
        self.previous_normalized_action = action.copy()

        progress_multiplier = (
            tracking["progress_gate"] if clipped_axial_progress > 0.0 else 1.0
        )
        reward_axial_progress = (
            self.axial_progress_scale
            * clipped_axial_progress
            * progress_multiplier
        )
        reward_tracking = -tracking["cost"]
        reward_lateral_energy_progress = (
            self.lateral_energy_progress_scale
            * clipped_lateral_energy_progress
        )
        reward_lateral_energy_state = (
            -self.lateral_energy_state_weight * lateral["energy"]
        )
        reward_terminal_box_progress = (
            self.terminal_box_progress_scale
            * clipped_terminal_box_progress
            * readiness["zone_fraction"]
        )
        reward_unready_inward = -readiness["cost"]
        reward_box_state = -self.box_state_cost_weight * bounded_box_cost
        reward_geometry_warning = -warnings["geometry_cost"]
        reward_velocity_warning = -warnings["velocity_cost"]
        reward_alignment_warning = -warnings["alignment_cost"]
        reward_attitude_margin = -warnings["attitude_margin_cost"]
        outward_velocity_mps = max(float(np.asarray(state)[2]), 0.0)
        outward_velocity_depth = float(
            np.clip(
                outward_velocity_mps / self.outward_velocity_scale_mps,
                0.0,
                self.outward_velocity_depth_clip,
            )
        )
        reward_outward_velocity = (
            -self.outward_velocity_weight * outward_velocity_depth**2
        )
        reward_action = -float(np.dot(self.action_weights, action**2))
        reward_action_delta = -float(
            np.dot(self.action_delta_weights, action_delta**2)
        )
        reward_success = self.success_bonus if info.get("success", False) else 0.0
        reward_safety_failure = (
            -self.safety_failure_penalty if info.get("failure", False) else 0.0
        )
        reward_timeout = -self.timeout_penalty if truncated else 0.0
        reward = (
            reward_axial_progress
            + reward_tracking
            + reward_lateral_energy_progress
            + reward_lateral_energy_state
            + reward_terminal_box_progress
            + reward_unready_inward
            + reward_box_state
            + reward_geometry_warning
            + reward_velocity_warning
            + reward_alignment_warning
            + reward_attitude_margin
            + reward_outward_velocity
            + reward_action
            + reward_action_delta
            + reward_success
            + reward_safety_failure
            + reward_timeout
        )

        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "success_set": "validated_X_P3_testing",
            "safety_violation_termination_enabled": True,
            "capture_alignment_safety_enabled": True,
            "axial_box_gap_m": axial_gap,
            "axial_progress_raw_m": raw_axial_progress,
            "axial_progress_clipped_m": clipped_axial_progress,
            "axial_progress_gate": tracking["progress_gate"],
            "target_vx_mps": float(self.core_env.p3_target[2]),
            "tracking_cost": tracking["cost"],
            "tracking_vx_weight": tracking["vx_weight"],
            "tracking_vx_near_fraction": tracking["vx_near_fraction"],
            "tracking_vx_ratio": float(tracking["ratios"][0]),
            "tracking_ry_ratio": float(tracking["ratios"][1]),
            "tracking_vy_ratio": float(tracking["ratios"][2]),
            "tracking_attitude_ratio": float(tracking["ratios"][3]),
            "tracking_rate_ratio": float(tracking["ratios"][4]),
            "lateral_position_error_m": abs(float(lateral["errors"][0])),
            "lateral_velocity_error_mps": abs(float(lateral["errors"][1])),
            "lateral_position_energy_ratio": float(lateral["ratios"][0]),
            "lateral_velocity_energy_ratio": float(lateral["ratios"][1]),
            "lateral_energy": lateral["energy"],
            "lateral_energy_bounded": lateral["bounded_energy"],
            "lateral_energy_progress_raw": raw_lateral_energy_progress,
            "lateral_energy_progress_clipped": clipped_lateral_energy_progress,
            "terminal_box_cost": terminal_box["cost"],
            "terminal_box_progress_raw": raw_terminal_box_progress,
            "terminal_box_progress_clipped": clipped_terminal_box_progress,
            "terminal_box_progress_zone_multiplier": readiness[
                "zone_fraction"
            ],
            "terminal_sync_zone_fraction": readiness["zone_fraction"],
            "terminal_unready_depth": readiness["unready_depth"],
            "terminal_lateral_position_unready_depth": readiness[
                "lateral_position_unready_depth"
            ],
            "terminal_inward_speed_mps": readiness["inward_speed_mps"],
            "terminal_inward_speed_depth": readiness["inward_speed_depth"],
            "progress_gate_raw": tracking["raw_progress_gate"],
            "progress_gate_ry_excess": float(tracking["gate_excess"][0]),
            "progress_gate_vy_excess": float(tracking["gate_excess"][1]),
            "progress_gate_attitude_excess": float(tracking["gate_excess"][2]),
            "validated_p3_box_cost": box_cost,
            "validated_p3_box_cost_bounded": bounded_box_cost,
            "validated_p3_box_ratio_max": float(np.max(box_ratios)),
            "geometry_warning_depth": warnings["geometry_depth"],
            "velocity_warning_depth": warnings["velocity_depth"],
            "alignment_warning_depth": warnings["alignment_depth"],
            "attitude_margin_depth": warnings["attitude_margin_depth"],
            "outward_velocity_mps": outward_velocity_mps,
            "outward_velocity_depth": outward_velocity_depth,
            "action_delta_squared_norm": float(np.dot(action_delta, action_delta)),
            "reward_axial_progress": float(reward_axial_progress),
            "reward_tracking": float(reward_tracking),
            "reward_lateral_energy_progress": float(
                reward_lateral_energy_progress
            ),
            "reward_lateral_energy_state": float(reward_lateral_energy_state),
            "reward_terminal_box_progress": float(reward_terminal_box_progress),
            "reward_unready_inward": float(reward_unready_inward),
            "reward_unready_inward_generic": float(-readiness["generic_cost"]),
            "reward_unready_inward_lateral_position": float(
                -readiness["lateral_position_cost"]
            ),
            "reward_box_state": float(reward_box_state),
            "reward_geometry_warning": float(reward_geometry_warning),
            "reward_velocity_warning": float(reward_velocity_warning),
            "reward_alignment_warning": float(reward_alignment_warning),
            "reward_attitude_margin": float(reward_attitude_margin),
            "reward_outward_velocity": float(reward_outward_velocity),
            "reward_action": float(reward_action),
            "reward_action_delta": float(reward_action_delta),
            "reward_success": float(reward_success),
            "reward_safety_failure": float(reward_safety_failure),
            "reward_timeout": float(reward_timeout),
            "reward_total": float(reward),
        }
        return float(reward), components
