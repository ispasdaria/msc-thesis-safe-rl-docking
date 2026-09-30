"""Reward computation for the standard 3-DoF Gym environment."""

import numpy as np


def compute_sac_reward(
    env,
    state,
    normalized_action,
    physical_action,
    info,
    terminated,
    truncated,
):
    """Return the phase reward and its diagnostic component mapping."""
    self = env

    q = self.core_env.get_derived_quantities(state)

    port_distance = q["port_distance"]
    port_speed = q["port_speed"]
    psi_abs = abs(q["psi_e"])
    omega_abs = abs(q["omega"])

    action_scale = np.array(
        [
            self.core_env.ax_max,
            self.core_env.ay_max,
            self.core_env.alphaz_max,
        ],
        dtype=np.float64,
    )

    action_norm = np.linalg.norm(physical_action / action_scale)

    closing_reward_time_scale = 1.0
    if (
        self.phase == "closing"
        and self.closing_reward_version in {"v15_dt", "v15_dt_safe"}
    ):
        closing_reward_time_scale = (
            self.core_env.dt / self.closing_reward_reference_dt
        )

    reward_components = {
        "phase": self.phase,
        "closing_reward_version": self.closing_reward_version,
        "closing_reward_time_scale": float(closing_reward_time_scale),

        # Raw diagnostic quantities
        "port_distance": float(port_distance),
        "port_speed": float(port_speed),
        "psi_abs": float(psi_abs),
        "omega_abs": float(omega_abs),
        "action_norm": float(action_norm),
        "s": float(q["s"]),
        "rho": float(q["rho"]),
        "s_dot": float(q["s_dot"]),

        # Error diagnostics
        "pos_error": None,
        "vel_error": None,
        "attitude_error": None,
        "rate_error": None,

        # Final-approach-specific diagnostics
        "desired_s_dot": None,
        "s_dot_error": None,
        "too_fast": None,
        "moving_away": None,
        "v_lateral": None,
        "progress_port": None,
        "progress_s": None,
        "rho_progress": None,
        "rho_excess": None,
        "cone_radius_at_s": None,
        "rho_ratio": None,
        "center_quality": None,
        "near_field_weight": None,
        "p3_position_ratio": None,
        "p3_lateral_ratio": None,
        "p3_lateral_violation": None,
        "p3_handoff_gate": None,
        "psi_ratio": None,
        "omega_ratio": None,
        "speed_ratio": None,
        "rho_gate": None,
        "speed_gate": None,
        "psi_gate": None,
        "omega_gate": None,
        "alignment_gate": None,
        "velocity_progress_gate": None,
        "alignment_progress_gate": None,
        "geometry_progress_gate": None,
        "progress_gate": None,
        "safe_progress": None,
        "unsafe_forward_progress": None,
        "no_progress": None,
        "attitude_progress": None,
        "p3_axial_ratio": None,
        "p3_vx_ratio": None,
        "p3_vy_ratio": None,
        "p3_attitude_ratio": None,
        "p3_rate_ratio": None,
        "p3_normalized_error_mean": None,
        "p3_normalized_error_max": None,
        "p3_terminal_weight": None,
        "p3_log_error_mean": None,
        "p3_log_error_smooth_max": None,
        "p3_terminal_cost": None,
        "p3_terminal_cost_progress": None,
        "p3_terminal_progress_scale": None,

        # Reward terms
        "reward_position": 0.0,
        "reward_velocity": 0.0,
        "reward_s_dot": 0.0,
        "reward_too_fast": 0.0,
        "reward_moving_away": 0.0,
        "reward_lateral": 0.0,
        "reward_rho_corridor": 0.0,
        "reward_lateral_speed": 0.0,
        "reward_attitude": 0.0,
        "reward_attitude_progress": 0.0,
        "reward_rate_progress": 0.0,
        "reward_rate": 0.0,
        "reward_action": 0.0,
        "reward_alpha": 0.0,
        "reward_progress": 0.0,
        "reward_axial_overshoot": 0.0,
        "reward_no_progress": 0.0,
        "reward_unsafe_forward_progress": 0.0,
        "reward_speed_margin": 0.0,
        "reward_port_speed_guard": 0.0,
        "reward_axial_speed_guard": 0.0,
        "reward_rho_keep": 0.0,
        "reward_rho_progress": 0.0,
        "reward_cone_preference": 0.0,
        "reward_near_field_centering": 0.0,
        "reward_mid_field_centering": 0.0,
        "reward_psi_margin": 0.0,
        "reward_omega_margin": 0.0,
        "reward_terminal_corridor": 0.0,
        "reward_terminal_alignment": 0.0,
        "reward_tolerance_aggregate": 0.0,
        "reward_tolerance_worst": 0.0,
        "reward_tolerance_progress": 0.0,
        "reward_tolerance_state": 0.0,
        "reward_warning": 0.0,
        "reward_failure": 0.0,
        "reward_success": 0.0,
        "reward_timeout": 0.0,
        "reward_total": 0.0,
    }

    if self.phase == "closing":
        if self.core_env.p3_target is not None:
            target = self.core_env.p3_target

            error = state - target
            error[4] = self.core_env.wrap_to_pi(state[4] - target[4])

            pos_error = np.linalg.norm(error[0:2])
            vel_error = np.linalg.norm(error[2:4])
            attitude_error = abs(error[4])
            rate_error = abs(error[5])
        else:
            pos_error = port_distance
            vel_error = port_speed
            attitude_error = psi_abs
            rate_error = omega_abs
            error = np.zeros(6, dtype=np.float64)

        distance_progress = 0.0
        if hasattr(self, "prev_closing_pos_error"):
            distance_progress = self.prev_closing_pos_error - pos_error

        self.prev_closing_pos_error = pos_error
        lateral_error = (
            abs(error[1])
            if self.core_env.p3_target is not None
            else q["rho"]
        )
        lateral_progress = 0.0
        if hasattr(self, "prev_closing_lateral_error"):
            lateral_progress = self.prev_closing_lateral_error - lateral_error

        self.prev_closing_lateral_error = lateral_error

        attitude_progress = 0.0
        if hasattr(self, "prev_closing_attitude_error"):
            attitude_progress = (
                self.prev_closing_attitude_error - attitude_error
            )

        self.prev_closing_attitude_error = attitude_error
        rate_progress = 0.0
        if hasattr(self, "prev_closing_rate_error"):
            rate_progress = self.prev_closing_rate_error - rate_error

        self.prev_closing_rate_error = rate_error

        vp_safe = self.core_env.vp_safe_close
        vp_unsafe = self.core_env.vp_unsafe_close
        sdot_safe = self.core_env.sdot_safe_close
        sdot_unsafe = self.core_env.sdot_unsafe_close
        speed_ratio = port_speed / max(vp_safe, 1e-6)
        sdot_ratio = abs(q["s_dot"]) / max(sdot_safe, 1e-6)
        speed_gate = max(speed_ratio, sdot_ratio)
        # Closing reward v12: reward task progress uniformly throughout the
        # state space. Velocity safety remains safety-critical through the
        # warning and hard-failure rewards below, while the progress term
        # no longer prescribes a conservative operating speed inside the
        # safe velocity set. Keep this diagnostic at one so saved rollout
        # formats remain compatible with earlier reward versions.
        safe_progress_scale = 1.0
        cone_radius_at_s = self.core_env.cone_slope * max(q["s"], 1e-6)
        rho_ratio = q["rho"] / max(cone_radius_at_s, 1e-6)
        corridor_ratio = q["rho"] / max(self.core_env.corridor_radius, 1e-6)
        corridor_excess = max(0.0, corridor_ratio - 1.0)
        psi_safe_ratio = psi_abs / max(self.core_env.psi_safe, 1e-6)
        omega_safe_ratio = omega_abs / max(self.core_env.omega_safe, 1e-6)
        alignment_gate = max(psi_safe_ratio, omega_safe_ratio)
        if self.core_env.p3_target is not None:
            p3_tol = self.core_env.p3_target_tolerance
            p3_x_ratio = abs(error[0]) / max(p3_tol[0], 1e-6)
            p3_lateral_ratio = abs(error[1]) / max(p3_tol[1], 1e-6)
            p3_position_ratio = np.hypot(p3_x_ratio, p3_lateral_ratio)
            p3_lateral_violation = max(0.0, p3_lateral_ratio - 1.0)
            p3_handoff_gate = 1.0 / (1.0 + p3_position_ratio)
            p3_psi_ratio = attitude_error / max(p3_tol[4], 1e-6)
            p3_omega_ratio = rate_error / max(p3_tol[5], 1e-6)
            p3_alignment_ratio = max(p3_psi_ratio, p3_omega_ratio)
            axial_overshoot = max(0.0, -error[0] - p3_tol[0])
        else:
            p3_position_ratio = 0.0
            p3_lateral_ratio = 0.0
            p3_lateral_violation = 0.0
            p3_handoff_gate = 0.0
            p3_psi_ratio = 0.0
            p3_omega_ratio = 0.0
            p3_alignment_ratio = 0.0
            axial_overshoot = 0.0

        reward_position = (
            -0.002 * pos_error * closing_reward_time_scale
        )
        reward_velocity = (
            -0.5 * vel_error * closing_reward_time_scale
        )
        # Reward v14 strengthens the persistent P3-relative attitude
        # penalty while retaining v13's attitude-progress term.
        reward_attitude = (
            -5.0 * attitude_error * closing_reward_time_scale
        )
        # Reward v13 adds only P3-relative attitude progress to v12. This
        # rewards reducing alignment error and penalizes the observed
        # rotation toward the 180-degree anti-aligned state without
        # prescribing an attitude trajectory or penalizing angular action.
        reward_attitude_progress = 50.0 * attitude_progress
        reward_rate = -0.1 * rate_error * closing_reward_time_scale
        reward_action = -0.01 * action_norm * closing_reward_time_scale
        # Reward v12 retains v10's lateral P3-relative velocity penalty. It
        # does not prescribe axial closing speed and does not retain reward
        # v9's angular-action penalty.
        lateral_velocity_error = (
            abs(error[3])
            if self.core_env.p3_target is not None
            else abs(q["v_p"][1])
        )
        reward_lateral_speed = (
            -5.0
            * lateral_velocity_error
            * closing_reward_time_scale
        )
        # Apply no axial-speed penalty inside the safe set. Once the axial
        # speed enters the warning band, increase the penalty smoothly up
        # to the hard safety boundary.
        sdot_warning_depth = np.clip(
            (abs(q["s_dot"]) - sdot_safe)
            / max(sdot_unsafe - sdot_safe, 1e-6),
            0.0,
            1.0,
        )
        reward_s_dot = (
            -20.0
            * sdot_warning_depth**2
            * closing_reward_time_scale
        )
        # Apply the same warning-depth shaping to total docking-port speed.
        # This remains zero throughout the safe set and therefore does not
        # prescribe a velocity schedule. It only distinguishes safe states
        # from increasingly critical states inside the warning band.
        port_speed_warning_depth = np.clip(
            (port_speed - vp_safe)
            / max(vp_unsafe - vp_safe, 1e-6),
            0.0,
            1.0,
        )
        reward_speed_margin = (
            -20.0
            * port_speed_warning_depth**2
            * closing_reward_time_scale
        )
        # Reward v15 uses symmetric P3-distance progress together with
        # persistent lateral position and velocity shaping. Axial
        # overshoot remains available as a rollout diagnostic, but it is
        # deliberately not part of this reward version.
        reward_progress = 20.0 * distance_progress
        reward_axial_overshoot = 0.0
        no_progress_distance = (
            0.01 * closing_reward_time_scale
        )
        no_progress = (
            pos_error > 50.0
            and distance_progress < no_progress_distance
        )
        reward_no_progress = (
            -0.05 * closing_reward_time_scale
            if no_progress
            else 0.0
        )
        # Reward v15 adds a persistent P3-relative lateral-position cost.
        # Together with reward_lateral_speed, this creates a position-plus-
        # damping objective without prescribing a lateral velocity profile.
        reward_lateral = (
            -0.01 * lateral_error * closing_reward_time_scale
        )

        reward_rate_progress = 0.0
        reward_port_speed_guard = 0.0
        reward_axial_speed_guard = 0.0
        if (
            self.closing_reward_version == "v15_dt_safe"
            and self.core_env.p3_target is not None
        ):
            # Bound persistent alignment costs so long safe trajectories
            # cannot become intrinsically worse than immediate failure.
            attitude_ratio = attitude_error / max(p3_tol[4], 1e-12)
            rate_ratio = rate_error / max(p3_tol[5], 1e-12)
            reward_attitude = (
                -self.closing_safe_attitude_state_scale
                * np.tanh(attitude_ratio / 4.0) ** 2
                * closing_reward_time_scale
            )
            reward_rate = (
                -self.closing_safe_rate_state_scale
                * np.tanh(rate_ratio / 4.0) ** 2
                * closing_reward_time_scale
            )

            previous_attitude_ratio = max(
                0.0,
                attitude_error + attitude_progress,
            ) / max(p3_tol[4], 1e-12)
            previous_rate_ratio = max(
                0.0,
                rate_error + rate_progress,
            ) / max(p3_tol[5], 1e-12)
            reward_attitude_progress = (
                self.closing_safe_attitude_progress_scale
                * (
                    np.log1p(previous_attitude_ratio)
                    - np.log1p(attitude_ratio)
                )
            )
            reward_rate_progress = (
                self.closing_safe_rate_progress_scale
                * (
                    np.log1p(previous_rate_ratio)
                    - np.log1p(rate_ratio)
                )
            )

            # Begin braking guidance before the existing safe/warning
            # boundary. The original warning-depth costs remain active and
            # become progressively stronger closer to the hard boundary.
            port_guard_start = (
                self.closing_safe_speed_guard_fraction * vp_safe
            )
            axial_guard_start = (
                self.closing_safe_speed_guard_fraction * sdot_safe
            )
            port_guard_depth = np.clip(
                (port_speed - port_guard_start)
                / max(vp_unsafe - port_guard_start, 1e-12),
                0.0,
                1.0,
            )
            axial_guard_depth = np.clip(
                (abs(q["s_dot"]) - axial_guard_start)
                / max(sdot_unsafe - axial_guard_start, 1e-12),
                0.0,
                1.0,
            )
            reward_port_speed_guard = (
                -self.closing_safe_speed_guard_scale
                * port_guard_depth**2
                * closing_reward_time_scale
            )
            reward_axial_speed_guard = (
                -self.closing_safe_speed_guard_scale
                * axial_guard_depth**2
                * closing_reward_time_scale
            )

        # Reward v16 extends v15 with P3-tolerance-normalized terminal
        # shaping. Each state component is divided by the exact tolerance
        # used by the P3 success condition, so all six conditions must
        # become small together. The mean rewards balanced convergence;
        # the maximum prevents a single bad component from being hidden by
        # the other five. Ratio clipping bounds the scale and avoids a
        # distance gate that the policy could exploit by remaining just
        # outside the shaped region.
        p3_error_ratios = np.zeros(6, dtype=np.float64)
        clipped_p3_error_ratios = np.zeros(6, dtype=np.float64)
        p3_normalized_error_mean = 0.0
        p3_normalized_error_max = 0.0
        p3_terminal_weight = 0.0
        p3_log_error_mean = 0.0
        p3_log_error_smooth_max = 0.0
        p3_terminal_cost = 0.0
        p3_terminal_cost_progress = 0.0
        terminal_progress_scale = 0.0
        reward_tolerance_aggregate = 0.0
        reward_tolerance_worst = 0.0
        reward_tolerance_progress = 0.0
        reward_tolerance_state = 0.0
        if self.core_env.p3_target is not None:
            p3_error_ratios = np.abs(error) / np.maximum(p3_tol, 1e-12)

        if (
            self.closing_reward_version == "v16"
            and self.core_env.p3_target is not None
        ):
            clipped_p3_error_ratios = np.clip(
                p3_error_ratios,
                0.0,
                self.closing_v16_ratio_clip,
            )
            p3_normalized_error_mean = float(
                np.mean(clipped_p3_error_ratios)
            )
            p3_normalized_error_max = float(
                np.max(clipped_p3_error_ratios)
            )
            p3_terminal_weight = 1.0
            reward_tolerance_aggregate = (
                -self.closing_v16_aggregate_weight
                * p3_terminal_weight
                * p3_normalized_error_mean
            )
            reward_tolerance_worst = (
                -self.closing_v16_worst_weight
                * p3_terminal_weight
                * p3_normalized_error_max
            )
        elif (
            self.closing_reward_version in {"v17", "v18", "v19", "v20"}
            and self.core_env.p3_target is not None
        ):
            terminal = self._closing_v17_terminal_cost(error, p3_tol)
            p3_log_error_mean = terminal["aggregate"]
            p3_log_error_smooth_max = terminal["smooth_worst"]
            p3_terminal_cost = terminal["cost"]
            previous_terminal_cost = getattr(
                self,
                "prev_closing_terminal_cost",
                p3_terminal_cost,
            )
            p3_terminal_cost_progress = (
                previous_terminal_cost - p3_terminal_cost
            )
            self.prev_closing_terminal_cost = p3_terminal_cost
            p3_normalized_error_mean = float(np.mean(p3_error_ratios))
            p3_normalized_error_max = float(np.max(p3_error_ratios))
            p3_terminal_weight = 1.0
            if self.closing_reward_version in {"v19", "v20"}:
                reward_tolerance_state = (
                    -self.closing_v19_state_cost_scale * p3_terminal_cost
                )
            else:
                terminal_progress_scale = (
                    self.closing_v18_progress_scale
                    if self.closing_reward_version == "v18"
                    else self.closing_v17_progress_scale
                )
                reward_tolerance_progress = (
                    terminal_progress_scale * p3_terminal_cost_progress
                )

        if self.closing_reward_version in {"v19", "v20"}:
            # Do not mix tolerance-normalized terminal control with v15's
            # differently-scaled distance, attitude, and progress terms.
            reward_position = 0.0
            reward_velocity = 0.0
            reward_attitude = 0.0
            reward_attitude_progress = 0.0
            reward_rate = 0.0
            reward_lateral_speed = 0.0
            reward_progress = 0.0
            reward_axial_overshoot = 0.0
            reward_no_progress = 0.0
            reward_lateral = 0.0
            if self.closing_reward_version == "v19":
                reward_s_dot = 0.0
                reward_speed_margin = 0.0
            reward = (
                reward_tolerance_state
                + reward_action
                + reward_s_dot
                + reward_speed_margin
            )
        else:
            reward = (
                reward_position
                + reward_velocity
                + reward_attitude
                + reward_attitude_progress
                + reward_rate
                + reward_rate_progress
                + reward_action
                + reward_lateral_speed
                + reward_s_dot
                + reward_progress
                + reward_axial_overshoot
                + reward_no_progress
                + reward_lateral
                + reward_speed_margin
                + reward_port_speed_guard
                + reward_axial_speed_guard
                + reward_tolerance_aggregate
                + reward_tolerance_worst
                + reward_tolerance_progress
            )

        reward_components.update(
            {
                "pos_error": float(pos_error),
                "vel_error": float(vel_error),
                "attitude_error": float(attitude_error),
                "rate_error": float(rate_error),
                "v_lateral": float(lateral_error),
                "progress_port": float(distance_progress),
                "rho_progress": float(lateral_progress),
                "speed_ratio": float(speed_ratio),
                "progress_gate": float(safe_progress_scale),
                "cone_radius_at_s": float(cone_radius_at_s),
                "rho_ratio": float(rho_ratio),
                "corridor_excess": float(corridor_excess),
                "rho_excess": float(
                    max(0.0, q["rho"] - self.core_env.corridor_radius)
                ),
                "p3_position_ratio": float(p3_position_ratio),
                "p3_lateral_ratio": float(p3_lateral_ratio),
                "p3_lateral_violation": float(p3_lateral_violation),
                "p3_handoff_gate": float(p3_handoff_gate),
                "p3_alignment_ratio": float(p3_alignment_ratio),
                "axial_overshoot": float(axial_overshoot),
                "psi_ratio": float(psi_safe_ratio),
                "omega_ratio": float(omega_safe_ratio),
                "rho_gate": float(corridor_ratio),
                "psi_gate": float(p3_psi_ratio),
                "omega_gate": float(p3_omega_ratio),
                "alignment_gate": float(alignment_gate),
                "no_progress": float(no_progress),
                "attitude_progress": float(attitude_progress),
                "rate_progress": float(rate_progress),
                "p3_axial_ratio": float(p3_error_ratios[0]),
                "p3_vx_ratio": float(p3_error_ratios[2]),
                "p3_vy_ratio": float(p3_error_ratios[3]),
                "p3_attitude_ratio": float(p3_error_ratios[4]),
                "p3_rate_ratio": float(p3_error_ratios[5]),
                "p3_normalized_error_mean": float(
                    p3_normalized_error_mean
                ),
                "p3_normalized_error_max": float(
                    p3_normalized_error_max
                ),
                "p3_terminal_weight": float(p3_terminal_weight),
                "p3_log_error_mean": float(p3_log_error_mean),
                "p3_log_error_smooth_max": float(
                    p3_log_error_smooth_max
                ),
                "p3_terminal_cost": float(p3_terminal_cost),
                "p3_terminal_cost_progress": float(
                    p3_terminal_cost_progress
                ),
                "p3_terminal_progress_scale": float(
                    terminal_progress_scale
                ),
                "too_fast": float(
                    max(0.0, speed_ratio - 0.75)
                    + max(0.0, sdot_ratio - 0.75)
                ),
                "reward_position": float(reward_position),
                "reward_velocity": float(reward_velocity),
                "reward_attitude": float(reward_attitude),
                "reward_attitude_progress": float(
                    reward_attitude_progress
                ),
                "reward_rate": float(reward_rate),
                "reward_rate_progress": float(reward_rate_progress),
                "reward_action": float(reward_action),
                "reward_lateral_speed": float(reward_lateral_speed),
                "reward_s_dot": float(reward_s_dot),
                "reward_progress": float(reward_progress),
                "reward_axial_overshoot": float(reward_axial_overshoot),
                "reward_no_progress": float(reward_no_progress),
                "reward_lateral": float(reward_lateral),
                "reward_speed_margin": float(reward_speed_margin),
                "reward_port_speed_guard": float(
                    reward_port_speed_guard
                ),
                "reward_axial_speed_guard": float(
                    reward_axial_speed_guard
                ),
                "reward_tolerance_aggregate": float(
                    reward_tolerance_aggregate
                ),
                "reward_tolerance_worst": float(
                    reward_tolerance_worst
                ),
                "reward_tolerance_progress": float(
                    reward_tolerance_progress
                ),
                "reward_tolerance_state": float(
                    reward_tolerance_state
                ),
            }
        )

    elif self.phase == "final":

        s = q["s"]
        rho = q["rho"]
        s_dot = q["s_dot"]

        progress_port = 0.0
        progress_s = 0.0

        if hasattr(self, "prev_port_distance"):
            progress_port = self.prev_port_distance - port_distance

        if hasattr(self, "prev_s"):
            progress_s = self.prev_s - s

        vp_safe = getattr(
            self.core_env,
            "vp_safe_final",
            getattr(self.core_env, "vp_safe", 1.0),
        )
        speed_ratio = port_speed / vp_safe

        psi_ratio = psi_abs / self.core_env.psi_safe
        omega_ratio = omega_abs / self.core_env.omega_safe

        if hasattr(self, "prev_rho"):
            rho_progress = self.prev_rho - rho
        else:
            rho_progress = 0.0

        cone_radius_at_s = self.core_env.cone_slope * max(s, 1e-6)
        rho_ratio = rho / max(cone_radius_at_s, 1e-6)
        center_quality = np.exp(-3.0 * rho_ratio**2)
        near_field_weight = np.exp(-s / 50.0)

        # General port-level nominal-control reward.
        # Safety-specific shaping is intentionally kept light so that the
        # later CBF filter can be evaluated as the explicit safety layer.
        reward_position = -0.03 * port_distance
        reward_velocity = -3.0 * port_speed
        reward_lateral = -0.5 * rho
        reward_attitude = -5.0 * psi_abs
        reward_rate = -1.0 * omega_abs
        reward_progress = 45.0 * progress_port * (0.25 + 0.75 * center_quality)
        reward_rho_progress = 5.0 * rho_progress
        reward_cone_preference = -10.0 * max(0.0, rho_ratio - 0.4) ** 2
        reward_near_field_centering = -25.0 * near_field_weight * rho_ratio**2
        reward_mid_field_centering = 0.0
        reward_psi_margin = -25.0 * max(0.0, psi_ratio - 0.5) ** 2
        reward_speed_margin = -20.0 * max(0.0, speed_ratio - 0.7) ** 2
        reward_action = -0.02 * action_norm**2
        reward_alpha = -2.0 * normalized_action[2] ** 2

        reward = (
            reward_position
            + reward_velocity
            + reward_lateral
            + reward_attitude
            + reward_rate
            + reward_action
            + reward_alpha
            + reward_progress
            + reward_rho_progress
            + reward_cone_preference
            + reward_near_field_centering
            + reward_psi_margin
            + reward_speed_margin
        )

        self.prev_port_distance = port_distance
        self.prev_s = s
        self.prev_rho = rho

        reward_components.update(
            {
                # Error diagnostics
                "pos_error": float(port_distance),
                "vel_error": float(port_speed),
                "attitude_error": float(psi_abs),
                "rate_error": float(omega_abs),

                # Final-approach-specific diagnostics
                "rho": float(rho),
                "s": float(s),
                "s_dot": float(s_dot),
                "v_lateral": float(abs(q["v_p"][1])),
                "progress_port": float(progress_port),
                "progress_s": float(progress_s),
                "rho_progress": float(rho_progress),
                "rho_excess": float(max(0.0, rho - self.core_env.corridor_radius)),
                "cone_radius_at_s": float(cone_radius_at_s),
                "rho_ratio": float(rho_ratio),
                "center_quality": float(center_quality),
                "near_field_weight": float(near_field_weight),
                "psi_ratio": float(psi_ratio),
                "omega_ratio": float(omega_ratio),
                "speed_ratio": float(speed_ratio),

                # Reward components
                "reward_position": float(reward_position),
                "reward_velocity": float(reward_velocity),
                "reward_lateral": float(reward_lateral),
                "reward_attitude": float(reward_attitude),
                "reward_rate": float(reward_rate),
                "reward_action": float(reward_action),
                "reward_alpha": float(reward_alpha),
                "reward_progress": float(reward_progress),
                "reward_rho_progress": float(reward_rho_progress),
                "reward_cone_preference": float(reward_cone_preference),
                "reward_near_field_centering": float(
                    reward_near_field_centering
                ),
                "reward_mid_field_centering": float(
                    reward_mid_field_centering
                ),
                "reward_psi_margin": float(reward_psi_margin),
                "reward_speed_margin": float(reward_speed_margin),
            }
        )

    else:
        raise ValueError("Unknown phase.")

    if info.get("warning", False):
        reward_warning = -10.0 * closing_reward_time_scale
        reward += reward_warning
        reward_components["reward_warning"] = float(reward_warning)

    if info.get("failure", False):
        reward_failure = (
            self.closing_safe_failure_reward
            if (
                self.phase == "closing"
                and self.closing_reward_version == "v15_dt_safe"
            )
            else -1000.0
        )
        reward += reward_failure
        reward_components["reward_failure"] = float(reward_failure)

    if info.get("success", False):
        reward_success = (
            self.closing_safe_success_reward
            if (
                self.phase == "closing"
                and self.closing_reward_version == "v15_dt_safe"
            )
            else 1000.0
        )
        reward += reward_success
        reward_components["reward_success"] = float(reward_success)

    if truncated:
        reward_timeout = (
            self.closing_safe_timeout_reward
            if (
                self.phase == "closing"
                and self.closing_reward_version == "v15_dt_safe"
            )
            else -200.0
        )
        reward += reward_timeout
        reward_components["reward_timeout"] = float(reward_timeout)

    reward_components["reward_total"] = float(reward)

    return reward, reward_components
