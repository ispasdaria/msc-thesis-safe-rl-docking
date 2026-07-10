import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.docking3dof.core import Docking3DoFEnv


class Docking3DoFGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the existing Docking3DoFEnv.

    SAC sees phase-specific observations.

    Final approach:
        observation = [rpx, rpy, vpx, vpy, psi_e, omegaz]

        rpx, rpy:
            chaser docking-port center relative to the target docking-port
            center, expressed in the target-centered orbital frame [m]
        vpx, vpy:
            chaser docking-port velocity relative to the target docking-port
            velocity, expressed in the target-centered orbital frame [m/s]
        psi_e:
            wrapped relative attitude error [rad]
        omegaz:
            relative angular velocity [rad/s]
        action      = normalized action in [-1, 1]^3

    Internally:
        action is scaled to physical accelerations:
            [ax_Bc, ay_Bc, alphaz_Bc]

    Closing:
        observation = [rx, ry, vx, vy, psi, omegaz]

    The simulator state remains [rx, ry, vx, vy, psi, omegaz]. Only the final
    approach agent observes docking-port errors because docking is accepted at
    port level.
    The phase changes:
        - reset distribution
        - terminal condition
        - safety logic
        - reward definition
        - default maximum episode time
    """

    metadata = {"render_modes": []}
    closing_observation_labels = ("rx", "ry", "vx", "vy", "psi", "omegaz")
    final_observation_labels = ("rpx", "rpy", "vpx", "vpy", "psi_e", "omegaz")

    def __init__(
        self,
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=None,
    ):
        super().__init__()

        if phase not in ["closing", "final"]:
            raise ValueError("phase must be either 'closing' or 'final'.")

        if mode not in ["nominal", "testing", "training"]:
            raise ValueError("mode must be one of: 'nominal', 'testing', or 'training'.")

        if max_time is None:
            if phase == "closing":
                max_time = 12000.0
            elif phase == "final":
                max_time = 5000.0

        self.phase = phase
        self.mode = mode
        self.normalize_action = normalize_action
        self.max_time = max_time

        self.core_env = Docking3DoFEnv(
            dt=dt,
            max_time=max_time,
            integrator=integrator,
            phase=phase,
            seed=seed,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        if self.normalize_action:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=self.core_env.action_low.astype(np.float32),
                high=self.core_env.action_high.astype(np.float32),
                dtype=np.float32,
            )

    def _scale_action(self, action):
        """
        Converts SAC action to physical body-frame acceleration command.
        """

        action = np.asarray(action, dtype=np.float64)

        if self.normalize_action:
            physical_action = action * np.array(
                [
                    self.core_env.ax_max,
                    self.core_env.ay_max,
                    self.core_env.alphaz_max,
                ],
                dtype=np.float64,
            )
        else:
            physical_action = action

        return physical_action

    def _get_obs(self):
        """
        Returns the current Gymnasium observation.

        In the final phase, the SAC policy receives the docking-port relative
        position and velocity used by the docking acceptance conditions. In the
        closing phase, it keeps the original CoM-relative state observation.
        """

        if self.phase == "closing":
            return self.core_env.state.astype(np.float32)

        q = self.core_env.get_derived_quantities(self.core_env.state)

        obs = np.array(
            [
                q["r_p"][0],
                q["r_p"][1],
                q["v_p"][0],
                q["v_p"][1],
                q["psi_e"],
                q["omega"],
            ],
            dtype=np.float32,
        )

        return obs

    def _get_observation_labels(self):
        if self.phase == "final":
            return self.final_observation_labels

        return self.closing_observation_labels

    def _get_reset_info(self):
        """
        Returns diagnostic information after reset.
        """

        q = self.core_env.get_derived_quantities(self.core_env.state)
        monitor = self.core_env.monitor_safety(self.core_env.state)

        return {
            "phase": self.phase,
            "mode": self.mode,
            "max_time": self.max_time,
            "safe": monitor["safe"],
            "warning": monitor["warning"],
            "unsafe": monitor["unsafe"],
            "port_distance": q["port_distance"],
            "port_speed": q["port_speed"],
            "com_distance": q["com_distance"],
            "com_speed": q["com_speed"],
            "s": q["s"],
            "rho": q["rho"],
            "s_dot": q["s_dot"],
            "psi_e": q["psi_e"],
            "omega": q["omega"],
            "observation_labels": self._get_observation_labels(),
        }

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset API.

        Returns:
            obs, info
        """

        super().reset(seed=seed)

        if seed is not None:
            self.core_env.rng = np.random.default_rng(seed)

        initial_state = None
        mode = self.mode

        if options is not None:
            initial_state = options.get("initial_state", None)
            mode = options.get("mode", self.mode)

            if mode not in ["nominal", "testing", "training"]:
                raise ValueError(
                    "mode must be one of: 'nominal', 'testing', or 'training'."
                )

        self.core_env.reset(
            initial_state=initial_state,
            mode=mode,
        )

        q = self.core_env.get_derived_quantities(self.core_env.state)

        self.prev_port_distance = q["port_distance"]
        self.prev_s = q["s"]
        self.prev_rho = q["rho"]
        if self.phase == "closing" and self.core_env.p3_target is not None:
            target_error = self.core_env.state - self.core_env.p3_target
            target_error[4] = self.core_env.wrap_to_pi(
                self.core_env.state[4] - self.core_env.p3_target[4]
            )
            self.prev_closing_pos_error = np.linalg.norm(target_error[0:2])
            self.prev_closing_lateral_error = abs(target_error[1])

        obs = self._get_obs()
        info = self._get_reset_info()

        return obs, info

    def step(self, action):
        """
        Gymnasium step API.

        Returns:
            obs, reward, terminated, truncated, info
        """

        physical_action = self._scale_action(action)

        _, _, done, info = self.core_env.step(physical_action)
        obs = self._get_obs()

        terminated = bool(
            info.get("success", False)
            or info.get("failure", False)
        )

        truncated = bool(info.get("time_limit", False))

        reward, reward_components = self._compute_sac_reward(
            state=self.core_env.state,
            normalized_action=np.asarray(action, dtype=np.float64),
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        info["normalized_action"] = np.asarray(action, dtype=np.float32)
        info["physical_action"] = physical_action.astype(np.float32)
        info["reward_components"] = reward_components

        return obs, float(reward), terminated, truncated, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        """
        SAC reward with diagnostic reward components.

        This returns a dictionary with individual reward terms so that reward
        behavior can be inspected during rollouts.
        """

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

        reward_components = {
            "phase": self.phase,

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
            "reward_rate": 0.0,
            "reward_action": 0.0,
            "reward_alpha": 0.0,
            "reward_progress": 0.0,
            "reward_no_progress": 0.0,
            "reward_unsafe_forward_progress": 0.0,
            "reward_speed_margin": 0.0,
            "reward_rho_keep": 0.0,
            "reward_rho_progress": 0.0,
            "reward_cone_preference": 0.0,
            "reward_near_field_centering": 0.0,
            "reward_mid_field_centering": 0.0,
            "reward_psi_margin": 0.0,
            "reward_omega_margin": 0.0,
            "reward_terminal_corridor": 0.0,
            "reward_terminal_alignment": 0.0,
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

            vp_safe = self.core_env.vp_safe_close
            sdot_safe = self.core_env.sdot_safe_close
            speed_ratio = port_speed / max(vp_safe, 1e-6)
            sdot_ratio = abs(q["s_dot"]) / max(sdot_safe, 1e-6)
            speed_gate = max(speed_ratio, sdot_ratio)
            safe_progress_scale = np.clip(
                1.0 - max(0.0, speed_gate - 0.8) / 0.4,
                0.0,
                1.0,
            )
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
            else:
                p3_position_ratio = 0.0
                p3_lateral_ratio = 0.0
                p3_lateral_violation = 0.0
                p3_handoff_gate = 0.0
                p3_psi_ratio = 0.0
                p3_omega_ratio = 0.0
                p3_alignment_ratio = 0.0

            reward_position = -0.002 * pos_error
            reward_velocity = -0.5 * vel_error
            reward_attitude = -0.5 * attitude_error
            reward_rate = -0.1 * rate_error
            reward_action = -0.01 * action_norm
            reward_progress = 20.0 * distance_progress * safe_progress_scale
            no_progress = (
                pos_error > 50.0
                and distance_progress < 0.01
                and speed_gate <= 0.75
            )
            reward_no_progress = -0.05 if no_progress else 0.0
            reward_lateral = 0.0
            reward_speed_margin = (
                -80.0 * max(0.0, speed_ratio - 0.75) ** 2
                -80.0 * max(0.0, sdot_ratio - 0.75) ** 2
            )

            reward = (
                reward_position
                + reward_velocity
                + reward_attitude
                + reward_rate
                + reward_action
                + reward_progress
                + reward_no_progress
                + reward_lateral
                + reward_speed_margin
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
                    "psi_ratio": float(psi_safe_ratio),
                    "omega_ratio": float(omega_safe_ratio),
                    "rho_gate": float(corridor_ratio),
                    "psi_gate": float(p3_psi_ratio),
                    "omega_gate": float(p3_omega_ratio),
                    "alignment_gate": float(alignment_gate),
                    "no_progress": float(no_progress),
                    "too_fast": float(
                        max(0.0, speed_ratio - 0.75)
                        + max(0.0, sdot_ratio - 0.75)
                    ),
                    "reward_position": float(reward_position),
                    "reward_velocity": float(reward_velocity),
                    "reward_attitude": float(reward_attitude),
                    "reward_rate": float(reward_rate),
                    "reward_action": float(reward_action),
                    "reward_progress": float(reward_progress),
                    "reward_no_progress": float(reward_no_progress),
                    "reward_lateral": float(reward_lateral),
                    "reward_speed_margin": float(reward_speed_margin),
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
            reward_warning = -10.0
            reward += reward_warning
            reward_components["reward_warning"] = float(reward_warning)

        if info.get("failure", False):
            reward_failure = -1000.0
            reward += reward_failure
            reward_components["reward_failure"] = float(reward_failure)

        if info.get("success", False):
            reward_success = 1000.0
            reward += reward_success
            reward_components["reward_success"] = float(reward_success)

        if truncated:
            reward_timeout = -200.0
            reward += reward_timeout
            reward_components["reward_timeout"] = float(reward_timeout)

        reward_components["reward_total"] = float(reward)

        return reward, reward_components

    def render(self):
        """
        No visual renderer yet.
        Use core_env.get_history_arrays() for plotting.
        """
        pass

    def close(self):
        pass

    def get_history_arrays(self):
        return self.core_env.get_history_arrays()
