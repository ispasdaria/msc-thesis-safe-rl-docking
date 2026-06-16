import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.docking3dof.core import Docking3DoFEnv


class Docking3DoFGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the existing Docking3DoFEnv.

    SAC sees:
        observation = [rx, ry, vx, vy, psi, omegaz]
        action      = normalized action in [-1, 1]^3

    Internally:
        action is scaled to physical accelerations:
            [ax_Bc, ay_Bc, alphaz_Bc]

    The same observation structure is used for both phases.
    The phase changes:
        - reset distribution
        - terminal condition
        - safety logic
        - reward definition
        - default maximum episode time
    """

    metadata = {"render_modes": []}

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

        # Phase-dependent default max_time.
        # Still able to override it manually by passing max_time=...
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

        # Observation is the physical simulator state:
        # [rx, ry, vx, vy, psi, omegaz]
        #
        # The observation structure is the same for closing and final approach.
        # The actual observation at each step is self.core_env.state.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # SAC outputs actions in [-1, 1], then they are scaled to the physical
        # acceleration limits from the core environment.
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

        If normalize_action=True:
            SAC action is in [-1, 1]^3.

            physical action =
                [
                    action[0] * ax_max,
                    action[1] * ay_max,
                    action[2] * alphaz_max
                ]

        If normalize_action=False:
            action is already physical.
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
        Returns the current simulator state as the Gymnasium observation.
        """

        return self.core_env.state.astype(np.float32)

    def _get_reset_info(self):
        """
        Returns diagnostic information after reset.

        This information is useful for debugging and logging, but it is not
        part of the observation seen by SAC.
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

        obs, _, done, info = self.core_env.step(physical_action)

        obs = obs.astype(np.float32)

        # In Gymnasium:
        # terminated = task ended naturally, either success or failure.
        # truncated  = episode was cut off by a time limit.
        terminated = bool(
            info.get("success", False)
            or info.get("failure", False)
        )

        truncated = bool(info.get("time_limit", False))

        reward, reward_components = self._compute_sac_reward(
            state=self.core_env.state,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        info["normalized_action"] = np.asarray(action, dtype=np.float32)
        info["physical_action"] = physical_action.astype(np.float32)

        # Reward diagnostics for plotting/debugging.
        # These are not part of the observation; SAC does not directly use them.
        info["reward_components"] = reward_components

        return obs, float(reward), terminated, truncated, info

    def _compute_sac_reward(
        self,
        state,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        """
        First SAC reward version with diagnostic reward components.

        The total reward is unchanged compared to the previous version.
        This function now also returns a dictionary with individual reward terms
        so that reward behavior can be inspected during rollouts.
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

        # Initialize diagnostics with default values.
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

            # Reward terms
            "reward_position": 0.0,
            "reward_velocity": 0.0,
            "reward_s_dot": 0.0,
            "reward_too_fast": 0.0,
            "reward_moving_away": 0.0,
            "reward_lateral": 0.0,
            "reward_lateral_speed": 0.0,
            "reward_attitude": 0.0,
            "reward_rate": 0.0,
            "reward_action": 0.0,
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

            reward_position = -0.002 * pos_error
            reward_velocity = -0.5 * vel_error
            reward_attitude = -0.5 * attitude_error
            reward_rate = -0.1 * rate_error
            reward_action = -0.01 * action_norm
            v_lateral = abs(q["v_p"][1])
            reward_lateral_speed = -10.0 * v_lateral

            reward = (
                reward_position
                + reward_velocity
                + reward_attitude
                + reward_rate
                + reward_action
                + reward_lateral_speed
            )

            reward_components.update(
                {
                    "pos_error": float(pos_error),
                    "vel_error": float(vel_error),
                    "attitude_error": float(attitude_error),
                    "rate_error": float(rate_error),
                    "reward_position": float(reward_position),
                    "reward_velocity": float(reward_velocity),
                    "reward_attitude": float(reward_attitude),
                    "reward_rate": float(reward_rate),
                    "reward_action": float(reward_action),
                    "reward_lateral_speed": float(reward_lateral_speed),
                }
            )

        elif self.phase == "final":
            # ------------------------------------------------------------
            # Reward v2 for final approach
            # ------------------------------------------------------------
            #
            # Main design idea:
            #   Reward v1 encouraged reducing port_distance, but this term dominated
            #   the reward and led to unsafe velocity/alignment behaviour.
            #
            # Reward v2 therefore:
            #   - reduces the dominance of the distance term
            #   - adds desired closing-speed shaping using s_dot
            #   - penalizes lateral offset rho
            #   - strengthens attitude/alignment penalties
            #   - keeps action effort penalty small
            # ------------------------------------------------------------

            s = q["s"]
            rho = q["rho"]
            s_dot = q["s_dot"]
            v_lateral = abs(q["v_p"][1])

            # Distance-dependent desired axial closing speed.
            # Convention: s_dot < 0 means approaching the docking point.
            #
            # Far away: allow a slightly faster approach.
            # Closer in: require slower approach.
            if s > 200.0:
                desired_s_dot = -0.035
            elif s > 100.0:
                desired_s_dot = -0.025
            elif s > 30.0:
                desired_s_dot = -0.015
            else:
                desired_s_dot = -0.005

            s_dot_error = abs(s_dot - desired_s_dot)

            # Extra penalty only when approaching too fast.
            # If s_dot is more negative than desired, the vehicle is closing too quickly.
            too_fast = max(0.0, desired_s_dot - s_dot)

            # Extra penalty if moving away from the docking point.
            # If s_dot > 0, the vehicle is increasing axial distance.
            moving_away = max(0.0, s_dot)

            # Reward terms.
            #
            # Compared to v1:
            #   - position coefficient reduced from 2.0 to 0.5
            #   - port speed coefficient increased from 1.0 to 5.0
            #   - attitude coefficient increased from 0.5 to 5.0
            #   - rate coefficient increased from 0.1 to 1.0
            #   - added s_dot tracking
            #   - added too-fast penalty
            #   - added moving-away penalty
            #   - added lateral rho penalty
            reward_position = -0.5 * port_distance
            reward_velocity = -5.0 * port_speed
            reward_s_dot = -20.0 * s_dot_error
            reward_too_fast = -50.0 * too_fast
            reward_moving_away = -20.0 * moving_away
            reward_lateral = -1.0 * rho - 0.05 * rho**2
            reward_attitude = -15.0 * psi_abs
            reward_rate = -3.0 * omega_abs
            reward_action = -0.01 * action_norm
            reward_lateral_speed = -10.0 * v_lateral

            reward = (
                reward_position
                + reward_velocity
                + reward_s_dot
                + reward_too_fast
                + reward_moving_away
                + reward_lateral
                + reward_attitude
                + reward_rate
                + reward_action
                + reward_lateral_speed 
            )

            reward_components.update(
                {
                    # Error diagnostics
                    "pos_error": float(port_distance),
                    "vel_error": float(port_speed),
                    "attitude_error": float(psi_abs),
                    "rate_error": float(omega_abs),

                    # Final-approach-specific diagnostics
                    "desired_s_dot": float(desired_s_dot),
                    "s_dot_error": float(s_dot_error),
                    "too_fast": float(too_fast),
                    "moving_away": float(moving_away),
                    "rho": float(rho),
                    "s": float(s),
                    "s_dot": float(s_dot),
                    "v_lateral": float(v_lateral),

                    # Reward components
                    "reward_position": float(reward_position),
                    "reward_velocity": float(reward_velocity),
                    "reward_s_dot": float(reward_s_dot),
                    "reward_too_fast": float(reward_too_fast),
                    "reward_moving_away": float(reward_moving_away),
                    "reward_lateral": float(reward_lateral),
                    "reward_attitude": float(reward_attitude),
                    "reward_rate": float(reward_rate),
                    "reward_action": float(reward_action),
                    "reward_lateral_speed": float(reward_lateral_speed),
                }
            )

        else:
            raise ValueError("Unknown phase.")

        if info.get("warning", False):
            reward_warning = -10.0
            reward += reward_warning
            reward_components["reward_warning"] = float(reward_warning)

        if info.get("failure", False):
            reward_failure = -100.0
            reward += reward_failure
            reward_components["reward_failure"] = float(reward_failure)

        if info.get("success", False):
            reward_success = 200.0
            reward += reward_success
            reward_components["reward_success"] = float(reward_success)

        if truncated:
            reward_timeout = -10.0
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