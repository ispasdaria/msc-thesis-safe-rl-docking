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

        # SAC output actions in [-1, 1], then scale them to the physical
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

        reward = self._compute_sac_reward(
            state=self.core_env.state,
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        info["normalized_action"] = np.asarray(action, dtype=np.float32)
        info["physical_action"] = physical_action.astype(np.float32)

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
        First SAC reward version.

        This is intentionally simple and stable:
        - penalize distance/error to the target
        - penalize velocity
        - penalize attitude error
        - penalize control effort
        - penalize warning regions
        - strongly penalize failure
        - strongly reward success
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

            reward = (
                -0.002 * pos_error
                -0.5 * vel_error
                -0.5 * attitude_error
                -0.1 * rate_error
                -0.01 * action_norm
            )

        elif self.phase == "final":
            reward = (
                -2.0 * port_distance
                -1.0 * port_speed
                -0.5 * psi_abs
                -0.1 * omega_abs
                -0.01 * action_norm
            )

        else:
            raise ValueError("Unknown phase.")

        if info.get("warning", False):
            reward -= 1.0

        if info.get("failure", False):
            reward -= 100.0

        if info.get("success", False):
            reward += 200.0

        if truncated:
            reward -= 10.0

        return reward

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