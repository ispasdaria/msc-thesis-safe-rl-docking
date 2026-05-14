import numpy as np


class Docking3DoFEnv:
    """
    3-DoF target-centered orbital-frame docking environment.

    Frame convention:
        F_O = {V-bar, H-bar, R-bar}

        x = V-bar = along-track / docking-axis direction / approximately target velocity direction
        y = H-bar = orbit-normal direction
        z = R-bar = radial direction toward Earth

    This is a simplified 3-DoF nonlinear relative dynamics in the V-H plane.
    The radial displacement along R-bar is assumed to be zero.
    Therefore, the chaser inertial radius is approximated as:

        rc = sqrt(r0^2 + rx^2 + ry^2)

    This model keeps the along-track and orbit-normal relative motion while
    neglecting radial relative motion and the full rotating-frame coupling terms.

    State:
        state = [rx, ry, vx, vy, psi, omegaz]

        rx: relative position along V-bar of chaser CoM w.r.t. target CoM [m]
        ry: relative position along H-bar of chaser CoM w.r.t. target CoM [m]
        vx: relative velocity along V-bar w.r.t. target CoM [m/s]
        vy: relative velocity along H-bar w.r.t. target CoM [m/s]
        psi: relative attitude angle about R-bar [rad]
        omegaz: relative angular velocity about R-bar [rad/s]

    Action:
        action = [ax_Bc, ay_Bc, alphaz_Bc]

        ax_Bc: commanded translational acceleration along the chaser body x-axis [m/s^2]
        ay_Bc: commanded translational acceleration along the chaser body y-axis [m/s^2]
        alphaz_Bc: commanded angular acceleration about the chaser body z-axis / R-bar direction [rad/s^2]

        The action is clipped in the chaser body frame and then transformed into
        the target-centered orbital frame before being used in the translational EoM.

    Phase logic:
        phase = "closing":
            initial condition is sampled from X_P2
            successful terminal condition is reaching X_P3

        phase = "final":
            initial condition is sampled from X_P3
            successful terminal condition is reaching X_dock
    """

    def __init__(
        self,
        dt=1.0,
        max_time=3600.0,
        mu=3.986004418e14,
        r0=6_778_000.0,
        ax_max=0.01,
        ay_max=0.01,
        alphaz_max=0.003,
        integrator="rk4",
        phase="closing",
        r_c_body=np.array([4.1, 0.0]),
        r_t_orbital=np.array([20.0, 0.0]),
        seed=None,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if max_time <= 0.0:
            raise ValueError("max_time must be positive.")
        if mu <= 0.0:
            raise ValueError("mu must be positive.")
        if r0 <= 0.0:
            raise ValueError("r0 must be positive.")

        ax_max = abs(ax_max)
        ay_max = abs(ay_max)
        alphaz_max = abs(alphaz_max)

        if not np.isfinite(ax_max) or not np.isfinite(ay_max) or not np.isfinite(alphaz_max):
            raise ValueError("Control limits must be finite.")

        if integrator not in ["euler", "rk4"]:
            raise ValueError("integrator must be either 'euler' or 'rk4'.")

        if phase not in ["closing", "final"]:
            raise ValueError("phase must be either 'closing' or 'final'.")

        self.dt = dt
        self.max_time = max_time

        # Phase definition
        self.phase = phase

        # Earth gravitational parameter and target orbital radius
        self.mu = mu
        self.r0 = r0

        # Circular-orbit mean motion
        self.n = np.sqrt(self.mu / self.r0**3)

        # Control limits in the chaser body frame
        self.ax_max = ax_max
        self.ay_max = ay_max
        self.alphaz_max = alphaz_max

        self.action_low = np.array(
            [-self.ax_max, -self.ay_max, -self.alphaz_max],
            dtype=float,
        )

        self.action_high = np.array(
            [self.ax_max, self.ay_max, self.alphaz_max],
            dtype=float,
        )

        self.integrator = integrator

        # Docking-port geometry in the reduced 2D V-H plane.
        #
        # r_c_body:
        #     Chaser docking-port center relative to chaser CoM,
        #     expressed in the chaser body frame.
        #
        # r_t_orbital:
        #     Target docking-port center relative to target CoM,
        #     expressed in the target-centered orbital frame.
        #
        # If both are zero, the docking-port relative position reduces to
        # the CoM relative position [rx, ry].
        if r_c_body is None:
            self.r_c_body = np.zeros(2, dtype=float)
        else:
            self.r_c_body = np.asarray(r_c_body, dtype=float)
            if self.r_c_body.shape != (2,):
                raise ValueError("r_c_body must have shape (2,), i.e. [x_c, y_c].")

        if r_t_orbital is None:
            self.r_t_orbital = np.zeros(2, dtype=float)
        else:
            self.r_t_orbital = np.asarray(r_t_orbital, dtype=float)
            if self.r_t_orbital.shape != (2,):
                raise ValueError("r_t_orbital must have shape (2,), i.e. [x_t, y_t].")

        self.rng = np.random.default_rng(seed)

        self.state = None
        self.time = 0.0
        self.history = []


    def wrap_to_pi(self, angle_rad):
        return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi
    

    def rotation_body_to_orbital_2d(self, psi):
        """
        Returns the 2D rotation matrix from the chaser body frame to the
        target-centered orbital frame.

        At nominal alignment psi = 0, the chaser body x-axis points opposite
        to the orbital x-axis. Therefore:

            ax_O = -ax_Bc
            ay_O = -ay_Bc

        for psi = 0.

        Matrix:

            R_Bc_to_O(psi) =
                [ -cos(psi)    sin(psi) ]
                [ -sin(psi)   -cos(psi) ]
        """

        c = np.cos(psi)
        s = np.sin(psi)

        return np.array(
            [
                [-c,  s],
                [-s, -c],
            ],
            dtype=float,
        )
    

    def clip_action(self, action):
        """
        Clips the commanded action in the chaser body frame.

        Input:
            action = [ax_Bc, ay_Bc, alphaz_Bc]

        Output:
            action_clipped = [ax_Bc_clipped, ay_Bc_clipped, alphaz_Bc_clipped]
        """

        action = np.asarray(action, dtype=float)

        if action.shape != (3,):
            raise ValueError(
                "Action must have shape (3,), "
                "i.e. [ax_Bc, ay_Bc, alphaz_Bc]."
            )

        return np.clip(action, self.action_low, self.action_high)
    

    def transform_action_to_orbital_frame(self, action_body, psi):
        """
        Transforms the clipped chaser-body-frame action into the
        target-centered orbital frame.

        Body-frame action:
            action_body = [ax_Bc, ay_Bc, alphaz_Bc]

        Orbital-frame action:
            action_orbital = [ax_O, ay_O, alphaz_O]

        The translational part is transformed using R_Bc_to_O(psi).
        The angular acceleration is kept as alphaz_O = alphaz_Bc, because
        the reduced model only includes rotation about the common R-bar axis.
        """

        action_body = np.asarray(action_body, dtype=float)

        if action_body.shape != (3,):
            raise ValueError(
                "Body-frame action must have shape (3,), "
                "i.e. [ax_Bc, ay_Bc, alphaz_Bc]."
            )

        R_Bc_to_O = self.rotation_body_to_orbital_2d(psi)

        a_body = action_body[:2]
        alphaz_body = action_body[2]

        a_orbital = R_Bc_to_O @ a_body

        action_orbital = np.array(
            [
                a_orbital[0],
                a_orbital[1],
                alphaz_body,
            ],
            dtype=float,
        )

        return action_orbital
    

    def get_port_relative_position(self, state):
        """
        Computes the chaser docking-port position relative to the target
        docking-port position in the target-centered orbital frame.

        Reduced 2D version of:

            r_p = r + R(q) r_c - r_t

        where:
            r   = chaser CoM position relative to target CoM
            R   = rotation from chaser body frame to target/orbital frame
            r_c = chaser docking-port position relative to chaser CoM
            r_t = target docking-port position relative to target CoM

        In this 3-DoF model:

            r   = [rx, ry]
            R   = R_Bc_to_O(psi)
            r_c = self.r_c_body
            r_t = self.r_t_orbital
        """

        rx, ry, vx, vy, psi, omegaz = state

        r_com = np.array([rx, ry], dtype=float)

        R_Bc_to_O = self.rotation_body_to_orbital_2d(psi)
        r_c_orbital = R_Bc_to_O @ self.r_c_body

        r_p = r_com + r_c_orbital - self.r_t_orbital

        return r_p
    

    def get_port_relative_velocity(self, state):
        """
        Computes the relative velocity of the chaser docking port with respect
        to the target docking port in the target-centered orbital frame.

        Reduced 2D version of:

            v_p = v + omega x (R(q) r_c)

        In this 3-DoF model:

            v = [vx, vy]
            omega = [0, 0, omegaz]
            R(q) r_c = r_c_orbital = [x_c, y_c]

        Therefore:

            omega x r_c_orbital =
                [-omegaz * y_c,
                  omegaz * x_c]

        The target is passive and fixed in the target-centered frame, so the
        target docking-port velocity is zero in this relative-frame model.
        """

        rx, ry, vx, vy, psi, omegaz = state

        v_com = np.array([vx, vy], dtype=float)

        R_Bc_to_O = self.rotation_body_to_orbital_2d(psi)
        r_c_orbital = R_Bc_to_O @ self.r_c_body

        omega_cross_r_c = np.array(
            [
                -omegaz * r_c_orbital[1],
                 omegaz * r_c_orbital[0],
            ],
            dtype=float,
        )

        v_p = v_com + omega_cross_r_c

        return v_p
    

    def dynamics(self, state, action):
        """
        Computes state derivative using the simplified nonlinear 3-DoF EoM.

        The input action is interpreted as a chaser-body-frame command:

            action = [ax_Bc, ay_Bc, alphaz_Bc]

        The action is first clipped in the chaser body frame. Then it is
        transformed into the target-centered orbital frame:

            action_orbital = [ax_O, ay_O, alphaz_O]

        The transformed action is then used in the equations of motion.
        """

        rx, ry, vx, vy, psi, omegaz = state

        # 1. Clip in chaser body frame
        action_body_clipped = self.clip_action(action)

        # 2. Transform body-frame action into orbital-frame action
        action_orbital = self.transform_action_to_orbital_frame(
            action_body=action_body_clipped,
            psi=psi,
        )

        ax, ay, alphaz = action_orbital

        rc = np.sqrt(rx**2 + ry**2 + self.r0**2)

        rx_dot = vx
        ry_dot = vy

        vx_dot = -self.mu * rx / rc**3 + self.n**2 * rx + ax
        vy_dot = -self.mu * ry / rc**3 + ay

        psi_dot = omegaz
        omegaz_dot = alphaz

        state_dot = np.array(
            [
                rx_dot,
                ry_dot,
                vx_dot,
                vy_dot,
                psi_dot,
                omegaz_dot,
            ],
            dtype=float,
        )

        return state_dot
    

    def integrate(self, state, action):
        """
        Propagates the state by one timestep.
        """

        if self.integrator == "euler":
            return state + self.dt * self.dynamics(state, action)

        if self.integrator == "rk4":
            dt = self.dt

            k1 = self.dynamics(state, action)
            k2 = self.dynamics(state + 0.5 * dt * k1, action)
            k3 = self.dynamics(state + 0.5 * dt * k2, action)
            k4 = self.dynamics(state + dt * k3, action)

            return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        raise ValueError("integrator must be either 'euler' or 'rk4'.")
    

    def sample_X_P2(self, mode):
        """
        Samples an initial state from the P2 initialization set.

        Reduced 2D interpretation:

            s          = rx
            rho        = |ry|
            v_parallel = vx
            v_perp     = vy
            psi_e      = psi
            omega      = omegaz
        """

        if mode == "nominal":
            rx0 = 1700.0
            ry0 = 0.0

            vx0 = -0.10
            vy0 = 0.0

            psi0 = np.deg2rad(0.0)
            omegaz0 = np.deg2rad(0.0)

        elif mode == "testing":
            rx0 = self.rng.uniform(1600.0, 1800.0)
            ry0 = self.rng.uniform(-100.0, 100.0)

            vx0 = self.rng.uniform(-0.15, -0.05)
            vy0 = self.rng.uniform(-0.02, 0.02)

            psi0 = np.deg2rad(self.rng.uniform(-3.0, 3.0))
            omegaz0 = np.deg2rad(self.rng.uniform(-0.05, 0.05))

        elif mode == "training":
            rx0 = self.rng.uniform(1500.0, 1900.0)
            ry0 = self.rng.uniform(-150.0, 150.0)

            vx0 = self.rng.uniform(-0.20, -0.03)
            vy0 = self.rng.uniform(-0.03, 0.03)

            psi0 = np.deg2rad(self.rng.uniform(-5.0, 5.0))
            omegaz0 = np.deg2rad(self.rng.uniform(-0.10, 0.10))

        else:
            raise ValueError(
                "mode must be one of: 'nominal', 'testing', or 'training'."
            )

        return np.array(
            [rx0, ry0, vx0, vy0, psi0, omegaz0],
            dtype=float,
        )
    

    def sample_X_P3(self, mode):
        """
        Samples an initial state from the P3 initialization set.

        Used as the reset set for the final approach phase.

        Reduced 2D interpretation:

            s          = rx
            rho        = |ry|
            v_parallel = vx
            v_perp     = vy
            psi_e      = psi
            omega      = omegaz
        """

        if mode == "nominal":
            rx0 = 300.0
            ry0 = 0.0

            vx0 = -0.03
            vy0 = 0.0

            psi0 = np.deg2rad(0.0)
            omegaz0 = np.deg2rad(0.0)

        elif mode == "testing":
            rx0 = self.rng.uniform(290.0, 310.0)
            ry0 = self.rng.uniform(-3.0, 3.0)

            vx0 = self.rng.uniform(-0.05, -0.01)
            vy0 = self.rng.uniform(-0.005, 0.005)

            psi0 = np.deg2rad(self.rng.uniform(-2.0, 2.0))
            omegaz0 = np.deg2rad(self.rng.uniform(-0.03, 0.03))

        elif mode == "training":
            rx0 = self.rng.uniform(280.0, 320.0)
            ry0 = self.rng.uniform(-5.0, 5.0)

            vx0 = self.rng.uniform(-0.06, -0.01)
            vy0 = self.rng.uniform(-0.01, 0.01)

            psi0 = np.deg2rad(self.rng.uniform(-3.0, 3.0))
            omegaz0 = np.deg2rad(self.rng.uniform(-0.05, 0.05))

        else:
            raise ValueError(
                "mode must be one of: 'nominal', 'testing', or 'training'."
            )

        return np.array(
            [rx0, ry0, vx0, vy0, psi0, omegaz0],
            dtype=float,
        )
    

    def is_in_X_P3(self, state, mode="testing"):
        """
        Checks whether the state belongs to the P3 set.

        For the closing phase, this is used as the successful terminal set.

        Reduced 2D interpretation:

            s          = rx
            rho        = |ry|
            v_parallel = vx
            v_perp     = vy
            psi_e      = psi
            omega      = omegaz
        """

        rx, ry, vx, vy, psi, omegaz = state

        if mode == "nominal":
            d_min = 300.0
            d_max = 300.0
            rho_max = 0.0
            v_min = -0.03
            v_max = -0.03
            v_perp_max = 0.0
            psi_max = np.deg2rad(0.0)
            omega_max = np.deg2rad(0.0)

        elif mode == "testing":
            d_min = 290.0
            d_max = 310.0
            rho_max = 3.0
            v_min = -0.05
            v_max = -0.01
            v_perp_max = 0.005
            psi_max = np.deg2rad(2.0)
            omega_max = np.deg2rad(0.03)

        elif mode == "training":
            d_min = 280.0
            d_max = 320.0
            rho_max = 5.0
            v_min = -0.06
            v_max = -0.01
            v_perp_max = 0.01
            psi_max = np.deg2rad(3.0)
            omega_max = np.deg2rad(0.05)

        else:
            raise ValueError("mode must be one of: 'nominal', 'testing', or 'training'.")

        return (
            d_min <= rx <= d_max
            and abs(ry) <= rho_max
            and v_min <= vx <= v_max
            and abs(vy) <= v_perp_max
            and abs(self.wrap_to_pi(psi)) <= psi_max
            and abs(omegaz) <= omega_max
        )
    

    def is_in_X_dock(self, state):
        """
        Checks whether the state satisfies the terminal docking set.

        Terminal docking set:

            X_dock = {
                ||r_p|| <= 0.1 m,
                ||v_p|| <= 0.1 m/s,
                |psi_e| <= 4 deg,
                |omega| <= 0.2 deg/s
            }

        The terminal constraints are imposed at docking-port level, not
        only at CoM level.
        """

        rx, ry, vx, vy, psi, omegaz = state

        r_p = self.get_port_relative_position(state)
        v_p = self.get_port_relative_velocity(state)

        port_distance = np.linalg.norm(r_p)
        port_speed = np.linalg.norm(v_p)

        return (
            port_distance <= 0.1
            and port_speed <= 0.1
            and abs(self.wrap_to_pi(psi)) <= np.deg2rad(4.0)
            and abs(omegaz) <= np.deg2rad(0.2)
        )
    

    def reset(self, initial_state=None, mode="training"):
        """
        Resets the environment.

        Parameters
        ----------
        initial_state : np.ndarray or None
            If provided, this state is used directly.
            Expected shape: (6,)
            State order: [rx, ry, vx, vy, psi, omegaz]

        mode : str
            Defines the initialization range used when initial_state is None.

            Options:
                "nominal"  : deterministic nominal state
                "testing"  : randomized initial state within testing bounds
                "training" : randomized initial state within training bounds

        Phase-dependent behavior:
            phase = "closing":
                reset samples from X_P2

            phase = "final":
                reset samples from X_P3
        """

        self.time = 0.0
        self.history = []

        if initial_state is not None:
            initial_state = np.asarray(initial_state, dtype=float)

            if initial_state.shape != (6,):
                raise ValueError(
                    "Initial state must have shape (6,), "
                    "i.e. [rx, ry, vx, vy, psi, omegaz]."
                )

            self.state = initial_state.copy()
            self.state[4] = self.wrap_to_pi(self.state[4])

        else:
            if self.phase == "closing":
                self.state = self.sample_X_P2(mode)

            elif self.phase == "final":
                self.state = self.sample_X_P3(mode)

            else:
                raise ValueError("Unknown phase.")

        zero_action_body_raw = np.zeros(3)
        zero_action_body_clipped = self.clip_action(zero_action_body_raw)

        zero_action_orbital = self.transform_action_to_orbital_frame(
            action_body=zero_action_body_clipped,
            psi=self.state[4],
        )

        initial_reward = self.compute_reward(self.state, zero_action_body_clipped)

        initial_info = {
            "phase": self.phase,
            "initial_set": "X_P2" if self.phase == "closing" else "X_P3",
            "port_relative_position": self.get_port_relative_position(self.state),
            "port_relative_velocity": self.get_port_relative_velocity(self.state),
            "port_distance": np.linalg.norm(self.get_port_relative_position(self.state)),
            "port_speed": np.linalg.norm(self.get_port_relative_velocity(self.state)),
        }

        self.log_step(
            action_body_raw=zero_action_body_raw,
            action_body_clipped=zero_action_body_clipped,
            action_orbital=zero_action_orbital,
            reward=initial_reward,
            done=False,
            info=initial_info,
        )

        return self.state.copy()
    

    def step(self, action):
        """
        Applies one control input and advances the simulator by one timestep.

        The external action is interpreted as a chaser-body-frame command:

            action = [ax_Bc, ay_Bc, alphaz_Bc]

        The action is clipped in the chaser body frame and transformed into the
        target-centered orbital frame inside dynamics().
        """

        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action = np.asarray(action, dtype=float)

        if action.shape != (3,):
            raise ValueError(
                "Action must have shape (3,), "
                "i.e. [ax_Bc, ay_Bc, alphaz_Bc]."
            )

        # For logging and reward calculation at the start of the step
        psi_before = self.state[4]

        action_body_clipped = self.clip_action(action)

        action_orbital_before_step = self.transform_action_to_orbital_frame(
            action_body=action_body_clipped,
            psi=psi_before,
        )

        # Propagate using the original action.
        # dynamics() will clip in the body frame and transform internally.
        self.state = self.integrate(self.state, action)

        # Keep attitude angle bounded for numerical clarity
        self.state[4] = self.wrap_to_pi(self.state[4])

        self.time += self.dt

        # Numerical validity check
        if not np.all(np.isfinite(self.state)):
            reward = 0.0

            done = True
            info = {
                "success": False,
                "failure": True,
                "time_limit": False,
                "failure_reason": "non_finite_state",
                "phase": self.phase,
            }

            info["action_body_raw"] = action.copy()
            info["action_body_clipped"] = action_body_clipped.copy()
            info["action_orbital_before_step"] = action_orbital_before_step.copy()

            self.log_step(
                action_body_raw=action,
                action_body_clipped=action_body_clipped,
                action_orbital=action_orbital_before_step,
                reward=reward,
                done=done,
                info=info,
            )

            return self.state.copy(), reward, done, info

        reward = self.compute_reward(self.state, action_body_clipped)

        done, info = self.check_done(self.state)

        info["action_body_raw"] = action.copy()
        info["action_body_clipped"] = action_body_clipped.copy()
        info["action_orbital_before_step"] = action_orbital_before_step.copy()

        self.log_step(
            action_body_raw=action,
            action_body_clipped=action_body_clipped,
            action_orbital=action_orbital_before_step,
            reward=reward,
            done=done,
            info=info,
        )

        return self.state.copy(), reward, done, info
    

    def compute_reward(self, state, action):
        """
        Preliminary reward function.

        Here, action is the clipped chaser-body-frame action:

            action = [ax_Bc, ay_Bc, alphaz_Bc]

        This is intentionally still a placeholder. Reward shaping, terminal
        bonuses, penalties, and safety-set penalties can be refined later.
        """

        rx, ry, vx, vy, psi, omegaz = state
        ax_Bc, ay_Bc, alphaz_Bc = action

        position_error = np.sqrt(rx**2 + ry**2)
        velocity_error = np.sqrt(vx**2 + vy**2)
        attitude_error = abs(self.wrap_to_pi(psi))
        angular_rate_error = abs(omegaz)
        control_effort = np.sqrt(ax_Bc**2 + ay_Bc**2 + alphaz_Bc**2)

        reward = (
            -1.0 * position_error
            -10.0 * velocity_error
            -5.0 * attitude_error
            -2.0 * angular_rate_error
            -0.1 * control_effort
        )

        return reward
    

    def check_done(self, state):
        """
        Checks whether the episode should terminate.

        Phase-dependent success logic:

            Closing phase:
                success if state enters X_P3.

            Final approach phase:
                success if state enters X_dock.

        Failure and safety-set logic are still placeholders and can be refined
        later when X_safe_close and X_safe_final are implemented.
        """

        rx, ry, vx, vy, psi, omegaz = state

        r_p = self.get_port_relative_position(state)
        v_p = self.get_port_relative_velocity(state)

        port_distance = np.linalg.norm(r_p)
        port_speed = np.linalg.norm(v_p)

        com_distance = np.sqrt(rx**2 + ry**2)
        com_speed = np.sqrt(vx**2 + vy**2)

        info = {
            "success": False,
            "failure": False,
            "time_limit": False,
            "phase": self.phase,
            "port_relative_position": r_p,
            "port_relative_velocity": v_p,
            "port_distance": port_distance,
            "port_speed": port_speed,
            "com_distance": com_distance,
            "com_speed": com_speed,
        }

        if self.phase == "closing":
            if self.is_in_X_P3(state, mode="testing"):
                info["success"] = True
                info["terminal_set"] = "X_P3"
                return True, info

        elif self.phase == "final":
            if self.is_in_X_dock(state):
                info["success"] = True
                info["terminal_set"] = "X_dock"
                return True, info

        else:
            raise ValueError("Unknown phase.")

        # Placeholder failure condition: too far from target.
        # More realistic phase-dependent safety failures can be added later.
        if com_distance > 5000.0:
            info["failure"] = True
            info["failure_reason"] = "too_far_from_target"
            return True, info

        # Time limit
        if self.time >= self.max_time:
            info["time_limit"] = True
            return True, info

        return False, info
    

    def log_step(
        self,
        action_body_raw,
        action_body_clipped,
        action_orbital,
        reward,
        done,
        info,
    ):
        """
        Stores simulation history for plotting and debugging.

        The action is logged in three forms:

            action_body_raw:
                Raw external command before clipping,
                expressed in the chaser body frame.

            action_body_clipped:
                Command after applying body-frame acceleration limits,
                still expressed in the chaser body frame.

            action_orbital:
                Clipped command transformed into the target-centered
                orbital frame. This is the translational/angular acceleration
                actually used by the equations of motion.
        """

        self.history.append(
            {
                "time": self.time,
                "state": self.state.copy(),
                "action_body_raw": np.asarray(action_body_raw, dtype=float).copy(),
                "action_body_clipped": np.asarray(action_body_clipped, dtype=float).copy(),
                "action_orbital": np.asarray(action_orbital, dtype=float).copy(),
                "reward": reward,
                "done": done,
                "info": info.copy(),
            }
        )
        

    def get_history_arrays(self):
        """
        Converts logged history into arrays for plotting.
        """

        times = np.array([entry["time"] for entry in self.history])
        states = np.array([entry["state"] for entry in self.history])

        actions_body_raw = np.array(
            [entry["action_body_raw"] for entry in self.history]
        )

        actions_body_clipped = np.array(
            [entry["action_body_clipped"] for entry in self.history]
        )

        actions_orbital = np.array(
            [entry["action_orbital"] for entry in self.history]
        )

        rewards = np.array([entry["reward"] for entry in self.history])

        return (
            times,
            states,
            actions_body_raw,
            actions_body_clipped,
            actions_orbital,
            rewards,
        )