import numpy as np 

class Docking3DoFEnv: 
    """ 
    3-DoF target-centered orbital-frame docking environment. 

    Frame convention: 
        F_O = {V-bar, H-bar, R-bar}

        x = V-bar = along-track / docking-axis direction / approximately target velocity direction 
        y = H-bar = orbit-normal direction 
        z = R-bar = radial direction toward Earth

    State: 
        state = [rx, ry, vx, vy, psi, omegaz]

        rx: relative position along V-bar of chaser CoM w.r.t. target CoM [m] 
        ry: relative position along H-bar of chaser CoM w.r.t. target CoM [m]
        vx: relative velocity along V-bar w.r.t. target CoM [m/s]
        vy: relative velocity along H-bar w.r.t. target CoM [m/s]
        psi: relative attitude angle about R-bar [rad]
        omegaz: relative angular velocity about R-bar [rad/s]

    Action: 
        action = [ax, ay, alphaz]
        ax: commanded translational acceleration along V-bar [m/s^2]
        ay: commanded translational acceleration along H-bar [m/s^2]
        alphaz: commanded angular acceleration about R-bar [rad/s^2]
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
        if ax_max == 0.0 or ay_max == 0.0 or alphaz_max == 0.0:
            raise ValueError("Control limits must be positive.")

        self.dt = dt
        self.max_time = max_time

        # Earth gravitational parameter and target orbital radius
        self.mu = mu
        self.r0 = r0

        # Circular-orbit mean motion
        self.n = np.sqrt(self.mu / self.r0**3)

        # Control limits
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

        if integrator not in ["euler", "rk4"]:
            raise ValueError("integrator must be either 'euler' or 'rk4'.")

        self.integrator = integrator

        self.rng = np.random.default_rng(seed)

        self.state = None
        self.time = 0.0
        self.history = []

    def wrap_to_pi(self, angle_rad):
        return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    def dynamics(self, state, action):
        """
        Computes state derivative using the nonlinear 3-DoF EoM.

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

        Transformation:
            ax_O = -cos(psi) * ax_Bc + sin(psi) * ay_Bc
            ay_O = -sin(psi) * ax_Bc - cos(psi) * ay_Bc
            alphaz_O = alphaz_Bc
        """

        ax_Bc, ay_Bc, alphaz_Bc = action_body

        c = np.cos(psi)
        s = np.sin(psi)

        ax_O = -c * ax_Bc + s * ay_Bc
        ay_O = -s * ax_Bc - c * ay_Bc
        alphaz_O = alphaz_Bc

        return np.array([ax_O, ay_O, alphaz_O], dtype=float)

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

            return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        raise ValueError("integrator must be either 'euler' or 'rk4'.")

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
            Defines the P2 initialization range used when initial_state is None.

            Options:
                "nominal"  : deterministic nominal P2 state
                "testing"  : randomized initial state within P2 testing bounds
                "training" : randomized initial state within P2 training bounds
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

        else:
            if mode == "nominal":
                rx0 = 1700.0
                ry0 = 0.0

                vx0 = -0.10
                vy0 = 0.0

                psi0 = np.deg2rad(0.0)
                omegaz0 = np.deg2rad(0.0)

            elif mode == "testing":
                rx0 = self.rng.uniform(1600.0, 1800.0)      # m
                ry0 = self.rng.uniform(-100.0, 100.0)       # m

                vx0 = self.rng.uniform(-0.15, -0.05)        # m/s
                vy0 = self.rng.uniform(-0.02, 0.02)         # m/s

                psi0 = np.deg2rad(self.rng.uniform(-3.0, 3.0))          # rad
                omegaz0 = np.deg2rad(self.rng.uniform(-0.05, 0.05))     # rad/s

            elif mode == "training":
                rx0 = self.rng.uniform(1500.0, 1900.0)      # m
                ry0 = self.rng.uniform(-150.0, 150.0)       # m

                vx0 = self.rng.uniform(-0.20, -0.03)        # m/s
                vy0 = self.rng.uniform(-0.03, 0.03)         # m/s

                psi0 = np.deg2rad(self.rng.uniform(-5.0, 5.0))          # rad
                omegaz0 = np.deg2rad(self.rng.uniform(-0.10, 0.10))     # rad/s

            else:
                raise ValueError(
                    "mode must be one of: 'nominal', 'testing', or 'training'."
                )

            self.state = np.array(
                [rx0, ry0, vx0, vy0, psi0, omegaz0],
                dtype=float,
            )

        zero_action = np.zeros(3)
        initial_reward = self.compute_reward(self.state, zero_action)
        self.log_step(action=zero_action, reward=initial_reward, done=False, info={})

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

        self.time += self.dt

        # Reward uses body-frame clipped effort, because the physical limits
        # are defined in the chaser body frame.
        reward = self.compute_reward(self.state, action_body_clipped)

        done, info = self.check_done(self.state)

        info["action_body_raw"] = action.copy()
        info["action_body_clipped"] = action_body_clipped.copy()
        info["action_orbital_before_step"] = action_orbital_before_step.copy()

        self.log_step(action_body_clipped, reward, done, info)

        return self.state.copy(), reward, done, info

    def compute_reward(self, state, action):
        """
        Preliminary reward function.

        Here, action is the clipped chaser-body-frame action:
            action = [ax_Bc, ay_Bc, alphaz_Bc]
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

        This is a simple placeholder. Later, replace this with your
        docking success/failure/safety-set logic.
        """

        rx, ry, vx, vy, psi, omegaz = state

        distance = np.sqrt(rx**2 + ry**2)
        speed = np.sqrt(vx**2 + vy**2)

        info = {
            "success": False,
            "failure": False,
            "time_limit": False,
        }

        # Example docking success condition
        if (
            distance <= 0.1
            and speed <= 0.1
            and abs(psi) <= np.deg2rad(4.0)
            and abs(omegaz) <= np.deg2rad(0.2)
        ):
            info["success"] = True
            return True, info

        # Example failure condition: too far from target
        if distance > 5000.0:
            info["failure"] = True
            return True, info

        # Time limit
        if self.time >= self.max_time:
            info["time_limit"] = True
            return True, info

        return False, info

    def log_step(self, action, reward, done, info):
        """
        Stores simulation history for plotting and debugging.
        """

        self.history.append(
            {
                "time": self.time,
                "state": self.state.copy(),
                "action": np.asarray(action, dtype=float).copy(),
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
        actions = np.array([entry["action"] for entry in self.history])
        rewards = np.array([entry["reward"] for entry in self.history])

        return times, states, actions, rewards










