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

    Key methods:
        reset(): initialize the episode state for the selected phase
        step(): propagate the dynamics and return (state, reward, done, info)
        monitor_safety(): classify the state into safe/warn/unsafe sets
        get_history_arrays(): export logged trajectories for analysis/plotting
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
        r_c_body=None,
        r_t_orbital=None,
        p3_target_tolerance=None,
        seed=None,
    ):
        """
        Initializes the environment parameters and safety thresholds.
        """
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
            self.r_c_body = np.array([4.1, 0.0], dtype=float)
        else:
            self.r_c_body = np.asarray(r_c_body, dtype=float)
            if self.r_c_body.shape != (2,):
                raise ValueError("r_c_body must have shape (2,), i.e. [x_c, y_c].")

        if r_t_orbital is None:
            self.r_t_orbital = np.array([20.0, 0.0], dtype=float)
        else:
            self.r_t_orbital = np.asarray(r_t_orbital, dtype=float)
            if self.r_t_orbital.shape != (2,):
                raise ValueError("r_t_orbital must have shape (2,), i.e. [x_t, y_t].")

        self.rng = np.random.default_rng(seed)

        # Safety geometry parameters
        self.ae_x = 2000.0      # [m] approach ellipse semi-axis along docking axis
        self.ae_y = 1000.0      # [m] approach ellipse semi-axis lateral
        self.r_kos = 200.0      # [m] keep-out circle radius
        self.cone_length = 2000.0
        self.cone_base_radius = 200.0
        self.cone_slope = self.cone_base_radius / self.cone_length

        # Phase distances (set per episode in reset from port axial distance)
        self.d_P2 = None
        self.d_P3 = None
        self.p3_target = None

        if p3_target_tolerance is None:
            self.p3_target_tolerance = np.array(
                [
                    1.0,
                    0.5,
                    0.01,
                    0.005,
                    np.deg2rad(0.5),
                    np.deg2rad(0.01),
                ],
                dtype=float,
            )
        else:
            self.p3_target_tolerance = np.asarray(p3_target_tolerance, dtype=float)
            if self.p3_target_tolerance.shape != (6,):
                raise ValueError(
                    "p3_target_tolerance must have shape (6,), "
                    "i.e. [rx, ry, vx, vy, psi, omegaz] tolerances."
                )

        # Corridor
        self.corridor_radius = 5.0

        # Alignment limits
        self.psi_safe = np.deg2rad(3.0)
        self.psi_unsafe = np.deg2rad(4.0)

        self.omega_safe = np.deg2rad(0.15)
        self.omega_unsafe = np.deg2rad(0.2)

        # Velocity limits: closing
        self.vp_safe_close = 0.25
        self.vp_unsafe_close = 0.35
        self.sdot_safe_close = 0.15
        self.sdot_unsafe_close = 0.25

        # Velocity limits: final approach
        self.vp_safe_final = 0.15
        self.vp_unsafe_final = 0.2
        self.sdot_safe_final = 0.1
        self.sdot_unsafe_final = 0.15

        self.state = None
        self.time = 0.0
        self.history = []


    ### Helpers and transforms
    def wrap_to_pi(self, angle_rad):
        """
        Wraps an angle to the interval [-pi, pi).
        """
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


    def get_derived_quantities(self, state):
        """
        Computes the 3-DoF quantities used by the safety monitor.

        In the reduced planar case:

            r_p = r + R(psi) r_c - r_t
            v_p = v + omega x (R(psi) r_c)

            e_d = [1, 0]
            s = dot(e_d, r_p) = r_p[0]
            rho = |r_p[1]|

        The sign convention is kept consistent with the current simulator:
        the chaser approaches the docking point from positive x toward smaller x.
        """

        rx, ry, vx, vy, psi, omegaz = state

        r_com = np.array([rx, ry], dtype=float)
        v_com = np.array([vx, vy], dtype=float)

        r_p = self.get_port_relative_position(state)
        v_p = self.get_port_relative_velocity(state)

        s = r_p[0]
        rho = abs(r_p[1])

        # Since e_d = [1, 0], s_dot is the x-component of port velocity.
        s_dot = v_p[0]

        q = {
            "r_com": r_com,
            "v_com": v_com,
            "r_p": r_p,
            "v_p": v_p,
            "s": s,
            "rho": rho,
            "s_dot": s_dot,
            "port_distance": np.linalg.norm(r_p),
            "port_speed": np.linalg.norm(v_p),
            "com_distance": np.linalg.norm(r_com),
            "com_speed": np.linalg.norm(v_com),
            "psi_e": self.wrap_to_pi(psi),
            "omega": omegaz,
        }

        return q


    ### Safety and termination helpers
    def is_in_E_AE(self, q):
        """
        Checks membership in the approach ellipse E_AE in the V-H plane.
        """
        r = q["r_com"]
        value = (r[0] / self.ae_x)**2 + (r[1] / self.ae_y)**2
        return value <= 1.0


    def is_in_S_KOS(self, q):
        """
        Checks membership in the keep-out sphere/circle S_KOS.
        """
        return q["com_distance"] <= self.r_kos


    def is_in_C_dock(self, q):
        """
        Checks membership in the docking cone C_dock.
        """
        s = q["s"]
        rho = q["rho"]

        return (
            0.0 <= s <= self.cone_length
            and rho <= self.cone_slope * s
        )


    def is_in_C_corr(self, q):
        """
        Checks membership in the final-approach corridor C_corr.
        """
        s = q["s"]
        rho = q["rho"]

        return (
            0.0 <= s <= self.d_P3
            and rho <= self.corridor_radius
        )


    def classify_alignment(self, q):
        """
        Classifies attitude alignment as safe/warn/unsafe.
        """
        psi_abs = abs(q["psi_e"])
        omega_abs = abs(q["omega"])

        safe = (
            psi_abs <= self.psi_safe
            and omega_abs <= self.omega_safe
        )

        unsafe = (
            psi_abs > self.psi_unsafe
            or omega_abs > self.omega_unsafe
        )

        warn = (
            not safe
            and not unsafe
        )

        return {
            "safe": safe,
            "warn": warn,
            "unsafe": unsafe,
            "psi_abs": psi_abs,
            "omega_abs": omega_abs,
        }


    def classify_velocity(self, q, phase):
        """
        Classifies relative velocity limits for the given phase.
        """
        vp_norm = q["port_speed"]
        s_dot_abs = abs(q["s_dot"])

        if phase == "closing":
            vp_safe = self.vp_safe_close
            vp_unsafe = self.vp_unsafe_close
            sdot_safe = self.sdot_safe_close
            sdot_unsafe = self.sdot_unsafe_close

        elif phase == "final":
            vp_safe = self.vp_safe_final
            vp_unsafe = self.vp_unsafe_final
            sdot_safe = self.sdot_safe_final
            sdot_unsafe = self.sdot_unsafe_final

        else:
            raise ValueError("phase must be either 'closing' or 'final'.")

        safe = (
            vp_norm <= vp_safe
            and s_dot_abs <= sdot_safe
        )

        unsafe = (
            vp_norm > vp_unsafe
            or s_dot_abs > sdot_unsafe
        )

        warn = (
            not safe
            and not unsafe
        )

        return {
            "safe": safe,
            "warn": warn,
            "unsafe": unsafe,
            "vp_norm": vp_norm,
            "s_dot_abs": s_dot_abs,
        }


    def classify_geometry_closing(self, q):
        """
        Classifies closing-phase geometry as safe/warn/unsafe.
        """
        in_E = self.is_in_E_AE(q)
        in_KOS = self.is_in_S_KOS(q)
        in_cone = self.is_in_C_dock(q)

        s = q["s"]

        in_phase_axial_range = self.d_P3 <= s <= self.d_P2
        passed_P3_region = s < self.d_P3

        safe = (
            in_E
            and not in_KOS
            and in_cone
            and in_phase_axial_range
        )

        unsafe = (
            (in_KOS and not in_cone)
            or not in_E
        )

        warn = (
            (
                in_E
                and not in_KOS
                and not in_cone
            )
            or (
                in_KOS
                and in_cone
            )
            or (
                in_E
                and in_cone
                and passed_P3_region
            )
        )

        warn = warn and not unsafe

        return {
            "safe": safe,
            "warn": warn,
            "unsafe": unsafe,
            "in_E_AE": in_E,
            "in_S_KOS": in_KOS,
            "in_C_dock": in_cone,
            "in_phase_axial_range": in_phase_axial_range,
            "passed_P3_region": passed_P3_region,
        }


    def classify_geometry_final(self, q):
        """
        Classifies final-approach geometry as safe/warn/unsafe.
        """
        in_cone = self.is_in_C_dock(q)
        in_corr = self.is_in_C_corr(q)

        # G_safe_final = C_dock ∩ C_corr
        safe = in_cone and in_corr

        # G_unsafe_final = C_dock_complement
        unsafe = not in_cone

        # G_warn_final = C_dock \ C_corr
        warn = in_cone and not in_corr

        return {
            "safe": safe,
            "warn": warn,
            "unsafe": unsafe,
            "in_C_dock": in_cone,
            "in_C_corr": in_corr,
        }


    def monitor_safety(self, state):
        """
        Classifies the current state as safe, warning, or unsafe.

        This implements:

            Closing:
                X_safe_close = G_safe_close ∩ V_safe_close
                X_unsafe_close = G_unsafe_close ∪ V_unsafe_close
                X_warn_close = (G_warn_close ∪ V_warn_close) minus X_unsafe_close

            Final:
                X_safe_final = G_safe_final ∩ V_safe_final ∩ A_safe
                X_unsafe_final = G_unsafe_final ∪ V_unsafe_final ∪ A_unsafe
                X_warn_final = (G_warn_final ∪ V_warn_final ∪ A_warn) minus X_unsafe_final
        """

        q = self.get_derived_quantities(state)

        if self.phase == "closing":
            geometry = self.classify_geometry_closing(q)
            velocity = self.classify_velocity(q, phase="closing")
            alignment = self.classify_alignment(q)

            safe = geometry["safe"] and velocity["safe"]

            unsafe = geometry["unsafe"] or velocity["unsafe"]

            warn = (
                geometry["warn"]
                or velocity["warn"]
            ) and not unsafe

        elif self.phase == "final":
            geometry = self.classify_geometry_final(q)
            velocity = self.classify_velocity(q, phase="final")
            alignment = self.classify_alignment(q)

            safe = (
                geometry["safe"]
                and velocity["safe"]
                and alignment["safe"]
            )

            unsafe = (
                geometry["unsafe"]
                or velocity["unsafe"]
                or alignment["unsafe"]
            )

            warn = (
                geometry["warn"]
                or velocity["warn"]
                or alignment["warn"]
            ) and not unsafe

        else:
            raise ValueError("Unknown phase.")

        violated_constraints = []

        if geometry["unsafe"]:
            violated_constraints.append("geometry")

        if velocity["unsafe"]:
            violated_constraints.append("velocity")

        if self.phase == "final" and alignment["unsafe"]:
            violated_constraints.append("alignment")

        monitor = {
            "safe": safe,
            "warning": warn,
            "unsafe": unsafe,
            "violated_constraints": violated_constraints,
            "derived": q,
            "geometry": geometry,
            "velocity": velocity,
            "alignment": alignment,
        }

        return monitor


    ### Dynamics and integration
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


    ### Sampling and terminal sets
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


    def is_at_P3_target(self, state):
        """
        Checks whether the state is close to the specific sampled P3 target.
        """

        if self.p3_target is None:
            raise RuntimeError("P3 target has not been sampled yet.")

        state = np.asarray(state, dtype=float)

        return np.all(
            np.abs(state - self.p3_target) <= self.p3_target_tolerance
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


    ### Reset, step, and logging
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

        self.p3_target = None
        self.d_P2 = None
        self.d_P3 = None

        self.time = 0.0
        self.history = []
        self.mode = mode

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
                self.p3_target = self.sample_X_P3(mode)

            elif self.phase == "final":
                self.state = self.sample_X_P3(mode)
                self.p3_target = None

            else:
                raise ValueError("Unknown phase.")

        # Store phase distances as port-axial distance to the docking point.
        q0 = self.get_derived_quantities(self.state)
        if self.phase == "closing":
            self.d_P2 = q0["s"]
            if self.p3_target is None:
                self.p3_target = self.sample_X_P3(mode)
            q3 = self.get_derived_quantities(self.p3_target)
            self.d_P3 = q3["s"]
        elif self.phase == "final":
            self.d_P3 = q0["s"]

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
        Placeholder reward function.

        The actual task-specific reward will be designed later when implementing
        the controllers. For now, this keeps the environment API complete.
        """
        return 0.0


    def check_done(self, state):
        """
        Checks whether the episode should terminate.

        Success:
            closing phase: reach sampled P3 target without safety violation
            final phase: satisfy docking conditions without safety violation

        Failure:
            hard safety violation

        Warning sets do not terminate the episode.
        """

        monitor = self.monitor_safety(state)
        q = monitor["derived"]

        info = {
            "success": False,
            "failure": False,
            "time_limit": False,
            "warning": monitor["warning"],
            "safe": monitor["safe"],
            "unsafe": monitor["unsafe"],
            "phase": self.phase,
            "violated_constraints": monitor["violated_constraints"],

            "port_relative_position": q["r_p"],
            "port_relative_velocity": q["v_p"],
            "port_distance": q["port_distance"],
            "port_speed": q["port_speed"],
            "com_distance": q["com_distance"],
            "com_speed": q["com_speed"],
            "s": q["s"],
            "rho": q["rho"],
            "s_dot": q["s_dot"],
            "psi_e": q["psi_e"],
            "omega": q["omega"],

            "geometry": monitor["geometry"],
            "velocity": monitor["velocity"],
            "alignment": monitor["alignment"],
        }

        # 1. Hard safety failure first
        if monitor["unsafe"]:
            info["failure"] = True
            info["failure_reason"] = "safety_violation"
            info["termination_reason"] = "safety_violation"
            return True, info

        # 2. Success only if no hard safety violation
        if self.phase == "closing":
            if self.is_at_P3_target(state):
                info["success"] = True
                info["terminal_set"] = "P3_target"
                info["termination_reason"] = "success_P3_target"
                return True, info

        elif self.phase == "final":
            if self.is_in_X_dock(state):
                info["success"] = True
                info["terminal_set"] = "X_dock"
                info["termination_reason"] = "success_docking"
                return True, info

        else:
            raise ValueError("Unknown phase.")

        # 3. Time limit
        if self.time >= self.max_time:
            info["time_limit"] = True
            info["termination_reason"] = "timeout"
            return True, info

        # 4. Continue episode
        info["termination_reason"] = None
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