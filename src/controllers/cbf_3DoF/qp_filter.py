"""Minimally invasive quadratic-program safety filter."""

from dataclasses import replace
import importlib.util

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, minimize

from .barriers import FinalApproachCBFConfig, build_final_approach_constraints
from .closing_barriers import ClosingCBFConfig, build_closing_constraints
from .port_dynamics import evaluate_port_acceleration
from .cbf_types import FilterResult


class FinalApproachCBFFilter:
    """Filter a nominal physical action through final-approach CBF rows."""

    def __init__(self, core_env, config=None):
        if core_env.phase != self.required_phase:
            raise ValueError(
                f"{type(self).__name__} requires a {self.required_phase}-phase "
                "environment."
            )
        self.core_env = core_env
        self.config = config or self.default_config()
        self.action_limits = np.asarray(core_env.action_high, dtype=np.float64)
        if self.action_limits.shape != (3,) or np.any(self.action_limits <= 0.0):
            raise ValueError("Expected three positive physical action limits.")
        if getattr(self.config, "enable_alignment", False):
            if self.config.attitude_max > core_env.psi_unsafe:
                raise ValueError(
                    "attitude_max cannot exceed the simulator's unsafe limit."
                )
            if self.config.angular_rate_max > core_env.omega_unsafe:
                raise ValueError(
                    "angular_rate_max cannot exceed the simulator's unsafe limit."
                )

    required_phase = "final"

    @staticmethod
    def default_config():
        return FinalApproachCBFConfig()

    def _build_constraints(self, state):
        return build_final_approach_constraints(self.core_env, state, self.config)

    def with_enabled_constraints(self, *, cone, speed, alignment):
        """Return a filter sharing the environment with selected CBF families."""
        return FinalApproachCBFFilter(
            self.core_env,
            replace(
                self.config,
                enable_cone=cone,
                enable_speed=speed,
                enable_alignment=alignment,
            ),
        )

    def filter(self, state, action_nominal):
        state = np.asarray(state, dtype=np.float64)
        action_nominal = np.asarray(action_nominal, dtype=np.float64)
        if state.shape != (6,):
            raise ValueError(f"Physical state must have shape (6,), got {state.shape}.")
        if action_nominal.shape != (3,):
            raise ValueError(
                f"Nominal physical action must have shape (3,), got {action_nominal.shape}."
            )
        if not np.all(np.isfinite(action_nominal)):
            raise ValueError("Nominal physical action must contain finite values.")

        nominal = np.clip(action_nominal, -self.action_limits, self.action_limits)
        constraints = self._build_constraints(state)
        solution, slack, status, backend, failed = self._solve_qp(
            nominal,
            constraints.A,
            constraints.b,
            constraints.names,
        )

        if failed:
            # Deliberate evaluation behavior: preserve the nominal SAC fallback
            # while marking the result as failed.  The paired evaluator records
            # an independent hard-feasibility diagnosis for every such step;
            # callers must not interpret this fallback as CBF-certified safety.
            solution = nominal.copy()
            slack = np.zeros(len(constraints.names), dtype=np.float64)

        raw_residual = constraints.A @ solution - constraints.b
        relaxed_residual = raw_residual + slack
        difference = solution - nominal
        scaled_difference = difference / self.action_limits

        barrier_derivatives = dict(constraints.barrier_derivatives)
        port_acceleration = None
        if "port_speed" in constraints.barrier_values:
            port_acceleration = evaluate_port_acceleration(
                constraints.port_acceleration,
                solution,
            )
            barrier_derivatives["port_speed"] = float(
                -2.0 * constraints.kinematics.v_p @ port_acceleration
            )
        if "axial_port_speed" in constraints.barrier_values:
            if port_acceleration is None:
                port_acceleration = evaluate_port_acceleration(
                    constraints.port_acceleration,
                    solution,
                )
            barrier_derivatives["axial_port_speed"] = float(
                -2.0 * constraints.kinematics.s_dot * port_acceleration[0]
            )
        if "angular_rate_upper" in constraints.barrier_values:
            angular_acceleration = float(solution[2])
            barrier_derivatives["angular_rate_upper"] = -angular_acceleration
            barrier_derivatives["angular_rate_lower"] = angular_acceleration

        return FilterResult(
            action_filtered=solution.copy(),
            action_nominal=nominal.copy(),
            solver_status=status,
            solver_backend=backend,
            intervened=bool(
                np.linalg.norm(scaled_difference) > self.config.intervention_eps
            ),
            intervention_norm=float(np.linalg.norm(difference)),
            intervention_scaled_norm=float(np.linalg.norm(scaled_difference)),
            barrier_values=dict(constraints.barrier_values),
            barrier_derivatives=barrier_derivatives,
            cbf_residuals={
                name: float(value)
                for name, value in zip(constraints.names, raw_residual)
            },
            relaxed_cbf_residuals={
                name: float(value)
                for name, value in zip(constraints.names, relaxed_residual)
            },
            slack_values={
                name: float(value)
                for name, value in zip(constraints.names, slack)
            },
            qp_failed=failed,
        )

    def _slack_penalties(self, names):
        return np.asarray(
            [
                self.config.speed_slack_penalty
                if name == "port_speed"
                else self.config.alignment_slack_penalty
                if name.startswith(("attitude_", "angular_rate_"))
                else self.config.cone_slack_penalty
                for name in names
            ],
            dtype=np.float64,
        )

    def _build_qp(self, nominal, cbf_A, cbf_b, names):
        n_constraints = len(names)
        n_slack = n_constraints if self.config.allow_slack else 0
        n_variables = 3 + n_slack

        action_weights = 1.0 / self.action_limits**2
        diagonal = list(action_weights)
        if n_slack:
            diagonal.extend(self._slack_penalties(names))

        P = sparse.diags(diagonal, format="csc")
        q = np.zeros(n_variables, dtype=np.float64)
        q[:3] = -action_weights * nominal

        cbf_matrix = np.zeros((n_constraints, n_variables), dtype=np.float64)
        cbf_matrix[:, :3] = cbf_A
        if n_slack:
            cbf_matrix[:, 3:] = np.eye(n_constraints, dtype=np.float64)

        action_matrix = np.zeros((3, n_variables), dtype=np.float64)
        action_matrix[:, :3] = np.eye(3, dtype=np.float64)

        matrices = [cbf_matrix, action_matrix]
        lower = [cbf_b, -self.action_limits]
        upper = [np.full(n_constraints, np.inf), self.action_limits]

        if n_slack:
            slack_matrix = np.zeros((n_constraints, n_variables), dtype=np.float64)
            slack_matrix[:, 3:] = np.eye(n_constraints, dtype=np.float64)
            matrices.append(slack_matrix)
            lower.append(np.zeros(n_constraints, dtype=np.float64))
            upper.append(np.full(n_constraints, np.inf))

        matrix = sparse.csc_matrix(np.vstack(matrices))
        return (
            P,
            q,
            matrix,
            np.concatenate(lower),
            np.concatenate(upper),
            n_slack,
        )

    def _select_backend(self):
        requested = self.config.solver_backend
        osqp_available = importlib.util.find_spec("osqp") is not None
        if requested == "osqp" and not osqp_available:
            raise RuntimeError(
                "OSQP was requested but is not installed. Install requirements.txt "
                "or select solver_backend='scipy'."
            )
        if requested == "auto":
            return "osqp" if osqp_available else "scipy"
        return requested

    def _solve_qp(self, nominal, cbf_A, cbf_b, names):
        P, q, matrix, lower, upper, n_slack = self._build_qp(
            nominal,
            cbf_A,
            cbf_b,
            names,
        )
        try:
            backend = self._select_backend()
            if backend == "osqp":
                vector, status, success = self._solve_osqp(P, q, matrix, lower, upper)
            else:
                vector, status, success = self._solve_scipy(
                    P,
                    q,
                    matrix,
                    lower,
                    upper,
                    nominal,
                    cbf_A,
                    cbf_b,
                    n_slack,
                )
        except Exception as exc:  # solver failures must be visible to evaluation
            return (
                nominal.copy(),
                np.zeros(len(names), dtype=np.float64),
                f"exception:{type(exc).__name__}:{exc}",
                self.config.solver_backend,
                True,
            )

        if not success or vector is None or not np.all(np.isfinite(vector)):
            return (
                nominal.copy(),
                np.zeros(len(names), dtype=np.float64),
                status,
                backend,
                True,
            )

        action = np.clip(vector[:3], -self.action_limits, self.action_limits)
        slack = (
            np.maximum(vector[3:], 0.0)
            if n_slack
            else np.zeros(len(names), dtype=np.float64)
        )
        return action, slack, status, backend, False

    def _solve_osqp(self, P, q, matrix, lower, upper):
        import osqp

        solver = osqp.OSQP()
        settings = dict(
            verbose=False,
            eps_abs=self.config.solver_eps_abs,
            eps_rel=self.config.solver_eps_rel,
            max_iter=self.config.solver_max_iter,
        )
        try:
            solver.setup(
                P=P,
                q=q,
                A=matrix,
                l=lower,
                u=upper,
                polishing=False,
                **settings,
            )
        except TypeError:  # compatibility with OSQP 0.x
            solver.setup(
                P=P,
                q=q,
                A=matrix,
                l=lower,
                u=upper,
                polish=False,
                **settings,
            )
        try:
            result = solver.solve(raise_error=False)
        except TypeError:  # compatibility with OSQP 0.x
            result = solver.solve()

        status = str(result.info.status)
        success = status.lower() in {"solved", "solved inaccurate"}
        return result.x, status, success

    def _solve_scipy(
        self,
        P,
        q,
        matrix,
        lower,
        upper,
        nominal,
        cbf_A,
        cbf_b,
        n_slack,
    ):
        dense_P = P.toarray()
        dense_matrix = matrix.toarray()
        n_variables = dense_P.shape[0]
        initial_physical = np.zeros(n_variables, dtype=np.float64)
        initial_physical[:3] = nominal
        if n_slack:
            initial_physical[3:] = (
                np.maximum(cbf_b - cbf_A @ nominal, 0.0) + 1.0e-10
            )

        # SLSQP is sensitive to the very different physical action and slack
        # scales.  Solve in variables whose quadratic weights are one, then
        # map the result back to physical units.
        scaling = 1.0 / np.sqrt(np.diag(dense_P))
        scaled_P = (scaling[:, None] * dense_P) * scaling[None, :]
        scaled_q = q * scaling
        scaled_matrix = dense_matrix * scaling[None, :]
        initial = initial_physical / scaling

        def objective(vector):
            return 0.5 * float(vector @ scaled_P @ vector) + float(scaled_q @ vector)

        def gradient(vector):
            return scaled_P @ vector + scaled_q

        result = minimize(
            objective,
            initial,
            jac=gradient,
            constraints=[LinearConstraint(scaled_matrix, lower, upper)],
            method="SLSQP",
            options={
                "ftol": self.config.solver_eps_abs,
                "maxiter": self.config.solver_max_iter,
                "disp": False,
            },
        )
        status = f"scipy:{result.status}:{result.message}"
        return scaling * result.x, status, bool(result.success)


class ClosingCBFFilter(FinalApproachCBFFilter):
    """Filter a nominal action through closing cone, speed, and alignment CBFs."""

    required_phase = "closing"

    @staticmethod
    def default_config():
        return ClosingCBFConfig()

    def __init__(self, core_env, config=None):
        super().__init__(core_env, config)
        if self.config.port_speed_max > core_env.vp_unsafe_close:
            raise ValueError(
                "port_speed_max cannot exceed the simulator's closing unsafe limit."
            )
        if self.config.axial_port_speed_max > core_env.sdot_unsafe_close:
            raise ValueError(
                "axial_port_speed_max cannot exceed the simulator's closing "
                "unsafe limit."
            )

    def with_enabled_constraints(
        self,
        *,
        port_speed,
        axial_port_speed,
        cone=None,
        alignment=None,
    ):
        """Return a closing filter with selected CBF families."""
        return ClosingCBFFilter(
            self.core_env,
            replace(
                self.config,
                enable_cone=(
                    self.config.enable_cone if cone is None else cone
                ),
                enable_port_speed=port_speed,
                enable_axial_port_speed=axial_port_speed,
                enable_alignment=(
                    self.config.enable_alignment
                    if alignment is None
                    else alignment
                ),
            ),
        )

    def _build_constraints(self, state):
        return build_closing_constraints(self.core_env, state, self.config)

    def _slack_penalties(self, names):
        return np.asarray(
            [
                self.config.cone_slack_penalty
                if name.startswith("cone_")
                else self.config.alignment_slack_penalty
                if name.startswith(("attitude_", "angular_rate_"))
                else self.config.axial_port_speed_slack_penalty
                if name == "axial_port_speed"
                else self.config.port_speed_slack_penalty
                for name in names
            ],
            dtype=np.float64,
        )
