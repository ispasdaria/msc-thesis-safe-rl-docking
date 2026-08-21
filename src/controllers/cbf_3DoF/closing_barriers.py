"""Docking-cone, speed, and alignment CBFs for the closing phase."""

from dataclasses import dataclass

import numpy as np

from .cbf_types import CBFConstraintSet
from .barriers import (
    alignment_barrier_values,
    cone_barrier_derivatives,
    cone_barrier_values,
)
from .port_dynamics import (
    compute_port_acceleration_affine,
    compute_port_kinematics,
)


@dataclass(frozen=True)
class ClosingCBFConfig:
    """Configuration for closing-phase cone, speed, and alignment CBFs.

    The defaults match the closing phase's fully safe velocity thresholds,
    following the same conservative-envelope convention as the existing
    final-approach speed CBF.  Cone geometry is always read from the simulator;
    only the HOCBF response gains are configurable.
    """

    cone_k1: float = 0.1
    cone_k2: float = 0.1
    cone_interior_margin: float = 1.0e-4
    port_speed_max: float = 0.25
    axial_port_speed_max: float = 0.15
    port_speed_alpha: float = 0.5
    axial_port_speed_alpha: float = 0.5
    attitude_max: float = np.deg2rad(3.0)
    angular_rate_max: float = np.deg2rad(0.15)
    attitude_k1: float = 0.3
    attitude_k2: float = 0.3
    angular_rate_alpha: float = 0.5
    cone_slack_penalty: float = 1.0e8
    port_speed_slack_penalty: float = 1.0e8
    axial_port_speed_slack_penalty: float = 1.0e8
    alignment_slack_penalty: float = 1.0e8
    numerical_eps: float = 1.0e-9
    intervention_eps: float = 1.0e-7
    solver_eps_abs: float = 1.0e-8
    solver_eps_rel: float = 1.0e-8
    solver_max_iter: int = 20_000
    allow_slack: bool = True
    enable_cone: bool = True
    enable_port_speed: bool = True
    enable_axial_port_speed: bool = True
    enable_alignment: bool = True
    solver_backend: str = "auto"

    def __post_init__(self):
        positive_fields = {
            "cone_k1": self.cone_k1,
            "cone_k2": self.cone_k2,
            "port_speed_max": self.port_speed_max,
            "axial_port_speed_max": self.axial_port_speed_max,
            "port_speed_alpha": self.port_speed_alpha,
            "axial_port_speed_alpha": self.axial_port_speed_alpha,
            "attitude_max": self.attitude_max,
            "angular_rate_max": self.angular_rate_max,
            "attitude_k1": self.attitude_k1,
            "attitude_k2": self.attitude_k2,
            "angular_rate_alpha": self.angular_rate_alpha,
            "cone_slack_penalty": self.cone_slack_penalty,
            "port_speed_slack_penalty": self.port_speed_slack_penalty,
            "axial_port_speed_slack_penalty": self.axial_port_speed_slack_penalty,
            "alignment_slack_penalty": self.alignment_slack_penalty,
            "solver_eps_abs": self.solver_eps_abs,
            "solver_eps_rel": self.solver_eps_rel,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.numerical_eps < 0.0 or self.intervention_eps < 0.0:
            raise ValueError("Numerical tolerances must be nonnegative.")
        if (
            not np.isfinite(self.cone_interior_margin)
            or self.cone_interior_margin < 0.0
        ):
            raise ValueError("cone_interior_margin must be finite and nonnegative.")
        if self.solver_max_iter <= 0:
            raise ValueError("solver_max_iter must be positive.")
        if self.attitude_max >= np.pi:
            raise ValueError("attitude_max must be smaller than pi radians.")
        if (
            not self.enable_cone
            and not self.enable_port_speed
            and not self.enable_axial_port_speed
            and not self.enable_alignment
        ):
            raise ValueError("At least one closing CBF constraint must be enabled.")
        if self.solver_backend not in {"auto", "osqp", "scipy"}:
            raise ValueError("solver_backend must be 'auto', 'osqp', or 'scipy'.")


def axial_speed_barrier_value(s_dot, s_dot_max):
    """Return ``h = s_dot_max^2 - s_dot^2`` for either axial direction."""
    return float(s_dot_max**2 - float(s_dot) ** 2)


def build_closing_constraints(core_env, state, config=None):
    """Build enabled closing-phase inequalities as ``A u >= b``.

    The three finite-cone barriers have relative degree two with respect to
    the physical body action ``u = [a_x, a_y, alpha_z]``:

    ``h_upper = cone_slope * s - y_p``
    ``h_lower = cone_slope * s + y_p``
    ``h_axial_upper = cone_length - s``

    The lateral pair also implies ``s >= 0``.  Each cone row uses the
    exponential HOCBF condition ``h_ddot + (k1+k2)h_dot + k1*k2*h >= 0``.

    Both speed barriers have relative degree one with respect to the physical
    action ``u = [a_x, a_y, alpha_z]``:

    ``h_total = v_max^2 - v_p.T v_p``
    ``h_axial = s_dot_max^2 - s_dot^2``

    and each row enforces ``h_dot + alpha h >= 0``. Alignment adds upper and
    lower attitude HOCBF rows plus upper and lower angular-rate CBF rows. These
    rows protect the closing handoff envelope even though closing-phase
    simulator termination does not classify alignment as a failure reason.
    """
    if config is None:
        config = ClosingCBFConfig()

    kinematics = compute_port_kinematics(core_env, state)
    acceleration = compute_port_acceleration_affine(core_env, state)

    rows = []
    bounds = []
    names = []
    barrier_values = {}
    barrier_derivatives = {}

    if config.enable_cone:
        cone_slope = float(core_env.cone_slope)
        cone_length = float(core_env.cone_length)
        h_upper, h_lower = cone_barrier_values(
            kinematics.s,
            kinematics.y_p,
            cone_slope,
        )
        # A sub-millimetre inset prevents solver/integration roundoff from
        # toggling the simulator's exact cone-membership Boolean at h = 0.
        h_upper_cbf = h_upper - config.cone_interior_margin
        h_lower_cbf = h_lower - config.cone_interior_margin
        h_upper_dot, h_lower_dot = cone_barrier_derivatives(
            kinematics.s_dot,
            kinematics.y_p_dot,
            cone_slope,
        )
        h_axial_upper = float(cone_length - kinematics.s)
        h_axial_upper_cbf = h_axial_upper - config.cone_interior_margin
        h_axial_upper_dot = float(-kinematics.s_dot)

        b_s, b_y = acceleration.drift
        g_s = acceleration.control_matrix[0]
        g_y = acceleration.control_matrix[1]
        gain_sum = config.cone_k1 + config.cone_k2
        gain_product = config.cone_k1 * config.cone_k2

        rows.extend(
            (
                cone_slope * g_s - g_y,
                cone_slope * g_s + g_y,
                -g_s,
            )
        )
        bounds.extend(
            (
                -cone_slope * b_s
                + b_y
                - gain_sum * h_upper_dot
                - gain_product * h_upper_cbf,
                -cone_slope * b_s
                - b_y
                - gain_sum * h_lower_dot
                - gain_product * h_lower_cbf,
                b_s
                + gain_sum * kinematics.s_dot
                - gain_product * h_axial_upper_cbf,
            )
        )
        names.extend(("cone_upper", "cone_lower", "cone_axial_upper"))
        barrier_values.update(
            cone_upper=h_upper,
            cone_lower=h_lower,
            cone_axial_upper=h_axial_upper,
        )
        barrier_derivatives.update(
            cone_upper=h_upper_dot,
            cone_lower=h_lower_dot,
            cone_axial_upper=h_axial_upper_dot,
        )

    if config.enable_port_speed:
        h_total = float(
            config.port_speed_max**2 - np.dot(kinematics.v_p, kinematics.v_p)
        )
        rows.append(-2.0 * kinematics.v_p @ acceleration.control_matrix)
        bounds.append(
            2.0 * float(kinematics.v_p @ acceleration.drift)
            - config.port_speed_alpha * h_total
        )
        names.append("port_speed")
        barrier_values["port_speed"] = h_total
        barrier_derivatives["port_speed_drift"] = float(
            -2.0 * kinematics.v_p @ acceleration.drift
        )

    if config.enable_axial_port_speed:
        h_axial = axial_speed_barrier_value(
            kinematics.s_dot,
            config.axial_port_speed_max,
        )
        g_s = acceleration.control_matrix[0]
        b_s = float(acceleration.drift[0])
        rows.append(-2.0 * kinematics.s_dot * g_s)
        bounds.append(
            2.0 * kinematics.s_dot * b_s
            - config.axial_port_speed_alpha * h_axial
        )
        names.append("axial_port_speed")
        barrier_values["axial_port_speed"] = h_axial
        barrier_derivatives["axial_port_speed_drift"] = float(
            -2.0 * kinematics.s_dot * b_s
        )

    if config.enable_alignment:
        psi_error = float(core_env.wrap_to_pi(state[4]))
        omega = float(state[5])
        alignment_values = alignment_barrier_values(
            psi_error,
            omega,
            config.attitude_max,
            config.angular_rate_max,
        )
        angular_column = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        attitude_gain_sum = config.attitude_k1 + config.attitude_k2
        attitude_gain_product = config.attitude_k1 * config.attitude_k2

        rows.extend((-angular_column, angular_column))
        bounds.extend(
            (
                attitude_gain_sum * omega
                - attitude_gain_product * alignment_values["attitude_upper"],
                -attitude_gain_sum * omega
                - attitude_gain_product * alignment_values["attitude_lower"],
            )
        )
        names.extend(("attitude_upper", "attitude_lower"))

        rows.extend((-angular_column, angular_column))
        bounds.extend(
            (
                -config.angular_rate_alpha
                * alignment_values["angular_rate_upper"],
                -config.angular_rate_alpha
                * alignment_values["angular_rate_lower"],
            )
        )
        names.extend(("angular_rate_upper", "angular_rate_lower"))
        barrier_values.update(alignment_values)
        barrier_derivatives.update(
            attitude_upper=-omega,
            attitude_lower=omega,
        )

    return CBFConstraintSet(
        A=np.asarray(rows, dtype=np.float64).reshape(-1, 3),
        b=np.asarray(bounds, dtype=np.float64),
        names=tuple(names),
        barrier_values=barrier_values,
        barrier_derivatives=barrier_derivatives,
        kinematics=kinematics,
        port_acceleration=acceleration,
    )
