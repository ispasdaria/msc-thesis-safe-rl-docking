"""Final-approach docking-cone and docking-port-speed CBF inequalities."""

from dataclasses import dataclass

import numpy as np

from .cbf_types import CBFConstraintSet
from .port_dynamics import (
    compute_port_acceleration_affine,
    compute_port_kinematics,
)


@dataclass(frozen=True)
class FinalApproachCBFConfig:
    """Configuration for the final-approach CBF-QP safety filter.

    The gain defaults are conservative starting values for validation, not
    claimed optimal values.  They must be frozen before the final test set is
    evaluated.
    """

    kappa: float = 0.1
    port_speed_max: float = 0.15
    cone_k1: float = 0.1
    cone_k2: float = 0.1
    speed_alpha: float = 0.5
    attitude_max: float = np.deg2rad(3.0)
    angular_rate_max: float = np.deg2rad(0.15)
    attitude_k1: float = 0.3
    attitude_k2: float = 0.3
    angular_rate_alpha: float = 0.5
    cone_slack_penalty: float = 1.0e8
    speed_slack_penalty: float = 1.0e8
    alignment_slack_penalty: float = 1.0e8
    numerical_eps: float = 1.0e-9
    intervention_eps: float = 1.0e-7
    solver_eps_abs: float = 1.0e-8
    solver_eps_rel: float = 1.0e-8
    solver_max_iter: int = 20_000
    allow_slack: bool = True
    enable_cone: bool = True
    enable_speed: bool = True
    enable_alignment: bool = True
    solver_backend: str = "auto"

    def __post_init__(self):
        positive_fields = {
            "kappa": self.kappa,
            "port_speed_max": self.port_speed_max,
            "cone_k1": self.cone_k1,
            "cone_k2": self.cone_k2,
            "speed_alpha": self.speed_alpha,
            "attitude_max": self.attitude_max,
            "angular_rate_max": self.angular_rate_max,
            "attitude_k1": self.attitude_k1,
            "attitude_k2": self.attitude_k2,
            "angular_rate_alpha": self.angular_rate_alpha,
            "cone_slack_penalty": self.cone_slack_penalty,
            "speed_slack_penalty": self.speed_slack_penalty,
            "alignment_slack_penalty": self.alignment_slack_penalty,
            "solver_eps_abs": self.solver_eps_abs,
            "solver_eps_rel": self.solver_eps_rel,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.numerical_eps < 0.0 or self.intervention_eps < 0.0:
            raise ValueError("Numerical tolerances must be nonnegative.")
        if self.solver_max_iter <= 0:
            raise ValueError("solver_max_iter must be positive.")
        if self.attitude_max >= np.pi:
            raise ValueError("attitude_max must be smaller than pi radians.")
        if not self.enable_cone and not self.enable_speed and not self.enable_alignment:
            raise ValueError("At least one CBF constraint must be enabled.")
        if self.solver_backend not in {"auto", "osqp", "scipy"}:
            raise ValueError("solver_backend must be 'auto', 'osqp', or 'scipy'.")


def cone_barrier_values(s, y_p, kappa):
    """Return the upper and lower final-approach cone barriers."""
    return float(kappa * s - y_p), float(kappa * s + y_p)


def cone_barrier_derivatives(s_dot, y_p_dot, kappa):
    """Return the time derivatives of the two cone barriers."""
    return (
        float(kappa * s_dot - y_p_dot),
        float(kappa * s_dot + y_p_dot),
    )


def speed_barrier_value(v_p, v_max):
    """Return the squared docking-port speed margin."""
    v_p = np.asarray(v_p, dtype=np.float64)
    return float(v_max**2 - np.dot(v_p, v_p))


def alignment_barrier_values(psi_error, omega, attitude_max, angular_rate_max):
    """Return upper/lower attitude and angular-rate box barriers."""
    return {
        "attitude_upper": float(attitude_max - psi_error),
        "attitude_lower": float(attitude_max + psi_error),
        "angular_rate_upper": float(angular_rate_max - omega),
        "angular_rate_lower": float(angular_rate_max + omega),
    }


def build_final_approach_constraints(core_env, state, config=None):
    """Build enabled final-approach inequalities with convention ``A u >= b``."""
    if config is None:
        config = FinalApproachCBFConfig()

    kinematics = compute_port_kinematics(core_env, state)
    acceleration = compute_port_acceleration_affine(core_env, state)

    rows = []
    bounds = []
    names = []
    barrier_values = {}
    barrier_derivatives = {}

    if config.enable_cone:
        h_plus, h_minus = cone_barrier_values(
            kinematics.s,
            kinematics.y_p,
            config.kappa,
        )
        h_plus_dot, h_minus_dot = cone_barrier_derivatives(
            kinematics.s_dot,
            kinematics.y_p_dot,
            config.kappa,
        )

        b_s, b_y = acceleration.drift
        g_s = acceleration.control_matrix[0]
        g_y = acceleration.control_matrix[1]
        gain_sum = config.cone_k1 + config.cone_k2
        gain_product = config.cone_k1 * config.cone_k2

        rows.extend((config.kappa * g_s - g_y, config.kappa * g_s + g_y))
        bounds.extend(
            (
                -config.kappa * b_s
                + b_y
                - gain_sum * h_plus_dot
                - gain_product * h_plus,
                -config.kappa * b_s
                - b_y
                - gain_sum * h_minus_dot
                - gain_product * h_minus,
            )
        )
        names.extend(("cone_upper", "cone_lower"))
        barrier_values.update(cone_upper=h_plus, cone_lower=h_minus)
        barrier_derivatives.update(
            cone_upper=h_plus_dot,
            cone_lower=h_minus_dot,
        )

    if config.enable_speed:
        h_speed = speed_barrier_value(
            kinematics.v_p,
            config.port_speed_max,
        )
        speed_row = -2.0 * kinematics.v_p @ acceleration.control_matrix
        speed_bound = (
            2.0 * float(kinematics.v_p @ acceleration.drift)
            - config.speed_alpha * h_speed
        )
        rows.append(speed_row)
        bounds.append(speed_bound)
        names.append("port_speed")
        barrier_values["port_speed"] = h_speed
        barrier_derivatives["port_speed_drift"] = float(
            -2.0 * kinematics.v_p @ acceleration.drift
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
        attitude_gain_sum = config.attitude_k1 + config.attitude_k2
        attitude_gain_product = config.attitude_k1 * config.attitude_k2
        angular_column = np.array([0.0, 0.0, 1.0], dtype=np.float64)

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
