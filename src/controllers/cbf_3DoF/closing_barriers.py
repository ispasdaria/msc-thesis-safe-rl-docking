"""Docking-port speed CBF inequalities for the closing phase."""

from dataclasses import dataclass

import numpy as np

from .cbf_types import CBFConstraintSet
from .port_dynamics import (
    compute_port_acceleration_affine,
    compute_port_kinematics,
)


@dataclass(frozen=True)
class ClosingCBFConfig:
    """Configuration for the two closing-phase speed CBFs.

    The defaults match the closing phase's fully safe velocity thresholds,
    following the same conservative-envelope convention as the existing
    final-approach speed CBF.  The warning and unsafe thresholds remain
    evaluation diagnostics rather than CBF boundaries.
    """

    port_speed_max: float = 0.25
    axial_port_speed_max: float = 0.15
    port_speed_alpha: float = 0.5
    axial_port_speed_alpha: float = 0.5
    port_speed_slack_penalty: float = 1.0e8
    axial_port_speed_slack_penalty: float = 1.0e8
    numerical_eps: float = 1.0e-9
    intervention_eps: float = 1.0e-7
    solver_eps_abs: float = 1.0e-8
    solver_eps_rel: float = 1.0e-8
    solver_max_iter: int = 20_000
    allow_slack: bool = True
    enable_port_speed: bool = True
    enable_axial_port_speed: bool = True
    solver_backend: str = "auto"

    def __post_init__(self):
        positive_fields = {
            "port_speed_max": self.port_speed_max,
            "axial_port_speed_max": self.axial_port_speed_max,
            "port_speed_alpha": self.port_speed_alpha,
            "axial_port_speed_alpha": self.axial_port_speed_alpha,
            "port_speed_slack_penalty": self.port_speed_slack_penalty,
            "axial_port_speed_slack_penalty": self.axial_port_speed_slack_penalty,
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
        if not self.enable_port_speed and not self.enable_axial_port_speed:
            raise ValueError("At least one closing CBF constraint must be enabled.")
        if self.solver_backend not in {"auto", "osqp", "scipy"}:
            raise ValueError("solver_backend must be 'auto', 'osqp', or 'scipy'.")


def axial_speed_barrier_value(s_dot, s_dot_max):
    """Return ``h = s_dot_max^2 - s_dot^2`` for either axial direction."""
    return float(s_dot_max**2 - float(s_dot) ** 2)


def build_closing_constraints(core_env, state, config=None):
    """Build enabled closing-phase speed inequalities as ``A u >= b``.

    Both barriers have relative degree one with respect to the physical body
    action ``u = [a_x, a_y, alpha_z]``:

    ``h_total = v_max^2 - v_p.T v_p``
    ``h_axial = s_dot_max^2 - s_dot^2``

    and each row enforces ``h_dot + alpha h >= 0``.
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

    return CBFConstraintSet(
        A=np.asarray(rows, dtype=np.float64).reshape(-1, 3),
        b=np.asarray(bounds, dtype=np.float64),
        names=tuple(names),
        barrier_values=barrier_values,
        barrier_derivatives=barrier_derivatives,
        kinematics=kinematics,
        port_acceleration=acceleration,
    )
