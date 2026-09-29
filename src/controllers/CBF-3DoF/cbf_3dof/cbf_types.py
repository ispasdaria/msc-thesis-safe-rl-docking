"""Shared data structures used by the final-approach CBF filter."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortKinematics:
    s: float
    y_p: float
    s_dot: float
    y_p_dot: float
    v_p: np.ndarray


@dataclass(frozen=True)
class PortAccelerationAffine:
    """Docking-port acceleration ``drift + control_matrix @ u``."""

    drift: np.ndarray
    control_matrix: np.ndarray


@dataclass(frozen=True)
class CBFConstraintSet:
    """Linear CBF inequalities represented with the convention ``A u >= b``."""

    A: np.ndarray
    b: np.ndarray
    names: tuple[str, ...]
    barrier_values: dict[str, float]
    barrier_derivatives: dict[str, float]
    kinematics: PortKinematics
    port_acceleration: PortAccelerationAffine


@dataclass(frozen=True)
class FilterResult:
    action_filtered: np.ndarray
    action_nominal: np.ndarray
    solver_status: str
    solver_backend: str
    intervened: bool
    intervention_norm: float
    intervention_scaled_norm: float
    barrier_values: dict[str, float]
    barrier_derivatives: dict[str, float]
    cbf_residuals: dict[str, float]
    relaxed_cbf_residuals: dict[str, float]
    slack_values: dict[str, float]
    qp_failed: bool
