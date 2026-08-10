"""Feasibility diagnostics for final-approach CBF constraints."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class HardCBFFeasibilityResult:
    """Result of maximizing the minimum raw CBF-row residual."""

    feasible: bool
    margin: float
    action: np.ndarray | None
    solver_status: str


def cone_hocbf_admissibility(constraints, config):
    """Return ``psi_1 = h_dot + k1*h`` for both cone sides.

    A relative-degree-two HOCBF forward-invariance claim starts from states
    satisfying both ``h >= 0`` and ``psi_1 >= 0``.
    """
    required = ("cone_upper", "cone_lower")
    if any(name not in constraints.barrier_values for name in required):
        raise ValueError("Cone constraints must be enabled.")

    return {
        name: float(
            constraints.barrier_derivatives[name]
            + config.cone_k1 * constraints.barrier_values[name]
        )
        for name in required
    }


def attitude_hocbf_admissibility(constraints, config):
    """Return ``psi_1 = h_dot + k1*h`` for both attitude limits."""
    required = ("attitude_upper", "attitude_lower")
    if any(name not in constraints.barrier_values for name in required):
        raise ValueError("Alignment constraints must be enabled.")

    return {
        name: float(
            constraints.barrier_derivatives[name]
            + config.attitude_k1 * constraints.barrier_values[name]
        )
        for name in required
    }


def hard_cbf_feasibility_margin(constraints, action_limits, tolerance=1.0e-9):
    """Check whether actuator-bounded controls can satisfy ``A u >= b``.

    The LP maximizes a scalar ``gamma`` such that every raw row residual obeys
    ``A_i u - b_i >= gamma``.  Its sign is an exact hard-feasibility test:
    nonnegative means the original rows are feasible.  Because different CBF
    families have different physical units, the magnitude is diagnostic only
    and must not be compared across differently scaled formulations.
    """
    limits = np.asarray(action_limits, dtype=np.float64)
    if limits.shape != (3,) or np.any(limits <= 0.0):
        raise ValueError("action_limits must contain three positive values.")

    A = np.asarray(constraints.A, dtype=np.float64)
    b = np.asarray(constraints.b, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 3 or b.shape != (A.shape[0],):
        raise ValueError("Expected CBF constraints with A shape (m, 3).")

    # -A u + gamma <= -b is equivalent to A u - b >= gamma.
    inequality_matrix = np.column_stack((-A, np.ones(A.shape[0])))
    objective = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    bounds = [(-limit, limit) for limit in limits] + [(None, None)]
    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=-b,
        bounds=bounds,
        method="highs",
    )

    if not result.success or result.x is None:
        return HardCBFFeasibilityResult(
            feasible=False,
            margin=float("nan"),
            action=None,
            solver_status=f"scipy-highs:{result.status}:{result.message}",
        )

    margin = float(result.x[-1])
    return HardCBFFeasibilityResult(
        feasible=margin >= -tolerance,
        margin=margin,
        action=np.asarray(result.x[:3], dtype=np.float64),
        solver_status=f"scipy-highs:{result.status}:{result.message}",
    )
